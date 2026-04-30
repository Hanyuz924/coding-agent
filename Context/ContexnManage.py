import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Any, Literal, cast
import logging
import litellm

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel
import json

ContextSizeKind = Literal["messages", "tokens", "fraction"]
ContextSize = tuple[ContextSizeKind, int | float]

logger = logging.getLogger("myagent.context")

_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_MAX_INPUT_TOKENS = 400_000
_DEFAULT_MESSAGES_TO_KEEP = 10
_DEFAULT_TOKLENS_TO_KEEP = 4000
_DEFAULT_SUMMARIZE_TOKENS_TRI =  50_000

class ContextManagerConfig(BaseModel):
    task_model_name: str

    summarization_model_name: str
    """Name of the model used for summarizaiton. """

    model_api_key: str

    summarization_template: str
    """Summarization prompt."""

    trigger_condition: ContextSize = ("tokens", _DEFAULT_SUMMARIZE_TOKENS_TRI)
    """Trigger condition for summarization

    Trigger summarization when 50 messages is reached
    ("messages", 50)

    Trigger summarization when 3000 tokens is reached
    ("tokens", 3000)

    Trigger summarization either when 80% of model's max input tokens
    is reached or when 100 messages is reached (whichever comes first)
    [("fraction", 0.8), ("messages", 100)]
    """

    trim_tokens_to_summarization: int | None = _DEFAULT_TRIM_TOKEN_LIMIT
    """
    trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for
    the summarization call.
    """

    message_keep: ContextSize = ("tokens", _DEFAULT_TOKLENS_TO_KEEP)

    token_limits: int = _DEFAULT_MAX_INPUT_TOKENS
    """
    
    """

class BaseContextManager:
    def __init__(self, config: ContextManagerConfig, **kwargs) -> None:
        self.config = config
        self.summarization_cost = 0.0

    def ContextManage(self, chat_message: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Summarize old messages and keep the recent suffix intact."""
        if not chat_message:
            return chat_message
        total_message_token_count = self._count_token_num(
            model_name= self.config.task_model_name,
            message= chat_message
        )
        logger.info(f"ContextManage: Processing {len(chat_message)} total messages, with total token account {total_message_token_count}")

        
        if not self._should_summarize(message=chat_message):
            logger.info("ContextManage: No summarization needed, returning original messages")
            return chat_message

        logger.info("ContextManage: Summarization triggered")
        
        system_prefix, working_messages = self._split_leading_system_messages(chat_message)
        logger.info(f"ContextManage: Found {len(system_prefix)} system messages, {len(working_messages)} working messages")
        
        if not working_messages:
            return chat_message

        cut_off_index = self._find_cut_index(message=working_messages)
        logger.info(f"ContextManage: Cutoff point found at index {cut_off_index}")
        
        if cut_off_index > 0:
            cutoff_msg = working_messages[cut_off_index] if cut_off_index < len(working_messages) else None
            logger.info(f"ContextManage: Message at cutoff point - role: {cutoff_msg.get('role') if cutoff_msg else 'N/A'}, has tool_call_id: {'tool_call_id' in cutoff_msg if cutoff_msg else False}")
        
        if cut_off_index <= 0:
            logger.info("ContextManage: Cutoff index <= 0, returning original messages")
            return chat_message
        
        #TODO Add trim message for summarization , need to add this feature later ...
        msg_to_summarize = working_messages[:cut_off_index]
        kept_messages = working_messages[cut_off_index:]
        
        logger.info(f"ContextManage: Will summarize {len(msg_to_summarize)} messages, keeping {len(kept_messages)} messages")

        try:
            llm_response = self._generate_summary(msg_to_summarize)
            summarization_string = self._parse_llm_response(llm_response).strip()
            self.summarization_cost = self._calculate_cost(llm_response)
            logger.info(f"ContextManage: Summarization cost: ${self.summarization_cost:.4f}")
            logger.info(f"ContextManage: Summary result length: {len(summarization_string)} chars")
            logger.info(f"ContextManage: Summary preview: {summarization_string[:200]}...")
        except Exception as e:
            logger.exception(f"Failed to generate summary: {e}; falling back to original messages")
            return chat_message

        if not summarization_string:
            logger.warning("ContextManage: Summarization returned empty string")
            return chat_message

        summary_message = self._build_summary_message(summarization_string)
        logger.info(f"ContextManage: Built summary message with {len(summary_message['content'])} chars")
        
        result_messages = [*system_prefix, summary_message, *kept_messages]
        logger.info(f"ContextManage: Final message count after summarization: {len(result_messages)} (was {len(chat_message)})")
        logger.info(f"ContextManage: Message structure - system: {len(system_prefix)}, summary: 1, kept: {len(kept_messages)}")
        
        return result_messages
    
    @staticmethod
    def _split_leading_system_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        prefix: list[dict[str, Any]] = []
        idx = 0
        while idx < len(messages) and messages[idx].get("role") == "system":
            prefix.append(messages[idx])
            idx += 1
        return prefix, messages[idx:]

    def _should_summarize(self, message: list[dict]):
        total_tokens = self._count_token_num(model_name=self.config.task_model_name, message=message)
        method = self.config.trigger_condition[0]
        condition = self.config.trigger_condition[1]
        
        logger.debug(f"_should_summarize: method={method}, condition={condition}, total_tokens={total_tokens}, message_count={len(message)}")
        
        if method == "messages":
            should = len(message) > condition
            logger.debug(f"_should_summarize (messages): {len(message)} > {condition} = {should}")
            if should:
                return True
        if method == "tokens":
            should = total_tokens > condition
            logger.debug(f"_should_summarize (tokens): {total_tokens} > {condition} = {should}")
            if should:
                return True
        if method == "fraction":
            max_input_token = self.config.token_limits
            threshold_value = int(max_input_token * condition)
            should = total_tokens > threshold_value
            logger.debug(f"_should_summarize (fraction): {total_tokens} > {threshold_value} ({condition*100}% of {max_input_token}) = {should}")
            if should:
                return True
        
        logger.debug("_should_summarize: No trigger condition met")
        return False
            
    def _find_safe_cutoff_point(self, message: list[dict[str, Any]], cut_index: int) -> int:
        """
        If cut_index lands on a tool message, move cutoff backward to the assistant
        message that issued the matching tool_call_id.

        message[:cut_index] -> summarize
        message[cut_index:] -> keep
        """
        if cut_index >= len(message):
            logger.debug(f"_find_safe_cutoff_point: cut_index {cut_index} >= len(message) {len(message)}, returning as-is")
            return cut_index

        cutoff_message = message[cut_index]
        if cutoff_message.get("role") != "tool":
            logger.debug(f"_find_safe_cutoff_point: Message at index {cut_index} is not tool message (role={cutoff_message.get('role')}), returning as-is")
            return cut_index

        tool_call_id = cutoff_message.get("tool_call_id")
        if not tool_call_id:
            logger.warning("_find_safe_cutoff_point: cutoff point is a tool call but no tool call id found")
            return cut_index
        
        logger.info(f"_find_safe_cutoff_point: Cutoff lands on tool message, searching backward for matching tool_call_id={tool_call_id}")
        
        for i in range(cut_index - 1, -1, -1):
            msg = message[i]
            if msg.get("role") != "assistant":
                continue

            tool_calls = msg.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue

            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("id") == tool_call_id:
                    logger.info(f"_find_safe_cutoff_point: Found matching assistant message at index {i}, moving cutoff from {cut_index} to {i}")
                    return i

        logger.warning(f"_find_safe_cutoff_point: Could not find matching assistant message for tool_call_id={tool_call_id}")
        return cut_index

    def _find_cut_index(self, message: list[dict[str, Any]]) -> int:
        """Default to `message_keep` for deciding the cutoff."""
        keep_method, value = self.config.message_keep[0], self.config.message_keep[1]
        logger.debug(f"_find_cut_index: keep_method={keep_method}, value={value}, total_messages={len(message)}")

        if keep_method in ["tokens", "fraction"]:
            return self._find_token_based_cutoff(message)

        messages_to_keep = cast(int, value)
        if len(message) <= messages_to_keep:
            logger.debug(f"_find_cut_index: total_messages ({len(message)}) <= messages_to_keep ({messages_to_keep}), no cutoff needed")
            return 0

        cut_off_index = len(message) - messages_to_keep
        logger.info(f"_find_cut_index: Calculated cut_off_index = {len(message)} - {messages_to_keep} = {cut_off_index}")
        
        safe_index = self._find_safe_cutoff_point(message, cut_off_index)
        logger.info(f"_find_cut_index: Final safe cutoff index = {safe_index}")
        return safe_index

    def _find_token_based_cutoff(self, message: list[dict[str, Any]]) -> int:
        if not message:
            return 0

        keep_method, value = self.config.message_keep[0], self.config.message_keep[1]
        if keep_method == "fraction":
            token_to_keep = int(self.config.token_limits * value)
        else:
            token_to_keep = int(value)

        if token_to_keep <= 0:
            token_to_keep = 1

        total_tokens = self._count_token_num(model_name=self.config.task_model_name, message=message)
        logger.debug(f"_find_token_based_cutoff: keep_method={keep_method}, token_to_keep={token_to_keep}, total_tokens={total_tokens}")

        if total_tokens <= token_to_keep:
            logger.debug(f"_find_token_based_cutoff: total_tokens ({total_tokens}) <= token_to_keep ({token_to_keep}), no cutoff needed")
            return 0

        # Find the earliest index whose suffix fits inside token_to_keep.
        left, right = 0, len(message)
        cut_off_message_index = len(message)
        max_iterations = len(message).bit_length() + 1

        logger.debug(f"_find_token_based_cutoff: Binary search starting, left=0, right={len(message)}, max_iterations={max_iterations}")

        for iteration in range(max_iterations):
            if left >= right:
                break
            mid = (left + right) // 2
            mid_suffix_tokens = self._count_token_num(model_name=self.config.task_model_name, message=message[mid:])
            
            if mid_suffix_tokens <= token_to_keep:
                cut_off_message_index = mid
                right = mid
                logger.debug(f"_find_token_based_cutoff: Iteration {iteration}: mid={mid}, suffix_tokens={mid_suffix_tokens} <= {token_to_keep}, moving right")
            else:
                left = mid + 1
                logger.debug(f"_find_token_based_cutoff: Iteration {iteration}: mid={mid}, suffix_tokens={mid_suffix_tokens} > {token_to_keep}, moving left")

        if cut_off_message_index == len(message):
            cut_off_message_index = left

        if cut_off_message_index >= len(message):
            if len(message) == 1:
                return 0
            cut_off_message_index = len(message) - 1

        logger.info(f"_find_token_based_cutoff: Final cut_off_message_index={cut_off_message_index}")
        safe_index = self._find_safe_cutoff_point(message, cut_off_message_index)
        return safe_index

    def _count_token_num(self, model_name: str, message: list[dict]):
        token_nums = litellm.token_counter(
            model=model_name,
            messages= message
        )
        return token_nums
    
    def _get_summarization_prompt(self, message_to_summarize: list[dict[str, Any]]) -> str:
        rendered_messages = json.dumps(message_to_summarize, ensure_ascii=False, indent=2)
        prompt = Template(
            self.config.summarization_template,
            undefined=StrictUndefined,
        ).render(messages=rendered_messages)
        logger.debug(f"_get_summarization_prompt: Rendered prompt with {len(prompt)} chars from {len(message_to_summarize)} messages")
        return prompt

    def _generate_summary(self, message_to_summarize: list[dict[str, Any]]) -> litellm.ModelResponse:
        prompt = self._get_summarization_prompt(message_to_summarize)
        try:
            logger.info(
                f"_generate_summary: Calling LLM API with model={self.config.summarization_model_name}, "
                f"prompt_chars={len(prompt)}, messages_to_summarize={len(message_to_summarize)}"
            )
            logger.debug(f"_generate_summary: Prompt preview (first 300 chars): {prompt[:300]}...")
            
            response = litellm.completion(
                model=self.config.summarization_model_name,
                messages=[{"role": "user", "content": prompt}],
                api_key=self.config.model_api_key,
            )

            logger.info("_generate_summary: LLM API call successful")

            usage = getattr(response, "usage", None)
            hidden_params = getattr(response, "_hidden_params", {}) or {}

            logger.debug(
                "_generate_summary: usage input_tokens=%s, output_tokens=%s, total_tokens=%s, response_cost=%s",
                getattr(usage, "prompt_tokens", None) if usage else None,
                getattr(usage, "completion_tokens", None) if usage else None,
                getattr(usage, "total_tokens", None) if usage else None,
                hidden_params.get("response_cost"),
            )
            
            # self._update_summarization_cost(response)
            return response                             # type: ignore
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"_generate_summary: Authentication error: {e}")
            if hasattr(e, "message"):
                e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise

    def _parse_llm_response(self, llm_response: litellm.ModelResponse) -> str:
        raw_content = llm_response.choices[0].message.content
        if raw_content is None:
            return ""
        return raw_content

    def _build_summary_message(self, summarization_string: str) -> dict[str, str]:
        summary_message = {
            "role": "user",
            "content": (
                "Here is a summary of the conversation so far. "
                "This is not original raw history; it is condensed context to help continue the task.\n\n"
                f"{summarization_string}"
            ),
        }
        logger.info(f"_build_summary_message: Created new summary message with content length={len(summary_message['content'])} chars")
        logger.debug(f"_build_summary_message: Summary message content preview (first 200 chars): {summary_message['content'][:200]}...")
        return summary_message
    
    def _calculate_cost(self, response: litellm.ModelResponse) -> float:
        cost = litellm.completion_cost(completion_response=response)
        return cost
        