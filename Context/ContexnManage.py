import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import litellm
from WorkSpace import LocalWorkSpace
from pathlib import Path
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel
import exceptions as ex
from utils import utils
import json
logger = logging.getLogger("myagent.contex")


_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_MAX_INPUT_TOKENS = 100_000
_DEFAULT_MESSAGES_TO_KEEP = 20

class ContextManagerConfig(BaseModel):
    summarization_model_name: str
    """Name of the model used for summarizaiton. """

    model_api_key: str

    summarization_prompt_template: str
    """Summarization prompt."""

    trigger_condition: tuple | list[tuple] 
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

    message_keep: tuple = ("message",_DEFAULT_MESSAGES_TO_KEEP)

    token_limits: int = _DEFAULT_MAX_INPUT_TOKENS
    """
    
    """

class BaseContextManager:
    def __init__(self, config: ContextManagerConfig, **kwargs) -> None:
        self.config = config
        self.summarization_cost = 0.0

    def ContextManage(self, chat_message: list[dict]) -> list[dict]:
        if not self._should_summarize(message=chat_message):
            return chat_message
        cut_off_index = self._find_cut_index(message=chat_message)
        if cut_off_index <= 0 :
            return chat_message
        
        msg_to_summerize = chat_message[:cut_off_index]

        #TODO 1. parse the summerization message from llm 
        #TODO 2.build the new message summerization + kept message , need to figure out whether need some value to notice the llm that the previous message is not the orignal messages , but has been summerized !!
        
        return [{}]


    def _should_summarize(self, message: list[dict]):
        total_tokens = self._count_token_num(message=message)
        for method, condition in self.config.trigger_condition:
            if method == "message":
                if len(message) > condition:
                    return True
            if method == "tokens":
                if total_tokens > condition:
                    return True
            if method == "fraction":
                max_input_token = self.config.token_limits
                threshold_value = int(max_input_token * condition)
                if total_tokens > threshold_value:
                    return True
        return False
            
    def _valid_cut_index(self, message: list[dict], cut_index:int) -> int:
        """
        message[:cut_index] -> got summarize.

        message[cut_index:] -> keep .

        If cut_index is a tool call obseravtion(tool call result), we need to search back
        to find the correspondiong assitant message that evoke the tool calls.
        This ensures tool call requests and responses stay together.

        Returns:
                int: The new cutoff index (i <= cut_indx).
        """
        cutoff_message = message[cut_index]
        if cutoff_message.get("role", None) == "tool":
            tool_call_id = cutoff_message.get("id", None)
            if tool_call_id is not None:
                logger.warning(f"cut off point is tool call observation, but tool call id is None")
                return cut_index
            for i in range(cut_index -1 , -1 ,-1):
                msg = message[i]
                if msg.get("role", None) == "assistant" and msg.get("tool_calls", None):
                    if msg["tool_calls"].get("id", None) == tool_call_id:
                        return i
        return cut_index

    def _find_cut_index(self, message:list[dict]):
        """
        default to use message based keep method to find the index 
        """
        keep_method, value = self.config.message_keep

        if keep_method in ["tokens", "fraction"]:
            return self._find_token_based_cutoff(message)
        else:
            if value is None:
                value = _DEFAULT_MESSAGES_TO_KEEP
            if len(message) <= value:
                return 0
            else:
                cut_off_index = len(message) - value
                return self._valid_cut_index(message, cut_off_index)

    def _find_token_based_cutoff(self, message: list[dict]) -> int:
        if not message:
            return 0
        keep_method, value = self.config.message_keep
        if keep_method == "fraction":
            token_to_keep = int(self.config.token_limits * value)
        else:
            token_to_keep = value
        if self._count_token_num(message) <= token_to_keep:
            return 0
        cut_off_message_index = len(message) -1 
        for index in range(len(message), -1, -1):
            cut_off_message_index = index
            if self._count_token_num(message[index:]) >= token_to_keep:
                break
        return self._valid_cut_index(message, cut_off_message_index)

    def _count_token_num(self, message: list[dict]):
        token_nums = litellm.token_counter(
            model=self.config.summarization_model_name,
            messages= message
        )
        return token_nums
    def _get_summarization_prompt(self,message_to_summariza:list[dict]):
        message_to_summariza_str = json.dumps(message_to_summariza)
        return Template(self.config.summarization_prompt_template, undefined=StrictUndefined).render(messages = message_to_summariza_str)

    def _generate_summary(self, message_to_summariza:list[dict]):
        prompt = self._get_summarization_prompt(message_to_summariza)
        try:
            logger.info(f"Calling LLM API with model={self.config.summarization_model_name}, {len(prompt)} messages")
            logger.info(f"Prompt messages: {prompt}") 
            response = litellm.completion(
                model=self.config.summarization_model_name,
                messages=prompt,
                api_key=self.config.model_api_key
            )
            logger.info(f"LLM API response received: {response}")
            return response
        except litellm.exceptions.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e


        