import json
import time
import logging
import anthropic
from typing import Generator
from enum import Enum
from jinja2 import StrictUndefined, Template
from pydantic import BaseModel, Field

logger = logging.getLogger("myagent.model")

# Pricing per million tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}


class TextChunk:
    def __init__(self, text): self.text = text

class ThinkingChunk:
    def __init__(self, text): self.text = text

class AssistantTurn:
    """Completed assistant turn with text + tool_calls."""
    def __init__(self, text, tool_calls, in_tokens, out_tokens, stop_reason, usage):
        self.text        = text
        self.tool_calls  = tool_calls   # list of {id, name, input}
        self.in_tokens   = in_tokens
        self.out_tokens  = out_tokens
        self.stop_reason = stop_reason
        self.usage = usage
class FinishChunk:
    def __init__(self, text):
        self.text = text
class UsageChunk:
    def __init__(self, cost):
        self.cost = cost


#TODO change the model config and prompt 
class ModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict = Field(default_factory=dict)

    format_error_template: str = "{{ error }}"
    observation_template: str = (
        "<returncode>{{output.returncode}}</returncode>\n"
        "<output>\n{{output.output}}</output>"
    )


class AnthropicModelClass:
    def __init__(self, model_name: str, config: ModelConfig) -> None:
        self.model_name = model_name
        self.config = config
        self.client = anthropic.Anthropic()
        logger.info(f"AnthropicModelClass initialized with model: {model_name}")
        logger.debug(f"Model config: {config}")

    def get_template_variable(self) -> dict:
        return self.config.model_dump()

    def pre_process_prompt(self):
        pass

    # ------------------------------------------------------------------ #
    # Streaming                                                            #
    # ------------------------------------------------------------------ #

    """
    Anthropic API need merge the tool result from different tools to a single user message :
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "1", ...},
            {"type": "tool_result", "tool_use_id": "2", ...},
            {"type": "tool_result", "tool_use_id": "3", ...}
        ]
    }
    And:  Anthropic's API takes system as a top-level parameter, not as a message in the messages array.
    So not system message here ..
    """
    def prepare_anthropic_message(self, messages: list) -> list:
        result = []
        index = 0
        while index < len(messages):
            m = messages[index]
            role = m["role"]

            if role == "user":
                result.append({"role": "user", "content": m["content"]})
                index += 1

            elif role == "assistant":
                contents = []
                text = m.get("content", "")
                if text:
                    contents.append({"type": "text", "text": text})
                for tool_call in m.get("tool_calls", []):
                    contents.append({
                        "type":  "tool_use",
                        "id":    tool_call["id"],
                        "name":  tool_call["name"],
                        "input": tool_call["input"],
                    })
                result.append({"role": "assistant", "content": contents})
                index += 1

            elif role == "tool":
                tool_result_blocks = []
                while index < len(messages) and messages[index]["role"] == "tool":
                    tr = messages[index]
                    tool_result_blocks.append({
                        "type":        "tool_result",
                        "content":     tr["content"],
                        "tool_use_id": tr["tool_call_id"],
                    })
                    index += 1
                result.append({"role": "user", "content": tool_result_blocks})

            else:  # system — skip
                index += 1

        return result 

    def stream_anthropic(
            self, 
            system_message:str,
            tool_use_schemas:list,
            messages: list,
        ) -> Generator:
        """
        Stream the LLM response, yielding typed chunk dicts:

          {"type": ChunkType.TEXT,      "content": str}
          {"type": ChunkType.THINKING,  "content": str}
          {"type": ChunkType.TOOL_CALL, "index": int, "id": str,
                                        "name": str, "arguments": str}
              - First chunk per tool call has id/name populated; subsequent
                argument-fragment chunks have id="" and name="".
          {"type": ChunkType.FINISH,    "reason": str}
              - reason: "end_turn" | "tool_use" | "max_tokens" | "stop_sequence"
          {"type": ChunkType.USAGE,
                   "input_tokens": int, "output_tokens": int,
                   "cache_read_tokens": int, "cache_creation_tokens": int}
        """

        try:
            #TODO get from config and set a default value 
            max_tokens = 16000
            kwargs = {
                "model": self.model_name,
                "max_tokens":max_tokens,
                "system":system_message,
                "message": self.prepare_anthropic_message(messages),
                "tools":tool_use_schemas                   
            }
            text = ""
            
            tool_calls = []
            with self.client.messages.stream(
                **kwargs,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text += delta.text
                            yield TextChunk(delta.text)
                        elif delta.type == "thinking_delta":
                            yield ThinkingChunk(delta.thinking)
                # Usage is complete only after the stream closes
                final = stream.get_final_message()

                for block in final.content:
                    if block.type == "tool_use":
                        #TODO may be we need a generic output format for tool use wrapper ? ??
                        tool_calls.append({
                            "id" : block.id,
                            "name": block.name,
                            "input": block.input
                        })

                yield AssistantTurn(
                    text, tool_calls,
                    final.usage.input_tokens,
                    final.usage.output_tokens,
                    stop_reason=final.stop_reason,
                    usage=final.usage
                )
        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise

    def collect_stream(
            self, 
            system_message:str,
            tool_use_schemas:list,
            messages: list,
        ) -> anthropic.types.Message:
        """
        Call the API with streaming and return the complete Message object.
        Use this when you need the full response for parse_actions / parse_response
        without consuming streaming chunks manually.
        """
        kwargs = dict(self.config.model_kwargs)
        max_tokens = kwargs.pop("max_tokens", 16000)
        with self.client.messages.stream(
            model=self.model_name,
            system=system_message,
            messages=self.prepare_anthropic_message(messages),
            tools=tool_use_schemas,
            max_tokens=max_tokens,
        ) as stream:
            final = stream.get_final_message()
        logger.info(
            f"collect_stream complete: stop_reason={final.stop_reason}, "
            f"usage={final.usage}"
        )
        return final

    # ------------------------------------------------------------------ #
    # Parsing                                                              #
    # ------------------------------------------------------------------ #

    def parse_response(self, llm_response: anthropic.types.Message) -> dict:
        """
        Convert an Anthropic Message into the agent's parsed-response dict.

        Keys:
          role        - always "assistant"
          content     - text string (may be None if the turn is tool-only)
          raw_content - the original list of Anthropic content blocks;
                        use this as the "content" when appending the assistant
                        turn to chat_history so tool_use blocks are preserved.
          tool_calls  - list of {id, type, function: {name, arguments}} dicts
          cost        - float USD
          timestamp   - float unix time
        """
        text_blocks = [b for b in llm_response.content if b.type == "text"]
        raw_content = text_blocks[0].text if text_blocks else None

        tool_calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {
                    "name": b.name,
                    "arguments": json.dumps(b.input),
                },
            }
            for b in llm_response.content
            if b.type == "tool_use"
        ]

        return {
            "role": llm_response.role,
            "content": raw_content,
            # raw_content preserves tool_use blocks — the agent must append
            # {"role": "assistant", "content": parsed["raw_content"]} to
            # chat_history (not the text-only "content" string).
            "raw_content": llm_response.content,
            "tool_calls": tool_calls,
            "cost": self._calculate_cost(llm_response),
            "timestamp": time.time(),
        }


    # ------------------------------------------------------------------ #
    # Cost                                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_cost(self, response: anthropic.types.Message) -> float:
        pricing = _MODEL_PRICING.get(self.model_name)
        if not pricing:
            return 0.0
        input_price, output_price = pricing
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return (
            usage.input_tokens * input_price / 1_000_000
            + usage.output_tokens * output_price / 1_000_000
            + cache_read * input_price * 0.1 / 1_000_000
            + cache_creation * input_price * 1.25 / 1_000_000
        )
