from __future__ import annotations

# stdlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional, cast

# third-party
import anthropic
from anthropic.types import MessageParam, ToolParam

# local
from Model.types import Usage
if TYPE_CHECKING:
    from Agent.types import Message

logger = logging.getLogger("myagent.model")

@dataclass
class ThinkingConfig:
    type: Literal["adaptive", "enabled", "disabled"]
    budget_tokens: int = 8000


# Pricing per million tokens (input, output)
_MODEL_PRICING: dict[str, tuple[float, float]] = {  # (input $/M, output $/M)
    "claude-opus-4-6":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}

MODEL_MAX_OUTPUT = {
    "claude-opus-4-7":              16_000,
    "claude-opus-4-20250514":       16_000,
    "claude-sonnet-4-6":            16_000,
    "claude-sonnet-4-20250514":     16_000,
    "claude-haiku-4-5-20251001":    16_000,
}

MODEL_CONTEXT = {
    "claude-opus-4-6": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-4-20250514": 200000,
    "claude-haiku-4-5-20251001": 200000,
    "claude-opus-4-20250514": 200000,
}
def get_max_output_tokens(model:str) -> int:
    return 16_000
    # return MODEL_MAX_OUTPUT.get(model, 16_000)

def get_context_window(model: str) -> int:
    return 50000
    #return MODEL_CONTEXT.get(model, 200000)

def calculate_cost(model_name: str, usage: Usage) -> float:
    pricing = _MODEL_PRICING.get(model_name)
    if not pricing:
        return 0.0
    input_price, output_price = pricing
    return usage.cost(input_price, output_price)



def normalize_message_api(message_list: list[Message], api_cache_enable: bool = True) -> list[dict]:
    ret = []
    for m in message_list:
        # Build content from message.content and message.tool_calls
        content = list(m.content) if isinstance(m.content, list) else (
            [{"type": "text", "text": m.content}] if m.content else []
        )
        # Append tool_calls from the dedicated field
        for tc in m.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["name"],
                "input": tc["input"],
            })
        ret.append({"role": m.role, "content": content})

    if api_cache_enable and ret:
        last = ret[-1]
        blocks = last["content"]
        if blocks:  # only add cache_control if there are content blocks
            blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
            last["content"] = blocks

    return ret


def make_api_system_message(system_messages: list[Message], api_cache_enable: bool = True) -> list:
    if api_cache_enable:
        return [{"type": "text", "text": m.content, "cache_control": {"type": "ephemeral"}} for m in system_messages]
    else:
        return [{"type": "text", "text": m.content} for m in system_messages]

def make_api_tool_message(tool_use_schema: list[dict], api_cache_enable: bool = True) -> list[dict]:
    if not api_cache_enable or not tool_use_schema:
        return tool_use_schema.copy()       
    tools = tool_use_schema.copy()
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools

@dataclass


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
        self.usage       = Usage.from_anthropic(usage)
class FinishChunk:
    def __init__(self, text):
        self.text = text
class UsageChunk:
    def __init__(self, cost):
        self.cost = cost


class AnthropicModelClass:
    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        self.model_name = model_name
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        logger.info(f"AnthropicModelClass initialized with model: {model_name}")

    # ------------------------------------------------------------------ #
    # Streaming                                                          #
    # ------------------------------------------------------------------ #

    def _make_cached_system(self, system_messages: list[Message]) -> list:
        return [{"type": "text", "text": m.content, "cache_control": {"type": "ephemeral"}} for m in system_messages]

    def _make_cached_tool_schema(self, tool_use_schema: list[dict]) -> list[dict]:
        tools = tool_use_schema.copy()
        if tools:
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        return tools

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
    
    async def side_query(
        self,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: Optional[list[dict[str, Any]]] = None,
        # json_schema dict for structured output, e.g.
        #   {"type": "json_schema", "schema": {...}}
        output_format: Optional[dict[str, Any]] = None,
        max_tokens: int = 1024,
        max_retries: int = 2,
        thinking:bool =False,
        stop_sequences: Optional[list[str]] = None,
        query_source: str = "side_query",
    ) -> Any:
        """Make a single non-agentic API call and return the raw response message."""

        system_blocks: list[dict[str, Any]] = []
        if isinstance(system, list):
            system_blocks.extend(system)
        elif system:
            system_blocks.append({"type": "text", "text": system})

        # ── assemble optional params ──────────────────────────────────────────────
        params: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system_blocks:
            params["system"] = system_blocks
        if tools:
            params["tools"] = tools
        if output_format:
            params["output_config"] = {"format": output_format}
        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        if thinking is True:
            params["thinking"] = {"type": "adaptive"}
        else:
            params["thinking"] = {"type": "disabled"}

        response = await self.client.with_options(max_retries=max_retries).messages.create(**params)

        u = response.usage
        logger.info(
            "[%s] model=%s in=%d out=%d cache_read=%d cache_create=%d",
            query_source,
            self.model_name,
            u.input_tokens,
            u.output_tokens,
            u.cache_read_input_tokens or 0,
            u.cache_creation_input_tokens or 0,
        )
        return response

        

    async def stream_anthropic(
            self,
            system_message:list[str],
            tool_use_schemas:list[dict],
            messages: list,
        ) -> AsyncGenerator:
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
            max_tokens = 32000
            sys_has_cache = any(
                isinstance(s, dict) and "cache_control" in s
                for s in (system_message or [])
            )
            tools_has_cache = any(
                isinstance(t, dict) and "cache_control" in t
                for t in (tool_use_schemas or [])
            )
            logger.debug(
                f"[stream_anthropic] sys_cache={sys_has_cache} "
                f"tools_cache={tools_has_cache} "
                f"sys_len={len(system_message or [])} "
                f"tools_len={len(tool_use_schemas or [])} "
                f"msgs_len={len(messages or [])}"
            )
            kwargs = {
                "model": self.model_name,
                "max_tokens":max_tokens,
                "system":system_message,
                "messages": messages,
                "tools":tool_use_schemas
            }
            text = ""
            
            tool_calls = []
            async with self.client.messages.stream(
                **kwargs,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            text += delta.text
                            yield TextChunk(delta.text)
                        elif delta.type == "thinking_delta":
                            yield ThinkingChunk(delta.thinking)
                # Usage is complete only after the stream closes
                final = await stream.get_final_message()

                for block in final.content:
                    if block.type == "tool_use":
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
        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise
        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise

    async def count_tokens_api(
        self,
        system_messages: list[Message],
        tool_use_schemas: list,
        messages: list[Message],
    ) -> int:
        response = await self.client.messages.count_tokens(
            model=self.model_name,
            system=self._make_cached_system(system_messages),
            messages=cast(list[MessageParam], normalize_message_api(messages)),
            tools=cast(list[ToolParam], self._make_cached_tool_schema(tool_use_schemas)),
        )
        return response.input_tokens

    async def collect_stream(
            self,
            system_messages: list[Message],
            tool_use_schemas: list,
            messages: list,
        ) -> anthropic.types.Message:
        """
        Call the API with streaming and return the complete Message object.
        """
        max_tokens = 16000
        async with self.client.messages.stream(
            model=self.model_name,
            system=self._make_cached_system(system_messages),
            messages=messages,
            tools=tool_use_schemas,
            max_tokens=max_tokens,
        ) as stream:
            final = await stream.get_final_message()
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
            "cost": calculate_cost(self.model_name, Usage.from_anthropic(llm_response.usage)),
            "timestamp": time.time(),
        }


    # ------------------------------------------------------------------ #
    # Cost                                                                 #
    # ------------------------------------------------------------------ #

