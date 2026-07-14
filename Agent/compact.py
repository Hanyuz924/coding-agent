from __future__ import annotations

# stdlib
import asyncio
import logging
from typing import TYPE_CHECKING, AsyncGenerator
import anthropic
# local
from Agent.compact_prompt import get_compact_prompt, get_compact_user_summary_message, get_partial_compact_prompt
from Agent.events import CompactResult
from Agent.types import AgentDefinition, AgentRunContext, Message, SystemCompactMessage
from ForkAgent.fork_agent import CacheSafeParams, ForkedAgentResult, build_cache_safe_params, run_fork_agent
from Model.anthropic_base import (
    AssistantTurn, TextChunk, ThinkingChunk,
    calculate_cost, get_max_output_tokens, make_api_system_message, normalize_message_api,
)
from Tools.registry import get_read_only_tool_schemas
if TYPE_CHECKING:
    from Agent.types import AgentState

logger = logging.getLogger("myagent.agent")

MICROCOMPACT_IDLE_S = 5 * 60
KEEP_RECENT_RESULTS = 3
SNIP_PLACEHOLDER = "[Content snipped - re-read if needed]"
SNIP_READ_PLACEHOLDER = "[File content snipped - superseded by a more recent read]"
SNIP_TURN = 1
SNIP_READ_TURN = 1
AUTO_COMPACT_THRESHOLD = 0.9
MAX_COMPACT_STREAMING_RETRIES = 2


def estimate_tokens(messages: list[Message]) -> int:
    total_chars = 0
    message_count = 0
    for message in messages:
        message_count += 1
        content = message.content
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    for v in block.values():
                        if isinstance(v, str):
                            total_chars += len(v)
        else:
            total_chars += len(content)
    content_tokens = int(total_chars / 2.8)
    framing_tokens = message_count * 4
    return int((content_tokens + framing_tokens) * 1.1)


async def check_and_compact(
    agent_def: AgentDefinition,
    state: AgentState,
    ctx: AgentRunContext,
) -> AsyncGenerator[CompactResult, None]:
    if ctx.disable_compact:
        return
    if state.context_tokens > agent_def.context_window * AUTO_COMPACT_THRESHOLD:
        logger.info(f"[compact] context at {state.context_tokens}/{agent_def.context_window} tokens, triggering compact")
        if ctx.rendered_system_prompt:
            async for event in compact_conversation(agent_state=state, agent_def=agent_def, ctx=ctx):
                if isinstance(event, CompactResult):
                    if event.success:
                        state.context_tokens = estimate_tokens(get_messages_after_compact_boundary(state.messages))
                    else:
                        logger.error(f"[compact] compaction failed via path={event.path!r}, conversation context may be stale")
                        raise RuntimeError(f"Compaction failed (path={event.path!r})")
                yield event

def _extract_summary_text(messages: list[Message]) -> str:
    """Pull the plain summary text out of the fork's assistant output message(s).

    The fork returns its last assistant message, whose content is a list of
    content blocks (or a bare string). We only want the text — it then gets
    re-wrapped as a framed *user* message (CC parity: the summary re-enters the
    conversation as user-provided context, not as the model's own turn).
    """
    parts: list[str] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            parts.append(content)
            continue
        for block in content or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
    return "\n".join(p for p in parts if p).strip()


async def compact_conversation(
    agent_state: AgentState,
    agent_def: AgentDefinition,
    ctx: AgentRunContext,
    customInstructions: str | None = None,
) -> AsyncGenerator[CompactResult | TextChunk | ThinkingChunk, None]:
    compact_prompt = get_compact_prompt(customInstructions)
    compact_message = Message(role="user", content=compact_prompt)
    #decide which part of the message need to be compact 
    message_to_compact = get_messages_after_compact_boundary(agent_state.messages)
    pre_compact_token = estimate_tokens(message_to_compact)
    cachesafeparams = build_cache_safe_params(
        model=ctx.options.model,
        system_prompt=ctx.rendered_system_prompt,  # type: ignore
        tools=ctx.options.tools,
        fork_context_messages=message_to_compact,
    )
    compactresult = CompactResult(success=False)
    try:
        async for event in run_fork_agent(
            parent_ctx=ctx,
            agent_def=agent_def,
            cachesafeparams=cachesafeparams,
            user_message=compact_message,
            agent_type="forked-sub-agent",
            disable_compact=True,
        ):
            if isinstance(event, ForkedAgentResult):
                if event.message:
                    cost = calculate_cost(ctx.options.model, event.usage)
                    agent_state.usage.accumulate(event.usage)
                    agent_state.current_cost += cost
                    logger.info(
                        f"[compact] summary usage: in={event.usage.input_tokens} "
                        f"out={event.usage.output_tokens} "
                        f"cache_read={event.usage.cache_read_input_tokens} "
                        f"cache_create={event.usage.cache_creation_input_tokens} "
                        f"cost=${cost:.6f}"
                    )
                    compactresult.success = True
                    compactresult.path = "fork"
                    compactresult.cache_read_tokens = event.usage.cache_read_input_tokens
                    compactresult.cache_creation_tokens = event.usage.cache_creation_input_tokens
                    #update the agent message here
                    compact_boundary_marker = create_compact_boundary_message(pre_compact_token=pre_compact_token)
                    # Re-wrap the model-produced summary as a framed USER message
                    # (CC parity), not the fork's raw assistant message.
                    summary_text = _extract_summary_text(event.message)
                    summary_message = Message(
                        role="user",
                        content=get_compact_user_summary_message(summary_text),
                    )
                    agent_state.messages.append(compact_boundary_marker)
                    agent_state.messages.append(summary_message)
            else:
                yield event
    except Exception:
        logger.exception("[compact] fork agent failed, falling back to direct streaming")

    if compactresult.success:
        agent_state.context_tokens = estimate_tokens(get_messages_after_compact_boundary(agent_state.messages))
        yield compactresult
        return

    # Fallback: direct streaming without cache sharing
    logger.info("[compact] falling back to direct streaming for compaction")
    sys_message = Message(role="system", content="You are a helpful AI assistant tasked with summarizing conversations.")
    api_sys_message = make_api_system_message([sys_message], api_cache_enable=False)
    api_messages = normalize_message_api(message_to_compact)
    api_messages.append({"role": "user", "content": compact_prompt})

    assistant_turn = None
    for attempt in range(MAX_COMPACT_STREAMING_RETRIES):

        assistant_turn = None
        try:
            async for stream in agent_def.model.stream_anthropic(
                system_message=api_sys_message,
                tool_use_schemas=get_read_only_tool_schemas(),
                messages=api_messages,
            ):
                if isinstance(stream, (TextChunk, ThinkingChunk)):
                    yield stream
                elif isinstance(stream, AssistantTurn):
                    assistant_turn = stream
            if assistant_turn is not None:
                break
        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            wait = 2 ** (attempt + 1)
            logger.warning(f"{type(e).__name__}, retrying in {wait}s (attempt {attempt + 1}/{MAX_COMPACT_STREAMING_RETRIES})")
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
    if not assistant_turn:
        logger.warning("[compact] fallback produced no summary text, skipping state update")
        yield CompactResult(success=False, path="fallback")
        return
    cost = calculate_cost(ctx.options.model, assistant_turn.usage)
    agent_state.usage.accumulate(assistant_turn.usage)
    agent_state.current_cost += cost
    compact_boundary_marker = create_compact_boundary_message(pre_compact_token=pre_compact_token)
    # Same as the fork path: the summary re-enters as a framed USER message.
    summary_message = Message(
        role="user",
        content=get_compact_user_summary_message(assistant_turn.text),
    )
    agent_state.messages.extend([compact_boundary_marker, summary_message])

    yield CompactResult(success=True, path="fallback")



def create_compact_boundary_message(pre_compact_token:int, trigger:str =" auto", ) -> SystemCompactMessage:
    return SystemCompactMessage(compact_metadata={
        "trigger":trigger,
        "pre_compact_tokens":pre_compact_token,
    })

def get_messages_after_compact_boundary(messages:list[Message]) -> list[Message]:
    boundaryIndex = find_last_compact_boundary_index(messages)
    if boundaryIndex == -1:
        return messages
    return messages[boundaryIndex + 1:]

def find_last_compact_boundary_index(messages:list[Message]) -> int:
    if not messages:
        return -1
    for i in range(len(messages) -1, -1, -1):
        if isinstance(messages[i],SystemCompactMessage) :
            return i
    return -1

