from __future__ import annotations

# stdlib
import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator
import asyncio

# local
from Agent.agent import create_agent_runcontext, query_loop
from Agent.types import AgentDefinition, AgentRunContext, AgentState, Message
from Model.anthropic_base import TextChunk, ThinkingChunk
from Model.types import Usage

logger = logging.getLogger("myagent.fork")

@dataclass(frozen=True)
class CacheSafeParams:
    """Snapshot of the parent's cache key inputs.

    The Anthropic prompt cache is keyed on model + tools + system + the
    messages prefix (rendered in that order). A fork sub-agent must resend
    all of these byte-identically to hit the parent's cache entry.
    """
    model: str
    system_prompt: list[Message]
    tools: list[dict[str, Any]]
    fork_context_messages: list[Message]  # parent messages used as prefix

# TODO: add run_fork_agent_background(... on_done=None) -> asyncio.Task for fire-and-forget forks
# (memory extraction, indexing, logging) so the main agent doesn't block waiting for the result.
@dataclass
class ForkedAgentResult:
    message: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))

    

def build_cache_safe_params(
    *,
    model: str,
    system_prompt: list[Message],
    tools: list[dict[str, Any]],
    fork_context_messages: list[Message],
) -> CacheSafeParams:
    return CacheSafeParams(
        model=model,
        system_prompt=system_prompt,
        tools=tools,
        fork_context_messages=fork_context_messages,
    )

def create_fork_runcontext(
    parent: AgentRunContext,
    cachesafeparams: CacheSafeParams,
    *,
    agent_type: str | None = None,
    disable_compact: bool = True,
    max_turns: int | None = None,
) -> AgentRunContext | None:
    sub_depth_budget = parent.query_depth - 1
    if sub_depth_budget < 0:
        logger.warning("fork sub-agent nesting limit reached, not spawning")
        return None

    return create_agent_runcontext(
        model_name=cachesafeparams.model,
        filecachestate=copy.deepcopy(parent.filecachestate),
        tool_schemas=cachesafeparams.tools,
        agent_id=f"sub-{uuid.uuid4().hex[:12]}",
        agent_type=agent_type,
        sub_agent_query_depth=sub_depth_budget,
        parent_system_prompt=cachesafeparams.system_prompt,
        disable_compact=disable_compact,
        max_turn=max_turns if max_turns is not None else 12,
    )


async def run_fork_agent(
    parent_ctx: AgentRunContext,
    agent_def: AgentDefinition,
    cachesafeparams: CacheSafeParams,
    user_message: Message,
    *,
    agent_type: str | None = None,
    disable_compact: bool = True,
    max_turns: int | None = None,
) -> AsyncGenerator[TextChunk | ThinkingChunk | ForkedAgentResult, None]:
    # Debug: compare cachesafeparams with parent_ctx to diagnose cache misses
    fork_sys_hash = hash(cachesafeparams.system_prompt[0].content) if cachesafeparams.system_prompt else None
    parent_sys_hash = hash(parent_ctx.rendered_system_prompt[0].content) if parent_ctx.rendered_system_prompt else None
    logger.debug(
        f"[run_fork_agent] cachesafeparams: model={cachesafeparams.model} "
        f"sys_count={len(cachesafeparams.system_prompt)} "
        f"sys_hash={fork_sys_hash} "
        f"tools_count={len(cachesafeparams.tools)} "
        f"msgs_count={len(cachesafeparams.fork_context_messages)}"
    )
    logger.debug(
        f"[run_fork_agent] parent_ctx: model={parent_ctx.options.model} "
        f"sys_count={len(parent_ctx.rendered_system_prompt) if parent_ctx.rendered_system_prompt else 0} "
        f"sys_hash={parent_sys_hash} "
        f"tools_count={len(parent_ctx.options.tools)} "
        f"sys_match={fork_sys_hash == parent_sys_hash} "
        f"model_match={cachesafeparams.model == parent_ctx.options.model} "
        f"tools_match={len(cachesafeparams.tools) == len(parent_ctx.options.tools)}"
    )
    ctx = create_fork_runcontext(
        parent_ctx, 
        cachesafeparams, 
        agent_type=agent_type, 
        disable_compact=disable_compact, 
        max_turns=max_turns
    )
    if ctx is None:
        return

    state = AgentState()
    state.messages = copy.deepcopy(cachesafeparams.fork_context_messages)  # copy so fork appends don't mutate parent's list
    state.filecachestate = copy.deepcopy(ctx.filecachestate)
    try:
        async for event in query_loop(
            agent_def=agent_def,
            state=state,
            user_message=user_message,
            system_prompt=cachesafeparams.system_prompt,
            run_ctx=ctx,
        ):
            if isinstance(event, (TextChunk, ThinkingChunk)):
                yield event
    except Exception:
        # TODO: differentiate exceptions — e.g. rate limit (retry), context limit (abort), tool error (log+continue)
        logger.exception("fork agent failed")
        return
    # TODO: find last assistant message rather than last message to handle tool-call endings
    last_assistant = next(
        (m for m in reversed(state.messages) if m.role == "assistant"), None
    )
    yield ForkedAgentResult(
        message=[last_assistant] if last_assistant else [],
        usage=state.usage,
    )


# TODO: add run_fork_agent_background(... on_done=None) -> asyncio.Task for fire-and-forget forks
# (memory extraction, indexing, logging) so the main agent doesn't block waiting for the result.

def run_fork_agent_background(
    parent_ctx: AgentRunContext,
    agent_def: AgentDefinition,
    cachesafeparams: CacheSafeParams,
    user_message: Message,
    *,
    agent_type: str | None = None,
    disable_compact: bool = True,
    max_turns: int | None = None,
    on_done=None,
) -> asyncio.Task:
    async def _run():
        logger.debug(f"[fork_bg] started agent_type={agent_type} max_turns={max_turns} disable_compact={disable_compact}")
        try:
            result = None
            async for event in run_fork_agent(
                parent_ctx=parent_ctx,
                agent_def=agent_def,
                cachesafeparams=cachesafeparams,
                user_message=user_message,
                agent_type=agent_type,
                disable_compact=disable_compact,
                max_turns=max_turns,
            ):
                if isinstance(event, ForkedAgentResult):
                    result = event
                    break
            if result is None:
                logger.warning(f"[fork_bg] agent_type={agent_type} produced no ForkedAgentResult")
            else:
                logger.debug(f"[fork_bg] done agent_type={agent_type} in={result.usage.input_tokens} out={result.usage.output_tokens} msg_count={len(result.message)}")
            if on_done is not None:
                on_done(result)
            return result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(f"[fork_bg] agent_type={agent_type} failed")
            if on_done is not None:
                on_done(None)
            return None

    task = asyncio.create_task(_run())
    logger.debug(f"[fork_bg] task created agent_type={agent_type}")
    return task
    