from __future__ import annotations

# stdlib
import asyncio
import logging
import os
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

# third-party
import anthropic
from langfuse import get_client, propagate_attributes

# local — model layer
from Model import anthropic_base
from Model.anthropic_base import (
    AssistantTurn, TextChunk, ThinkingChunk,
    get_context_window, make_api_system_message, make_api_tool_message, normalize_message_api,
)
from Model.types import ThinkingConfig

# local — agent layer
# Agent.compact imports are deferred inside query_loop to break the circular import:
# compact → fork_agent → agent → compact
from Agent.events import CompactResult, MaxTurnsReached, PermissionRequest, ToolEnd, ToolStart, TurnDone
from Agent.types import AgentDefinition, AgentRunContext, AgentState, Message, Options

# local — tools / infra
from MCP.client import MCPManager
from Tools import BaseTool
from Tools.ReadTool.read_tool import get_read_tool_name
from Tools.filestateCache import FileStateCache
from Tools.permission_check import check_permission
from Tools.registry import dispatch, get_tool_concurrent_safe, get_tool_schemas
from utils.tracer import _truncate_for_langfuse

logger = logging.getLogger("myagent.agent")

NONPERSISTABLE_TOOLS = ["Read"]
MAX_TOOL_RESULT_CHARS = 50_000
TOOL_RESULT_PERSIST_THRESHOLD = 30 * 1024
PREVIEW_TOOL_RESULT_LINES = 150

# Type alias for all events yielded by query_loop
QueryLoopEvent = Union[
    TextChunk,
    ThinkingChunk,
    ToolStart,
    ToolEnd,
    PermissionRequest,
    TurnDone,
    MaxTurnsReached,
    CompactResult,
]

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def truncate_tool_result(result: str) -> str:
    if len(result) <= MAX_TOOL_RESULT_CHARS:
        return result
    head_tail_keep = (MAX_TOOL_RESULT_CHARS - 60) // 2
    return (
        result[:head_tail_keep]
        + f"\n\n[... truncated {len(result) - head_tail_keep * 2} chars ...]\n\n"
        + result[-head_tail_keep:]
    )


def persistLargeResult(tool_name: str, tool_result: str, tool_call_id: int) -> str:
    if tool_name in NONPERSISTABLE_TOOLS:
        return tool_result
    encoded = tool_result.encode()
    if len(encoded) <= TOOL_RESULT_PERSIST_THRESHOLD:
        return tool_result
    save_path = Path.home() / "coding-agent" / "tmp" / "tool_result"
    os.makedirs(save_path, exist_ok=True)
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + tool_name + "_" + str(tool_call_id)
    save_file_path = Path.joinpath(save_path, file_name)
    with open(save_file_path, mode="w") as f:
        f.write(tool_result)
    lines = tool_result.split("\n")
    preview = "\n".join(lines[:PREVIEW_TOOL_RESULT_LINES])
    return (
        f"Result too large ({len(encoded) // 1024} KB, {len(lines)} lines). "
        f"Full output saved to {save_file_path}, use {get_read_tool_name()} to read it.\n\n"
        f"Preview (first {PREVIEW_TOOL_RESULT_LINES} lines):\n{preview}"
    )




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def _observe(agent_def: AgentDefinition, **kwargs):
    if agent_def.langfuse_use and agent_def.langfuse:
        return agent_def.langfuse.start_as_current_observation(**kwargs)
    return nullcontext()


async def estimate_tokens_by_api(
    agent_def: AgentDefinition,
    system_prompt: list[Message],
    messages: list[Message],
) -> int:
    return await agent_def.model.count_tokens_api(
        system_messages=system_prompt,
        messages=messages,
        tool_use_schemas=agent_def.tool_schemas,
    )

def create_agent_runcontext(
    model_name: str,
    filecachestate: FileStateCache,
    tool_schemas: list[dict],
    agent_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    sub_agent_query_depth: int = 4,
    thinking_config: ThinkingConfig = ThinkingConfig("adaptive"),
    max_turn: int = 12,
    parent_system_prompt: Optional[list[Message]] = None,
    disable_compact: bool = False,
) -> AgentRunContext:
    options = Options(model=model_name, tools=tool_schemas, thinking=thinking_config, max_turns=max_turn)
    return AgentRunContext(
        options=options,
        filecachestate=filecachestate,
        agent_id=agent_id,
        agent_type=agent_type,
        query_depth=sub_agent_query_depth,
        rendered_system_prompt=parent_system_prompt,
        disable_compact=disable_compact,
    )

def create_agent_definition(
    model_name: str,
    api_key: str,
    langfuse_use: bool = False,
    tool_schemas: list | None = None,
) -> AgentDefinition:
    model = anthropic_base.AnthropicModelClass(model_name=model_name, api_key=api_key)
    return AgentDefinition(
        model=model,
        tool_schemas=tool_schemas if tool_schemas is not None else get_tool_schemas(),
        mcp_manager=MCPManager(),
        context_window=get_context_window(model_name),
        langfuse_use=langfuse_use,
        langfuse=get_client() if langfuse_use else None,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

async def query_loop(
    agent_def: AgentDefinition,
    state: AgentState,
    user_message: Message,
    system_prompt: list[Message],
    run_ctx:AgentRunContext
) -> AsyncGenerator[QueryLoopEvent, None]:
    """Drive one user turn through the agent loop.

    Yields: TextChunk | ThinkingChunk | ToolStart | ToolEnd |
            PermissionRequest | TurnDone | MaxTurnsReached | CompactResult
    """
    # Deferred to break circular import: compact → fork_agent → agent → compact
    from Agent.compact import check_and_compact, estimate_tokens, get_messages_after_compact_boundary
    state.messages.append(user_message)
    state.current_turn_number += 1
    state.turn_count = 0
    relevant_memory_prefetch = None
    #start the memoryprefetch here 
    if run_ctx.agent_type is None:  # only inject memories for main agent
        from Memory.MemoryPrefetch import MemoryPrefetch
        from Memory.SelectMemories import select_relevant_memories, read_mem_file, RelevantMemoriesAttachment
        from Memory.ExtractMemories import scan_memory_files, get_auto_mem_path
        mem_headers = scan_memory_files(get_auto_mem_path())
        if mem_headers:
            relevant_memory_prefetch = MemoryPrefetch(
                task=select_relevant_memories(user_query=user_message, memory_list=mem_headers)
            )


    prop_ctx = (
        propagate_attributes(session_id=state.session_id)
        if agent_def.langfuse_use
        else nullcontext()
    )
    with prop_ctx:
        with _observe(
            agent_def,
            as_type="span",
            name=f"turn_{state.current_turn_number}",
            input={"user_message": user_message},
        ) as turn_span:

            # Define _exec_one once, closing over agent_def and state
            async def _exec_one(tool_call: dict, permission: bool = True):
                """Execute a single tool call. Returns (api_dict, ToolResult | None)."""
                with _observe(
                    agent_def,
                    as_type="span",
                    name=f"execute_tool {tool_call['name']}",
                    input=_truncate_for_langfuse(tool_call["input"]),
                ) as tool_span:
                    if not permission:
                        if tool_span:
                            tool_span.update(level="WARNING", status_message="denied")
                        return (
                            {"type": "tool_result", "tool_use_id": tool_call["id"],
                             "content": "Action denied.", "is_error": True},
                            None,
                        )

                    try:
                        tool_result = await asyncio.to_thread(
                            lambda: dispatch(
                                tool_name=tool_call["name"],
                                tool_input=tool_call["input"],
                                ctx=run_ctx,
                            )
                        )
                    except Exception as e:
                        logger.error(f"Tool execution error for {tool_call['name']}: {type(e).__name__}: {e}")
                        tool_result = BaseTool.ToolResult(
                            success=False, data=None,
                            error=f"{type(e).__name__}: {str(e)}",
                            metadata={"tool_name": tool_call["name"], "error_type": type(e).__name__},
                        )
                    if tool_span:
                        tool_span.update(
                            output=_truncate_for_langfuse(tool_result.to_string()),
                            level="ERROR" if not tool_result.success else "DEFAULT",
                            status_message=tool_result.error,
                        )
                    return (
                        {"type": "tool_result", "tool_use_id": tool_call["id"],
                         "content": tool_result.to_api_content()},
                        tool_result,
                    )

            while True:
                if run_ctx.options.max_turns is not None and state.turn_count >= run_ctx.options.max_turns:
                    logger.info(f"[query_loop] max_turns={run_ctx.options.max_turns} reached, prompting user")
                    event = MaxTurnsReached(max_turns=run_ctx.options.max_turns)
                    yield event
                    if event.continue_turn:
                        state.turn_count = 0
                        logger.info("[query_loop] user chose to continue, turn_count reset")
                    else:
                        break

                assistant_turn: AssistantTurn | None = None
                async for event in check_and_compact(agent_def, state, run_ctx):
                    yield event
                
                # Include compacted summary in the query (if it exists)
                # Before: get_messages_after_compact_boundary() would skip the summary
                # Now: use all state.messages so the model sees the compacted context
                message_for_query = state.messages
                logger.debug(f"[query_loop] using {len(message_for_query)} messages for next query (after compaction)")
                last_message = state.messages[-1] if state.messages else None
                with _observe(
                    agent_def,
                    as_type="generation",
                    name=f"chat {agent_def.model.model_name}",
                    model=agent_def.model.model_name,
                    input={
                        "est_input_tokens": estimate_tokens(state.messages),
                        "turn": state.current_turn_number,
                        "last_message_role": last_message.role if last_message else None,
                        "last_message_preview": str(last_message.content)[:200] if last_message else "",
                    },
                ) as gen:
                    api_sys_message = make_api_system_message(system_prompt, run_ctx.api_cache_enable)
                    api_tool_message = make_api_tool_message(agent_def.tool_schemas, run_ctx.api_cache_enable)
                    api_messages = normalize_message_api(message_for_query)
                    sys_text = system_prompt[0].content if system_prompt else ""
                    logger.debug(
                        f"[query_loop] agent={run_ctx.agent_type or 'main'} "
                        f"sys_len={len(sys_text)} hash={hash(sys_text)} "
                        f"tools={len(agent_def.tool_schemas)} cache={run_ctx.api_cache_enable}"
                    )
                    for attempt in range(agent_def.max_api_retry + 1):
                        try:
                            async for stream in agent_def.model.stream_anthropic(
                                system_message=api_sys_message,
                                tool_use_schemas=api_tool_message,
                                messages=api_messages,
                            ):
                                if isinstance(stream, (ThinkingChunk, TextChunk)):
                                    yield stream
                                elif isinstance(stream, AssistantTurn):
                                    assistant_turn = stream
                            if assistant_turn is not None:
                                break
                        except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                            if attempt >= agent_def.max_api_retry:
                                raise
                            wait = 2 ** (attempt + 1)
                            logger.warning(f"{type(e).__name__}, retrying in {wait}s (attempt {attempt + 1}/{agent_def.max_api_retry})")
                            await asyncio.sleep(wait)
                        except Exception as e:
                            logger.error(f"Unexpected error during API call: {type(e).__name__}: {e}")
                            raise

                    if assistant_turn is not None and gen:
                        gen.update(
                            output=assistant_turn.text,
                            usage_details={
                                "input": assistant_turn.in_tokens,
                                "output": assistant_turn.out_tokens,
                            },
                            metadata={"stop_reason": assistant_turn.stop_reason},
                        )

                if assistant_turn is None:
                    if turn_span:
                        turn_span.update(level="ERROR", status_message="All retries exhausted (rate limit)")
                    break

                state.last_api_call_time = time.time()

                assistant_content = []
                if assistant_turn.text:
                    assistant_content.append({"type": "text", "text": assistant_turn.text})
                
                tool_calls = []
                for tc in assistant_turn.tool_calls:
                    tool_calls.append({
                        "id": tc["id"],
                        "name": tc["name"],
                        "input": tc["input"],
                    })
                
                state.messages.append(Message(role="assistant", content=assistant_content, tool_calls=tool_calls))
                state.turn_count += 1
                state.usage.accumulate(assistant_turn.usage)
                total_input = (
                    assistant_turn.usage.cache_read_input_tokens 
                    + assistant_turn.usage.cache_creation_input_tokens
                    + assistant_turn.usage.input_tokens
                )
                state.context_tokens = total_input
                yield TurnDone(
                    assistant_turn.in_tokens,
                    assistant_turn.out_tokens,
                    assistant_turn.stop_reason,
                    cache_read_tokens=assistant_turn.usage.cache_read_input_tokens,
                    cache_creation_tokens=assistant_turn.usage.cache_creation_input_tokens,
                )

                # TODO: handle other stop reasons — max_tokens (retry or summarize),
                # pause_turn (model explicitly pausing), stop_sequence (custom stop marker)
                if not assistant_turn.tool_calls:
                    if turn_span:
                        turn_span.update(
                            output=assistant_turn.text,
                            metadata={
                                "total_turns": state.turn_count,
                                "total_input_tokens": state.usage.input_tokens,
                                "total_output_tokens": state.usage.output_tokens,
                            },
                        )
                    if run_ctx.agent_type is None:  # only run on main agent, not forks
                        from Memory.ExtractMemories import MemExtractionState, memory_extraction
                        def _on_done(t: asyncio.Task) -> None:
                            if t.cancelled():
                                return
                            exc = t.exception()
                            if exc is not None:
                                logger.error("memory extraction failed: %s", exc, exc_info=exc)
                                return                   
                            state.last_memory_message_id = state.messages[-1].message_id   # advance ONLY on success
                        mem_task = memory_extraction(
                            messages=get_messages_after_compact_boundary(state.messages),
                            agent_def=agent_def,
                            ctx=run_ctx,
                            mem_extraction_state=MemExtractionState(
                                last_memory_message_id=state.last_memory_message_id,
                            ),
                        )
                        if mem_task is not None:
                            mem_task.add_done_callback(_on_done)
                            await asyncio.sleep(0)  # yield to event loop so task starts before query_loop returns
                    break

                # Classify and execute tool calls
                permissions_check: dict[str, bool] = {}
                parallel_batch = []
                sequential_batch = []
                for tool_call in assistant_turn.tool_calls:
                    permissions = check_permission(tool_call["name"], tool_call["input"])
                    if permissions["action"] == "deny":
                        permissions_check[tool_call["id"]] = False
                    elif permissions["action"] == "confirm":
                        perm_req = PermissionRequest(description=permissions["message"])
                        yield perm_req
                        permissions_check[tool_call["id"]] = perm_req.granted
                    else:
                        permissions_check[tool_call["id"]] = True
                    if get_tool_concurrent_safe(tool_name=tool_call["name"]):
                        parallel_batch.append(tool_call)
                    else:
                        sequential_batch.append(tool_call)

                results_ordered: dict[str, dict] = {}
                logger.debug(f"[tool_dispatch] total={len(assistant_turn.tool_calls)} parallel={len(parallel_batch)} sequential={len(sequential_batch)}")

                if parallel_batch:
                    logger.debug(f"[tool_dispatch] parallel batch: {[tc['name'] for tc in parallel_batch]}")
                    for tc in parallel_batch:
                        yield ToolStart(tc["name"], tc["input"])
                        logger.debug(f"[tool_dispatch] submitting parallel: {tc['name']} id={tc['id']}")
                    gather_results = await asyncio.gather(
                        *[_exec_one(tc, permissions_check[tc["id"]]) for tc in parallel_batch],
                        return_exceptions=True,
                    )
                    for tc, result in zip(parallel_batch, gather_results):
                        if isinstance(result, BaseException):
                            logger.error(f"[tool_dispatch] parallel error: {tc['name']} id={tc['id']} {type(result).__name__}: {result}")
                            api_dict = {"type": "tool_result", "tool_use_id": tc["id"],
                                        "content": f"{type(result).__name__}: {result}", "is_error": True}
                            tool_result = None
                        else:
                            api_dict, tool_result = result
                            logger.debug(f"[tool_dispatch] parallel done: {tc['name']} id={tc['id']} success={tool_result.success if tool_result else False}")
                        yield ToolEnd(tc["name"], tool_result, permitted=permissions_check[tc["id"]])
                        results_ordered[tc["id"]] = api_dict
                    logger.debug(f"[tool_dispatch] parallel batch complete")

                if sequential_batch:
                    logger.debug(f"[tool_dispatch] sequential batch: {[tc['name'] for tc in sequential_batch]}")
                for tc in sequential_batch:
                    logger.debug(f"[tool_dispatch] running sequential: {tc['name']} id={tc['id']}")
                    yield ToolStart(tc["name"], tc["input"])
                    api_dict, tool_result = await _exec_one(tc, permissions_check[tc["id"]])
                    logger.debug(f"[tool_dispatch] sequential done: {tc['name']} id={tc['id']} success={tool_result.success if tool_result else False}")
                    yield ToolEnd(tc["name"], tool_result, permitted=permissions_check[tc["id"]])
                    results_ordered[tc["id"]] = api_dict
                # make sure the order of tool use result is same with the order of tool call in the api response
                tool_calls_result = [
                    results_ordered[tc["id"]]
                    for tc in assistant_turn.tool_calls
                    if tc["id"] in results_ordered
                ]
                for tc in assistant_turn.tool_calls:
                    state.tool_id_to_turn[tc["id"]] = {
                        "tool_call_turn": state.current_turn_number,
                        "tool_name": tc["name"],
                    }

                # gather memory blocks IF the prefetch is ready — consume once + dedup
                memory_blocks = []
                if (relevant_memory_prefetch
                        and relevant_memory_prefetch.settled
                        and not relevant_memory_prefetch.consumed):
                    memory_blocks = relevant_memory_prefetch.take_task_result()
                # append — bundle memory WITH the tool_results (same user message)
                if tool_calls_result:
                    # tool_results MUST immediately follow the assistant tool_use → one user message.
                    # tool_results FIRST (paired), memory AFTER.
                    state.messages.append(
                        Message(role="user", content=tool_calls_result + memory_blocks)
                    )
                elif memory_blocks:
                    state.messages.append(Message(role="user", content=memory_blocks))

