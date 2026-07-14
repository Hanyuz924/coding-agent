---
name: Agent architecture refactor - async/type-safe redesign
description: Major ongoing refactor moving from sync to async-first with new type-safe dataclasses
type: project
---

Large refactoring underway converting the agent system from synchronous to async-first architecture:

- Moved from class-based BaseAgent to function-based query_loop generator
- Extracted dataclasses: AgentDefinition, AgentRunContext, AgentState, Message, Options, events (ToolStart, ToolEnd, TurnDone, PermissionRequest, MaxTurnsReached, CompactResult)
- New Model/types.py with ThinkingConfig and Usage dataclasses
- New Agent/events.py and Agent/types.py modules
- Tool execution now supports parallel batching via get_tool_concurrent_safe()
- Message system changed from plain dicts to Message dataclass objects
- Added async token estimation and async tool dispatch

**Why:** Move toward cleaner async handling, type safety, and parallel tool execution support.

**How to apply:** Expect type changes in tool invocations and message handling. Review PRs for concurrent safety and async type hints. Check that Message serialization doesn't lose data during API calls.

**Message.tool_calls coordination (FIXED):**
Previously, agent.py didn't populate Message.tool_calls and normalize_message_api() didn't read it, breaking the source-of-truth contract.

Fixes implemented:
1. agent.py:321-334 — now extracts tool_calls into separate list and populates Message(role="assistant", content=..., tool_calls=tool_calls)
2. anthropic_base.py:66-90 — normalize_message_api() now iterates over m.tool_calls and appends each as content block before serializing

**Concurrent tool execution safety (FRAGILE):**
Design correctly prevents race conditions IF tools are properly marked: read-only tools (Glob, Grep, Read) have concurrent_safe=True and run in parallel; write tools (Edit, Bash, Write) have concurrent_safe=False and run sequentially. This ensures no concurrent file mutations.

**HOWEVER — Critical validation gap (code review 2026-07-14):**
- `get_tool_concurrent_safe(tool_name)` silently returns False if tool is missing from registry (registry.py:75-79), instead of raising an error
- If a tool is removed but code still references it, concurrent execution degrades to sequential with NO warning
- Risk: Mislabeled tool (marked concurrent_safe=True when should be False) will cause concurrent file mutations on write tools
- Recommendation: Add explicit validation to fail fast on missing/invalid tools

**AsyncGenerator typing (FIXED):**
Created QueryLoopEvent = Union[TextChunk, ThinkingChunk, ToolStart, ToolEnd, PermissionRequest, TurnDone, MaxTurnsReached, CompactResult] type alias and updated query_loop return type to AsyncGenerator[QueryLoopEvent, None]. Now type checkers validate event handling in callers, IDEs provide autocomplete, and API is self-documenting.

**Message type contract violation (CRITICAL - code review 2026-07-14):**
- `Message.tool_calls` field is only populated on assistant messages (agent.py:346), but there's no type enforcement
- `normalize_message_api()` iterates all messages and reads tool_calls without validating role=="assistant" (anthropic_base.py:74)
- Risk: If user/system messages accidentally get tool_calls added, they'll silently disappear during API serialization
- Root cause: Implicit contract between code paths — not enforced at type level
- Recommendation: Either add explicit validation (`assert msg.role == "assistant" if msg.tool_calls`) or refactor to separate AssistantMessage subclass

**Token estimation inaccuracy (MAJOR - code review 2026-07-14):**
- `estimate_tokens()` uses character-based heuristic (total_chars / 2.8), NOT actual token counts (compact.py:33-49)
- `state.context_tokens` is set from this heuristic (compact.py:65, agent.py:354)
- Compaction trigger depends on heuristic (compact.py:59): `if state.context_tokens > agent_def.context_window * 0.9`
- `get_context_window(model)` hardcoded to 50,000 (anthropic_base.py:54), likely a conservative workaround for heuristic variance
- Risk: If heuristic systematically over/underestimates, compaction triggers at wrong time
- Recommendation: After compaction completes, call `count_tokens_api()` for true token count instead of estimate

**Token estimation logging vs API call mismatch (CRITICAL - code review follow-up 2026-07-14):**
- agent.py:279 logs `estimate_tokens(state.messages)` using **full** state messages
- agent.py:287 sends `normalize_message_api(message_for_query)` using **filtered** messages (after compaction boundary via `get_messages_after_compact_boundary()`)
- **Symptom:** Langfuse observability logs show token count that doesn't match what was actually sent to API
- **Root cause:** Message filtering happens after token estimation logging, so estimate is stale/inaccurate
- **Fix needed:** Change line 279 to use `estimate_tokens(message_for_query)` instead of full state.messages
- **Design note:** This reveals that the message filtering strategy (line 270) is intentional—model only sees messages post-compaction boundary, not the full conversation history

**Duplicate MCPManager import (FIXED):**
Removed duplicate MCPManager import from Agent/types.py TYPE_CHECKING block (was on lines 15 and 17).

**Unnecessary inline import (FIXED):**
Removed inline import of MaxTurnsReached from agent.py (was at line 258). The import already exists at module level (line 28).

**Checkpoint feature (PLANNED):**
Implement session checkpointing to enable rollback and recovery. Design:
1. New Checkpoint dataclass: checkpoint_id, timestamp, turn_number, messages, usage, context_tokens, session_id
2. New Agent/checkpoint.py manager: save/load/list/cleanup checkpoints
3. Storage: ~/.coding-agent/checkpoints/{session_id}/{turn_number}_{timestamp}.json
4. Integration: Auto-save after each turn in query_loop, background async save via asyncio.to_thread()
5. CLI commands: /checkpoint list, /checkpoint save [name], /checkpoint load [id], /checkpoint delete [id]
6. Phases: Phase 1 (simple JSON save/load), Phase 2 (compress/incremental), Phase 3 (time-travel debugging with replay)

**Checkpoint feature implementation status:**
- Phase 1: Create Checkpoint dataclass in Agent/types.py with fields: checkpoint_id (UUID hex), timestamp, turn_number, messages, usage, context_tokens, session_id
- Phase 2: Create Agent/checkpoint.py with CheckpointManager class: async def save(), load(), list(), cleanup() for managing checkpoints on disk at ~/.coding-agent/checkpoints/{session_id}/{turn_number}_{timestamp}.json
- Phase 3 (deferred): Compression, incremental saves, time-travel debugging with replay

**Blocking issues found in latest refactor (code review 2026-07-14):**

1. **Incomplete function implementation (CRITICAL - RESOLVED):** `find_last_compact_boundary_index()` in compact.py:217-224 appeared truncated in git diff output, but actual file read (2026-07-14) showed it was complete (7 lines: while loop to find boundary index, return -1 if not found). Was a git truncation artifact, not a code issue.

2. **Missing type hints on AsyncGenerators (CRITICAL - FIXED 2026-07-14):** `check_and_compact()` (compact.py:56) and `compact_conversation()` (compact.py:98) now properly typed:
   - Line 56: `AsyncGenerator[CompactResult, None]`
   - Line 98: `AsyncGenerator[CompactResult | TextChunk | ThinkingChunk, None]`
   - Enables type checkers to validate event handling in callers and provides IDE autocomplete.

3. **Message filtering logic (MAJOR - CLARIFIED 2026-07-14):** After compaction, agent.py:270 uses `message_for_query = get_messages_after_compact_boundary(state.messages)`. This filters to only messages after the boundary marker. **Design is intentional**: model intentionally sees only post-compaction messages to avoid re-processing the entire compacted context. However, there was a **logging mismatch** discovered: token estimation (line 279) uses full state.messages while API call (line 287) uses filtered messages—this is being tracked as a separate critical issue (see "Token estimation logging vs API call mismatch" above).

4. **PEP 8 formatting violations (MINOR):** Inconsistent spacing in type hints noted but not yet addressed. Low priority given other CRITICAL issues.

Status: Code review 2026-07-14 completed. Two critical issues resolved (function completion, AsyncGenerator typing), one critical issue identified for follow-up (token estimation mismatch), one FRAGILE design area flagged (tool registry validation).
