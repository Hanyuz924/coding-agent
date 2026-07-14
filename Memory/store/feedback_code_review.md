---
name: Code review approach - concurrent/async safety
description: Prioritize concurrency and type safety issues in reviews; check for race conditions and type hints
type: feedback
---

When reviewing async/concurrent code, focus on:
1. Race conditions in parallel execution (e.g., multiple tools writing to same file)
2. Type safety of async generators and union types
3. Message serialization roundtrips that might lose data

**Why:** The refactoring moved from sync to async-first design with new parallel batching. These architectural changes introduce subtle bugs that are hard to debug later.

**How to apply:** For any refactor touching async, concurrency, or message/state passing, explicitly check for: (a) concurrent access to shared mutable state, (b) explicit type hints on generators yielding multiple types, (c) data loss in format conversions.

**Examples caught and fixed:**

1. **Message normalization bug (bidirectional):** agent.py wasn't populating Message.tool_calls, and normalize_message_api() wasn't reading it. Coordinated fix across both modules: agent.py now populates tool_calls field separately (lines 321-334), and normalize_message_api() now extracts from that field and appends to content blocks (lines 66-90 in anthropic_base.py).

2. **Compaction context loss bug (FIXED):** After check_and_compact() in Agent/agent.py:267 completes and yields CompactResult, the subsequent API query (line 270) was filtering messages incorrectly. The old code used `message_for_query = get_messages_after_compact_boundary(state.messages)`, which skips BOTH the boundary marker AND the compacted summary message. Result: model never sees the conversation summary; turn ends prematurely without continuing to fix the original issue. Fix (commit 09407554a5): changed to `message_for_query = state.messages` to include the full history including the summary. The compacted context is now visible to the model on the next API call.
   - **Root cause:** State mutations (append boundary + summary in compact.py) followed by filtering function (get_messages_after_compact_boundary) that skips those new elements. 
   - **Watch for:** After mutations that append/prepend messages, verify that downstream code doesn't filter them out. Functions that slice message lists by index (like "skip to after boundary") are fragile when state structure changes.

3. **Type contract violations in message system (CRITICAL - NEW):** 
   - `Message.tool_calls` is only populated on assistant messages (agent.py:346), but `normalize_message_api()` iterates all messages looking for tool_calls without type validation (anthropic_base.py:74)
   - Risk: If code accidentally adds tool_calls to user/system messages, they'll silently disappear during serialization
   - **Watch for:** Implicit contracts between dataclass fields and code paths. Add explicit validation or enforce via types (frozen=True on Message, or separate AssistantMessage subclass)

4. **Tool registry validation missing (CRITICAL - NEW):**
   - `get_tool_concurrent_safe(tool_name)` returns False if tool doesn't exist, instead of raising/logging (registry.py:75-79)
   - If a tool is removed but still referenced, concurrent execution silently downgrades to sequential with no warning
   - **Watch for:** Sentinel returns (False, None, empty list) that mask missing/invalid state. Prefer explicit errors so misconfigurations fail fast

5. **Token estimation logging mismatch (CRITICAL - NEW, code review follow-up 2026-07-14):**
   - agent.py:279 logs `estimate_tokens(state.messages)` with full state before message filtering
   - agent.py:287 sends `normalize_message_api(message_for_query)` with filtered messages after compaction boundary
   - **Symptom:** Langfuse logs show token count that doesn't match actual API call
   - **Fix:** Line 279 should use `estimate_tokens(message_for_query)` to match what's actually sent
   - **Design insight:** Message filtering strategy (line 270) is intentional—model intentionally sees only post-compaction messages, not full history
