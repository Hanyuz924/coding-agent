# coding-agent TODO

## Pending

- [ ] **Fix `prepare_anthropic_message()` performance**
  - Currently re-processes the entire message list on every API call
  - Option: incremental conversion — cache prepared list + track `_prepared_until` index in `AgentState`
  - Option: store messages in Anthropic wire format directly, eliminating conversion entirely

- [ ] **Concurrent tool call execution**
  - Currently tool calls in a single turn run sequentially
  - Refactor the tool dispatch loop in `queryLoop` to run independent tool calls concurrently
  - Use `asyncio` or `concurrent.futures.ThreadPoolExecutor`
  - Must handle permission checks and result collection safely

- [ ] **Test GrepTool with all output modes**
  - `output_mode: "content"` — test with context lines (-A/-B/-C), line numbers (-n), head_limit
  - `output_mode: "files_with_matches"` — test with head_limit
  - `output_mode: "count"` — test match counts per file
  - Test case-insensitive flag (-i)
  - Test with glob pattern and type filter
