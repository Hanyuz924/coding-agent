# coding-agent TODO

## 2026-07-07

- [ ] **Microcompact: study CC's `cachedMicrocompactPath` implementation**
  - CC has a cache-aware microcompact path that avoids breaking the prompt cache when clearing old tool results
  - Learn how `cachedMicrocompactPath` works and apply the same approach here

- [ ] **Snip strategy: deliberate design required**
  - Snipping frees context window space but may cause cache misses on the next API call
  - Anthropic prompt cache is keyed on the exact message prefix — any snipped content breaks the cache hit
  - Trade-off: context savings vs. cache miss cost (cache miss = full input token cost instead of 0.1x)
  - Need to decide: only snip when context pressure is high enough that the savings outweigh the cache miss penalty, or snip lazily after a compact boundary (cache is already broken at that point anyway)


- [ ] **Implement memory / session persistence**
  - Persist `AgentState` (messages, usage, context_tokens) to disk at end of session
  - Reload and resume a previous session by session ID
  - Store compact boundary markers so resumed sessions know where history ends and model-visible context begins

- [√] **Decide position of auto compaction trigger**
  - Currently triggered after tool results are appended to messages
  - Evaluate whether compaction should also be checked before each LLM API call
  - Ensure split point always preserves at least the last plain user message
  - inside the while loop or out side the while loop

- [ ] **Add checkpoint system**
  - Create checkpoints at key points in the conversation
  - Enable rollback to previous checkpoints if needed
  - Track checkpoint history with timestamps

