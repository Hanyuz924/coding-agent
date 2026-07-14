from __future__ import annotations

# stdlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional
from abc import ABC, abstractmethod

# local
from Model.types import ThinkingConfig, Usage
from Tools.filestateCache import FileStateCache

if TYPE_CHECKING:
    from MCP.client import MCPManager
    from Model.anthropic_base import AnthropicModelClass

class Attachment(ABC):
    @abstractmethod
    def render(self) -> list[dict]:
        """Flatten to API content blocks (the normalizeMessagesForAPI step)."""


@dataclass(frozen=True)
class FileAttachment(Attachment):
    filename: str
    content: str
    def render(self) -> list[dict]:
        return [{"type": "text", "text": f"<file path={self.filename}>\n{self.content}\n</file>"}]



@dataclass
class Message:
    role: str
    content: str | list
    tool_calls: list = field(default_factory=list)   # assistant messages only
    tool_call_id: str = ""                            # tool result messages only
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    is_compact_boundary: bool = False

@dataclass
class SystemCompactMessage(Message):
    role: str = "system"
    content: str = "Conversation compacted"
    is_compact_boundary: bool = True
    compact_metadata: dict = field(default_factory=dict)


# Immutable per-run settings passed into the model call: which model, which
# tools, how to think, and how to format output. Frozen so it can be safely
# shared across concurrent sub-agents without accidental mutation.
@dataclass(frozen=True)
class Options:
    model: str
    tools: list[dict]
    thinking: ThinkingConfig = field(default_factory=lambda: ThinkingConfig("adaptive"))
    max_turns: int = 12


# Runtime context threaded through a single agent invocation. Combines the
# immutable Options with per-call bookkeeping (file cache, agent identity,
# nesting depth, and the parent's frozen system prompt for cache sharing).
@dataclass
class AgentRunContext:
    options: Options
    filecachestate: FileStateCache
    rendered_system_prompt: Optional[list[Message]] = None 
    api_cache_enable: bool = True
    disable_compact: bool = False  # set True for fork agents to prevent recursive self-compaction
    agent_id: Optional[str] = None
    agent_type: Optional[str] = None
    query_depth: int = 0


# Stateless description of what an agent IS: the model it uses, the tools it
# can call, observability config, and the context window size used for
# compaction decisions. Create once and reuse across many AgentState instances.
@dataclass
class AgentDefinition:
    model: AnthropicModelClass
    tool_schemas: list
    mcp_manager: MCPManager
    context_window: int
    langfuse_use: bool = False
    langfuse: Any = None
    max_api_retry: int = 3


# All mutable state that belongs to one conversation: the message history,
# token counts, file cache, and timing info. A fresh AgentState starts a new
# conversation; the same AgentDefinition can be reused with a different
# AgentState to run multiple independent sessions concurrently.
@dataclass
class AgentState:
    # This is what claude does !!!!!!!
    # Full conversation history. After compaction, the model only sees messages
    # after the compact boundary (summary + newer turns); older messages are
    # retained here for reference but not sent to the API.
    messages: list[Message] = field(default_factory=list)
    tool_id_to_turn: dict[str, dict] = field(default_factory=dict)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))  # cumulative, never reset
    context_tokens: int = 0  # current context window fill level, reset after compaction
    turn_count: int = 0
    transition_reason: str | None = None
    current_turn_number: int = 0
    filecachestate: FileStateCache = field(default_factory=FileStateCache)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    last_api_call_time: float | None = None
    current_cost: float = 0.0
    last_memory_message_id: str | None = None  # ID of last message when memory was extracted
