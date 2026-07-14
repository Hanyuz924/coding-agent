from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Tools.BaseTool import ToolResult


@dataclass
class ToolStart:
    name: str
    inputs: dict


@dataclass
class ToolEnd:
    name: str
    result: ToolResult | None
    permitted: bool = True


@dataclass
class TurnDone:
    input_tokens: int
    output_tokens: int
    turn_done_reason: str
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


@dataclass
class PermissionRequest:
    description: str
    granted: bool = False


@dataclass
class MaxTurnsReached:
    max_turns: int
    continue_turn: bool = False  # CLI sets True to reset counter and keep going


@dataclass
class CompactResult:
    success: bool
    path: str = ""  # "fork" | "fallback" | ""
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    
