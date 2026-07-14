from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ThinkingConfig:
    type: Literal["adaptive", "enabled", "disabled"]
    budget_tokens: int = 8000


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_anthropic(cls, usage) -> Usage:
        return cls(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    def accumulate(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens

    def cost(self, input_price_per_m: float, output_price_per_m: float) -> float:
        return (
            self.input_tokens * input_price_per_m / 1_000_000
            + self.output_tokens * output_price_per_m / 1_000_000
            + self.cache_read_input_tokens * input_price_per_m * 0.1 / 1_000_000
            + self.cache_creation_input_tokens * input_price_per_m * 1.25 / 1_000_000
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
