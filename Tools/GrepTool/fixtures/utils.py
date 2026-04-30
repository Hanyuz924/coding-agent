"""Utility functions for arithmetic and string operations."""

from typing import Union

Number = Union[int, float]


def add(a: Number, b: Number) -> Number:
    """Return the sum of a and b."""
    return a + b


def subtract(a: Number, b: Number) -> Number:
    """Return the difference of a and b."""
    return a - b


def multiply(a: Number, b: Number) -> Number:
    """Return the product of a and b."""
    return a * b


def divide(a: Number, b: Number) -> float:
    """Return a divided by b. Raises ZeroDivisionError if b is 0."""
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b


def clamp(value: Number, lo: Number, hi: Number) -> Number:
    """Clamp value to the range [lo, hi]."""
    return max(lo, min(hi, value))


def is_even(n: int) -> bool:
    return n % 2 == 0


def is_odd(n: int) -> bool:
    return not is_even(n)


def factorial(n: int) -> int:
    if n < 0:
        raise ValueError("factorial not defined for negative numbers")
    if n == 0:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
