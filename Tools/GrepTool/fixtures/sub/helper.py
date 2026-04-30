"""Helper constants and small utilities used across the project."""

HELLO = "hello"
BYE = "bye"
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

_INTERNAL_FLAG = True


def retry(func, times: int = MAX_RETRIES):
    """Call func up to `times` times, returning on first success."""
    last_err = None
    for _ in range(times):
        try:
            return func()
        except Exception as e:
            last_err = e
    raise last_err


def truncate(text: str, max_len: int = 80) -> str:
    """Truncate text to max_len characters, appending '...' if cut."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
