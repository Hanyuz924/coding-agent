"""Main module for the sample project."""

GREETING = "Hello"
FAREWELL = "Goodbye"
VERSION = "1.0.0"


def hello(name: str = "World") -> str:
    """Return a greeting string."""
    return f"Hello, {name}!"


def goodbye(name: str = "World") -> str:
    """Return a farewell string."""
    return f"Goodbye, {name}!"


def greet_all(names: list) -> list:
    """Greet multiple people."""
    return [hello(n) for n in names]


class Greeter:
    """A stateful greeter that remembers who it has greeted."""

    def __init__(self, prefix: str = "Hello"):
        self.prefix = prefix
        self.history: list[str] = []

    def greet(self, name: str) -> str:
        msg = f"{self.prefix}, {name}!"
        self.history.append(name)
        return msg

    def greet_count(self) -> int:
        return len(self.history)


if __name__ == "__main__":
    print(hello())
    print(goodbye())
