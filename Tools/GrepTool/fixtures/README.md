# Sample Project

This project demonstrates a simple greeting library.

## Features

- Says Hello to anyone
- Says Goodbye too
- Tracks greeting history
- Arithmetic utilities

## Usage

```python
from main import hello, goodbye, Greeter

print(hello("Alice"))   # Hello, Alice!
print(goodbye("Bob"))   # Goodbye, Bob!

g = Greeter(prefix="Hi")
g.greet("Charlie")
print(g.greet_count())  # 1
```

## Installation

```bash
pip install sample-project
```

## TODO

- Add async support
- Add logging
- Write more tests
