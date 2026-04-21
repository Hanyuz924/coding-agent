from typing import Literal,TypedDict, Optional


"""
Current message schema for anthropic  apis 
"""
class UserMessage(TypedDict):
    role: Literal["user"]
    content: str
    images: Optional[list]

class AssistantMessage(TypedDict):
    role: Literal["assistant"]
    content: str
    tool_calls: Optional[list]

class ToolMessage(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str