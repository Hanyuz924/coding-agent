"""Base class for tool implementations using class-based architecture."""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar
from dataclasses import dataclass, field
from Tools.filestateCache import FileStateCache
import logging
import json
import dataclasses


#Top level tool call logger.
#All the tool call use this logger.
logger = logging.getLogger("myagent.tools")
T = TypeVar("T")

@dataclass
class ToolUseContext:
    filecachestate: FileStateCache
    tooluse_id: str
    max_tokens_read: int = 0


@dataclass
class ToolResult(Generic[T]):
    """Result of a tool execution.

    Generic over T — the type of data returned:
      ToolResult[str]   text output (BashTool, ReadTool)
      ToolResult[dict]  structured output
      ToolResult[list]  multiple items

    Fields:
      success      Whether the tool succeeded
      data         Typed output — None on failure
      error        Error message when success=False
      metadata     Extra info (returncode, truncated, etc.)
    """
    success:  bool
    data:     Optional[T] = None
    error:    str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_string(self) -> str:
        if self.success:
            return str(self.data) if self.data is not None else ""
        return f"Error: {self.error}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data":  dataclasses.asdict(self.data) if dataclasses.is_dataclass(self.data) else self.data,
            "error": self.error,
            "metadata": self.metadata,
        }

    def to_api_content(self) -> str:
        try:
            return json.dumps(self.to_dict())
        except (TypeError, ValueError) as e:
            return json.dumps({"success": False, "data": None, "error": str(e), "metadata": {}})


class BaseTool(ABC):
    """Abstract base class for all tool implementations.
    
    Provides a common interface for tools with schema definition,
    execution logic, and metadata.
    """
    
    def __init__(self):
        """Initialize the tool."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Return the tool's input schema (for Claude API)
        This schema will send to llm , let llm know how to use tool.
        
        Should be a dict with 'type', 'properties', and 'required' keys.
        """
        pass
    
    @property
    def schema(self) -> Dict[str, Any]:
        """Return the complete tool schema (name, description, input_schema)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
    
    @property
    def read_only(self) -> bool:
        """Return True if this tool only reads data (no side effects)."""
        return False
    
    @property
    def concurrent_safe(self) -> bool:
        """Return True if this tool can be executed concurrently."""
        return True
    
    @abstractmethod
    def execute(self, context: ToolUseContext, **kwargs) -> ToolResult:
        """Execute the tool with given parameters.
        
        Args:
            context:  ToolUseContext providing FileStateCache, tooluse_id, and token limits
            **kwargs: Parameters matching the input_schema
            
        Returns:
            ToolResult: Structured result containing success status, data, error info
        """
        pass
    
    def __call__(self, context: ToolUseContext, **kwargs) -> ToolResult:
        """Allow tool instances to be called directly."""
        return self.execute(context, **kwargs)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"
