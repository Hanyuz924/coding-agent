"""
Tool registry — single source of truth for all available tools.

Usage:
    from Tools.registry import tool_schemas, tool_registry

    # Pass to Anthropic API
    stream_anthropic(system_message=..., tool_use_schemas=tool_schemas, messages=...)

    # Dispatch a tool call returned by the LLM
    tool = tool_registry[tool_call.name]
    result = tool.execute(**tool_call.input)
"""
from Tools.BashTool.BashTool import BashTool
from Tools.GrepTool.GrepTool import GrepTool
from Tools.GlobTool.GlobTool import GlobTool
from Tools.ReadTool.read_tool import ReadTool
from Tools.WriteTool.writefile_tool import WriteTool
from Tools.EditTool.EditTool import EditTool
from Tools.BaseTool import ToolResult, ToolUseContext




TOOLS = [
    BashTool(),
    GrepTool(),
    GlobTool(),
    ReadTool(),
    WriteTool(),
    EditTool(),
]
TOOL_REGISTRY : dict[str, object] = {t.name: t for t in TOOLS}

tool_schemas: list[dict] = [t.schema for t in TOOLS]


def get_tool_schemas():
    return tool_schemas

def dispatch(tool_name: str, tool_input: dict, ctx: ToolUseContext) -> ToolResult:
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return ToolResult(success=False, error=f"Unknown tool: {tool_name!r}")
    return tool.execute(ctx, **tool_input) # type: ignore


