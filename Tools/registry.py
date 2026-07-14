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
# stdlib
from typing import List

# local — agent
from Agent.types import AgentRunContext

# local — MCP
from MCP.client import MCPManager
from MCP.mcp_config import load_mcp_config
from MCP.mcp_types import MCPTool

# local — tools
from Tools.BaseTool import ToolResult
from Tools.BashTool.BashTool import BashTool
from Tools.EditTool.EditTool import EditTool
from Tools.GlobTool.GlobTool import GlobTool
from Tools.GrepTool.GrepTool import GrepTool
from Tools.ReadTool.read_tool import ReadTool
from Tools.WriteTool.writefile_tool import WriteTool


MCP_TOOLS :List[MCPTool] | None = None


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



def connect_mcp_servers(mcp_manager: MCPManager):
    mcp_server_config = load_mcp_config()
    for server_name, server_config in mcp_server_config.items():
        mcp_manager.add_server(server_config)
    MCP_TOOLS = mcp_manager.all_tools()
    for mcp_tool in MCP_TOOLS:
        tool_schemas.append(mcp_tool.to_tool_schema())

def register_skill_tool() -> None:
    """Register the model-invocable SkillTool.

    Call this AFTER load_all_skills() so the tool's schema (whose description +
    skill_name enum are built from the registry) reflects the loaded skills.
    Imported lazily to avoid a circular import (SkillTool -> Skill.registry ->
    Agent.agent -> Tools.registry).
    """
    from Tools.SkillTool.SkillTool import SkillTool

    tool = SkillTool()
    TOOL_REGISTRY[tool.name] = tool
    tool_schemas.append(tool.schema)


def get_tool_concurrent_safe(tool_name: str):
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return False
    return tool.concurrent_safe # type: ignore

def get_tool_schemas():
    return tool_schemas

def get_read_only_tool_schemas() -> list[dict]:
    return [t.schema for t in TOOLS if t.read_only]

def dispatch(tool_name: str, tool_input: dict, ctx: AgentRunContext , mcp_manager:MCPManager | None = None) -> ToolResult:
    if tool_name.startswith("mcp__"):
        if mcp_manager is not None:
            result = mcp_manager.call_tool(qualified_name=tool_name, arguments=tool_input)
            return ToolResult(success=True, data=result)
        else:
            return ToolResult(success=False, error=f"MCP server is not running tool is unavaiable")
    
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return ToolResult(success=False, error=f"Unknown tool: {tool_name!r}")
    return tool.execute(ctx, **tool_input) # type: ignore


