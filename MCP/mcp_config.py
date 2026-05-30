from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from MCP.mcp_types import MCPServerConfig


USER_MCP_CONFIG  = Path.home() / ".cc" / "mcp.json"
PROJECT_MCP_NAME = ".mcp.json"   # looked up relative to cwd

def _load_file(path:Path) -> Dict[str, Dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("mcpServers", {})

def load_mcp_config() -> Dict[str, MCPServerConfig]:
    #user level home level
    mcp_servers: Dict[str, Dict] = _load_file(USER_MCP_CONFIG) 

    path =Path.cwd()
    project_configs = []
    for _ in range(5):
        file_name = path / PROJECT_MCP_NAME
        if file_name.exists():
            project_configs.append(_load_file(file_name))
        if file_name.parent == Path.home():
            break
        path = path.parent
    for config in reversed(project_configs):
        mcp_servers.update(config)
    return{
        name: MCPServerConfig.from_dict(name, d = config)
        for name,config in mcp_servers.items()
    }

def list_config_file() -> list[Path]:
    files = []
    if USER_MCP_CONFIG.exists():
        files.append(USER_MCP_CONFIG)
    path = Path.cwd()
    for _ in range(10):
        file_name = path / PROJECT_MCP_NAME
        if file_name.exists():
            files.append(file_name)
        if file_name.parent == Path.home():
            break
    return files


