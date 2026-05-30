import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from MCP.mcp_types import MCPServerConfig, MCPTransport
from MCP.client import MCPClient

TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    raise RuntimeError("GITHUB_TOKEN not set in environment or .env file")

config = MCPServerConfig(
    name="github",
    transport=MCPTransport.HTTP,
    url="https://api.githubcopilot.com/mcp/",
    headers={"Authorization": f"Bearer {TOKEN}"},
)

client = MCPClient(config)

print("Connecting...")
client.connect()
print(f"State: {client.state}")
print(f"Server info: {client._server_info}")
print(f"Capabilities: {client._capabilities}")

print("\nFetching tools...")
tools = client.server_tools()
print(f"Found {len(tools)} tools:")
for t in tools:
    print(f"  - {t.qualified_name}: {t.description[:60]}")

print("\nCalling a tool...")
result = client.call_tool("get_me", {})
print(f"get_me result: {result}")

client.disconnect()
