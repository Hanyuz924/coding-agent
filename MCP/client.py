import json
import threading
from MCP.mcp_types import MCPServerConfig, make_request, make_notification, MCPServerState,MCPTransport,INIT_PARAMS, MCPTool
from MCP.oauth import OAuthSession
import httpx
from typing import Optional, Any, List, Dict
import logging
logger = logging.getLogger("myagent.MCP")


class HTTPClient():
    def __init__(self, config : MCPServerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._client = None
        self._oauth_lock = threading.Lock()
        self._authentication_session:OAuthSession | None = None
        self._next_id = 1
        self._oauth_token = None
        self._session_url = self._config.url

    def _build_client(self, token: str | None = None) -> httpx.Client:
        headers = {
            "Accept": "application/json, text/event-stream",
            **self._config.headers,
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return httpx.Client(
            headers=headers,
            timeout=self._config.timeout,
            follow_redirects=True,
        )

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            logger.debug(f"[{self._config.name}] building httpx client (no auth)")
            self._client = self._build_client()
        return self._client

    def _oauth(self) -> None:
        with self._oauth_lock:
            if self._authentication_session is None:
                logger.info(f"[{self._config.name}] starting OAuth session")
                extra = {k: v for k, v in self._config.headers.items()
                         if not k.lower().startswith("authorization")}
                self._authentication_session = OAuthSession(self._config.name, self._config.url, extra)
            logger.debug(f"[{self._config.name}] fetching OAuth token")
            token = self._authentication_session.get_token()
            self._oauth_token = token
            if self._client:
                self._client.close()
            self._client = self._build_client(token=token)
            logger.info(f"[{self._config.name}] OAuth token acquired, client rebuilt")

    def request(self, method: str, params: Optional[dict] = None, timeout: Optional[int] = None) -> dict:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
        msg = make_request(method, params, req_id)
        client = self._client if self._client is not None else self._get_client()
        wait_secs = timeout or self._config.timeout
        url = self._session_url
        result = None
        logger.debug(f"[{self._config.name}] → {method} id={req_id}")
        for attempt in range(2):
            res = client.post(url, json=msg, timeout=wait_secs)
            if res.status_code == 401:
                has_static_auth = any(k.lower() == "authorization" for k in self._config.headers)
                if has_static_auth:
                    raise RuntimeError(f"MCP server '{self._config.name}' rejected static auth token (401)")
                logger.info(f"[{self._config.name}] 401 on {method}, triggering OAuth (attempt {attempt+1})")
                self._oauth()
                continue
            res.raise_for_status()
            content_type = res.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                for line in res.text.splitlines():
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        if data and data != "[DONE]":
                            result = json.loads(data)
                            break
            else:
                result = res.json()
            if result is not None:
                break
        if result is None:
            raise TimeoutError(f"MCP server '{self._config.name}' timed out on '{method}'")
        if "error" in result:
            err = result["error"]
            logger.error(f"[{self._config.name}] ← {method} error {err.get('code')}: {err.get('message')}")
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
        logger.debug(f"[{self._config.name}] ← {method} ok")
        return result.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        client = self._client if self._client is not None else self._get_client()
        msg = make_notification(method, params)
        url = self._session_url or self._config.url
        logger.debug(f"[{self._config.name}] notify {method}")
        try:
            client.post(url, json=msg)
        except Exception as e:
            logger.warning(f"[{self._config.name}] notify {method} failed: {e}")

    def start(self) -> None:
        """For SSE transport: connect to the /sse endpoint and get session URL."""
        if self._config.transport == MCPTransport.SSE:
            pass
        else:
            self._session_url = self._config.url
            logger.info(f"[{self._config.name}] streamable HTTP transport, url={self._session_url}")

    def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                self._client.close()
                logger.info(f"[{self._config.name}] httpx client closed")
            except Exception as e:
                logger.warning(f"[{self._config.name}] error closing client: {e}")
            self._client = None

class MCPClient():
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.state = MCPServerState.DISCONNECTED
        self._transport: Optional[Any] = None
        self._server_info: dict = {}
        self._capabilities: dict = {}
        self._tools: List[MCPTool] = []
        self._error: str = ""

    def connect(self) -> None:
        if self.state == MCPServerState.CONNECTED:
            logger.debug(f"[{self.config.name}] already connected, skipping")
            return
        logger.info(f"[{self.config.name}] connecting (transport={self.config.transport.value})")
        self.state = MCPServerState.CONNECTING
        self._error = ""
        try:
            self._transport = self._MCP_transport_type()
            self._transport.start()
            self._handshake()
            self.state = MCPServerState.CONNECTED
            logger.info(f"[{self.config.name}] connected — server={self._server_info.get('name')} v{self._server_info.get('version')} caps={list(self._capabilities.keys())}")
        except Exception as e:
            self.state = MCPServerState.ERROR
            self._error = str(e)
            logger.error(f"[{self.config.name}] connection failed: {e}")
            raise

    def _MCP_transport_type(self):
        t = self.config.transport
        if t == MCPTransport.STDIO:
            pass
        if t in (MCPTransport.SSE, MCPTransport.HTTP):
            return HTTPClient(self.config)
        raise ValueError(f"Unsupported MCP transport: {t}")

    def _handshake(self) -> None:
        logger.debug(f"[{self.config.name}] sending initialize")
        result = self._transport.request("initialize", INIT_PARAMS, timeout=15) # type: ignore
        # e.g. {"name": "my-mcp-server", "version": "1.2.0"}
        self._server_info = result.get("serverInfo", {})
        # e.g. {"tools": {}, "resources": {}, "prompts": {}}
        # use to feature-check before calling tools/list, resources/list, etc.
        self._capabilities = result.get("capabilities", {})
        self._transport.notify("notifications/initialized") # type: ignore
        logger.debug(f"[{self.config.name}] handshake complete")

    def disconnect(self) -> None:
        logger.info(f"[{self.config.name}] disconnecting")
        if self._transport:
            self._transport.stop()
            self._transport = None
        self.state = MCPServerState.DISCONNECTED

    def reconnect(self) -> None:
        logger.info(f"[{self.config.name}] reconnecting")
        self.disconnect()
        self.connect()
    
    def _parse_mcp_tools(self, tool_info:dict) -> MCPTool:
        tool_name = tool_info.get("name", "")
        if not tool_name:
            raise RuntimeError(f"MCP server '{self.config.name}' tool name is not found ")
        mcp_tool_name = f"mcp__{self.config.name}__{tool_name}"
        # Sanitize: replace non-alphanumeric with _ for API compatibility
        mcp_tool_name = "".join(c if c.isalnum() or c == "_" else "_" for c in mcp_tool_name)
        annotations = tool_info.get("annotations", {})
        readOnlyHint = bool(annotations.get("readOnlyHint", False))
        description = tool_info.get("description", "")
        inputSchema = tool_info.get("inputSchema", {"type": "object", "properties": {}})
        if not isinstance(inputSchema, dict):
            raise RuntimeError(
                f"MCP server '{self.config.name}' tool '{tool_name}' "
                f"has invalid inputSchema type {type(inputSchema).__name__}: {inputSchema!r}"
            )
        return MCPTool(
            server_name=self.config.name,
            tool_name=tool_name,
            qualified_name= mcp_tool_name,
            description=description,
            input_schema=inputSchema,
            read_only=readOnlyHint
        )

    def server_tools(self) -> List[MCPTool]:
        if self.state != MCPServerState.CONNECTED:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")
        if self._capabilities.get("tools", None) is None:
            logger.debug(f"[{self.config.name}] server does not advertise tools capability")
            self._tools = []
            return self._tools
        result = self._transport.request("tools/list", timeout=30) # type: ignore
        self._tools = [self._parse_mcp_tools(tool_info) for tool_info in result.get("tools", [])]
        logger.info(f"[{self.config.name}] {len(self._tools)} tool(s) loaded: {[t.tool_name for t in self._tools]}")
        return self._tools

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self.state != MCPServerState.CONNECTED:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")
        logger.debug(f"[{self.config.name}] call_tool {tool_name} args={arguments}")
        params = {"name": tool_name, "arguments": arguments}
        result = self._transport.request("tools/call", params, timeout=self.config.timeout) # type: ignore

        # result looks like:
        # {"content": [{"type": "text", "text": "..."}], "isError": false}
        is_error = result.get("isError", False)
        content = result.get("content", [])
        parts = []
        for block in content:
            block_type = block.get("type", "")
            if block_type == "text":
                parts.append(block.get("text", ""))
            elif block_type == "image":
                parts.append(f"[image: {block.get('mimeType')} base64 data]")
            elif block_type == "resource":
                res = block.get("resource", {})
                parts.append(f"[resource: {res.get('uri', '')}]")
        text = "\n".join(parts) if parts else str(result)
        if is_error:
            logger.warning(f"[{self.config.name}] tool '{tool_name}' returned isError=true")
            return f"[MCP tool error]\n{text}"
        logger.debug(f"[{self.config.name}] call_tool '{tool_name}' ok ({len(text)} chars)")
        return text
    # ── Status ────────────────────────────────────────────────────────────────

    def status_line(self) -> str:
        icon = {"connected": "✓", "connecting": "…", "disconnected": "○", "error": "✗"}.get(
            self.state.value, "?"
        )
        server = self._server_info.get("name", self.config.name)
        version = self._server_info.get("version", "")
        tool_count = len(self._tools)
        line = f"{icon} {self.config.name}"
        if server and server != self.config.name:
            line += f" ({server}"
            if version:
                line += f" v{version}"
            line += ")"
        if self.state == MCPServerState.CONNECTED:
            line += f"  [{tool_count} tool(s)]"
        if self.state == MCPServerState.ERROR:
            line += f"  error: {self._error}"
        return line
        



class MCPManager:
    """Singleton that manages all configured MCP server connections."""
    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Normalize a server name to match the qualified tool name format."""
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    def add_server(self, config: MCPServerConfig) -> MCPClient:
        """Register a server. Replaces any existing client with the same name."""
        key = self._sanitize_name(config.name)
        if key in self._clients:
            logger.info(f"[MCPManager] replacing existing server '{config.name}'")
            try:
                self._clients[key].disconnect()
            except Exception:
                pass
        client = MCPClient(config)
        self._clients[key] = client
        logger.info(f"[MCPManager] registered server '{config.name}' (transport={config.transport.value})")
        return client

    def connect_server(self, name: str) -> MCPClient:
        """Connect (or reconnect) a single server by name."""
        client = self._clients.get(self._sanitize_name(name))
        if client is None:
            raise KeyError(f"MCP server '{name}' not configured")
        if client.state != MCPServerState.CONNECTED:
            client.connect()
            client.server_tools()
        else:
            logger.debug(f"[MCPManager] '{name}' already connected")
        return client

    def get_tools(self, server_name: str) -> list[MCPTool]:
        client = self._clients.get(self._sanitize_name(server_name))
        if client is None:
            raise KeyError(f"MCP server '{server_name}' not configured")
        if client.state != MCPServerState.CONNECTED:
            client.connect()
            return client.server_tools()
        else:
            logger.debug(f"[MCPManager] get_tools '{server_name}' returning {len(client._tools)} cached tool(s)")
            return client._tools

    def all_tools(self) -> List[MCPTool]:
        """Return all tools from all connected servers."""
        tools: List[MCPTool] = []
        for client in self._clients.values():
            if client.state == MCPServerState.CONNECTED:
                tools.extend(client._tools)
            else:
                try:
                    client.connect()
                    tools.extend(client.server_tools())
                except Exception as e:
                    logger.warning(f"[MCPManager] skipping '{client.config.name}' during all_tools: {e}")
        logger.debug(f"[MCPManager] all_tools returning {len(tools)} tool(s) across {len(self._clients)} server(s)")
        return tools

    def call_tool(self, qualified_name: str, arguments: dict) -> str:
        """Dispatch a tool call by qualified name (mcp__server__tool)."""
        parts = qualified_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError(f"Invalid MCP tool name: {qualified_name}")
        server_name = parts[1]
        tool_name = parts[2]

        client = self._clients.get(server_name)
        if client is None:
            raise RuntimeError(f"MCP server '{server_name}' not configured")

        if client.state == MCPServerState.DISCONNECTED:
            logger.info(f"[MCPManager] '{server_name}' disconnected, auto-reconnecting")
            client.reconnect()
            client.server_tools()

        original_name = tool_name
        for t in client._tools:
            if t.qualified_name == qualified_name:
                original_name = t.tool_name
                break
        logger.debug(f"[MCPManager] dispatching '{qualified_name}' → '{original_name}' on '{server_name}'")
        return client.call_tool(original_name, arguments)

    def list_servers(self) -> List[MCPClient]:
        return list(self._clients.values())

    def disconnect_all(self) -> None:
        logger.info(f"[MCPManager] disconnecting all {len(self._clients)} server(s)")
        for client in self._clients.values():
            try:
                client.disconnect()
            except Exception as e:
                logger.warning(f"[MCPManager] error disconnecting '{client.config.name}': {e}")

    def reload_server(self, name: str) -> None:
        logger.info(f"[MCPManager] reloading '{name}'")
        client = self._clients.get(self._sanitize_name(name))
        if client:
            client.reconnect()
            client.server_tools()
        else:
            logger.warning(f"[MCPManager] reload_server: '{name}' not found")

    