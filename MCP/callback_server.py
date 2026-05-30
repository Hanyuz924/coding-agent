from __future__ import annotations

import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        code = params.get("code", [None])[0]
        state = params.get("state", [None])[0]
        error = params.get("error", [None])[0]

        if error:
            self.server.result = {"error": error, "error_description": params.get("error_description", [None])[0]} # type: ignore
            body = f"<h2>Authorization failed: {error}</h2><p>You can close this tab.</p>"
        elif code and state == self.server.expected_state: # pyright: ignore[reportAttributeAccessIssue]
            self.server.result = {"code": code, "state": state} # pyright: ignore[reportAttributeAccessIssue]
            body = "<h2>Authorization successful!</h2><p>You can close this tab.</p>"
        else:
            self.server.result = {"error": "state_mismatch"} # pyright: ignore[reportAttributeAccessIssue]
            body = "<h2>Invalid state — possible CSRF attack.</h2>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args) -> None:
        pass  # suppress request logs


class CallbackServer:
    """Temporary local HTTP server that catches the OAuth redirect."""

    def __init__(self, port: int, expected_state: str) -> None:
        self._port = port
        self._expected_state = expected_state
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def redirect_uri(self) -> str:
        return f"http://localhost:{self._port}/callback"

    def start(self) -> None:
        self._server = HTTPServer(("localhost", self._port), _CallbackHandler)
        self._server.expected_state = self._expected_state  # type: ignore[attr-defined]
        self._server.result = None  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def wait_for_code(self, timeout: int = 120) -> dict:
        """Block until the callback arrives or timeout (seconds). Returns result dict."""
        if not self._server:
            raise RuntimeError("Server not started")
        self._thread.join(timeout=timeout)  # type: ignore[union-attr]
        result = self._server.result  # type: ignore[attr-defined]
        if result is None:
            self._server.shutdown()
            raise TimeoutError(f"OAuth callback not received within {timeout}s")
        return result
