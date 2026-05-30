from __future__ import annotations

import json
import os
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional
import httpx
import socket
from authlib.integrations.httpx_client import OAuth2Client
from MCP.callback_server import CallbackServer


_TOKEN_STORE = Path.home() / ".cc" / "mcp_oauth.json"


# ── Token persistence ─────────────────────────────────────────────────────────

def _load_store() -> dict:
    try:
        return json.loads(_TOKEN_STORE.read_text())
    except Exception:
        return {}


def _save_store(data: dict) -> None:
    _TOKEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    # Restrict the parent dir too — best-effort, ignored on filesystems that
    # don't honour POSIX modes (e.g. some Windows setups).
    try:
        os.chmod(_TOKEN_STORE.parent, 0o700)
    except OSError:
        pass
    # Write atomically with 0600 perms so refresh tokens aren't world-readable.
    tmp = _TOKEN_STORE.with_suffix(_TOKEN_STORE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, _TOKEN_STORE)

def _register_client(registration_endpoint: str, server_name: str, redirect_uri: str) -> str:
    """Register a new OAuth client and return the client_id."""
    import httpx
    payload = {
        "client_name": f"nano-cc-{server_name}",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    r = httpx.post(registration_endpoint, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["client_id"]

def _refresh_token(token_endpoint: str, client_id: str, refresh_token: str) -> dict:
    import httpx
    r = httpx.post(token_endpoint, data={
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def _discover(target_url: str, extra_headers: dict) -> dict:
    parsed = urllib.parse.urlparse(target_url)
    path_suffix = parsed.path.lstrip("/")
    well_known_candidates = []
    if path_suffix:
        well_known_candidates.append(
            f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource/{path_suffix}"
        )
    well_known_candidates.append(
        f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource"
    )
    target_meta_data = None
    for url in well_known_candidates:
        try:
            res = httpx.get(url=url, headers=extra_headers, timeout=10, follow_redirects=True)
            if res.status_code == 200:
                target_meta_data = res.json()
                break
        except Exception:
            continue

    if not target_meta_data:
        raise RuntimeError(f"Could not discover OAuth metadata for {target_url}")
    auth_server = target_meta_data.get("authorization_servers", [])
    if not auth_server:
        raise RuntimeError(f"No authorization_servers in OAuth metadata for {target_url}")
    as_base = auth_server[0].rstrip("/")
    as_meta_url = f"{as_base}/.well-known/oauth-authorization-server"
    r = httpx.get(as_meta_url, timeout=10, follow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch authorization server metadata from {as_meta_url}: {r.status_code}")
    return r.json()

class OAuthSession():
    def __init__(self, server_name: str, target_url: str, 
                 extra_headers : Optional[dict] = None):
        self._name = server_name
        self._target_url = target_url
        self._extra_headers = extra_headers or {}
        self._lock = threading.Lock()
        self._auth_server_meta: Optional[dict] = None   

    def get_token(self) -> str:
        with self._lock:
            store = _load_store()
            entry = store.get(self._name, {})

            # Valid non-expired token
            if entry.get("access_token") and not self._is_expired(entry):
                return entry["access_token"]

            # Try refresh first
            if entry.get("refresh_token"):
                try:
                    meta = self._auth_server_metadata()
                    tokens = _refresh_token(
                        meta["token_endpoint"],
                        entry["client_id"],
                        entry["refresh_token"],
                    )
                    entry = self._merge_tokens(entry, tokens)
                    store[self._name] = entry
                    _save_store(store)
                    return entry["access_token"]
                except Exception:
                    pass  # fall through to full re-auth

            # Full interactive auth
            entry = self._authorize(entry)
            store[self._name] = entry
            _save_store(store)
            return entry["access_token"]

    def _auth_server_metadata(self):
        if self._auth_server_meta is None:
            self._auth_server_meta = _discover(self._target_url, self._extra_headers)
        return self._auth_server_meta
    
    
    def _authorize(self, client_data: dict) -> dict:
        server_meta_data = self._auth_server_metadata()
        auth_endpoint  = server_meta_data["authorization_endpoint"]
        token_endpoint = server_meta_data["token_endpoint"]
        reg_endpoint   = server_meta_data.get("registration_endpoint", None)


        port = self._pick_port()
        redirect_uri = f"http://localhost:{port}/callback"

        client_id = client_data.get("client_id",None)
        if client_id is None and reg_endpoint is not None:
            client_id = _register_client(reg_endpoint,self._name, redirect_uri)
            client_data["client_id"] = client_id
        else:                 
            raise RuntimeError(
                f"MCP server '{self._name}' requires OAuth but has no client_id "
                "and does not support dynamic registration. "
                "Add 'oauth_client_id' to its mcp.json entry."
            )
        client = OAuth2Client(
            client_id=client_id,
            redirect_uri=redirect_uri,
            code_challenge_method="S256",
        )
        auth_url, state = client.create_authorization_url(
                auth_endpoint,
                scope=self._pick_scope(server_meta_data) or None,
        )
        callbackserver = CallbackServer(port= port, expected_state= state)
        callbackserver.start()

        #TODO add log to indicate the MCP local call back server start
        webbrowser.open(auth_url)

        code = callbackserver.wait_for_code(timeout=120)
        if not code:
            raise RuntimeError(f"OAuth timed out waiting for callback for '{self._name}'")

        token = client.fetch_token(token_endpoint, code=code)
        client_data["client_id"] = client_id
        return self._merge_tokens(client_data, token)
    
    @staticmethod
    def _is_expired(entry: dict) -> bool:
        exp = entry.get("expires_at")
        if exp is None:
            return False
        # Refresh 60 s before actual expiry
        return time.time() >= exp - 60
    
    @staticmethod
    def _merge_tokens(entry: dict, tokens: dict) -> dict:
        entry["access_token"] = tokens["access_token"]
        if "refresh_token" in tokens:
            entry["refresh_token"] = tokens["refresh_token"]
        if "expires_in" in tokens:
            entry["expires_at"] = time.time() + int(tokens["expires_in"])
        else:
            entry.pop("expires_at", None)
        return entry
    
    @staticmethod
    def _pick_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
        
    @staticmethod
    def _pick_scope(meta: dict) -> Optional[str]:
        """Choose a scope the AS advertises. Prefer 'mcp', else the first one,
        else None (which means: don't send a scope parameter at all).

        Hardcoding 'mcp' broke servers whose scopes_supported didn't include it
        (invalid_scope error from the AS).
        """
        supported = meta.get("scopes_supported") or []
        if "mcp" in supported:
            return "mcp"
        if supported:
            return supported[0]
        return None
