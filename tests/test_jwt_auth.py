"""JWT authentication tests for Mnemory."""

from __future__ import annotations

import importlib
import json
import os
import threading
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _make_keypair(tmp_path: Path) -> tuple[object, str]:
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )
    public_key_path = tmp_path / "cognis-public.pem"
    public_key_path.write_text(public_pem, encoding="utf-8")
    return private_key, str(public_key_path)


def _make_jwt(
    private_key: object, *, sub: str, aud: list[str], agent_id: str | None = None
) -> str:
    payload = {
        "sub": sub,
        "aud": aud,
        "iss": "cognis",
    }
    if agent_id is not None:
        payload["agent_id"] = agent_id
    return jwt.encode(
        payload, private_key, algorithm="ES256", headers={"kid": "test-key"}
    )


class _JWKSHandler(BaseHTTPRequestHandler):
    jwks_body = b"{}"

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.jwks_body)))
        self.end_headers()
        self.wfile.write(self.jwks_body)

    def log_message(self, format, *args):  # noqa: A003
        return


class _JWKSFixture:
    def __init__(self, public_key: object) -> None:
        jwk = json.loads(ECAlgorithm.to_jwk(public_key))
        jwk["use"] = "sig"
        jwk["kid"] = "test-key"
        _JWKSHandler.jwks_body = json.dumps({"keys": [jwk]}).encode("utf-8")
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _JWKSHandler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/jwks.json"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self._thread.start()
        return self.url

    def __exit__(self, exc_type, exc, tb) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _create_client(env: dict[str, str]) -> TestClient:
    with patch.dict(os.environ, env, clear=False):
        import mnemory.auth as auth
        import mnemory.server as srv

        auth._validator = None
        auth._validator_config = None
        srv._config = None
        srv._service = None
        srv._signing_key = None
        importlib.reload(srv)

        async def _whoami(request):
            return JSONResponse(
                {
                    "user_id": srv._session_user_id.get(),
                    "agent_id": srv._session_agent_id.get(),
                    "timezone": srv._session_timezone.get(),
                    "can_switch_user": not srv._session_user_bound.get(),
                }
            )

        app = Starlette(
            routes=[Route("/api/whoami", _whoami)],
            middleware=[Middleware(srv.APIKeyMiddleware)],
        )
        return TestClient(app)


def _load_server_module(env: dict[str, str]):
    with patch.dict(os.environ, env, clear=False):
        import mnemory.auth as auth
        import mnemory.server as srv

        auth._validator = None
        auth._validator_config = None
        srv._config = None
        srv._service = None
        srv._signing_key = None
        importlib.reload(srv)
        return srv


def _paths(app: Starlette) -> set[str]:
    return {route.path for route in app.routes if hasattr(route, "path")}


class TestJWTAuth:
    def test_public_key_jwt_sets_bound_identity(self, tmp_path):
        private_key, public_key_path = _make_keypair(tmp_path)
        token = _make_jwt(
            private_key,
            sub="alice@example.com",
            aud=["mnemory"],
            agent_id="agent-1",
        )

        client = _create_client(
            {
                "LLM_API_KEY": "test-key",
                "MNEMORY_JWT_PUBLIC_KEY": public_key_path,
                "MCP_API_KEY": "",
                "MCP_API_KEYS": "",
            }
        )
        with client:
            response = client.get(
                "/api/whoami",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        assert response.json() == {
            "user_id": "alice@example.com",
            "agent_id": "agent-1",
            "timezone": None,
            "can_switch_user": False,
        }

    def test_header_agent_fallback_when_claim_missing(self, tmp_path):
        private_key, public_key_path = _make_keypair(tmp_path)
        token = _make_jwt(
            private_key,
            sub="alice@example.com",
            aud=["mnemory"],
        )

        client = _create_client(
            {
                "LLM_API_KEY": "test-key",
                "MNEMORY_JWT_PUBLIC_KEY": public_key_path,
                "MCP_API_KEY": "",
                "MCP_API_KEYS": "",
            }
        )
        with client:
            response = client.get(
                "/api/whoami",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": "agent-header",
                },
            )

        assert response.status_code == 200
        assert response.json()["agent_id"] == "agent-header"

    def test_claim_header_agent_mismatch_rejected(self, tmp_path):
        private_key, public_key_path = _make_keypair(tmp_path)
        token = _make_jwt(
            private_key,
            sub="alice@example.com",
            aud=["mnemory"],
            agent_id="agent-claim",
        )

        client = _create_client(
            {
                "LLM_API_KEY": "test-key",
                "MNEMORY_JWT_PUBLIC_KEY": public_key_path,
                "MCP_API_KEY": "",
                "MCP_API_KEYS": "",
            }
        )
        with client:
            response = client.get(
                "/api/whoami",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": "agent-header",
                },
            )

        assert response.status_code == 401

    def test_openwebui_header_does_not_override_jwt_subject(self, tmp_path):
        private_key, public_key_path = _make_keypair(tmp_path)
        token = _make_jwt(
            private_key,
            sub="alice@example.com",
            aud=["mnemory"],
        )

        client = _create_client(
            {
                "LLM_API_KEY": "test-key",
                "MNEMORY_JWT_PUBLIC_KEY": public_key_path,
                "MCP_API_KEY": "",
                "MCP_API_KEYS": "",
            }
        )
        with client:
            response = client.get(
                "/api/whoami",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-OpenWebUI-User-Email": "bob@example.com",
                },
            )

        assert response.status_code == 200
        assert response.json()["user_id"] == "alice@example.com"

    def test_wrong_audience_rejected(self, tmp_path):
        private_key, public_key_path = _make_keypair(tmp_path)
        token = _make_jwt(
            private_key,
            sub="alice@example.com",
            aud=["intaris"],
        )

        client = _create_client(
            {
                "LLM_API_KEY": "test-key",
                "MNEMORY_JWT_PUBLIC_KEY": public_key_path,
                "MCP_API_KEY": "",
                "MCP_API_KEYS": "",
            }
        )
        with client:
            response = client.get(
                "/api/whoami",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 401

    def test_invalid_jwt_falls_back_to_api_key(self, tmp_path):
        _, public_key_path = _make_keypair(tmp_path)
        fallback_token = "not.a.jwt"

        client = _create_client(
            {
                "LLM_API_KEY": "test-key",
                "MNEMORY_JWT_PUBLIC_KEY": public_key_path,
                "MCP_API_KEY": fallback_token,
                "MCP_API_KEYS": "",
            }
        )
        with client:
            response = client.get(
                "/api/whoami",
                headers={
                    "Authorization": f"Bearer {fallback_token}",
                    "X-User-Id": "fallback-user",
                },
            )

        assert response.status_code == 200
        assert response.json()["user_id"] == "fallback-user"

    def test_jwks_url_validation(self, tmp_path):
        private_key, _ = _make_keypair(tmp_path)
        token = _make_jwt(
            private_key,
            sub="alice@example.com",
            aud=["mnemory"],
            agent_id="agent-1",
        )

        with _JWKSFixture(private_key.public_key()) as jwks_url:
            client = _create_client(
                {
                    "LLM_API_KEY": "test-key",
                    "MNEMORY_JWKS_URL": jwks_url,
                    "MNEMORY_JWT_PUBLIC_KEY": "",
                    "MCP_API_KEY": "",
                    "MCP_API_KEYS": "",
                }
            )
            with client:
                response = client.get(
                    "/api/whoami",
                    headers={"Authorization": f"Bearer {token}"},
                )

        assert response.status_code == 200
        assert response.json()["user_id"] == "alice@example.com"


class TestManagementRoutes:
    def test_main_app_keeps_health_and_metrics_when_mgmt_port_set(self):
        srv = _load_server_module(
            {
                "LLM_API_KEY": "test-key",
                "MCP_API_KEY": "test-api-key",
                "MCP_API_KEYS": "",
                "MGMT_PORT": "9090",
                "ENABLE_METRICS": "true",
            }
        )

        app = srv.create_app()

        assert "/health" in _paths(app)
        assert "/metrics" in _paths(app)

    def test_mgmt_app_exposes_health_and_metrics_without_auth(self):
        srv = _load_server_module(
            {
                "LLM_API_KEY": "test-key",
                "MCP_API_KEY": "test-api-key",
                "MCP_API_KEYS": "",
                "MGMT_PORT": "9090",
                "ENABLE_METRICS": "true",
            }
        )

        mgmt_app = srv.create_mgmt_app()

        assert "/health" in _paths(mgmt_app)
        assert "/metrics" in _paths(mgmt_app)

    def test_main_app_health_requires_auth_when_mgmt_port_set(self):
        srv = _load_server_module(
            {
                "LLM_API_KEY": "test-key",
                "MCP_API_KEY": "test-api-key",
                "MCP_API_KEYS": "",
                "MGMT_PORT": "9090",
            }
        )

        @asynccontextmanager
        async def _noop_lifespan(app):
            yield

        srv.lifespan = _noop_lifespan
        client = TestClient(srv.create_app())
        with client:
            response = client.get("/health")

        assert response.status_code == 401

    def test_mgmt_app_health_skips_auth_when_mgmt_port_set(self):
        srv = _load_server_module(
            {
                "LLM_API_KEY": "test-key",
                "MCP_API_KEY": "test-api-key",
                "MCP_API_KEYS": "",
                "MGMT_PORT": "9090",
            }
        )

        client = TestClient(srv.create_mgmt_app())
        with client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
