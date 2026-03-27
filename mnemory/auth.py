"""JWT validation helpers for Cognis service authentication.

Supports local PEM validation and remote JWKS validation for Cognis-issued
ES256 service tokens. JWT validation is optional and additive — callers can
fall back to existing API key authentication when validation fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jwt
from jwt import InvalidTokenError, PyJWKClient

logger = logging.getLogger("mnemory")

_EXPECTED_ISSUER = "cognis"
_EXPECTED_AUDIENCE = "mnemory"
_JWKS_CACHE_TTL_SECONDS = 300
_JWKS_TIMEOUT_SECONDS = 5


@dataclass(frozen=True)
class JWTAuthContext:
    """Validated Cognis JWT identity."""

    user_id: str
    agent_id: str | None = None


class CognisJWTValidator:
    """Validate Cognis-issued service JWTs for Mnemory."""

    def __init__(self, *, public_key_path: str = "", jwks_url: str = "") -> None:
        self._public_key_path = public_key_path
        self._jwks_url = jwks_url
        self._public_key: str | None = None
        self._jwks_client: PyJWKClient | None = None

        if self._public_key_path:
            with open(self._public_key_path, encoding="utf-8") as f:
                self._public_key = f.read()

    @property
    def enabled(self) -> bool:
        """Whether JWT validation is configured."""
        return bool(self._public_key_path or self._jwks_url)

    def validate(
        self, token: str, header_agent_id: str | None = None
    ) -> JWTAuthContext:
        """Validate a Cognis JWT and resolve the effective agent_id.

        Args:
            token: Bearer token value.
            header_agent_id: Optional X-Agent-Id header fallback.

        Returns:
            Validated auth context.

        Raises:
            InvalidTokenError: If the JWT is invalid.
        """
        claims = self._decode(token)
        user_id = claims.get("sub")
        if not isinstance(user_id, str) or not user_id.strip():
            raise InvalidTokenError("JWT is missing a valid sub claim")

        claim_agent_id = claims.get("agent_id")
        if claim_agent_id is not None and not isinstance(claim_agent_id, str):
            raise InvalidTokenError("JWT agent_id claim must be a string when present")

        if claim_agent_id and header_agent_id and claim_agent_id != header_agent_id:
            raise InvalidTokenError("X-Agent-Id does not match JWT agent_id claim")

        return JWTAuthContext(
            user_id=user_id, agent_id=claim_agent_id or header_agent_id
        )

    def _decode(self, token: str) -> dict:
        if self._public_key is not None:
            return self._decode_with_key(token, self._public_key)

        last_error: Exception | None = None
        for force_refresh in (False, True):
            try:
                key = (
                    self._get_jwks_client(force_refresh)
                    .get_signing_key_from_jwt(token)
                    .key
                )
                return self._decode_with_key(token, key)
            except InvalidTokenError as e:
                last_error = e
                if not force_refresh:
                    logger.info(
                        "JWT validation failed with cached JWKS key; refreshing"
                    )
                    continue
                raise
            except Exception as e:  # pragma: no cover - defensive wrapper
                last_error = e
                if not force_refresh:
                    logger.info("JWKS lookup failed; refreshing JWKS cache")
                    continue
                raise InvalidTokenError("Failed to resolve JWT signing key") from e

        raise InvalidTokenError("JWT validation failed") from last_error

    @staticmethod
    def _decode_with_key(token: str, key: object) -> dict:
        return jwt.decode(
            token,
            key,
            algorithms=["ES256"],
            issuer=_EXPECTED_ISSUER,
            audience=_EXPECTED_AUDIENCE,
        )

    def _get_jwks_client(self, force_refresh: bool = False) -> PyJWKClient:
        if force_refresh or self._jwks_client is None:
            if not self._jwks_url:
                raise InvalidTokenError("JWT validation is not configured")
            self._jwks_client = PyJWKClient(
                self._jwks_url,
                cache_jwk_set=True,
                lifespan=_JWKS_CACHE_TTL_SECONDS,
                timeout=_JWKS_TIMEOUT_SECONDS,
            )
        return self._jwks_client


_validator: CognisJWTValidator | None = None
_validator_config: tuple[str, str] | None = None


def get_jwt_validator(
    public_key_path: str = "",
    jwks_url: str = "",
) -> CognisJWTValidator | None:
    """Return a cached validator for the current JWT configuration."""
    global _validator, _validator_config

    if not public_key_path and not jwks_url:
        _validator = None
        _validator_config = None
        return None

    config_key = (public_key_path, jwks_url)
    if _validator is None or _validator_config != config_key:
        _validator = CognisJWTValidator(
            public_key_path=public_key_path,
            jwks_url=jwks_url,
        )
        _validator_config = config_key
    return _validator


def looks_like_jwt(token: str) -> bool:
    """Return True when the token has the shape of a JWT."""
    return token.count(".") == 2
