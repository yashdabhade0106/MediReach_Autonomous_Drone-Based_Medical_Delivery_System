# -*- coding: utf-8 -*-
"""
MediReach — Token Blacklist (Redis-based Replay Prevention).

Prevents replay attacks by tracking used QR tokens in Redis
with automatic TTL-based expiry.
"""

from __future__ import annotations

import time
from typing import Optional

from src.utils.constants import QRConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = QRConfig()


class TokenBlacklist:
    """Redis-based token blacklist for replay attack prevention."""

    PREFIX = "medireach:token_blacklist:"

    def __init__(self, redis_client: Optional[object] = None) -> None:
        self.redis = redis_client
        self._fallback: dict = {}  # In-memory fallback if Redis unavailable
        self._use_redis = redis_client is not None
        logger.info("TokenBlacklist init (redis=%s)", self._use_redis)

    def is_blacklisted(self, token_id: str) -> bool:
        key = f"{self.PREFIX}{token_id}"
        if self._use_redis:
            try:
                return bool(self.redis.exists(key))  # type: ignore[union-attr]
            except Exception as exc:
                logger.error("Redis check error: %s", exc)
                return token_id in self._fallback
        return token_id in self._fallback

    def blacklist(
        self, token_id: str, expiry_seconds: int = _CFG.BLACKLIST_EXPIRY_SECONDS
    ) -> None:
        key = f"{self.PREFIX}{token_id}"
        if self._use_redis:
            try:
                self.redis.setex(key, expiry_seconds, "1")  # type: ignore[union-attr]
                logger.info("Token %s blacklisted (TTL=%ds)", token_id, expiry_seconds)
                return
            except Exception as exc:
                logger.error("Redis blacklist error: %s", exc)

        self._fallback[token_id] = time.time() + expiry_seconds
        logger.info("Token %s blacklisted in-memory", token_id)

    def remove(self, token_id: str) -> None:
        key = f"{self.PREFIX}{token_id}"
        if self._use_redis:
            try:
                self.redis.delete(key)  # type: ignore[union-attr]
            except Exception:
                pass
        self._fallback.pop(token_id, None)

    def cleanup_expired(self) -> int:
        """Remove expired entries from in-memory fallback."""
        now = time.time()
        expired = [k for k, v in self._fallback.items() if v < now]
        for k in expired:
            del self._fallback[k]
        return len(expired)

    def count(self) -> int:
        if self._use_redis:
            try:
                keys = self.redis.keys(f"{self.PREFIX}*")  # type: ignore[union-attr]
                return len(keys)
            except Exception:
                pass
        return len(self._fallback)
