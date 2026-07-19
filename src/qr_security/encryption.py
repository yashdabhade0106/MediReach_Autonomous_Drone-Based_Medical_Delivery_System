# -*- coding: utf-8 -*-
"""
MediReach — AES-256 Encryption & HMAC-SHA256 Utilities.

Cryptographic functions for securing QR tokens, telemetry
payloads, and inter-service communication.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EncryptionManager:
    """AES-256 (Fernet) encryption and HMAC-SHA256 signing.

    Provides symmetric encryption for data-at-rest and
    HMAC signatures for data integrity verification.
    """

    def __init__(
        self,
        aes_key: Optional[bytes] = None,
        hmac_secret: Optional[str] = None,
    ) -> None:
        aes = aes_key or os.getenv("AES_KEY", "").encode()
        secret = hmac_secret or os.getenv("HMAC_SECRET", "medireach-default-hmac")

        if not aes or len(aes) < 10:
            aes = Fernet.generate_key()
            logger.warning("No AES_KEY set — generated ephemeral key")

        self.fernet = Fernet(aes)
        self.hmac_secret = secret.encode() if isinstance(secret, str) else secret
        logger.info("EncryptionManager initialised")

    def encrypt(self, data: str) -> str:
        return self.fernet.encrypt(data.encode()).decode()

    def decrypt(self, token: str) -> Optional[str]:
        try:
            return self.fernet.decrypt(token.encode()).decode()
        except InvalidToken:
            logger.warning("Decryption failed — invalid token")
            return None

    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        return self.encrypt(json.dumps(data, sort_keys=True, default=str))

    def decrypt_dict(self, token: str) -> Optional[Dict[str, Any]]:
        plaintext = self.decrypt(token)
        if plaintext is None:
            return None
        try:
            return json.loads(plaintext)
        except json.JSONDecodeError:
            logger.warning("Decrypted data is not valid JSON")
            return None

    def sign(self, data: str) -> str:
        return hmac.new(self.hmac_secret, data.encode(), hashlib.sha256).hexdigest()

    def verify_signature(self, data: str, signature: str) -> bool:
        expected = self.sign(data)
        return hmac.compare_digest(expected, signature)

    def sign_dict(self, data: Dict[str, Any]) -> str:
        canonical = json.dumps(data, sort_keys=True, default=str)
        return self.sign(canonical)

    def hash_data(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def generate_aes_key() -> bytes:
        return Fernet.generate_key()
