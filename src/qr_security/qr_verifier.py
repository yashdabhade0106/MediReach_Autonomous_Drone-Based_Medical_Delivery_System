# -*- coding: utf-8 -*-
"""
MediReach — QR Verification Engine.

Drone-side QR verification: decrypt → validate HMAC →
check expiry → verify order → check blacklist → confirm.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import cv2
import numpy as np

from src.qr_security.encryption import EncryptionManager
from src.qr_security.token_blacklist import TokenBlacklist
from src.utils.constants import QRConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = QRConfig()


class QRVerifier:
    """Drone-side QR verification engine."""

    def __init__(
        self,
        aes_key: Optional[bytes] = None,
        hmac_secret: Optional[str] = None,
        redis_client: Optional[object] = None,
    ) -> None:
        self.crypto = EncryptionManager(aes_key=aes_key, hmac_secret=hmac_secret)
        self.blacklist = TokenBlacklist(redis_client=redis_client)
        logger.info("QRVerifier initialised")

    def verify_from_camera(
        self, frame: np.ndarray, expected_order_id: str
    ) -> Dict[str, Any]:
        """Complete verification pipeline from camera frame."""
        qr_data = self._decode_qr_from_frame(frame)
        if qr_data is None:
            return self._fail_result("No QR code detected in frame")
        return self.verify_from_string(qr_data, expected_order_id)

    def verify_from_string(
        self, encrypted_token: str, expected_order_id: str
    ) -> Dict[str, Any]:
        """Verify token from encrypted string."""
        # 1. Decrypt
        token_data = self.crypto.decrypt_dict(encrypted_token)
        if token_data is None:
            return self._fail_result("Decryption failed — invalid or tampered token")

        # 2. Validate HMAC
        if not self._validate_hmac(token_data):
            return self._fail_result("HMAC signature validation failed")

        # 3. Check expiry
        if self._is_expired(token_data):
            return self._fail_result("Token has expired")

        # 4. Verify order ID
        if token_data.get("order_id") != expected_order_id:
            return self._fail_result(
                f"Order ID mismatch: expected {expected_order_id}, "
                f"got {token_data.get('order_id')}"
            )

        # 5. Check blacklist (replay prevention)
        token_id = token_data.get("token_id", "")
        if self.blacklist.is_blacklisted(token_id):
            return self._fail_result("Token already used (replay attack prevented)")

        # 6. All checks passed — blacklist and return success
        self.blacklist.blacklist(token_id)

        result = {
            "verified": True,
            "order_id": token_data["order_id"],
            "patient_id": token_data.get("patient_id", ""),
            "medicines": token_data.get("medicines", []),
            "failure_reason": None,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "token_id": token_id,
        }
        logger.info("QR VERIFIED: order=%s, patient=%s", result["order_id"], result["patient_id"])
        return result

    def scan_and_verify_continuous(
        self,
        camera_source: int,
        expected_order_id: str,
        timeout_seconds: int = _CFG.SCAN_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        """Continuously scan camera until QR verified or timeout."""
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            return self._fail_result("Cannot open camera")

        start = time.time()
        attempts = 0

        try:
            while time.time() - start < timeout_seconds:
                ret, frame = cap.read()
                if not ret:
                    continue

                result = self.verify_from_camera(frame, expected_order_id)
                attempts += 1

                if result["verified"]:
                    result["scan_attempts"] = attempts
                    return result

                time.sleep(_CFG.SCAN_INTERVAL_SECONDS)

        finally:
            cap.release()

        return self._fail_result(f"Scan timeout after {timeout_seconds}s ({attempts} attempts)")

    def _decode_qr_from_frame(self, frame: np.ndarray) -> Optional[str]:
        """Decode QR code from camera frame."""
        try:
            from pyzbar.pyzbar import decode
            decoded = decode(frame)
            if decoded:
                return decoded[0].data.decode("utf-8")
        except ImportError:
            # Fallback to OpenCV QR detector
            detector = cv2.QRCodeDetector()
            data, _, _ = detector.detectAndDecode(frame)
            if data:
                return data
        except Exception as exc:
            logger.debug("QR decode error: %s", exc)
        return None

    def _validate_hmac(self, token_data: Dict[str, Any]) -> bool:
        stored_sig = token_data.pop("hmac_signature", None)
        if stored_sig is None:
            return False
        computed = self.crypto.sign_dict(token_data)
        token_data["hmac_signature"] = stored_sig  # Restore
        return self.crypto.verify_signature(
            computed, computed  # Dummy — use direct comparison
        ) and stored_sig == computed

    def _is_expired(self, token_data: Dict[str, Any]) -> bool:
        expires_str = token_data.get("expires_at", "")
        try:
            expires = datetime.fromisoformat(expires_str)
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            return datetime.now(timezone.utc) > expires
        except (ValueError, TypeError):
            return True

    @staticmethod
    def _fail_result(reason: str) -> Dict[str, Any]:
        logger.warning("QR verification FAILED: %s", reason)
        return {
            "verified": False,
            "order_id": None,
            "patient_id": None,
            "failure_reason": reason,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
