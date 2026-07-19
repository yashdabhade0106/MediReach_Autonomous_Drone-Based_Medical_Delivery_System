# -*- coding: utf-8 -*-
"""
MediReach — QR Token Generator.

Generates time-bound, HMAC-signed, AES-encrypted QR tokens
for secure payload delivery verification.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src.qr_security.encryption import EncryptionManager
from src.utils.constants import QRConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)
_CFG = QRConfig()


class QRTokenGenerator:
    """Generates tamper-proof, time-bound QR tokens.

    Token lifecycle: Created → Encrypted → QR encoded → Scanned → Verified
    """

    def __init__(
        self,
        aes_key: Optional[bytes] = None,
        hmac_secret: Optional[str] = None,
        validity_minutes: int = _CFG.TOKEN_VALIDITY_MINUTES,
    ) -> None:
        self.crypto = EncryptionManager(aes_key=aes_key, hmac_secret=hmac_secret)
        self.validity_minutes = validity_minutes
        logger.info("QRTokenGenerator init (validity=%d min)", validity_minutes)

    def generate_token(
        self,
        order_id: str,
        patient_id: str,
        medicine_list: List[str],
    ) -> Dict[str, Any]:
        """Generate a complete QR token package.

        Returns dict with token_id, qr_image_base64, token_data,
        expires_at, and hmac_signature.
        """
        now = datetime.now(timezone.utc)
        token_payload = {
            "token_id": str(uuid.uuid4()),
            "order_id": order_id,
            "patient_id": patient_id,
            "medicines": medicine_list,
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(minutes=self.validity_minutes)).isoformat(),
            "nonce": str(uuid.uuid4())[:8],
        }

        # Sign with HMAC (before adding signature to payload)
        payload_for_signing = {k: v for k, v in token_payload.items()}
        signature = self.crypto.sign_dict(payload_for_signing)
        token_payload["hmac_signature"] = signature

        # Encrypt the full payload
        encrypted = self.crypto.encrypt_dict(token_payload)

        # Generate QR image
        qr_base64 = self._create_qr_base64(encrypted)

        result = {
            "token_id": token_payload["token_id"],
            "qr_image_base64": qr_base64,
            "token_data": encrypted,
            "expires_at": token_payload["expires_at"],
            "hmac_signature": signature,
        }

        logger.info("QR token generated: id=%s, order=%s", token_payload["token_id"], order_id)
        return result

    def _create_qr_base64(self, data: str) -> str:
        """Generate QR code image as base64 string."""
        try:
            import qrcode
            from PIL import Image

            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=_CFG.QR_BOX_SIZE,
                border=_CFG.QR_BORDER,
            )
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")

            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            return base64.b64encode(buffer.getvalue()).decode()

        except ImportError:
            logger.warning("qrcode/PIL not available — returning empty base64")
            return ""

    def generate_for_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate token from a complete Team A order payload."""
        return self.generate_token(
            order_id=order_data["order_id"],
            patient_id=order_data["patient"]["id"],
            medicine_list=[m["name"] for m in order_data["medicines"]],
        )
