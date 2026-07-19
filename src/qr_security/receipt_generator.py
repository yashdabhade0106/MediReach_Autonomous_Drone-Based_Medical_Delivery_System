# -*- coding: utf-8 -*-
"""
MediReach — e-Receipt Generator.

Creates digital delivery receipts with hash verification
for audit trail and confirmation to Team A.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.qr_security.encryption import EncryptionManager
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReceiptGenerator:
    """Generates tamper-evident digital delivery receipts."""

    def __init__(
        self, aes_key: Optional[bytes] = None, hmac_secret: Optional[str] = None
    ) -> None:
        self.crypto = EncryptionManager(aes_key=aes_key, hmac_secret=hmac_secret)
        logger.info("ReceiptGenerator initialised")

    def generate_receipt(
        self,
        mission_id: str,
        order_id: str,
        patient_id: str,
        drone_id: str,
        medicines: List[str],
        delivery_lat: float,
        delivery_long: float,
        total_distance_km: float,
        flight_time_minutes: float,
        battery_used_percent: float,
        qr_verified_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a complete delivery receipt.

        Returns:
            Receipt dictionary with hash for integrity verification.
        """
        now = datetime.now(timezone.utc)

        receipt = {
            "receipt_id": str(uuid.uuid4()),
            "mission_id": mission_id,
            "order_id": order_id,
            "patient_id": patient_id,
            "drone_id": drone_id,
            "medicines_delivered": medicines,
            "qr_verified": True,
            "qr_verified_at": qr_verified_at or now.isoformat(),
            "delivery_location": {"lat": delivery_lat, "long": delivery_long},
            "total_distance_km": round(total_distance_km, 2),
            "flight_time_minutes": round(flight_time_minutes, 1),
            "battery_used_percent": round(battery_used_percent, 1),
            "delivered_at": now.isoformat(),
            "generated_at": now.isoformat(),
        }

        # Generate integrity hash
        receipt_str = json.dumps(receipt, sort_keys=True, default=str)
        receipt["receipt_hash"] = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt["hmac_signature"] = self.crypto.sign(receipt_str)

        logger.info(
            "Receipt generated: %s for order %s",
            receipt["receipt_id"], order_id,
        )
        return receipt

    def verify_receipt(self, receipt: Dict[str, Any]) -> bool:
        """Verify receipt integrity via hash."""
        stored_hash = receipt.pop("receipt_hash", None)
        receipt.pop("hmac_signature", None)
        receipt_str = json.dumps(receipt, sort_keys=True, default=str)
        computed = hashlib.sha256(receipt_str.encode()).hexdigest()
        receipt["receipt_hash"] = stored_hash
        return stored_hash == computed

    def format_for_team_a(self, receipt: Dict[str, Any]) -> Dict[str, Any]:
        """Format receipt for Team A webhook payload."""
        return {
            "event": "delivery_confirmed",
            "order_id": receipt["order_id"],
            "mission_id": receipt["mission_id"],
            "delivered_at": receipt["delivered_at"],
            "delivery_location": receipt["delivery_location"],
            "receipt_hash": receipt.get("receipt_hash", ""),
            "flight_metrics": {
                "distance_km": receipt["total_distance_km"],
                "time_minutes": receipt["flight_time_minutes"],
                "battery_used_percent": receipt["battery_used_percent"],
            },
        }
