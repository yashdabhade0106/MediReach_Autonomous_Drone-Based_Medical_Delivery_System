# -*- coding: utf-8 -*-
"""
MediReach — SQLAlchemy ORM Models.

Defines the complete database schema for drone missions,
telemetry, delivery receipts, drone inventory, and users.
Supports both SQLite (prototype) and MySQL (production).
"""

import uuid
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _generate_uuid() -> str:
    """Generate a new UUID4 string for primary keys."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


# ═══════════════════════════════════════════════════════════
#  Drone Mission Model
# ═══════════════════════════════════════════════════════════

class DroneMission(db.Model):  # type: ignore[name-defined]
    """Represents a single drone delivery mission.

    Lifecycle: PENDING → DISPATCHED → IN_FLIGHT → APPROACHING
    → LANDING → QR_PENDING → DELIVERED | FAILED | RETURNED
    """
    __tablename__ = "drone_missions"

    id = db.Column(
        db.String(36), primary_key=True, default=_generate_uuid,
    )
    order_id = db.Column(
        db.String(50), nullable=False, index=True, unique=True,
    )
    drone_id = db.Column(db.String(20), nullable=False, index=True)
    status = db.Column(
        db.String(20),
        nullable=False,
        default="PENDING",
        index=True,
    )
    # Pickup location (pharmacy / warehouse)
    pickup_lat = db.Column(db.Float, nullable=False)
    pickup_long = db.Column(db.Float, nullable=False)
    # Delivery location (patient)
    delivery_lat = db.Column(db.Float, nullable=False)
    delivery_long = db.Column(db.Float, nullable=False)
    # Optimised route from RL engine (JSON list of waypoints)
    optimized_path = db.Column(db.JSON, nullable=True)
    # Order metadata
    order_priority = db.Column(db.String(20), default="standard")
    patient_id = db.Column(db.String(50), nullable=False)
    patient_name = db.Column(db.String(200), nullable=True)
    patient_phone = db.Column(db.String(20), nullable=True)
    address_type = db.Column(db.String(20), nullable=True)
    medicines = db.Column(db.JSON, nullable=True)
    # QR verification
    qr_token = db.Column(db.Text, nullable=True)
    qr_verified = db.Column(db.Boolean, default=False)
    qr_verified_at = db.Column(db.DateTime, nullable=True)
    # Route metrics
    total_distance_km = db.Column(db.Float, nullable=True)
    estimated_time_minutes = db.Column(db.Float, nullable=True)
    route_risk_score = db.Column(db.Float, nullable=True)
    battery_usage_percent = db.Column(db.Float, nullable=True)
    # Timestamps
    dispatched_at = db.Column(db.DateTime, nullable=True)
    delivered_at = db.Column(db.DateTime, nullable=True)
    failed_at = db.Column(db.DateTime, nullable=True)
    failure_reason = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    # Relationships
    telemetry = db.relationship(
        "DroneTelemetry",
        backref="mission",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )
    receipt = db.relationship(
        "DeliveryReceipt",
        backref="mission",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<DroneMission {self.id} order={self.order_id} status={self.status}>"

    def to_dict(self) -> dict:
        """Serialise mission to dictionary."""
        return {
            "id": self.id,
            "order_id": self.order_id,
            "drone_id": self.drone_id,
            "status": self.status,
            "pickup": {"lat": self.pickup_lat, "long": self.pickup_long},
            "delivery": {"lat": self.delivery_lat, "long": self.delivery_long},
            "order_priority": self.order_priority,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "medicines": self.medicines,
            "qr_verified": self.qr_verified,
            "qr_verified_at": self.qr_verified_at.isoformat() if self.qr_verified_at else None,
            "total_distance_km": self.total_distance_km,
            "estimated_time_minutes": self.estimated_time_minutes,
            "route_risk_score": self.route_risk_score,
            "battery_usage_percent": self.battery_usage_percent,
            "optimized_path": self.optimized_path,
            "dispatched_at": self.dispatched_at.isoformat() if self.dispatched_at else None,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "failure_reason": self.failure_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# ═══════════════════════════════════════════════════════════
#  Drone Telemetry Model
# ═══════════════════════════════════════════════════════════

class DroneTelemetry(db.Model):  # type: ignore[name-defined]
    """Time-series telemetry record from a drone in flight."""
    __tablename__ = "drone_telemetry"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    mission_id = db.Column(
        db.String(36),
        db.ForeignKey("drone_missions.id"),
        nullable=False,
        index=True,
    )
    drone_id = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.Float, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    altitude = db.Column(db.Float, nullable=False)
    speed_ms = db.Column(db.Float, nullable=False)
    heading_degrees = db.Column(db.Float, nullable=False)
    battery_percent = db.Column(db.Float, nullable=False)
    signal_strength = db.Column(db.Integer, nullable=False, default=-50)
    status = db.Column(db.String(20), nullable=False)
    obstacle_near = db.Column(db.Boolean, default=False)
    eta_seconds = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return (
            f"<DroneTelemetry mission={self.mission_id} "
            f"t={self.timestamp} batt={self.battery_percent}%>"
        )

    def to_dict(self) -> dict:
        """Serialise telemetry to dictionary."""
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "drone_id": self.drone_id,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "speed_ms": self.speed_ms,
            "heading_degrees": self.heading_degrees,
            "battery_percent": self.battery_percent,
            "signal_strength": self.signal_strength,
            "status": self.status,
            "obstacle_near": self.obstacle_near,
            "eta_seconds": self.eta_seconds,
        }


# ═══════════════════════════════════════════════════════════
#  Delivery Receipt Model
# ═══════════════════════════════════════════════════════════

class DeliveryReceipt(db.Model):  # type: ignore[name-defined]
    """Digital receipt generated upon successful delivery."""
    __tablename__ = "delivery_receipts"

    id = db.Column(
        db.String(36), primary_key=True, default=_generate_uuid,
    )
    mission_id = db.Column(
        db.String(36),
        db.ForeignKey("drone_missions.id"),
        nullable=False,
        unique=True,
    )
    order_id = db.Column(db.String(50), nullable=False, index=True)
    patient_id = db.Column(db.String(50), nullable=False)
    drone_id = db.Column(db.String(20), nullable=False)
    medicines_delivered = db.Column(db.JSON, nullable=False)
    qr_verified = db.Column(db.Boolean, default=True)
    qr_verified_at = db.Column(db.DateTime, nullable=False)
    delivery_lat = db.Column(db.Float, nullable=False)
    delivery_long = db.Column(db.Float, nullable=False)
    total_distance_km = db.Column(db.Float, nullable=True)
    flight_time_minutes = db.Column(db.Float, nullable=True)
    battery_used_percent = db.Column(db.Float, nullable=True)
    receipt_hash = db.Column(db.String(64), nullable=False)
    delivered_at = db.Column(db.DateTime, nullable=False, default=_utcnow)
    created_at = db.Column(db.DateTime, default=_utcnow)

    def __repr__(self) -> str:
        return f"<DeliveryReceipt {self.id} order={self.order_id}>"

    def to_dict(self) -> dict:
        """Serialise receipt to dictionary."""
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "order_id": self.order_id,
            "patient_id": self.patient_id,
            "drone_id": self.drone_id,
            "medicines_delivered": self.medicines_delivered,
            "qr_verified": self.qr_verified,
            "qr_verified_at": self.qr_verified_at.isoformat() if self.qr_verified_at else None,
            "delivery_location": {
                "lat": self.delivery_lat,
                "long": self.delivery_long,
            },
            "total_distance_km": self.total_distance_km,
            "flight_time_minutes": self.flight_time_minutes,
            "battery_used_percent": self.battery_used_percent,
            "receipt_hash": self.receipt_hash,
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
        }


# ═══════════════════════════════════════════════════════════
#  Drone Inventory Model
# ═══════════════════════════════════════════════════════════

class DroneInventory(db.Model):  # type: ignore[name-defined]
    """Registry of all drone units and their operational state."""
    __tablename__ = "drone_inventory"

    id = db.Column(db.String(20), primary_key=True)  # e.g. DRN-001
    model = db.Column(db.String(50), default="MediReach-X1")
    status = db.Column(db.String(20), default="AVAILABLE", index=True)
    battery_percent = db.Column(db.Float, default=100.0)
    current_lat = db.Column(db.Float, nullable=True)
    current_long = db.Column(db.Float, nullable=True)
    current_altitude = db.Column(db.Float, default=0.0)
    home_lat = db.Column(db.Float, nullable=False, default=18.5204)
    home_long = db.Column(db.Float, nullable=False, default=73.8567)
    total_flights = db.Column(db.Integer, default=0)
    total_distance_km = db.Column(db.Float, default=0.0)
    last_maintenance = db.Column(db.DateTime, nullable=True)
    firmware_version = db.Column(db.String(20), default="1.0.0")
    current_mission_id = db.Column(db.String(36), nullable=True)
    created_at = db.Column(db.DateTime, default=_utcnow)
    updated_at = db.Column(db.DateTime, default=_utcnow, onupdate=_utcnow)

    def __repr__(self) -> str:
        return f"<DroneInventory {self.id} status={self.status} batt={self.battery_percent}%>"

    def to_dict(self) -> dict:
        """Serialise drone inventory to dictionary."""
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status,
            "battery_percent": self.battery_percent,
            "current_position": {
                "lat": self.current_lat,
                "long": self.current_long,
                "altitude": self.current_altitude,
            },
            "home_position": {
                "lat": self.home_lat,
                "long": self.home_long,
            },
            "total_flights": self.total_flights,
            "total_distance_km": self.total_distance_km,
            "firmware_version": self.firmware_version,
            "current_mission_id": self.current_mission_id,
            "last_maintenance": (
                self.last_maintenance.isoformat() if self.last_maintenance else None
            ),
        }


# ═══════════════════════════════════════════════════════════
#  User Model (for JWT auth)
# ═══════════════════════════════════════════════════════════

class User(db.Model):  # type: ignore[name-defined]
    """System user for API authentication."""
    __tablename__ = "users"

    id = db.Column(
        db.String(36), primary_key=True, default=_generate_uuid,
    )
    username = db.Column(
        db.String(80), unique=True, nullable=False, index=True,
    )
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), default="operator")
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=_utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<User {self.username} role={self.role}>"

    def to_dict(self) -> dict:
        """Serialise user to dictionary (excludes password)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }
