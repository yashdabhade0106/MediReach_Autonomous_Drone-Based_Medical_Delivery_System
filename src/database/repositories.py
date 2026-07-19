# -*- coding: utf-8 -*-
"""
MediReach — Data Access Layer (Repository Pattern).

Provides CRUD operations and query abstractions for all
database models.  Business logic should call these methods
rather than querying the ORM directly.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import desc, asc
from werkzeug.security import generate_password_hash, check_password_hash

from src.database.models import (
    db,
    DroneMission,
    DroneTelemetry,
    DeliveryReceipt,
    DroneInventory,
    User,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
#  Mission Repository
# ═══════════════════════════════════════════════════════════

class MissionRepository:
    """Data access layer for DroneMission records."""

    @staticmethod
    def create(data: Dict) -> DroneMission:
        """Create a new mission record.

        Args:
            data: Dictionary of mission fields.

        Returns:
            Newly created DroneMission instance.
        """
        mission = DroneMission(**data)
        db.session.add(mission)
        db.session.commit()
        logger.info("Created mission %s for order %s", mission.id, mission.order_id)
        return mission

    @staticmethod
    def get_by_id(mission_id: str) -> Optional[DroneMission]:
        """Retrieve a mission by its primary key."""
        return db.session.get(DroneMission, mission_id)

    @staticmethod
    def get_by_order_id(order_id: str) -> Optional[DroneMission]:
        """Retrieve a mission by its Team A order ID."""
        return DroneMission.query.filter_by(order_id=order_id).first()

    @staticmethod
    def get_active_missions() -> List[DroneMission]:
        """Return all missions that are currently in progress."""
        active_statuses = [
            "PENDING", "DISPATCHED", "IN_FLIGHT",
            "APPROACHING", "LANDING", "QR_PENDING",
        ]
        return DroneMission.query.filter(
            DroneMission.status.in_(active_statuses)
        ).order_by(desc(DroneMission.created_at)).all()

    @staticmethod
    def get_history(
        page: int = 1,
        per_page: int = 20,
        sort_by: str = "created_at",
        sort_order: str = "desc",
        status_filter: Optional[str] = None,
        drone_id_filter: Optional[str] = None,
    ) -> Tuple[List[DroneMission], int]:
        """Paginated mission history with filters.

        Args:
            page: Page number (1-indexed).
            per_page: Items per page.
            sort_by: Column name to sort by.
            sort_order: 'asc' or 'desc'.
            status_filter: Optional status to filter by.
            drone_id_filter: Optional drone ID to filter by.

        Returns:
            Tuple of (list of missions, total count).
        """
        query = DroneMission.query

        if status_filter:
            query = query.filter_by(status=status_filter)
        if drone_id_filter:
            query = query.filter_by(drone_id=drone_id_filter)

        sort_col = getattr(DroneMission, sort_by, DroneMission.created_at)
        order_fn = desc if sort_order == "desc" else asc
        query = query.order_by(order_fn(sort_col))

        total = query.count()
        missions = query.offset((page - 1) * per_page).limit(per_page).all()
        return missions, total

    @staticmethod
    def update_status(
        mission_id: str,
        status: str,
        extra_fields: Optional[Dict] = None,
    ) -> Optional[DroneMission]:
        """Update mission status and optional extra fields.

        Args:
            mission_id: Mission primary key.
            status: New status string.
            extra_fields: Additional column updates.

        Returns:
            Updated mission or None if not found.
        """
        mission = db.session.get(DroneMission, mission_id)
        if mission is None:
            logger.warning("Mission %s not found for status update", mission_id)
            return None

        mission.status = status
        mission.updated_at = datetime.now(timezone.utc)

        if extra_fields:
            for key, value in extra_fields.items():
                if hasattr(mission, key):
                    setattr(mission, key, value)

        db.session.commit()
        logger.info("Mission %s status → %s", mission_id, status)
        return mission

    @staticmethod
    def mark_delivered(
        mission_id: str,
        qr_verified_at: datetime,
    ) -> Optional[DroneMission]:
        """Mark a mission as successfully delivered."""
        return MissionRepository.update_status(
            mission_id,
            "DELIVERED",
            extra_fields={
                "qr_verified": True,
                "qr_verified_at": qr_verified_at,
                "delivered_at": datetime.now(timezone.utc),
            },
        )

    @staticmethod
    def mark_failed(
        mission_id: str,
        reason: str,
    ) -> Optional[DroneMission]:
        """Mark a mission as failed with a reason."""
        return MissionRepository.update_status(
            mission_id,
            "FAILED",
            extra_fields={
                "failed_at": datetime.now(timezone.utc),
                "failure_reason": reason,
            },
        )


# ═══════════════════════════════════════════════════════════
#  Telemetry Repository
# ═══════════════════════════════════════════════════════════

class TelemetryRepository:
    """Data access layer for DroneTelemetry records."""

    @staticmethod
    def create(data: Dict) -> DroneTelemetry:
        """Store a telemetry data point."""
        telemetry = DroneTelemetry(**data)
        db.session.add(telemetry)
        db.session.commit()
        return telemetry

    @staticmethod
    def bulk_create(records: List[Dict]) -> int:
        """Bulk-insert telemetry records for efficiency.

        Args:
            records: List of telemetry dictionaries.

        Returns:
            Number of records inserted.
        """
        objects = [DroneTelemetry(**r) for r in records]
        db.session.bulk_save_objects(objects)
        db.session.commit()
        return len(objects)

    @staticmethod
    def get_by_mission(
        mission_id: str,
        limit: Optional[int] = None,
    ) -> List[DroneTelemetry]:
        """Get telemetry history for a mission, ordered by timestamp."""
        query = DroneTelemetry.query.filter_by(
            mission_id=mission_id
        ).order_by(asc(DroneTelemetry.timestamp))

        if limit:
            query = query.limit(limit)
        return query.all()

    @staticmethod
    def get_latest(mission_id: str) -> Optional[DroneTelemetry]:
        """Get most recent telemetry for a mission."""
        return DroneTelemetry.query.filter_by(
            mission_id=mission_id
        ).order_by(desc(DroneTelemetry.timestamp)).first()


# ═══════════════════════════════════════════════════════════
#  Receipt Repository
# ═══════════════════════════════════════════════════════════

class ReceiptRepository:
    """Data access layer for DeliveryReceipt records."""

    @staticmethod
    def create(data: Dict) -> DeliveryReceipt:
        """Create a delivery receipt."""
        receipt = DeliveryReceipt(**data)
        db.session.add(receipt)
        db.session.commit()
        logger.info("Created receipt %s for order %s", receipt.id, receipt.order_id)
        return receipt

    @staticmethod
    def get_by_mission(mission_id: str) -> Optional[DeliveryReceipt]:
        """Get receipt for a specific mission."""
        return DeliveryReceipt.query.filter_by(mission_id=mission_id).first()

    @staticmethod
    def get_by_order(order_id: str) -> Optional[DeliveryReceipt]:
        """Get receipt by Team A order ID."""
        return DeliveryReceipt.query.filter_by(order_id=order_id).first()


# ═══════════════════════════════════════════════════════════
#  Drone Inventory Repository
# ═══════════════════════════════════════════════════════════

class DroneRepository:
    """Data access layer for DroneInventory records."""

    @staticmethod
    def create(data: Dict) -> DroneInventory:
        """Register a new drone."""
        drone = DroneInventory(**data)
        db.session.add(drone)
        db.session.commit()
        logger.info("Registered drone %s", drone.id)
        return drone

    @staticmethod
    def get_by_id(drone_id: str) -> Optional[DroneInventory]:
        """Get drone by its ID."""
        return db.session.get(DroneInventory, drone_id)

    @staticmethod
    def get_available() -> List[DroneInventory]:
        """Get all drones with AVAILABLE status."""
        return DroneInventory.query.filter_by(
            status="AVAILABLE"
        ).order_by(desc(DroneInventory.battery_percent)).all()

    @staticmethod
    def get_best_available() -> Optional[DroneInventory]:
        """Get the best available drone (highest battery)."""
        return DroneInventory.query.filter_by(
            status="AVAILABLE"
        ).order_by(desc(DroneInventory.battery_percent)).first()

    @staticmethod
    def update_status(
        drone_id: str,
        status: str,
        mission_id: Optional[str] = None,
    ) -> Optional[DroneInventory]:
        """Update drone status and optionally assign a mission."""
        drone = db.session.get(DroneInventory, drone_id)
        if drone is None:
            return None

        drone.status = status
        drone.current_mission_id = mission_id
        drone.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return drone

    @staticmethod
    def update_position(
        drone_id: str,
        lat: float,
        lon: float,
        altitude: float,
        battery: float,
    ) -> Optional[DroneInventory]:
        """Update drone telemetry position."""
        drone = db.session.get(DroneInventory, drone_id)
        if drone is None:
            return None

        drone.current_lat = lat
        drone.current_long = lon
        drone.current_altitude = altitude
        drone.battery_percent = battery
        drone.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return drone

    @staticmethod
    def increment_flights(drone_id: str, distance_km: float) -> None:
        """Increment flight counter and total distance."""
        drone = db.session.get(DroneInventory, drone_id)
        if drone:
            drone.total_flights = (drone.total_flights or 0) + 1
            drone.total_distance_km = (drone.total_distance_km or 0.0) + distance_km
            db.session.commit()

    @staticmethod
    def get_all() -> List[DroneInventory]:
        """Get all registered drones."""
        return DroneInventory.query.all()


# ═══════════════════════════════════════════════════════════
#  User Repository
# ═══════════════════════════════════════════════════════════

class UserRepository:
    """Data access layer for User records."""

    @staticmethod
    def create(
        username: str,
        email: str,
        password: str,
        role: str = "operator",
    ) -> User:
        """Create a new user with hashed password."""
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
        )
        db.session.add(user)
        db.session.commit()
        logger.info("Created user %s with role %s", username, role)
        return user

    @staticmethod
    def get_by_username(username: str) -> Optional[User]:
        """Find user by username."""
        return User.query.filter_by(username=username).first()

    @staticmethod
    def get_by_id(user_id: str) -> Optional[User]:
        """Find user by primary key."""
        return db.session.get(User, user_id)

    @staticmethod
    def verify_password(user: User, password: str) -> bool:
        """Check a plaintext password against the stored hash."""
        return check_password_hash(user.password_hash, password)

    @staticmethod
    def update_last_login(user_id: str) -> None:
        """Record the last login timestamp."""
        user = db.session.get(User, user_id)
        if user:
            user.last_login = datetime.now(timezone.utc)
            db.session.commit()
