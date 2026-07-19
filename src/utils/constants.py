# -*- coding: utf-8 -*-
"""
MediReach — Project-Wide Constants & Enumerations.

Centralised configuration constants used across all modules.
All magic numbers and string literals should be defined here
to ensure single-source-of-truth and easy maintenance.
"""

from enum import Enum, unique
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════
#  Mission Status Enum
# ═══════════════════════════════════════════════════════════

@unique
class MissionStatus(str, Enum):
    """Lifecycle states for a drone delivery mission."""
    PENDING = "PENDING"
    DISPATCHED = "DISPATCHED"
    IN_FLIGHT = "IN_FLIGHT"
    APPROACHING = "APPROACHING"
    LANDING = "LANDING"
    QR_PENDING = "QR_PENDING"
    DELIVERED = "DELIVERED"
    FAILED = "FAILED"
    RETURNED = "RETURNED"
    ABORTED = "ABORTED"


@unique
class DroneStatus(str, Enum):
    """Operational states for a drone unit."""
    AVAILABLE = "AVAILABLE"
    DISPATCHED = "DISPATCHED"
    IN_FLIGHT = "IN_FLIGHT"
    CHARGING = "CHARGING"
    MAINTENANCE = "MAINTENANCE"
    OFFLINE = "OFFLINE"
    EMERGENCY = "EMERGENCY"


@unique
class OrderPriority(str, Enum):
    """Order priority levels from Team A."""
    EMERGENCY = "emergency"
    STANDARD = "standard"


@unique
class LandingZoneType(str, Enum):
    """Types of landing zones detected by CV module."""
    SAFE_FLAT_GROUND = "safe_flat_ground"
    DRIVEWAY = "driveway"
    BALCONY = "balcony"
    ROOFTOP = "rooftop"
    WATER = "water"
    CROWD = "crowd"
    SLOPE = "slope"
    VEHICLE = "vehicle"
    OBSTACLE = "obstacle"


@unique
class WeatherProfile(str, Enum):
    """Weather condition profiles for simulation."""
    CLEAR = "clear"
    WINDY = "windy"
    RAINY = "rainy"
    STORM = "storm"
    FOGGY = "foggy"


@unique
class DroneAction(int, Enum):
    """Discrete action space for RL navigation."""
    MOVE_NORTH = 0
    MOVE_SOUTH = 1
    MOVE_EAST = 2
    MOVE_WEST = 3
    ASCEND = 4
    DESCEND = 5
    HOVER = 6


@unique
class AlertType(str, Enum):
    """Types of MQTT alert messages."""
    BATTERY_LOW = "battery_low"
    BATTERY_CRITICAL = "battery_critical"
    OBSTACLE_DETECTED = "obstacle_detected"
    NO_FLY_VIOLATION = "no_fly_violation"
    WEATHER_UNSAFE = "weather_unsafe"
    TAMPER_DETECTED = "tamper_detected"
    GPS_LOST = "gps_lost"
    DELIVERY_COMPLETE = "delivery_complete"
    MISSION_ABORTED = "mission_aborted"


# ═══════════════════════════════════════════════════════════
#  Drone Hardware Constants
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DroneConfig:
    """Physical and operational drone parameters."""
    MAX_ALTITUDE_M: float = 100.0
    MIN_ALTITUDE_M: float = 10.0
    MAX_SPEED_MS: float = 15.0
    CRUISE_ALTITUDE_M: float = 50.0
    MAX_RANGE_KM: float = 25.0
    MAX_PAYLOAD_KG: float = 2.5
    BATTERY_CAPACITY_MAH: int = 5200
    LOW_BATTERY_THRESHOLD: float = 0.20
    CRITICAL_BATTERY_THRESHOLD: float = 0.10
    RETURN_BATTERY_RESERVE: float = 0.15
    HOME_LAT: float = 18.5204
    HOME_LONG: float = 73.8567
    LANDING_APPROACH_ALTITUDE_M: float = 15.0
    LANDING_TOLERANCE_M: float = 10.0
    QR_SCAN_ALTITUDE_M: float = 3.0


# ═══════════════════════════════════════════════════════════
#  RL Navigation Constants
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RLConfig:
    """Reinforcement learning training and inference parameters."""
    # Observation space dimensions
    OBS_DRONE_POS: int = 3
    OBS_TARGET_POS: int = 2
    OBS_BATTERY: int = 1
    OBS_WIND_SPEED: int = 1
    OBS_WIND_DIR: int = 1
    OBS_RAIN: int = 1
    OBS_OBSTACLE_MAP: int = 8
    OBS_NO_FLY: int = 5
    OBS_DISTANCE: int = 1
    OBS_TIME: int = 1
    TOTAL_OBS_DIM: int = 24

    # Action space
    NUM_ACTIONS: int = 7

    # Grid environment
    GRID_SIZE: int = 100
    GRID_CELL_SIZE_M: float = 50.0

    # PPO hyperparameters
    LEARNING_RATE: float = 3e-4
    N_STEPS: int = 2048
    BATCH_SIZE: int = 64
    N_EPOCHS: int = 10
    GAMMA: float = 0.99
    GAE_LAMBDA: float = 0.95
    CLIP_RANGE: float = 0.2
    ENT_COEF: float = 0.01

    # Training
    TOTAL_TIMESTEPS: int = 500_000
    EVAL_FREQ: int = 10_000
    CHECKPOINT_FREQ: int = 50_000
    STOP_REWARD_THRESHOLD: float = 90.0
    MAX_EPISODE_STEPS: int = 500

    # Movement
    STEP_SIZE_GRID: float = 1.0
    ALTITUDE_STEP_M: float = 5.0
    BATTERY_DRAIN_PER_STEP: float = 0.002
    BATTERY_DRAIN_WIND_FACTOR: float = 0.001
    TARGET_RADIUS_GRID: float = 2.0


# ═══════════════════════════════════════════════════════════
#  CV Detection Constants
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CVConfig:
    """Computer vision model and detection parameters."""
    IMAGE_SIZE: int = 640
    CONFIDENCE_THRESHOLD: float = 0.65
    NMS_THRESHOLD: float = 0.45
    SAFE_CLASSES: Tuple[str, ...] = (
        "safe_flat_ground", "driveway", "balcony", "rooftop"
    )
    UNSAFE_CLASSES: Tuple[str, ...] = (
        "water", "crowd", "slope", "vehicle", "obstacle"
    )
    ALL_CLASSES: Tuple[str, ...] = (
        "safe_flat_ground", "driveway", "balcony", "rooftop",
        "water", "crowd", "slope", "vehicle", "obstacle"
    )
    MIN_LANDING_AREA_PIXELS: int = 5000
    EPOCHS: int = 100
    BATCH_SIZE: int = 16
    PATIENCE: int = 20
    EDGE_TARGET_FPS: int = 5


# ═══════════════════════════════════════════════════════════
#  QR Security Constants
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class QRConfig:
    """QR token generation and verification parameters."""
    TOKEN_VALIDITY_MINUTES: int = 30
    QR_ERROR_CORRECTION: str = "H"
    QR_BOX_SIZE: int = 10
    QR_BORDER: int = 4
    BLACKLIST_EXPIRY_SECONDS: int = 3600
    MAX_SCAN_ATTEMPTS: int = 10
    SCAN_TIMEOUT_SECONDS: int = 60
    SCAN_INTERVAL_SECONDS: float = 0.5


# ═══════════════════════════════════════════════════════════
#  GPIO Pin Mapping (Raspberry Pi BCM)
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GPIOPins:
    """BCM pin assignments for Raspberry Pi GPIO."""
    SERVO_PWM: int = 18
    LED_LOCKED: int = 17
    LED_UNLOCKED: int = 27
    LED_SCANNING: int = 22
    TAMPER_SENSOR: int = 23
    BUZZER: int = 24
    LED_SAFE_ZONE: int = 5
    LED_UNSAFE_ZONE: int = 6


# ═══════════════════════════════════════════════════════════
#  MQTT Topics
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MQTTTopics:
    """MQTT topic templates. Format with drone_id."""
    TELEMETRY: str = "medireach/drone/{}/telemetry"
    COMMANDS: str = "medireach/drone/{}/commands"
    ALERTS: str = "medireach/drone/{}/alerts"
    STATUS: str = "medireach/drone/{}/status"
    LANDING: str = "medireach/drone/{}/landing"
    QR_RESULT: str = "medireach/drone/{}/qr_result"


# ═══════════════════════════════════════════════════════════
#  API Constants
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class APIConfig:
    """REST API configuration constants."""
    API_VERSION: str = "v1"
    API_PREFIX: str = "/api/v1"
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100
    RATE_LIMIT_DEFAULT: str = "100/hour"
    RATE_LIMIT_DISPATCH: str = "30/hour"
    SSE_PUBLISH_INTERVAL_SEC: float = 2.0
    JWT_ACCESS_TOKEN_EXPIRES_HOURS: int = 24
    JWT_REFRESH_TOKEN_EXPIRES_DAYS: int = 30


# ═══════════════════════════════════════════════════════════
#  Weather Thresholds
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WeatherThresholds:
    """Safety thresholds for flight weather conditions."""
    MAX_WIND_SPEED_MS: float = 35.0
    MAX_RAIN_INTENSITY: float = 0.9
    MIN_VISIBILITY: float = 0.2
    WIND_WARNING_MS: float = 25.0
    RAIN_WARNING: float = 0.7
