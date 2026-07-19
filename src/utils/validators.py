# -*- coding: utf-8 -*-
"""
MediReach — Input Validation Schemas.

Marshmallow schemas for validating all API request payloads,
Team A order JSON, GPS coordinates, and configuration values.
"""

from marshmallow import Schema, fields, validate, validates, ValidationError, post_load
from typing import Any, Dict

from src.utils.constants import OrderPriority, MissionStatus


# ═══════════════════════════════════════════════════════════
#  Coordinate Schemas
# ═══════════════════════════════════════════════════════════

class CoordinateSchema(Schema):
    """Validates a GPS coordinate pair."""
    lat = fields.Float(
        required=True,
        validate=validate.Range(min=-90.0, max=90.0),
        metadata={"description": "Latitude in decimal degrees"},
    )
    long = fields.Float(
        required=True,
        validate=validate.Range(min=-180.0, max=180.0),
        metadata={"description": "Longitude in decimal degrees"},
    )


# ═══════════════════════════════════════════════════════════
#  Patient Schema
# ═══════════════════════════════════════════════════════════

class PatientSchema(Schema):
    """Validates patient information from Team A."""
    id = fields.String(
        required=True,
        validate=validate.Length(min=1, max=50),
        metadata={"description": "Patient identifier from Team A"},
    )
    name = fields.String(
        required=True,
        validate=validate.Length(min=1, max=200),
    )
    phone = fields.String(
        required=True,
        validate=validate.Regexp(
            r"^\+?[1-9]\d{6,14}$",
            error="Invalid phone number format",
        ),
    )
    delivery_coords = fields.Nested(
        CoordinateSchema,
        required=True,
    )
    address_type = fields.String(
        required=True,
        validate=validate.OneOf(["house", "apartment", "clinic"]),
    )


# ═══════════════════════════════════════════════════════════
#  Medicine Schema
# ═══════════════════════════════════════════════════════════

class MedicineSchema(Schema):
    """Validates a single medicine item in an order."""
    name = fields.String(
        required=True,
        validate=validate.Length(min=1, max=200),
    )
    quantity = fields.Integer(
        required=True,
        validate=validate.Range(min=1, max=100),
    )
    priority = fields.String(
        required=False,
        load_default="normal",
        validate=validate.OneOf(["emergency", "normal"]),
    )


# ═══════════════════════════════════════════════════════════
#  Dispatch Request Schema (Team A → Our API)
# ═══════════════════════════════════════════════════════════

class DispatchRequestSchema(Schema):
    """Validates the full drone dispatch request from Team A.

    This is the primary input schema for ``POST /api/v1/drone/dispatch``.
    """
    order_id = fields.String(
        required=True,
        validate=validate.Regexp(
            r"^ORD-\d{4}-[A-Z0-9]{5,}$",
            error="Order ID must match format ORD-YYYY-XXXXX",
        ),
        metadata={"description": "Unique order ID from Team A"},
    )
    patient = fields.Nested(
        PatientSchema,
        required=True,
    )
    medicines = fields.List(
        fields.Nested(MedicineSchema),
        required=True,
        validate=validate.Length(min=1, max=20),
    )
    qr_token = fields.String(
        required=True,
        validate=validate.Length(min=10),
        metadata={"description": "Encrypted QR token string"},
    )
    token_expiry = fields.DateTime(
        required=True,
        format="iso",
        metadata={"description": "Token expiry timestamp (ISO 8601)"},
    )
    pickup_coords = fields.Nested(
        CoordinateSchema,
        required=True,
    )
    order_priority = fields.String(
        required=False,
        load_default="standard",
        validate=validate.OneOf([p.value for p in OrderPriority]),
    )

    @validates("medicines")
    def validate_medicines_not_empty(self, value: list) -> None:
        """Ensure at least one medicine is present."""
        if not value:
            raise ValidationError("At least one medicine is required.")


# ═══════════════════════════════════════════════════════════
#  Mission Status Update Schema
# ═══════════════════════════════════════════════════════════

class MissionStatusUpdateSchema(Schema):
    """Validates mission status update requests."""
    status = fields.String(
        required=True,
        validate=validate.OneOf([s.value for s in MissionStatus]),
    )
    notes = fields.String(
        required=False,
        validate=validate.Length(max=500),
    )


# ═══════════════════════════════════════════════════════════
#  Auth Schemas
# ═══════════════════════════════════════════════════════════

class LoginSchema(Schema):
    """Validates login credentials."""
    username = fields.String(
        required=True,
        validate=validate.Length(min=3, max=80),
    )
    password = fields.String(
        required=True,
        validate=validate.Length(min=8, max=128),
    )


class RegisterSchema(Schema):
    """Validates user registration."""
    username = fields.String(
        required=True,
        validate=validate.Length(min=3, max=80),
    )
    password = fields.String(
        required=True,
        validate=validate.Length(min=8, max=128),
    )
    email = fields.Email(required=True)
    role = fields.String(
        required=False,
        load_default="operator",
        validate=validate.OneOf(["admin", "operator", "viewer"]),
    )


# ═══════════════════════════════════════════════════════════
#  Telemetry Ingest Schema
# ═══════════════════════════════════════════════════════════

class TelemetryIngestSchema(Schema):
    """Validates incoming telemetry packets from drones."""
    drone_id = fields.String(required=True)
    mission_id = fields.String(required=True)
    timestamp = fields.Float(required=True)
    latitude = fields.Float(
        required=True,
        validate=validate.Range(min=-90.0, max=90.0),
    )
    longitude = fields.Float(
        required=True,
        validate=validate.Range(min=-180.0, max=180.0),
    )
    altitude = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=500.0),
    )
    speed_ms = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=50.0),
    )
    heading_degrees = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=360.0),
    )
    battery_percent = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=100.0),
    )
    signal_strength = fields.Integer(
        required=True,
        validate=validate.Range(min=-120, max=0),
    )
    status = fields.String(required=True)
    obstacle_near = fields.Boolean(required=False, load_default=False)
    eta_seconds = fields.Integer(
        required=False,
        load_default=0,
        validate=validate.Range(min=0),
    )


# ═══════════════════════════════════════════════════════════
#  Route Request Schema
# ═══════════════════════════════════════════════════════════

class NoFlyZoneSchema(Schema):
    """Validates a no-fly zone definition."""
    zone_id = fields.String(required=True)
    center = fields.Nested(CoordinateSchema, required=True)
    radius_m = fields.Float(
        required=True,
        validate=validate.Range(min=10.0, max=10000.0),
    )
    active = fields.Boolean(required=False, load_default=True)


class WeatherDataSchema(Schema):
    """Validates weather data for route optimisation."""
    wind_speed = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=100.0),
    )
    wind_direction = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=360.0),
    )
    rain_intensity = fields.Float(
        required=True,
        validate=validate.Range(min=0.0, max=1.0),
    )
    visibility = fields.Float(
        required=False,
        load_default=1.0,
        validate=validate.Range(min=0.0, max=1.0),
    )


class RouteRequestSchema(Schema):
    """Validates route optimisation requests."""
    pickup_coords = fields.Nested(CoordinateSchema, required=True)
    delivery_coords = fields.Nested(CoordinateSchema, required=True)
    weather_data = fields.Nested(WeatherDataSchema, required=False)
    no_fly_zones = fields.List(
        fields.Nested(NoFlyZoneSchema),
        required=False,
        load_default=[],
    )
    battery_level = fields.Float(
        required=False,
        load_default=1.0,
        validate=validate.Range(min=0.0, max=1.0),
    )
    priority = fields.String(
        required=False,
        load_default="standard",
        validate=validate.OneOf(["emergency", "standard"]),
    )


# ═══════════════════════════════════════════════════════════
#  Pagination Schema
# ═══════════════════════════════════════════════════════════

class PaginationSchema(Schema):
    """Validates pagination query parameters."""
    page = fields.Integer(
        required=False,
        load_default=1,
        validate=validate.Range(min=1),
    )
    per_page = fields.Integer(
        required=False,
        load_default=20,
        validate=validate.Range(min=1, max=100),
    )
    sort_by = fields.String(
        required=False,
        load_default="created_at",
    )
    sort_order = fields.String(
        required=False,
        load_default="desc",
        validate=validate.OneOf(["asc", "desc"]),
    )


# ═══════════════════════════════════════════════════════════
#  Validation Helper
# ═══════════════════════════════════════════════════════════

def validate_request(schema: Schema, data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate request data against a marshmallow schema.

    Args:
        schema: Marshmallow schema instance.
        data: Raw request data dictionary.

    Returns:
        Validated and deserialized data dictionary.

    Raises:
        ValidationError: If validation fails.
    """
    return schema.load(data)
