from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class Location(BaseModel):
    lat: float
    lng: float
    alt: Optional[float] = 0.0

class PatientInfo(BaseModel):
    id: str
    name: str
    phone: str

class OrderPayload(BaseModel):
    order_id: str
    pharmacy_id: str
    patient: PatientInfo
    delivery_location: Location
    package_weight_kg: float
    priority: str = Field(default="normal", description="normal, urgent")
    qr_token_hash: str = Field(description="Hash of the QR token the patient will scan")

class StatusWebhookPayload(BaseModel):
    order_id: str
    drone_id: str
    status: str = Field(description="dispatched, in_transit, landing, delivered, failed, returning")
    current_location: Location
    battery_percent: float
    timestamp: str
    message: Optional[str] = None
