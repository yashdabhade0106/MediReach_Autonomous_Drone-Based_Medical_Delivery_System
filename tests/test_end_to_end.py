import pytest
from src.models.contracts import OrderPayload

def test_order_payload_validation():
    payload = {
        "order_id": "123",
        "pharmacy_id": "456",
        "patient": {"id": "1", "name": "John", "phone": "123"},
        "delivery_location": {"lat": 1.0, "lng": 2.0},
        "package_weight_kg": 1.0,
        "qr_token_hash": "abc"
    }
    order = OrderPayload(**payload)
    assert order.order_id == "123"

def test_simulated_hardware():
    from src.hardware.simulated_hardware import SimulatedServoBox
    box = SimulatedServoBox()
    assert box.is_locked == True
    box.unlock()
    assert box.is_locked == False

def test_drone_nav_env():
    from src.rl_navigation.env import DroneNavEnv
    env = DroneNavEnv()
    obs, info = env.reset()
    assert len(obs) == 4
