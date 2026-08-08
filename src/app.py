from flask import Flask, request, jsonify
import requests
import datetime
from models.contracts import OrderPayload
from pydantic import ValidationError

app = Flask(__name__)

# Config
MOCK_API_URL = "http://localhost:5001"

from src.hardware.hardware_manager import hw_manager

current_drone_state = {
    "drone_id": "drone-01",
    "status": "idle",
    "current_mission": None,
    "battery": 100.0,
    "location": {"lat": 37.7700, "lng": -122.4100, "alt": 0.0}
}

def send_status_webhook(status: str, message: str = ""):
    current_drone_state["status"] = status
    payload = {
        "order_id": current_drone_state["current_mission"]["order_id"] if current_drone_state["current_mission"] else "",
        "drone_id": current_drone_state["drone_id"],
        "status": status,
        "current_location": current_drone_state["location"],
        "battery_percent": current_drone_state["battery"],
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "message": message
    }
    try:
        requests.post(f"{MOCK_API_URL}/mock/webhook/status", json=payload, timeout=2)
    except Exception as e:
        print(f"Failed to send webhook: {e}")

@app.route('/drone/dispatch', methods=['POST'])
def dispatch_drone():
    """Receives order from Team A and starts mission."""
    try:
        data = request.json
        order = OrderPayload(**data)
        
        current_drone_state["current_mission"] = order.model_dump() if hasattr(order, 'model_dump') else order.dict()
        current_drone_state["battery"] = 100.0 # reset for new flight
        
        send_status_webhook("dispatched", "Drone is taking off")
        return jsonify({"status": "success", "message": "Mission started"})
    except ValidationError as e:
        return jsonify({"status": "error", "message": "Invalid order payload", "details": e.errors()}), 400

@app.route('/drone/status', methods=['GET'])
def get_drone_status():
    return jsonify(current_drone_state)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
