from flask import Flask, request, jsonify
import uuid
import datetime
import hashlib

app = Flask(__name__)

# In-memory storage for webhooks received
received_webhooks = []

@app.route('/mock/generate_order', methods=['POST'])
def generate_order():
    order_id = str(uuid.uuid4())
    qr_secret = "secret_123"
    qr_token = f"{order_id}_{qr_secret}"
    qr_token_hash = hashlib.sha256(qr_token.encode()).hexdigest()
    
    order_payload = {
        "order_id": order_id,
        "pharmacy_id": "pharm_01",
        "patient": {
            "id": "pat_123",
            "name": "Jane Doe",
            "phone": "+1234567890"
        },
        "delivery_location": {
            "lat": 37.7749,
            "lng": -122.4194,
            "alt": 0.0
        },
        "package_weight_kg": 1.2,
        "priority": "normal",
        "qr_token_hash": qr_token_hash
    }
    return jsonify({"status": "success", "order": order_payload})

@app.route('/mock/webhook/status', methods=['POST'])
def receive_status_webhook():
    data = request.json
    received_webhooks.append(data)
    print(f"Received webhook: {data}")
    return jsonify({"status": "received"})

@app.route('/mock/webhooks', methods=['GET'])
def get_webhooks():
    return jsonify(received_webhooks)

if __name__ == '__main__':
    app.run(port=5001, debug=True)
