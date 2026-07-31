# Team A Integration Requirements

This document outlines the API contracts and integration requirements that Team B needs from Team A to successfully connect the MediReach Drone Delivery System.

## What we need, Week 1:
- API documentation for order endpoints.
- Sample dispatch JSON schemas (especially how delivery locations and patient info are structured).
- QR token generation logic + shared HMAC secret (so our verifier matches your generator).
- Patient schema fields (what fields are guaranteed vs optional).
- Staging test credentials to hit your sandbox environment.

## What we need, Weeks 3–5:
- A working `/order/{id}` endpoint on staging.
- Live QR token generator on staging to test our verifier against real, freshly minted tokens.
- A webhook endpoint configured on your side to receive our status pushes (`/drone/status`).
- A pharmacy "packed & ready" confirmation API.

## What we need, Weeks 7–8:
- A full end-to-end staging environment.
- Test patients and pharmacies configured in the database.
- The ability to trigger simulated orders on demand.
- Team A's frontend showing our drone tracking updates live as they arrive via webhooks.

## The exact contract we're building against right now
**Order JSON Schema (Incoming to `/drone/dispatch`)**
```json
{
  "order_id": "string",
  "pharmacy_id": "string",
  "patient": {
    "id": "string",
    "name": "string",
    "phone": "string"
  },
  "delivery_location": {
    "lat": "number",
    "lng": "number",
    "alt": "number (optional)"
  },
  "package_weight_kg": "number",
  "priority": "normal | urgent",
  "qr_token_hash": "string (hash of the token the patient scans)"
}
```

**Status Webhook Schema (Outgoing from Team B to Team A)**
```json
{
  "order_id": "string",
  "drone_id": "string",
  "status": "dispatched | in_transit | landing | delivered | failed | returning",
  "current_location": {
    "lat": "number",
    "lng": "number",
    "alt": "number"
  },
  "battery_percent": "number",
  "timestamp": "ISO-8601 string",
  "message": "string (optional details)"
}
```

**Exposed Endpoints (Team B):**
- `POST /drone/dispatch`: Start a delivery mission.
- `GET /drone/status/{drone_id}`: Fetch current live status.
- `POST /drone/delivery/confirm`: Confirm delivery (used internally/testing).

## What happens if Team A is late:
We will continue testing against our own mock server (`mock_order_api.py`). Swapping to your real API should only ever require a URL/config change on our end, never new development. All our internal logic relies on the established schemas above.
