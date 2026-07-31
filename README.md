# MediReach - Autonomous Drone Delivery System (Team B)

This repository contains the software layer for Team B's autonomous drone operations, including Reinforcement Learning navigation, CV-based landing, QR-based security verification, and the primary Flask API that integrates with Team A's patient application.

## Quick Start (Virtual Pipeline)

Currently, the system is designed to run in a **fully simulated, virtual mode**. You do not need physical hardware or trained models to verify the logic flow. 

### Prerequisites

1. Python 3.9+
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 1. Run the Full Simulation End-to-End

We have provided a demo script that executes the entire pipeline (Dispatch -> Route -> Navigate -> CV Scan -> QR Verify -> Unlock Box -> Receipt) synchronously against simulated hardware and dummy values:

```bash
python run_full_demo.py
```
*Note: This will output a structured log showing every step succeeding, proving the components are wired together correctly.*

### 2. Run the Mock Team A Server

If you want to test the Flask API live, first start the mock Team A server (which acts as the order generator and webhook listener):
```bash
python mock_team_a/mock_order_api.py
```
This runs on port `5001`.

### 3. Start the Drone Flask API

In a new terminal window:
```bash
python src/app.py
```
This runs on port `5000`. You can now dispatch orders to `http://localhost:5000/drone/dispatch`.

### 4. Run the Automated Test Suite

We use `pytest` to assert that all components behave securely and correctly:
```bash
pytest tests/test_end_to_end.py -v --html=report.html
```

## Documentation

- **`docs/TEAM_A_REQUIREMENTS.md`**: The strict API contracts and expectations from Team A.
- **`docs/SYSTEM_STATUS.md`**: What is currently built vs what is remaining.
- **`docs/HARDWARE_INTEGRATION_GUIDE.md`**: Instructions on how to swap out the simulated `src/hardware` classes for physical GPIO/UART hardware classes when deploying to the real edge device.
