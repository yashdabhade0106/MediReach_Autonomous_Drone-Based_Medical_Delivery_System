import os

# Create model directories
os.makedirs('models/rl', exist_ok=True)
os.makedirs('models/cv', exist_ok=True)

print("Generating dummy PPO model...")
from src.rl_navigation.agent import PPOAgent
from src.rl_navigation.environment import MediReachEnv
from src.utils.constants import RLConfig

try:
    env = MediReachEnv()
    agent = PPOAgent(env, RLConfig())
    agent.build_model()
    agent.model.save("models/rl/best_model.zip")
    print("Successfully generated models/rl/best_model.zip")
except Exception as e:
    print(f"Failed to generate PPO model: {e}")

print("Generating dummy YOLO model...")
try:
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.save("models/cv/landing_model.pt")
    print("Successfully generated models/cv/landing_model.pt")
except Exception as e:
    print(f"Failed to generate YOLO model: {e}")
