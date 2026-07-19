# -*- coding: utf-8 -*-
"""
MediReach — RL Training Entry Point.

CLI-based training launcher for the PPO navigation agent.

Usage:
    python -m src.rl_navigation.train --timesteps 500000
    python -m src.rl_navigation.train --eval-only --model models/rl/best_model.zip
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

from src.rl_navigation.environment import MediReachEnv
from src.rl_navigation.agent import PPOAgent
from src.rl_navigation.visualizer import RouteVisualizer
from src.utils.constants import RLConfig
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="MediReach RL Navigation Agent Training",
    )
    parser.add_argument(
        "--timesteps", type=int,
        default=int(os.getenv("RL_TOTAL_TIMESTEPS", "500000")),
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--grid-size", type=int, default=100,
        help="Grid world size (default: 100)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500,
        help="Max episode steps (default: 500)",
    )
    parser.add_argument(
        "--save-dir", type=str, default="models/rl",
        help="Model save directory (default: models/rl)",
    )
    parser.add_argument(
        "--run-name", type=str, default="medireach_ppo",
        help="Run name prefix for saved models",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training, only evaluate an existing model",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to trained model for evaluation",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=100,
        help="Number of evaluation episodes (default: 100)",
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Enable rendering during evaluation",
    )
    return parser.parse_args()


def main() -> None:
    """Main training/evaluation entry point."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("MediReach RL Navigation — Training Pipeline")
    logger.info("=" * 60)

    # Create environments
    render_mode = "human" if args.render else None
    train_env = MediReachEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode=None,  # Never render during training
    )
    eval_env = MediReachEnv(
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        render_mode=render_mode,
    )

    agent = PPOAgent(
        env=train_env,
        config=RLConfig(),
    )

    if args.eval_only:
        # Evaluation mode
        model_path = args.model
        if model_path is None:
            model_path = os.path.join(args.save_dir, "best_model.zip")

        logger.info("Loading model from %s", model_path)
        agent = PPOAgent(env=eval_env)
        agent.load(model_path)

        logger.info("Running evaluation: %d episodes", args.eval_episodes)
        result = agent.evaluate(n_episodes=args.eval_episodes)

        logger.info("=" * 40)
        logger.info("Evaluation Results:")
        logger.info("  Mean Reward  : %.2f ± %.2f", result.mean_reward, result.std_reward)
        logger.info("  Success Rate : %.1f%%", result.success_rate * 100)
        logger.info("  Avg Steps    : %.1f", result.avg_steps)
        logger.info("  Avg Battery  : %.1f%%", result.avg_battery_remaining * 100)
        logger.info("  Min Reward   : %.2f", result.min_reward)
        logger.info("  Max Reward   : %.2f", result.max_reward)
        logger.info("=" * 40)

    else:
        # Training mode
        logger.info("Configuration:")
        logger.info("  Timesteps    : %d", args.timesteps)
        logger.info("  Grid Size    : %d", args.grid_size)
        logger.info("  Max Steps    : %d", args.max_steps)
        logger.info("  Save Dir     : %s", args.save_dir)

        agent.build_model()
        start = time.time()
        result = agent.train(
            total_timesteps=args.timesteps,
            eval_env=eval_env,
            save_dir=args.save_dir,
            run_name=args.run_name,
        )
        elapsed = time.time() - start

        logger.info("=" * 40)
        logger.info("Training Complete!")
        logger.info("  Duration     : %.1f minutes", elapsed / 60)
        logger.info("  Mean Reward  : %.2f ± %.2f", result.mean_reward, result.std_reward)
        logger.info("  Success Rate : %.1f%%", result.success_rate * 100)
        logger.info("  Model Path   : %s", result.model_path)
        logger.info("=" * 40)

    # Cleanup
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
