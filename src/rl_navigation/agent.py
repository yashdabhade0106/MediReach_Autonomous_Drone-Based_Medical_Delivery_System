# -*- coding: utf-8 -*-
"""
MediReach — PPO Agent for Drone Navigation.

Wraps Stable-Baselines3 PPO with custom MLP policy,
training callbacks, evaluation, model persistence,
and route inference.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger
from src.utils.constants import RLConfig, DroneAction

logger = get_logger(__name__)

_CFG = RLConfig()


@dataclass
class TrainingResult:
    """Container for training run results."""
    total_timesteps: int
    mean_reward: float
    std_reward: float
    success_rate: float
    best_mean_reward: float
    model_path: str


@dataclass
class EvaluationResult:
    """Container for evaluation metrics."""
    n_episodes: int
    mean_reward: float
    std_reward: float
    success_rate: float
    avg_steps: float
    avg_battery_remaining: float
    min_reward: float
    max_reward: float


class PPOAgent:
    """PPO-based navigation agent for the MediReach environment.

    Handles model creation, training with callbacks, evaluation,
    saving/loading, and route inference.
    """

    def __init__(
        self,
        env: Any,
        config: Optional[RLConfig] = None,
        tensorboard_log: str = "./logs/tensorboard/",
    ) -> None:
        """Initialise the PPO agent.

        Args:
            env: Gymnasium-compatible environment.
            config: RL configuration parameters.
            tensorboard_log: Path for TensorBoard logs.
        """
        self.env = env
        self.config = config or RLConfig()
        self.tensorboard_log = tensorboard_log
        self.model: Any = None
        self._model_built = False

        logger.info("PPOAgent initialised")

    def build_model(self) -> None:
        """Build the PPO model with custom MLP architecture.

        Network: pi=[256, 128, 64], vf=[256, 128, 64]
        """
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            logger.error("stable-baselines3 not installed: %s", exc)
            raise

        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config.LEARNING_RATE,
            n_steps=self.config.N_STEPS,
            batch_size=self.config.BATCH_SIZE,
            n_epochs=self.config.N_EPOCHS,
            gamma=self.config.GAMMA,
            gae_lambda=self.config.GAE_LAMBDA,
            clip_range=self.config.CLIP_RANGE,
            ent_coef=self.config.ENT_COEF,
            verbose=1,
            tensorboard_log=self.tensorboard_log,
            policy_kwargs={
                "net_arch": [
                    dict(pi=[256, 128, 64], vf=[256, 128, 64])
                ],
            },
        )
        self._model_built = True
        logger.info(
            "PPO model built: lr=%.4f, steps=%d, batch=%d, γ=%.2f",
            self.config.LEARNING_RATE,
            self.config.N_STEPS,
            self.config.BATCH_SIZE,
            self.config.GAMMA,
        )

    def train(
        self,
        total_timesteps: Optional[int] = None,
        eval_env: Optional[Any] = None,
        save_dir: str = "models/rl",
        run_name: str = "medireach_ppo",
    ) -> TrainingResult:
        """Run full training with evaluation and checkpoint callbacks.

        Args:
            total_timesteps: Override default timesteps.
            eval_env: Separate environment for evaluation callbacks.
            save_dir: Directory for model checkpoints.
            run_name: Name prefix for saved models.

        Returns:
            TrainingResult with metrics.

        Raises:
            RuntimeError: If model not built.
        """
        if not self._model_built or self.model is None:
            raise RuntimeError("Call build_model() before train()")

        from stable_baselines3.common.callbacks import (
            EvalCallback,
            CheckpointCallback,
            StopTrainingOnRewardThreshold,
            CallbackList,
        )

        timesteps = total_timesteps or self.config.TOTAL_TIMESTEPS
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        callbacks = []

        # Evaluation callback
        if eval_env is None:
            eval_env = self.env

        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=self.config.STOP_REWARD_THRESHOLD,
            verbose=1,
        )

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=os.path.join(save_dir, "eval_logs"),
            eval_freq=self.config.EVAL_FREQ,
            n_eval_episodes=10,
            deterministic=True,
            callback_after_eval=stop_callback,
            verbose=1,
        )
        callbacks.append(eval_callback)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.CHECKPOINT_FREQ,
            save_path=save_dir,
            name_prefix=run_name,
            verbose=1,
        )
        callbacks.append(checkpoint_callback)

        callback_list = CallbackList(callbacks)

        logger.info(
            "Starting PPO training: %d timesteps, eval every %d, "
            "checkpoint every %d",
            timesteps,
            self.config.EVAL_FREQ,
            self.config.CHECKPOINT_FREQ,
        )

        self.model.learn(
            total_timesteps=timesteps,
            callback=callback_list,
            progress_bar=True,
        )

        # Save final model
        final_path = os.path.join(save_dir, f"{run_name}_final")
        self.model.save(final_path)
        logger.info("Final model saved to %s", final_path)

        # Evaluate final model
        eval_result = self.evaluate(n_episodes=50)

        return TrainingResult(
            total_timesteps=timesteps,
            mean_reward=eval_result.mean_reward,
            std_reward=eval_result.std_reward,
            success_rate=eval_result.success_rate,
            best_mean_reward=eval_result.max_reward,
            model_path=final_path + ".zip",
        )

    def evaluate(self, n_episodes: int = 100) -> EvaluationResult:
        """Run evaluation episodes and compute metrics.

        Args:
            n_episodes: Number of episodes to run.

        Returns:
            EvaluationResult with statistics.

        Raises:
            RuntimeError: If model not built.
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call build_model() or load().")

        rewards = []
        steps_list = []
        successes = 0
        battery_remaining = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0.0
            done = False
            step_count = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(int(action))
                episode_reward += reward
                step_count += 1
                done = terminated or truncated

            rewards.append(episode_reward)
            steps_list.append(step_count)
            battery_remaining.append(info.get("battery_level", 0.0))

            if info.get("reached_target", False):
                successes += 1

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))
        success_rate = successes / n_episodes

        logger.info(
            "Evaluation (%d episodes): mean_reward=%.2f±%.2f, "
            "success=%.1f%%, avg_steps=%.1f",
            n_episodes, mean_reward, std_reward,
            success_rate * 100, np.mean(steps_list),
        )

        return EvaluationResult(
            n_episodes=n_episodes,
            mean_reward=mean_reward,
            std_reward=std_reward,
            success_rate=success_rate,
            avg_steps=float(np.mean(steps_list)),
            avg_battery_remaining=float(np.mean(battery_remaining)),
            min_reward=float(np.min(rewards)),
            max_reward=float(np.max(rewards)),
        )

    def optimize_route(
        self,
        start_pos: List[float],
        end_pos: List[float],
        weather: Optional[Dict[str, Any]] = None,
        no_fly_zones: Optional[List[Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference to generate an optimised route.

        Args:
            start_pos: [x, y, altitude] start position.
            end_pos: [x, y] target position.
            weather: Optional weather conditions.
            no_fly_zones: Optional list of no-fly zone dicts.

        Returns:
            List of waypoint dictionaries with positions and actions.

        Raises:
            RuntimeError: If model not loaded.
        """
        if self.model is None:
            raise RuntimeError("No model loaded.")

        # Reset environment with specified positions
        options = {
            "start_pos": start_pos,
            "target_pos": end_pos,
        }
        obs, info = self.env.reset(options=options)

        if weather:
            self.env.weather_sim.set_from_api_data(weather)

        waypoints: List[Dict[str, Any]] = []
        done = False
        step = 0

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action_int = int(action)
            obs, reward, terminated, truncated, info = self.env.step(action_int)
            done = terminated or truncated

            waypoint = {
                "step": step,
                "x": float(self.env.drone_pos[0]),
                "y": float(self.env.drone_pos[1]),
                "altitude": float(self.env.drone_pos[2]),
                "action": DroneAction(action_int).name.lower(),
                "battery": float(self.env.battery),
                "distance_remaining": float(info.get("distance_to_target", 0)),
            }
            waypoints.append(waypoint)
            step += 1

        logger.info(
            "Route generated: %d waypoints, success=%s, battery=%.1f%%",
            len(waypoints),
            info.get("reached_target", False),
            self.env.battery * 100,
        )

        return waypoints

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: File path (without extension).
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: Path to the saved model file (.zip).
        """
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            logger.error("stable-baselines3 not installed: %s", exc)
            raise

        self.model = PPO.load(path, env=self.env)
        self._model_built = True
        logger.info("Model loaded from %s", path)
