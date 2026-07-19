# -*- coding: utf-8 -*-
"""
MediReach — Reward Shaping Functions for RL Navigation.

Provides a configurable reward calculator with positive rewards
for progress, delivery, efficiency and negative penalties for
collisions, no-fly violations, battery depletion, and idle hovering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RewardConfig:
    """Configurable reward and penalty magnitudes.

    All values are tunable hyper-parameters for reward shaping.
    """
    # Positive rewards
    PROGRESS_REWARD_SCALE: float = 5.0
    DELIVERY_BONUS: float = 100.0
    EFFICIENCY_BONUS: float = 10.0
    WEATHER_ADAPT_BONUS: float = 5.0

    # Negative penalties
    NO_FLY_PENALTY: float = 50.0
    COLLISION_PENALTY: float = 100.0
    BATTERY_PENALTY: float = 75.0
    HOVER_PENALTY: float = 1.0
    TIME_PENALTY_PER_STEP: float = 0.1

    # Bonus thresholds
    BATTERY_EFFICIENT_THRESHOLD: float = 0.30
    CLOSE_APPROACH_RADIUS: float = 5.0
    CLOSE_APPROACH_BONUS: float = 2.0


class RewardCalculator:
    """Calculates composite reward for each environment step.

    The reward signal combines multiple components to encourage
    the agent to reach the target quickly, efficiently, and safely.

    Components:
        + Progress reward (proportional to distance reduction)
        + Delivery completion bonus
        + Battery efficiency bonus
        + Weather adaptation bonus
        + Close-approach shaping bonus
        − No-fly zone violation penalty
        − Obstacle collision penalty
        − Battery depletion penalty
        − Unnecessary hover penalty
        − Time penalty per step
    """

    def __init__(self, config: RewardConfig) -> None:
        """Initialise with reward configuration.

        Args:
            config: RewardConfig dataclass with magnitudes.
        """
        self.cfg = config

    def calculate_reward(
        self,
        prev_distance: float,
        current_distance: float,
        action: int,
        info: Dict[str, Any],
    ) -> float:
        """Calculate the total reward for the current step.

        Args:
            prev_distance: Distance to target before action.
            current_distance: Distance to target after action.
            action: Action taken (0–6).
            info: Dictionary with boolean flags from environment step.

        Returns:
            Total scalar reward.
        """
        reward = 0.0

        # ── Positive rewards ──────────────────────────────

        # 1. Progress reward: proportional to distance reduction
        progress = prev_distance - current_distance
        reward += progress * self.cfg.PROGRESS_REWARD_SCALE

        # 2. Delivery completion bonus
        if info.get("reached_target", False):
            reward += self.cfg.DELIVERY_BONUS
            logger.debug("Delivery bonus applied: +%.1f", self.cfg.DELIVERY_BONUS)

        # 3. Battery efficiency bonus (arrived with battery to spare)
        if info.get("battery_efficient", False):
            reward += self.cfg.EFFICIENCY_BONUS

        # 4. Weather adaptation (hovering/descending in high wind)
        if info.get("weather_adapted", False):
            reward += self.cfg.WEATHER_ADAPT_BONUS

        # 5. Close approach shaping (extra reward when getting close)
        if current_distance < self.cfg.CLOSE_APPROACH_RADIUS:
            approach_bonus = (
                (self.cfg.CLOSE_APPROACH_RADIUS - current_distance)
                / self.cfg.CLOSE_APPROACH_RADIUS
                * self.cfg.CLOSE_APPROACH_BONUS
            )
            reward += approach_bonus

        # ── Negative penalties ────────────────────────────

        # 6. No-fly zone violation
        if info.get("no_fly_violation", False):
            reward -= self.cfg.NO_FLY_PENALTY

        # 7. Obstacle collision
        if info.get("collision", False):
            reward -= self.cfg.COLLISION_PENALTY

        # 8. Battery depleted
        if info.get("battery_dead", False):
            reward -= self.cfg.BATTERY_PENALTY

        # 9. Hovering penalty (only when not near obstacles)
        if action == 6 and not info.get("obstacle_near", False):
            reward -= self.cfg.HOVER_PENALTY

        # 10. Time penalty (constant per-step cost)
        reward -= self.cfg.TIME_PENALTY_PER_STEP

        return reward

    def get_config(self) -> Dict[str, float]:
        """Return current reward configuration as a dictionary.

        Returns:
            Dictionary of all reward parameters.
        """
        return {
            "progress_scale": self.cfg.PROGRESS_REWARD_SCALE,
            "delivery_bonus": self.cfg.DELIVERY_BONUS,
            "efficiency_bonus": self.cfg.EFFICIENCY_BONUS,
            "weather_adapt_bonus": self.cfg.WEATHER_ADAPT_BONUS,
            "no_fly_penalty": self.cfg.NO_FLY_PENALTY,
            "collision_penalty": self.cfg.COLLISION_PENALTY,
            "battery_penalty": self.cfg.BATTERY_PENALTY,
            "hover_penalty": self.cfg.HOVER_PENALTY,
            "time_penalty": self.cfg.TIME_PENALTY_PER_STEP,
            "close_approach_radius": self.cfg.CLOSE_APPROACH_RADIUS,
            "close_approach_bonus": self.cfg.CLOSE_APPROACH_BONUS,
        }
