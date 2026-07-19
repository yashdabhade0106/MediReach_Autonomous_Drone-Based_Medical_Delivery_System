# -*- coding: utf-8 -*-
"""
MediReach — Drone Navigation Gymnasium Environment.

Custom OpenAI Gymnasium environment for autonomous drone
route optimisation with weather, obstacles, and no-fly zones.

Observation Space (24-dim continuous):
    drone_position    : [lat, long, altitude]     (3)
    target_position   : [lat, long]               (2)
    battery_level     : [0.0–1.0]                 (1)
    wind_speed        : [0.0–50.0 m/s]            (1)
    wind_direction    : [0–360 deg]               (1)
    rain_intensity    : [0.0–1.0]                 (1)
    obstacle_map      : [8 directional sensors]   (8)
    no_fly_active     : [binary × 5 zones]        (5)
    distance_to_target: [0.0–max_range]           (1)
    time_elapsed      : [0.0–max_time]            (1)

Action Space (Discrete, 7):
    0=North, 1=South, 2=East, 3=West,
    4=Ascend, 5=Descend, 6=Hover
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.rl_navigation.reward_functions import RewardCalculator, RewardConfig
from src.rl_navigation.weather_simulator import WeatherSimulator
from src.rl_navigation.obstacles import ObstacleManager, NoFlyZone
from src.utils.logger import get_logger
from src.utils.constants import DroneAction, RLConfig

logger = get_logger(__name__)

# Default RL configuration
_CFG = RLConfig()


class MediReachEnv(gym.Env):
    """Gymnasium environment for MediReach drone navigation.

    The drone must navigate from a pickup location to a delivery
    location while avoiding obstacles, no-fly zones, and adverse
    weather, all under battery and time constraints.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = _CFG.GRID_SIZE,
        max_steps: int = _CFG.MAX_EPISODE_STEPS,
        render_mode: Optional[str] = None,
        num_obstacles: int = 15,
        num_no_fly_zones: int = 5,
        weather_profile: Optional[str] = None,
    ) -> None:
        """Initialise the MediReach navigation environment.

        Args:
            grid_size: Side length of the square grid world.
            max_steps: Maximum timesteps per episode.
            render_mode: 'human' for matplotlib, 'rgb_array' for image.
            num_obstacles: Number of random obstacles to place.
            num_no_fly_zones: Number of no-fly zones.
            weather_profile: Fixed weather profile or None for random.
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.num_obstacles = num_obstacles
        self.num_no_fly_zones = num_no_fly_zones
        self.fixed_weather = weather_profile

        # --- Observation space (24-dim continuous) ---
        low = np.array(
            [0.0, 0.0, 0.0]          # drone pos: x, y, alt
            + [0.0, 0.0]              # target pos: x, y
            + [0.0]                   # battery
            + [0.0]                   # wind speed
            + [0.0]                   # wind direction
            + [0.0]                   # rain intensity
            + [0.0] * 8              # 8 obstacle sensors
            + [0.0] * 5              # 5 no-fly zone flags
            + [0.0]                   # distance to target
            + [0.0],                  # time elapsed (normalised)
            dtype=np.float32,
        )
        high = np.array(
            [float(grid_size)] * 3    # drone pos
            + [float(grid_size)] * 2  # target pos
            + [1.0]                   # battery
            + [50.0]                  # wind speed
            + [360.0]                 # wind direction
            + [1.0]                   # rain intensity
            + [1.0] * 8              # obstacle sensors (normalised)
            + [1.0] * 5              # no-fly flags
            + [float(grid_size * 2)]  # max possible distance
            + [1.0],                  # time elapsed (normalised)
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(7)

        # Sub-systems
        self.weather_sim = WeatherSimulator()
        self.obstacle_mgr = ObstacleManager(grid_size=grid_size)
        self.reward_calc = RewardCalculator(RewardConfig())

        # Episode state (initialised in reset)
        self.drone_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.target_pos: np.ndarray = np.zeros(2, dtype=np.float32)
        self.battery: float = 1.0
        self.current_step: int = 0
        self.path_history: List[np.ndarray] = []
        self.total_reward: float = 0.0
        self._fig = None  # matplotlib figure for rendering

        logger.info(
            "MediReachEnv created: grid=%d, max_steps=%d, obs=%d, actions=%d",
            grid_size, max_steps,
            self.observation_space.shape[0],
            self.action_space.n,
        )

    # ───────────────────────────────────────────────────────
    #  Core Gym API
    # ───────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment for a new episode.

        Randomises start/end positions, obstacles, weather.

        Args:
            seed: Random seed for reproducibility.
            options: Optional overrides (start_pos, target_pos).

        Returns:
            Tuple of (observation, info dict).
        """
        super().reset(seed=seed)
        rng = self.np_random

        # Drone start position (randomised within inner grid)
        margin = int(self.grid_size * 0.1)
        if options and "start_pos" in options:
            self.drone_pos = np.array(options["start_pos"], dtype=np.float32)
        else:
            self.drone_pos = np.array([
                rng.integers(margin, self.grid_size - margin),
                rng.integers(margin, self.grid_size - margin),
                rng.uniform(30.0, 60.0),  # altitude in grid units
            ], dtype=np.float32)

        # Target position (ensure minimum distance)
        if options and "target_pos" in options:
            self.target_pos = np.array(options["target_pos"], dtype=np.float32)
        else:
            min_dist = self.grid_size * 0.3
            while True:
                target = np.array([
                    rng.integers(margin, self.grid_size - margin),
                    rng.integers(margin, self.grid_size - margin),
                ], dtype=np.float32)
                dist = np.linalg.norm(self.drone_pos[:2] - target)
                if dist >= min_dist:
                    self.target_pos = target
                    break

        # Reset state
        self.battery = 1.0
        self.current_step = 0
        self.path_history = [self.drone_pos.copy()]
        self.total_reward = 0.0

        # Randomise obstacles and no-fly zones
        self.obstacle_mgr.reset(
            num_obstacles=self.num_obstacles,
            num_no_fly_zones=self.num_no_fly_zones,
            rng=rng,
            drone_pos=self.drone_pos[:2],
            target_pos=self.target_pos,
        )

        # Initialise weather
        if self.fixed_weather:
            self.weather_sim.set_profile(self.fixed_weather)
        else:
            self.weather_sim.randomise(rng)

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one timestep.

        Args:
            action: Integer action (0–6).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        prev_distance = self._calculate_distance()
        prev_pos = self.drone_pos.copy()

        # Apply action
        self._apply_action(action)
        self.current_step += 1

        # Check collisions and violations
        collision = self.obstacle_mgr.check_collision(
            self.drone_pos[0], self.drone_pos[1]
        )
        no_fly_violation = self.obstacle_mgr.check_no_fly_zone(
            self.drone_pos[0], self.drone_pos[1]
        )

        # Update battery (higher drain in wind)
        wind_factor = self.weather_sim.get_wind_resistance_factor(
            self.weather_sim.get_current_conditions()["wind_speed"],
            self.weather_sim.get_current_conditions()["wind_direction"],
            self._get_heading(prev_pos, self.drone_pos),
        )
        movement_drain = _CFG.BATTERY_DRAIN_PER_STEP
        if action != DroneAction.HOVER.value:
            movement_drain += _CFG.BATTERY_DRAIN_WIND_FACTOR * wind_factor
        self.battery = max(0.0, self.battery - movement_drain)

        # Update weather (stochastic)
        self.weather_sim.step_weather()

        # Calculate distances
        current_distance = self._calculate_distance()
        reached_target = current_distance <= _CFG.TARGET_RADIUS_GRID
        battery_dead = self.battery <= 0.0
        time_exceeded = self.current_step >= self.max_steps

        # Determine obstacle proximity for hover penalty
        obstacle_near = self.obstacle_mgr.is_obstacle_nearby(
            self.drone_pos[0], self.drone_pos[1], radius=3.0
        )

        # Weather adaptation check
        weather = self.weather_sim.get_current_conditions()
        weather_adapted = (
            weather["wind_speed"] > 20.0
            and action in (DroneAction.HOVER.value, DroneAction.DESCEND.value)
        )

        # Battery efficiency check
        battery_efficient = (
            reached_target and self.battery >= 0.3
        )

        # Build info dict
        info: Dict[str, Any] = {
            "reached_target": reached_target,
            "collision": collision,
            "no_fly_violation": no_fly_violation,
            "battery_dead": battery_dead,
            "battery_level": self.battery,
            "distance_to_target": current_distance,
            "obstacle_near": obstacle_near,
            "weather_adapted": weather_adapted,
            "battery_efficient": battery_efficient,
            "step": self.current_step,
            "prev_distance": prev_distance,
            "current_distance": current_distance,
        }

        # Calculate reward
        reward = self.reward_calc.calculate_reward(
            prev_distance=prev_distance,
            current_distance=current_distance,
            action=action,
            info=info,
        )
        self.total_reward += reward

        # Terminal conditions
        terminated = reached_target or collision or no_fly_violation or battery_dead
        truncated = time_exceeded

        # Record path
        self.path_history.append(self.drone_pos.copy())

        if terminated and reached_target:
            logger.debug("Episode: TARGET REACHED in %d steps, battery=%.2f",
                         self.current_step, self.battery)
        elif terminated:
            reason = (
                "collision" if collision
                else "no_fly" if no_fly_violation
                else "battery"
            )
            logger.debug("Episode: TERMINATED (%s) at step %d", reason, self.current_step)

        info["total_reward"] = self.total_reward
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Render the current environment state.

        Returns:
            RGB array if render_mode='rgb_array', else None.
        """
        try:
            import matplotlib
            if self.render_mode == "rgb_array":
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for rendering")
            return None

        if self._fig is None:
            self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 8))

        ax = self._ax
        ax.clear()

        # Grid
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)
        ax.set_aspect("equal")
        ax.set_title(
            f"MediReach Nav — Step {self.current_step} | "
            f"Battery: {self.battery:.0%} | "
            f"Reward: {self.total_reward:.1f}"
        )
        ax.grid(True, alpha=0.2)

        # Obstacles
        for obs_pos in self.obstacle_mgr.obstacles:
            ax.plot(obs_pos[0], obs_pos[1], "rs", markersize=6, alpha=0.6)

        # No-fly zones
        for nfz in self.obstacle_mgr.no_fly_zones:
            circle = plt.Circle(
                (nfz.center_x, nfz.center_y), nfz.radius,
                color="red", alpha=0.15, linewidth=2, linestyle="--",
                fill=True,
            )
            ax.add_patch(circle)

        # Path
        if len(self.path_history) > 1:
            path = np.array(self.path_history)
            ax.plot(path[:, 0], path[:, 1], "b-", linewidth=1.5, alpha=0.7)

        # Drone
        ax.plot(
            self.drone_pos[0], self.drone_pos[1],
            "g^", markersize=14, label="Drone",
        )

        # Target
        ax.plot(
            self.target_pos[0], self.target_pos[1],
            "r*", markersize=16, label="Target",
        )

        ax.legend(loc="upper right")

        if self.render_mode == "human":
            plt.pause(0.01)
            return None
        elif self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            data = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
            return data

        return None

    def close(self) -> None:
        """Clean up rendering resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None

    # ───────────────────────────────────────────────────────
    #  Private Methods
    # ───────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build the 24-dimensional observation vector.

        Returns:
            Flat numpy array of observations.
        """
        weather = self.weather_sim.get_current_conditions()
        obstacle_sensors = self.obstacle_mgr.get_sensor_readings(
            self.drone_pos[0], self.drone_pos[1], num_directions=8
        )
        no_fly_flags = self.obstacle_mgr.get_no_fly_flags(
            self.drone_pos[0], self.drone_pos[1]
        )
        distance = self._calculate_distance()
        time_norm = self.current_step / max(self.max_steps, 1)

        obs = np.array(
            [self.drone_pos[0], self.drone_pos[1], self.drone_pos[2]]  # 3
            + [self.target_pos[0], self.target_pos[1]]                 # 2
            + [self.battery]                                            # 1
            + [weather["wind_speed"]]                                   # 1
            + [weather["wind_direction"]]                               # 1
            + [weather["rain_intensity"]]                               # 1
            + list(obstacle_sensors)                                    # 8
            + list(no_fly_flags)                                        # 5
            + [distance]                                                # 1
            + [time_norm],                                              # 1
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Build initial info dictionary."""
        return {
            "distance_to_target": self._calculate_distance(),
            "battery_level": self.battery,
            "step": self.current_step,
            "weather": self.weather_sim.get_current_conditions(),
        }

    def _apply_action(self, action: int) -> None:
        """Apply the action to update drone position.

        Clips position to stay within grid bounds.

        Args:
            action: Integer action index.
        """
        step = _CFG.STEP_SIZE_GRID
        alt_step = _CFG.ALTITUDE_STEP_M

        if action == DroneAction.MOVE_NORTH.value:
            self.drone_pos[1] += step
        elif action == DroneAction.MOVE_SOUTH.value:
            self.drone_pos[1] -= step
        elif action == DroneAction.MOVE_EAST.value:
            self.drone_pos[0] += step
        elif action == DroneAction.MOVE_WEST.value:
            self.drone_pos[0] -= step
        elif action == DroneAction.ASCEND.value:
            self.drone_pos[2] += alt_step
        elif action == DroneAction.DESCEND.value:
            self.drone_pos[2] -= alt_step
        elif action == DroneAction.HOVER.value:
            pass  # No position change

        # Clip to grid bounds
        self.drone_pos[0] = np.clip(self.drone_pos[0], 0, self.grid_size - 1)
        self.drone_pos[1] = np.clip(self.drone_pos[1], 0, self.grid_size - 1)
        self.drone_pos[2] = np.clip(self.drone_pos[2], 0, self.grid_size)

    def _calculate_distance(self) -> float:
        """Calculate 2D Euclidean distance from drone to target.

        Returns:
            Distance in grid units.
        """
        return float(np.linalg.norm(self.drone_pos[:2] - self.target_pos))

    def _get_heading(self, from_pos: np.ndarray, to_pos: np.ndarray) -> float:
        """Calculate heading angle between two positions.

        Args:
            from_pos: Origin position.
            to_pos: Destination position.

        Returns:
            Heading in degrees (0-360).
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        angle = math.degrees(math.atan2(dx, dy))
        return (angle + 360.0) % 360.0
