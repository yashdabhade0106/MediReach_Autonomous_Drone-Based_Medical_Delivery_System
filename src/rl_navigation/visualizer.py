# -*- coding: utf-8 -*-
"""
MediReach — Route & Path Visualisation.

Generates static plots (matplotlib) and interactive maps (folium)
to visualise drone routes, training episodes, and obstacle environments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RouteVisualizer:
    """Visualisation utilities for drone navigation routes."""

    def __init__(self, output_dir: str = "outputs/visualisations") -> None:
        """Initialise visualiser.

        Args:
            output_dir: Directory to save generated visualisations.
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def plot_episode(
        self,
        path: List[np.ndarray],
        obstacles: List[np.ndarray],
        no_fly_zones: List[Any],
        target: np.ndarray,
        grid_size: int = 100,
        title: str = "MediReach Navigation Episode",
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot a single training/inference episode on the grid.

        Args:
            path: List of drone positions [x, y, alt].
            obstacles: List of obstacle positions.
            no_fly_zones: List of NoFlyZone objects.
            target: Target position [x, y].
            grid_size: Grid world size.
            title: Plot title.
            save_path: Custom save path, or auto-generated.

        Returns:
            Path to saved image, or None on error.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # Grid background
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")

        # No-fly zones
        for nfz in no_fly_zones:
            circle = plt.Circle(
                (nfz.center_x, nfz.center_y), nfz.radius,
                color="red", alpha=0.12, linewidth=1.5,
                linestyle="--", fill=True, label="No-Fly Zone",
            )
            ax.add_patch(circle)
            ax.text(
                nfz.center_x, nfz.center_y, nfz.zone_id,
                ha="center", va="center", fontsize=7, color="red",
            )

        # Obstacles
        if obstacles:
            obs_arr = np.array(obstacles)
            ax.scatter(
                obs_arr[:, 0], obs_arr[:, 1],
                c="red", s=30, marker="s", alpha=0.5,
                label="Obstacles", zorder=3,
            )

        # Flight path
        if len(path) > 1:
            path_arr = np.array(path)
            # Colour by altitude
            colours = path_arr[:, 2] if path_arr.shape[1] > 2 else None
            if colours is not None:
                scatter = ax.scatter(
                    path_arr[:, 0], path_arr[:, 1],
                    c=colours, cmap="viridis", s=10, zorder=4,
                )
                plt.colorbar(scatter, ax=ax, label="Altitude")
            ax.plot(
                path_arr[:, 0], path_arr[:, 1],
                "b-", linewidth=1.2, alpha=0.6, zorder=4,
            )

        # Start and end markers
        if len(path) > 0:
            ax.plot(
                path[0][0], path[0][1], "go",
                markersize=14, label="Start", zorder=5,
            )
        ax.plot(
            target[0], target[1], "r*",
            markersize=18, label="Target", zorder=5,
        )

        ax.legend(loc="upper right", fontsize=9)
        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = os.path.join(self.output_dir, "episode_plot.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Episode plot saved to %s", save_path)
        return save_path

    def plot_training_curve(
        self,
        rewards: List[float],
        title: str = "Training Reward Curve",
        window: int = 50,
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot training reward curve with moving average.

        Args:
            rewards: List of episode rewards.
            title: Plot title.
            window: Moving average window size.
            save_path: Custom save path.

        Returns:
            Path to saved image, or None on error.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))

        episodes = range(len(rewards))
        ax.plot(episodes, rewards, alpha=0.3, color="blue", label="Episode Reward")

        # Moving average
        if len(rewards) >= window:
            moving_avg = np.convolve(
                rewards, np.ones(window) / window, mode="valid"
            )
            ax.plot(
                range(window - 1, len(rewards)), moving_avg,
                color="red", linewidth=2,
                label=f"Moving Avg ({window})",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_curve.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Training curve saved to %s", save_path)
        return save_path

    def generate_folium_map(
        self,
        waypoints: List[Dict[str, Any]],
        pickup: Dict[str, float],
        delivery: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Generate an interactive Folium map of the delivery route.

        Args:
            waypoints: List of GPS waypoints with lat/long.
            pickup: Pickup GPS coordinates.
            delivery: Delivery GPS coordinates.
            save_path: Custom save path for HTML.

        Returns:
            Path to saved HTML file, or None on error.
        """
        try:
            import folium
        except ImportError:
            logger.warning("folium not available")
            return None

        # Centre map on midpoint
        mid_lat = (pickup["lat"] + delivery["lat"]) / 2
        mid_lon = (pickup["long"] + delivery["long"]) / 2

        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=13)

        # Pickup marker
        folium.Marker(
            [pickup["lat"], pickup["long"]],
            popup="Pickup (Pharmacy)",
            icon=folium.Icon(color="green", icon="plus-sign"),
        ).add_to(m)

        # Delivery marker
        folium.Marker(
            [delivery["lat"], delivery["long"]],
            popup="Delivery (Patient)",
            icon=folium.Icon(color="red", icon="home"),
        ).add_to(m)

        # Route polyline
        if waypoints:
            route_coords = [
                [wp["lat"], wp["long"]] for wp in waypoints
            ]
            folium.PolyLine(
                route_coords,
                color="blue",
                weight=3,
                opacity=0.7,
                tooltip="Drone Route",
            ).add_to(m)

            # Waypoint markers
            for i, wp in enumerate(waypoints):
                folium.CircleMarker(
                    [wp["lat"], wp["long"]],
                    radius=3,
                    color="blue",
                    fill=True,
                    popup=f"WP {i}: alt={wp.get('altitude', 0)}m",
                ).add_to(m)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "route_map.html")
        m.save(save_path)
        logger.info("Folium route map saved to %s", save_path)
        return save_path

    def plot_battery_profile(
        self,
        waypoints: List[Dict[str, Any]],
        save_path: Optional[str] = None,
    ) -> Optional[str]:
        """Plot battery level over the route.

        Args:
            waypoints: Waypoints with 'battery' field.
            save_path: Custom save path.

        Returns:
            Path to saved image, or None.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None

        batteries = [wp.get("battery", 100) for wp in waypoints]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.plot(batteries, "g-", linewidth=2)
        ax.axhline(y=20, color="orange", linestyle="--", label="Low Battery")
        ax.axhline(y=10, color="red", linestyle="--", label="Critical")
        ax.fill_between(range(len(batteries)), batteries, alpha=0.15, color="green")
        ax.set_xlabel("Waypoint")
        ax.set_ylabel("Battery (%)")
        ax.set_title("Battery Profile During Flight")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "battery_profile.png")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Battery profile saved to %s", save_path)
        return save_path
