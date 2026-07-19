# -*- coding: utf-8 -*-
"""
MediReach — GPS & Geographic Utilities.

Provides Haversine distance, bearing calculations,
coordinate conversions, and bounding-box helpers used
by RL navigation, route optimisation, and telemetry.
"""

import math
from dataclasses import dataclass
from typing import List, Tuple

# Earth radius in metres (WGS-84 mean)
EARTH_RADIUS_M: float = 6_371_000.0


@dataclass(frozen=True)
class GeoPoint:
    """Immutable geographic coordinate."""
    latitude: float
    longitude: float
    altitude: float = 0.0


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the great-circle distance between two points.

    Uses the Haversine formula for accuracy on a spherical Earth.

    Args:
        lat1: Latitude of point 1 in decimal degrees.
        lon1: Longitude of point 1 in decimal degrees.
        lat2: Latitude of point 2 in decimal degrees.
        lon2: Longitude of point 2 in decimal degrees.

    Returns:
        Distance in metres.
    """
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    return EARTH_RADIUS_M * c


def haversine_distance_3d(
    lat1: float, lon1: float, alt1: float,
    lat2: float, lon2: float, alt2: float,
) -> float:
    """3-D distance combining horizontal Haversine with altitude difference.

    Args:
        lat1, lon1, alt1: Position 1 (degrees, degrees, metres).
        lat2, lon2, alt2: Position 2 (degrees, degrees, metres).

    Returns:
        Euclidean-approximated 3-D distance in metres.
    """
    horizontal = haversine_distance(lat1, lon1, lat2, lon2)
    vertical = abs(alt2 - alt1)
    return math.sqrt(horizontal ** 2 + vertical ** 2)


def calculate_bearing(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Calculate the initial compass bearing from point 1 to point 2.

    Args:
        lat1: Latitude of origin in decimal degrees.
        lon1: Longitude of origin in decimal degrees.
        lat2: Latitude of destination in decimal degrees.
        lon2: Longitude of destination in decimal degrees.

    Returns:
        Bearing in degrees (0–360), where 0 = North.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlon_r = math.radians(lon2 - lon1)

    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = (
        math.cos(lat1_r) * math.sin(lat2_r)
        - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    )
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360.0) % 360.0


def destination_point(
    lat: float,
    lon: float,
    bearing_deg: float,
    distance_m: float,
) -> Tuple[float, float]:
    """Calculate destination point given start, bearing, and distance.

    Args:
        lat: Start latitude in decimal degrees.
        lon: Start longitude in decimal degrees.
        bearing_deg: Bearing in degrees (0 = North).
        distance_m: Distance in metres.

    Returns:
        Tuple of (latitude, longitude) of destination in decimal degrees.
    """
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    bearing_r = math.radians(bearing_deg)
    angular_dist = distance_m / EARTH_RADIUS_M

    dest_lat = math.asin(
        math.sin(lat_r) * math.cos(angular_dist)
        + math.cos(lat_r) * math.sin(angular_dist) * math.cos(bearing_r)
    )
    dest_lon = lon_r + math.atan2(
        math.sin(bearing_r) * math.sin(angular_dist) * math.cos(lat_r),
        math.cos(angular_dist) - math.sin(lat_r) * math.sin(dest_lat),
    )

    return math.degrees(dest_lat), math.degrees(dest_lon)


def grid_to_gps(
    grid_x: int,
    grid_y: int,
    origin_lat: float,
    origin_lon: float,
    cell_size_m: float,
) -> Tuple[float, float]:
    """Convert grid coordinates to GPS (lat, lon).

    Grid X maps to East (longitude) and Grid Y maps to North (latitude).

    Args:
        grid_x: Grid column index.
        grid_y: Grid row index.
        origin_lat: Latitude of grid origin (bottom-left).
        origin_lon: Longitude of grid origin (bottom-left).
        cell_size_m: Size of each grid cell in metres.

    Returns:
        Tuple of (latitude, longitude) in decimal degrees.
    """
    north_dist = grid_y * cell_size_m
    east_dist = grid_x * cell_size_m

    lat = origin_lat + (north_dist / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon = origin_lon + (
        east_dist / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat)))
    ) * (180.0 / math.pi)

    return lat, lon


def gps_to_grid(
    lat: float,
    lon: float,
    origin_lat: float,
    origin_lon: float,
    cell_size_m: float,
) -> Tuple[int, int]:
    """Convert GPS coordinates to grid (x, y).

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        origin_lat: Latitude of grid origin (bottom-left).
        origin_lon: Longitude of grid origin (bottom-left).
        cell_size_m: Size of each grid cell in metres.

    Returns:
        Tuple of (grid_x, grid_y) as integer indices.
    """
    north_dist = haversine_distance(origin_lat, origin_lon, lat, origin_lon)
    if lat < origin_lat:
        north_dist = -north_dist

    east_dist = haversine_distance(origin_lat, origin_lon, origin_lat, lon)
    if lon < origin_lon:
        east_dist = -east_dist

    grid_y = int(round(north_dist / cell_size_m))
    grid_x = int(round(east_dist / cell_size_m))

    return grid_x, grid_y


def is_within_radius(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    radius_m: float,
) -> bool:
    """Check whether two points are within a given radius.

    Args:
        lat1, lon1: Point 1.
        lat2, lon2: Point 2.
        radius_m: Tolerance radius in metres.

    Returns:
        True if points are within radius_m of each other.
    """
    return haversine_distance(lat1, lon1, lat2, lon2) <= radius_m


def bounding_box(
    center_lat: float,
    center_lon: float,
    radius_m: float,
) -> Tuple[float, float, float, float]:
    """Calculate a bounding box around a center point.

    Args:
        center_lat: Center latitude in decimal degrees.
        center_lon: Center longitude in decimal degrees.
        radius_m: Radius in metres.

    Returns:
        Tuple of (min_lat, min_lon, max_lat, max_lon).
    """
    lat_delta = (radius_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    lon_delta = (
        radius_m / (EARTH_RADIUS_M * math.cos(math.radians(center_lat)))
    ) * (180.0 / math.pi)

    return (
        center_lat - lat_delta,
        center_lon - lon_delta,
        center_lat + lat_delta,
        center_lon + lon_delta,
    )


def point_in_polygon(
    lat: float,
    lon: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """Ray-casting algorithm for point-in-polygon test.

    Args:
        lat: Test point latitude.
        lon: Test point longitude.
        polygon: List of (lat, lon) vertices defining the polygon.

    Returns:
        True if the point is inside the polygon.
    """
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        yi, xi = polygon[i]
        yj, xj = polygon[j]

        if ((yi > lon) != (yj > lon)) and (
            lat < (xj - xi) * (lon - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i

    return inside


def estimate_flight_time(
    distance_m: float,
    speed_ms: float,
    headwind_ms: float = 0.0,
) -> float:
    """Estimate flight time accounting for headwind.

    Args:
        distance_m: Distance to travel in metres.
        speed_ms: Drone airspeed in m/s.
        headwind_ms: Headwind component in m/s (positive = opposing).

    Returns:
        Estimated flight time in seconds.

    Raises:
        ValueError: If effective speed is non-positive.
    """
    effective_speed = speed_ms - headwind_ms
    if effective_speed <= 0:
        raise ValueError(
            f"Effective speed is non-positive ({effective_speed:.1f} m/s). "
            "Wind too strong for flight."
        )
    return distance_m / effective_speed


def metres_to_km(metres: float) -> float:
    """Convert metres to kilometres."""
    return metres / 1000.0


def km_to_metres(km: float) -> float:
    """Convert kilometres to metres."""
    return km * 1000.0
