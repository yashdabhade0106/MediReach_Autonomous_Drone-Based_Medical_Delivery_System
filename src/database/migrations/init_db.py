# -*- coding: utf-8 -*-
"""
MediReach — Database Schema Initialisation.

Creates all tables and seeds default data (drone inventory,
admin user) for first-time setup.

Usage:
    python -m src.database.migrations.init_db
"""

import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dotenv import load_dotenv
from flask import Flask

from src.database.models import db, DroneInventory, User
from src.database.repositories import UserRepository
from src.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


def create_app_for_migration() -> Flask:
    """Create a minimal Flask app for database operations.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv(
        "DATABASE_URL", "sqlite:///medireach.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)
    return app


def init_database(app: Flask) -> None:
    """Create all tables defined by SQLAlchemy models.

    Args:
        app: Flask application with database configured.
    """
    with app.app_context():
        db.create_all()
        logger.info("All database tables created successfully.")


def seed_drones(app: Flask) -> None:
    """Seed the drone inventory with default units.

    Args:
        app: Flask application with database configured.
    """
    default_drones = [
        {
            "id": "DRN-001",
            "model": "MediReach-X1",
            "status": "AVAILABLE",
            "battery_percent": 100.0,
            "home_lat": 18.5204,
            "home_long": 73.8567,
            "current_lat": 18.5204,
            "current_long": 73.8567,
        },
        {
            "id": "DRN-002",
            "model": "MediReach-X1",
            "status": "AVAILABLE",
            "battery_percent": 95.0,
            "home_lat": 18.5204,
            "home_long": 73.8567,
            "current_lat": 18.5204,
            "current_long": 73.8567,
        },
        {
            "id": "DRN-003",
            "model": "MediReach-X2",
            "status": "CHARGING",
            "battery_percent": 40.0,
            "home_lat": 18.5204,
            "home_long": 73.8567,
            "current_lat": 18.5204,
            "current_long": 73.8567,
        },
    ]

    with app.app_context():
        for drone_data in default_drones:
            existing = db.session.get(DroneInventory, drone_data["id"])
            if existing is None:
                drone = DroneInventory(**drone_data)
                db.session.add(drone)
                logger.info("Seeded drone %s", drone_data["id"])
            else:
                logger.info("Drone %s already exists, skipping.", drone_data["id"])

        db.session.commit()
        logger.info("Drone inventory seeding complete.")


def seed_admin_user(app: Flask) -> None:
    """Create the default admin user if it doesn't exist.

    Args:
        app: Flask application with database configured.
    """
    with app.app_context():
        admin_username = os.getenv("ADMIN_USERNAME", "admin")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@medireach.local")
        admin_password = os.getenv("ADMIN_PASSWORD", "MediReach@2024")

        existing = User.query.filter_by(username=admin_username).first()
        if existing is None:
            UserRepository.create(
                username=admin_username,
                email=admin_email,
                password=admin_password,
                role="admin",
            )
            logger.info("Admin user '%s' created.", admin_username)
        else:
            logger.info("Admin user '%s' already exists, skipping.", admin_username)


def run_migration() -> None:
    """Execute full database initialisation and seeding."""
    logger.info("=" * 60)
    logger.info("MediReach — Database Initialisation")
    logger.info("=" * 60)

    app = create_app_for_migration()

    logger.info("Step 1/3: Creating database tables...")
    init_database(app)

    logger.info("Step 2/3: Seeding drone inventory...")
    seed_drones(app)

    logger.info("Step 3/3: Creating admin user...")
    seed_admin_user(app)

    logger.info("=" * 60)
    logger.info("Database initialisation COMPLETE.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_migration()
