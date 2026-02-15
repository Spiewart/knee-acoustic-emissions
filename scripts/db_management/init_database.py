#!/usr/bin/env python3
"""Database initialization and testing script.

This script helps set up and verify the PostgreSQL database for acoustic emissions processing.

By default, operations target the production database (AE_DATABASE_URL).
Use --test flag to target the test database instead.

Usage:
    # Initialize production database
    python scripts/init_database.py --init

    # Initialize test database
    python scripts/init_database.py --init --test

    # Test connection to production database
    python scripts/init_database.py --test-connection

    # Create sample data in test database
    python scripts/init_database.py --sample --test

    # Drop all tables (DANGER!)
    python scripts/init_database.py --drop
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# Load environment variables from .env.local
from dotenv import load_dotenv

load_dotenv(repo_root / ".env.local")

from src.db import Base, get_engine, get_session, init_db
from src.db.models import AudioProcessingRecord, ParticipantRecord, StudyRecord
from src.db.repository import Repository
from src.metadata import AudioProcessing


def get_database_url(use_test_db: bool = False):
    """Get the appropriate database URL.

    Args:
        use_test_db: If True, return test database URL
    """
    if use_test_db:
        # Check for explicit test database URL
        test_url = os.getenv("AE_TEST_DATABASE_URL")
        if test_url:
            return test_url

        # Fall back to production URL with _test suffix
        prod_url = os.getenv("AE_DATABASE_URL")
        if prod_url and "acoustic_emissions" in prod_url:
            return prod_url.replace("acoustic_emissions", "acoustic_emissions_test")

        # Default test database
        return "postgresql+psycopg://postgres@localhost/acoustic_emissions_test"
    else:
        # Production database
        prod_url = os.getenv("AE_DATABASE_URL")
        if not prod_url:
            raise RuntimeError("AE_DATABASE_URL not set. Please configure in .env.local")
        return prod_url


def init_database(use_test_db: bool = False):
    """Initialize database by creating all tables."""
    db_type = "test" if use_test_db else "production"
    print(f"Initializing {db_type} database...")
    try:
        db_url = get_database_url(use_test_db)
        print(f"  Database: {db_url.split('@')[-1]}")  # Show host/db without credentials

        engine = get_engine()
        # Override engine URL if using test database
        if use_test_db:
            from sqlalchemy import create_engine as _create_engine

            engine = _create_engine(db_url)

        init_db(engine)
        print("✅ Database initialized successfully!")
        print(f"   Tables created: {', '.join(Base.metadata.tables.keys())}")
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        sys.exit(1)


def test_connection():
    """Test database connection."""
    print("Testing database connection...")
    try:
        engine = get_engine()
        with engine.connect() as conn:
            print("✅ Successfully connected to database!")
            print(f"   Database URL: {engine.url}")

        # Try a simple query
        with get_session(engine) as session:
            study_count = session.query(StudyRecord).count()
            participant_count = session.query(ParticipantRecord).count()
            audio_count = session.query(AudioProcessingRecord).count()

            print("   Current records:")
            print(f"     • Studies: {study_count}")
            print(f"     • Participants: {participant_count}")
            print(f"     • Audio files: {audio_count}")

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nMake sure:")
        print("  1. PostgreSQL is running")
        print("  2. AE_DATABASE_URL is set in .env.local")
        print("  3. Database exists (createdb acoustic_emissions)")
        sys.exit(1)


def create_sample_data():
    """Create sample data for testing."""
    print("Creating sample data...")
    try:
        engine = get_engine()

        # Create sample audio processing record
        audio = AudioProcessing(
            study="AOA",
            study_id=9999,  # Test participant
            audio_file_name="sample_test.bin",
            device_serial="TEST123",
            firmware_version=1,
            file_time=datetime(2024, 1, 1, 10, 0, 0),
            file_size_mb=100.0,
            recording_date=datetime(2024, 1, 1),
            recording_time=datetime(2024, 1, 1, 10, 0, 0),
            knee="left",
            maneuver="walk",
            num_channels=4,
            sample_rate=46875.0,
            mic_1_position="IPM",
            mic_2_position="IPL",
            mic_3_position="SPM",
            mic_4_position="SPL",
            linked_biomechanics=False,
            pass_number=1,
            speed="normal",
        )

        with get_session(engine) as session:
            repo = Repository(session)
            record = repo.save_audio_processing(audio)
            print(f"✅ Created sample audio processing record (ID: {record.id})")
            print(f"   Study: {audio.study}")
            print(f"   Participant: {audio.study_id}")
            print(f"   Audio file: {audio.audio_file_name}")
            print(f"   Knee: {audio.knee}, Maneuver: {audio.maneuver}")

    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        sys.exit(1)


def drop_database():
    """Drop all tables (DANGER!)."""
    print("⚠️  WARNING: This will delete ALL data from the database!")
    response = input("Type 'YES' to confirm: ")

    if response != "YES":
        print("Aborted.")
        return

    print("Dropping all tables...")
    try:
        engine = get_engine()
        Base.metadata.drop_all(engine)
        print("✅ All tables dropped successfully!")
    except Exception as e:
        print(f"❌ Failed to drop tables: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Database initialization and testing tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize production database
  python scripts/init_database.py --init

  # Initialize test database
  python scripts/init_database.py --init --test

  # Test connection to production database
  python scripts/init_database.py --test-connection

  # Create sample data in test database
  python scripts/init_database.py --sample --test
        """,
    )

    parser.add_argument("--init", action="store_true", help="Initialize database (create all tables)")
    parser.add_argument("--test-connection", action="store_true", help="Test database connection")
    parser.add_argument("--sample", action="store_true", help="Create sample data for testing")
    parser.add_argument("--drop", action="store_true", help="Drop all tables (DANGER!)")
    parser.add_argument("--test", action="store_true", help="Target test database instead of production")

    args = parser.parse_args()

    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    use_test_db = args.test

    # Execute requested operations in order
    if args.drop:
        drop_database(use_test_db)

    if args.init:
        init_database(use_test_db)

    if args.test_connection:
        test_connection(use_test_db)

    if args.sample:
        create_sample_data(use_test_db)


if __name__ == "__main__":
    main()
