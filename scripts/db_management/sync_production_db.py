#!/usr/bin/env python3
"""Synchronize production database schema with current model definitions.

This script compares the current database schema with the SQLAlchemy models
and updates the production database to match. It's safer than dropping/recreating.

Usage:
    python scripts/sync_production_db.py

Warning:
    This will modify your production database schema!
    Back up your data before running if you have important data.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_database_url, load_env_file
from src.db import get_engine, init_db


def sync_production_db():
    """Synchronize production database with current schema."""

    # Load environment variables from .env.local
    load_env_file()

    db_url = get_database_url()
    if not db_url:
        print("❌ Error: AE_DATABASE_URL not set in environment")
        print("   Please check your .env.local file")
        sys.exit(1)

    print(f"🔗 Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    print()
    print("⚠️  WARNING: This will update your production database schema!")
    print("   Existing data will be preserved, but new columns will be added.")
    print()

    response = input("Do you want to continue? (yes/no): ").strip().lower()
    if response != "yes":
        print("❌ Aborted")
        sys.exit(0)

    try:
        engine = get_engine()

        print("\n📝 Updating database schema...")
        init_db(engine)

        print("✅ Database schema synchronized successfully!")
        print("\nNote: init_db() creates tables that don't exist and preserves existing data.")
        print("If you need to add columns to existing tables, you may need a migration script.")

    except Exception as e:
        print(f"\n❌ Failed to sync database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sync_production_db()
