#!/usr/bin/env python3
"""Migration script to add biomechanics_import_id column to audio_processing table.

This migration adds the foreign key column that links audio processing records
to biomechanics import records.

Usage:
    python scripts/migrate_add_biomechanics_import_id.py

Environment:
    Requires AE_DATABASE_URL to be set (production database)
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text

from src.config import get_database_url, load_env_file
from src.db import get_engine


def migrate_add_biomechanics_import_id():
    """Add biomechanics_import_id column to audio_processing table."""

    # Load environment variables from .env.local
    load_env_file()

    db_url = get_database_url()
    if not db_url:
        print("❌ Error: AE_DATABASE_URL environment variable not set")
        print("   Please set it to your production database URL")
        sys.exit(1)

    print(f"🔗 Connecting to database: {db_url.split('@')[-1] if '@' in db_url else db_url}")

    try:
        engine = get_engine()

        with engine.begin() as conn:
            # Check if column already exists
            result = conn.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name='audio_processing'
                AND column_name='biomechanics_import_id'
            """))

            if result.fetchone():
                print("✅ Column 'biomechanics_import_id' already exists in audio_processing table")
                return

            print("📝 Adding biomechanics_import_id column to audio_processing table...")

            # Add the column
            conn.execute(text("""
                ALTER TABLE audio_processing
                ADD COLUMN biomechanics_import_id INTEGER
            """))

            # Add foreign key constraint
            conn.execute(text("""
                ALTER TABLE audio_processing
                ADD CONSTRAINT fk_audio_biomechanics
                FOREIGN KEY (biomechanics_import_id)
                REFERENCES biomechanics_import(id)
                ON DELETE SET NULL
            """))

            print("✅ Successfully added biomechanics_import_id column")
            print("✅ Successfully added foreign key constraint")

        print("\n🎉 Migration completed successfully!")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    migrate_add_biomechanics_import_id()
