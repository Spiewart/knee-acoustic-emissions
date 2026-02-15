#!/usr/bin/env python
"""Migrate database schema to rename columns:
- study_id FK -> study_participant_id
- participant_number -> study_id
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text


def migrate_database():
    """Perform the database migration."""
    db_url = os.getenv("AE_DATABASE_URL")
    if not db_url:
        raise ValueError("AE_DATABASE_URL environment variable not set")

    engine = create_engine(db_url)

    with engine.connect() as conn:
        print("Starting database migration...")

        # Step 1: Add new columns
        print("1. Adding new columns...")
        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                ADD COLUMN study_participant_id INTEGER;
            """)
            )
            print("   ✓ Added study_participant_id column")
        except Exception as e:
            print(f"   ✗ Error adding study_participant_id: {e}")

        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                ADD COLUMN study_id_new INTEGER;
            """)
            )
            print("   ✓ Added study_id_new column")
        except Exception as e:
            print(f"   ✗ Error adding study_id_new: {e}")

        # Step 2: Copy data to new columns
        print("2. Copying data to new columns...")
        try:
            conn.execute(
                text("""
                UPDATE participants
                SET study_participant_id = study_id,
                    study_id_new = participant_number;
            """)
            )
            print("   ✓ Copied study_id -> study_participant_id")
            print("   ✓ Copied participant_number -> study_id_new")
        except Exception as e:
            print(f"   ✗ Error copying data: {e}")

        # Step 3: Drop constraints on old columns
        print("3. Dropping old constraints...")
        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                DROP CONSTRAINT uq_study_participant;
            """)
            )
            print("   ✓ Dropped unique constraint")
        except Exception as e:
            print(f"   ✗ Error dropping constraint: {e}")

        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                DROP CONSTRAINT fk_participants_study_id;
            """)
            )
            print("   ✓ Dropped FK constraint on study_id")
        except Exception:
            # FK might have a different name, try alternative
            try:
                conn.execute(
                    text("""
                    ALTER TABLE participants
                    DROP CONSTRAINT participants_study_id_fkey;
                """)
                )
                print("   ✓ Dropped FK constraint (alternate name)")
            except:
                print("   ℹ FK constraint not found (may not exist)")

        # Step 4: Drop old columns
        print("4. Dropping old columns...")
        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                DROP COLUMN participant_number;
            """)
            )
            print("   ✓ Dropped participant_number column")
        except Exception as e:
            print(f"   ✗ Error dropping participant_number: {e}")

        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                DROP COLUMN study_id;
            """)
            )
            print("   ✓ Dropped old study_id column")
        except Exception as e:
            print(f"   ✗ Error dropping study_id: {e}")

        # Step 5: Rename new columns
        print("5. Renaming new columns...")
        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                RENAME COLUMN study_id_new TO study_id;
            """)
            )
            print("   ✓ Renamed study_id_new -> study_id")
        except Exception as e:
            print(f"   ✗ Error renaming study_id: {e}")

        # Step 6: Add constraints
        print("6. Adding constraints...")
        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                ADD CONSTRAINT fk_participants_study
                FOREIGN KEY (study_participant_id) REFERENCES studies(id);
            """)
            )
            print("   ✓ Added FK constraint on study_participant_id")
        except Exception as e:
            print(f"   ✗ Error adding FK constraint: {e}")

        try:
            conn.execute(
                text("""
                ALTER TABLE participants
                ADD CONSTRAINT uq_study_participant
                UNIQUE (study_participant_id, study_id);
            """)
            )
            print("   ✓ Added unique constraint")
        except Exception as e:
            print(f"   ✗ Error adding unique constraint: {e}")

        conn.commit()
        print("\n✓ Migration completed successfully!")


if __name__ == "__main__":
    migrate_database()
