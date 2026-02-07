#!/usr/bin/env python
"""Clean up stale test data from the production database."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text


def get_db_url():
    """Get database URL from environment."""
    url = os.getenv("AE_DATABASE_URL")
    if not url:
        raise ValueError("AE_DATABASE_URL not set")
    return url

def clean_database():
    """Delete old test participant records that have wrong IDs."""
    db_url = get_db_url()
    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Check what's in the database
        result = conn.execute(text("""
            SELECT p.id, p.study_id, s.name
            FROM participants p
            JOIN studies s ON p.study_participant_id = s.id
            WHERE p.study_id = 1016
            ORDER BY p.id
        """))

        rows = result.fetchall()
        print(f"Found {len(rows)} participant(s) with study_id 1016:")
        for row in rows:
            print(f"  ID={row[0]}, study_id={row[1]}, study={row[2]}")

        if len(rows) > 1:
            print("\nMultiple records found! Deleting all but the last one...")
            for row in rows[:-1]:
                print(f"  Deleting participant ID={row[0]}")
                conn.execute(text("DELETE FROM participants WHERE id = :id"), {"id": row[0]})
            conn.commit()
            print("Done!")
        else:
            print("Only one record found, no action needed")

if __name__ == "__main__":
    try:
        clean_database()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
