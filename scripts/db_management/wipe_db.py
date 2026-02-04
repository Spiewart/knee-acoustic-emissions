#!/usr/bin/env python
"""Wipe the database and recreate schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sqlalchemy import create_engine, text

from src.db.models import Base

DB_URL = "postgresql://spiewart@localhost/acoustic_emissions"

engine = create_engine(DB_URL)

print("Dropping all tables...")
with engine.begin() as conn:
    conn.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
    conn.execute(text("CREATE SCHEMA public"))

print("Recreating schema...")
Base.metadata.create_all(engine)

print("Database wiped and recreated successfully!")
