"""CLI helpers for database integration.

Provides utilities for setting up database connections from environment
variables and command-line arguments.
"""

import logging
import os
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)


def get_database_url(explicit_url: Optional[str] = None) -> Optional[str]:
    """Get database URL from explicit argument or environment variable.

    Args:
        explicit_url: Explicit database URL (takes precedence)

    Returns:
        Database URL or None if not configured
    """
    if explicit_url:
        return explicit_url

    url = os.getenv("AE_DATABASE_URL")
    if url:
        logger.debug("Using AE_DATABASE_URL from environment")
    return url


def create_db_session(db_url: Optional[str] = None, echo: bool = False) -> Optional[Session]:
    """Create a database session.

    Args:
        db_url: Database URL (uses AE_DATABASE_URL if not provided)
        echo: Whether to echo SQL statements

    Returns:
        SQLAlchemy Session or None if database not configured
    """
    url = get_database_url(db_url)
    if not url:
        logger.debug("Database not configured (AE_DATABASE_URL not set)")
        return None

    try:
        engine = create_engine(url, echo=echo)
        # Test connection
        with engine.connect() as conn:
            pass
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        logger.info("Database connection established")
        return session
    except Exception as e:
        logger.warning(f"Failed to create database session: {e}")
        return None


def close_db_session(session: Optional[Session]) -> None:
    """Close a database session.

    Args:
        session: Session to close (no-op if None)
    """
    if session is not None:
        try:
            session.close()
            logger.debug("Database session closed")
        except Exception as e:
            logger.warning(f"Error closing database session: {e}")
