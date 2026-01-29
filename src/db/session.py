"""Database session management and connection configuration."""

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.db.models import Base


def get_database_url() -> str:
    """Get database URL from environment variable.

    Returns:
        Database URL string

    Raises:
        ValueError: If AE_DATABASE_URL is not set
    """
    url = os.getenv("AE_DATABASE_URL")
    if not url:
        raise ValueError(
            "AE_DATABASE_URL environment variable not set. "
            "Please set it in .env.local (e.g., postgresql+psycopg://user:password@localhost:5432/acoustic_emissions)"
        )
    return url


def get_engine(echo: bool = False) -> Engine:
    """Create and return SQLAlchemy engine.

    Args:
        echo: If True, log all SQL statements

    Returns:
        SQLAlchemy engine instance
    """
    url = get_database_url()
    return create_engine(url, echo=echo, pool_pre_ping=True)


def init_db(engine: Engine | None = None) -> None:
    """Initialize database by creating all tables.

    Args:
        engine: SQLAlchemy engine. If None, creates new engine.
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


@contextmanager
def get_session(engine: Engine | None = None) -> Generator[Session, None, None]:
    """Context manager for database sessions.

    Args:
        engine: SQLAlchemy engine. If None, creates new engine.

    Yields:
        Database session

    Example:
        >>> with get_session() as session:
        ...     record = session.query(StudyRecord).first()
    """
    if engine is None:
        engine = get_engine()

    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
