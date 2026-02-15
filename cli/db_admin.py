"""Database initialization and migration utilities."""

import logging

import click

from src.db import get_engine, init_db

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Database management commands."""
    pass


@cli.command()
@click.option("--echo", is_flag=True, help="Echo SQL statements")
def init(echo: bool):
    """Initialize database schema.

    Creates all tables if they don't exist.
    Safe to run multiple times (won't drop existing data).

    Example:
        ae-db-init
        ae-db-init --echo  # Show SQL statements
    """
    try:
        engine = get_engine(echo=echo)
        init_db(engine)
        click.echo("✓ Database schema initialized successfully")
        click.echo(f"  Connected to: {engine.url}")
    except ValueError as e:
        click.echo(f"✗ Configuration error: {e}", err=True)
        click.echo("\nPlease set AE_DATABASE_URL in your .env.local file:", err=True)
        click.echo("  AE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/acoustic_emissions", err=True)
        raise click.Abort() from e
    except Exception as e:
        click.echo(f"✗ Database initialization failed: {e}", err=True)
        raise


@cli.command()
@click.option("--echo", is_flag=True, help="Echo SQL statements")
def check(echo: bool):
    """Check database connection.

    Verifies that the database is accessible.

    Example:
        ae-db-check
    """
    try:
        engine = get_engine(echo=echo)
        with engine.connect():
            click.echo("✓ Database connection successful")
            click.echo(f"  Connected to: {engine.url}")
    except ValueError as e:
        click.echo(f"✗ Configuration error: {e}", err=True)
        click.echo("\nPlease set AE_DATABASE_URL in your .env.local file:", err=True)
        click.echo("  AE_DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/acoustic_emissions", err=True)
        raise click.Abort() from e
    except Exception as e:
        click.echo(f"✗ Database connection failed: {e}", err=True)
        raise


if __name__ == "__main__":
    cli()
