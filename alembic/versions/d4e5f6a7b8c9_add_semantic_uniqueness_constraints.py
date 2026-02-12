"""Add semantic uniqueness constraints to synchronizations and movement_cycles

Revision ID: d4e5f6a7b8c9
Revises: c3f7a2b91d04
Create Date: 2026-02-12

Adds knee/maneuver columns and replaces filename-based unique constraints
with semantic multi-column constraints:
- synchronizations: (participant_id, knee, maneuver, pass_number, speed)
- movement_cycles: (participant_id, knee, maneuver, pass_number, speed, cycle_index)

Also adds a partial unique index for non-walk maneuvers (where pass_number
and speed are NULL) since PostgreSQL treats each NULL combo as distinct.

NOTE: This migration is NOT backward-compatible. Flush the DB before running:
  alembic downgrade base && alembic upgrade head
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'c3f7a2b91d04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add knee/maneuver columns and semantic uniqueness constraints."""
    # ── synchronizations table ──────────────────────────────────────────

    # Add knee and maneuver columns
    op.add_column('synchronizations', sa.Column('knee', sa.String(10), nullable=False))
    op.add_column('synchronizations', sa.Column('maneuver', sa.String(20), nullable=False))

    # Add check constraints
    op.create_check_constraint('sync_valid_knee', 'synchronizations',
                               "knee IN ('left', 'right')")
    op.create_check_constraint('sync_valid_maneuver', 'synchronizations',
                               "maneuver IN ('fe', 'sts', 'walk')")

    # Drop old filename-based unique constraint
    op.drop_constraint('uq_synchronization', 'synchronizations', type_='unique')

    # Add new semantic unique constraint
    op.create_unique_constraint('uq_synchronization', 'synchronizations',
                                ['participant_id', 'knee', 'maneuver',
                                 'pass_number', 'speed'])

    # Partial unique index for non-walk maneuvers (pass_number IS NULL)
    # PostgreSQL treats NULLs as distinct in UNIQUE constraints, so
    # without this, two "left fe" records for the same participant would
    # both be allowed (both have pass_number=NULL, speed=NULL).
    op.execute(
        "CREATE UNIQUE INDEX uq_sync_nonwalk "
        "ON synchronizations (participant_id, knee, maneuver) "
        "WHERE pass_number IS NULL"
    )

    # ── movement_cycles table ───────────────────────────────────────────

    # Add knee and maneuver columns
    op.add_column('movement_cycles', sa.Column('knee', sa.String(10), nullable=False))
    op.add_column('movement_cycles', sa.Column('maneuver', sa.String(20), nullable=False))

    # Add check constraints
    op.create_check_constraint('cycle_valid_knee', 'movement_cycles',
                               "knee IN ('left', 'right')")
    op.create_check_constraint('cycle_valid_maneuver', 'movement_cycles',
                               "maneuver IN ('fe', 'sts', 'walk')")

    # Drop old filename-based unique constraint
    op.drop_constraint('uq_movement_cycle', 'movement_cycles', type_='unique')

    # Add new semantic unique constraint
    op.create_unique_constraint('uq_movement_cycle', 'movement_cycles',
                                ['participant_id', 'knee', 'maneuver',
                                 'pass_number', 'speed', 'cycle_index'])


def downgrade() -> None:
    """Remove semantic uniqueness columns and restore filename-based constraints."""
    # ── movement_cycles table ───────────────────────────────────────────
    op.drop_constraint('uq_movement_cycle', 'movement_cycles', type_='unique')
    op.drop_constraint('cycle_valid_maneuver', 'movement_cycles', type_='check')
    op.drop_constraint('cycle_valid_knee', 'movement_cycles', type_='check')
    op.drop_column('movement_cycles', 'maneuver')
    op.drop_column('movement_cycles', 'knee')

    # Restore old unique constraint
    op.create_unique_constraint('uq_movement_cycle', 'movement_cycles',
                                ['participant_id', 'cycle_file'])

    # ── synchronizations table ──────────────────────────────────────────
    op.execute("DROP INDEX IF EXISTS uq_sync_nonwalk")
    op.drop_constraint('uq_synchronization', 'synchronizations', type_='unique')
    op.drop_constraint('sync_valid_maneuver', 'synchronizations', type_='check')
    op.drop_constraint('sync_valid_knee', 'synchronizations', type_='check')
    op.drop_column('synchronizations', 'maneuver')
    op.drop_column('synchronizations', 'knee')

    # Restore old unique constraint
    op.create_unique_constraint('uq_synchronization', 'synchronizations',
                                ['participant_id', 'sync_file_name'])
