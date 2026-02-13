"""Add is_active soft-delete column to downstream tables

Revision ID: g7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-02-13

Adds is_active BOOLEAN NOT NULL DEFAULT TRUE to:
- audio_processing
- biomechanics_imports
- synchronizations
- movement_cycles

Records are never deleted. Instead, records not present in the latest
processing run are marked is_active=False. Re-processing that matches
a previously-inactive record re-activates it (is_active=True) while
preserving the original PK.
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'g7b8c9d0e1f2'
down_revision = 'f6a7b8c9d0e1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ('audio_processing', 'biomechanics_imports',
                  'synchronizations', 'movement_cycles'):
        op.add_column(
            table,
            sa.Column('is_active', sa.Boolean(), nullable=False,
                       server_default=sa.text('true')),
        )


def downgrade() -> None:
    for table in ('audio_processing', 'biomechanics_imports',
                  'synchronizations', 'movement_cycles'):
        op.drop_column(table, 'is_active')
