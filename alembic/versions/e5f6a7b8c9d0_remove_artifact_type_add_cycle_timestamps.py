"""Remove redundant artifact type columns, add cycle dropout/continuous timestamps

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-02-12

Two changes:
1. Drop 5 redundant qc_artifact_type columns from audio_processing
   (these stored ["Intermittent", "Continuous"] strings that duplicated
   the separate boolean + segment columns).
2. Add 22 columns to movement_cycles for audio-stage dropout/continuous
   artifacts trimmed to cycle boundaries (11 dropout + 11 continuous,
   each with aggregate + per-channel bool fail + ARRAY(Float) timestamps).

NOTE: This migration is NOT backward-compatible. Flush the DB before running:
  alembic downgrade base && alembic upgrade head
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Drop artifact type columns, add dropout/continuous cycle timestamps."""
    # ── audio_processing: remove redundant artifact type columns ─────────
    op.drop_column('audio_processing', 'qc_artifact_type')
    op.drop_column('audio_processing', 'qc_artifact_type_ch1')
    op.drop_column('audio_processing', 'qc_artifact_type_ch2')
    op.drop_column('audio_processing', 'qc_artifact_type_ch3')
    op.drop_column('audio_processing', 'qc_artifact_type_ch4')

    # ── movement_cycles: add audio-stage dropout artifact columns ────────
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_fail', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_fail_ch1', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_fail_ch2', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_fail_ch3', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_fail_ch4', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))

    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_timestamps', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_timestamps_ch1', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_timestamps_ch2', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_timestamps_ch3', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_dropout_timestamps_ch4', sa.ARRAY(sa.Float()), nullable=True))

    # ── movement_cycles: add audio-stage continuous artifact columns ─────
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_fail', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_fail_ch1', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_fail_ch2', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_fail_ch3', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_fail_ch4', sa.Boolean(),
        nullable=False, server_default=sa.text('false')))

    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_timestamps', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_timestamps_ch1', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_timestamps_ch2', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_timestamps_ch3', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_continuous_timestamps_ch4', sa.ARRAY(sa.Float()), nullable=True))

    # Remove server defaults now that existing rows have been populated
    op.alter_column('movement_cycles', 'audio_artifact_dropout_fail',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_dropout_fail_ch1',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_dropout_fail_ch2',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_dropout_fail_ch3',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_dropout_fail_ch4',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_continuous_fail',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_continuous_fail_ch1',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_continuous_fail_ch2',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_continuous_fail_ch3',
                    server_default=None)
    op.alter_column('movement_cycles', 'audio_artifact_continuous_fail_ch4',
                    server_default=None)


def downgrade() -> None:
    """Restore artifact type columns, remove dropout/continuous cycle columns."""
    # ── movement_cycles: remove continuous artifact columns ───────────────
    op.drop_column('movement_cycles', 'audio_artifact_continuous_timestamps_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_timestamps_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_timestamps_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_timestamps_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_timestamps')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_fail_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_fail_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_fail_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_fail_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_continuous_fail')

    # ── movement_cycles: remove dropout artifact columns ─────────────────
    op.drop_column('movement_cycles', 'audio_artifact_dropout_timestamps_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_timestamps_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_timestamps_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_timestamps_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_timestamps')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_fail_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_fail_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_fail_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_fail_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_dropout_fail')

    # ── audio_processing: restore artifact type columns ──────────────────
    op.add_column('audio_processing', sa.Column(
        'qc_artifact_type', sa.ARRAY(sa.String()), nullable=True))
    op.add_column('audio_processing', sa.Column(
        'qc_artifact_type_ch1', sa.ARRAY(sa.String()), nullable=True))
    op.add_column('audio_processing', sa.Column(
        'qc_artifact_type_ch2', sa.ARRAY(sa.String()), nullable=True))
    op.add_column('audio_processing', sa.Column(
        'qc_artifact_type_ch3', sa.ARRAY(sa.String()), nullable=True))
    op.add_column('audio_processing', sa.Column(
        'qc_artifact_type_ch4', sa.ARRAY(sa.String()), nullable=True))
