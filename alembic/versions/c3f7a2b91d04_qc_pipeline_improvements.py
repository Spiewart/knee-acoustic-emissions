"""QC pipeline improvements: drop sync QC columns, add periodic artifact columns

Revision ID: c3f7a2b91d04
Revises: bb05b38aaebb
Create Date: 2026-02-11

Phase A: Remove unused sync_qc_fail and sync_qc_notes from synchronizations
         (sync-level QC is not meaningful; QC is done at cycle level)
Phase E: Add periodic artifact detection columns to synchronizations and
         movement_cycles (periodic artifacts detected on full exercise portion,
         propagated to individual cycles)
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3f7a2b91d04'
down_revision: Union[str, None] = 'bb05b38aaebb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Apply QC pipeline improvements."""
    # Phase A: Drop unused sync-level QC columns from synchronizations
    op.drop_column('synchronizations', 'sync_qc_fail')
    op.drop_column('synchronizations', 'sync_qc_notes')

    # Phase E: Add periodic artifact detection columns to synchronizations
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_detected', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_detected_ch1', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_detected_ch2', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_detected_ch3', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_detected_ch4', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_segments', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_segments_ch1', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_segments_ch2', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_segments_ch3', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('synchronizations', sa.Column(
        'periodic_artifact_segments_ch4', sa.ARRAY(sa.Float()), nullable=True))

    # Phase E: Add periodic artifact columns to movement_cycles
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_fail', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_fail_ch1', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_fail_ch2', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_fail_ch3', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_fail_ch4', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_timestamps', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_timestamps_ch1', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_timestamps_ch2', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_timestamps_ch3', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column(
        'audio_artifact_periodic_timestamps_ch4', sa.ARRAY(sa.Float()), nullable=True))


def downgrade() -> None:
    """Reverse QC pipeline improvements."""
    # Reverse Phase E: Drop periodic artifact columns from movement_cycles
    op.drop_column('movement_cycles', 'audio_artifact_periodic_timestamps_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_timestamps_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_timestamps_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_timestamps_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_timestamps')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_fail_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_fail_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_fail_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_fail_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_periodic_fail')

    # Reverse Phase E: Drop periodic artifact columns from synchronizations
    op.drop_column('synchronizations', 'periodic_artifact_segments_ch4')
    op.drop_column('synchronizations', 'periodic_artifact_segments_ch3')
    op.drop_column('synchronizations', 'periodic_artifact_segments_ch2')
    op.drop_column('synchronizations', 'periodic_artifact_segments_ch1')
    op.drop_column('synchronizations', 'periodic_artifact_segments')
    op.drop_column('synchronizations', 'periodic_artifact_detected_ch4')
    op.drop_column('synchronizations', 'periodic_artifact_detected_ch3')
    op.drop_column('synchronizations', 'periodic_artifact_detected_ch2')
    op.drop_column('synchronizations', 'periodic_artifact_detected_ch1')
    op.drop_column('synchronizations', 'periodic_artifact_detected')

    # Reverse Phase A: Re-add sync QC columns
    op.add_column('synchronizations', sa.Column(
        'sync_qc_fail', sa.Boolean(), nullable=False, server_default=sa.false()))
    op.add_column('synchronizations', sa.Column(
        'sync_qc_notes', sa.Text(), nullable=True))
