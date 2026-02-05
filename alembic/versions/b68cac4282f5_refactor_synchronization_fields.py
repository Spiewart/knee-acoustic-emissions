"""refactor_synchronization_fields

Revision ID: b68cac4282f5
Revises:
Create Date: 2026-02-05 10:34:20.040253

Refactors synchronization table fields to:
1. Remove redundant audio sync time fields
2. Clarify that biomechanics is synced to audio (audio t=0 = sync t=0)
3. Support multiple stomp detection methods (audio, consensus, biomechanics)
4. Add optional audio sync time fields for each leg
5. Remove QC version fields (QC done at other stages)

See SYNCHRONIZATION_SCHEMA_CHANGES.md for full details.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b68cac4282f5'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - refactor synchronization fields."""

    # Remove redundant fields
    op.drop_column('synchronizations', 'audio_sync_time')
    op.drop_column('synchronizations', 'aligned_audio_sync_time')
    op.drop_column('synchronizations', 'aligned_biomechanics_sync_time')
    op.drop_column('synchronizations', 'audio_qc_version')
    op.drop_column('synchronizations', 'biomech_qc_version')
    op.drop_column('synchronizations', 'cycle_qc_version')

    # Rename fields for clarity
    op.alter_column('synchronizations', 'sync_offset', new_column_name='bio_sync_offset')
    op.alter_column('synchronizations', 'selected_audio_sync_time', new_column_name='bio_selected_sync_time')
    op.alter_column('synchronizations', 'contra_selected_audio_sync_time', new_column_name='contra_bio_selected_sync_time')
    op.alter_column('synchronizations', 'audio_stomp_method', new_column_name='selected_stomp_method')

    # Add new fields
    op.add_column('synchronizations', sa.Column('aligned_sync_time', sa.Float(), nullable=True,
                                                comment='Unified aligned sync time on merged dataframes'))
    op.add_column('synchronizations', sa.Column('stomp_detection_methods', postgresql.ARRAY(sa.String()), nullable=True,
                                                comment='List of methods used: audio, consensus, biomechanics'))
    op.add_column('synchronizations', sa.Column('audio_sync_time_left', sa.Float(), nullable=True,
                                                comment='Time between mic on and participant stopping (left leg)'))
    op.add_column('synchronizations', sa.Column('audio_sync_time_right', sa.Float(), nullable=True,
                                                comment='Time between mic on and participant stopping (right leg)'))
    op.add_column('synchronizations', sa.Column('audio_sync_offset', sa.Float(), nullable=True,
                                                comment='Required if both left and right audio sync times present'))
    op.add_column('synchronizations', sa.Column('selected_audio_sync_time', sa.Float(), nullable=True,
                                                comment='Required if audio in stomp_detection_methods'))
    op.add_column('synchronizations', sa.Column('contra_selected_audio_sync_time', sa.Float(), nullable=True,
                                                comment='Required if audio in stomp_detection_methods'))


def downgrade() -> None:
    """Downgrade schema - revert synchronization field changes."""

    # Remove new fields
    op.drop_column('synchronizations', 'contra_selected_audio_sync_time')
    op.drop_column('synchronizations', 'selected_audio_sync_time')
    op.drop_column('synchronizations', 'audio_sync_offset')
    op.drop_column('synchronizations', 'audio_sync_time_right')
    op.drop_column('synchronizations', 'audio_sync_time_left')
    op.drop_column('synchronizations', 'stomp_detection_methods')
    op.drop_column('synchronizations', 'aligned_sync_time')

    # Revert renames
    op.alter_column('synchronizations', 'selected_stomp_method', new_column_name='audio_stomp_method')
    op.alter_column('synchronizations', 'contra_bio_selected_sync_time', new_column_name='contra_selected_audio_sync_time')
    op.alter_column('synchronizations', 'bio_selected_sync_time', new_column_name='selected_audio_sync_time')
    op.alter_column('synchronizations', 'bio_sync_offset', new_column_name='sync_offset')

    # Re-add removed fields
    op.add_column('synchronizations', sa.Column('cycle_qc_version', sa.String(20), nullable=True))
    op.add_column('synchronizations', sa.Column('biomech_qc_version', sa.String(20), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_qc_version', sa.String(20), nullable=True))
    op.add_column('synchronizations', sa.Column('aligned_biomechanics_sync_time', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('aligned_audio_sync_time', sa.Float(), nullable=True))
    op.add_column('synchronizations', sa.Column('audio_sync_time', sa.Float(), nullable=True))
