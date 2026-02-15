"""Update cycles and audio qc columns

Revision ID: bb05b38aaebb
Revises: a8dcba1a1181
Create Date: 2026-02-06 10:24:15.301117

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'bb05b38aaebb'
down_revision: Union[str, None] = 'a8dcba1a1181'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # AudioProcessingRecord: Rename qc_artifact columns to qc_continuous_artifact
    op.alter_column('audio_processing', 'qc_artifact', new_column_name='qc_continuous_artifact')
    op.alter_column('audio_processing', 'qc_artifact_segments', new_column_name='qc_continuous_artifact_segments')
    op.alter_column('audio_processing', 'qc_artifact_ch1', new_column_name='qc_continuous_artifact_ch1')
    op.alter_column('audio_processing', 'qc_artifact_segments_ch1', new_column_name='qc_continuous_artifact_segments_ch1')
    op.alter_column('audio_processing', 'qc_artifact_ch2', new_column_name='qc_continuous_artifact_ch2')
    op.alter_column('audio_processing', 'qc_artifact_segments_ch2', new_column_name='qc_continuous_artifact_segments_ch2')
    op.alter_column('audio_processing', 'qc_artifact_ch3', new_column_name='qc_continuous_artifact_ch3')
    op.alter_column('audio_processing', 'qc_artifact_segments_ch3', new_column_name='qc_continuous_artifact_segments_ch3')
    op.alter_column('audio_processing', 'qc_artifact_ch4', new_column_name='qc_continuous_artifact_ch4')
    op.alter_column('audio_processing', 'qc_artifact_segments_ch4', new_column_name='qc_continuous_artifact_segments_ch4')

    # MovementCycleRecord: Drop old columns and add new ones
    # First rename audio_start_time/audio_end_time to start_time/end_time
    op.alter_column('movement_cycles', 'audio_start_time', new_column_name='start_time')
    op.alter_column('movement_cycles', 'audio_end_time', new_column_name='end_time')

    # Drop bio_start_time and bio_end_time (replaced by start_time/end_time)
    op.drop_column('movement_cycles', 'bio_start_time')
    op.drop_column('movement_cycles', 'bio_end_time')

    # Remove default from is_outlier (make it required without default)
    op.alter_column('movement_cycles', 'is_outlier',
                    existing_type=sa.Boolean(),
                    nullable=False,
                    server_default=None)

    # Add new QC columns
    op.add_column('movement_cycles', sa.Column('audio_qc_fail', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('movement_cycles', sa.Column('audio_qc_failures', sa.ARRAY(sa.String()), nullable=True))
    op.add_column('movement_cycles', sa.Column('audio_artifact_intermittent_fail', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('movement_cycles', sa.Column('audio_artifact_intermittent_fail_ch1', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('movement_cycles', sa.Column('audio_artifact_intermittent_fail_ch2', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('movement_cycles', sa.Column('audio_artifact_intermittent_fail_ch3', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('movement_cycles', sa.Column('audio_artifact_intermittent_fail_ch4', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('movement_cycles', sa.Column('audio_artifact_timestamps', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column('audio_artifact_timestamps_ch1', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column('audio_artifact_timestamps_ch2', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column('audio_artifact_timestamps_ch3', sa.ARRAY(sa.Float()), nullable=True))
    op.add_column('movement_cycles', sa.Column('audio_artifact_timestamps_ch4', sa.ARRAY(sa.Float()), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    # MovementCycleRecord: Remove new columns
    op.drop_column('movement_cycles', 'audio_artifact_timestamps_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_timestamps_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_timestamps_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_timestamps_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_timestamps')
    op.drop_column('movement_cycles', 'audio_artifact_intermittent_fail_ch4')
    op.drop_column('movement_cycles', 'audio_artifact_intermittent_fail_ch3')
    op.drop_column('movement_cycles', 'audio_artifact_intermittent_fail_ch2')
    op.drop_column('movement_cycles', 'audio_artifact_intermittent_fail_ch1')
    op.drop_column('movement_cycles', 'audio_artifact_intermittent_fail')
    op.drop_column('movement_cycles', 'audio_qc_failures')
    op.drop_column('movement_cycles', 'audio_qc_fail')

    # Restore default for is_outlier
    op.alter_column('movement_cycles', 'is_outlier',
                    existing_type=sa.Boolean(),
                    nullable=False,
                    server_default='false')

    # Restore bio_start_time and bio_end_time columns
    op.add_column('movement_cycles', sa.Column('bio_end_time', sa.DateTime(), nullable=True))
    op.add_column('movement_cycles', sa.Column('bio_start_time', sa.DateTime(), nullable=True))

    # Rename start_time/end_time back to audio_start_time/audio_end_time
    op.alter_column('movement_cycles', 'end_time', new_column_name='audio_end_time')
    op.alter_column('movement_cycles', 'start_time', new_column_name='audio_start_time')

    # AudioProcessingRecord: Rename qc_continuous_artifact columns back to qc_artifact
    op.alter_column('audio_processing', 'qc_continuous_artifact_segments_ch4', new_column_name='qc_artifact_segments_ch4')
    op.alter_column('audio_processing', 'qc_continuous_artifact_ch4', new_column_name='qc_artifact_ch4')
    op.alter_column('audio_processing', 'qc_continuous_artifact_segments_ch3', new_column_name='qc_artifact_segments_ch3')
    op.alter_column('audio_processing', 'qc_continuous_artifact_ch3', new_column_name='qc_artifact_ch3')
    op.alter_column('audio_processing', 'qc_continuous_artifact_segments_ch2', new_column_name='qc_artifact_segments_ch2')
    op.alter_column('audio_processing', 'qc_continuous_artifact_ch2', new_column_name='qc_artifact_ch2')
    op.alter_column('audio_processing', 'qc_continuous_artifact_segments_ch1', new_column_name='qc_artifact_segments_ch1')
    op.alter_column('audio_processing', 'qc_continuous_artifact_ch1', new_column_name='qc_artifact_ch1')
    op.alter_column('audio_processing', 'qc_continuous_artifact_segments', new_column_name='qc_artifact_segments')
    op.alter_column('audio_processing', 'qc_continuous_artifact', new_column_name='qc_artifact')
