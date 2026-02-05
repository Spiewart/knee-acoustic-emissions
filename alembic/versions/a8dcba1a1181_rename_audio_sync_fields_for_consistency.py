"""rename_audio_sync_fields_for_consistency

Revision ID: a8dcba1a1181
Revises: b68cac4282f5
Create Date: 2026-02-05 14:09:40.345963

Renames audio sync fields for consistency:
- selected_audio_sync_time -> audio_selected_sync_time
- contra_selected_audio_sync_time -> contra_audio_selected_sync_time

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a8dcba1a1181'
down_revision: Union[str, None] = 'b68cac4282f5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Rename audio sync fields for consistency."""
    # Rename selected_audio_sync_time to audio_selected_sync_time
    op.alter_column('synchronization', 'selected_audio_sync_time',
                   new_column_name='audio_selected_sync_time')

    # Rename contra_selected_audio_sync_time to contra_audio_selected_sync_time
    op.alter_column('synchronization', 'contra_selected_audio_sync_time',
                   new_column_name='contra_audio_selected_sync_time')


def downgrade() -> None:
    """Reverse the field renames."""
    # Reverse: audio_selected_sync_time back to selected_audio_sync_time
    op.alter_column('synchronization', 'audio_selected_sync_time',
                   new_column_name='selected_audio_sync_time')

    # Reverse: contra_audio_selected_sync_time back to contra_selected_audio_sync_time
    op.alter_column('synchronization', 'contra_audio_selected_sync_time',
                   new_column_name='contra_selected_audio_sync_time')
