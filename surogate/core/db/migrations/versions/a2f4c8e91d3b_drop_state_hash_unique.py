"""drop state_hash unique constraint

Revision ID: a2f4c8e91d3b
Revises: 38b1cd7fe61c
Create Date: 2026-04-14 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a2f4c8e91d3b'
down_revision: Union[str, None] = '38b1cd7fe61c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('chat_turns') as batch_op:
        batch_op.drop_index('ix_chat_turns_state_hash')
        batch_op.create_index('ix_chat_turns_state_hash', ['state_hash'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('chat_turns') as batch_op:
        batch_op.drop_index('ix_chat_turns_state_hash')
        batch_op.create_index('ix_chat_turns_state_hash', ['state_hash'], unique=True)
