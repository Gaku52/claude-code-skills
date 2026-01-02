"""add user profiles table

Revision ID: abc123def456
Revises: xyz789
Create Date: 2025-01-03 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'abc123def456'
down_revision = 'xyz789'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create user_profiles table with all necessary constraints and indexes"""

    # Create table
    op.create_table(
        'user_profiles',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('avatar_url', sa.String(length=500), nullable=True),
        sa.Column('website', sa.String(length=500), nullable=True),
        sa.Column('location', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'],
                                name='fk_user_profiles_user_id',
                                ondelete='CASCADE')
    )

    # Create indexes
    op.create_index(
        'ix_user_profiles_user_id',
        'user_profiles',
        ['user_id'],
        unique=True
    )

    # Create trigger for updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    op.execute("""
        CREATE TRIGGER update_user_profiles_updated_at
        BEFORE UPDATE ON user_profiles
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """)


def downgrade() -> None:
    """Drop user_profiles table and related objects"""

    # Drop trigger
    op.execute('DROP TRIGGER IF EXISTS update_user_profiles_updated_at ON user_profiles')

    # Drop function (if no other tables use it)
    # op.execute('DROP FUNCTION IF EXISTS update_updated_at_column()')

    # Drop index
    op.drop_index('ix_user_profiles_user_id', table_name='user_profiles')

    # Drop table
    op.drop_table('user_profiles')
