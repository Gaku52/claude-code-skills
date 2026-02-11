"""migrate user data from first_name and last_name to full_name

Revision ID: def456abc789
Revises: abc123def456
Create Date: 2025-01-03 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'def456abc789'
down_revision = 'abc123def456'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Migrate user data: first_name + last_name -> full_name"""

    # Step 1: Add new column (NULL allowed)
    op.add_column('users', sa.Column('full_name', sa.String(length=100), nullable=True))

    # Step 2: Migrate existing data in batches
    connection = op.get_bind()

    batch_size = 1000
    offset = 0

    print("Starting data migration...")

    while True:
        # Fetch batch of users
        result = connection.execute(
            sa.text("""
                SELECT id, first_name, last_name
                FROM users
                WHERE full_name IS NULL
                LIMIT :limit OFFSET :offset
            """),
            {'limit': batch_size, 'offset': offset}
        )

        rows = result.fetchall()
        if not rows:
            break

        # Update full_name for each user
        for row in rows:
            full_name = f"{row.first_name or ''} {row.last_name or ''}".strip()

            if full_name:  # Only update if not empty
                connection.execute(
                    sa.text("""
                        UPDATE users
                        SET full_name = :full_name
                        WHERE id = :user_id
                    """),
                    {'full_name': full_name, 'user_id': row.id}
                )

        connection.commit()
        offset += batch_size
        print(f"Migrated {offset} users...")

    print("Data migration completed.")

    # Step 3: Set NOT NULL constraint
    op.alter_column('users', 'full_name', nullable=False)

    # Step 4: Drop old columns (optional - can be done in a later migration)
    # op.drop_column('users', 'first_name')
    # op.drop_column('users', 'last_name')


def downgrade() -> None:
    """Restore first_name and last_name from full_name"""

    # Step 1: Add back old columns
    op.add_column('users', sa.Column('first_name', sa.String(length=50), nullable=True))
    op.add_column('users', sa.Column('last_name', sa.String(length=50), nullable=True))

    # Step 2: Restore data from full_name
    connection = op.get_bind()

    connection.execute(
        sa.text("""
            UPDATE users
            SET
                first_name = SPLIT_PART(full_name, ' ', 1),
                last_name = SPLIT_PART(full_name, ' ', 2)
            WHERE full_name IS NOT NULL
        """)
    )

    # Step 3: Set NOT NULL constraint on old columns
    op.alter_column('users', 'first_name', nullable=False)
    op.alter_column('users', 'last_name', nullable=False)

    # Step 4: Drop full_name column
    op.drop_column('users', 'full_name')
