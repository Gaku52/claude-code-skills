import { MigrationInterface, QueryRunner, Table, TableIndex, TableForeignKey } from 'typeorm'

export class InitialSchema1704272400000 implements MigrationInterface {
  name = 'InitialSchema1704272400000'

  public async up(queryRunner: QueryRunner): Promise<void> {
    // ========================================
    // Users Table
    // ========================================
    await queryRunner.createTable(
      new Table({
        name: 'users',
        columns: [
          {
            name: 'id',
            type: 'serial',
            isPrimary: true,
          },
          {
            name: 'username',
            type: 'varchar',
            length: '50',
            isUnique: true,
            isNullable: false,
          },
          {
            name: 'email',
            type: 'varchar',
            length: '255',
            isUnique: true,
            isNullable: false,
          },
          {
            name: 'password_hash',
            type: 'varchar',
            length: '255',
            isNullable: false,
          },
          {
            name: 'created_at',
            type: 'timestamp with time zone',
            default: 'CURRENT_TIMESTAMP',
            isNullable: false,
          },
          {
            name: 'updated_at',
            type: 'timestamp with time zone',
            default: 'CURRENT_TIMESTAMP',
            isNullable: false,
          },
        ],
      }),
      true
    )

    // Create indexes
    await queryRunner.createIndex(
      'users',
      new TableIndex({
        name: 'idx_users_email',
        columnNames: ['email'],
      })
    )

    await queryRunner.createIndex(
      'users',
      new TableIndex({
        name: 'idx_users_username',
        columnNames: ['username'],
      })
    )

    // ========================================
    // Posts Table
    // ========================================
    await queryRunner.createTable(
      new Table({
        name: 'posts',
        columns: [
          {
            name: 'id',
            type: 'serial',
            isPrimary: true,
          },
          {
            name: 'user_id',
            type: 'integer',
            isNullable: false,
          },
          {
            name: 'title',
            type: 'varchar',
            length: '255',
            isNullable: false,
          },
          {
            name: 'content',
            type: 'text',
            isNullable: true,
          },
          {
            name: 'slug',
            type: 'varchar',
            length: '255',
            isUnique: true,
            isNullable: false,
          },
          {
            name: 'published_at',
            type: 'timestamp with time zone',
            isNullable: true,
          },
          {
            name: 'created_at',
            type: 'timestamp with time zone',
            default: 'CURRENT_TIMESTAMP',
            isNullable: false,
          },
          {
            name: 'updated_at',
            type: 'timestamp with time zone',
            default: 'CURRENT_TIMESTAMP',
            isNullable: false,
          },
        ],
      }),
      true
    )

    // Foreign key
    await queryRunner.createForeignKey(
      'posts',
      new TableForeignKey({
        name: 'fk_posts_user_id',
        columnNames: ['user_id'],
        referencedTableName: 'users',
        referencedColumnNames: ['id'],
        onDelete: 'CASCADE',
      })
    )

    // Indexes
    await queryRunner.createIndex(
      'posts',
      new TableIndex({
        name: 'idx_posts_user_id',
        columnNames: ['user_id'],
      })
    )

    await queryRunner.createIndex(
      'posts',
      new TableIndex({
        name: 'idx_posts_slug',
        columnNames: ['slug'],
      })
    )

    // Partial index for published posts
    await queryRunner.query(`
      CREATE INDEX idx_posts_published_at
      ON posts(published_at)
      WHERE published_at IS NOT NULL
    `)

    // ========================================
    // Update Trigger Function
    // ========================================
    await queryRunner.query(`
      CREATE OR REPLACE FUNCTION update_updated_at_column()
      RETURNS TRIGGER AS $$
      BEGIN
        NEW.updated_at = CURRENT_TIMESTAMP;
        RETURN NEW;
      END;
      $$ language 'plpgsql';
    `)

    await queryRunner.query(`
      CREATE TRIGGER update_users_updated_at
      BEFORE UPDATE ON users
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();
    `)

    await queryRunner.query(`
      CREATE TRIGGER update_posts_updated_at
      BEFORE UPDATE ON posts
      FOR EACH ROW
      EXECUTE FUNCTION update_updated_at_column();
    `)
  }

  public async down(queryRunner: QueryRunner): Promise<void> {
    // Drop triggers
    await queryRunner.query('DROP TRIGGER IF EXISTS update_posts_updated_at ON posts')
    await queryRunner.query('DROP TRIGGER IF EXISTS update_users_updated_at ON users')
    await queryRunner.query('DROP FUNCTION IF EXISTS update_updated_at_column()')

    // Drop posts table (foreign keys are dropped automatically)
    await queryRunner.dropTable('posts', true)

    // Drop users table
    await queryRunner.dropTable('users', true)
  }
}
