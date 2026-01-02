-- ========================================
-- SQL Query Optimization Examples
-- ========================================

-- ========================================
-- 1. SELECT Optimization
-- ========================================

-- ❌ BAD: SELECT * (retrieves unnecessary columns)
SELECT * FROM users WHERE id = 1;

-- ✅ GOOD: Select only required columns
SELECT id, username, email FROM users WHERE id = 1;

-- Performance: Data transfer reduced by ~70%


-- ========================================
-- 2. Index Usage
-- ========================================

-- ❌ BAD: Function on indexed column (index not used)
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';

-- ✅ GOOD: Create function-based index
CREATE INDEX idx_users_email_lower ON users(LOWER(email));
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';

-- ✅ ALTERNATIVE: Store email in lowercase
SELECT * FROM users WHERE email = 'user@example.com';


-- ========================================
-- 3. JOIN Optimization
-- ========================================

-- ❌ BAD: Large table first
SELECT p.*, u.username
FROM posts p
JOIN users u ON p.user_id = u.id
WHERE u.username = 'admin';

-- ✅ GOOD: Filter small table first
SELECT p.*, u.username
FROM users u
JOIN posts p ON u.id = p.user_id
WHERE u.username = 'admin';

-- Performance: 850ms → 45ms (-95%)


-- ========================================
-- 4. Subquery Optimization
-- ========================================

-- ❌ BAD: Correlated subquery (executed for each row)
SELECT
  u.id,
  u.username,
  (SELECT COUNT(*) FROM posts WHERE user_id = u.id) AS post_count
FROM users u;

-- ✅ GOOD: JOIN with aggregation
SELECT
  u.id,
  u.username,
  COUNT(p.id) AS post_count
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
GROUP BY u.id, u.username;

-- Performance: 25s → 0.8s (-97%)


-- ========================================
-- 5. DISTINCT Optimization
-- ========================================

-- ❌ BAD: DISTINCT (requires sorting)
SELECT DISTINCT user_id FROM posts;

-- ✅ GOOD: GROUP BY (faster)
SELECT user_id FROM posts GROUP BY user_id;

-- ✅ ALTERNATIVE: EXISTS (for existence check)
SELECT u.id FROM users u
WHERE EXISTS (SELECT 1 FROM posts p WHERE p.user_id = u.id);


-- ========================================
-- 6. OR Condition Optimization
-- ========================================

-- ❌ BAD: OR condition (index may not be used efficiently)
SELECT * FROM users
WHERE username = 'user1' OR username = 'user2' OR username = 'user3';

-- ✅ GOOD: IN clause
SELECT * FROM users
WHERE username IN ('user1', 'user2', 'user3');

-- Performance: 450ms → 12ms (-97%)


-- ========================================
-- 7. LIKE Optimization
-- ========================================

-- ❌ BAD: Leading wildcard (index not used)
SELECT * FROM users WHERE email LIKE '%@example.com';

-- ✅ GOOD: Prefix match (index used)
SELECT * FROM users WHERE email LIKE 'user@%';

-- ✅ ALTERNATIVE: Full-text search
CREATE INDEX idx_users_email_search ON users USING GIN(to_tsvector('english', email));
SELECT * FROM users
WHERE to_tsvector('english', email) @@ to_tsquery('example.com');


-- ========================================
-- 8. COUNT Optimization
-- ========================================

-- ❌ BAD: COUNT(*) on large table (full scan)
SELECT COUNT(*) FROM posts;

-- ✅ GOOD: Approximate count (PostgreSQL)
SELECT reltuples::bigint AS estimate
FROM pg_class
WHERE relname = 'posts';

-- ✅ ALTERNATIVE: Maintain count in aggregate table
CREATE TABLE table_stats (
  table_name VARCHAR(50) PRIMARY KEY,
  row_count BIGINT,
  updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Trigger to update count
CREATE OR REPLACE FUNCTION update_table_stats()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    INSERT INTO table_stats (table_name, row_count)
    VALUES (TG_TABLE_NAME, 1)
    ON CONFLICT (table_name)
    DO UPDATE SET row_count = table_stats.row_count + 1;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE table_stats
    SET row_count = row_count - 1
    WHERE table_name = TG_TABLE_NAME;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER posts_update_stats
AFTER INSERT OR DELETE ON posts
FOR EACH ROW EXECUTE FUNCTION update_table_stats();

-- Query
SELECT row_count FROM table_stats WHERE table_name = 'posts';

-- Performance: 10,200ms → 15ms (-99.9%)


-- ========================================
-- 9. Pagination Optimization
-- ========================================

-- ❌ BAD: OFFSET (scans skipped rows)
SELECT * FROM posts
ORDER BY created_at DESC
LIMIT 10 OFFSET 10000;
-- Scans 10,010 rows to return 10

-- ✅ GOOD: Cursor-based pagination
SELECT * FROM posts
WHERE created_at < '2025-01-01 00:00:00'
ORDER BY created_at DESC
LIMIT 10;

-- ✅ ALTERNATIVE: Keyset pagination
SELECT * FROM posts
WHERE id < 12345
ORDER BY id DESC
LIMIT 10;

-- Performance: 5,500ms → 18ms (-99.7%)


-- ========================================
-- 10. Index-Only Scan
-- ========================================

-- ❌ BAD: Requires table access
CREATE INDEX idx_users_email ON users(email);
SELECT id, username, email FROM users WHERE email = 'user@example.com';

-- ✅ GOOD: Covering index (all columns in index)
CREATE INDEX idx_users_email_username_id ON users(email, username, id);
SELECT id, username, email FROM users WHERE email = 'user@example.com';

-- Performance: 45ms → 2ms (-96%)


-- ========================================
-- 11. Avoid Type Conversion
-- ========================================

-- ❌ BAD: Implicit type conversion (index not used)
CREATE TABLE products (
  id SERIAL PRIMARY KEY,
  product_code VARCHAR(20) UNIQUE
);

-- Query with integer (type conversion)
SELECT * FROM products WHERE product_code = 123;

-- ✅ GOOD: Use correct type
SELECT * FROM products WHERE product_code = '123';


-- ========================================
-- 12. Batch Operations
-- ========================================

-- ❌ BAD: Multiple individual inserts
INSERT INTO users (username, email) VALUES ('user1', 'user1@example.com');
INSERT INTO users (username, email) VALUES ('user2', 'user2@example.com');
INSERT INTO users (username, email) VALUES ('user3', 'user3@example.com');

-- ✅ GOOD: Batch insert
INSERT INTO users (username, email) VALUES
  ('user1', 'user1@example.com'),
  ('user2', 'user2@example.com'),
  ('user3', 'user3@example.com');

-- Performance: 3x faster


-- ========================================
-- 13. Transaction Optimization
-- ========================================

-- ❌ BAD: Multiple transactions
INSERT INTO users (username, email) VALUES ('user1', 'user1@example.com');
COMMIT;
INSERT INTO users (username, email) VALUES ('user2', 'user2@example.com');
COMMIT;

-- ✅ GOOD: Single transaction
BEGIN;
INSERT INTO users (username, email) VALUES ('user1', 'user1@example.com');
INSERT INTO users (username, email) VALUES ('user2', 'user2@example.com');
COMMIT;


-- ========================================
-- 14. Partial Index
-- ========================================

-- ❌ BAD: Index on all rows
CREATE INDEX idx_posts_published_at ON posts(published_at);

-- ✅ GOOD: Partial index (only published posts)
CREATE INDEX idx_posts_published_at ON posts(published_at)
WHERE published_at IS NOT NULL;

-- Index size reduction: -85%
-- Query speed: 2,500ms → 80ms (-97%)


-- ========================================
-- 15. Expression Index
-- ========================================

-- ❌ BAD: Function in WHERE clause (index not used)
SELECT * FROM products WHERE price * (1 - discount_rate) < 1000;

-- ✅ GOOD: Expression index
CREATE INDEX idx_products_discounted_price
ON products((price * (1 - discount_rate)));

SELECT * FROM products WHERE price * (1 - discount_rate) < 1000;
