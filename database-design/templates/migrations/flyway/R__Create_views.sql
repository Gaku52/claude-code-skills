-- Description: Create or replace database views
-- Type: Repeatable migration (runs on every change)
-- Author: Developer

-- ========================================
-- User Statistics View
-- ========================================
CREATE OR REPLACE VIEW user_stats AS
SELECT
    u.id,
    u.username,
    u.email,
    COUNT(DISTINCT p.id) AS post_count,
    COUNT(DISTINCT c.id) AS comment_count,
    MAX(p.created_at) AS last_post_at,
    MAX(c.created_at) AS last_comment_at
FROM users u
LEFT JOIN posts p ON u.id = p.user_id
LEFT JOIN comments c ON u.id = c.user_id
GROUP BY u.id, u.username, u.email;

COMMENT ON VIEW user_stats IS 'User activity statistics';

-- ========================================
-- Published Posts View
-- ========================================
CREATE OR REPLACE VIEW published_posts AS
SELECT
    p.id,
    p.title,
    p.slug,
    p.content,
    p.published_at,
    p.created_at,
    u.id AS author_id,
    u.username AS author_username,
    COUNT(DISTINCT c.id) AS comment_count
FROM posts p
INNER JOIN users u ON p.user_id = u.id
LEFT JOIN comments c ON p.id = c.post_id
WHERE p.published_at IS NOT NULL
  AND p.published_at <= CURRENT_TIMESTAMP
GROUP BY p.id, p.title, p.slug, p.content, p.published_at, p.created_at,
         u.id, u.username;

COMMENT ON VIEW published_posts IS 'All published posts with author information';

-- ========================================
-- Recent Activity View
-- ========================================
CREATE OR REPLACE VIEW recent_activity AS
(
    SELECT
        'post' AS activity_type,
        p.id AS activity_id,
        p.user_id,
        u.username,
        p.title AS activity_title,
        p.created_at
    FROM posts p
    INNER JOIN users u ON p.user_id = u.id
)
UNION ALL
(
    SELECT
        'comment' AS activity_type,
        c.id AS activity_id,
        c.user_id,
        u.username,
        SUBSTRING(c.content, 1, 50) AS activity_title,
        c.created_at
    FROM comments c
    INNER JOIN users u ON c.user_id = u.id
)
ORDER BY created_at DESC
LIMIT 100;

COMMENT ON VIEW recent_activity IS 'Recent user activity (posts and comments)';
