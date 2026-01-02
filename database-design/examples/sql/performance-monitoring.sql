-- ========================================
-- Database Performance Monitoring Queries
-- ========================================

-- ========================================
-- PostgreSQL Performance Monitoring
-- ========================================

-- Enable pg_stat_statements extension
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 1. Most executed queries
SELECT
  query,
  calls,
  total_exec_time,
  mean_exec_time,
  max_exec_time,
  rows
FROM pg_stat_statements
ORDER BY calls DESC
LIMIT 20;

-- 2. Slowest queries by total time
SELECT
  query,
  calls,
  total_exec_time,
  mean_exec_time,
  max_exec_time,
  stddev_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- 3. Slowest queries by average time
SELECT
  query,
  calls,
  mean_exec_time,
  max_exec_time,
  total_exec_time
FROM pg_stat_statements
WHERE calls > 100  -- Ignore queries with few executions
ORDER BY mean_exec_time DESC
LIMIT 20;

-- 4. Table sizes
SELECT
  schemaname,
  tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
  pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) -
                 pg_relation_size(schemaname||'.'||tablename)) AS index_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 5. Index sizes
SELECT
  schemaname,
  tablename,
  indexname,
  pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexname::regclass) DESC;

-- 6. Table access statistics
SELECT
  schemaname,
  tablename,
  seq_scan,           -- Full table scans
  seq_tup_read,       -- Rows read by full scans
  idx_scan,           -- Index scans
  idx_tup_fetch,      -- Rows fetched by index scans
  n_tup_ins,          -- Inserts
  n_tup_upd,          -- Updates
  n_tup_del,          -- Deletes
  n_live_tup,         -- Estimated live rows
  n_dead_tup,         -- Estimated dead rows
  last_vacuum,
  last_autovacuum
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY seq_scan DESC;

-- 7. Unused indexes
SELECT
  schemaname,
  tablename,
  indexname,
  idx_scan,
  pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND schemaname = 'public'
  AND indexname NOT LIKE '%_pkey'  -- Exclude primary keys
ORDER BY pg_relation_size(indexrelid) DESC;

-- 8. Index usage efficiency
SELECT
  schemaname,
  tablename,
  indexname,
  idx_scan,
  idx_tup_read,
  idx_tup_fetch,
  CASE
    WHEN idx_scan = 0 THEN 0
    ELSE ROUND((idx_tup_fetch::numeric / idx_tup_read) * 100, 2)
  END AS efficiency_percent
FROM pg_stat_user_indexes
WHERE schemaname = 'public'
  AND idx_scan > 0
ORDER BY idx_scan DESC;

-- 9. Tables with high sequential scans
SELECT
  schemaname,
  tablename,
  seq_scan,
  seq_tup_read,
  idx_scan,
  n_live_tup,
  CASE
    WHEN seq_scan = 0 THEN 0
    ELSE ROUND((seq_tup_read::numeric / seq_scan), 2)
  END AS avg_rows_per_scan
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND seq_scan > 0
ORDER BY seq_scan DESC
LIMIT 20;

-- 10. Database bloat (dead tuples)
SELECT
  schemaname,
  tablename,
  n_live_tup,
  n_dead_tup,
  CASE
    WHEN n_live_tup = 0 THEN 0
    ELSE ROUND((n_dead_tup::numeric / (n_live_tup + n_dead_tup)) * 100, 2)
  END AS bloat_percent,
  last_autovacuum
FROM pg_stat_user_tables
WHERE schemaname = 'public'
  AND n_dead_tup > 0
ORDER BY bloat_percent DESC;

-- 11. Cache hit ratio
SELECT
  sum(heap_blks_read) AS heap_read,
  sum(heap_blks_hit) AS heap_hit,
  ROUND(
    (sum(heap_blks_hit)::numeric / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0)) * 100,
    2
  ) AS cache_hit_ratio_percent
FROM pg_statio_user_tables;

-- 12. Active connections
SELECT
  datname,
  count(*) AS connections,
  max(now() - query_start) AS longest_query
FROM pg_stat_activity
WHERE state != 'idle'
GROUP BY datname
ORDER BY connections DESC;

-- 13. Long-running queries
SELECT
  pid,
  now() - query_start AS duration,
  usename,
  datname,
  state,
  query
FROM pg_stat_activity
WHERE state != 'idle'
  AND now() - query_start > interval '1 minute'
ORDER BY duration DESC;

-- 14. Blocking queries
SELECT
  blocked_locks.pid AS blocked_pid,
  blocked_activity.usename AS blocked_user,
  blocking_locks.pid AS blocking_pid,
  blocking_activity.usename AS blocking_user,
  blocked_activity.query AS blocked_statement,
  blocking_activity.query AS blocking_statement
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks
  ON blocking_locks.locktype = blocked_locks.locktype
  AND blocking_locks.database IS NOT DISTINCT FROM blocked_locks.database
  AND blocking_locks.relation IS NOT DISTINCT FROM blocked_locks.relation
  AND blocking_locks.page IS NOT DISTINCT FROM blocked_locks.page
  AND blocking_locks.tuple IS NOT DISTINCT FROM blocked_locks.tuple
  AND blocking_locks.virtualxid IS NOT DISTINCT FROM blocked_locks.virtualxid
  AND blocking_locks.transactionid IS NOT DISTINCT FROM blocked_locks.transactionid
  AND blocking_locks.classid IS NOT DISTINCT FROM blocked_locks.classid
  AND blocking_locks.objid IS NOT DISTINCT FROM blocked_locks.objid
  AND blocking_locks.objsubid IS NOT DISTINCT FROM blocked_locks.objsubid
  AND blocking_locks.pid != blocked_locks.pid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;

-- 15. Duplicate indexes
SELECT
  pg_size_pretty(sum(pg_relation_size(idx))::bigint) AS size,
  (array_agg(idx))[1] AS idx1,
  (array_agg(idx))[2] AS idx2,
  (array_agg(idx))[3] AS idx3,
  (array_agg(idx))[4] AS idx4
FROM (
  SELECT
    indexrelid::regclass AS idx,
    indrelid::regclass AS tbl,
    array_agg(attname ORDER BY attnum) AS cols
  FROM pg_index
  JOIN pg_attribute ON attrelid = indrelid AND attnum = ANY(indkey)
  WHERE indisprimary = false
  GROUP BY indexrelid, indrelid
) sub
GROUP BY tbl, cols
HAVING count(*) > 1;


-- ========================================
-- MySQL Performance Monitoring
-- ========================================

-- Enable Performance Schema (add to my.cnf)
-- [mysqld]
-- performance_schema = ON

-- 1. Most executed queries
SELECT
  DIGEST_TEXT,
  COUNT_STAR AS executions,
  SUM_TIMER_WAIT / 1000000000000 AS total_sec,
  AVG_TIMER_WAIT / 1000000000000 AS avg_sec,
  MAX_TIMER_WAIT / 1000000000000 AS max_sec
FROM performance_schema.events_statements_summary_by_digest
ORDER BY COUNT_STAR DESC
LIMIT 20;

-- 2. Slowest queries by total time
SELECT
  DIGEST_TEXT,
  COUNT_STAR AS executions,
  SUM_TIMER_WAIT / 1000000000000 AS total_sec,
  AVG_TIMER_WAIT / 1000000000000 AS avg_sec
FROM performance_schema.events_statements_summary_by_digest
ORDER BY SUM_TIMER_WAIT DESC
LIMIT 20;

-- 3. Table I/O statistics
SELECT
  OBJECT_SCHEMA,
  OBJECT_NAME,
  COUNT_READ,
  COUNT_WRITE,
  COUNT_FETCH,
  SUM_TIMER_WAIT / 1000000000000 AS total_sec
FROM performance_schema.table_io_waits_summary_by_table
WHERE OBJECT_SCHEMA = 'mydb'
ORDER BY SUM_TIMER_WAIT DESC;

-- 4. Index usage
SELECT
  OBJECT_SCHEMA,
  OBJECT_NAME,
  INDEX_NAME,
  COUNT_STAR,
  SUM_TIMER_WAIT / 1000000000000 AS total_sec
FROM performance_schema.table_io_waits_summary_by_index_usage
WHERE OBJECT_SCHEMA = 'mydb'
  AND INDEX_NAME IS NOT NULL
ORDER BY COUNT_STAR DESC;

-- 5. Table sizes
SELECT
  table_schema,
  table_name,
  ROUND(((data_length + index_length) / 1024 / 1024), 2) AS total_mb,
  ROUND((data_length / 1024 / 1024), 2) AS data_mb,
  ROUND((index_length / 1024 / 1024), 2) AS index_mb,
  table_rows
FROM information_schema.TABLES
WHERE table_schema = 'mydb'
ORDER BY (data_length + index_length) DESC;

-- 6. Active connections
SELECT
  ID,
  USER,
  HOST,
  DB,
  COMMAND,
  TIME,
  STATE,
  INFO
FROM information_schema.PROCESSLIST
WHERE COMMAND != 'Sleep'
ORDER BY TIME DESC;


-- ========================================
-- Monitoring Script Creation
-- ========================================

-- Create monitoring table to store metrics
CREATE TABLE IF NOT EXISTS performance_metrics (
  id SERIAL PRIMARY KEY,
  metric_name VARCHAR(100) NOT NULL,
  metric_value NUMERIC,
  metric_unit VARCHAR(50),
  recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_metrics_name_date
ON performance_metrics(metric_name, recorded_at DESC);

-- Function to record metrics
CREATE OR REPLACE FUNCTION record_performance_metric(
  p_metric_name VARCHAR,
  p_metric_value NUMERIC,
  p_metric_unit VARCHAR DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
  INSERT INTO performance_metrics (metric_name, metric_value, metric_unit)
  VALUES (p_metric_name, p_metric_value, p_metric_unit);
END;
$$ LANGUAGE plpgsql;

-- Example usage
SELECT record_performance_metric('cache_hit_ratio', 95.5, 'percent');
SELECT record_performance_metric('active_connections', 150, 'count');
