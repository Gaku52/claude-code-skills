# SQL とクエリマスタリー

> SQL はデータ操作の共通言語。基本構文から高度なウィンドウ関数、クエリ最適化、実行計画分析、データベース固有機能まで、SQL の全てを体系的に解説する。

## このSkillの対象者

- SQL を体系的に学びたいエンジニア
- クエリパフォーマンスを最適化したい方
- データベース設計と運用に関わる方

## 前提知識

- リレーショナルデータベースの基礎概念
- 基本的な SQL（SELECT/INSERT/UPDATE/DELETE）

## 学習ガイド

### 00-basics — SQL の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-basics/00-sql-fundamentals.md]] | SELECT/FROM/WHERE/ORDER BY/LIMIT、データ型、NULL 処理 |
| 01 | [[docs/00-basics/01-joins-and-subqueries.md]] | JOIN 全種類、サブクエリ、EXISTS、CTE（WITH句） |
| 02 | [[docs/00-basics/02-aggregation.md]] | GROUP BY、HAVING、集約関数、GROUPING SETS/CUBE/ROLLUP |
| 03 | [[docs/00-basics/03-dml-and-ddl.md]] | INSERT/UPDATE/DELETE/MERGE、CREATE/ALTER/DROP、制約 |

### 01-advanced — 高度な SQL

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-advanced/00-window-functions.md]] | ROW_NUMBER/RANK/DENSE_RANK/LEAD/LAG/SUM OVER/PARTITION BY |
| 01 | [[docs/01-advanced/01-recursive-cte.md]] | 再帰 CTE、階層データ、グラフ探索、連番生成 |
| 02 | [[docs/01-advanced/02-json-and-arrays.md]] | JSON 関数（PostgreSQL/MySQL）、配列操作、JSONB |
| 03 | [[docs/01-advanced/03-advanced-patterns.md]] | PIVOT/UNPIVOT、Gap and Island、Running Total、Median |

### 02-optimization — クエリ最適化

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-optimization/00-explain-and-execution-plan.md]] | EXPLAIN ANALYZE、実行計画の読み方、コストモデル |
| 01 | [[docs/02-optimization/01-indexing-strategy.md]] | B-Tree/Hash/GIN/GiST、複合インデックス、カバリングインデックス |
| 02 | [[docs/02-optimization/02-query-tuning.md]] | N+1 問題、バッチ処理、パーティショニング、マテリアライズドビュー |
| 03 | [[docs/02-optimization/03-database-tuning.md]] | 接続プーリング、バキューム、統計情報、設定チューニング |

### 03-specific — データベース固有機能

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-specific/00-postgresql.md]] | PostgreSQL 固有機能、拡張、pg_stat、全文検索 |
| 01 | [[docs/03-specific/01-mysql.md]] | MySQL/MariaDB 固有機能、InnoDB、レプリケーション |
| 02 | [[docs/03-specific/02-sqlite.md]] | SQLite、組み込み DB、WAL モード、Turso/LibSQL |

## クイックリファレンス

```
SQL パフォーマンスチェックリスト:
  ✓ EXPLAIN ANALYZE で実行計画を確認
  ✓ 適切なインデックスを作成
  ✓ SELECT * を避け、必要なカラムのみ
  ✓ N+1 クエリを JOIN で解決
  ✓ 大量データは LIMIT/OFFSET → カーソルベース
  ✓ 集約は DB 側で実行
```

## 参考文献

1. PostgreSQL. "Documentation." postgresql.org/docs, 2024.
2. Winand, M. "SQL Performance Explained." use-the-index-luke.com, 2012.
3. Molinaro, A. "SQL Cookbook." O'Reilly, 2020.
