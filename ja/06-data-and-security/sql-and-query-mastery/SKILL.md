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

### 01-advanced — 高度な SQL

| # | ファイル | 内容 |
|---|---------|------|

### 02-optimization — クエリ最適化

| # | ファイル | 内容 |
|---|---------|------|

### 03-specific — データベース固有機能

| # | ファイル | 内容 |
|---|---------|------|

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
