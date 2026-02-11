# システム設計ガイド

> システム設計はエンジニアリングの総合力。スケーラビリティ、可用性、一貫性のトレードオフを理解し、実際のシステムを設計するための体系的な知識を解説する。

## このSkillの対象者

- システム設計面接の準備をしているエンジニア
- 大規模システムの設計に関わる方
- アーキテクチャ判断の根拠を学びたい方

## 前提知識

- Web 開発の基礎知識
- データベースの基礎
- ネットワークの基礎

## 学習ガイド

### 00-fundamentals — 基礎概念

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-scalability.md]] | スケーラビリティ（水平/垂直）、CAP 定理、PACELC |
| 01 | [[docs/00-fundamentals/01-reliability.md]] | 可用性、冗長性、フェイルオーバー、SLA/SLO/SLI |
| 02 | [[docs/00-fundamentals/02-consistency-models.md]] | 強一貫性、結果整合性、因果一貫性、ACID vs BASE |
| 03 | [[docs/00-fundamentals/03-estimation.md]] | バックオブエンベロープ計算、QPS/帯域/ストレージ推定 |

### 01-building-blocks — 構成要素

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-building-blocks/00-load-balancing.md]] | ロードバランサー、アルゴリズム、L4/L7、ヘルスチェック |
| 01 | [[docs/01-building-blocks/01-caching.md]] | キャッシュ戦略、Redis、CDN、Cache Invalidation |
| 02 | [[docs/01-building-blocks/02-message-queues.md]] | メッセージキュー、Kafka、RabbitMQ、イベント駆動 |
| 03 | [[docs/01-building-blocks/03-databases.md]] | DB 選定、シャーディング、レプリケーション、NewSQL |
| 04 | [[docs/01-building-blocks/04-api-design.md]] | REST vs GraphQL vs gRPC、API バージョニング、Rate Limiting |

### 02-patterns — 設計パターン

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-patterns/00-microservices.md]] | マイクロサービス設計、サービス分割、通信パターン |
| 01 | [[docs/02-patterns/01-event-driven.md]] | イベント駆動、CQRS、Event Sourcing、Saga |
| 02 | [[docs/02-patterns/02-data-intensive.md]] | データパイプライン、バッチ/ストリーム処理、Lambda Architecture |

### 03-case-studies — ケーススタディ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-case-studies/00-url-shortener.md]] | URL 短縮サービス設計（面接頻出） |
| 01 | [[docs/03-case-studies/01-chat-system.md]] | チャットシステム設計（WebSocket、プレゼンス） |
| 02 | [[docs/03-case-studies/02-notification-system.md]] | 通知システム設計（Push/Email/SMS） |
| 03 | [[docs/03-case-studies/03-rate-limiter.md]] | レートリミッター設計（Token Bucket/Sliding Window） |
| 04 | [[docs/03-case-studies/04-search-engine.md]] | 検索エンジン設計（インデックス、ランキング） |

## クイックリファレンス

```
システム設計フレームワーク:
  1. 要件定義（機能/非機能/制約）
  2. 概算（QPS/ストレージ/帯域）
  3. 高レベル設計（コンポーネント図）
  4. 詳細設計（API/DB スキーマ/アルゴリズム）
  5. スケーラビリティ/ボトルネック対策
```

## 参考文献

1. Xu, A. "System Design Interview." ByteByteGo, 2023.
2. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
3. Fowler, M. "Patterns of Enterprise Application Architecture." Addison-Wesley, 2002.
