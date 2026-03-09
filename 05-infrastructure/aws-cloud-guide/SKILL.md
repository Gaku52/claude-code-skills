# AWS クラウドガイド

> AWS はクラウドコンピューティングの事実上の標準。EC2/S3/Lambda の基礎からネットワーク設計、データベース選定、サーバーレスアーキテクチャ、コンテナ運用、セキュリティ、コスト最適化まで、AWS の全てを体系的に解説する。

## このSkillの対象者

- AWS を使ったインフラ構築を学びたいエンジニア
- AWS 認定資格（SAA/SAP）の取得を目指す方
- オンプレミスからクラウドへの移行を計画する方

## 前提知識

- Linux の基本操作
- ネットワークの基礎（TCP/IP、DNS、HTTP）
- コンテナの基礎知識（Docker）

## 学習ガイド

### 00-fundamentals — AWS の基礎

| # | ファイル | 内容 |
|---|---------|------|

### 01-compute — コンピューティング

| # | ファイル | 内容 |
|---|---------|------|

### 02-storage — ストレージ

| # | ファイル | 内容 |
|---|---------|------|

### 03-database — データベース

| # | ファイル | 内容 |
|---|---------|------|

### 04-networking — ネットワーキング

| # | ファイル | 内容 |
|---|---------|------|

### 05-serverless — サーバーレス

| # | ファイル | 内容 |
|---|---------|------|

### 06-containers — コンテナサービス

| # | ファイル | 内容 |
|---|---------|------|

### 07-devops — DevOps サービス

| # | ファイル | 内容 |
|---|---------|------|

### 08-security — セキュリティサービス

| # | ファイル | 内容 |
|---|---------|------|

### 09-cost-management — コスト管理

| # | ファイル | 内容 |
|---|---------|------|

## クイックリファレンス

```
AWS サービス選定チャート:

  コンピューティング:
    コンテナ → ECS Fargate（推奨）or EKS
    サーバーレス → Lambda + API Gateway
    VM → EC2 + Auto Scaling
    PaaS → Elastic Beanstalk

  データベース:
    RDB → Aurora（推奨）or RDS
    NoSQL → DynamoDB
    キャッシュ → ElastiCache Redis
    全文検索 → OpenSearch

  ストレージ:
    オブジェクト → S3
    ファイル → EFS
    ブロック → EBS

  ネットワーク:
    DNS → Route 53
    CDN → CloudFront
    ロードバランサー → ALB

  コスト削減:
    ✓ Savings Plans / Reserved Instances
    ✓ スポットインスタンス（耐障害性あるワークロード）
    ✓ S3 ライフサイクル（Glacier 移行）
    ✓ Lambda（低トラフィック時）
```

## 参考文献

1. AWS. "Documentation." docs.aws.amazon.com, 2024.
2. AWS. "Well-Architected Framework." aws.amazon.com/architecture, 2024.
3. AWS. "Pricing Calculator." calculator.aws, 2024.
