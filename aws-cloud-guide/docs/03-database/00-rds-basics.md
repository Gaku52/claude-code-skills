# Amazon RDS 基礎

> AWS のフルマネージドリレーショナルデータベースサービスを理解し、MySQL/PostgreSQL の運用・マルチ AZ・リードレプリカを実践的に学ぶ

## この章で学ぶこと

1. **RDS の基本アーキテクチャ** — エンジン選択、インスタンスクラス、ストレージタイプの設計判断
2. **高可用性の実現** — マルチ AZ 配置、自動フェイルオーバー、バックアップ戦略
3. **読み取りスケーリング** — リードレプリカの構築・活用パターンとレプリケーション遅延の管理

---

## 1. RDS アーキテクチャ概要

### RDS の位置づけ

```
+----------------------------------------------------------+
|                      AWS Cloud                           |
|  +----------------------------------------------------+  |
|  |                    VPC                              |  |
|  |  +--------------------+  +----------------------+  |  |
|  |  |  Public Subnet     |  |  Private Subnet      |  |  |
|  |  |  +-------------+   |  |  +----------------+  |  |  |
|  |  |  | EC2 / ECS   |   |  |  | RDS Primary    |  |  |  |
|  |  |  | (App Layer)  |-------->| (MySQL/PgSQL)  |  |  |  |
|  |  |  +-------------+   |  |  +-------+--------+  |  |  |
|  |  +--------------------+  |          |            |  |  |
|  |                          |          | Sync Repl  |  |  |
|  |  +--------------------+  |  +-------v--------+   |  |  |
|  |  |  Another AZ        |  |  | RDS Standby    |   |  |  |
|  |  |                    |  |  | (Multi-AZ)     |   |  |  |
|  |  +--------------------+  |  +----------------+   |  |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
```

### コード例 1: RDS インスタンスの作成（AWS CLI）

```bash
# MySQL 8.0 の RDS インスタンスを作成
aws rds create-db-instance \
  --db-instance-identifier my-mysql-db \
  --db-instance-class db.r6g.large \
  --engine mysql \
  --engine-version 8.0.35 \
  --master-username admin \
  --master-user-password 'SecureP@ssw0rd!' \
  --allocated-storage 100 \
  --storage-type gp3 \
  --storage-encrypted \
  --kms-key-id alias/rds-key \
  --multi-az \
  --vpc-security-group-ids sg-0abc123def456 \
  --db-subnet-group-name my-db-subnet-group \
  --backup-retention-period 7 \
  --preferred-backup-window "03:00-04:00" \
  --preferred-maintenance-window "Mon:04:00-Mon:05:00" \
  --auto-minor-version-upgrade \
  --tags Key=Environment,Value=production
```

### コード例 2: Terraform による RDS 定義

```hcl
resource "aws_db_instance" "main" {
  identifier     = "app-mysql-prod"
  engine         = "mysql"
  engine_version = "8.0.35"
  instance_class = "db.r6g.large"

  # ストレージ
  allocated_storage     = 100
  max_allocated_storage = 500   # オートスケーリング上限
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.rds.arn

  # ネットワーク
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false

  # 高可用性
  multi_az = true

  # 認証
  username = "admin"
  password = var.db_password  # Secrets Manager 推奨

  # バックアップ
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  # パラメータ
  parameter_group_name = aws_db_parameter_group.mysql80.name

  # 削除保護
  deletion_protection = true
  skip_final_snapshot = false
  final_snapshot_identifier = "app-mysql-prod-final"

  tags = {
    Environment = "production"
  }
}

resource "aws_db_parameter_group" "mysql80" {
  family = "mysql8.0"
  name   = "app-mysql80-params"

  parameter {
    name  = "character_set_server"
    value = "utf8mb4"
  }

  parameter {
    name  = "slow_query_log"
    value = "1"
  }

  parameter {
    name  = "long_query_time"
    value = "1"
  }
}
```

---

## 2. エンジン比較

### RDS 対応エンジン比較表

| エンジン | バージョン例 | 最大ストレージ | 特徴 | ユースケース |
|---|---|---|---|---|
| **MySQL** | 8.0, 8.4 | 64 TiB | 広い互換性、コミュニティ大 | Web アプリ全般 |
| **PostgreSQL** | 15, 16 | 64 TiB | 拡張性、JSON対応、GIS | 分析系、地理データ |
| **MariaDB** | 10.6, 10.11 | 64 TiB | MySQL 互換、追加機能 | MySQL 代替 |
| **Oracle** | 19c, 21c | 64 TiB | エンタープライズ機能 | 基幹系移行 |
| **SQL Server** | 2019, 2022 | 16 TiB | Windows 統合 | .NET アプリ |
| **Aurora MySQL** | 3 (MySQL 8.0互換) | 128 TiB | 高性能、自動スケール | 高負荷 Web |
| **Aurora PostgreSQL** | 15, 16 互換 | 128 TiB | 高性能、Babelfish | エンタープライズ |

### MySQL vs PostgreSQL 選定基準

| 観点 | MySQL | PostgreSQL |
|---|---|---|
| **学習コスト** | 低い | やや高い |
| **JSON 操作** | 基本的 | 高度（JSONB、インデックス対応） |
| **全文検索** | あり | 高度（tsvector/tsquery） |
| **地理空間** | 基本 | PostGIS で高度対応 |
| **パーティション** | RANGE/LIST/HASH | 宣言的パーティション |
| **レプリケーション** | binlog | WAL ベース（論理/物理） |
| **拡張性** | プラグイン | Extension で柔軟 |
| **同時接続性能** | 高い | 中〜高（接続プール推奨） |

---

## 3. ストレージタイプの選択

```
ストレージ選択フローチャート
============================

開始
 |
 v
IOPS が 3,000 以下で十分?
 |           |
 Yes         No
 |           |
 v           v
gp3        IOPS 要件は?
(汎用)      |         |
           ~64,000   ~256,000
            |         |
            v         v
          gp3       io2 Block
        (IOPS指定)  Express
```

### コード例 3: ストレージ自動スケーリングと監視

```bash
# 既存インスタンスにストレージ自動スケーリングを追加
aws rds modify-db-instance \
  --db-instance-identifier my-mysql-db \
  --max-allocated-storage 500 \
  --apply-immediately

# ストレージ使用量の監視（CloudWatch）
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name FreeStorageSpace \
  --dimensions Name=DBInstanceIdentifier,Value=my-mysql-db \
  --start-time 2026-02-10T00:00:00Z \
  --end-time 2026-02-11T00:00:00Z \
  --period 3600 \
  --statistics Average \
  --unit Bytes
```

---

## 4. マルチ AZ 配置

### フェイルオーバーの仕組み

```
正常時:
+----------+    同期レプリケーション    +----------+
| Primary  | ========================> | Standby  |
| (AZ-1a)  |                          | (AZ-1c)  |
+----+-----+                          +----------+
     ^
     |  DNS: mydb.xxxx.ap-northeast-1.rds.amazonaws.com
     |
+----+-----+
| App      |
+----------+

障害発生時:
+----------+                           +----------+
| Primary  |   X  接続断              | Standby  |
| (AZ-1a)  |                          | -> Primary|
+----------+                          +----+-----+
  障害                                      ^
                                           |  DNS 自動切替
                                           |  (60-120秒)
                                      +----+-----+
                                      | App      |
                                      +----------+
```

### コード例 4: マルチ AZ フェイルオーバーテスト

```bash
# フェイルオーバーの手動実行（テスト用）
aws rds reboot-db-instance \
  --db-instance-identifier my-mysql-db \
  --force-failover

# フェイルオーバーイベントの確認
aws rds describe-events \
  --source-type db-instance \
  --source-identifier my-mysql-db \
  --duration 60

# マルチ AZ 状態の確認
aws rds describe-db-instances \
  --db-instance-identifier my-mysql-db \
  --query 'DBInstances[0].{MultiAZ:MultiAZ,AZ:AvailabilityZone,SecondaryAZ:SecondaryAvailabilityZone}'
```

---

## 5. リードレプリカ

### コード例 5: リードレプリカの作成と活用

```bash
# リードレプリカの作成
aws rds create-db-instance-read-replica \
  --db-instance-identifier my-mysql-db-read1 \
  --source-db-instance-identifier my-mysql-db \
  --db-instance-class db.r6g.large \
  --availability-zone ap-northeast-1c

# クロスリージョンリードレプリカ
aws rds create-db-instance-read-replica \
  --db-instance-identifier my-mysql-db-us-read \
  --source-db-instance-identifier arn:aws:rds:ap-northeast-1:123456789:db:my-mysql-db \
  --db-instance-class db.r6g.large \
  --region us-east-1
```

### コード例 6: アプリケーションでの読み書き分離（Python）

```python
import pymysql
from contextlib import contextmanager

class DatabaseRouter:
    """読み書き分離を行うデータベースルーター"""

    def __init__(self):
        self.writer_config = {
            'host': 'my-mysql-db.xxxx.ap-northeast-1.rds.amazonaws.com',
            'user': 'admin',
            'password': 'secret',
            'database': 'myapp',
            'charset': 'utf8mb4',
        }
        self.reader_configs = [
            {
                'host': 'my-mysql-db-read1.xxxx.ap-northeast-1.rds.amazonaws.com',
                'user': 'readonly',
                'password': 'secret',
                'database': 'myapp',
                'charset': 'utf8mb4',
            },
            {
                'host': 'my-mysql-db-read2.xxxx.ap-northeast-1.rds.amazonaws.com',
                'user': 'readonly',
                'password': 'secret',
                'database': 'myapp',
                'charset': 'utf8mb4',
            },
        ]
        self._reader_index = 0

    @contextmanager
    def writer(self):
        """書き込み用コネクション"""
        conn = pymysql.connect(**self.writer_config)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def reader(self):
        """読み取り用コネクション（ラウンドロビン）"""
        config = self.reader_configs[self._reader_index]
        self._reader_index = (self._reader_index + 1) % len(self.reader_configs)
        conn = pymysql.connect(**config)
        try:
            yield conn
        finally:
            conn.close()

# 使用例
db = DatabaseRouter()

# 書き込みはプライマリへ
with db.writer() as conn:
    with conn.cursor() as cur:
        cur.execute("INSERT INTO users (name, email) VALUES (%s, %s)",
                    ("Taro", "taro@example.com"))

# 読み取りはリードレプリカへ
with db.reader() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (1,))
        user = cur.fetchone()
```

---

## 6. バックアップとリカバリ

### コード例 7: ポイントインタイムリカバリ

```bash
# 特定時刻の状態にリストア（新インスタンスとして作成）
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier my-mysql-db \
  --target-db-instance-identifier my-mysql-db-restored \
  --restore-time "2026-02-10T15:30:00Z" \
  --db-instance-class db.r6g.large \
  --db-subnet-group-name my-db-subnet-group \
  --vpc-security-group-ids sg-0abc123def456

# 手動スナップショットの作成
aws rds create-db-snapshot \
  --db-instance-identifier my-mysql-db \
  --db-snapshot-identifier my-mysql-db-snap-20260211

# スナップショットからの復元
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier my-mysql-db-from-snap \
  --db-snapshot-identifier my-mysql-db-snap-20260211 \
  --db-instance-class db.r6g.large
```

---

## 7. 監視とパフォーマンスチューニング

### コード例 8: Performance Insights の活用

```bash
# Performance Insights を有効化
aws rds modify-db-instance \
  --db-instance-identifier my-mysql-db \
  --enable-performance-insights \
  --performance-insights-retention-period 731 \
  --performance-insights-kms-key-id alias/rds-pi-key \
  --apply-immediately

# トップ待機イベントの取得
aws pi get-resource-metrics \
  --service-type RDS \
  --identifier db-XXXXXXXXXXXXXXXXXXXX \
  --metric-queries '[{
    "Metric": "db.load.avg",
    "GroupBy": {
      "Group": "db.wait_event",
      "Limit": 5
    }
  }]' \
  --start-time 2026-02-10T00:00:00Z \
  --end-time 2026-02-11T00:00:00Z \
  --period-in-seconds 3600
```

---

## アンチパターン

### 1. パブリックアクセス有効での運用

```
[NG] パブリックアクセス有効
=============================================
Internet --> RDS (publicly_accessible=true)
  - ポートスキャンの対象になる
  - SG 設定ミスで即座に侵害される

[OK] プライベートサブネット配置
=============================================
Internet --> ALB --> App (Private) --> RDS (Private)
  - RDS は VPC 内からのみアクセス
  - 開発者は Bastion / SSM 経由
```

**問題**: `publicly_accessible = true` にすると、インターネットから直接 RDS にアクセス可能な状態になる。セキュリティグループで制限していても、設定ミスのリスクが常に存在する。

**対策**: RDS は必ずプライベートサブネットに配置し、`publicly_accessible = false` を設定する。開発者のアクセスは SSM Session Manager や Bastion Host 経由とする。

### 2. 単一 AZ でのプロダクション運用

**問題**: コスト削減のためマルチ AZ を無効にすると、AZ 障害時にデータベースが完全に停止する。手動での復旧に数時間を要する可能性がある。

**対策**: プロダクション環境では必ず `multi_az = true` を設定する。マルチ AZ のコスト（約2倍）は、ダウンタイムのビジネスインパクトと比較すれば正当化できる。開発・ステージング環境では単一 AZ で問題ない。

---

## FAQ

### Q1: RDS と Aurora はどちらを選ぶべきですか？

**A**: 判断基準は以下の通りです。
- **RDS を選ぶ場合**: コストを抑えたい、既存の MySQL/PostgreSQL からの移行でそのままの動作を期待、シンプルな要件
- **Aurora を選ぶ場合**: 高い読み取りスループットが必要（最大15リードレプリカ）、ストレージの自動スケーリングが必要、より高速なフェイルオーバー（30秒以下）が必要
- Aurora は RDS の 3〜5 倍の性能を謳いますが、コストも高くなるため、ワークロードに応じて判断してください。

### Q2: リードレプリカのレプリケーション遅延が問題になる場合の対処法は？

**A**: 以下の対策を組み合わせます。
1. **書き込み直後の読み取り** はプライマリから行う（Read-after-Write consistency）
2. **レプリカラグの監視** を CloudWatch の `ReplicaLag` メトリクスで行い、閾値超過時にアラート
3. **インスタンスクラスのスケールアップ** でレプリカの処理能力を上げる
4. **並列レプリケーション** を有効化（MySQL: `replica_parallel_workers`）

### Q3: RDS のコストを最適化するには？

**A**: 主な最適化手法:
- **リザーブドインスタンス**: 1年/3年の予約で最大60%割引
- **インスタンスの適正化**: Performance Insights で実使用率を確認し、オーバープロビジョニングを解消
- **ストレージタイプの見直し**: gp2 から gp3 への移行で同じ IOPS をより低コストで実現
- **開発環境の停止**: 夜間・休日に不要なインスタンスを停止（最大7日間）

---

## まとめ

| 項目 | 要点 |
|---|---|
| RDS とは | フルマネージド RDB サービス。パッチ適用・バックアップ・フェイルオーバーを自動化 |
| エンジン選択 | Web アプリ → MySQL、分析・拡張性 → PostgreSQL、高性能 → Aurora |
| ストレージ | gp3 が標準。IOPS 要件が高い場合は io2 |
| マルチ AZ | プロダクションでは必須。同期レプリケーションで自動フェイルオーバー |
| リードレプリカ | 読み取り負荷分散。非同期のためレプリケーション遅延に注意 |
| バックアップ | 自動バックアップ + ポイントインタイムリカバリで RPO を最小化 |
| 監視 | Performance Insights で待機イベント分析、CloudWatch でメトリクス監視 |
| セキュリティ | プライベートサブネット配置、暗号化、IAM 認証の活用 |

## 次に読むべきガイド

- [DynamoDB](./01-dynamodb.md) — NoSQL データベースの設計と運用
- [ElastiCache](./02-elasticache.md) — キャッシュレイヤーの構築
- [VPC 基礎](../04-networking/00-vpc-basics.md) — RDS を配置するネットワーク設計

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon RDS ユーザーガイド](https://docs.aws.amazon.com/ja_jp/AmazonRDS/latest/UserGuide/) — エンジン別の詳細設定リファレンス
2. **AWS Well-Architected Framework**: [信頼性の柱 - データベース設計](https://docs.aws.amazon.com/ja_jp/wellarchitected/latest/reliability-pillar/) — 信頼性の柱におけるデータベース設計指針
3. **Amazon RDS ベストプラクティス**: [Performance Insights を使用した DB 負荷の分析](https://docs.aws.amazon.com/ja_jp/AmazonRDS/latest/UserGuide/USER_PerfInsights.html) — パフォーマンス分析の実践ガイド
