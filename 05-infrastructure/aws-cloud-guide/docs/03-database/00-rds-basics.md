# Amazon RDS 基礎

> AWS のフルマネージドリレーショナルデータベースサービスを理解し、MySQL/PostgreSQL の運用・マルチ AZ・リードレプリカを実践的に学ぶ

## この章で学ぶこと

1. **RDS の基本アーキテクチャ** — エンジン選択、インスタンスクラス、ストレージタイプの設計判断
2. **高可用性の実現** — マルチ AZ 配置、自動フェイルオーバー、バックアップ戦略
3. **読み取りスケーリング** — リードレプリカの構築・活用パターンとレプリケーション遅延の管理
4. **セキュリティ設計** — VPC 配置、暗号化、IAM 認証、監査ログの設定
5. **Infrastructure as Code** — CloudFormation / CDK による RDS の宣言的管理

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

### RDS のマネージド範囲

```
+-------------------------------+-------------------------------+
|      ユーザーの責任             |      RDS が管理                |
+-------------------------------+-------------------------------+
| アプリケーションの最適化        | OS パッチ適用                  |
| クエリチューニング             | データベースエンジンの更新       |
| スキーマ設計                   | 自動バックアップ               |
| インデックス管理               | スナップショット管理            |
| パラメータグループのチューニング | マルチ AZ フェイルオーバー      |
| セキュリティグループ設定        | ストレージの自動スケーリング    |
| バックアップ保持期間の決定      | ヘルスモニタリング             |
| 暗号化設定                    | ハードウェア障害時の自動復旧    |
+-------------------------------+-------------------------------+
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
  --enable-performance-insights \
  --performance-insights-retention-period 731 \
  --monitoring-interval 60 \
  --monitoring-role-arn arn:aws:iam::123456789012:role/rds-monitoring-role \
  --enable-cloudwatch-logs-exports '["audit","error","general","slowquery"]' \
  --copy-tags-to-snapshot \
  --deletion-protection \
  --tags Key=Environment,Value=production Key=Team,Value=backend

# 作成状態の確認
aws rds wait db-instance-available \
  --db-instance-identifier my-mysql-db

# エンドポイントの取得
aws rds describe-db-instances \
  --db-instance-identifier my-mysql-db \
  --query 'DBInstances[0].Endpoint.{Address:Address,Port:Port}'
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

  # 監視
  performance_insights_enabled          = true
  performance_insights_retention_period = 731
  monitoring_interval                   = 60
  monitoring_role_arn                   = aws_iam_role.rds_monitoring.arn
  enabled_cloudwatch_logs_exports       = ["audit", "error", "slowquery"]

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
    name  = "collation_server"
    value = "utf8mb4_unicode_ci"
  }

  parameter {
    name  = "slow_query_log"
    value = "1"
  }

  parameter {
    name  = "long_query_time"
    value = "1"
  }

  parameter {
    name  = "log_output"
    value = "FILE"
  }

  parameter {
    name         = "innodb_buffer_pool_size"
    value        = "{DBInstanceClassMemory*3/4}"
    apply_method = "pending-reboot"
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "app-db-subnet-group"
  subnet_ids = var.private_subnet_ids

  tags = {
    Name = "app-db-subnet-group"
  }
}

resource "aws_security_group" "rds" {
  name_prefix = "rds-"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 3306
    to_port         = 3306
    protocol        = "tcp"
    security_groups = [var.app_security_group_id]
    description     = "MySQL from application layer"
  }

  tags = {
    Name = "rds-security-group"
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
| **Window 関数** | 8.0 で対応 | 高度に対応 |
| **CTE (再帰)** | 8.0 で対応 | 早期から対応 |

### インスタンスクラスの選択

```
インスタンスクラス選定フロー
==============================

開始
 |
 v
本番環境?
 |           |
 Yes         No (開発/テスト)
 |           |
 v           v
メモリ集約型  汎用インスタンス
db.r6g/r7g   db.t3/t4g
 |           (バースト対応)
 |
 v
Graviton 対応エンジン?
 |           |
 Yes         No
 |           |
 v           v
db.r7g      db.r6i
(コスパ最良)  (Intel)

インスタンスクラスの命名規則:
  db.r6g.2xlarge
  |  | | |
  |  | | +-- サイズ (large, xlarge, 2xlarge, ...)
  |  | +---- プロセッサ (g=Graviton, i=Intel, なし=デフォルト)
  |  +------ 世代 (6, 7)
  +--------- ファミリー (r=メモリ最適化, m=汎用, t=バースト)
```

| クラス | vCPU | メモリ | 用途 | 月額概算 (東京) |
|-------|------|--------|------|---------------|
| db.t3.micro | 2 | 1 GiB | 開発・テスト | ~$25 |
| db.t3.medium | 2 | 4 GiB | 小規模本番 | ~$100 |
| db.r6g.large | 2 | 16 GiB | 中規模本番 | ~$250 |
| db.r6g.xlarge | 4 | 32 GiB | 大規模本番 | ~$500 |
| db.r6g.2xlarge | 8 | 64 GiB | 高負荷本番 | ~$1,000 |
| db.r7g.4xlarge | 16 | 128 GiB | 大規模エンタープライズ | ~$2,000 |

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

### ストレージタイプ詳細比較

| 項目 | gp3 | gp2 (旧) | io1 | io2 | io2 Block Express |
|------|-----|---------|-----|-----|-------------------|
| ベースライン IOPS | 3,000 | 容量比例 | 指定 | 指定 | 指定 |
| 最大 IOPS | 16,000 | 16,000 | 64,000 | 64,000 | 256,000 |
| 最大スループット | 1,000 MiB/s | 250 MiB/s | 1,000 MiB/s | 1,000 MiB/s | 4,000 MiB/s |
| IOPS/GiB 比 | 独立 | 3:1 | 50:1 | 500:1 | 1,000:1 |
| 耐久性 | 99.8-99.9% | 99.8-99.9% | 99.8-99.9% | 99.999% | 99.999% |
| コスト | 最安 | やや高い | 高い | 高い | 非常に高い |

### コード例 3: ストレージ自動スケーリングと監視

```bash
# 既存インスタンスにストレージ自動スケーリングを追加
aws rds modify-db-instance \
  --db-instance-identifier my-mysql-db \
  --max-allocated-storage 500 \
  --apply-immediately

# gp2 から gp3 への移行（コスト削減）
aws rds modify-db-instance \
  --db-instance-identifier my-mysql-db \
  --storage-type gp3 \
  --iops 3000 \
  --storage-throughput 125 \
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

# ストレージ使用量のアラーム設定（残り 10GB で通知）
aws cloudwatch put-metric-alarm \
  --alarm-name "RDS-FreeStorage-Low" \
  --metric-name FreeStorageSpace \
  --namespace AWS/RDS \
  --dimensions Name=DBInstanceIdentifier,Value=my-mysql-db \
  --statistic Average \
  --period 300 \
  --evaluation-periods 3 \
  --threshold 10737418240 \
  --comparison-operator LessThanThreshold \
  --alarm-actions "arn:aws:sns:ap-northeast-1:123456789012:alerts" \
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

マルチ AZ クラスター（新方式）:
+----------+    同期    +-----------+    同期    +-----------+
| Writer   | ========> | Reader 1  | ========> | Reader 2  |
| (AZ-1a)  |           | (AZ-1c)   |           | (AZ-1d)   |
+----------+           +-----------+           +-----------+
  ↑ 書き込み              ↑ 読み取り               ↑ 読み取り
  rw-endpoint             ro-endpoint             ro-endpoint

  フェイルオーバー時間: 約 35 秒（従来のマルチ AZ より高速）
```

### マルチ AZ インスタンス vs マルチ AZ クラスター

| 項目 | マルチ AZ インスタンス | マルチ AZ クラスター |
|------|---------------------|-------------------|
| Standby | 1 台（読み取り不可） | 2 台（読み取り可能） |
| フェイルオーバー | 60-120 秒 | 約 35 秒 |
| 読み取りエンドポイント | なし | あり |
| 対応エンジン | 全エンジン | MySQL, PostgreSQL |
| ストレージ | EBS | ローカル NVMe + EBS |
| コスト | 約 2 倍 | 約 3 倍 |
| ユースケース | 標準的な HA | 高性能 HA + 読み取りスケール |

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

# EventBridge でフェイルオーバーを検知
aws events put-rule \
  --name rds-failover-notification \
  --event-pattern '{
    "source": ["aws.rds"],
    "detail-type": ["RDS DB Instance Event"],
    "detail": {
      "EventCategories": ["failover"],
      "SourceType": ["DB_INSTANCE"]
    }
  }'

aws events put-targets \
  --rule rds-failover-notification \
  --targets '[{
    "Id": "sns-target",
    "Arn": "arn:aws:sns:ap-northeast-1:123456789012:alerts"
  }]'
```

---

## 5. リードレプリカ

### リードレプリカのアーキテクチャ

```
リードレプリカ構成

+----------+   非同期レプリケーション   +-----------+
| Primary  | =======================> | Read      |
| (Writer) |                          | Replica 1 |
|          |                          | (AZ-1c)   |
|          |                          +-----------+
|          |
|          |   非同期レプリケーション   +-----------+
|          | =======================> | Read      |
|          |                          | Replica 2 |
|          |                          | (AZ-1d)   |
+----------+                          +-----------+

  ↑ 書き込み                            ↑ 読み取り
  (プライマリエンドポイント)              (各レプリカのエンドポイント)

注意: 非同期のため、レプリケーション遅延が発生する
      ReplicaLag メトリクスで監視が必要
```

### コード例 5: リードレプリカの作成と活用

```bash
# リードレプリカの作成
aws rds create-db-instance-read-replica \
  --db-instance-identifier my-mysql-db-read1 \
  --source-db-instance-identifier my-mysql-db \
  --db-instance-class db.r6g.large \
  --availability-zone ap-northeast-1c \
  --storage-type gp3 \
  --max-allocated-storage 500 \
  --enable-performance-insights \
  --performance-insights-retention-period 731 \
  --monitoring-interval 60 \
  --monitoring-role-arn arn:aws:iam::123456789012:role/rds-monitoring-role

# 2 台目のリードレプリカ
aws rds create-db-instance-read-replica \
  --db-instance-identifier my-mysql-db-read2 \
  --source-db-instance-identifier my-mysql-db \
  --db-instance-class db.r6g.large \
  --availability-zone ap-northeast-1d

# クロスリージョンリードレプリカ（DR 用）
aws rds create-db-instance-read-replica \
  --db-instance-identifier my-mysql-db-us-read \
  --source-db-instance-identifier arn:aws:rds:ap-northeast-1:123456789:db:my-mysql-db \
  --db-instance-class db.r6g.large \
  --region us-east-1

# レプリケーション遅延の監視
aws cloudwatch get-metric-statistics \
  --namespace AWS/RDS \
  --metric-name ReplicaLag \
  --dimensions Name=DBInstanceIdentifier,Value=my-mysql-db-read1 \
  --start-time "$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 60 \
  --statistics Average

# レプリケーション遅延アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "RDS-ReplicaLag-High" \
  --metric-name ReplicaLag \
  --namespace AWS/RDS \
  --dimensions Name=DBInstanceIdentifier,Value=my-mysql-db-read1 \
  --statistic Average \
  --period 60 \
  --evaluation-periods 3 \
  --threshold 30 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions "arn:aws:sns:ap-northeast-1:123456789012:alerts"

# リードレプリカをスタンドアロン DB に昇格
aws rds promote-read-replica \
  --db-instance-identifier my-mysql-db-read1
```

### コード例 6: アプリケーションでの読み書き分離（Python）

```python
import pymysql
from contextlib import contextmanager
import random
import time

class DatabaseRouter:
    """読み書き分離を行うデータベースルーター"""

    def __init__(self):
        self.writer_config = {
            'host': 'my-mysql-db.xxxx.ap-northeast-1.rds.amazonaws.com',
            'user': 'admin',
            'password': 'secret',
            'database': 'myapp',
            'charset': 'utf8mb4',
            'connect_timeout': 5,
            'read_timeout': 30,
        }
        self.reader_configs = [
            {
                'host': 'my-mysql-db-read1.xxxx.ap-northeast-1.rds.amazonaws.com',
                'user': 'readonly',
                'password': 'secret',
                'database': 'myapp',
                'charset': 'utf8mb4',
                'connect_timeout': 5,
                'read_timeout': 30,
            },
            {
                'host': 'my-mysql-db-read2.xxxx.ap-northeast-1.rds.amazonaws.com',
                'user': 'readonly',
                'password': 'secret',
                'database': 'myapp',
                'charset': 'utf8mb4',
                'connect_timeout': 5,
                'read_timeout': 30,
            },
        ]
        self._reader_index = 0

    def _connect_with_retry(self, config, max_retries=3):
        """リトライ付きの接続"""
        for attempt in range(max_retries):
            try:
                return pymysql.connect(**config)
            except pymysql.OperationalError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))

    @contextmanager
    def writer(self):
        """書き込み用コネクション"""
        conn = self._connect_with_retry(self.writer_config)
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
        conn = self._connect_with_retry(config)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def consistent_reader(self):
        """一貫性が必要な読み取り（プライマリから読む）"""
        conn = self._connect_with_retry(self.writer_config)
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

# 書き込み直後の読み取り（レプリケーション遅延回避）
with db.consistent_reader() as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", ("taro@example.com",))
        user = cur.fetchone()
```

### コード例: RDS Proxy を使った接続管理

```bash
# RDS Proxy の作成
aws rds create-db-proxy \
  --db-proxy-name my-app-proxy \
  --engine-family MYSQL \
  --auth '[{
    "AuthScheme": "SECRETS",
    "SecretArn": "arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:my-db-creds",
    "IAMAuth": "DISABLED"
  }]' \
  --role-arn arn:aws:iam::123456789012:role/rds-proxy-role \
  --vpc-subnet-ids subnet-aaa subnet-bbb subnet-ccc \
  --vpc-security-group-ids sg-proxy123 \
  --require-tls \
  --idle-client-timeout 1800

# ターゲットグループの登録
aws rds register-db-proxy-targets \
  --db-proxy-name my-app-proxy \
  --db-instance-identifiers my-mysql-db

# Proxy エンドポイントの取得
aws rds describe-db-proxies \
  --db-proxy-name my-app-proxy \
  --query 'DBProxies[0].Endpoint'

# Proxy のリーダーエンドポイントを作成（読み取り用）
aws rds create-db-proxy-endpoint \
  --db-proxy-name my-app-proxy \
  --db-proxy-endpoint-name my-app-proxy-reader \
  --target-role READ_ONLY \
  --vpc-subnet-ids subnet-aaa subnet-bbb subnet-ccc
```

```
RDS Proxy のメリット:
1. 接続プーリング: Lambda 等の短命な接続を効率的に管理
2. フェイルオーバー高速化: Proxy がフェイルオーバーを隠蔽（切替時間短縮）
3. IAM 認証: データベースパスワードの代わりに IAM 認証を使用可能
4. TLS 強制: クライアント-Proxy 間の暗号化を強制
5. ピン留め: 同一セッション内のクエリを同一接続に固定
```

---

## 6. バックアップとリカバリ

### バックアップの仕組み

```
RDS バックアップ戦略

自動バックアップ:
  毎日のスナップショット (バックアップウィンドウ内)
  + トランザクションログ (5 分間隔)
  = ポイントインタイムリカバリ (PITR)
  保持期間: 1-35 日（デフォルト 7 日）

  +--+--+--+--+--+--+--+--+--+--+--+--+
  |日|月|火|水|木|金|土|日|月|火|水|木|
  +--+--+--+--+--+--+--+--+--+--+--+--+
   ↑  ↑  ↑  ↑  ↑  ↑  ↑             ↑
   スナップショット                    最新の復元可能な時点
                                    (Latest Restorable Time)

手動スナップショット:
  - 自動削除されない（手動で削除するまで保持）
  - リージョン間コピー可能（DR 用）
  - アカウント間共有可能
```

### コード例 7: ポイントインタイムリカバリ

```bash
# 復元可能な最新時刻の確認
aws rds describe-db-instances \
  --db-instance-identifier my-mysql-db \
  --query 'DBInstances[0].LatestRestorableTime'

# 特定時刻の状態にリストア（新インスタンスとして作成）
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier my-mysql-db \
  --target-db-instance-identifier my-mysql-db-restored \
  --restore-time "2026-02-10T15:30:00Z" \
  --db-instance-class db.r6g.large \
  --db-subnet-group-name my-db-subnet-group \
  --vpc-security-group-ids sg-0abc123def456 \
  --multi-az \
  --storage-type gp3 \
  --copy-tags-to-snapshot

# 最新の復元可能な時点にリストア
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier my-mysql-db \
  --target-db-instance-identifier my-mysql-db-latest \
  --use-latest-restorable-time \
  --db-instance-class db.r6g.large

# 手動スナップショットの作成
aws rds create-db-snapshot \
  --db-instance-identifier my-mysql-db \
  --db-snapshot-identifier my-mysql-db-snap-20260211

# スナップショットからの復元
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier my-mysql-db-from-snap \
  --db-snapshot-identifier my-mysql-db-snap-20260211 \
  --db-instance-class db.r6g.large

# スナップショットの別リージョンへのコピー（DR 用）
aws rds copy-db-snapshot \
  --source-db-snapshot-identifier arn:aws:rds:ap-northeast-1:123456789012:snapshot:my-mysql-db-snap-20260211 \
  --target-db-snapshot-identifier my-mysql-db-snap-us-copy \
  --region us-east-1 \
  --kms-key-id alias/rds-dr-key

# スナップショットの別アカウントへの共有
aws rds modify-db-snapshot-attribute \
  --db-snapshot-identifier my-mysql-db-snap-20260211 \
  --attribute-name restore \
  --values-to-add "987654321098"
```

### コード例: 自動バックアップのクロスリージョンレプリケーション

```bash
# 自動バックアップの別リージョンへのレプリケーション
aws rds start-db-instance-automated-backups-replication \
  --source-db-instance-arn arn:aws:rds:ap-northeast-1:123456789012:db:my-mysql-db \
  --backup-retention-period 7 \
  --kms-key-id alias/rds-dr-key \
  --region us-east-1

# レプリケーション状態の確認
aws rds describe-db-instance-automated-backups \
  --db-instance-automated-backups-arn arn:aws:rds:us-east-1:123456789012:auto-backup:xxx \
  --region us-east-1
```

---

## 7. 監視とパフォーマンスチューニング

### 主要 CloudWatch メトリクス

| メトリクス | 閾値（目安） | 対応 |
|-----------|------------|------|
| CPUUtilization | > 80% | インスタンスクラスのスケールアップ |
| FreeableMemory | < 256 MB | スケールアップ、クエリ最適化 |
| FreeStorageSpace | < 10 GB | ストレージ拡張、古いデータの削除 |
| ReadIOPS / WriteIOPS | ベースライン超過 | gp3 IOPS 増加、io2 への変更 |
| ReplicaLag | > 30 秒 | レプリカのスケールアップ、並列レプリケーション |
| DatabaseConnections | > 80% of max | RDS Proxy 導入、接続プール |
| SwapUsage | > 0 | メモリ不足、スケールアップ |
| DiskQueueDepth | > 64 | ストレージ IOPS の増加 |

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

# トップ SQL の取得
aws pi get-resource-metrics \
  --service-type RDS \
  --identifier db-XXXXXXXXXXXXXXXXXXXX \
  --metric-queries '[{
    "Metric": "db.load.avg",
    "GroupBy": {
      "Group": "db.sql",
      "Limit": 10
    }
  }]' \
  --start-time 2026-02-10T00:00:00Z \
  --end-time 2026-02-11T00:00:00Z \
  --period-in-seconds 3600

# Enhanced Monitoring の有効化（OS レベルのメトリクス）
aws rds modify-db-instance \
  --db-instance-identifier my-mysql-db \
  --monitoring-interval 60 \
  --monitoring-role-arn arn:aws:iam::123456789012:role/rds-monitoring-role \
  --apply-immediately
```

### コード例: CloudWatch ダッシュボードの作成

```bash
# RDS 監視ダッシュボードの作成
aws cloudwatch put-dashboard \
  --dashboard-name "RDS-Monitoring" \
  --dashboard-body '{
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "title": "CPU Utilization",
          "metrics": [
            ["AWS/RDS", "CPUUtilization", "DBInstanceIdentifier", "my-mysql-db"]
          ],
          "period": 300,
          "stat": "Average",
          "region": "ap-northeast-1",
          "yAxis": {"left": {"min": 0, "max": 100}}
        },
        "width": 12, "height": 6, "x": 0, "y": 0
      },
      {
        "type": "metric",
        "properties": {
          "title": "Database Connections",
          "metrics": [
            ["AWS/RDS", "DatabaseConnections", "DBInstanceIdentifier", "my-mysql-db"]
          ],
          "period": 300,
          "stat": "Average",
          "region": "ap-northeast-1"
        },
        "width": 12, "height": 6, "x": 12, "y": 0
      },
      {
        "type": "metric",
        "properties": {
          "title": "IOPS",
          "metrics": [
            ["AWS/RDS", "ReadIOPS", "DBInstanceIdentifier", "my-mysql-db"],
            ["AWS/RDS", "WriteIOPS", "DBInstanceIdentifier", "my-mysql-db"]
          ],
          "period": 300,
          "stat": "Average",
          "region": "ap-northeast-1"
        },
        "width": 12, "height": 6, "x": 0, "y": 6
      },
      {
        "type": "metric",
        "properties": {
          "title": "Replica Lag",
          "metrics": [
            ["AWS/RDS", "ReplicaLag", "DBInstanceIdentifier", "my-mysql-db-read1"],
            ["AWS/RDS", "ReplicaLag", "DBInstanceIdentifier", "my-mysql-db-read2"]
          ],
          "period": 60,
          "stat": "Average",
          "region": "ap-northeast-1"
        },
        "width": 12, "height": 6, "x": 12, "y": 6
      }
    ]
  }'
```

---

## 8. セキュリティ

### 8.1 VPC とネットワーク

```bash
# DB サブネットグループの作成
aws rds create-db-subnet-group \
  --db-subnet-group-name my-db-subnet-group \
  --db-subnet-group-description "Private subnets for RDS" \
  --subnet-ids subnet-aaa111 subnet-bbb222 subnet-ccc333

# セキュリティグループの作成
SG_ID=$(aws ec2 create-security-group \
  --group-name rds-mysql-sg \
  --description "Security group for RDS MySQL" \
  --vpc-id vpc-xxx \
  --query 'GroupId' --output text)

# アプリケーション層からのみ MySQL ポートを許可
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 3306 \
  --source-group sg-app-layer

# Bastion / SSM からの接続も許可（管理者用）
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp \
  --port 3306 \
  --source-group sg-bastion
```

### 8.2 IAM データベース認証

```bash
# IAM 認証の有効化
aws rds modify-db-instance \
  --db-instance-identifier my-mysql-db \
  --enable-iam-database-authentication \
  --apply-immediately

# MySQL で IAM 認証ユーザーの作成
# mysql> CREATE USER 'iam_user'@'%' IDENTIFIED WITH AWSAuthenticationPlugin AS 'RDS';
# mysql> GRANT SELECT ON myapp.* TO 'iam_user'@'%';

# IAM ポリシーの例
# {
#   "Version": "2012-10-17",
#   "Statement": [{
#     "Effect": "Allow",
#     "Action": "rds-db:connect",
#     "Resource": "arn:aws:rds-db:ap-northeast-1:123456789012:dbuser:cluster-xxx/iam_user"
#   }]
# }

# IAM 認証トークンの取得と接続
TOKEN=$(aws rds generate-db-auth-token \
  --hostname my-mysql-db.xxxx.ap-northeast-1.rds.amazonaws.com \
  --port 3306 \
  --username iam_user)

mysql -h my-mysql-db.xxxx.ap-northeast-1.rds.amazonaws.com \
  -u iam_user \
  --password=$TOKEN \
  --ssl-mode=REQUIRED \
  --ssl-ca=global-bundle.pem
```

### 8.3 暗号化

```bash
# 暗号化済みインスタンスの確認
aws rds describe-db-instances \
  --db-instance-identifier my-mysql-db \
  --query 'DBInstances[0].{StorageEncrypted:StorageEncrypted,KmsKeyId:KmsKeyId}'

# 非暗号化インスタンスの暗号化（スナップショット経由）
# 1. スナップショット取得
aws rds create-db-snapshot \
  --db-instance-identifier my-mysql-db-unencrypted \
  --db-snapshot-identifier my-mysql-db-unenc-snap

# 2. 暗号化コピーを作成
aws rds copy-db-snapshot \
  --source-db-snapshot-identifier my-mysql-db-unenc-snap \
  --target-db-snapshot-identifier my-mysql-db-encrypted-snap \
  --kms-key-id alias/rds-key

# 3. 暗号化スナップショットから復元
aws rds restore-db-instance-from-db-snapshot \
  --db-instance-identifier my-mysql-db-encrypted \
  --db-snapshot-identifier my-mysql-db-encrypted-snap

# 4. アプリの接続先を切り替え後、旧インスタンスを削除

# SSL/TLS 接続の強制（パラメータグループ）
# MySQL: require_secure_transport = 1
# PostgreSQL: rds.force_ssl = 1
```

---

## 9. インスタンスの停止と起動

```bash
# RDS インスタンスの一時停止（最大 7 日間）
aws rds stop-db-instance \
  --db-instance-identifier my-dev-db

# 7 日後に自動起動されるため、継続的に停止したい場合はスクリプトが必要

# RDS インスタンスの起動
aws rds start-db-instance \
  --db-instance-identifier my-dev-db

# Lambda で開発環境の定時停止・起動を自動化
# EventBridge ルール: 平日 20:00 に停止、翌朝 8:00 に起動
aws events put-rule \
  --name stop-dev-rds-nightly \
  --schedule-expression "cron(0 11 ? * MON-FRI *)" \
  --description "Stop dev RDS at 20:00 JST"

aws events put-rule \
  --name start-dev-rds-morning \
  --schedule-expression "cron(0 23 ? * SUN-THU *)" \
  --description "Start dev RDS at 8:00 JST"
```

---

## 10. CloudFormation / CDK による構築

### 10.1 CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Production RDS MySQL with Multi-AZ and Read Replica

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
  PrivateSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
  AppSecurityGroupId:
    Type: AWS::EC2::SecurityGroup::Id
  DBPassword:
    Type: String
    NoEcho: true
    Description: Master password (Secrets Manager recommended)

Resources:
  # KMS キー
  RDSKey:
    Type: AWS::KMS::Key
    Properties:
      Description: RDS encryption key
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Sid: AllowRootAccount
            Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:root'
            Action: 'kms:*'
            Resource: '*'

  # DB サブネットグループ
  DBSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Private subnets for RDS
      SubnetIds: !Ref PrivateSubnetIds

  # セキュリティグループ
  DBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: RDS MySQL Security Group
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          SourceSecurityGroupId: !Ref AppSecurityGroupId
          Description: MySQL from app layer

  # パラメータグループ
  DBParameterGroup:
    Type: AWS::RDS::DBParameterGroup
    Properties:
      Family: mysql8.0
      Description: Custom MySQL 8.0 parameters
      Parameters:
        character_set_server: utf8mb4
        collation_server: utf8mb4_unicode_ci
        slow_query_log: '1'
        long_query_time: '1'
        require_secure_transport: '1'

  # プライマリインスタンス
  DBInstance:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot
    Properties:
      DBInstanceIdentifier: !Sub '${AWS::StackName}-mysql'
      Engine: mysql
      EngineVersion: '8.0.35'
      DBInstanceClass: db.r6g.large
      AllocatedStorage: 100
      MaxAllocatedStorage: 500
      StorageType: gp3
      StorageEncrypted: true
      KmsKeyId: !Ref RDSKey
      MultiAZ: true
      MasterUsername: admin
      MasterUserPassword: !Ref DBPassword
      DBSubnetGroupName: !Ref DBSubnetGroup
      VPCSecurityGroups:
        - !Ref DBSecurityGroup
      DBParameterGroupName: !Ref DBParameterGroup
      BackupRetentionPeriod: 7
      PreferredBackupWindow: '03:00-04:00'
      PreferredMaintenanceWindow: 'Mon:04:00-Mon:05:00'
      AutoMinorVersionUpgrade: true
      DeletionProtection: true
      CopyTagsToSnapshot: true
      EnablePerformanceInsights: true
      PerformanceInsightsRetentionPeriod: 731
      MonitoringInterval: 60
      MonitoringRoleArn: !GetAtt MonitoringRole.Arn
      EnableCloudwatchLogsExports:
        - audit
        - error
        - slowquery

  # リードレプリカ
  ReadReplica:
    Type: AWS::RDS::DBInstance
    DependsOn: DBInstance
    Properties:
      DBInstanceIdentifier: !Sub '${AWS::StackName}-mysql-read1'
      SourceDBInstanceIdentifier: !Ref DBInstance
      DBInstanceClass: db.r6g.large
      StorageType: gp3
      EnablePerformanceInsights: true
      PerformanceInsightsRetentionPeriod: 731
      MonitoringInterval: 60
      MonitoringRoleArn: !GetAtt MonitoringRole.Arn

  # Enhanced Monitoring 用 IAM ロール
  MonitoringRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: monitoring.rds.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonRDSEnhancedMonitoringRole

  # CPU 使用率アラーム
  CPUAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub '${AWS::StackName}-rds-cpu-high'
      MetricName: CPUUtilization
      Namespace: AWS/RDS
      Dimensions:
        - Name: DBInstanceIdentifier
          Value: !Ref DBInstance
      Statistic: Average
      Period: 300
      EvaluationPeriods: 3
      Threshold: 80
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Sub 'arn:aws:sns:${AWS::Region}:${AWS::AccountId}:alerts'

Outputs:
  PrimaryEndpoint:
    Value: !GetAtt DBInstance.Endpoint.Address
    Description: Primary DB endpoint
  PrimaryPort:
    Value: !GetAtt DBInstance.Endpoint.Port
    Description: Primary DB port
  ReadReplicaEndpoint:
    Value: !GetAtt ReadReplica.Endpoint.Address
    Description: Read replica endpoint
```

### 10.2 CDK (TypeScript) による構築

```typescript
import * as cdk from 'aws-cdk-lib';
import * as rds from 'aws-cdk-lib/aws-rds';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as cw_actions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';

interface RdsStackProps extends cdk.StackProps {
  vpc: ec2.IVpc;
  appSecurityGroup: ec2.ISecurityGroup;
}

export class RdsStack extends cdk.Stack {
  public readonly dbInstance: rds.DatabaseInstance;
  public readonly readReplica: rds.DatabaseInstanceReadReplica;

  constructor(scope: Construct, id: string, props: RdsStackProps) {
    super(scope, id, props);

    // KMS キー
    const encryptionKey = new kms.Key(this, 'RdsKey', {
      description: 'RDS encryption key',
      enableKeyRotation: true,
    });

    // セキュリティグループ
    const dbSg = new ec2.SecurityGroup(this, 'DbSg', {
      vpc: props.vpc,
      description: 'RDS MySQL Security Group',
      allowAllOutbound: false,
    });
    dbSg.addIngressRule(
      props.appSecurityGroup,
      ec2.Port.tcp(3306),
      'MySQL from app layer'
    );

    // パラメータグループ
    const parameterGroup = new rds.ParameterGroup(this, 'Params', {
      engine: rds.DatabaseInstanceEngine.mysql({
        version: rds.MysqlEngineVersion.VER_8_0_35,
      }),
      parameters: {
        character_set_server: 'utf8mb4',
        collation_server: 'utf8mb4_unicode_ci',
        slow_query_log: '1',
        long_query_time: '1',
        require_secure_transport: '1',
      },
    });

    // プライマリインスタンス
    this.dbInstance = new rds.DatabaseInstance(this, 'Primary', {
      engine: rds.DatabaseInstanceEngine.mysql({
        version: rds.MysqlEngineVersion.VER_8_0_35,
      }),
      instanceType: ec2.InstanceType.of(
        ec2.InstanceClass.R6G,
        ec2.InstanceSize.LARGE
      ),
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [dbSg],
      multiAz: true,
      allocatedStorage: 100,
      maxAllocatedStorage: 500,
      storageType: rds.StorageType.GP3,
      storageEncrypted: true,
      storageEncryptionKey: encryptionKey,
      parameterGroup,
      backupRetention: cdk.Duration.days(7),
      preferredBackupWindow: '03:00-04:00',
      preferredMaintenanceWindow: 'Mon:04:00-Mon:05:00',
      deletionProtection: true,
      removalPolicy: cdk.RemovalPolicy.SNAPSHOT,
      copyTagsToSnapshot: true,
      enablePerformanceInsights: true,
      performanceInsightRetention: rds.PerformanceInsightRetention.MONTHS_25,
      monitoringInterval: cdk.Duration.seconds(60),
      cloudwatchLogsExports: ['audit', 'error', 'slowquery'],
      credentials: rds.Credentials.fromGeneratedSecret('admin', {
        secretName: 'rds/mysql/prod/credentials',
      }),
    });

    // リードレプリカ
    this.readReplica = new rds.DatabaseInstanceReadReplica(this, 'ReadReplica', {
      sourceDatabaseInstance: this.dbInstance,
      instanceType: ec2.InstanceType.of(
        ec2.InstanceClass.R6G,
        ec2.InstanceSize.LARGE
      ),
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [dbSg],
      storageType: rds.StorageType.GP3,
      enablePerformanceInsights: true,
      performanceInsightRetention: rds.PerformanceInsightRetention.MONTHS_25,
      monitoringInterval: cdk.Duration.seconds(60),
    });

    // CPU 使用率アラーム
    const alertTopic = sns.Topic.fromTopicArn(
      this, 'AlertTopic',
      `arn:aws:sns:${this.region}:${this.account}:alerts`
    );

    const cpuAlarm = this.dbInstance.metricCPUUtilization({
      period: cdk.Duration.minutes(5),
    }).createAlarm(this, 'CpuAlarm', {
      threshold: 80,
      evaluationPeriods: 3,
      alarmDescription: 'RDS CPU utilization > 80%',
    });
    cpuAlarm.addAlarmAction(new cw_actions.SnsAction(alertTopic));

    // レプリケーション遅延アラーム
    const replicaLagAlarm = new cloudwatch.Alarm(this, 'ReplicaLagAlarm', {
      metric: new cloudwatch.Metric({
        namespace: 'AWS/RDS',
        metricName: 'ReplicaLag',
        dimensionsMap: {
          DBInstanceIdentifier: this.readReplica.instanceIdentifier,
        },
        period: cdk.Duration.minutes(1),
        statistic: 'Average',
      }),
      threshold: 30,
      evaluationPeriods: 3,
      alarmDescription: 'RDS ReplicaLag > 30 seconds',
    });
    replicaLagAlarm.addAlarmAction(new cw_actions.SnsAction(alertTopic));

    // 出力
    new cdk.CfnOutput(this, 'PrimaryEndpoint', {
      value: this.dbInstance.dbInstanceEndpointAddress,
    });
    new cdk.CfnOutput(this, 'ReadReplicaEndpoint', {
      value: this.readReplica.dbInstanceEndpointAddress,
    });
    new cdk.CfnOutput(this, 'SecretArn', {
      value: this.dbInstance.secret!.secretArn,
    });
  }
}
```

---

## 11. アンチパターン

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

### 3. バックアップ保持期間を 0 にする

**問題**: バックアップ保持期間を 0 にすると、自動バックアップが無効化され、PITR（ポイントインタイムリカバリ）が使えなくなる。データの誤操作やアプリケーションバグによるデータ破損からの復旧が困難になる。

**対策**: 本番環境ではバックアップ保持期間を最低 7 日、重要なシステムでは 14-35 日に設定する。加えて、定期的な手動スナップショットも取得し、別リージョンにコピーする。

### 4. デフォルトのパラメータグループを使い続ける

**問題**: デフォルトのパラメータグループはカスタマイズできず、チューニングの余地がない。文字コード設定やスロークエリログが無効のまま運用される。

**対策**: 必ずカスタムパラメータグループを作成し、文字コード (utf8mb4)、スロークエリログ、InnoDB バッファプールサイズなどを適切に設定する。

### 5. Secrets Manager を使わずにパスワードを管理する

```
# 悪い例
- パスワードを環境変数にハードコード
- .env ファイルに平文で記載
- Terraform の state ファイルに残る

# 良い例
- Secrets Manager でパスワードを管理
- IAM 認証を使用
- RDS Proxy 経由で IAM 認証
```

---

## 12. FAQ

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
5. **RDS Proxy** のリーダーエンドポイントを使って負荷分散

### Q3: RDS のコストを最適化するには？

**A**: 主な最適化手法:
- **リザーブドインスタンス**: 1年/3年の予約で最大60%割引
- **インスタンスの適正化**: Performance Insights で実使用率を確認し、オーバープロビジョニングを解消
- **ストレージタイプの見直し**: gp2 から gp3 への移行で同じ IOPS をより低コストで実現
- **開発環境の停止**: 夜間・休日に不要なインスタンスを停止（最大7日間）
- **Graviton インスタンス**: db.r6g/r7g に切り替えで約 20% コスト削減

### Q4: RDS のメンテナンスウィンドウ中にダウンタイムは発生しますか？

**A**: メンテナンスの種類によります。
- **マイナーバージョンアップグレード**: 数分のダウンタイムが発生。マルチ AZ の場合、Standby を先に更新してからフェイルオーバーするため、ダウンタイムはフェイルオーバー時間（60-120秒）のみ。
- **パッチ適用**: ほとんどの場合ダウンタイムなし。OS パッチの一部でリブートが必要な場合あり。
- **メジャーバージョンアップグレード**: 数十分のダウンタイムが発生する場合がある。Blue/Green デプロイメントの活用を推奨。

### Q5: RDS の Blue/Green デプロイメントとは？

**A**: RDS のメジャーバージョンアップグレードやパラメータ変更を安全に行うための機能。現在の環境（Blue）のコピー（Green）を作成し、Green 側で変更を適用してテスト。問題なければ DNS を切り替えて Green を本番にする。切り替えは 1 分以下で完了する。

```bash
# Blue/Green デプロイメントの作成
aws rds create-blue-green-deployment \
  --blue-green-deployment-name mysql-upgrade \
  --source arn:aws:rds:ap-northeast-1:123456789012:db:my-mysql-db \
  --target-engine-version 8.0.36 \
  --target-db-parameter-group-name new-params

# 状態の確認
aws rds describe-blue-green-deployments \
  --blue-green-deployment-identifier bgd-xxx

# 切り替え実行
aws rds switchover-blue-green-deployment \
  --blue-green-deployment-identifier bgd-xxx \
  --switchover-timeout 300
```

---

## 13. まとめ

| 項目 | 要点 |
|---|---|
| RDS とは | フルマネージド RDB サービス。パッチ適用・バックアップ・フェイルオーバーを自動化 |
| エンジン選択 | Web アプリ → MySQL、分析・拡張性 → PostgreSQL、高性能 → Aurora |
| インスタンスクラス | Graviton (r6g/r7g) がコスパ最良。t3 は開発・テスト用 |
| ストレージ | gp3 が標準。IOPS 要件が高い場合は io2 |
| マルチ AZ | プロダクションでは必須。同期レプリケーションで自動フェイルオーバー |
| リードレプリカ | 読み取り負荷分散。非同期のためレプリケーション遅延に注意 |
| RDS Proxy | Lambda や短命接続の効率化、フェイルオーバーの高速化 |
| バックアップ | 自動バックアップ + ポイントインタイムリカバリで RPO を最小化 |
| 監視 | Performance Insights で待機イベント分析、CloudWatch でメトリクス監視 |
| セキュリティ | プライベートサブネット配置、暗号化、IAM 認証の活用 |
| IaC | CloudFormation / CDK で宣言的に管理 |

## 次に読むべきガイド

- [DynamoDB](./01-dynamodb.md) — NoSQL データベースの設計と運用
- [ElastiCache](./02-elasticache.md) — キャッシュレイヤーの構築
- [VPC 基礎](../04-networking/00-vpc-basics.md) — RDS を配置するネットワーク設計

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon RDS ユーザーガイド](https://docs.aws.amazon.com/ja_jp/AmazonRDS/latest/UserGuide/) — エンジン別の詳細設定リファレンス
2. **AWS Well-Architected Framework**: [信頼性の柱 - データベース設計](https://docs.aws.amazon.com/ja_jp/wellarchitected/latest/reliability-pillar/) — 信頼性の柱におけるデータベース設計指針
3. **Amazon RDS ベストプラクティス**: [Performance Insights を使用した DB 負荷の分析](https://docs.aws.amazon.com/ja_jp/AmazonRDS/latest/UserGuide/USER_PerfInsights.html) — パフォーマンス分析の実践ガイド
4. **RDS Proxy ドキュメント**: [Amazon RDS Proxy の使用](https://docs.aws.amazon.com/ja_jp/AmazonRDS/latest/UserGuide/rds-proxy.html) — 接続管理の最適化
5. **RDS Blue/Green デプロイメント**: [Blue/Green Deployments の概要](https://docs.aws.amazon.com/ja_jp/AmazonRDS/latest/UserGuide/blue-green-deployments.html) — 安全なアップグレード手法
