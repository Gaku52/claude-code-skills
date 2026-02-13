# IaC セキュリティ

> tfsec、Checkov によるインフラコードの自動セキュリティチェック、ポリシー as コードによるガバナンス適用まで、Infrastructure as Code のセキュリティを体系的に学ぶ

## 前提知識

- Terraform / CloudFormation の基本的な構文と概念
- クラウドインフラ (AWS/GCP/Azure) の基本的なサービス理解
- CI/CD パイプラインの基礎知識

## この章で学ぶこと

1. **IaC のセキュリティリスク** — Terraform/CloudFormation コードに潜む設定ミスとその影響
2. **静的解析ツール** — tfsec、Checkov、KICS、Trivy によるセキュリティポリシーの自動検証
3. **ポリシー as コード** — OPA/Rego、Sentinel によるカスタムポリシーの実装
4. **ドリフト検知** — IaC と実インフラの乖離を検出し是正する手法
5. **CI/CD 統合** — セキュリティゲートをパイプラインに組み込む実践手法
6. **シークレット管理** — IaC コードからの機密情報の排除

---

## 1. IaC セキュリティの重要性

### なぜ IaC セキュリティが必要なのか

IaC (Infrastructure as Code) は本来、インフラ構成の一貫性と再現性を高めるために導入される。しかし、IaC コード自体にセキュリティ上の問題が含まれている場合、その問題はコードを適用するたびに**大規模に再現**される。手動設定であれば1つの環境にのみ影響する設定ミスが、IaC では全環境に同時にデプロイされてしまう。

#### IaC セキュリティの3つの脅威モデル

```
+------------------------------------------------------------------+
|          IaC セキュリティにおける脅威モデル                          |
|------------------------------------------------------------------|
|                                                                  |
|  [脅威1: 設定ミス (Misconfiguration)]                             |
|  +-- 最も頻発するリスク                                           |
|  +-- S3 バケットの公開設定、SG の過剰許可など                      |
|  +-- 2023年のクラウドセキュリティインシデントの 60%以上が設定ミス    |
|  +-- IaC により設定ミスが大規模に展開される                        |
|                                                                  |
|  [脅威2: シークレット漏洩 (Secrets Exposure)]                     |
|  +-- ハードコードされた認証情報                                    |
|  +-- Git 履歴に残るシークレット                                   |
|  +-- Terraform state ファイル内の機密データ                       |
|  +-- 出力変数 (output) によるシークレットの露出                    |
|                                                                  |
|  [脅威3: サプライチェーン攻撃 (Supply Chain)]                     |
|  +-- 悪意のある Terraform モジュール                               |
|  +-- 改竄された Provider プラグイン                                |
|  +-- 信頼されないレジストリからのモジュール取得                     |
|  +-- モジュールのバージョン固定忘れ                                |
|                                                                  |
+------------------------------------------------------------------+
```

### IaC で起きるセキュリティ問題の分類

```
+------------------------------------------------------------------+
|         IaC の典型的なセキュリティ問題                               |
|------------------------------------------------------------------|
|                                                                  |
|  [ネットワーク]                                                   |
|  +-- Security Group で 0.0.0.0/0:22 を許可                      |
|  +-- NACL のデフォルト全許可                                      |
|  +-- VPC ピアリングの過剰な許可                                   |
|  +-- VPC Endpoint の未設定（パブリック経路でのAPI通信）             |
|  +-- パブリックサブネットへの不要なリソース配置                     |
|                                                                  |
|  [データ保護]                                                     |
|  +-- S3 バケットのパブリックアクセス                               |
|  +-- RDS/EBS の暗号化未設定                                      |
|  +-- ログの暗号化未設定                                           |
|  +-- バックアップの暗号化・保持期間未設定                          |
|  +-- クロスリージョンレプリケーション未設定                         |
|                                                                  |
|  [認証・認可]                                                     |
|  +-- IAM ポリシーの * (全許可)                                    |
|  +-- ハードコードされた認証情報                                    |
|  +-- MFA 未設定のリソース                                         |
|  +-- 過剰な権限のサービスロール                                    |
|  +-- AssumeRole の Principal 制限不足                             |
|                                                                  |
|  [ログ・監視]                                                     |
|  +-- CloudTrail 無効                                            |
|  +-- VPC Flow Logs 未設定                                        |
|  +-- アクセスログ未有効化                                         |
|  +-- CloudWatch アラーム未設定                                   |
|  +-- GuardDuty/SecurityHub 未有効化                              |
|                                                                  |
|  [コンプライアンス]                                                |
|  +-- タグ付け規則の不遵守                                         |
|  +-- リージョン制限の未適用                                       |
|  +-- データ保持ポリシーの未実装                                    |
|  +-- 暗号化基準の不遵守                                           |
|                                                                  |
+------------------------------------------------------------------+
```

### IaC のセキュリティチェックのタイミング

セキュリティチェックは「シフトレフト」の原則に従い、できるだけ早い段階で実施することが重要である。問題の発見が遅れるほど修正コストは指数関数的に増加する。

```
開発者 PC         CI/CD              デプロイ前            ランタイム
    |                |                    |                    |
  [pre-commit]    [ビルド]            [Plan/Apply]         [ドリフト検知]
    |                |                    |                    |
  tfsec           Checkov             Sentinel/OPA         AWS Config
  trivy           KICS                (ポリシーゲート)       Prowler
  (IDE連携)       tfsec                                     (定期スキャン)
  git-secrets     trivy                                    driftctl
                  Snyk IaC
    |                |                    |                    |
  コスト:低       コスト:中             コスト:高             コスト:最高
  発見速度:最速   発見速度:速           発見速度:中           発見速度:遅
```

#### セキュリティチェックのレイヤー構造

```
┌─────────────────────────────────────────────────────────┐
│              IaC セキュリティチェック レイヤー              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 5: ランタイム監視                                  │
│  ├── AWS Config Rules（リソース準拠チェック）              │
│  ├── Prowler（CIS ベンチマーク定期スキャン）               │
│  └── ドリフト検知（terraform plan -detailed-exitcode）   │
│                                                         │
│  Layer 4: ポリシーゲート                                  │
│  ├── OPA/Conftest（terraform plan JSON 検証）            │
│  ├── Sentinel（Terraform Cloud/Enterprise）              │
│  └── AWS Service Control Policies                       │
│                                                         │
│  Layer 3: CI/CD パイプライン                              │
│  ├── Checkov（マルチフレームワーク静的解析）               │
│  ├── tfsec / trivy config（Terraform 特化解析）          │
│  ├── KICS（Checkmarx IaC スキャナ）                      │
│  └── SARIF → GitHub Security タブ連携                   │
│                                                         │
│  Layer 2: Pre-commit フック                               │
│  ├── tfsec / trivy（ローカルスキャン）                    │
│  ├── terraform fmt / validate                           │
│  ├── git-secrets / gitleaks（シークレット検知）           │
│  └── tflint（Terraform リンター）                        │
│                                                         │
│  Layer 1: IDE 統合                                        │
│  ├── VS Code tfsec 拡張機能                              │
│  ├── VS Code Checkov 拡張機能                            │
│  └── IntelliJ Terraform プラグイン                       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. tfsec (Terraform セキュリティスキャナ)

### tfsec の内部アーキテクチャ

tfsec は Terraform のHCLコードを解析し、セキュリティルール違反を検出する静的解析ツールである。現在は Aqua Security の Trivy に統合されつつあるが、tfsec 単体としても広く使われている。

```
┌─────────────────── tfsec 内部処理フロー ───────────────────┐
│                                                            │
│  1. HCL パーサー                                           │
│     ├── .tf ファイルを AST（抽象構文木）に変換               │
│     ├── 変数の解決（variables.tf, terraform.tfvars）       │
│     └── モジュール参照の解決                                │
│                                                            │
│  2. リソースグラフ構築                                       │
│     ├── リソース間の依存関係を解析                           │
│     ├── 属性の参照チェーン（例: SG → EC2）を追跡             │
│     └── data source の評価                                 │
│                                                            │
│  3. ルールエンジン                                          │
│     ├── 組み込みルール（~1000 ルール）                      │
│     ├── カスタムルール（YAML/JSON/Rego）                    │
│     └── 各ルールがリソースを検査                            │
│                                                            │
│  4. 結果レポーター                                          │
│     ├── テキスト / JSON / SARIF / CSV / JUnit             │
│     ├── 重大度レベル: CRITICAL, HIGH, MEDIUM, LOW          │
│     └── 修正推奨事項の提示                                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### tfsec の使い方

```bash
# インストール
brew install tfsec

# または Trivy 経由（推奨：tfsec は Trivy に統合済み）
brew install trivy
trivy config .

# 基本スキャン実行
tfsec .

# 特定の重大度以上のみ
tfsec --minimum-severity HIGH .

# JSON 出力 (CI/CD 向け)
tfsec --format json --out results.json .

# SARIF 出力 (GitHub Security タブ連携)
tfsec --format sarif --out results.sarif .

# JUnit 出力 (Jenkins 連携)
tfsec --format junit --out results.xml .

# 特定のディレクトリを除外
tfsec --exclude-path modules/legacy .

# 特定のルールを無効化
tfsec --exclude aws-s3-enable-versioning .

# カスタムルールファイルを指定
tfsec --custom-check-dir ./custom-rules .

# Terraform 変数ファイルを指定
tfsec --tfvars-file production.tfvars .

# ソフトフェイル（CI を止めずに結果だけ出力）
tfsec --soft-fail .
```

### tfsec の検出例と修正

```hcl
# NG: tfsec が検出する問題（複数のセキュリティ違反）
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"
  # aws-s3-enable-bucket-encryption: 暗号化未設定
  # aws-s3-enable-bucket-logging: アクセスログ未設定
  # aws-s3-enable-versioning: バージョニング未設定
  # aws-s3-block-public-acls: パブリックアクセスブロック未設定
}

resource "aws_security_group_rule" "ssh" {
  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]  # aws-vpc-no-public-ingress-sgr
  security_group_id = aws_security_group.main.id
}

resource "aws_db_instance" "main" {
  engine         = "postgres"
  instance_class = "db.t3.medium"
  # aws-rds-encrypt-instance-storage-data: 暗号化未設定
  # aws-rds-no-public-db-access: パブリックアクセス制御未設定
  # aws-rds-enable-performance-insights: パフォーマンスインサイト未設定
}

# OK: 修正後（セキュリティベストプラクティス準拠）
resource "aws_s3_bucket" "data" {
  bucket = "my-data-bucket"

  tags = {
    Environment = "production"
    Security    = "high"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_logging" "data" {
  bucket        = aws_s3_bucket.data.id
  target_bucket = aws_s3_bucket.logs.id
  target_prefix = "s3-access-logs/"
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "archive-old-objects"
    status = "Enabled"

    transition {
      days          = 90
      storage_class = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

resource "aws_db_instance" "main" {
  engine                  = "postgres"
  engine_version          = "15.4"
  instance_class          = "db.t3.medium"
  storage_encrypted       = true
  kms_key_id              = aws_kms_key.rds.arn
  publicly_accessible     = false
  multi_az                = true
  backup_retention_period = 35
  deletion_protection     = true

  performance_insights_enabled    = true
  performance_insights_kms_key_id = aws_kms_key.rds.arn

  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.private.name

  tags = {
    Environment = "production"
  }
}
```

### tfsec のインライン抑制

```hcl
# 特定のルールを正当な理由で抑制
resource "aws_security_group_rule" "https" {
  type              = "ingress"
  from_port         = 443
  to_port           = 443
  protocol          = "tcp"
  cidr_blocks       = ["0.0.0.0/0"]  #tfsec:ignore:aws-vpc-no-public-ingress-sgr -- Public HTTPS endpoint
  security_group_id = aws_security_group.alb.id
}

# 複数ルールの同時抑制
resource "aws_s3_bucket" "public_assets" {
  bucket = "my-public-assets"
  #tfsec:ignore:aws-s3-block-public-acls -- CDN origin for public assets
  #tfsec:ignore:aws-s3-block-public-policy -- Intentionally public
}

# 有効期限付き抑制 (tfsec 1.28+)
resource "aws_instance" "legacy" {
  ami           = "ami-xxx"
  instance_type = "t3.micro"
  #tfsec:ignore:aws-ec2-enforce-http-token-imds:exp:2024-12-31 -- Migration planned
}
```

### tfsec カスタムルール

```yaml
# .tfsec/custom_checks.yaml
---
checks:
  - code: CUS001
    description: "S3 bucket must have environment tag"
    impact: "Cannot track resource ownership"
    resolution: "Add 'Environment' tag to the bucket"
    requiredTypes:
      - resource
    requiredLabels:
      - aws_s3_bucket
    severity: MEDIUM
    matchSpec:
      name: tags
      action: contains
      value: Environment

  - code: CUS002
    description: "RDS instance must have deletion protection"
    impact: "Database can be accidentally deleted"
    resolution: "Set deletion_protection to true"
    requiredTypes:
      - resource
    requiredLabels:
      - aws_db_instance
    severity: HIGH
    matchSpec:
      name: deletion_protection
      action: equals
      value: true

  - code: CUS003
    description: "EC2 instance must use IMDSv2"
    impact: "SSRF attacks can access instance metadata"
    resolution: "Set metadata_options http_tokens to required"
    requiredTypes:
      - resource
    requiredLabels:
      - aws_instance
    severity: CRITICAL
    matchSpec:
      name: metadata_options
      action: isPresent
      subMatch:
        name: http_tokens
        action: equals
        value: required
```

---

## 3. Checkov (マルチフレームワーク対応)

### Checkov の特徴

| 項目 | tfsec | Checkov | KICS | Trivy |
|------|-------|---------|------|-------|
| 対応 IaC | Terraform | TF, CFn, K8s, ARM, Docker, Helm | 多数 | TF, CFn, K8s, Docker, Helm |
| ルール数 | ~1000 | ~2500 | ~2000 | ~1500 |
| カスタムポリシー | YAML/Rego | Python/YAML/Rego | Rego | Rego |
| グラフベース解析 | 部分的 | あり (依存関係解析) | なし | 部分的 |
| SCA 機能 | なし | あり (OSS脆弱性) | なし | あり |
| CI/CD 統合 | GitHub Action | GitHub Action, pre-commit | GitHub Action | GitHub Action |
| 修正提案 | 部分的 | あり (自動修正PR) | なし | 部分的 |
| ライセンス管理 | なし | あり | なし | あり |
| 実行速度 | 高速 | 中程度 | 中程度 | 高速 |

### Checkov の内部アーキテクチャ

```
┌─────────────────── Checkov 内部処理フロー ─────────────────┐
│                                                            │
│  1. フレームワーク検出                                      │
│     ├── ファイル拡張子/内容からフレームワークを判定           │
│     ├── .tf → Terraform                                   │
│     ├── template.yaml → CloudFormation                    │
│     ├── Dockerfile → Dockerfile                           │
│     └── deployment.yaml → Kubernetes                      │
│                                                            │
│  2. パーサー（フレームワーク別）                              │
│     ├── TerraformParser → HCL AST                         │
│     ├── CFNParser → YAML/JSON DOM                         │
│     ├── KubernetesParser → YAML DOM                       │
│     └── DockerfileParser → 命令リスト                      │
│                                                            │
│  3. リソースグラフ（Checkov 独自機能）                       │
│     ├── リソース間の参照・依存関係を解析                     │
│     ├── 例: SG ← EC2 の関係を追跡                          │
│     ├── グラフベースのポリシー評価が可能                     │
│     └── 複数リソースにまたがるルールを記述可能               │
│                                                            │
│  4. チェックランナー                                        │
│     ├── Python チェック（BaseResourceCheck）                │
│     ├── YAML チェック                                      │
│     ├── グラフチェック                                      │
│     └── External チェック（GitHub URL から取得）             │
│                                                            │
│  5. 結果レポーター                                          │
│     ├── CLI / JSON / SARIF / JUnit / CSV / CycloneDX     │
│     └── Bridgecrew プラットフォーム連携                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Checkov の使い方

```bash
# インストール
pip install checkov

# Terraform スキャン
checkov -d . --framework terraform

# Kubernetes マニフェストスキャン
checkov -d ./k8s/ --framework kubernetes

# Dockerfile スキャン
checkov --file Dockerfile --framework dockerfile

# Helm チャートスキャン
checkov -d ./charts/ --framework helm

# CloudFormation スキャン
checkov --file template.yaml --framework cloudformation

# 複数フレームワーク同時スキャン
checkov -d . --framework terraform,kubernetes,dockerfile

# 特定のチェックのみ実行
checkov -d . --check CKV_AWS_18,CKV_AWS_19,CKV_AWS_21

# 特定のチェックをスキップ
checkov -d . --skip-check CKV_AWS_999

# 出力形式
checkov -d . -o json > checkov-results.json
checkov -d . -o sarif > checkov-results.sarif
checkov -d . -o junitxml > checkov-results.xml
checkov -d . -o cyclonedx > checkov-sbom.xml

# カスタムポリシーディレクトリを指定
checkov -d . --external-checks-dir ./custom_checks

# 外部チェック（GitHub から取得）
checkov -d . --external-checks-git "https://github.com/myorg/checkov-policies"

# コンパクト出力（パスしたチェックを非表示）
checkov -d . --compact

# 既知の問題を無視（ベースライン）
checkov -d . --baseline checkov-baseline.json

# ベースラインの作成
checkov -d . --create-baseline

# SCA スキャン (依存関係の脆弱性チェック)
checkov -d . --framework sca_package
```

### Checkov カスタムポリシー (Python)

```python
# custom_checks/s3_naming_convention.py
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck
from checkov.common.models.enums import CheckResult, CheckCategories

class S3NamingConvention(BaseResourceCheck):
    """S3 バケット名が命名規則に従っているか"""

    def __init__(self):
        name = "S3 bucket follows naming convention: {env}-{service}-{purpose}"
        id = "CKV_CUSTOM_1"
        supported_resources = ["aws_s3_bucket"]
        categories = [CheckCategories.CONVENTION]
        super().__init__(name=name, id=id, categories=categories,
                        supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        bucket_name = conf.get("bucket", [""])[0]
        # 命名規則: {env}-{service}-{purpose}
        valid_prefixes = ["prod-", "stg-", "dev-", "shared-"]
        if any(bucket_name.startswith(prefix) for prefix in valid_prefixes):
            return CheckResult.PASSED
        return CheckResult.FAILED

check = S3NamingConvention()
```

```python
# custom_checks/rds_backup_retention.py
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck
from checkov.common.models.enums import CheckResult, CheckCategories

class RDSBackupRetention(BaseResourceCheck):
    """RDS のバックアップ保持期間が 30 日以上であること"""

    def __init__(self):
        name = "RDS backup retention period is at least 30 days"
        id = "CKV_CUSTOM_2"
        supported_resources = ["aws_db_instance"]
        categories = [CheckCategories.BACKUP_AND_RECOVERY]
        super().__init__(name=name, id=id, categories=categories,
                        supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        retention = conf.get("backup_retention_period", [0])
        if isinstance(retention, list):
            retention = retention[0]
        if isinstance(retention, int) and retention >= 30:
            return CheckResult.PASSED
        return CheckResult.FAILED

check = RDSBackupRetention()
```

```python
# custom_checks/ec2_imdsv2.py
from checkov.terraform.checks.resource.base_resource_check import BaseResourceCheck
from checkov.common.models.enums import CheckResult, CheckCategories

class EC2IMDSv2Required(BaseResourceCheck):
    """EC2 インスタンスで IMDSv2 が必須であること"""

    def __init__(self):
        name = "EC2 instance requires IMDSv2"
        id = "CKV_CUSTOM_3"
        supported_resources = ["aws_instance", "aws_launch_template"]
        categories = [CheckCategories.GENERAL_SECURITY]
        super().__init__(name=name, id=id, categories=categories,
                        supported_resources=supported_resources)

    def scan_resource_conf(self, conf):
        metadata_options = conf.get("metadata_options", [{}])
        if isinstance(metadata_options, list):
            metadata_options = metadata_options[0] if metadata_options else {}

        http_tokens = metadata_options.get("http_tokens", ["optional"])
        if isinstance(http_tokens, list):
            http_tokens = http_tokens[0]

        if http_tokens == "required":
            return CheckResult.PASSED
        return CheckResult.FAILED

check = EC2IMDSv2Required()
```

### Checkov グラフベースポリシー (YAML)

```yaml
# custom_checks/graph/s3_has_encryption_and_logging.yaml
---
metadata:
  id: "CKV_GRAPH_1"
  name: "S3 bucket has both encryption and logging configured"
  category: "ENCRYPTION"
  severity: "HIGH"
definition:
  and:
    - resource_types:
        - aws_s3_bucket
      connected_resource_types:
        - aws_s3_bucket_server_side_encryption_configuration
      operator: exists
    - resource_types:
        - aws_s3_bucket
      connected_resource_types:
        - aws_s3_bucket_logging
      operator: exists
```

### Checkov のインライン抑制

```hcl
# Checkov のインライン抑制（HCL コメント）
resource "aws_s3_bucket" "public_website" {
  bucket = "my-public-website"
  #checkov:skip=CKV_AWS_18: "Intentionally public - static website hosting"
  #checkov:skip=CKV_AWS_19: "Public website does not require encryption"
}

# 複数チェックの同時抑制
resource "aws_instance" "bastion" {
  #checkov:skip=CKV_AWS_88: "Bastion host requires public IP"
  #checkov:skip=CKV_AWS_135: "EBS optimization not needed for t3.micro"
  ami                         = "ami-xxx"
  instance_type               = "t3.micro"
  associate_public_ip_address = true
}
```

### .checkov.yaml 設定ファイル

```yaml
# .checkov.yaml
---
# グローバル設定
compact: true
directory:
  - "terraform/"
  - "k8s/"
framework:
  - terraform
  - kubernetes
  - dockerfile

# 除外するチェック
skip-check:
  - CKV_AWS_999  # 組織のポリシーで例外

# 除外するパス
skip-path:
  - "terraform/modules/legacy/"
  - "terraform/sandbox/"

# カスタムチェック
external-checks-dir:
  - "custom_checks/"

# 出力形式
output:
  - cli
  - sarif

# ソフトフェイル（CI を止めない）
soft-fail: false

# ソフトフェイル対象のチェック
soft-fail-on:
  - CKV_AWS_18  # 一時的に許可

# ハードフェイル対象のチェック
hard-fail-on:
  - CKV_AWS_145  # S3 暗号化は絶対必須
  - CKV_AWS_19   # S3 暗号化（旧ルール）
```

---

## 4. KICS と Trivy（追加の静的解析ツール）

### KICS (Keeping Infrastructure as Code Secure)

```bash
# Docker で実行
docker run -v $(pwd):/path checkmarx/kics:latest scan \
  --path /path \
  --output-path /path/results \
  --type Terraform,Kubernetes,Dockerfile

# レポート形式
docker run -v $(pwd):/path checkmarx/kics:latest scan \
  --path /path \
  --report-formats "sarif,json,html" \
  --output-path /path/results
```

### Trivy (統合セキュリティスキャナ)

```bash
# IaC スキャン（tfsec 後継）
trivy config .

# 特定フレームワーク
trivy config --tf-vars production.tfvars .

# 特定の重大度のみ
trivy config --severity HIGH,CRITICAL .

# JSON 出力
trivy config --format json --output results.json .

# Rego カスタムポリシー
trivy config --policy ./policies --namespaces custom .

# コンテナイメージスキャン（IaC + 脆弱性統合）
trivy image myapp:latest

# ファイルシステムスキャン（IaC + シークレット + 脆弱性）
trivy fs --scanners vuln,secret,misconfig .
```

### ツール選定ガイド

```
┌─────────────────── ツール選定フローチャート ─────────────────┐
│                                                             │
│  Q: どのフレームワークを使っているか?                        │
│                                                             │
│  Terraform のみ                                             │
│  └→ tfsec / trivy config（高速でシンプル）                  │
│                                                             │
│  Terraform + Kubernetes + Dockerfile                        │
│  └→ Checkov（マルチフレームワーク対応 + グラフ解析）        │
│                                                             │
│  多数のフレームワーク + カスタムポリシー重視                  │
│  └→ KICS（広範なフレームワーク対応）                        │
│                                                             │
│  統合セキュリティ（IaC + コンテナ + SCA）                   │
│  └→ Trivy（Aqua Security 統合プラットフォーム）             │
│                                                             │
│  エンタープライズ（Terraform Cloud/Enterprise 使用）         │
│  └→ Sentinel（HashiCorp 純正ポリシーエンジン）              │
│                                                             │
│  推奨: 複数ツールの併用                                      │
│  └→ tfsec(ローカル) + Checkov(CI) + OPA(ポリシーゲート)    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. CI/CD 統合

### GitHub Actions での統合パイプライン

```yaml
# .github/workflows/iac-security.yaml
name: IaC Security
on:
  pull_request:
    paths: ['terraform/**', 'k8s/**', 'Dockerfile*']

jobs:
  iac-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
      pull-requests: write

    steps:
      - uses: actions/checkout@v4

      # --- tfsec ---
      - name: tfsec
        uses: aquasecurity/tfsec-action@v1.0.0
        with:
          working_directory: terraform/
          soft_fail: false
          format: sarif
          sarif_file: tfsec.sarif

      # --- Checkov ---
      - name: Checkov
        uses: bridgecrewio/checkov-action@master
        with:
          directory: terraform/
          framework: terraform
          output_format: sarif
          output_file_path: checkov.sarif
          soft_fail: false
          skip_check: CKV_AWS_999

      # --- KICS ---
      - name: KICS Scan
        uses: Checkmarx/kics-github-action@v2.1.0
        with:
          path: 'terraform/,k8s/'
          output_path: kics-results/
          output_formats: 'sarif,json'
          fail_on: high

      # --- Trivy ---
      - name: Trivy Config Scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'config'
          scan-ref: 'terraform/'
          format: 'sarif'
          output: 'trivy.sarif'
          severity: 'HIGH,CRITICAL'

      # --- SARIF アップロード ---
      - name: Upload tfsec SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: tfsec.sarif
          category: tfsec

      - name: Upload Checkov SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: checkov.sarif
          category: checkov

      # --- PR コメント ---
      - name: Post Results to PR
        if: always() && github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            // 各ツールの結果を読み取ってPRにコメント
            const body = `## IaC Security Scan Results
            - tfsec: ✅ / ❌
            - Checkov: ✅ / ❌
            - KICS: ✅ / ❌
            - Trivy: ✅ / ❌`;
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

### GitLab CI での統合

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - security-scan
  - plan
  - apply

tfsec:
  stage: security-scan
  image: aquasec/tfsec:latest
  script:
    - tfsec terraform/ --minimum-severity HIGH --format json --out tfsec.json
  artifacts:
    reports:
      sast: tfsec.json
    when: always

checkov:
  stage: security-scan
  image: bridgecrew/checkov:latest
  script:
    - checkov -d terraform/ --framework terraform -o junitxml > checkov.xml
  artifacts:
    reports:
      junit: checkov.xml
    when: always

terraform-plan:
  stage: plan
  needs: ['tfsec', 'checkov']
  script:
    - terraform init
    - terraform plan -out=tfplan
    - terraform show -json tfplan > tfplan.json
    - conftest test tfplan.json --policy policy/
```

### Pre-commit フック設定

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/antonbabenko/pre-commit-terraform
    rev: v1.86.0
    hooks:
      - id: terraform_fmt
      - id: terraform_validate
      - id: terraform_tflint
        args:
          - '--args=--config=__GIT_WORKING_DIR__/.tflint.hcl'
      - id: terraform_tfsec
        args:
          - '--args=--minimum-severity HIGH'
      - id: terraform_checkov
        args:
          - '--args=--compact --quiet'

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.18.0
    hooks:
      - id: gitleaks

  - repo: https://github.com/awslabs/git-secrets
    rev: master
    hooks:
      - id: git-secrets
```

---

## 6. ポリシー as コード (OPA / Sentinel)

### OPA (Open Policy Agent) + Rego

OPA は CNCF 卒業プロジェクトであり、汎用的なポリシーエンジンとして広く採用されている。Rego はOPA のポリシー記述言語で、宣言的にポリシーを定義できる。

#### Rego の基本構文

```rego
# Rego の基本構文
package terraform.rules

# インポート
import input
import future.keywords.in
import future.keywords.if
import future.keywords.contains

# ヘルパー関数
is_aws_resource(type) if startswith(type, "aws_")

# 定数定義
allowed_regions := {"ap-northeast-1", "us-east-1", "eu-west-1"}

# deny ルール: 条件を満たすとポリシー違反
deny contains msg if {
    some resource_type, name
    resource := input.resource[resource_type][name]
    is_aws_resource(resource_type)
    not has_required_tags(resource)
    msg := sprintf(
        "%s.%s: Required tags (Environment, Team, CostCenter) are missing",
        [resource_type, name]
    )
}

# ヘルパー: 必須タグの存在チェック
has_required_tags(resource) if {
    required := {"Environment", "Team", "CostCenter"}
    tags := object.keys(resource.tags)
    missing := required - {tag | some tag in tags}
    count(missing) == 0
}
```

#### S3 セキュリティポリシー

```rego
# policy/terraform/s3.rego
package terraform.s3

import future.keywords.in
import future.keywords.if
import future.keywords.contains

# S3 バケットの暗号化を必須化
deny contains msg if {
    some name
    resource := input.resource.aws_s3_bucket[name]
    not has_encryption(name)
    msg := sprintf("S3 bucket '%s' must have server-side encryption enabled", [name])
}

has_encryption(bucket_name) if {
    some config_name
    config := input.resource.aws_s3_bucket_server_side_encryption_configuration[config_name]
    config.bucket == bucket_name
}

# パブリックアクセスブロックを必須化
deny contains msg if {
    some name
    resource := input.resource.aws_s3_bucket[name]
    not has_public_access_block(name)
    msg := sprintf("S3 bucket '%s' must have public access block", [name])
}

has_public_access_block(bucket_name) if {
    some block_name
    block := input.resource.aws_s3_bucket_public_access_block[block_name]
    block.bucket == bucket_name
    block.block_public_acls == true
    block.block_public_policy == true
    block.ignore_public_acls == true
    block.restrict_public_buckets == true
}

# バージョニングを必須化
deny contains msg if {
    some name
    resource := input.resource.aws_s3_bucket[name]
    not has_versioning(name)
    msg := sprintf("S3 bucket '%s' must have versioning enabled", [name])
}

has_versioning(bucket_name) if {
    some ver_name
    ver := input.resource.aws_s3_bucket_versioning[ver_name]
    ver.bucket == bucket_name
    ver.versioning_configuration.status == "Enabled"
}

# KMS 暗号化を推奨 (SSE-S3 ではなく SSE-KMS)
warn contains msg if {
    some config_name
    config := input.resource.aws_s3_bucket_server_side_encryption_configuration[config_name]
    rule := config.rule
    sse := rule.apply_server_side_encryption_by_default
    sse.sse_algorithm != "aws:kms"
    msg := sprintf(
        "S3 encryption config '%s': SSE-KMS is recommended over SSE-S3 for better key management",
        [config_name]
    )
}
```

#### IAM セキュリティポリシー

```rego
# policy/terraform/iam.rego
package terraform.iam

import future.keywords.in
import future.keywords.if
import future.keywords.contains

# ワイルドカード Action の禁止
deny contains msg if {
    some name
    policy := input.resource.aws_iam_policy[name]
    statement := json.unmarshal(policy.policy).Statement[_]
    statement.Effect == "Allow"
    action := statement.Action
    action == "*"
    msg := sprintf("IAM policy '%s': Wildcard (*) actions are not allowed", [name])
}

# ワイルドカード Action の禁止 (配列内)
deny contains msg if {
    some name
    policy := input.resource.aws_iam_policy[name]
    statement := json.unmarshal(policy.policy).Statement[_]
    statement.Effect == "Allow"
    some action in statement.Action
    action == "*"
    msg := sprintf("IAM policy '%s': Wildcard (*) actions are not allowed", [name])
}

# ワイルドカード Resource の禁止
deny contains msg if {
    some name
    policy := input.resource.aws_iam_policy[name]
    statement := json.unmarshal(policy.policy).Statement[_]
    statement.Effect == "Allow"
    statement.Resource == "*"
    msg := sprintf("IAM policy '%s': Wildcard (*) resource is not allowed. Specify exact ARN.", [name])
}

# 管理者ポリシー (AdministratorAccess) の直接アタッチ禁止
deny contains msg if {
    some name
    attachment := input.resource.aws_iam_policy_attachment[name]
    contains(attachment.policy_arn, "AdministratorAccess")
    msg := sprintf("IAM policy attachment '%s': AdministratorAccess should not be directly attached", [name])
}

# インラインポリシーの使用を警告
warn contains msg if {
    some name
    policy := input.resource.aws_iam_role_policy[name]
    msg := sprintf("IAM role policy '%s': Prefer managed policies over inline policies for better reusability", [name])
}
```

#### ネットワークセキュリティポリシー

```rego
# policy/terraform/network.rego
package terraform.network

import future.keywords.in
import future.keywords.if
import future.keywords.contains

# SSH の全世界公開を禁止
deny contains msg if {
    some name
    rule := input.resource.aws_security_group_rule[name]
    rule.type == "ingress"
    rule.from_port <= 22
    rule.to_port >= 22
    some cidr in rule.cidr_blocks
    cidr == "0.0.0.0/0"
    msg := sprintf("Security group rule '%s': SSH (port 22) must not be open to 0.0.0.0/0", [name])
}

# RDP の全世界公開を禁止
deny contains msg if {
    some name
    rule := input.resource.aws_security_group_rule[name]
    rule.type == "ingress"
    rule.from_port <= 3389
    rule.to_port >= 3389
    some cidr in rule.cidr_blocks
    cidr == "0.0.0.0/0"
    msg := sprintf("Security group rule '%s': RDP (port 3389) must not be open to 0.0.0.0/0", [name])
}

# データベースポートの全世界公開を禁止
db_ports := {3306, 5432, 1433, 27017, 6379}

deny contains msg if {
    some name
    rule := input.resource.aws_security_group_rule[name]
    rule.type == "ingress"
    some port in db_ports
    rule.from_port <= port
    rule.to_port >= port
    some cidr in rule.cidr_blocks
    cidr == "0.0.0.0/0"
    msg := sprintf("Security group rule '%s': Database port %d must not be open to the internet", [name, port])
}
```

### Conftest による OPA ポリシーテスト

```bash
# Terraform plan を JSON に変換
terraform plan -out=tfplan
terraform show -json tfplan > tfplan.json

# OPA ポリシーでテスト
conftest test tfplan.json --policy policy/

# 出力例:
# FAIL - tfplan.json - terraform.s3 - S3 bucket 'data' must have encryption
# FAIL - tfplan.json - terraform.iam - IAM policy 'admin': Wildcard (*) actions not allowed
# WARN - tfplan.json - terraform.iam - Prefer managed policies over inline policies
# 5 tests, 2 passed, 1 warnings, 2 failures

# 特定のネームスペースのみテスト
conftest test tfplan.json --policy policy/ --namespace terraform.s3

# JSON 出力
conftest test tfplan.json --policy policy/ --output json

# ポリシーのユニットテスト
conftest verify --policy policy/
```

### OPA ポリシーのユニットテスト

```rego
# policy/tests/s3_test.rego
package terraform.s3

# テストデータ: 暗号化なし → deny
test_deny_s3_without_encryption {
    deny with input as {
        "resource": {
            "aws_s3_bucket": {
                "test_bucket": {
                    "bucket": "test-bucket"
                }
            }
        }
    }
}

# テストデータ: 暗号化あり → deny なし
test_allow_s3_with_encryption {
    count(deny) == 0 with input as {
        "resource": {
            "aws_s3_bucket": {
                "test_bucket": {
                    "bucket": "test-bucket"
                }
            },
            "aws_s3_bucket_server_side_encryption_configuration": {
                "test_config": {
                    "bucket": "test_bucket",
                    "rule": {
                        "apply_server_side_encryption_by_default": {
                            "sse_algorithm": "aws:kms"
                        }
                    }
                }
            },
            "aws_s3_bucket_public_access_block": {
                "test_block": {
                    "bucket": "test_bucket",
                    "block_public_acls": true,
                    "block_public_policy": true,
                    "ignore_public_acls": true,
                    "restrict_public_buckets": true
                }
            },
            "aws_s3_bucket_versioning": {
                "test_ver": {
                    "bucket": "test_bucket",
                    "versioning_configuration": {
                        "status": "Enabled"
                    }
                }
            }
        }
    }
}
```

```bash
# ポリシーテストの実行
opa test policy/ -v
# policy/tests/s3_test.rego:
# data.terraform.s3.test_deny_s3_without_encryption: PASS
# data.terraform.s3.test_allow_s3_with_encryption: PASS
```

### HashiCorp Sentinel (Terraform Cloud/Enterprise)

```python
# sentinel/s3-encryption.sentinel
import "tfplan/v2" as tfplan

# S3 バケットの暗号化ルール
s3_buckets = filter tfplan.resource_changes as _, rc {
    rc.type is "aws_s3_bucket" and
    rc.mode is "managed" and
    (rc.change.actions contains "create" or rc.change.actions contains "update")
}

# 全 S3 バケットが暗号化設定を持つことを検証
encryption_required = rule {
    all s3_buckets as _, bucket {
        bucket.change.after is not null
    }
}

# メインルール
main = rule {
    encryption_required
}
```

### ポリシーリポジトリ構成

```
policy/
  ├── terraform/
  │   ├── s3.rego           # S3 ポリシー
  │   ├── iam.rego          # IAM ポリシー
  │   ├── network.rego      # ネットワークポリシー
  │   ├── encryption.rego   # 暗号化ポリシー
  │   ├── tagging.rego      # タグ付けポリシー
  │   └── compliance.rego   # コンプライアンスポリシー
  ├── kubernetes/
  │   ├── pod_security.rego
  │   ├── network_policy.rego
  │   └── rbac.rego
  ├── dockerfile/
  │   ├── base_image.rego
  │   └── user.rego
  ├── tests/
  │   ├── s3_test.rego      # ポリシーのユニットテスト
  │   ├── iam_test.rego
  │   └── network_test.rego
  ├── lib/
  │   └── helpers.rego      # 共通ヘルパー関数
  └── README.md
```

---

## 7. シークレット管理と IaC

### IaC コードにおけるシークレット漏洩の防止

```
┌──────────── シークレット漏洩の経路 ─────────────┐
│                                                  │
│  1. ハードコード                                 │
│     ├── .tf ファイルに直接記述                    │
│     ├── terraform.tfvars に平文記述              │
│     └── 変数のデフォルト値に記述                  │
│                                                  │
│  2. State ファイル                               │
│     ├── terraform.tfstate に平文保存              │
│     ├── plan ファイルに含まれる値                 │
│     └── output 変数の sensitive 未設定            │
│                                                  │
│  3. Git 履歴                                     │
│     ├── 過去のコミットにシークレットが残存         │
│     ├── .gitignore の不足                        │
│     └── ブランチ削除しても履歴は残る              │
│                                                  │
│  4. CI/CD ログ                                   │
│     ├── terraform plan の出力に値が表示           │
│     ├── 環境変数のデバッグ出力                    │
│     └── エラーメッセージに含まれる値              │
│                                                  │
└──────────────────────────────────────────────────┘
```

### シークレット検知ツール

```bash
# gitleaks: Git リポジトリのシークレットスキャン
gitleaks detect --source . --report-format json --report-path gitleaks-report.json

# git-secrets: AWS 認証情報に特化
git secrets --scan

# trufflehog: エントロピーベースの検知
trufflehog filesystem --directory . --json

# Trivy のシークレットスキャン
trivy fs --scanners secret .
```

### Terraform でのシークレット管理ベストプラクティス

```hcl
# NG: ハードコードされたシークレット
resource "aws_db_instance" "main" {
  username = "admin"
  password = "SuperSecret123!"  # 絶対にやってはいけない
}

# OK: Secrets Manager から取得
data "aws_secretsmanager_secret_version" "db_credentials" {
  secret_id = "myapp/db/credentials"
}

locals {
  db_creds = jsondecode(data.aws_secretsmanager_secret_version.db_credentials.secret_string)
}

resource "aws_db_instance" "main" {
  username = local.db_creds.username
  password = local.db_creds.password
}

# OK: 変数 + CI/CD 環境変数
variable "db_password" {
  type      = string
  sensitive = true  # plan 出力で非表示
}

resource "aws_db_instance" "main" {
  username = "admin"
  password = var.db_password  # TF_VAR_db_password 環境変数から
}

# output の sensitive マーク
output "db_endpoint" {
  value     = aws_db_instance.main.endpoint
  sensitive = false  # エンドポイントは公開可
}

output "db_password" {
  value     = aws_db_instance.main.password
  sensitive = true  # シークレットは非表示
}
```

---

## 8. ドリフト検知

### ドリフトとは

IaC コード (あるべき状態) と実際のインフラ (現在の状態) の間に生じる乖離をドリフトと呼ぶ。ドリフトは手動変更、別のツールによる変更、または外部要因によって発生する。

```
IaC コード (あるべき状態)     実際のインフラ (現在の状態)
+-----------------------+    +-----------------------+
| SG: port 443 のみ許可 |    | SG: port 443 + 22     |
|                       | != |  (手動で SSH 追加)     |
| 暗号化: 有効           |    | 暗号化: 有効           |
| タグ: Env=production  |    | タグ: Env=prod (変更) |
+-----------------------+    +-----------------------+
                                  ↑
                              ドリフト (乖離)

発生原因:
  1. コンソールからの手動変更
  2. 他チームの直接 API 操作
  3. AWS の自動更新（セキュリティパッチなど）
  4. 別の IaC コードによる変更
  5. インシデント対応時の緊急変更
```

### ドリフト検知の手法

```bash
# Terraform でドリフト検知
terraform plan -detailed-exitcode
# Exit code 0 = 変更なし
# Exit code 1 = エラー
# Exit code 2 = ドリフトあり

# 定期実行スクリプト
#!/bin/bash
terraform init -backend=true
terraform plan -detailed-exitcode -out=drift-check.plan 2>&1
EXIT_CODE=$?

if [ $EXIT_CODE -eq 2 ]; then
    echo "DRIFT DETECTED!"
    terraform show -json drift-check.plan > drift-report.json
    # Slack 通知等
fi

# AWS Config でドリフト検知 (CloudFormation)
aws cloudformation detect-stack-drift --stack-name my-stack
aws cloudformation describe-stack-drift-detection-status \
    --stack-drift-detection-id <detection-id>

# driftctl (専用ツール - 非推奨、後継は snyk)
driftctl scan --from tfstate://terraform.tfstate

# Terraform Cloud/Enterprise
# ドリフト検知が自動的に実行される（Health Assessment）
```

### ドリフト修復戦略

```
┌─────────────── ドリフト修復戦略 ────────────────┐
│                                                  │
│  戦略1: コードに合わせる（推奨）                  │
│  ├── terraform apply で IaC コードの状態に戻す    │
│  ├── 手動変更を完全に巻き戻す                    │
│  └── IaC が Single Source of Truth              │
│                                                  │
│  戦略2: コードを実態に合わせる                    │
│  ├── terraform import で取り込み                  │
│  ├── .tf ファイルを現状に合わせて更新             │
│  └── 変更に正当な理由がある場合                   │
│                                                  │
│  戦略3: 選択的修復                                │
│  ├── セキュリティ関連のドリフトは即時修復          │
│  ├── 非セキュリティのドリフトは計画的に対応        │
│  └── -target オプションでリソース単位の適用       │
│                                                  │
│  推奨アプローチ:                                  │
│  ├── 本番環境: 手動変更を禁止するポリシー          │
│  ├── ドリフト検知を CI/CD で定期実行              │
│  └── セキュリティドリフトは自動修復               │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 9. Terraform モジュールのセキュリティ

### サプライチェーンリスク

```
┌──────── Terraform モジュール サプライチェーン ────────┐
│                                                      │
│  リスク1: 悪意のあるモジュール                        │
│  ├── 非公式レジストリからのモジュール                  │
│  ├── 見た目は正常だがバックドアを含む                  │
│  └── IAM ロールの追加作成、データ外部送信など          │
│                                                      │
│  リスク2: バージョン固定忘れ                          │
│  ├── source = "terraform-aws-modules/vpc/aws"        │
│  ├── version 未指定 → 最新版を自動取得               │
│  └── 破壊的変更やセキュリティ劣化のリスク             │
│                                                      │
│  リスク3: Provider プラグインの改竄                   │
│  ├── 非公式ミラーからの取得                           │
│  ├── チェックサム未検証                               │
│  └── .terraform.lock.hcl の不管理                    │
│                                                      │
│  対策:                                               │
│  ├── 公式レジストリのモジュールのみ使用               │
│  ├── version 制約を必ず記述                           │
│  ├── .terraform.lock.hcl をバージョン管理             │
│  ├── モジュールのコードレビュー                       │
│  └── プライベートレジストリの運用                     │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### セキュアなモジュール利用

```hcl
# NG: バージョン未固定
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  # version 未指定 → 危険
}

# OK: バージョン固定 + 制約
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"  # 5.x の範囲で最新
}

# OK: 厳密な固定
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.4.0"  # 完全固定
}

# OK: プライベートレジストリ
module "vpc" {
  source  = "app.terraform.io/myorg/vpc/aws"
  version = "2.1.0"
}

# OK: Git リポジトリ（タグ固定）
module "vpc" {
  source = "git::https://github.com/myorg/terraform-modules.git//vpc?ref=v2.1.0"
}
```

### .terraform.lock.hcl の管理

```hcl
# .terraform.lock.hcl はバージョン管理に含める
# → Provider のバージョンとハッシュを固定

provider "registry.terraform.io/hashicorp/aws" {
  version     = "5.31.0"
  constraints = "~> 5.0"
  hashes = [
    "h1:XXXXXXXXXXXXXXXXXXXXXXXXXX=",
    "zh:XXXXXXXXXXXXXXXXXXXXXXXXXX",
  ]
}
```

```bash
# lock ファイルの更新
terraform init -upgrade

# プラットフォーム固定（CI/CD 環境を指定）
terraform providers lock \
  -platform=linux_amd64 \
  -platform=darwin_amd64 \
  -platform=darwin_arm64
```

---

## 10. アンチパターン

### アンチパターン 1: Terraform state ファイルの不安全な管理

```hcl
# NG: ローカルに state を保存 (暗号化なし、共有不可)
terraform {
  backend "local" {}
}

# NG: S3 バックエンドだが暗号化・ロックなし
terraform {
  backend "s3" {
    bucket = "my-terraform-state"
    key    = "terraform.tfstate"
    region = "ap-northeast-1"
    # encrypt 未指定 → 暗号化されない
    # dynamodb_table 未指定 → ロックなし
  }
}

# OK: リモート state + 暗号化 + ロック + アクセス制限
terraform {
  backend "s3" {
    bucket         = "my-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "ap-northeast-1"
    encrypt        = true
    kms_key_id     = "arn:aws:kms:ap-northeast-1:123456:key/xxx"
    dynamodb_table = "terraform-state-lock"

    # state バケットへのアクセスを制限
    # (バケットポリシーで特定 IAM ロールのみ許可)
  }
}
```

**影響**: state ファイルにはシークレット情報 (パスワード、APIキーなど) が含まれる可能性がある。平文でS3に保存された場合、バケットへのアクセス権を持つ全員がシークレットを閲覧できる。ロックがない場合、同時実行でインフラが破壊される可能性がある。

### アンチパターン 2: IaC スキャンの CI/CD 非統合

```
NG: 開発者がローカルでのみスキャンを実行
  → 忘れたり、結果を無視したりする
  → レビュアーがセキュリティ問題を見逃す
  → 本番環境に問題のあるコードがデプロイされる

OK: CI/CD でスキャンを強制し、失敗時はマージをブロック
  → PR のマージ条件に tfsec/Checkov のパスを含める
  → Branch Protection Rule で必須ステータスチェックに設定
  → 結果を SARIF で GitHub Security タブに集約
  → Slack/Teams で結果を通知
```

### アンチパターン 3: 例外の無管理

```
NG: 抑制コメントを理由なく大量に追加
  #tfsec:ignore:aws-s3-enable-bucket-encryption
  #checkov:skip=CKV_AWS_19

  → セキュリティチェックが形骸化
  → 本来修正すべき問題が放置される

OK: 例外管理プロセスの確立
  → 抑制には必ず理由コメントを付ける
  → 有効期限を設定する（:exp:2024-12-31）
  → 定期的に抑制をレビューし、不要な例外を削除
  → 例外の数をメトリクスとして監視
```

### アンチパターン 4: IaC とコンソールの並行管理

```
NG: 一部のリソースは IaC、一部はコンソールで管理
  → ドリフトが常態化
  → 「どちらが正しいか」が不明に
  → 障害時の切り戻しが困難

OK: IaC を Single Source of Truth とする
  → 全リソースを IaC で管理
  → コンソール操作は読み取り専用（ReadOnlyAccess）
  → 緊急時のコンソール変更は事後に IaC に反映
  → terraform import で既存リソースを取り込み
```

---

## 11. 演習問題

### 演習 1: tfsec / Checkov による脆弱性検出と修正

以下の Terraform コードに含まれるセキュリティ問題を全て特定し、修正せよ。

```hcl
# 演習用コード（問題あり）
provider "aws" {
  region     = "ap-northeast-1"
  access_key = "AKIA..."       # 問題1
  secret_key = "wJal..."       # 問題2
}

resource "aws_s3_bucket" "data" {
  bucket = "company-data"
  acl    = "public-read"       # 問題3
}

resource "aws_instance" "web" {
  ami           = "ami-xxx"
  instance_type = "t3.micro"
  # 問題4: metadata_options 未設定（IMDSv1 がデフォルト）
  # 問題5: vpc_security_group_ids 未設定
  # 問題6: monitoring 未有効化
}

resource "aws_security_group" "web" {
  name = "web-sg"

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]  # 問題7
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]  # 問題8（意図的ならOK）
  }
}

resource "aws_db_instance" "main" {
  engine            = "mysql"
  instance_class    = "db.t3.medium"
  username          = "admin"
  password          = "password123"   # 問題9
  publicly_accessible = true          # 問題10
  # 問題11: storage_encrypted 未設定
  # 問題12: backup_retention_period 未設定
}
```

**目標**: 全12個の問題を修正し、tfsec / Checkov でエラー0件にすること。

### 演習 2: OPA ポリシーの作成

以下の要件を満たす OPA ポリシーを Rego で作成せよ。

1. 全 EC2 インスタンスに `Environment`, `Team`, `CostCenter` タグが必須
2. RDS インスタンスは `publicly_accessible = false` であること
3. S3 バケット名は `{env}-{team}-` で始まること（env は prod/stg/dev のいずれか）
4. Security Group のインバウンドルールで port 22 が 0.0.0.0/0 に公開されていないこと
5. 各ルールに対応するユニットテストを作成すること

### 演習 3: CI/CD パイプラインの構築

以下の要件を満たす GitHub Actions ワークフローを作成せよ。

1. PR 作成時に tfsec, Checkov, gitleaks を実行
2. 全ツールの結果を SARIF で GitHub Security タブにアップロード
3. CRITICAL/HIGH の検出がある場合は PR のマージをブロック
4. 結果のサマリーを PR コメントに投稿
5. terraform plan の出力に対して Conftest でポリシーチェック

---

## 12. FAQ

### Q1. tfsec と Checkov のどちらを使うべきか?

Terraform のみを使っている場合は tfsec (現在は Trivy に統合) が高速でシンプルである。Terraform に加えて Kubernetes、Dockerfile、CloudFormation なども管理している場合は Checkov のマルチフレームワーク対応が有利である。両方を併用することで検出漏れを減らせる。エンタープライズ環境では Checkov の SCA 機能やグラフベース解析が価値を発揮する。

### Q2. OPA ポリシーの管理はどうすべきか?

ポリシーは専用の Git リポジトリで管理し、CI/CD で自動テストする。ポリシーの変更にもレビュープロセスを適用する。OPA のテストフレームワーク (`opa test`) でポリシーのユニットテストを書き、意図しない許可・拒否を防ぐ。ポリシーのバージョニングを行い、段階的なロールアウト (warn → deny) を推奨する。

### Q3. 既存インフラを IaC 化する際のセキュリティ考慮は?

`terraform import` で既存リソースを IaC に取り込んだ後、即座に tfsec/Checkov でスキャンする。多数のセキュリティ問題が見つかる場合は優先度をつけて段階的に修正する。ドリフト検知を有効にして IaC 外の手動変更を検出する。取り込み時に state ファイルにシークレットが含まれるため、バックエンドの暗号化を確認すること。

### Q4. IaC スキャンで大量の false positive が出る場合はどうするか?

まず、false positive のパターンを分析する。特定のルールが常に false positive を出す場合は `.checkov.yaml` や tfsec の設定で除外する。正当な例外は理由付きのインラインコメントで抑制する。組織固有の要件は、組み込みルールを無効にした上でカスタムポリシーで代替する。false positive 率をメトリクスとして追跡し、定期的にルールセットを見直す。

### Q5. Terraform Cloud/Enterprise の Sentinel と OPA のどちらを使うべきか?

Terraform Cloud/Enterprise を使用しているなら Sentinel が自然な選択である（terraform plan のコンテキストに直接アクセスできる）。マルチツール環境（Terraform + Kubernetes + CI/CD パイプライン）では OPA が汎用的で統一的なポリシーエンジンとして機能する。OPA は CNCF 卒業プロジェクトであり、エコシステムが広い。

### Q6. IaC セキュリティの成熟度をどう評価するか?

以下の段階で評価する。Level 1: ローカルでのアドホックなスキャン。Level 2: CI/CD に統合されたスキャン。Level 3: ポリシー as コードによるゲート。Level 4: ドリフト検知と自動修復。Level 5: 全リソースの IaC 化 + 継続的コンプライアンス監視。まずは Level 2 を目指し、段階的に成熟度を上げていく。

---

## まとめ

| 項目 | 要点 |
|------|------|
| IaC のリスク | 設定ミスがコードに残り、大規模に展開される |
| tfsec / Trivy | Terraform 特化の高速スキャナ。Trivy に統合済み |
| Checkov | マルチフレームワーク対応の包括的スキャナ。グラフベース解析が強み |
| KICS | Checkmarx 提供の広範なフレームワーク対応スキャナ |
| OPA/Rego | カスタムポリシーの定義と自動適用。CNCF 卒業プロジェクト |
| Sentinel | Terraform Cloud/Enterprise 専用のポリシーエンジン |
| CI/CD 統合 | PR のマージ条件にスキャンパスを必須化。SARIF で結果集約 |
| ドリフト検知 | IaC と実インフラの乖離を継続的に検出。定期実行を推奨 |
| State 管理 | リモート + 暗号化 + ロックで安全に管理 |
| シークレット | IaC コードにハードコードしない。Secrets Manager を活用 |
| モジュール | バージョン固定、公式レジストリ、lock ファイル管理 |

---

## 次に読むべきガイド

- [AWSセキュリティ](./01-aws-security.md) — IaC で設定する AWS セキュリティサービス
- [クラウドセキュリティ基礎](./00-cloud-security-basics.md) — IAM と暗号化の基礎
- [コンテナセキュリティ](../04-application-security/02-container-security.md) — Kubernetes マニフェストのセキュリティ

---

## 参考文献

1. **Checkov Documentation** — https://www.checkov.io/
2. **tfsec Documentation** — https://aquasecurity.github.io/tfsec/
3. **Trivy Documentation** — https://aquasecurity.github.io/trivy/
4. **Open Policy Agent (OPA) Documentation** — https://www.openpolicyagent.org/docs/latest/
5. **Rego Policy Language** — https://www.openpolicyagent.org/docs/latest/policy-language/
6. **HashiCorp Sentinel Documentation** — https://developer.hashicorp.com/sentinel/docs
7. **KICS Documentation** — https://docs.kics.io/
8. **Conftest** — https://www.conftest.dev/
9. **OWASP IaC Security** — https://owasp.org/www-project-devsecops-guideline/latest/02d-Infrastructure-as-Code
10. **CIS Benchmarks for Terraform** — https://www.cisecurity.org/benchmark/terraform
