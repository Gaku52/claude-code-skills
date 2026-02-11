# クラウドコンピューティング概要

> インターネット経由でコンピューティングリソースをオンデマンドで利用する仕組みを体系的に理解する

## この章で学ぶこと

1. クラウドコンピューティングの定義と5つの本質的特性を説明できる
2. IaaS / PaaS / SaaS の責任分界モデルを区別し、適切なサービスを選択できる
3. AWS / GCP / Azure の主要サービスを比較し、プロジェクトに最適なクラウドを判断できる

---

## 1. クラウドコンピューティングとは

### 1.1 NIST による定義

米国国立標準技術研究所 (NIST SP 800-145) はクラウドコンピューティングを次のように定義している。

> 共有プールの構成可能なコンピューティングリソース（ネットワーク、サーバー、ストレージ、アプリケーション、サービス）に対して、最小限の管理労力またはサービスプロバイダとのやり取りで迅速にプロビジョニングおよびリリースできる、便利なオンデマンドのネットワークアクセスを可能にするモデル

### 1.2 5つの本質的特性

```
+--------------------------------------------------+
|         クラウドの5つの本質的特性                    |
+--------------------------------------------------+
| 1. オンデマンド・セルフサービス                      |
|    - 人手を介さずリソースを即時確保                   |
| 2. 幅広いネットワークアクセス                        |
|    - 標準プロトコルでどこからでもアクセス              |
| 3. リソースプーリング                               |
|    - マルチテナントで物理リソースを共有               |
| 4. 迅速な弾力性 (Rapid Elasticity)                 |
|    - 需要に応じて自動スケール                        |
| 5. 従量課金 (Measured Service)                     |
|    - 使った分だけ支払い                              |
+--------------------------------------------------+
```

### 1.3 オンプレミス vs クラウド

```
+---------------------+      +---------------------+
|   オンプレミス        |      |   クラウド            |
+---------------------+      +---------------------+
| ハードウェア購入 必要  |      | ハードウェア購入 不要  |
| 初期費用 大           |      | 初期費用 小          |
| スケール 数週間~数月   |      | スケール 数分        |
| 運用 自社             |      | 運用 共有責任        |
| 減価償却 あり          |      | 減価償却 なし(OPEX)  |
+---------------------+      +---------------------+
```

---

## 2. サービスモデル — IaaS / PaaS / SaaS

### 2.1 責任分界モデル

```
管理責任の範囲（上に行くほどユーザー責任）

  +--------------+---------+---------+---------+
  |              | IaaS    | PaaS    | SaaS    |
  +--------------+---------+---------+---------+
  | アプリ       | ユーザー | ユーザー | ベンダー |
  | データ       | ユーザー | ユーザー | ベンダー |
  | ランタイム   | ユーザー | ベンダー | ベンダー |
  | ミドルウェア | ユーザー | ベンダー | ベンダー |
  | OS           | ユーザー | ベンダー | ベンダー |
  | 仮想化       | ベンダー | ベンダー | ベンダー |
  | サーバー     | ベンダー | ベンダー | ベンダー |
  | ストレージ   | ベンダー | ベンダー | ベンダー |
  | ネットワーク | ベンダー | ベンダー | ベンダー |
  +--------------+---------+---------+---------+
```

### 2.2 各モデルの代表サービス

| モデル | 概要 | AWS 例 | GCP 例 | Azure 例 |
|--------|------|--------|--------|----------|
| IaaS | 仮想マシン・ネットワークを提供 | EC2, VPC | Compute Engine | Virtual Machines |
| PaaS | アプリ実行基盤を提供 | Elastic Beanstalk, App Runner | App Engine | App Service |
| SaaS | 完成したアプリケーション | WorkMail, Chime | Google Workspace | Microsoft 365 |

### 2.3 コード例: IaaS (EC2) の起動

```bash
# EC2インスタンスを起動する最小コマンド
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.micro \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --count 1
```

### 2.4 コード例: PaaS (Elastic Beanstalk) デプロイ

```bash
# Elastic Beanstalk CLIで環境を作成
eb init my-app --platform python-3.11 --region ap-northeast-1
eb create my-env --instance-type t3.small
eb deploy
```

### 2.5 コード例: SaaS 連携 (AWS SES メール送信)

```python
import boto3

client = boto3.client('ses', region_name='ap-northeast-1')

response = client.send_email(
    Source='sender@example.com',
    Destination={'ToAddresses': ['recipient@example.com']},
    Message={
        'Subject': {'Data': 'テスト通知'},
        'Body': {'Text': {'Data': 'クラウドからのメール送信テストです'}}
    }
)
print(f"MessageId: {response['MessageId']}")
```

---

## 3. デプロイメントモデル

| モデル | 説明 | ユースケース |
|--------|------|-------------|
| パブリッククラウド | プロバイダが所有・運用する共有インフラ | スタートアップ、Web サービス |
| プライベートクラウド | 単一組織専用のクラウドインフラ | 金融機関、政府機関 |
| ハイブリッドクラウド | パブリック+オンプレミスの組合せ | 段階的移行、規制対応 |
| マルチクラウド | 複数プロバイダの組合せ | ベンダーロックイン回避 |

---

## 4. AWS vs GCP vs Azure — 主要サービス比較

### 4.1 コンピュート比較

| カテゴリ | AWS | GCP | Azure |
|----------|-----|-----|-------|
| 仮想マシン | EC2 | Compute Engine | Virtual Machines |
| コンテナ(マネージド) | ECS / EKS | GKE | AKS |
| サーバーレス | Lambda | Cloud Functions | Azure Functions |
| コンテナサーバーレス | Fargate | Cloud Run | Container Apps |

### 4.2 ストレージ/DB比較

| カテゴリ | AWS | GCP | Azure |
|----------|-----|-----|-------|
| オブジェクトストレージ | S3 | Cloud Storage | Blob Storage |
| RDB(マネージド) | RDS / Aurora | Cloud SQL / AlloyDB | Azure SQL |
| NoSQL | DynamoDB | Firestore / Bigtable | Cosmos DB |
| キャッシュ | ElastiCache | Memorystore | Azure Cache for Redis |

### 4.3 コード例: 各クラウドの CLI でバケット作成

```bash
# AWS
aws s3 mb s3://my-bucket-2024 --region ap-northeast-1

# GCP
gsutil mb -l asia-northeast1 gs://my-bucket-2024

# Azure
az storage container create --name my-bucket-2024 \
  --account-name mystorageaccount
```

### 4.4 コード例: 各クラウドの SDK — ファイルアップロード (Python)

```python
# === AWS S3 ===
import boto3
s3 = boto3.client('s3')
s3.upload_file('local.txt', 'my-bucket', 'remote.txt')

# === GCP Cloud Storage ===
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('my-bucket')
blob = bucket.blob('remote.txt')
blob.upload_from_filename('local.txt')

# === Azure Blob Storage ===
from azure.storage.blob import BlobServiceClient
blob_service = BlobServiceClient.from_connection_string(conn_str)
blob_client = blob_service.get_blob_client('my-container', 'remote.txt')
with open('local.txt', 'rb') as data:
    blob_client.upload_blob(data)
```

---

## 5. クラウド導入のメリットと課題

```
  メリット                          課題
  +-------------------------+      +-------------------------+
  | 初期投資の削減 (CAPEX→OPEX) |   | ベンダーロックイン        |
  | グローバル展開の容易さ      |   | データ主権・コンプライアンス|
  | 高可用性・耐障害性          |   | ネットワーク遅延          |
  | 自動スケーリング            |   | コスト管理の複雑さ        |
  | マネージドサービス活用      |   | セキュリティ責任の理解     |
  +-------------------------+      +-------------------------+
```

---

## 6. アンチパターン

### アンチパターン 1: リフト&シフトで終わらせる

オンプレミスの構成をそのままクラウドに移すだけでは、クラウドのメリット（自動スケーリング、マネージドサービス）を活かせず、むしろコストが高くなるケースが多い。移行後に「クラウドネイティブ化」のフェーズを計画すべきである。

```
# 悪い例: オンプレと同じ構成をそのまま再現
EC2 (常時起動 x 10台) + 自前 MySQL + 自前 Redis
↓
# 良い例: マネージドサービスを活用
Lambda/Fargate + Aurora Serverless + ElastiCache
```

### アンチパターン 2: 全てをひとつのクラウドアカウントで運用する

本番環境・開発環境・ステージング環境を単一アカウントで管理すると、権限分離やコスト把握が困難になる。AWS Organizations で環境ごとにアカウントを分離すべきである。

```
# 悪い例
1つのAWSアカウントに全環境を配置
↓
# 良い例
AWS Organizations
├── Management Account (請求・ガバナンス)
├── Production Account
├── Staging Account
└── Development Account
```

---

## 7. FAQ

### Q1. クラウドは本当にコスト削減になるのか？

必ずしもそうではない。常時稼働のワークロードはオンプレミスの方が安くなる場合がある。クラウドの真のメリットは「弾力性」と「俊敏性」にあり、変動するワークロードや新規プロジェクトの立ち上げでコスト優位性が高い。Reserved Instances や Savings Plans を活用すれば、固定ワークロードでも最大 72% の割引が可能。

### Q2. AWS / GCP / Azure のどれを選ぶべきか？

チームのスキルセット、既存の技術スタック、必要なサービスの成熟度で判断する。一般的に AWS はサービスの幅が最も広く、GCP はデータ分析・ML に強み、Azure は Microsoft エコシステム (Active Directory, Office 365) との親和性が高い。マルチクラウド戦略も有効だが、運用複雑性が増すため慎重に検討する。

### Q3. クラウドのセキュリティはオンプレミスより弱いのか？

「共有責任モデル」において、クラウドプロバイダは物理インフラのセキュリティを担い、ユーザーはデータとアクセス管理を担う。主要プロバイダは SOC 2、ISO 27001、PCI DSS などの認証を取得しており、適切に設定すればオンプレミスと同等以上のセキュリティを実現できる。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| クラウドの定義 | オンデマンドでリソースを確保・解放でき、従量課金で利用するモデル |
| サービスモデル | IaaS(インフラ) → PaaS(プラットフォーム) → SaaS(アプリ) の順に抽象度が上がる |
| デプロイモデル | パブリック、プライベート、ハイブリッド、マルチクラウドの4種 |
| AWS の強み | サービス数最多、グローバルリージョン最多、エコシステム成熟 |
| コスト最適化 | 従量課金 + 予約割引 + スポット活用の3層戦略が基本 |
| セキュリティ | 共有責任モデルを理解し、ユーザー側の設定を確実に行う |

---

## 次に読むべきガイド

- [01-aws-account-setup.md](./01-aws-account-setup.md) — AWS アカウントの作成と初期設定
- [02-aws-cli-sdk.md](./02-aws-cli-sdk.md) — CLI/SDK のセットアップと認証情報管理

---

## 参考文献

1. NIST SP 800-145 "The NIST Definition of Cloud Computing" — https://csrc.nist.gov/publications/detail/sp/800-145/final
2. AWS Well-Architected Framework — https://docs.aws.amazon.com/wellarchitected/latest/framework/welcome.html
3. Gartner "Magic Quadrant for Cloud Infrastructure and Platform Services" — https://www.gartner.com/en/documents/cloud-infrastructure-platform-services
4. AWS 共有責任モデル — https://aws.amazon.com/compliance/shared-responsibility-model/
