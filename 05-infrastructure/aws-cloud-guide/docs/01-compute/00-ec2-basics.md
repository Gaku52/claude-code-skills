# EC2 基礎

> AWS の仮想サーバー EC2 の基本概念 — インスタンスタイプ、AMI、セキュリティグループ、キーペア、EBS を体系的に理解する

## この章で学ぶこと

1. EC2 インスタンスのライフサイクルとインスタンスタイプの選定基準を理解できる
2. AMI、セキュリティグループ、キーペアを適切に設定し、安全にインスタンスを起動できる
3. EBS ボリュームの種類と特性を理解し、ワークロードに最適なストレージを選択できる

---

## 1. EC2 とは

Amazon Elastic Compute Cloud (EC2) は、AWS 上で仮想サーバー (インスタンス) をオンデマンドで起動・管理できるサービスである。

### 1.1 EC2 の主要コンポーネント

```
EC2 インスタンスの構成要素
+--------------------------------------------------+
|                  EC2 Instance                     |
|                                                   |
|  +-------------+  +---------------------------+  |
|  |    AMI      |  |   Instance Type            |  |
|  | (OS+ソフト)  |  | (CPU, メモリ, ネットワーク)  |  |
|  +-------------+  +---------------------------+  |
|                                                   |
|  +-------------+  +---------------------------+  |
|  |  Key Pair   |  |   Security Group           |  |
|  | (SSH認証)    |  | (ファイアウォール)           |  |
|  +-------------+  +---------------------------+  |
|                                                   |
|  +---------------------------------------------+ |
|  |           EBS Volume (ストレージ)              | |
|  |  ルートボリューム + 追加ボリューム              | |
|  +---------------------------------------------+ |
|                                                   |
|  +---------------------------------------------+ |
|  |           VPC / Subnet (ネットワーク)          | |
|  +---------------------------------------------+ |
+--------------------------------------------------+
```

### 1.2 インスタンスのライフサイクル

```
              +----------+
              | pending  |  ← 起動中
              +----+-----+
                   |
                   v
+--------+   +---------+   +----------+
| stopped| <-| running | ->| stopping |
+---+----+   +----+----+   +----------+
    |             |
    |             v
    |        +------------+
    +------> | terminated |  ← 削除（復元不可）
             +------------+

  起動: stopped → pending → running
  停止: running → stopping → stopped
  終了: running → shutting-down → terminated
```

---

## 2. インスタンスタイプ

### 2.1 命名規則

```
  m  5  .  xlarge
  |  |     |
  |  |     +-- サイズ (nano, micro, small, medium, large, xlarge, 2xlarge...)
  |  +-------- 世代番号
  +----------- ファミリー (汎用, コンピュート最適化, メモリ最適化...)

  追加属性の例:
  m5a.xlarge  → a: AMD プロセッサ
  m5g.xlarge  → g: Graviton (ARM) プロセッサ
  m5n.xlarge  → n: ネットワーク強化
```

### 2.2 インスタンスファミリー比較

| ファミリー | プレフィックス | 特徴 | ユースケース |
|-----------|-------------|------|-------------|
| 汎用 | t3, m5, m6i, m7g | CPU/メモリバランス | Web サーバー、小中規模 DB |
| コンピュート最適化 | c5, c6i, c7g | 高 CPU 性能 | バッチ処理、機械学習推論 |
| メモリ最適化 | r5, r6i, x2idn | 大容量メモリ | インメモリ DB、ビッグデータ |
| ストレージ最適化 | i3, d3, h1 | 高 I/O | データウェアハウス、ログ処理 |
| 高速コンピューティング | p4, g5, inf2 | GPU / 推論チップ | 機械学習訓練、動画処理 |

### 2.3 T系インスタンスのバーストモデル

| 項目 | t3.micro | t3.small | t3.medium |
|------|----------|----------|-----------|
| vCPU | 2 | 2 | 2 |
| メモリ | 1 GiB | 2 GiB | 4 GiB |
| ベースライン CPU | 10% | 20% | 20% |
| CPU クレジット/時 | 12 | 24 | 24 |
| 料金 (東京, Linux) | ~$0.0136/h | ~$0.0272/h | ~$0.0544/h |

```
T3 バーストモデル
CPU使用率
100% |     *****
     |    *     *
 20% |---*-------*----------  ← ベースライン
     |  *         **********
  0% +--+----+----+----------> 時間
     クレジット  クレジット
     消費      蓄積
```

---

## 3. AMI (Amazon Machine Image)

### 3.1 AMI の種類

| 種類 | 提供元 | 例 |
|------|--------|-----|
| AWS 公式 AMI | Amazon | Amazon Linux 2023, Ubuntu, Windows Server |
| マーケットプレイス AMI | サードパーティ | WordPress, NGINX Plus |
| コミュニティ AMI | 一般ユーザー | カスタムビルド |
| カスタム AMI | 自組織 | 社内標準構成 |

### 3.2 コード例: AMI の検索と起動

```bash
# Amazon Linux 2023 の最新 AMI を検索
aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=al2023-ami-2023*-x86_64" \
    "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' \
  --output text

# Ubuntu 22.04 の最新 AMI を検索
aws ec2 describe-images \
  --owners 099720109477 \
  --filters \
    "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
    "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text
```

### 3.3 コード例: カスタム AMI の作成

```bash
# 稼働中のインスタンスから AMI を作成
aws ec2 create-image \
  --instance-id i-0123456789abcdef0 \
  --name "my-app-v1.2.0-$(date +%Y%m%d)" \
  --description "My App v1.2.0 with NGINX + Node.js" \
  --no-reboot

# AMI 一覧（自分のアカウント）
aws ec2 describe-images --owners self \
  --query 'Images[].[ImageId,Name,CreationDate]' \
  --output table
```

---

## 4. セキュリティグループ

### 4.1 セキュリティグループの特性

```
セキュリティグループ = ステートフルファイアウォール

  インバウンドルール               アウトバウンドルール
  (外→内)                        (内→外)
  +-------------------+          +-------------------+
  | 許可ルールのみ     |          | 許可ルールのみ     |
  | デフォルト: 全拒否  |          | デフォルト: 全許可  |
  | ステートフル       |          | ステートフル       |
  +-------------------+          +-------------------+

  ステートフル = インバウンドで許可した通信の戻りは自動許可
```

### 4.2 コード例: セキュリティグループの作成

```bash
# Web サーバー用セキュリティグループを作成
SG_ID=$(aws ec2 create-security-group \
  --group-name web-server-sg \
  --description "Security group for web servers" \
  --vpc-id vpc-0123456789abcdef0 \
  --query 'GroupId' --output text)

# SSH (管理用、特定 IP のみ)
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 22 \
  --cidr 203.0.113.0/32

# HTTP
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 80 \
  --cidr 0.0.0.0/0

# HTTPS
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --protocol tcp --port 443 \
  --cidr 0.0.0.0/0

echo "Created Security Group: $SG_ID"
```

### 4.3 セキュリティグループ設計例

| 層 | SG 名 | インバウンド | ソース |
|----|--------|------------|--------|
| ALB | alb-sg | 80, 443 | 0.0.0.0/0 |
| Web | web-sg | 8080 | alb-sg |
| App | app-sg | 3000 | web-sg |
| DB | db-sg | 3306 | app-sg |

---

## 5. キーペア

### 5.1 コード例: キーペアの作成と SSH 接続

```bash
# キーペアを作成（Ed25519 推奨）
aws ec2 create-key-pair \
  --key-name my-key-pair \
  --key-type ed25519 \
  --query 'KeyMaterial' \
  --output text > my-key-pair.pem

# パーミッション設定
chmod 400 my-key-pair.pem

# SSH 接続
ssh -i my-key-pair.pem ec2-user@<パブリックIP>

# EC2 Instance Connect（キーペア不要で接続）
aws ec2-instance-connect send-ssh-public-key \
  --instance-id i-0123456789abcdef0 \
  --instance-os-user ec2-user \
  --ssh-public-key file://~/.ssh/id_ed25519.pub
```

---

## 6. EBS (Elastic Block Store)

### 6.1 EBS ボリュームタイプ比較

| タイプ | 名称 | IOPS | スループット | 用途 |
|--------|------|------|-------------|------|
| gp3 | 汎用 SSD | 3,000-16,000 | 125-1,000 MB/s | 一般用途（推奨） |
| gp2 | 汎用 SSD | 100-16,000 (容量連動) | 128-250 MB/s | レガシー |
| io2 | プロビジョンド IOPS | 最大 64,000 | 1,000 MB/s | 高性能 DB |
| st1 | スループット最適化 HDD | 500 | 500 MB/s | ビッグデータ |
| sc1 | コールド HDD | 250 | 250 MB/s | アーカイブ |

### 6.2 コード例: EBS ボリュームの作成とアタッチ

```bash
# gp3 ボリュームを作成（100GB, 5000 IOPS）
VOL_ID=$(aws ec2 create-volume \
  --volume-type gp3 \
  --size 100 \
  --iops 5000 \
  --throughput 250 \
  --availability-zone ap-northeast-1a \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=data-vol}]' \
  --query 'VolumeId' --output text)

# インスタンスにアタッチ
aws ec2 attach-volume \
  --volume-id $VOL_ID \
  --instance-id i-0123456789abcdef0 \
  --device /dev/sdf

# スナップショットの作成
aws ec2 create-snapshot \
  --volume-id $VOL_ID \
  --description "Daily backup $(date +%Y-%m-%d)" \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=daily-backup}]'
```

### 6.3 EBS とインスタンスストアの比較

| 特性 | EBS | インスタンスストア |
|------|-----|------------------|
| 永続性 | インスタンス停止後も保持 | インスタンス停止で消失 |
| スナップショット | 可能 | 不可 |
| サイズ変更 | 可能（オンライン） | 不可 |
| レイテンシ | ネットワーク経由 | ローカルディスク（低レイテンシ） |
| 用途 | OS、DB、永続データ | キャッシュ、一時ファイル |

---

## 7. EC2 インスタンスの起動 — 完全な例

```bash
# 全要素を指定してインスタンスを起動
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.small \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --block-device-mappings '[
    {
      "DeviceName": "/dev/xvda",
      "Ebs": {
        "VolumeSize": 30,
        "VolumeType": "gp3",
        "Iops": 3000,
        "Throughput": 125,
        "DeleteOnTermination": true,
        "Encrypted": true
      }
    }
  ]' \
  --iam-instance-profile Name=EC2-S3-ReadOnly-Profile \
  --user-data file://startup.sh \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Name,Value=web-server-01},{Key=Environment,Value=production}]' \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled" \
  --count 1

# User Data スクリプト例 (startup.sh)
#!/bin/bash
yum update -y
yum install -y nginx
systemctl start nginx
systemctl enable nginx
```

---

## 8. アンチパターン

### アンチパターン 1: セキュリティグループで 0.0.0.0/0 に SSH を公開する

```bash
# 悪い例 — 全世界に SSH ポートを公開
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx --protocol tcp --port 22 --cidr 0.0.0.0/0

# 良い例 — 管理元 IP のみ許可 + Session Manager を併用
aws ec2 authorize-security-group-ingress \
  --group-id sg-xxx --protocol tcp --port 22 --cidr 203.0.113.10/32

# さらに良い例 — SSH 不要、Session Manager で接続
aws ssm start-session --target i-0123456789abcdef0
```

### アンチパターン 2: EBS を暗号化せずに使用する

機密データを含むボリュームは必ず暗号化すべきである。デフォルト暗号化を有効にしておけば、忘れを防止できる。

```bash
# アカウントレベルで EBS デフォルト暗号化を有効化
aws ec2 enable-ebs-encryption-by-default --region ap-northeast-1

# 暗号化状態の確認
aws ec2 get-ebs-encryption-by-default --region ap-northeast-1
```

---

## 9. FAQ

### Q1. t3.micro と t3.small のどちらを選ぶべきか？

t3.micro (1 GiB メモリ) は軽量な Web サーバーやテスト用途に適している。メモリ 2 GiB 以上が必要なアプリケーション（WordPress + MySQL など）は t3.small 以上を選択する。無料枠を使う場合は t2.micro (12ヶ月無料) も検討対象。

### Q2. インスタンスを停止するとデータはどうなるか？

EBS ルートボリューム (DeleteOnTermination=true) はインスタンス「終了」で削除されるが、「停止」では保持される。インスタンスストアは停止・終了の両方でデータが失われる。重要なデータは EBS スナップショットや S3 にバックアップする。

### Q3. IMDSv2 とは何か？なぜ必要か？

Instance Metadata Service v2 は、トークンベースの認証をメタデータアクセスに追加するセキュリティ強化。SSRF 攻撃によるメタデータ漏洩を防止する。`HttpTokens=required` で IMDSv2 を強制すべきである。

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| インスタンスタイプ | ワークロードに応じてファミリーとサイズを選定 |
| AMI | Amazon Linux 2023 / Ubuntu が一般的、カスタム AMI で標準化 |
| セキュリティグループ | ステートフル、最小限のポート開放、ソースに SG ID を指定 |
| キーペア | Ed25519 推奨、Session Manager への移行を検討 |
| EBS | gp3 がデフォルト推奨、暗号化を必ず有効化 |
| User Data | 起動時の自動セットアップに活用 |
| IMDSv2 | 必ず有効化し SSRF 対策を実施 |

---

## 次に読むべきガイド

- [01-ec2-advanced.md](./01-ec2-advanced.md) — Auto Scaling、ALB、スポットインスタンス
- [../04-networking/00-vpc-basics.md](../04-networking/00-vpc-basics.md) — VPC の基礎

---

## 参考文献

1. Amazon EC2 ユーザーガイド — https://docs.aws.amazon.com/ec2/latest/userguide/
2. EC2 インスタンスタイプ — https://aws.amazon.com/ec2/instance-types/
3. EBS ボリュームタイプ — https://docs.aws.amazon.com/ebs/latest/userguide/ebs-volume-types.html
4. EC2 セキュリティベストプラクティス — https://docs.aws.amazon.com/ec2/latest/userguide/ec2-security.html
