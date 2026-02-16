# EC2 基礎

> AWS の仮想サーバー EC2 の基本概念 — インスタンスタイプ、AMI、セキュリティグループ、キーペア、EBS を体系的に理解する

## この章で学ぶこと

1. EC2 インスタンスのライフサイクルとインスタンスタイプの選定基準を理解できる
2. AMI、セキュリティグループ、キーペアを適切に設定し、安全にインスタンスを起動できる
3. EBS ボリュームの種類と特性を理解し、ワークロードに最適なストレージを選択できる
4. User Data とメタデータサービスを活用した自動化とセキュリティ強化を実装できる
5. CloudFormation / CDK を使って EC2 環境を Infrastructure as Code で管理できる

---

## 1. EC2 とは

Amazon Elastic Compute Cloud (EC2) は、AWS 上で仮想サーバー (インスタンス) をオンデマンドで起動・管理できるサービスである。物理サーバーの調達・設置・保守を不要にし、分単位の課金で柔軟にコンピューティングリソースを利用できる。

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
|                                                   |
|  +---------------------------------------------+ |
|  |       IAM Instance Profile (権限)             | |
|  +---------------------------------------------+ |
+--------------------------------------------------+
```

### 1.2 EC2 の料金体系

EC2 の料金は主に以下の要素で構成される。

| 料金要素 | 説明 | 課金単位 |
|---------|------|---------|
| インスタンス料金 | vCPU・メモリに基づく時間課金 | 秒単位（最低60秒） |
| EBS ボリューム | ストレージ容量と IOPS | GB/月 + IOPS/月 |
| データ転送 | リージョン外へのアウトバウンド | GB あたり |
| Elastic IP | 未使用の EIP に対する課金 | 時間あたり |
| EBS スナップショット | S3 に保存されるバックアップ | GB/月 |

```bash
# EC2 の料金見積もりに役立つコマンド
# インスタンスタイプの料金情報を取得（AWS Pricing API）
aws pricing get-products \
  --service-code AmazonEC2 \
  --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=t3.small" \
    "Type=TERM_MATCH,Field=location,Value=Asia Pacific (Tokyo)" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
    "Type=TERM_MATCH,Field=tenancy,Value=Shared" \
    "Type=TERM_MATCH,Field=preInstalledSw,Value=NA" \
  --region us-east-1 \
  --query 'PriceList[0]' \
  --output json | jq '.terms.OnDemand | to_entries[0].value.priceDimensions | to_entries[0].value.pricePerUnit.USD'
```

### 1.3 インスタンスのライフサイクル

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
  休止: running → stopping → stopped (メモリ内容を EBS に保存)
```

各状態での課金:

| 状態 | インスタンス課金 | EBS 課金 | Elastic IP 課金 |
|------|---------------|---------|---------------|
| running | あり | あり | なし（アタッチ時） |
| stopped | なし | あり | あり（未アタッチ時） |
| terminated | なし | なし（削除済み） | あり（未アタッチ時） |

### 1.4 EC2 のネットワーク構成

```
EC2 のネットワーク配置
+--------------------------------------------------+
|  VPC (10.0.0.0/16)                                |
|                                                   |
|  +--------------------+  +--------------------+  |
|  | Public Subnet      |  | Private Subnet     |  |
|  | (10.0.1.0/24)      |  | (10.0.2.0/24)      |  |
|  |  +-------------+   |  |  +-------------+   |  |
|  |  | Web Server  |   |  |  | App Server  |   |  |
|  |  | (EC2)       |   |  |  | (EC2)       |   |  |
|  |  | Public IP   |   |  |  | Private IP  |   |  |
|  |  +------+------+   |  |  +------+------+   |  |
|  |         |           |  |         |           |  |
|  +----+----+-----+-----+  +----+----+-----+-----+  |
|       |          |              |          |        |
|  +----v----+ +---v----+   +----v----+ +---v----+  |
|  | IGW     | | NAT GW |   | NAT GW | |  VPC   |  |
|  |(Internet)| |        |   |(経由)   | |Endpoint|  |
|  +---------+ +--------+   +--------+ +--------+  |
+--------------------------------------------------+
```

---

## 2. インスタンスタイプ

### 2.1 命名規則

```
  m  5  a  .  xlarge
  |  |  |     |
  |  |  |     +-- サイズ (nano, micro, small, medium, large, xlarge, 2xlarge...)
  |  |  +-------- 追加属性 (a: AMD, g: Graviton, n: ネットワーク強化, d: ローカルストレージ)
  |  +----------- 世代番号
  +-------------- ファミリー (汎用, コンピュート最適化, メモリ最適化...)

  追加属性の例:
  m5a.xlarge   → a: AMD プロセッサ（コスト効率が良い）
  m7g.xlarge   → g: Graviton (ARM) プロセッサ（高性能・低コスト）
  m5n.xlarge   → n: ネットワーク強化（最大 100Gbps）
  m5d.xlarge   → d: ローカル NVMe SSD 付き
  m5ad.xlarge  → a+d: AMD + ローカル NVMe SSD
  c7gn.xlarge  → g+n: Graviton + ネットワーク強化
  r6idn.xlarge → i+d+n: Intel + ローカル NVMe + ネットワーク強化
```

### 2.2 インスタンスファミリー比較

| ファミリー | プレフィックス | 特徴 | ユースケース |
|-----------|-------------|------|-------------|
| 汎用 | t3, m5, m6i, m7g | CPU/メモリバランス | Web サーバー、小中規模 DB |
| コンピュート最適化 | c5, c6i, c7g | 高 CPU 性能 | バッチ処理、機械学習推論 |
| メモリ最適化 | r5, r6i, x2idn | 大容量メモリ | インメモリ DB、ビッグデータ |
| ストレージ最適化 | i3, d3, h1 | 高 I/O | データウェアハウス、ログ処理 |
| 高速コンピューティング | p4, g5, inf2 | GPU / 推論チップ | 機械学習訓練、動画処理 |
| HPC 最適化 | hpc6a, hpc7g | 高帯域ネットワーク | 科学計算、シミュレーション |

### 2.3 Graviton プロセッサの選定

AWS Graviton は ARM ベースのカスタムプロセッサで、同等の Intel/AMD インスタンスと比較して最大 40% のコストパフォーマンス改善を実現する。

```
Graviton 世代比較
+-------------------+-------------------+-------------------+
| Graviton2         | Graviton3         | Graviton4         |
| (2020〜)          | (2022〜)          | (2024〜)          |
+-------------------+-------------------+-------------------+
| m6g, c6g, r6g     | m7g, c7g, r7g     | m8g, c8g, r8g     |
| t4g               | c7gn (ネットワーク)| (最新世代)          |
| 前世代比 40%↑     | Graviton2比 25%↑  | Graviton3比 30%↑  |
+-------------------+-------------------+-------------------+

対応ソフトウェア確認ポイント:
- Docker コンテナ: linux/arm64 イメージが必要
- Node.js / Python / Java: ほぼそのまま動作
- C/C++ ネイティブ: ARM 向けコンパイルが必要
- .NET: .NET 6+ で ARM ネイティブ対応
```

```bash
# Graviton インスタンスの料金比較
# Intel vs Graviton の料金差を確認
echo "=== t3.medium (Intel x86_64) ==="
aws pricing get-products \
  --service-code AmazonEC2 \
  --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=t3.medium" \
    "Type=TERM_MATCH,Field=location,Value=Asia Pacific (Tokyo)" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
  --region us-east-1 --output json 2>/dev/null | jq -r '.PriceList[0]' | jq -r '.terms.OnDemand | to_entries[0].value.priceDimensions | to_entries[0].value.pricePerUnit.USD'

echo "=== t4g.medium (Graviton2 ARM64) ==="
aws pricing get-products \
  --service-code AmazonEC2 \
  --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=t4g.medium" \
    "Type=TERM_MATCH,Field=location,Value=Asia Pacific (Tokyo)" \
    "Type=TERM_MATCH,Field=operatingSystem,Value=Linux" \
  --region us-east-1 --output json 2>/dev/null | jq -r '.PriceList[0]' | jq -r '.terms.OnDemand | to_entries[0].value.priceDimensions | to_entries[0].value.pricePerUnit.USD'
```

### 2.4 T系インスタンスのバーストモデル

| 項目 | t3.nano | t3.micro | t3.small | t3.medium | t3.large |
|------|---------|----------|----------|-----------|----------|
| vCPU | 2 | 2 | 2 | 2 | 2 |
| メモリ | 0.5 GiB | 1 GiB | 2 GiB | 4 GiB | 8 GiB |
| ベースライン CPU | 5% | 10% | 20% | 20% | 30% |
| CPU クレジット/時 | 6 | 12 | 24 | 24 | 36 |
| 最大クレジット残高 | 144 | 288 | 576 | 576 | 864 |
| 料金 (東京, Linux) | ~$0.0068/h | ~$0.0136/h | ~$0.0272/h | ~$0.0544/h | ~$0.1088/h |

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

T3 Unlimited モード:
- ベースライン超過分を追加料金で継続使用可能
- vCPU あたり $0.05/時（Linux）の追加課金
- バッチ処理など一時的な高負荷に有用
- 予期せぬ高額課金に注意が必要
```

```bash
# T3 バーストモードの確認と設定
# 現在のクレジットバランスを確認
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUCreditBalance \
  --dimensions Name=InstanceId,Value=i-0123456789abcdef0 \
  --start-time "$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 300 \
  --statistics Average

# Unlimited モードの有効化
aws ec2 modify-instance-credit-specification \
  --instance-credit-specifications '[{
    "InstanceId": "i-0123456789abcdef0",
    "CpuCredits": "unlimited"
  }]'

# Unlimited モードの無効化（standard に戻す）
aws ec2 modify-instance-credit-specification \
  --instance-credit-specifications '[{
    "InstanceId": "i-0123456789abcdef0",
    "CpuCredits": "standard"
  }]'
```

### 2.5 インスタンスタイプの選定フローチャート

```
インスタンスタイプ選定フロー
============================

用途は？
├─ Web サーバー / API サーバー
│   ├─ 負荷が断続的 → t3 / t4g
│   ├─ 負荷が安定的 → m6i / m7g
│   └─ CPU集約型 → c6i / c7g
│
├─ データベース
│   ├─ 小〜中規模 → r6i / r7g (メモリ最適化)
│   ├─ インメモリDB → x2idn (大容量メモリ)
│   └─ 高IOPS → i3 (ローカルNVMe)
│
├─ 機械学習
│   ├─ 訓練 → p4d / p5 (GPU)
│   ├─ 推論 → inf2 (Inferentia)
│   └─ データ前処理 → c6i / c7g
│
├─ バッチ処理
│   ├─ CPU集約 → c6i / c7g
│   ├─ メモリ集約 → r6i / r7g
│   └─ I/O集約 → i3 / d3
│
└─ 開発・テスト
    ├─ 最低コスト → t3.micro / t4g.micro
    └─ 無料枠 → t2.micro (12ヶ月無料)
```

---

## 3. AMI (Amazon Machine Image)

### 3.1 AMI の種類

| 種類 | 提供元 | 例 | 特徴 |
|------|--------|-----|------|
| AWS 公式 AMI | Amazon | Amazon Linux 2023, Ubuntu, Windows Server | 定期的にパッチ適用、無料 |
| マーケットプレイス AMI | サードパーティ | WordPress, NGINX Plus, Databricks | ライセンス料が含まれる場合あり |
| コミュニティ AMI | 一般ユーザー | カスタムビルド | セキュリティリスクに注意 |
| カスタム AMI | 自組織 | 社内標準構成 | ゴールデンイメージとして運用 |

### 3.2 AMI のアーキテクチャ

```
AMI の構造
+-------------------------------------------+
|  AMI (ami-0abcdef1234567890)              |
|                                            |
|  +--------------------------------------+ |
|  | ルート EBS スナップショット             | |
|  | - OS (Amazon Linux 2023)              | |
|  | - インストール済みソフトウェア          | |
|  | - 設定ファイル                         | |
|  +--------------------------------------+ |
|                                            |
|  +--------------------------------------+ |
|  | 追加 EBS スナップショット (オプション)   | |
|  | - データボリューム                     | |
|  +--------------------------------------+ |
|                                            |
|  +--------------------------------------+ |
|  | メタデータ                            | |
|  | - アーキテクチャ (x86_64 / arm64)     | |
|  | - 仮想化タイプ (hvm)                  | |
|  | - ブートモード (uefi / legacy-bios)   | |
|  | - ブロックデバイスマッピング           | |
|  +--------------------------------------+ |
+-------------------------------------------+
```

### 3.3 コード例: AMI の検索と起動

```bash
# Amazon Linux 2023 の最新 AMI を検索
aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=al2023-ami-2023*-x86_64" \
    "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].[ImageId,Name]' \
  --output text

# Amazon Linux 2023 ARM64 (Graviton) の最新 AMI
aws ec2 describe-images \
  --owners amazon \
  --filters \
    "Name=name,Values=al2023-ami-2023*-arm64" \
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

# Ubuntu 24.04 の最新 AMI を検索
aws ec2 describe-images \
  --owners 099720109477 \
  --filters \
    "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" \
    "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text

# SSM パラメータストアから最新 AMI を取得（推奨）
aws ssm get-parameter \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64 \
  --query 'Parameter.Value' --output text
```

### 3.4 コード例: カスタム AMI の作成と管理

```bash
# 稼働中のインスタンスから AMI を作成
aws ec2 create-image \
  --instance-id i-0123456789abcdef0 \
  --name "my-app-v1.2.0-$(date +%Y%m%d)" \
  --description "My App v1.2.0 with NGINX + Node.js" \
  --no-reboot \
  --tag-specifications '[
    {
      "ResourceType": "image",
      "Tags": [
        {"Key": "Name", "Value": "my-app-v1.2.0"},
        {"Key": "Version", "Value": "1.2.0"},
        {"Key": "CreatedBy", "Value": "automation"}
      ]
    },
    {
      "ResourceType": "snapshot",
      "Tags": [
        {"Key": "Name", "Value": "my-app-v1.2.0-snapshot"}
      ]
    }
  ]'

# AMI 一覧（自分のアカウント）
aws ec2 describe-images --owners self \
  --query 'Images[].[ImageId,Name,CreationDate,State]' \
  --output table

# AMI のリージョン間コピー
aws ec2 copy-image \
  --source-region ap-northeast-1 \
  --source-image-id ami-0abcdef1234567890 \
  --name "my-app-v1.2.0-us-east-1" \
  --description "Cross-region copy of my-app-v1.2.0" \
  --region us-east-1

# 古い AMI の登録解除とスナップショット削除
AMI_ID="ami-0abcdef1234567890"
# スナップショット ID を取得
SNAP_IDS=$(aws ec2 describe-images --image-ids $AMI_ID \
  --query 'Images[0].BlockDeviceMappings[*].Ebs.SnapshotId' --output text)
# AMI 登録解除
aws ec2 deregister-image --image-id $AMI_ID
# スナップショット削除
for SNAP_ID in $SNAP_IDS; do
  aws ec2 delete-snapshot --snapshot-id $SNAP_ID
done
```

### 3.5 ゴールデン AMI パイプライン

```
ゴールデン AMI の自動ビルドフロー
===================================

  +-------------------+
  | EC2 Image Builder |
  | パイプライン       |
  +--------+----------+
           |
    +------v------+
    | ベース AMI  |  (Amazon Linux 2023)
    +------+------+
           |
    +------v------+
    | ビルド       |  - パッケージインストール
    | コンポーネント|  - セキュリティ設定
    |              |  - アプリケーション配置
    +------+------+
           |
    +------v------+
    | テスト       |  - CIS ベンチマーク
    | コンポーネント|  - Inspector スキャン
    |              |  - 動作確認
    +------+------+
           |
    +------v------+
    | ゴールデン AMI|  → 各リージョンに配布
    | 配布         |  → Auto Scaling で利用
    +-------------+
```

```bash
# EC2 Image Builder のパイプラインを作成する例
# まずコンポーネントを定義
aws imagebuilder create-component \
  --name "install-web-server" \
  --semantic-version "1.0.0" \
  --platform Linux \
  --data '
name: InstallWebServer
schemaVersion: 1.0
phases:
  - name: build
    steps:
      - name: InstallNginx
        action: ExecuteBash
        inputs:
          commands:
            - dnf install -y nginx
            - systemctl enable nginx
      - name: InstallNodeJS
        action: ExecuteBash
        inputs:
          commands:
            - dnf install -y nodejs npm
  - name: validate
    steps:
      - name: ValidateNginx
        action: ExecuteBash
        inputs:
          commands:
            - nginx -t
            - node --version
'
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

  セキュリティグループ vs ネットワーク ACL
  +---------------------------+---------------------------+
  | セキュリティグループ        | ネットワーク ACL            |
  +---------------------------+---------------------------+
  | インスタンスレベル          | サブネットレベル            |
  | ステートフル               | ステートレス               |
  | 許可ルールのみ             | 許可 + 拒否ルール           |
  | 全ルールを評価             | 番号順に評価（最初の一致）   |
  | ENI に関連付け             | サブネットに関連付け        |
  +---------------------------+---------------------------+
```

### 4.2 コード例: セキュリティグループの作成

```bash
# Web サーバー用セキュリティグループを作成
SG_ID=$(aws ec2 create-security-group \
  --group-name web-server-sg \
  --description "Security group for web servers" \
  --vpc-id vpc-0123456789abcdef0 \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=Name,Value=web-server-sg}]' \
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

# IPv6 対応
aws ec2 authorize-security-group-ingress \
  --group-id $SG_ID \
  --ip-permissions '[
    {"IpProtocol": "tcp", "FromPort": 80, "ToPort": 80, "Ipv6Ranges": [{"CidrIpv6": "::/0"}]},
    {"IpProtocol": "tcp", "FromPort": 443, "ToPort": 443, "Ipv6Ranges": [{"CidrIpv6": "::/0"}]}
  ]'

echo "Created Security Group: $SG_ID"

# セキュリティグループのルール一覧
aws ec2 describe-security-group-rules \
  --filter Name=group-id,Values=$SG_ID \
  --query 'SecurityGroupRules[].[IsEgress,IpProtocol,FromPort,ToPort,CidrIpv4,ReferencedGroupInfo.GroupId]' \
  --output table
```

### 4.3 セキュリティグループ設計例（多層アーキテクチャ）

| 層 | SG 名 | インバウンド | ソース | 説明 |
|----|--------|------------|--------|------|
| ALB | alb-sg | 80, 443 | 0.0.0.0/0 | インターネットからの HTTP/HTTPS |
| Web | web-sg | 8080 | alb-sg | ALB からのみアクセス可能 |
| App | app-sg | 3000 | web-sg | Web 層からのみアクセス可能 |
| DB | db-sg | 3306 | app-sg | App 層からのみアクセス可能 |
| Cache | cache-sg | 6379 | app-sg | App 層からのみアクセス可能 |
| Bastion | bastion-sg | 22 | 社内IP/32 | 管理用踏み台 |

```bash
# 多層セキュリティグループを一括作成するスクリプト
#!/bin/bash
VPC_ID="vpc-0123456789abcdef0"
OFFICE_IP="203.0.113.0/32"

# ALB 用 SG
ALB_SG=$(aws ec2 create-security-group \
  --group-name alb-sg --description "ALB SG" --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $ALB_SG \
  --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $ALB_SG \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Web 用 SG（ALB からのみ）
WEB_SG=$(aws ec2 create-security-group \
  --group-name web-sg --description "Web SG" --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $WEB_SG \
  --protocol tcp --port 8080 --source-group $ALB_SG

# App 用 SG（Web からのみ）
APP_SG=$(aws ec2 create-security-group \
  --group-name app-sg --description "App SG" --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $APP_SG \
  --protocol tcp --port 3000 --source-group $WEB_SG

# DB 用 SG（App からのみ）
DB_SG=$(aws ec2 create-security-group \
  --group-name db-sg --description "DB SG" --vpc-id $VPC_ID \
  --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG \
  --protocol tcp --port 3306 --source-group $APP_SG

echo "ALB: $ALB_SG | Web: $WEB_SG | App: $APP_SG | DB: $DB_SG"
```

### 4.4 セキュリティグループのベストプラクティス

1. **最小権限の原則**: 必要なポートとソースのみ許可する
2. **SG ID をソースに指定**: CIDR ブロックではなく SG ID を参照することで、動的な IP 変更に対応
3. **説明 (Description) を必ず記載**: 各ルールの目的を明記する
4. **定期的な棚卸し**: 不要なルールを削除する
5. **デフォルト SG を使わない**: 専用の SG を作成して明示的に設定する

---

## 5. キーペアと接続方法

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

### 5.2 Session Manager による接続（推奨）

Session Manager を使えば、SSH ポートを開放せずにインスタンスに接続できる。

```bash
# Session Manager で接続（SSH ポート不要）
aws ssm start-session --target i-0123456789abcdef0

# ポートフォワーディング（ローカルの 8080 → リモートの 80）
aws ssm start-session \
  --target i-0123456789abcdef0 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["80"],"localPortNumber":["8080"]}'

# RDS へのポートフォワーディング（踏み台不要）
aws ssm start-session \
  --target i-0123456789abcdef0 \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{
    "host": ["my-db.xxxx.ap-northeast-1.rds.amazonaws.com"],
    "portNumber": ["3306"],
    "localPortNumber": ["3306"]
  }'
```

Session Manager を使うための IAM ポリシー:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ssm:StartSession",
        "ssm:TerminateSession",
        "ssm:ResumeSession"
      ],
      "Resource": [
        "arn:aws:ec2:ap-northeast-1:123456789012:instance/*",
        "arn:aws:ssm:ap-northeast-1:123456789012:document/AWS-StartPortForwardingSession",
        "arn:aws:ssm:ap-northeast-1:123456789012:document/AWS-StartPortForwardingSessionToRemoteHost"
      ],
      "Condition": {
        "StringLike": {
          "ssm:resourceTag/Environment": ["production", "staging"]
        }
      }
    }
  ]
}
```

### 5.3 EC2 インスタンスの IAM ロール設定

```bash
# EC2 用の IAM ロールを作成
aws iam create-role \
  --role-name EC2-WebServer-Role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {"Service": "ec2.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }
    ]
  }'

# SSM 用ポリシーをアタッチ（Session Manager に必要）
aws iam attach-role-policy \
  --role-name EC2-WebServer-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

# S3 読み取り用ポリシーをアタッチ
aws iam attach-role-policy \
  --role-name EC2-WebServer-Role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# インスタンスプロファイルを作成してロールを紐付け
aws iam create-instance-profile \
  --instance-profile-name EC2-WebServer-Profile

aws iam add-role-to-instance-profile \
  --instance-profile-name EC2-WebServer-Profile \
  --role-name EC2-WebServer-Role

# 既存のインスタンスにプロファイルをアタッチ
aws ec2 associate-iam-instance-profile \
  --instance-id i-0123456789abcdef0 \
  --iam-instance-profile Name=EC2-WebServer-Profile
```

---

## 6. EBS (Elastic Block Store)

### 6.1 EBS ボリュームタイプ比較

| タイプ | 名称 | IOPS | スループット | 最大容量 | 用途 |
|--------|------|------|-------------|---------|------|
| gp3 | 汎用 SSD | 3,000-16,000 | 125-1,000 MB/s | 16 TiB | 一般用途（推奨） |
| gp2 | 汎用 SSD | 100-16,000 (容量連動) | 128-250 MB/s | 16 TiB | レガシー |
| io2 | プロビジョンド IOPS | 最大 64,000 | 1,000 MB/s | 16 TiB | 高性能 DB |
| io2 Block Express | 超高性能 SSD | 最大 256,000 | 4,000 MB/s | 64 TiB | SAP HANA 等 |
| st1 | スループット最適化 HDD | 500 | 500 MB/s | 16 TiB | ビッグデータ |
| sc1 | コールド HDD | 250 | 250 MB/s | 16 TiB | アーカイブ |

### 6.2 gp2 から gp3 への移行

gp3 は gp2 と比較して同じ性能で最大 20% のコスト削減が可能。IOPS とスループットを個別に設定できる利点もある。

```bash
# gp2 ボリュームを gp3 に変更
aws ec2 modify-volume \
  --volume-id vol-0123456789abcdef0 \
  --volume-type gp3 \
  --iops 3000 \
  --throughput 125

# 変更状態の確認
aws ec2 describe-volumes-modifications \
  --volume-ids vol-0123456789abcdef0 \
  --query 'VolumesModifications[0].[ModificationState,TargetVolumeType,TargetIops,TargetThroughput,Progress]' \
  --output table

# 全 gp2 ボリュームを一覧表示（移行候補の特定）
aws ec2 describe-volumes \
  --filters "Name=volume-type,Values=gp2" \
  --query 'Volumes[].[VolumeId,Size,Iops,State,Attachments[0].InstanceId]' \
  --output table
```

### 6.3 コード例: EBS ボリュームの作成とアタッチ

```bash
# gp3 ボリュームを作成（100GB, 5000 IOPS）
VOL_ID=$(aws ec2 create-volume \
  --volume-type gp3 \
  --size 100 \
  --iops 5000 \
  --throughput 250 \
  --availability-zone ap-northeast-1a \
  --encrypted \
  --kms-key-id alias/ebs-key \
  --tag-specifications 'ResourceType=volume,Tags=[{Key=Name,Value=data-vol}]' \
  --query 'VolumeId' --output text)

# インスタンスにアタッチ
aws ec2 attach-volume \
  --volume-id $VOL_ID \
  --instance-id i-0123456789abcdef0 \
  --device /dev/sdf

# Linux でのボリュームフォーマットとマウント
# (User Data またはSSH接続後に実行)
# デバイスの確認
lsblk
# ファイルシステム作成
sudo mkfs -t xfs /dev/nvme1n1
# マウントポイント作成
sudo mkdir /data
# マウント
sudo mount /dev/nvme1n1 /data
# 永続マウント設定
UUID=$(sudo blkid -o value -s UUID /dev/nvme1n1)
echo "UUID=$UUID /data xfs defaults,nofail 0 2" | sudo tee -a /etc/fstab

# スナップショットの作成
aws ec2 create-snapshot \
  --volume-id $VOL_ID \
  --description "Daily backup $(date +%Y-%m-%d)" \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=daily-backup},{Key=RetainDays,Value=7}]'
```

### 6.4 EBS スナップショットのライフサイクル管理

```bash
# Data Lifecycle Manager (DLM) でスナップショットを自動管理
aws dlm create-lifecycle-policy \
  --description "Daily EBS snapshots with 7-day retention" \
  --state ENABLED \
  --execution-role-arn arn:aws:iam::123456789012:role/AWSDataLifecycleManagerDefaultRole \
  --policy-details '{
    "PolicyType": "EBS_SNAPSHOT_MANAGEMENT",
    "ResourceTypes": ["VOLUME"],
    "TargetTags": [{"Key": "Backup", "Value": "true"}],
    "Schedules": [{
      "Name": "DailySnapshots",
      "CreateRule": {
        "Interval": 24,
        "IntervalUnit": "HOURS",
        "Times": ["03:00"]
      },
      "RetainRule": {
        "Count": 7
      },
      "CopyTags": true,
      "TagsToAdd": [{"Key": "CreatedBy", "Value": "DLM"}]
    }]
  }'
```

### 6.5 EBS とインスタンスストアの比較

| 特性 | EBS | インスタンスストア |
|------|-----|------------------|
| 永続性 | インスタンス停止後も保持 | インスタンス停止で消失 |
| スナップショット | 可能 | 不可 |
| サイズ変更 | 可能（オンライン） | 不可 |
| レイテンシ | ネットワーク経由 | ローカルディスク（低レイテンシ） |
| IOPS | 最大 256,000 (io2 BE) | 最大数百万 (NVMe) |
| 用途 | OS、DB、永続データ | キャッシュ、一時ファイル、バッファ |
| 暗号化 | KMS / デフォルト暗号化 | ハードウェアレベル |

---

## 7. EC2 インスタンスの起動 — 完全な例

### 7.1 AWS CLI での起動

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
  --iam-instance-profile Name=EC2-WebServer-Profile \
  --user-data file://startup.sh \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=Name,Value=web-server-01},{Key=Environment,Value=production}]' \
    'ResourceType=volume,Tags=[{Key=Name,Value=web-server-01-root}]' \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1" \
  --count 1 \
  --monitoring Enabled=true
```

### 7.2 User Data スクリプト例

```bash
#!/bin/bash
# startup.sh - EC2 起動時に自動実行されるスクリプト
set -euxo pipefail

# ログ出力先
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

# システム更新
dnf update -y

# 必要なパッケージのインストール
dnf install -y nginx nodejs npm git

# NGINX 設定
cat > /etc/nginx/conf.d/app.conf << 'NGINX_EOF'
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    location /health {
        access_log off;
        return 200 'healthy';
        add_header Content-Type text/plain;
    }
}
NGINX_EOF

# NGINX 起動
systemctl start nginx
systemctl enable nginx

# CloudWatch Agent のインストール
dnf install -y amazon-cloudwatch-agent
cat > /opt/aws/amazon-cloudwatch-agent/etc/config.json << 'CW_EOF'
{
  "metrics": {
    "metrics_collected": {
      "mem": {"measurement": ["mem_used_percent"]},
      "disk": {"measurement": ["disk_used_percent"], "resources": ["/"]}
    }
  },
  "logs": {
    "logs_collected": {
      "files": {
        "collect_list": [
          {
            "file_path": "/var/log/nginx/access.log",
            "log_group_name": "/ec2/nginx/access",
            "log_stream_name": "{instance_id}"
          },
          {
            "file_path": "/var/log/nginx/error.log",
            "log_group_name": "/ec2/nginx/error",
            "log_stream_name": "{instance_id}"
          }
        ]
      }
    }
  }
}
CW_EOF
/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config -m ec2 -s -c file:/opt/aws/amazon-cloudwatch-agent/etc/config.json

echo "User data script completed successfully"
```

### 7.3 CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: EC2 Web Server with best practices

Parameters:
  EnvironmentName:
    Type: String
    Default: production
    AllowedValues: [production, staging, development]
  InstanceType:
    Type: String
    Default: t3.small
    AllowedValues: [t3.micro, t3.small, t3.medium, t3.large, m6i.large]
  VpcId:
    Type: AWS::EC2::VPC::Id
  SubnetId:
    Type: AWS::EC2::Subnet::Id
  LatestAmiId:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>
    Default: /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64

Resources:
  # セキュリティグループ
  WebServerSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Security group for web server
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-web-sg

  # IAM ロール
  WebServerRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
      Tags:
        - Key: Environment
          Value: !Ref EnvironmentName

  # インスタンスプロファイル
  WebServerInstanceProfile:
    Type: AWS::IAM::InstanceProfile
    Properties:
      Roles:
        - !Ref WebServerRole

  # EC2 インスタンス
  WebServerInstance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: !Ref LatestAmiId
      InstanceType: !Ref InstanceType
      IamInstanceProfile: !Ref WebServerInstanceProfile
      SubnetId: !Ref SubnetId
      SecurityGroupIds:
        - !Ref WebServerSecurityGroup
      BlockDeviceMappings:
        - DeviceName: /dev/xvda
          Ebs:
            VolumeSize: 30
            VolumeType: gp3
            Iops: 3000
            Throughput: 125
            Encrypted: true
            DeleteOnTermination: true
      MetadataOptions:
        HttpTokens: required
        HttpEndpoint: enabled
        HttpPutResponseHopLimit: 1
      Monitoring: true
      UserData:
        Fn::Base64: |
          #!/bin/bash
          dnf update -y
          dnf install -y nginx
          systemctl start nginx
          systemctl enable nginx
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-web-server
        - Key: Environment
          Value: !Ref EnvironmentName

Outputs:
  InstanceId:
    Value: !Ref WebServerInstance
  PrivateIp:
    Value: !GetAtt WebServerInstance.PrivateIp
  SecurityGroupId:
    Value: !Ref WebServerSecurityGroup
```

### 7.4 CDK (TypeScript) での EC2 定義

```typescript
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export class Ec2Stack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // VPC（既存の VPC を参照する場合）
    const vpc = ec2.Vpc.fromLookup(this, 'Vpc', {
      vpcId: 'vpc-0123456789abcdef0',
    });

    // セキュリティグループ
    const webSg = new ec2.SecurityGroup(this, 'WebSG', {
      vpc,
      description: 'Security group for web servers',
      allowAllOutbound: true,
    });
    webSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'Allow HTTP');
    webSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'Allow HTTPS');

    // IAM ロール
    const role = new iam.Role(this, 'WebServerRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy'),
      ],
    });

    // EC2 インスタンス
    const instance = new ec2.Instance(this, 'WebServer', {
      vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      instanceType: ec2.InstanceType.of(ec2.InstanceClass.T3, ec2.InstanceSize.SMALL),
      machineImage: ec2.MachineImage.latestAmazonLinux2023({
        cpuType: ec2.AmazonLinuxCpuType.X86_64,
      }),
      securityGroup: webSg,
      role,
      blockDevices: [{
        deviceName: '/dev/xvda',
        volume: ec2.BlockDeviceVolume.ebs(30, {
          volumeType: ec2.EbsDeviceVolumeType.GP3,
          iops: 3000,
          throughput: 125,
          encrypted: true,
          deleteOnTermination: true,
        }),
      }],
      requireImdsv2: true,
      detailedMonitoring: true,
    });

    // User Data
    instance.addUserData(
      'dnf update -y',
      'dnf install -y nginx',
      'systemctl start nginx',
      'systemctl enable nginx',
    );

    // 出力
    new cdk.CfnOutput(this, 'InstanceId', { value: instance.instanceId });
    new cdk.CfnOutput(this, 'PrivateIp', { value: instance.instancePrivateIp });
  }
}
```

---

## 8. メタデータサービス (IMDS)

### 8.1 IMDSv2 の使い方

```bash
# IMDSv2 でトークンを取得してメタデータにアクセス
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

# インスタンス ID
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id

# インスタンスタイプ
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-type

# パブリック IP
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4

# IAM ロールの一時的な認証情報
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/iam/security-credentials/

# アベイラビリティゾーン
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/availability-zone

# インスタンスのタグ（設定が必要）
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/tags/instance/Name
```

### 8.2 IMDSv2 の強制設定

```bash
# 新規インスタンスで IMDSv2 を強制
aws ec2 run-instances \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1" \
  ...

# 既存インスタンスで IMDSv2 を強制
aws ec2 modify-instance-metadata-options \
  --instance-id i-0123456789abcdef0 \
  --http-tokens required \
  --http-endpoint enabled \
  --http-put-response-hop-limit 1

# アカウントレベルで IMDSv2 をデフォルトに設定
aws ec2 modify-instance-metadata-defaults \
  --http-tokens required \
  --http-put-response-hop-limit 1 \
  --http-endpoint enabled \
  --region ap-northeast-1

# IMDSv1 を使用しているインスタンスの検出
aws ec2 describe-instances \
  --query 'Reservations[].Instances[?MetadataOptions.HttpTokens==`optional`].[InstanceId,Tags[?Key==`Name`].Value|[0]]' \
  --output table
```

---

## 9. EC2 の監視

### 9.1 CloudWatch メトリクス

```bash
# CPU 使用率の取得
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-0123456789abcdef0 \
  --start-time "$(date -u -v-1H +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 300 \
  --statistics Average Maximum

# CPU アラームの作成
aws cloudwatch put-metric-alarm \
  --alarm-name "ec2-high-cpu" \
  --alarm-description "CPU usage exceeds 80% for 5 minutes" \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-0123456789abcdef0 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --alarm-actions arn:aws:sns:ap-northeast-1:123456789012:alerts

# ステータスチェックアラーム（自動リカバリ）
aws cloudwatch put-metric-alarm \
  --alarm-name "ec2-auto-recovery" \
  --alarm-description "Auto-recover when status check fails" \
  --namespace AWS/EC2 \
  --metric-name StatusCheckFailed_System \
  --dimensions Name=InstanceId,Value=i-0123456789abcdef0 \
  --statistic Maximum \
  --period 60 \
  --threshold 1 \
  --comparison-operator GreaterThanOrEqualToThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:automate:ap-northeast-1:ec2:recover
```

### 9.2 主要メトリクス一覧

| メトリクス | 説明 | 通常値 | アラート閾値 |
|-----------|------|--------|------------|
| CPUUtilization | CPU 使用率 (%) | 10-60% | > 80% |
| NetworkIn/Out | ネットワーク I/O (bytes) | ワークロード依存 | 急激な増加 |
| DiskReadOps/WriteOps | ディスク I/O 操作数 | ワークロード依存 | キュー長増加 |
| StatusCheckFailed | システム/インスタンスチェック | 0 | > 0 |
| CPUCreditBalance | T系のクレジット残高 | > 100 | < 20 |
| mem_used_percent | メモリ使用率（CW Agent） | 30-70% | > 85% |
| disk_used_percent | ディスク使用率（CW Agent） | < 60% | > 80% |

---

## 10. アンチパターン

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

# デフォルト KMS キーの変更
aws ec2 modify-ebs-default-kms-key-id \
  --kms-key-id alias/ebs-custom-key \
  --region ap-northeast-1
```

### アンチパターン 3: IAM ロールを使わずにアクセスキーを埋め込む

```bash
# 悪い例 — EC2 内にアクセスキーをハードコード
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
aws s3 ls  # キー漏洩のリスク

# 良い例 — IAM ロール（インスタンスプロファイル）を使用
# EC2 に IAM ロールをアタッチし、一時的な認証情報を自動取得
aws s3 ls  # インスタンスプロファイルの認証情報を自動使用
```

### アンチパターン 4: User Data にシークレットを直接記述する

```bash
# 悪い例 — User Data にパスワードをハードコード
#!/bin/bash
export DB_PASSWORD="MySecretPassword123!"

# 良い例 — Secrets Manager から取得
#!/bin/bash
DB_PASSWORD=$(aws secretsmanager get-secret-value \
  --secret-id my-db-password \
  --query 'SecretString' --output text)
export DB_PASSWORD
```

### アンチパターン 5: インスタンスサイズを大きくしすぎる

```
# 悪い例 — 「念のため」で巨大インスタンスを選択
→ m5.4xlarge (16 vCPU, 64 GiB) で CPU 使用率 5%

# 良い例 — 適切なサイズで開始し、モニタリングに基づいてスケール
→ t3.medium (2 vCPU, 4 GiB) で開始
→ CPU 使用率が 60% を超えたら t3.large に変更
→ AWS Compute Optimizer の推奨を定期的に確認
```

```bash
# AWS Compute Optimizer で推奨インスタンスタイプを確認
aws compute-optimizer get-ec2-instance-recommendations \
  --instance-arns arn:aws:ec2:ap-northeast-1:123456789012:instance/i-0123456789abcdef0 \
  --query 'instanceRecommendations[].[instanceArn,currentInstanceType,recommendationOptions[0].instanceType,finding]' \
  --output table
```

---

## 11. EC2 運用のベストプラクティス

### 11.1 タグ付け戦略

| タグキー | 説明 | 例 |
|---------|------|-----|
| Name | リソースの識別名 | web-server-01 |
| Environment | 環境 | production / staging / development |
| Team | 所有チーム | backend / frontend / infra |
| CostCenter | コスト配分先 | CC-12345 |
| Application | アプリケーション名 | my-web-app |
| ManagedBy | 管理方法 | terraform / cloudformation / manual |
| Backup | バックアップ対象 | true / false |

```bash
# タグ付けポリシーの適用
aws ec2 create-tags \
  --resources i-0123456789abcdef0 \
  --tags \
    Key=Name,Value=web-server-01 \
    Key=Environment,Value=production \
    Key=Team,Value=backend \
    Key=CostCenter,Value=CC-12345 \
    Key=Application,Value=my-web-app \
    Key=ManagedBy,Value=terraform \
    Key=Backup,Value=true
```

### 11.2 パッチ管理

```bash
# SSM Patch Manager でパッチ適用を自動化
aws ssm create-patch-baseline \
  --name "AmazonLinux2023-Custom" \
  --operating-system AMAZON_LINUX_2023 \
  --approval-rules '{
    "PatchRules": [{
      "PatchFilterGroup": {
        "PatchFilters": [
          {"Key": "SEVERITY", "Values": ["Critical", "Important"]},
          {"Key": "CLASSIFICATION", "Values": ["Security", "Bugfix"]}
        ]
      },
      "ApproveAfterDays": 3,
      "ComplianceLevel": "CRITICAL"
    }]
  }'

# パッチ適用のメンテナンスウィンドウを作成
aws ssm create-maintenance-window \
  --name "WeeklyPatching" \
  --schedule "cron(0 2 ? * SUN *)" \
  --duration 3 \
  --cutoff 1 \
  --allow-unassociated-targets
```

---

## 12. FAQ

### Q1. t3.micro と t3.small のどちらを選ぶべきか？

t3.micro (1 GiB メモリ) は軽量な Web サーバーやテスト用途に適している。メモリ 2 GiB 以上が必要なアプリケーション（WordPress + MySQL など）は t3.small 以上を選択する。無料枠を使う場合は t2.micro (12ヶ月無料) も検討対象。コスト効率を重視するなら Graviton ベースの t4g ファミリーも有力な選択肢である。

### Q2. インスタンスを停止するとデータはどうなるか？

EBS ルートボリューム (DeleteOnTermination=true) はインスタンス「終了」で削除されるが、「停止」では保持される。インスタンスストアは停止・終了の両方でデータが失われる。重要なデータは EBS スナップショットや S3 にバックアップする。なお、停止中もEBSの課金は継続される点に注意。

### Q3. IMDSv2 とは何か？なぜ必要か？

Instance Metadata Service v2 は、トークンベースの認証をメタデータアクセスに追加するセキュリティ強化。SSRF 攻撃によるメタデータ漏洩を防止する。`HttpTokens=required` で IMDSv2 を強制すべきである。Capital One の2019年データ漏洩事件は、IMDSv1 の脆弱性を悪用したものであり、このインシデントが IMDSv2 開発の契機となった。

### Q4. EC2 インスタンスのバックアップ戦略はどうすべきか？

以下の3層でバックアップを構成するのが推奨である。

1. **EBS スナップショット**: Data Lifecycle Manager (DLM) で日次自動取得、7-30日保持
2. **AMI**: 週次でゴールデン AMI を作成、EC2 Image Builder で自動化
3. **AWS Backup**: 組織全体のバックアップを一元管理、クロスリージョンコピーにも対応

### Q5. Elastic IP は必要か？

Elastic IP は静的なパブリック IP アドレスであり、インスタンスの停止・起動でも IP が変わらない。ただし、未使用の Elastic IP には課金が発生する。多くのケースでは DNS (Route 53) と ALB の組み合わせの方が柔軟で推奨される。直接 IP でアクセスする必要がある場合（VPN接続先の指定など）に限り Elastic IP を使用する。

### Q6. EC2 のリージョン・AZ の選定基準は？

| 考慮事項 | 説明 |
|---------|------|
| レイテンシ | ユーザーに近いリージョンを選択 |
| コスト | リージョンにより料金が異なる（東京 > バージニア） |
| コンプライアンス | データ主権要件によるリージョン制約 |
| サービス提供状況 | 一部サービスは特定リージョンのみ |
| AZ 分散 | 可用性のため最低2つの AZ を使用 |

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| インスタンスタイプ | ワークロードに応じてファミリーとサイズを選定、Graviton を積極活用 |
| AMI | Amazon Linux 2023 / Ubuntu が一般的、SSM パラメータストアで最新 AMI を取得 |
| セキュリティグループ | ステートフル、最小限のポート開放、ソースに SG ID を指定 |
| キーペア | Ed25519 推奨、Session Manager への移行を検討 |
| EBS | gp3 がデフォルト推奨、暗号化を必ず有効化、DLM でスナップショット自動化 |
| User Data | 起動時の自動セットアップに活用、シークレットは Secrets Manager から取得 |
| IMDSv2 | 必ず有効化し SSRF 対策を実施、アカウントレベルで強制 |
| 監視 | CloudWatch + CloudWatch Agent でメトリクスとログを収集 |
| IaC | CloudFormation / CDK でインフラをコード管理 |
| コスト | Compute Optimizer で適正サイズを確認、不要リソースを削除 |

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
5. AWS Graviton プロセッサ — https://aws.amazon.com/ec2/graviton/
6. EC2 Image Builder — https://docs.aws.amazon.com/imagebuilder/latest/userguide/
7. AWS Systems Manager Session Manager — https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html
8. AWS Compute Optimizer — https://docs.aws.amazon.com/compute-optimizer/latest/ug/
