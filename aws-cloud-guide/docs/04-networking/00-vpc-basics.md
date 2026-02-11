# Amazon VPC 基礎

> AWS のネットワーク基盤である VPC を理解し、サブネット設計・ルートテーブル・IGW/NAT GW を使った本番ネットワーク構成を実践的に習得する

## この章で学ぶこと

1. **VPC の基本アーキテクチャ** — CIDR 設計、サブネット分割、AZ 配置の設計判断
2. **ルーティングとゲートウェイ** — ルートテーブル、IGW、NAT GW の役割と構成
3. **セキュリティ制御** — セキュリティグループ、ネットワーク ACL、VPC エンドポイント

---

## 1. VPC アーキテクチャ全体像

```
+----------------------------------------------------------------------+
|  AWS Region (ap-northeast-1)                                         |
|  +----------------------------------------------------------------+  |
|  |  VPC: 10.0.0.0/16 (65,536 IPs)                                |  |
|  |                                                                |  |
|  |  +-- AZ-1a ----------------+  +-- AZ-1c ----------------+     |  |
|  |  |                         |  |                         |     |  |
|  |  |  Public Subnet          |  |  Public Subnet          |     |  |
|  |  |  10.0.1.0/24            |  |  10.0.2.0/24            |     |  |
|  |  |  [ALB] [NAT GW]        |  |  [ALB] [NAT GW]        |     |  |
|  |  |                         |  |                         |     |  |
|  |  |  Private Subnet (App)   |  |  Private Subnet (App)   |     |  |
|  |  |  10.0.11.0/24           |  |  10.0.12.0/24           |     |  |
|  |  |  [ECS/EC2]              |  |  [ECS/EC2]              |     |  |
|  |  |                         |  |                         |     |  |
|  |  |  Private Subnet (DB)    |  |  Private Subnet (DB)    |     |  |
|  |  |  10.0.21.0/24           |  |  10.0.22.0/24           |     |  |
|  |  |  [RDS] [ElastiCache]    |  |  [RDS Standby]         |     |  |
|  |  +-------------------------+  +-------------------------+     |  |
|  +----------------------------------------------------------------+  |
|       |                                                              |
|  +----+----+                                                         |
|  |   IGW   | <--> Internet                                          |
|  +---------+                                                         |
+----------------------------------------------------------------------+
```

### コード例 1: VPC とサブネットの作成（AWS CLI）

```bash
# VPC の作成
VPC_ID=$(aws ec2 create-vpc \
  --cidr-block 10.0.0.0/16 \
  --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=prod-vpc}]' \
  --query 'Vpc.VpcId' --output text)

# DNS 有効化
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-support
aws ec2 modify-vpc-attribute --vpc-id $VPC_ID --enable-dns-hostnames

# パブリックサブネット（AZ-1a, AZ-1c）
PUB_1A=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.1.0/24 --availability-zone ap-northeast-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=pub-1a}]' \
  --query 'Subnet.SubnetId' --output text)

PUB_1C=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.2.0/24 --availability-zone ap-northeast-1c \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=pub-1c}]' \
  --query 'Subnet.SubnetId' --output text)

# プライベートサブネット（App層 / DB層）
PRIV_APP_1A=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.11.0/24 --availability-zone ap-northeast-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=priv-app-1a}]' \
  --query 'Subnet.SubnetId' --output text)

PRIV_DB_1A=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.21.0/24 --availability-zone ap-northeast-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=priv-db-1a}]' \
  --query 'Subnet.SubnetId' --output text)
```

---

## 2. CIDR 設計

### CIDR ブロックサイズ早見表

| CIDR | IP 数 | 利用可能 IP | 用途例 |
|---|---|---|---|
| /16 | 65,536 | 65,531 | VPC 全体（推奨） |
| /20 | 4,096 | 4,091 | 大規模サブネット |
| /24 | 256 | 251 | 標準サブネット |
| /26 | 64 | 59 | 小規模サブネット |
| /28 | 16 | 11 | 最小サブネット |

> AWS はサブネットごとに 5 IP を予約する（ネットワーク、VPC ルーター、DNS、将来予約、ブロードキャスト）

### 推奨 CIDR 設計パターン

```
VPC: 10.0.0.0/16 の設計例
============================

Public Subnets     (各 /24 = 251 IP)
  AZ-a: 10.0.1.0/24
  AZ-c: 10.0.2.0/24
  AZ-d: 10.0.3.0/24

Private App        (各 /20 = 4,091 IP)
  AZ-a: 10.0.16.0/20
  AZ-c: 10.0.32.0/20
  AZ-d: 10.0.48.0/20

Private DB         (各 /24 = 251 IP)
  AZ-a: 10.0.64.0/24
  AZ-c: 10.0.65.0/24
  AZ-d: 10.0.66.0/24

予備               10.0.128.0/17 (将来の拡張用に確保)
```

### コード例 2: Terraform による VPC 定義

```hcl
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.5.0"

  name = "prod-vpc"
  cidr = "10.0.0.0/16"

  azs              = ["ap-northeast-1a", "ap-northeast-1c", "ap-northeast-1d"]
  public_subnets   = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  private_subnets  = ["10.0.16.0/20", "10.0.32.0/20", "10.0.48.0/20"]
  database_subnets = ["10.0.64.0/24", "10.0.65.0/24", "10.0.66.0/24"]

  enable_nat_gateway     = true
  single_nat_gateway     = false  # 本番: AZ ごとに NAT GW
  one_nat_gateway_per_az = true

  enable_dns_hostnames = true
  enable_dns_support   = true

  create_database_subnet_group       = true
  create_database_subnet_route_table = true

  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_iam_role             = true

  tags = {
    Environment = "production"
    Terraform   = "true"
  }
}
```

---

## 3. ルートテーブルとゲートウェイ

```
ルーティングの仕組み
======================

[Public Subnet ルートテーブル]
+--------------------+-----------+
| Destination        | Target    |
+--------------------+-----------+
| 10.0.0.0/16        | local     |  <-- VPC 内通信
| 0.0.0.0/0          | igw-xxx   |  <-- インターネットへ
+--------------------+-----------+

[Private App Subnet ルートテーブル]
+--------------------+-----------+
| Destination        | Target    |
+--------------------+-----------+
| 10.0.0.0/16        | local     |  <-- VPC 内通信
| 0.0.0.0/0          | nat-xxx   |  <-- NAT GW 経由
+--------------------+-----------+

[Private DB Subnet ルートテーブル]
+--------------------+-----------+
| Destination        | Target    |
+--------------------+-----------+
| 10.0.0.0/16        | local     |  <-- VPC 内通信のみ
+--------------------+-----------+
```

### コード例 3: IGW と NAT GW の設定

```bash
# Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=prod-igw}]' \
  --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID

# Elastic IP + NAT Gateway
EIP_ID=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
NAT_ID=$(aws ec2 create-nat-gateway \
  --subnet-id $PUB_1A --allocation-id $EIP_ID \
  --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-1a}]' \
  --query 'NatGateway.NatGatewayId' --output text)

# Public ルートテーブル
PUB_RT=$(aws ec2 create-route-table --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=pub-rt}]' \
  --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id $PUB_RT \
  --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID
aws ec2 associate-route-table --route-table-id $PUB_RT --subnet-id $PUB_1A

# Private ルートテーブル（NAT GW 経由）
PRIV_RT=$(aws ec2 create-route-table --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=priv-rt-1a}]' \
  --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id $PRIV_RT \
  --destination-cidr-block 0.0.0.0/0 --nat-gateway-id $NAT_ID
aws ec2 associate-route-table --route-table-id $PRIV_RT --subnet-id $PRIV_APP_1A
```

---

## 4. セキュリティグループとネットワーク ACL

### SG vs NACL 比較表

| 特性 | セキュリティグループ (SG) | ネットワーク ACL (NACL) |
|---|---|---|
| **適用レベル** | ENI（インスタンス単位） | サブネット単位 |
| **ステート** | ステートフル（戻りは自動許可） | ステートレス（戻りも明示必要） |
| **ルール** | 許可のみ | 許可 + 拒否 |
| **評価順序** | 全ルールを評価 | 番号順に評価、最初の一致 |
| **デフォルト** | 全アウトバウンド許可 | 全トラフィック許可 |
| **推奨用途** | 主要なアクセス制御 | 追加の防御層（サブネットレベル） |

### コード例 4: 3層アーキテクチャの SG 設計

```bash
# ALB 用 SG
ALB_SG=$(aws ec2 create-security-group \
  --group-name alb-sg --description "ALB SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $ALB_SG \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# App 用 SG（ALB からのみ受信）
APP_SG=$(aws ec2 create-security-group \
  --group-name app-sg --description "App SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $APP_SG \
  --protocol tcp --port 8080 --source-group $ALB_SG

# DB 用 SG（App からのみ受信）
DB_SG=$(aws ec2 create-security-group \
  --group-name db-sg --description "DB SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $DB_SG \
  --protocol tcp --port 3306 --source-group $APP_SG
aws ec2 authorize-security-group-ingress --group-id $DB_SG \
  --protocol tcp --port 6379 --source-group $APP_SG
```

---

## 5. VPC エンドポイント

```
VPC エンドポイントの種類
==========================

Gateway Endpoint (無料)
  対応: S3, DynamoDB
  ルートテーブルにエントリ追加
  App --> Route Table --> S3 (AWS 内部ネットワーク)

Interface Endpoint (有料: ~$0.014/時 + データ転送)
  対応: ほぼ全 AWS サービス
  サブネットに ENI を作成
  App --> ENI --> AWS サービス (PrivateLink)
```

### コード例 5: VPC エンドポイントの作成

```bash
# Gateway エンドポイント（S3）- 無料
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.s3 \
  --route-table-ids $PRIV_RT \
  --vpc-endpoint-type Gateway

# Gateway エンドポイント（DynamoDB）- 無料
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.dynamodb \
  --route-table-ids $PRIV_RT \
  --vpc-endpoint-type Gateway

# Interface エンドポイント（Secrets Manager）- 有料
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.secretsmanager \
  --vpc-endpoint-type Interface \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --security-group-ids $ENDPOINT_SG \
  --private-dns-enabled
```

---

## 6. VPC Peering と Transit Gateway

### コード例 6: VPC Peering の設定

```bash
# VPC Peering 接続の作成
PEERING_ID=$(aws ec2 create-vpc-peering-connection \
  --vpc-id $VPC_ID --peer-vpc-id vpc-0abc123 \
  --tag-specifications 'ResourceType=vpc-peering-connection,Tags=[{Key=Name,Value=prod-to-shared}]' \
  --query 'VpcPeeringConnection.VpcPeeringConnectionId' --output text)

# 承認
aws ec2 accept-vpc-peering-connection --vpc-peering-connection-id $PEERING_ID

# ルートテーブルに Peering ルート追加
aws ec2 create-route --route-table-id $PRIV_RT \
  --destination-cidr-block 10.1.0.0/16 \
  --vpc-peering-connection-id $PEERING_ID
```

---

## アンチパターン

### 1. 全リソースをパブリックサブネットに配置

**問題**: EC2、RDS、ElastiCache をすべてパブリックサブネットに配置すると、セキュリティグループの設定ミスで内部リソースがインターネットに露出するリスクがある。多層防御の原則に反する。

**対策**: 3層サブネット設計を採用する。パブリックには ALB/NAT GW のみ配置し、アプリケーションとデータベースはプライベートサブネットに配置する。

### 2. CIDR の過小設計

**問題**: VPC を `/24` のような小さい CIDR で作成すると、EKS ノード、Lambda ENI、ElastiCache ノードなど予想外に IP を消費するサービスで IP 枯渇が発生する。VPC の CIDR は後から変更できない。

**対策**: VPC は `/16` で作成し、サブネットは用途に応じて `/20` 〜 `/24` で分割する。将来の拡張用に CIDR 空間の半分は予約しておく。

---

## FAQ

### Q1: NAT Gateway のコストが高い場合の対策は？

**A**: NAT GW は約 $0.062/時 + データ処理 $0.062/GB で、月額約 $45 + データ転送量です。コスト削減策:
1. **開発環境**: NAT Instance（t4g.nano: 約 $3/月）で代替
2. **VPC エンドポイント**: S3・DynamoDB は Gateway Endpoint（無料）で NAT GW を経由しない
3. **ECR Image Pull**: VPC エンドポイントで NAT GW トラフィックを削減
4. **シングル NAT GW**: 開発環境では AZ ごとではなく1つの NAT GW を共有

### Q2: Peering と Transit Gateway はどう使い分けますか？

**A**:
- **VPC Peering**: 2-3 VPC の接続。無料（データ転送料のみ）。1対1接続
- **Transit Gateway**: 多数の VPC/オンプレミス接続。ハブ&スポーク構成。時間課金（約 $0.07/時）
VPC が 3 つ以下なら Peering、4 つ以上や VPN 接続がある場合は Transit Gateway が効率的です。

### Q3: VPC Flow Logs は有効にすべきですか？

**A**: プロダクション環境では必須です。セキュリティインシデント調査、ネットワークトラブルシュート、コンプライアンスで必要です。コスト最適化のため、送信先は S3（CloudWatch Logs より安価）を選び、カスタムフォーマットで必要なフィールドのみ記録してください。

---

## まとめ

| 項目 | 要点 |
|---|---|
| VPC 設計 | /16 CIDR、3層サブネット（Public/Private App/Private DB）、マルチ AZ |
| サブネット | Public: ALB/NAT、Private App: ECS/EC2、Private DB: RDS/Cache |
| ルーティング | Public -> IGW、Private -> NAT GW、DB -> local のみ |
| セキュリティ | SG でアクセス制御（主）、NACL で追加防御（補助） |
| VPC エンドポイント | S3/DynamoDB は Gateway（無料）、その他は Interface |
| コスト注意 | NAT GW が主要コスト要素。VPC エンドポイントで削減 |

## 次に読むべきガイド

- [RDS 基礎](../03-database/00-rds-basics.md) — VPC 内でのデータベース配置
- [ElastiCache](../03-database/02-elasticache.md) — プライベートサブネットでのキャッシュ構築
- [DynamoDB](../03-database/01-dynamodb.md) — VPC エンドポイントでの接続最適化

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon VPC ユーザーガイド](https://docs.aws.amazon.com/ja_jp/vpc/latest/userguide/) — VPC の全機能リファレンス
2. **AWS Well-Architected Framework**: [セキュリティの柱](https://docs.aws.amazon.com/ja_jp/wellarchitected/latest/security-pillar/) — ネットワークセキュリティのベストプラクティス
3. **AWS ブログ**: [VPC ベストプラクティス](https://aws.amazon.com/blogs/networking-and-content-delivery/) — 実践的な VPC 設計パターン
