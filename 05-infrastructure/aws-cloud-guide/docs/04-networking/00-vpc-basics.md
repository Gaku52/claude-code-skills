# Amazon VPC 基礎

> AWS のネットワーク基盤である VPC を理解し、サブネット設計・ルートテーブル・IGW/NAT GW を使った本番ネットワーク構成を実践的に習得する

## この章で学ぶこと

1. **VPC の基本アーキテクチャ** — CIDR 設計、サブネット分割、AZ 配置の設計判断
2. **ルーティングとゲートウェイ** — ルートテーブル、IGW、NAT GW の役割と構成
3. **セキュリティ制御** — セキュリティグループ、ネットワーク ACL、VPC エンドポイント
4. **VPC 間接続** — VPC Peering、Transit Gateway、PrivateLink の使い分け
5. **VPC Flow Logs と監視** — ネットワークトラフィックの可視化とトラブルシューティング

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

### VPC の主要コンポーネント一覧

| コンポーネント | 説明 | スコープ |
|---|---|---|
| VPC | 仮想ネットワークの論理的な隔離空間 | リージョン |
| サブネット | VPC 内の IP アドレス範囲 | アベイラビリティゾーン |
| ルートテーブル | サブネットのトラフィックルーティング規則 | VPC |
| Internet Gateway (IGW) | VPC とインターネット間の接続ポイント | VPC |
| NAT Gateway | プライベートサブネットからの外向きインターネット接続 | AZ |
| セキュリティグループ | インスタンスレベルのステートフルファイアウォール | VPC |
| ネットワーク ACL | サブネットレベルのステートレスファイアウォール | VPC |
| VPC エンドポイント | VPC 内から AWS サービスへのプライベート接続 | VPC |
| Elastic IP | 静的なパブリック IPv4 アドレス | リージョン |
| ENI | 仮想ネットワークインターフェースカード | AZ |

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

PRIV_APP_1C=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.12.0/24 --availability-zone ap-northeast-1c \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=priv-app-1c}]' \
  --query 'Subnet.SubnetId' --output text)

PRIV_DB_1A=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.21.0/24 --availability-zone ap-northeast-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=priv-db-1a}]' \
  --query 'Subnet.SubnetId' --output text)

PRIV_DB_1C=$(aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 10.0.22.0/24 --availability-zone ap-northeast-1c \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=priv-db-1c}]' \
  --query 'Subnet.SubnetId' --output text)

# パブリックサブネットの自動パブリック IP 割り当て有効化
aws ec2 modify-subnet-attribute \
  --subnet-id $PUB_1A \
  --map-public-ip-on-launch

aws ec2 modify-subnet-attribute \
  --subnet-id $PUB_1C \
  --map-public-ip-on-launch
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

### RFC 1918 プライベート IP アドレス範囲

| 範囲 | CIDR | IP 数 | 推奨用途 |
|---|---|---|---|
| 10.0.0.0 - 10.255.255.255 | 10.0.0.0/8 | 16,777,216 | 大規模ネットワーク（推奨） |
| 172.16.0.0 - 172.31.255.255 | 172.16.0.0/12 | 1,048,576 | 中規模ネットワーク |
| 192.168.0.0 - 192.168.255.255 | 192.168.0.0/16 | 65,536 | 小規模ネットワーク |

### マルチ VPC 環境での CIDR 割当計画

```
マルチアカウント・マルチ VPC の IP 割当例:
=============================================

本番環境 (Production Account)
  prod-vpc:       10.0.0.0/16
  shared-svc-vpc: 10.1.0.0/16

ステージング環境 (Staging Account)
  staging-vpc:    10.2.0.0/16

開発環境 (Development Account)
  dev-vpc:        10.3.0.0/16

セキュリティ環境 (Security Account)
  security-vpc:   10.4.0.0/16

ログ集約環境 (Logging Account)
  logging-vpc:    10.5.0.0/16

ポイント:
  - VPC 間で CIDR が重複しないように計画する
  - VPC Peering / Transit Gateway で接続する場合は重複不可
  - 10.0.0.0/8 の範囲を複数 /16 に分割して割当
  - オンプレミスのネットワークとも重複しないよう調整
  - Secondary CIDR の追加も検討（最大 5 つ）
```

### コード例 2: Secondary CIDR の追加

```bash
# VPC に Secondary CIDR ブロックを追加
aws ec2 associate-vpc-cidr-block \
  --vpc-id $VPC_ID \
  --cidr-block 100.64.0.0/16

# 追加した CIDR でサブネットを作成
aws ec2 create-subnet --vpc-id $VPC_ID \
  --cidr-block 100.64.1.0/24 \
  --availability-zone ap-northeast-1a \
  --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=eks-pods-1a}]'

# 利用シーン:
# - EKS Pod のカスタムネットワーキング（Pod に VPC IP を割当）
# - IP アドレス空間が不足した場合の拡張
# - RFC 6598 (100.64.0.0/10) は CGN 用だが VPC 内では利用可能
```

### コード例 3: Terraform による VPC 定義

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

  # パブリックサブネットのタグ（EKS ALB Ingress Controller 用）
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  # プライベートサブネットのタグ（EKS 内部 LB 用）
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }

  tags = {
    Environment = "production"
    Terraform   = "true"
  }
}

# VPC のアウトプット
output "vpc_id" {
  value = module.vpc.vpc_id
}

output "private_subnet_ids" {
  value = module.vpc.private_subnets
}

output "public_subnet_ids" {
  value = module.vpc.public_subnets
}

output "database_subnet_group_name" {
  value = module.vpc.database_subnet_group_name
}
```

### コード例 4: CloudFormation による VPC 定義

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Production VPC with 3-tier subnet architecture

Parameters:
  EnvironmentName:
    Type: String
    Default: prod
  VpcCIDR:
    Type: String
    Default: 10.0.0.0/16

Resources:
  # VPC
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCIDR
      EnableDnsSupport: true
      EnableDnsHostnames: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-vpc

  # Internet Gateway
  InternetGateway:
    Type: AWS::EC2::InternetGateway
    Properties:
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-igw

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # Public Subnets
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.1.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-pub-1a

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.2.0/24
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-pub-1c

  # Private App Subnets
  PrivateAppSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.11.0/24
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-priv-app-1a

  PrivateAppSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.12.0/24
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-priv-app-1c

  # Private DB Subnets
  PrivateDBSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [0, !GetAZs '']
      CidrBlock: 10.0.21.0/24
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-priv-db-1a

  PrivateDBSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      AvailabilityZone: !Select [1, !GetAZs '']
      CidrBlock: 10.0.22.0/24
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-priv-db-1c

  # NAT Gateway (AZ-1a)
  NatGateway1EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc

  NatGateway1:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway1EIP.AllocationId
      SubnetId: !Ref PublicSubnet1
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-nat-1a

  # NAT Gateway (AZ-1c)
  NatGateway2EIP:
    Type: AWS::EC2::EIP
    DependsOn: InternetGatewayAttachment
    Properties:
      Domain: vpc

  NatGateway2:
    Type: AWS::EC2::NatGateway
    Properties:
      AllocationId: !GetAtt NatGateway2EIP.AllocationId
      SubnetId: !Ref PublicSubnet2
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-nat-1c

  # Public Route Table
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-pub-rt

  DefaultPublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet1

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PublicRouteTable
      SubnetId: !Ref PublicSubnet2

  # Private Route Tables (AZ ごと)
  PrivateRouteTable1:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-priv-rt-1a

  DefaultPrivateRoute1:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway1

  PrivateAppSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable1
      SubnetId: !Ref PrivateAppSubnet1

  PrivateRouteTable2:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC
      Tags:
        - Key: Name
          Value: !Sub ${EnvironmentName}-priv-rt-1c

  DefaultPrivateRoute2:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable2
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway2

  PrivateAppSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      RouteTableId: !Ref PrivateRouteTable2
      SubnetId: !Ref PrivateAppSubnet2

Outputs:
  VpcId:
    Value: !Ref VPC
    Export:
      Name: !Sub ${EnvironmentName}-VpcId

  PublicSubnets:
    Value: !Join [',', [!Ref PublicSubnet1, !Ref PublicSubnet2]]
    Export:
      Name: !Sub ${EnvironmentName}-PublicSubnets

  PrivateAppSubnets:
    Value: !Join [',', [!Ref PrivateAppSubnet1, !Ref PrivateAppSubnet2]]
    Export:
      Name: !Sub ${EnvironmentName}-PrivateAppSubnets

  PrivateDBSubnets:
    Value: !Join [',', [!Ref PrivateDBSubnet1, !Ref PrivateDBSubnet2]]
    Export:
      Name: !Sub ${EnvironmentName}-PrivateDBSubnets
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

### ルートテーブルの評価ルール

ルートテーブルでは、宛先 IP に対して最も具体的な（プレフィックスが長い）ルートが優先される。例えば、`10.1.0.0/16` と `0.0.0.0/0` の両方が存在する場合、`10.1.x.x` 宛のトラフィックは `10.1.0.0/16` のルートに従う。`local` ルートは常に最優先で、削除できない。

### コード例 5: IGW と NAT GW の設定

```bash
# Internet Gateway
IGW_ID=$(aws ec2 create-internet-gateway \
  --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=prod-igw}]' \
  --query 'InternetGateway.InternetGatewayId' --output text)
aws ec2 attach-internet-gateway --internet-gateway-id $IGW_ID --vpc-id $VPC_ID

# Elastic IP + NAT Gateway (AZ-1a)
EIP_1A=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
NAT_1A=$(aws ec2 create-nat-gateway \
  --subnet-id $PUB_1A --allocation-id $EIP_1A \
  --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-1a}]' \
  --query 'NatGateway.NatGatewayId' --output text)

# NAT Gateway の作成完了を待機
aws ec2 wait nat-gateway-available --nat-gateway-ids $NAT_1A

# Elastic IP + NAT Gateway (AZ-1c)
EIP_1C=$(aws ec2 allocate-address --domain vpc --query 'AllocationId' --output text)
NAT_1C=$(aws ec2 create-nat-gateway \
  --subnet-id $PUB_1C --allocation-id $EIP_1C \
  --tag-specifications 'ResourceType=natgateway,Tags=[{Key=Name,Value=nat-1c}]' \
  --query 'NatGateway.NatGatewayId' --output text)

aws ec2 wait nat-gateway-available --nat-gateway-ids $NAT_1C

# Public ルートテーブル
PUB_RT=$(aws ec2 create-route-table --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=pub-rt}]' \
  --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id $PUB_RT \
  --destination-cidr-block 0.0.0.0/0 --gateway-id $IGW_ID
aws ec2 associate-route-table --route-table-id $PUB_RT --subnet-id $PUB_1A
aws ec2 associate-route-table --route-table-id $PUB_RT --subnet-id $PUB_1C

# Private ルートテーブル AZ-1a（NAT GW 経由）
PRIV_RT_1A=$(aws ec2 create-route-table --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=priv-rt-1a}]' \
  --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id $PRIV_RT_1A \
  --destination-cidr-block 0.0.0.0/0 --nat-gateway-id $NAT_1A
aws ec2 associate-route-table --route-table-id $PRIV_RT_1A --subnet-id $PRIV_APP_1A
aws ec2 associate-route-table --route-table-id $PRIV_RT_1A --subnet-id $PRIV_DB_1A

# Private ルートテーブル AZ-1c（NAT GW 経由）
PRIV_RT_1C=$(aws ec2 create-route-table --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=route-table,Tags=[{Key=Name,Value=priv-rt-1c}]' \
  --query 'RouteTable.RouteTableId' --output text)
aws ec2 create-route --route-table-id $PRIV_RT_1C \
  --destination-cidr-block 0.0.0.0/0 --nat-gateway-id $NAT_1C
aws ec2 associate-route-table --route-table-id $PRIV_RT_1C --subnet-id $PRIV_APP_1C
aws ec2 associate-route-table --route-table-id $PRIV_RT_1C --subnet-id $PRIV_DB_1C
```

### NAT Gateway vs NAT Instance 比較

| 項目 | NAT Gateway | NAT Instance |
|---|---|---|
| **可用性** | AZ 内で高可用（AWS 管理） | 手動でフェイルオーバー構成 |
| **帯域幅** | 最大 100 Gbps | インスタンスタイプ依存 |
| **メンテナンス** | AWS 管理（パッチ不要） | ユーザー管理 |
| **コスト** | ~$0.062/時 + $0.062/GB | インスタンス料金のみ |
| **セキュリティグループ** | 関連付け不可 | 関連付け可能 |
| **ポートフォワーディング** | 非対応 | 対応 |
| **Bastion ホスト兼用** | 不可 | 可能 |
| **推奨** | 本番環境 | 開発/テスト環境（コスト重視） |

### コード例 6: NAT Instance による低コスト構成（開発環境向け）

```bash
# NAT Instance の作成（Amazon Linux 2023 AMI）
NAT_INSTANCE=$(aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t4g.nano \
  --subnet-id $PUB_1A \
  --security-group-ids $NAT_SG \
  --key-name my-key \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nat-instance}]' \
  --query 'Instances[0].InstanceId' --output text)

# Source/Destination Check を無効化（NAT に必須）
aws ec2 modify-instance-attribute \
  --instance-id $NAT_INSTANCE \
  --no-source-dest-check

# NAT Instance 用のセキュリティグループ
NAT_SG=$(aws ec2 create-security-group \
  --group-name nat-sg --description "NAT Instance SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)

# プライベートサブネットからの全トラフィックを許可
aws ec2 authorize-security-group-ingress --group-id $NAT_SG \
  --protocol -1 --cidr 10.0.0.0/16

# プライベートルートテーブルのデフォルトルートを NAT Instance に設定
aws ec2 create-route --route-table-id $PRIV_RT_1A \
  --destination-cidr-block 0.0.0.0/0 \
  --instance-id $NAT_INSTANCE
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

### コード例 7: 3層アーキテクチャの SG 設計

```bash
# ALB 用 SG
ALB_SG=$(aws ec2 create-security-group \
  --group-name alb-sg --description "ALB SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $ALB_SG \
  --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --group-id $ALB_SG \
  --protocol tcp --port 80 --cidr 0.0.0.0/0

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

# Bastion 用 SG（特定 IP からのみ SSH）
BASTION_SG=$(aws ec2 create-security-group \
  --group-name bastion-sg --description "Bastion SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $BASTION_SG \
  --protocol tcp --port 22 --cidr 203.0.113.0/32

# App SG に Bastion からの SSH を追加
aws ec2 authorize-security-group-ingress --group-id $APP_SG \
  --protocol tcp --port 22 --source-group $BASTION_SG

# VPC Endpoint 用 SG
ENDPOINT_SG=$(aws ec2 create-security-group \
  --group-name endpoint-sg --description "VPC Endpoint SG" \
  --vpc-id $VPC_ID --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --group-id $ENDPOINT_SG \
  --protocol tcp --port 443 --cidr 10.0.0.0/16
```

### コード例 8: ネットワーク ACL によるサブネットレベルの防御

```bash
# DB サブネット用 NACL
DB_NACL=$(aws ec2 create-network-acl --vpc-id $VPC_ID \
  --tag-specifications 'ResourceType=network-acl,Tags=[{Key=Name,Value=db-nacl}]' \
  --query 'NetworkAcl.NetworkAclId' --output text)

# インバウンド: App サブネットからの MySQL/Redis のみ許可
aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 100 --protocol tcp --port-range From=3306,To=3306 \
  --cidr-block 10.0.11.0/24 --rule-action allow --ingress

aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 110 --protocol tcp --port-range From=3306,To=3306 \
  --cidr-block 10.0.12.0/24 --rule-action allow --ingress

aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 120 --protocol tcp --port-range From=6379,To=6379 \
  --cidr-block 10.0.11.0/24 --rule-action allow --ingress

aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 130 --protocol tcp --port-range From=6379,To=6379 \
  --cidr-block 10.0.12.0/24 --rule-action allow --ingress

# アウトバウンド: エフェメラルポート（戻りトラフィック）のみ許可
aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 100 --protocol tcp --port-range From=1024,To=65535 \
  --cidr-block 10.0.11.0/24 --rule-action allow --egress

aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 110 --protocol tcp --port-range From=1024,To=65535 \
  --cidr-block 10.0.12.0/24 --rule-action allow --egress

# 全拒否ルール（デフォルトルールだが明示的に記載）
aws ec2 create-network-acl-entry --network-acl-id $DB_NACL \
  --rule-number 32767 --protocol -1 --cidr-block 0.0.0.0/0 \
  --rule-action deny --ingress

# NACL をサブネットに関連付け
aws ec2 replace-network-acl-association \
  --association-id aclassoc-xxxxx \
  --network-acl-id $DB_NACL
```

### セキュリティグループのベストプラクティス

```
1. ソースには CIDR ではなく SG ID を指定
   ✕ --cidr 10.0.11.0/24
   ○ --source-group sg-app-xxxxx
   理由: IP が変わっても追従する。意図が明確。

2. 用途ごとに SG を分離
   ✕ 1つの SG に全ルールを集約
   ○ ALB用、App用、DB用、管理用で分離
   理由: 最小権限の原則。変更の影響範囲を限定。

3. 説明フィールドを必ず記載
   aws ec2 authorize-security-group-ingress --group-id $SG \
     --ip-permissions '[{
       "IpProtocol": "tcp",
       "FromPort": 443,
       "ToPort": 443,
       "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "HTTPS from Internet"}]
     }]'

4. 定期的な棚卸し
   # 使用されていない SG の検出
   aws ec2 describe-security-groups \
     --query 'SecurityGroups[?length(IpPermissions)==`0` && length(IpPermissionsEgress)==`1`].[GroupId,GroupName]' \
     --output table
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

Gateway Load Balancer Endpoint
  対応: サードパーティアプライアンス
  ネットワークトラフィックのインスペクション用
  App --> GWLB Endpoint --> Firewall Appliance --> 宛先
```

### コード例 9: VPC エンドポイントの作成

```bash
# Gateway エンドポイント（S3）- 無料
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.s3 \
  --route-table-ids $PRIV_RT_1A $PRIV_RT_1C \
  --vpc-endpoint-type Gateway \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=s3-endpoint}]'

# Gateway エンドポイント（DynamoDB）- 無料
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.dynamodb \
  --route-table-ids $PRIV_RT_1A $PRIV_RT_1C \
  --vpc-endpoint-type Gateway \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=dynamodb-endpoint}]'

# Interface エンドポイント（Secrets Manager）- 有料
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.secretsmanager \
  --vpc-endpoint-type Interface \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --security-group-ids $ENDPOINT_SG \
  --private-dns-enabled \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=secretsmanager-endpoint}]'

# Interface エンドポイント（ECR - Docker Pull 用）
# ECR は 2 つのエンドポイントが必要
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.ecr.dkr \
  --vpc-endpoint-type Interface \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --security-group-ids $ENDPOINT_SG \
  --private-dns-enabled \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=ecr-dkr-endpoint}]'

aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.ecr.api \
  --vpc-endpoint-type Interface \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --security-group-ids $ENDPOINT_SG \
  --private-dns-enabled \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=ecr-api-endpoint}]'

# Interface エンドポイント（CloudWatch Logs）
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.logs \
  --vpc-endpoint-type Interface \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --security-group-ids $ENDPOINT_SG \
  --private-dns-enabled \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=logs-endpoint}]'

# Interface エンドポイント（STS - IAM ロール引き受け用）
aws ec2 create-vpc-endpoint \
  --vpc-id $VPC_ID \
  --service-name com.amazonaws.ap-northeast-1.sts \
  --vpc-endpoint-type Interface \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --security-group-ids $ENDPOINT_SG \
  --private-dns-enabled \
  --tag-specifications 'ResourceType=vpc-endpoint,Tags=[{Key=Name,Value=sts-endpoint}]'
```

### ECS/EKS で必要な VPC エンドポイント一覧

| サービス | エンドポイントタイプ | 必要性 | 用途 |
|---|---|---|---|
| S3 | Gateway (無料) | 必須 | ECR イメージレイヤーの取得 |
| ECR (dkr) | Interface | 必須 | Docker イメージの Pull |
| ECR (api) | Interface | 必須 | ECR API コール |
| CloudWatch Logs | Interface | 推奨 | ログ送信 |
| STS | Interface | EKS で必須 | IAM Roles for Service Accounts |
| Secrets Manager | Interface | 推奨 | シークレット取得 |
| SSM | Interface | 推奨 | パラメータストア、Session Manager |

### S3 Gateway エンドポイントのポリシー設定

```bash
# S3 エンドポイントへのポリシー設定（特定バケットのみ許可）
aws ec2 modify-vpc-endpoint \
  --vpc-endpoint-id vpce-xxxxx \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Sid": "AllowSpecificBuckets",
        "Effect": "Allow",
        "Principal": "*",
        "Action": [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ],
        "Resource": [
          "arn:aws:s3:::my-app-bucket",
          "arn:aws:s3:::my-app-bucket/*",
          "arn:aws:s3:::prod-*"
        ]
      },
      {
        "Sid": "AllowECRBucket",
        "Effect": "Allow",
        "Principal": "*",
        "Action": "s3:GetObject",
        "Resource": "arn:aws:s3:::prod-ap-northeast-1-starport-layer-bucket/*"
      }
    ]
  }'
```

---

## 6. VPC Peering と Transit Gateway

### VPC Peering

```
VPC Peering の構成:

  VPC-A (10.0.0.0/16) <----> VPC-B (10.1.0.0/16)
       |                          |
       +---- Peering 接続 --------+
       |                          |
  ルートテーブルに          ルートテーブルに
  10.1.0.0/16 -> pcx-xxx   10.0.0.0/16 -> pcx-xxx

制約:
  - 推移的ルーティング不可 (A-B, B-C でも A-C は通信不可)
  - CIDR 重複不可
  - リージョン間対応 (Inter-Region Peering)
```

### コード例 10: VPC Peering の設定

```bash
# VPC Peering 接続の作成
PEERING_ID=$(aws ec2 create-vpc-peering-connection \
  --vpc-id $VPC_ID --peer-vpc-id vpc-0abc123 \
  --tag-specifications 'ResourceType=vpc-peering-connection,Tags=[{Key=Name,Value=prod-to-shared}]' \
  --query 'VpcPeeringConnection.VpcPeeringConnectionId' --output text)

# 承認（相手側アカウントで実行、またはクロスアカウントの場合）
aws ec2 accept-vpc-peering-connection --vpc-peering-connection-id $PEERING_ID

# 双方のルートテーブルに Peering ルート追加
# VPC-A 側
aws ec2 create-route --route-table-id $PRIV_RT_1A \
  --destination-cidr-block 10.1.0.0/16 \
  --vpc-peering-connection-id $PEERING_ID

# VPC-B 側
aws ec2 create-route --route-table-id rtb-shared-xxx \
  --destination-cidr-block 10.0.0.0/16 \
  --vpc-peering-connection-id $PEERING_ID

# DNS 解決の有効化（Peering 先の プライベート DNS を解決）
aws ec2 modify-vpc-peering-connection-options \
  --vpc-peering-connection-id $PEERING_ID \
  --requester-peering-connection-options AllowDnsResolutionFromRemoteVpc=true \
  --accepter-peering-connection-options AllowDnsResolutionFromRemoteVpc=true
```

### Transit Gateway

```
Transit Gateway の構成（ハブ&スポーク):

                    +-------------------+
                    |  Transit Gateway  |
                    |  (ハブ)           |
                    +---+-----+-----+--+
                        |     |     |
            +-----------+     |     +-----------+
            |                 |                 |
  +----+----+---+   +---+----+----+   +---+----+----+
  | VPC-Prod    |   | VPC-Staging |   | VPC-Shared  |
  | 10.0.0.0/16 |   | 10.2.0.0/16|   | 10.1.0.0/16 |
  +-------------+   +------------+   +-------------+
                                             |
                                      VPN / Direct Connect
                                             |
                                      +------+------+
                                      | On-Premises |
                                      +-------------+

利点:
  - 推移的ルーティング対応 (A-B, B-C → A-C 通信可能)
  - ルートテーブルで通信制御
  - VPN / Direct Connect もアタッチ可能
  - 複数アカウント対応 (RAM で共有)
```

### コード例 11: Transit Gateway の作成と VPC アタッチ

```bash
# Transit Gateway の作成
TGW_ID=$(aws ec2 create-transit-gateway \
  --description "Central hub for all VPCs" \
  --options '{
    "AmazonSideAsn": 64512,
    "AutoAcceptSharedAttachments": "enable",
    "DefaultRouteTableAssociation": "enable",
    "DefaultRouteTablePropagation": "enable",
    "DnsSupport": "enable",
    "VpnEcmpSupport": "enable"
  }' \
  --tag-specifications 'ResourceType=transit-gateway,Tags=[{Key=Name,Value=central-tgw}]' \
  --query 'TransitGateway.TransitGatewayId' --output text)

# VPC アタッチメントの作成
aws ec2 create-transit-gateway-vpc-attachment \
  --transit-gateway-id $TGW_ID \
  --vpc-id $VPC_ID \
  --subnet-ids $PRIV_APP_1A $PRIV_APP_1C \
  --tag-specifications 'ResourceType=transit-gateway-attachment,Tags=[{Key=Name,Value=prod-vpc-attach}]'

# VPC のルートテーブルに TGW 経由のルート追加
aws ec2 create-route --route-table-id $PRIV_RT_1A \
  --destination-cidr-block 10.1.0.0/16 \
  --transit-gateway-id $TGW_ID

aws ec2 create-route --route-table-id $PRIV_RT_1A \
  --destination-cidr-block 10.2.0.0/16 \
  --transit-gateway-id $TGW_ID

# TGW ルートテーブルの確認
aws ec2 search-transit-gateway-routes \
  --transit-gateway-route-table-id tgw-rtb-xxxxx \
  --filters "Name=type,Values=propagated"
```

### Peering vs Transit Gateway 使い分け

| 項目 | VPC Peering | Transit Gateway |
|---|---|---|
| **接続トポロジー** | ポイントツーポイント | ハブ&スポーク |
| **推移的ルーティング** | 不可 | 可能 |
| **最大接続数** | VPC あたり 125 | 5,000 アタッチメント |
| **コスト** | データ転送料のみ | $0.07/時 + データ転送料 |
| **帯域幅** | 制限なし | VPC アタッチメントあたり 50 Gbps |
| **VPN/DX 統合** | 不可 | 可能 |
| **推奨** | 2-3 VPC の少数接続 | 4+ VPC、VPN/DX 統合 |

---

## 7. VPC Flow Logs

### VPC Flow Logs の概要

VPC Flow Logs は VPC 内のネットワークインターフェース間のトラフィック情報を記録する機能である。セキュリティ分析、ネットワーク監視、トラブルシューティングに不可欠である。

```
Flow Log の記録レベル:

VPC レベル        → VPC 内の全 ENI のトラフィックを記録
サブネットレベル  → 特定サブネット内の全 ENI のトラフィックを記録
ENI レベル        → 特定の ENI のトラフィックのみ記録

送信先:
  CloudWatch Logs  → リアルタイム分析、メトリクスフィルター
  S3               → 長期保存、Athena でクエリ（推奨）
  Kinesis Firehose → リアルタイム加工、SIEM 連携
```

### コード例 12: VPC Flow Logs の設定

```bash
# CloudWatch Logs への Flow Log 設定
aws ec2 create-flow-logs \
  --resource-type VPC \
  --resource-ids $VPC_ID \
  --traffic-type ALL \
  --log-destination-type cloud-watch-logs \
  --log-group-name /vpc/flow-logs/prod \
  --deliver-logs-permission-arn arn:aws:iam::123456789012:role/flowlogs-role \
  --max-aggregation-interval 60 \
  --tag-specifications 'ResourceType=vpc-flow-log,Tags=[{Key=Name,Value=prod-flow-log}]'

# S3 への Flow Log 設定（推奨: 長期保存・Athena 分析向き）
aws ec2 create-flow-logs \
  --resource-type VPC \
  --resource-ids $VPC_ID \
  --traffic-type ALL \
  --log-destination-type s3 \
  --log-destination arn:aws:s3:::my-flowlogs-bucket/prod/ \
  --log-format '${version} ${account-id} ${interface-id} ${srcaddr} ${dstaddr} ${srcport} ${dstport} ${protocol} ${packets} ${bytes} ${start} ${end} ${action} ${log-status} ${vpc-id} ${subnet-id} ${az-id} ${sublocation-type} ${sublocation-id} ${pkt-srcaddr} ${pkt-dstaddr} ${region} ${pkt-src-aws-service} ${pkt-dst-aws-service} ${flow-direction} ${traffic-path}' \
  --max-aggregation-interval 60

# Flow Logs 用の IAM ロール
cat > flowlogs-trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "vpc-flow-logs.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
  --role-name flowlogs-role \
  --assume-role-policy-document file://flowlogs-trust-policy.json

aws iam put-role-policy \
  --role-name flowlogs-role \
  --policy-name flowlogs-policy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Action": [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ],
        "Effect": "Allow",
        "Resource": "*"
      }
    ]
  }'
```

### コード例 13: Athena で Flow Logs を分析

```sql
-- Athena テーブルの作成（S3 に保存した Flow Logs 用）
CREATE EXTERNAL TABLE IF NOT EXISTS vpc_flow_logs (
  version int,
  account_id string,
  interface_id string,
  srcaddr string,
  dstaddr string,
  srcport int,
  dstport int,
  protocol bigint,
  packets bigint,
  bytes bigint,
  start bigint,
  `end` bigint,
  action string,
  log_status string,
  vpc_id string,
  subnet_id string,
  az_id string,
  sublocation_type string,
  sublocation_id string,
  pkt_srcaddr string,
  pkt_dstaddr string,
  region string,
  pkt_src_aws_service string,
  pkt_dst_aws_service string,
  flow_direction string,
  traffic_path int
)
PARTITIONED BY (dt string)
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ' '
LOCATION 's3://my-flowlogs-bucket/prod/AWSLogs/123456789012/vpcflowlogs/ap-northeast-1/'
TBLPROPERTIES ("skip.header.line.count"="1");

-- 拒否されたトラフィックの分析
SELECT srcaddr, dstaddr, dstport, protocol, action, SUM(packets) as total_packets
FROM vpc_flow_logs
WHERE action = 'REJECT'
  AND dt = '2026/02/15'
GROUP BY srcaddr, dstaddr, dstport, protocol, action
ORDER BY total_packets DESC
LIMIT 20;

-- 特定 IP からのトラフィック量分析
SELECT dstport, protocol, SUM(bytes) as total_bytes, COUNT(*) as flow_count
FROM vpc_flow_logs
WHERE srcaddr = '10.0.11.15'
  AND dt >= '2026/02/01'
GROUP BY dstport, protocol
ORDER BY total_bytes DESC;

-- NAT Gateway 経由のトラフィック量（コスト分析用）
SELECT pkt_dstaddr, dstport, SUM(bytes) as total_bytes
FROM vpc_flow_logs
WHERE interface_id IN (
  SELECT network_interface_id FROM nat_gw_enis
)
AND flow_direction = 'egress'
GROUP BY pkt_dstaddr, dstport
ORDER BY total_bytes DESC
LIMIT 50;
```

---

## 8. AWS CDK による VPC 構築

```typescript
// lib/vpc-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

export class VpcStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // VPC の作成（3層サブネット）
    this.vpc = new ec2.Vpc(this, 'ProdVpc', {
      vpcName: 'prod-vpc',
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
      maxAzs: 3,
      natGateways: 3,  // AZ ごとに 1 つ

      subnetConfiguration: [
        {
          subnetType: ec2.SubnetType.PUBLIC,
          name: 'Public',
          cidrMask: 24,
        },
        {
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          name: 'PrivateApp',
          cidrMask: 20,
        },
        {
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          name: 'PrivateDB',
          cidrMask: 24,
        },
      ],

      // Flow Logs
      flowLogs: {
        's3': {
          destination: ec2.FlowLogDestination.toS3(),
          trafficType: ec2.FlowLogTrafficType.ALL,
        },
      },
    });

    // S3 Gateway Endpoint（無料）
    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    // DynamoDB Gateway Endpoint（無料）
    this.vpc.addGatewayEndpoint('DynamoDBEndpoint', {
      service: ec2.GatewayVpcEndpointAwsService.DYNAMODB,
    });

    // ECR Interface Endpoints
    this.vpc.addInterfaceEndpoint('EcrDockerEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
    });

    this.vpc.addInterfaceEndpoint('EcrApiEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.ECR,
    });

    // CloudWatch Logs Interface Endpoint
    this.vpc.addInterfaceEndpoint('CloudWatchLogsEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
    });

    // セキュリティグループ: ALB
    const albSg = new ec2.SecurityGroup(this, 'AlbSg', {
      vpc: this.vpc,
      description: 'Security group for ALB',
      allowAllOutbound: true,
    });
    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), 'HTTPS');
    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), 'HTTP redirect');

    // セキュリティグループ: App
    const appSg = new ec2.SecurityGroup(this, 'AppSg', {
      vpc: this.vpc,
      description: 'Security group for App tier',
      allowAllOutbound: true,
    });
    appSg.addIngressRule(albSg, ec2.Port.tcp(8080), 'From ALB');

    // セキュリティグループ: DB
    const dbSg = new ec2.SecurityGroup(this, 'DbSg', {
      vpc: this.vpc,
      description: 'Security group for DB tier',
      allowAllOutbound: false,
    });
    dbSg.addIngressRule(appSg, ec2.Port.tcp(3306), 'MySQL from App');
    dbSg.addIngressRule(appSg, ec2.Port.tcp(6379), 'Redis from App');

    // アウトプット
    new cdk.CfnOutput(this, 'VpcId', { value: this.vpc.vpcId });
    new cdk.CfnOutput(this, 'PublicSubnets', {
      value: this.vpc.publicSubnets.map(s => s.subnetId).join(','),
    });
    new cdk.CfnOutput(this, 'PrivateSubnets', {
      value: this.vpc.privateSubnets.map(s => s.subnetId).join(','),
    });
    new cdk.CfnOutput(this, 'IsolatedSubnets', {
      value: this.vpc.isolatedSubnets.map(s => s.subnetId).join(','),
    });
  }
}
```

---

## 9. IPv6 対応

### デュアルスタック VPC の構成

```
デュアルスタック VPC:

  IPv4 CIDR: 10.0.0.0/16 (プライベート)
  IPv6 CIDR: 2600:1f18:xxxx::/56 (AWS 割当パブリック)

  サブネット:
    Public:  10.0.1.0/24 + 2600:1f18:xxxx:0100::/64
    Private: 10.0.11.0/24 + 2600:1f18:xxxx:0b00::/64

  ルーティング:
    Public:  ::/0 → igw-xxx (IPv6 インターネット直接)
    Private: ::/0 → eigw-xxx (Egress-only Internet Gateway)

  Egress-only Internet Gateway:
    - IPv6 のアウトバウンドのみ許可（NAT GW の IPv6 版）
    - インバウンドは拒否
    - 無料
```

### コード例 14: IPv6 対応 VPC の設定

```bash
# VPC に IPv6 CIDR を関連付け
aws ec2 associate-vpc-cidr-block \
  --vpc-id $VPC_ID \
  --amazon-provided-ipv6-cidr-block

# サブネットに IPv6 CIDR を割当
aws ec2 associate-subnet-cidr-block \
  --subnet-id $PUB_1A \
  --ipv6-cidr-block 2600:1f18:xxxx:0100::/64

# Egress-only Internet Gateway の作成
EIGW_ID=$(aws ec2 create-egress-only-internet-gateway \
  --vpc-id $VPC_ID \
  --query 'EgressOnlyInternetGateway.EgressOnlyInternetGatewayId' --output text)

# プライベートサブネットの IPv6 ルート（アウトバウンドのみ）
aws ec2 create-route --route-table-id $PRIV_RT_1A \
  --destination-ipv6-cidr-block ::/0 \
  --egress-only-internet-gateway-id $EIGW_ID

# パブリックサブネットの IPv6 ルート（双方向）
aws ec2 create-route --route-table-id $PUB_RT \
  --destination-ipv6-cidr-block ::/0 \
  --gateway-id $IGW_ID
```

---

## アンチパターン

### 1. 全リソースをパブリックサブネットに配置

**問題**: EC2、RDS、ElastiCache をすべてパブリックサブネットに配置すると、セキュリティグループの設定ミスで内部リソースがインターネットに露出するリスクがある。多層防御の原則に反する。

**対策**: 3層サブネット設計を採用する。パブリックには ALB/NAT GW のみ配置し、アプリケーションとデータベースはプライベートサブネットに配置する。

### 2. CIDR の過小設計

**問題**: VPC を `/24` のような小さい CIDR で作成すると、EKS ノード、Lambda ENI、ElastiCache ノードなど予想外に IP を消費するサービスで IP 枯渇が発生する。VPC の CIDR は後から変更できない。

**対策**: VPC は `/16` で作成し、サブネットは用途に応じて `/20` 〜 `/24` で分割する。将来の拡張用に CIDR 空間の半分は予約しておく。

### 3. シングル AZ 構成

**問題**: コスト削減のため 1 つの AZ にのみリソースを配置すると、AZ 障害時にサービス全体が停止する。AWS の AZ 障害は年に数回発生している。

**対策**: 最低 2 AZ、可能であれば 3 AZ 構成にする。NAT Gateway もマルチ AZ にすることで、単一 AZ の障害がプライベートサブネットのインターネットアクセスに影響しないようにする。

### 4. セキュリティグループの過剰許可

**問題**: 開発の便宜のために `0.0.0.0/0` からの全ポート許可を設定し、本番にそのまま持ち込む。

**対策**: SG のソースには他の SG の ID を指定する。ポートは必要最小限に限定する。AWS Config の `restricted-ssh` や `restricted-common-ports` ルールで自動検出する。

### 5. VPC エンドポイントを使わない NAT Gateway 経由のアクセス

**問題**: S3 や DynamoDB へのアクセスを NAT Gateway 経由で行うと、不要な NAT Gateway 料金（$0.062/GB）が発生する。

**対策**: S3 と DynamoDB は Gateway Endpoint（無料）を使用する。ECR やその他の AWS サービスも Interface Endpoint を検討し、NAT Gateway のデータ処理量を削減する。

---

## FAQ

### Q1: NAT Gateway のコストが高い場合の対策は？

**A**: NAT GW は約 $0.062/時 + データ処理 $0.062/GB で、月額約 $45 + データ転送量です。コスト削減策:
1. **開発環境**: NAT Instance（t4g.nano: 約 $3/月）で代替
2. **VPC エンドポイント**: S3・DynamoDB は Gateway Endpoint（無料）で NAT GW を経由しない
3. **ECR Image Pull**: VPC エンドポイントで NAT GW トラフィックを削減
4. **シングル NAT GW**: 開発環境では AZ ごとではなく1つの NAT GW を共有
5. **Flow Logs 分析**: NAT GW を経由しているトラフィックの内訳を分析し、エンドポイント化可能なものを特定

### Q2: Peering と Transit Gateway はどう使い分けますか？

**A**:
- **VPC Peering**: 2-3 VPC の接続。無料（データ転送料のみ）。1対1接続
- **Transit Gateway**: 多数の VPC/オンプレミス接続。ハブ&スポーク構成。時間課金（約 $0.07/時）
VPC が 3 つ以下なら Peering、4 つ以上や VPN 接続がある場合は Transit Gateway が効率的です。

### Q3: VPC Flow Logs は有効にすべきですか？

**A**: プロダクション環境では必須です。セキュリティインシデント調査、ネットワークトラブルシュート、コンプライアンスで必要です。コスト最適化のため、送信先は S3（CloudWatch Logs より安価）を選び、カスタムフォーマットで必要なフィールドのみ記録してください。

### Q4: AWS Network Firewall は必要ですか？

**A**: 基本的な要件はセキュリティグループと NACL で十分ですが、以下の場合に Network Firewall を検討してください:
- **IDS/IPS が必要**: Suricata ベースのルールで侵入検知・防止
- **ドメインフィルタリング**: 特定のドメインへのアウトバウンドのみ許可
- **TLS インスペクション**: 暗号化されたトラフィックの検査が必要
- **コンプライアンス要件**: PCI DSS や HIPAA で要求される場合

### Q5: セキュリティグループの上限に達した場合はどうしますか？

**A**: デフォルトでは VPC あたり 2,500 SG、SG あたり 60 インバウンドルール + 60 アウトバウンドルールです。対策:
1. **プレフィックスリストの活用**: 複数の CIDR を 1 つのプレフィックスリストにまとめる
2. **SG の整理**: 未使用の SG を削除、類似ルールの SG を統合
3. **Service Quotas で上限引き上げ**: AWS Support から引き上げを申請
4. **NACL の活用**: サブネットレベルのルールを NACL に移行して SG ルールを削減

### Q6: VPC の DNS 設定でハマりやすいポイントは？

**A**: 以下の点に注意してください:
1. **enableDnsSupport**: true にしないと VPC 内の DNS 解決が動作しない
2. **enableDnsHostnames**: true にしないと EC2 インスタンスにパブリック DNS ホスト名が付与されない
3. **DHCP オプションセット**: カスタム DNS サーバーを指定する場合に変更
4. **Route 53 Resolver**: オンプレミスとの DNS 統合に必要（インバウンド/アウトバウンドエンドポイント）
5. **プライベートホストゾーン**: VPC 内部のサービスディスカバリに活用

---

## まとめ

| 項目 | 要点 |
|---|---|
| VPC 設計 | /16 CIDR、3層サブネット（Public/Private App/Private DB）、マルチ AZ |
| サブネット | Public: ALB/NAT、Private App: ECS/EC2、Private DB: RDS/Cache |
| ルーティング | Public -> IGW、Private -> NAT GW、DB -> local のみ |
| セキュリティ | SG でアクセス制御（主）、NACL で追加防御（補助） |
| VPC エンドポイント | S3/DynamoDB は Gateway（無料）、その他は Interface |
| VPC 間接続 | 少数は Peering、多数は Transit Gateway |
| Flow Logs | S3 + Athena でコスト効率の良い分析 |
| コスト注意 | NAT GW が主要コスト要素。VPC エンドポイントで削減 |
| IPv6 | デュアルスタック対応、Egress-only IGW でプライベートアクセス |

## 次に読むべきガイド

- [RDS 基礎](../03-database/00-rds-basics.md) — VPC 内でのデータベース配置
- [ElastiCache](../03-database/02-elasticache.md) — プライベートサブネットでのキャッシュ構築
- [DynamoDB](../03-database/01-dynamodb.md) — VPC エンドポイントでの接続最適化
- [Route 53](./01-route53.md) — VPC のプライベートホストゾーンと DNS 設計

## 参考文献

1. **AWS 公式ドキュメント**: [Amazon VPC ユーザーガイド](https://docs.aws.amazon.com/ja_jp/vpc/latest/userguide/) — VPC の全機能リファレンス
2. **AWS Well-Architected Framework**: [セキュリティの柱](https://docs.aws.amazon.com/ja_jp/wellarchitected/latest/security-pillar/) — ネットワークセキュリティのベストプラクティス
3. **AWS ブログ**: [VPC ベストプラクティス](https://aws.amazon.com/blogs/networking-and-content-delivery/) — 実践的な VPC 設計パターン
4. **AWS re:Invent**: [NET305 - Advanced VPC design and new capabilities](https://www.youtube.com/results?search_query=aws+reinvent+advanced+vpc+design) — 高度な VPC 設計
5. **AWS ドキュメント**: [VPC Flow Logs](https://docs.aws.amazon.com/vpc/latest/userguide/flow-logs.html) — トラフィック分析と監視
