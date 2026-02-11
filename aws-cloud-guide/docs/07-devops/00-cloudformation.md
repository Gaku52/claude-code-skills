# AWS CloudFormation

> AWS リソースをコードで定義・管理する CloudFormation のテンプレート構文、スタック管理、クロススタック参照、ドリフト検出までを体系的に学ぶ。

---

## この章で学ぶこと

1. **テンプレート構文の理解** -- YAML/JSON によるリソース定義、パラメータ、マッピング、条件、組み込み関数を習得する
2. **スタックの管理と運用** -- スタックの作成・更新・削除、変更セット、ネストスタックの設計を理解する
3. **クロススタック参照とドリフト検出** -- 複数スタック間のリソース共有と、実際の構成との差分検出を身につける

---

## 1. CloudFormation の基本概念

### 1.1 Infrastructure as Code (IaC)

```
CloudFormation のワークフロー:

テンプレート (YAML/JSON)
    |
    v
+------------------+
| CloudFormation   |
| サービス         |
+------------------+
    |
    | リソースのプロビジョニング
    |
    v
+------------------+
| スタック         |
| +------+ +-----+|
| | VPC  | | EC2 ||
| +------+ +-----+|
| +------+ +-----+|
| | RDS  | | SG  ||
| +------+ +-----+|
+------------------+
    |
    | 状態追跡・変更管理
    v
+------------------+
| スタックイベント  |
| ドリフト検出      |
+------------------+
```

### 1.2 テンプレート構造の全体像

```
テンプレートのセクション:

+-------------------------------------------+
| AWSTemplateFormatVersion (バージョン)       |
+-------------------------------------------+
| Description (説明)                         |
+-------------------------------------------+
| Metadata (メタデータ)                      |
+-------------------------------------------+
| Parameters (パラメータ -- 入力値)           |
+-------------------------------------------+
| Mappings (マッピング -- 静的な定数テーブル)  |
+-------------------------------------------+
| Conditions (条件 -- リソース作成の制御)     |
+-------------------------------------------+
| Resources (リソース -- 必須セクション)      |
+-------------------------------------------+
| Outputs (出力 -- エクスポート値)            |
+-------------------------------------------+
```

---

## 2. テンプレート構文

### 2.1 基本的なテンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'Web アプリケーション基盤テンプレート'

Parameters:
  EnvironmentName:
    Type: String
    Default: dev
    AllowedValues: [dev, stg, prod]
    Description: デプロイ先環境名

  InstanceType:
    Type: String
    Default: t3.micro
    AllowedValues: [t3.micro, t3.small, t3.medium]
    Description: EC2 インスタンスタイプ

  VpcCidr:
    Type: String
    Default: '10.0.0.0/16'
    AllowedPattern: '(\d{1,3}\.){3}\d{1,3}/\d{1,2}'
    Description: VPC の CIDR ブロック

Mappings:
  RegionAMI:
    ap-northeast-1:
      HVM64: ami-0abcdef1234567890
    us-east-1:
      HVM64: ami-0fedcba9876543210

Conditions:
  IsProduction: !Equals [!Ref EnvironmentName, prod]
  CreateReadReplica: !Equals [!Ref EnvironmentName, prod]

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCidr
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-vpc'

  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: !Select [0, !Cidr [!Ref VpcCidr, 4, 8]]
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-1'

  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      InstanceType: !If [IsProduction, t3.medium, !Ref InstanceType]
      ImageId: !FindInMap [RegionAMI, !Ref 'AWS::Region', HVM64]
      SubnetId: !Ref PublicSubnet1
      SecurityGroupIds:
        - !Ref WebSecurityGroup
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-web-server'

  WebSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Web server security group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

Outputs:
  VpcId:
    Description: VPC ID
    Value: !Ref VPC
    Export:
      Name: !Sub '${EnvironmentName}-VpcId'

  WebServerPublicIP:
    Description: Web サーバーのパブリック IP
    Value: !GetAtt WebServer.PublicIp
```

### 2.2 主要な組み込み関数

| 関数 | 用途 | 使用例 |
|------|------|--------|
| `!Ref` | パラメータ/リソースの参照 | `!Ref VPC` |
| `!Sub` | 文字列内の変数展開 | `!Sub '${Env}-vpc'` |
| `!GetAtt` | リソースの属性取得 | `!GetAtt EC2.PublicIp` |
| `!Join` | 文字列結合 | `!Join ['-', [a, b, c]]` |
| `!Select` | リストから要素選択 | `!Select [0, !GetAZs '']` |
| `!Split` | 文字列分割 | `!Split [',', 'a,b,c']` |
| `!If` | 条件分岐 | `!If [IsProd, t3.large, t3.micro]` |
| `!FindInMap` | マッピング検索 | `!FindInMap [Map, Key1, Key2]` |
| `!ImportValue` | 別スタックの出力参照 | `!ImportValue 'vpc-id'` |
| `!Cidr` | CIDR 分割 | `!Cidr [!Ref VpcCidr, 4, 8]` |
| `!GetAZs` | AZ リスト取得 | `!GetAZs ''` |

### 2.3 組み込み関数の活用例

```yaml
# !Sub の高度な使い方
BucketPolicy:
  Type: AWS::S3::BucketPolicy
  Properties:
    Bucket: !Ref MyBucket
    PolicyDocument:
      Statement:
        - Effect: Allow
          Principal:
            AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:role/${RoleName}'
          Action: 's3:GetObject'
          Resource: !Sub 'arn:aws:s3:::${MyBucket}/*'

# !Cidr による自動サブネット計算
Subnets:
  # 10.0.0.0/16 から 4つの /24 サブネットを自動計算
  # → 10.0.0.0/24, 10.0.1.0/24, 10.0.2.0/24, 10.0.3.0/24
  - !Select [0, !Cidr [!Ref VpcCidr, 4, 8]]
  - !Select [1, !Cidr [!Ref VpcCidr, 4, 8]]
  - !Select [2, !Cidr [!Ref VpcCidr, 4, 8]]
  - !Select [3, !Cidr [!Ref VpcCidr, 4, 8]]
```

---

## 3. スタック管理

### 3.1 スタックの CRUD 操作

```bash
# スタックの作成
aws cloudformation create-stack \
  --stack-name my-web-stack \
  --template-body file://template.yaml \
  --parameters \
    ParameterKey=EnvironmentName,ParameterValue=prod \
    ParameterKey=InstanceType,ParameterValue=t3.small \
  --capabilities CAPABILITY_NAMED_IAM \
  --tags Key=Project,Value=MyApp

# 変更セットの作成 (更新前にプレビュー)
aws cloudformation create-change-set \
  --stack-name my-web-stack \
  --change-set-name update-instance-type \
  --template-body file://template-v2.yaml \
  --parameters \
    ParameterKey=InstanceType,ParameterValue=t3.medium

# 変更セットの確認
aws cloudformation describe-change-set \
  --stack-name my-web-stack \
  --change-set-name update-instance-type

# 変更セットの実行
aws cloudformation execute-change-set \
  --stack-name my-web-stack \
  --change-set-name update-instance-type

# スタックの削除
aws cloudformation delete-stack \
  --stack-name my-web-stack
```

### 3.2 スタック更新時のリソース影響

```
更新の影響レベル:

更新なし (No Interruption):
  タグ変更、セキュリティグループルール追加
  → サービス影響なし

一時中断 (Some Interruption):
  EC2 インスタンスタイプ変更
  → 一時的にサービス停止

置換 (Replacement):
  RDS エンジン変更、EC2 AMI 変更
  → 新リソースを作成し旧リソースを削除
  ⚠ データ損失の可能性あり！

変更セットで事前に影響を確認することが必須
```

### 3.3 スタックポリシー

```json
{
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "Update:*",
      "Principal": "*",
      "Resource": "*"
    },
    {
      "Effect": "Deny",
      "Action": "Update:Replace",
      "Principal": "*",
      "Resource": "LogicalResourceId/Database",
      "Condition": {
        "StringEquals": {
          "ResourceType": ["AWS::RDS::DBInstance"]
        }
      }
    }
  ]
}
```

---

## 4. クロススタック参照

### 4.1 Export/Import パターン

```
クロススタック参照:

ネットワークスタック              アプリケーションスタック
+------------------+            +------------------+
| VPC              |            | EC2 Instance     |
| Subnets          |  Export    | SubnetId:        |
| Security Groups  | ========> |   !ImportValue    |
|                  |  VpcId    |   'prod-VpcId'   |
| Outputs:         |  SubnetIds|                  |
|   Export:        |           | Security Group:   |
|   'prod-VpcId'   |           |   !ImportValue    |
|   'prod-SubnetIds'|          |   'prod-SG'      |
+------------------+            +------------------+
```

```yaml
# network-stack.yaml (エクスポート側)
Outputs:
  VpcId:
    Value: !Ref VPC
    Export:
      Name: !Sub '${EnvironmentName}-VpcId'

  PublicSubnetIds:
    Value: !Join [',', [!Ref PublicSubnet1, !Ref PublicSubnet2]]
    Export:
      Name: !Sub '${EnvironmentName}-PublicSubnetIds'

# app-stack.yaml (インポート側)
Resources:
  WebServer:
    Type: AWS::EC2::Instance
    Properties:
      SubnetId: !Select
        - 0
        - !Split [',', !ImportValue 'prod-PublicSubnetIds']
      SecurityGroupIds:
        - !ImportValue 'prod-WebSG'
```

### 4.2 ネストスタック

```yaml
# parent-stack.yaml
Resources:
  NetworkStack:
    Type: AWS::CloudFormation::Stack
    Properties:
      TemplateURL: https://s3.amazonaws.com/my-bucket/network.yaml
      Parameters:
        EnvironmentName: !Ref EnvironmentName
        VpcCidr: !Ref VpcCidr

  DatabaseStack:
    Type: AWS::CloudFormation::Stack
    DependsOn: NetworkStack
    Properties:
      TemplateURL: https://s3.amazonaws.com/my-bucket/database.yaml
      Parameters:
        VpcId: !GetAtt NetworkStack.Outputs.VpcId
        SubnetIds: !GetAtt NetworkStack.Outputs.PrivateSubnetIds

  ApplicationStack:
    Type: AWS::CloudFormation::Stack
    DependsOn: [NetworkStack, DatabaseStack]
    Properties:
      TemplateURL: https://s3.amazonaws.com/my-bucket/application.yaml
      Parameters:
        VpcId: !GetAtt NetworkStack.Outputs.VpcId
        DatabaseEndpoint: !GetAtt DatabaseStack.Outputs.Endpoint
```

---

## 5. ドリフト検出

### 5.1 ドリフトとは

```
ドリフト検出の仕組み:

CloudFormation テンプレート           実際のリソース状態
+--------------------+              +--------------------+
| SG: port 80, 443   |              | SG: port 80, 443,  |
|                    |   !=          |     8080           |
|                    |              | (手動で追加された)   |
+--------------------+              +--------------------+

ドリフト検出結果:
  - リソース: WebSecurityGroup
  - ドリフトステータス: MODIFIED
  - 差分: SecurityGroupIngress に port 8080 が追加
```

### 5.2 ドリフト検出の実行

```bash
# ドリフト検出の開始
aws cloudformation detect-stack-drift \
  --stack-name my-web-stack

# ドリフト検出結果の確認
aws cloudformation describe-stack-drift-detection-status \
  --stack-drift-detection-id <detection-id>

# リソースごとのドリフト詳細
aws cloudformation describe-stack-resource-drifts \
  --stack-name my-web-stack \
  --stack-resource-drift-status-filters MODIFIED DELETED
```

| ドリフトステータス | 意味 |
|------------------|------|
| IN_SYNC | テンプレートと一致 |
| MODIFIED | プロパティが変更されている |
| DELETED | リソースが削除されている |
| NOT_CHECKED | 未チェック |

---

## 6. アンチパターン

### 6.1 巨大な単一テンプレート

```
[悪い例]
1つのテンプレートに全リソース (VPC, EC2, RDS, Lambda, IAM...) を定義
→ 500行超のテンプレート
→ 更新時のリスクが高い、チーム分業が困難

[良い例]
レイヤー別にスタックを分割:
  network-stack.yaml   → VPC, Subnet, NAT GW
  security-stack.yaml  → IAM, SG, KMS
  database-stack.yaml  → RDS, ElastiCache
  app-stack.yaml       → EC2, ECS, Lambda
  monitoring-stack.yaml → CloudWatch, SNS

各スタック間は Export/Import で連携
```

### 7.2 DeletionPolicy 未設定でのデータベース運用

**問題点**: スタック削除時にデータベースも一緒に削除され、データが失われる。

**改善**: RDS や DynamoDB などのステートフルリソースには `DeletionPolicy: Snapshot` または `DeletionPolicy: Retain` を設定する。

```yaml
Database:
  Type: AWS::RDS::DBInstance
  DeletionPolicy: Snapshot
  UpdateReplacePolicy: Snapshot
  Properties:
    # ...
```

---

## 7. FAQ

### Q1. CloudFormation と Terraform のどちらを使うべきですか？

AWS のみを利用する場合は CloudFormation が AWS サービスとの統合が深く、変更セットやドリフト検出など運用機能も充実している。マルチクラウドやオンプレミスも管理する場合は Terraform が適している。組織の既存スキルセットも重要な判断要素である。

### Q2. テンプレートの最大サイズは？

テンプレート本文は 51,200 バイト(約50KB)が上限。S3 に配置したテンプレートを参照する場合は 460,800 バイト(約450KB)まで拡大される。大規模な構成ではネストスタックで分割するか、CDK の利用を検討する。

### Q3. スタック更新に失敗した場合はどうなりますか？

デフォルトでは自動ロールバックが実行され、更新前の状態に戻る。`--disable-rollback` オプションを使うとロールバックを無効化でき、失敗原因の調査がしやすくなるが、スタックは UPDATE_FAILED 状態のままになる。本番環境では自動ロールバックを有効にしておくべきである。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| テンプレート | YAML/JSON でリソースを宣言的に定義 |
| パラメータ | 環境ごとの設定値を外部化 |
| 組み込み関数 | !Ref, !Sub, !GetAtt, !If などで動的な値を構成 |
| 変更セット | 更新前に影響範囲をプレビュー |
| クロススタック参照 | Export/Import でスタック間の値を共有 |
| ドリフト検出 | テンプレートと実際の構成の差分を検出 |

---

## 次に読むべきガイド

- [AWS CDK](./01-cdk.md) -- プログラミング言語でインフラを定義
- [CodePipeline](./02-codepipeline.md) -- CloudFormation を CI/CD に統合
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- CloudFormation で使う IAM の設計

---

## 参考文献

1. AWS 公式ドキュメント「AWS CloudFormation ユーザーガイド」 https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/
2. AWS 公式「CloudFormation ベストプラクティス」 https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/best-practices.html
3. AWS 公式「リソースおよびプロパティリファレンス」 https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html
