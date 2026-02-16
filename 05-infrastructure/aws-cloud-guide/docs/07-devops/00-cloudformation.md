# AWS CloudFormation

> AWS リソースをコードで定義・管理する CloudFormation のテンプレート構文、スタック管理、クロススタック参照、ドリフト検出までを体系的に学ぶ。カスタムリソース、マクロ、スタックセット、CI/CD 統合、トラブルシューティングまで含めた実践的な運用知識を網羅する。

---

## この章で学ぶこと

1. **テンプレート構文の理解** -- YAML/JSON によるリソース定義、パラメータ、マッピング、条件、組み込み関数を習得する
2. **スタックの管理と運用** -- スタックの作成・更新・削除、変更セット、ネストスタックの設計を理解する
3. **クロススタック参照とドリフト検出** -- 複数スタック間のリソース共有と、実際の構成との差分検出を身につける
4. **カスタムリソースとマクロ** -- Lambda ベースのカスタムリソースやテンプレートマクロで CloudFormation を拡張する方法を学ぶ
5. **スタックセットとマルチアカウント管理** -- AWS Organizations と連携した大規模デプロイメントを理解する
6. **CI/CD 統合とトラブルシューティング** -- CloudFormation を CI/CD パイプラインに組み込み、障害対応のスキルを身につける

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
| Rules (ルール -- パラメータ検証)             |
+-------------------------------------------+
| Mappings (マッピング -- 静的な定数テーブル)  |
+-------------------------------------------+
| Conditions (条件 -- リソース作成の制御)     |
+-------------------------------------------+
| Transform (変換 -- マクロの適用)            |
+-------------------------------------------+
| Resources (リソース -- 必須セクション)      |
+-------------------------------------------+
| Outputs (出力 -- エクスポート値)            |
+-------------------------------------------+
```

### 1.3 CloudFormation vs 他の IaC ツール

| 特性 | CloudFormation | Terraform | CDK | Pulumi |
|------|---------------|-----------|-----|--------|
| 提供元 | AWS | HashiCorp | AWS | Pulumi |
| 対応クラウド | AWS のみ | マルチクラウド | AWS のみ | マルチクラウド |
| 言語 | YAML/JSON | HCL | TypeScript/Python等 | TypeScript/Python等 |
| 状態管理 | AWS マネージド | tfstate ファイル | CloudFormation | Pulumi Cloud |
| 変更プレビュー | 変更セット | plan | diff | preview |
| ドリフト検出 | あり | あり | あり (CFn経由) | あり |
| 費用 | 無料 | 無料/有料 | 無料 | 無料/有料 |
| 学習コスト | 低〜中 | 中 | 中〜高 | 中〜高 |
| エコシステム | AWS 密連携 | 非常に広い | AWS 密連携 | 広い |

```
選択の指針:

AWS のみ + YAML 派                → CloudFormation
AWS のみ + プログラミング言語派    → CDK
マルチクラウド + 宣言的            → Terraform
マルチクラウド + プログラミング派   → Pulumi
既存 CFn 資産あり                 → CloudFormation または CDK
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
| `!Base64` | Base64 エンコード | `!Base64 !Sub 'script'` |
| `!Transform` | マクロの適用 | `!Transform {Name: macro}` |

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

### 2.4 パラメータの高度な設定

```yaml
Parameters:
  # SSM Parameter Store からの動的参照
  LatestAmiId:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>
    Default: /aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2
    Description: 最新の Amazon Linux 2 AMI ID

  # 正規表現によるバリデーション
  ProjectName:
    Type: String
    MinLength: 3
    MaxLength: 20
    AllowedPattern: '[a-z][a-z0-9-]*'
    ConstraintDescription: 小文字英数字とハイフンのみ。先頭は英字。

  # 複数選択可能なパラメータ
  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: デプロイ先サブネット

  # NoEcho でパスワードを隠す
  DatabasePassword:
    Type: String
    NoEcho: true
    MinLength: 8
    MaxLength: 128
    AllowedPattern: '[a-zA-Z0-9!@#$%^&*()_+-=]*'
    Description: RDS マスターパスワード

  # Secrets Manager からの動的参照
  DatabaseCredentials:
    Type: String
    Default: '{{resolve:secretsmanager:prod/db/credentials:SecretString:password}}'
```

### 2.5 Rules セクション (パラメータ検証)

```yaml
Rules:
  # 本番環境では t3.micro を使用禁止
  ProdInstanceTypeRule:
    RuleCondition: !Equals [!Ref EnvironmentName, prod]
    Assertions:
      - Assert: !Not [!Equals [!Ref InstanceType, t3.micro]]
        AssertDescription: 本番環境では t3.micro は使用できません

  # マルチ AZ は本番環境でのみ必須
  MultiAZRule:
    RuleCondition: !Equals [!Ref EnvironmentName, prod]
    Assertions:
      - Assert: !Equals [!Ref MultiAZDatabase, true]
        AssertDescription: 本番環境ではマルチ AZ を有効にしてください
```

### 2.6 Metadata セクション

```yaml
Metadata:
  # コンソールでのパラメータグループ化
  AWS::CloudFormation::Interface:
    ParameterGroups:
      - Label:
          default: ネットワーク設定
        Parameters:
          - VpcCidr
          - SubnetIds
      - Label:
          default: コンピューティング設定
        Parameters:
          - InstanceType
          - KeyPairName
      - Label:
          default: データベース設定
        Parameters:
          - DatabasePassword
          - MultiAZDatabase
    ParameterLabels:
      VpcCidr:
        default: VPC CIDR ブロック
      InstanceType:
        default: EC2 インスタンスタイプ
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

# スタック作成完了まで待機
aws cloudformation wait stack-create-complete \
  --stack-name my-web-stack

# スタックイベントの確認
aws cloudformation describe-stack-events \
  --stack-name my-web-stack \
  --query 'StackEvents[0:10].{Time:Timestamp,Status:ResourceStatus,Type:ResourceType,Reason:ResourceStatusReason}' \
  --output table
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

```bash
# スタックポリシーの設定
aws cloudformation set-stack-policy \
  --stack-name my-web-stack \
  --stack-policy-body file://stack-policy.json

# スタックポリシーの確認
aws cloudformation get-stack-policy \
  --stack-name my-web-stack
```

### 3.4 ロールバック設定

```bash
# ロールバック設定付きでスタック作成
aws cloudformation create-stack \
  --stack-name my-web-stack \
  --template-body file://template.yaml \
  --rollback-configuration \
    RollbackTriggers='[{Arn=arn:aws:cloudwatch:ap-northeast-1:123456789012:alarm:HighErrorRate,Type=AWS::CloudWatch::Alarm}]',\
    MonitoringTimeInMinutes=10

# ロールバック無効化 (デバッグ時)
aws cloudformation update-stack \
  --stack-name my-web-stack \
  --template-body file://template-v2.yaml \
  --disable-rollback
```

### 3.5 スタックのインポート

```bash
# 既存リソースをスタックにインポート
# 1. インポートテンプレートを準備 (DeletionPolicy: Retain 必須)
# 2. 変更セットを作成
aws cloudformation create-change-set \
  --stack-name my-web-stack \
  --change-set-name import-existing-resources \
  --change-set-type IMPORT \
  --template-body file://import-template.yaml \
  --resources-to-import '[
    {
      "ResourceType": "AWS::EC2::SecurityGroup",
      "LogicalResourceId": "WebSecurityGroup",
      "ResourceIdentifier": {"GroupId": "sg-12345678"}
    }
  ]'

# 変更セットの確認と実行
aws cloudformation describe-change-set \
  --stack-name my-web-stack \
  --change-set-name import-existing-resources

aws cloudformation execute-change-set \
  --stack-name my-web-stack \
  --change-set-name import-existing-resources
```

```yaml
# インポート用テンプレート (DeletionPolicy: Retain が必須)
Resources:
  WebSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    DeletionPolicy: Retain
    Properties:
      GroupDescription: Imported security group
      VpcId: vpc-12345678
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

### 4.3 Export/Import vs ネストスタック

```
Export/Import:
  ✓ スタック間の疎結合
  ✓ 個別にデプロイ・更新可能
  ✓ 異なるチームが個別管理
  ✗ Export 値変更時にインポート側への影響
  ✗ 削除順序の管理が必要

ネストスタック:
  ✓ 親スタックで一括管理
  ✓ パラメータの受け渡しが明示的
  ✓ 一括デプロイ・削除
  ✗ 密結合
  ✗ 更新時の影響範囲が広い
  ✗ テンプレートを S3 に配置する必要

推奨パターン:
  レイヤー間 (Network ↔ App) → Export/Import
  同一レイヤー内の分割       → ネストスタック
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

# 特定リソースのドリフト検出
aws cloudformation detect-stack-resource-drift \
  --stack-name my-web-stack \
  --logical-resource-id WebSecurityGroup
```

| ドリフトステータス | 意味 |
|------------------|------|
| IN_SYNC | テンプレートと一致 |
| MODIFIED | プロパティが変更されている |
| DELETED | リソースが削除されている |
| NOT_CHECKED | 未チェック |

### 5.3 ドリフト検出の自動化

```yaml
# EventBridge + Lambda でドリフトを定期チェック
Resources:
  DriftCheckSchedule:
    Type: AWS::Events::Rule
    Properties:
      Description: 日次ドリフト検出
      ScheduleExpression: 'cron(0 9 * * ? *)'
      State: ENABLED
      Targets:
        - Arn: !GetAtt DriftCheckFunction.Arn
          Id: DriftCheckTarget

  DriftCheckFunction:
    Type: AWS::Lambda::Function
    Properties:
      Runtime: python3.12
      Handler: index.handler
      Role: !GetAtt DriftCheckRole.Arn
      Code:
        ZipFile: |
          import boto3
          import json

          cfn = boto3.client('cloudformation')
          sns = boto3.client('sns')

          def handler(event, context):
              # 全スタックのドリフト検出
              stacks = cfn.list_stacks(
                  StackStatusFilter=['CREATE_COMPLETE', 'UPDATE_COMPLETE']
              )

              results = []
              for stack in stacks['StackSummaries']:
                  stack_name = stack['StackName']
                  try:
                      response = cfn.detect_stack_drift(
                          StackName=stack_name
                      )
                      results.append({
                          'stack': stack_name,
                          'detection_id': response['StackDriftDetectionId']
                      })
                  except Exception as e:
                      print(f"Error checking {stack_name}: {e}")

              return {'statusCode': 200, 'results': results}
```

---

## 6. カスタムリソース

### 6.1 Lambda ベースのカスタムリソース

```yaml
# カスタムリソースの定義
Resources:
  # Lambda 関数
  CustomResourceFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub '${AWS::StackName}-custom-resource'
      Runtime: python3.12
      Handler: index.handler
      Timeout: 300
      Role: !GetAtt CustomResourceRole.Arn
      Code:
        ZipFile: |
          import json
          import urllib.request
          import boto3

          def handler(event, context):
              response_data = {}
              physical_resource_id = event.get('PhysicalResourceId', 'custom-resource')

              try:
                  if event['RequestType'] == 'Create':
                      # リソース作成ロジック
                      # 例: S3 バケットにデフォルトファイルをアップロード
                      s3 = boto3.client('s3')
                      bucket_name = event['ResourceProperties']['BucketName']
                      s3.put_object(
                          Bucket=bucket_name,
                          Key='config/default.json',
                          Body=json.dumps({'version': '1.0'})
                      )
                      physical_resource_id = f"{bucket_name}-init"
                      response_data['Message'] = 'Created successfully'

                  elif event['RequestType'] == 'Update':
                      # リソース更新ロジック
                      response_data['Message'] = 'Updated successfully'

                  elif event['RequestType'] == 'Delete':
                      # リソース削除ロジック (クリーンアップ)
                      s3 = boto3.client('s3')
                      bucket_name = event['ResourceProperties']['BucketName']
                      # バケット内のオブジェクトを削除
                      objects = s3.list_objects_v2(Bucket=bucket_name)
                      if 'Contents' in objects:
                          delete_objects = [{'Key': obj['Key']} for obj in objects['Contents']]
                          s3.delete_objects(
                              Bucket=bucket_name,
                              Delete={'Objects': delete_objects}
                          )
                      response_data['Message'] = 'Deleted successfully'

                  send_response(event, context, 'SUCCESS', response_data, physical_resource_id)

              except Exception as e:
                  print(f"Error: {e}")
                  send_response(event, context, 'FAILED', {'Error': str(e)}, physical_resource_id)

          def send_response(event, context, status, data, physical_resource_id):
              response_body = json.dumps({
                  'Status': status,
                  'Reason': f"See CloudWatch Log Stream: {context.log_stream_name}",
                  'PhysicalResourceId': physical_resource_id,
                  'StackId': event['StackId'],
                  'RequestId': event['RequestId'],
                  'LogicalResourceId': event['LogicalResourceId'],
                  'Data': data
              })

              req = urllib.request.Request(
                  event['ResponseURL'],
                  data=response_body.encode('utf-8'),
                  headers={'Content-Type': 'application/json'},
                  method='PUT'
              )
              urllib.request.urlopen(req)

  # カスタムリソースの呼び出し
  BucketInitializer:
    Type: Custom::BucketInit
    DependsOn: MyBucket
    Properties:
      ServiceToken: !GetAtt CustomResourceFunction.Arn
      BucketName: !Ref MyBucket

  MyBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: !Sub '${AWS::StackName}-data-bucket'
```

### 6.2 cfn-response モジュール

```yaml
# cfn-response を使ったシンプルなカスタムリソース
CustomResourceFunction:
  Type: AWS::Lambda::Function
  Properties:
    Runtime: python3.12
    Handler: index.handler
    Role: !GetAtt CustomResourceRole.Arn
    Code:
      ZipFile: |
        import cfnresponse
        import boto3

        def handler(event, context):
            try:
                if event['RequestType'] == 'Create':
                    # 処理
                    response_data = {'Result': 'Success'}
                    cfnresponse.send(event, context, cfnresponse.SUCCESS, response_data)
                elif event['RequestType'] == 'Delete':
                    cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
                else:
                    cfnresponse.send(event, context, cfnresponse.SUCCESS, {})
            except Exception as e:
                cfnresponse.send(event, context, cfnresponse.FAILED, {'Error': str(e)})
```

### 6.3 AWS::CloudFormation::CustomResource vs Custom::*

```
Custom::BucketInit (推奨):
  - Type 名でリソースの目的が明確
  - 複数のカスタムリソースを区別しやすい
  - CloudFormation コンソールで見やすい

AWS::CloudFormation::CustomResource:
  - 公式のリソースタイプ名
  - Custom:: と機能は同一
  - やや冗長
```

---

## 7. CloudFormation マクロ

### 7.1 マクロの仕組み

```
マクロの処理フロー:

テンプレート → CloudFormation → マクロ (Lambda) → 変換後テンプレート → リソース作成
                    |                                     ↑
                    | Transform セクション                 |
                    +-------------------------------------+

組み込みマクロ:
  AWS::Include     → 外部テンプレート断片のインクルード
  AWS::Serverless  → SAM テンプレートの変換
```

```yaml
# AWS::Include マクロの使用例
Resources:
  MyResource:
    Fn::Transform:
      Name: AWS::Include
      Parameters:
        Location: s3://my-bucket/resource-snippet.yaml

# SAM テンプレート (AWS::Serverless マクロ)
Transform: AWS::Serverless-2016-10-31

Resources:
  MyFunction:
    Type: AWS::Serverless::Function
    Properties:
      Runtime: python3.12
      Handler: app.handler
      Events:
        Api:
          Type: Api
          Properties:
            Path: /hello
            Method: get
```

### 7.2 カスタムマクロの作成

```yaml
# マクロ登録テンプレート
Resources:
  MacroFunction:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: EnvVarInjector
      Runtime: python3.12
      Handler: index.handler
      Role: !GetAtt MacroRole.Arn
      Code:
        ZipFile: |
          import copy

          def handler(event, context):
              """全 Lambda 関数に共通環境変数を自動注入するマクロ"""
              fragment = event['fragment']
              common_env = event['templateParameterValues'].get('CommonEnvVars', {})

              for resource_name, resource in fragment.get('Resources', {}).items():
                  if resource['Type'] == 'AWS::Lambda::Function':
                      props = resource.get('Properties', {})
                      env = props.get('Environment', {})
                      variables = env.get('Variables', {})
                      variables.update({
                          'ENVIRONMENT': fragment.get('Parameters', {}).get('EnvironmentName', {}).get('Default', 'dev'),
                          'REGION': {'Ref': 'AWS::Region'},
                          'STACK_NAME': {'Ref': 'AWS::StackName'}
                      })
                      env['Variables'] = variables
                      props['Environment'] = env
                      resource['Properties'] = props

              return {
                  'requestId': event['requestId'],
                  'status': 'success',
                  'fragment': fragment
              }

  EnvVarInjectorMacro:
    Type: AWS::CloudFormation::Macro
    Properties:
      Name: EnvVarInjector
      FunctionName: !GetAtt MacroFunction.Arn
```

```yaml
# マクロの使用
Transform: EnvVarInjector

Resources:
  MyFunction:
    Type: AWS::Lambda::Function
    Properties:
      Runtime: python3.12
      Handler: index.handler
      # Environment は自動で注入される
```

---

## 8. スタックセット (マルチアカウント・マルチリージョン)

### 8.1 スタックセットの概念

```
スタックセット:

管理アカウント
+---------------------+
| StackSet            |
| (テンプレート定義)   |
+---------------------+
        |
        | デプロイ
        |
        +--→ アカウント A / ap-northeast-1  → Stack Instance
        |
        +--→ アカウント A / us-east-1       → Stack Instance
        |
        +--→ アカウント B / ap-northeast-1  → Stack Instance
        |
        +--→ アカウント B / us-east-1       → Stack Instance

デプロイモデル:
  Self-managed: 手動で IAM ロールを設定
  Service-managed: AWS Organizations と自動連携
```

### 8.2 スタックセットの作成と管理

```bash
# スタックセットの作成 (Service-managed)
aws cloudformation create-stack-set \
  --stack-set-name security-baseline \
  --template-body file://security-baseline.yaml \
  --permission-model SERVICE_MANAGED \
  --auto-deployment Enabled=true,RetainStacksOnAccountRemoval=false \
  --capabilities CAPABILITY_NAMED_IAM

# スタックインスタンスのデプロイ (OU 指定)
aws cloudformation create-stack-instances \
  --stack-set-name security-baseline \
  --deployment-targets OrganizationalUnitIds='["ou-xxxx-yyyyyyy"]' \
  --regions ap-northeast-1 us-east-1 \
  --operation-preferences \
    FailureTolerancePercentage=10,\
    MaxConcurrentPercentage=25,\
    RegionConcurrencyType=PARALLEL

# スタックセットの更新
aws cloudformation update-stack-set \
  --stack-set-name security-baseline \
  --template-body file://security-baseline-v2.yaml

# スタックインスタンスの状態確認
aws cloudformation list-stack-instances \
  --stack-set-name security-baseline \
  --query 'Summaries[].{Account:Account,Region:Region,Status:Status}' \
  --output table
```

### 8.3 セキュリティベースラインテンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'セキュリティベースライン (全アカウント共通)'

Resources:
  # CloudTrail 有効化
  CloudTrail:
    Type: AWS::CloudTrail::Trail
    Properties:
      TrailName: organization-trail
      IsLogging: true
      IsMultiRegionTrail: true
      EnableLogFileValidation: true
      S3BucketName: !Sub 'cloudtrail-${AWS::AccountId}'

  # GuardDuty 有効化
  GuardDutyDetector:
    Type: AWS::GuardDuty::Detector
    Properties:
      Enable: true
      DataSources:
        S3Logs:
          Enable: true
        Kubernetes:
          AuditLogs:
            Enable: true

  # Config 有効化
  ConfigRecorder:
    Type: AWS::Config::ConfigurationRecorder
    Properties:
      Name: default
      RoleARN: !GetAtt ConfigRole.Arn
      RecordingGroup:
        AllSupported: true
        IncludeGlobalResourceTypes: true

  # デフォルト VPC の削除を防止する Config ルール
  RestrictedSSH:
    Type: AWS::Config::ConfigRule
    DependsOn: ConfigRecorder
    Properties:
      ConfigRuleName: restricted-ssh
      Source:
        Owner: AWS
        SourceIdentifier: INCOMING_SSH_DISABLED

  # パスワードポリシー
  PasswordPolicy:
    Type: Custom::PasswordPolicy
    Properties:
      ServiceToken: !GetAtt PasswordPolicyFunction.Arn
      MinimumPasswordLength: 14
      RequireSymbols: true
      RequireNumbers: true
      RequireUppercaseCharacters: true
      RequireLowercaseCharacters: true
      MaxPasswordAge: 90
      PasswordReusePrevention: 12

  # S3 パブリックアクセスブロック (アカウントレベル)
  S3PublicAccessBlock:
    Type: AWS::S3::AccountPublicAccessBlock
    Properties:
      BlockPublicAcls: true
      BlockPublicPolicy: true
      IgnorePublicAcls: true
      RestrictPublicBuckets: true
```

---

## 9. CI/CD 統合

### 9.1 CodePipeline + CloudFormation

```yaml
# CI/CD パイプラインテンプレート
Resources:
  Pipeline:
    Type: AWS::CodePipeline::Pipeline
    Properties:
      Name: infra-pipeline
      RoleArn: !GetAtt PipelineRole.Arn
      ArtifactStore:
        Type: S3
        Location: !Ref ArtifactBucket
      Stages:
        # ソースステージ
        - Name: Source
          Actions:
            - Name: SourceAction
              ActionTypeId:
                Category: Source
                Owner: AWS
                Provider: CodeStarSourceConnection
                Version: "1"
              Configuration:
                ConnectionArn: !Ref CodeStarConnection
                FullRepositoryId: my-org/infra-repo
                BranchName: main
              OutputArtifacts:
                - Name: SourceOutput

        # テストステージ
        - Name: Test
          Actions:
            - Name: CFnLint
              ActionTypeId:
                Category: Build
                Owner: AWS
                Provider: CodeBuild
                Version: "1"
              Configuration:
                ProjectName: !Ref LintProject
              InputArtifacts:
                - Name: SourceOutput

        # ステージング環境
        - Name: Staging
          Actions:
            - Name: CreateChangeSet
              ActionTypeId:
                Category: Deploy
                Owner: AWS
                Provider: CloudFormation
                Version: "1"
              Configuration:
                ActionMode: CHANGE_SET_REPLACE
                StackName: staging-stack
                ChangeSetName: staging-changeset
                TemplatePath: SourceOutput::template.yaml
                TemplateConfiguration: SourceOutput::config/staging.json
                Capabilities: CAPABILITY_NAMED_IAM
                RoleArn: !GetAtt CloudFormationRole.Arn
              InputArtifacts:
                - Name: SourceOutput
              RunOrder: 1

            - Name: ExecuteChangeSet
              ActionTypeId:
                Category: Deploy
                Owner: AWS
                Provider: CloudFormation
                Version: "1"
              Configuration:
                ActionMode: CHANGE_SET_EXECUTE
                StackName: staging-stack
                ChangeSetName: staging-changeset
              RunOrder: 2

        # 承認ステージ
        - Name: Approval
          Actions:
            - Name: ManualApproval
              ActionTypeId:
                Category: Approval
                Owner: AWS
                Provider: Manual
                Version: "1"
              Configuration:
                NotificationArn: !Ref ApprovalTopic
                CustomData: "本番環境へのデプロイを承認してください"

        # 本番環境
        - Name: Production
          Actions:
            - Name: CreateChangeSet
              ActionTypeId:
                Category: Deploy
                Owner: AWS
                Provider: CloudFormation
                Version: "1"
              Configuration:
                ActionMode: CHANGE_SET_REPLACE
                StackName: production-stack
                ChangeSetName: production-changeset
                TemplatePath: SourceOutput::template.yaml
                TemplateConfiguration: SourceOutput::config/production.json
                Capabilities: CAPABILITY_NAMED_IAM
                RoleArn: !GetAtt CloudFormationRole.Arn
              InputArtifacts:
                - Name: SourceOutput
              RunOrder: 1

            - Name: ExecuteChangeSet
              ActionTypeId:
                Category: Deploy
                Owner: AWS
                Provider: CloudFormation
                Version: "1"
              Configuration:
                ActionMode: CHANGE_SET_EXECUTE
                StackName: production-stack
                ChangeSetName: production-changeset
              RunOrder: 2
```

### 9.2 GitHub Actions + CloudFormation

```yaml
# .github/workflows/deploy.yml
name: Deploy Infrastructure

on:
  push:
    branches: [main]
    paths:
      - 'cloudformation/**'
  pull_request:
    branches: [main]
    paths:
      - 'cloudformation/**'

permissions:
  id-token: write
  contents: read
  pull-requests: write

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: cfn-lint
        uses: scottbrenner/cfn-lint-action@v2
        with:
          command: cfn-lint cloudformation/**/*.yaml

      - name: cfn-nag
        uses: stelligent/cfn_nag@master
        with:
          input_path: cloudformation/

  deploy-staging:
    needs: lint
    if: github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-role
          aws-region: ap-northeast-1

      - name: Deploy to Staging
        run: |
          aws cloudformation deploy \
            --template-file cloudformation/template.yaml \
            --stack-name staging-stack \
            --parameter-overrides \
              EnvironmentName=stg \
              InstanceType=t3.small \
            --capabilities CAPABILITY_NAMED_IAM \
            --no-fail-on-empty-changeset

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-role
          aws-region: ap-northeast-1

      - name: Create Change Set
        run: |
          aws cloudformation create-change-set \
            --stack-name production-stack \
            --change-set-name "deploy-${GITHUB_SHA}" \
            --template-body file://cloudformation/template.yaml \
            --parameters \
              ParameterKey=EnvironmentName,ParameterValue=prod \
            --capabilities CAPABILITY_NAMED_IAM

          aws cloudformation wait change-set-create-complete \
            --stack-name production-stack \
            --change-set-name "deploy-${GITHUB_SHA}"

      - name: Describe Change Set
        run: |
          aws cloudformation describe-change-set \
            --stack-name production-stack \
            --change-set-name "deploy-${GITHUB_SHA}" \
            --output json

      - name: Execute Change Set
        run: |
          aws cloudformation execute-change-set \
            --stack-name production-stack \
            --change-set-name "deploy-${GITHUB_SHA}"

          aws cloudformation wait stack-update-complete \
            --stack-name production-stack
```

### 9.3 テンプレートの Lint とテスト

```bash
# cfn-lint (構文チェック)
pip install cfn-lint
cfn-lint template.yaml

# cfn-nag (セキュリティチェック)
gem install cfn-nag
cfn_nag_scan --input-path template.yaml

# TaskCat (マルチリージョンテスト)
pip install taskcat
taskcat test run

# Rain (CloudFormation の便利ツール)
# テンプレートのフォーマット
rain fmt template.yaml

# テンプレートのデプロイ (対話的)
rain deploy template.yaml my-stack

# スタック情報の表示
rain ls
rain watch my-stack
```

```yaml
# taskcat の設定ファイル (.taskcat.yml)
project:
  name: my-infra
  regions:
    - ap-northeast-1
    - us-east-1
tests:
  default:
    template: template.yaml
    parameters:
      EnvironmentName: test
      InstanceType: t3.micro
```

---

## 10. トラブルシューティング

### 10.1 よくあるエラーと対処法

```
エラー 1: CREATE_FAILED - Resource already exists
原因: 同名のリソースが既に存在
対処:
  - リソース名を変更する
  - 既存リソースをインポートする
  - 既存リソースを削除してからスタック作成

エラー 2: UPDATE_ROLLBACK_FAILED
原因: ロールバック中にもエラーが発生
対処:
  - ContinueUpdateRollback を実行
  aws cloudformation continue-update-rollback \
    --stack-name my-stack \
    --resources-to-skip LogicalResourceId1

エラー 3: DELETE_FAILED
原因: リソースが他から参照されている / 手動変更されている
対処:
  - 依存リソースを先に削除
  - retain-resources オプションで強制削除
  aws cloudformation delete-stack \
    --stack-name my-stack \
    --retain-resources WebSecurityGroup

エラー 4: Template validation error
原因: テンプレート構文エラー
対処:
  - aws cloudformation validate-template で事前検証
  aws cloudformation validate-template \
    --template-body file://template.yaml

エラー 5: InsufficientCapabilities
原因: IAM リソース作成に capabilities が不足
対処:
  - --capabilities CAPABILITY_NAMED_IAM を追加
  - CAPABILITY_AUTO_EXPAND (マクロ使用時)
```

### 10.2 UPDATE_ROLLBACK_FAILED の対処

```bash
# 1. 失敗したリソースの特定
aws cloudformation describe-stack-events \
  --stack-name my-stack \
  --query 'StackEvents[?ResourceStatus==`UPDATE_FAILED`].{Resource:LogicalResourceId,Reason:ResourceStatusReason}' \
  --output table

# 2. 問題のあるリソースをスキップしてロールバック続行
aws cloudformation continue-update-rollback \
  --stack-name my-stack \
  --resources-to-skip MyLambdaFunction MySecurityGroup

# 3. ロールバック完了を待機
aws cloudformation wait stack-rollback-complete \
  --stack-name my-stack
```

### 10.3 スタック削除のスタック

```bash
# 削除保護の解除
aws cloudformation update-termination-protection \
  --no-enable-termination-protection \
  --stack-name my-stack

# 依存関係の確認
aws cloudformation list-imports \
  --export-name 'my-stack-VpcId'

# S3 バケットが空でない場合の対処
aws s3 rm s3://my-bucket --recursive
aws cloudformation delete-stack --stack-name my-stack

# 強制削除 (一部リソースを残す)
aws cloudformation delete-stack \
  --stack-name my-stack \
  --retain-resources SecurityGroup VPCEndpoint
```

### 10.4 デバッグテクニック

```yaml
# cfn-init と cfn-signal を使った EC2 初期化とシグナル

Resources:
  WebServer:
    Type: AWS::EC2::Instance
    CreationPolicy:
      ResourceSignal:
        Count: 1
        Timeout: PT15M  # 15分でタイムアウト
    Metadata:
      AWS::CloudFormation::Init:
        configSets:
          full_install:
            - install_packages
            - configure_app
        install_packages:
          packages:
            yum:
              httpd: []
              php: []
          services:
            sysvinit:
              httpd:
                enabled: true
                ensureRunning: true
        configure_app:
          files:
            /var/www/html/index.html:
              content: |
                <html><body>Hello from CloudFormation!</body></html>
              mode: '000644'
    Properties:
      ImageId: !Ref LatestAmiId
      InstanceType: !Ref InstanceType
      UserData:
        Fn::Base64: !Sub |
          #!/bin/bash -xe
          yum update -y
          yum install -y aws-cfn-bootstrap

          # cfn-init の実行
          /opt/aws/bin/cfn-init -v \
            --stack ${AWS::StackName} \
            --resource WebServer \
            --configsets full_install \
            --region ${AWS::Region}

          # 結果を cfn-signal で報告
          /opt/aws/bin/cfn-signal -e $? \
            --stack ${AWS::StackName} \
            --resource WebServer \
            --region ${AWS::Region}
```

---

## 11. 高度なテンプレートパターン

### 11.1 DeletionPolicy と UpdateReplacePolicy

```yaml
Resources:
  # スナップショット取得後に削除
  Database:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot
    UpdateReplacePolicy: Snapshot
    Properties:
      DBInstanceClass: db.t3.medium
      Engine: mysql
      MasterUsername: admin
      MasterUserPassword: !Ref DatabasePassword

  # 削除しない (スタック削除時もリソースを残す)
  ImportantBucket:
    Type: AWS::S3::Bucket
    DeletionPolicy: Retain
    Properties:
      BucketName: !Sub '${AWS::StackName}-important-data'

  # 通常削除 (デフォルト)
  TempBucket:
    Type: AWS::S3::Bucket
    DeletionPolicy: Delete
    Properties:
      BucketName: !Sub '${AWS::StackName}-temp'
```

### 11.2 DependsOn と WaitCondition

```yaml
Resources:
  # 明示的な依存関係
  ApplicationServer:
    Type: AWS::EC2::Instance
    DependsOn:
      - Database
      - CacheCluster
    Properties:
      # ...

  # WaitCondition (外部プロセスからのシグナル待ち)
  WaitHandle:
    Type: AWS::CloudFormation::WaitConditionHandle

  WaitCondition:
    Type: AWS::CloudFormation::WaitCondition
    DependsOn: ApplicationServer
    Properties:
      Handle: !Ref WaitHandle
      Timeout: '600'
      Count: 1
```

### 11.3 条件分岐の活用

```yaml
Conditions:
  IsProduction: !Equals [!Ref EnvironmentName, prod]
  IsNotProduction: !Not [!Equals [!Ref EnvironmentName, prod]]
  CreateReplica: !And
    - !Equals [!Ref EnvironmentName, prod]
    - !Equals [!Ref EnableReadReplica, true]
  UseCustomDomain: !Not [!Equals [!Ref DomainName, '']]

Resources:
  # 条件付きリソース作成
  ReadReplica:
    Type: AWS::RDS::DBInstance
    Condition: CreateReplica
    Properties:
      SourceDBInstanceIdentifier: !Ref Database
      DBInstanceClass: db.t3.medium

  # 条件付きプロパティ
  ALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Scheme: !If [IsProduction, internet-facing, internal]
      Subnets: !If
        - IsProduction
        - [!Ref PublicSubnet1, !Ref PublicSubnet2, !Ref PublicSubnet3]
        - [!Ref PublicSubnet1, !Ref PublicSubnet2]

  # 条件付き出力
  Outputs:
    ReplicaEndpoint:
      Condition: CreateReplica
      Value: !GetAtt ReadReplica.Endpoint.Address
```

### 11.4 動的参照 (Dynamic References)

```yaml
Resources:
  Database:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceClass: db.t3.medium
      Engine: mysql
      # SSM Parameter Store からの参照
      MasterUsername: '{{resolve:ssm:/prod/db/username}}'
      # Secrets Manager からの参照
      MasterUserPassword: '{{resolve:secretsmanager:prod/db/credentials:SecretString:password}}'

  # SSM SecureString からの参照
  ApiServer:
    Type: AWS::ECS::TaskDefinition
    Properties:
      ContainerDefinitions:
        - Name: app
          Environment:
            - Name: API_KEY
              Value: '{{resolve:ssm-secure:/prod/api/key}}'
```

---

## 12. 本格的なテンプレート例

### 12.1 3 層 Web アプリケーション

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: '3層 Web アプリケーション基盤'

Parameters:
  EnvironmentName:
    Type: String
    Default: prod
    AllowedValues: [dev, stg, prod]

  VpcCidr:
    Type: String
    Default: '10.0.0.0/16'

  DatabasePassword:
    Type: String
    NoEcho: true
    MinLength: 8

Conditions:
  IsProduction: !Equals [!Ref EnvironmentName, prod]

Resources:
  # ============ VPC ============
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCidr
      EnableDnsHostnames: true
      EnableDnsSupport: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-vpc'

  InternetGateway:
    Type: AWS::EC2::InternetGateway

  InternetGatewayAttachment:
    Type: AWS::EC2::VPCGatewayAttachment
    Properties:
      InternetGatewayId: !Ref InternetGateway
      VpcId: !Ref VPC

  # パブリックサブネット
  PublicSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: !Select [0, !Cidr [!Ref VpcCidr, 6, 8]]
      AvailabilityZone: !Select [0, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-1'

  PublicSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: !Select [1, !Cidr [!Ref VpcCidr, 6, 8]]
      AvailabilityZone: !Select [1, !GetAZs '']
      MapPublicIpOnLaunch: true
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-public-2'

  # プライベートサブネット
  PrivateSubnet1:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: !Select [2, !Cidr [!Ref VpcCidr, 6, 8]]
      AvailabilityZone: !Select [0, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-private-1'

  PrivateSubnet2:
    Type: AWS::EC2::Subnet
    Properties:
      VpcId: !Ref VPC
      CidrBlock: !Select [3, !Cidr [!Ref VpcCidr, 6, 8]]
      AvailabilityZone: !Select [1, !GetAZs '']
      Tags:
        - Key: Name
          Value: !Sub '${EnvironmentName}-private-2'

  # NAT Gateway
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

  # ルートテーブル
  PublicRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC

  PublicRoute:
    Type: AWS::EC2::Route
    DependsOn: InternetGatewayAttachment
    Properties:
      RouteTableId: !Ref PublicRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      GatewayId: !Ref InternetGateway

  PublicSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet1
      RouteTableId: !Ref PublicRouteTable

  PublicSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PublicSubnet2
      RouteTableId: !Ref PublicRouteTable

  PrivateRouteTable:
    Type: AWS::EC2::RouteTable
    Properties:
      VpcId: !Ref VPC

  PrivateRoute:
    Type: AWS::EC2::Route
    Properties:
      RouteTableId: !Ref PrivateRouteTable
      DestinationCidrBlock: 0.0.0.0/0
      NatGatewayId: !Ref NatGateway1

  PrivateSubnet1RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PrivateSubnet1
      RouteTableId: !Ref PrivateRouteTable

  PrivateSubnet2RouteTableAssociation:
    Type: AWS::EC2::SubnetRouteTableAssociation
    Properties:
      SubnetId: !Ref PrivateSubnet2
      RouteTableId: !Ref PrivateRouteTable

  # ============ ALB ============
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: ALB Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 443
          ToPort: 443
          CidrIp: 0.0.0.0/0
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: !Sub '${EnvironmentName}-alb'
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2
      SecurityGroups:
        - !Ref ALBSecurityGroup
      Tags:
        - Key: Environment
          Value: !Ref EnvironmentName

  ALBTargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: !Sub '${EnvironmentName}-tg'
      Port: 80
      Protocol: HTTP
      VpcId: !Ref VPC
      TargetType: ip
      HealthCheckPath: /healthz
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  ALBListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Port: 80
      Protocol: HTTP
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref ALBTargetGroup

  # ============ RDS ============
  DatabaseSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Database Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 3306
          ToPort: 3306
          SourceSecurityGroupId: !Ref AppSecurityGroup

  DatabaseSubnetGroup:
    Type: AWS::RDS::DBSubnetGroup
    Properties:
      DBSubnetGroupDescription: Database subnet group
      SubnetIds:
        - !Ref PrivateSubnet1
        - !Ref PrivateSubnet2

  Database:
    Type: AWS::RDS::DBInstance
    DeletionPolicy: Snapshot
    UpdateReplacePolicy: Snapshot
    Properties:
      DBInstanceIdentifier: !Sub '${EnvironmentName}-db'
      DBInstanceClass: !If [IsProduction, db.r6g.large, db.t3.medium]
      Engine: mysql
      EngineVersion: '8.0'
      MasterUsername: admin
      MasterUserPassword: !Ref DatabasePassword
      AllocatedStorage: 100
      StorageType: gp3
      MultiAZ: !If [IsProduction, true, false]
      DBSubnetGroupName: !Ref DatabaseSubnetGroup
      VPCSecurityGroups:
        - !Ref DatabaseSecurityGroup
      BackupRetentionPeriod: !If [IsProduction, 30, 7]
      StorageEncrypted: true
      DeletionProtection: !If [IsProduction, true, false]

  # ============ ECS ============
  AppSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Application Security Group
      VpcId: !Ref VPC
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8080
          ToPort: 8080
          SourceSecurityGroupId: !Ref ALBSecurityGroup

  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: !Sub '${EnvironmentName}-cluster'
      ClusterSettings:
        - Name: containerInsights
          Value: enabled

  TaskExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: !Sub '${EnvironmentName}-app'
      NetworkMode: awsvpc
      RequiresCompatibilities: [FARGATE]
      Cpu: '512'
      Memory: '1024'
      ExecutionRoleArn: !GetAtt TaskExecutionRole.Arn
      ContainerDefinitions:
        - Name: app
          Image: !Sub '${AWS::AccountId}.dkr.ecr.${AWS::Region}.amazonaws.com/my-app:latest'
          PortMappings:
            - ContainerPort: 8080
          Environment:
            - Name: DB_HOST
              Value: !GetAtt Database.Endpoint.Address
            - Name: ENVIRONMENT
              Value: !Ref EnvironmentName
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref 'AWS::Region'
              awslogs-stream-prefix: app

  Service:
    Type: AWS::ECS::Service
    DependsOn: ALBListener
    Properties:
      ServiceName: !Sub '${EnvironmentName}-app-svc'
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: !If [IsProduction, 3, 1]
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          Subnets:
            - !Ref PrivateSubnet1
            - !Ref PrivateSubnet2
          SecurityGroups:
            - !Ref AppSecurityGroup
      LoadBalancers:
        - ContainerName: app
          ContainerPort: 8080
          TargetGroupArn: !Ref ALBTargetGroup

  LogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: !Sub '/ecs/${EnvironmentName}-app'
      RetentionInDays: !If [IsProduction, 90, 14]

Outputs:
  VpcId:
    Value: !Ref VPC
    Export:
      Name: !Sub '${EnvironmentName}-VpcId'

  ALBDnsName:
    Value: !GetAtt ApplicationLoadBalancer.DNSName

  DatabaseEndpoint:
    Value: !GetAtt Database.Endpoint.Address

  ECSClusterName:
    Value: !Ref ECSCluster
    Export:
      Name: !Sub '${EnvironmentName}-ECSCluster'
```

---

## 13. アンチパターン

### 13.1 巨大な単一テンプレート

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

### 13.2 DeletionPolicy 未設定でのデータベース運用

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

### 13.3 ハードコードされた値

```yaml
# [悪い例]
Resources:
  Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: ami-0abcdef1234567890  # ハードコード
      SubnetId: subnet-12345678       # ハードコード
      SecurityGroupIds:
        - sg-87654321                 # ハードコード

# [良い例]
Parameters:
  LatestAmiId:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>
    Default: /aws/service/ami-amazon-linux-latest/amzn2-ami-hvm-x86_64-gp2

Resources:
  Instance:
    Type: AWS::EC2::Instance
    Properties:
      ImageId: !Ref LatestAmiId
      SubnetId: !ImportValue 'network-PublicSubnet1Id'
      SecurityGroupIds:
        - !ImportValue 'security-WebSGId'
```

### 13.4 変更セットを使わない直接更新

```
[悪い例]
aws cloudformation update-stack --stack-name prod-stack ...
→ 意図しない置換 (Replacement) が発生する可能性

[良い例]
1. aws cloudformation create-change-set ...
2. aws cloudformation describe-change-set ...  ← 影響を確認
3. aws cloudformation execute-change-set ...
```

### 13.5 削除保護なしの本番スタック

```bash
# 本番スタックには必ず削除保護を有効化
aws cloudformation update-termination-protection \
  --enable-termination-protection \
  --stack-name production-stack
```

---

## 14. FAQ

### Q1. CloudFormation と Terraform のどちらを使うべきですか？

AWS のみを利用する場合は CloudFormation が AWS サービスとの統合が深く、変更セットやドリフト検出など運用機能も充実している。マルチクラウドやオンプレミスも管理する場合は Terraform が適している。組織の既存スキルセットも重要な判断要素である。

### Q2. テンプレートの最大サイズは？

テンプレート本文は 51,200 バイト(約50KB)が上限。S3 に配置したテンプレートを参照する場合は 460,800 バイト(約450KB)まで拡大される。大規模な構成ではネストスタックで分割するか、CDK の利用を検討する。

### Q3. スタック更新に失敗した場合はどうなりますか？

デフォルトでは自動ロールバックが実行され、更新前の状態に戻る。`--disable-rollback` オプションを使うとロールバックを無効化でき、失敗原因の調査がしやすくなるが、スタックは UPDATE_FAILED 状態のままになる。本番環境では自動ロールバックを有効にしておくべきである。

### Q4. CloudFormation で管理できないリソースがある場合は？

カスタムリソース (Lambda-backed) を使って、CloudFormation が直接サポートしていないリソースも管理できる。例えば、サードパーティ API の設定や、AWS の新サービスがサポートされる前のリソース作成にも活用できる。また、AWS CloudFormation Registry にサードパーティのリソースタイプを登録する方法もある。

### Q5. CDK と CloudFormation の関係は？

CDK は TypeScript や Python などのプログラミング言語で記述し、最終的に CloudFormation テンプレートを生成・デプロイする。CDK の裏側では CloudFormation が動作しているため、CloudFormation の知識は CDK を使う上でも必須である。複雑なロジックや再利用性が求められる場合は CDK が適しており、シンプルなテンプレートには CloudFormation を直接使うのが効率的である。

### Q6. スタックの数に上限はありますか？

デフォルトでは 1 リージョンあたり 2,000 スタックが上限。Service Quotas からの引き上げリクエストが可能。Export 値は 1 リージョンあたり 5,000 が上限で、こちらは引き上げできないため、大規模構成では SSM Parameter Store を代替として使うことを検討する。

### Q7. CloudFormation のデプロイを高速化するには？

テンプレートの分割、並列リソース作成 (DependsOn の最小化)、不要なリソースの除外が基本。また、`aws cloudformation deploy` コマンドは変更セットの作成と実行を自動化するため、スクリプトでの利用に適している。CI/CD パイプラインでは、ステージング環境のテスト結果をキャッシュして本番環境のデプロイ時間を短縮する戦略も有効である。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| テンプレート | YAML/JSON でリソースを宣言的に定義 |
| パラメータ | 環境ごとの設定値を外部化、SSM/Secrets Manager との動的参照も可能 |
| 組み込み関数 | !Ref, !Sub, !GetAtt, !If などで動的な値を構成 |
| 変更セット | 更新前に影響範囲をプレビュー (本番更新の必須プロセス) |
| クロススタック参照 | Export/Import でスタック間の値を共有 |
| ネストスタック | 大規模テンプレートの分割と再利用 |
| ドリフト検出 | テンプレートと実際の構成の差分を検出・自動化可能 |
| カスタムリソース | Lambda で CloudFormation を拡張 |
| スタックセット | マルチアカウント・マルチリージョンデプロイ |
| CI/CD 統合 | CodePipeline / GitHub Actions で自動デプロイ |
| トラブルシューティング | ロールバック対処、リソースインポート、デバッグ手法 |

---

## 次に読むべきガイド

- [AWS CDK](./01-cdk.md) -- プログラミング言語でインフラを定義
- [CodePipeline](./02-codepipeline.md) -- CloudFormation を CI/CD に統合
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- CloudFormation で使う IAM の設計
- [ECS 基礎](../06-containers/00-ecs-basics.md) -- CloudFormation で ECS を構築

---

## 参考文献

1. AWS 公式ドキュメント「AWS CloudFormation ユーザーガイド」 https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/
2. AWS 公式「CloudFormation ベストプラクティス」 https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/best-practices.html
3. AWS 公式「リソースおよびプロパティリファレンス」 https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/aws-template-resource-type-ref.html
4. cfn-lint GitHub リポジトリ https://github.com/aws-cloudformation/cfn-lint
5. Rain (CloudFormation CLI ツール) https://github.com/aws-cloudformation/rain
6. TaskCat (テストフレームワーク) https://github.com/aws-ia/taskcat
