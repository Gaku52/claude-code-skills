# AWS CDK (Cloud Development Kit)

> プログラミング言語で AWS インフラを定義する AWS CDK の基本概念、TypeScript/Python での実装、L1/L2/L3 コンストラクト、テスト、CI/CD 統合までを体系的に学ぶ。

---

## この章で学ぶこと

1. **CDK の基本概念とプロジェクト構成** -- App、Stack、Construct の階層構造と、CDK プロジェクトの初期化・デプロイフローを理解する
2. **L1/L2/L3 コンストラクトの使い分け** -- 抽象化レベルの異なるコンストラクトを適切に選択して効率的にインフラを定義する
3. **テストと CI/CD 統合** -- スナップショットテスト、ファイングレインドアサーション、パイプラインでの自動デプロイを実践する
4. **マルチスタック設計とクロススタック参照** -- 複数のスタックを連携させた大規模インフラの設計パターンを習得する
5. **CDK Pipelines による自動デプロイ** -- CDK 自身をパイプラインで管理するセルフミューテーションパターンを理解する

---

## 1. CDK の基本概念

### 1.1 CDK のアーキテクチャ

```
CDK のワークフロー:

TypeScript / Python コード
    |
    | cdk synth
    v
+----------------------+
| CloudFormation       |
| テンプレート (JSON)   |
+----------------------+
    |
    | cdk deploy
    v
+----------------------+
| CloudFormation       |
| スタック             |
+----------------------+
    |
    v
+----------------------+
| AWS リソース         |
| (VPC, Lambda, etc.)  |
+----------------------+

CDK の階層:
+-----------------------------------+
| App                               |
|   +-----------------------------+ |
|   | Stack A (dev)               | |
|   |   +-----+ +-----+ +-----+ | |
|   |   | L2  | | L2  | | L3  | | |
|   |   | VPC | | Lambda| | API | | |
|   |   +-----+ +-----+ +-----+ | |
|   +-----------------------------+ |
|   +-----------------------------+ |
|   | Stack B (prod)              | |
|   |   +-----+ +-----+ +-----+ | |
|   |   | L2  | | L2  | | L3  | | |
|   |   +-----+ +-----+ +-----+ | |
|   +-----------------------------+ |
+-----------------------------------+
```

### 1.2 CDK vs CloudFormation vs Terraform

| 特性 | CDK | CloudFormation | Terraform |
|------|-----|---------------|-----------|
| 言語 | TypeScript, Python, Java, Go, C# | YAML/JSON | HCL |
| 抽象化 | L1/L2/L3 コンストラクト | なし | モジュール |
| ループ/条件 | 言語ネイティブ | 限定的 | count, for_each |
| テスト | 単体テスト可能 | cfn-lint | terraform plan |
| 状態管理 | CloudFormation | CloudFormation | tfstate |
| マルチクラウド | AWS のみ | AWS のみ | マルチ対応 |
| 学習コスト | プログラマに低い | 中程度 | 中程度 |
| IDE サポート | 型補完、リファクタリング | VSCode プラグイン | VSCode プラグイン |
| ドリフト検出 | CloudFormation 経由 | ネイティブ | plan で検出 |
| エコシステム | Construct Hub | Registry | Registry |

### 1.3 CDK v2 の特徴

CDK v2 では全てのモジュールが `aws-cdk-lib` 単一パッケージに統合された。v1 では個別にインストールしていた `@aws-cdk/aws-s3` や `@aws-cdk/aws-lambda` が不要になり、依存関係の管理が大幅に簡素化された。

```typescript
// CDK v1 (旧)
import * as s3 from '@aws-cdk/aws-s3';
import * as lambda from '@aws-cdk/aws-lambda';

// CDK v2 (現行)
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as lambda from 'aws-cdk-lib/aws-lambda';
```

主な変更点:
- **単一パッケージ**: `aws-cdk-lib` に全モジュール統合
- **Constructs ライブラリ分離**: `constructs` パッケージが独立
- **安定 API のみ含有**: 実験的 API は `@aws-cdk` スコープの alpha パッケージに分離
- **後方互換性**: マイナーバージョン間で後方互換性が保証

---

## 2. CDK プロジェクトの構成

### 2.1 プロジェクト初期化

```bash
# CDK CLI のインストール
npm install -g aws-cdk

# バージョン確認
cdk --version

# TypeScript プロジェクトの作成
mkdir my-cdk-app && cd my-cdk-app
cdk init app --language typescript

# Python プロジェクトの作成
mkdir my-cdk-app && cd my-cdk-app
cdk init app --language python
source .venv/bin/activate
pip install -r requirements.txt

# Java プロジェクトの作成
mkdir my-cdk-app && cd my-cdk-app
cdk init app --language java

# Go プロジェクトの作成
mkdir my-cdk-app && cd my-cdk-app
cdk init app --language go
```

### 2.2 TypeScript プロジェクト構造

```
my-cdk-app/
├── bin/
│   └── my-cdk-app.ts        # App エントリポイント
├── lib/
│   ├── network-stack.ts      # ネットワークスタック
│   ├── app-stack.ts          # アプリケーションスタック
│   ├── database-stack.ts     # データベーススタック
│   ├── monitoring-stack.ts   # 監視スタック
│   └── constructs/           # カスタムコンストラクト
│       ├── api-construct.ts
│       └── vpc-construct.ts
├── test/
│   ├── network-stack.test.ts # テスト
│   ├── app-stack.test.ts
│   └── snapshot/             # スナップショット
├── lambda/                   # Lambda 関数のソースコード
│   ├── handler.py
│   └── requirements.txt
├── cdk.json                  # CDK 設定
├── cdk.context.json          # コンテキスト値のキャッシュ
├── package.json
└── tsconfig.json
```

### 2.3 App エントリポイント

```typescript
// bin/my-cdk-app.ts
import * as cdk from 'aws-cdk-lib';
import { NetworkStack } from '../lib/network-stack';
import { DatabaseStack } from '../lib/database-stack';
import { AppStack } from '../lib/app-stack';
import { MonitoringStack } from '../lib/monitoring-stack';

const app = new cdk.App();

// 環境設定
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION || 'ap-northeast-1',
};

// コンテキストから環境名を取得
const environmentName = app.node.tryGetContext('env') || 'dev';

// スタック間の依存関係を定義
const networkStack = new NetworkStack(app, `${environmentName}-NetworkStack`, {
  env,
  environmentName,
});

const databaseStack = new DatabaseStack(app, `${environmentName}-DatabaseStack`, {
  env,
  vpc: networkStack.vpc,
  environmentName,
});

const appStack = new AppStack(app, `${environmentName}-AppStack`, {
  env,
  vpc: networkStack.vpc,
  table: databaseStack.table,
  environmentName,
});

new MonitoringStack(app, `${environmentName}-MonitoringStack`, {
  env,
  lambdaFunction: appStack.handler,
  apiGateway: appStack.api,
  environmentName,
});

// タグをアプリ全体に適用
cdk.Tags.of(app).add('Project', 'MyApp');
cdk.Tags.of(app).add('Environment', environmentName);
cdk.Tags.of(app).add('ManagedBy', 'CDK');
```

### 2.4 cdk.json の設定

```json
{
  "app": "npx ts-node --prefer-ts-exts bin/my-cdk-app.ts",
  "watch": {
    "include": ["**"],
    "exclude": [
      "README.md",
      "cdk*.json",
      "**/*.d.ts",
      "**/*.js",
      "tsconfig.json",
      "package*.json",
      "yarn.lock",
      "node_modules",
      "test"
    ]
  },
  "context": {
    "@aws-cdk/aws-lambda:recognizeLayerVersion": true,
    "@aws-cdk/core:checkSecretUsage": true,
    "@aws-cdk/core:target-partitions": ["aws", "aws-cn"],
    "@aws-cdk/aws-ecs:arnFormatIncludesClusterName": true,
    "@aws-cdk/aws-apigateway:usagePlanKeyOrderInsensitiveId": true,
    "@aws-cdk/core:stackRelativeExports": true,
    "@aws-cdk/aws-rds:lowercaseDbIdentifier": true,
    "@aws-cdk/aws-lambda:recognizeVersionProps": true,
    "@aws-cdk/aws-ec2:restrictDefaultSecurityGroup": true,
    "env": "dev",
    "vpcCidr": "10.0.0.0/16",
    "maxAzs": 3
  }
}
```

---

## 3. L1/L2/L3 コンストラクト

### 3.1 コンストラクトの抽象化レベル

```
コンストラクトの階層:

L3 (Patterns): 複数リソースの組み合わせ
  aws-ecs-patterns.ApplicationLoadBalancedFargateService
  → ALB + ECS Service + タスク定義 + CloudWatch を一括作成
       |
       v
L2 (High-level): 個別リソースの高レベル抽象化
  aws-ec2.Vpc, aws-lambda.Function, aws-s3.Bucket
  → デフォルト値、ヘルパーメソッド、型安全
       |
       v
L1 (CFn Resources): CloudFormation リソースの 1:1 マッピング
  aws-ec2.CfnVPC, aws-lambda.CfnFunction
  → CloudFormation テンプレートと同じ粒度
```

### 3.2 L2 コンストラクトでの VPC 定義

```typescript
// lib/network-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import { Construct } from 'constructs';

interface NetworkStackProps extends cdk.StackProps {
  environmentName: string;
}

export class NetworkStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string, props: NetworkStackProps) {
    super(scope, id, props);

    // L2 コンストラクト: デフォルトで NAT GW、IGW、ルートテーブルを自動作成
    this.vpc = new ec2.Vpc(this, 'Vpc', {
      maxAzs: 3,
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
      subnetConfiguration: [
        {
          name: 'Public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'Private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
        {
          name: 'Isolated',
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
          cidrMask: 24,
        },
      ],
      natGateways: props.environmentName === 'prod' ? 3 : 1,
      // VPC フローログの有効化
      flowLogs: {
        's3FlowLog': {
          destination: ec2.FlowLogDestination.toS3(),
          trafficType: ec2.FlowLogTrafficType.REJECT,
        },
      },
    });

    // VPC エンドポイントの追加（プライベートサブネットからの AWS サービスアクセス）
    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });

    this.vpc.addGatewayEndpoint('DynamoDBEndpoint', {
      service: ec2.GatewayVpcEndpointAwsService.DYNAMODB,
    });

    this.vpc.addInterfaceEndpoint('SecretsManagerEndpoint', {
      service: ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
      subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });

    // タグ付け
    cdk.Tags.of(this.vpc).add('Environment', props.environmentName);

    // 出力
    new cdk.CfnOutput(this, 'VpcId', {
      value: this.vpc.vpcId,
      exportName: `${props.environmentName}-VpcId`,
    });

    new cdk.CfnOutput(this, 'PrivateSubnetIds', {
      value: this.vpc.privateSubnets.map(s => s.subnetId).join(','),
      exportName: `${props.environmentName}-PrivateSubnetIds`,
    });
  }
}
```

### 3.3 L2 コンストラクトでの Lambda 定義

```typescript
// lib/app-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as apigateway from 'aws-cdk-lib/aws-apigateway';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

interface AppStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  table: dynamodb.Table;
  environmentName: string;
}

export class AppStack extends cdk.Stack {
  public readonly handler: lambda.Function;
  public readonly api: apigateway.RestApi;

  constructor(scope: Construct, id: string, props: AppStackProps) {
    super(scope, id, props);

    // Lambda レイヤー（共通ライブラリ）
    const commonLayer = new lambda.LayerVersion(this, 'CommonLayer', {
      code: lambda.Code.fromAsset('lambda/layers/common'),
      compatibleRuntimes: [lambda.Runtime.PYTHON_3_12],
      description: '共通ユーティリティライブラリ',
    });

    // Lambda 関数
    this.handler = new lambda.Function(this, 'ApiHandler', {
      runtime: lambda.Runtime.PYTHON_3_12,
      code: lambda.Code.fromAsset('lambda/'),
      handler: 'handler.lambda_handler',
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      environment: {
        TABLE_NAME: props.table.tableName,
        ENVIRONMENT: props.environmentName,
        POWERTOOLS_SERVICE_NAME: 'my-api',
        LOG_LEVEL: props.environmentName === 'prod' ? 'INFO' : 'DEBUG',
      },
      tracing: lambda.Tracing.ACTIVE,
      layers: [commonLayer],
      logRetention: logs.RetentionDays.ONE_MONTH,
      // VPC 内で実行する場合
      // vpc: props.vpc,
      // vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });

    // IAM 権限の付与 (CDK が自動的に最小権限ポリシーを生成)
    props.table.grantReadWriteData(this.handler);

    // API Gateway
    this.api = new apigateway.RestApi(this, 'ItemsApi', {
      restApiName: `items-api-${props.environmentName}`,
      description: 'Items CRUD API',
      deployOptions: {
        stageName: props.environmentName,
        tracingEnabled: true,
        loggingLevel: apigateway.MethodLoggingLevel.INFO,
        dataTraceEnabled: props.environmentName !== 'prod',
        metricsEnabled: true,
        throttlingRateLimit: 1000,
        throttlingBurstLimit: 500,
      },
      defaultCorsPreflightOptions: {
        allowOrigins: apigateway.Cors.ALL_ORIGINS,
        allowMethods: apigateway.Cors.ALL_METHODS,
        allowHeaders: ['Content-Type', 'Authorization'],
      },
    });

    const items = this.api.root.addResource('items');
    items.addMethod('GET', new apigateway.LambdaIntegration(this.handler));
    items.addMethod('POST', new apigateway.LambdaIntegration(this.handler));

    const item = items.addResource('{id}');
    item.addMethod('GET', new apigateway.LambdaIntegration(this.handler));
    item.addMethod('PUT', new apigateway.LambdaIntegration(this.handler));
    item.addMethod('DELETE', new apigateway.LambdaIntegration(this.handler));

    // 出力
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: this.api.url,
      description: 'API Gateway URL',
    });
  }
}
```

### 3.4 L3 コンストラクト (パターン) の例

```typescript
import * as cdk from 'aws-cdk-lib';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as certificatemanager from 'aws-cdk-lib/aws-certificatemanager';
import * as route53 from 'aws-cdk-lib/aws-route53';

// L3: ALB + Fargate サービスを一行で作成
const cluster = new ecs.Cluster(this, 'Cluster', {
  vpc,
  containerInsights: true,
});

const certificate = certificatemanager.Certificate.fromCertificateArn(
  this, 'Cert',
  'arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx'
);

const service = new ecsPatterns.ApplicationLoadBalancedFargateService(
  this, 'WebService', {
    cluster,
    taskImageOptions: {
      image: ecs.ContainerImage.fromAsset('./app'),
      containerPort: 8080,
      environment: {
        NODE_ENV: 'production',
      },
      logDriver: ecs.LogDrivers.awsLogs({
        streamPrefix: 'web-service',
      }),
    },
    cpu: 512,
    memoryLimitMiB: 1024,
    desiredCount: 3,
    publicLoadBalancer: true,
    certificate,
    redirectHTTP: true,
    circuitBreaker: { rollback: true },
    healthCheckGracePeriod: cdk.Duration.seconds(60),
  }
);

// ヘルスチェックの設定
service.targetGroup.configureHealthCheck({
  path: '/health',
  interval: cdk.Duration.seconds(30),
  timeout: cdk.Duration.seconds(5),
  healthyThresholdCount: 2,
  unhealthyThresholdCount: 3,
});

// Auto Scaling の追加
const scaling = service.service.autoScaleTaskCount({
  minCapacity: 2,
  maxCapacity: 20,
});
scaling.scaleOnCpuUtilization('CpuScaling', {
  targetUtilizationPercent: 70,
  scaleInCooldown: cdk.Duration.seconds(60),
  scaleOutCooldown: cdk.Duration.seconds(60),
});
scaling.scaleOnMemoryUtilization('MemoryScaling', {
  targetUtilizationPercent: 80,
});
```

### 3.5 カスタム L2 コンストラクトの作成

```typescript
// lib/constructs/secure-bucket.ts
import * as cdk from 'aws-cdk-lib';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface SecureBucketProps {
  /**
   * バケット名のプレフィックス。アカウント ID が自動付与される。
   */
  bucketNamePrefix: string;
  /**
   * 環境名 (dev/staging/prod)
   */
  environmentName: string;
  /**
   * ライフサイクルルール: 日数後に Glacier に移行
   * @default 90
   */
  glacierTransitionDays?: number;
  /**
   * ライフサイクルルール: 日数後に削除
   * @default 365
   */
  expirationDays?: number;
  /**
   * バージョニングの有効化
   * @default true
   */
  versioned?: boolean;
}

export class SecureBucket extends Construct {
  public readonly bucket: s3.Bucket;
  public readonly encryptionKey: kms.Key;

  constructor(scope: Construct, id: string, props: SecureBucketProps) {
    super(scope, id);

    // KMS キーの作成
    this.encryptionKey = new kms.Key(this, 'Key', {
      alias: `${props.bucketNamePrefix}-${props.environmentName}`,
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      description: `Encryption key for ${props.bucketNamePrefix}`,
    });

    // セキュアな S3 バケットの作成
    this.bucket = new s3.Bucket(this, 'Bucket', {
      bucketName: `${props.bucketNamePrefix}-${props.environmentName}-${cdk.Stack.of(this).account}`,
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: this.encryptionKey,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      versioned: props.versioned ?? true,
      removalPolicy: props.environmentName === 'prod'
        ? cdk.RemovalPolicy.RETAIN
        : cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: props.environmentName !== 'prod',
      lifecycleRules: [
        {
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: cdk.Duration.days(30),
            },
            {
              storageClass: s3.StorageClass.GLACIER,
              transitionAfter: cdk.Duration.days(props.glacierTransitionDays ?? 90),
            },
          ],
          expiration: cdk.Duration.days(props.expirationDays ?? 365),
          noncurrentVersionExpiration: cdk.Duration.days(30),
        },
      ],
      serverAccessLogsBucket: new s3.Bucket(this, 'AccessLogBucket', {
        bucketName: `${props.bucketNamePrefix}-access-logs-${cdk.Stack.of(this).account}`,
        encryption: s3.BucketEncryption.S3_MANAGED,
        blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
        lifecycleRules: [
          { expiration: cdk.Duration.days(90) },
        ],
      }),
    });

    // バケットポリシー: HTTPS のみ許可
    this.bucket.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'DenyInsecureTransport',
      effect: iam.Effect.DENY,
      principals: [new iam.AnyPrincipal()],
      actions: ['s3:*'],
      resources: [this.bucket.bucketArn, `${this.bucket.bucketArn}/*`],
      conditions: {
        Bool: { 'aws:SecureTransport': 'false' },
      },
    }));
  }

  /**
   * 指定のロールにバケットへの読み取り権限を付与
   */
  grantRead(grantee: iam.IGrantable): iam.Grant {
    this.encryptionKey.grantDecrypt(grantee);
    return this.bucket.grantRead(grantee);
  }

  /**
   * 指定のロールにバケットへの読み書き権限を付与
   */
  grantReadWrite(grantee: iam.IGrantable): iam.Grant {
    this.encryptionKey.grantEncryptDecrypt(grantee);
    return this.bucket.grantReadWrite(grantee);
  }
}
```

---

## 4. Python での CDK

### 4.1 Python スタック定義

```python
# lib/app_stack.py
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    CfnOutput,
    aws_lambda as lambda_,
    aws_dynamodb as dynamodb,
    aws_apigateway as apigateway,
    aws_logs as logs,
    aws_iam as iam,
)
from constructs import Construct


class AppStack(Stack):
    def __init__(self, scope: Construct, id: str, environment_name: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # DynamoDB
        table = dynamodb.Table(
            self, "ItemsTable",
            partition_key=dynamodb.Attribute(
                name="id",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="created_at",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
            stream=dynamodb.StreamViewType.NEW_AND_OLD_IMAGES,
        )

        # GSI の追加
        table.add_global_secondary_index(
            index_name="StatusIndex",
            partition_key=dynamodb.Attribute(
                name="status",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="updated_at",
                type=dynamodb.AttributeType.STRING,
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # Lambda
        handler = lambda_.Function(
            self, "ApiHandler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            code=lambda_.Code.from_asset("lambda/"),
            handler="handler.lambda_handler",
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "TABLE_NAME": table.table_name,
                "ENVIRONMENT": environment_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.ONE_MONTH,
        )

        # 権限付与
        table.grant_read_write_data(handler)

        # API Gateway
        api = apigateway.RestApi(
            self, "ItemsApi",
            rest_api_name=f"items-api-{environment_name}",
            deploy_options=apigateway.StageOptions(
                stage_name=environment_name,
                tracing_enabled=True,
                metrics_enabled=True,
            ),
        )

        items = api.root.add_resource("items")
        items.add_method("GET", apigateway.LambdaIntegration(handler))
        items.add_method("POST", apigateway.LambdaIntegration(handler))

        item = items.add_resource("{id}")
        item.add_method("GET", apigateway.LambdaIntegration(handler))
        item.add_method("PUT", apigateway.LambdaIntegration(handler))
        item.add_method("DELETE", apigateway.LambdaIntegration(handler))

        CfnOutput(self, "ApiUrl", value=api.url)
        CfnOutput(self, "TableName", value=table.table_name)
```

### 4.2 Python でのカスタムコンストラクト

```python
# lib/constructs/monitored_lambda.py
from aws_cdk import (
    Duration,
    aws_lambda as lambda_,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_logs as logs,
)
from constructs import Construct


class MonitoredLambda(Construct):
    """Lambda 関数に CloudWatch アラーム・ダッシュボードを自動付与するコンストラクト"""

    def __init__(
        self,
        scope: Construct,
        id: str,
        *,
        function_name: str,
        handler: str,
        code: lambda_.Code,
        runtime: lambda_.Runtime = lambda_.Runtime.PYTHON_3_12,
        timeout: Duration = Duration.seconds(30),
        memory_size: int = 256,
        environment: dict = None,
        alarm_topic: sns.ITopic = None,
        error_rate_threshold: float = 5.0,
        duration_threshold_ms: float = 10000,
    ):
        super().__init__(scope, id)

        # Lambda 関数
        self.function = lambda_.Function(
            self, "Function",
            function_name=function_name,
            runtime=runtime,
            code=code,
            handler=handler,
            timeout=timeout,
            memory_size=memory_size,
            environment=environment or {},
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # エラー率アラーム
        errors = self.function.metric_errors(period=Duration.minutes(5))
        invocations = self.function.metric_invocations(period=Duration.minutes(5))

        error_rate = cloudwatch.MathExpression(
            expression="(errors / invocations) * 100",
            using_metrics={
                "errors": errors,
                "invocations": invocations,
            },
            label="Error Rate (%)",
            period=Duration.minutes(5),
        )

        self.error_alarm = cloudwatch.Alarm(
            self, "ErrorRateAlarm",
            metric=error_rate,
            threshold=error_rate_threshold,
            evaluation_periods=3,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description=f"{function_name} のエラー率が {error_rate_threshold}% を超過",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # レイテンシアラーム
        self.duration_alarm = cloudwatch.Alarm(
            self, "DurationAlarm",
            metric=self.function.metric_duration(
                statistic="p99",
                period=Duration.minutes(5),
            ),
            threshold=duration_threshold_ms,
            evaluation_periods=3,
            alarm_description=f"{function_name} の p99 レイテンシが {duration_threshold_ms}ms を超過",
        )

        # SNS 通知
        if alarm_topic:
            self.error_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))
            self.duration_alarm.add_alarm_action(cw_actions.SnsAction(alarm_topic))
```

---

## 5. テスト

### 5.1 テストの種類

```
CDK テストの種類:

+--------------------+     +--------------------+     +--------------------+
| スナップショット    |     | ファイングレインド  |     | バリデーション      |
| テスト             |     | アサーション        |     | テスト             |
+--------------------+     +--------------------+     +--------------------+
| 生成される CFn     |     | 特定のリソースや   |     | コンテキストに     |
| テンプレート全体を |     | プロパティの存在   |     | よるカスタム       |
| スナップショットと |     | を検証             |     | バリデーション     |
| 比較               |     |                    |     |                    |
+--------------------+     +--------------------+     +--------------------+
```

### 5.2 テストコード (TypeScript)

```typescript
// test/app-stack.test.ts
import * as cdk from 'aws-cdk-lib';
import { Template, Match, Capture } from 'aws-cdk-lib/assertions';
import { AppStack } from '../lib/app-stack';
import { NetworkStack } from '../lib/network-stack';
import { DatabaseStack } from '../lib/database-stack';

describe('AppStack', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const networkStack = new NetworkStack(app, 'TestNetworkStack', {
      environmentName: 'test',
    });
    const databaseStack = new DatabaseStack(app, 'TestDatabaseStack', {
      vpc: networkStack.vpc,
      environmentName: 'test',
    });
    const appStack = new AppStack(app, 'TestAppStack', {
      vpc: networkStack.vpc,
      table: databaseStack.table,
      environmentName: 'test',
    });
    template = Template.fromStack(appStack);
  });

  // ファイングレインドアサーション
  test('DynamoDB テーブルが PAY_PER_REQUEST で作成される', () => {
    template.hasResourceProperties('AWS::DynamoDB::Table', {
      BillingMode: 'PAY_PER_REQUEST',
      PointInTimeRecoverySpecification: {
        PointInTimeRecoveryEnabled: true,
      },
    });
  });

  test('Lambda 関数が正しいランタイムで作成される', () => {
    template.hasResourceProperties('AWS::Lambda::Function', {
      Runtime: 'python3.12',
      Timeout: 30,
      MemorySize: 256,
      TracingConfig: {
        Mode: 'Active',
      },
    });
  });

  test('Lambda に DynamoDB への読み書き権限が付与される', () => {
    template.hasResourceProperties('AWS::IAM::Policy', {
      PolicyDocument: {
        Statement: Match.arrayWith([
          Match.objectLike({
            Action: Match.arrayWith([
              'dynamodb:BatchGetItem',
              'dynamodb:GetItem',
              'dynamodb:PutItem',
            ]),
            Effect: 'Allow',
          }),
        ]),
      },
    });
  });

  // リソース数のテスト
  test('API Gateway REST API が1つ作成される', () => {
    template.resourceCountIs('AWS::ApiGateway::RestApi', 1);
  });

  // Capture を使ったテスト
  test('Lambda 環境変数に TABLE_NAME が設定される', () => {
    const envCapture = new Capture();
    template.hasResourceProperties('AWS::Lambda::Function', {
      Environment: {
        Variables: {
          TABLE_NAME: envCapture,
        },
      },
    });
    // Capture した値を検証
    expect(envCapture.asObject()).toBeDefined();
  });

  // 特定のリソースが存在しないことを検証
  test('パブリック S3 バケットが作成されない', () => {
    const resources = template.findResources('AWS::S3::Bucket', {
      Properties: {
        PublicAccessBlockConfiguration: Match.absent(),
      },
    });
    expect(Object.keys(resources)).toHaveLength(0);
  });

  // スナップショットテスト
  test('テンプレートがスナップショットと一致する', () => {
    expect(template.toJSON()).toMatchSnapshot();
  });
});
```

### 5.3 テストの実行

```bash
# テストの実行
npm test

# カバレッジ付きテスト
npm test -- --coverage

# 特定のテストファイルのみ実行
npm test -- --testPathPattern app-stack

# スナップショットの更新
npm test -- -u

# CDK diff (デプロイ前の差分確認)
cdk diff

# CDK synth (テンプレート生成確認)
cdk synth

# 特定スタックのテンプレート生成
cdk synth AppStack > template.yaml
```

### 5.4 Python でのテスト

```python
# test/test_app_stack.py
import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template, Match, Capture

from lib.app_stack import AppStack


@pytest.fixture
def template():
    app = cdk.App()
    stack = AppStack(app, "TestStack", environment_name="test")
    return Template.from_stack(stack)


def test_dynamodb_table_created(template):
    """DynamoDB テーブルが正しい設定で作成される"""
    template.has_resource_properties("AWS::DynamoDB::Table", {
        "BillingMode": "PAY_PER_REQUEST",
    })


def test_lambda_function_runtime(template):
    """Lambda 関数が Python 3.12 ランタイムで作成される"""
    template.has_resource_properties("AWS::Lambda::Function", {
        "Runtime": "python3.12",
        "MemorySize": 256,
    })


def test_api_gateway_created(template):
    """API Gateway が1つ作成される"""
    template.resource_count_is("AWS::ApiGateway::RestApi", 1)


def test_lambda_has_table_name_env(template):
    """Lambda に TABLE_NAME 環境変数が設定される"""
    env_capture = Capture()
    template.has_resource_properties("AWS::Lambda::Function", {
        "Environment": {
            "Variables": {
                "TABLE_NAME": env_capture,
            },
        },
    })
    assert env_capture.as_object() is not None


def test_no_public_s3_buckets(template):
    """パブリックアクセス可能な S3 バケットが存在しない"""
    template.has_resource_properties("AWS::S3::Bucket", {
        "PublicAccessBlockConfiguration": {
            "BlockPublicAcls": True,
            "BlockPublicPolicy": True,
            "IgnorePublicAcls": True,
            "RestrictPublicBuckets": True,
        },
    })
```

---

## 6. CDK Pipelines（セルフミューテーション）

### 6.1 CDK Pipelines の概念

```
CDK Pipelines のセルフミューテーション:

1. パイプライン定義を CDK で記述
2. パイプライン自身がソースの変更を検知
3. パイプラインが自分自身を更新 (Self-Mutation)
4. 更新後のパイプラインがアプリケーションをデプロイ

┌──────────────────────────────────────────────────────────────┐
│                    CDK Pipeline                              │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐│
│  │ Source    │  │ Build    │  │ Update   │  │ Deploy       ││
│  │ (GitHub) │→│ (Synth)  │→│ Pipeline │→│ Stages       ││
│  │          │  │          │  │ (Self    │  │ (Dev→Stg→Prd)││
│  │          │  │          │  │  Mutate) │  │              ││
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘│
│                                                              │
│  Self-Mutation: パイプライン定義が変わったら                   │
│  自動的にパイプライン自身を更新する                           │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 CDK Pipelines の実装

```typescript
// lib/pipeline-stack.ts
import * as cdk from 'aws-cdk-lib';
import * as codecommit from 'aws-cdk-lib/aws-codecommit';
import * as pipelines from 'aws-cdk-lib/pipelines';
import { Construct } from 'constructs';
import { AppStage } from './app-stage';

export class PipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // ソースリポジトリ
    const repo = codecommit.Repository.fromRepositoryName(
      this, 'Repo', 'my-cdk-app'
    );

    // パイプラインの定義
    const pipeline = new pipelines.CodePipeline(this, 'Pipeline', {
      pipelineName: 'MyAppPipeline',
      crossAccountKeys: true,
      synth: new pipelines.ShellStep('Synth', {
        input: pipelines.CodePipelineSource.codeCommit(repo, 'main'),
        commands: [
          'npm ci',
          'npm run build',
          'npx cdk synth',
        ],
        primaryOutputDirectory: 'cdk.out',
      }),
      // Docker ビルドを使用するかどうか
      dockerEnabledForSynth: true,
      dockerEnabledForSelfMutation: true,
    });

    // 開発環境ステージ
    const devStage = pipeline.addStage(new AppStage(this, 'Dev', {
      env: { account: '111111111111', region: 'ap-northeast-1' },
      environmentName: 'dev',
    }));

    // 開発環境のポストデプロイテスト
    devStage.addPost(new pipelines.ShellStep('IntegrationTest', {
      commands: [
        'curl -f $API_URL/health || exit 1',
      ],
      envFromCfnOutputs: {
        API_URL: devStage.apiUrlOutput,
      },
    }));

    // ステージング環境ステージ
    const stagingStage = pipeline.addStage(new AppStage(this, 'Staging', {
      env: { account: '222222222222', region: 'ap-northeast-1' },
      environmentName: 'staging',
    }));

    // 本番環境ステージ（手動承認付き）
    const prodStage = pipeline.addStage(new AppStage(this, 'Prod', {
      env: { account: '333333333333', region: 'ap-northeast-1' },
      environmentName: 'prod',
    }));

    prodStage.addPre(new pipelines.ManualApprovalStep('PromoteToProd', {
      comment: 'ステージング環境の動作を確認後、本番デプロイを承認してください',
    }));
  }
}
```

### 6.3 App Stage の定義

```typescript
// lib/app-stage.ts
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { NetworkStack } from './network-stack';
import { DatabaseStack } from './database-stack';
import { AppStack } from './app-stack';

interface AppStageProps extends cdk.StageProps {
  environmentName: string;
}

export class AppStage extends cdk.Stage {
  public readonly apiUrlOutput: cdk.CfnOutput;

  constructor(scope: Construct, id: string, props: AppStageProps) {
    super(scope, id, props);

    const networkStack = new NetworkStack(this, 'Network', {
      environmentName: props.environmentName,
    });

    const databaseStack = new DatabaseStack(this, 'Database', {
      vpc: networkStack.vpc,
      environmentName: props.environmentName,
    });

    const appStack = new AppStack(this, 'App', {
      vpc: networkStack.vpc,
      table: databaseStack.table,
      environmentName: props.environmentName,
    });

    this.apiUrlOutput = appStack.apiUrl;
  }
}
```

---

## 7. CDK デプロイコマンド

```bash
# ブートストラップ (初回のみ、各アカウント・リージョンごと)
cdk bootstrap aws://123456789012/ap-northeast-1

# クロスアカウントブートストラップ（信頼関係の設定）
cdk bootstrap aws://222222222222/ap-northeast-1 \
  --trust 111111111111 \
  --cloudformation-execution-policies 'arn:aws:iam::aws:policy/AdministratorAccess' \
  --trust-for-lookup 111111111111

# テンプレートの合成
cdk synth

# 差分確認
cdk diff

# 全スタックのデプロイ
cdk deploy --all

# 特定スタックのデプロイ
cdk deploy AppStack

# コンテキスト値を渡してデプロイ
cdk deploy --all --context env=prod --context vpcCidr=10.1.0.0/16

# 承認なしでデプロイ (CI/CD 向け)
cdk deploy --all --require-approval never

# ロールバック無効化（デバッグ時）
cdk deploy --all --no-rollback

# ホットスワップデプロイ（開発環境向け高速デプロイ）
cdk deploy --hotswap

# ウォッチモード（ファイル変更を検知して自動デプロイ）
cdk watch

# スタックの一覧
cdk list

# スタックの削除
cdk destroy --all

# コンテキストキャッシュのクリア
cdk context --clear
```

### 7.1 環境変数の活用

```bash
# CDK で使える主な環境変数
export CDK_DEFAULT_ACCOUNT=123456789012
export CDK_DEFAULT_REGION=ap-northeast-1
export CDK_NEW_BOOTSTRAP=1

# AWS プロファイルの指定
cdk deploy --all --profile my-profile

# verbose モード（デバッグ出力）
cdk deploy --all -v

# 出力を JSON 形式で取得
cdk synth --json
```

---

## 8. CDK のベストプラクティス

### 8.1 スタック分割戦略

```
推奨されるスタック分割:

1. ライフサイクルベース:
   ┌─────────────────────────────────────────────┐
   │ 変更頻度: 低 → 高                            │
   │                                             │
   │ NetworkStack    DatabaseStack    AppStack   │
   │ (VPC, Subnet)  (RDS, DynamoDB)  (Lambda,   │
   │                                  API GW)    │
   │ 月1回程度      月数回           日次        │
   └─────────────────────────────────────────────┘

2. チームベース:
   Platform Team → NetworkStack, SecurityStack
   Backend Team  → AppStack, DatabaseStack
   Frontend Team → CDNStack, S3StaticStack

3. 環境ベース:
   Dev   → dev-NetworkStack, dev-AppStack, ...
   Stg   → stg-NetworkStack, stg-AppStack, ...
   Prod  → prod-NetworkStack, prod-AppStack, ...
```

### 8.2 設定の外部化

```typescript
// 環境ごとの設定ファイル
// config/dev.ts
export const devConfig = {
  environmentName: 'dev',
  vpcCidr: '10.0.0.0/16',
  maxAzs: 2,
  natGateways: 1,
  lambdaMemorySize: 256,
  desiredCount: 1,
  domainName: 'dev.example.com',
  logRetentionDays: 7,
  enableWaf: false,
};

// config/prod.ts
export const prodConfig = {
  environmentName: 'prod',
  vpcCidr: '10.1.0.0/16',
  maxAzs: 3,
  natGateways: 3,
  lambdaMemorySize: 1024,
  desiredCount: 3,
  domainName: 'api.example.com',
  logRetentionDays: 365,
  enableWaf: true,
};

// bin/app.ts で設定を使用
import { devConfig } from '../config/dev';
import { prodConfig } from '../config/prod';

const config = process.env.CDK_ENV === 'prod' ? prodConfig : devConfig;
```

### 8.3 Aspects による横断的関心事

```typescript
// lib/aspects/tagging-aspect.ts
import * as cdk from 'aws-cdk-lib';
import { IConstruct } from 'constructs';

/**
 * 全リソースに必須タグを自動付与する Aspect
 */
class MandatoryTagsAspect implements cdk.IAspect {
  constructor(
    private readonly tags: Record<string, string>
  ) {}

  visit(node: IConstruct): void {
    if (cdk.CfnResource.isCfnResource(node)) {
      Object.entries(this.tags).forEach(([key, value]) => {
        cdk.Tags.of(node).add(key, value);
      });
    }
  }
}

/**
 * S3 バケットのパブリックアクセスを禁止する Aspect
 */
class BucketSecurityAspect implements cdk.IAspect {
  visit(node: IConstruct): void {
    if (node instanceof cdk.aws_s3.CfnBucket) {
      node.addPropertyOverride('PublicAccessBlockConfiguration', {
        BlockPublicAcls: true,
        BlockPublicPolicy: true,
        IgnorePublicAcls: true,
        RestrictPublicBuckets: true,
      });
    }
  }
}

/**
 * Lambda 関数の設定を強制する Aspect
 */
class LambdaSecurityAspect implements cdk.IAspect {
  visit(node: IConstruct): void {
    if (node instanceof cdk.aws_lambda.CfnFunction) {
      // X-Ray トレーシングの強制
      if (!node.tracingConfig) {
        node.addPropertyOverride('TracingConfig.Mode', 'Active');
      }
      // 予約済み同時実行数の上限設定
      if (!node.reservedConcurrentExecutions) {
        node.addPropertyOverride('ReservedConcurrentExecutions', 100);
      }
    }
  }
}

// 使用例
const app = new cdk.App();
cdk.Aspects.of(app).add(new MandatoryTagsAspect({
  Project: 'MyApp',
  CostCenter: 'Engineering',
  ManagedBy: 'CDK',
}));
cdk.Aspects.of(app).add(new BucketSecurityAspect());
cdk.Aspects.of(app).add(new LambdaSecurityAspect());
```

---

## 9. Construct Hub とサードパーティコンストラクト

### 9.1 Construct Hub の活用

Construct Hub (https://constructs.dev/) は CDK コンストラクトのレジストリで、AWS 公式および OSS コミュニティのコンストラクトを検索・利用できる。

```bash
# よく使われるサードパーティコンストラクト
npm install @aws-solutions-constructs/aws-lambda-dynamodb
npm install @aws-solutions-constructs/aws-cloudfront-s3
npm install cdk-nag
```

### 9.2 cdk-nag によるセキュリティチェック

```typescript
import { Aspects } from 'aws-cdk-lib';
import { AwsSolutionsChecks, NagSuppressions } from 'cdk-nag';

const app = new cdk.App();
// AWS Solutions のベストプラクティスチェック
Aspects.of(app).add(new AwsSolutionsChecks({ verbose: true }));

// 特定の警告を抑制する場合
NagSuppressions.addStackSuppressions(myStack, [
  {
    id: 'AwsSolutions-IAM4',
    reason: '開発環境では AWS マネージドポリシーの使用を許可',
  },
]);
```

### 9.3 AWS Solutions Constructs の活用

```typescript
import { LambdaToDynamoDB } from '@aws-solutions-constructs/aws-lambda-dynamodb';

// Lambda + DynamoDB のベストプラクティス構成を一括作成
const lambdaDdb = new LambdaToDynamoDB(this, 'LambdaToDdb', {
  lambdaFunctionProps: {
    runtime: lambda.Runtime.PYTHON_3_12,
    code: lambda.Code.fromAsset('lambda/'),
    handler: 'handler.lambda_handler',
  },
  dynamoTableProps: {
    partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
    billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
  },
});

// 自動的に以下が設定される:
// - Lambda に DynamoDB への最小権限
// - Lambda の CloudWatch Logs
// - DynamoDB の暗号化
// - Dead Letter Queue
```

---

## 10. アンチパターン

### 10.1 ハードコードされた値

```typescript
// [悪い例]
const bucket = new s3.Bucket(this, 'Bucket', {
  bucketName: 'my-company-prod-data-bucket',  // 環境ごとに変更が必要
});

// [良い例]
const bucket = new s3.Bucket(this, 'Bucket', {
  bucketName: `${props.environmentName}-data-${this.account}`,
  // または bucketName を省略して CDK に自動生成させる
});
```

### 10.2 L1 コンストラクトの多用

**問題点**: L1 (CfnXxx) コンストラクトを使うと、CDK の恩恵(自動的な IAM ポリシー生成、デフォルト値、型安全性)を失い、CloudFormation を直接書くのと変わらなくなる。

**改善**: 可能な限り L2 コンストラクトを使い、L2 で対応していないプロパティのみ `addOverride` や `node.defaultChild` でカスタマイズする。

```typescript
// [悪い例]: L1 で VPC を定義
const cfnVpc = new ec2.CfnVPC(this, 'Vpc', {
  cidrBlock: '10.0.0.0/16',
  enableDnsHostnames: true,
  enableDnsSupport: true,
});
// サブネット、ルートテーブル、IGW、NAT GW を全て手動で作成する必要がある

// [良い例]: L2 + エスケープハッチ
const vpc = new ec2.Vpc(this, 'Vpc', {
  maxAzs: 3,
});
// L1 にアクセスしてカスタマイズ
const cfnVpc = vpc.node.defaultChild as ec2.CfnVPC;
cfnVpc.addPropertyOverride('InstanceTenancy', 'dedicated');
```

### 10.3 巨大な単一スタック

```typescript
// [悪い例]: 全リソースを1つのスタックに詰め込む
class MonolithStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);
    // VPC, Lambda, DynamoDB, S3, CloudFront, WAF, ...
    // 500+ リソース → CloudFormation の上限 (500) に近づく
    // デプロイに 30 分以上かかる
  }
}

// [良い例]: 責務ごとにスタックを分割
// NetworkStack → VPC, Subnet, NAT GW
// DatabaseStack → DynamoDB, RDS
// AppStack → Lambda, API Gateway
// CDNStack → CloudFront, WAF
// MonitoringStack → CloudWatch, SNS
```

### 10.4 コンストラクト ID の変更

```typescript
// [危険]: コンストラクト ID を変更すると、リソースの論理 ID が変わり、
// 既存リソースが削除されて新しいリソースが作成される
// (ステートフルリソースの場合、データ損失のリスク)

// 変更前
const table = new dynamodb.Table(this, 'ItemsTable', { ... });
// 変更後（論理 ID が変わる → テーブルが再作成される）
const table = new dynamodb.Table(this, 'ItemsTableV2', { ... });

// 安全な対処法: RemovalPolicy.RETAIN を事前に設定
const table = new dynamodb.Table(this, 'ItemsTable', {
  removalPolicy: cdk.RemovalPolicy.RETAIN,  // スタック削除時もテーブルを残す
});
```

---

## 11. FAQ

### Q1. CDK と CloudFormation のどちらから始めるべきですか？

プログラミング経験があるなら CDK から始めることを推奨する。CDK は CloudFormation テンプレートを内部で生成するため、CDK を使いながら CloudFormation の概念も自然に学べる。`cdk synth` で生成されるテンプレートを確認することで理解が深まる。

### Q2. CDK のバージョンアップはどう管理すべきですか？

CDK v2 では全モジュールが `aws-cdk-lib` に統合されているため、バージョン管理が簡素化されている。`package.json` の `aws-cdk-lib` バージョンを更新し、テストを実行して互換性を確認する。マイナーバージョンアップは後方互換性が保たれている。

```bash
# CDK のバージョン確認
npx cdk --version

# パッケージの更新
npm update aws-cdk-lib constructs

# CLI の更新
npm install -g aws-cdk@latest

# 更新後のテスト実行
npm test
cdk diff
```

### Q3. CDK でステートフルリソースを安全に管理するには？

DynamoDB テーブルや RDS インスタンスなどのステートフルリソースには `removalPolicy: RemovalPolicy.RETAIN` を設定し、スタック削除時にリソースが残るようにする。また、リソースの論理 ID が変わらないよう、コンストラクト ID の変更には注意が必要である。

### Q4. cdk synth で生成されるテンプレートが巨大すぎる場合の対策は？

CloudFormation テンプレートの上限はパッケージ化後で 1 MB。大規模な場合はスタックを分割する。ネストされたスタック（`NestedStack`）を使う方法もあるが、CDK では独立したスタック間のクロススタック参照がよりシンプルで推奨される。

### Q5. CDK で既存の AWS リソースをインポートするには？

```typescript
// 既存 VPC のインポート
const existingVpc = ec2.Vpc.fromLookup(this, 'ExistingVpc', {
  vpcId: 'vpc-12345678',
});

// 既存 S3 バケットのインポート
const existingBucket = s3.Bucket.fromBucketName(
  this, 'ExistingBucket', 'my-existing-bucket'
);

// 既存 DynamoDB テーブルのインポート
const existingTable = dynamodb.Table.fromTableName(
  this, 'ExistingTable', 'existing-table'
);

// 既存 Lambda 関数のインポート
const existingFunction = lambda.Function.fromFunctionArn(
  this, 'ExistingFunc',
  'arn:aws:lambda:ap-northeast-1:123456789012:function:my-function'
);
```

### Q6. CDK のデプロイが遅い場合の高速化方法は？

```bash
# ホットスワップデプロイ（Lambda、ECS、Step Functions 等に対応）
# CloudFormation を経由せずに直接リソースを更新
cdk deploy --hotswap

# ウォッチモード（ファイル変更を検知して自動ホットスワップ）
cdk watch

# 並列デプロイ（依存関係のないスタックを並列実行）
cdk deploy --all --concurrency 5
```

### Q7. マルチリージョンデプロイはどう実装する？

```typescript
const regions = ['ap-northeast-1', 'us-east-1', 'eu-west-1'];

for (const region of regions) {
  new AppStack(app, `AppStack-${region}`, {
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region,
    },
    environmentName: 'prod',
  });
}

// CloudFormation StackSets を使用する場合は AWS SDK 経由で管理
```

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CDK とは | プログラミング言語で AWS インフラを定義するフレームワーク |
| コンストラクト | L1(CFn 1:1)、L2(高レベル抽象化)、L3(パターン) |
| 言語 | TypeScript, Python, Java, Go, C# |
| テスト | スナップショットテスト + ファイングレインドアサーション |
| デプロイ | cdk synth -> cdk diff -> cdk deploy |
| CDK Pipelines | セルフミューテーションで CI/CD を自動化 |
| Aspects | 横断的関心事（タグ付け、セキュリティ）を一括適用 |
| ベストプラクティス | スタック分割、設定外部化、cdk-nag でセキュリティチェック |
| 利点 | 型安全、テスト可能、IDE 補完、ループ/条件がネイティブ |

---

## 次に読むべきガイド

- [CloudFormation](./00-cloudformation.md) -- CDK の内部で生成されるテンプレートの理解
- [CodePipeline](./02-codepipeline.md) -- CDK を CI/CD パイプラインに統合
- [Lambda 基礎](../05-serverless/00-lambda-basics.md) -- CDK でデプロイする Lambda の設計

---

## 参考文献

1. AWS 公式ドキュメント「AWS CDK v2 デベロッパーガイド」 https://docs.aws.amazon.com/cdk/v2/guide/
2. AWS CDK API リファレンス https://docs.aws.amazon.com/cdk/api/v2/
3. CDK Patterns https://cdkpatterns.com/
4. Matt Coulter「The CDK Book」 https://www.thecdkbook.com/
5. Construct Hub https://constructs.dev/
6. AWS Solutions Constructs https://docs.aws.amazon.com/solutions/latest/constructs/
7. cdk-nag https://github.com/cdklabs/cdk-nag
