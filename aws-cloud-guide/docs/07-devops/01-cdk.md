# AWS CDK (Cloud Development Kit)

> プログラミング言語で AWS インフラを定義する AWS CDK の基本概念、TypeScript/Python での実装、L1/L2/L3 コンストラクト、テスト、CI/CD 統合までを体系的に学ぶ。

---

## この章で学ぶこと

1. **CDK の基本概念とプロジェクト構成** -- App、Stack、Construct の階層構造と、CDK プロジェクトの初期化・デプロイフローを理解する
2. **L1/L2/L3 コンストラクトの使い分け** -- 抽象化レベルの異なるコンストラクトを適切に選択して効率的にインフラを定義する
3. **テストと CI/CD 統合** -- スナップショットテスト、ファイングレインドアサーション、パイプラインでの自動デプロイを実践する

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

---

## 2. CDK プロジェクトの構成

### 2.1 プロジェクト初期化

```bash
# CDK CLI のインストール
npm install -g aws-cdk

# TypeScript プロジェクトの作成
mkdir my-cdk-app && cd my-cdk-app
cdk init app --language typescript

# Python プロジェクトの作成
mkdir my-cdk-app && cd my-cdk-app
cdk init app --language python
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.2 TypeScript プロジェクト構造

```
my-cdk-app/
├── bin/
│   └── my-cdk-app.ts        # App エントリポイント
├── lib/
│   ├── network-stack.ts      # ネットワークスタック
│   ├── app-stack.ts          # アプリケーションスタック
│   └── constructs/           # カスタムコンストラクト
│       └── api-construct.ts
├── test/
│   ├── network-stack.test.ts # テスト
│   └── app-stack.test.ts
├── cdk.json                  # CDK 設定
├── package.json
└── tsconfig.json
```

### 2.3 App エントリポイント

```typescript
// bin/my-cdk-app.ts
import * as cdk from 'aws-cdk-lib';
import { NetworkStack } from '../lib/network-stack';
import { AppStack } from '../lib/app-stack';

const app = new cdk.App();

const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION || 'ap-northeast-1',
};

const networkStack = new NetworkStack(app, 'NetworkStack', {
  env,
  environmentName: 'prod',
});

new AppStack(app, 'AppStack', {
  env,
  vpc: networkStack.vpc,
  environmentName: 'prod',
});
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
    });

    // タグ付け
    cdk.Tags.of(this.vpc).add('Environment', props.environmentName);
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
import { Construct } from 'constructs';

interface AppStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  environmentName: string;
}

export class AppStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: AppStackProps) {
    super(scope, id, props);

    // DynamoDB テーブル
    const table = new dynamodb.Table(this, 'ItemsTable', {
      partitionKey: { name: 'id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      pointInTimeRecovery: true,
    });

    // Lambda 関数
    const handler = new lambda.Function(this, 'ApiHandler', {
      runtime: lambda.Runtime.PYTHON_3_12,
      code: lambda.Code.fromAsset('lambda/'),
      handler: 'handler.lambda_handler',
      timeout: cdk.Duration.seconds(30),
      memorySize: 256,
      environment: {
        TABLE_NAME: table.tableName,
        ENVIRONMENT: props.environmentName,
      },
      tracing: lambda.Tracing.ACTIVE,
    });

    // IAM 権限の付与 (CDK が自動的に最小権限ポリシーを生成)
    table.grantReadWriteData(handler);

    // API Gateway
    const api = new apigateway.RestApi(this, 'ItemsApi', {
      restApiName: `items-api-${props.environmentName}`,
      deployOptions: {
        stageName: props.environmentName,
        tracingEnabled: true,
      },
    });

    const items = api.root.addResource('items');
    items.addMethod('GET', new apigateway.LambdaIntegration(handler));
    items.addMethod('POST', new apigateway.LambdaIntegration(handler));

    const item = items.addResource('{id}');
    item.addMethod('GET', new apigateway.LambdaIntegration(handler));
    item.addMethod('PUT', new apigateway.LambdaIntegration(handler));
    item.addMethod('DELETE', new apigateway.LambdaIntegration(handler));

    // 出力
    new cdk.CfnOutput(this, 'ApiUrl', {
      value: api.url,
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

// L3: ALB + Fargate サービスを一行で作成
const service = new ecsPatterns.ApplicationLoadBalancedFargateService(
  this, 'WebService', {
    cluster: new ecs.Cluster(this, 'Cluster', { vpc }),
    taskImageOptions: {
      image: ecs.ContainerImage.fromAsset('./app'),
      containerPort: 8080,
      environment: {
        NODE_ENV: 'production',
      },
    },
    cpu: 512,
    memoryLimitMiB: 1024,
    desiredCount: 3,
    publicLoadBalancer: true,
  }
);

// Auto Scaling の追加
const scaling = service.service.autoScaleTaskCount({
  minCapacity: 2,
  maxCapacity: 20,
});
scaling.scaleOnCpuUtilization('CpuScaling', {
  targetUtilizationPercent: 70,
});
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
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
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
            },
        )

        # 権限付与
        table.grant_read_write_data(handler)

        # API Gateway
        api = apigateway.RestApi(
            self, "ItemsApi",
            rest_api_name=f"items-api-{environment_name}",
        )

        items = api.root.add_resource("items")
        items.add_method("GET", apigateway.LambdaIntegration(handler))
        items.add_method("POST", apigateway.LambdaIntegration(handler))

        CfnOutput(self, "ApiUrl", value=api.url)
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

describe('AppStack', () => {
  let template: Template;

  beforeAll(() => {
    const app = new cdk.App();
    const networkStack = new NetworkStack(app, 'TestNetworkStack', {
      environmentName: 'test',
    });
    const appStack = new AppStack(app, 'TestAppStack', {
      vpc: networkStack.vpc,
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

# CDK diff (デプロイ前の差分確認)
cdk diff

# CDK synth (テンプレート生成確認)
cdk synth
```

---

## 6. CDK デプロイコマンド

```bash
# ブートストラップ (初回のみ、各アカウント・リージョンごと)
cdk bootstrap aws://123456789012/ap-northeast-1

# テンプレートの合成
cdk synth

# 差分確認
cdk diff

# デプロイ
cdk deploy --all

# 特定スタックのデプロイ
cdk deploy AppStack

# 承認なしでデプロイ (CI/CD 向け)
cdk deploy --all --require-approval never

# スタックの削除
cdk destroy --all
```

---

## 7. アンチパターン

### 7.1 ハードコードされた値

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

### 7.2 L1 コンストラクトの多用

**問題点**: L1 (CfnXxx) コンストラクトを使うと、CDK の恩恵(自動的な IAM ポリシー生成、デフォルト値、型安全性)を失い、CloudFormation を直接書くのと変わらなくなる。

**改善**: 可能な限り L2 コンストラクトを使い、L2 で対応していないプロパティのみ `addOverride` や `node.defaultChild` でカスタマイズする。

---

## 8. FAQ

### Q1. CDK と CloudFormation のどちらから始めるべきですか？

プログラミング経験があるなら CDK から始めることを推奨する。CDK は CloudFormation テンプレートを内部で生成するため、CDK を使いながら CloudFormation の概念も自然に学べる。`cdk synth` で生成されるテンプレートを確認することで理解が深まる。

### Q2. CDK のバージョンアップはどう管理すべきですか？

CDK v2 では全モジュールが `aws-cdk-lib` に統合されているため、バージョン管理が簡素化されている。`package.json` の `aws-cdk-lib` バージョンを更新し、テストを実行して互換性を確認する。マイナーバージョンアップは後方互換性が保たれている。

### Q3. CDK でステートフルリソースを安全に管理するには？

DynamoDB テーブルや RDS インスタンスなどのステートフルリソースには `removalPolicy: RemovalPolicy.RETAIN` を設定し、スタック削除時にリソースが残るようにする。また、リソースの論理 ID が変わらないよう、コンストラクト ID の変更には注意が必要である。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| CDK とは | プログラミング言語で AWS インフラを定義するフレームワーク |
| コンストラクト | L1(CFn 1:1)、L2(高レベル抽象化)、L3(パターン) |
| 言語 | TypeScript, Python, Java, Go, C# |
| テスト | スナップショットテスト + ファイングレインドアサーション |
| デプロイ | cdk synth -> cdk diff -> cdk deploy |
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
