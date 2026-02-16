# Elastic Beanstalk

> アプリケーションのデプロイ・スケーリング・監視を自動化する AWS の PaaS サービスを使いこなす

## この章で学ぶこと

1. Elastic Beanstalk の対応プラットフォームとアーキテクチャを理解し、適切な構成を選択できる
2. 4つのデプロイ戦略の特性を比較し、ダウンタイムなしのデプロイを実現できる
3. .ebextensions と環境変数を使ったカスタマイズとモニタリング設定ができる
4. Blue/Green デプロイによる安全なリリースとロールバックを実装できる
5. Docker プラットフォームを使ったコンテナベースのデプロイを実現できる

---

## 1. Elastic Beanstalk とは

### 1.1 アーキテクチャ概要

```
Elastic Beanstalk 環境構成
+----------------------------------------------------------+
|  Elastic Beanstalk Environment                            |
|                                                           |
|  +--------------------------------------------------+    |
|  |  ALB (ロードバランサー)                              |   |
|  +--------------------------------------------------+    |
|              |              |              |               |
|  +-----------v--+ +---------v----+ +-------v------+       |
|  | EC2 Instance | | EC2 Instance | | EC2 Instance |       |
|  | (AZ-1a)      | | (AZ-1c)      | | (AZ-1d)     |       |
|  | +----------+ | | +----------+ | | +----------+ |       |
|  | | App      | | | | App      | | | | App      | |       |
|  | | Runtime  | | | | Runtime  | | | | Runtime  | |       |
|  | | OS       | | | | OS       | | | | OS       | |       |
|  | +----------+ | | +----------+ | | +----------+ |       |
|  +--------------+ +--------------+ +--------------+       |
|                                                           |
|  +--------------------------------------------------+    |
|  | Auto Scaling Group                                |    |
|  +--------------------------------------------------+    |
|                                                           |
|  +--------------------------------------------------+    |
|  | Security Groups + CloudWatch + S3 (ログ)          |    |
|  +--------------------------------------------------+    |
+----------------------------------------------------------+
```

### 1.2 Elastic Beanstalk の責任分界

```
+------------------------+------------------------+
|    ユーザーの責任        |   Beanstalk が管理      |
+------------------------+------------------------+
| アプリケーションコード   | EC2 プロビジョニング     |
| 環境変数の設定          | Auto Scaling 設定        |
| デプロイ戦略の選択      | ロードバランサー管理      |
| .ebextensions 設定     | OS パッチ (マネージド更新) |
| アプリケーション監視     | ヘルスモニタリング        |
| カスタムドメイン設定     | ログ収集                 |
| SSL 証明書の準備        | セキュリティグループ作成  |
+------------------------+------------------------+
```

### 1.3 環境タイプ

| 環境タイプ | 説明 | 構成要素 | ユースケース |
|-----------|------|---------|------------|
| Web サーバー環境 | HTTP リクエストを処理 | ALB + EC2 + ASG | Web アプリ、API |
| ワーカー環境 | バックグラウンドジョブを処理 | SQS + EC2 + ASG | バッチ処理、非同期タスク |

```
Web サーバー環境 vs ワーカー環境

Web サーバー環境:
  クライアント → ALB → EC2 (アプリケーション)
                          ↓ (非同期タスクを SQS に投入)
ワーカー環境:
  SQS キュー → sqsd デーモン → EC2 (ワーカーアプリ)
                                  ↓ (処理完了で SQS からメッセージ削除)
```

---

## 2. 対応プラットフォーム

### 2.1 サポートプラットフォーム一覧

| 言語/フレームワーク | プラットフォーム | コンテナ | デフォルトポート |
|-------------------|---------------|---------|---------------|
| Node.js | Node.js 18/20 on Amazon Linux 2023 | AL2023 | 8080 |
| Python | Python 3.11/3.12 on Amazon Linux 2023 | AL2023 | 8000 |
| Java | Corretto 17/21 on Amazon Linux 2023 | AL2023 | 5000 |
| Go | Go 1.21 on Amazon Linux 2023 | AL2023 | 5000 |
| .NET | .NET 6/8 on Amazon Linux 2023 | AL2023 | 5000 |
| Ruby | Ruby 3.2/3.3 on Amazon Linux 2023 | AL2023 | 8080 |
| PHP | PHP 8.2/8.3 on Amazon Linux 2023 | AL2023 | 80 (Apache) |
| Docker | Docker on Amazon Linux 2023 | AL2023 | 80 |
| Multi-container Docker | ECS managed Docker | ECS | 各コンテナによる |

### 2.2 プラットフォーム選定ガイド

```
プラットフォーム選定フロー
==========================

コンテナ化されている？
├─ Yes → Docker プラットフォーム
│   ├─ 単一コンテナ → Docker on AL2023
│   └─ 複数コンテナ → Multi-container Docker (ECS)
│
└─ No → 言語/フレームワークに応じて選択
    ├─ Python (Django/Flask) → Python on AL2023
    ├─ Node.js (Express/NestJS) → Node.js on AL2023
    ├─ Java (Spring Boot) → Corretto on AL2023
    ├─ Go (Gin/Echo) → Go on AL2023
    ├─ .NET (ASP.NET Core) → .NET on AL2023
    ├─ Ruby (Rails) → Ruby on AL2023
    └─ PHP (Laravel) → PHP on AL2023

注意: Amazon Linux 2 は 2025年6月にサポート終了
→ 必ず Amazon Linux 2023 ベースを選択すること
```

### 2.3 コード例: EB CLI のインストールと初期化

```bash
# EB CLI インストール
pip install awsebcli

# バージョン確認
eb --version

# プロジェクトを初期化
cd /path/to/my-app
eb init

# 対話形式で設定
# 1. リージョン選択: ap-northeast-1
# 2. アプリケーション名: my-web-app
# 3. プラットフォーム: Python 3.12
# 4. CodeCommit 連携: No
# 5. SSH キーペア: 既存のキーを選択

# 非対話形式で初期化
eb init my-web-app \
  --platform "Python 3.12 running on 64bit Amazon Linux 2023" \
  --region ap-northeast-1 \
  --keyname my-key-pair
```

### 2.4 コード例: 環境の作成

```bash
# 環境を作成
eb create production-env \
  --instance-type t3.small \
  --scale 2 \
  --elb-type application \
  --region ap-northeast-1 \
  --tags Environment=production,Team=backend \
  --vpc.id vpc-0123456789abcdef0 \
  --vpc.elbsubnets subnet-pub-a,subnet-pub-c \
  --vpc.ec2subnets subnet-priv-a,subnet-priv-c \
  --vpc.elbpublic \
  --vpc.publicip

# 環境の状態確認
eb status

# ログの確認
eb logs

# ヘルスチェック
eb health

# 環境一覧
eb list

# 環境の終了（注意: リソースが全て削除される）
eb terminate production-env
```

### 2.5 アプリケーション構成例（Python/Django）

```
my-django-app/
├── .ebextensions/
│   ├── 01-packages.config
│   ├── 02-django.config
│   ├── 03-logging.config
│   └── 04-https.config
├── .platform/
│   ├── hooks/
│   │   ├── prebuild/
│   │   │   └── 01_install_deps.sh
│   │   ├── predeploy/
│   │   │   └── 01_migrate.sh
│   │   └── postdeploy/
│   │       └── 01_health_check.sh
│   └── nginx/
│       └── conf.d/
│           └── custom.conf
├── myapp/
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── production.py
│   │   └── development.py
│   ├── urls.py
│   └── wsgi.py
├── manage.py
├── requirements.txt
├── Procfile
└── .ebignore
```

```python
# Procfile — EB がアプリケーションの起動方法を知るためのファイル
# web: gunicorn myapp.wsgi --bind :8000 --workers 3 --threads 2
```

```bash
# .ebignore — デプロイに含めないファイル
.git
__pycache__
*.pyc
.env
.venv
node_modules
.DS_Store
*.sqlite3
```

---

## 3. デプロイ戦略

### 3.1 5つのデプロイ戦略比較

```
All at Once (一括更新)
+---------+---------+---------+
| v1→v2   | v1→v2   | v1→v2   |  全インスタンスを同時に更新
+---------+---------+---------+  ダウンタイム: あり

Rolling (ローリング)
+---------+---------+---------+
| v1→v2   | v1      | v1      |  バッチごとに順次更新
+---------+---------+---------+  ダウンタイム: なし（容量一時低下）
    ↓
+---------+---------+---------+
| v2      | v1→v2   | v1      |
+---------+---------+---------+
    ↓
+---------+---------+---------+
| v2      | v2      | v1→v2   |
+---------+---------+---------+

Rolling with Additional Batch (追加バッチ付き)
+---------+---------+---------+---------+
| v1→v2   | v1      | v1      | v2(新)  |  追加インスタンスで容量維持
+---------+---------+---------+---------+  ダウンタイム: なし

Immutable (イミュータブル)
+---------+---------+---------+   +---------+---------+---------+
| v1      | v1      | v1      |   | v2      | v2      | v2      |
+---------+---------+---------+   +---------+---------+---------+
  旧 ASG (ヘルスチェック後削除)      新 ASG (ヘルスチェック後に切替)
  ダウンタイム: なし、ロールバック: 高速

Traffic Splitting (トラフィック分割)
+---------+---------+---------+   +---------+
| v1      | v1      | v1      |   | v2      |
+---------+---------+---------+   +---------+
  旧 TG (90% のトラフィック)         新 TG (10% のトラフィック)
  → 段階的にトラフィックを移行       → カナリアリリース的な手法
```

### 3.2 デプロイ戦略比較表

| 戦略 | ダウンタイム | デプロイ速度 | コスト | ロールバック | 推奨環境 |
|------|-----------|-----------|--------|-----------|---------|
| All at Once | あり | 最速 | 追加コストなし | 再デプロイ必要 | 開発環境 |
| Rolling | なし | 中 | 追加コストなし | 再デプロイ必要 | ステージング |
| Rolling + Batch | なし | 中 | 一時的追加 | 再デプロイ必要 | 本番（低リスク） |
| Immutable | なし | 遅い | 一時的に2倍 | 高速（旧環境に戻す） | 本番（推奨） |
| Traffic Splitting | なし | 遅い | 一時的に追加 | 高速 | 本番（カナリア） |
| Blue/Green | なし | 遅い | 常に2倍 | 最速（URL スワップ） | 本番（最高安全性） |

### 3.3 コード例: デプロイ設定 (.ebextensions)

```yaml
# .ebextensions/01-deploy.config
option_settings:
  aws:elasticbeanstalk:command:
    DeploymentPolicy: RollingWithAdditionalBatch
    BatchSizeType: Percentage
    BatchSize: 25
    Timeout: 600
  aws:autoscaling:updatepolicy:rollingupdate:
    RollingUpdateEnabled: true
    MaxBatchSize: 1
    MinInstancesInService: 1
```

### 3.4 コード例: Traffic Splitting の設定

```yaml
# .ebextensions/traffic-splitting.config
option_settings:
  aws:elasticbeanstalk:command:
    DeploymentPolicy: TrafficSplitting
  aws:elasticbeanstalk:trafficsplitting:
    NewVersionPercent: 10
    EvaluationTime: 10
```

### 3.5 デプロイの実行

```bash
# 現在のディレクトリのコードをデプロイ
eb deploy

# 特定のバージョンをデプロイ
eb deploy --version v1.2.0

# ステージングにラベル付きでデプロイ
eb deploy staging-env --label "release-2026-02-16" --message "Feature X release"

# デプロイ状態の監視
eb events --follow

# ロールバック（前のバージョンに戻す）
eb deploy --version previous-version-label

# アプリケーションバージョンの一覧
aws elasticbeanstalk describe-application-versions \
  --application-name my-web-app \
  --query 'ApplicationVersions[].[VersionLabel,DateCreated,Status]' \
  --output table
```

---

## 4. 設定カスタマイズ

### 4.1 .ebextensions の構造

```
my-app/
├── .ebextensions/
│   ├── 01-packages.config      # パッケージインストール
│   ├── 02-files.config         # ファイル配置
│   ├── 03-commands.config      # コマンド実行
│   ├── 04-options.config       # 環境設定
│   ├── 05-resources.config     # CloudFormation リソース
│   └── 06-logging.config       # ログ設定
├── .platform/
│   ├── hooks/
│   │   ├── prebuild/           # ビルド前フック
│   │   ├── predeploy/          # デプロイ前フック
│   │   └── postdeploy/         # デプロイ後フック
│   ├── confighooks/
│   │   ├── prebuild/           # 設定変更時のビルド前フック
│   │   └── predeploy/          # 設定変更時のデプロイ前フック
│   └── nginx/
│       ├── nginx.conf          # NGINX メイン設定（完全上書き）
│       └── conf.d/
│           └── custom.conf     # NGINX カスタム設定（追加）
├── application.py
└── requirements.txt

.ebextensions の実行順序:
1. packages       — OS パッケージインストール
2. groups         — Linux グループ作成
3. users          — Linux ユーザー作成
4. sources        — アーカイブ展開
5. files          — ファイル配置
6. commands       — アプリデプロイ前のコマンド
7. services       — サービス起動/有効化
8. container_commands — アプリデプロイ後のコマンド（leader_only 対応）
```

### 4.2 コード例: パッケージインストールと設定

```yaml
# .ebextensions/01-packages.config
packages:
  yum:
    git: []
    jq: []
    htop: []

files:
  "/etc/nginx/conf.d/proxy.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      client_max_body_size 50M;
      proxy_read_timeout 300;
      proxy_connect_timeout 60;
      proxy_send_timeout 300;

  "/opt/elasticbeanstalk/tasks/taillogs.d/app-logs.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      /var/log/app/*.log

commands:
  01_install_node:
    command: |
      curl -fsSL https://rpm.nodesource.com/setup_20.x | bash -
      yum install -y nodejs
    test: "! node --version 2>/dev/null"

container_commands:
  01_migrate:
    command: "python manage.py migrate"
    leader_only: true
  02_collectstatic:
    command: "python manage.py collectstatic --noinput"
  03_create_superuser:
    command: "python manage.py createsuperuser --noinput || true"
    leader_only: true
    env:
      DJANGO_SUPERUSER_USERNAME: admin
      DJANGO_SUPERUSER_EMAIL: admin@example.com
      DJANGO_SUPERUSER_PASSWORD: InitialPassword123!
```

### 4.3 コード例: 環境変数の設定

```bash
# CLI で環境変数を設定
eb setenv \
  DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/mydb \
  REDIS_URL=redis://elasticache-endpoint:6379 \
  SECRET_KEY=my-secret-key-123 \
  DEBUG=false \
  ALLOWED_HOSTS=.example.com \
  AWS_STORAGE_BUCKET_NAME=my-static-bucket \
  SENTRY_DSN=https://xxx@sentry.io/123

# 環境変数の確認
eb printenv

# .ebextensions で環境変数を設定
# .ebextensions/04-options.config
option_settings:
  aws:elasticbeanstalk:application:environment:
    DJANGO_SETTINGS_MODULE: myapp.settings.production
    PYTHONPATH: /var/app/current
    LOG_LEVEL: INFO

# AWS CLI で環境変数を設定
aws elasticbeanstalk update-environment \
  --environment-name production-env \
  --option-settings '[
    {"Namespace": "aws:elasticbeanstalk:application:environment", "OptionName": "DEBUG", "Value": "false"},
    {"Namespace": "aws:elasticbeanstalk:application:environment", "OptionName": "LOG_LEVEL", "Value": "WARNING"}
  ]'
```

### 4.4 コード例: NGINX カスタム設定

```nginx
# .platform/nginx/conf.d/custom.conf
upstream backend {
    server 127.0.0.1:8000;
    keepalive 256;
}

server {
    listen 80;

    # gzip 圧縮
    gzip on;
    gzip_comp_level 6;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    # セキュリティヘッダー
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # 静的ファイル
    location /static/ {
        alias /var/app/current/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
        access_log off;
    }

    # メディアファイル
    location /media/ {
        alias /var/app/current/media/;
        expires 7d;
        add_header Cache-Control "public";
    }

    # ヘルスチェック（ログ不要）
    location /health {
        proxy_pass http://backend;
        access_log off;
    }

    # アプリケーション
    location / {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        proxy_buffering off;
    }

    # WebSocket
    location /ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 4.5 コード例: CloudFormation リソースの追加

```yaml
# .ebextensions/05-resources.config
Resources:
  sslSecurityGroupIngress:
    Type: AWS::EC2::SecurityGroupIngress
    Properties:
      GroupId: {"Fn::GetAtt": ["AWSEBSecurityGroup", "GroupId"]}
      IpProtocol: tcp
      ToPort: 443
      FromPort: 443
      CidrIp: 0.0.0.0/0

  AWSEBAutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300

  # CloudWatch アラーム
  CPUAlarmHigh:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmDescription: "CPU usage > 80%"
      Namespace: AWS/EC2
      MetricName: CPUUtilization
      Dimensions:
        - Name: AutoScalingGroupName
          Value: {"Ref": "AWSEBAutoScalingGroup"}
      Statistic: Average
      Period: 300
      EvaluationPeriods: 2
      Threshold: 80
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - {"Ref": "NotificationTopic"}

  NotificationTopic:
    Type: AWS::SNS::Topic
    Properties:
      Subscription:
        - Protocol: email
          Endpoint: alerts@example.com
```

### 4.6 コード例: Auto Scaling の設定

```yaml
# .ebextensions/autoscaling.config
option_settings:
  # インスタンスタイプ
  aws:autoscaling:launchconfiguration:
    InstanceType: t3.small
    IamInstanceProfile: aws-elasticbeanstalk-ec2-role
    SecurityGroups: sg-0123456789abcdef0
    RootVolumeType: gp3
    RootVolumeSize: 30

  # Auto Scaling 設定
  aws:autoscaling:asg:
    MinSize: 2
    MaxSize: 8
    Cooldown: 300

  # スケーリングトリガー
  aws:autoscaling:trigger:
    MeasureName: CPUUtilization
    Statistic: Average
    Unit: Percent
    Period: 1
    BreachDuration: 5
    UpperThreshold: 70
    UpperBreachScaleIncrement: 1
    LowerThreshold: 30
    LowerBreachScaleIncrement: -1

  # スケジュールスケーリング
  aws:autoscaling:scheduledaction:
    # 平日の朝にスケールアウト
    - ResourceId: AWSEBAutoScalingGroup
      Schedule: "cron(0 0 * * MON-FRI)"
      MinSize: 4
      MaxSize: 8
      DesiredCapacity: 4

  # ロードバランサー設定
  aws:elasticbeanstalk:environment:
    LoadBalancerType: application

  aws:elbv2:listener:443:
    Protocol: HTTPS
    SSLCertificateArns: arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx
    SSLPolicy: ELBSecurityPolicy-TLS13-1-2-2021-06

  aws:elbv2:listener:80:
    Protocol: HTTP
    DefaultProcess: default
    ListenerEnabled: true

  # ヘルスチェック
  aws:elasticbeanstalk:application:
    Application Healthcheck URL: /health

  aws:elasticbeanstalk:environment:process:default:
    HealthCheckPath: /health
    HealthCheckInterval: 30
    HealthyThresholdCount: 2
    UnhealthyThresholdCount: 3
    DeregistrationDelay: 30
    StickinessEnabled: true
    StickinessLBCookieDuration: 3600
```

### 4.7 Platform Hooks の使い方

```bash
#!/bin/bash
# .platform/hooks/predeploy/01_migrate.sh
# デプロイ前にデータベースマイグレーションを実行

set -euo pipefail

echo "Running database migrations..."
cd /var/app/staging

# 仮想環境を有効化
source /var/app/venv/*/bin/activate

# マイグレーション実行
python manage.py migrate --noinput

echo "Database migrations completed successfully"
```

```bash
#!/bin/bash
# .platform/hooks/postdeploy/01_health_check.sh
# デプロイ後にヘルスチェックを実行

set -euo pipefail

echo "Running post-deploy health check..."

# アプリケーションが起動するまで待機
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Health check passed!"
    exit 0
  fi
  echo "Waiting for application to start... ($i/30)"
  sleep 2
done

echo "Health check failed after 60 seconds"
exit 1
```

---

## 5. モニタリングとトラブルシューティング

### 5.1 ヘルスモニタリング

```
EB ヘルスダッシュボード

  環境全体: ● OK (Green)

  インスタンス別ヘルス:
  +-------------+--------+---------+----------+--------+
  | Instance ID | Status | CPU (%) | Req/sec  | P99(ms)|
  +-------------+--------+---------+----------+--------+
  | i-aaa       | ● OK   | 35      | 120      | 45     |
  | i-bbb       | ● OK   | 42      | 115      | 52     |
  | i-ccc       | ▲ Warn | 78      | 95       | 250    |
  +-------------+--------+---------+----------+--------+

  ヘルスカラー:
  ● Green  = 正常
  ● Yellow = 警告（デプロイ中含む）
  ● Red    = 異常（アクションが必要）
  ● Grey   = 情報不足

  拡張ヘルスのメトリクス:
  - ApplicationRequests*: 各 HTTP ステータスコードのリクエスト数
  - ApplicationLatencyP*: レイテンシのパーセンタイル (P50, P90, P99)
  - InstanceHealth: インスタンスレベルのヘルス情報
  - CPUUtilization, LoadAverage, RootFilesystemUtil
```

### 5.2 コード例: ログの取得と確認

```bash
# 最新のログを取得
eb logs

# 完全なログバンドルを取得
eb logs --all

# 特定のインスタンスのログ
eb logs --instance i-0123456789abcdef0

# ログをストリーミング
eb logs --stream

# SSH で直接確認
eb ssh
# EB エンジンログ:       /var/log/eb-engine.log
# NGINX アクセスログ:     /var/log/nginx/access.log
# NGINX エラーログ:       /var/log/nginx/error.log
# アプリ stdout:          /var/log/web.stdout.log
# アプリ stderr:          /var/log/web.stderr.log
# EB フックログ:          /var/log/eb-hooks.log
# デプロイログ:           /var/log/eb-activity.log
```

### 5.3 CloudWatch Logs ストリーミング

```yaml
# .ebextensions/06-logging.config
option_settings:
  aws:elasticbeanstalk:cloudwatch:logs:
    StreamLogs: true
    DeleteOnTerminate: false
    RetentionInDays: 30

  aws:elasticbeanstalk:cloudwatch:logs:health:
    HealthStreamingEnabled: true
    DeleteOnTerminate: false
    RetentionInDays: 7

# カスタムログファイルの追加
files:
  "/opt/elasticbeanstalk/tasks/bundlelogs.d/app-logs.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      /var/app/current/logs/*.log

  "/opt/elasticbeanstalk/tasks/taillogs.d/app-logs.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      /var/app/current/logs/*.log
```

### 5.4 トラブルシューティングチェックリスト

| 症状 | 確認箇所 | 対処法 |
|------|---------|--------|
| デプロイ失敗 | `/var/log/eb-engine.log` | コマンド実行エラーを確認 |
| 502 Bad Gateway | NGINX → アプリの接続 | ポート番号、アプリ起動状態を確認 |
| ヘルスチェック失敗 | `/health` エンドポイント | SG、パス、レスポンスコードを確認 |
| 環境が Red | 拡張ヘルス詳細 | `eb health` でインスタンス別状態確認 |
| メモリ不足 | CloudWatch メトリクス | インスタンスタイプのスケールアップ |
| ディスク容量不足 | `/var/log`, `/tmp` | 古いデプロイバージョンの削除 |

```bash
# トラブルシューティング用のコマンド集

# 環境の詳細状態
eb health --refresh

# イベント一覧（エラーを確認）
eb events -f

# 環境設定のダンプ
eb config

# 環境の再構築（最終手段）
eb rebuild

# SSH 接続してログを確認
eb ssh --command "tail -100 /var/log/eb-engine.log"

# AWS CLI でインスタンスのヘルスを確認
aws elasticbeanstalk describe-instances-health \
  --environment-name production-env \
  --attribute-names All \
  --output table
```

---

## 6. Blue/Green デプロイ

```
Blue/Green デプロイフロー

  1. 現在の環境 (Blue)
     production.example.com → Blue 環境

  2. 新環境を作成 (Green)
     eb clone production-env --clone-name green-env

  3. Green 環境にデプロイ・テスト
     eb deploy green-env

  4. URL スワップ
     eb swap production-env --destination-name green-env
     production.example.com → Green 環境

  5. 問題があれば再度スワップでロールバック
     eb swap production-env --destination-name green-env
     production.example.com → Blue 環境（元に戻る）
```

```bash
# Blue/Green デプロイの実行
# 1. 現在の環境をクローン
eb clone production-env --clone-name green-env \
  --tags Environment=green,Release=v2.0

# 2. Green 環境の作成完了を待つ
eb status green-env

# 3. Green 環境にデプロイ
eb deploy green-env --label v2.0

# 4. Green 環境でテスト
GREEN_URL=$(aws elasticbeanstalk describe-environments \
  --environment-names green-env \
  --query 'Environments[0].CNAME' --output text)
curl -f "http://$GREEN_URL/health"

# 5. テスト完了後、URL スワップ
eb swap production-env --destination-name green-env

# 6. 旧環境を削除（問題なければ）
eb terminate green-env --force
```

### 6.1 Blue/Green デプロイの自動化スクリプト

```bash
#!/bin/bash
# blue-green-deploy.sh
set -euo pipefail

APP_NAME="my-web-app"
BLUE_ENV="production-env"
GREEN_ENV="green-env"
VERSION_LABEL="v$(date +%Y%m%d-%H%M%S)"
HEALTH_CHECK_URL="/health"

echo "=== Blue/Green Deploy: $VERSION_LABEL ==="

# 1. Green 環境の作成（既存の場合はスキップ）
if aws elasticbeanstalk describe-environments \
  --environment-names $GREEN_ENV \
  --query 'Environments[?Status!=`Terminated`]' \
  --output text | grep -q "$GREEN_ENV"; then
  echo "Green environment already exists, reusing..."
else
  echo "Creating green environment..."
  eb clone $BLUE_ENV --clone-name $GREEN_ENV
  echo "Waiting for green environment to be ready..."
  aws elasticbeanstalk wait environment-updated \
    --environment-name $GREEN_ENV
fi

# 2. Green 環境にデプロイ
echo "Deploying to green environment..."
eb deploy $GREEN_ENV --label $VERSION_LABEL

# 3. ヘルスチェック
echo "Running health checks..."
GREEN_URL=$(aws elasticbeanstalk describe-environments \
  --environment-names $GREEN_ENV \
  --query 'Environments[0].CNAME' --output text)

for i in $(seq 1 10); do
  if curl -sf "http://$GREEN_URL$HEALTH_CHECK_URL" > /dev/null; then
    echo "Health check passed!"
    break
  fi
  echo "Health check attempt $i/10 failed, retrying..."
  sleep 10
done

# 4. URL スワップ
echo "Swapping URLs..."
eb swap $BLUE_ENV --destination-name $GREEN_ENV

echo "=== Blue/Green Deploy Complete ==="
echo "New production URL: $GREEN_URL"
```

---

## 7. Docker プラットフォーム

### 7.1 単一コンテナ Docker

```dockerfile
# Dockerfile
FROM node:20-slim AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM node:20-slim
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package.json .
EXPOSE 8080
CMD ["node", "dist/server.js"]
```

```json
// Dockerrun.aws.json (v1 — 単一コンテナ)
{
  "AWSEBDockerrunVersion": "1",
  "Image": {
    "Name": "123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest",
    "Update": "true"
  },
  "Ports": [
    {
      "ContainerPort": 8080,
      "HostPort": 80
    }
  ],
  "Logging": "/var/log/app"
}
```

### 7.2 マルチコンテナ Docker (docker-compose)

```yaml
# docker-compose.yml
version: '3.8'
services:
  web:
    image: 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:latest
    ports:
      - "80:8080"
    environment:
      - NODE_ENV=production
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    depends_on:
      - redis
    restart: always
    logging:
      driver: awslogs
      options:
        awslogs-group: /eb/my-app/web
        awslogs-region: ap-northeast-1
        awslogs-stream-prefix: web

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: always

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - web
    restart: always
```

---

## 8. マネージドプラットフォーム更新

### 8.1 マネージドプラットフォーム更新の設定

```yaml
# .ebextensions/managed-updates.config
option_settings:
  aws:elasticbeanstalk:managedactions:
    ManagedActionsEnabled: true
    PreferredStartTime: "Sun:02:00"
    ServiceRoleForManagedUpdates: "aws-elasticbeanstalk-service-role"

  aws:elasticbeanstalk:managedactions:platformupdate:
    UpdateLevel: minor
    InstanceRefreshEnabled: true
```

```bash
# プラットフォーム更新の状態確認
aws elasticbeanstalk describe-environment-managed-actions \
  --environment-name production-env

# 手動でプラットフォーム更新を適用
aws elasticbeanstalk apply-environment-managed-action \
  --environment-name production-env \
  --action-id managed-action-id

# 利用可能なプラットフォームバージョンの確認
aws elasticbeanstalk list-available-solution-stacks \
  --query 'SolutionStacks[?contains(@, `Python`)]'
```

---

## 9. RDS との連携

### 9.1 EB 環境に RDS を直接関連付ける方法（開発用）

```yaml
# .ebextensions/rds.config（開発環境のみ推奨）
option_settings:
  aws:rds:dbinstance:
    DBEngine: postgres
    DBEngineVersion: 16.1
    DBInstanceClass: db.t3.micro
    DBAllocatedStorage: 20
    DBPassword: initial-password
    DBUser: ebroot
    DBDeletionPolicy: Delete
    MultiAZDatabase: false
```

### 9.2 外部 RDS を使用する方法（本番推奨）

```bash
# 環境変数で RDS 接続情報を設定
eb setenv \
  RDS_HOSTNAME=my-db.xxxx.ap-northeast-1.rds.amazonaws.com \
  RDS_PORT=5432 \
  RDS_DB_NAME=myapp \
  RDS_USERNAME=admin \
  RDS_PASSWORD=secret-from-secrets-manager

# Secrets Manager から動的に取得する場合
# .platform/hooks/predeploy/00_fetch_secrets.sh
#!/bin/bash
SECRET=$(aws secretsmanager get-secret-value \
  --secret-id my-app/production/db \
  --query 'SecretString' --output text)
export RDS_PASSWORD=$(echo $SECRET | jq -r '.password')
```

---

## 10. アンチパターン

### アンチパターン 1: 全設定をコンソールで手動管理する

コンソールで変更した設定は再現性がなく、環境再構築時に漏れが発生する。`.ebextensions` と `.platform` でコード化し、Git で管理すべきである。

```
# 悪い例
コンソールで NGINX 設定を手動変更
→ 環境再構築時に設定が消える
→ 他のチームメンバーが設定を知らない

# 良い例
.platform/nginx/conf.d/custom.conf に設定を記述
→ デプロイのたびに自動適用
→ Git で変更履歴を追跡可能
```

### アンチパターン 2: All at Once デプロイを本番で使う

全インスタンスが同時に更新されるため、デプロイ中にダウンタイムが発生する。本番環境では Rolling with Additional Batch または Immutable を使用すべきである。

### アンチパターン 3: EB 環境に RDS を直接関連付ける（本番）

EB 環境を終了すると関連付けられた RDS も削除される。本番環境では必ず外部の RDS を使用し、環境変数で接続情報を渡すべきである。

```
# 悪い例（本番）
EB 環境に RDS を関連付け
→ eb terminate で RDS も一緒に削除される
→ Blue/Green デプロイで別の DB が作られてしまう

# 良い例（本番）
RDS は Terraform/CloudFormation で別管理
→ EB 環境の終了に影響されない
→ Blue/Green の両環境で同じ DB を共有
```

### アンチパターン 4: 環境変数にシークレットを直接設定する

```bash
# 悪い例
eb setenv DB_PASSWORD=PlainTextPassword123!
# → EB コンソールで誰でも閲覧可能

# 良い例 — Secrets Manager を使用
eb setenv DB_SECRET_ARN=arn:aws:secretsmanager:ap-northeast-1:123456789012:secret:my-db-creds
# → アプリ側で Secrets Manager から動的に取得
```

### アンチパターン 5: .ebignore を設定しない

```bash
# 悪い例 — 全ファイルをデプロイ
# .git ディレクトリや node_modules が含まれ、デプロイが遅い

# 良い例 — .ebignore で不要ファイルを除外
# .ebignore
.git
__pycache__
*.pyc
.env
.venv
node_modules
tests/
docs/
*.sqlite3
.DS_Store
```

---

## 11. FAQ

### Q1. Elastic Beanstalk と ECS / Fargate のどちらを選ぶべきか？

| 観点 | Elastic Beanstalk | ECS / Fargate |
|------|------------------|--------------|
| 運用複雑度 | 低い | 中〜高い |
| カスタマイズ性 | 中程度 | 高い |
| コンテナ対応 | Docker プラットフォーム | ネイティブ |
| サイドカー | 困難 | 容易 |
| サービスメッシュ | 非対応 | App Mesh 対応 |
| 対象ユーザー | 小〜中規模チーム | 大規模・マイクロサービス |

Beanstalk は「アプリケーションをデプロイするだけ」のシンプルさが利点。コンテナオーケストレーションの詳細な制御（サイドカーパターン、サービスメッシュ等）が必要なら ECS/Fargate を選択する。

### Q2. Elastic Beanstalk のコストは？

Beanstalk 自体は無料で、裏側の EC2、ALB、RDS などの料金のみが発生する。ただし Beanstalk が自動作成する ALB やスケーリング設定が想定以上のコストを生む場合があるので、作成されるリソースを確認する。不要な環境は速やかに終了すること。

### Q3. カスタムドメインと HTTPS をどう設定するか？

Route 53 でドメインを ALB に ALIAS レコードで向け、ACM (AWS Certificate Manager) で SSL 証明書を取得して ALB のリスナーに設定する。`.ebextensions` で HTTPS リスナーの設定を自動化できる。

```yaml
# .ebextensions/https.config
option_settings:
  aws:elbv2:listener:443:
    Protocol: HTTPS
    SSLCertificateArns: arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx
    SSLPolicy: ELBSecurityPolicy-TLS13-1-2-2021-06

  # HTTP → HTTPS リダイレクト
  aws:elbv2:listener:80:
    DefaultProcess: default
    ListenerEnabled: true
    Protocol: HTTP
    Rules: redirect-to-https

  aws:elbv2:listenerrule:redirect-to-https:
    PathPatterns: /*
    Process: default
    Priority: 1
```

### Q4. EB の環境構成を別のアカウントに移行するには？

```bash
# 環境設定をエクスポート
eb config save production-env --cfg saved-config

# エクスポートされたファイルを確認
cat .elasticbeanstalk/saved_configs/saved-config.cfg.yml

# 別のアカウントで設定を適用
eb create new-production-env --cfg saved-config
```

### Q5. EB 環境のインスタンスに SSH する方法は？

```bash
# EB CLI で SSH 接続
eb ssh

# 特定のインスタンスに接続
eb ssh --instance i-0123456789abcdef0

# Session Manager の方が推奨
aws ssm start-session --target i-0123456789abcdef0
```

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| 位置づけ | EC2 + ALB + ASG をまとめて管理する PaaS |
| プラットフォーム | Node.js, Python, Java, Go, .NET, Docker 等（AL2023 必須） |
| デプロイ戦略 | 本番は Immutable か Traffic Splitting 推奨 |
| カスタマイズ | .ebextensions と .platform で宣言的に管理 |
| Blue/Green | eb swap で URL を切り替え、高速ロールバック |
| モニタリング | 拡張ヘルス + CloudWatch Logs ストリーミング |
| RDS | 本番は外部 RDS を使用、環境変数で接続情報を渡す |
| シークレット | Secrets Manager で管理、環境変数に平文を置かない |
| Docker | 単一コンテナまたは docker-compose で柔軟にデプロイ |

---

## 13. CloudFormation / CDK による EB 環境の定義

### 13.1 CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Elastic Beanstalk Application with VPC-aware Environment

Parameters:
  AppName:
    Type: String
    Default: my-web-app
  VpcId:
    Type: AWS::EC2::VPC::Id
  PublicSubnets:
    Type: List<AWS::EC2::Subnet::Id>
    Description: ALB 用パブリックサブネット
  PrivateSubnets:
    Type: List<AWS::EC2::Subnet::Id>
    Description: EC2 用プライベートサブネット
  CertificateArn:
    Type: String
    Description: ACM 証明書の ARN
  InstanceType:
    Type: String
    Default: t3.small
    AllowedValues: [t3.micro, t3.small, t3.medium, t3.large]

Resources:
  Application:
    Type: AWS::ElasticBeanstalk::Application
    Properties:
      ApplicationName: !Ref AppName
      Description: !Sub '${AppName} Elastic Beanstalk Application'
      ResourceLifecycleConfig:
        ServiceRole: !Sub 'arn:aws:iam::${AWS::AccountId}:role/aws-elasticbeanstalk-service-role'
        VersionLifecycleConfig:
          MaxCountRule:
            DeleteSourceFromS3: true
            Enabled: true
            MaxCount: 10

  Environment:
    Type: AWS::ElasticBeanstalk::Environment
    Properties:
      ApplicationName: !Ref Application
      EnvironmentName: !Sub '${AppName}-production'
      SolutionStackName: '64bit Amazon Linux 2023 v6.1.0 running Python 3.12'
      Tier:
        Name: WebServer
        Type: Standard
      OptionSettings:
        # VPC 設定
        - Namespace: aws:ec2:vpc
          OptionName: VPCId
          Value: !Ref VpcId
        - Namespace: aws:ec2:vpc
          OptionName: Subnets
          Value: !Join [',', !Ref PrivateSubnets]
        - Namespace: aws:ec2:vpc
          OptionName: ELBSubnets
          Value: !Join [',', !Ref PublicSubnets]
        - Namespace: aws:ec2:vpc
          OptionName: AssociatePublicIpAddress
          Value: 'false'

        # インスタンス設定
        - Namespace: aws:autoscaling:launchconfiguration
          OptionName: InstanceType
          Value: !Ref InstanceType
        - Namespace: aws:autoscaling:launchconfiguration
          OptionName: IamInstanceProfile
          Value: aws-elasticbeanstalk-ec2-role

        # Auto Scaling
        - Namespace: aws:autoscaling:asg
          OptionName: MinSize
          Value: '2'
        - Namespace: aws:autoscaling:asg
          OptionName: MaxSize
          Value: '6'

        # ALB
        - Namespace: aws:elasticbeanstalk:environment
          OptionName: LoadBalancerType
          Value: application
        - Namespace: aws:elbv2:listener:443
          OptionName: Protocol
          Value: HTTPS
        - Namespace: aws:elbv2:listener:443
          OptionName: SSLCertificateArns
          Value: !Ref CertificateArn

        # デプロイ戦略
        - Namespace: aws:elasticbeanstalk:command
          OptionName: DeploymentPolicy
          Value: Immutable
        - Namespace: aws:elasticbeanstalk:command
          OptionName: Timeout
          Value: '600'

        # ヘルスチェック
        - Namespace: aws:elasticbeanstalk:application
          OptionName: Application Healthcheck URL
          Value: /health

        # 拡張ヘルスレポート
        - Namespace: aws:elasticbeanstalk:healthreporting:system
          OptionName: SystemType
          Value: enhanced

        # CloudWatch Logs
        - Namespace: aws:elasticbeanstalk:cloudwatch:logs
          OptionName: StreamLogs
          Value: 'true'
        - Namespace: aws:elasticbeanstalk:cloudwatch:logs
          OptionName: RetentionInDays
          Value: '30'

        # マネージド更新
        - Namespace: aws:elasticbeanstalk:managedactions
          OptionName: ManagedActionsEnabled
          Value: 'true'
        - Namespace: aws:elasticbeanstalk:managedactions
          OptionName: PreferredStartTime
          Value: 'Sun:02:00'
        - Namespace: aws:elasticbeanstalk:managedactions:platformupdate
          OptionName: UpdateLevel
          Value: minor

Outputs:
  EnvironmentURL:
    Value: !GetAtt Environment.EndpointURL
    Description: EB 環境の URL
  EnvironmentName:
    Value: !Ref Environment
    Description: EB 環境名
```

### 13.2 CDK (TypeScript) による定義

```typescript
import * as cdk from 'aws-cdk-lib';
import * as elasticbeanstalk from 'aws-cdk-lib/aws-elasticbeanstalk';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import { Construct } from 'constructs';

export class EbStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // EC2 インスタンスプロファイル
    const role = new iam.Role(this, 'EbInstanceRole', {
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AWSElasticBeanstalkWebTier'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
      ],
    });

    const instanceProfile = new iam.CfnInstanceProfile(this, 'EbInstanceProfile', {
      roles: [role.roleName],
    });

    // アプリケーション
    const app = new elasticbeanstalk.CfnApplication(this, 'App', {
      applicationName: 'my-web-app',
      resourceLifecycleConfig: {
        serviceRole: `arn:aws:iam::${this.account}:role/aws-elasticbeanstalk-service-role`,
        versionLifecycleConfig: {
          maxCountRule: {
            enabled: true,
            maxCount: 10,
            deleteSourceFromS3: true,
          },
        },
      },
    });

    // 環境
    const env = new elasticbeanstalk.CfnEnvironment(this, 'Env', {
      applicationName: app.applicationName!,
      environmentName: 'my-web-app-production',
      solutionStackName: '64bit Amazon Linux 2023 v6.1.0 running Python 3.12',
      optionSettings: [
        { namespace: 'aws:autoscaling:launchconfiguration', optionName: 'IamInstanceProfile', value: instanceProfile.ref },
        { namespace: 'aws:autoscaling:launchconfiguration', optionName: 'InstanceType', value: 't3.small' },
        { namespace: 'aws:autoscaling:asg', optionName: 'MinSize', value: '2' },
        { namespace: 'aws:autoscaling:asg', optionName: 'MaxSize', value: '6' },
        { namespace: 'aws:elasticbeanstalk:environment', optionName: 'LoadBalancerType', value: 'application' },
        { namespace: 'aws:elasticbeanstalk:command', optionName: 'DeploymentPolicy', value: 'Immutable' },
        { namespace: 'aws:elasticbeanstalk:healthreporting:system', optionName: 'SystemType', value: 'enhanced' },
        { namespace: 'aws:elasticbeanstalk:cloudwatch:logs', optionName: 'StreamLogs', value: 'true' },
        { namespace: 'aws:elasticbeanstalk:cloudwatch:logs', optionName: 'RetentionInDays', value: '30' },
      ],
    });

    env.addDependency(app);

    new cdk.CfnOutput(this, 'EndpointURL', {
      value: env.attrEndpointUrl,
    });
  }
}
```

---

## 次に読むべきガイド

- [../02-storage/00-s3-basics.md](../02-storage/00-s3-basics.md) — S3 の基礎
- [../03-database/00-rds-basics.md](../03-database/00-rds-basics.md) — RDS の基礎

---

## 参考文献

1. AWS Elastic Beanstalk 開発者ガイド — https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/
2. EB CLI コマンドリファレンス — https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3.html
3. Elastic Beanstalk デプロイポリシー — https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/using-features.rolling-version-deploy.html
4. .ebextensions 設定ガイド — https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/ebextensions.html
5. Platform Hooks — https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/platforms-linux-extend.html
6. Docker プラットフォーム — https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create_deploy_docker.html
