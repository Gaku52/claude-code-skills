# Elastic Beanstalk

> アプリケーションのデプロイ・スケーリング・監視を自動化する AWS の PaaS サービスを使いこなす

## この章で学ぶこと

1. Elastic Beanstalk の対応プラットフォームとアーキテクチャを理解し、適切な構成を選択できる
2. 4つのデプロイ戦略の特性を比較し、ダウンタイムなしのデプロイを実現できる
3. .ebextensions と環境変数を使ったカスタマイズとモニタリング設定ができる

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
|                        | ヘルスモニタリング        |
|                        | ログ収集                 |
+------------------------+------------------------+
```

---

## 2. 対応プラットフォーム

### 2.1 サポートプラットフォーム一覧

| 言語/フレームワーク | プラットフォーム | コンテナ |
|-------------------|---------------|---------|
| Node.js | Node.js 18/20 on Amazon Linux 2023 | AL2023 |
| Python | Python 3.11/3.12 on Amazon Linux 2023 | AL2023 |
| Java | Corretto 17/21 on Amazon Linux 2023 | AL2023 |
| Go | Go 1.21 on Amazon Linux 2023 | AL2023 |
| .NET | .NET 6/8 on Amazon Linux 2023 | AL2023 |
| Ruby | Ruby 3.2/3.3 on Amazon Linux 2023 | AL2023 |
| PHP | PHP 8.2/8.3 on Amazon Linux 2023 | AL2023 |
| Docker | Docker on Amazon Linux 2023 | AL2023 |
| Multi-container Docker | ECS managed Docker | ECS |

### 2.2 コード例: EB CLI のインストールと初期化

```bash
# EB CLI インストール
pip install awsebcli

# プロジェクトを初期化
cd /path/to/my-app
eb init

# 対話形式で設定
# 1. リージョン選択: ap-northeast-1
# 2. アプリケーション名: my-web-app
# 3. プラットフォーム: Python 3.11
# 4. CodeCommit 連携: No
# 5. SSH キーペア: 既存のキーを選択
```

### 2.3 コード例: 環境の作成

```bash
# 環境を作成
eb create production-env \
  --instance-type t3.small \
  --scale 2 \
  --elb-type application \
  --region ap-northeast-1 \
  --tags Environment=production,Team=backend

# 環境の状態確認
eb status

# ログの確認
eb logs

# ヘルスチェック
eb health
```

---

## 3. デプロイ戦略

### 3.1 4つのデプロイ戦略比較

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
```

### 3.2 デプロイ戦略比較表

| 戦略 | ダウンタイム | デプロイ速度 | コスト | ロールバック |
|------|-----------|-----------|--------|-----------|
| All at Once | あり | 最速 | 追加コストなし | 再デプロイ必要 |
| Rolling | なし | 中 | 追加コストなし | 再デプロイ必要 |
| Rolling + Batch | なし | 中 | 一時的追加 | 再デプロイ必要 |
| Immutable | なし | 遅い | 一時的に2倍 | 高速（旧環境に戻す） |
| Blue/Green | なし | 遅い | 常に2倍 | 最速（URL スワップ） |

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
│   └── 05-resources.config     # CloudFormation リソース
├── .platform/
│   ├── hooks/
│   │   ├── prebuild/           # ビルド前フック
│   │   ├── predeploy/          # デプロイ前フック
│   │   └── postdeploy/         # デプロイ後フック
│   └── nginx/
│       └── conf.d/
│           └── custom.conf     # NGINX カスタム設定
├── application.py
└── requirements.txt
```

### 4.2 コード例: パッケージインストールと設定

```yaml
# .ebextensions/01-packages.config
packages:
  yum:
    git: []
    jq: []

files:
  "/etc/nginx/conf.d/proxy.conf":
    mode: "000644"
    owner: root
    group: root
    content: |
      client_max_body_size 50M;
      proxy_read_timeout 300;

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
```

### 4.3 コード例: 環境変数の設定

```bash
# CLI で環境変数を設定
eb setenv \
  DATABASE_URL=postgresql://user:pass@rds-endpoint:5432/mydb \
  REDIS_URL=redis://elasticache-endpoint:6379 \
  SECRET_KEY=my-secret-key-123 \
  DEBUG=false \
  ALLOWED_HOSTS=.example.com

# .ebextensions で環境変数を設定
# .ebextensions/04-options.config
option_settings:
  aws:elasticbeanstalk:application:environment:
    DJANGO_SETTINGS_MODULE: myapp.settings.production
    PYTHONPATH: /var/app/current
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

    location /static/ {
        alias /var/app/current/staticfiles/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
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
  ● Yellow = 警告
  ● Red    = 異常
  ● Grey   = 情報不足
```

### 5.2 コード例: ログの取得と確認

```bash
# 最新のログを取得
eb logs

# 完全なログバンドルを取得
eb logs --all

# CloudWatch Logs へのストリーミングを有効化
# .ebextensions/cloudwatch-logs.config
option_settings:
  aws:elasticbeanstalk:cloudwatch:logs:
    StreamLogs: true
    DeleteOnTerminate: false
    RetentionInDays: 30

# SSH で直接確認
eb ssh
# /var/log/eb-engine.log        — EB エンジンログ
# /var/log/nginx/access.log     — NGINX アクセスログ
# /var/log/web.stdout.log       — アプリケーション stdout
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
```

```bash
# Blue/Green デプロイの実行
# 1. 現在の環境をクローン
eb clone production-env --clone-name green-env

# 2. Green 環境にデプロイ
eb deploy green-env

# 3. テスト完了後、URL スワップ
eb swap production-env --destination-name green-env

# 4. 旧環境を削除（問題なければ）
eb terminate green-env
```

---

## 7. アンチパターン

### アンチパターン 1: 全設定をコンソールで手動管理する

コンソールで変更した設定は再現性がなく、環境再構築時に漏れが発生する。`.ebextensions` と `.platform` でコード化し、Git で管理すべきである。

```
# 悪い例
コンソールで NGINX 設定を手動変更
→ 環境再構築時に設定が消える

# 良い例
.platform/nginx/conf.d/custom.conf に設定を記述
→ デプロイのたびに自動適用
```

### アンチパターン 2: All at Once デプロイを本番で使う

全インスタンスが同時に更新されるため、デプロイ中にダウンタイムが発生する。本番環境では Rolling with Additional Batch または Immutable を使用すべきである。

---

## 8. FAQ

### Q1. Elastic Beanstalk と ECS / Fargate のどちらを選ぶべきか？

Beanstalk は「アプリケーションをデプロイするだけ」のシンプルさが利点。コンテナオーケストレーションの詳細な制御（サイドカーパターン、サービスメッシュ等）が必要なら ECS/Fargate を選択する。Docker プラットフォームを使えば Beanstalk でもコンテナを実行できるが、複雑な構成には向かない。

### Q2. Elastic Beanstalk のコストは？

Beanstalk 自体は無料で、裏側の EC2、ALB、RDS などの料金のみが発生する。ただし Beanstalk が自動作成する ALB やスケーリング設定が想定以上のコストを生む場合があるので、作成されるリソースを確認する。

### Q3. カスタムドメインと HTTPS をどう設定するか？

Route 53 でドメインを ALB に ALIAS レコードで向け、ACM (AWS Certificate Manager) で SSL 証明書を取得して ALB のリスナーに設定する。`.ebextensions` で HTTPS リスナーの設定を自動化できる。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| 位置づけ | EC2 + ALB + ASG をまとめて管理する PaaS |
| プラットフォーム | Node.js, Python, Java, Go, .NET, Docker 等 |
| デプロイ戦略 | 本番は Rolling + Batch か Immutable 推奨 |
| カスタマイズ | .ebextensions と .platform で宣言的に管理 |
| Blue/Green | eb swap で URL を切り替え、高速ロールバック |
| モニタリング | 拡張ヘルス + CloudWatch Logs ストリーミング |

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
