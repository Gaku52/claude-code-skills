# EC2 応用

> Auto Scaling、ロードバランサー、スポットインスタンス、Savings Plans で EC2 をプロダクションレベルで運用する

## この章で学ぶこと

1. Auto Scaling グループを設計し、負荷に応じたスケーリングポリシーを実装できる
2. ALB / NLB の特性を理解し、適切なロードバランサーを選択・設定できる
3. スポットインスタンスと Savings Plans を活用してコストを最適化できる

---

## 1. Auto Scaling

### 1.1 Auto Scaling の構成要素

```
Auto Scaling アーキテクチャ
+----------------------------------------------------------+
|                                                           |
|  +-------------------+     +--------------------------+  |
|  | 起動テンプレート   |     | Auto Scaling グループ      |  |
|  | - AMI             | --> | - 最小: 2                  |  |
|  | - インスタンスタイプ |     | - 希望: 2                  |  |
|  | - セキュリティGrp  |     | - 最大: 10                 |  |
|  | - User Data       |     | - AZ: 1a, 1c, 1d          |  |
|  +-------------------+     +--------------------------+  |
|                                    |                      |
|                                    v                      |
|  +---------------------------------------------------+   |
|  | スケーリングポリシー                                  |  |
|  | - ターゲット追跡: CPU 60% 維持                       |  |
|  | - ステップ: CPU 80%→+2台, 90%→+4台                 |  |
|  | - スケジュール: 平日 9時に5台                         |  |
|  +---------------------------------------------------+   |
+----------------------------------------------------------+
```

### 1.2 コード例: 起動テンプレートの作成

```bash
# 起動テンプレートを作成
aws ec2 create-launch-template \
  --launch-template-name web-server-template \
  --version-description "v1.0 - NGINX + Node.js" \
  --launch-template-data '{
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "t3.small",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-0123456789abcdef0"],
    "IamInstanceProfile": {
      "Name": "EC2-WebServer-Profile"
    },
    "BlockDeviceMappings": [
      {
        "DeviceName": "/dev/xvda",
        "Ebs": {
          "VolumeSize": 30,
          "VolumeType": "gp3",
          "Encrypted": true
        }
      }
    ],
    "MetadataOptions": {
      "HttpTokens": "required",
      "HttpEndpoint": "enabled"
    },
    "UserData": "'$(base64 -w 0 startup.sh)'"
  }'
```

### 1.3 コード例: Auto Scaling グループの作成

```bash
# Auto Scaling グループを作成
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name web-asg \
  --launch-template LaunchTemplateName=web-server-template,Version='$Latest' \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 2 \
  --vpc-zone-identifier "subnet-aaa,subnet-bbb,subnet-ccc" \
  --target-group-arns "arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:targetgroup/web-tg/xxx" \
  --health-check-type ELB \
  --health-check-grace-period 300 \
  --tags '[
    {"Key": "Name", "Value": "web-server", "PropagateAtLaunch": true},
    {"Key": "Environment", "Value": "production", "PropagateAtLaunch": true}
  ]'
```

### 1.4 スケーリングポリシーの種類

| ポリシー | 仕組み | ユースケース |
|---------|--------|-------------|
| ターゲット追跡 | メトリクスを目標値に維持 | CPU 使用率 60% を維持 |
| ステップスケーリング | 閾値超過量に応じて段階的に増減 | 急激な負荷変動 |
| シンプルスケーリング | 1つの閾値で固定台数を増減 | 単純なルール |
| スケジュール | 時刻指定でキャパシティ変更 | 営業時間の負荷パターン |
| 予測スケーリング | ML で需要を予測し事前スケール | 周期的なトラフィック |

### 1.5 コード例: ターゲット追跡ポリシー

```bash
# CPU 使用率 60% を維持するポリシー
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name web-asg \
  --policy-name cpu-target-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ASGAverageCPUUtilization"
    },
    "TargetValue": 60.0,
    "ScaleInCooldown": 300,
    "ScaleOutCooldown": 60
  }'

# リクエスト数ベースのポリシー
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name web-asg \
  --policy-name request-count-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ALBRequestCountPerTarget",
      "ResourceLabel": "app/my-alb/xxx/targetgroup/web-tg/yyy"
    },
    "TargetValue": 1000.0
  }'
```

---

## 2. Elastic Load Balancing

### 2.1 ロードバランサーの種類

```
                       クライアント
                           |
              +------------+------------+
              |            |            |
         +----v----+  +---v----+  +---v----+
         |   ALB   |  |  NLB   |  |  GLB   |
         | (L7)    |  | (L4)   |  | (L3)   |
         +---------+  +--------+  +--------+
         HTTP/HTTPS   TCP/UDP/TLS  アプライアンス
         パスルーティング 超低レイテンシ  透過型
         ホストルーティング 固定IP      IDS/IPS
         WebSocket    NLB→ALB連携
```

### 2.2 ALB vs NLB 比較

| 特性 | ALB | NLB |
|------|-----|-----|
| OSI レイヤー | L7 (HTTP/HTTPS) | L4 (TCP/UDP/TLS) |
| ルーティング | パス、ホスト、ヘッダー、クエリ | ポートベース |
| レイテンシ | 数ミリ秒 | 超低レイテンシ (数百マイクロ秒) |
| 固定 IP | 不可 (DNS 名) | 可能 (Elastic IP) |
| SSL 終端 | 可能 | 可能 |
| WebSocket | 対応 | 対応 |
| gRPC | 対応 | TCP で対応 |
| 料金 | やや高い | やや安い |

### 2.3 コード例: ALB の作成

```bash
# ALB を作成
ALB_ARN=$(aws elbv2 create-load-balancer \
  --name web-alb \
  --subnets subnet-aaa subnet-bbb subnet-ccc \
  --security-groups sg-alb-xxx \
  --scheme internet-facing \
  --type application \
  --ip-address-type ipv4 \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# ターゲットグループを作成
TG_ARN=$(aws elbv2 create-target-group \
  --name web-tg \
  --protocol HTTP --port 80 \
  --vpc-id vpc-xxx \
  --target-type instance \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

# HTTPS リスナーを作成
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTPS --port 443 \
  --certificates CertificateArn=arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx \
  --default-actions Type=forward,TargetGroupArn=$TG_ARN \
  --ssl-policy ELBSecurityPolicy-TLS13-1-2-2021-06

# HTTP → HTTPS リダイレクト
aws elbv2 create-listener \
  --load-balancer-arn $ALB_ARN \
  --protocol HTTP --port 80 \
  --default-actions '[{
    "Type": "redirect",
    "RedirectConfig": {
      "Protocol": "HTTPS",
      "Port": "443",
      "StatusCode": "HTTP_301"
    }
  }]'
```

### 2.4 ALB パスベースルーティング

```
                    ALB
                     |
        +------------+------------+
        |            |            |
   /api/*       /static/*     /* (デフォルト)
        |            |            |
   +----v----+  +---v----+  +---v----+
   | API TG  |  | S3     |  | Web TG |
   | (Fargate)|  | (固定応答)|  | (EC2)  |
   +---------+  +--------+  +--------+
```

```bash
# パスベースルーティングルールを追加
aws elbv2 create-rule \
  --listener-arn $LISTENER_ARN \
  --conditions '[{
    "Field": "path-pattern",
    "Values": ["/api/*"]
  }]' \
  --actions '[{
    "Type": "forward",
    "TargetGroupArn": "'$API_TG_ARN'"
  }]' \
  --priority 10
```

---

## 3. スポットインスタンス

### 3.1 購入オプション比較

| オプション | 割引率 | コミットメント | 中断リスク | ユースケース |
|-----------|--------|-------------|-----------|------------|
| オンデマンド | 0% | なし | なし | ベースライン |
| リザーブド (1年) | 最大 40% | 1年 | なし | 定常ワークロード |
| リザーブド (3年) | 最大 60% | 3年 | なし | 長期利用 |
| Savings Plans | 最大 72% | 1-3年 | なし | 柔軟なコミットメント |
| スポット | 最大 90% | なし | あり (2分通知) | バッチ、耐障害性あるワークロード |

### 3.2 コード例: スポットインスタンスのリクエスト

```bash
# スポットインスタンスリクエスト
aws ec2 request-spot-instances \
  --spot-price "0.05" \
  --instance-count 5 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "c5.xlarge",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-xxx"],
    "SubnetId": "subnet-xxx"
  }'

# スポットフリートを起動（複数インスタンスタイプ）
aws ec2 request-spot-fleet \
  --spot-fleet-request-config '{
    "IamFleetRole": "arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-role",
    "TargetCapacity": 10,
    "SpotPrice": "0.10",
    "LaunchSpecifications": [
      {"InstanceType": "c5.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-aaa"},
      {"InstanceType": "c5a.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-bbb"},
      {"InstanceType": "c5d.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-ccc"}
    ],
    "AllocationStrategy": "capacityOptimized"
  }'
```

### 3.3 スポットの中断対策

```
スポット中断ハンドリングフロー

  EC2 メタデータ          EventBridge
  (2分前通知)             (中断イベント)
       |                       |
       v                       v
  +----------+          +-----------+
  | 処理中の   |          | Lambda    |
  | ジョブを   |          | ASG から   |
  | チェック   |          | デタッチ   |
  | ポイント   |          | 代替起動   |
  +----------+          +-----------+
```

```bash
# EC2 メタデータでスポット中断通知を確認するスクリプト
#!/bin/bash
while true; do
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-aws-ec2-metadata-token: $(curl -s -X PUT \
      http://169.254.169.254/latest/api/token \
      -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')" \
    http://169.254.169.254/latest/meta-data/spot/instance-action)

  if [ "$RESPONSE" == "200" ]; then
    echo "Spot interruption notice received!"
    # グレースフルシャットダウン処理
    /usr/local/bin/graceful-shutdown.sh
    break
  fi
  sleep 5
done
```

---

## 4. Savings Plans

### 4.1 Savings Plans の種類

```
  +-----------------------------------------------+
  | Compute Savings Plans                          |
  | - EC2, Fargate, Lambda に適用                   |
  | - リージョン・ファミリー・OS 変更自由            |
  | - 割引率: 最大 66%                              |
  +-----------------------------------------------+
  | EC2 Instance Savings Plans                     |
  | - 特定リージョン・ファミリーに限定               |
  | - インスタンスサイズ・OS は変更可能              |
  | - 割引率: 最大 72%                              |
  +-----------------------------------------------+
```

### 4.2 コード例: Savings Plans の情報取得

```bash
# 推奨 Savings Plans を確認
aws savingsplans describe-savings-plans-offerings \
  --service-codes AmazonEC2 \
  --payment-options NoUpfront \
  --plan-types ComputeSavingsPlans \
  --region us-east-1

# 現在の Savings Plans 一覧
aws savingsplans describe-savings-plans \
  --query 'savingsPlans[].[savingsPlanId,savingsPlanType,commitment,state]' \
  --output table
```

---

## 5. コスト最適化戦略

```
EC2 コスト最適化ピラミッド

         /\
        /  \  スポット (中断許容)
       /    \    最大 90% 割引
      /------\
     /        \  Savings Plans / RI
    /          \    40-72% 割引
   /------------\
  /              \  右サイジング
 /                \   不要リソース削除
/------------------\
  オンデマンド (ベースライン)
```

| 戦略 | 効果 | 実装難易度 | 優先度 |
|------|------|-----------|--------|
| 未使用リソース削除 | 即効性あり | 低 | 最優先 |
| 右サイジング | 20-30% 削減 | 中 | 高 |
| Savings Plans | 40-72% 削減 | 低 | 高 |
| スポット活用 | 最大 90% 削減 | 中-高 | 中 |
| Graviton 移行 | 20-40% 削減 | 中 | 中 |

---

## 6. アンチパターン

### アンチパターン 1: 単一 AZ に全インスタンスを配置する

```
# 悪い例 — 1つの AZ のみ
Auto Scaling Group → subnet-1a のみ
→ AZ 障害で全滅

# 良い例 — 複数 AZ に分散
Auto Scaling Group → subnet-1a, subnet-1c, subnet-1d
→ 1つの AZ が障害でも残りで継続
```

### アンチパターン 2: Auto Scaling の最小値を 0 にする

最小値 0 ではスケールイン時に全インスタンスが終了する可能性がある。本番環境では最小 2（マルチ AZ）を確保すべきである。

```bash
# 悪い例
--min-size 0 --desired-capacity 1

# 良い例（本番環境）
--min-size 2 --desired-capacity 2 --max-size 10
```

### アンチパターン 3: ヘルスチェックなしでスケーリングする

EC2 のステータスチェックだけでは、アプリケーションレベルの障害を検知できない。ELB ヘルスチェックとカスタムヘルスチェックを組み合わせるべきである。

---

## 7. FAQ

### Q1. ALB と NLB のどちらを選ぶべきか？

HTTP/HTTPS ベースの Web アプリケーションには ALB を選択する。TCP/UDP レベルのルーティング、超低レイテンシ、固定 IP が必要な場合は NLB を選択する。gRPC は ALB が L7 レベルでサポートしているが、NLB でも TCP として通すことは可能。

### Q2. スポットインスタンスの中断はどのくらいの頻度で起きるか？

リージョンとインスタンスタイプに依存するが、capacityOptimized 戦略を使い複数タイプを指定すると中断頻度は大幅に下がる。AWS Spot Instance Advisor で中断率を事前確認できる。バッチ処理やコンテナベースのステートレスワークロードが最適。

### Q3. Savings Plans とリザーブドインスタンスの違いは？

Savings Plans は「コミット金額/時」ベースで柔軟性が高く、インスタンスファミリー・リージョン・OS の変更が可能（Compute SP の場合）。RI は特定インスタンスタイプ・AZ に紐づく。新規購入では Savings Plans を推奨する。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| Auto Scaling | 起動テンプレート + ASG + スケーリングポリシーの3層構成 |
| ALB | L7 ルーティング、パス/ホストベース、HTTP/HTTPS |
| NLB | L4 ルーティング、超低レイテンシ、固定 IP |
| スポット | 最大 90% 割引、中断対策必須、capacityOptimized 推奨 |
| Savings Plans | Compute SP で柔軟にコスト削減、1年/3年コミット |
| コスト最適化 | 削除→右サイジング→SP/RI→スポットの順で実施 |

---

## 次に読むべきガイド

- [02-elastic-beanstalk.md](./02-elastic-beanstalk.md) — Elastic Beanstalk によるマネージドデプロイ
- [../04-networking/00-vpc-basics.md](../04-networking/00-vpc-basics.md) — VPC の設計

---

## 参考文献

1. Amazon EC2 Auto Scaling ユーザーガイド — https://docs.aws.amazon.com/autoscaling/ec2/userguide/
2. Elastic Load Balancing ユーザーガイド — https://docs.aws.amazon.com/elasticloadbalancing/latest/userguide/
3. Amazon EC2 スポットインスタンスベストプラクティス — https://docs.aws.amazon.com/ec2/latest/userguide/spot-best-practices.html
4. Savings Plans ユーザーガイド — https://docs.aws.amazon.com/savingsplans/latest/userguide/
