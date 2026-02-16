# EC2 応用

> Auto Scaling、ロードバランサー、スポットインスタンス、Savings Plans で EC2 をプロダクションレベルで運用する

## この章で学ぶこと

1. Auto Scaling グループを設計し、負荷に応じたスケーリングポリシーを実装できる
2. ALB / NLB の特性を理解し、適切なロードバランサーを選択・設定できる
3. スポットインスタンスと Savings Plans を活用してコストを最適化できる
4. CloudFormation / CDK で Auto Scaling + ALB のインフラをコード管理できる
5. 混合インスタンスポリシーで Graviton とスポットを組み合わせた高コスパ構成を実現できる

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
|  | - 予測: ML ベースの需要予測                           |  |
|  +---------------------------------------------------+   |
|                                                           |
|  +---------------------------------------------------+   |
|  | ライフサイクルフック                                   |  |
|  | - 起動時: 設定完了まで待機                            |  |
|  | - 終了時: ログ退避・接続ドレイン                       |  |
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
    "Monitoring": {
      "Enabled": true
    },
    "TagSpecifications": [
      {
        "ResourceType": "instance",
        "Tags": [
          {"Key": "Name", "Value": "web-server"},
          {"Key": "Environment", "Value": "production"}
        ]
      }
    ],
    "UserData": "'$(base64 -w 0 startup.sh)'"
  }'

# 起動テンプレートの新バージョンを作成
aws ec2 create-launch-template-version \
  --launch-template-name web-server-template \
  --version-description "v2.0 - Graviton migration" \
  --source-version 1 \
  --launch-template-data '{
    "ImageId": "ami-0fedcba9876543210",
    "InstanceType": "t4g.small"
  }'

# デフォルトバージョンの設定
aws ec2 modify-launch-template \
  --launch-template-name web-server-template \
  --default-version 2

# 起動テンプレートのバージョン一覧
aws ec2 describe-launch-template-versions \
  --launch-template-name web-server-template \
  --query 'LaunchTemplateVersions[].[VersionNumber,VersionDescription,LaunchTemplateData.InstanceType]' \
  --output table
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
  --default-cooldown 300 \
  --termination-policies '["OldestLaunchTemplate", "OldestInstance"]' \
  --new-instances-protected-from-scale-in \
  --capacity-rebalance \
  --tags '[
    {"Key": "Name", "Value": "web-server", "PropagateAtLaunch": true},
    {"Key": "Environment", "Value": "production", "PropagateAtLaunch": true}
  ]'

# ASG の状態確認
aws autoscaling describe-auto-scaling-groups \
  --auto-scaling-group-names web-asg \
  --query 'AutoScalingGroups[0].{Min:MinSize,Max:MaxSize,Desired:DesiredCapacity,Instances:Instances[*].{Id:InstanceId,AZ:AvailabilityZone,Health:HealthStatus,State:LifecycleState}}' \
  --output json
```

### 1.4 混合インスタンスポリシー（Graviton + スポット）

コスト最適化の決定版として、Graviton インスタンスとスポットインスタンスを組み合わせる構成がある。

```bash
# 混合インスタンスポリシーで ASG を作成
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name web-asg-mixed \
  --mixed-instances-policy '{
    "LaunchTemplate": {
      "LaunchTemplateSpecification": {
        "LaunchTemplateName": "web-server-template",
        "Version": "$Latest"
      },
      "Overrides": [
        {"InstanceType": "t4g.small", "WeightedCapacity": "1"},
        {"InstanceType": "t4g.medium", "WeightedCapacity": "2"},
        {"InstanceType": "t3.small", "WeightedCapacity": "1"},
        {"InstanceType": "t3.medium", "WeightedCapacity": "2"},
        {"InstanceType": "m6g.large", "WeightedCapacity": "4"},
        {"InstanceType": "m5.large", "WeightedCapacity": "4"}
      ]
    },
    "InstancesDistribution": {
      "OnDemandBaseCapacity": 2,
      "OnDemandPercentageAboveBaseCapacity": 25,
      "SpotAllocationStrategy": "capacity-optimized",
      "SpotMaxPrice": ""
    }
  }' \
  --min-size 2 \
  --max-size 20 \
  --desired-capacity 4 \
  --vpc-zone-identifier "subnet-aaa,subnet-bbb,subnet-ccc" \
  --target-group-arns "arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:targetgroup/web-tg/xxx" \
  --health-check-type ELB \
  --health-check-grace-period 300

# 結果:
# - ベースの 2 台はオンデマンド（安定性確保）
# - 追加分の 75% はスポット（コスト削減）
# - 追加分の 25% はオンデマンド（可用性確保）
# - capacity-optimized で中断リスクが低いプールを自動選択
```

### 1.5 スケーリングポリシーの種類

| ポリシー | 仕組み | ユースケース | 設定の複雑さ |
|---------|--------|-------------|------------|
| ターゲット追跡 | メトリクスを目標値に維持 | CPU 使用率 60% を維持 | 低 |
| ステップスケーリング | 閾値超過量に応じて段階的に増減 | 急激な負荷変動 | 中 |
| シンプルスケーリング | 1つの閾値で固定台数を増減 | 単純なルール | 低 |
| スケジュール | 時刻指定でキャパシティ変更 | 営業時間の負荷パターン | 低 |
| 予測スケーリング | ML で需要を予測し事前スケール | 周期的なトラフィック | 低 |

### 1.6 コード例: ターゲット追跡ポリシー

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
    "ScaleOutCooldown": 60,
    "DisableScaleIn": false
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

# カスタムメトリクスベースのポリシー（SQS キュー長）
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name worker-asg \
  --policy-name sqs-queue-tracking \
  --policy-type TargetTrackingScaling \
  --target-tracking-configuration '{
    "CustomizedMetricSpecification": {
      "MetricName": "ApproximateNumberOfMessagesVisible",
      "Namespace": "AWS/SQS",
      "Dimensions": [
        {"Name": "QueueName", "Value": "my-worker-queue"}
      ],
      "Statistic": "Average"
    },
    "TargetValue": 10.0
  }'
```

### 1.7 コード例: ステップスケーリングポリシー

```bash
# スケールアウト: CPU に応じて段階的に台数追加
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name web-asg \
  --policy-name cpu-step-scale-out \
  --policy-type StepScaling \
  --adjustment-type ChangeInCapacity \
  --step-adjustments '[
    {"MetricIntervalLowerBound": 0, "MetricIntervalUpperBound": 20, "ScalingAdjustment": 1},
    {"MetricIntervalLowerBound": 20, "MetricIntervalUpperBound": 40, "ScalingAdjustment": 2},
    {"MetricIntervalLowerBound": 40, "ScalingAdjustment": 4}
  ]' \
  --metric-aggregation-type Average

# 対応する CloudWatch アラーム
aws cloudwatch put-metric-alarm \
  --alarm-name "web-asg-cpu-high" \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=AutoScalingGroupName,Value=web-asg \
  --statistic Average \
  --period 60 \
  --evaluation-periods 2 \
  --threshold 60 \
  --comparison-operator GreaterThanThreshold \
  --alarm-actions "arn:aws:autoscaling:ap-northeast-1:123456789012:scalingPolicy:xxx:autoScalingGroupName/web-asg:policyName/cpu-step-scale-out"
```

### 1.8 コード例: スケジュールスケーリング

```bash
# 平日 9:00 (JST) にスケールアウト
aws autoscaling put-scheduled-update-group-action \
  --auto-scaling-group-name web-asg \
  --scheduled-action-name weekday-scale-out \
  --recurrence "0 0 * * MON-FRI" \
  --min-size 4 \
  --max-size 10 \
  --desired-capacity 4 \
  --time-zone "Asia/Tokyo"

# 平日 22:00 (JST) にスケールイン
aws autoscaling put-scheduled-update-group-action \
  --auto-scaling-group-name web-asg \
  --scheduled-action-name weekday-scale-in \
  --recurrence "0 13 * * MON-FRI" \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 2 \
  --time-zone "Asia/Tokyo"

# 週末は最小構成
aws autoscaling put-scheduled-update-group-action \
  --auto-scaling-group-name web-asg \
  --scheduled-action-name weekend-scale-down \
  --recurrence "0 0 * * SAT" \
  --min-size 2 \
  --max-size 4 \
  --desired-capacity 2 \
  --time-zone "Asia/Tokyo"

# スケジュール一覧の確認
aws autoscaling describe-scheduled-actions \
  --auto-scaling-group-name web-asg \
  --output table
```

### 1.9 予測スケーリング

```bash
# 予測スケーリングを有効化
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name web-asg \
  --policy-name predictive-scaling \
  --policy-type PredictiveScaling \
  --predictive-scaling-configuration '{
    "MetricSpecifications": [{
      "TargetValue": 60.0,
      "PredefinedMetricPairSpecification": {
        "PredefinedMetricType": "ASGCPUUtilization"
      }
    }],
    "Mode": "ForecastAndScale",
    "SchedulingBufferTime": 300,
    "MaxCapacityBreachBehavior": "HonorMaxCapacity"
  }'

# 予測結果の確認
aws autoscaling get-predictive-scaling-forecast \
  --auto-scaling-group-name web-asg \
  --policy-name predictive-scaling \
  --start-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u -v+2d +%Y-%m-%dT%H:%M:%SZ)"
```

### 1.10 ライフサイクルフック

```bash
# 起動時のライフサイクルフック（設定完了まで待機）
aws autoscaling put-lifecycle-hook \
  --auto-scaling-group-name web-asg \
  --lifecycle-hook-name launch-hook \
  --lifecycle-transition autoscaling:EC2_INSTANCE_LAUNCHING \
  --heartbeat-timeout 300 \
  --default-result CONTINUE \
  --notification-target-arn arn:aws:sns:ap-northeast-1:123456789012:asg-lifecycle

# 終了時のライフサイクルフック（ログ退避・接続ドレイン）
aws autoscaling put-lifecycle-hook \
  --auto-scaling-group-name web-asg \
  --lifecycle-hook-name terminate-hook \
  --lifecycle-transition autoscaling:EC2_INSTANCE_TERMINATING \
  --heartbeat-timeout 600 \
  --default-result CONTINUE \
  --notification-target-arn arn:aws:sns:ap-northeast-1:123456789012:asg-lifecycle

# ライフサイクルアクション完了通知
aws autoscaling complete-lifecycle-action \
  --auto-scaling-group-name web-asg \
  --lifecycle-hook-name launch-hook \
  --instance-id i-0123456789abcdef0 \
  --lifecycle-action-result CONTINUE
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
         gRPC
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
| ヘルスチェック | HTTP/HTTPS | TCP/HTTP/HTTPS |
| スティッキーセッション | Cookie ベース | なし（ソース IP ハッシュ） |
| クロスゾーン | デフォルト有効 | デフォルト無効 |
| 料金 | やや高い | やや安い |
| PrivateLink | 不可 | 対応 |
| WAF 連携 | 対応 | 不可 |

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
  --tags Key=Environment,Value=production \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# ALB のアクセスログを S3 に出力
aws elbv2 modify-load-balancer-attributes \
  --load-balancer-arn $ALB_ARN \
  --attributes '[
    {"Key": "access_logs.s3.enabled", "Value": "true"},
    {"Key": "access_logs.s3.bucket", "Value": "my-alb-logs-bucket"},
    {"Key": "access_logs.s3.prefix", "Value": "web-alb"},
    {"Key": "idle_timeout.timeout_seconds", "Value": "60"},
    {"Key": "routing.http.drop_invalid_header_fields.enabled", "Value": "true"},
    {"Key": "routing.http2.enabled", "Value": "true"}
  ]'

# ターゲットグループを作成
TG_ARN=$(aws elbv2 create-target-group \
  --name web-tg \
  --protocol HTTP --port 80 \
  --vpc-id vpc-xxx \
  --target-type instance \
  --health-check-path /health \
  --health-check-interval-seconds 30 \
  --health-check-timeout-seconds 5 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 3 \
  --matcher '{"HttpCode": "200-299"}' \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

# ターゲットグループの属性設定
aws elbv2 modify-target-group-attributes \
  --target-group-arn $TG_ARN \
  --attributes '[
    {"Key": "deregistration_delay.timeout_seconds", "Value": "30"},
    {"Key": "slow_start.duration_seconds", "Value": "60"},
    {"Key": "stickiness.enabled", "Value": "true"},
    {"Key": "stickiness.type", "Value": "lb_cookie"},
    {"Key": "stickiness.lb_cookie.duration_seconds", "Value": "3600"}
  ]'

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
        +------------+------------+-----------+
        |            |            |           |
   /api/*       /static/*     /ws/*      /* (デフォルト)
        |            |            |           |
   +----v----+  +---v----+  +---v----+  +---v----+
   | API TG  |  | S3     |  | WS TG  |  | Web TG |
   | (Fargate)|  | (固定応答)|  | (WS)   |  | (EC2)  |
   +---------+  +--------+  +--------+  +--------+
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

# ホストベースルーティング
aws elbv2 create-rule \
  --listener-arn $LISTENER_ARN \
  --conditions '[{
    "Field": "host-header",
    "Values": ["api.example.com"]
  }]' \
  --actions '[{
    "Type": "forward",
    "TargetGroupArn": "'$API_TG_ARN'"
  }]' \
  --priority 20

# 複合条件（パス + ヘッダー）
aws elbv2 create-rule \
  --listener-arn $LISTENER_ARN \
  --conditions '[
    {"Field": "path-pattern", "Values": ["/api/v2/*"]},
    {"Field": "http-header", "HttpHeaderConfig": {"HttpHeaderName": "X-Api-Version", "Values": ["2"]}}
  ]' \
  --actions '[{
    "Type": "forward",
    "TargetGroupArn": "'$API_V2_TG_ARN'"
  }]' \
  --priority 5

# 加重ターゲットグループ（カナリアリリース）
aws elbv2 create-rule \
  --listener-arn $LISTENER_ARN \
  --conditions '[{"Field": "path-pattern", "Values": ["/feature/*"]}]' \
  --actions '[{
    "Type": "forward",
    "ForwardConfig": {
      "TargetGroups": [
        {"TargetGroupArn": "'$STABLE_TG_ARN'", "Weight": 90},
        {"TargetGroupArn": "'$CANARY_TG_ARN'", "Weight": 10}
      ],
      "TargetGroupStickinessConfig": {
        "Enabled": true,
        "DurationSeconds": 3600
      }
    }
  }]' \
  --priority 15
```

### 2.5 コード例: NLB の作成

```bash
# NLB を作成（固定 IP 付き）
NLB_ARN=$(aws elbv2 create-load-balancer \
  --name tcp-nlb \
  --type network \
  --subnets subnet-aaa subnet-bbb subnet-ccc \
  --scheme internet-facing \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# 固定 Elastic IP を割り当てる場合
NLB_ARN=$(aws elbv2 create-load-balancer \
  --name tcp-nlb-eip \
  --type network \
  --subnet-mappings '[
    {"SubnetId": "subnet-aaa", "AllocationId": "eipalloc-aaa"},
    {"SubnetId": "subnet-bbb", "AllocationId": "eipalloc-bbb"}
  ]' \
  --scheme internet-facing \
  --query 'LoadBalancers[0].LoadBalancerArn' --output text)

# TCP ターゲットグループ
NLB_TG_ARN=$(aws elbv2 create-target-group \
  --name tcp-tg \
  --protocol TCP --port 443 \
  --vpc-id vpc-xxx \
  --target-type instance \
  --health-check-protocol TCP \
  --health-check-interval-seconds 10 \
  --healthy-threshold-count 2 \
  --unhealthy-threshold-count 2 \
  --query 'TargetGroups[0].TargetGroupArn' --output text)

# TLS リスナー（NLB で TLS 終端）
aws elbv2 create-listener \
  --load-balancer-arn $NLB_ARN \
  --protocol TLS --port 443 \
  --certificates CertificateArn=arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx \
  --default-actions Type=forward,TargetGroupArn=$NLB_TG_ARN \
  --ssl-policy ELBSecurityPolicy-TLS13-1-2-2021-06
```

### 2.6 ALB + NLB の連携パターン

```
  クライアント
       |
  +----v----+
  |   NLB   |  ← 固定 IP / PrivateLink 用
  |  (L4)   |
  +----+----+
       |
  +----v----+
  |   ALB   |  ← L7 ルーティング / WAF
  |  (L7)   |
  +----+----+
       |
  +----v----+
  | Target  |
  | Group   |
  +---------+
```

---

## 3. スポットインスタンス

### 3.1 購入オプション比較

| オプション | 割引率 | コミットメント | 中断リスク | ユースケース |
|-----------|--------|-------------|-----------|------------|
| オンデマンド | 0% | なし | なし | ベースライン |
| リザーブド (1年) | 最大 40% | 1年 | なし | 定常ワークロード |
| リザーブド (3年) | 最大 60% | 3年 | なし | 長期利用 |
| Savings Plans (Compute) | 最大 66% | 1-3年 | なし | 柔軟なコミットメント |
| Savings Plans (EC2) | 最大 72% | 1-3年 | なし | 特定ファミリー固定 |
| スポット | 最大 90% | なし | あり (2分通知) | バッチ、耐障害性あるワークロード |

### 3.2 スポットインスタンスの割り当て戦略

| 戦略 | 説明 | 推奨ユースケース |
|------|------|----------------|
| capacity-optimized | 最も利用可能な容量のプールから割り当て | 一般的なワークロード（推奨） |
| capacity-optimized-prioritized | 優先度付きで容量最適化 | 特定タイプを優先したい場合 |
| lowest-price | 最低価格のプールから割り当て | コスト最重視 |
| diversified | 複数プールに均等分配 | 大規模フリート |
| price-capacity-optimized | 価格と容量のバランス | コストと可用性の両立 |

### 3.3 コード例: スポットインスタンスのリクエスト

```bash
# スポットインスタンスリクエスト（推奨: ASG の混合ポリシーを使う）
aws ec2 request-spot-instances \
  --instance-count 5 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0abcdef1234567890",
    "InstanceType": "c5.xlarge",
    "KeyName": "my-key-pair",
    "SecurityGroupIds": ["sg-xxx"],
    "SubnetId": "subnet-xxx",
    "IamInstanceProfile": {"Name": "BatchWorkerProfile"}
  }'

# スポットフリートを起動（複数インスタンスタイプ）
aws ec2 request-spot-fleet \
  --spot-fleet-request-config '{
    "IamFleetRole": "arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-role",
    "TargetCapacity": 10,
    "LaunchSpecifications": [
      {"InstanceType": "c5.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-aaa", "WeightedCapacity": 1},
      {"InstanceType": "c5a.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-bbb", "WeightedCapacity": 1},
      {"InstanceType": "c5d.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-ccc", "WeightedCapacity": 1},
      {"InstanceType": "c6i.xlarge", "ImageId": "ami-xxx", "SubnetId": "subnet-aaa", "WeightedCapacity": 1},
      {"InstanceType": "c6g.xlarge", "ImageId": "ami-xxx-arm64", "SubnetId": "subnet-bbb", "WeightedCapacity": 1}
    ],
    "AllocationStrategy": "capacityOptimized",
    "TerminateInstancesWithExpiration": true,
    "Type": "maintain",
    "ReplaceUnhealthyInstances": true
  }'

# スポット価格履歴の確認
aws ec2 describe-spot-price-history \
  --instance-types c5.xlarge c5a.xlarge c6i.xlarge \
  --product-descriptions "Linux/UNIX" \
  --start-time "$(date -u -v-1d +%Y-%m-%dT%H:%M:%SZ)" \
  --query 'SpotPriceHistory[].[InstanceType,AvailabilityZone,SpotPrice,Timestamp]' \
  --output table
```

### 3.4 スポットの中断対策

```
スポット中断ハンドリングフロー

  EC2 メタデータ          EventBridge            ASG
  (2分前通知)             (中断イベント)          (自動代替)
       |                       |                    |
       v                       v                    v
  +----------+          +-----------+         +-----------+
  | 処理中の   |          | Lambda    |         | 新しい     |
  | ジョブを   |          | SQS再投入  |         | インスタンス |
  | チェック   |          | Slack通知  |         | 自動起動   |
  | ポイント   |          | メトリクス  |         |           |
  +----------+          +-----------+         +-----------+
```

```bash
# EC2 メタデータでスポット中断通知を確認するスクリプト
#!/bin/bash
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

while true; do
  RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/spot/instance-action)

  if [ "$RESPONSE" == "200" ]; then
    ACTION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
      http://169.254.169.254/latest/meta-data/spot/instance-action)
    echo "Spot interruption notice received: $ACTION"

    # グレースフルシャットダウン処理
    # 1. 新しいリクエストの受付を停止
    /usr/local/bin/stop-accepting-requests.sh
    # 2. 進行中のジョブをチェックポイント
    /usr/local/bin/checkpoint-jobs.sh
    # 3. ログを S3 に退避
    /usr/local/bin/flush-logs-to-s3.sh
    # 4. ロードバランサーからデタッチ
    /usr/local/bin/deregister-from-lb.sh
    break
  fi
  sleep 5
done
```

```bash
# EventBridge でスポット中断をキャッチする Lambda
# EventBridge ルール
aws events put-rule \
  --name "spot-interruption-handler" \
  --event-pattern '{
    "source": ["aws.ec2"],
    "detail-type": ["EC2 Spot Instance Interruption Warning"]
  }'

# Lambda ターゲットの設定
aws events put-targets \
  --rule "spot-interruption-handler" \
  --targets '[{
    "Id": "SpotInterruptionHandler",
    "Arn": "arn:aws:lambda:ap-northeast-1:123456789012:function:handle-spot-interruption"
  }]'
```

### 3.5 スポット活用のベストプラクティス

1. **複数インスタンスタイプ**: 最低6つ以上のインスタンスタイプを指定
2. **複数 AZ**: 全ての利用可能な AZ を使用
3. **capacity-optimized**: 中断リスクが低いプールを自動選択
4. **x86 + ARM 混合**: Graviton を含めることでプール拡大
5. **チェックポイント**: 長時間ジョブは定期的に状態を保存
6. **ASG 統合**: スポットフリート単独より ASG の混合ポリシーを推奨

---

## 4. Savings Plans

### 4.1 Savings Plans の種類

```
  +-----------------------------------------------+
  | Compute Savings Plans                          |
  | - EC2, Fargate, Lambda に適用                   |
  | - リージョン・ファミリー・OS 変更自由            |
  | - 割引率: 最大 66%                              |
  | - 最も柔軟な選択肢                              |
  +-----------------------------------------------+
  | EC2 Instance Savings Plans                     |
  | - 特定リージョン・ファミリーに限定               |
  | - インスタンスサイズ・OS は変更可能              |
  | - 割引率: 最大 72%                              |
  | - 高い割引率を求める場合                        |
  +-----------------------------------------------+
  | SageMaker Savings Plans                        |
  | - SageMaker インスタンスに適用                  |
  | - 割引率: 最大 64%                              |
  +-----------------------------------------------+
```

### 4.2 支払いオプション比較

| 支払い方法 | 割引率 | キャッシュフロー |
|-----------|--------|---------------|
| 全額前払い (All Upfront) | 最大 | 初期一括支払い |
| 一部前払い (Partial Upfront) | 中間 | 半額前払い + 月額 |
| 前払いなし (No Upfront) | 最小 | 月額のみ |

### 4.3 コード例: Savings Plans の情報取得

```bash
# 推奨 Savings Plans を確認
aws savingsplans describe-savings-plans-offerings \
  --service-codes AmazonEC2 \
  --payment-options NoUpfront \
  --plan-types ComputeSavingsPlans \
  --region us-east-1

# 現在の Savings Plans 一覧
aws savingsplans describe-savings-plans \
  --query 'savingsPlans[].[savingsPlanId,savingsPlanType,commitment,state,start,end]' \
  --output table

# Savings Plans の利用率確認
aws ce get-savings-plans-utilization \
  --time-period Start=2026-01-01,End=2026-02-01 \
  --query 'Total.{Utilization:Utilization.UtilizationPercentage,TotalCommitment:AmortizedCommitment.TotalAmortizedCommitment,TotalSavings:SavingsPlansSavings}'

# Cost Explorer で Savings Plans の推奨を取得
aws ce get-savings-plans-purchase-recommendation \
  --savings-plans-type COMPUTE_SP \
  --payment-option NO_UPFRONT \
  --term-in-years ONE_YEAR \
  --lookback-period-in-days THIRTY_DAYS
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
  /              \  右サイジング + Graviton
 /                \   20-40% 削減
/------------------\
  オンデマンド (ベースライン)
```

| 戦略 | 効果 | 実装難易度 | 優先度 |
|------|------|-----------|--------|
| 未使用リソース削除 | 即効性あり | 低 | 最優先 |
| 右サイジング | 20-30% 削減 | 中 | 高 |
| gp2 → gp3 移行 | 20% 削減（EBS） | 低 | 高 |
| Graviton 移行 | 20-40% 削減 | 中 | 高 |
| Savings Plans | 40-72% 削減 | 低 | 高 |
| スポット活用 | 最大 90% 削減 | 中-高 | 中 |
| 開発環境の停止 | 50-70% 削減 | 低 | 高 |

### 5.1 コスト最適化の実装例

```bash
# 未使用の Elastic IP を検出・削除
aws ec2 describe-addresses \
  --query 'Addresses[?AssociationId==null].[AllocationId,PublicIp]' \
  --output table

# 低使用率インスタンスの検出（CPU 使用率 5% 以下）
aws cloudwatch get-metric-statistics \
  --namespace AWS/EC2 \
  --metric-name CPUUtilization \
  --dimensions Name=InstanceId,Value=i-0123456789abcdef0 \
  --start-time "$(date -u -v-7d +%Y-%m-%dT%H:%M:%SZ)" \
  --end-time "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --period 86400 \
  --statistics Average

# 未アタッチの EBS ボリュームを検出
aws ec2 describe-volumes \
  --filters "Name=status,Values=available" \
  --query 'Volumes[].[VolumeId,Size,VolumeType,CreateTime]' \
  --output table

# 古いスナップショットの検出（90日以上前）
aws ec2 describe-snapshots --owner-ids self \
  --query "Snapshots[?StartTime<='$(date -u -v-90d +%Y-%m-%dT%H:%M:%SZ)'].[SnapshotId,VolumeSize,StartTime,Description]" \
  --output table

# AWS Compute Optimizer の推奨を取得
aws compute-optimizer get-ec2-instance-recommendations \
  --query 'instanceRecommendations[].{Instance:instanceArn,Current:currentInstanceType,Recommended:recommendationOptions[0].instanceType,Finding:finding,Savings:recommendationOptions[0].estimatedMonthlySavings.value}' \
  --output table
```

### 5.2 開発環境の自動停止・起動

```bash
# EventBridge + Lambda で開発環境を夜間停止
# Lambda 関数例（Python）
# 停止対象: Environment=development タグのインスタンス

# 停止スケジュール（毎日 20:00 JST）
aws events put-rule \
  --name stop-dev-instances \
  --schedule-expression "cron(0 11 * * ? *)" \
  --state ENABLED

# 起動スケジュール（毎日 09:00 JST）
aws events put-rule \
  --name start-dev-instances \
  --schedule-expression "cron(0 0 ? * MON-FRI *)" \
  --state ENABLED
```

---

## 6. CloudFormation テンプレート（ALB + ASG）

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: Production-grade ALB + Auto Scaling Group

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
  PublicSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
  PrivateSubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
  CertificateArn:
    Type: String
  LatestAmiId:
    Type: AWS::SSM::Parameter::Value<AWS::EC2::Image::Id>
    Default: /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64

Resources:
  # ALB セキュリティグループ
  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: ALB Security Group
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

  # EC2 セキュリティグループ
  EC2SecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: EC2 Security Group
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          SourceSecurityGroupId: !Ref ALBSecurityGroup

  # ALB
  ApplicationLoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: web-alb
      Type: application
      Scheme: internet-facing
      Subnets: !Ref PublicSubnetIds
      SecurityGroups:
        - !Ref ALBSecurityGroup
      LoadBalancerAttributes:
        - Key: idle_timeout.timeout_seconds
          Value: '60'
        - Key: routing.http.drop_invalid_header_fields.enabled
          Value: 'true'

  # ターゲットグループ
  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: web-tg
      Protocol: HTTP
      Port: 80
      VpcId: !Ref VpcId
      TargetType: instance
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3
      TargetGroupAttributes:
        - Key: deregistration_delay.timeout_seconds
          Value: '30'

  # HTTPS リスナー
  HTTPSListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Protocol: HTTPS
      Port: 443
      Certificates:
        - CertificateArn: !Ref CertificateArn
      SslPolicy: ELBSecurityPolicy-TLS13-1-2-2021-06
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref TargetGroup

  # HTTP → HTTPS リダイレクト
  HTTPListener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ApplicationLoadBalancer
      Protocol: HTTP
      Port: 80
      DefaultActions:
        - Type: redirect
          RedirectConfig:
            Protocol: HTTPS
            Port: '443'
            StatusCode: HTTP_301

  # 起動テンプレート
  LaunchTemplate:
    Type: AWS::EC2::LaunchTemplate
    Properties:
      LaunchTemplateName: web-server-lt
      LaunchTemplateData:
        ImageId: !Ref LatestAmiId
        InstanceType: t3.small
        SecurityGroupIds:
          - !Ref EC2SecurityGroup
        MetadataOptions:
          HttpTokens: required
          HttpEndpoint: enabled
        Monitoring:
          Enabled: true
        BlockDeviceMappings:
          - DeviceName: /dev/xvda
            Ebs:
              VolumeSize: 30
              VolumeType: gp3
              Encrypted: true

  # Auto Scaling グループ
  AutoScalingGroup:
    Type: AWS::AutoScaling::AutoScalingGroup
    Properties:
      AutoScalingGroupName: web-asg
      LaunchTemplate:
        LaunchTemplateId: !Ref LaunchTemplate
        Version: !GetAtt LaunchTemplate.LatestVersionNumber
      MinSize: 2
      MaxSize: 10
      DesiredCapacity: 2
      VPCZoneIdentifier: !Ref PrivateSubnetIds
      TargetGroupARNs:
        - !Ref TargetGroup
      HealthCheckType: ELB
      HealthCheckGracePeriod: 300
      Tags:
        - Key: Name
          Value: web-server
          PropagateAtLaunch: true

  # スケーリングポリシー
  CPUScalingPolicy:
    Type: AWS::AutoScaling::ScalingPolicy
    Properties:
      AutoScalingGroupName: !Ref AutoScalingGroup
      PolicyType: TargetTrackingScaling
      TargetTrackingConfiguration:
        PredefinedMetricSpecification:
          PredefinedMetricType: ASGAverageCPUUtilization
        TargetValue: 60
        ScaleInCooldown: 300
        ScaleOutCooldown: 60

Outputs:
  ALBDNSName:
    Value: !GetAtt ApplicationLoadBalancer.DNSName
  ALBHostedZoneId:
    Value: !GetAtt ApplicationLoadBalancer.CanonicalHostedZoneID
```

---

## 7. アンチパターン

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

```bash
# 悪い例 — EC2 ステータスチェックのみ
--health-check-type EC2

# 良い例 — ELB ヘルスチェック（HTTP レベル）
--health-check-type ELB --health-check-grace-period 300
```

### アンチパターン 4: スポットインスタンスを単一タイプで使う

```bash
# 悪い例 — 1つのインスタンスタイプのみ
"LaunchSpecifications": [
  {"InstanceType": "c5.xlarge", ...}
]
# → そのプールの容量不足で全て中断される可能性

# 良い例 — 複数タイプ・複数 AZ
"Overrides": [
  {"InstanceType": "c5.xlarge"},
  {"InstanceType": "c5a.xlarge"},
  {"InstanceType": "c5d.xlarge"},
  {"InstanceType": "c6i.xlarge"},
  {"InstanceType": "c6g.xlarge"},
  {"InstanceType": "m5.xlarge"}
]
```

### アンチパターン 5: デプロイ時にヘルスチェック猶予期間を設定しない

```bash
# 悪い例 — 猶予期間なし
# → アプリケーション起動前に unhealthy 判定され、無限ループ
--health-check-grace-period 0

# 良い例 — アプリケーション起動時間を考慮
# アプリが起動完了するまで 5 分待つ
--health-check-grace-period 300
```

---

## 8. FAQ

### Q1. ALB と NLB のどちらを選ぶべきか？

HTTP/HTTPS ベースの Web アプリケーションには ALB を選択する。TCP/UDP レベルのルーティング、超低レイテンシ、固定 IP が必要な場合は NLB を選択する。gRPC は ALB が L7 レベルでサポートしているが、NLB でも TCP として通すことは可能。WAF を使いたい場合は ALB が必須。PrivateLink で VPC 間のサービス公開が必要な場合は NLB を使う。

### Q2. スポットインスタンスの中断はどのくらいの頻度で起きるか？

リージョンとインスタンスタイプに依存するが、capacityOptimized 戦略を使い複数タイプを指定すると中断頻度は大幅に下がる。AWS Spot Instance Advisor で中断率を事前確認できる。一般的に、6つ以上のインスタンスタイプと3つ以上の AZ を使うと、中断率は 5% 以下に抑えられることが多い。

### Q3. Savings Plans とリザーブドインスタンスの違いは？

Savings Plans は「コミット金額/時」ベースで柔軟性が高く、インスタンスファミリー・リージョン・OS の変更が可能（Compute SP の場合）。RI は特定インスタンスタイプ・AZ に紐づく。新規購入では Savings Plans を推奨する。RI は既に所有している場合のみ継続利用し、新規購入は SP を選択すべきである。

### Q4. Auto Scaling のスケールアウトが遅い場合の対策は？

1. **予測スケーリング**: ML ベースで事前にスケールする
2. **ウォームプール**: 事前に停止状態のインスタンスを準備
3. **ゴールデン AMI**: 起動時のセットアップ時間を最小化
4. **スケールアウト冷却期間の短縮**: 60秒程度に設定
5. **ステップスケーリング**: 急激な負荷に対応

```bash
# ウォームプールの設定
aws autoscaling put-warm-pool \
  --auto-scaling-group-name web-asg \
  --pool-state Stopped \
  --min-size 2 \
  --max-group-prepared-capacity 5
```

### Q5. ロードバランサーのヘルスチェックが失敗し続ける場合は？

1. **ヘルスチェックパスの確認**: アプリケーションが `/health` で 200 を返すか
2. **セキュリティグループの確認**: ALB → EC2 のポートが開いているか
3. **猶予期間の確認**: アプリ起動に十分な時間が設定されているか
4. **ヘルスチェック間隔とタイムアウト**: タイムアウトが短すぎないか
5. **アプリケーションログの確認**: エラーが発生していないか

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| Auto Scaling | 起動テンプレート + ASG + スケーリングポリシーの3層構成 |
| 混合ポリシー | Graviton + スポット + オンデマンドの組み合わせでコスト最適化 |
| ALB | L7 ルーティング、パス/ホストベース、HTTP/HTTPS、WAF 連携 |
| NLB | L4 ルーティング、超低レイテンシ、固定 IP、PrivateLink |
| スポット | 最大 90% 割引、中断対策必須、capacityOptimized 推奨、複数タイプ |
| Savings Plans | Compute SP で柔軟にコスト削減、1年/3年コミット |
| コスト最適化 | 削除→右サイジング→Graviton→SP/RI→スポットの順で実施 |
| ライフサイクル | フックで起動・終了時のカスタム処理を実装 |
| IaC | CloudFormation / CDK で ALB + ASG を一括管理 |

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
5. Auto Scaling 混合インスタンスポリシー — https://docs.aws.amazon.com/autoscaling/ec2/userguide/ec2-auto-scaling-mixed-instances-groups.html
6. ALB リスナールール — https://docs.aws.amazon.com/elasticloadbalancing/latest/application/listener-update-rules.html
