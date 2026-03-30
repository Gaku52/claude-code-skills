# Amazon Route 53

> AWS のフルマネージド DNS サービスを理解し、ドメイン管理・ルーティングポリシー・ヘルスチェックを活用した高可用性アーキテクチャを構築する

## この章で学ぶこと

1. **Route 53 の基本概念** — ホストゾーン、レコードタイプ、DNS 解決の仕組み
2. **ルーティングポリシー** — シンプル、加重、レイテンシ、フェイルオーバー、位置情報ルーティング
3. **ヘルスチェックと DNS フェイルオーバー** — エンドポイント監視と自動切替
4. **ドメイン管理** — ドメイン登録、移管、DNSSEC の設定
5. **Traffic Flow** — ビジュアルエディタによる高度なルーティング設計
6. **Resolver** — ハイブリッド DNS とオンプレミス連携


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [Amazon VPC 基礎](./00-vpc-basics.md) の内容を理解していること

---

## 1. Route 53 とは

Route 53 は AWS のスケーラブルな DNS サービスで、ドメイン登録、DNS ルーティング、ヘルスチェックの 3 つの機能を提供する。100% の可用性 SLA を持つ唯一の AWS サービスである。

### 図解 1: DNS 解決の流れ

```
ユーザーが www.example.com にアクセス:

  ブラウザ
    │
    ▼
  ┌──────────────────┐
  │ ローカル DNS      │ ← キャッシュあれば即返却
  │ リゾルバ          │
  └────────┬─────────┘
           │ キャッシュなし
           ▼
  ┌──────────────────┐
  │ ルート DNS        │ → .com の権威サーバーを返却
  │ サーバー (.)      │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ TLD DNS サーバー   │ → example.com の NS を返却
  │ (.com)            │
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ Route 53          │ → www.example.com の IP を返却
  │ (権威 DNS)        │    (ルーティングポリシーに基づく)
  │                   │
  │ Hosted Zone:      │
  │ example.com       │
  └────────┬─────────┘
           │
           ▼
  ブラウザが IP で接続 → ALB/CloudFront/EC2
```

### Route 53 の料金体系

| 項目 | 料金 |
|------|------|
| ホストゾーン | $0.50/月（最初の25ゾーン） |
| DNS クエリ（標準） | $0.40/100万クエリ |
| DNS クエリ（Alias） | 無料（AWS リソース宛） |
| DNS クエリ（レイテンシ） | $0.60/100万クエリ |
| DNS クエリ（Geo） | $0.70/100万クエリ |
| ヘルスチェック（基本） | $0.50/月 |
| ヘルスチェック（HTTPS） | $0.75/月 |
| ヘルスチェック（文字列検索付き） | $1.00/月 |
| ドメイン登録（.com） | ~$13/年 |

---

## 2. ホストゾーンとレコード

### パブリック vs プライベートホストゾーン

```
パブリックホストゾーン:
======================
  インターネット上のどこからでも DNS 解決可能
  example.com → 203.0.113.10

プライベートホストゾーン:
========================
  VPC 内部からのみ DNS 解決可能
  internal.example.com → 10.0.1.50

  複数の VPC を関連付け可能:
  ┌───────────────────────────────────────┐
  │ Private Hosted Zone                   │
  │ internal.example.com                  │
  │                                       │
  │  ┌─── VPC-A (ap-northeast-1)         │
  │  ├─── VPC-B (ap-northeast-1)         │
  │  └─── VPC-C (us-east-1)             │
  └───────────────────────────────────────┘
```

### コード例 1: ホストゾーンの作成

```bash
# パブリックホストゾーンの作成
aws route53 create-hosted-zone \
  --name example.com \
  --caller-reference "$(date +%s)" \
  --hosted-zone-config Comment="Production zone"

# プライベートホストゾーンの作成（VPC 内部用）
aws route53 create-hosted-zone \
  --name internal.example.com \
  --caller-reference "$(date +%s)" \
  --vpc VPCRegion=ap-northeast-1,VPCId=vpc-0abc1234 \
  --hosted-zone-config Comment="Internal DNS",PrivateZone=true

# プライベートホストゾーンに追加の VPC を関連付け
aws route53 associate-vpc-with-hosted-zone \
  --hosted-zone-id Z0123456789ABCDEF \
  --vpc VPCRegion=ap-northeast-1,VPCId=vpc-0def5678

# ホストゾーン一覧の取得
aws route53 list-hosted-zones \
  --query 'HostedZones[*].{Name:Name,Id:Id,Private:Config.PrivateZone}' \
  --output table

# ホストゾーンのレコード一覧
aws route53 list-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --output table
```

### コード例 2: DNS レコードの登録

```bash
# A レコード（ALB の Alias）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "my-alb-1234567890.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# AAAA レコード（IPv6 Alias）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "AAAA",
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "my-alb-1234567890.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# MX レコード（メール）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "example.com",
        "Type": "MX",
        "TTL": 3600,
        "ResourceRecords": [
          {"Value": "10 mail1.example.com"},
          {"Value": "20 mail2.example.com"}
        ]
      }
    }]
  }'

# TXT レコード（SPF/DKIM）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "example.com",
        "Type": "TXT",
        "TTL": 300,
        "ResourceRecords": [
          {"Value": "\"v=spf1 include:_spf.google.com ~all\""}
        ]
      }
    }]
  }'

# CNAME レコード（外部サービス）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "blog.example.com",
        "Type": "CNAME",
        "TTL": 300,
        "ResourceRecords": [
          {"Value": "example-blog.netlify.app"}
        ]
      }
    }]
  }'

# CAA レコード（証明書認証局制限）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "example.com",
        "Type": "CAA",
        "TTL": 3600,
        "ResourceRecords": [
          {"Value": "0 issue \"amazon.com\""},
          {"Value": "0 issue \"letsencrypt.org\""},
          {"Value": "0 issuewild \"amazon.com\""},
          {"Value": "0 iodef \"mailto:security@example.com\""}
        ]
      }
    }]
  }'

# 複数レコードの一括変更
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Comment": "Initial DNS setup",
    "Changes": [
      {
        "Action": "CREATE",
        "ResourceRecordSet": {
          "Name": "example.com",
          "Type": "A",
          "AliasTarget": {
            "HostedZoneId": "Z2FDTNDATAQYW2",
            "DNSName": "d111111abcdef8.cloudfront.net",
            "EvaluateTargetHealth": false
          }
        }
      },
      {
        "Action": "CREATE",
        "ResourceRecordSet": {
          "Name": "api.example.com",
          "Type": "A",
          "AliasTarget": {
            "HostedZoneId": "Z14GRHDCWA56QT",
            "DNSName": "api-alb-1234.ap-northeast-1.elb.amazonaws.com",
            "EvaluateTargetHealth": true
          }
        }
      }
    ]
  }'
```

### レコードタイプ一覧

```
┌──────────┬──────────────────────────────────────┐
│ レコード │ 用途                                 │
├──────────┼──────────────────────────────────────┤
│ A        │ IPv4 アドレス                        │
│ AAAA     │ IPv6 アドレス                        │
│ CNAME    │ 別ドメインへの転送（Zone Apex 不可） │
│ Alias    │ AWS リソースへのエイリアス（推奨）   │
│ MX       │ メールサーバー                       │
│ TXT      │ テキスト（SPF, DKIM, 検証用）        │
│ NS       │ ネームサーバー                       │
│ SOA      │ ゾーン管理情報                       │
│ SRV      │ サービスロケーション                 │
│ CAA      │ 証明書認証局制限                     │
│ NAPTR    │ Name Authority Pointer               │
│ DS       │ DNSSEC 委任署名者                    │
└──────────┴──────────────────────────────────────┘
```

---

## 3. ルーティングポリシー

### 図解 2: ルーティングポリシーの比較

```
1. Simple (シンプル):
   DNS Query → 1 つの値を返却（複数値ならランダム）

2. Weighted (加重):
   DNS Query → 重みに基づいて分散
   ┌─ 70% → us-east-1 (v2)
   └─ 30% → us-east-1 (v1)   ← カナリアデプロイに最適

3. Latency (レイテンシ):
   DNS Query → 最もレイテンシが低いリージョンに振分
   東京ユーザー → ap-northeast-1
   米国ユーザー → us-east-1

4. Failover (フェイルオーバー):
   DNS Query → Primary 正常なら Primary、異常なら Secondary
   ┌─ Primary (ap-northeast-1)  ← ヘルスチェック OK
   └─ Secondary (us-west-2)     ← Primary 異常時に切替

5. Geolocation (地理的位置):
   DNS Query → ユーザーの地理的位置に基づいて振分
   日本 → ap-northeast-1
   米国 → us-east-1
   デフォルト → eu-west-1

6. Multivalue Answer:
   DNS Query → 最大 8 個の正常なレコードを返却
   ※ ヘルスチェック付きの簡易ロードバランシング

7. Geoproximity (地理的近接性):
   DNS Query → バイアス値で地理的範囲を調整
   ※ Traffic Flow でのみ使用可能

8. IP-based (IPベース):
   DNS Query → クライアント IP の CIDR 範囲に基づいて振分
   ※ ISP 毎の最適なルーティング等に使用
```

### コード例 3: 加重ルーティング（カナリアデプロイ）

```bash
# v2（新バージョン）に 90% の重みを設定
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "v2-primary",
        "Weight": 90,
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-v2.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# v1（旧バージョン）に 10% の重みを設定
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "v1-canary",
        "Weight": 10,
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-v1.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# カナリアデプロイの段階的な重み変更
# Phase 1: 90/10 → Phase 2: 70/30 → Phase 3: 0/100
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "v2-primary",
        "Weight": 0,
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-v2.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    },
    {
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "v1-canary",
        "Weight": 100,
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-v1.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

### コード例 3b: レイテンシルーティング

```bash
# 東京リージョン
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "tokyo",
        "Region": "ap-northeast-1",
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-tokyo.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# バージニアリージョン
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "virginia",
        "Region": "us-east-1",
        "AliasTarget": {
          "HostedZoneId": "Z35SXDOTRQ7X7K",
          "DNSName": "alb-virginia.us-east-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# フランクフルトリージョン
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "frankfurt",
        "Region": "eu-central-1",
        "AliasTarget": {
          "HostedZoneId": "Z215JYRZR1TBD5",
          "DNSName": "alb-frankfurt.eu-central-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

### コード例 4: フェイルオーバールーティング

```bash
# ヘルスチェックの作成
aws route53 create-health-check \
  --caller-reference "primary-$(date +%s)" \
  --health-check-config '{
    "IPAddress": "203.0.113.1",
    "Port": 443,
    "Type": "HTTPS",
    "ResourcePath": "/health",
    "RequestInterval": 10,
    "FailureThreshold": 3,
    "EnableSNI": true,
    "FullyQualifiedDomainName": "api.example.com"
  }'
# → HealthCheckId: hc-primary-001

# Primary レコード
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "primary",
        "Failover": "PRIMARY",
        "HealthCheckId": "hc-primary-001",
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-primary.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# Secondary レコード（DR リージョン）
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.example.com",
        "Type": "A",
        "SetIdentifier": "secondary",
        "Failover": "SECONDARY",
        "AliasTarget": {
          "HostedZoneId": "Z1H1FL5HABSF5",
          "DNSName": "alb-dr.us-west-2.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# S3 静的ウェブサイトを Secondary（メンテナンスページ）に使用
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "SetIdentifier": "maintenance",
        "Failover": "SECONDARY",
        "AliasTarget": {
          "HostedZoneId": "Z2M4EHUR26P7ZW",
          "DNSName": "s3-website-ap-northeast-1.amazonaws.com",
          "EvaluateTargetHealth": false
        }
      }
    }]
  }'
```

### コード例 4b: 地理的位置ルーティング

```bash
# 日本からのアクセス → 東京リージョン
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "SetIdentifier": "japan",
        "GeoLocation": {
          "CountryCode": "JP"
        },
        "AliasTarget": {
          "HostedZoneId": "Z14GRHDCWA56QT",
          "DNSName": "alb-tokyo.ap-northeast-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# 米国からのアクセス → バージニアリージョン
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "SetIdentifier": "us",
        "GeoLocation": {
          "CountryCode": "US"
        },
        "AliasTarget": {
          "HostedZoneId": "Z35SXDOTRQ7X7K",
          "DNSName": "alb-virginia.us-east-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'

# デフォルト（その他の地域）→ フランクフルトリージョン
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890 \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "www.example.com",
        "Type": "A",
        "SetIdentifier": "default",
        "GeoLocation": {
          "CountryCode": "*"
        },
        "AliasTarget": {
          "HostedZoneId": "Z215JYRZR1TBD5",
          "DNSName": "alb-frankfurt.eu-central-1.elb.amazonaws.com",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

---

## 4. ヘルスチェック

### 図解 3: ヘルスチェックの種類

```
1. エンドポイントヘルスチェック:
   Route 53 ──→ HTTPS GET /health ──→ App
                     │
                     ├─ 2xx/3xx → Healthy
                     └─ Timeout/5xx → Unhealthy

   ※ 世界中の 15+ ヘルスチェッカーから監視
   ※ 18% 以上が Healthy と判定 → 全体 Healthy

2. 計算済みヘルスチェック:
   ┌─ HC-1 (us-east-1)  → Healthy ─┐
   ├─ HC-2 (eu-west-1)  → Healthy ─┼→ AND/OR → 全体結果
   └─ HC-3 (ap-ne-1)    → Unhealthy┘

3. CloudWatch アラームヘルスチェック:
   CloudWatch Alarm → Route 53 Health Check
   (DynamoDB, Lambda 等の内部リソース監視に使用)
```

### コード例 5: さまざまなヘルスチェック設定

```bash
# HTTPS エンドポイントヘルスチェック（文字列検索付き）
aws route53 create-health-check \
  --caller-reference "api-health-$(date +%s)" \
  --health-check-config '{
    "Type": "HTTPS_STR_MATCH",
    "FullyQualifiedDomainName": "api.example.com",
    "Port": 443,
    "ResourcePath": "/health",
    "SearchString": "\"status\":\"ok\"",
    "RequestInterval": 10,
    "FailureThreshold": 3,
    "EnableSNI": true,
    "Regions": ["us-east-1", "eu-west-1", "ap-southeast-1"]
  }'

# TCP ヘルスチェック（データベース等）
aws route53 create-health-check \
  --caller-reference "db-health-$(date +%s)" \
  --health-check-config '{
    "Type": "TCP",
    "IPAddress": "10.0.1.50",
    "Port": 5432,
    "RequestInterval": 30,
    "FailureThreshold": 3
  }'

# 計算済みヘルスチェック（子ヘルスチェックの組み合わせ）
aws route53 create-health-check \
  --caller-reference "calculated-$(date +%s)" \
  --health-check-config '{
    "Type": "CALCULATED",
    "ChildHealthChecks": [
      "hc-api-001",
      "hc-db-001",
      "hc-cache-001"
    ],
    "HealthThreshold": 2
  }'

# CloudWatch アラーム連動ヘルスチェック
aws route53 create-health-check \
  --caller-reference "cw-alarm-$(date +%s)" \
  --health-check-config '{
    "Type": "CLOUDWATCH_METRIC",
    "AlarmIdentifier": {
      "Region": "ap-northeast-1",
      "Name": "API-5xx-Error-Rate"
    },
    "InsufficientDataHealthStatus": "Unhealthy"
  }'

# ヘルスチェックにタグを付ける
aws route53 change-tags-for-resource \
  --resource-type healthcheck \
  --resource-id hc-api-001 \
  --add-tags Key=Name,Value="API Health Check" Key=Environment,Value=production

# ヘルスチェックの状態確認
aws route53 get-health-check-status \
  --health-check-id hc-api-001 \
  --query 'HealthCheckObservations[*].{Region:Region,Status:StatusReport.Status}'
```

### コード例 6: Terraform でヘルスチェック付きフェイルオーバー

```hcl
# ヘルスチェック
resource "aws_route53_health_check" "primary" {
  fqdn              = "api.example.com"
  port               = 443
  type               = "HTTPS"
  resource_path      = "/health"
  failure_threshold  = 3
  request_interval   = 10

  regions = [
    "us-east-1",
    "eu-west-1",
    "ap-southeast-1",
  ]

  tags = {
    Name = "primary-health-check"
  }
}

# CloudWatch アラーム連動ヘルスチェック
resource "aws_route53_health_check" "cloudwatch" {
  type                            = "CLOUDWATCH_METRIC"
  cloudwatch_alarm_name           = aws_cloudwatch_metric_alarm.api_error.alarm_name
  cloudwatch_alarm_region         = "ap-northeast-1"
  insufficient_data_health_status = "Unhealthy"
}

# 計算済みヘルスチェック
resource "aws_route53_health_check" "calculated" {
  type                   = "CALCULATED"
  child_health_threshold = 2
  child_healthchecks = [
    aws_route53_health_check.primary.id,
    aws_route53_health_check.cloudwatch.id,
  ]

  tags = {
    Name = "calculated-health-check"
  }
}

# Primary レコード
resource "aws_route53_record" "primary" {
  zone_id         = aws_route53_zone.main.zone_id
  name            = "api.example.com"
  type            = "A"
  set_identifier  = "primary"
  health_check_id = aws_route53_health_check.primary.id

  failover_routing_policy {
    type = "PRIMARY"
  }

  alias {
    name                   = aws_lb.primary.dns_name
    zone_id                = aws_lb.primary.zone_id
    evaluate_target_health = true
  }
}

# Secondary レコード
resource "aws_route53_record" "secondary" {
  zone_id        = aws_route53_zone.main.zone_id
  name           = "api.example.com"
  type           = "A"
  set_identifier = "secondary"

  failover_routing_policy {
    type = "SECONDARY"
  }

  alias {
    name                   = aws_lb.secondary.dns_name
    zone_id                = aws_lb.secondary.zone_id
    evaluate_target_health = true
  }
}

# レイテンシルーティング（マルチリージョン）
resource "aws_route53_record" "latency_tokyo" {
  zone_id        = aws_route53_zone.main.zone_id
  name           = "global.example.com"
  type           = "A"
  set_identifier = "tokyo"

  latency_routing_policy {
    region = "ap-northeast-1"
  }

  alias {
    name                   = aws_lb.tokyo.dns_name
    zone_id                = aws_lb.tokyo.zone_id
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "latency_virginia" {
  zone_id        = aws_route53_zone.main.zone_id
  name           = "global.example.com"
  type           = "A"
  set_identifier = "virginia"

  latency_routing_policy {
    region = "us-east-1"
  }

  alias {
    name                   = aws_lb.virginia.dns_name
    zone_id                = aws_lb.virginia.zone_id
    evaluate_target_health = true
  }
}
```

---

## 5. ルーティングポリシー比較

### 比較表 1: ルーティングポリシー選定

| ポリシー | 用途 | ヘルスチェック | 複雑さ |
|----------|------|---------------|--------|
| **Simple** | 単一リソース | なし | 低 |
| **Weighted** | カナリア / A-B テスト | 対応 | 低 |
| **Latency** | マルチリージョン最適化 | 対応 | 中 |
| **Failover** | Active-Passive DR | 必須 | 中 |
| **Geolocation** | 地域制限 / コンプライアンス | 対応 | 中 |
| **Multivalue** | 簡易ロードバランシング | 対応 | 低 |
| **Geoproximity** | 地理的範囲調整 | 対応 | 高 |
| **IP-based** | ISP / ネットワーク最適化 | 対応 | 高 |

### 比較表 2: Alias vs CNAME

| 項目 | Alias | CNAME |
|------|-------|-------|
| **Zone Apex 対応** | 可 (example.com) | 不可 |
| **DNS クエリ課金** | 無料（AWS リソース宛） | 有料 |
| **ヘルスチェック** | EvaluateTargetHealth で連携 | 別途設定 |
| **対応先** | ALB, CloudFront, S3, API GW 等 | 任意のドメイン |
| **TTL** | AWS が自動管理 | 自分で設定 |
| **推奨** | AWS リソースには必ず Alias | 外部サービスのみ |

### 対応する AWS リソースの Alias ホストゾーン ID

```
主要サービスの Alias ターゲット HostedZoneId:
=============================================

CloudFront:       Z2FDTNDATAQYW2 (全リージョン共通)
API Gateway:      リージョンごとに異なる
S3 Website:       リージョンごとに異なる

ALB/NLB:
  ap-northeast-1: Z14GRHDCWA56QT
  us-east-1:      Z35SXDOTRQ7X7K
  us-west-2:      Z1H1FL5HABSF5
  eu-west-1:      Z32O12XQLNTSW2
  eu-central-1:   Z215JYRZR1TBD5
```

---

## 6. DNSSEC

DNSSEC（DNS Security Extensions）は、DNS レスポンスの真正性を検証するためのセキュリティ拡張である。

```bash
# DNSSEC の有効化
# Step 1: KSK（Key Signing Key）の作成
aws route53 create-key-signing-key \
  --hosted-zone-id Z1234567890 \
  --name my-ksk-key \
  --key-management-service-arn arn:aws:kms:us-east-1:123456789012:key/xxx-xxx \
  --status ACTIVE

# Step 2: DNSSEC 署名の有効化
aws route53 enable-hosted-zone-dnssec \
  --hosted-zone-id Z1234567890

# Step 3: DS レコードを親ゾーン（レジストラ）に登録
# Route 53 で登録したドメインの場合:
aws route53domains enable-domain-transfer-lock \
  --domain-name example.com

# DNSSEC の状態確認
aws route53 get-dnssec \
  --hosted-zone-id Z1234567890
```

---

## 7. Route 53 Resolver

ハイブリッド環境でのDNS解決を実現する。

```
Route 53 Resolver のアーキテクチャ:
====================================

オンプレミス DNS                        AWS VPC
+------------------+                  +------------------+
|                  |                  |                  |
| Corporate DNS    |  ←── Outbound   | Route 53         |
| (10.0.0.53)     |       Endpoint   | Resolver         |
|                  |                  |                  |
|                  |  ──→ Inbound    |                  |
|                  |       Endpoint   |                  |
+------------------+                  +------------------+

Inbound Endpoint:  オンプレミス → AWS の DNS 解決
Outbound Endpoint: AWS → オンプレミスの DNS 解決
Resolver Rules:    条件付き転送ルール
```

```bash
# Inbound Endpoint の作成（オンプレミスからの DNS 解決用）
aws route53resolver create-resolver-endpoint \
  --creator-request-id "inbound-$(date +%s)" \
  --name "inbound-resolver" \
  --security-group-ids sg-0abc123 \
  --direction INBOUND \
  --ip-addresses SubnetId=subnet-0123,Ip=10.0.1.10 SubnetId=subnet-0456,Ip=10.0.2.10

# Outbound Endpoint の作成（AWS からオンプレミスへの DNS 解決用）
aws route53resolver create-resolver-endpoint \
  --creator-request-id "outbound-$(date +%s)" \
  --name "outbound-resolver" \
  --security-group-ids sg-0abc123 \
  --direction OUTBOUND \
  --ip-addresses SubnetId=subnet-0123 SubnetId=subnet-0456

# 転送ルールの作成（特定ドメインをオンプレミスに転送）
aws route53resolver create-resolver-rule \
  --creator-request-id "forward-$(date +%s)" \
  --name "forward-to-onprem" \
  --rule-type FORWARD \
  --domain-name "corp.internal" \
  --resolver-endpoint-id rslvr-out-xxx \
  --target-ips Ip=10.0.0.53,Port=53

# ルールを VPC に関連付け
aws route53resolver associate-resolver-rule \
  --resolver-rule-id rslvr-rr-xxx \
  --vpc-id vpc-0abc1234
```

### コード例 7: Resolver Terraform 定義

```hcl
resource "aws_route53_resolver_endpoint" "inbound" {
  name               = "inbound-resolver"
  direction          = "INBOUND"
  security_group_ids = [aws_security_group.resolver.id]

  ip_address {
    subnet_id = aws_subnet.private_a.id
    ip        = "10.0.1.10"
  }

  ip_address {
    subnet_id = aws_subnet.private_c.id
    ip        = "10.0.2.10"
  }

  tags = { Name = "inbound-resolver" }
}

resource "aws_route53_resolver_endpoint" "outbound" {
  name               = "outbound-resolver"
  direction          = "OUTBOUND"
  security_group_ids = [aws_security_group.resolver.id]

  ip_address {
    subnet_id = aws_subnet.private_a.id
  }

  ip_address {
    subnet_id = aws_subnet.private_c.id
  }

  tags = { Name = "outbound-resolver" }
}

resource "aws_route53_resolver_rule" "forward_to_onprem" {
  domain_name          = "corp.internal"
  name                 = "forward-to-onprem"
  rule_type            = "FORWARD"
  resolver_endpoint_id = aws_route53_resolver_endpoint.outbound.id

  target_ip {
    ip   = "10.0.0.53"
    port = 53
  }

  target_ip {
    ip   = "10.0.0.54"
    port = 53
  }
}

resource "aws_route53_resolver_rule_association" "forward" {
  resolver_rule_id = aws_route53_resolver_rule.forward_to_onprem.id
  vpc_id           = aws_vpc.main.id
}
```

---

## 8. Route 53 Profiles

複数アカウント/VPC での DNS 設定を一元管理する機能。

```
Route 53 Profiles:
==================

Profile
  ├── DNS Firewall Rule Group Association
  ├── Private Hosted Zone Association
  └── Resolver Rule Association

  → 1つの Profile を複数の VPC に関連付け
  → AWS Organizations 全体で共有可能
  → DNS 設定の一貫性を保証
```

---

## 9. アンチパターン

### アンチパターン 1: TTL を極端に短くする

```
[悪い例]
  TTL = 5 秒 (全レコード)
  → DNS クエリが増加しコストが上がる
  → DNS リゾルバへの負荷が増加
  → ユーザーのレイテンシが微増

[良い例]
  通常レコード:    TTL = 300 秒 (5 分)
  フェイルオーバー: TTL = 60 秒 (1 分)
  移行中:          TTL = 60 秒 → 移行後に 300 秒に戻す

  ポイント:
  - 移行前に TTL を下げておく（旧 TTL の 2 倍の時間前）
  - 移行完了後は TTL を戻す
  - Alias レコードは TTL が自動管理される
```

### アンチパターン 2: ヘルスチェックなしのフェイルオーバー

```
[悪い例]
  Primary → ALB (ヘルスチェックなし)
  Secondary → S3 Static Site

  Primary の ALB が異常でもフェイルオーバーが発生しない
  → ユーザーはエラーページを見続ける

[良い例]
  Primary → ALB (ヘルスチェック: /health を監視)
  Secondary → S3 Static Site ("メンテナンス中" ページ)

  ヘルスチェック設定:
  - Type: HTTPS
  - Path: /health
  - Interval: 10 秒
  - Failure Threshold: 3
  - EvaluateTargetHealth: true (ALB の背後も監視)
```

### アンチパターン 3: Zone Apex に CNAME を使用する

```
[悪い例]
  example.com → CNAME → d111111.cloudfront.net
  → RFC 違反: Zone Apex に CNAME は設定不可
  → DNS エラーが発生する

[良い例]
  example.com → Alias → d111111.cloudfront.net
  → Zone Apex でも使用可能
  → DNS クエリ無料
  → TTL 自動管理
```

### アンチパターン 4: Geolocation でデフォルトを設定しない

```
[悪い例]
  JP → ap-northeast-1
  US → us-east-1
  → JP/US 以外のユーザーは DNS 解決に失敗（NXDOMAIN）

[良い例]
  JP → ap-northeast-1
  US → us-east-1
  * (デフォルト) → eu-west-1  ← 必ずデフォルトを設定
```

---

## 10. DNS Firewall

Route 53 Resolver DNS Firewall は、VPC からの DNS クエリをフィルタリングし、悪意のあるドメインへのアクセスをブロックする。

```bash
# DNS Firewall ドメインリストの作成
aws route53resolver create-firewall-domain-list \
  --name "blocked-domains" \
  --creator-request-id "blocked-$(date +%s)"

# ドメインの追加
aws route53resolver update-firewall-domains \
  --firewall-domain-list-id rslvr-fdl-xxx \
  --operation ADD \
  --domains "*.malware.example.com" "phishing.example.com" "*.crypto-mining.example.com"

# ファイアウォールルールグループの作成
aws route53resolver create-firewall-rule-group \
  --name "security-rules" \
  --creator-request-id "rules-$(date +%s)"

# ルールの追加（ブロック）
aws route53resolver create-firewall-rule \
  --firewall-rule-group-id rslvr-frg-xxx \
  --firewall-domain-list-id rslvr-fdl-xxx \
  --priority 100 \
  --action BLOCK \
  --block-response NXDOMAIN \
  --name "block-malware"

# AWS Managed ドメインリストの利用（推奨）
# AmazonGuardDutyThreatList - GuardDuty が検出した脅威ドメイン
# AmazonRegisteredDomains - AWS に登録されたドメイン

# ルールグループを VPC に関連付け
aws route53resolver associate-firewall-rule-group \
  --firewall-rule-group-id rslvr-frg-xxx \
  --vpc-id vpc-0abc1234 \
  --priority 101 \
  --name "protect-vpc"
```

### Terraform による DNS Firewall 設定

```hcl
resource "aws_route53_resolver_firewall_domain_list" "blocked" {
  name    = "blocked-domains"
  domains = ["*.malware.example.com", "*.crypto-mining.example.com"]
}

resource "aws_route53_resolver_firewall_rule_group" "security" {
  name = "security-rules"
}

resource "aws_route53_resolver_firewall_rule" "block" {
  name                    = "block-malware"
  action                  = "BLOCK"
  block_response          = "NXDOMAIN"
  firewall_domain_list_id = aws_route53_resolver_firewall_domain_list.blocked.id
  firewall_rule_group_id  = aws_route53_resolver_firewall_rule_group.security.id
  priority                = 100
}

resource "aws_route53_resolver_firewall_rule_group_association" "main" {
  name                   = "protect-vpc"
  firewall_rule_group_id = aws_route53_resolver_firewall_rule_group.security.id
  vpc_id                 = aws_vpc.main.id
  priority               = 101
}
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---

## 11. FAQ

### Q1: Route 53 でドメインを購入できますか？

**A:** はい、Route 53 はドメインレジストラとしても機能する。`.com`、`.jp`、`.io` など多数の TLD に対応している。購入したドメインは自動的にホストゾーンが作成される。年間登録料はドメインによって異なる（.com は約 $13/年）。ただし .co.jp などの一部ドメインは Route 53 では購入できないため、外部レジストラ（お名前.com 等）で購入し、NS レコードを Route 53 に向ける方法が一般的。

### Q2: CNAME と Alias の使い分けは？

**A:** AWS リソース（ALB、CloudFront、S3 等）への参照には必ず Alias を使用する。Alias は Zone Apex（example.com）にも設定でき、DNS クエリも無料。CNAME は Zone Apex に設定できず、クエリ課金もされる。外部サービス（Heroku、Vercel 等）へのポイントには CNAME を使用する。

### Q3: マルチリージョンの DNS 設計はどうすべきですか？

**A:** レイテンシルーティング + ヘルスチェック + フェイルオーバーを組み合わせる。まずレイテンシルーティングで最寄りリージョンに振り分け、ヘルスチェックで異常を検知したらフェイルオーバーする。この構成を Terraform の `aws_route53_record` で宣言的に管理し、IaC で全リージョンを統一管理するのがベストプラクティスである。

### Q4: Route 53 で DNS の変更が反映されるまでの時間は？

**A:** Route 53 自体への変更は通常 60 秒以内に全世界のエッジロケーションに伝播する。ただし、既存のレコードの変更の場合、以前の TTL が切れるまでキャッシュが残る。そのため、DNS 移行時は事前に TTL を短く（60秒程度に）設定しておき、変更後に TTL を戻すのがベストプラクティスである。

### Q5: Route 53 のクエリログを取得するには？

**A:** Route 53 Query Logging を使用する。CloudWatch Logs にクエリログを送信でき、クエリされたドメイン名、レコードタイプ、レスポンスコード、ソース IP 等を記録できる。セキュリティ監査やトラブルシューティングに有用。

```bash
# クエリログの有効化
aws route53resolver create-resolver-query-log-config \
  --name "dns-query-log" \
  --destination-arn "arn:aws:logs:ap-northeast-1:123456789012:log-group:/route53/query-log"

# VPC との関連付け
aws route53resolver associate-resolver-query-log-config \
  --resolver-query-log-config-id rqlc-xxx \
  --resource-id vpc-0abc1234
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| ホストゾーン | パブリック（インターネット）とプライベート（VPC 内部）を使い分け |
| レコードタイプ | AWS リソースには Alias を優先。Zone Apex にも対応 |
| ルーティング | 用途に応じてポリシーを選択。フェイルオーバーは本番必須 |
| ヘルスチェック | エンドポイント監視 + EvaluateTargetHealth を組合せ |
| TTL | 通常 300 秒。移行時は事前に短縮、完了後に戻す |
| コスト | Alias クエリは無料。ヘルスチェックは数百円/月 |
| DNSSEC | ゾーン署名でDNSレスポンスの真正性を検証 |
| Resolver | ハイブリッド環境での DNS 解決（Inbound/Outbound） |
| セキュリティ | CAA レコード + DNSSEC + クエリログで保護 |

---

## 次に読むべきガイド

- [02-api-gateway.md](./02-api-gateway.md) — Route 53 と連携する API Gateway
- [00-vpc-basics.md](./00-vpc-basics.md) — プライベートホストゾーンの基盤となる VPC
- [02-waf-shield.md](../08-security/02-waf-shield.md) — Route 53 + Shield による DDoS 対策

---

## 参考文献

1. **AWS 公式ドキュメント** — Amazon Route 53 開発者ガイド
   https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/
2. **AWS Route 53 ルーティングポリシー** — 各ポリシーの詳細と設定方法
   https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/routing-policy.html
3. **AWS Well-Architected — Reliability Pillar** — DNS とフェイルオーバー設計
   https://docs.aws.amazon.com/wellarchitected/latest/reliability-pillar/
4. **AWS Route 53 Resolver** — ハイブリッド DNS 設計ガイド
   https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/resolver.html
5. **DNSSEC 署名** — Route 53 での DNSSEC 設定
   https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/dns-configuring-dnssec.html
