# Amazon Route 53

> AWS のフルマネージド DNS サービスを理解し、ドメイン管理・ルーティングポリシー・ヘルスチェックを活用した高可用性アーキテクチャを構築する

## この章で学ぶこと

1. **Route 53 の基本概念** — ホストゾーン、レコードタイプ、DNS 解決の仕組み
2. **ルーティングポリシー** — シンプル、加重、レイテンシ、フェイルオーバー、位置情報ルーティング
3. **ヘルスチェックと DNS フェイルオーバー** — エンドポイント監視と自動切替

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

---

## 2. ホストゾーンとレコード

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

### コード例 5: Terraform でヘルスチェック付きフェイルオーバー

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

### 比較表 2: Alias vs CNAME

| 項目 | Alias | CNAME |
|------|-------|-------|
| **Zone Apex 対応** | 可 (example.com) | 不可 |
| **DNS クエリ課金** | 無料（AWS リソース宛） | 有料 |
| **ヘルスチェック** | EvaluateTargetHealth で連携 | 別途設定 |
| **対応先** | ALB, CloudFront, S3, API GW 等 | 任意のドメイン |
| **TTL** | AWS が自動管理 | 自分で設定 |
| **推奨** | AWS リソースには必ず Alias | 外部サービスのみ |

---

## 6. アンチパターン

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

---

## 7. FAQ

### Q1: Route 53 でドメインを購入できますか？

**A:** はい、Route 53 はドメインレジストラとしても機能する。`.com`、`.jp`、`.io` など多数の TLD に対応している。購入したドメインは自動的にホストゾーンが作成される。年間登録料はドメインによって異なる（.com は約 $13/年）。ただし .co.jp などの一部ドメインは Route 53 では購入できないため、外部レジストラ（お名前.com 等）で購入し、NS レコードを Route 53 に向ける方法が一般的。

### Q2: CNAME と Alias の使い分けは？

**A:** AWS リソース（ALB、CloudFront、S3 等）への参照には必ず Alias を使用する。Alias は Zone Apex（example.com）にも設定でき、DNS クエリも無料。CNAME は Zone Apex に設定できず、クエリ課金もされる。外部サービス（Heroku、Vercel 等）へのポイントには CNAME を使用する。

### Q3: マルチリージョンの DNS 設計はどうすべきですか？

**A:** レイテンシルーティング + ヘルスチェック + フェイルオーバーを組み合わせる。まずレイテンシルーティングで最寄りリージョンに振り分け、ヘルスチェックで異常を検知したらフェイルオーバーする。この構成を Terraform の `aws_route53_record` で宣言的に管理し、IaC で全リージョンを統一管理するのがベストプラクティスである。

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
