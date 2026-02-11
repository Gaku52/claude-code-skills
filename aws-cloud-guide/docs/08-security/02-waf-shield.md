# WAF / Shield — WAF ルールと DDoS 対策

> AWS WAF でアプリケーション層（L7）の攻撃を防御し、AWS Shield で DDoS 攻撃（L3/L4）から保護するための実践ガイド。

---

## この章で学ぶこと

1. **AWS WAF** のルール設計と Web ACL によるリクエストフィルタリング
2. **AWS Shield Standard / Advanced** による DDoS 防御アーキテクチャ
3. **WAF + CloudFront + ALB** を組み合わせた多層防御パターン

---

## 1. AWS WAF の全体アーキテクチャ

### 1.1 WAF の配置と処理フロー

```
┌─────────┐     ┌──────────────┐     ┌───────────┐     ┌─────────┐
│ Client  │ ──→ │ CloudFront   │ ──→ │  ALB      │ ──→ │ EC2/ECS │
│ (攻撃者)│     │ + WAF        │     │ + WAF     │     │ App     │
└─────────┘     │ (エッジ防御) │     │ (リージョン│     └─────────┘
                └──────┬───────┘     │  防御)    │
                       │             └─────┬─────┘
                 Web ACL 評価          Web ACL 評価
                       │                   │
                ┌──────▼───────┐     ┌─────▼─────┐
                │ Allow/Block  │     │ Allow/Block│
                │ /Count/CAPTCHA│    │ /Count    │
                └──────────────┘     └───────────┘
```

### 1.2 WAF のルール評価順序

```
┌─────────────────────────────────────────────────┐
│              Web ACL ルール評価                   │
│                                                 │
│  Priority 0:  IP ブロックリスト (Block)          │
│       │                                         │
│       ▼ マッチしなければ次へ                     │
│  Priority 1:  AWS Managed Rules - Common        │
│       │       (SQLi, XSS 等を Block)            │
│       ▼                                         │
│  Priority 2:  レートベースルール                 │
│       │       (2000 req/5min 超過で Block)       │
│       ▼                                         │
│  Priority 3:  Geo マッチルール                   │
│       │       (特定国からの Block)               │
│       ▼                                         │
│  Priority 4:  カスタムルール                     │
│       │       (Bot 検知等)                       │
│       ▼                                         │
│  Default Action: Allow                          │
│  (どのルールにもマッチしなかったリクエスト)       │
└─────────────────────────────────────────────────┘
```

---

## 2. WAF ルール設計

### 2.1 AWS マネージドルールの適用（CloudFormation）

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  WebACL:
    Type: AWS::WAFv2::WebACL
    Properties:
      Name: production-web-acl
      Scope: REGIONAL    # ALB/API Gateway 用。CloudFront は CLOUDFRONT
      DefaultAction:
        Allow: {}
      VisibilityConfig:
        CloudWatchMetricsEnabled: true
        MetricName: production-web-acl
        SampledRequestsEnabled: true
      Rules:
        # AWS マネージドルール: 一般的な脆弱性
        - Name: AWSManagedRulesCommonRuleSet
          Priority: 0
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesCommonRuleSet
              ExcludedRules:
                - Name: SizeRestrictions_BODY    # 大きいPOSTが必要な場合
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: CommonRuleSet
            SampledRequestsEnabled: true

        # AWS マネージドルール: SQLインジェクション
        - Name: AWSManagedRulesSQLiRuleSet
          Priority: 1
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesSQLiRuleSet
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: SQLiRuleSet
            SampledRequestsEnabled: true

        # レートベースルール
        - Name: RateLimitRule
          Priority: 2
          Action:
            Block: {}
          Statement:
            RateBasedStatement:
              Limit: 2000           # 5分間で2000リクエスト
              AggregateKeyType: IP
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: RateLimitRule
            SampledRequestsEnabled: true
```

### 2.2 カスタムルール: 特定パスの保護

```yaml
        # /admin パスへのアクセスを IP 制限
        - Name: AdminPathIPRestriction
          Priority: 3
          Action:
            Block: {}
          Statement:
            AndStatement:
              Statements:
                - ByteMatchStatement:
                    SearchString: /admin
                    FieldToMatch:
                      UriPath: {}
                    TextTransformations:
                      - Priority: 0
                        Type: LOWERCASE
                    PositionalConstraint: STARTS_WITH
                - NotStatement:
                    Statement:
                      IPSetReferenceStatement:
                        Arn: !GetAtt AllowedAdminIPs.Arn
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: AdminIPRestriction
            SampledRequestsEnabled: true

  AllowedAdminIPs:
    Type: AWS::WAFv2::IPSet
    Properties:
      Name: allowed-admin-ips
      Scope: REGIONAL
      IPAddressVersion: IPV4
      Addresses:
        - 203.0.113.0/24     # オフィス IP
        - 198.51.100.10/32   # VPN IP
```

### 2.3 Terraform での WAF 構成

```hcl
resource "aws_wafv2_web_acl" "main" {
  name        = "production-web-acl"
  scope       = "REGIONAL"
  description = "Production WAF ACL"

  default_action {
    allow {}
  }

  # Bot Control
  rule {
    name     = "BotControl"
    priority = 0

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesBotControlRuleSet"

        managed_rule_group_configs {
          aws_managed_rules_bot_control_rule_set {
            inspection_level = "COMMON"
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "BotControl"
      sampled_requests_enabled   = true
    }
  }

  # カスタム: ログインエンドポイントへのレート制限
  rule {
    name     = "LoginRateLimit"
    priority = 1

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 100    # 5分間で100回
        aggregate_key_type = "IP"

        scope_down_statement {
          byte_match_statement {
            positional_constraint = "STARTS_WITH"
            search_string         = "/api/login"

            field_to_match {
              uri_path {}
            }

            text_transformation {
              priority = 0
              type     = "LOWERCASE"
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "LoginRateLimit"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "production-web-acl"
    sampled_requests_enabled   = true
  }
}
```

### 2.4 WAF ログの有効化と分析

```bash
# WAF ログを S3 に送信
aws wafv2 put-logging-configuration \
  --logging-configuration '{
    "ResourceArn": "arn:aws:wafv2:ap-northeast-1:123456789012:regional/webacl/production-web-acl/xxxx",
    "LogDestinationConfigs": [
      "arn:aws:s3:::my-waf-logs-bucket"
    ],
    "RedactedFields": [
      {"SingleHeader": {"Name": "authorization"}}
    ]
  }'
```

```python
# Athena でブロックされたリクエストを分析
QUERY = """
SELECT
  httprequest.clientip,
  httprequest.uri,
  terminatingruleid,
  COUNT(*) as block_count
FROM waf_logs
WHERE action = 'BLOCK'
  AND from_unixtime(timestamp/1000) > current_timestamp - interval '24' hour
GROUP BY 1, 2, 3
ORDER BY block_count DESC
LIMIT 20
"""
```

---

## 3. AWS Shield

### 3.1 Shield Standard vs Advanced

```
┌──────────────────────────────────────────────────────────┐
│                   Shield Standard                        │
│  (全 AWS アカウントに自動適用・無料)                      │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │ L3/L4 DDoS 防御                                │      │
│  │ - SYN/UDP Flood                                │      │
│  │ - Reflection 攻撃                              │      │
│  │ - CloudFront / Route 53 で自動軽減             │      │
│  └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   Shield Advanced                        │
│  ($3,000/月 + データ転送料、1年コミット)                  │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │ Standard の全機能 +                             │      │
│  │ - L7 DDoS 検知・自動緩和                       │      │
│  │ - DDoS Response Team (DRT) 24/7 サポート       │      │
│  │ - コスト保護（DDoS 起因のスケールアウト費用）   │      │
│  │ - リアルタイム攻撃可視化                       │      │
│  │ - WAF 料金が Shield Advanced に含まれる         │      │
│  │ - Health-based 検知（Route 53 ヘルスチェック）  │      │
│  └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 多層防御アーキテクチャ

### 4.1 完全な防御構成

```
Internet
    │
    ▼
┌──────────────────┐
│ Route 53         │  ← Shield Standard (DNS DDoS 防御)
│ (DNS)            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CloudFront       │  ← Shield Standard/Advanced + WAF
│ (CDN + Edge WAF) │     Geo Restriction
│                  │     Origin Access Control
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ALB              │  ← WAF (Regional) + Security Group
│ (Load Balancer)  │     リクエスト検証
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ECS / EC2        │  ← Security Group (ALB からのみ)
│ (Application)    │     アプリケーション側バリデーション
└──────────────────┘
```

---

## 5. 比較表

### 5.1 WAF マネージドルールグループ比較

| ルールグループ | 防御対象 | WCU | 推奨 |
|--------------|---------|-----|------|
| **AWSManagedRulesCommonRuleSet** | XSS, パストラバーサル, ファイルインジェクション | 700 | 全アプリ必須 |
| **AWSManagedRulesSQLiRuleSet** | SQL インジェクション | 200 | DB 利用アプリ必須 |
| **AWSManagedRulesKnownBadInputsRuleSet** | Log4j, 既知の脆弱性 | 200 | Java アプリは必須 |
| **AWSManagedRulesBotControlRuleSet** | Bot 検知・制御 | 50 | EC サイト推奨 |
| **AWSManagedRulesATPRuleSet** | アカウント乗っ取り防止 | 50 | 認証ありアプリ |
| **AWSManagedRulesAnonymousIPList** | VPN/Tor/プロキシ | 50 | セキュリティ重視 |

### 5.2 DDoS 対策レイヤー比較

| レイヤー | 攻撃例 | 防御サービス | 自動/手動 |
|---------|--------|-------------|----------|
| **L3 (ネットワーク)** | UDP Flood, ICMP Flood | Shield Standard | 自動 |
| **L4 (トランスポート)** | SYN Flood, TCP RST | Shield Standard | 自動 |
| **L7 (アプリケーション)** | HTTP Flood, Slowloris | WAF + Shield Advanced | ルール設定必要 |
| **DNS** | DNS Query Flood | Route 53 + Shield | 自動 |
| **API** | API 呼び出し集中 | API Gateway Throttling + WAF | 設定必要 |

---

## 6. アンチパターン

### 6.1 WAF ルールを Count モードのまま放置

```
NG: 導入時に Count モードで開始 → そのまま数ヶ月放置
    → 攻撃をログに記録するだけでブロックしていない

OK: 段階的な移行プロセス
    Week 1-2: Count モードで誤検知を監視
    Week 3:   誤検知ルールを ExcludedRules に追加
    Week 4:   Block モードに切り替え
    継続:     CloudWatch アラームでブロック率を監視
```

**対策**: Count モードでの運用期間をプロジェクト計画に組み込み、切り替えの判断基準を事前に定義する。

### 6.2 WAF を ALB にだけ設定し CloudFront を経由しない

```
NG:
  Client → ALB (WAF) → App
  問題: DDoS が ALB に直撃、オリジン IP が露出

OK:
  Client → CloudFront (WAF + Shield) → ALB (SG: CloudFront のみ許可) → App
  利点: エッジで攻撃軽減、ALB への直接アクセスを遮断
```

**対策**: ALB のセキュリティグループで CloudFront の IP レンジのみを許可し、直接アクセスをブロックする。AWS が公開する CloudFront IP リストを AWS Managed Prefix List で参照可能。

---

## 7. FAQ

### Q1. WAF の WCU（Web ACL Capacity Unit）制限にどう対処する？

**A.** Web ACL のデフォルト上限は 5,000 WCU。マネージドルールの WCU を確認し、不要なルールグループを削除するか、scope-down statement でルール適用範囲を限定する。それでも不足する場合は AWS サポートに上限緩和をリクエストできる。

### Q2. WAF のルール変更はどのくらいで反映される？

**A.** REGIONAL スコープ（ALB/API Gateway）は通常数秒〜1分。CLOUDFRONT スコープはエッジロケーションへの伝播に数分かかる場合がある。本番環境では Count モードで事前テストすることを推奨。

### Q3. Shield Advanced は本当に $3,000/月の価値がある？

**A.** 以下の条件に当てはまる場合は検討の価値あり:
- DDoS 攻撃による売上損失が月 $3,000 を超える可能性がある
- 24/7 の DDoS Response Team サポートが必要
- DDoS によるスケールアウト費用の保護が必要
- WAF を大規模に使用しており WAF 料金の節約になる

小〜中規模サービスでは Shield Standard + WAF + CloudFront の組み合わせで十分な場合が多い。

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **AWS WAF** | Web ACL でルールを評価、マネージドルールで即座に一般的攻撃を防御 |
| **Shield Standard** | 全アカウント無料、L3/L4 DDoS 自動防御 |
| **Shield Advanced** | $3,000/月、L7 DDoS 対応、DRT サポート、コスト保護 |
| **多層防御** | CloudFront (Edge) + ALB (Regional) の 2 段 WAF が理想 |
| **運用** | Count → Block の段階的移行、CloudWatch でブロック率監視 |

---

## 次に読むべきガイド

- [01-secrets-management.md](./01-secrets-management.md) — シークレット管理で機密情報を保護
- IAM ポリシー設計ガイド — 最小権限の原則を実装
- VPC セキュリティガイド — ネットワーク層の防御

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS WAF Developer Guide" — https://docs.aws.amazon.com/waf/latest/developerguide/
2. **AWS公式ドキュメント** — "AWS Shield Developer Guide" — https://docs.aws.amazon.com/waf/latest/developerguide/shield-chapter.html
3. **AWS公式ブログ** — "AWS WAF を使った一般的な Web 攻撃の防御" — https://aws.amazon.com/blogs/security/
4. **AWS Well-Architected Framework** — Security Pillar — "Infrastructure protection" — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
