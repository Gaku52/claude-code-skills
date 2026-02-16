# WAF / Shield — WAF ルールと DDoS 対策

> AWS WAF でアプリケーション層（L7）の攻撃を防御し、AWS Shield で DDoS 攻撃（L3/L4）から保護するための実践ガイド。

---

## この章で学ぶこと

1. **AWS WAF** のルール設計と Web ACL によるリクエストフィルタリング
2. **AWS Shield Standard / Advanced** による DDoS 防御アーキテクチャ
3. **WAF + CloudFront + ALB** を組み合わせた多層防御パターン
4. **WAF ログの分析と運用** — Athena、CloudWatch、自動対応
5. **CDK/Terraform による WAF の IaC 管理** — 再現可能なセキュリティ構成

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

### 1.3 WAF が対応するリソース

| リソースタイプ | スコープ | 用途 |
|--------------|---------|------|
| **Amazon CloudFront** | CLOUDFRONT | エッジでのグローバル防御 |
| **Application Load Balancer** | REGIONAL | リージョナルなL7防御 |
| **Amazon API Gateway REST API** | REGIONAL | API エンドポイントの保護 |
| **AWS AppSync GraphQL API** | REGIONAL | GraphQL API の保護 |
| **Amazon Cognito User Pool** | REGIONAL | 認証エンドポイントの保護 |
| **AWS App Runner** | REGIONAL | コンテナサービスの保護 |
| **AWS Verified Access** | REGIONAL | ゼロトラストアクセスの保護 |

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

        # AWS マネージドルール: 既知の脆弱性入力
        - Name: AWSManagedRulesKnownBadInputsRuleSet
          Priority: 2
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesKnownBadInputsRuleSet
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: KnownBadInputs
            SampledRequestsEnabled: true

        # AWS マネージドルール: Linux OS 固有攻撃
        - Name: AWSManagedRulesLinuxRuleSet
          Priority: 3
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesLinuxRuleSet
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: LinuxRuleSet
            SampledRequestsEnabled: true

        # レートベースルール
        - Name: RateLimitRule
          Priority: 4
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

        # IPブロックリスト
        - Name: IPBlockList
          Priority: 5
          Action:
            Block:
              CustomResponse:
                ResponseCode: 403
                CustomResponseBodyKey: BlockedResponse
          Statement:
            IPSetReferenceStatement:
              Arn: !GetAtt BlockedIPSet.Arn
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: IPBlockList
            SampledRequestsEnabled: true

      CustomResponseBodies:
        BlockedResponse:
          ContentType: APPLICATION_JSON
          Content: '{"error":"Access denied","code":"BLOCKED"}'

  BlockedIPSet:
    Type: AWS::WAFv2::IPSet
    Properties:
      Name: blocked-ips
      Scope: REGIONAL
      IPAddressVersion: IPV4
      Addresses: []    # CLI/API で動的に追加

  # ALB との関連付け
  WebACLAssociation:
    Type: AWS::WAFv2::WebACLAssociation
    Properties:
      ResourceArn: !Ref ApplicationLoadBalancerArn
      WebACLArn: !GetAtt WebACL.Arn
```

### 2.2 カスタムルール: 特定パスの保護

```yaml
        # /admin パスへのアクセスを IP 制限
        - Name: AdminPathIPRestriction
          Priority: 6
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

        # CAPTCHA チャレンジ（Bot対策）
        - Name: CAPTCHAForSensitivePaths
          Priority: 7
          Action:
            Captcha:
              CustomRequestHandling:
                InsertHeaders:
                  - Name: x-waf-captcha-verified
                    Value: "true"
          Statement:
            OrStatement:
              Statements:
                - ByteMatchStatement:
                    SearchString: /api/signup
                    FieldToMatch:
                      UriPath: {}
                    TextTransformations:
                      - Priority: 0
                        Type: LOWERCASE
                    PositionalConstraint: EXACTLY
                - ByteMatchStatement:
                    SearchString: /api/contact
                    FieldToMatch:
                      UriPath: {}
                    TextTransformations:
                      - Priority: 0
                        Type: LOWERCASE
                    PositionalConstraint: EXACTLY
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: CAPTCHAChallenge
            SampledRequestsEnabled: true

        # 特定ヘッダーの検査
        - Name: RequireAPIKey
          Priority: 8
          Action:
            Block: {}
          Statement:
            AndStatement:
              Statements:
                - ByteMatchStatement:
                    SearchString: /api/
                    FieldToMatch:
                      UriPath: {}
                    TextTransformations:
                      - Priority: 0
                        Type: LOWERCASE
                    PositionalConstraint: STARTS_WITH
                - NotStatement:
                    Statement:
                      SizeConstraintStatement:
                        ComparisonOperator: GT
                        Size: 0
                        FieldToMatch:
                          SingleHeader:
                            Name: x-api-key
                        TextTransformations:
                          - Priority: 0
                            Type: NONE
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: RequireAPIKey
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

  # Geo制限: 特定国からのアクセスをブロック
  rule {
    name     = "GeoRestriction"
    priority = 2

    action {
      block {}
    }

    statement {
      geo_match_statement {
        country_codes = ["CN", "RU", "KP"]  # 必要に応じて調整
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "GeoRestriction"
      sampled_requests_enabled   = true
    }
  }

  # リクエストボディサイズ制限
  rule {
    name     = "RequestBodySizeLimit"
    priority = 3

    action {
      block {}
    }

    statement {
      size_constraint_statement {
        comparison_operator = "GT"
        size               = 10485760  # 10MB

        field_to_match {
          body {
            oversize_handling = "MATCH"
          }
        }

        text_transformation {
          priority = 0
          type     = "NONE"
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RequestBodySizeLimit"
      sampled_requests_enabled   = true
    }
  }

  # Account Takeover Prevention (ATP)
  rule {
    name     = "AccountTakeoverPrevention"
    priority = 4

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        vendor_name = "AWS"
        name        = "AWSManagedRulesATPRuleSet"

        managed_rule_group_configs {
          aws_managed_rules_atp_rule_set {
            login_path = "/api/login"

            request_inspection {
              payload_type = "JSON"

              username_field {
                identifier = "/username"
              }

              password_field {
                identifier = "/password"
              }
            }

            response_inspection {
              status_code {
                success_codes = [200, 201]
                failure_codes = [401, 403]
              }
            }
          }
        }
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "ATP"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "production-web-acl"
    sampled_requests_enabled   = true
  }

  tags = {
    Environment = "Production"
    ManagedBy   = "Terraform"
  }
}

# WAF と ALB の関連付け
resource "aws_wafv2_web_acl_association" "main" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
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
      {"SingleHeader": {"Name": "authorization"}},
      {"SingleHeader": {"Name": "cookie"}}
    ],
    "LoggingFilter": {
      "DefaultBehavior": "KEEP",
      "Filters": [
        {
          "Behavior": "KEEP",
          "Conditions": [
            {
              "ActionCondition": {
                "Action": "BLOCK"
              }
            }
          ],
          "Requirement": "MEETS_ANY"
        }
      ]
    }
  }'

# WAF ログを CloudWatch Logs に送信
aws wafv2 put-logging-configuration \
  --logging-configuration '{
    "ResourceArn": "arn:aws:wafv2:ap-northeast-1:123456789012:regional/webacl/production-web-acl/xxxx",
    "LogDestinationConfigs": [
      "arn:aws:logs:ap-northeast-1:123456789012:log-group:aws-waf-logs-production"
    ]
  }'

# Kinesis Data Firehose 経由で S3 + OpenSearch に送信
aws wafv2 put-logging-configuration \
  --logging-configuration '{
    "ResourceArn": "arn:aws:wafv2:ap-northeast-1:123456789012:regional/webacl/production-web-acl/xxxx",
    "LogDestinationConfigs": [
      "arn:aws:firehose:ap-northeast-1:123456789012:deliverystream/aws-waf-logs-stream"
    ]
  }'
```

### 2.5 Athena による WAF ログ分析

```sql
-- Athena テーブルの作成
CREATE EXTERNAL TABLE waf_logs (
  timestamp bigint,
  formatVersion int,
  webaclId string,
  terminatingRuleId string,
  terminatingRuleType string,
  action string,
  terminatingRuleMatchDetails array<struct<
    conditionType:string,
    sensitivityLevel:string,
    location:string,
    matchedData:array<string>
  >>,
  httpSourceName string,
  httpSourceId string,
  ruleGroupList array<struct<
    ruleGroupId:string,
    terminatingRule:struct<ruleId:string,action:string,ruleMatchDetails:string>,
    nonTerminatingMatchingRules:array<struct<ruleId:string,action:string>>,
    excludedRules:array<struct<ruleId:string,exclusionType:string>>
  >>,
  rateBasedRuleList array<struct<
    rateBasedRuleId:string,
    limitKey:string,
    maxRateAllowed:int
  >>,
  nonTerminatingMatchingRules array<struct<
    ruleId:string,
    action:string,
    ruleMatchDetails:array<struct<
      conditionType:string,
      sensitivityLevel:string,
      location:string,
      matchedData:array<string>
    >>
  >>,
  requestHeadersInserted string,
  responseCodeSent int,
  httpRequest struct<
    clientIp:string,
    country:string,
    headers:array<struct<name:string,value:string>>,
    uri:string,
    args:string,
    httpVersion:string,
    httpMethod:string,
    requestId:string
  >,
  labels array<struct<name:string>>
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://my-waf-logs-bucket/AWSLogs/123456789012/WAFLogs/ap-northeast-1/production-web-acl/';

-- ブロックされたリクエストの分析
SELECT
  httprequest.clientip,
  httprequest.uri,
  httprequest.country,
  terminatingruleid,
  COUNT(*) as block_count
FROM waf_logs
WHERE action = 'BLOCK'
  AND from_unixtime(timestamp/1000) > current_timestamp - interval '24' hour
GROUP BY 1, 2, 3, 4
ORDER BY block_count DESC
LIMIT 20;

-- 国別のリクエスト分布
SELECT
  httprequest.country,
  action,
  COUNT(*) as request_count
FROM waf_logs
WHERE from_unixtime(timestamp/1000) > current_timestamp - interval '7' day
GROUP BY 1, 2
ORDER BY request_count DESC;

-- SQLi/XSS 検出の詳細
SELECT
  httprequest.clientip,
  httprequest.uri,
  httprequest.httpmethod,
  terminatingruleid,
  terminatingrulematchdetails
FROM waf_logs
WHERE terminatingruleid IN ('AWSManagedRulesSQLiRuleSet', 'CrossSiteScripting_BODY')
  AND from_unixtime(timestamp/1000) > current_timestamp - interval '24' hour
LIMIT 50;

-- レートベースルールでブロックされたIP
SELECT
  httprequest.clientip,
  httprequest.country,
  COUNT(*) as blocked_count,
  MIN(from_unixtime(timestamp/1000)) as first_blocked,
  MAX(from_unixtime(timestamp/1000)) as last_blocked
FROM waf_logs
WHERE terminatingruleid = 'RateLimitRule'
  AND from_unixtime(timestamp/1000) > current_timestamp - interval '24' hour
GROUP BY 1, 2
ORDER BY blocked_count DESC
LIMIT 20;
```

### 2.6 CloudWatch メトリクスとアラーム

```yaml
# CloudFormation: WAF の CloudWatch アラーム
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  # ブロック率の急増アラーム
  HighBlockRateAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: waf-high-block-rate
      AlarmDescription: "WAF block rate exceeds threshold"
      Namespace: AWS/WAFV2
      MetricName: BlockedRequests
      Dimensions:
        - Name: WebACL
          Value: production-web-acl
        - Name: Region
          Value: ap-northeast-1
        - Name: Rule
          Value: ALL
      Statistic: Sum
      Period: 300            # 5分間
      EvaluationPeriods: 2   # 2期間連続
      Threshold: 1000        # 5分間で1000ブロック
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref SecurityAlertTopic

  # レートベースルールのトリガーアラーム
  RateLimitAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: waf-rate-limit-triggered
      AlarmDescription: "Rate limit rule triggered frequently"
      Namespace: AWS/WAFV2
      MetricName: BlockedRequests
      Dimensions:
        - Name: WebACL
          Value: production-web-acl
        - Name: Region
          Value: ap-northeast-1
        - Name: Rule
          Value: RateLimitRule
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 1
      Threshold: 100
      ComparisonOperator: GreaterThanThreshold
      AlarmActions:
        - !Ref SecurityAlertTopic

  # 全リクエスト数の異常検知
  RequestAnomalyAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: waf-request-anomaly
      AlarmDescription: "Unusual request volume detected"
      Namespace: AWS/WAFV2
      MetricName: AllowedRequests
      Dimensions:
        - Name: WebACL
          Value: production-web-acl
        - Name: Region
          Value: ap-northeast-1
        - Name: Rule
          Value: ALL
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 3
      Threshold: 50000
      ComparisonOperator: GreaterThanThreshold
      TreatMissingData: notBreaching
      AlarmActions:
        - !Ref SecurityAlertTopic

  SecurityAlertTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: waf-security-alerts
      Subscription:
        - Protocol: email
          Endpoint: security@example.com
```

### 2.7 IP セットの動的管理

```python
import boto3
import json
from datetime import datetime

waf = boto3.client("wafv2", region_name="ap-northeast-1")

def add_ip_to_blocklist(ip_address: str, ip_set_name: str, ip_set_id: str):
    """IP アドレスをブロックリストに追加"""
    # 現在の IP セットを取得
    response = waf.get_ip_set(
        Name=ip_set_name,
        Scope="REGIONAL",
        Id=ip_set_id,
    )

    addresses = response["IPSet"]["Addresses"]
    lock_token = response["LockToken"]

    # CIDR 表記に変換
    if "/" not in ip_address:
        ip_address = f"{ip_address}/32"

    if ip_address not in addresses:
        addresses.append(ip_address)

        waf.update_ip_set(
            Name=ip_set_name,
            Scope="REGIONAL",
            Id=ip_set_id,
            Addresses=addresses,
            LockToken=lock_token,
        )
        print(f"Added {ip_address} to blocklist at {datetime.now()}")
    else:
        print(f"{ip_address} already in blocklist")

def remove_ip_from_blocklist(ip_address: str, ip_set_name: str, ip_set_id: str):
    """IP アドレスをブロックリストから削除"""
    response = waf.get_ip_set(
        Name=ip_set_name,
        Scope="REGIONAL",
        Id=ip_set_id,
    )

    addresses = response["IPSet"]["Addresses"]
    lock_token = response["LockToken"]

    if "/" not in ip_address:
        ip_address = f"{ip_address}/32"

    if ip_address in addresses:
        addresses.remove(ip_address)

        waf.update_ip_set(
            Name=ip_set_name,
            Scope="REGIONAL",
            Id=ip_set_id,
            Addresses=addresses,
            LockToken=lock_token,
        )
        print(f"Removed {ip_address} from blocklist")

# 自動ブロック: WAF ログから攻撃元 IP を自動追加
def auto_block_from_logs(threshold: int = 100):
    """CloudWatch Logs Insights で攻撃元 IP を検出してブロック"""
    logs = boto3.client("logs", region_name="ap-northeast-1")

    query = f"""
    fields httprequest.clientip as ip, count(*) as cnt
    | filter action = "BLOCK"
    | stats count(*) as cnt by httprequest.clientip
    | filter cnt > {threshold}
    | sort cnt desc
    | limit 50
    """

    response = logs.start_query(
        logGroupName="aws-waf-logs-production",
        startTime=int((datetime.now().timestamp() - 3600)),  # 過去1時間
        endTime=int(datetime.now().timestamp()),
        queryString=query,
    )

    # クエリ結果を取得して IP をブロック
    import time
    time.sleep(10)

    results = logs.get_query_results(queryId=response["queryId"])
    for result in results.get("results", []):
        ip = None
        cnt = 0
        for field in result:
            if field["field"] == "ip":
                ip = field["value"]
            elif field["field"] == "cnt":
                cnt = int(field["value"])

        if ip and cnt > threshold:
            add_ip_to_blocklist(
                ip, "blocked-ips", "ip-set-id-xxxx"
            )
```

### 2.8 AWS WAF JavaScript SDK（Bot検知）

```html
<!-- WAF CAPTCHA/Challenge の統合 -->
<script type="text/javascript"
  src="https://xxxxx.token.awswaf.com/xxxxx/challenge.js"
  defer>
</script>

<script>
// WAF Token の取得と送信
async function submitForm() {
  try {
    // WAF Challenge トークンを取得
    const token = await AwsWafIntegration.getToken();

    const response = await fetch('/api/login', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-aws-waf-token': token,
      },
      body: JSON.stringify({
        username: document.getElementById('username').value,
        password: document.getElementById('password').value,
      }),
    });

    const data = await response.json();
    console.log('Login result:', data);
  } catch (error) {
    console.error('WAF challenge failed:', error);
  }
}
</script>
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
│  │ - SRT による proactive engagement              │      │
│  └────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Shield Advanced の設定

```bash
# Shield Advanced の有効化（年間コミット）
aws shield create-subscription

# 保護対象リソースの追加
aws shield create-protection \
  --name "Production-ALB" \
  --resource-arn "arn:aws:elasticloadbalancing:ap-northeast-1:123456789012:loadbalancer/app/prod-alb/xxxx"

aws shield create-protection \
  --name "Production-CloudFront" \
  --resource-arn "arn:aws:cloudfront::123456789012:distribution/EXXXXXXXXXX"

aws shield create-protection \
  --name "Production-EIP" \
  --resource-arn "arn:aws:ec2:ap-northeast-1:123456789012:eip-allocation/eipalloc-xxxx"

# 保護一覧の確認
aws shield list-protections

# DRT へのアクセス権限付与
aws shield associate-drt-role \
  --role-arn "arn:aws:iam::123456789012:role/ShieldDRTAccessRole"

aws shield associate-drt-log-bucket \
  --log-bucket "my-waf-logs-bucket"

# Proactive Engagement の有効化
aws shield enable-proactive-engagement

# Emergency Contact の設定
aws shield associate-health-check \
  --protection-id "protection-id-xxxx" \
  --health-check-arn "arn:aws:route53:::healthcheck/xxxx"

aws shield update-emergency-contact-settings \
  --emergency-contact-list '[
    {"EmailAddress": "security@example.com", "PhoneNumber": "+81-90-xxxx-xxxx", "ContactNotes": "Security Team Lead"},
    {"EmailAddress": "oncall@example.com", "PhoneNumber": "+81-80-xxxx-xxxx", "ContactNotes": "On-call Engineer"}
  ]'
```

### 3.3 Shield Advanced のイベント監視

```python
import boto3
from datetime import datetime, timedelta

def get_ddos_attacks(hours: int = 24) -> list[dict]:
    """過去N時間のDDoS攻撃イベントを取得"""
    shield = boto3.client("shield", region_name="us-east-1")  # Shield はus-east-1

    start_time = datetime.utcnow() - timedelta(hours=hours)
    end_time = datetime.utcnow()

    response = shield.list_attacks(
        StartTime={"FromInclusive": start_time, "ToExclusive": end_time},
        MaxResults=100,
    )

    attacks = []
    for attack in response.get("AttackSummaries", []):
        detail = shield.describe_attack(AttackId=attack["AttackId"])
        attack_detail = detail["Attack"]

        attacks.append({
            "AttackId": attack_detail["AttackId"],
            "ResourceArn": attack_detail["ResourceArn"],
            "StartTime": str(attack_detail.get("StartTime")),
            "EndTime": str(attack_detail.get("EndTime")),
            "Vectors": [
                {
                    "VectorType": v["VectorType"],
                    "Counters": [
                        {"Name": c["Name"], "Max": c["Max"], "Average": c["Average"], "Sum": c["Sum"]}
                        for c in v.get("VectorCounters", [])
                    ]
                }
                for v in attack_detail.get("AttackProperties", [])
            ],
            "Mitigations": attack_detail.get("Mitigations", []),
        })

    return attacks

# 実行
attacks = get_ddos_attacks(24)
for attack in attacks:
    print(f"Attack: {attack['AttackId']}")
    print(f"  Resource: {attack['ResourceArn']}")
    print(f"  Duration: {attack['StartTime']} - {attack['EndTime']}")
    for vector in attack["Vectors"]:
        print(f"  Vector: {vector['VectorType']}")
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
│ (DNS)            │     DNSSEC 有効化
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ CloudFront       │  ← Shield Standard/Advanced + WAF
│ (CDN + Edge WAF) │     Geo Restriction
│                  │     Origin Access Control
│                  │     TLS 1.2+ 強制
└────────┬─────────┘
         │ Origin Access (署名付きリクエスト)
         ▼
┌──────────────────┐
│ ALB              │  ← WAF (Regional) + Security Group
│ (Load Balancer)  │     リクエスト検証
│                  │     SG: CloudFront Prefix List のみ許可
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ ECS / EC2        │  ← Security Group (ALB からのみ)
│ (Application)    │     アプリケーション側バリデーション
│                  │     OWASP Top 10 対策
└──────────────────┘
```

### 4.2 CloudFront + ALB の二段 WAF 構成

```yaml
# CloudFront WAF（エッジ防御）
AWSTemplateFormatVersion: "2010-09-09"
Resources:
  # エッジ WAF（us-east-1 で作成する必要あり）
  EdgeWebACL:
    Type: AWS::WAFv2::WebACL
    Properties:
      Name: edge-web-acl
      Scope: CLOUDFRONT
      DefaultAction:
        Allow: {}
      Rules:
        # Geo制限（エッジで早期ブロック）
        - Name: GeoBlock
          Priority: 0
          Action:
            Block: {}
          Statement:
            NotStatement:
              Statement:
                GeoMatchStatement:
                  CountryCodes: ["JP", "US", "SG"]  # 許可国のみ
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: GeoBlock
            SampledRequestsEnabled: true

        # IP レピュテーション
        - Name: AmazonIPReputation
          Priority: 1
          OverrideAction:
            None: {}
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesAmazonIpReputationList
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: IPReputation
            SampledRequestsEnabled: true

        # Anonymous IP（VPN/Tor）
        - Name: AnonymousIP
          Priority: 2
          OverrideAction:
            Count: {}    # まず Count で様子見
          Statement:
            ManagedRuleGroupStatement:
              VendorName: AWS
              Name: AWSManagedRulesAnonymousIpList
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: AnonymousIP
            SampledRequestsEnabled: true

        # グローバルレート制限
        - Name: GlobalRateLimit
          Priority: 3
          Action:
            Block: {}
          Statement:
            RateBasedStatement:
              Limit: 5000    # エッジでの全体制限
              AggregateKeyType: IP
          VisibilityConfig:
            CloudWatchMetricsEnabled: true
            MetricName: GlobalRateLimit
            SampledRequestsEnabled: true
      VisibilityConfig:
        CloudWatchMetricsEnabled: true
        MetricName: edge-web-acl
        SampledRequestsEnabled: true
```

### 4.3 ALB セキュリティグループの設定

```bash
# CloudFront のマネージドプレフィックスリストを使用
aws ec2 describe-managed-prefix-lists \
  --filters Name=prefix-list-name,Values=com.amazonaws.global.cloudfront.origin-facing

# ALB のセキュリティグループ: CloudFront からのみ許可
aws ec2 authorize-security-group-ingress \
  --group-id sg-alb-xxxx \
  --ip-permissions '[
    {
      "IpProtocol": "tcp",
      "FromPort": 443,
      "ToPort": 443,
      "PrefixListIds": [
        {"PrefixListId": "pl-3b927c52"}
      ]
    }
  ]'

# CloudFront のカスタムヘッダーで検証
# CloudFront → ALB にカスタムヘッダーを追加
# ALB のリスナールールでヘッダーを検証
```

---

## 5. CDK による WAF の完全構成

### 5.1 CDK での WAF スタック

```typescript
import * as cdk from 'aws-cdk-lib';
import * as wafv2 from 'aws-cdk-lib/aws-wafv2';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct } from 'constructs';

export class WafStack extends cdk.Stack {
  public readonly webAcl: wafv2.CfnWebACL;

  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // IP ブロックリスト
    const blockedIpSet = new wafv2.CfnIPSet(this, 'BlockedIPs', {
      name: 'blocked-ips',
      scope: 'REGIONAL',
      ipAddressVersion: 'IPV4',
      addresses: [],
    });

    // 管理者許可 IP セット
    const adminIpSet = new wafv2.CfnIPSet(this, 'AdminIPs', {
      name: 'admin-allowed-ips',
      scope: 'REGIONAL',
      ipAddressVersion: 'IPV4',
      addresses: ['203.0.113.0/24', '198.51.100.10/32'],
    });

    // Web ACL
    this.webAcl = new wafv2.CfnWebACL(this, 'WebACL', {
      name: 'production-web-acl',
      scope: 'REGIONAL',
      defaultAction: { allow: {} },
      visibilityConfig: {
        cloudWatchMetricsEnabled: true,
        metricName: 'production-web-acl',
        sampledRequestsEnabled: true,
      },
      rules: [
        // IP ブロックリスト
        {
          name: 'IPBlockList',
          priority: 0,
          action: { block: {} },
          statement: {
            ipSetReferenceStatement: { arn: blockedIpSet.attrArn },
          },
          visibilityConfig: {
            cloudWatchMetricsEnabled: true,
            metricName: 'IPBlockList',
            sampledRequestsEnabled: true,
          },
        },
        // AWS マネージドルール: Common
        {
          name: 'AWSCommonRules',
          priority: 1,
          overrideAction: { none: {} },
          statement: {
            managedRuleGroupStatement: {
              vendorName: 'AWS',
              name: 'AWSManagedRulesCommonRuleSet',
            },
          },
          visibilityConfig: {
            cloudWatchMetricsEnabled: true,
            metricName: 'CommonRules',
            sampledRequestsEnabled: true,
          },
        },
        // AWS マネージドルール: SQLi
        {
          name: 'AWSSQLiRules',
          priority: 2,
          overrideAction: { none: {} },
          statement: {
            managedRuleGroupStatement: {
              vendorName: 'AWS',
              name: 'AWSManagedRulesSQLiRuleSet',
            },
          },
          visibilityConfig: {
            cloudWatchMetricsEnabled: true,
            metricName: 'SQLiRules',
            sampledRequestsEnabled: true,
          },
        },
        // レート制限
        {
          name: 'RateLimit',
          priority: 3,
          action: { block: {} },
          statement: {
            rateBasedStatement: {
              limit: 2000,
              aggregateKeyType: 'IP',
            },
          },
          visibilityConfig: {
            cloudWatchMetricsEnabled: true,
            metricName: 'RateLimit',
            sampledRequestsEnabled: true,
          },
        },
      ],
    });

    // CloudWatch アラーム
    const alertTopic = new sns.Topic(this, 'WafAlerts', {
      topicName: 'waf-security-alerts',
    });

    new cloudwatch.Alarm(this, 'HighBlockRate', {
      alarmName: 'waf-high-block-rate',
      metric: new cloudwatch.Metric({
        namespace: 'AWS/WAFV2',
        metricName: 'BlockedRequests',
        dimensionsMap: {
          WebACL: 'production-web-acl',
          Region: this.region,
          Rule: 'ALL',
        },
        statistic: 'Sum',
        period: cdk.Duration.minutes(5),
      }),
      threshold: 1000,
      evaluationPeriods: 2,
    });
  }
}
```

---

## 6. 比較表

### 6.1 WAF マネージドルールグループ比較

| ルールグループ | 防御対象 | WCU | 推奨 |
|--------------|---------|-----|------|
| **AWSManagedRulesCommonRuleSet** | XSS, パストラバーサル, ファイルインジェクション | 700 | 全アプリ必須 |
| **AWSManagedRulesSQLiRuleSet** | SQL インジェクション | 200 | DB 利用アプリ必須 |
| **AWSManagedRulesKnownBadInputsRuleSet** | Log4j, 既知の脆弱性 | 200 | Java アプリは必須 |
| **AWSManagedRulesBotControlRuleSet** | Bot 検知・制御 | 50 | EC サイト推奨 |
| **AWSManagedRulesATPRuleSet** | アカウント乗っ取り防止 | 50 | 認証ありアプリ |
| **AWSManagedRulesAnonymousIPList** | VPN/Tor/プロキシ | 50 | セキュリティ重視 |
| **AWSManagedRulesAmazonIpReputationList** | 悪意あるIP | 25 | 全アプリ推奨 |
| **AWSManagedRulesLinuxRuleSet** | Linux固有攻撃 | 200 | Linux環境 |
| **AWSManagedRulesWindowsRuleSet** | Windows固有攻撃 | 200 | Windows環境 |
| **AWSManagedRulesPHPRuleSet** | PHP固有攻撃 | 100 | PHP アプリ |
| **AWSManagedRulesWordPressRuleSet** | WordPress固有攻撃 | 100 | WordPress |

### 6.2 DDoS 対策レイヤー比較

| レイヤー | 攻撃例 | 防御サービス | 自動/手動 |
|---------|--------|-------------|----------|
| **L3 (ネットワーク)** | UDP Flood, ICMP Flood | Shield Standard | 自動 |
| **L4 (トランスポート)** | SYN Flood, TCP RST | Shield Standard | 自動 |
| **L7 (アプリケーション)** | HTTP Flood, Slowloris | WAF + Shield Advanced | ルール設定必要 |
| **DNS** | DNS Query Flood | Route 53 + Shield | 自動 |
| **API** | API 呼び出し集中 | API Gateway Throttling + WAF | 設定必要 |

### 6.3 Shield Standard vs Advanced 詳細比較

| 機能 | Shield Standard | Shield Advanced |
|------|----------------|----------------|
| **料金** | 無料 | $3,000/月 + データ転送 |
| **L3/L4 防御** | 自動 | 自動 + 高度な検知 |
| **L7 防御** | なし | WAF 統合で自動緩和 |
| **DRT サポート** | なし | 24/7 対応 |
| **コスト保護** | なし | DDoS起因のスケールアウト費用を補填 |
| **攻撃可視化** | 基本的 | リアルタイムダッシュボード |
| **WAF 料金** | 別途 | Shield Advanced に含む |
| **ヘルスチェック連携** | なし | Route 53 ヘルスチェック |
| **Proactive Engagement** | なし | DRT による事前対応 |
| **対象リソース** | CloudFront, Route53 | ALB, EIP, CloudFront, Route53, GA |

---

## 7. アンチパターン

### 7.1 WAF ルールを Count モードのまま放置

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

### 7.2 WAF を ALB にだけ設定し CloudFront を経由しない

```
NG:
  Client → ALB (WAF) → App
  問題: DDoS が ALB に直撃、オリジン IP が露出

OK:
  Client → CloudFront (WAF + Shield) → ALB (SG: CloudFront のみ許可) → App
  利点: エッジで攻撃軽減、ALB への直接アクセスを遮断
```

**対策**: ALB のセキュリティグループで CloudFront の IP レンジのみを許可し、直接アクセスをブロックする。AWS が公開する CloudFront IP リストを AWS Managed Prefix List で参照可能。

### 7.3 全てのマネージドルールを無条件に適用

```
NG:
  全マネージドルールを Block モードで一括適用
  → 正常なリクエストまでブロックされる（誤検知）
  → WCU 上限に到達

OK:
  段階的導入:
  1. まず Common + SQLi を Count モードで適用
  2. 1-2週間ログを監視して誤検知を確認
  3. ExcludedRules で誤検知ルールを除外
  4. Block モードに切り替え
  5. 次のルールグループを追加
```

### 7.4 WAF ログを保存・分析していない

```
NG:
  WAF ログを有効化していない
  → 攻撃の傾向が把握できない
  → 誤検知の原因調査ができない

OK:
  WAF ログを S3 + Athena で分析
  → ブロックされたリクエストの傾向を定期的にレビュー
  → ダッシュボードで攻撃状況を可視化
  → 自動アラートで異常検知
```

### 7.5 レート制限が緩すぎる/厳しすぎる

```
NG:
  レート制限を全エンドポイントに一律 10,000 req/5min で設定
  → ログインページへのブルートフォースを防げない
  → 正常な API 利用がブロックされる

OK:
  エンドポイント別のレート制限:
  - /api/login    → 100 req/5min (厳しく)
  - /api/signup   → 50 req/5min (厳しく)
  - /api/data     → 5000 req/5min (通常)
  - Global        → 10000 req/5min (全体の安全弁)
```

---

## 8. FAQ

### Q1. WAF の WCU（Web ACL Capacity Unit）制限にどう対処する？

**A.** Web ACL のデフォルト上限は 5,000 WCU。マネージドルールの WCU を確認し、不要なルールグループを削除するか、scope-down statement でルール適用範囲を限定する。それでも不足する場合は AWS サポートに上限緩和をリクエストできる。

### Q2. WAF のルール変更はどのくらいで反映される？

**A.** REGIONAL スコープ（ALB/API Gateway）は通常数秒から1分。CLOUDFRONT スコープはエッジロケーションへの伝播に数分かかる場合がある。本番環境では Count モードで事前テストすることを推奨。

### Q3. Shield Advanced は本当に $3,000/月の価値がある？

**A.** 以下の条件に当てはまる場合は検討の価値あり:
- DDoS 攻撃による売上損失が月 $3,000 を超える可能性がある
- 24/7 の DDoS Response Team サポートが必要
- DDoS によるスケールアウト費用の保護が必要
- WAF を大規模に使用しており WAF 料金の節約になる

小から中規模サービスでは Shield Standard + WAF + CloudFront の組み合わせで十分な場合が多い。

### Q4. WAF の誤検知が発生した場合の対処方法は？

**A.** (1) WAF ログから該当リクエストの terminatingRuleId を特定、(2) 該当ルールを ExcludedRules に追加するか Count モードに変更、(3) 必要に応じて scope-down statement で適用範囲を限定（特定パスのみ等）、(4) カスタムルールで代替の保護を実装。誤検知の頻度が高い場合はルールのラベル機能を使って詳細な制御を行う。

### Q5. API Gateway と ALB の WAF を使い分ける基準は？

**A.** API Gateway REST API には直接 WAF を適用可能。ALB の場合も同様。CloudFront を前段に置く場合は CloudFront の WAF が最前線になる。推奨構成は CloudFront (WAF: Geo制限/IP制限/レート制限) + ALB (WAF: SQLi/XSS/アプリケーション固有ルール) の二段構成。API Gateway を使う場合は API Gateway の WAF + API キー + Usage Plan による追加の保護を組み合わせる。

---

## 9. まとめ

| 項目 | ポイント |
|------|---------|
| **AWS WAF** | Web ACL でルールを評価、マネージドルールで即座に一般的攻撃を防御 |
| **Shield Standard** | 全アカウント無料、L3/L4 DDoS 自動防御 |
| **Shield Advanced** | $3,000/月、L7 DDoS 対応、DRT サポート、コスト保護 |
| **多層防御** | CloudFront (Edge) + ALB (Regional) の 2 段 WAF が理想 |
| **ルール設計** | マネージドルール + カスタムルールの組み合わせ、エンドポイント別レート制限 |
| **運用** | Count → Block の段階的移行、CloudWatch でブロック率監視 |
| **ログ分析** | S3 + Athena でブロックパターンを分析、EventBridge で自動アラート |
| **IaC** | CDK/Terraform で WAF 構成をコード管理、環境間の一貫性を確保 |

---

## 次に読むべきガイド

- [01-secrets-management.md](./01-secrets-management.md) — シークレット管理で機密情報を保護
- [00-iam-deep-dive.md](./00-iam-deep-dive.md) — IAM ポリシー設計ガイド
- VPC セキュリティガイド — ネットワーク層の防御

---

## 参考文献

1. **AWS公式ドキュメント** — "AWS WAF Developer Guide" — https://docs.aws.amazon.com/waf/latest/developerguide/
2. **AWS公式ドキュメント** — "AWS Shield Developer Guide" — https://docs.aws.amazon.com/waf/latest/developerguide/shield-chapter.html
3. **AWS公式ブログ** — "AWS WAF を使った一般的な Web 攻撃の防御" — https://aws.amazon.com/blogs/security/
4. **AWS Well-Architected Framework** — Security Pillar — "Infrastructure protection" — https://docs.aws.amazon.com/wellarchitected/latest/security-pillar/
5. **AWS WAF Security Automations** — "AWS WAF Security Automations Solution" — https://aws.amazon.com/solutions/implementations/aws-waf-security-automations/
6. **OWASP Top 10** — "OWASP Top 10 Web Application Security Risks" — https://owasp.org/www-project-top-ten/
