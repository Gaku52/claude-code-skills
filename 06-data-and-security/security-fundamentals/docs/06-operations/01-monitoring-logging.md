# 監視/ログ

> SIEM によるログの集約・相関分析、効果的なログ収集アーキテクチャ、異常検知の手法まで、セキュリティ監視の基盤を体系的に学ぶ

## 前提知識

このガイドを理解するために、以下の知識が必要である:

- ネットワークの基礎知識（TCP/IP、DNS、HTTP）
- Linux の基本操作（syslog、systemd の基礎）
- クラウドサービス（AWS）の基本概念
- JSON フォーマットの理解

関連ガイド: [セキュリティ原則](../00-basics/02-security-principles.md) | [クラウドセキュリティ基礎](../05-cloud-security/00-cloud-security-basics.md)

## この章で学ぶこと

1. **SIEM の仕組みと活用** — ログの集約、相関分析、アラート生成のメカニズム
2. **ログ集約アーキテクチャ** — 大規模環境でのログ収集・保存・検索の設計
3. **異常検知** — ルールベースと機械学習ベースの検知手法
4. **ダッシュボード設計** — 効果的なセキュリティ可視化の原則
5. **運用のベストプラクティス** — ログローテーション、保持ポリシー、パフォーマンスチューニング

---

## 1. セキュリティ監視の全体像

### なぜセキュリティ監視が必要か

セキュリティ監視は、攻撃の検知、インシデントの調査、コンプライアンスの証跡確保という3つの目的を達成するための基盤である。NIST SP 800-92 では、ログ管理を情報セキュリティの重要な構成要素として位置づけている。

監視を行わないシステムは「目を閉じて運転する車」に等しい。攻撃者の平均滞在期間（Dwell Time）は業界平均で約 200 日とされており、早期検知なしには被害が拡大し続ける。

### 監視アーキテクチャ

```
+----------------------------------------------------------+
|                    データソース                             |
|----------------------------------------------------------|
|  OS ログ    | アプリログ  | ネットワーク | クラウド           |
|  (syslog,   | (stdout,   | (VPC Flow,   | (CloudTrail,     |
|   auditd)   |  JSON)     |  pcap)       |  GuardDuty)      |
+----------------------------------------------------------+
          |            |            |            |
          v            v            v            v
+----------------------------------------------------------+
|              ログ収集・転送層                               |
|  Fluentd / Fluent Bit / Vector / CloudWatch Agent        |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|              ログ保存・インデックス層                        |
|  +-- Hot:  OpenSearch / Elasticsearch (直近 30 日)       |
|  +-- Warm: S3 + Athena (30-365 日)                       |
|  +-- Cold: S3 Glacier (365 日以上)                        |
+----------------------------------------------------------+
          |
          v
+----------------------------------------------------------+
|              分析・検知層 (SIEM)                            |
|  +-- ルールベース検知 (相関ルール)                          |
|  +-- 機械学習ベース異常検知                                |
|  +-- ダッシュボード / 可視化                               |
|  +-- アラート生成 → PagerDuty / Slack                     |
+----------------------------------------------------------+
```

### 監視の3つの層

セキュリティ監視を設計する際は、以下の3層を意識する必要がある:

```
+----------------------------------------------------------+
|                   監視の3層モデル                           |
|----------------------------------------------------------|
|                                                          |
|  Layer 1: コレクション (Collection)                        |
|  +-- 目的: 生データの収集と転送                            |
|  +-- ツール: Fluent Bit, Vector, CloudWatch Agent         |
|  +-- 課題: データ量の制御、ネットワーク帯域               |
|  +-- 設計原則: 「できるだけ早く、できるだけ完全に」        |
|                                                          |
|  Layer 2: ストレージ (Storage)                             |
|  +-- 目的: ログの保存、インデックス化、検索性確保          |
|  +-- ツール: OpenSearch, S3, Glacier                      |
|  +-- 課題: コスト最適化、保持期間の管理                    |
|  +-- 設計原則: 「階層化ストレージでコスト最適化」          |
|                                                          |
|  Layer 3: アナリティクス (Analytics)                       |
|  +-- 目的: 相関分析、異常検知、可視化                     |
|  +-- ツール: SIEM, ML エンジン, ダッシュボード             |
|  +-- 課題: 偽陽性の制御、アラート疲れの防止               |
|  +-- 設計原則: 「アクションにつながるアラートのみ」        |
+----------------------------------------------------------+
```

### ログの分類と優先度マトリクス

```
+----------------------------------------------------------+
|             ログ分類マトリクス                              |
|----------------------------------------------------------|
|                                                          |
|      高頻度     +-----------+-----------+                |
|                 | 認証ログ  | アプリ    |                |
|                 | (重要)    | アクセス  |                |
|                 |           | ログ      |                |
|                 +-----------+-----------+                |
|                 | ネット    | デバッグ  |                |
|                 | フロー    | ログ      |                |
|      低頻度     | ログ      | (低優先)  |                |
|                 +-----------+-----------+                |
|                  高セキュリティ  低セキュリティ              |
|                  価値            価値                     |
|                                                          |
|  優先度: 認証ログ > ネットフロー > アプリログ > デバッグ    |
+----------------------------------------------------------+
```

---

## 2. ログ収集の設計

### 収集すべきログ

| ログソース | 内容 | 保存期間 | 優先度 | フォーマット |
|-----------|------|---------|--------|------------|
| CloudTrail | AWS API 呼び出し | 1年以上 | 必須 | JSON |
| VPC Flow Logs | ネットワークフロー | 90日 | 必須 | テキスト/Parquet |
| ALB/NLB Access Logs | HTTP リクエスト | 90日 | 必須 | テキスト |
| GuardDuty Findings | 脅威検知結果 | 90日 | 必須 | JSON |
| OS syslog/auditd | OS レベルのイベント | 90日 | 高 | syslog |
| Application Logs | アプリケーション動作 | 30-90日 | 高 | JSON |
| DNS Query Logs | DNS クエリ | 30日 | 中 | JSON |
| WAF Logs | WAF 判定結果 | 30日 | 中 | JSON |
| S3 Access Logs | S3 操作ログ | 90日 | 中 | テキスト |
| RDS Audit Logs | DB 操作ログ | 90日 | 高 | テキスト |
| Lambda Logs | 関数実行ログ | 30日 | 中 | JSON |
| Config Changes | 設定変更履歴 | 1年以上 | 必須 | JSON |

### 構造化ログのフォーマット

構造化ログは、ログの検索性と分析可能性を飛躍的に向上させる。以下は推奨フォーマットである:

```json
{
  "timestamp": "2025-03-15T14:30:00.000Z",
  "level": "WARN",
  "service": "auth-service",
  "version": "2.1.0",
  "environment": "production",
  "traceId": "abc-123-def",
  "spanId": "span-456",
  "requestId": "req-456",
  "event": "login_failed",
  "userId": "user-789",
  "sourceIp": "203.0.113.50",
  "userAgent": "Mozilla/5.0...",
  "geoLocation": {
    "country": "JP",
    "region": "Tokyo"
  },
  "details": {
    "reason": "invalid_password",
    "attemptCount": 5,
    "accountLocked": false,
    "mfaEnabled": true
  },
  "metadata": {
    "hostname": "auth-pod-abc123",
    "containerId": "docker-xyz",
    "kubernetes": {
      "namespace": "production",
      "pod": "auth-service-7b9f4d-abc12"
    }
  }
}
```

### 構造化ログの実装（Python）

```python
# コード例1: 構造化ログライブラリの実装
import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextvars import ContextVar
import uuid

# リクエストコンテキストをスレッドセーフに保持
_request_context: ContextVar[Dict[str, Any]] = ContextVar(
    'request_context', default={}
)

class StructuredLogFormatter(logging.Formatter):
    """構造化ログフォーマッタ

    ログをJSON形式で出力し、トレースID、リクエストID等の
    コンテキスト情報を自動的に付与する。

    セキュリティログでは以下の情報が特に重要:
    - timestamp: UTC形式のISO 8601タイムスタンプ
    - sourceIp: リクエスト元のIPアドレス
    - userId: 操作を行ったユーザーのID
    - event: セキュリティイベントの種類
    """

    # ログに含めてはいけない機密フィールド
    SENSITIVE_FIELDS = {
        'password', 'secret', 'token', 'api_key',
        'credit_card', 'ssn', 'cvv', 'pin',
        'authorization', 'cookie', 'session_id',
    }

    def __init__(self, service_name: str, environment: str):
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service_name,
            "environment": self.environment,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # リクエストコンテキストの追加
        ctx = _request_context.get()
        if ctx:
            log_entry.update({
                "traceId": ctx.get("trace_id"),
                "requestId": ctx.get("request_id"),
                "userId": ctx.get("user_id"),
                "sourceIp": ctx.get("source_ip"),
            })

        # 追加フィールドの処理（機密データのマスク）
        if hasattr(record, 'extra_fields'):
            sanitized = self._sanitize(record.extra_fields)
            log_entry["details"] = sanitized

        # 例外情報の追加
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "stacktrace": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, ensure_ascii=False, default=str)

    def _sanitize(self, data: Any) -> Any:
        """機密データをマスクする"""
        if isinstance(data, dict):
            return {
                k: "***REDACTED***" if k.lower() in self.SENSITIVE_FIELDS
                else self._sanitize(v)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            return [self._sanitize(item) for item in data]
        return data


class SecurityLogger:
    """セキュリティイベント専用ロガー"""

    def __init__(self, service_name: str, environment: str = "production"):
        self.logger = logging.getLogger(f"security.{service_name}")
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            StructuredLogFormatter(service_name, environment)
        )
        self.logger.addHandler(handler)

    def log_auth_event(self, event: str, user_id: str,
                       success: bool, **kwargs):
        """認証イベントをログに記録"""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Authentication event: {event}", (), None
        )
        record.extra_fields = {
            "event_type": "authentication",
            "event": event,
            "user_id": user_id,
            "success": success,
            **kwargs,
        }
        self.logger.handle(record)

    def log_access_event(self, resource: str, action: str,
                         allowed: bool, **kwargs):
        """アクセス制御イベントをログに記録"""
        level = logging.INFO if allowed else logging.WARNING
        record = self.logger.makeRecord(
            self.logger.name, level, "", 0,
            f"Access event: {action} on {resource}", (), None
        )
        record.extra_fields = {
            "event_type": "access_control",
            "resource": resource,
            "action": action,
            "allowed": allowed,
            **kwargs,
        }
        self.logger.handle(record)

    def log_data_event(self, operation: str, data_type: str,
                       record_count: int, **kwargs):
        """データ操作イベントをログに記録"""
        record = self.logger.makeRecord(
            self.logger.name, logging.INFO, "", 0,
            f"Data event: {operation} on {data_type}", (), None
        )
        record.extra_fields = {
            "event_type": "data_operation",
            "operation": operation,
            "data_type": data_type,
            "record_count": record_count,
            **kwargs,
        }
        self.logger.handle(record)


# 使用例
sec_logger = SecurityLogger("auth-service")
sec_logger.log_auth_event(
    event="login_failed",
    user_id="user-789",
    success=False,
    reason="invalid_password",
    attempt_count=5,
    source_ip="203.0.113.50",
)
```

### Fluent Bit の設定

```ini
# /etc/fluent-bit/fluent-bit.conf

[SERVICE]
    Flush         5
    Daemon        Off
    Log_Level     info
    Parsers_File  parsers.conf
    # メトリクスの公開（Prometheus 形式）
    HTTP_Server   On
    HTTP_Listen   0.0.0.0
    HTTP_Port     2020
    # バッファ管理
    storage.path              /var/log/fluent-bit/buffer/
    storage.sync              normal
    storage.checksum          off
    storage.backlog.mem_limit 50M

# アプリケーションログ
[INPUT]
    Name              tail
    Path              /var/log/app/*.log
    Parser            json
    Tag               app.*
    Refresh_Interval  5
    Mem_Buf_Limit     50MB
    # バッファをファイルに永続化（データ損失防止）
    storage.type      filesystem
    Skip_Long_Lines   On
    DB                /var/log/fluent-bit/app.db

# OS syslog
[INPUT]
    Name              systemd
    Tag               system.*
    Systemd_Filter    _SYSTEMD_UNIT=sshd.service
    Read_From_Tail    On

# Kubernetes メタデータの付与
[FILTER]
    Name              kubernetes
    Match             app.*
    Kube_URL          https://kubernetes.default.svc
    Kube_CA_File      /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    Kube_Token_File   /var/run/secrets/kubernetes.io/serviceaccount/token
    Merge_Log         On

# メタデータの追加
[FILTER]
    Name              record_modifier
    Match             *
    Record hostname   ${HOSTNAME}
    Record env        production
    Record cluster    prod-ap-northeast-1

# 機密データのマスキング
[FILTER]
    Name              lua
    Match             app.*
    script            /etc/fluent-bit/scripts/mask_sensitive.lua
    call              mask_fields

# OpenSearch に送信
[OUTPUT]
    Name              opensearch
    Match             app.*
    Host              opensearch.internal.example.com
    Port              443
    TLS               On
    Index             app-logs
    Type              _doc
    Suppress_Type_Name On
    Logstash_Format   On
    Logstash_Prefix   app-logs
    # リトライ設定
    Retry_Limit       5
    # バッファリング
    Buffer_Size       512KB
    # 認証
    HTTP_User         fluent-bit
    HTTP_Passwd       ${OPENSEARCH_PASSWORD}

# S3 にバックアップ
[OUTPUT]
    Name              s3
    Match             *
    bucket            security-logs-archive
    region            ap-northeast-1
    total_file_size   100M
    upload_timeout    10m
    s3_key_format     /logs/%Y/%m/%d/$TAG/%H-%M-%S
    # 圧縮
    compression       gzip
    # サーバサイド暗号化
    use_put_object    On
```

### Fluent Bit の機密データマスキングスクリプト

```lua
-- /etc/fluent-bit/scripts/mask_sensitive.lua
-- 機密データをログ出力前にマスクする

local sensitive_patterns = {
    -- クレジットカード番号（16桁）
    {pattern = "%d%d%d%d[- ]?%d%d%d%d[- ]?%d%d%d%d[- ]?%d%d%d%d",
     replacement = "****-****-****-XXXX"},
    -- メールアドレス
    {pattern = "[%w%.%-]+@[%w%.%-]+%.%w+",
     replacement = "***@***.***"},
    -- AWS アクセスキー
    {pattern = "AKIA[0-9A-Z]%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d",
     replacement = "AKIA****************"},
}

function mask_fields(tag, timestamp, record)
    local modified = false
    for key, value in pairs(record) do
        if type(value) == "string" then
            for _, pat in ipairs(sensitive_patterns) do
                local new_value = string.gsub(value, pat.pattern,
                                              pat.replacement)
                if new_value ~= value then
                    record[key] = new_value
                    modified = true
                end
            end
        end
    end
    if modified then
        return 1, timestamp, record
    end
    return 0, timestamp, record
end
```

### Vector による高性能ログ収集

Fluent Bit の代替として、Rust 製の Vector は高いパフォーマンスと柔軟な設定が特徴である。

```toml
# /etc/vector/vector.toml

[api]
enabled = true
address = "127.0.0.1:8686"

# ソース: アプリケーションログ
[sources.app_logs]
type = "file"
include = ["/var/log/app/*.log"]
read_from = "beginning"

# ソース: Kubernetes ログ
[sources.k8s_logs]
type = "kubernetes_logs"
auto_partial_merge = true

# トランスフォーム: JSON パース
[transforms.parse_json]
type = "remap"
inputs = ["app_logs"]
source = '''
. = parse_json!(string!(.message))
.timestamp = to_timestamp!(.timestamp)
.environment = "production"
.cluster = "prod-ap-northeast-1"
'''

# トランスフォーム: 機密データのリダクション
[transforms.redact_sensitive]
type = "remap"
inputs = ["parse_json"]
source = '''
if exists(.details.password) {
    .details.password = "***REDACTED***"
}
if exists(.details.token) {
    .details.token = "***REDACTED***"
}
# クレジットカード番号のマスク
if exists(.details.card_number) {
    card = string!(.details.card_number)
    .details.card_number = "****-****-****-" + slice!(card, -4)
}
'''

# トランスフォーム: セキュリティイベントのフィルタリング
[transforms.security_filter]
type = "filter"
inputs = ["redact_sensitive"]
condition = '''
.level == "ERROR" || .level == "WARN" ||
.event_type == "authentication" ||
.event_type == "access_control"
'''

# シンク: OpenSearch
[sinks.opensearch]
type = "elasticsearch"
inputs = ["redact_sensitive"]
endpoints = ["https://opensearch.internal.example.com:443"]
bulk.index = "app-logs-%Y-%m-%d"
auth.strategy = "basic"
auth.user = "vector"
auth.password = "${OPENSEARCH_PASSWORD}"

# シンク: セキュリティイベントを専用インデックスに
[sinks.security_opensearch]
type = "elasticsearch"
inputs = ["security_filter"]
endpoints = ["https://opensearch.internal.example.com:443"]
bulk.index = "security-events-%Y-%m-%d"

# シンク: S3 アーカイブ
[sinks.s3_archive]
type = "aws_s3"
inputs = ["redact_sensitive"]
bucket = "security-logs-archive"
region = "ap-northeast-1"
key_prefix = "logs/%Y/%m/%d/"
compression = "gzip"
encoding.codec = "json"
```

### ログ収集エージェントの比較

| 項目 | Fluent Bit | Fluentd | Vector | CloudWatch Agent |
|------|-----------|---------|--------|-----------------|
| 言語 | C | Ruby/C | Rust | Go |
| メモリ使用量 | ~1MB | ~40MB | ~10MB | ~30MB |
| スループット | 高 | 中 | 非常に高 | 中 |
| プラグイン数 | 中 | 多い | 中 | 少ない |
| Kubernetes 対応 | 良好 | 良好 | 良好 | 限定的 |
| 設定の柔軟性 | 中 | 高 | 高 | 低 |
| ユースケース | エッジ/コンテナ | サーバ集約 | 汎用 | AWS 限定 |

---

## 3. SIEM

### SIEM の内部動作

SIEM (Security Information and Event Management) は、ログの集約、正規化、相関分析、アラート生成を一貫して行うプラットフォームである。

```
+----------------------------------------------------------+
|                SIEM の内部処理フロー                        |
|----------------------------------------------------------|
|                                                          |
|  1. 収集 (Collection)                                     |
|  +-- Syslog, API, Agent, ファイルビート                   |
|       |                                                  |
|       v                                                  |
|  2. パース・正規化 (Parsing & Normalization)               |
|  +-- 異なるフォーマットを共通スキーマに変換                |
|  +-- ECS (Elastic Common Schema)                         |
|  +-- OCSF (Open Cybersecurity Schema Framework)          |
|       |                                                  |
|       v                                                  |
|  3. エンリッチメント (Enrichment)                          |
|  +-- GeoIP 情報の付与                                    |
|  +-- 脅威インテリジェンスとの突合                          |
|  +-- ユーザ/アセット情報の付与                            |
|       |                                                  |
|       v                                                  |
|  4. インデックス・保存 (Indexing & Storage)                |
|  +-- 全文検索インデックスの構築                            |
|  +-- 時系列データの最適化                                 |
|       |                                                  |
|       v                                                  |
|  5. 相関分析 (Correlation)                                |
|  +-- ルールベースマッチング                               |
|  +-- 統計的異常検知                                      |
|  +-- 機械学習モデル                                      |
|       |                                                  |
|       v                                                  |
|  6. アラート・対応 (Alerting & Response)                   |
|  +-- チケット自動生成                                    |
|  +-- SOAR 連携（自動対応）                               |
|  +-- ダッシュボード更新                                  |
+----------------------------------------------------------+
```

### SIEM ツールの比較

| 項目 | Splunk | Elastic SIEM | Amazon Security Lake | Datadog SIEM | Sumo Logic |
|------|--------|-------------|---------------------|-------------|-----------|
| デプロイ | オンプレ/SaaS | オンプレ/クラウド | SaaS (AWS) | SaaS | SaaS |
| コスト | 高い (データ量課金) | OSS 版あり | S3 保存量 | 中程度 | 中程度 |
| 相関分析 | 高度 (SPL) | KQL | Athena (SQL) | ログパイプライン | CSE |
| 機械学習 | MLTK | ML Jobs | -- | Anomaly Detection | CSE Insight |
| カスタムルール | SPL | Detection Rules | Lambda | Detection Rules | CSE Rules |
| スケーラビリティ | 高 | 中-高 | 高 (S3 ベース) | 高 | 高 |
| 初期学習コスト | 高 | 中 | 低-中 | 低 | 中 |
| SOAR 統合 | Splunk SOAR | Elastic SOAR | Step Functions | Workflow | SOAR 連携 |

### コスト比較（月間 100GB のログ取り込み想定）

| 項目 | Splunk Cloud | Elastic Cloud | Amazon Security Lake | Datadog SIEM |
|------|-------------|--------------|---------------------|-------------|
| 月額概算 | $4,500-6,000 | $1,500-3,000 | $500-1,000 | $2,000-3,500 |
| 課金モデル | GB/日 | ノード数+ストレージ | S3保存量+クエリ | GB/月 |
| 無料枠 | なし | 14日トライアル | S3のみ | 15日トライアル |
| 長期保存コスト | 高い | 中程度 | 非常に低い | 中程度 |

### 検知ルールの作成 (Sigma ルール)

Sigma は SIEM 間で検知ルールを共有するための共通フォーマットである。YARA が マルウェア検知のための共通フォーマットであるのと同様に、Sigma はログ検知のための共通フォーマットとして位置づけられる。

```yaml
# sigma/rules/credential_access/brute_force_ssh.yml
title: SSH Brute Force Attack
id: a1234567-b890-1234-cdef-567890abcdef
status: stable
description: 同一 IP から短時間に多数の SSH ログイン失敗を検知
author: Security Team
date: 2025/01/15
modified: 2025/03/15
tags:
  - attack.credential_access
  - attack.t1110.001
  - cve.none
references:
  - https://attack.mitre.org/techniques/T1110/001/
logsource:
  category: authentication
  product: linux
detection:
  selection:
    eventid: 'sshd'
    action: 'Failed'
  filter:
    source_ip|cidr:
      - '10.0.0.0/8'
      - '172.16.0.0/12'
  timeframe: 5m
  condition: selection and not filter | count(source_ip) > 10
level: high
falsepositives:
  - 正当なパスワードリセット作業
  - 自動化スクリプトの設定ミス
```

```yaml
# sigma/rules/persistence/new_iam_user.yml
title: AWS IAM User Created
id: b2345678-c901-2345-defg-678901bcdefg
status: stable
description: 新しい IAM ユーザの作成を検知（不正アクセスの持続化の兆候）
author: Security Team
date: 2025/02/01
tags:
  - attack.persistence
  - attack.t1136.003
logsource:
  product: aws
  service: cloudtrail
detection:
  selection:
    eventName: 'CreateUser'
    eventSource: 'iam.amazonaws.com'
  filter_automation:
    userIdentity.arn|contains:
      - 'terraform'
      - 'cloudformation'
  condition: selection and not filter_automation
level: medium
falsepositives:
  - IaC による正当なユーザ作成
  - オンボーディング作業
```

```yaml
# sigma/rules/exfiltration/large_s3_download.yml
title: Large S3 Data Download
id: c3456789-d012-3456-efgh-789012cdefgh
status: experimental
description: S3 バケットからの大量データダウンロードを検知
author: Security Team
date: 2025/03/01
tags:
  - attack.exfiltration
  - attack.t1530
logsource:
  product: aws
  service: cloudtrail
detection:
  selection:
    eventName: 'GetObject'
    eventSource: 's3.amazonaws.com'
  timeframe: 1h
  condition: selection | count(requestParameters.bucketName) > 1000
level: high
falsepositives:
  - バックアップジョブ
  - データ移行作業
  - 正当な大量ダウンロード処理
```

### Sigma ルールを各 SIEM に変換

```bash
# Sigma CLI で各 SIEM のクエリ形式に変換
pip install sigma-cli

# 利用可能なバックエンドの確認
sigma list backends

# Splunk SPL に変換
sigma convert -t splunk -p sysmon sigma/rules/

# Elastic EQL に変換
sigma convert -t elasticsearch sigma/rules/

# OpenSearch に変換
sigma convert -t opensearch sigma/rules/

# Microsoft Sentinel KQL に変換
sigma convert -t microsoft365defender sigma/rules/

# 変換例 (Splunk SPL):
# source=sshd action="Failed"
# NOT (source_ip="10.0.0.0/8" OR source_ip="172.16.0.0/12")
# | stats count by source_ip
# | where count > 10

# バッチ変換とテスト
sigma convert -t splunk sigma/rules/ -o splunk_rules/
sigma check sigma/rules/  # ルールの構文チェック
```

### 相関ルールの実装例

```python
# コード例2: SIEM 相関ルールの実装（概念実装）
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict
import heapq

@dataclass
class LogEvent:
    """ログイベントの共通構造"""
    timestamp: datetime
    source: str
    event_type: str
    severity: str
    source_ip: str
    user_id: Optional[str] = None
    details: Dict = field(default_factory=dict)

@dataclass
class Alert:
    """相関分析結果のアラート"""
    rule_id: str
    title: str
    severity: str
    description: str
    events: List[LogEvent]
    created_at: datetime = field(
        default_factory=lambda: datetime.utcnow()
    )
    mitre_tactic: Optional[str] = None
    mitre_technique: Optional[str] = None

class CorrelationEngine:
    """SIEM 相関分析エンジン

    複数のログイベントを時系列で分析し、
    単体のイベントでは検知できない攻撃パターンを検出する。

    内部動作:
    1. イベントを時系列でバッファリング
    2. 各相関ルールとマッチングを実行
    3. 閾値を超えた場合にアラートを生成
    4. 重複排除と優先度計算を行う
    """

    def __init__(self, window_minutes: int = 60):
        self.window = timedelta(minutes=window_minutes)
        self.event_buffer: List[LogEvent] = []
        self.rules: List[callable] = []
        self.alerts: List[Alert] = []
        # ルールごとのイベントカウンタ
        self._counters: Dict[str, Dict] = defaultdict(
            lambda: defaultdict(int)
        )

    def add_event(self, event: LogEvent) -> List[Alert]:
        """イベントを追加し、相関ルールをチェック"""
        self.event_buffer.append(event)
        self._cleanup_old_events()

        new_alerts = []
        for rule in self.rules:
            alert = rule(event, self.event_buffer)
            if alert:
                # 重複アラートの排除
                if not self._is_duplicate(alert):
                    new_alerts.append(alert)
                    self.alerts.append(alert)

        return new_alerts

    def _cleanup_old_events(self):
        """ウィンドウ外の古いイベントを削除"""
        cutoff = datetime.utcnow() - self.window
        self.event_buffer = [
            e for e in self.event_buffer
            if e.timestamp > cutoff
        ]

    def _is_duplicate(self, alert: Alert) -> bool:
        """過去1時間以内の同一ルールIDのアラートを重複とみなす"""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        return any(
            a.rule_id == alert.rule_id and a.created_at > cutoff
            for a in self.alerts
        )

    def register_rule(self, rule_func: callable):
        """相関ルールを登録"""
        self.rules.append(rule_func)


def brute_force_rule(event: LogEvent,
                     buffer: List[LogEvent]) -> Optional[Alert]:
    """ブルートフォース検知ルール

    5分以内に同一IPから10回以上のログイン失敗を検知
    """
    if event.event_type != "login_failed":
        return None

    window = timedelta(minutes=5)
    cutoff = event.timestamp - window

    failed_from_same_ip = [
        e for e in buffer
        if e.event_type == "login_failed"
        and e.source_ip == event.source_ip
        and e.timestamp > cutoff
    ]

    if len(failed_from_same_ip) >= 10:
        return Alert(
            rule_id="BRUTE_FORCE_001",
            title=f"ブルートフォース攻撃検知: {event.source_ip}",
            severity="HIGH",
            description=(
                f"IP {event.source_ip} から5分間に"
                f"{len(failed_from_same_ip)}回のログイン失敗"
            ),
            events=failed_from_same_ip,
            mitre_tactic="Credential Access",
            mitre_technique="T1110.001",
        )
    return None


def impossible_travel_rule(event: LogEvent,
                           buffer: List[LogEvent]) -> Optional[Alert]:
    """不可能な移動検知ルール

    短時間内に地理的に離れた場所からのログインを検知
    """
    if event.event_type != "login_success":
        return None

    window = timedelta(hours=1)
    cutoff = event.timestamp - window

    recent_logins = [
        e for e in buffer
        if e.event_type == "login_success"
        and e.user_id == event.user_id
        and e.timestamp > cutoff
        and e.source_ip != event.source_ip
    ]

    for prev_login in recent_logins:
        # 地理的距離を計算（実装略）
        distance = calculate_geo_distance(
            prev_login.details.get("geo"),
            event.details.get("geo"),
        )
        time_diff = (event.timestamp - prev_login.timestamp).total_seconds()

        # 時速1000km以上の移動は不可能と判定
        if distance > 0 and time_diff > 0:
            speed_kmh = (distance / time_diff) * 3600
            if speed_kmh > 1000:
                return Alert(
                    rule_id="IMPOSSIBLE_TRAVEL_001",
                    title=f"不可能な移動検知: {event.user_id}",
                    severity="HIGH",
                    description=(
                        f"ユーザ {event.user_id} が"
                        f"{time_diff/60:.0f}分間で"
                        f"{distance:.0f}km 移動"
                    ),
                    events=[prev_login, event],
                    mitre_tactic="Initial Access",
                    mitre_technique="T1078",
                )
    return None


# エンジンの使用例
engine = CorrelationEngine(window_minutes=60)
engine.register_rule(brute_force_rule)
engine.register_rule(impossible_travel_rule)
```

---

## 4. 異常検知

### ルールベース vs 機械学習ベース

```
+----------------------------------------------------------+
|          異常検知アプローチの詳細比較                       |
|----------------------------------------------------------|
|                                                          |
|  ルールベース検知:                                         |
|  +-- 既知の攻撃パターンに効果的                            |
|  +-- 偽陽性の制御が容易                                   |
|  +-- 新規の攻撃パターンには対応不可                        |
|  +-- 例: "5分間にSSHログイン失敗10回以上"                 |
|  +-- メリット: 透明性が高い、チューニングが容易            |
|  +-- デメリット: 未知の脅威を検知できない                  |
|                                                          |
|  統計ベース検知:                                          |
|  +-- ベースラインからの標準偏差を計算                     |
|  +-- 比較的シンプルなモデル                               |
|  +-- 例: "平均の3σ以上のAPIコール数"                      |
|  +-- メリット: 解釈が容易、計算コストが低い               |
|  +-- デメリット: 季節変動への対応が必要                   |
|                                                          |
|  機械学習ベース検知:                                       |
|  +-- ベースラインからの逸脱を検知                         |
|  +-- 未知の攻撃パターンにも対応可能                       |
|  +-- 偽陽性のチューニングが必要                           |
|  +-- 例: "通常と異なるデータ転送パターン"                 |
|  +-- メリット: 複雑なパターンを自動学習                   |
|  +-- デメリット: ブラックボックス化のリスク               |
+----------------------------------------------------------+
```

### 検知手法の比較表

| 項目 | ルールベース | 統計ベース | 教師なしML | 教師ありML |
|------|-----------|----------|----------|----------|
| 検知対象 | 既知の攻撃 | 異常な統計値 | 未知の異常 | 既知の攻撃+類似 |
| 学習期間 | 不要 | 1-2週間 | 2-4週間 | 大量の学習データ |
| 偽陽性率 | 低い | 中程度 | 高い | 低-中 |
| 偽陰性率 | 高い（未知攻撃） | 中程度 | 低い | 中程度 |
| 運用コスト | 低い | 低い | 高い | 高い |
| 説明可能性 | 高い | 高い | 低い | 中程度 |
| ユースケース | コンプライアンス | 性能監視 | 内部脅威検知 | マルウェア検知 |

### 検知すべき異常パターン

```
+----------------------------------------------------------+
|            セキュリティ異常検知パターン                      |
|----------------------------------------------------------|
|                                                          |
|  [認証・アクセス]                                          |
|  +-- 短時間の大量ログイン失敗 (ブルートフォース)            |
|  +-- 通常と異なる時間帯のアクセス                          |
|  +-- 地理的に不可能なログイン (Impossible Travel)          |
|  +-- 権限昇格の試行                                      |
|  +-- 普段と異なるユーザーエージェント                     |
|  +-- 休日・深夜のアクセス                                |
|                                                          |
|  [データ]                                                 |
|  +-- 大量データの外部転送 (Exfiltration)                  |
|  +-- 通常と異なるDB クエリパターン                         |
|  +-- 機密データへの異常なアクセス頻度                       |
|  +-- バックアップデータへの予期しないアクセス               |
|  +-- データベースのスキーマ変更                            |
|                                                          |
|  [ネットワーク]                                           |
|  +-- C2 通信パターン (ビーコニング)                        |
|  +-- DNS トンネリング                                     |
|  +-- ポートスキャン                                       |
|  +-- 内部ネットワークのラテラルムーブメント                  |
|  +-- 通常使用しないプロトコルの通信                       |
|  +-- 暗号化されたトラフィックの急増                       |
|                                                          |
|  [システム]                                               |
|  +-- プロセスの異常な動作                                 |
|  +-- ファイルの大量暗号化 (ランサムウェア)                  |
|  +-- 設定変更 (CloudTrail 無効化等)                       |
|  +-- 予期しないサービスの起動                             |
|  +-- カーネルモジュールのロード                           |
+----------------------------------------------------------+
```

### CloudWatch Metric Filter + アラーム

```python
# コード例3: CloudWatch による異常検知の自動構築
import boto3
import json

logs = boto3.client('logs')
cloudwatch = boto3.client('cloudwatch')
sns = boto3.client('sns')

class SecurityAlarmBuilder:
    """CloudWatch セキュリティアラームの自動構築

    以下のベストプラクティスに従って構築する:
    - 重要なセキュリティイベントのみアラート化
    - 閾値は環境の特性に合わせて調整
    - 通知先は重大度に応じて分離
    """

    def __init__(self, log_group: str, sns_topic_arn: str):
        self.log_group = log_group
        self.sns_topic_arn = sns_topic_arn

    def create_root_account_alarm(self):
        """Root アカウント使用の検知"""
        logs.put_metric_filter(
            logGroupName=self.log_group,
            filterName='RootAccountUsage',
            filterPattern=(
                '{ $.userIdentity.type = "Root" '
                '&& $.userIdentity.invokedBy NOT EXISTS '
                '&& $.eventType != "AwsServiceEvent" }'
            ),
            metricTransformations=[{
                'metricName': 'RootAccountUsageCount',
                'metricNamespace': 'SecurityMetrics',
                'metricValue': '1',
            }],
        )

        cloudwatch.put_metric_alarm(
            AlarmName='RootAccountUsage',
            MetricName='RootAccountUsageCount',
            Namespace='SecurityMetrics',
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=1,
            Threshold=1,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            AlarmActions=[self.sns_topic_arn],
            TreatMissingData='notBreaching',
        )

    def create_unauthorized_api_alarm(self):
        """未認可 API 呼び出しの検知"""
        logs.put_metric_filter(
            logGroupName=self.log_group,
            filterName='UnauthorizedAPICalls',
            filterPattern=(
                '{ ($.errorCode = "*UnauthorizedAccess*") '
                '|| ($.errorCode = "AccessDenied*") }'
            ),
            metricTransformations=[{
                'metricName': 'UnauthorizedAPICallCount',
                'metricNamespace': 'SecurityMetrics',
                'metricValue': '1',
            }],
        )

        cloudwatch.put_metric_alarm(
            AlarmName='UnauthorizedAPICalls',
            MetricName='UnauthorizedAPICallCount',
            Namespace='SecurityMetrics',
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=1,
            Threshold=5,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            AlarmActions=[self.sns_topic_arn],
            TreatMissingData='notBreaching',
        )

    def create_console_login_failure_alarm(self):
        """コンソールログイン失敗の検知"""
        logs.put_metric_filter(
            logGroupName=self.log_group,
            filterName='ConsoleLoginFailures',
            filterPattern=(
                '{ ($.eventName = "ConsoleLogin") '
                '&& ($.errorMessage = "Failed authentication") }'
            ),
            metricTransformations=[{
                'metricName': 'ConsoleLoginFailureCount',
                'metricNamespace': 'SecurityMetrics',
                'metricValue': '1',
            }],
        )

        cloudwatch.put_metric_alarm(
            AlarmName='ConsoleLoginFailures',
            MetricName='ConsoleLoginFailureCount',
            Namespace='SecurityMetrics',
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=1,
            Threshold=3,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            AlarmActions=[self.sns_topic_arn],
            TreatMissingData='notBreaching',
        )

    def create_security_group_change_alarm(self):
        """セキュリティグループの変更検知"""
        logs.put_metric_filter(
            logGroupName=self.log_group,
            filterName='SecurityGroupChanges',
            filterPattern=(
                '{ ($.eventName = "AuthorizeSecurityGroupIngress") '
                '|| ($.eventName = "AuthorizeSecurityGroupEgress") '
                '|| ($.eventName = "RevokeSecurityGroupIngress") '
                '|| ($.eventName = "RevokeSecurityGroupEgress") '
                '|| ($.eventName = "CreateSecurityGroup") '
                '|| ($.eventName = "DeleteSecurityGroup") }'
            ),
            metricTransformations=[{
                'metricName': 'SecurityGroupChangeCount',
                'metricNamespace': 'SecurityMetrics',
                'metricValue': '1',
            }],
        )

        cloudwatch.put_metric_alarm(
            AlarmName='SecurityGroupChanges',
            MetricName='SecurityGroupChangeCount',
            Namespace='SecurityMetrics',
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=1,
            Threshold=1,
            ComparisonOperator='GreaterThanOrEqualToThreshold',
            AlarmActions=[self.sns_topic_arn],
            TreatMissingData='notBreaching',
        )

    def create_all_alarms(self):
        """全セキュリティアラームを一括作成"""
        self.create_root_account_alarm()
        self.create_unauthorized_api_alarm()
        self.create_console_login_failure_alarm()
        self.create_security_group_change_alarm()
        print("All security alarms created successfully")


# 使用例
builder = SecurityAlarmBuilder(
    log_group='/aws/cloudtrail',
    sns_topic_arn='arn:aws:sns:ap-northeast-1:123456:security-alerts',
)
builder.create_all_alarms()
```

### ベースラインベースの異常検知実装

```python
# コード例4: 統計的異常検知の実装
import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AnomalyResult:
    """異常検知結果"""
    metric_name: str
    current_value: float
    baseline_mean: float
    baseline_stddev: float
    z_score: float
    is_anomaly: bool
    severity: str  # low, medium, high, critical
    timestamp: datetime

class StatisticalAnomalyDetector:
    """統計的ベースライン異常検知器

    指数移動平均（EMA）と標準偏差を用いて
    メトリクスのベースラインを動的に計算し、
    異常値を検出する。

    動作原理:
    1. 過去のデータポイントからEMAを計算
    2. 標準偏差を計算
    3. 新しいデータポイントのZスコアを算出
    4. Zスコアが閾値を超えた場合に異常と判定

    Zスコア = (観測値 - 平均) / 標準偏差
    - |Z| > 2: 低リスク異常
    - |Z| > 3: 中リスク異常
    - |Z| > 4: 高リスク異常
    """

    def __init__(self, window_hours: int = 168,  # 1週間
                 min_data_points: int = 100):
        self.window_hours = window_hours
        self.min_data_points = min_data_points
        # メトリクス名 → (timestamp, value) のリスト
        self.data_points: dict = defaultdict(list)

    def add_data_point(self, metric_name: str, value: float,
                       timestamp: Optional[datetime] = None):
        """データポイントを追加"""
        ts = timestamp or datetime.utcnow()
        self.data_points[metric_name].append((ts, value))
        self._cleanup(metric_name)

    def check_anomaly(self, metric_name: str,
                      value: float) -> Optional[AnomalyResult]:
        """異常値チェック"""
        points = self.data_points.get(metric_name, [])

        if len(points) < self.min_data_points:
            return None  # データ不足

        values = [p[1] for p in points]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = math.sqrt(variance) if variance > 0 else 0.001

        z_score = (value - mean) / stddev

        severity = self._classify_severity(abs(z_score))
        is_anomaly = abs(z_score) > 2.0

        return AnomalyResult(
            metric_name=metric_name,
            current_value=value,
            baseline_mean=mean,
            baseline_stddev=stddev,
            z_score=z_score,
            is_anomaly=is_anomaly,
            severity=severity,
            timestamp=datetime.utcnow(),
        )

    def _classify_severity(self, abs_z: float) -> str:
        if abs_z > 4.0:
            return "critical"
        elif abs_z > 3.0:
            return "high"
        elif abs_z > 2.0:
            return "medium"
        else:
            return "low"

    def _cleanup(self, metric_name: str):
        """ウィンドウ外の古いデータを削除"""
        cutoff = datetime.utcnow() - timedelta(hours=self.window_hours)
        self.data_points[metric_name] = [
            p for p in self.data_points[metric_name]
            if p[0] > cutoff
        ]


# 使用例
detector = StatisticalAnomalyDetector()

# ベースラインの構築（過去データのシミュレーション）
import random
for i in range(200):
    # 通常のAPI呼び出し数: 平均100, 標準偏差20
    normal_value = random.gauss(100, 20)
    detector.add_data_point(
        "api_calls_per_minute",
        max(0, normal_value),
        datetime.utcnow() - timedelta(hours=200-i),
    )

# 異常値のチェック
result = detector.check_anomaly("api_calls_per_minute", 250)
if result and result.is_anomaly:
    print(f"ANOMALY DETECTED: {result.metric_name}")
    print(f"  Current: {result.current_value}")
    print(f"  Baseline: {result.baseline_mean:.1f} +/- "
          f"{result.baseline_stddev:.1f}")
    print(f"  Z-score: {result.z_score:.2f}")
    print(f"  Severity: {result.severity}")
```

---

## 5. ダッシュボード設計

### セキュリティダッシュボードの構成

```
+----------------------------------------------------------+
|  Security Operations Dashboard                           |
|----------------------------------------------------------|
|                                                          |
|  [概要パネル]                                             |
|  +-- アクティブアラート数 (Critical/High/Medium/Low)       |
|  +-- 過去24時間のインシデント数                            |
|  +-- 平均検知時間 (MTTD)                                  |
|  +-- 平均対応時間 (MTTR)                                  |
|  +-- セキュリティスコア (0-100)                            |
|                                                          |
|  [トレンドグラフ]                                         |
|  +-- アラート推移 (日次/週次)                              |
|  +-- ログインエラーの推移                                 |
|  +-- ネットワークトラフィック量の推移                       |
|  +-- 異常検知イベントの推移                               |
|                                                          |
|  [地理マップ]                                             |
|  +-- ソース IP の地理分布                                 |
|  +-- 異常な接続元の国別表示                               |
|  +-- ブロックされた IP の分布                             |
|                                                          |
|  [テーブル]                                               |
|  +-- 最新のアラート一覧                                   |
|  +-- トップ攻撃者 IP                                      |
|  +-- 最もアクセスされたエンドポイント                      |
|  +-- 未対応の脆弱性一覧                                  |
+----------------------------------------------------------+
```

### ダッシュボード設計のベストプラクティス

```
+----------------------------------------------------------+
|          ダッシュボード設計の7原則                          |
|----------------------------------------------------------|
|                                                          |
|  1. 5秒ルール                                             |
|     → 画面を見て5秒以内に状況を把握できること             |
|     → 重要な KPI は画面上部に大きく表示                   |
|                                                          |
|  2. アクション駆動                                        |
|     → 「見るだけ」のデータは排除                         |
|     → 各パネルに「何をすべきか」が明確                   |
|                                                          |
|  3. 階層化                                               |
|     → 概要 → 詳細 → 生データのドリルダウン              |
|     → エグゼクティブ向け / アナリスト向けの分離           |
|                                                          |
|  4. リアルタイム性                                        |
|     → Critical アラートは 1分以内に反映                   |
|     → トレンドは 5分間隔で更新                           |
|                                                          |
|  5. コンテキスト                                          |
|     → 数値には比較対象を付ける（前日比、前週比）          |
|     → 閾値を超えたら色で強調                             |
|                                                          |
|  6. ノイズ排除                                            |
|     → 情報過多を避ける（1画面に7個以下のパネル）         |
|     → フィルタリング機能の提供                           |
|                                                          |
|  7. 一貫性                                               |
|     → 色の意味を統一（赤=Critical, 黄=Warning等）        |
|     → 時間軸の統一                                      |
+----------------------------------------------------------+
```

### OpenSearch Dashboards のセキュリティダッシュボード構築

```python
# コード例5: OpenSearch ダッシュボードの自動構築スクリプト
import json
import requests
from typing import Dict, List

class SecurityDashboardBuilder:
    """OpenSearch Dashboards のセキュリティダッシュボード自動構築"""

    def __init__(self, opensearch_url: str, auth: tuple):
        self.url = opensearch_url
        self.auth = auth
        self.headers = {"Content-Type": "application/json"}

    def create_index_pattern(self, pattern: str, time_field: str):
        """インデックスパターンの作成"""
        body = {
            "attributes": {
                "title": pattern,
                "timeFieldName": time_field,
            }
        }
        requests.post(
            f"{self.url}/api/saved_objects/index-pattern/{pattern}",
            json=body, auth=self.auth, headers=self.headers,
        )

    def create_failed_login_visualization(self) -> dict:
        """ログイン失敗の時系列グラフ"""
        return {
            "title": "Failed Login Attempts Over Time",
            "visState": json.dumps({
                "title": "Failed Login Attempts",
                "type": "line",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                    },
                    {
                        "id": "2",
                        "type": "date_histogram",
                        "schema": "segment",
                        "params": {
                            "field": "@timestamp",
                            "interval": "5m",
                        },
                    },
                ],
                "params": {
                    "seriesParams": [{
                        "show": True,
                        "type": "line",
                        "mode": "normal",
                    }],
                },
            }),
            "kibanaSavedObjectMeta": {
                "searchSourceJSON": json.dumps({
                    "query": {
                        "bool": {
                            "must": [
                                {"match": {"event": "login_failed"}}
                            ]
                        }
                    },
                    "index": "security-events-*",
                }),
            },
        }

    def create_top_attackers_table(self) -> dict:
        """攻撃元 IP トップ10テーブル"""
        return {
            "title": "Top Attacking IPs",
            "visState": json.dumps({
                "title": "Top Attacking IPs",
                "type": "table",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                        "params": {"customLabel": "Attack Count"},
                    },
                    {
                        "id": "2",
                        "type": "terms",
                        "schema": "bucket",
                        "params": {
                            "field": "sourceIp.keyword",
                            "size": 10,
                            "order": "desc",
                            "orderBy": "1",
                            "customLabel": "Source IP",
                        },
                    },
                ],
            }),
        }

    def create_geo_map(self) -> dict:
        """攻撃元の地理分布マップ"""
        return {
            "title": "Attack Source Geographic Distribution",
            "visState": json.dumps({
                "title": "Attack Sources Map",
                "type": "tile_map",
                "aggs": [
                    {
                        "id": "1",
                        "type": "count",
                        "schema": "metric",
                    },
                    {
                        "id": "2",
                        "type": "geohash_grid",
                        "schema": "segment",
                        "params": {
                            "field": "geoLocation",
                            "precision": 3,
                        },
                    },
                ],
            }),
        }

    def build_security_dashboard(self):
        """セキュリティダッシュボードの構築"""
        self.create_index_pattern("security-events-*", "@timestamp")

        visualizations = [
            self.create_failed_login_visualization(),
            self.create_top_attackers_table(),
            self.create_geo_map(),
        ]

        for viz in visualizations:
            requests.post(
                f"{self.url}/api/saved_objects/visualization",
                json={"attributes": viz},
                auth=self.auth,
                headers=self.headers,
            )

        print(f"Dashboard created with {len(visualizations)} visualizations")
```

---

## 6. ログ保存とパフォーマンス

### 階層型ストレージ設計

```
+----------------------------------------------------------+
|          ログの階層型ストレージ戦略                         |
|----------------------------------------------------------|
|                                                          |
|  Hot Tier (0-30日)                                       |
|  +-- ストレージ: OpenSearch / Elasticsearch               |
|  +-- 特性: 即座に検索可能、高コスト                       |
|  +-- 用途: リアルタイム分析、アラート検知                 |
|  +-- コスト: ~$0.10/GB/月（インスタンスコスト含む）       |
|                                                          |
|  Warm Tier (30-90日)                                      |
|  +-- ストレージ: S3 Standard + Athena                     |
|  +-- 特性: SQL クエリで検索、中コスト                     |
|  +-- 用途: インシデント調査、フォレンジック               |
|  +-- コスト: ~$0.023/GB/月                               |
|                                                          |
|  Cold Tier (90-365日)                                     |
|  +-- ストレージ: S3 Infrequent Access                     |
|  +-- 特性: アクセス料金あり、低コスト                     |
|  +-- 用途: コンプライアンス要件の充足                     |
|  +-- コスト: ~$0.0125/GB/月                              |
|                                                          |
|  Archive Tier (365日以上)                                  |
|  +-- ストレージ: S3 Glacier Deep Archive                  |
|  +-- 特性: 復元に12時間、最低コスト                      |
|  +-- 用途: 法的保持要件、訴訟対応                        |
|  +-- コスト: ~$0.00099/GB/月                             |
+----------------------------------------------------------+
```

### S3 ライフサイクルポリシーの設定

```python
# コード例6: S3 ライフサイクルポリシーの設定
import boto3

s3 = boto3.client('s3')

def configure_log_lifecycle(bucket_name: str):
    """セキュリティログ用のライフサイクルポリシーを設定"""

    lifecycle_config = {
        'Rules': [
            {
                'ID': 'SecurityLogLifecycle',
                'Status': 'Enabled',
                'Filter': {
                    'Prefix': 'logs/',
                },
                'Transitions': [
                    {
                        'Days': 30,
                        'StorageClass': 'STANDARD_IA',
                    },
                    {
                        'Days': 90,
                        'StorageClass': 'GLACIER',
                    },
                    {
                        'Days': 365,
                        'StorageClass': 'DEEP_ARCHIVE',
                    },
                ],
                # PCI DSS/SOC 2 要件: 最低1年保持
                # 法的保持要件により7年保持する場合もある
                'Expiration': {
                    'Days': 2555,  # 7年
                },
                'NoncurrentVersionExpiration': {
                    'NoncurrentDays': 30,
                },
            },
        ],
    }

    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration=lifecycle_config,
    )

    # バケット暗号化の設定
    s3.put_bucket_encryption(
        Bucket=bucket_name,
        ServerSideEncryptionConfiguration={
            'Rules': [{
                'ApplyServerSideEncryptionByDefault': {
                    'SSEAlgorithm': 'aws:kms',
                    'KMSMasterKeyID': 'alias/security-logs-key',
                },
                'BucketKeyEnabled': True,
            }]
        },
    )

    # オブジェクトロック（改竄防止）
    s3.put_object_lock_configuration(
        Bucket=bucket_name,
        ObjectLockConfiguration={
            'ObjectLockEnabled': 'Enabled',
            'Rule': {
                'DefaultRetention': {
                    'Mode': 'COMPLIANCE',
                    'Days': 365,
                }
            }
        },
    )

    print(f"Lifecycle policy configured for {bucket_name}")

configure_log_lifecycle("security-logs-archive")
```

### パフォーマンスチューニング

| チューニング対象 | 問題 | 対策 | 効果 |
|----------------|------|------|------|
| OpenSearch インデックス | 検索が遅い | シャード数の最適化（50GB/シャード） | 検索速度 2-5x 向上 |
| Fluent Bit バッファ | メモリ不足 | filesystem ストレージの使用 | データ損失防止 |
| S3 アーカイブ | Athena クエリが遅い | Parquet 形式+パーティショニング | クエリ速度 10x 向上 |
| ログ転送 | ネットワーク帯域 | 圧縮（gzip）の有効化 | 帯域 70-80% 削減 |
| インデックスローテーション | ディスク不足 | ILM ポリシーの設定 | 自動的なストレージ管理 |

---

## 7. エッジケース

### エッジケース 1: ログの欠損

ネットワーク障害やエージェントの異常により、ログが欠損するケースがある。これを検知するために「ハートビートログ」を実装する。

```python
# コード例7: ログ欠損検知の実装
from datetime import datetime, timedelta
from typing import Dict, Set

class LogGapDetector:
    """ログ欠損を検知するモニター

    各ログソースからのログの到着間隔を監視し、
    期待される間隔を超えた場合にアラートを発行する。
    """

    def __init__(self):
        # ソース名 → (最終受信時刻, 期待間隔)
        self.sources: Dict[str, tuple] = {}
        # 期待されるソースの一覧
        self.expected_sources: Set[str] = set()

    def register_source(self, source_name: str,
                        expected_interval: timedelta):
        """ログソースを登録"""
        self.expected_sources.add(source_name)
        self.sources[source_name] = (None, expected_interval)

    def record_log(self, source_name: str):
        """ログの到着を記録"""
        if source_name in self.sources:
            _, interval = self.sources[source_name]
            self.sources[source_name] = (datetime.utcnow(), interval)

    def check_gaps(self) -> list:
        """ログの欠損を検出"""
        alerts = []
        now = datetime.utcnow()

        for source_name in self.expected_sources:
            if source_name not in self.sources:
                alerts.append({
                    'source': source_name,
                    'type': 'source_not_registered',
                    'message': f"ログソース {source_name} が未登録",
                })
                continue

            last_seen, expected_interval = self.sources[source_name]

            if last_seen is None:
                alerts.append({
                    'source': source_name,
                    'type': 'no_logs_received',
                    'message': f"{source_name} からログが一度も到着していない",
                })
            elif (now - last_seen) > expected_interval * 2:
                gap_minutes = (now - last_seen).total_seconds() / 60
                alerts.append({
                    'source': source_name,
                    'type': 'log_gap',
                    'message': (
                        f"{source_name} から{gap_minutes:.0f}分間"
                        f"ログが到着していない"
                    ),
                    'last_seen': last_seen.isoformat(),
                    'gap_minutes': gap_minutes,
                })

        return alerts
```

### エッジケース 2: タイムゾーンの不一致

異なるソースからのログでタイムゾーンが混在すると、相関分析の精度が低下する。すべてのログを UTC で統一することが必須である。

### エッジケース 3: ログのタンパリング

攻撃者がログを改竄・削除する可能性がある。これを防ぐために以下の対策が必要:

- S3 Object Lock による改竄防止
- CloudTrail のログファイル整合性検証の有効化
- ログの複数箇所への同時書き出し（Write-Ahead Log パターン）
- WORM (Write Once Read Many) ストレージの使用

---

## 8. アンチパターン

### アンチパターン 1: ログを収集するが分析しない

```
NG:
  → S3 にログを保存するだけ
  → 誰もログを見ない
  → インシデント発生後に初めて確認する

OK:
  → SIEM で相関分析ルールを設定
  → アラートを自動生成し通知
  → 定期的なログレビューを実施 (週次)
  → KPI (MTTD/MTTR) を測定し改善

影響:
  → 平均検知時間が 200日以上に
  → インシデント発生時の調査に膨大な時間
  → コンプライアンス監査で指摘事項に
```

### アンチパターン 2: アラート疲れ (Alert Fatigue)

```
NG:
  → 1日に数百件のアラートが発生
  → 大半が偽陽性
  → 重要なアラートが埋もれて見逃される

OK:
  → アラートの重大度を適切に設定
  → 偽陽性を減らすためのチューニングを継続
  → 低重大度はダッシュボード表示のみ
  → 通知は High 以上に限定
  → 抑制ルールを文書化して管理

改善プロセス:
  1. 月次でアラートの偽陽性率を集計
  2. 偽陽性率 > 30% のルールをレビュー
  3. 閾値の調整またはルールの無効化
  4. 改善後の効果を測定
```

### アンチパターン 3: 機密データのログ出力

```
NG:
  → パスワードをログに出力
  → クレジットカード番号をログに出力
  → API キーをログに出力
  → セッショントークンをログに出力

OK:
  → 機密データは自動マスキング
  → 構造化ログでフィールドレベルの制御
  → ログレビュープロセスで機密データの混入を確認
  → PII 検出ツールの導入

技術的対策:
  → Fluent Bit の Lua フィルタでマスキング
  → アプリケーション層での事前サニタイズ
  → DLP ツールによる事後検出
```

---

## 9. 演習

### 演習 1: 基礎 — 構造化ログの実装

以下の要件を満たす構造化ログシステムを構築せよ:

1. JSON 形式でログを出力する Python ロガーを実装
2. トレース ID、リクエスト ID を自動的に付与
3. 機密データ（password, token, api_key）を自動マスク
4. ログレベル別のフィルタリング機能を実装

**検証ポイント:**
- ログが JSON として正しくパースできること
- 機密フィールドがマスクされていること
- コンテキスト情報が正しく付与されていること

### 演習 2: 応用 — SIEM 相関ルールの設計

以下の攻撃シナリオを検知する Sigma ルールを作成せよ:

1. 認証情報のスプレー攻撃（複数ユーザに対する同一パスワードでのログイン試行）
2. 権限昇格の試行（通常ユーザが管理者操作を実行）
3. データ流出の兆候（営業時間外の大量データダウンロード）

**検証ポイント:**
- Sigma CLI で構文エラーがないこと
- 偽陽性のシナリオを 2 つ以上列挙していること
- MITRE ATT&CK の TTP がタグ付けされていること

### 演習 3: 発展 — 異常検知システムの構築

以下の要件を満たす異常検知プロトタイプを構築せよ:

1. 統計的ベースライン検知（Zスコアベース）を実装
2. 少なくとも 3 種類のメトリクス（API コール数、データ転送量、ログイン試行数）を監視
3. ベースラインの動的更新（スライディングウィンドウ）を実装
4. 異常検知結果を Slack / PagerDuty に通知する機能を実装

**検証ポイント:**
- 通常値で異常アラートが発生しないこと
- 異常値で確実にアラートが発生すること
- ベースラインが時間経過とともに更新されること

---

## 10. FAQ

### Q1. SIEM のログ保存期間はどのくらい必要か?

PCI DSS では 1 年間（直近 3 ヶ月は即座に検索可能）、SOC 2 では監査期間分（通常 12 ヶ月）が求められる。GDPR では特定の保持期間は定めていないが、データ最小化原則に基づき必要最小限の期間とする。コスト最適化のため、Hot (30日/OpenSearch) → Warm (90日/S3+Athena) → Cold (365日+/Glacier) の段階的保存が効果的である。

### Q2. 小規模チームでも SIEM は必要か?

専用の SIEM 製品でなくても、CloudWatch Logs + Athena + EventBridge の組み合わせで基本的な監視は構築できる。まず CloudTrail、VPC Flow Logs、アプリログの収集を開始し、重要なアラートルールを数個設定するところから始めるのが現実的である。月間コスト $100-200 程度から始められる。

### Q3. 機械学習ベースの異常検知は信頼できるか?

学習期間（通常 2-4 週間）のベースラインデータの品質に依存する。異常なイベントが学習期間に含まれると誤ったベースラインになる。機械学習はルールベース検知を補完するものであり、置き換えるものではない。まずルールベースの検知を充実させ、その上で機械学習を追加するのが推奨順序である。

### Q4. ログのフォーマットは統一すべきか?

可能な限り統一すべきである。ECS (Elastic Common Schema) や OCSF (Open Cybersecurity Schema Framework) などの共通スキーマを採用することで、異なるソースからのログを一貫して分析できる。既存のログフォーマットを変更できない場合は、SIEM 側でパーサーを使って正規化する。

### Q5. ログの暗号化は必要か?

セキュリティログには個人情報や認証情報が含まれる可能性があるため、保存時暗号化（SSE-KMS）と転送時暗号化（TLS）の両方が必要である。特にコンプライアンス要件（PCI DSS、HIPAA）がある場合は必須である。暗号化キーの管理には AWS KMS を使用し、キーのローテーションも設定する。

### Q6. マルチクラウド環境のログ集約はどうするか?

Amazon Security Lake は OCSF スキーマに基づいてマルチクラウドのログを統合する。あるいは、各クラウドのログを共通のバケット（S3）に集約し、Athena や OpenSearch で横断検索する方法もある。ログ転送にはリージョン間のネットワーク料金を考慮する必要がある。

---

## 11. トラブルシューティング

### よくある問題と対処法

| 問題 | 原因 | 対処法 |
|------|------|--------|
| ログが欠損する | エージェントのバッファオーバーフロー | バッファサイズの拡大、filesystem ストレージの使用 |
| 検索が遅い | シャード数の過不足 | 50GB/シャードを目安に調整 |
| アラートが来ない | メトリクスフィルタのパターン不一致 | CloudWatch でフィルタパターンをテスト |
| コストが高い | 全ログを Hot ストレージに保存 | ILM ポリシーで階層化ストレージに移行 |
| 偽陽性が多い | ルールの閾値が低すぎる | ベースラインを分析して閾値を調整 |
| ディスクフル | ログローテーション未設定 | logrotate + ILM ポリシーの設定 |
| タイムスタンプが不正 | NTP の同期不良 | chrony/ntpd の設定確認 |

---

## まとめ

| 項目 | 要点 |
|------|------|
| ログ収集 | CloudTrail、VPC Flow Logs、アプリログを必ず収集 |
| 構造化ログ | JSON 形式で一貫したフォーマットを使用 |
| 機密データ保護 | ログ内の機密データは自動マスキング |
| SIEM | ログの集約・相関分析・アラート生成を自動化 |
| 検知ルール | Sigma ルールで SIEM 間のポータビリティを確保 |
| 異常検知 | ルールベース + 統計ベース + 機械学習の組み合わせ |
| ログ保存 | Hot/Warm/Cold/Archive の段階的保存でコスト最適化 |
| アラート管理 | Alert Fatigue を防ぎ、High 以上を即座に対応 |
| パフォーマンス | インデックス設計、圧縮、パーティショニングの最適化 |

---

## 次に読むべきガイド

- [インシデント対応](./00-incident-response.md) — 検知後のインシデント対応フロー
- [コンプライアンス](./02-compliance.md) — ログ保存の法的要件
- [AWSセキュリティ](../05-cloud-security/01-aws-security.md) — CloudTrail・GuardDuty の詳細
- [SAST/DAST](../04-application-security/03-sast-dast.md) — アプリケーション脆弱性の検出

---

## 参考文献

1. **NIST SP 800-92 — Guide to Computer Security Log Management** — https://csrc.nist.gov/publications/detail/sp/800-92/final
2. **NIST SP 800-137 — Information Security Continuous Monitoring** — https://csrc.nist.gov/publications/detail/sp/800-137/final
3. **Sigma Rules Repository** — https://github.com/SigmaHQ/sigma
4. **Elastic Common Schema (ECS)** — https://www.elastic.co/guide/en/ecs/current/index.html
5. **OCSF (Open Cybersecurity Schema Framework)** — https://schema.ocsf.io/
6. **Elastic SIEM Documentation** — https://www.elastic.co/guide/en/security/current/index.html
7. **MITRE ATT&CK Framework** — https://attack.mitre.org/ — 攻撃手法の分類体系
8. **Fluent Bit Documentation** — https://docs.fluentbit.io/
9. **Vector Documentation** — https://vector.dev/docs/
10. **AWS Security Logging Best Practices** — https://docs.aws.amazon.com/prescriptive-guidance/latest/logging-monitoring-for-application-owners/
