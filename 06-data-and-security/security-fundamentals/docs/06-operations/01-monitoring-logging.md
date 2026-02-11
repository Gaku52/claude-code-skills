# 監視/ログ

> SIEM によるログの集約・相関分析、効果的なログ収集アーキテクチャ、異常検知の手法まで、セキュリティ監視の基盤を体系的に学ぶ

## この章で学ぶこと

1. **SIEM の仕組みと活用** — ログの集約、相関分析、アラート生成のメカニズム
2. **ログ集約アーキテクチャ** — 大規模環境でのログ収集・保存・検索の設計
3. **異常検知** — ルールベースと機械学習ベースの検知手法

---

## 1. セキュリティ監視の全体像

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

---

## 2. ログ収集の設計

### 収集すべきログ

| ログソース | 内容 | 保存期間 | 優先度 |
|-----------|------|---------|--------|
| CloudTrail | AWS API 呼び出し | 1年以上 | 必須 |
| VPC Flow Logs | ネットワークフロー | 90日 | 必須 |
| ALB/NLB Access Logs | HTTP リクエスト | 90日 | 必須 |
| GuardDuty Findings | 脅威検知結果 | 90日 | 必須 |
| OS syslog/auditd | OS レベルのイベント | 90日 | 高 |
| Application Logs | アプリケーション動作 | 30-90日 | 高 |
| DNS Query Logs | DNS クエリ | 30日 | 中 |
| WAF Logs | WAF 判定結果 | 30日 | 中 |

### 構造化ログのフォーマット

```json
{
  "timestamp": "2025-03-15T14:30:00.000Z",
  "level": "WARN",
  "service": "auth-service",
  "traceId": "abc-123-def",
  "requestId": "req-456",
  "event": "login_failed",
  "userId": "user-789",
  "sourceIp": "203.0.113.50",
  "userAgent": "Mozilla/5.0...",
  "details": {
    "reason": "invalid_password",
    "attemptCount": 5
  }
}
```

### Fluent Bit の設定

```ini
# /etc/fluent-bit/fluent-bit.conf

[SERVICE]
    Flush         5
    Daemon        Off
    Log_Level     info
    Parsers_File  parsers.conf

# アプリケーションログ
[INPUT]
    Name              tail
    Path              /var/log/app/*.log
    Parser            json
    Tag               app.*
    Refresh_Interval  5
    Mem_Buf_Limit     50MB

# OS syslog
[INPUT]
    Name              systemd
    Tag               system.*
    Systemd_Filter    _SYSTEMD_UNIT=sshd.service
    Read_From_Tail    On

# メタデータの追加
[FILTER]
    Name              record_modifier
    Match             *
    Record hostname   ${HOSTNAME}
    Record env        production

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

# S3 にバックアップ
[OUTPUT]
    Name              s3
    Match             *
    bucket            security-logs-archive
    region            ap-northeast-1
    total_file_size   100M
    upload_timeout    10m
    s3_key_format     /logs/%Y/%m/%d/$TAG/%H-%M-%S
```

---

## 3. SIEM

### SIEM ツールの比較

| 項目 | Splunk | Elastic SIEM | Amazon Security Lake | Datadog SIEM |
|------|--------|-------------|---------------------|-------------|
| デプロイ | オンプレ/SaaS | オンプレ/クラウド | SaaS (AWS) | SaaS |
| コスト | 高い (データ量課金) | OSS 版あり | S3 保存量 | 中程度 |
| 相関分析 | 高度 (SPL) | KQL | Athena (SQL) | ログパイプライン |
| 機械学習 | MLTK | ML Jobs | -- | Anomaly Detection |
| カスタムルール | SPL | Detection Rules | Lambda | Detection Rules |
| スケーラビリティ | 高 | 中-高 | 高 (S3 ベース) | 高 |

### 検知ルールの作成 (Sigma ルール)

```yaml
# sigma/rules/credential_access/brute_force_ssh.yml
title: SSH Brute Force Attack
id: a1234567-b890-1234-cdef-567890abcdef
status: stable
description: 同一 IP から短時間に多数の SSH ログイン失敗を検知
author: Security Team
date: 2025/01/15
tags:
  - attack.credential_access
  - attack.t1110.001
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

### Sigma ルールを各 SIEM に変換

```bash
# Sigma CLI で各 SIEM のクエリ形式に変換
pip install sigma-cli

# Splunk SPL に変換
sigma convert -t splunk -p sysmon sigma/rules/

# Elastic EQL に変換
sigma convert -t elasticsearch sigma/rules/

# OpenSearch に変換
sigma convert -t opensearch sigma/rules/

# 変換例 (Splunk SPL):
# source=sshd action="Failed"
# NOT (source_ip="10.0.0.0/8" OR source_ip="172.16.0.0/12")
# | stats count by source_ip
# | where count > 10
```

---

## 4. 異常検知

### ルールベース vs 機械学習ベース

```
ルールベース検知:
  +-- 既知の攻撃パターンに効果的
  +-- 偽陽性の制御が容易
  +-- 新規の攻撃パターンには対応不可
  +-- 例: "5分間にSSHログイン失敗10回以上"

機械学習ベース検知:
  +-- ベースラインからの逸脱を検知
  +-- 未知の攻撃パターンにも対応可能
  +-- 偽陽性のチューニングが必要
  +-- 例: "通常と異なるデータ転送量"
```

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
|                                                          |
|  [データ]                                                 |
|  +-- 大量データの外部転送 (Exfiltration)                  |
|  +-- 通常と異なるDB クエリパターン                         |
|  +-- 機密データへの異常なアクセス頻度                       |
|                                                          |
|  [ネットワーク]                                           |
|  +-- C2 通信パターン (ビーコニング)                        |
|  +-- DNS トンネリング                                     |
|  +-- ポートスキャン                                       |
|  +-- 内部ネットワークのラテラルムーブメント                  |
|                                                          |
|  [システム]                                               |
|  +-- プロセスの異常な動作                                 |
|  +-- ファイルの大量暗号化 (ランサムウェア)                  |
|  +-- 設定変更 (CloudTrail 無効化等)                       |
+----------------------------------------------------------+
```

### CloudWatch Metric Filter + アラーム

```python
import boto3

logs = boto3.client('logs')
cloudwatch = boto3.client('cloudwatch')
sns = boto3.client('sns')

# メトリクスフィルタ: Root アカウントの使用を検知
logs.put_metric_filter(
    logGroupName='/aws/cloudtrail',
    filterName='RootAccountUsage',
    filterPattern='{ $.userIdentity.type = "Root" '
                  '&& $.userIdentity.invokedBy NOT EXISTS '
                  '&& $.eventType != "AwsServiceEvent" }',
    metricTransformations=[{
        'metricName': 'RootAccountUsageCount',
        'metricNamespace': 'SecurityMetrics',
        'metricValue': '1',
    }],
)

# アラーム: Root 使用時に通知
cloudwatch.put_metric_alarm(
    AlarmName='RootAccountUsage',
    MetricName='RootAccountUsageCount',
    Namespace='SecurityMetrics',
    Statistic='Sum',
    Period=300,
    EvaluationPeriods=1,
    Threshold=1,
    ComparisonOperator='GreaterThanOrEqualToThreshold',
    AlarmActions=['arn:aws:sns:ap-northeast-1:123456:security-alerts'],
    TreatMissingData='notBreaching',
)
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
|                                                          |
|  [トレンドグラフ]                                         |
|  +-- アラート推移 (日次/週次)                              |
|  +-- ログインスロットエラーの推移                          |
|  +-- ネットワークトラフィック量の推移                       |
|                                                          |
|  [地理マップ]                                             |
|  +-- ソース IP の地理分布                                 |
|  +-- 異常な接続元の国別表示                               |
|                                                          |
|  [テーブル]                                               |
|  +-- 最新のアラート一覧                                   |
|  +-- トップ攻撃者 IP                                      |
|  +-- 最もアクセスされたエンドポイント                      |
+----------------------------------------------------------+
```

---

## 6. アンチパターン

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
```

---

## 7. FAQ

### Q1. SIEM のログ保存期間はどのくらい必要か?

PCI DSS では 1 年間 (直近 3 ヶ月は即座に検索可能)、SOC 2 では監査期間分 (通常 12 ヶ月) が求められる。コスト最適化のため、Hot (30日/OpenSearch) → Warm (90日/S3+Athena) → Cold (365日+/Glacier) の段階的保存が効果的である。

### Q2. 小規模チームでも SIEM は必要か?

専用の SIEM 製品でなくても、CloudWatch Logs + Athena + EventBridge の組み合わせで基本的な監視は構築できる。まず CloudTrail、VPC Flow Logs、アプリログの収集を開始し、重要なアラートルールを数個設定するところから始めるのが現実的である。

### Q3. 機械学習ベースの異常検知は信頼できるか?

学習期間 (通常 2-4 週間) のベースラインデータの品質に依存する。異常なイベントが学習期間に含まれると誤ったベースラインになる。機械学習はルールベース検知を補完するものであり、置き換えるものではない。まずルールベースの検知を充実させ、その上で機械学習を追加するのが推奨順序である。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ログ収集 | CloudTrail、VPC Flow Logs、アプリログを必ず収集 |
| 構造化ログ | JSON 形式で一貫したフォーマットを使用 |
| SIEM | ログの集約・相関分析・アラート生成を自動化 |
| 検知ルール | Sigma ルールで SIEM 間のポータビリティを確保 |
| 異常検知 | ルールベース + 機械学習の組み合わせ |
| ログ保存 | Hot/Warm/Cold の段階的保存でコスト最適化 |
| アラート管理 | Alert Fatigue を防ぎ、High 以上を即座に対応 |

---

## 次に読むべきガイド

- [インシデント対応](./00-incident-response.md) — 検知後のインシデント対応フロー
- [コンプライアンス](./02-compliance.md) — ログ保存の法的要件
- [AWSセキュリティ](../05-cloud-security/01-aws-security.md) — CloudTrail・GuardDuty の詳細

---

## 参考文献

1. **NIST SP 800-92 — Guide to Computer Security Log Management** — https://csrc.nist.gov/publications/detail/sp/800-92/final
2. **Sigma Rules Repository** — https://github.com/SigmaHQ/sigma
3. **Elastic SIEM Documentation** — https://www.elastic.co/guide/en/security/current/index.html
4. **MITRE ATT&CK Framework** — https://attack.mitre.org/ — 攻撃手法の分類体系
