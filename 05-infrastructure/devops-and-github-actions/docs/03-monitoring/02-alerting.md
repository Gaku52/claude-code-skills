# アラート戦略

> アラート設計の原則、エスカレーションポリシー、ポストモーテムの運用を習得し、アラート疲れのない持続可能なオンコール体制を構築する

## この章で学ぶこと

1. **効果的なアラート設計** — アラート疲れを防ぎ、本当に対応が必要なアラートだけを発報する設計原則
2. **エスカレーションとオンコール体制** — 段階的なエスカレーションポリシーとオンコールローテーションの構築
3. **ポストモーテムと継続的改善** — 障害から学び、再発防止を組織的に推進するプロセス
4. **自動修復 (Auto-remediation)** — 人間の介入なしに障害を自動復旧する仕組みの構築
5. **インシデント管理プロセス** — SEV レベルの定義、コミュニケーション、ステータスページの運用

---

## 1. アラート設計の全体像

```
┌──────────────────────────────────────────────────────────┐
│                  アラート設計のピラミッド                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│                    ┌──────────┐                          │
│                    │ PAGE     │  即座に対応が必要           │
│                    │ (緊急)   │  SLO 違反、サービス停止     │
│                    └────┬─────┘                          │
│                   ┌─────▼──────┐                         │
│                   │  TICKET    │  営業時間内に対応          │
│                   │ (重要)     │  パフォーマンス劣化        │
│                   └─────┬──────┘                         │
│              ┌──────────▼───────────┐                    │
│              │     NOTIFICATION     │  情報として通知       │
│              │     (参考)           │  容量警告、証明書期限  │
│              └──────────┬───────────┘                    │
│         ┌───────────────▼────────────────┐               │
│         │        DASHBOARD ONLY          │  ダッシュボード │
│         │        (記録のみ)               │  で確認可能     │
│         └────────────────────────────────┘               │
│                                                          │
│  原則: 上に行くほど数を絞る。PAGE は月に数回が理想。       │
└──────────────────────────────────────────────────────────┘
```

### 1.1 アラート設計の原則

```
アラートの品質を決める5つの原則:

1. アクショナブル (Actionable)
   ─ アラートを受けたら「今すぐやるべきこと」が明確
   ─ 対応不要なアラートは存在すべきではない

2. コンテキスト付き (Contextual)
   ─ アラートメッセージに十分な情報を含める
   ─ Runbook URL、ダッシュボードリンク、影響範囲

3. SLO ベース (SLO-based)
   ─ 静的な閾値ではなく、SLO 違反の予測に基づく
   ─ バーンレートアラートでビジネスインパクトを反映

4. 適切な粒度 (Right Granularity)
   ─ 症状 (symptom) でアラート、原因 (cause) はデバッグ時
   ─ 「CPU 80%」ではなく「レスポンスタイムがSLO違反」

5. 継続的に改善 (Continuously Improved)
   ─ 月次でアラートを棚卸し
   ─ False positive / False negative を追跡
```

### 1.2 アラート品質メトリクス

```
アラートの品質を定量的に計測する指標:

┌─────────────────────────────────────────────────┐
│ 指標                     │ 目標値              │
├──────────────────────────┼─────────────────────┤
│ PAGE 数/月               │ < 10                │
│ False positive 率        │ < 10%               │
│ MTTA (平均応答時間)       │ < 5分               │
│ MTTR (平均復旧時間)       │ < 30分              │
│ アラート → アクション率   │ > 90%               │
│ Runbook カバレッジ        │ 100%                │
│ 自動修復率               │ > 30%               │
│ オンコール満足度          │ > 4/5               │
└──────────────────────────┴─────────────────────┘

計測方法:
- PagerDuty / Opsgenie のレポート機能
- カスタムダッシュボードでアラート統計を可視化
- 月次のアラートレビューミーティングで振り返り
```

### 1.3 症状ベース vs 原因ベースのアラート

```
症状ベースアラート (推奨):
┌────────────────────────────────────────────────────┐
│ ユーザーが体験する問題に直接紐づく                    │
│                                                    │
│ 例:                                                │
│ ・エラーレートが SLO を超えている                    │
│ ・レスポンスタイムの p99 が 2秒を超えている           │
│ ・注文処理の成功率が 99% を下回っている              │
│                                                    │
│ 利点:                                              │
│ ・ビジネスインパクトが明確                          │
│ ・原因が複合的でも検知できる                        │
│ ・False positive が少ない                          │
└────────────────────────────────────────────────────┘

原因ベースアラート (補助的に使用):
┌────────────────────────────────────────────────────┐
│ インフラの状態に基づく                               │
│                                                    │
│ 例:                                                │
│ ・CPU 使用率 > 90%                                 │
│ ・ディスク使用率 > 85%                              │
│ ・メモリ使用率 > 90%                                │
│                                                    │
│ 注意:                                              │
│ ・症状に影響しない場合がある (CPU 90% でも正常動作)  │
│ ・予防的な TICKET/NOTIFICATION レベルで使用          │
│ ・PAGE には使わない                                 │
└────────────────────────────────────────────────────┘
```

---

## 2. アラートルールの設計

### 2.1 SLO バーンレートアラート

```yaml
# prometheus-alerts.yml — Prometheus アラートルール
groups:
  - name: slo-alerts
    rules:
      # SLO バーンレートアラート (Multi-window, Multi-burn-rate)
      # 99.9% SLO: 30日で許容エラー = 0.1% = 43.2分のダウンタイム
      - alert: HighErrorBurnRate_Critical
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m]))
            / sum(rate(http_requests_total[5m]))
          ) > (14.4 * 0.001)
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[1h]))
            / sum(rate(http_requests_total[1h]))
          ) > (14.4 * 0.001)
        for: 2m
        labels:
          severity: critical
          team: backend
          slo: availability
        annotations:
          summary: "エラーバーンレートが危険水準 (Critical)"
          description: >
            直近5分と1時間のエラー率がSLOバーンレートの14.4倍を超えています。
            このペースでは2日以内に月間エラーバジェットを使い切ります。
          runbook: "https://wiki.example.com/runbooks/high-error-rate"
          dashboard: "https://grafana.example.com/d/slo-overview"
          impact: "全ユーザーに影響の可能性"

      - alert: HighErrorBurnRate_Warning
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[30m]))
            / sum(rate(http_requests_total[30m]))
          ) > (6 * 0.001)
          and
          (
            sum(rate(http_requests_total{status=~"5.."}[6h]))
            / sum(rate(http_requests_total[6h]))
          ) > (6 * 0.001)
        for: 5m
        labels:
          severity: warning
          team: backend
          slo: availability
        annotations:
          summary: "エラーバーンレートが警告水準 (Warning)"
          description: >
            直近30分と6時間のエラー率がSLOバーンレートの6倍を超えています。
          runbook: "https://wiki.example.com/runbooks/high-error-rate"

      # レイテンシ SLO (p99 < 500ms を 99.5% のリクエストで達成)
      - alert: HighLatencyBurnRate_Critical
        expr: |
          (
            1 - (
              sum(rate(http_request_duration_seconds_bucket{le="0.5"}[5m]))
              / sum(rate(http_request_duration_seconds_count[5m]))
            )
          ) > (14.4 * 0.005)
          and
          (
            1 - (
              sum(rate(http_request_duration_seconds_bucket{le="0.5"}[1h]))
              / sum(rate(http_request_duration_seconds_count[1h]))
            )
          ) > (14.4 * 0.005)
        for: 2m
        labels:
          severity: critical
          team: backend
          slo: latency
        annotations:
          summary: "レイテンシバーンレートが危険水準"
          description: >
            p99レイテンシが500msを超えるリクエストの割合が急増しています。
          runbook: "https://wiki.example.com/runbooks/high-latency"
```

### 2.2 バーンレート計算の詳細

```
バーンレートアラートの設計理論:

SLO: 99.9% (30日間)
エラーバジェット: 0.1% = 30日 × 24時間 × 60分 × 0.001 = 43.2分

バーンレート = 実際のエラー率 / 許容エラー率

  バーンレート × ウィンドウ → バジェット消費
  ┌─────────┬──────────┬────────────────────────┐
  │ バーン   │ ウィンドウ │ バジェット全消費までの    │
  │ レート   │          │ 時間                    │
  ├─────────┼──────────┼────────────────────────┤
  │ 14.4x   │ 5m + 1h  │ 2日 (PAGE)             │
  │ 6x      │ 30m + 6h │ 5日 (TICKET)           │
  │ 3x      │ 2h + 1d  │ 10日 (NOTIFICATION)    │
  │ 1x      │ 6h + 3d  │ 30日 (ちょうどSLO消費)  │
  └─────────┴──────────┴────────────────────────┘

  Multi-window の意味:
  ┌───────────────────────────────────────────────┐
  │ 短いウィンドウ (5m)                            │
  │   → 瞬間的なスパイクを検知                     │
  │   → 単独だと false positive が多い             │
  │                                               │
  │ 長いウィンドウ (1h)                            │
  │   → 持続的な問題を確認                         │
  │   → 単独だと検知が遅い                         │
  │                                               │
  │ 両方の AND 条件                                │
  │   → 瞬間スパイクを除外しつつ                   │
  │     持続的な問題を素早く検知                    │
  └───────────────────────────────────────────────┘
```

### 2.3 インフラストラクチャアラート

```yaml
  - name: infrastructure-alerts
    rules:
      - alert: HighMemoryUsage
        expr: |
          (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) > 0.9
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "メモリ使用率が90%を超過"
          description: "{{ $labels.instance }} のメモリ使用率が {{ $value | humanizePercentage }} です"
          runbook: "https://wiki.example.com/runbooks/high-memory"

      - alert: DiskSpaceRunningOut
        expr: |
          predict_linear(node_filesystem_avail_bytes{mountpoint="/"}[6h], 24*3600) < 0
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "24時間以内にディスク容量が枯渇する予測"
          description: "{{ $labels.instance }} のディスクが24時間以内に満杯になります"
          runbook: "https://wiki.example.com/runbooks/disk-space"

      - alert: DiskSpaceCritical
        expr: |
          (1 - (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})) > 0.95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "ディスク使用率が95%を超過"
          description: "{{ $labels.instance }} のディスク使用率が {{ $value | humanizePercentage }} です。即時対応が必要です。"
          runbook: "https://wiki.example.com/runbooks/disk-space-critical"

      - alert: HighCPUUsage
        expr: |
          100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 90
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "CPU使用率が90%を超過 (15分間持続)"
          description: "{{ $labels.instance }} のCPU使用率が {{ $value }}% です"

      # SSL 証明書の有効期限チェック
      - alert: SSLCertExpiringSoon
        expr: |
          (probe_ssl_earliest_cert_expiry - time()) / 86400 < 30
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "SSL証明書の有効期限が30日以内"
          description: "{{ $labels.instance }} の証明書が {{ $value | humanize }}日後に期限切れ"

      - alert: SSLCertExpiringSoon_Critical
        expr: |
          (probe_ssl_earliest_cert_expiry - time()) / 86400 < 7
        for: 1h
        labels:
          severity: critical
        annotations:
          summary: "SSL証明書の有効期限が7日以内"
          description: "{{ $labels.instance }} の証明書が {{ $value | humanize }}日後に期限切れです。即時更新してください。"

  - name: kubernetes-alerts
    rules:
      # Pod が CrashLoopBackOff
      - alert: PodCrashLooping
        expr: |
          rate(kube_pod_container_status_restarts_total[15m]) * 60 * 15 > 3
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Pod が CrashLoopBackOff 状態"
          description: >
            {{ $labels.namespace }}/{{ $labels.pod }} が15分間に3回以上再起動しています。
          runbook: "https://wiki.example.com/runbooks/pod-crashloop"

      # Deployment のレプリカ不足
      - alert: DeploymentReplicasMismatch
        expr: |
          kube_deployment_spec_replicas != kube_deployment_status_available_replicas
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Deployment のレプリカ数が不一致"
          description: >
            {{ $labels.namespace }}/{{ $labels.deployment }} の
            期待レプリカ数: {{ $labels.spec_replicas }},
            実際のレプリカ数: {{ $value }}

      # Node の NotReady 状態
      - alert: KubeNodeNotReady
        expr: |
          kube_node_status_condition{condition="Ready", status="true"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kubernetes Node が NotReady"
          description: "{{ $labels.node }} が5分以上 NotReady 状態です"

      # PersistentVolume の容量
      - alert: PersistentVolumeRunningOut
        expr: |
          kubelet_volume_stats_available_bytes / kubelet_volume_stats_capacity_bytes < 0.15
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "PersistentVolume の空き容量が15%未満"
          description: "{{ $labels.namespace }}/{{ $labels.persistentvolumeclaim }} の残り容量: {{ $value | humanizePercentage }}"

      # HPA のスケーリング限界
      - alert: HPAMaxedOut
        expr: |
          kube_horizontalpodautoscaler_status_current_replicas == kube_horizontalpodautoscaler_spec_max_replicas
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "HPA が最大レプリカ数に達しています"
          description: >
            {{ $labels.namespace }}/{{ $labels.horizontalpodautoscaler }} が
            最大レプリカ数 ({{ $value }}) で15分以上動作しています。
            スケーリング上限の引き上げを検討してください。
```

### 2.4 ビジネスアラート

```yaml
  - name: business-alerts
    rules:
      # 注文数の急激な減少
      - alert: OrderRateDropped
        expr: |
          sum(rate(orders_created_total[30m]))
          < sum(rate(orders_created_total[30m] offset 1d)) * 0.5
        for: 15m
        labels:
          severity: critical
          team: business
        annotations:
          summary: "注文数が前日比50%以下に急減"
          description: >
            直近30分の注文レートが前日同時刻と比較して50%以下に低下しています。
            決済システム障害またはフロントエンド障害の可能性があります。
          runbook: "https://wiki.example.com/runbooks/order-rate-drop"

      # 決済成功率の低下
      - alert: PaymentSuccessRateLow
        expr: |
          sum(rate(payment_transactions_total{status="success"}[10m]))
          / sum(rate(payment_transactions_total[10m]))
          < 0.95
        for: 5m
        labels:
          severity: critical
          team: payments
        annotations:
          summary: "決済成功率が95%を下回っています"
          description: >
            直近10分の決済成功率: {{ $value | humanizePercentage }}
            決済プロバイダの障害またはアプリケーションバグの可能性があります。

      # ユーザー登録の異常
      - alert: UserRegistrationAnomaly
        expr: |
          abs(
            sum(rate(user_registrations_total[1h]))
            - sum(rate(user_registrations_total[1h] offset 7d))
          ) / sum(rate(user_registrations_total[1h] offset 7d))
          > 2
        for: 30m
        labels:
          severity: warning
          team: growth
        annotations:
          summary: "ユーザー登録数に異常検知"
          description: "前週同時刻と比較して200%以上の変動があります"

      # キューの滞留
      - alert: QueueBacklogHigh
        expr: |
          sum(queue_messages_pending) by (queue_name) > 10000
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "メッセージキューの滞留が10,000件を超過"
          description: >
            キュー {{ $labels.queue_name }} に {{ $value }} 件のメッセージが滞留しています。
            コンシューマのスケールアウトまたは処理遅延の調査が必要です。
```

---

## 3. Alertmanager 設定

### 3.1 ルーティングと通知設定

```yaml
# alertmanager.yml — エスカレーション設定
global:
  resolve_timeout: 5m
  slack_api_url: "https://hooks.slack.com/services/XXX/YYY/ZZZ"
  pagerduty_url: "https://events.pagerduty.com/v2/enqueue"

# ルーティングツリー
route:
  receiver: default-notification
  group_by: ['alertname', 'team', 'service']
  group_wait: 30s       # 同じグループのアラートを30秒待って集約
  group_interval: 5m    # 同じグループの再通知間隔
  repeat_interval: 4h   # 同じアラートの繰り返し通知間隔

  routes:
    # Critical → PagerDuty + Slack
    - match:
        severity: critical
      receiver: pagerduty-critical
      continue: true  # 次のルートも評価
    - match:
        severity: critical
      receiver: slack-critical
      group_wait: 0s  # 即座に通知

    # Warning → Slack のみ
    - match:
        severity: warning
      receiver: slack-warning
      repeat_interval: 12h

    # チーム別ルーティング
    - match:
        team: backend
      receiver: slack-backend
    - match:
        team: frontend
      receiver: slack-frontend
    - match:
        team: payments
      receiver: slack-payments
      routes:
        # 決済チームの Critical は専用 PagerDuty へ
        - match:
            severity: critical
          receiver: pagerduty-payments

    # ビジネスアラート → ビジネスチャンネル
    - match:
        team: business
      receiver: slack-business
      group_wait: 5m

# 通知先の定義
receivers:
  - name: default-notification
    slack_configs:
      - channel: '#alerts-general'
        send_resolved: true
        title: '{{ .CommonLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: pagerduty-critical
    pagerduty_configs:
      - routing_key: "YOUR_PAGERDUTY_ROUTING_KEY"
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'
        details:
          alert: '{{ .CommonLabels.alertname }}'
          description: '{{ .CommonAnnotations.description }}'
          runbook: '{{ .CommonAnnotations.runbook }}'
          dashboard: '{{ .CommonAnnotations.dashboard }}'
          num_firing: '{{ .Alerts.Firing | len }}'
        links:
          - href: '{{ .CommonAnnotations.runbook }}'
            text: 'Runbook'
          - href: '{{ .CommonAnnotations.dashboard }}'
            text: 'Dashboard'

  - name: pagerduty-payments
    pagerduty_configs:
      - routing_key: "PAYMENTS_TEAM_ROUTING_KEY"
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'

  - name: slack-critical
    slack_configs:
      - channel: '#alerts-critical'
        color: 'danger'
        title: '{{ .Status | toUpper }}: {{ .CommonLabels.alertname }}'
        text: >-
          *Summary:* {{ .CommonAnnotations.summary }}
          *Description:* {{ .CommonAnnotations.description }}
          *Severity:* {{ .CommonLabels.severity }}
          *Service:* {{ .CommonLabels.service }}
        actions:
          - type: button
            text: 'Runbook'
            url: '{{ .CommonAnnotations.runbook }}'
          - type: button
            text: 'Dashboard'
            url: '{{ .CommonAnnotations.dashboard }}'
          - type: button
            text: 'Silence'
            url: '{{ template "__alertmanagerURL" . }}/#/silences/new?filter=%7Balertname%3D%22{{ .CommonLabels.alertname }}%22%7D'
        send_resolved: true

  - name: slack-warning
    slack_configs:
      - channel: '#alerts-warning'
        color: 'warning'
        title: '{{ .CommonLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
        send_resolved: true

  - name: slack-backend
    slack_configs:
      - channel: '#team-backend-alerts'
        send_resolved: true

  - name: slack-frontend
    slack_configs:
      - channel: '#team-frontend-alerts'
        send_resolved: true

  - name: slack-payments
    slack_configs:
      - channel: '#team-payments-alerts'
        send_resolved: true

  - name: slack-business
    slack_configs:
      - channel: '#business-alerts'
        send_resolved: true

# 抑制ルール
inhibit_rules:
  # Critical 発報中は Warning を抑制
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['alertname', 'team']

  # サービス全体が停止中は個別エンドポイントのアラートを抑制
  - source_match:
      alertname: ServiceDown
    target_match_re:
      alertname: 'High.*BurnRate.*'
    equal: ['service']

  # Node が NotReady なら、その Node 上の Pod アラートを抑制
  - source_match:
      alertname: KubeNodeNotReady
    target_match_re:
      alertname: 'Pod.*'
    equal: ['node']

# テンプレート
templates:
  - '/etc/alertmanager/templates/*.tmpl'
```

### 3.2 Alertmanager テンプレートのカスタマイズ

```go
{{/* /etc/alertmanager/templates/slack.tmpl */}}
{{ define "slack.custom.title" }}
{{ if eq .Status "firing" }}🔥{{ else }}✅{{ end }} [{{ .Status | toUpper }}] {{ .CommonLabels.alertname }}
{{ end }}

{{ define "slack.custom.text" }}
{{ range .Alerts }}
*Alert:* {{ .Labels.alertname }}
*Severity:* {{ .Labels.severity }}
*Service:* {{ .Labels.service | default "unknown" }}

*Summary:* {{ .Annotations.summary }}
*Description:* {{ .Annotations.description }}

{{ if .Annotations.runbook }}📖 <{{ .Annotations.runbook }}|Runbook>{{ end }}
{{ if .Annotations.dashboard }}📊 <{{ .Annotations.dashboard }}|Dashboard>{{ end }}

*Started:* {{ .StartsAt.Format "2006-01-02 15:04:05 JST" }}
{{ if .EndsAt }}*Ended:* {{ .EndsAt.Format "2006-01-02 15:04:05 JST" }}{{ end }}
---
{{ end }}
{{ end }}

{{ define "slack.custom.footer" }}
Alertmanager | {{ .ExternalURL }}
{{ end }}
```

### 3.3 Silence (一時的なアラート抑制) の管理

```bash
#!/bin/bash
# silence-management.sh — Alertmanager Silence の管理スクリプト

ALERTMANAGER_URL="http://alertmanager:9093"

# Silence の作成 (メンテナンスウィンドウ用)
create_maintenance_silence() {
  local duration="${1:-2h}"
  local comment="${2:-Scheduled maintenance}"
  local creator="${3:-sre-team}"

  curl -X POST "${ALERTMANAGER_URL}/api/v2/silences" \
    -H "Content-Type: application/json" \
    -d "{
      \"matchers\": [
        {
          \"name\": \"severity\",
          \"value\": \"warning\",
          \"isRegex\": false,
          \"isEqual\": true
        }
      ],
      \"startsAt\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",
      \"endsAt\": \"$(date -u -d \"+${duration}\" +%Y-%m-%dT%H:%M:%S.000Z)\",
      \"createdBy\": \"${creator}\",
      \"comment\": \"${comment}\"
    }"
}

# 特定のアラートを Silence
silence_alert() {
  local alertname="$1"
  local duration="${2:-1h}"
  local reason="$3"

  curl -X POST "${ALERTMANAGER_URL}/api/v2/silences" \
    -H "Content-Type: application/json" \
    -d "{
      \"matchers\": [
        {
          \"name\": \"alertname\",
          \"value\": \"${alertname}\",
          \"isRegex\": false,
          \"isEqual\": true
        }
      ],
      \"startsAt\": \"$(date -u +%Y-%m-%dT%H:%M:%S.000Z)\",
      \"endsAt\": \"$(date -u -d \"+${duration}\" +%Y-%m-%dT%H:%M:%S.000Z)\",
      \"createdBy\": \"$(whoami)\",
      \"comment\": \"${reason}\"
    }"
}

# アクティブな Silence の一覧
list_silences() {
  curl -s "${ALERTMANAGER_URL}/api/v2/silences" | \
    jq '.[] | select(.status.state == "active") | {id: .id, createdBy: .createdBy, comment: .comment, endsAt: .endsAt}'
}

# Silence の削除
expire_silence() {
  local silence_id="$1"
  curl -X DELETE "${ALERTMANAGER_URL}/api/v2/silence/${silence_id}"
}
```

---

## 4. エスカレーションフローとオンコール

### 4.1 エスカレーションフロー詳細

```
エスカレーションフロー:

  アラート発報
      │
      ▼
  ┌──────────────┐
  │ 自動対応可能？ │──── Yes ──► 自動修復 (Auto-remediation)
  └──────┬───────┘              例: Pod 再起動、スケールアウト
         │ No
         ▼
  ┌──────────────┐
  │  Severity?   │
  └──────┬───────┘
         │
    ┌────┼────────────────┐
    ▼    ▼                ▼
  Critical  Warning     Info
    │       │             │
    ▼       ▼             ▼
  PagerDuty  Slack       Slack
  + Slack    チャンネル   チャンネル
    │        (営業時間)   (記録のみ)
    ▼
  5分以内に
  応答なし？
    │
    ▼
  2次オンコール
  へエスカレート
    │
    ▼
  15分以内に
  応答なし？
    │
    ▼
  マネージャー
  へエスカレート
    │
    ▼
  30分以内に
  応答なし？
    │
    ▼
  VP/CTO
  へエスカレート
```

### 4.2 インシデント重大度 (SEV) の定義

```
インシデント SEV レベル:

┌──────┬────────────────┬──────────────┬──────────────┬──────────────┐
│ SEV  │ 定義           │ 例            │ 対応時間     │ 通知先        │
├──────┼────────────────┼──────────────┼──────────────┼──────────────┤
│ SEV-1│ サービス全停止  │ 全API 500    │ 即時         │ 全エンジニア  │
│      │ 全ユーザー影響  │ データ消失    │ (24/7)       │ + 経営層      │
├──────┼────────────────┼──────────────┼──────────────┼──────────────┤
│ SEV-2│ 主要機能停止   │ 決済不可     │ 15分以内     │ オンコール    │
│      │ 一部ユーザー   │ 検索不可      │ (24/7)       │ + チームリード│
├──────┼────────────────┼──────────────┼──────────────┼──────────────┤
│ SEV-3│ 機能劣化       │ レスポンス遅延│ 営業時間内   │ チーム        │
│      │ 回避策あり     │ UI バグ       │ 4時間以内    │              │
├──────┼────────────────┼──────────────┼──────────────┼──────────────┤
│ SEV-4│ 軽微な問題     │ ログ警告     │ 営業時間内   │ チケット作成  │
│      │ ユーザー影響なし│ 非本番環境    │ 次スプリント │              │
└──────┴────────────────┴──────────────┴──────────────┴──────────────┘
```

### 4.3 PagerDuty / Opsgenie 連携

```typescript
// oncall-rotation.ts — オンコールローテーション設計
interface OncallSchedule {
  team: string;
  rotationType: 'weekly' | 'daily';
  members: string[];
  escalationPolicy: EscalationLevel[];
  overrides: Override[];
  handoffTime: string; // "09:00" (月曜朝に交代)
}

interface EscalationLevel {
  level: number;
  targets: string[];
  timeout: number;  // 分
  notificationChannels: ('phone' | 'sms' | 'push' | 'email')[];
}

interface Override {
  user: string;
  startDate: string;
  endDate: string;
  reason: string;
}

const backendOncall: OncallSchedule = {
  team: 'backend',
  rotationType: 'weekly',
  handoffTime: '09:00',
  members: [
    'engineer-a@example.com',
    'engineer-b@example.com',
    'engineer-c@example.com',
    'engineer-d@example.com',
    'engineer-e@example.com', // 最低5名のローテーション
  ],
  escalationPolicy: [
    {
      level: 1,
      targets: ['current-oncall'],
      timeout: 5,
      notificationChannels: ['push', 'phone'],
    },
    {
      level: 2,
      targets: ['secondary-oncall'],
      timeout: 10,
      notificationChannels: ['push', 'phone', 'sms'],
    },
    {
      level: 3,
      targets: ['engineering-manager'],
      timeout: 15,
      notificationChannels: ['phone', 'sms'],
    },
    {
      level: 4,
      targets: ['vp-engineering'],
      timeout: 30,
      notificationChannels: ['phone'],
    },
  ],
  overrides: [],
};
```

### 4.4 PagerDuty Terraform 設定

```hcl
# pagerduty.tf — PagerDuty リソース管理
terraform {
  required_providers {
    pagerduty = {
      source  = "PagerDuty/pagerduty"
      version = "~> 3.0"
    }
  }
}

provider "pagerduty" {
  token = var.pagerduty_token
}

# チーム
resource "pagerduty_team" "backend" {
  name        = "Backend Team"
  description = "Backend engineering team"
}

# ユーザー
resource "pagerduty_user" "engineers" {
  for_each = toset([
    "engineer-a@example.com",
    "engineer-b@example.com",
    "engineer-c@example.com",
    "engineer-d@example.com",
    "engineer-e@example.com",
  ])

  name  = each.key
  email = each.key
  role  = "user"
}

# オンコールスケジュール
resource "pagerduty_schedule" "backend_primary" {
  name      = "Backend Primary On-Call"
  time_zone = "Asia/Tokyo"

  layer {
    name                         = "Primary"
    start                        = "2025-01-06T09:00:00+09:00"
    rotation_virtual_start       = "2025-01-06T09:00:00+09:00"
    rotation_turn_length_seconds = 604800  # 1週間

    users = [for u in pagerduty_user.engineers : u.id]
  }
}

resource "pagerduty_schedule" "backend_secondary" {
  name      = "Backend Secondary On-Call"
  time_zone = "Asia/Tokyo"

  layer {
    name                         = "Secondary"
    start                        = "2025-01-13T09:00:00+09:00"
    rotation_virtual_start       = "2025-01-13T09:00:00+09:00"
    rotation_turn_length_seconds = 604800

    users = [for u in pagerduty_user.engineers : u.id]
  }
}

# エスカレーションポリシー
resource "pagerduty_escalation_policy" "backend" {
  name      = "Backend Escalation Policy"
  num_loops = 2  # 全レベル2回繰り返し

  rule {
    escalation_delay_in_minutes = 5
    target {
      type = "schedule_reference"
      id   = pagerduty_schedule.backend_primary.id
    }
  }

  rule {
    escalation_delay_in_minutes = 10
    target {
      type = "schedule_reference"
      id   = pagerduty_schedule.backend_secondary.id
    }
  }

  rule {
    escalation_delay_in_minutes = 15
    target {
      type = "user_reference"
      id   = pagerduty_user.engineers["engineer-a@example.com"].id  # EM
    }
  }
}

# サービス
resource "pagerduty_service" "order_service" {
  name                    = "Order Service"
  description             = "Order processing service"
  escalation_policy       = pagerduty_escalation_policy.backend.id
  auto_resolve_timeout    = 14400  # 4時間で自動解決
  acknowledgement_timeout = 600    # 10分でエスカレート

  alert_creation = "create_alerts_and_incidents"

  incident_urgency_rule {
    type    = "constant"
    urgency = "high"
  }

  auto_pause_notifications_parameters {
    enabled = true
    timeout = 300  # 5分間のフラッピング防止
  }
}

# サービス統合 (Prometheus Alertmanager → PagerDuty)
resource "pagerduty_service_integration" "prometheus" {
  name    = "Prometheus Alertmanager"
  service = pagerduty_service.order_service.id
  vendor  = data.pagerduty_vendor.prometheus.id
}

data "pagerduty_vendor" "prometheus" {
  name = "Prometheus"
}
```

---

## 5. 自動修復 (Auto-remediation)

### 5.1 Kubernetes CronJob による自動修復

```yaml
# auto-remediation.yml — 自動修復ジョブ
apiVersion: batch/v1
kind: CronJob
metadata:
  name: cleanup-stuck-pods
  namespace: monitoring
spec:
  schedule: "*/5 * * * *"  # 5分ごと
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: auto-remediation
          containers:
            - name: remediation
              image: bitnami/kubectl:latest
              command:
                - /bin/bash
                - -c
                - |
                  # CrashLoopBackOff の Pod を検出して削除
                  kubectl get pods --all-namespaces -o json | \
                    jq -r '.items[] |
                      select(.status.containerStatuses[]?.state.waiting.reason == "CrashLoopBackOff") |
                      select(.status.containerStatuses[]?.restartCount > 10) |
                      "\(.metadata.namespace) \(.metadata.name)"' | \
                  while read namespace pod; do
                    echo "Deleting CrashLoopBackOff pod: ${namespace}/${pod}"
                    kubectl delete pod -n "${namespace}" "${pod}" --grace-period=0
                  done

                  # Evicted Pod のクリーンアップ
                  kubectl get pods --all-namespaces --field-selector=status.phase=Failed -o json | \
                    jq -r '.items[] | select(.status.reason == "Evicted") | "\(.metadata.namespace) \(.metadata.name)"' | \
                  while read namespace pod; do
                    echo "Cleaning up evicted pod: ${namespace}/${pod}"
                    kubectl delete pod -n "${namespace}" "${pod}"
                  done
          restartPolicy: OnFailure
```

### 5.2 Alertmanager Webhook による自動修復

```typescript
// auto-remediation-webhook.ts — アラートに基づく自動修復
import express from 'express';
import { KubernetesObjectApi, KubeConfig } from '@kubernetes/client-node';

const app = express();
app.use(express.json());

interface AlertmanagerWebhook {
  status: 'firing' | 'resolved';
  alerts: {
    status: string;
    labels: Record<string, string>;
    annotations: Record<string, string>;
  }[];
}

// 自動修復アクションの定義
const remediationActions: Record<string, (alert: any) => Promise<void>> = {
  // Pod の再起動
  PodCrashLooping: async (alert) => {
    const namespace = alert.labels.namespace;
    const pod = alert.labels.pod;
    console.log(`Auto-remediation: Restarting pod ${namespace}/${pod}`);

    const kc = new KubeConfig();
    kc.loadFromDefault();
    const k8sApi = kc.makeApiClient(KubernetesObjectApi);

    await k8sApi.delete({
      apiVersion: 'v1',
      kind: 'Pod',
      metadata: { name: pod, namespace: namespace },
    });

    // Slack 通知
    await notifySlack(`自動修復: Pod ${namespace}/${pod} を再起動しました`);
  },

  // HPA のスケールアウト
  HPAMaxedOut: async (alert) => {
    const namespace = alert.labels.namespace;
    const hpaName = alert.labels.horizontalpodautoscaler;
    console.log(`Auto-remediation: Scaling up HPA ${namespace}/${hpaName}`);

    const kc = new KubeConfig();
    kc.loadFromDefault();
    const k8sApi = kc.makeApiClient(KubernetesObjectApi);

    // 最大レプリカ数を 50% 増加
    const hpa = await k8sApi.read({
      apiVersion: 'autoscaling/v2',
      kind: 'HorizontalPodAutoscaler',
      metadata: { name: hpaName, namespace: namespace },
    });

    const currentMax = (hpa.body as any).spec.maxReplicas;
    const newMax = Math.ceil(currentMax * 1.5);

    await k8sApi.patch({
      apiVersion: 'autoscaling/v2',
      kind: 'HorizontalPodAutoscaler',
      metadata: { name: hpaName, namespace: namespace },
    }, [
      { op: 'replace', path: '/spec/maxReplicas', value: newMax },
    ]);

    await notifySlack(
      `自動修復: HPA ${namespace}/${hpaName} の最大レプリカ数を ${currentMax} → ${newMax} に変更しました`
    );
  },

  // ディスク容量のクリーンアップ
  DiskSpaceCritical: async (alert) => {
    const instance = alert.labels.instance;
    console.log(`Auto-remediation: Cleaning disk on ${instance}`);

    // SSH 経由でクリーンアップ (実際は Ansible / SSM 経由)
    // 古いログファイルの削除
    // Docker の未使用イメージの削除
    await notifySlack(
      `自動修復: ${instance} のディスククリーンアップを実行しました`
    );
  },
};

// Webhook エンドポイント
app.post('/webhook/alertmanager', async (req, res) => {
  const payload: AlertmanagerWebhook = req.body;

  for (const alert of payload.alerts) {
    if (alert.status !== 'firing') continue;

    const alertName = alert.labels.alertname;
    const action = remediationActions[alertName];

    if (action) {
      try {
        await action(alert);
        console.log(`Remediation succeeded for ${alertName}`);
      } catch (error) {
        console.error(`Remediation failed for ${alertName}:`, error);
        await notifySlack(
          `自動修復失敗: ${alertName} — ${(error as Error).message}`
        );
      }
    }
  }

  res.status(200).send('ok');
});

async function notifySlack(message: string): Promise<void> {
  await fetch(process.env.SLACK_WEBHOOK_URL!, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      channel: '#auto-remediation',
      text: message,
    }),
  });
}

app.listen(8080, () => {
  console.log('Auto-remediation webhook listening on :8080');
});
```

---

## 6. ポストモーテムテンプレート

### 6.1 ポストモーテムドキュメント

```markdown
<!-- postmortem-template.md -->
# ポストモーテム: [インシデントタイトル]

## 概要
- **日時**: 2025-03-15 14:30 〜 15:45 JST (75分間)
- **影響範囲**: 全ユーザーの決済処理が不可
- **影響度**: SEV-1 (サービス全体の重大障害)
- **検知方法**: SLO バーンレートアラート (自動検知)
- **対応者**: @engineer-a (1次), @engineer-b (支援)
- **インシデントコマンダー**: @engineer-a

## タイムライン
| 時刻 | イベント |
|------|---------|
| 14:30 | デプロイ完了 (v2.5.0) |
| 14:32 | エラーレート急上昇、アラート発報 |
| 14:35 | オンコール担当がアクノレッジ |
| 14:38 | Slack の #incident-20250315 チャンネル作成 |
| 14:40 | 決済 API の 500 エラーを確認 |
| 14:45 | ステータスページを「Degraded」に更新 |
| 14:50 | 原因特定: DB マイグレーションで決済テーブルのカラム名変更 |
| 15:00 | ロールバック開始 (v2.4.3 へ) |
| 15:15 | ロールバック完了、エラーレート正常化 |
| 15:30 | ステータスページを「Operational」に更新 |
| 15:45 | 全メトリクス正常を確認、インシデントクローズ |

## 根本原因
DB マイグレーションで `payment_status` カラムを `status` にリネームしたが、
旧バージョンのコードが `payment_status` を参照していた。
Rolling Update 中に新旧バージョンが混在し、旧コードがカラム未検出エラーを起こした。

## 影響の定量化
- 影響を受けたユーザー数: 約 3,200 名
- 失敗したトランザクション: 847 件
- 推定売上損失: ¥4,235,000
- SLO エラーバジェット消費: 月間バジェットの 12.3%

## 5 Whys 分析
1. **Why** 決済が失敗した？ → カラム名の不一致
2. **Why** カラム名が不一致？ → 破壊的変更のマイグレーション
3. **Why** 破壊的変更が本番適用された？ → Expand-Contract パターン未使用
4. **Why** パターン未使用？ → DBマイグレーションガイドラインがない
5. **Why** ガイドラインがない？ → マイグレーション戦略の文書化が未実施

## 再発防止策
| アクション | 担当 | 期限 | 優先度 | ステータス |
|-----------|------|------|--------|-----------|
| DBマイグレーションのExpand-Contractパターン必須化 | @engineer-a | 2025-03-22 | P0 | 完了 |
| デプロイ後の自動 Smoke Test を追加 | @engineer-b | 2025-03-29 | P1 | 進行中 |
| ロールバック手順を自動化 | @engineer-c | 2025-04-05 | P1 | 未着手 |
| DBマイグレーションガイドラインの文書化 | @engineer-a | 2025-04-12 | P1 | 未着手 |
| マイグレーションの自動互換性チェック (CI) | @engineer-d | 2025-04-19 | P2 | 未着手 |

## 教訓
- 破壊的な DB 変更は Expand-Contract パターンで段階的に行う
- デプロイ直後のメトリクス監視期間を設ける (最低10分)
- ロールバック判断の基準を事前に定義しておく
- ステータスページの更新を迅速に行い、ユーザーへの影響を透明化する

## うまくいったこと
- SLO バーンレートアラートが2分以内に検知した
- オンコール担当が3分以内にアクノレッジした
- ロールバック自体は15分で完了した (手順が整備されていた)
```

### 6.2 ポストモーテムプロセス

```
ポストモーテムの運用プロセス:

  インシデント発生
      │
      ▼
  ┌──────────────────┐
  │ インシデント対応   │  ← 今は対応に集中。記録は後で。
  │ (復旧最優先)      │
  └──────┬───────────┘
         │ 復旧完了
         ▼
  ┌──────────────────┐
  │ 48時間以内に       │
  │ ポストモーテム作成 │  ← 記憶が新しいうちに
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │ ポストモーテム     │  ← 関係者全員参加
  │ レビューミーティング│     blame-free の原則
  └──────┬───────────┘     30-60分
         │
         ▼
  ┌──────────────────┐
  │ アクションアイテム │  ← Jira / Linear チケット化
  │ のトラッキング     │     期限と担当者を明確に
  └──────┬───────────┘
         │
         ▼
  ┌──────────────────┐
  │ 月次アラート       │  ← アクション完了の確認
  │ レビュー           │     新たなアラートの評価
  └──────────────────┘
```

### 6.3 ポストモーテムの GitHub Issue 自動生成

```yaml
# .github/workflows/create-postmortem.yml
name: Create Postmortem Issue

on:
  workflow_dispatch:
    inputs:
      incident_title:
        description: 'インシデントのタイトル'
        required: true
      severity:
        description: 'SEV レベル'
        required: true
        type: choice
        options: ['SEV-1', 'SEV-2', 'SEV-3']
      start_time:
        description: '開始時刻 (JST)'
        required: true
      end_time:
        description: '終了時刻 (JST)'
        required: true
      incident_commander:
        description: 'インシデントコマンダー (GitHub username)'
        required: true

jobs:
  create-issue:
    runs-on: ubuntu-latest
    steps:
      - name: Create Postmortem Issue
        uses: actions/github-script@v7
        with:
          script: |
            const body = `# ポストモーテム: ${{ inputs.incident_title }}

            ## 概要
            - **日時**: ${{ inputs.start_time }} 〜 ${{ inputs.end_time }} JST
            - **影響度**: ${{ inputs.severity }}
            - **インシデントコマンダー**: @${{ inputs.incident_commander }}

            ## タイムライン
            | 時刻 | イベント |
            |------|---------|
            | | |

            ## 根本原因
            _（記入してください）_

            ## 影響の定量化
            - 影響を受けたユーザー数:
            - 失敗したリクエスト数:
            - SLO エラーバジェット消費:

            ## 5 Whys 分析
            1. **Why**
            2. **Why**
            3. **Why**
            4. **Why**
            5. **Why**

            ## 再発防止策
            | アクション | 担当 | 期限 | 優先度 |
            |-----------|------|------|--------|
            | | | | |

            ## 教訓
            -

            ## うまくいったこと
            -

            ---
            **期限: ${new Date(Date.now() + 48 * 60 * 60 * 1000).toISOString().split('T')[0]} (48時間以内に完成)**
            `;

            const issue = await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `[Postmortem] ${{ inputs.incident_title }}`,
              body: body,
              labels: ['postmortem', '${{ inputs.severity }}'],
              assignees: ['${{ inputs.incident_commander }}'],
            });

            console.log(`Created postmortem issue: ${issue.data.html_url}`);
```

---

## 7. インシデント対応プロセス

### 7.1 インシデントコマンダー (IC) の役割

```
インシデントコマンダーの責務:

┌────────────────────────────────────────────────────┐
│                                                    │
│  1. 状況把握 (Assess)                              │
│     ・影響範囲の確認                               │
│     ・SEV レベルの判定                             │
│     ・対応チームの招集                             │
│                                                    │
│  2. コミュニケーション (Communicate)               │
│     ・#incident チャンネルの作成                    │
│     ・定期的なステータス更新 (15分ごと)             │
│     ・ステータスページの更新                       │
│     ・ステークホルダーへの報告                     │
│                                                    │
│  3. 委任 (Delegate)                                │
│     ・調査担当の指名                               │
│     ・対外コミュニケーション担当の指名             │
│     ・自身は調査に没頭しない (指揮に専念)          │
│                                                    │
│  4. 意思決定 (Decide)                              │
│     ・ロールバックの判断                           │
│     ・エスカレーションの判断                       │
│     ・インシデントクローズの判断                   │
│                                                    │
│  5. 記録 (Record)                                  │
│     ・タイムラインの記録                           │
│     ・ポストモーテムの手配                         │
│                                                    │
└────────────────────────────────────────────────────┘
```

### 7.2 インシデント対応チェックリスト

```
□ アラートの確認とアクノレッジ
  └─ PagerDuty/Opsgenie でアクノレッジ
  └─ アラートの内容を確認

□ 初期評価 (最初の5分)
  └─ 影響範囲を確認
  └─ SEV レベルを判定
  └─ Runbook を確認

□ コミュニケーション開始
  └─ #incident-YYYYMMDD チャンネルを作成
  └─ IC (インシデントコマンダー) を宣言
  └─ 初期ステータスを投稿

□ 調査と対応
  └─ ダッシュボード/ログ/トレースを確認
  └─ 直近の変更 (デプロイ、設定変更) を確認
  └─ 原因の仮説を立てて検証

□ 緩和策の実行
  └─ ロールバック / スケールアウト / フェイルオーバー
  └─ 影響の軽減策を実行

□ ステータスページの更新
  └─ SEV-1/2: 15分ごとに更新
  └─ SEV-3: 1時間ごとに更新

□ 復旧確認
  └─ メトリクスが正常範囲に戻ったことを確認
  └─ ユーザー影響がなくなったことを確認
  └─ ステータスページを「Operational」に更新

□ インシデントクローズ
  └─ #incident チャンネルにクローズを宣言
  └─ ポストモーテム作成の手配 (48時間以内)
  └─ 関係者への最終報告
```

### 7.3 ステータスページの運用

```typescript
// statuspage-updater.ts — ステータスページの自動更新
import fetch from 'node-fetch';

interface StatusPageConfig {
  apiKey: string;
  pageId: string;
  componentMap: Record<string, string>; // service name → component ID
}

class StatusPageUpdater {
  constructor(private config: StatusPageConfig) {}

  // コンポーネントのステータスを更新
  async updateComponentStatus(
    serviceName: string,
    status: 'operational' | 'degraded_performance' | 'partial_outage' | 'major_outage'
  ): Promise<void> {
    const componentId = this.config.componentMap[serviceName];
    if (!componentId) throw new Error(`Unknown service: ${serviceName}`);

    await fetch(
      `https://api.statuspage.io/v1/pages/${this.config.pageId}/components/${componentId}`,
      {
        method: 'PATCH',
        headers: {
          'Authorization': `OAuth ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          component: { status },
        }),
      }
    );
  }

  // インシデントの作成
  async createIncident(
    name: string,
    body: string,
    impactOverride: 'none' | 'minor' | 'major' | 'critical',
    componentIds: string[],
    componentStatus: string
  ): Promise<string> {
    const response = await fetch(
      `https://api.statuspage.io/v1/pages/${this.config.pageId}/incidents`,
      {
        method: 'POST',
        headers: {
          'Authorization': `OAuth ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          incident: {
            name,
            body,
            status: 'investigating',
            impact_override: impactOverride,
            component_ids: componentIds,
            components: Object.fromEntries(
              componentIds.map(id => [id, componentStatus])
            ),
          },
        }),
      }
    );

    const data = await response.json() as any;
    return data.id;
  }

  // インシデントの更新
  async updateIncident(
    incidentId: string,
    status: 'investigating' | 'identified' | 'monitoring' | 'resolved',
    body: string
  ): Promise<void> {
    await fetch(
      `https://api.statuspage.io/v1/pages/${this.config.pageId}/incidents/${incidentId}`,
      {
        method: 'PATCH',
        headers: {
          'Authorization': `OAuth ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          incident: {
            status,
            body,
          },
        }),
      }
    );
  }
}

// 使用例
const statusPage = new StatusPageUpdater({
  apiKey: process.env.STATUSPAGE_API_KEY!,
  pageId: 'your-page-id',
  componentMap: {
    'order-service': 'component-id-1',
    'payment-service': 'component-id-2',
    'user-service': 'component-id-3',
  },
});

// インシデント発生時
const incidentId = await statusPage.createIncident(
  '決済処理の遅延',
  '決済処理に通常より時間がかかっています。調査中です。',
  'major',
  ['component-id-2'],
  'degraded_performance'
);

// 原因特定後
await statusPage.updateIncident(
  incidentId,
  'identified',
  '決済プロバイダ側の一時的な障害を確認しました。復旧を待っています。'
);

// 復旧後
await statusPage.updateIncident(
  incidentId,
  'resolved',
  '決済プロバイダの障害が復旧し、全サービスが正常に動作しています。'
);
```

---

## 8. 比較表

| アラートツール | Alertmanager | PagerDuty | Opsgenie | Datadog Monitors |
|--------------|-------------|-----------|----------|-----------------|
| 運用形態 | OSS | SaaS | SaaS | SaaS |
| エスカレーション | 基本的 | 高度 | 高度 | 基本的 |
| オンコール管理 | なし (外部連携) | 充実 | 充実 | なし (外部連携) |
| モバイルアプリ | なし | あり | あり | あり |
| インシデント管理 | なし | あり | あり | あり |
| 自動修復 | Webhook | Event Orchestration | Webhook | Workflow Automation |
| 料金 | 無料 | $21/user/月〜 | $9/user/月〜 | 含む |
| Terraform | 対応 | 対応 | 対応 | 対応 |

| ポストモーテムツール | Google Docs | Jeli | incident.io | Notion | GitHub Issues |
|--------------------|------------|------|-------------|--------|---------------|
| テンプレート管理 | 手動 | 自動 | 自動 | 手動 | 自動 (Actions) |
| タイムライン自動生成 | 不可 | 対応 | 対応 | 不可 | 不可 |
| メトリクス埋め込み | 不可 | 対応 | 対応 | 不可 | 不可 |
| アクション追跡 | 不可 | 対応 | 対応 | 手動 | Issue 連携 |
| Slack 統合 | 不可 | 対応 | 対応 | 対応 | 対応 |
| コスト | 無料 | 有料 | 有料 | 無料/有料 | 無料 |

| ステータスページ | Statuspage | Instatus | Cachet | Better Uptime |
|----------------|-----------|----------|--------|--------------|
| 運用形態 | SaaS | SaaS | OSS | SaaS |
| API 対応 | 充実 | 対応 | 対応 | 対応 |
| 自動ステータス更新 | 対応 | 対応 | 限定的 | 対応 |
| カスタムドメイン | 対応 | 対応 | 対応 | 対応 |
| 料金 | $29/月〜 | $20/月〜 | 無料 | $20/月〜 |

---

## 9. アンチパターン

### アンチパターン 1: アラート疲れ (Alert Fatigue)

```
[悪い例]
- 1日に50件以上のアラートが発報
- 大半が「対応不要」で無視される
- 本当に重要なアラートも見逃される
- オンコール担当が疲弊し離職

[良い例]
- アラートは「今すぐ人間が対応すべきもの」に限定
- 月次でアラートを棚卸し:
  - 対応不要で無視したアラート → 削除またはしきい値調整
  - 自動復旧できるアラート → 自動修復に変更
  - 頻発するアラート → 根本原因を修正
- 目標: PAGE は月に数回、1アラートにつき必ずアクションが伴う
```

### アンチパターン 2: Runbook なしのアラート

```
[悪い例]
- アラートが来たが対応方法が不明
- 「前回どう対応した？」を Slack で検索
- 深夜のオンコールで経験者に電話して確認
- 同じ障害に毎回異なる対応をしてしまう

[良い例]
- 全アラートに Runbook URL を紐付け
- Runbook の内容:
  1. アラートの意味と影響範囲
  2. 確認すべきダッシュボード/ログ
  3. ステップバイステップの対応手順
  4. エスカレーション判断基準
  5. 過去のインシデントリンク
- Runbook は障害対応後に更新する習慣をつける
```

### アンチパターン 3: ポストモーテムの形骸化

```
[悪い例]
- ポストモーテムを書くが、アクションアイテムが放置される
- 同じ原因のインシデントが繰り返し発生
- 「個人の注意不足」が根本原因に挙がる
- ポストモーテムが blame (非難) の場になる

[良い例]
- アクションアイテムは必ずチケット化し、期限と担当者を設定
- 月次でアクションアイテムの進捗を確認
- 根本原因は「システムの改善余地」として記述
  ×「Aさんがテストを忘れた」
  ○「テスト自動化の仕組みがなく、手動テストに依存していた」
- ポストモーテムの目的を繰り返し周知:
  「Who failed?」ではなく「What failed?」
```

### アンチパターン 4: エスカレーション不足

```
[悪い例]
- オンコール担当が一人で2時間以上格闘
- 「もう少しで解決できそう」と思ってエスカレートしない
- 結果的にダウンタイムが長引く
- 事後に「なぜもっと早くエスカレートしなかった？」

[良い例]
- エスカレーション判断の明文化:
  ・15分以内に原因が特定できない → エスカレート
  ・SEV-1/2 のインシデント → 即時エスカレート
  ・自分の専門外の領域 → 即時エスカレート
- エスカレーションは「弱さ」ではなく「判断力」
- IC はエスカレーションを促す役割を持つ
```

---

## 10. 月次アラートレビュー

### 10.1 レビューミーティングのアジェンダ

```
月次アラートレビュー (60分):

  1. 先月のアラート統計 (10分)
     ・PAGE 数 / TICKET 数 / NOTIFICATION 数
     ・MTTA (平均応答時間) / MTTR (平均復旧時間)
     ・False positive 率
     ・最も発報回数が多いアラート Top 5

  2. アラートの棚卸し (20分)
     ・False positive が多いアラート → 閾値調整 or 削除
     ・対応不要だったアラート → レベル変更 or 削除
     ・新しく追加すべきアラート → 作成
     ・自動修復に移行できるアラート → 自動化

  3. Runbook の更新 (10分)
     ・新しいアラートの Runbook 作成
     ・既存 Runbook の更新 (最新の対応手順)
     ・Runbook カバレッジの確認 (100% 目標)

  4. ポストモーテムのアクション確認 (10分)
     ・未完了のアクションアイテムの進捗
     ・完了したアクションの効果確認

  5. オンコール体験の振り返り (10分)
     ・オンコール担当からのフィードバック
     ・改善提案
     ・次月のスケジュール確認
```

### 10.2 アラートレビュー用 PromQL

```promql
# 先月の PAGE 数
count(ALERTS{severity="critical", alertstate="firing"})

# アラート別の発報回数 (Top 10)
topk(10,
  count(ALERTS{alertstate="firing"}) by (alertname)
)

# アラートの平均 firing 時間
avg(
  time() - ALERTS_FOR_STATE{alertstate="firing"}
) by (alertname)

# False positive 率の推定
# (5分以内に resolve されたアラートの割合)
count(ALERTS{alertstate="firing"} < 300) by (alertname)
/
count(ALERTS{alertstate="firing"}) by (alertname)
```

---

## 11. FAQ

### Q1: アラートのしきい値はどう決めるべきですか？

SLO ベースのバーンレートアラートを基本とし、静的なしきい値（CPU > 80% など）は補助的に使います。バーンレートアラートは「このペースでエラーが続くとSLOを割る」という予測的なアラートで、ビジネスインパクトに直結します。しきい値は最初はゆるめに設定し、運用しながら誤検知（false positive）と見逃し（false negative）のバランスで調整してください。

### Q2: オンコールローテーションの適切な人数は？

最低 4〜5 名のローテーションが推奨です。週次ローテーションで月に1回程度の当番が理想です。2〜3 名では頻度が高すぎて燃え尽きのリスクがあります。また、1次担当と 2次担当を分け、1次が対応不可の場合に自動エスカレートする仕組みを構築してください。

### Q3: ポストモーテムで最も重要なことは何ですか？

**blame-free（非難しない）文化**の徹底です。ポストモーテムの目的は「誰が悪かったか」ではなく「システムのどこに改善余地があるか」を見つけることです。個人を責めると情報が隠蔽され、組織の学習能力が低下します。再発防止策は「人が気をつける」ではなく「システムで防止する」方向で設計してください。

### Q4: Alertmanager の高可用性をどう実現しますか？

Alertmanager はネイティブでクラスタリングをサポートしています。複数のインスタンスを `--cluster.peer` フラグで相互接続し、アラートの重複排除（deduplication）と自動フェイルオーバーを実現します。最低 2 台、推奨 3 台の構成で、全 Prometheus インスタンスから全 Alertmanager インスタンスに通知を送信する設定にしてください。

### Q5: ビジネスメトリクスのアラートは誰が管理すべきですか？

ビジネスメトリクスのアラートは、エンジニアリングチームとビジネスチームの協業で管理すべきです。エンジニアがアラートの技術的な実装（PromQL / Datadog クエリ）を担当し、ビジネスチームが閾値とビジネスインパクトの定義を担当します。ビジネスアラートは通常 TICKET レベルで、営業時間内に対応する設計にします。

### Q6: アラートの Silence (抑制) を使う際の注意点は？

Silence は計画メンテナンス時や、既知の問題で繰り返しアラートが発報される場合に使用します。注意点として、(1) Silence には必ず期限を設定する（永続的な Silence は禁止）、(2) Silence の理由をコメントに記載する、(3) Silence の作成/削除を Slack に通知する、(4) 週次で不要な Silence がないか確認する、があります。

---

## まとめ

| 項目 | 要点 |
|------|------|
| アラート設計 | PAGE/TICKET/NOTIFICATION の階層化。PAGE は最小限に |
| バーンレートアラート | SLO ベースの予測的アラート。Multi-window で誤検知を低減 |
| エスカレーション | 段階的な通知先。タイムアウトで自動エスカレート |
| オンコール | 4〜5名ローテーション。Runbook を必ず整備 |
| 自動修復 | 定型的な障害は Webhook/CronJob で自動復旧 |
| ポストモーテム | blame-free で実施。アクションアイテムは必ずチケット化 |
| アラート疲れ | 月次棚卸しで不要アラートを削除。自動修復を推進 |
| インシデント管理 | SEV レベル定義、IC の役割、ステータスページの運用 |

---

## 次に読むべきガイド

- [00-observability.md](./00-observability.md) — オブザーバビリティの基礎
- [01-monitoring-tools.md](./01-monitoring-tools.md) — 監視ツールの選定と構築
- [03-performance-monitoring.md](./03-performance-monitoring.md) — パフォーマンス監視

---

## 参考文献

1. **Google SRE Book - Alerting on SLOs** — https://sre.google/workbook/alerting-on-slos/ — SLO ベースのアラート設計
2. **PagerDuty Incident Response Guide** — https://response.pagerduty.com/ — インシデント対応のベストプラクティス
3. **Alertmanager Documentation** — https://prometheus.io/docs/alerting/latest/alertmanager/ — Alertmanager 公式ドキュメント
4. **Etsy Debriefing Facilitation Guide** — https://github.com/etsy/DebriefingFacilitationGuide — ポストモーテムのファシリテーション手法
5. **incident.io** — https://incident.io/ — インシデント管理プラットフォーム
6. **Jeli** — https://www.jeli.io/ — ポストモーテム管理ツール
7. **Learning from Incidents in Software** — https://www.learningfromincidents.io/ — インシデントからの学習
