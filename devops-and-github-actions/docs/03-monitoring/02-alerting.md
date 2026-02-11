# アラート戦略

> アラート設計の原則、エスカレーションポリシー、ポストモーテムの運用を習得し、アラート疲れのない持続可能なオンコール体制を構築する

## この章で学ぶこと

1. **効果的なアラート設計** — アラート疲れを防ぎ、本当に対応が必要なアラートだけを発報する設計原則
2. **エスカレーションとオンコール体制** — 段階的なエスカレーションポリシーとオンコールローテーションの構築
3. **ポストモーテムと継続的改善** — 障害から学び、再発防止を組織的に推進するプロセス

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

---

## 2. アラートルールの設計

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
        annotations:
          summary: "エラーバーンレートが危険水準 (Critical)"
          description: >
            直近5分と1時間のエラー率がSLOバーンレートの14.4倍を超えています。
            このペースでは2日以内に月間エラーバジェットを使い切ります。
          runbook: "https://wiki.example.com/runbooks/high-error-rate"

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
        annotations:
          summary: "エラーバーンレートが警告水準 (Warning)"
          description: >
            直近30分と6時間のエラー率がSLOバーンレートの6倍を超えています。
          runbook: "https://wiki.example.com/runbooks/high-error-rate"

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

      - alert: DiskSpaceRunningOut
        expr: |
          predict_linear(node_filesystem_avail_bytes{mountpoint="/"}[6h], 24*3600) < 0
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "24時間以内にディスク容量が枯渇する予測"
          description: "{{ $labels.instance }} のディスクが24時間以内に満杯になります"
```

---

## 3. Alertmanager 設定

```yaml
# alertmanager.yml — エスカレーション設定
global:
  resolve_timeout: 5m
  slack_api_url: "https://hooks.slack.com/services/XXX/YYY/ZZZ"

# ルーティングツリー
route:
  receiver: default-notification
  group_by: ['alertname', 'team']
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

# 通知先の定義
receivers:
  - name: default-notification
    slack_configs:
      - channel: '#alerts-general'

  - name: pagerduty-critical
    pagerduty_configs:
      - routing_key: "YOUR_PAGERDUTY_ROUTING_KEY"
        severity: critical
        description: '{{ .CommonAnnotations.summary }}'
        details:
          alert: '{{ .CommonLabels.alertname }}'
          description: '{{ .CommonAnnotations.description }}'
          runbook: '{{ .CommonAnnotations.runbook }}'

  - name: slack-critical
    slack_configs:
      - channel: '#alerts-critical'
        color: 'danger'
        title: '{{ .CommonLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
        actions:
          - type: button
            text: 'Runbook'
            url: '{{ .CommonAnnotations.runbook }}'
          - type: button
            text: 'Dashboard'
            url: 'https://grafana.example.com/d/main'

  - name: slack-warning
    slack_configs:
      - channel: '#alerts-warning'
        color: 'warning'
        title: '{{ .CommonLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'

# 抑制ルール (Critical 発報中は Warning を抑制)
inhibit_rules:
  - source_match:
      severity: critical
    target_match:
      severity: warning
    equal: ['alertname', 'team']
```

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
```

---

## 4. PagerDuty / Opsgenie 連携

```typescript
// oncall-rotation.ts — オンコールローテーション設計
interface OncallSchedule {
  team: string;
  rotationType: 'weekly' | 'daily';
  members: string[];
  escalationPolicy: EscalationLevel[];
  overrides: Override[];
}

interface EscalationLevel {
  level: number;
  targets: string[];
  timeout: number;  // 分
}

const backendOncall: OncallSchedule = {
  team: 'backend',
  rotationType: 'weekly',
  members: [
    'engineer-a@example.com',
    'engineer-b@example.com',
    'engineer-c@example.com',
    'engineer-d@example.com',
  ],
  escalationPolicy: [
    { level: 1, targets: ['current-oncall'], timeout: 5 },
    { level: 2, targets: ['secondary-oncall'], timeout: 10 },
    { level: 3, targets: ['engineering-manager'], timeout: 15 },
  ],
  overrides: [],
};
```

---

## 5. ポストモーテムテンプレート

```markdown
<!-- postmortem-template.md -->
# ポストモーテム: [インシデントタイトル]

## 概要
- **日時**: 2025-03-15 14:30 〜 15:45 JST (75分間)
- **影響範囲**: 全ユーザーの決済処理が不可
- **影響度**: SEV-1 (サービス全体の重大障害)
- **検知方法**: SLO バーンレートアラート (自動検知)
- **対応者**: @engineer-a (1次), @engineer-b (支援)

## タイムライン
| 時刻 | イベント |
|------|---------|
| 14:30 | デプロイ完了 (v2.5.0) |
| 14:32 | エラーレート急上昇、アラート発報 |
| 14:35 | オンコール担当がアクノレッジ |
| 14:40 | 決済 API の 500 エラーを確認 |
| 14:50 | 原因特定: DB マイグレーションで決済テーブルのカラム名変更 |
| 15:00 | ロールバック開始 (v2.4.3 へ) |
| 15:15 | ロールバック完了、エラーレート正常化 |
| 15:45 | 全メトリクス正常を確認、インシデントクローズ |

## 根本原因
DB マイグレーションで `payment_status` カラムを `status` にリネームしたが、
旧バージョンのコードが `payment_status` を参照していた。
Rolling Update 中に新旧バージョンが混在し、旧コードがカラム未検出エラーを起こした。

## 再発防止策
| アクション | 担当 | 期限 | 優先度 |
|-----------|------|------|--------|
| DBマイグレーションのExpand-Contract パターンを必須化 | @engineer-a | 2025-03-22 | P0 |
| デプロイ後の自動 Smoke Test を追加 | @engineer-b | 2025-03-29 | P1 |
| ロールバック手順を自動化 | @engineer-c | 2025-04-05 | P1 |

## 教訓
- 破壊的な DB 変更は Expand-Contract パターンで段階的に行う
- デプロイ直後のメトリクス監視期間を設ける (最低10分)
- ロールバック判断の基準を事前に定義しておく
```

---

## 6. 比較表

| アラートツール | Alertmanager | PagerDuty | Opsgenie | Datadog Monitors |
|--------------|-------------|-----------|----------|-----------------|
| 運用形態 | OSS | SaaS | SaaS | SaaS |
| エスカレーション | 基本的 | 高度 | 高度 | 基本的 |
| オンコール管理 | なし (外部連携) | 充実 | 充実 | なし (外部連携) |
| モバイルアプリ | なし | あり | あり | あり |
| インシデント管理 | なし | あり | あり | あり |
| 料金 | 無料 | $21/user/月〜 | $9/user/月〜 | 含む |

| ポストモーテムツール | Google Docs | Jeli | incident.io | Notion |
|--------------------|------------|------|-------------|--------|
| テンプレート管理 | 手動 | 自動 | 自動 | 手動 |
| タイムライン自動生成 | 不可 | 対応 | 対応 | 不可 |
| メトリクス埋め込み | 不可 | 対応 | 対応 | 不可 |
| アクション追跡 | 不可 | 対応 | 対応 | 手動 |
| コスト | 無料 | 有料 | 有料 | 無料/有料 |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: アラートのしきい値はどう決めるべきですか？

SLO ベースのバーンレートアラートを基本とし、静的なしきい値（CPU > 80% など）は補助的に使います。バーンレートアラートは「このペースでエラーが続くとSLOを割る」という予測的なアラートで、ビジネスインパクトに直結します。しきい値は最初はゆるめに設定し、運用しながら誤検知（false positive）と見逃し（false negative）のバランスで調整してください。

### Q2: オンコールローテーションの適切な人数は？

最低 4〜5 名のローテーションが推奨です。週次ローテーションで月に1回程度の当番が理想です。2〜3 名では頻度が高すぎて燃え尽きのリスクがあります。また、1次担当と 2次担当を分け、1次が対応不可の場合に自動エスカレートする仕組みを構築してください。

### Q3: ポストモーテムで最も重要なことは何ですか？

**blame-free（非難しない）文化**の徹底です。ポストモーテムの目的は「誰が悪かったか」ではなく「システムのどこに改善余地があるか」を見つけることです。個人を責めると情報が隠蔽され、組織の学習能力が低下します。再発防止策は「人が気をつける」ではなく「システムで防止する」方向で設計してください。

---

## まとめ

| 項目 | 要点 |
|------|------|
| アラート設計 | PAGE/TICKET/NOTIFICATION の階層化。PAGE は最小限に |
| バーンレートアラート | SLO ベースの予測的アラート。Multi-window で誤検知を低減 |
| エスカレーション | 段階的な通知先。タイムアウトで自動エスカレート |
| オンコール | 4〜5名ローテーション。Runbook を必ず整備 |
| ポストモーテム | blame-free で実施。再発防止策はシステム改善で対応 |
| アラート疲れ | 月次棚卸しで不要アラートを削除。自動修復を推進 |

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
