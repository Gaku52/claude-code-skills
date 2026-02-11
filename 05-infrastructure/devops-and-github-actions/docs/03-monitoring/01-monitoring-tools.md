# 監視ツール

> Datadog、Grafana、CloudWatch の特徴を理解し、システム規模と要件に応じた最適な監視基盤を構築する

## この章で学ぶこと

1. **Grafana + Prometheus によるOSS監視スタック** — メトリクス収集、ダッシュボード構築、PromQL の活用
2. **Datadog によるフルスタック監視** — SaaS ベースの統合監視プラットフォームの活用
3. **AWS CloudWatch による AWS ネイティブ監視** — メトリクス、ログ、アラーム、ダッシュボードの構成

---

## 1. 監視ツールの全体像

```
┌──────────────────────────────────────────────────────┐
│               監視ツールの選択肢                       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  OSS スタック (自前運用)                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │Prometheus│─►│ Grafana  │  │  Loki    │           │
│  │(メトリクス)│  │(可視化)   │  │(ログ)    │           │
│  └──────────┘  └──────────┘  └──────────┘           │
│  ┌──────────┐  ┌──────────┐                         │
│  │  Jaeger  │  │ Alertmgr │                         │
│  │(トレース) │  │(アラート) │                         │
│  └──────────┘  └──────────┘                         │
│                                                      │
│  SaaS (マネージド)                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Datadog  │  │ New Relic│  │CloudWatch│           │
│  │(フルスタック)│ │(APM重視) │  │(AWS特化) │           │
│  └──────────┘  └──────────┘  └──────────┘           │
└──────────────────────────────────────────────────────┘
```

---

## 2. Prometheus + Grafana スタック

```yaml
# docker-compose.monitoring.yml
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:v2.50.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning

  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    volumes:
      - loki-data:/loki

  node-exporter:
    image: prom/node-exporter:v1.7.0
    ports:
      - "9100:9100"
    pid: host
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro

volumes:
  prometheus-data:
  grafana-data:
  loki-data:
```

```yaml
# prometheus.yml — Prometheus 設定
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/*.yml"

scrape_configs:
  # Prometheus 自身の監視
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # Node Exporter (サーバーメトリクス)
  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  # アプリケーション (Express/Fastify)
  - job_name: "app"
    metrics_path: /metrics
    scrape_interval: 10s
    static_configs:
      - targets: ["app:3000"]
        labels:
          service: "order-service"
          environment: "production"

  # Kubernetes Service Discovery
  - job_name: "kubernetes-pods"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
```

---

## 3. PromQL の基本

```promql
# 基本的な PromQL クエリ例

# 1. HTTP リクエストレート (1秒あたり)
rate(http_requests_total[5m])

# 2. エラーレート (%)
sum(rate(http_requests_total{status=~"5.."}[5m]))
/
sum(rate(http_requests_total[5m]))
* 100

# 3. p95 レイテンシ
histogram_quantile(0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
)

# 4. CPU 使用率 (%)
100 - (avg by(instance) (
  irate(node_cpu_seconds_total{mode="idle"}[5m])
) * 100)

# 5. メモリ使用率 (%)
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# 6. ディスク使用率 (%)
(1 - (node_filesystem_avail_bytes{mountpoint="/"}
     / node_filesystem_size_bytes{mountpoint="/"})) * 100
```

```
PromQL のデータモデル:

  時系列 (Time Series)
  ┌────────────────────────────────────────────┐
  │ メトリクス名{ラベル}                         │
  │                                            │
  │ http_requests_total{method="GET",status="200"}  │
  │                                            │
  │  時刻        値                             │
  │  T1    ──── 1000                           │
  │  T2    ──── 1050                           │
  │  T3    ──── 1120                           │
  │  T4    ──── 1200                           │
  │                                            │
  │  rate() → 瞬間的な増加率を算出              │
  │  (1200 - 1000) / (T4 - T1) = X req/sec    │
  └────────────────────────────────────────────┘

  メトリクスの型:
  ┌─────────┬──────────────────────────────────┐
  │ Counter │ 単調増加 (リクエスト数、エラー数)   │
  │ Gauge   │ 上下する値 (CPU使用率、温度)       │
  │Histogram│ 分布 (レイテンシ、サイズ)          │
  │ Summary │ 分位数の直接計算                   │
  └─────────┴──────────────────────────────────┘
```

---

## 4. Datadog による統合監視

```typescript
// datadog-apm.ts — Datadog APM のセットアップ
import tracer from 'dd-trace';

tracer.init({
  service: 'order-service',
  env: process.env.NODE_ENV ?? 'development',
  version: process.env.APP_VERSION ?? '0.0.0',
  logInjection: true,  // ログにトレースIDを自動挿入
  runtimeMetrics: true, // Node.js ランタイムメトリクス
  profiling: true,      // Continuous Profiling
  appsec: true,         // Application Security Monitoring
});

export default tracer;
```

```yaml
# datadog-agent.yaml — Datadog Agent の Kubernetes DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: datadog-agent
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: datadog-agent
  template:
    metadata:
      labels:
        app: datadog-agent
    spec:
      containers:
        - name: agent
          image: gcr.io/datadoghq/agent:7
          env:
            - name: DD_API_KEY
              valueFrom:
                secretKeyRef:
                  name: datadog-secrets
                  key: api-key
            - name: DD_SITE
              value: "ap1.datadoghq.com"
            - name: DD_APM_ENABLED
              value: "true"
            - name: DD_LOGS_ENABLED
              value: "true"
            - name: DD_PROCESS_AGENT_ENABLED
              value: "true"
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          volumeMounts:
            - name: dockersocket
              mountPath: /var/run/docker.sock
            - name: procdir
              mountPath: /host/proc
              readOnly: true
      volumes:
        - name: dockersocket
          hostPath:
            path: /var/run/docker.sock
        - name: procdir
          hostPath:
            path: /proc
```

---

## 5. AWS CloudWatch

```typescript
// cloudwatch-custom-metrics.ts — CloudWatch カスタムメトリクスの送信
import {
  CloudWatchClient,
  PutMetricDataCommand,
} from '@aws-sdk/client-cloudwatch';

const cloudwatch = new CloudWatchClient({ region: 'ap-northeast-1' });

async function publishMetric(
  metricName: string,
  value: number,
  unit: 'Count' | 'Milliseconds' | 'Percent',
  dimensions: { Name: string; Value: string }[]
) {
  await cloudwatch.send(
    new PutMetricDataCommand({
      Namespace: 'MyApp/Production',
      MetricData: [
        {
          MetricName: metricName,
          Value: value,
          Unit: unit,
          Dimensions: dimensions,
          Timestamp: new Date(),
        },
      ],
    })
  );
}

// 使用例: API レイテンシの記録
await publishMetric('ApiLatency', 125, 'Milliseconds', [
  { Name: 'Service', Value: 'order-service' },
  { Name: 'Endpoint', Value: '/api/orders' },
]);

// 使用例: ビジネスメトリクスの記録
await publishMetric('OrdersCreated', 1, 'Count', [
  { Name: 'PaymentMethod', Value: 'credit_card' },
]);
```

---

## 6. 比較表

| 特性 | Prometheus + Grafana | Datadog | CloudWatch |
|------|---------------------|---------|------------|
| 運用形態 | セルフホスト | SaaS | AWS マネージド |
| メトリクス | Prometheus | 独自 | 独自 |
| ログ | Loki | Log Management | CloudWatch Logs |
| トレース | Jaeger/Tempo | APM | X-Ray |
| ダッシュボード | Grafana (強力) | 内蔵 (高機能) | 基本的 |
| アラート | Alertmanager | Monitors | CloudWatch Alarms |
| 月額コスト (中規模) | インフラ費のみ | $500〜$5,000+ | $100〜$500 |
| 学習コスト | 高い (複数ツール) | 中 | 低い (AWS 利用者) |

| ダッシュボードツール | Grafana | Datadog Dashboard | CloudWatch Dashboard |
|---------------------|---------|-------------------|---------------------|
| データソース数 | 100+ | Datadog内 | AWS内 |
| テンプレート変数 | 強力 | 対応 | 限定的 |
| 共有/埋め込み | 対応 | 対応 | 限定的 |
| アラート統合 | Alertmanager | 内蔵 | SNS 連携 |
| モバイル対応 | アプリあり | アプリあり | なし |
| IaC サポート | Terraform/Jsonnet | Terraform | CloudFormation |

---

## 7. アンチパターン

### アンチパターン 1: ダッシュボードの乱立

```
[悪い例]
- チームメンバーが個人でダッシュボードを大量作成
- 同じメトリクスを異なるクエリで表示 → 数値が一致しない
- 重要なダッシュボードが見つからない（50個以上のダッシュボード）
- メンテナンスされず古いメトリクスを参照し続ける

[良い例]
- ダッシュボードの階層設計:
  Level 0: サービス全体のヘルスチェック (経営層向け)
  Level 1: サービス別の主要メトリクス (チームリード向け)
  Level 2: 詳細な技術メトリクス (エンジニア向け)
- ダッシュボードを IaC (Terraform/Jsonnet) で管理
- 四半期ごとに不要なダッシュボードを棚卸し
```

### アンチパターン 2: 高カーディナリティラベル

```promql
# 悪い例: ユーザーIDをラベルにする → 数百万の時系列が生成
http_requests_total{user_id="user-123", method="GET", path="/api/items"}
# ユーザー100万人 × メソッド4種 × パス50種 = 2億の時系列!

# 良い例: 集計に意味のあるラベルのみ使用
http_requests_total{method="GET", path="/api/items", status="200"}
# メソッド4種 × パス50種 × ステータス10種 = 2,000の時系列

# ユーザー単位の分析はログやトレースで行う
```

---

## 8. FAQ

### Q1: OSS スタックと SaaS、どちらを選ぶべきですか？

運用チームの規模とスキルが判断基準です。専任の SRE/インフラチーム（2名以上）がいれば OSS（Prometheus + Grafana）でコストを抑えつつ高いカスタマイズ性が得られます。少人数チームで監視基盤の運用に時間を割けない場合は、Datadog のような SaaS を選択してください。AWS に閉じたシステムなら CloudWatch が最もシンプルです。

### Q2: Prometheus のデータ保持期間はどのくらいが適切ですか？

ローカルストレージでは 15〜30日が現実的です。長期保存が必要な場合は Thanos や Cortex などのリモートストレージソリューションを導入してください。メトリクスの解像度を下げる（ダウンサンプリング）ことで、1年以上のデータも効率的に保持できます。

### Q3: Grafana のダッシュボードをコード管理する方法は？

3つのアプローチがあります。(1) **Grafana Provisioning**: YAML + JSON ファイルで Git 管理し、起動時に自動読み込み。(2) **Terraform provider**: `grafana_dashboard` リソースで IaC 管理。(3) **Grafonnet (Jsonnet)**: プログラマブルにダッシュボードを生成。チーム規模が大きい場合は Terraform、小規模なら Provisioning が推奨です。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Prometheus | Pull 型メトリクス収集。PromQL で柔軟なクエリ |
| Grafana | 多数のデータソース対応。最も柔軟なダッシュボード |
| Loki | Grafana 連携のログ集約。ラベルベースのインデックス |
| Datadog | フルスタック SaaS。APM/ログ/メトリクス統合 |
| CloudWatch | AWS ネイティブ。追加設定なしで AWS リソース監視 |
| ダッシュボード設計 | 階層化し、IaC で管理。カーディナリティに注意 |

---

## 次に読むべきガイド

- [00-observability.md](./00-observability.md) — オブザーバビリティの3本柱
- [02-alerting.md](./02-alerting.md) — アラート戦略とエスカレーション
- [03-performance-monitoring.md](./03-performance-monitoring.md) — APM とパフォーマンス監視

---

## 参考文献

1. **Prometheus: Up & Running** — Brian Brazil (O'Reilly, 2018) — Prometheus の実践ガイド
2. **Grafana Documentation** — https://grafana.com/docs/ — Grafana 公式ドキュメント
3. **Datadog Documentation** — https://docs.datadoghq.com/ — Datadog 公式リファレンス
4. **AWS CloudWatch Documentation** — https://docs.aws.amazon.com/AmazonCloudWatch/ — CloudWatch 公式ガイド
