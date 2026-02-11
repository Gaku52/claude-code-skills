# モニタリング

> Prometheus / Grafana / cAdvisor / Lokiを組み合わせて、Dockerコンテナ環境の包括的な監視・ログ集約・アラート基盤を構築する。

---

## この章で学ぶこと

1. **Prometheus + cAdvisorによるメトリクス収集**のアーキテクチャと設定を理解する
2. **Grafanaダッシュボード**の構築とアラートルールの設定を習得する
3. **Loki / ELKによるログ集約**と相関分析の手法を把握する

---

## 1. コンテナモニタリングの全体像

### 監視スタックのアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Host                             │
│                                                             │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐                     │
│  │  App A  │ │  App B  │ │  App C  │  ← 監視対象          │
│  └────┬────┘ └────┬────┘ └────┬────┘                     │
│       │           │           │                             │
│  ┌────▼───────────▼───────────▼────┐                      │
│  │          cAdvisor                │  ← コンテナメトリクス │
│  │  CPU, Memory, Network, Disk I/O │     収集              │
│  └──────────────┬──────────────────┘                      │
│                 │ :8080/metrics                             │
│  ┌──────────────▼──────────────────┐                      │
│  │          Prometheus             │  ← メトリクス保存     │
│  │  Pull型メトリクス収集            │     クエリエンジン    │
│  │  PromQL クエリ                  │                      │
│  └──────┬───────────┬──────────────┘                      │
│         │           │                                      │
│  ┌──────▼──────┐ ┌──▼──────────────┐                     │
│  │  Grafana    │ │  Alertmanager   │                     │
│  │ ダッシュボード│ │  Slack/Email    │                     │
│  │  :3000      │ │  PagerDuty      │                     │
│  └─────────────┘ └─────────────────┘                     │
│                                                             │
│  ┌─────────────────────────────────┐                      │
│  │          Loki                   │  ← ログ集約          │
│  │  ログのインデックス・検索        │                      │
│  └──────────────┬──────────────────┘                      │
│                 │                                          │
│  ┌──────────────▼──────────────────┐                      │
│  │        Promtail / Alloy         │  ← ログ収集エージェント│
│  └─────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### 監視ツール比較表

| ツール | 種類 | 役割 | データ型 | 特徴 |
|--------|------|------|---------|------|
| Prometheus | メトリクス | 時系列データ収集・保存 | 数値 | Pull型、PromQL |
| cAdvisor | エクスポーター | コンテナリソースメトリクス | 数値 | Googleが開発 |
| Grafana | 可視化 | ダッシュボード・アラート | - | 多データソース対応 |
| Alertmanager | アラート | 通知ルーティング・抑制 | - | グルーピング、サイレンス |
| Loki | ログ | ログ集約・検索 | テキスト | Prometheusライクなラベル |
| Promtail | ログ収集 | ログ転送エージェント | テキスト | Loki専用 |
| ELK Stack | ログ | ログ集約・全文検索 | テキスト | 高機能、リソース消費大 |

---

## 2. Prometheus + cAdvisor によるメトリクス収集

### コード例1: 監視スタックの Docker Compose 構成

```yaml
# docker-compose.monitoring.yml
version: "3.9"

services:
  # === メトリクス収集 ===
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: prometheus
    restart: unless-stopped
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"
      - "--web.enable-lifecycle"    # APIでリロード可能
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/alert-rules.yml:/etc/prometheus/alert-rules.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    networks:
      - monitoring

  # === コンテナメトリクス ===
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.49.1
    container_name: cadvisor
    restart: unless-stopped
    privileged: true
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    ports:
      - "8080:8080"
    networks:
      - monitoring

  # === Node Exporter（ホストメトリクス） ===
  node-exporter:
    image: prom/node-exporter:v1.8.0
    container_name: node-exporter
    restart: unless-stopped
    command:
      - "--path.rootfs=/host"
    volumes:
      - /:/host:ro,rslave
    pid: host
    networks:
      - monitoring

  # === ダッシュボード ===
  grafana:
    image: grafana/grafana:10.4.0
    container_name: grafana
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    ports:
      - "3000:3000"
    networks:
      - monitoring

  # === アラートマネージャー ===
  alertmanager:
    image: prom/alertmanager:v0.27.0
    container_name: alertmanager
    restart: unless-stopped
    volumes:
      - ./alertmanager/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
    ports:
      - "9093:9093"
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus-data:
  grafana-data:
```

### コード例2: Prometheus設定ファイル

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s        # メトリクス収集間隔
  evaluation_interval: 15s    # ルール評価間隔
  scrape_timeout: 10s

# アラートルール
rule_files:
  - "alert-rules.yml"

# Alertmanager連携
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

# スクレイプ対象の定義
scrape_configs:
  # Prometheus自身のメトリクス
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # cAdvisor（コンテナメトリクス）
  - job_name: "cadvisor"
    static_configs:
      - targets: ["cadvisor:8080"]
    metric_relabel_configs:
      # 不要なメトリクスを除外（ストレージ節約）
      - source_labels: [__name__]
        regex: "container_tasks_state|container_memory_failures_total"
        action: drop

  # Node Exporter（ホストメトリクス）
  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  # アプリケーションメトリクス（/metrics エンドポイント）
  - job_name: "app-metrics"
    static_configs:
      - targets: ["api:8080", "worker:8080"]
    metrics_path: /metrics

  # Docker Engine メトリクス
  - job_name: "docker-engine"
    static_configs:
      - targets: ["host.docker.internal:9323"]

  # Dockerサービスディスカバリ（ラベルベース）
  - job_name: "docker-sd"
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 30s
    relabel_configs:
      - source_labels: [__meta_docker_container_label_prometheus_scrape]
        regex: "true"
        action: keep
      - source_labels: [__meta_docker_container_label_prometheus_port]
        target_label: __address__
        regex: (.+)
        replacement: "${1}"
```

---

## 3. アラートルール

### コード例3: Prometheus アラートルール

```yaml
# prometheus/alert-rules.yml
groups:
  - name: container-alerts
    rules:
      # コンテナダウン
      - alert: ContainerDown
        expr: absent(container_last_seen{name=~".+"})
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "コンテナが停止しています"
          description: "{{ $labels.name }} が1分以上停止しています"

      # CPU使用率が高い
      - alert: ContainerHighCPU
        expr: >
          (sum(rate(container_cpu_usage_seconds_total{name=~".+"}[5m])) by (name)
          / sum(container_spec_cpu_quota{name=~".+"}/container_spec_cpu_period{name=~".+"}) by (name)
          * 100) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "コンテナCPU使用率が高い ({{ $value | printf \"%.1f\" }}%)"
          description: "{{ $labels.name }} のCPU使用率が5分間80%を超えています"

      # メモリ使用率が高い
      - alert: ContainerHighMemory
        expr: >
          (container_memory_usage_bytes{name=~".+"}
          / container_spec_memory_limit_bytes{name=~".+"} * 100) > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "コンテナメモリ使用率が高い ({{ $value | printf \"%.1f\" }}%)"
          description: "{{ $labels.name }} のメモリ使用率が5分間85%を超えています"

      # OOM Kill 発生
      - alert: ContainerOOMKilled
        expr: >
          increase(container_oom_events_total{name=~".+"}[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "OOM Kill が発生"
          description: "{{ $labels.name }} がOOM Killされました"

      # ディスク使用率
      - alert: HostHighDiskUsage
        expr: >
          (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "ディスク使用率が高い ({{ $value | printf \"%.1f\" }}%)"

  - name: application-alerts
    rules:
      # ヘルスチェック失敗
      - alert: HealthCheckFailing
        expr: up{job="app-metrics"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "アプリケーションのヘルスチェックが失敗"
          description: "{{ $labels.instance }} が2分間応答していません"

      # レスポンスタイム劣化
      - alert: HighResponseTime
        expr: >
          histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "95パーセンタイルレスポンスタイムが1秒超"
```

### Alertmanager設定

```yaml
# alertmanager/alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ["alertname", "cluster"]
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: "slack-notifications"

  routes:
    - match:
        severity: critical
      receiver: "pagerduty"
      repeat_interval: 1h
    - match:
        severity: warning
      receiver: "slack-notifications"

receivers:
  - name: "slack-notifications"
    slack_configs:
      - api_url: "${SLACK_WEBHOOK_URL}"
        channel: "#alerts"
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'

  - name: "pagerduty"
    pagerduty_configs:
      - service_key: "${PAGERDUTY_SERVICE_KEY}"
        severity: '{{ if eq .GroupLabels.severity "critical" }}critical{{ else }}warning{{ end }}'
```

---

## 4. Grafana ダッシュボード

### コード例4: Grafana プロビジョニング設定

```yaml
# grafana/provisioning/datasources/datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false
```

```yaml
# grafana/provisioning/dashboards/dashboards.yml
apiVersion: 1

providers:
  - name: "Docker Monitoring"
    orgId: 1
    folder: "Docker"
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
```

### 重要なPromQLクエリ集

```promql
# === CPU メトリクス ===
# コンテナ別CPU使用率（%）
sum(rate(container_cpu_usage_seconds_total{name=~".+"}[5m])) by (name) * 100

# === メモリメトリクス ===
# コンテナ別メモリ使用量
container_memory_usage_bytes{name=~".+"} / 1024 / 1024  # MB単位

# メモリ使用率（%）
container_memory_usage_bytes{name=~".+"} / container_spec_memory_limit_bytes{name=~".+"} * 100

# === ネットワークメトリクス ===
# 受信バイト数（毎秒）
sum(rate(container_network_receive_bytes_total{name=~".+"}[5m])) by (name)

# 送信バイト数（毎秒）
sum(rate(container_network_transmit_bytes_total{name=~".+"}[5m])) by (name)

# === ディスク I/O ===
# 読み取りバイト数（毎秒）
sum(rate(container_fs_reads_bytes_total{name=~".+"}[5m])) by (name)

# 書き込みバイト数（毎秒）
sum(rate(container_fs_writes_bytes_total{name=~".+"}[5m])) by (name)
```

### ダッシュボードレイアウト

```
┌─────────────────────────────────────────────────────────┐
│  Docker Monitoring Dashboard                    [24h ▼] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│  │ Containers│ │  CPU   │ │ Memory  │ │ Alerts  │    │
│  │    12    │ │  34%   │ │  62%    │ │    2    │    │
│  │ running  │ │ avg    │ │ avg     │ │ active  │    │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘    │
│                                                         │
│  CPU Usage by Container            Memory Usage         │
│  ┌─────────────────────┐   ┌─────────────────────┐    │
│  │  ████               │   │  ██████             │    │
│  │  ██████████         │   │  ████████████       │    │
│  │  ████████           │   │  ██████████         │    │
│  │  ──────── time ──►  │   │  ──────── time ──►  │    │
│  └─────────────────────┘   └─────────────────────┘    │
│                                                         │
│  Network I/O                    Container Restarts      │
│  ┌─────────────────────┐   ┌─────────────────────┐    │
│  │  rx: ────           │   │  api: 0             │    │
│  │  tx: ----           │   │  web: 2  ← 要注意   │    │
│  │                     │   │  db:  0             │    │
│  └─────────────────────┘   └─────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 5. ログ集約

### Loki vs ELK 比較表

| 特性 | Grafana Loki | ELK Stack |
|------|-------------|-----------|
| アーキテクチャ | 軽量（ラベルのみインデックス） | 全文インデックス |
| リソース消費 | 低い | 高い（Elasticsearch） |
| クエリ言語 | LogQL | KQL / Lucene |
| スケーラビリティ | 水平スケーリング容易 | 管理が複雑 |
| Grafana連携 | ネイティブ | プラグイン |
| セットアップ | 簡単 | 複雑 |
| 検索速度 | ラベルベース高速 | 全文検索高速 |
| 適用規模 | 中小〜中規模 | 大規模 |

### コード例5: Loki + Promtail構成

```yaml
# docker-compose.logging.yml
version: "3.9"

services:
  loki:
    image: grafana/loki:2.9.6
    container_name: loki
    restart: unless-stopped
    command: -config.file=/etc/loki/local-config.yaml
    volumes:
      - ./loki/loki-config.yaml:/etc/loki/local-config.yaml:ro
      - loki-data:/loki
    ports:
      - "3100:3100"
    networks:
      - monitoring

  promtail:
    image: grafana/promtail:2.9.6
    container_name: promtail
    restart: unless-stopped
    command: -config.file=/etc/promtail/config.yml
    volumes:
      - ./promtail/config.yml:/etc/promtail/config.yml:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    networks:
      - monitoring

volumes:
  loki-data:

networks:
  monitoring:
    external: true
```

```yaml
# loki/loki-config.yaml
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2024-01-01
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

limits_config:
  retention_period: 30d
  max_query_length: 720h

compactor:
  working_directory: /loki/retention
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
```

```yaml
# promtail/config.yml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      # コンテナ名をラベルとして付与
      - source_labels: ["__meta_docker_container_name"]
        target_label: "container"
        regex: "/(.+)"
      # Composeサービス名
      - source_labels: ["__meta_docker_container_label_com_docker_compose_service"]
        target_label: "service"
      # Composeプロジェクト名
      - source_labels: ["__meta_docker_container_label_com_docker_compose_project"]
        target_label: "project"
    pipeline_stages:
      # JSON ログのパース
      - json:
          expressions:
            level: level
            message: message
            timestamp: timestamp
      - labels:
          level:
      - timestamp:
          source: timestamp
          format: RFC3339
```

### LogQLクエリ例

```logql
# 特定コンテナのログを表示
{container="api"} |= "error"

# JSON構造化ログのフィルタリング
{service="api"} | json | level="error" | status >= 500

# エラーログの発生率
rate({service="api"} |= "error" [5m])

# レスポンスタイムの統計
{service="api"} | json | unwrap duration_ms | quantile_over_time(0.95, [5m])
```

---

## 6. アプリケーションメトリクスの計装

### コード例6: Prometheusクライアントライブラリ（Node.js）

```javascript
// metrics.js - Prometheus メトリクスの計装
const client = require("prom-client");

// デフォルトメトリクス（CPU, メモリ, GC等）
client.collectDefaultMetrics({ prefix: "app_" });

// カスタムメトリクス
const httpRequestDuration = new client.Histogram({
  name: "http_request_duration_seconds",
  help: "HTTPリクエストの処理時間",
  labelNames: ["method", "route", "status_code"],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
});

const httpRequestTotal = new client.Counter({
  name: "http_requests_total",
  help: "HTTPリクエストの総数",
  labelNames: ["method", "route", "status_code"],
});

const activeConnections = new client.Gauge({
  name: "http_active_connections",
  help: "現在のアクティブ接続数",
});

// Express ミドルウェア
function metricsMiddleware(req, res, next) {
  const end = httpRequestDuration.startTimer();
  activeConnections.inc();

  res.on("finish", () => {
    const labels = {
      method: req.method,
      route: req.route?.path || req.path,
      status_code: res.statusCode,
    };
    end(labels);
    httpRequestTotal.inc(labels);
    activeConnections.dec();
  });

  next();
}

// /metrics エンドポイント
async function metricsHandler(req, res) {
  res.set("Content-Type", client.register.contentType);
  res.end(await client.register.metrics());
}

module.exports = { metricsMiddleware, metricsHandler };
```

---

## アンチパターン

### アンチパターン1: モニタリングなしの本番運用

```yaml
# NG: アプリケーションだけデプロイ
services:
  app:
    image: my-app:latest
    ports:
      - "80:80"
# → 障害が発生しても気づけない、原因調査もできない

# OK: 監視スタックを同時にデプロイ
services:
  app:
    image: my-app:latest
  prometheus:
    image: prom/prometheus:latest
  grafana:
    image: grafana/grafana:latest
  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
```

**なぜ問題か**: 「観測できないものは管理できない」。障害検知が遅延し、MTTR（平均修復時間）が増大する。

### アンチパターン2: アラートの設定不足または過剰

```yaml
# NG: 閾値が低すぎてアラート疲れ
- alert: HighCPU
  expr: container_cpu_usage > 50  # 50%で発報 → 常にアラート
  for: 1m                         # 1分は短すぎる

# OK: 適切な閾値と持続時間
- alert: HighCPU
  expr: container_cpu_usage > 85  # 85%で発報
  for: 5m                         # 5分間継続した場合のみ
  labels:
    severity: warning              # 重要度を適切に設定
```

**なぜ問題か**: アラート過多は「アラート疲れ」を引き起こし、本当に重要なアラートが見過ごされる。逆に設定不足では障害を検知できない。

---

## FAQ

### Q1: PrometheusのPull型とPush型の違いは？

Prometheusはデフォルトで**Pull型**（サーバーがターゲットからメトリクスを取得する）を採用。一方、短命なバッチジョブ等には**Pushgateway**を使ってPush型も可能。Pull型の利点はターゲットの死活監視が自動的にできること、Pushgatewayが単一障害点にならないよう注意が必要。

### Q2: メトリクスの保持期間はどの程度が適切か？

一般的な指針:
- **高解像度（15秒間隔）**: 7-15日
- **中解像度（1分間隔にダウンサンプリング）**: 30-90日
- **低解像度（5分間隔）**: 1年以上

ストレージコストと分析需要のバランスで決定する。長期保存にはThanosやCortexなどのリモートストレージを検討。

### Q3: cAdvisorとDocker Engine Metricsの違いは？

cAdvisorはGoogleが開発したコンテナ特化のメトリクス収集ツールで、CPU/メモリ/ネットワーク/ファイルシステムの詳細なメトリクスを提供する。Docker Engine Metricsは実験的機能で、よりシンプルなメトリクスのみ。本番環境ではcAdvisorを推奨。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Prometheus | Pull型メトリクス収集。PromQLで柔軟なクエリ |
| cAdvisor | コンテナリソースメトリクスの収集。必須コンポーネント |
| Grafana | 統一ダッシュボード。Prometheus/Lokiと連携 |
| Alertmanager | アラートルーティング。重要度別に通知先を分離 |
| Loki | 軽量ログ集約。ラベルベースのインデックス |
| 計装 | アプリにPrometheusクライアントを組み込み。/metricsエンドポイント |
| アラート設計 | 適切な閾値と持続時間。アラート疲れを防ぐ |

---

## 次に読むべきガイド

- [Docker CI/CD](./02-ci-cd-docker.md) -- デプロイパイプラインへの監視統合
- [Kubernetes基礎](../05-orchestration/01-kubernetes-basics.md) -- K8s環境のモニタリング
- [本番ベストプラクティス](./00-production-best-practices.md) -- ヘルスチェックとログ戦略

---

## 参考文献

1. Prometheus公式ドキュメント -- https://prometheus.io/docs/
2. Grafana Loki公式ドキュメント -- https://grafana.com/docs/loki/latest/
3. Google cAdvisor GitHub -- https://github.com/google/cadvisor
4. Brian Brazil (2018) *Prometheus: Up & Running*, O'Reilly
5. Grafana Labs "Docker monitoring with Grafana" -- https://grafana.com/docs/grafana-cloud/monitor-infrastructure/integrations/integration-reference/integration-docker/
