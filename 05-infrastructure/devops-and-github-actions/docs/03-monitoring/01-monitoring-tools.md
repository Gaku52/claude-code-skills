# 監視ツール

> Datadog、Grafana、CloudWatch の特徴を理解し、システム規模と要件に応じた最適な監視基盤を構築する

## この章で学ぶこと

1. **Grafana + Prometheus によるOSS監視スタック** — メトリクス収集、ダッシュボード構築、PromQL の活用
2. **Datadog によるフルスタック監視** — SaaS ベースの統合監視プラットフォームの活用
3. **AWS CloudWatch による AWS ネイティブ監視** — メトリクス、ログ、アラーム、ダッシュボードの構成
4. **Grafana Loki によるログ集約** — ラベルベースの軽量ログ基盤の設計と LogQL クエリ
5. **長期保存とスケーリング** — Thanos・Cortex・Mimir によるマルチクラスタ監視の実現

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

### 1.1 監視ツール選定フレームワーク

監視ツールの選定は、組織のフェーズ・チーム規模・技術スタックに大きく依存する。以下のフレームワークで判断する。

```
選定判断フロー:

  ┌───────────────────┐
  │ AWS のみ使用？     │
  └────────┬──────────┘
           │
     ┌─────┼─────┐
     Yes         No
     │           │
     ▼           ▼
  CloudWatch   ┌───────────────────┐
  で十分か？    │ 専任 SRE チーム     │
     │         │ がいるか？          │
   ┌─┼─┐       └────────┬──────────┘
   Yes No            ┌───┼────┐
   │   │             Yes      No
   ▼   ▼             │        │
  CW   CW +          ▼        ▼
  のみ  Grafana      OSS      SaaS
       連携         スタック   (Datadog等)
```

| 判断基準 | OSS (Prometheus+Grafana) | Datadog | CloudWatch |
|----------|--------------------------|---------|------------|
| チーム規模 | SRE 2名以上 | 少人数でも可 | AWS チームなら即時 |
| 月額コスト | インフラ費のみ ($100-500) | $500-$10,000+ | $100-$500 |
| カスタマイズ性 | 非常に高い | 高い | 中程度 |
| 運用負荷 | 高い | 低い | 低い |
| マルチクラウド | 得意 | 得意 | AWS のみ |
| 立ち上げ速度 | 遅い (1-2週間) | 速い (数時間) | 最速 (即時) |
| ベンダーロックイン | なし | あり | あり (AWS) |

### 1.2 監視成熟度モデル

```
Level 0 — なし
  監視なし。障害はユーザー報告で気づく。

Level 1 — 基本的なインフラ監視
  CPU/メモリ/ディスクの閾値ベースアラート。
  ツール: CloudWatch 基本メトリクス、Zabbix

Level 2 — アプリケーション監視
  APM 導入。レイテンシ・エラー率・スループットを計測。
  ツール: Prometheus + Grafana、Datadog APM

Level 3 — SLO ベース監視
  SLI/SLO を定義し、エラーバジェットでアラート。
  ツール: Prometheus + バーンレートアラート、Datadog SLO

Level 4 — フルオブザーバビリティ
  ログ・メトリクス・トレースが統合。
  任意のリクエストを追跡可能。
  ツール: Grafana Stack (Prometheus+Loki+Tempo)、Datadog

Level 5 — 予測的監視
  異常検知・予測アラート。
  キャパシティプランニングの自動化。
  ツール: Datadog Watchdog、ML ベースの異常検知
```

---

## 2. Prometheus + Grafana スタック

### 2.1 Docker Compose による統合監視環境

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
      - ./alerts/:/etc/prometheus/alerts/
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.retention.size=10GB'
      - '--web.enable-lifecycle'            # /-/reload で設定リロード
      - '--web.enable-admin-api'            # 管理API有効化
      - '--storage.tsdb.wal-compression'    # WAL圧縮でディスク節約
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.3.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-piechart-panel
      - GF_AUTH_ANONYMOUS_ENABLED=true
      - GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer
      - GF_DASHBOARDS_DEFAULT_HOME_DASHBOARD_PATH=/var/lib/grafana/dashboards/home.json
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
      - loki
    restart: unless-stopped
    networks:
      - monitoring

  loki:
    image: grafana/loki:2.9.0
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      - loki-data:/loki
    command: -config.file=/etc/loki/local-config.yaml
    restart: unless-stopped
    networks:
      - monitoring

  promtail:
    image: grafana/promtail:2.9.0
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
    command: -config.file=/etc/promtail/config.yml
    depends_on:
      - loki
    restart: unless-stopped
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:v1.7.0
    ports:
      - "9100:9100"
    pid: host
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/rootfs'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    restart: unless-stopped
    networks:
      - monitoring

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:v0.47.0
    ports:
      - "8080:8080"
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    privileged: true
    restart: unless-stopped
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:v0.27.0
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    restart: unless-stopped
    networks:
      - monitoring

volumes:
  prometheus-data:
  grafana-data:
  loki-data:

networks:
  monitoring:
    driver: bridge
```

### 2.2 Prometheus 設定の詳細

```yaml
# prometheus.yml — Prometheus 設定
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  scrape_timeout: 10s
  external_labels:
    cluster: production
    region: ap-northeast-1

rule_files:
  - "alerts/*.yml"

# Alertmanager との連携
alerting:
  alertmanagers:
    - static_configs:
        - targets: ["alertmanager:9093"]

scrape_configs:
  # Prometheus 自身の監視
  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]

  # Node Exporter (サーバーメトリクス)
  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]

  # cAdvisor (コンテナメトリクス)
  - job_name: "cadvisor"
    static_configs:
      - targets: ["cadvisor:8080"]

  # アプリケーション (Express/Fastify)
  - job_name: "app"
    metrics_path: /metrics
    scrape_interval: 10s
    static_configs:
      - targets: ["app:3000"]
        labels:
          service: "order-service"
          environment: "production"

  # マルチターゲットの例 (複数サービス)
  - job_name: "microservices"
    scrape_interval: 10s
    static_configs:
      - targets: ["user-service:3001"]
        labels:
          service: "user-service"
      - targets: ["payment-service:3002"]
        labels:
          service: "payment-service"
      - targets: ["inventory-service:3003"]
        labels:
          service: "inventory-service"

  # Kubernetes Service Discovery
  - job_name: "kubernetes-pods"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      # prometheus.io/scrape: "true" アノテーションを持つ Pod のみ対象
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      # カスタムメトリクスパスの指定
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
        action: replace
        target_label: __metrics_path__
        regex: (.+)
      # カスタムポートの指定
      - source_labels: [__address__, __meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: ([^:]+)(?::\d+)?;(\d+)
        replacement: $1:$2
      # Pod のラベルをメトリクスラベルに転写
      - action: labelmap
        regex: __meta_kubernetes_pod_label_(.+)
      # Namespace と Pod 名を追加
      - source_labels: [__meta_kubernetes_namespace]
        action: replace
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_pod_name]
        action: replace
        target_label: kubernetes_pod_name

  # Kubernetes Service Discovery — Service レベル
  - job_name: "kubernetes-services"
    kubernetes_sd_configs:
      - role: service
    metrics_path: /probe
    params:
      module: [http_2xx]
    relabel_configs:
      - source_labels: [__meta_kubernetes_service_annotation_prometheus_io_probe]
        action: keep
        regex: true
      - source_labels: [__address__]
        target_label: __param_target
      - target_label: __address__
        replacement: blackbox-exporter:9115
      - source_labels: [__param_target]
        target_label: instance
      - source_labels: [__meta_kubernetes_namespace]
        target_label: kubernetes_namespace
      - source_labels: [__meta_kubernetes_service_name]
        target_label: kubernetes_service_name

  # EC2 Auto Discovery (AWS)
  - job_name: "ec2-instances"
    ec2_sd_configs:
      - region: ap-northeast-1
        port: 9100
        filters:
          - name: "tag:monitoring"
            values: ["enabled"]
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance_name
      - source_labels: [__meta_ec2_instance_id]
        target_label: instance_id
      - source_labels: [__meta_ec2_availability_zone]
        target_label: availability_zone

  # Blackbox Exporter (外形監視)
  - job_name: "blackbox-http"
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - https://api.example.com/health
          - https://www.example.com
          - https://admin.example.com
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### 2.3 Recording Rules による計算の事前集約

```yaml
# recording-rules.yml — よく使うクエリを事前計算
groups:
  - name: http_request_rules
    interval: 30s
    rules:
      # リクエストレート (サービス・メソッド・ステータス別)
      - record: service:http_requests:rate5m
        expr: sum(rate(http_requests_total[5m])) by (service, method, status_class)

      # エラーレート (サービス別)
      - record: service:http_error_rate:ratio_rate5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
          /
          sum(rate(http_requests_total[5m])) by (service)

      # p50/p95/p99 レイテンシ (サービス別)
      - record: service:http_request_duration_seconds:p50
        expr: |
          histogram_quantile(0.5,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:http_request_duration_seconds:p95
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

      - record: service:http_request_duration_seconds:p99
        expr: |
          histogram_quantile(0.99,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (service, le)
          )

  - name: node_rules
    interval: 60s
    rules:
      # CPU 使用率 (インスタンス別)
      - record: instance:node_cpu_utilization:ratio
        expr: |
          1 - avg by(instance) (
            irate(node_cpu_seconds_total{mode="idle"}[5m])
          )

      # メモリ使用率 (インスタンス別)
      - record: instance:node_memory_utilization:ratio
        expr: |
          1 - (
            node_memory_MemAvailable_bytes
            /
            node_memory_MemTotal_bytes
          )

      # ディスク使用率 (インスタンス・マウントポイント別)
      - record: instance:node_filesystem_utilization:ratio
        expr: |
          1 - (
            node_filesystem_avail_bytes{fstype!~"tmpfs|overlay"}
            /
            node_filesystem_size_bytes{fstype!~"tmpfs|overlay"}
          )

      # ネットワーク受信/送信レート
      - record: instance:node_network_receive_bytes:rate5m
        expr: sum(rate(node_network_receive_bytes_total{device!~"lo|veth.*|docker.*|br-.*"}[5m])) by (instance)

      - record: instance:node_network_transmit_bytes:rate5m
        expr: sum(rate(node_network_transmit_bytes_total{device!~"lo|veth.*|docker.*|br-.*"}[5m])) by (instance)
```

---

## 3. PromQL の基本と応用

### 3.1 基本クエリ

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

### 3.2 PromQL データモデルの詳細

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

  rate() vs irate():
  ┌─────────┬────────────────────────────────────────────┐
  │ rate()  │ 指定範囲全体の平均増加率。安定したグラフ向き  │
  │ irate() │ 直近2点の瞬間増加率。スパイク検知向き        │
  │ 推奨    │ アラートには rate()、ダッシュボードに irate()  │
  └─────────┴────────────────────────────────────────────┘
```

### 3.3 応用クエリパターン

```promql
# --- SLO 関連のクエリ ---

# SLO バーンレート (Multi-window)
# 短期ウィンドウ (5m) × 長期ウィンドウ (1h) の組み合わせ
(
  sum(rate(http_requests_total{status=~"5.."}[5m]))
  / sum(rate(http_requests_total[5m]))
) > (14.4 * 0.001)
and
(
  sum(rate(http_requests_total{status=~"5.."}[1h]))
  / sum(rate(http_requests_total[1h]))
) > (14.4 * 0.001)

# 可用性 SLI (30日間)
1 - (
  sum(increase(http_requests_total{status=~"5.."}[30d]))
  /
  sum(increase(http_requests_total[30d]))
)

# エラーバジェット残量 (%)
(1 - (
  sum(increase(http_requests_total{status=~"5.."}[30d]))
  /
  sum(increase(http_requests_total[30d]))
  /
  (1 - 0.999)  # SLO: 99.9%
)) * 100

# --- キャパシティプランニング ---

# ディスク枯渇予測 (24時間後の予測値)
predict_linear(
  node_filesystem_avail_bytes{mountpoint="/"}[6h],
  24 * 3600
)

# メモリ枯渇予測 (4時間後)
predict_linear(
  node_memory_MemAvailable_bytes[1h],
  4 * 3600
) < 0

# CPU 使用率のトレンド (1週間分の線形回帰)
predict_linear(
  instance:node_cpu_utilization:ratio[7d],
  30 * 24 * 3600  # 30日後
)

# --- トップN 分析 ---

# レイテンシが最も高い上位5エンドポイント
topk(5,
  histogram_quantile(0.95,
    sum(rate(http_request_duration_seconds_bucket[5m])) by (le, handler)
  )
)

# リクエスト数が最も多い上位10エンドポイント
topk(10,
  sum(rate(http_requests_total[5m])) by (handler)
)

# エラーレートが最も高い上位5サービス
topk(5,
  sum(rate(http_requests_total{status=~"5.."}[5m])) by (service)
  /
  sum(rate(http_requests_total[5m])) by (service)
)

# --- 前日比・前週比 ---

# リクエスト数の前日比
sum(rate(http_requests_total[1h]))
/
sum(rate(http_requests_total[1h] offset 1d))

# エラーレートの前週比
(
  sum(rate(http_requests_total{status=~"5.."}[1h]))
  / sum(rate(http_requests_total[1h]))
)
/
(
  sum(rate(http_requests_total{status=~"5.."}[1h] offset 7d))
  / sum(rate(http_requests_total[1h] offset 7d))
)

# --- コンテナ監視 ---

# コンテナ CPU 使用率 (%)
sum(rate(container_cpu_usage_seconds_total{name!=""}[5m])) by (name) * 100

# コンテナ メモリ使用量 (MB)
container_memory_working_set_bytes{name!=""} / 1024 / 1024

# コンテナ ネットワーク I/O (bytes/sec)
sum(rate(container_network_receive_bytes_total{name!=""}[5m])) by (name)

# Pod の再起動回数
sum(kube_pod_container_status_restarts_total) by (namespace, pod)
```

---

## 4. Grafana ダッシュボードの設計と管理

### 4.1 Grafana Provisioning によるダッシュボード管理

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
    jsonData:
      timeInterval: "15s"
      httpMethod: POST

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false
    jsonData:
      maxLines: 1000
      derivedFields:
        - datasourceUid: tempo
          matcherRegex: "traceID=(\\w+)"
          name: TraceID
          url: "$${__value.raw}"

  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    uid: tempo
    editable: false

  - name: Alertmanager
    type: alertmanager
    access: proxy
    url: http://alertmanager:9093
    editable: false
    jsonData:
      implementation: prometheus
```

```yaml
# grafana/provisioning/dashboards/dashboards.yml
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: 'Infrastructure'
    type: file
    disableDeletion: true
    editable: false
    updateIntervalSeconds: 30
    allowUiUpdates: false
    options:
      path: /var/lib/grafana/dashboards/infrastructure
      foldersFromFilesStructure: true

  - name: 'applications'
    orgId: 1
    folder: 'Applications'
    type: file
    disableDeletion: true
    editable: false
    options:
      path: /var/lib/grafana/dashboards/applications
      foldersFromFilesStructure: true

  - name: 'slo'
    orgId: 1
    folder: 'SLO'
    type: file
    disableDeletion: true
    editable: false
    options:
      path: /var/lib/grafana/dashboards/slo
```

### 4.2 Grafana ダッシュボード JSON の構造

```json
{
  "dashboard": {
    "title": "サービスヘルスダッシュボード",
    "uid": "service-health-main",
    "tags": ["service", "slo", "production"],
    "timezone": "Asia/Tokyo",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "templating": {
      "list": [
        {
          "name": "service",
          "type": "query",
          "datasource": "Prometheus",
          "query": "label_values(http_requests_total, service)",
          "refresh": 2,
          "multi": true,
          "includeAll": true,
          "current": { "text": "All", "value": "$__all" }
        },
        {
          "name": "environment",
          "type": "custom",
          "options": [
            { "text": "production", "value": "production" },
            { "text": "staging", "value": "staging" }
          ],
          "current": { "text": "production", "value": "production" }
        }
      ]
    },
    "panels": [
      {
        "title": "リクエストレート",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{service=~\"$service\"}[5m])) by (service)",
            "legendFormat": "{{service}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "custom": {
              "drawStyle": "line",
              "lineWidth": 2,
              "fillOpacity": 10
            }
          }
        }
      },
      {
        "title": "エラーレート (%)",
        "type": "stat",
        "gridPos": { "h": 4, "w": 6, "x": 12, "y": 0 },
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{service=~\"$service\",status=~\"5..\"}[5m])) / sum(rate(http_requests_total{service=~\"$service\"}[5m])) * 100"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                { "color": "green", "value": null },
                { "color": "yellow", "value": 0.1 },
                { "color": "red", "value": 1 }
              ]
            }
          }
        }
      },
      {
        "title": "レイテンシ (p50/p95/p99)",
        "type": "timeseries",
        "gridPos": { "h": 8, "w": 12, "x": 0, "y": 8 },
        "targets": [
          {
            "expr": "histogram_quantile(0.5, sum(rate(http_request_duration_seconds_bucket{service=~\"$service\"}[5m])) by (le))",
            "legendFormat": "p50"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service=~\"$service\"}[5m])) by (le))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket{service=~\"$service\"}[5m])) by (le))",
            "legendFormat": "p99"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "custom": { "drawStyle": "line", "lineWidth": 2 }
          }
        }
      }
    ]
  }
}
```

### 4.3 Terraform による Grafana ダッシュボード管理

```hcl
# grafana.tf — Terraform で Grafana ダッシュボード管理
terraform {
  required_providers {
    grafana = {
      source  = "grafana/grafana"
      version = "~> 2.0"
    }
  }
}

provider "grafana" {
  url  = "https://grafana.example.com"
  auth = var.grafana_api_key
}

# フォルダ作成
resource "grafana_folder" "infrastructure" {
  title = "Infrastructure"
}

resource "grafana_folder" "applications" {
  title = "Applications"
}

resource "grafana_folder" "slo" {
  title = "SLO Dashboards"
}

# ダッシュボード (JSON ファイルから読み込み)
resource "grafana_dashboard" "service_health" {
  folder    = grafana_folder.applications.id
  overwrite = true

  config_json = file("${path.module}/dashboards/service-health.json")
}

resource "grafana_dashboard" "node_overview" {
  folder    = grafana_folder.infrastructure.id
  overwrite = true

  config_json = file("${path.module}/dashboards/node-overview.json")
}

# データソース
resource "grafana_data_source" "prometheus" {
  type = "prometheus"
  name = "Prometheus"
  url  = "http://prometheus:9090"

  json_data_encoded = jsonencode({
    timeInterval = "15s"
    httpMethod   = "POST"
  })

  is_default = true
}

resource "grafana_data_source" "loki" {
  type = "loki"
  name = "Loki"
  url  = "http://loki:3100"

  json_data_encoded = jsonencode({
    maxLines = 1000
  })
}

# アラートルール
resource "grafana_rule_group" "slo_alerts" {
  name             = "SLO Alerts"
  folder_uid       = grafana_folder.slo.uid
  interval_seconds = 60

  rule {
    name      = "High Error Rate"
    condition = "C"

    data {
      ref_id = "A"
      relative_time_range {
        from = 300
        to   = 0
      }
      datasource_uid = grafana_data_source.prometheus.uid
      model = jsonencode({
        expr = "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m]))"
      })
    }

    data {
      ref_id = "C"
      relative_time_range {
        from = 0
        to   = 0
      }
      datasource_uid = "__expr__"
      model = jsonencode({
        type       = "threshold"
        conditions = [{ evaluator = { type = "gt", params = [0.01] } }]
      })
    }
  }
}

# 通知ポリシー
resource "grafana_notification_policy" "default" {
  contact_point = grafana_contact_point.slack.name
  group_by      = ["alertname", "service"]

  policy {
    matcher {
      label = "severity"
      match = "="
      value = "critical"
    }
    contact_point = grafana_contact_point.pagerduty.name
  }
}

resource "grafana_contact_point" "slack" {
  name = "Slack"

  slack {
    url     = var.slack_webhook_url
    channel = "#alerts"
  }
}

resource "grafana_contact_point" "pagerduty" {
  name = "PagerDuty"

  pagerduty {
    integration_key = var.pagerduty_key
    severity        = "critical"
  }
}
```

### 4.4 ダッシュボード階層設計

```
ダッシュボードの階層設計:

  Level 0: Executive Overview (経営層)
  ┌────────────────────────────────────────┐
  │ ・全サービスの稼働率 (SLA 達成状況)     │
  │ ・月間インシデント数と MTTR             │
  │ ・エラーバジェット消費率               │
  │ ・トラフィックトレンド (前月比)         │
  └────────────────────────────────────────┘
         │
         ▼
  Level 1: Service Overview (チームリード)
  ┌────────────────────────────────────────┐
  │ ・サービス別の RED メトリクス            │
  │   (Rate / Error / Duration)            │
  │ ・SLO 達成状況とバーンレート            │
  │ ・直近のデプロイとその影響              │
  │ ・依存サービスのヘルス状態              │
  └────────────────────────────────────────┘
         │
         ▼
  Level 2: Technical Detail (エンジニア)
  ┌────────────────────────────────────────┐
  │ ・エンドポイント別レイテンシ分布        │
  │ ・DB クエリパフォーマンス               │
  │ ・キャッシュヒット率                    │
  │ ・Pod/コンテナのリソース使用量          │
  │ ・外部 API 呼び出しのレイテンシ         │
  └────────────────────────────────────────┘
         │
         ▼
  Level 3: Debug (障害調査)
  ┌────────────────────────────────────────┐
  │ ・トレース一覧と Span 詳細              │
  │ ・ログストリーム (LogQL)                │
  │ ・ネットワークレイテンシ               │
  │ ・Goroutine / Thread dump              │
  └────────────────────────────────────────┘
```

---

## 5. Grafana Loki によるログ集約

### 5.1 Loki 設定

```yaml
# loki-config.yml — Loki サーバー設定
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9096
  log_level: warn

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
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20
  max_streams_per_user: 10000
  reject_old_samples: true
  reject_old_samples_max_age: 168h  # 7日
  retention_period: 744h            # 31日

storage_config:
  tsdb_shipper:
    active_index_directory: /loki/tsdb-index
    cache_location: /loki/tsdb-cache

compactor:
  working_directory: /loki/compactor
  compaction_interval: 10m
  retention_enabled: true
  retention_delete_delay: 2h
  retention_delete_worker_count: 150
```

### 5.2 Promtail 設定

```yaml
# promtail-config.yml — ログ収集エージェント設定
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push
    batchwait: 1s
    batchsize: 1048576  # 1MB
    tenant_id: default

scrape_configs:
  # Docker コンテナログ
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        regex: '/(.*)'
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'stream'
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'
    pipeline_stages:
      # JSON ログのパース
      - json:
          expressions:
            level: level
            msg: msg
            timestamp: timestamp
            traceId: traceId
      - labels:
          level:
          traceId:
      - timestamp:
          source: timestamp
          format: RFC3339Nano
      # 機密情報のマスク
      - replace:
          expression: '(password|token|secret|api_key)=\S+'
          replace: '$1=***REDACTED***'

  # システムログ (/var/log)
  - job_name: system
    static_configs:
      - targets: [localhost]
        labels:
          job: system
          __path__: /var/log/*.log
    pipeline_stages:
      - regex:
          expression: '^(?P<timestamp>\S+ \S+) (?P<hostname>\S+) (?P<service>\S+)\[(?P<pid>\d+)\]: (?P<message>.*)$'
      - labels:
          hostname:
          service:
      - timestamp:
          source: timestamp
          format: "2006-01-02T15:04:05.000Z"

  # Kubernetes Pod ログ
  - job_name: kubernetes-pods
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        target_label: app
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - source_labels: [__meta_kubernetes_pod_container_name]
        target_label: container
    pipeline_stages:
      - cri: {}
      - json:
          expressions:
            level: level
            msg: msg
      - labels:
          level:
```

### 5.3 LogQL クエリの実践

```logql
# --- 基本的な LogQL クエリ ---

# サービス名でフィルタ
{service="order-service"}

# 複数条件のフィルタ
{service="order-service", level="error"}

# ログ内容で絞り込み (パイプライン)
{service="order-service"} |= "payment failed"

# 正規表現フィルタ
{service="order-service"} |~ "timeout|connection refused"

# 除外フィルタ
{service="order-service"} != "healthcheck" !~ "GET /health"

# --- JSON パース ---

# JSON ログのフィールド抽出
{service="order-service"} | json | level="error"

# 特定フィールドの値でフィルタ
{service="order-service"} | json | status_code >= 500

# フィールド値をラベルとして使用
{service="order-service"} | json | line_format "{{.method}} {{.path}} {{.status_code}} {{.duration}}ms"

# --- メトリクスクエリ (Log-based Metrics) ---

# エラーログの発生レート
rate({service="order-service", level="error"}[5m])

# サービス別のログ量 (bytes/sec)
sum(bytes_rate({job="docker"}[5m])) by (service)

# エラーメッセージの Top 10
topk(10,
  sum(count_over_time({service="order-service", level="error"}[1h]))
  by (msg)
)

# レイテンシの p95 (JSON ログからパース)
quantile_over_time(0.95,
  {service="order-service"} | json | unwrap duration [5m]
) by (method, path)

# 特定のエラーパターンの出現回数
sum(count_over_time(
  {service="order-service"} |= "database connection" |= "timeout" [1h]
))

# --- コンテキスト調査 ---

# 特定のトレース ID に紐づくログ
{traceId="abc123def456"}

# 特定の時間範囲でのエラーログ
{service="order-service", level="error"}
  | json
  | timestamp >= "2025-03-15T14:30:00Z"
  | timestamp <= "2025-03-15T15:00:00Z"

# 特定ユーザーの操作ログ (userId をパース)
{service="order-service"} | json | userId="user-12345"
```

---

## 6. Datadog による統合監視

### 6.1 Datadog APM セットアップ

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

### 6.2 Datadog Agent の Kubernetes デプロイ

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
      serviceAccountName: datadog-agent
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
            - name: DD_APM_NON_LOCAL_TRAFFIC
              value: "true"
            - name: DD_LOGS_ENABLED
              value: "true"
            - name: DD_LOGS_CONFIG_CONTAINER_COLLECT_ALL
              value: "true"
            - name: DD_PROCESS_AGENT_ENABLED
              value: "true"
            - name: DD_DOGSTATSD_NON_LOCAL_TRAFFIC
              value: "true"
            - name: DD_CLUSTER_AGENT_ENABLED
              value: "true"
            - name: DD_CLUSTER_AGENT_AUTH_TOKEN
              valueFrom:
                secretKeyRef:
                  name: datadog-secrets
                  key: cluster-agent-token
            # Kubernetes イベント収集
            - name: DD_KUBERNETES_EVENTS_ENABLED
              value: "true"
            # Network Performance Monitoring
            - name: DD_SYSTEM_PROBE_ENABLED
              value: "true"
            - name: DD_SYSTEM_PROBE_NETWORK_ENABLED
              value: "true"
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          ports:
            - containerPort: 8125
              name: dogstatsd
              protocol: UDP
            - containerPort: 8126
              name: apm
              protocol: TCP
          volumeMounts:
            - name: dockersocket
              mountPath: /var/run/docker.sock
            - name: procdir
              mountPath: /host/proc
              readOnly: true
            - name: cgroups
              mountPath: /host/sys/fs/cgroup
              readOnly: true
      volumes:
        - name: dockersocket
          hostPath:
            path: /var/run/docker.sock
        - name: procdir
          hostPath:
            path: /proc
        - name: cgroups
          hostPath:
            path: /sys/fs/cgroup
```

### 6.3 Datadog カスタムメトリクスの送信

```typescript
// datadog-metrics.ts — カスタムメトリクスの送信
import StatsD from 'hot-shots';

const dogstatsd = new StatsD({
  host: process.env.DD_AGENT_HOST ?? 'localhost',
  port: 8125,
  prefix: 'myapp.',
  globalTags: [
    `env:${process.env.NODE_ENV}`,
    `service:order-service`,
    `version:${process.env.APP_VERSION}`,
  ],
  errorHandler: (error) => {
    console.error('StatsD error:', error);
  },
});

// ビジネスメトリクスの例
class BusinessMetrics {
  // 注文作成
  recordOrderCreated(paymentMethod: string, amount: number): void {
    dogstatsd.increment('orders.created', 1, [
      `payment_method:${paymentMethod}`,
    ]);
    dogstatsd.histogram('orders.amount', amount, [
      `payment_method:${paymentMethod}`,
    ]);
  }

  // 注文キャンセル
  recordOrderCancelled(reason: string): void {
    dogstatsd.increment('orders.cancelled', 1, [
      `reason:${reason}`,
    ]);
  }

  // 在庫数の追跡
  recordInventoryLevel(productId: string, quantity: number): void {
    dogstatsd.gauge('inventory.level', quantity, [
      `product_id:${productId}`,
    ]);
  }

  // 外部 API 呼び出しのレイテンシ
  recordExternalApiCall(
    provider: string,
    endpoint: string,
    durationMs: number,
    success: boolean
  ): void {
    dogstatsd.histogram('external_api.duration', durationMs, [
      `provider:${provider}`,
      `endpoint:${endpoint}`,
      `success:${success}`,
    ]);
    dogstatsd.increment('external_api.calls', 1, [
      `provider:${provider}`,
      `success:${success}`,
    ]);
  }

  // キャッシュヒット率
  recordCacheAccess(cacheName: string, hit: boolean): void {
    dogstatsd.increment('cache.access', 1, [
      `cache:${cacheName}`,
      `hit:${hit}`,
    ]);
  }
}

export const businessMetrics = new BusinessMetrics();
```

### 6.4 Datadog Monitor (Terraform)

```hcl
# datadog-monitors.tf — Terraform による Monitor 管理
terraform {
  required_providers {
    datadog = {
      source  = "DataDog/datadog"
      version = "~> 3.0"
    }
  }
}

provider "datadog" {
  api_key = var.datadog_api_key
  app_key = var.datadog_app_key
  api_url = "https://api.ap1.datadoghq.com/"
}

# エラーレート監視
resource "datadog_monitor" "error_rate" {
  name    = "[${var.environment}] ${var.service_name} - High Error Rate"
  type    = "query alert"
  message = <<-EOT
    ## エラーレートが閾値を超えています

    サービス: ${var.service_name}
    環境: ${var.environment}

    **対応手順:**
    1. [Runbook](https://wiki.example.com/runbooks/high-error-rate) を参照
    2. APM のエラートレースを確認
    3. 直近のデプロイを確認

    {{#is_alert}}@pagerduty-critical{{/is_alert}}
    {{#is_warning}}@slack-alerts-warning{{/is_warning}}
  EOT

  query = <<-EOT
    sum(last_5m):sum:trace.express.request.errors{service:${var.service_name},env:${var.environment}}.as_count()
    /
    sum:trace.express.request.hits{service:${var.service_name},env:${var.environment}}.as_count()
    > 0.05
  EOT

  monitor_thresholds {
    critical          = 0.05  # 5%
    warning           = 0.01  # 1%
    critical_recovery = 0.02
    warning_recovery  = 0.005
  }

  notify_no_data    = false
  renotify_interval = 60
  timeout_h         = 0

  tags = [
    "service:${var.service_name}",
    "env:${var.environment}",
    "team:backend",
  ]
}

# レイテンシ監視 (p95)
resource "datadog_monitor" "latency_p95" {
  name    = "[${var.environment}] ${var.service_name} - High Latency (p95)"
  type    = "query alert"
  message = <<-EOT
    ## p95 レイテンシが閾値を超えています

    サービス: ${var.service_name}
    現在値: {{value}} ms

    **確認事項:**
    1. DB クエリのパフォーマンス
    2. 外部 API のレスポンスタイム
    3. CPU/メモリのリソース状況

    {{#is_alert}}@pagerduty-critical{{/is_alert}}
    {{#is_warning}}@slack-alerts-warning{{/is_warning}}
  EOT

  query = "percentile(last_5m):p95:trace.express.request{service:${var.service_name},env:${var.environment}} > 2000"

  monitor_thresholds {
    critical = 2000  # 2秒
    warning  = 1000  # 1秒
  }

  tags = [
    "service:${var.service_name}",
    "env:${var.environment}",
  ]
}

# Anomaly Detection (異常検知)
resource "datadog_monitor" "request_anomaly" {
  name    = "[${var.environment}] ${var.service_name} - Request Rate Anomaly"
  type    = "query alert"
  message = <<-EOT
    ## リクエスト数に異常が検知されました

    通常のパターンから大きく逸脱しています。
    トラフィックの急増またはサービス障害の可能性があります。

    @slack-alerts-warning
  EOT

  query = "avg(last_4h):anomalies(sum:trace.express.request.hits{service:${var.service_name},env:${var.environment}}.as_count(), 'agile', 3, direction='both', interval=60, alert_window='last_30m', count_default_zero='true') >= 1"

  monitor_thresholds {
    critical = 1
  }

  monitor_threshold_windows {
    trigger_window  = "last_30m"
    recovery_window = "last_15m"
  }

  tags = [
    "service:${var.service_name}",
    "env:${var.environment}",
    "type:anomaly",
  ]
}

# SLO Monitor
resource "datadog_service_level_objective" "availability" {
  name = "${var.service_name} - Availability SLO"
  type = "monitor"

  monitor_ids = [
    datadog_monitor.error_rate.id,
  ]

  thresholds {
    timeframe = "30d"
    target    = 99.9
    warning   = 99.95
  }

  thresholds {
    timeframe = "7d"
    target    = 99.9
    warning   = 99.95
  }

  tags = [
    "service:${var.service_name}",
    "env:${var.environment}",
  ]
}
```

---

## 7. AWS CloudWatch

### 7.1 カスタムメトリクスの送信

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

### 7.2 CloudWatch メトリクスのバッチ送信

```typescript
// cloudwatch-batch-metrics.ts — 効率的なバッチ送信
import {
  CloudWatchClient,
  PutMetricDataCommand,
  MetricDatum,
  StandardUnit,
} from '@aws-sdk/client-cloudwatch';

class CloudWatchMetricsBatcher {
  private buffer: MetricDatum[] = [];
  private readonly maxBatchSize = 20; // CloudWatch の上限
  private readonly flushIntervalMs = 10000; // 10秒
  private timer: NodeJS.Timeout | null = null;

  constructor(
    private readonly client: CloudWatchClient,
    private readonly namespace: string,
  ) {
    this.startAutoFlush();
  }

  addMetric(
    metricName: string,
    value: number,
    unit: StandardUnit,
    dimensions: { Name: string; Value: string }[] = [],
  ): void {
    this.buffer.push({
      MetricName: metricName,
      Value: value,
      Unit: unit,
      Dimensions: dimensions,
      Timestamp: new Date(),
    });

    if (this.buffer.length >= this.maxBatchSize) {
      this.flush();
    }
  }

  // 統計値の送信 (集約済みデータ)
  addStatisticMetric(
    metricName: string,
    stats: { min: number; max: number; sum: number; count: number },
    unit: StandardUnit,
    dimensions: { Name: string; Value: string }[] = [],
  ): void {
    this.buffer.push({
      MetricName: metricName,
      StatisticValues: {
        Minimum: stats.min,
        Maximum: stats.max,
        Sum: stats.sum,
        SampleCount: stats.count,
      },
      Unit: unit,
      Dimensions: dimensions,
      Timestamp: new Date(),
    });
  }

  async flush(): Promise<void> {
    if (this.buffer.length === 0) return;

    const batches: MetricDatum[][] = [];
    while (this.buffer.length > 0) {
      batches.push(this.buffer.splice(0, this.maxBatchSize));
    }

    await Promise.all(
      batches.map((batch) =>
        this.client.send(
          new PutMetricDataCommand({
            Namespace: this.namespace,
            MetricData: batch,
          })
        )
      )
    );
  }

  private startAutoFlush(): void {
    this.timer = setInterval(() => this.flush(), this.flushIntervalMs);
  }

  async shutdown(): Promise<void> {
    if (this.timer) clearInterval(this.timer);
    await this.flush();
  }
}

// 使用例
const batcher = new CloudWatchMetricsBatcher(
  new CloudWatchClient({ region: 'ap-northeast-1' }),
  'MyApp/Production'
);

// メトリクスを追加 (バッファリングされる)
batcher.addMetric('ApiLatency', 125, 'Milliseconds', [
  { Name: 'Service', Value: 'order-service' },
]);

// アプリケーション終了時にフラッシュ
process.on('SIGTERM', async () => {
  await batcher.shutdown();
  process.exit(0);
});
```

### 7.3 CloudWatch Logs Insights クエリ

```
# --- CloudWatch Logs Insights クエリ例 ---

# エラーログの検索
fields @timestamp, @message
| filter @message like /ERROR/
| sort @timestamp desc
| limit 100

# JSON ログのパースと集計
fields @timestamp, @message
| parse @message '{"level":"*","msg":"*","service":"*","duration":*}' as level, msg, service, duration
| filter level = "error"
| stats count(*) as error_count by service
| sort error_count desc

# レイテンシの統計情報
fields @timestamp, @message
| parse @message '"duration":*,' as duration
| stats avg(duration) as avg_duration,
        pct(duration, 95) as p95_duration,
        pct(duration, 99) as p99_duration,
        max(duration) as max_duration
  by bin(5m)

# 特定のエラーパターンの発生頻度
fields @timestamp, @message
| filter @message like /database connection/
| stats count(*) as count by bin(1h)
| sort @timestamp desc

# Lambda 関数のコールドスタート分析
filter @type = "REPORT"
| parse @log /\/aws\/lambda\/(?<function>.*)/
| stats count(*) as invocations,
        sum(@initDuration > 0) as cold_starts,
        avg(@initDuration) as avg_init_duration,
        max(@duration) as max_duration,
        avg(@duration) as avg_duration,
        avg(@maxMemoryUsed / @memorySize * 100) as avg_memory_pct
  by function

# API Gateway のレイテンシ分析
fields @timestamp, @message
| parse @message '"httpMethod":"*","resourcePath":"*","status":"*","responseLatency":*' as method, path, status, latency
| filter status like /5\d\d/
| stats count(*) as error_count,
        avg(latency) as avg_latency
  by method, path
| sort error_count desc

# ユニークユーザー数のカウント
fields @timestamp, @message
| parse @message '"userId":"*"' as userId
| stats count_distinct(userId) as unique_users by bin(1h)
```

### 7.4 CloudWatch ダッシュボード (CloudFormation)

```yaml
# cloudwatch-dashboard.yml — CloudFormation テンプレート
AWSTemplateFormatVersion: '2010-09-09'
Description: CloudWatch Dashboard for Application Monitoring

Parameters:
  Environment:
    Type: String
    Default: production
  ServiceName:
    Type: String
    Default: order-service

Resources:
  ApplicationDashboard:
    Type: AWS::CloudWatch::Dashboard
    Properties:
      DashboardName: !Sub "${ServiceName}-${Environment}"
      DashboardBody: !Sub |
        {
          "widgets": [
            {
              "type": "metric",
              "x": 0, "y": 0, "width": 12, "height": 6,
              "properties": {
                "title": "API リクエストレート",
                "metrics": [
                  ["MyApp/Production", "RequestCount",
                   "Service", "${ServiceName}",
                   {"stat": "Sum", "period": 60}]
                ],
                "view": "timeSeries",
                "region": "ap-northeast-1",
                "period": 60
              }
            },
            {
              "type": "metric",
              "x": 12, "y": 0, "width": 12, "height": 6,
              "properties": {
                "title": "API レイテンシ (p50/p95/p99)",
                "metrics": [
                  ["MyApp/Production", "ApiLatency",
                   "Service", "${ServiceName}",
                   {"stat": "p50", "period": 60, "label": "p50"}],
                  ["...", {"stat": "p95", "period": 60, "label": "p95"}],
                  ["...", {"stat": "p99", "period": 60, "label": "p99"}]
                ],
                "view": "timeSeries",
                "region": "ap-northeast-1"
              }
            },
            {
              "type": "metric",
              "x": 0, "y": 6, "width": 8, "height": 6,
              "properties": {
                "title": "エラー数",
                "metrics": [
                  ["MyApp/Production", "ErrorCount",
                   "Service", "${ServiceName}",
                   {"stat": "Sum", "period": 60, "color": "#d62728"}]
                ],
                "view": "timeSeries",
                "region": "ap-northeast-1"
              }
            },
            {
              "type": "log",
              "x": 0, "y": 12, "width": 24, "height": 6,
              "properties": {
                "title": "直近のエラーログ",
                "query": "fields @timestamp, @message\n| filter @message like /ERROR/\n| sort @timestamp desc\n| limit 20",
                "region": "ap-northeast-1",
                "stacked": false,
                "view": "table"
              }
            }
          ]
        }

  # CloudWatch Alarm
  HighErrorRateAlarm:
    Type: AWS::CloudWatch::Alarm
    Properties:
      AlarmName: !Sub "${ServiceName}-${Environment}-HighErrorRate"
      AlarmDescription: "エラーレートが5%を超えています"
      MetricName: ErrorCount
      Namespace: MyApp/Production
      Dimensions:
        - Name: Service
          Value: !Ref ServiceName
      Statistic: Sum
      Period: 300
      EvaluationPeriods: 2
      Threshold: 50
      ComparisonOperator: GreaterThanThreshold
      TreatMissingData: notBreaching
      AlarmActions:
        - !Ref AlertSNSTopic

  AlertSNSTopic:
    Type: AWS::SNS::Topic
    Properties:
      TopicName: !Sub "${ServiceName}-${Environment}-alerts"
      Subscription:
        - Protocol: email
          Endpoint: oncall@example.com
```

---

## 8. 長期保存とスケーリング — Thanos・Mimir

### 8.1 Thanos によるマルチクラスタ監視

```yaml
# thanos-sidecar.yml — Prometheus に Thanos Sidecar を追加
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: prometheus
  namespace: monitoring
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: prometheus
          image: prom/prometheus:v2.50.0
          args:
            - '--config.file=/etc/prometheus/prometheus.yml'
            - '--storage.tsdb.retention.time=2h'  # ローカルは短く
            - '--storage.tsdb.min-block-duration=2h'
            - '--storage.tsdb.max-block-duration=2h'
            - '--web.enable-lifecycle'
          volumeMounts:
            - name: prometheus-data
              mountPath: /prometheus

        # Thanos Sidecar
        - name: thanos-sidecar
          image: thanosio/thanos:v0.34.0
          args:
            - sidecar
            - '--tsdb.path=/prometheus'
            - '--prometheus.url=http://localhost:9090'
            - '--objstore.config-file=/etc/thanos/objstore.yml'
            - '--grpc-address=0.0.0.0:10901'
          volumeMounts:
            - name: prometheus-data
              mountPath: /prometheus
            - name: thanos-config
              mountPath: /etc/thanos

---
# thanos-query.yml — Thanos Query (複数クラスタを横断クエリ)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-query
  namespace: monitoring
spec:
  replicas: 2
  template:
    spec:
      containers:
        - name: thanos-query
          image: thanosio/thanos:v0.34.0
          args:
            - query
            - '--grpc-address=0.0.0.0:10901'
            - '--http-address=0.0.0.0:9090'
            - '--store=dnssrv+_grpc._tcp.thanos-store.monitoring.svc'
            - '--store=dnssrv+_grpc._tcp.thanos-sidecar.monitoring.svc'
            - '--query.auto-downsampling'
            - '--query.replica-label=replica'
          ports:
            - containerPort: 9090
              name: http
            - containerPort: 10901
              name: grpc

---
# thanos-store.yml — Thanos Store Gateway (オブジェクトストレージからの読み取り)
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: thanos-store
  namespace: monitoring
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: thanos-store
          image: thanosio/thanos:v0.34.0
          args:
            - store
            - '--objstore.config-file=/etc/thanos/objstore.yml'
            - '--data-dir=/thanos/store'
            - '--index-cache-size=500MB'
            - '--chunk-pool-size=2GB'
          volumeMounts:
            - name: thanos-config
              mountPath: /etc/thanos
            - name: store-data
              mountPath: /thanos/store

---
# thanos-compactor.yml — Thanos Compactor (ダウンサンプリング)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: thanos-compactor
  namespace: monitoring
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: thanos-compactor
          image: thanosio/thanos:v0.34.0
          args:
            - compact
            - '--objstore.config-file=/etc/thanos/objstore.yml'
            - '--data-dir=/thanos/compact'
            - '--retention.resolution-raw=30d'      # 生データ: 30日
            - '--retention.resolution-5m=180d'       # 5分解像度: 180日
            - '--retention.resolution-1h=365d'       # 1時間解像度: 1年
            - '--compact.concurrency=2'
            - '--downsample.concurrency=2'
            - '--wait'
```

```yaml
# objstore.yml — Thanos オブジェクトストレージ設定 (S3)
type: S3
config:
  bucket: "thanos-metrics-production"
  endpoint: "s3.ap-northeast-1.amazonaws.com"
  region: "ap-northeast-1"
  access_key: "${AWS_ACCESS_KEY_ID}"
  secret_key: "${AWS_SECRET_ACCESS_KEY}"
  # SSE-S3 による暗号化
  sse_config:
    type: "SSE-S3"
```

### 8.2 長期保存ソリューション比較

```
長期保存ソリューション比較:

┌────────────┬──────────────┬──────────────┬──────────────┐
│ 特性       │ Thanos       │ Cortex       │ Mimir        │
├────────────┼──────────────┼──────────────┼──────────────┤
│ アーキテクチャ│ Sidecar 方式 │ Push 方式    │ Push 方式    │
│ 既存環境対応│ 容易         │ 設定変更必要 │ 設定変更必要 │
│ 複雑さ     │ 中           │ 高い         │ 中           │
│ スケール性 │ 高い         │ 非常に高い   │ 非常に高い   │
│ マルチテナント│ 限定的      │ ネイティブ   │ ネイティブ   │
│ 開発元     │ Improbable   │ Cortex Project│ Grafana Labs │
│ ダウンサンプリング│ 対応    │ 非対応       │ 対応         │
│ 推奨規模   │ 中〜大規模   │ 大規模       │ 中〜大規模   │
│ Grafana 統合│ 良好        │ 良好         │ 最高         │
└────────────┴──────────────┴──────────────┴──────────────┘

推奨:
- 既存 Prometheus に追加したい → Thanos
- Grafana Cloud / LGTM スタック → Mimir
- 大規模マルチテナント → Cortex or Mimir
```

---

## 9. 比較表

| 特性 | Prometheus + Grafana | Datadog | CloudWatch |
|------|---------------------|---------|------------|
| 運用形態 | セルフホスト | SaaS | AWS マネージド |
| メトリクス | Prometheus | 独自 | 独自 |
| ログ | Loki | Log Management | CloudWatch Logs |
| トレース | Jaeger/Tempo | APM | X-Ray |
| ダッシュボード | Grafana (強力) | 内蔵 (高機能) | 基本的 |
| アラート | Alertmanager | Monitors | CloudWatch Alarms |
| 異常検知 | なし (外部連携) | Watchdog (ML) | Anomaly Detection |
| 月額コスト (中規模) | インフラ費のみ | $500〜$5,000+ | $100〜$500 |
| 学習コスト | 高い (複数ツール) | 中 | 低い (AWS 利用者) |
| OpenTelemetry 対応 | ネイティブ | 対応 | 限定的 |

| ダッシュボードツール | Grafana | Datadog Dashboard | CloudWatch Dashboard |
|---------------------|---------|-------------------|---------------------|
| データソース数 | 100+ | Datadog内 | AWS内 |
| テンプレート変数 | 強力 | 対応 | 限定的 |
| 共有/埋め込み | 対応 | 対応 | 限定的 |
| アラート統合 | Alertmanager | 内蔵 | SNS 連携 |
| モバイル対応 | アプリあり | アプリあり | なし |
| IaC サポート | Terraform/Jsonnet | Terraform | CloudFormation |
| ダッシュボード as Code | Provisioning/API | API/Terraform | CloudFormation |

| ログ管理ツール | Loki | Datadog Logs | CloudWatch Logs | Elasticsearch |
|---------------|------|-------------|-----------------|---------------|
| インデックス方式 | ラベルのみ | フルテキスト | フルテキスト | フルテキスト |
| ストレージ効率 | 非常に高い | 中 | 中 | 低い |
| クエリ言語 | LogQL | 独自 | Insights | KQL/Lucene |
| Grafana 連携 | ネイティブ | プラグイン | プラグイン | プラグイン |
| 月額コスト (100GB/日) | インフラ費のみ | $2,000+ | $500+ | インフラ費 ($500+) |
| 運用負荷 | 中 | 低い | 低い | 高い |

---

## 10. アンチパターン

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
- ダッシュボードの命名規則を統一:
  [環境]-[サービス]-[用途]
  例: prod-order-service-overview
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

### アンチパターン 3: メトリクス命名の不統一

```
[悪い例]
- チームごとに異なる命名規則:
  orderCount, order_count, orders.total, num_orders
- 単位が不明: latency (ms? sec? us?)
- 同じ意味のメトリクスが異なる名前で存在

[良い例]
- OpenMetrics / Prometheus 命名規則に従う:
  - snake_case を使用
  - 単位をサフィックスに含める: _seconds, _bytes, _total
  - Counter は _total サフィックス: http_requests_total
  - Histogram は _bucket, _sum, _count サフィックス

- 命名テンプレート:
  {namespace}_{subsystem}_{name}_{unit}
  例:
    http_server_request_duration_seconds (Histogram)
    http_server_requests_total (Counter)
    process_resident_memory_bytes (Gauge)
    db_query_duration_seconds (Histogram)
```

### アンチパターン 4: 監視の盲点 (Blind Spots)

```
[悪い例]
- サーバーメトリクスだけ監視 (CPU/メモリ/ディスク)
- ビジネスメトリクスがない (注文数、収益、ユーザー登録数)
- 依存サービスの監視がない (外部 API、CDN、DNS)
- 合成監視がない (定期的な外形監視)

[良い例]
- 4つの監視レイヤーを網羅:
  1. インフラ: CPU, メモリ, ディスク, ネットワーク
  2. アプリケーション: レイテンシ, エラー率, スループット
  3. ビジネス: 注文数, 売上, コンバージョン率
  4. ユーザー体験: Core Web Vitals, エラー率, ファネル

- 依存サービスの監視:
  - 外部 API のレスポンスタイム
  - CDN のキャッシュヒット率
  - DNS 解決時間
  - SSL 証明書の有効期限
```

---

## 11. 運用のベストプラクティス

### 11.1 Prometheus 運用チェックリスト

```
□ ストレージ容量の見積もり
  - 時系列数 × サンプルサイズ(1-2bytes) × 保持期間
  - 例: 100,000 時系列 × 2 bytes × 15s間隔 × 30日 ≈ 30GB

□ WAL ディスクの監視
  - prometheus_tsdb_wal_segment_current で WAL サイズ確認
  - WAL 用に十分な IOPS を確保

□ Recording Rules の活用
  - ダッシュボードで頻繁に使うクエリは事前計算
  - 高カーディナリティのクエリを集約

□ Federation の検討
  - クラスタ間でメトリクスを集約する場合
  - match[] で必要なメトリクスのみ収集

□ Alertmanager の冗長化
  - 最低 2 台のクラスタ構成
  - --cluster.peer で相互接続

□ バックアップ
  - TSDB スナップショット API: POST /api/v1/admin/tsdb/snapshot
  - オブジェクトストレージへの定期コピー
```

### 11.2 Grafana 運用チェックリスト

```
□ 認証・認可
  - LDAP/OIDC/SAML による SSO
  - Organization / Team でアクセス制御
  - Viewer / Editor / Admin ロールの使い分け

□ バックアップ
  - grafana.db (SQLite) の定期バックアップ
  - ダッシュボード JSON のエクスポート
  - Provisioning による Git 管理が最善

□ プラグイン管理
  - GF_INSTALL_PLUGINS 環境変数で宣言的に管理
  - セキュリティアップデートの定期確認

□ パフォーマンス
  - 重いダッシュボードの特定 (ロード時間 > 5秒)
  - パネル数を適正に (1ダッシュボードあたり20パネル以下推奨)
  - Auto-refresh 間隔を適切に設定 (最低 30秒)
```

---

## 12. FAQ

### Q1: OSS スタックと SaaS、どちらを選ぶべきですか？

運用チームの規模とスキルが判断基準です。専任の SRE/インフラチーム（2名以上）がいれば OSS（Prometheus + Grafana）でコストを抑えつつ高いカスタマイズ性が得られます。少人数チームで監視基盤の運用に時間を割けない場合は、Datadog のような SaaS を選択してください。AWS に閉じたシステムなら CloudWatch が最もシンプルです。

### Q2: Prometheus のデータ保持期間はどのくらいが適切ですか？

ローカルストレージでは 15〜30日が現実的です。長期保存が必要な場合は Thanos や Mimir などのリモートストレージソリューションを導入してください。Thanos の Compactor を使えば、ダウンサンプリングにより1年以上のデータも効率的に保持できます（生データ30日 → 5分解像度180日 → 1時間解像度1年）。

### Q3: Grafana のダッシュボードをコード管理する方法は？

3つのアプローチがあります。(1) **Grafana Provisioning**: YAML + JSON ファイルで Git 管理し、起動時に自動読み込み。(2) **Terraform provider**: `grafana_dashboard` リソースで IaC 管理。(3) **Grafonnet (Jsonnet)**: プログラマブルにダッシュボードを生成。チーム規模が大きい場合は Terraform、小規模なら Provisioning が推奨です。

### Q4: Loki と Elasticsearch、どちらを選ぶべきですか？

Loki はラベルベースのインデックスで、ストレージ効率が非常に高い反面、全文検索の性能は Elasticsearch に劣ります。Grafana エコシステムを活用しており、ログの検索パターンが「ラベルで絞り込み → テキスト検索」であれば Loki が最適です。全文検索やログの複雑な集計が主用途の場合は Elasticsearch（OpenSearch）を選択してください。

### Q5: Datadog のコストを抑えるには？

以下の戦略でコスト最適化できます。(1) **カスタムメトリクス数の管理**: 高カーディナリティなタグを避け、メトリクス数を制御する。(2) **ログの取り込み量制御**: 不要なログレベル（DEBUG/INFO）をフィルタリングし、重要なログのみ Datadog に送信する。(3) **APM サンプリング**: 全トレースではなく、エラートレースと一定割合のサンプリングを使用する。(4) **インデックスの最適化**: ログのインデックスを使い分け、長期保存は Archive に移す。

### Q6: CloudWatch の制限事項は何ですか？

主な制限として、(1) カスタムメトリクスの PutMetricData は 1回あたり20メトリクスまで（バッチ送信が必要）、(2) ダッシュボードの表現力が Grafana/Datadog に比べて限定的、(3) クロスリージョン・クロスアカウントのメトリクス集約には追加設定が必要、(4) Logs Insights のクエリは最大15分のタイムアウトがある、などがあります。これらの制限を超える要件がある場合は、CloudWatch をデータソースとして Grafana で可視化する構成が有効です。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Prometheus | Pull 型メトリクス収集。PromQL で柔軟なクエリ。Recording Rules で事前集約 |
| Grafana | 多数のデータソース対応。最も柔軟なダッシュボード。Provisioning/Terraform で IaC 管理 |
| Loki | Grafana 連携のログ集約。ラベルベースのインデックスで高効率。LogQL でクエリ |
| Datadog | フルスタック SaaS。APM/ログ/メトリクス統合。異常検知 (Watchdog) あり |
| CloudWatch | AWS ネイティブ。追加設定なしで AWS リソース監視。Logs Insights でログ分析 |
| Thanos/Mimir | Prometheus の長期保存・マルチクラスタ対応。ダウンサンプリングで効率的な保持 |
| ダッシュボード設計 | 階層化し、IaC で管理。カーディナリティに注意。命名規則を統一 |

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
5. **Thanos Documentation** — https://thanos.io/tip/thanos/getting-started.md/ — Thanos 公式ガイド
6. **Grafana Loki Documentation** — https://grafana.com/docs/loki/latest/ — Loki 公式ドキュメント
7. **PromQL Cheat Sheet** — https://promlabs.com/promql-cheat-sheet/ — PromQL クイックリファレンス
8. **Grafana Mimir** — https://grafana.com/docs/mimir/latest/ — Mimir 長期ストレージ
