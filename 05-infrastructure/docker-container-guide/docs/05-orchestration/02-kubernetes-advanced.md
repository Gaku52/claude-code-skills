# Kubernetes 応用 (Kubernetes Advanced)

> Helm、Ingress、ConfigMap/Secret、HPA (Horizontal Pod Autoscaler) を活用し、本番環境で運用可能な Kubernetes クラスタのデプロイ・管理・スケーリング戦略を体系的に学ぶ。

## この章で学ぶこと

1. **Helm によるパッケージ管理とテンプレート化** -- Chart の作成・管理を通じて、再利用可能な Kubernetes マニフェストを構築する
2. **Ingress と ConfigMap/Secret による設定管理** -- 外部トラフィックのルーティングとアプリケーション設定の安全な管理手法を理解する
3. **HPA によるオートスケーリング** -- Pod の水平スケーリングを実装し、負荷に応じた自動調整で可用性とコスト効率を両立する

---

## 1. Helm

### 1.1 Helm の概要

```
+------------------------------------------------------------------+
|              Helm のアーキテクチャ                                  |
+------------------------------------------------------------------+
|                                                                  |
|  [開発者]                                                        |
|     |  helm install / upgrade / rollback                         |
|     v                                                            |
|  [Helm CLI]                                                      |
|     |  Chart (テンプレート + values)                              |
|     v                                                            |
|  [Kubernetes API Server]                                         |
|     |  マニフェスト適用                                           |
|     v                                                            |
|  [Kubernetes クラスタ]                                            |
|     +-- Deployment                                               |
|     +-- Service                                                  |
|     +-- Ingress                                                  |
|     +-- ConfigMap / Secret                                       |
|     +-- HPA                                                      |
|                                                                  |
|  Chart 構造:                                                     |
|    mychart/                                                      |
|      Chart.yaml          ← Chart メタデータ                      |
|      values.yaml         ← デフォルト値                          |
|      templates/          ← Go テンプレート                       |
|        deployment.yaml                                           |
|        service.yaml                                              |
|        ingress.yaml                                              |
|        _helpers.tpl      ← 共通ヘルパー                          |
|                                                                  |
+------------------------------------------------------------------+
```

### 1.2 Chart.yaml

```yaml
# mychart/Chart.yaml
apiVersion: v2
name: myapp
description: A Helm chart for MyApp
type: application
version: 0.1.0        # Chart バージョン
appVersion: "1.2.0"   # アプリバージョン

dependencies:
  - name: postgresql
    version: "15.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "19.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

### 1.3 values.yaml

```yaml
# mychart/values.yaml
replicaCount: 2

image:
  repository: ghcr.io/myorg/myapp
  tag: "1.2.0"
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 3000

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: app-tls
      hosts:
        - app.example.com

resources:
  requests:
    cpu: 100m
    memory: 128Mi
  limits:
    cpu: 500m
    memory: 256Mi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

env:
  NODE_ENV: production
  LOG_LEVEL: info

# サブチャートの設定
postgresql:
  enabled: true
  auth:
    postgresPassword: ""  # Secret で管理
    database: myapp

redis:
  enabled: true
  architecture: standalone
```

### 1.4 Deployment テンプレート

```yaml
# mychart/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "myapp.fullname" . }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "myapp.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "myapp.selectorLabels" . | nindent 8 }}
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
    spec:
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
          envFrom:
            - configMapRef:
                name: {{ include "myapp.fullname" . }}
            - secretRef:
                name: {{ include "myapp.fullname" . }}
          livenessProbe:
            httpGet:
              path: /health
              port: {{ .Values.service.targetPort }}
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: {{ .Values.service.targetPort }}
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

### 1.5 Helm コマンド

```bash
# Chart の依存関係を解決
helm dependency update ./mychart

# Dry-run (マニフェストをプレビュー)
helm install myapp ./mychart --dry-run --debug

# インストール
helm install myapp ./mychart -n production --create-namespace

# 値をオーバーライドしてインストール
helm install myapp ./mychart \
  -f values-production.yaml \
  --set image.tag=1.3.0

# アップグレード
helm upgrade myapp ./mychart \
  --set image.tag=1.3.0 \
  --wait --timeout 5m

# ロールバック
helm rollback myapp 1  # リビジョン 1 に戻す

# リリース一覧
helm list -n production

# 履歴
helm history myapp -n production

# アンインストール
helm uninstall myapp -n production
```

---

## 2. Ingress

### 2.1 Ingress の仕組み

```
+------------------------------------------------------------------+
|              Ingress のトラフィックフロー                           |
+------------------------------------------------------------------+
|                                                                  |
|  [インターネット]                                                 |
|       |                                                          |
|       v                                                          |
|  [Load Balancer] (クラウドプロバイダ提供)                          |
|       |                                                          |
|       v                                                          |
|  [Ingress Controller] (nginx / traefik / ALB)                    |
|       |                                                          |
|       +----> app.example.com/      → Service: frontend → Pods   |
|       |                                                          |
|       +----> app.example.com/api/  → Service: backend  → Pods   |
|       |                                                          |
|       +----> admin.example.com/    → Service: admin    → Pods   |
|                                                                  |
+------------------------------------------------------------------+
```

### 2.2 Ingress マニフェスト

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: myapp-ingress
  annotations:
    # NGINX Ingress Controller
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    # cert-manager による自動 TLS
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - app.example.com
        - api.example.com
      secretName: myapp-tls
  rules:
    - host: app.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend
                port:
                  number: 80
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 8080
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: backend
                port:
                  number: 8080
```

### 2.3 Ingress Controller 比較

| 項目 | NGINX Ingress | Traefik | AWS ALB | Istio Gateway |
|------|-------------|---------|---------|--------------|
| プロトコル | HTTP/HTTPS/gRPC | HTTP/HTTPS/TCP/UDP | HTTP/HTTPS | HTTP/HTTPS/gRPC/TCP |
| 設定方式 | Annotations | CRD / Labels | Annotations | CRD (VirtualService) |
| 自動TLS | cert-manager | 内蔵 (ACME) | ACM | cert-manager |
| レート制限 | Annotation | Middleware | WAF | EnvoyFilter |
| 認証 | Basic / OAuth | ForwardAuth | Cognito | JWT / OAuth |
| 可観測性 | Prometheus | 内蔵 Dashboard | CloudWatch | Kiali / Jaeger |
| 学習コスト | 低 | 中 | 低 (AWS) | 高 |

---

## 3. ConfigMap と Secret

### 3.1 ConfigMap

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: myapp-config
data:
  # キー=値
  NODE_ENV: production
  LOG_LEVEL: info
  APP_PORT: "3000"

  # ファイル全体を値として格納
  nginx.conf: |
    server {
      listen 80;
      server_name localhost;
      location / {
        proxy_pass http://localhost:3000;
      }
    }
```

### 3.2 Secret

```yaml
# secret.yaml (Base64 エンコード)
apiVersion: v1
kind: Secret
metadata:
  name: myapp-secret
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL3VzZXI6cGFzc0BkYjozMjQzMi9teWFwcA==
  JWT_SECRET: c3VwZXItc2VjcmV0LWtleQ==
  REDIS_PASSWORD: cmVkaXMtcGFzc3dvcmQ=

# stringData を使えばエンコード不要
# stringData:
#   DATABASE_URL: "postgresql://user:pass@db:5432/myapp"
```

### 3.3 Pod からの利用パターン

```yaml
# deployment.yaml
spec:
  containers:
    - name: app
      # パターン 1: 環境変数として全体を注入
      envFrom:
        - configMapRef:
            name: myapp-config
        - secretRef:
            name: myapp-secret

      # パターン 2: 個別のキーを環境変数にマッピング
      env:
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: myapp-config
              key: DB_HOST
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: myapp-secret
              key: DB_PASSWORD

      # パターン 3: ファイルとしてマウント
      volumeMounts:
        - name: config-volume
          mountPath: /etc/config
        - name: secret-volume
          mountPath: /etc/secrets
          readOnly: true

  volumes:
    - name: config-volume
      configMap:
        name: myapp-config
    - name: secret-volume
      secret:
        secretName: myapp-secret
```

### 3.4 External Secrets (外部シークレット管理)

```yaml
# external-secret.yaml (External Secrets Operator)
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: myapp-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secretsmanager
    kind: ClusterSecretStore
  target:
    name: myapp-secret
    creationPolicy: Owner
  data:
    - secretKey: DATABASE_URL
      remoteRef:
        key: myapp/production
        property: database_url
    - secretKey: JWT_SECRET
      remoteRef:
        key: myapp/production
        property: jwt_secret
```

---

## 4. HPA (Horizontal Pod Autoscaler)

### 4.1 HPA の仕組み

```
+------------------------------------------------------------------+
|              HPA のオートスケーリングフロー                          |
+------------------------------------------------------------------+
|                                                                  |
|  [Metrics Server] → CPU/メモリ使用率を収集                        |
|       |                                                          |
|       v                                                          |
|  [HPA Controller] → 15秒ごとにメトリクスを評価                    |
|       |                                                          |
|       v  目標: CPU 使用率 70%                                     |
|                                                                  |
|  現在: 3 Pods, CPU 使用率 90%                                    |
|  計算: ceil(3 * 90/70) = ceil(3.86) = 4 Pods                    |
|       |                                                          |
|       v  スケールアウト                                           |
|  結果: 4 Pods に増加                                             |
|                                                                  |
|  --- 負荷低下後 ---                                               |
|  現在: 4 Pods, CPU 使用率 30%                                    |
|  計算: ceil(4 * 30/70) = ceil(1.71) = 2 Pods                    |
|       |                                                          |
|       v  スケールイン (安定化ウィンドウ: 5分待機)                   |
|  結果: 2 Pods に減少                                             |
|                                                                  |
+------------------------------------------------------------------+
```

### 4.2 HPA マニフェスト

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: myapp-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: myapp
  minReplicas: 2
  maxReplicas: 20
  metrics:
    # CPU 使用率ベース
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    # メモリ使用率ベース
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # カスタムメトリクス (Prometheus 連携)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "100"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50          # 最大50%ずつ増加
          periodSeconds: 60
        - type: Pods
          value: 4           # 最大4 Pod ずつ増加
          periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300  # 5分の安定化ウィンドウ
      policies:
        - type: Percent
          value: 25          # 最大25%ずつ減少
          periodSeconds: 60
```

### 4.3 スケーリング戦略比較

| 戦略 | メトリクス | レスポンス速度 | 精度 | 用途 |
|------|----------|-------------|------|------|
| HPA (CPU) | CPU 使用率 | 中 (15秒間隔) | 中 | 一般的な Web アプリ |
| HPA (メモリ) | メモリ使用量 | 中 | 低 | メモリ集約型処理 |
| HPA (カスタム) | RPS, レイテンシ等 | 中 | 高 | API サーバー |
| KEDA | イベントソース | 高 | 高 | キューワーカー, FaaS |
| VPA | CPU/メモリ | 遅い | 高 | リソース最適化 |
| Cluster Autoscaler | ノードリソース | 遅い | - | ノード追加/削除 |

---

## 5. デプロイ戦略

### 5.1 ローリングアップデート

```yaml
# deployment.yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%         # 一度に追加する Pod の割合
      maxUnavailable: 25%   # 一度に停止する Pod の割合
```

### 5.2 Blue-Green / Canary (Helm)

```bash
# カナリアデプロイ (10% のトラフィック)
helm upgrade myapp ./mychart \
  --set canary.enabled=true \
  --set canary.weight=10 \
  --set canary.image.tag=1.3.0

# 問題なければ本番に昇格
helm upgrade myapp ./mychart \
  --set image.tag=1.3.0 \
  --set canary.enabled=false

# 問題があればロールバック
helm rollback myapp
```

---

## アンチパターン

### アンチパターン 1: Secret を Git にコミット

```yaml
# NG: Secret の値をそのまま Git にコミット
apiVersion: v1
kind: Secret
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL2FkbWluOnBAc3N3b3JkQGRiOjU0MzIvcHJvZA==
  # Base64 はエンコードであって暗号化ではない！

# OK: 外部シークレット管理を使用
# - External Secrets Operator + AWS Secrets Manager
# - Sealed Secrets (暗号化して Git にコミット)
# - SOPS (Mozilla) で暗号化
```

**問題点**: Base64 は単なるエンコードであり、`echo "cG9z..." | base64 -d` で簡単にデコードできる。Git 履歴に一度でもコミットされると完全な削除が困難。External Secrets Operator や Sealed Secrets を使い、暗号化された状態でのみ Git に保存する。

### アンチパターン 2: HPA と Deployment の replicas を同時指定

```yaml
# NG: HPA が管理する replicas を Deployment にも指定
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3  # ← HPA と競合！ helm upgrade のたびにリセットされる

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 2
  maxReplicas: 10

# OK: HPA 有効時は replicas を省略
apiVersion: apps/v1
kind: Deployment
spec:
  # replicas は HPA が管理するため省略
  selector:
    matchLabels:
      app: myapp
```

**問題点**: HPA が Pod 数を 7 に増やしていても、`helm upgrade` 実行時に `replicas: 3` で上書きされ、突然 Pod が 4 つ削除される。HPA 有効時は Deployment の `replicas` フィールドを省略する。

---

## FAQ

### Q1: Helm Chart はどこに保存するのがベストですか？

**A**: (1) アプリケーションリポジトリ内の `charts/` ディレクトリに同居させる (モノリポ方式)、(2) 専用の Helm Chart リポジトリに分離する (OCI レジストリ or ChartMuseum)。小〜中規模なら (1) が管理しやすく、アプリとチャートのバージョンを同期できる。大規模で複数チームが同じ Chart を使う場合は (2) で共有 Chart として管理する。OCI レジストリ (GHCR, ECR 等) が現在の推奨。

### Q2: ConfigMap を変更した場合、Pod は自動的に再起動されますか？

**A**: いいえ。ConfigMap を更新しても既存の Pod は自動再起動されない。対応方法は (1) Deployment の annotation に ConfigMap の checksum を含める (`checksum/config: {{ sha256sum }}`) ことで、ConfigMap 変更時に Pod がローリングアップデートされる、(2) `kubectl rollout restart deployment myapp` で手動再起動、(3) Reloader (stakater/Reloader) を導入して自動検知・再起動。

### Q3: KEDA と HPA の違いは何ですか？

**A**: HPA は CPU/メモリやカスタムメトリクスに基づいてスケールするが、最小 1 Pod が必要 (0 にスケールダウンできない)。KEDA (Kubernetes Event-Driven Autoscaling) はイベントソース (SQS キュー長、Kafka ラグ、Cron 等) に基づいてスケールし、0 Pod までスケールダウンできる (Scale-to-Zero)。API サーバーには HPA、バックグラウンドワーカーやイベントドリブン処理には KEDA が適する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Helm | Chart でマニフェストをテンプレート化。values.yaml で環境差異を吸収 |
| Ingress | ホスト名/パスベースルーティング。cert-manager で自動 TLS |
| ConfigMap | 設定値を外部化。checksum annotation で変更検知 |
| Secret | 機密情報の管理。External Secrets Operator で外部連携推奨 |
| HPA | CPU/メモリ/カスタムメトリクスで水平スケーリング |
| デプロイ戦略 | RollingUpdate (標準)、Canary (Helm/Argo)、Blue-Green |
| KEDA | イベントドリブンスケーリング。Scale-to-Zero 対応 |
| 監視 | Prometheus + Grafana でメトリクス可視化 |

## 次に読むべきガイド

- [コンテナセキュリティ](../06-security/00-container-security.md) -- K8s 上のセキュリティベストプラクティス
- [サプライチェーンセキュリティ](../06-security/01-supply-chain-security.md) -- イメージ署名と SBOM
- Docker Compose から Kubernetes への移行 -- Kompose を使った移行パターン

## 参考文献

1. **Helm 公式ドキュメント** -- https://helm.sh/docs/ -- Helm Chart の作成と管理の包括的リファレンス
2. **Kubernetes Ingress** -- https://kubernetes.io/docs/concepts/services-networking/ingress/ -- Ingress リソースの公式仕様
3. **HPA 公式ドキュメント** -- https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/ -- HPA の設定と動作の詳細
4. **External Secrets Operator** -- https://external-secrets.io/ -- 外部シークレット管理との統合
