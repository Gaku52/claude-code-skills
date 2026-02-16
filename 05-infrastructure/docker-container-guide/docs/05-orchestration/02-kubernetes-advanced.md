# Kubernetes 応用 (Kubernetes Advanced)

> Helm、Ingress、ConfigMap/Secret、HPA (Horizontal Pod Autoscaler) を活用し、本番環境で運用可能な Kubernetes クラスタのデプロイ・管理・スケーリング戦略を体系的に学ぶ。

## この章で学ぶこと

1. **Helm によるパッケージ管理とテンプレート化** -- Chart の作成・管理を通じて、再利用可能な Kubernetes マニフェストを構築する
2. **Ingress と ConfigMap/Secret による設定管理** -- 外部トラフィックのルーティングとアプリケーション設定の安全な管理手法を理解する
3. **HPA によるオートスケーリング** -- Pod の水平スケーリングを実装し、負荷に応じた自動調整で可用性とコスト効率を両立する
4. **デプロイ戦略の選択と実装** -- ローリングアップデート、Blue-Green、Canary デプロイの使い分けと Argo Rollouts 連携
5. **KEDA によるイベントドリブンスケーリング** -- キューベースのワークロードを 0 から N までスケーリングする実装パターン
6. **Prometheus / Grafana による監視基盤** -- メトリクス収集からアラート設定までの可観測性スタック構築

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
|      Chart.yaml          <- Chart メタデータ                      |
|      values.yaml         <- デフォルト値                          |
|      values-staging.yaml <- ステージング用オーバーライド            |
|      values-prod.yaml    <- 本番用オーバーライド                    |
|      charts/             <- 依存 Chart (サブチャート)              |
|      crds/               <- CustomResourceDefinition              |
|      templates/          <- Go テンプレート                       |
|        deployment.yaml                                           |
|        service.yaml                                              |
|        ingress.yaml                                              |
|        configmap.yaml                                            |
|        secret.yaml                                               |
|        hpa.yaml                                                  |
|        pdb.yaml                                                  |
|        serviceaccount.yaml                                       |
|        NOTES.txt         <- インストール後に表示されるメモ          |
|        _helpers.tpl      <- 共通ヘルパー                          |
|      tests/              <- Helm テスト                           |
|        test-connection.yaml                                      |
|                                                                  |
+------------------------------------------------------------------+
```

Helm は Kubernetes マニフェストのパッケージマネージャであり、テンプレートエンジンでもある。複数の YAML ファイルを一つの「Chart」として管理し、`values.yaml` の値を差し替えることで、同一テンプレートから開発・ステージング・本番など異なる環境向けのマニフェストを生成できる。

Helm 3 以降は Tiller (サーバーサイドコンポーネント) が廃止され、クライアントのみで動作するようになった。リリース情報は Kubernetes の Secret として保存される。

### 1.2 Chart.yaml

```yaml
# mychart/Chart.yaml
apiVersion: v2
name: myapp
description: A Helm chart for MyApp
type: application
version: 0.1.0        # Chart バージョン
appVersion: "1.2.0"   # アプリバージョン

# Chart のメンテナ情報
maintainers:
  - name: Platform Team
    email: platform@example.com

# キーワード (検索用)
keywords:
  - webapp
  - nodejs

# ソースコードの場所
sources:
  - https://github.com/myorg/myapp

dependencies:
  - name: postgresql
    version: "15.x"
    repository: https://charts.bitnami.com/bitnami
    condition: postgresql.enabled
  - name: redis
    version: "19.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
  - name: common
    version: "2.x"
    repository: https://charts.bitnami.com/bitnami
    tags:
      - bitnami-common
```

### 1.3 values.yaml

```yaml
# mychart/values.yaml
replicaCount: 2

image:
  repository: ghcr.io/myorg/myapp
  tag: "1.2.0"
  pullPolicy: IfNotPresent

# イメージプルシークレット (プライベートレジストリ用)
imagePullSecrets:
  - name: ghcr-credentials

# Service Account
serviceAccount:
  create: true
  annotations: {}
  name: ""

# Pod のアノテーション
podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "3000"
  prometheus.io/path: "/metrics"

# Pod のラベル
podLabels:
  team: backend
  cost-center: engineering

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
  targetMemoryUtilizationPercentage: 80

# Pod Disruption Budget
podDisruptionBudget:
  enabled: true
  minAvailable: 1
  # maxUnavailable: 1

# ノードセレクタ
nodeSelector:
  kubernetes.io/arch: amd64

# Toleration
tolerations: []

# Affinity (Pod の配置制御)
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app.kubernetes.io/name
                operator: In
                values:
                  - myapp
          topologyKey: kubernetes.io/hostname

# Topology Spread Constraints (AZ 分散)
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: DoNotSchedule
    labelSelector:
      matchLabels:
        app.kubernetes.io/name: myapp

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

### 1.4 _helpers.tpl (共通ヘルパーテンプレート)

```yaml
{{/*
mychart/templates/_helpers.tpl
*/}}

{{/*
アプリケーション名 (Chart 名をベースにオーバーライド可能)
*/}}
{{- define "myapp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
フルネーム (リリース名 + Chart 名)
*/}}
{{- define "myapp.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
共通ラベル
*/}}
{{- define "myapp.labels" -}}
helm.sh/chart: {{ include "myapp.chart" . }}
{{ include "myapp.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
セレクタラベル
*/}}
{{- define "myapp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "myapp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Chart 名 + バージョン
*/}}
{{- define "myapp.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ServiceAccount 名
*/}}
{{- define "myapp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "myapp.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

### 1.5 Deployment テンプレート

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
        {{- with .Values.podLabels }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
        checksum/secret: {{ include (print $.Template.BasePath "/secret.yaml") . | sha256sum }}
        {{- with .Values.podAnnotations }}
        {{- toYaml . | nindent 8 }}
        {{- end }}
    spec:
      serviceAccountName: {{ include "myapp.serviceAccountName" . }}
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      securityContext:
        runAsNonRoot: true
        runAsUser: 1001
        runAsGroup: 1001
        fsGroup: 1001
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          ports:
            - containerPort: {{ .Values.service.targetPort }}
              protocol: TCP
              name: http
          envFrom:
            - configMapRef:
                name: {{ include "myapp.fullname" . }}
            - secretRef:
                name: {{ include "myapp.fullname" . }}
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
            timeoutSeconds: 3
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 10
            periodSeconds: 5
            failureThreshold: 30
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir:
            sizeLimit: 64Mi
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.topologySpreadConstraints }}
      topologySpreadConstraints:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

### 1.6 Service テンプレート

```yaml
# mychart/templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "myapp.fullname" . }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.targetPort }}
      protocol: TCP
      name: http
  selector:
    {{- include "myapp.selectorLabels" . | nindent 4 }}
```

### 1.7 PodDisruptionBudget テンプレート

```yaml
# mychart/templates/pdb.yaml
{{- if .Values.podDisruptionBudget.enabled }}
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: {{ include "myapp.fullname" . }}
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
spec:
  {{- if .Values.podDisruptionBudget.minAvailable }}
  minAvailable: {{ .Values.podDisruptionBudget.minAvailable }}
  {{- end }}
  {{- if .Values.podDisruptionBudget.maxUnavailable }}
  maxUnavailable: {{ .Values.podDisruptionBudget.maxUnavailable }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "myapp.selectorLabels" . | nindent 6 }}
{{- end }}
```

### 1.8 Helm テスト

```yaml
# mychart/templates/tests/test-connection.yaml
apiVersion: v1
kind: Pod
metadata:
  name: "{{ include "myapp.fullname" . }}-test-connection"
  labels:
    {{- include "myapp.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": test
    "helm.sh/hook-delete-policy": before-hook-creation,hook-succeeded
spec:
  containers:
    - name: wget
      image: busybox:1.36
      command: ['wget']
      args: ['{{ include "myapp.fullname" . }}:{{ .Values.service.port }}/health']
  restartPolicy: Never
```

### 1.9 Helm コマンド

```bash
# Chart の依存関係を解決
helm dependency update ./mychart

# Dry-run (マニフェストをプレビュー)
helm install myapp ./mychart --dry-run --debug

# テンプレートのレンダリングのみ (クラスタ不要)
helm template myapp ./mychart -f values-production.yaml

# Lint (構文チェック)
helm lint ./mychart

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

# install と upgrade を統合 (存在しなければ install)
helm upgrade --install myapp ./mychart \
  -f values-production.yaml \
  --wait --timeout 5m \
  --atomic  # 失敗時は自動ロールバック

# ロールバック
helm rollback myapp 1  # リビジョン 1 に戻す

# リリース一覧
helm list -n production

# 特定リリースの現在の values を表示
helm get values myapp -n production

# 特定リリースの全マニフェストを表示
helm get manifest myapp -n production

# 履歴
helm history myapp -n production

# テスト実行
helm test myapp -n production

# アンインストール
helm uninstall myapp -n production

# OCI レジストリへの Chart プッシュ
helm package ./mychart
helm push myapp-0.1.0.tgz oci://ghcr.io/myorg/charts

# OCI レジストリからのインストール
helm install myapp oci://ghcr.io/myorg/charts/myapp --version 0.1.0
```

### 1.10 環境別 values ファイルの運用

```yaml
# values-staging.yaml (ステージング用オーバーライド)
replicaCount: 1

image:
  tag: "staging-latest"

resources:
  requests:
    cpu: 50m
    memory: 64Mi
  limits:
    cpu: 200m
    memory: 128Mi

autoscaling:
  enabled: false

ingress:
  hosts:
    - host: staging.app.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: staging-app-tls
      hosts:
        - staging.app.example.com

env:
  NODE_ENV: staging
  LOG_LEVEL: debug
```

```yaml
# values-production.yaml (本番用オーバーライド)
replicaCount: 3

image:
  tag: "1.2.0"

resources:
  requests:
    cpu: 200m
    memory: 256Mi
  limits:
    cpu: 1000m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 65

podDisruptionBudget:
  enabled: true
  minAvailable: 2

env:
  NODE_ENV: production
  LOG_LEVEL: warn
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
|       +----> app.example.com/      -> Service: frontend -> Pods  |
|       |                                                          |
|       +----> app.example.com/api/  -> Service: backend  -> Pods  |
|       |                                                          |
|       +----> admin.example.com/    -> Service: admin    -> Pods  |
|       |                                                          |
|       +----> ws.example.com/       -> Service: websocket -> Pods |
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
    # タイムアウト設定
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "10"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "60"
    # CORS 設定
    nginx.ingress.kubernetes.io/enable-cors: "true"
    nginx.ingress.kubernetes.io/cors-allow-origin: "https://app.example.com"
    nginx.ingress.kubernetes.io/cors-allow-methods: "GET, POST, PUT, DELETE, OPTIONS"
    nginx.ingress.kubernetes.io/cors-allow-headers: "Authorization, Content-Type"
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

### 2.3 WebSocket 対応 Ingress

```yaml
# ingress-websocket.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: websocket-ingress
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-http-version: "1.1"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
spec:
  ingressClassName: nginx
  rules:
    - host: ws.example.com
      http:
        paths:
          - path: /ws
            pathType: Prefix
            backend:
              service:
                name: websocket-service
                port:
                  number: 8080
```

### 2.4 gRPC 対応 Ingress

```yaml
# ingress-grpc.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grpc-ingress
  annotations:
    nginx.ingress.kubernetes.io/backend-protocol: "GRPC"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - grpc.example.com
      secretName: grpc-tls
  rules:
    - host: grpc.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: grpc-service
                port:
                  number: 50051
```

### 2.5 cert-manager による自動 TLS 証明書管理

```yaml
# cert-manager ClusterIssuer (Let's Encrypt)
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@example.com
    privateKeySecretRef:
      name: letsencrypt-prod-key
    solvers:
      - http01:
          ingress:
            class: nginx
      # DNS01 チャレンジ (ワイルドカード証明書用)
      - dns01:
          route53:
            region: ap-northeast-1
            hostedZoneID: Z1234567890
        selector:
          dnsZones:
            - "example.com"

---
# ワイルドカード証明書
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: wildcard-cert
  namespace: default
spec:
  secretName: wildcard-tls
  issuerRef:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - "*.example.com"
    - "example.com"
```

### 2.6 Ingress Controller 比較

| 項目 | NGINX Ingress | Traefik | AWS ALB | Istio Gateway |
|------|-------------|---------|---------|--------------|
| プロトコル | HTTP/HTTPS/gRPC | HTTP/HTTPS/TCP/UDP | HTTP/HTTPS | HTTP/HTTPS/gRPC/TCP |
| 設定方式 | Annotations | CRD / Labels | Annotations | CRD (VirtualService) |
| 自動TLS | cert-manager | 内蔵 (ACME) | ACM | cert-manager |
| レート制限 | Annotation | Middleware | WAF | EnvoyFilter |
| 認証 | Basic / OAuth | ForwardAuth | Cognito | JWT / OAuth |
| 可観測性 | Prometheus | 内蔵 Dashboard | CloudWatch | Kiali / Jaeger |
| WebSocket | サポート | サポート | サポート | サポート |
| gRPC | サポート | サポート | サポート | ネイティブサポート |
| カナリア | Annotation | Weighted | Weighted Target Group | Traffic Shifting |
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

      # ヘルスチェック用エンドポイント
      location /nginx-health {
        return 200 "ok";
        access_log off;
      }

      location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
      }
    }

  # JSON 設定ファイル
  app-config.json: |
    {
      "database": {
        "pool": {
          "min": 2,
          "max": 10,
          "idleTimeoutMillis": 30000
        }
      },
      "cache": {
        "ttl": 3600,
        "checkPeriod": 600
      },
      "logging": {
        "format": "json",
        "level": "info"
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
        # 特定のキーのみマウント
        - name: nginx-config
          mountPath: /etc/nginx/conf.d/default.conf
          subPath: nginx.conf
          readOnly: true

  volumes:
    - name: config-volume
      configMap:
        name: myapp-config
    - name: secret-volume
      secret:
        secretName: myapp-secret
        defaultMode: 0400  # 読み取り専用パーミッション
    - name: nginx-config
      configMap:
        name: myapp-config
        items:
          - key: nginx.conf
            path: nginx.conf
```

### 3.4 External Secrets (外部シークレット管理)

```yaml
# ClusterSecretStore (AWS Secrets Manager)
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secretsmanager
spec:
  provider:
    aws:
      service: SecretsManager
      region: ap-northeast-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets-sa
            namespace: external-secrets

---
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
    template:
      type: Opaque
      data:
        DATABASE_URL: "{{ .database_url }}"
        JWT_SECRET: "{{ .jwt_secret }}"
  data:
    - secretKey: database_url
      remoteRef:
        key: myapp/production
        property: database_url
    - secretKey: jwt_secret
      remoteRef:
        key: myapp/production
        property: jwt_secret
```

### 3.5 Sealed Secrets (GitOps 向け暗号化)

```bash
# Sealed Secrets Controller のインストール
helm repo add sealed-secrets https://bitnami-labs.github.io/sealed-secrets
helm install sealed-secrets sealed-secrets/sealed-secrets -n kube-system

# kubeseal CLI で暗号化
kubectl create secret generic myapp-secret \
  --from-literal=DATABASE_URL='postgresql://user:pass@db:5432/myapp' \
  --dry-run=client -o yaml | kubeseal --format yaml > sealed-secret.yaml
```

```yaml
# sealed-secret.yaml (暗号化済み、Git にコミット可能)
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: myapp-secret
  namespace: production
spec:
  encryptedData:
    DATABASE_URL: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...
    JWT_SECRET: AgCtr3j5PKTWL+QjUZAAB0sP44dGEFr...
```

---

## 4. HPA (Horizontal Pod Autoscaler)

### 4.1 HPA の仕組み

```
+------------------------------------------------------------------+
|              HPA のオートスケーリングフロー                          |
+------------------------------------------------------------------+
|                                                                  |
|  [Metrics Server] -> CPU/メモリ使用率を収集                        |
|       |                                                          |
|       v                                                          |
|  [HPA Controller] -> 15秒ごとにメトリクスを評価                    |
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
|  スケーリング公式:                                                |
|    desiredReplicas = ceil[currentReplicas                        |
|      * (currentMetricValue / desiredMetricValue)]                |
|                                                                  |
|  複数メトリクス使用時:                                             |
|    各メトリクスで desiredReplicas を計算し、最大値を採用            |
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
    # 外部メトリクス (SQS キュー長など)
    - type: External
      external:
        metric:
          name: sqs_messages_visible
          selector:
            matchLabels:
              queue: myapp-tasks
        target:
          type: AverageValue
          averageValue: "5"
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
      selectPolicy: Min  # 最も保守的なポリシーを選択
```

### 4.3 Metrics Server のインストールと確認

```bash
# Metrics Server のインストール
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# メトリクスの確認
kubectl top nodes
kubectl top pods -n production

# HPA の状態確認
kubectl get hpa -n production
kubectl describe hpa myapp-hpa -n production

# HPA のイベントログ確認
kubectl get events --field-selector involvedObject.name=myapp-hpa -n production
```

### 4.4 Prometheus Adapter (カスタムメトリクス HPA)

```yaml
# prometheus-adapter の設定
# Prometheus のメトリクスを Kubernetes のカスタムメトリクス API に変換
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-adapter-config
  namespace: monitoring
data:
  config.yaml: |
    rules:
      # HTTP RPS をカスタムメトリクスとして公開
      - seriesQuery: 'http_requests_total{namespace!="",pod!=""}'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          matches: "^(.*)_total$"
          as: "${1}_per_second"
        metricsQuery: 'rate(<<.Series>>{<<.LabelMatchers>>}[2m])'

      # レスポンスタイム P99
      - seriesQuery: 'http_request_duration_seconds_bucket{namespace!="",pod!=""}'
        resources:
          overrides:
            namespace: {resource: "namespace"}
            pod: {resource: "pod"}
        name:
          as: "http_request_duration_p99"
        metricsQuery: 'histogram_quantile(0.99, rate(<<.Series>>{<<.LabelMatchers>>}[5m]))'
```

### 4.5 スケーリング戦略比較

| 戦略 | メトリクス | レスポンス速度 | 精度 | 用途 |
|------|----------|-------------|------|------|
| HPA (CPU) | CPU 使用率 | 中 (15秒間隔) | 中 | 一般的な Web アプリ |
| HPA (メモリ) | メモリ使用量 | 中 | 低 | メモリ集約型処理 |
| HPA (カスタム) | RPS, レイテンシ等 | 中 | 高 | API サーバー |
| KEDA | イベントソース | 高 | 高 | キューワーカー, FaaS |
| VPA | CPU/メモリ | 遅い | 高 | リソース最適化 |
| Cluster Autoscaler | ノードリソース | 遅い | - | ノード追加/削除 |
| Karpenter | ノードリソース | 速い | 高 | AWS ノードプロビジョニング |

---

## 5. KEDA (Kubernetes Event-Driven Autoscaling)

### 5.1 KEDA の概要

```
+------------------------------------------------------------------+
|              KEDA のアーキテクチャ                                  |
+------------------------------------------------------------------+
|                                                                  |
|  [イベントソース]                                                  |
|    +-- AWS SQS                                                   |
|    +-- Apache Kafka                                              |
|    +-- RabbitMQ                                                  |
|    +-- Redis Streams                                             |
|    +-- Prometheus                                                |
|    +-- Cron                                                      |
|       |                                                          |
|       v                                                          |
|  [KEDA Operator]                                                 |
|    +-- ScaledObject  -> Deployment のスケーリング                  |
|    +-- ScaledJob     -> Job のスケーリング                        |
|       |                                                          |
|       v  メトリクス評価                                           |
|  [HPA] (KEDA が内部的に HPA を生成・管理)                         |
|       |                                                          |
|       v                                                          |
|  [Deployment / Job]                                              |
|    0 Pods <-----> N Pods (Scale-to-Zero 対応)                    |
|                                                                  |
+------------------------------------------------------------------+
```

### 5.2 KEDA のインストール

```bash
# Helm でインストール
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda -n keda --create-namespace
```

### 5.3 SQS キューワーカーのスケーリング

```yaml
# keda-sqs-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sqs-worker-scaler
  namespace: production
spec:
  scaleTargetRef:
    name: sqs-worker
  pollingInterval: 15          # 15秒ごとにチェック
  cooldownPeriod: 60           # スケールダウン前の待機時間
  minReplicaCount: 0           # Scale-to-Zero
  maxReplicaCount: 50
  triggers:
    - type: aws-sqs-queue
      metadata:
        queueURL: https://sqs.ap-northeast-1.amazonaws.com/123456789/myapp-tasks
        queueLength: "5"       # メッセージ5件あたり1 Pod
        awsRegion: ap-northeast-1
      authenticationRef:
        name: aws-credentials

---
# 認証情報の参照
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: aws-credentials
  namespace: production
spec:
  podIdentity:
    provider: aws-eks  # EKS Pod Identity / IRSA を使用
```

### 5.4 Kafka コンシューマーのスケーリング

```yaml
# keda-kafka-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: kafka-consumer-scaler
spec:
  scaleTargetRef:
    name: kafka-consumer
  minReplicaCount: 0
  maxReplicaCount: 30
  triggers:
    - type: kafka
      metadata:
        bootstrapServers: kafka-broker:9092
        consumerGroup: myapp-group
        topic: events
        lagThreshold: "100"      # ラグ100件あたり1 Pod
        activationLagThreshold: "10"  # ラグ10件で0->1にスケール
```

### 5.5 Cron ベースのスケーリング

```yaml
# keda-cron-scaledobject.yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: business-hours-scaler
spec:
  scaleTargetRef:
    name: myapp
  triggers:
    # 営業時間 (平日 9:00-18:00 JST) は多めに Pod を維持
    - type: cron
      metadata:
        timezone: Asia/Tokyo
        start: 0 9 * * 1-5
        end: 0 18 * * 1-5
        desiredReplicas: "10"
    # 夜間・休日は最小構成
    - type: cron
      metadata:
        timezone: Asia/Tokyo
        start: 0 18 * * 1-5
        end: 0 9 * * 2-6
        desiredReplicas: "2"
```

---

## 6. デプロイ戦略

### 6.1 ローリングアップデート

```yaml
# deployment.yaml
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%         # 一度に追加する Pod の割合
      maxUnavailable: 25%   # 一度に停止する Pod の割合
```

### 6.2 Blue-Green / Canary (Helm)

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

### 6.3 Argo Rollouts によるプログレッシブデリバリー

```yaml
# argo-rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: myapp
spec:
  replicas: 5
  revisionHistoryLimit: 3
  selector:
    matchLabels:
      app: myapp
  strategy:
    canary:
      canaryService: myapp-canary
      stableService: myapp-stable
      trafficRouting:
        nginx:
          stableIngress: myapp-ingress
      steps:
        # 5% のトラフィックをカナリアに振る
        - setWeight: 5
        - pause: {duration: 5m}
        # メトリクスを自動分析
        - analysis:
            templates:
              - templateName: success-rate
            args:
              - name: service-name
                value: myapp-canary
        # 20% に増加
        - setWeight: 20
        - pause: {duration: 5m}
        # 50% に増加
        - setWeight: 50
        - pause: {duration: 10m}
        # 100% (昇格)
        - setWeight: 100

---
# 分析テンプレート
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
spec:
  args:
    - name: service-name
  metrics:
    - name: success-rate
      interval: 1m
      successCondition: result[0] >= 0.99
      failureLimit: 3
      provider:
        prometheus:
          address: http://prometheus:9090
          query: |
            sum(rate(http_requests_total{
              service="{{args.service-name}}",
              status=~"2.."
            }[5m])) /
            sum(rate(http_requests_total{
              service="{{args.service-name}}"
            }[5m]))
```

### 6.4 デプロイ戦略比較

| 戦略 | ダウンタイム | ロールバック速度 | リソース使用量 | 複雑度 |
|------|-----------|-------------|-------------|-------|
| RollingUpdate | なし | 中 (Pod 入れ替え) | 低 (25% 増) | 低 |
| Blue-Green | なし | 即時 (切り替え) | 高 (2倍) | 中 |
| Canary | なし | 即時 | 低 (少量追加) | 高 |
| Argo Rollouts | なし | 自動 | 低 | 高 |

---

## 7. Network Policy

### 7.1 デフォルト拒否ポリシー

```yaml
# network-policy-deny-all.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all
  namespace: production
spec:
  podSelector: {}  # 全 Pod に適用
  policyTypes:
    - Ingress
    - Egress
```

### 7.2 アプリケーション固有のポリシー

```yaml
# network-policy-myapp.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: myapp-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Ingress Controller からのトラフィックのみ許可
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
          podSelector:
            matchLabels:
              app.kubernetes.io/name: ingress-nginx
      ports:
        - protocol: TCP
          port: 3000
  egress:
    # DNS クエリを許可
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
    # PostgreSQL への接続を許可
    - to:
        - podSelector:
            matchLabels:
              app: postgresql
      ports:
        - protocol: TCP
          port: 5432
    # Redis への接続を許可
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
    # 外部 API への HTTPS 接続を許可
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
            except:
              - 10.0.0.0/8
              - 172.16.0.0/12
              - 192.168.0.0/16
      ports:
        - protocol: TCP
          port: 443
```

---

## 8. Prometheus / Grafana 監視基盤

### 8.1 kube-prometheus-stack のインストール

```bash
# Prometheus Operator + Grafana のインストール
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  -f monitoring-values.yaml
```

```yaml
# monitoring-values.yaml
grafana:
  adminPassword: "change-me-in-production"
  ingress:
    enabled: true
    ingressClassName: nginx
    hosts:
      - grafana.internal.example.com

prometheus:
  prometheusSpec:
    retention: 15d
    storageSpec:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 50Gi

alertmanager:
  alertmanagerSpec:
    storage:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 10Gi
```

### 8.2 ServiceMonitor (アプリメトリクスの収集)

```yaml
# servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: myapp-monitor
  namespace: production
  labels:
    release: monitoring  # Prometheus Operator のセレクタに合わせる
spec:
  selector:
    matchLabels:
      app.kubernetes.io/name: myapp
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
      scrapeTimeout: 10s
```

### 8.3 PrometheusRule (アラートルール)

```yaml
# prometheusrule.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: myapp-alerts
  namespace: production
  labels:
    release: monitoring
spec:
  groups:
    - name: myapp.rules
      rules:
        # 高エラーレート
        - alert: HighErrorRate
          expr: |
            sum(rate(http_requests_total{service="myapp",status=~"5.."}[5m]))
            / sum(rate(http_requests_total{service="myapp"}[5m])) > 0.05
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "myapp のエラーレートが 5% を超えています"
            description: "直近5分間のエラーレート: {{ $value | humanizePercentage }}"

        # 高レイテンシ
        - alert: HighLatency
          expr: |
            histogram_quantile(0.99,
              sum(rate(http_request_duration_seconds_bucket{service="myapp"}[5m])) by (le)
            ) > 2
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "myapp の P99 レイテンシが 2秒を超えています"

        # Pod の再起動
        - alert: PodCrashLooping
          expr: |
            rate(kube_pod_container_status_restarts_total{
              namespace="production",
              pod=~"myapp-.*"
            }[15m]) * 60 * 15 > 3
          for: 5m
          labels:
            severity: critical
          annotations:
            summary: "Pod {{ $labels.pod }} が頻繁に再起動しています"
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
  replicas: 3  # <- HPA と競合！ helm upgrade のたびにリセットされる

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

### アンチパターン 3: リソースリクエスト/リミットを設定しない

```yaml
# NG: resources を未設定
spec:
  containers:
    - name: app
      image: myapp:latest
      # resources が未設定 -> 無制限にリソースを消費可能
      # -> 同一ノードの他 Pod に影響、OOMKiller のリスク

# OK: 適切なリソース設定
spec:
  containers:
    - name: app
      image: myapp:latest
      resources:
        requests:
          cpu: 100m
          memory: 128Mi
        limits:
          cpu: 500m
          memory: 256Mi
```

**問題点**: リソース制限を設定しない Pod は BestEffort QoS クラスに分類され、ノードのリソースが不足した際に最初に退去 (evict) される。また、1 つの Pod が CPU やメモリを占有して他の Pod に影響を与える「ノイジーネイバー」問題も発生する。requests を設定することでスケジューラが適切にノードを選択でき、limits を設定することで暴走を防止できる。

### アンチパターン 4: PodDisruptionBudget を設定しない

```yaml
# NG: PDB なしでは、ノードのメンテナンス時に全 Pod が同時停止する可能性

# OK: PDB で最小可用数を保証
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: myapp-pdb
spec:
  minAvailable: 2  # 常に最低2 Pod は稼働
  selector:
    matchLabels:
      app: myapp
```

**問題点**: Kubernetes ノードのアップグレードやスケールダウン時に `kubectl drain` が実行される。PDB がないと全 Pod が同時に退去させられ、サービスが一時的にダウンする。本番環境では必ず PDB を設定する。

---

## FAQ

### Q1: Helm Chart はどこに保存するのがベストですか？

**A**: (1) アプリケーションリポジトリ内の `charts/` ディレクトリに同居させる (モノリポ方式)、(2) 専用の Helm Chart リポジトリに分離する (OCI レジストリ or ChartMuseum)。小~中規模なら (1) が管理しやすく、アプリとチャートのバージョンを同期できる。大規模で複数チームが同じ Chart を使う場合は (2) で共有 Chart として管理する。OCI レジストリ (GHCR, ECR 等) が現在の推奨。

### Q2: ConfigMap を変更した場合、Pod は自動的に再起動されますか？

**A**: いいえ。ConfigMap を更新しても既存の Pod は自動再起動されない。対応方法は (1) Deployment の annotation に ConfigMap の checksum を含める (`checksum/config: {{ sha256sum }}`) ことで、ConfigMap 変更時に Pod がローリングアップデートされる、(2) `kubectl rollout restart deployment myapp` で手動再起動、(3) Reloader (stakater/Reloader) を導入して自動検知・再起動。

### Q3: KEDA と HPA の違いは何ですか？

**A**: HPA は CPU/メモリやカスタムメトリクスに基づいてスケールするが、最小 1 Pod が必要 (0 にスケールダウンできない)。KEDA (Kubernetes Event-Driven Autoscaling) はイベントソース (SQS キュー長、Kafka ラグ、Cron 等) に基づいてスケールし、0 Pod までスケールダウンできる (Scale-to-Zero)。API サーバーには HPA、バックグラウンドワーカーやイベントドリブン処理には KEDA が適する。

### Q4: VPA (Vertical Pod Autoscaler) と HPA を同時に使えますか？

**A**: CPU ベースの HPA と CPU ベースの VPA を同時に使うと競合する。推奨パターンは (1) HPA はカスタムメトリクス (RPS 等) でスケールし、VPA は CPU/メモリの requests を最適化する分担方式、(2) VPA を「UpdateMode: Off」で実行し、推奨値のレポートのみに使う (推奨値を人間が確認して手動で反映)。Multidimensional Pod Autoscaler (MPA) は両方を統合する試みだが、まだ成熟していない。

### Q5: Network Policy を設定したが通信できない場合のデバッグ方法は？

**A**: (1) `kubectl describe networkpolicy <name>` でルールが正しいか確認する。(2) Network Policy をサポートする CNI (Calico, Cilium 等) がインストールされているか確認する (デフォルトの Flannel はサポートしない)。(3) DNS (kube-dns / CoreDNS) へのアクセスが Egress ルールで許可されているか確認する (これを忘れると名前解決ができず全通信が失敗する)。(4) `kubectl run debug --rm -it --image=nicolaka/netshoot -- bash` でデバッグ Pod を作成し、`nslookup` や `curl` で接続テストを行う。

### Q6: Argo Rollouts と Flagger の違いは何ですか？

**A**: どちらもプログレッシブデリバリーを実現するが、アプローチが異なる。Argo Rollouts は Deployment の代替リソース (Rollout CRD) を使い、Argo CD との統合が強い。Flagger は既存の Deployment をそのまま使い、Ingress/Service Mesh レベルでトラフィック制御する。Argo エコシステム (Argo CD, Argo Workflows) を使っているなら Rollouts、Istio/Linkerd ベースのサービスメッシュを使っているなら Flagger が自然な選択。

---

## まとめ

| 項目 | 要点 |
|------|------|
| Helm | Chart でマニフェストをテンプレート化。values.yaml で環境差異を吸収 |
| Ingress | ホスト名/パスベースルーティング。cert-manager で自動 TLS |
| ConfigMap | 設定値を外部化。checksum annotation で変更検知 |
| Secret | 機密情報の管理。External Secrets Operator で外部連携推奨 |
| HPA | CPU/メモリ/カスタムメトリクスで水平スケーリング |
| KEDA | イベントドリブンスケーリング。Scale-to-Zero 対応 |
| デプロイ戦略 | RollingUpdate (標準)、Canary (Argo Rollouts)、Blue-Green |
| Network Policy | デフォルト拒否 + ホワイトリストで通信を制御 |
| PDB | ノードメンテナンス時の最小可用性を保証 |
| 監視 | Prometheus + Grafana でメトリクス可視化。PrometheusRule でアラート |

## 次に読むべきガイド

- [コンテナセキュリティ](../06-security/00-container-security.md) -- K8s 上のセキュリティベストプラクティス
- [サプライチェーンセキュリティ](../06-security/01-supply-chain-security.md) -- イメージ署名と SBOM
- Docker Compose から Kubernetes への移行 -- Kompose を使った移行パターン

## 参考文献

1. **Helm 公式ドキュメント** -- https://helm.sh/docs/ -- Helm Chart の作成と管理の包括的リファレンス
2. **Kubernetes Ingress** -- https://kubernetes.io/docs/concepts/services-networking/ingress/ -- Ingress リソースの公式仕様
3. **HPA 公式ドキュメント** -- https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/ -- HPA の設定と動作の詳細
4. **External Secrets Operator** -- https://external-secrets.io/ -- 外部シークレット管理との統合
5. **KEDA 公式ドキュメント** -- https://keda.sh/docs/ -- イベントドリブンオートスケーリングの設定ガイド
6. **Argo Rollouts** -- https://argo-rollouts.readthedocs.io/ -- プログレッシブデリバリーの実装リファレンス
7. **cert-manager** -- https://cert-manager.io/docs/ -- Kubernetes での TLS 証明書の自動管理
8. **Kyverno** -- https://kyverno.io/docs/ -- Kubernetes ポリシー管理とセキュリティ適用
