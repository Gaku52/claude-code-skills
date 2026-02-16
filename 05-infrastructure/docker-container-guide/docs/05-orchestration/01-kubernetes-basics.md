# Kubernetes基礎

> Pod / Service / Deployment の3大リソースとkubectlの基本操作を通じて、Kubernetesの宣言的なコンテナ管理を習得する。

---

## この章で学ぶこと

1. **Pod / Service / Deployment の役割と関係性**を理解する
2. **kubectl を使った基本的なクラスタ操作**を習得する
3. **マニフェストファイル（YAML）の記述方法**とminikubeでの実践ができるようになる
4. **ConfigMap / Secret / PersistentVolume** などの関連リソースを理解する
5. **Ingress / HPA / RBAC** の基礎概念を把握する

---

## 1. Kubernetesアーキテクチャ

### クラスタの構成

```
┌─────────────────── Control Plane ───────────────────────┐
│                                                         │
│  ┌────────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ API Server │  │   etcd   │  │ Controller Manager│  │
│  │            │  │ (KVS)    │  │                   │  │
│  │ 全操作の   │  │ クラスタ │  │ Desired State を  │  │
│  │ エントリ   │  │ 状態保存 │  │ 維持するループ    │  │
│  └─────┬──────┘  └──────────┘  └───────────────────┘  │
│        │                                                │
│  ┌─────▼──────┐                                        │
│  │ Scheduler  │  Podをどのノードに配置するか決定        │
│  └────────────┘                                        │
└─────────────────────────────────────────────────────────┘
         │
         │ kubelet通信
         ▼
┌─────────────────── Worker Node ─────────────────────────┐
│                                                         │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐ │
│  │  kubelet   │  │ kube-proxy │  │ Container Runtime│ │
│  │            │  │            │  │ (containerd)     │ │
│  │ Pod管理    │  │ ネットワーク │  │ コンテナ実行     │ │
│  └────────────┘  │ ルーティング│  └──────────────────┘ │
│                  └────────────┘                         │
│  ┌──────────────────────────────────────┐              │
│  │  Pod        Pod        Pod           │              │
│  │ ┌──────┐  ┌──────┐  ┌──────┐       │              │
│  │ │ ctr  │  │ ctr  │  │ ctr  │       │              │
│  │ └──────┘  └──────┘  │ ctr  │       │              │
│  │                      └──────┘       │              │
│  └──────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────┘
```

### Control Planeの各コンポーネント詳細

| コンポーネント | 役割 | 障害時の影響 |
|---|---|---|
| API Server | 全リクエストの認証・認可・受付 | kubectlやコントローラーが操作不能 |
| etcd | クラスタ状態の唯一の永続ストア | 全状態が失われる（最重要） |
| Controller Manager | Desired State維持のコントロールループ群 | 自動復旧・スケーリングが停止 |
| Scheduler | Podのノード配置決定 | 新規Podが配置されない |

### Worker Nodeの各コンポーネント詳細

| コンポーネント | 役割 | 障害時の影響 |
|---|---|---|
| kubelet | Pod の作成・監視・レポート | ノード上のPodが管理不能 |
| kube-proxy | Service のネットワークルール管理 | Service 経由のルーティング不能 |
| Container Runtime | コンテナの実行（containerd, CRI-O） | コンテナの起動・停止不能 |

### Kubernetesの宣言的管理モデル

Kubernetesの最も重要な設計思想は**宣言的管理（Declarative Management）**である。命令的（Imperative）な「コンテナを3つ起動せよ」ではなく、宣言的（Declarative）な「コンテナは3つあるべき」と記述する。

```
宣言的管理の流れ:

ユーザー                    API Server                Controller
  │                           │                         │
  │ "replicas: 3" を apply    │                         │
  │──────────────────────────►│                         │
  │                           │ etcdに保存              │
  │                           │────────►                │
  │                           │                         │
  │                           │  現在の状態を監視       │
  │                           │◄────────────────────────│
  │                           │                         │
  │                           │  「2つしかない」検知    │
  │                           │────────────────────────►│
  │                           │                         │
  │                           │  Pod 1つ追加            │
  │                           │◄────────────────────────│
  │                           │                         │
  │                           │  Desired = Current      │
  │                           │  → Reconciliation完了   │
```

この Reconciliation Loop（調整ループ）が常に動作しているため、障害やノードダウンが発生しても自動的に desired state に戻る。

---

## 2. Pod

Kubernetesの最小デプロイ単位。1つ以上のコンテナを含み、同一Podのコンテナはネットワークとストレージを共有する。

### コード例1: Pod マニフェスト

```yaml
# pod.yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    app: my-app
    version: v1
    environment: development
spec:
  containers:
    - name: app
      image: node:20-alpine
      command: ["node", "server.js"]
      ports:
        - containerPort: 3000
          protocol: TCP
      env:
        - name: NODE_ENV
          value: "production"
        - name: DB_HOST
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: database-host
      resources:
        requests:
          memory: "128Mi"
          cpu: "100m"      # 0.1 CPU
        limits:
          memory: "256Mi"
          cpu: "500m"      # 0.5 CPU
      livenessProbe:
        httpGet:
          path: /health
          port: 3000
        initialDelaySeconds: 10
        periodSeconds: 15
      readinessProbe:
        httpGet:
          path: /ready
          port: 3000
        initialDelaySeconds: 5
        periodSeconds: 10
      volumeMounts:
        - name: data
          mountPath: /app/data

  volumes:
    - name: data
      emptyDir: {}

  restartPolicy: Always
```

```bash
# Podの作成
kubectl apply -f pod.yaml

# Podの一覧
kubectl get pods
kubectl get pods -o wide  # ノード情報も表示

# Podの詳細
kubectl describe pod my-app

# Podのログ
kubectl logs my-app
kubectl logs my-app -f  # リアルタイム追従

# Pod内でコマンド実行
kubectl exec -it my-app -- /bin/sh

# Podの削除
kubectl delete pod my-app
```

### Pod のライフサイクル

```
┌──────────────────────────────────────────────┐
│                Pod ライフサイクル              │
│                                              │
│  Pending ──► Running ──► Succeeded           │
│     │           │                            │
│     │           └──► Failed                  │
│     │                                        │
│     └──► (スケジューリング不可)               │
│                                              │
│  Running 中:                                 │
│  ┌─────────────────────────────────┐        │
│  │  Liveness Probe  → 失敗 → 再起動│        │
│  │  Readiness Probe → 失敗 → Service│        │
│  │                     から除外     │        │
│  │  Startup Probe   → 起動完了判定 │        │
│  └─────────────────────────────────┘        │
└──────────────────────────────────────────────┘
```

### マルチコンテナパターン

1つのPodに複数のコンテナを配置するユースケースには、代表的な3つのパターンがある。

```yaml
# sidecar-pattern.yaml
# サイドカーパターン: ログ収集エージェントを同居
apiVersion: v1
kind: Pod
metadata:
  name: app-with-sidecar
spec:
  containers:
    # メインアプリケーション
    - name: app
      image: my-app:latest
      ports:
        - containerPort: 8080
      volumeMounts:
        - name: log-volume
          mountPath: /var/log/app

    # サイドカー: ログを収集して外部に転送
    - name: log-collector
      image: fluent/fluent-bit:latest
      volumeMounts:
        - name: log-volume
          mountPath: /var/log/app
          readOnly: true
        - name: fluentbit-config
          mountPath: /fluent-bit/etc

  volumes:
    - name: log-volume
      emptyDir: {}
    - name: fluentbit-config
      configMap:
        name: fluentbit-config
```

```yaml
# init-container.yaml
# Init Container: メインコンテナ起動前に前処理を実行
apiVersion: v1
kind: Pod
metadata:
  name: app-with-init
spec:
  initContainers:
    # DBマイグレーションを実行してから起動
    - name: db-migration
      image: my-app:latest
      command: ["npm", "run", "migrate"]
      env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url

    # 外部サービスの起動を待つ
    - name: wait-for-redis
      image: busybox:latest
      command: ['sh', '-c', 'until nc -z redis-service 6379; do echo waiting for redis; sleep 2; done']

  containers:
    - name: app
      image: my-app:latest
      ports:
        - containerPort: 8080
```

| パターン | 用途 | 例 |
|---|---|---|
| Sidecar | メインコンテナの補助 | ログ収集、プロキシ、モニタリング |
| Ambassador | 外部通信のプロキシ | DBプロキシ、APIゲートウェイ |
| Adapter | 出力の変換 | ログフォーマット変換、メトリクス変換 |

---

## 3. Deployment

Podのレプリカ管理、ローリングアップデート、ロールバックを担うコントローラー。本番環境では直接Podを作らず、Deploymentを使う。

### コード例2: Deployment マニフェスト

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  labels:
    app: web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 更新時に追加で作成するPod数
      maxUnavailable: 0  # 更新時に停止を許容するPod数
  template:
    metadata:
      labels:
        app: web-app
        version: v1.0.0
    spec:
      containers:
        - name: web
          image: my-app:1.0.0
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
      terminationGracePeriodSeconds: 30
```

```bash
# Deploymentの作成
kubectl apply -f deployment.yaml

# Deploymentの一覧
kubectl get deployments

# ローリングアップデート（イメージ変更）
kubectl set image deployment/web-app web=my-app:2.0.0

# アップデートの進捗確認
kubectl rollout status deployment/web-app

# 更新履歴の確認
kubectl rollout history deployment/web-app

# ロールバック（前のバージョンに戻す）
kubectl rollout undo deployment/web-app

# 特定リビジョンにロールバック
kubectl rollout undo deployment/web-app --to-revision=2

# スケーリング
kubectl scale deployment/web-app --replicas=5
```

### Deployment / ReplicaSet / Pod の関係

```
┌─────────────────────────────────────────────────┐
│  Deployment: web-app                            │
│  (ローリングアップデート、ロールバック管理)        │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  ReplicaSet: web-app-7d9b8c6f5            │ │
│  │  (現在のバージョンのPodレプリカを管理)      │ │
│  │                                           │ │
│  │  ┌────────┐  ┌────────┐  ┌────────┐     │ │
│  │  │ Pod 1  │  │ Pod 2  │  │ Pod 3  │     │ │
│  │  │ v1.0.0 │  │ v1.0.0 │  │ v1.0.0 │     │ │
│  │  └────────┘  └────────┘  └────────┘     │ │
│  └───────────────────────────────────────────┘ │
│                                                 │
│  ┌───────────────────────────────────────────┐ │
│  │  ReplicaSet: web-app-5f4d3e2a1 (旧)      │ │
│  │  replicas: 0 (ロールバック用に保持)        │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### ローリングアップデートの流れ

```
maxSurge: 1, maxUnavailable: 0 の場合:

Step 1: 新Pod 1つ追加
  旧v1: [●] [●] [●]    replicas=3
  新v2: [○]             追加中...

Step 2: 新Pod Ready → 旧Pod 1つ削除
  旧v1: [●] [●]         replicas=2
  新v2: [●]             Ready!

Step 3: 新Pod 追加 → 旧Pod 削除（繰り返し）
  旧v1: [●]             replicas=1
  新v2: [●] [●]

Step 4: 完了
  旧v1:                  replicas=0
  新v2: [●] [●] [●]    全Pod更新完了
```

### デプロイ戦略の比較

```yaml
# Recreate戦略: 全Pod停止 → 全Pod起動（ダウンタイムあり）
spec:
  strategy:
    type: Recreate

# RollingUpdate戦略: 段階的に更新（ダウンタイムなし）
spec:
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 25%         # replicas の 25% まで追加Pod許容
      maxUnavailable: 25%   # replicas の 25% まで停止許容
```

| 戦略 | ダウンタイム | リソース消費 | 用途 |
|---|---|---|---|
| RollingUpdate | なし | 一時的に増加 | 本番環境（デフォルト） |
| Recreate | あり | 一定 | DBスキーマ変更など同時稼働不可の場合 |
| Blue/Green（手動） | なし | 2倍 | 完全なバージョン切替が必要な場合 |
| Canary（手動） | なし | 微増 | 段階的リリースで影響を最小化 |

---

## 4. Service

Podの集合に対して安定したネットワークエンドポイント（IPアドレス、DNS名）を提供する抽象化レイヤー。Podは一時的でIPが変わるが、Serviceは不変。

### コード例3: Service マニフェスト

```yaml
# === ClusterIP (クラスタ内部からのみアクセス) ===
apiVersion: v1
kind: Service
metadata:
  name: api-service
spec:
  type: ClusterIP          # デフォルト
  selector:
    app: web-app           # このラベルを持つPodにルーティング
  ports:
    - port: 80             # Serviceが受けるポート
      targetPort: 8080     # Podのポート
      protocol: TCP

---
# === NodePort (各ノードのIPでアクセス) ===
apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080      # 30000-32767 の範囲

---
# === LoadBalancer (クラウドLBを自動プロビジョニング) ===
apiVersion: v1
kind: Service
metadata:
  name: web-lb
spec:
  type: LoadBalancer
  selector:
    app: web-app
  ports:
    - port: 80
      targetPort: 8080
```

### Service タイプの比較表

| タイプ | アクセス範囲 | IP | 用途 |
|--------|------------|-----|------|
| ClusterIP | クラスタ内部のみ | 仮想IP (10.x.x.x) | 内部通信（デフォルト） |
| NodePort | 外部 + 内部 | ノードIP:30000-32767 | 開発、オンプレ |
| LoadBalancer | 外部 + 内部 | クラウドLBのIP | 本番（クラウド） |
| ExternalName | DNS CNAME | 外部DNS名 | 外部サービスへのエイリアス |

### Serviceのルーティング

```
外部クライアント
    │
    │  LoadBalancer IP: 203.0.113.50:80
    ▼
┌──────────────────────────────────────────┐
│  Service: web-lb (10.96.0.10:80)        │
│  selector: app=web-app                   │
│                                          │
│  Endpoints:                              │
│    172.17.0.5:8080 (Pod 1)              │
│    172.17.0.6:8080 (Pod 2)              │
│    172.17.0.7:8080 (Pod 3)              │
│                                          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │  Pod 1  │ │  Pod 2  │ │  Pod 3  │  │
│  │  :8080  │ │  :8080  │ │  :8080  │  │
│  │app=     │ │app=     │ │app=     │  │
│  │web-app  │ │web-app  │ │web-app  │  │
│  └─────────┘ └─────────┘ └─────────┘  │
└──────────────────────────────────────────┘
```

### Service DNS

Kubernetesクラスタ内ではCoreDNSにより、Serviceに自動でDNS名が割り当てられる。

```
# Service DNS命名規則
<service-name>.<namespace>.svc.cluster.local

# 例
api-service.myapp.svc.cluster.local
postgres-service.myapp.svc.cluster.local

# 同一Namespace内では省略可能
api-service        # 同一Namespace
api-service.myapp  # 異なるNamespaceから
```

```yaml
# DNS名を使ったアプリケーション設定例
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: myapp
data:
  # 同一Namespace内のService名をそのまま使用
  DATABASE_HOST: "postgres-service"
  REDIS_HOST: "redis-service"
  # 異なるNamespaceのServiceにはFQDNで指定
  AUTH_SERVICE: "auth-service.auth-system.svc.cluster.local"
```

---

## 5. Namespace

リソースの論理的な分離単位。チーム別、環境別にリソースを分割する。

### コード例4: Namespace の管理

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: staging
  labels:
    environment: staging
---
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    environment: production
```

```bash
# Namespaceの作成
kubectl apply -f namespace.yaml

# 特定Namespaceにリソースをデプロイ
kubectl apply -f deployment.yaml -n staging

# Namespace内のリソース一覧
kubectl get all -n staging

# デフォルトNamespaceの変更
kubectl config set-context --current --namespace=staging

# 全Namespaceのリソース一覧
kubectl get pods --all-namespaces
kubectl get pods -A  # 短縮形
```

### ResourceQuota でNamespace単位のリソース制限

```yaml
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: staging-quota
  namespace: staging
spec:
  hard:
    # コンピューティングリソース
    requests.cpu: "4"
    requests.memory: 8Gi
    limits.cpu: "8"
    limits.memory: 16Gi
    # オブジェクト数
    pods: "20"
    services: "10"
    persistentvolumeclaims: "5"
    configmaps: "20"
    secrets: "20"
```

```yaml
# limit-range.yaml
# Namespace内の個別Pod/コンテナのデフォルト値と上限を設定
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: staging
spec:
  limits:
    - type: Container
      default:           # limits のデフォルト値
        cpu: "500m"
        memory: "256Mi"
      defaultRequest:    # requests のデフォルト値
        cpu: "100m"
        memory: "128Mi"
      max:               # 上限
        cpu: "2"
        memory: "2Gi"
      min:               # 下限
        cpu: "50m"
        memory: "64Mi"
```

---

## 6. ConfigMap と Secret

### ConfigMap: 非機密設定データの管理

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: myapp
data:
  # 単純なKey-Value
  DATABASE_HOST: "postgres-service"
  DATABASE_PORT: "5432"
  LOG_LEVEL: "info"
  CACHE_TTL: "3600"

  # ファイルとしてマウント可能な設定
  nginx.conf: |
    server {
      listen 80;
      server_name localhost;
      location / {
        proxy_pass http://api-service:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
      }
    }

  application.yaml: |
    server:
      port: 8080
    spring:
      datasource:
        url: jdbc:postgresql://postgres-service:5432/myapp
```

```bash
# ファイルからConfigMap作成
kubectl create configmap nginx-config --from-file=nginx.conf

# リテラルからConfigMap作成
kubectl create configmap app-config \
  --from-literal=DATABASE_HOST=postgres-service \
  --from-literal=LOG_LEVEL=info

# ConfigMapの内容確認
kubectl get configmap app-config -o yaml
```

### Secret: 機密データの管理

```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: app-secret
  namespace: myapp
type: Opaque
data:
  # base64エンコード済み（echo -n 'value' | base64）
  DB_PASSWORD: c2VjcmV0MTIz
  API_KEY: bXktYXBpLWtleS0xMjM0NQ==
  JWT_SECRET: c3VwZXItc2VjcmV0LWtleQ==

---
# stringData を使えばプレーンテキストで記述可能（適用時に自動でbase64変換）
apiVersion: v1
kind: Secret
metadata:
  name: app-secret-plain
  namespace: myapp
type: Opaque
stringData:
  DB_PASSWORD: secret123
  API_KEY: my-api-key-12345
```

```bash
# コマンドラインからSecret作成
kubectl create secret generic db-secret \
  --from-literal=password=secret123 \
  --from-literal=username=admin

# TLS Secret作成
kubectl create secret tls tls-secret \
  --cert=tls.crt \
  --key=tls.key

# Docker Registry Secret
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=myuser \
  --docker-password=mytoken

# Secret の内容確認（base64デコード）
kubectl get secret app-secret -o jsonpath='{.data.DB_PASSWORD}' | base64 -d
```

### ConfigMap / Secret の利用パターン

```yaml
# 環境変数として注入
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    spec:
      containers:
        - name: api
          image: my-api:latest
          env:
            # ConfigMap から個別Key
            - name: DB_HOST
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: DATABASE_HOST
            # Secret から個別Key
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: app-secret
                  key: DB_PASSWORD
          # ConfigMap の全Keyを一括で環境変数に
          envFrom:
            - configMapRef:
                name: app-config
            - secretRef:
                name: app-secret

          # ファイルとしてマウント
          volumeMounts:
            - name: config-volume
              mountPath: /etc/nginx/conf.d
              readOnly: true
            - name: secret-volume
              mountPath: /etc/secrets
              readOnly: true

      volumes:
        - name: config-volume
          configMap:
            name: app-config
            items:
              - key: nginx.conf
                path: default.conf
        - name: secret-volume
          secret:
            secretName: app-secret
            defaultMode: 0400  # 読み取り専用パーミッション
```

---

## 7. PersistentVolume と PersistentVolumeClaim

### 永続ストレージの概念

```
┌────────────────────────────────────────────────┐
│                Storageの3層構造                  │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  PersistentVolumeClaim (PVC)             │ │
│  │  「10Gi の ReadWriteOnce が必要」         │ │
│  │  → Podが要求するストレージの仕様          │ │
│  └─────────────────┬────────────────────────┘ │
│                    │ バインド                   │
│  ┌─────────────────▼────────────────────────┐ │
│  │  PersistentVolume (PV)                   │ │
│  │  「AWS EBS 10Gi がある」                  │ │
│  │  → 管理者がプロビジョニングした実体        │ │
│  └─────────────────┬────────────────────────┘ │
│                    │                           │
│  ┌─────────────────▼────────────────────────┐ │
│  │  StorageClass                            │ │
│  │  「gp3タイプのEBSを自動作成」             │ │
│  │  → 動的プロビジョニングのテンプレート      │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

### コード例: PersistentVolume / PVC

```yaml
# storage-class.yaml
# 動的プロビジョニング用のStorageClass
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-storage
provisioner: kubernetes.io/aws-ebs   # クラウドプロバイダに応じて変更
parameters:
  type: gp3
  fsType: ext4
reclaimPolicy: Retain   # Delete | Retain | Recycle
volumeBindingMode: WaitForFirstConsumer
allowVolumeExpansion: true

---
# pvc.yaml
# アプリケーションが要求するストレージ
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data
  namespace: myapp
spec:
  accessModes:
    - ReadWriteOnce     # 単一ノードからRead/Write
  storageClassName: fast-storage
  resources:
    requests:
      storage: 10Gi

---
# PVCをPodで使用
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: myapp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16-alpine
          ports:
            - containerPort: 5432
          volumeMounts:
            - name: postgres-storage
              mountPath: /var/lib/postgresql/data
              subPath: pgdata    # サブディレクトリにマウント
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: POSTGRES_PASSWORD
      volumes:
        - name: postgres-storage
          persistentVolumeClaim:
            claimName: postgres-data
```

### AccessMode の種類

| AccessMode | 略称 | 説明 |
|---|---|---|
| ReadWriteOnce | RWO | 単一ノードからRead/Write |
| ReadOnlyMany | ROX | 複数ノードからReadOnly |
| ReadWriteMany | RWX | 複数ノードからRead/Write |
| ReadWriteOncePod | RWOP | 単一PodからRead/Write（K8s 1.27+） |

```bash
# PVCの状態確認
kubectl get pvc -n myapp
# NAME            STATUS   VOLUME         CAPACITY   ACCESS MODES
# postgres-data   Bound    pvc-abc123     10Gi       RWO

# PVの一覧
kubectl get pv
# NAME         CAPACITY   RECLAIM POLICY   STATUS
# pvc-abc123   10Gi       Retain           Bound
```

---

## 8. Ingress

### L7ロードバランシング

IngressはHTTP/HTTPSレイヤーでのルーティングを提供し、1つのIPアドレスで複数のServiceにルーティングする。

```
インターネット
    │
    ▼
┌─────────────────────────────────────────┐
│  Ingress Controller (nginx, traefik等)   │
│                                          │
│  ルール:                                 │
│  api.example.com  → api-service:80      │
│  web.example.com  → web-service:80      │
│  example.com/docs → docs-service:80     │
└─────────────────────────────────────────┘
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌────────┐   ┌────────┐
│  API   │  │  Web   │   │  Docs  │
│Service │  │Service │   │Service │
└────────┘  └────────┘   └────────┘
```

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  namespace: myapp
  annotations:
    # nginx Ingress Controller用の設定
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    # cert-manager でTLS証明書を自動取得
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - api.example.com
        - web.example.com
      secretName: app-tls
  rules:
    # ホストベースのルーティング
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
    - host: web.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-service
                port:
                  number: 80
    # パスベースのルーティング
    - host: example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
          - path: /
            pathType: Prefix
            backend:
              service:
                name: web-service
                port:
                  number: 80
```

```bash
# minikube で Ingress を有効化
minikube addons enable ingress

# Ingress の状態確認
kubectl get ingress -n myapp
# NAME          CLASS   HOSTS                              ADDRESS        PORTS
# app-ingress   nginx   api.example.com,web.example.com    192.168.49.2   80, 443

# Ingress の詳細
kubectl describe ingress app-ingress -n myapp
```

---

## 9. HPA（Horizontal Pod Autoscaler）

### 負荷に応じた自動スケーリング

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: myapp
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    # CPU使用率ベース
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70   # CPU使用率70%を超えたらスケールアウト
    # メモリ使用率ベース
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60   # スケールアップの安定化期間
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60            # 60秒ごとに最大2Pod追加
    scaleDown:
      stabilizationWindowSeconds: 300  # スケールダウンの安定化期間（5分）
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60            # 60秒ごとに最大10%削減
```

```bash
# HPAの状態確認
kubectl get hpa -n myapp
# NAME      REFERENCE        TARGETS         MINPODS   MAXPODS   REPLICAS
# api-hpa   Deployment/api   45%/70%,60%/80%   2         10        3

# HPAの詳細
kubectl describe hpa api-hpa -n myapp

# metrics-server のインストール（minikube）
minikube addons enable metrics-server

# リアルタイムのリソース使用量確認
kubectl top pods -n myapp
kubectl top nodes
```

### HPA の動作フロー

```
                    metrics-server
                         │
                         │ メトリクス収集
                         ▼
┌────────────────────────────────────────┐
│  HPA Controller                        │
│                                        │
│  現在CPU: 85%  目標: 70%               │
│  現在Replicas: 3                       │
│                                        │
│  必要Replicas = ceil(3 × 85/70) = 4   │
│  → スケールアウト: 3 → 4 Pod          │
└────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────┐
│  Deployment: api (replicas: 3 → 4)    │
│  [Pod1] [Pod2] [Pod3] [Pod4 ← NEW]   │
└────────────────────────────────────────┘
```

---

## 10. RBAC（Role-Based Access Control）

### アクセス制御の基本

```yaml
# role.yaml
# Namespace スコープの権限定義
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: developer-role
  namespace: staging
rules:
  # Pod の閲覧・ログ確認
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  # Deployment の管理
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list", "watch", "create", "update", "patch"]
  # Service の管理
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["get", "list", "watch", "create", "update"]
  # ConfigMap の閲覧
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]

---
# rolebinding.yaml
# ユーザーに Role を割り当て
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: developer-binding
  namespace: staging
subjects:
  - kind: User
    name: developer@example.com
    apiGroup: rbac.authorization.k8s.io
  - kind: Group
    name: developers
    apiGroup: rbac.authorization.k8s.io
  - kind: ServiceAccount
    name: ci-deployer
    namespace: staging
roleRef:
  kind: Role
  name: developer-role
  apiGroup: rbac.authorization.k8s.io
```

```yaml
# cluster-role.yaml
# クラスタ全体スコープの権限定義
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: readonly-cluster
rules:
  - apiGroups: [""]
    resources: ["pods", "services", "configmaps", "namespaces"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: readonly-binding
subjects:
  - kind: Group
    name: viewers
    apiGroup: rbac.authorization.k8s.io
roleRef:
  kind: ClusterRole
  name: readonly-cluster
  apiGroup: rbac.authorization.k8s.io
```

```bash
# 権限の確認
kubectl auth can-i create deployments --namespace staging
# yes

kubectl auth can-i delete pods --namespace production
# no

# 特定ユーザーの権限確認
kubectl auth can-i list pods --namespace staging --as developer@example.com
```

---

## 11. minikubeでの実践

### コード例5: minikubeクラスタの構築と操作

```bash
# minikubeのインストール（macOS）
brew install minikube

# クラスタの起動
minikube start --driver=docker --memory=4096 --cpus=2

# クラスタの状態確認
minikube status
kubectl cluster-info
kubectl get nodes

# ダッシュボードの起動
minikube dashboard

# === 実践: Webアプリケーションのデプロイ ===

# 1. Deploymentの作成
kubectl create deployment hello-web --image=nginx:alpine --replicas=3

# 2. Serviceの作成（NodePort）
kubectl expose deployment hello-web --type=NodePort --port=80

# 3. minikubeでServiceにアクセス
minikube service hello-web --url
# → http://192.168.49.2:30123 のようなURLが表示される

# 4. スケーリング
kubectl scale deployment hello-web --replicas=5
kubectl get pods -w  # リアルタイムで監視

# 5. ローリングアップデート
kubectl set image deployment/hello-web nginx=nginx:1.25-alpine
kubectl rollout status deployment/hello-web

# 6. ロールバック
kubectl rollout undo deployment/hello-web

# 7. クリーンアップ
kubectl delete service hello-web
kubectl delete deployment hello-web

# minikubeの停止・削除
minikube stop
minikube delete
```

### minikube アドオン

```bash
# 利用可能なアドオン一覧
minikube addons list

# よく使うアドオンの有効化
minikube addons enable ingress          # Ingress Controller
minikube addons enable metrics-server   # HPAに必要
minikube addons enable dashboard        # Web UI
minikube addons enable registry         # ローカルレジストリ
minikube addons enable storage-provisioner  # 動的PV

# minikube 内のDockerデーモンを使用（ローカルイメージを使う）
eval $(minikube docker-env)
docker build -t my-app:latest .
# → imagePullPolicy: Never でローカルイメージを使用可能
```

---

## 12. 完全なアプリケーション例

### コード例6: フロントエンド + API + DB の構成

```yaml
# complete-app.yaml

# === Namespace ===
apiVersion: v1
kind: Namespace
metadata:
  name: myapp
---

# === ConfigMap ===
apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
  namespace: myapp
data:
  DATABASE_HOST: "postgres-service"
  DATABASE_NAME: "myapp"
---

# === Secret ===
apiVersion: v1
kind: Secret
metadata:
  name: db-secret
  namespace: myapp
type: Opaque
data:
  POSTGRES_PASSWORD: c2VjcmV0MTIz  # base64エンコード

---

# === PostgreSQL Deployment ===
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: myapp
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16-alpine
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: POSTGRES_PASSWORD
            - name: POSTGRES_DB
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: DATABASE_NAME
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
---

# === PostgreSQL Service ===
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: myapp
spec:
  selector:
    app: postgres
  ports:
    - port: 5432
      targetPort: 5432
  type: ClusterIP

---

# === API Deployment ===
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: myapp
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
        - name: api
          image: my-api:latest
          ports:
            - containerPort: 8080
          env:
            - name: DB_HOST
              valueFrom:
                configMapKeyRef:
                  name: app-config
                  key: DATABASE_HOST
            - name: DB_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: db-secret
                  key: POSTGRES_PASSWORD
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
          resources:
            requests:
              memory: "128Mi"
              cpu: "100m"
            limits:
              memory: "256Mi"
              cpu: "500m"

---

# === API Service ===
apiVersion: v1
kind: Service
metadata:
  name: api-service
  namespace: myapp
spec:
  selector:
    app: api
  ports:
    - port: 80
      targetPort: 8080
  type: LoadBalancer
```

```bash
# 一括デプロイ
kubectl apply -f complete-app.yaml

# 状態確認
kubectl get all -n myapp

# 出力例:
# NAME                           READY   STATUS    RESTARTS   AGE
# pod/postgres-6d4f5c7b8-x2k9p  1/1     Running   0          2m
# pod/api-7f8d9e6a5-abc12       1/1     Running   0          2m
# pod/api-7f8d9e6a5-def34       1/1     Running   0          2m
#
# NAME                      TYPE           CLUSTER-IP     EXTERNAL-IP
# service/postgres-service  ClusterIP      10.96.0.15     <none>
# service/api-service       LoadBalancer   10.96.0.20     <pending>
```

---

## 13. kubectl コマンドチートシート

### コード例7: よく使うkubectlコマンド

```bash
# === リソース確認 ===
kubectl get pods                      # Pod一覧
kubectl get pods -o wide              # 詳細一覧（ノード、IP含む）
kubectl get pods -o yaml              # YAML形式で出力
kubectl get all                       # 全リソース一覧
kubectl get events --sort-by='.lastTimestamp'  # イベント（トラブルシュート用）

# === デバッグ ===
kubectl describe pod <pod-name>       # Pod詳細（イベント含む）
kubectl logs <pod-name>               # ログ出力
kubectl logs <pod-name> -c <container># マルチコンテナPodの特定コンテナ
kubectl logs <pod-name> --previous    # 前回クラッシュ時のログ
kubectl exec -it <pod-name> -- sh    # Pod内シェル

# === リソース管理 ===
kubectl apply -f manifest.yaml        # 作成/更新
kubectl delete -f manifest.yaml       # 削除
kubectl diff -f manifest.yaml         # 差分確認

# === ポートフォワード（ローカルからPodに直接アクセス） ===
kubectl port-forward pod/my-pod 8080:80
kubectl port-forward svc/my-service 8080:80
```

### 高度なデバッグコマンド

```bash
# === トラブルシューティングフロー ===

# 1. Pod が起動しない場合
kubectl get pods -n myapp                     # STATUS確認
kubectl describe pod <pod-name> -n myapp      # Events欄を確認
kubectl get events -n myapp --sort-by='.lastTimestamp'

# 2. CrashLoopBackOff の場合
kubectl logs <pod-name> -n myapp --previous   # 前回のクラッシュログ
kubectl describe pod <pod-name> -n myapp      # Exit Code確認

# 3. ImagePullBackOff の場合
kubectl describe pod <pod-name> -n myapp      # イメージ名とタグを確認
# → イメージ名のtypo、タグの不存在、レジストリ認証の問題

# 4. Pending の場合
kubectl describe pod <pod-name> -n myapp      # Schedulerのメッセージ確認
kubectl get nodes                             # ノードの状態確認
kubectl describe node <node-name>             # ノードのリソース使用状況

# 5. ネットワーク問題のデバッグ
kubectl run debug --rm -it --image=busybox -- sh
# Pod内で:
nslookup api-service.myapp.svc.cluster.local
wget -qO- http://api-service.myapp:80/health

# 6. 一時的なデバッグコンテナ（ephemeral container）
kubectl debug -it <pod-name> --image=busybox --target=app

# === リソース使用量の確認 ===
kubectl top pods -n myapp
kubectl top pods -n myapp --sort-by=memory
kubectl top nodes

# === JSONPath でフィルタリング ===
# 全PodのIPアドレスを取得
kubectl get pods -o jsonpath='{.items[*].status.podIP}'

# Running状態のPod名を取得
kubectl get pods --field-selector=status.phase=Running -o name

# 特定ラベルのPodだけ取得
kubectl get pods -l app=api,version=v2

# カスタムカラム出力
kubectl get pods -o custom-columns=\
NAME:.metadata.name,\
STATUS:.status.phase,\
NODE:.spec.nodeName,\
IP:.status.podIP
```

### kubectl コンテキスト管理

```bash
# コンテキスト一覧
kubectl config get-contexts

# 現在のコンテキスト確認
kubectl config current-context

# コンテキスト切り替え
kubectl config use-context minikube
kubectl config use-context production-cluster

# デフォルトNamespace変更
kubectl config set-context --current --namespace=myapp

# kubectx / kubens（便利ツール）
brew install kubectx
kubectx minikube           # コンテキスト切り替え
kubens myapp               # Namespace切り替え
```

---

## 14. Helm入門

### パッケージマネージャの基本

Helmは Kubernetes のパッケージマネージャで、複雑なアプリケーションのデプロイをテンプレート化する。

```bash
# Helmのインストール
brew install helm

# リポジトリの追加
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo update

# Chart の検索
helm search repo nginx
helm search hub prometheus   # Artifact Hub を検索

# Chart のインストール
helm install my-nginx bitnami/nginx -n myapp --create-namespace

# values.yaml でカスタマイズ
helm install my-nginx bitnami/nginx -n myapp \
  -f custom-values.yaml

# リリース一覧
helm list -n myapp

# リリースのアップグレード
helm upgrade my-nginx bitnami/nginx -n myapp \
  --set replicaCount=3

# ロールバック
helm rollback my-nginx 1 -n myapp

# アンインストール
helm uninstall my-nginx -n myapp
```

```yaml
# custom-values.yaml の例（bitnami/nginx）
replicaCount: 3

resources:
  requests:
    memory: "128Mi"
    cpu: "100m"
  limits:
    memory: "256Mi"
    cpu: "500m"

service:
  type: ClusterIP

ingress:
  enabled: true
  hostname: web.example.com
  tls: true
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPU: 70
```

---

## 15. 本番運用のベストプラクティス

### リソース設定のガイドライン

```yaml
# production-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
  namespace: production
  labels:
    app: api
    version: v2.1.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: api
        version: v2.1.0
      annotations:
        # Prometheus メトリクス収集
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      # 同一ノードに集中しないよう分散
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: kubernetes.io/hostname
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: api

      # Pod の優先度
      priorityClassName: high-priority

      # サービスアカウント
      serviceAccountName: api-sa
      automountServiceAccountToken: false

      # セキュリティコンテキスト
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000

      containers:
        - name: api
          image: ghcr.io/myorg/api:v2.1.0   # latest禁止、バージョン固定
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8080
              name: http
          resources:
            requests:
              memory: "256Mi"
              cpu: "200m"
            limits:
              memory: "512Mi"
              cpu: "1000m"

          # セキュリティ設定
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]

          # ヘルスチェック
          startupProbe:
            httpGet:
              path: /health
              port: 8080
            failureThreshold: 30
            periodSeconds: 2
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            periodSeconds: 15
            timeoutSeconds: 3
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            periodSeconds: 5
            timeoutSeconds: 3

          # 一時ファイル用
          volumeMounts:
            - name: tmp
              mountPath: /tmp

      volumes:
        - name: tmp
          emptyDir: {}

      # Graceful Shutdown
      terminationGracePeriodSeconds: 60
```

### 本番チェックリスト

| カテゴリ | チェック項目 | 重要度 |
|---|---|---|
| イメージ | `latest` タグを使わずバージョン固定 | 必須 |
| イメージ | プライベートレジストリからの pull | 必須 |
| リソース | requests / limits 両方設定 | 必須 |
| Probe | liveness / readiness / startup 全設定 | 必須 |
| セキュリティ | runAsNonRoot: true | 必須 |
| セキュリティ | readOnlyRootFilesystem: true | 推奨 |
| セキュリティ | capabilities drop ALL | 推奨 |
| 可用性 | replicas 2以上 | 必須 |
| 可用性 | topologySpreadConstraints 設定 | 推奨 |
| 可用性 | PodDisruptionBudget 設定 | 推奨 |
| ネットワーク | NetworkPolicy で通信制限 | 推奨 |
| 監視 | Prometheus メトリクス公開 | 推奨 |

### PodDisruptionBudget

```yaml
# pdb.yaml
# ノードメンテナンス時にも最低限のPod数を維持
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: api-pdb
  namespace: production
spec:
  minAvailable: 2      # 最低2Podは常に稼働
  # または maxUnavailable: 1  # 最大1Podまで停止許容
  selector:
    matchLabels:
      app: api
```

### NetworkPolicy

```yaml
# network-policy.yaml
# API Pod への通信を制限
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Ingress Controller からのHTTPのみ許可
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # PostgreSQL への通信のみ許可
    - to:
        - podSelector:
            matchLabels:
              app: postgres
      ports:
        - protocol: TCP
          port: 5432
    # DNS解決を許可
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
        - protocol: TCP
          port: 53
```

---

## アンチパターン

### アンチパターン1: 直接Podを作成する

```yaml
# NG: Podを直接作成（障害時に自動復旧されない）
apiVersion: v1
kind: Pod
metadata:
  name: web
spec:
  containers:
    - name: web
      image: nginx:alpine

# OK: Deploymentで管理（自動復旧、ローリングアップデート）
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 1
  selector:
    matchLabels:
      app: web
  template:
    # ...
```

**なぜ問題か**: 直接作成したPodは、ノード障害やPodクラッシュ時に自動で再作成されない。Deploymentはレプリカ数を維持し、障害時も宣言された状態に自動復旧する。

### アンチパターン2: リソース制限の未設定

```yaml
# NG: リソース制限なし
containers:
  - name: app
    image: my-app:latest
    # resources 未設定 → ノードの全リソースを消費しうる

# OK: requests と limits を必ず設定
containers:
  - name: app
    image: my-app:latest
    resources:
      requests:
        memory: "128Mi"
        cpu: "100m"
      limits:
        memory: "256Mi"
        cpu: "500m"
```

**なぜ問題か**: リソース制限のないPodは、同一ノード上の他のPodのリソースを奪い、クラスタ全体の不安定化を引き起こす。Schedulerも適切な配置判断ができない。

### アンチパターン3: latest タグの使用

```yaml
# NG: latest タグ（どのバージョンかわからない）
containers:
  - name: app
    image: my-app:latest

# OK: セマンティックバージョニングで固定
containers:
  - name: app
    image: my-app:2.1.0
    # または image digest で完全固定
    # image: my-app@sha256:abc123...
```

**なぜ問題か**: `latest` タグはどのバージョンが動いているか追跡できず、ロールバックも不可能。デプロイの再現性が失われ、同じマニフェストでも異なるイメージが動く可能性がある。

### アンチパターン4: Secretをマニフェストにハードコード

```yaml
# NG: パスワードを平文でマニフェストに記載
containers:
  - name: app
    env:
      - name: DB_PASSWORD
        value: "my-secret-password"   # Gitに残る!

# OK: Secret リソースを使用
containers:
  - name: app
    env:
      - name: DB_PASSWORD
        valueFrom:
          secretKeyRef:
            name: db-secret
            key: password
```

**なぜ問題か**: マニフェストはGitで管理されるため、パスワードがリポジトリの履歴に残る。External Secrets Operator や Sealed Secrets を使い、暗号化された状態でGit管理するのがベストプラクティス。

### アンチパターン5: ヘルスチェック未設定

```yaml
# NG: Probe未設定（異常検知不能）
containers:
  - name: app
    image: my-app:1.0.0
    ports:
      - containerPort: 8080

# OK: 3種類のProbeを適切に設定
containers:
  - name: app
    image: my-app:1.0.0
    ports:
      - containerPort: 8080
    startupProbe:           # 起動完了判定
      httpGet:
        path: /health
        port: 8080
      failureThreshold: 30
      periodSeconds: 2
    livenessProbe:          # 生存確認
      httpGet:
        path: /health
        port: 8080
      periodSeconds: 15
    readinessProbe:         # リクエスト受付可否
      httpGet:
        path: /ready
        port: 8080
      periodSeconds: 5
```

**なぜ問題か**: Probeがないと、アプリケーションがデッドロックや無限ループに陥っても検知できず、異常なPodにリクエストが送り続けられる。startupProbeがないと、起動が遅いアプリケーションがlivenessProbeで誤って再起動される。

---

## FAQ

### Q1: requests と limits の違いは？

- **requests**: Schedulerがノード選択時に使用する最低保証値。この分のリソースは必ず確保される。
- **limits**: 上限値。超過するとCPUはスロットリング、メモリはOOM Killが発生する。

一般的な指針: `requests` は平常時の使用量、`limits` はピーク時の1.5-2倍に設定。

### Q2: livenessProbe と readinessProbe の違いは？

- **livenessProbe**: コンテナが「生きている」か確認。失敗するとコンテナが再起動される。デッドロック検知に有用。
- **readinessProbe**: コンテナが「リクエストを受けられる状態」か確認。失敗するとServiceのエンドポイントから除外される（再起動はしない）。起動時のウォームアップに有用。

### Q3: `kubectl apply` と `kubectl create` の違いは？

`create` は新規作成のみ（既存リソースがあるとエラー）。`apply` は宣言的管理で、リソースが存在しなければ作成、存在すれば更新する。本番運用では常に `apply` を使用する。

### Q4: ConfigMap と Secret はどう使い分ける？

- **ConfigMap**: 非機密の設定データ（DB_HOST、LOG_LEVEL、設定ファイル等）
- **Secret**: 機密データ（パスワード、APIキー、TLS証明書等）

SecretはデフォルトではBase64エンコードのみで暗号化されていない点に注意。本番環境ではEtcd暗号化（EncryptionConfiguration）やExternal Secrets Operatorの使用を推奨する。

### Q5: Namespace はどう設計すべき？

一般的なパターン:

```
# 環境別
default        # 使わない（リソースを置かない）
development    # 開発環境
staging        # ステージング環境
production     # 本番環境

# チーム × 環境
team-a-dev     # チームAの開発
team-a-prod    # チームAの本番
team-b-dev     # チームBの開発

# マイクロサービス別
auth-system    # 認証サービス群
payment        # 決済サービス群
notification   # 通知サービス群
```

ResourceQuota と LimitRange を組み合わせて、Namespace単位でリソースの公平な配分を実現する。

### Q6: minikube と kind の違いは？

| 項目 | minikube | kind |
|---|---|---|
| 用途 | ローカル開発・学習 | CI/CD・テスト向き |
| マルチノード | 可能（v1.10+） | 容易に可能 |
| 起動速度 | やや遅い | 高速 |
| アドオン | 豊富 | 最小限 |
| ドライバ | Docker, VirtualBox等 | Docker専用 |

学習にはminikube、CI/CDにはkindが適している。

### Q7: StatefulSet はいつ使う？

StatefulSetは順序付きの安定したPod管理が必要な場合に使用する。

```yaml
# Deployment との違い
# - Pod名が固定（postgres-0, postgres-1, postgres-2）
# - 起動・停止の順序が保証される
# - 各Podに固定のPersistentVolumeが紐づく
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres-headless  # Headless Service必須
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    # ...
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: fast-storage
        resources:
          requests:
            storage: 10Gi
```

主な用途: データベース、メッセージキュー（Kafka）、分散ストレージ（Elasticsearch）

### Q8: Kubernetes上でのログ管理はどうする？

```
アプリケーション
  │ stdout/stderr
  ▼
kubelet (ノード上のログファイル)
  │
  ▼
┌──────────────────────────────────────┐
│  ログ収集パターン                      │
│                                      │
│  1. サイドカー: Fluent Bit/Fluentd   │
│     → 各Podにログ転送コンテナを配置   │
│                                      │
│  2. DaemonSet: Fluent Bit/Fluentd    │
│     → 各ノードに1つのログ収集Pod     │
│     → ノード上のログファイルを収集     │
│                                      │
│  3. 直接転送: アプリから直接送信      │
│     → 集中ログサービスに直接出力      │
└──────────────────────────────────────┘
  │
  ▼
Elasticsearch / Loki / CloudWatch 等
  │
  ▼
Kibana / Grafana 等で可視化
```

推奨は DaemonSet パターン。アプリケーションはstdout/stderrにJSON形式でログを出力し、ノード単位のDaemonSetが収集・転送する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Pod | 最小デプロイ単位。直接作成せずDeploymentで管理 |
| Deployment | レプリカ管理、ローリングアップデート、ロールバック |
| Service | 安定したネットワークエンドポイント。ClusterIP/NodePort/LoadBalancer |
| Namespace | リソースの論理的分離。環境・チーム別に使用 |
| ConfigMap/Secret | 設定と機密情報を分離管理。環境変数またはファイルマウント |
| PersistentVolume | Pod再起動後もデータ保持。StorageClassで動的プロビジョニング |
| Ingress | L7ルーティング。ホスト/パスベースで複数Serviceへ振り分け |
| HPA | CPU/メモリベースの自動水平スケーリング |
| RBAC | Role/ClusterRoleでアクセス制御。最小権限の原則 |
| Helm | パッケージマネージャ。複雑なデプロイをテンプレート化 |
| kubectl | `apply` で宣言的管理。`describe` と `logs` でデバッグ |
| リソース制限 | requests/limits 必須。Scheduler配置とOOM防止 |
| Probe | startup=起動判定、liveness=再起動、readiness=Service除外 |

---

## 次に読むべきガイド

- [Kubernetes応用](./02-kubernetes-advanced.md) -- Helm、Ingress、HPA、永続ボリューム
- [オーケストレーション概要](./00-orchestration-overview.md) -- Swarmとの比較と選定基準
- [コンテナセキュリティ](../06-security/00-container-security.md) -- K8sのセキュリティ設定

---

## 参考文献

1. Kubernetes公式ドキュメント "Concepts" -- https://kubernetes.io/docs/concepts/
2. Kubernetes公式チュートリアル -- https://kubernetes.io/docs/tutorials/
3. Nigel Poulton (2023) *The Kubernetes Book*, Independently Published
4. Brendan Burns, Joe Beda, Kelsey Hightower (2022) *Kubernetes: Up and Running*, 3rd Edition, O'Reilly
5. minikube公式ドキュメント -- https://minikube.sigs.k8s.io/docs/
6. Helm公式ドキュメント -- https://helm.sh/docs/
7. Kubernetes Best Practices -- https://kubernetes.io/docs/concepts/configuration/overview/
