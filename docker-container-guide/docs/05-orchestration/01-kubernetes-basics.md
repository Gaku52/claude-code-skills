# Kubernetes基礎

> Pod / Service / Deployment の3大リソースとkubectlの基本操作を通じて、Kubernetesの宣言的なコンテナ管理を習得する。

---

## この章で学ぶこと

1. **Pod / Service / Deployment の役割と関係性**を理解する
2. **kubectl を使った基本的なクラスタ操作**を習得する
3. **マニフェストファイル（YAML）の記述方法**とminikubeでの実践ができるようになる

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

---

## 6. minikubeでの実践

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

---

## 7. 完全なアプリケーション例

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

## 8. kubectl コマンドチートシート

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

---

## まとめ

| 項目 | ポイント |
|------|---------|
| Pod | 最小デプロイ単位。直接作成せずDeploymentで管理 |
| Deployment | レプリカ管理、ローリングアップデート、ロールバック |
| Service | 安定したネットワークエンドポイント。ClusterIP/NodePort/LoadBalancer |
| Namespace | リソースの論理的分離。環境・チーム別に使用 |
| kubectl | `apply` で宣言的管理。`describe` と `logs` でデバッグ |
| リソース制限 | requests/limits 必須。Scheduler配置とOOM防止 |
| Probe | liveness=再起動、readiness=Service除外 |

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
