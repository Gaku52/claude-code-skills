# オーケストレーション概要

> 複数ホストにまたがるコンテナ群を自動管理するオーケストレーションの基本概念と、Docker Swarm / Kubernetes の選定基準を理解する。

---

## この章で学ぶこと

1. **コンテナオーケストレーションが解決する課題**と基本概念を理解する
2. **Docker SwarmとKubernetesの違い**を比較し、適切な選定基準を持つ
3. **オーケストレーションの主要コンポーネント**（スケジューリング、自己修復、スケーリング）を把握する
4. **Docker Swarm の実践的なクラスタ構築**とサービス管理を習得する
5. **マネージドKubernetesサービス**の特徴と選定指針を理解する

---

## 1. なぜオーケストレーションが必要か

単一ホスト上の `docker compose` は開発・小規模運用には十分だが、本番環境では以下の課題に直面する。

### オーケストレーションが解決する課題

```
┌─── 単一ホスト（docker compose）───┐     ┌─── オーケストレーション ────────┐
│                                    │     │                                │
│  ❌ ホスト障害 = 全サービス停止     │     │  ✅ 障害時に別ホストで自動復旧  │
│  ❌ スケールアウト不可              │     │  ✅ 複数ホストに自動分散         │
│  ❌ ゼロダウンタイムデプロイ困難    │     │  ✅ ローリングアップデート        │
│  ❌ 負荷分散は手動設定              │     │  ✅ 組み込みロードバランサー      │
│  ❌ シークレット管理が限定的        │     │  ✅ 暗号化シークレットストア      │
│                                    │     │  ✅ 宣言的な状態管理             │
└────────────────────────────────────┘     └────────────────────────────────┘
```

### オーケストレーションの主要機能

```
┌──────────────────────────────────────────────────────────┐
│               オーケストレーター                          │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ スケジューリング│  │  自己修復    │  │ スケーリング  │  │
│  │              │  │              │  │              │  │
│  │ どのホストに  │  │ 障害コンテナ │  │ 負荷に応じて  │  │
│  │ 配置するか   │  │ を自動再起動 │  │ 自動増減     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ サービス      │  │ ローリング   │  │ シークレット  │  │
│  │ ディスカバリ  │  │ アップデート │  │ 管理         │  │
│  │              │  │              │  │              │  │
│  │ DNS/LBで     │  │ ダウンタイム │  │ 暗号化して   │  │
│  │ 自動接続     │  │ なしで更新   │  │ 安全に配布   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 単一ホストから複数ホストへの移行判断基準

| 判断ポイント | Docker Compose で十分 | オーケストレーション必要 |
|-------------|---------------------|----------------------|
| ホスト障害の許容 | 数分のダウンタイム許容 | ゼロダウンタイム必須 |
| トラフィック量 | 単一サーバーで処理可能 | 水平スケーリングが必要 |
| デプロイ頻度 | 週1回程度 | 日に複数回 |
| チーム規模 | 1-3人 | 5人以上 |
| コンテナ数 | ~10 | 20以上 |
| 予算 | 最小限 | 運用コスト許容 |

---

## 2. Docker Swarm

Docker Engine に組み込まれたオーケストレーション機能。追加のインストールが不要で、Docker CLIの延長で使用できる。

### コード例1: Docker Swarm クラスタの構築

```bash
# マネージャーノードでSwarmを初期化
docker swarm init --advertise-addr 192.168.1.10

# 出力されたトークンでワーカーノードを参加させる
# Worker Node 1:
docker swarm join --token SWMTKN-1-xxx 192.168.1.10:2377

# Worker Node 2:
docker swarm join --token SWMTKN-1-xxx 192.168.1.10:2377

# クラスタの状態確認
docker node ls
# ID          HOSTNAME   STATUS    AVAILABILITY   MANAGER STATUS
# abc123 *    manager1   Ready     Active         Leader
# def456      worker1    Ready     Active
# ghi789      worker2    Ready     Active
```

### Swarm クラスタのアーキテクチャ

```
┌──────────────────────────────────────────────────────────┐
│                  Docker Swarm Cluster                      │
│                                                            │
│  ┌──────────────────┐                                     │
│  │  Manager Node 1  │  ← Leader                          │
│  │  (192.168.1.10)  │                                     │
│  │  ┌────────────┐  │  ・Raft合意プロトコル               │
│  │  │ Raft Store │  │  ・クラスタ状態管理                 │
│  │  │ Scheduler  │  │  ・タスクスケジューリング           │
│  │  │ Dispatcher │  │  ・サービスAPI                      │
│  │  │ Allocator  │  │  ・ネットワーク管理                 │
│  │  └────────────┘  │                                     │
│  └──────────────────┘                                     │
│                                                            │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │  Manager Node 2  │  │  Manager Node 3  │  ← HA構成   │
│  │  (Reachable)     │  │  (Reachable)     │              │
│  │  バックアップ     │  │  バックアップ     │              │
│  └──────────────────┘  └──────────────────┘              │
│                                                            │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │  Worker Node 1   │  │  Worker Node 2   │              │
│  │  (192.168.1.11)  │  │  (192.168.1.12)  │              │
│  │  ┌────────────┐  │  │  ┌────────────┐  │              │
│  │  │ Container  │  │  │  │ Container  │  │              │
│  │  │ Container  │  │  │  │ Container  │  │              │
│  │  │ Container  │  │  │  │ Container  │  │              │
│  │  └────────────┘  │  │  └────────────┘  │              │
│  └──────────────────┘  └──────────────────┘              │
└──────────────────────────────────────────────────────────┘
```

### 高可用性（HA）構成の設計

```bash
# HA構成では奇数のマネージャーノードが推奨
# マネージャー数と耐障害性の関係:

# マネージャー: 1 → 障害耐性: 0 (非推奨)
# マネージャー: 3 → 障害耐性: 1 (最小HA構成)
# マネージャー: 5 → 障害耐性: 2 (推奨HA構成)
# マネージャー: 7 → 障害耐性: 3 (大規模)

# マネージャーの追加
docker swarm join-token manager
# 表示されたコマンドを新ノードで実行

# ノードの役割変更
docker node promote worker1     # ワーカー → マネージャー
docker node demote manager3     # マネージャー → ワーカー

# ノードのメンテナンスモード
docker node update --availability drain worker1
# → worker1 上のタスクが他のノードに移動

# メンテナンス完了後
docker node update --availability active worker1
```

### コード例2: Swarmサービスのデプロイ

```bash
# サービスの作成
docker service create \
  --name web \
  --replicas 3 \
  --publish published=80,target=80 \
  --update-delay 10s \
  --update-parallelism 1 \
  --rollback-parallelism 1 \
  --rollback-monitor 30s \
  --restart-condition on-failure \
  --limit-memory 256M \
  --limit-cpu 0.5 \
  nginx:alpine

# サービスの状態確認
docker service ls
docker service ps web

# スケーリング
docker service scale web=5

# ローリングアップデート
docker service update --image nginx:1.25-alpine web

# ロールバック
docker service rollback web

# サービスのログ確認
docker service logs web
docker service logs -f --tail 100 web

# サービスの詳細情報
docker service inspect --pretty web
```

### コード例3: Docker Stack（Compose形式でSwarmデプロイ）

```yaml
# docker-stack.yml
version: "3.9"

services:
  web:
    image: nginx:alpine
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
        monitor: 30s
        order: start-first  # 新しいタスクを起動してから古いタスクを停止
      rollback_config:
        parallelism: 1
        order: stop-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
      placement:
        constraints:
          - node.role == worker
        preferences:
          - spread: node.id  # ノード間に均等分散
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
        reservations:
          cpus: "0.1"
          memory: 64M
    ports:
      - "80:80"
    networks:
      - frontend
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:80/"]
      interval: 10s
      timeout: 3s
      retries: 3

  api:
    image: my-api:latest
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.labels.tier == app
      labels:
        com.example.service: "api"
        com.example.environment: "production"
    environment:
      DATABASE_URL: "postgres://postgres-service:5432/myapp"
    networks:
      - frontend
      - backend
    secrets:
      - db_password
      - api_key

  db:
    image: postgres:16-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.tier == data
        max_replicas_per_node: 1  # 1ノードに1レプリカのみ
    volumes:
      - pgdata:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      POSTGRES_DB: myapp
    networks:
      - backend
    secrets:
      - db_password

  redis:
    image: redis:7-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.tier == data
    networks:
      - backend

  # ビジュアライザー（管理用）
  visualizer:
    image: dockersamples/visualizer:latest
    deploy:
      placement:
        constraints:
          - node.role == manager
    ports:
      - "8080:8080"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro

networks:
  frontend:
    driver: overlay
  backend:
    driver: overlay
    internal: true  # 外部アクセス不可

volumes:
  pgdata:
    driver: local

secrets:
  db_password:
    external: true
  api_key:
    external: true
```

```bash
# シークレットの作成
echo "MySecretPassword123" | docker secret create db_password -
echo "sk-1234567890" | docker secret create api_key -

# ノードラベルの設定
docker node update --label-add tier=app worker1
docker node update --label-add tier=app worker2
docker node update --label-add tier=data worker3

# Stack のデプロイ
docker stack deploy -c docker-stack.yml myapp

# Stack の状態確認
docker stack ls
docker stack services myapp
docker stack ps myapp

# 特定サービスのスケーリング
docker service scale myapp_api=4

# Stack の更新（YAMLファイル変更後）
docker stack deploy -c docker-stack.yml myapp  # 再デプロイで差分適用

# Stack の削除
docker stack rm myapp
```

### Swarm のネットワーキング

```
┌──────────────────────────────────────────────────────────┐
│                  Overlay Network                          │
│                                                          │
│  Node 1 (Manager)        Node 2 (Worker)                │
│  ┌──────────────┐       ┌──────────────┐               │
│  │  web.1       │       │  web.2       │               │
│  │  10.0.0.5    │       │  10.0.0.6    │               │
│  │              │       │              │               │
│  │  api.1       │       │  api.2       │               │
│  │  10.0.0.7    │       │  10.0.0.8    │               │
│  └──────┬───────┘       └──────┬───────┘               │
│         │     VXLAN Tunnel     │                        │
│         └──────────────────────┘                        │
│                                                          │
│  Routing Mesh（ルーティングメッシュ）:                     │
│  ┌─────────────────────────────────────────────┐        │
│  │  クライアント → 任意のノード:80               │        │
│  │       ↓                                      │        │
│  │  ingress network が適切なコンテナに転送       │        │
│  │  (web.1 or web.2 or web.3)                   │        │
│  └─────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

```bash
# Overlay ネットワークの作成
docker network create --driver overlay --attachable my-overlay

# 暗号化されたOverlayネットワーク
docker network create --driver overlay --opt encrypted my-secure-overlay

# ネットワーク一覧
docker network ls --filter driver=overlay
```

---

## 3. Docker Swarm vs Kubernetes

### アーキテクチャ比較

```
Docker Swarm                          Kubernetes
┌──────────────────┐                  ┌──────────────────────────┐
│  Manager Node    │                  │  Control Plane            │
│  ┌────────────┐  │                  │  ┌──────────────────┐    │
│  │ Raft合意   │  │                  │  │ API Server        │    │
│  │ スケジューラ│  │                  │  │ etcd              │    │
│  │ ルーティング│  │                  │  │ Scheduler         │    │
│  └────────────┘  │                  │  │ Controller Manager│    │
├──────────────────┤                  │  └──────────────────┘    │
│  Worker Node 1   │                  ├──────────────────────────┤
│  ┌────────────┐  │                  │  Worker Node 1           │
│  │ Container  │  │                  │  ┌──────────────────┐    │
│  │ Container  │  │                  │  │ kubelet           │    │
│  └────────────┘  │                  │  │ kube-proxy        │    │
├──────────────────┤                  │  │ Container Runtime │    │
│  Worker Node 2   │                  │  │ Pod  Pod  Pod     │    │
│  ┌────────────┐  │                  │  └──────────────────┘    │
│  │ Container  │  │                  ├──────────────────────────┤
│  │ Container  │  │                  │  Worker Node 2           │
│  └────────────┘  │                  │  ┌──────────────────┐    │
└──────────────────┘                  │  │ Pod  Pod  Pod     │    │
                                      │  └──────────────────┘    │
 シンプル・軽量                        └──────────────────────────┘
                                       高機能・エコシステム充実
```

### 詳細比較表

| 観点 | Docker Swarm | Kubernetes |
|------|-------------|------------|
| セットアップ | `docker swarm init` のみ | 複数コンポーネントの構築が必要 |
| 学習コスト | 低い（Docker CLI の延長） | 高い（独自の概念が多い） |
| スケーラビリティ | 数百ノード | 数千ノード |
| エコシステム | 限定的 | Helm, Istio, Argo等 豊富 |
| 自動スケーリング | 手動（docker service scale） | HPA/VPA で自動 |
| ストレージ | Docker Volume | PV/PVC, StorageClass |
| ネットワーク | Overlay（組み込み） | CNIプラグイン（選択肢豊富） |
| Ingress | ルーティングメッシュ | Ingress Controller |
| 運用負荷 | 低い | 高い（マネージドサービス推奨） |
| クラウドサポート | 限定的 | EKS/GKE/AKS等 充実 |
| コミュニティ | 縮小傾向 | 活発（業界標準） |
| リソース消費 | 低い（Dockerに統合） | 高い（Control Plane が重い） |
| バッチジョブ | 限定的 | Jobs, CronJobs で充実 |
| 設定管理 | Docker Configs | ConfigMap, Secret |
| RBAC | なし | 詳細なRBAC |
| カスタムリソース | なし | CRD で拡張可能 |
| サービスメッシュ | なし | Istio, Linkerd 等 |

### 用語の対応表

| 概念 | Docker Swarm | Kubernetes |
|------|-------------|------------|
| クラスタ管理ノード | Manager | Control Plane (Master) |
| 実行ノード | Worker | Worker Node |
| デプロイ単位 | Task | Pod |
| サービス定義 | Service | Deployment + Service |
| 設定ファイル | docker-stack.yml | Manifest (YAML) |
| スケーリング | docker service scale | kubectl scale / HPA |
| ネットワーク | Overlay | CNI Plugin |
| ストレージ | Volume | PersistentVolume |
| 機密情報 | Secret | Secret |
| 設定情報 | Config | ConfigMap |
| ロードバランサー | Routing Mesh | Service (LoadBalancer) |
| 名前空間 | なし | Namespace |

### 選定基準の判断フロー

```
プロジェクトの規模は？
    │
    ├── 小規模（〜10コンテナ、〜3ノード）
    │       │
    │       ├── Docker Compose で十分か？ ── Yes ──► Docker Compose
    │       │
    │       └── 高可用性が必要 ──► Docker Swarm
    │
    ├── 中規模（10〜100コンテナ、3〜20ノード）
    │       │
    │       ├── 運用チームが小さい（1-3人）──► Docker Swarm
    │       │
    │       ├── クラウドネイティブ ──► マネージドK8s（EKS/GKE/AKS）
    │       │
    │       └── 将来の拡張を見据える ──► Kubernetes (マネージド)
    │
    └── 大規模（100+コンテナ、20+ノード）
            │
            ├── マルチクラウド/ハイブリッド ──► Kubernetes
            │
            └── 単一クラウド ──► マネージドK8s 一択
```

### コスト比較（参考値）

```
┌─────────────────────────────────────────────────────────┐
│  運用コスト比較（中規模: 10サービス、3ノード）            │
│                                                          │
│  Docker Compose（単一ホスト）:                            │
│    サーバー: $50/月 × 1台 = $50/月                       │
│    運用工数: 低                                           │
│    合計: 〜$50/月                                        │
│                                                          │
│  Docker Swarm:                                           │
│    サーバー: $50/月 × 3台 = $150/月                      │
│    運用工数: 低〜中                                       │
│    合計: 〜$150/月                                       │
│                                                          │
│  Kubernetes（自己管理）:                                  │
│    サーバー: $50/月 × 3台 + Control Plane = $200/月      │
│    運用工数: 高（K8s専任エンジニア必要）                   │
│    合計: 〜$200/月 + 人件費                              │
│                                                          │
│  マネージドK8s（EKS/GKE/AKS）:                           │
│    Control Plane: $73/月（EKS）or $0（GKE Autopilot）   │
│    ワーカーノード: $150/月                                │
│    運用工数: 中                                           │
│    合計: 〜$150-223/月                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. オーケストレーションの主要概念

### コード例4: 宣言的な状態管理

```yaml
# 「あるべき状態」を宣言すると、オーケストレーターが自動的に現実を合わせる

# Docker Swarm: docker-stack.yml
services:
  web:
    image: nginx:alpine
    deploy:
      replicas: 3    # ← 常に3つのレプリカを維持

# Kubernetes: deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3        # ← 常に3つのPodを維持
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
        - name: web
          image: nginx:alpine
```

### 命令的 vs 宣言的 管理の違い

```
命令的（Imperative）:
  「nginx コンテナを3つ起動しろ」
  「2番目のコンテナを停止しろ」
  「新しいコンテナを1つ追加しろ」
  → 手順を逐一指示する必要がある
  → 障害時の自動復旧なし

宣言的（Declarative）:
  「nginx コンテナは常に3つ存在すべき」
  → オーケストレーターが現在の状態と望ましい状態を比較
  → 差分を自動的に解消
  → 障害時も自動復旧

┌────────────────────────────────────────────┐
│  制御ループ（Reconciliation Loop）          │
│                                            │
│  ┌─────────┐    比較    ┌─────────┐      │
│  │ Desired │  ◄────►  │ Current │      │
│  │ State   │           │ State   │      │
│  │ (YAML)  │           │ (実態)  │      │
│  └─────────┘           └─────────┘      │
│       │                     ▲            │
│       │     差分を検出      │            │
│       │     ┌────────┐     │            │
│       └────►│ Action │─────┘            │
│             │ 実行   │                  │
│             └────────┘                  │
│                                          │
│  例: replicas=3 だが Pod が2つしかない    │
│  → 自動的に1つ追加して3つにする          │
└────────────────────────────────────────────┘
```

### 自己修復メカニズム

```
Desired State: replicas=3

現在の状態:
┌────────┐ ┌────────┐ ┌────────┐
│ Pod 1  │ │ Pod 2  │ │ Pod 3  │
│  ✅    │ │  ✅    │ │  ✅    │
└────────┘ └────────┘ └────────┘

Pod 2 が障害で停止:
┌────────┐ ┌────────┐ ┌────────┐
│ Pod 1  │ │ Pod 2  │ │ Pod 3  │
│  ✅    │ │  ❌    │ │  ✅    │
└────────┘ └────────┘ └────────┘

オーケストレーターが検知 → 自動で新しいPodを起動:
┌────────┐ ┌────────┐ ┌────────┐
│ Pod 1  │ │ Pod 3  │ │ Pod 4  │
│  ✅    │ │  ✅    │ │  ✅ ★  │  ← 新規作成
└────────┘ └────────┘ └────────┘

ノード障害の場合:
┌────────────┐ ┌────────────┐
│  Node 1    │ │  Node 2    │
│  Pod 1 ✅  │ │  Pod 2 ✅  │
│  Pod 3 ✅  │ │            │
└────────────┘ └────────────┘
    ↓ Node 1 障害
┌────────────┐ ┌────────────┐
│  Node 1    │ │  Node 2    │
│  ❌ 全停止 │ │  Pod 2 ✅  │
│            │ │  Pod 4 ✅  │ ← Pod 1 の代替
│            │ │  Pod 5 ✅  │ ← Pod 3 の代替
└────────────┘ └────────────┘
```

### コード例5: ローリングアップデート

```bash
# Docker Swarm のローリングアップデート
docker service update \
  --image my-app:v2.0.0 \
  --update-parallelism 1 \
  --update-delay 10s \
  --update-failure-action rollback \
  --update-monitor 30s \
  --update-max-failure-ratio 0.25 \
  --update-order start-first \
  my-service
```

```yaml
# Kubernetes のローリングアップデート
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # 同時に追加できるPod数
      maxUnavailable: 0  # 停止を許容するPod数
  template:
    spec:
      containers:
        - name: app
          image: my-app:v2.0.0
```

```
ローリングアップデートの流れ (parallelism=1):

Step 1: Pod 1 を v2 に更新
┌─────────┐ ┌─────────┐ ┌─────────┐
│ v1 → v2 │ │   v1    │ │   v1    │
│ 更新中   │ │  稼働中  │ │  稼働中  │
└─────────┘ └─────────┘ └─────────┘

Step 2: ヘルスチェック OK → Pod 2 を更新
┌─────────┐ ┌─────────┐ ┌─────────┐
│   v2    │ │ v1 → v2 │ │   v1    │
│  稼働中  │ │ 更新中   │ │  稼働中  │
└─────────┘ └─────────┘ └─────────┘

Step 3: Pod 3 を更新
┌─────────┐ ┌─────────┐ ┌─────────┐
│   v2    │ │   v2    │ │ v1 → v2 │
│  稼働中  │ │  稼働中  │ │ 更新中   │
└─────────┘ └─────────┘ └─────────┘

完了: 全Pod が v2 に更新（ダウンタイムなし）
```

### デプロイ戦略の比較

| 戦略 | ダウンタイム | リスク | ロールバック | リソース消費 |
|------|------------|--------|------------|------------|
| ローリングアップデート | なし | 低 | 自動 | 少し多い |
| Blue/Green | なし | 低 | 即座 | 2倍必要 |
| Canary | なし | 最低 | 容易 | 少し多い |
| Recreate | あり | 高 | 遅い | 変化なし |

```
Blue/Green デプロイ:
┌──────────┐     ┌──────────┐
│  Blue    │     │  Green   │
│  v1 ✅   │     │  v2 ✅   │
│  (現行)  │     │  (新版)  │
└──────────┘     └──────────┘
      │                │
      └─── LB ────────┘
           ↓ 切替
      ┌──────────┐     ┌──────────┐
      │  Blue    │     │  Green   │
      │  v1      │     │  v2 ✅   │
      │  (待機)  │     │  (現行)  │
      └──────────┘     └──────────┘

Canary デプロイ:
┌──────────────────────────┐
│  v1  v1  v1  v1  v2     │
│  90%              10%    │  ← 10%のトラフィックで検証
│                          │
│  問題なし → 段階的に拡大  │
│  v1  v1  v2  v2  v2     │
│  40%         60%         │
│                          │
│  v2  v2  v2  v2  v2     │
│  100%                    │  ← 完全移行
└──────────────────────────┘
```

---

## 5. マネージドKubernetesサービス

### クラウドサービス比較表

| サービス | プロバイダー | Control Plane費用 | 特徴 |
|---------|------------|-------------------|------|
| EKS | AWS | $73/月 | AWS統合、Fargate対応 |
| GKE | Google Cloud | 無料（Autopilot） | 最も成熟、Autopilotモード |
| AKS | Azure | 無料 | Azure統合、Windows対応 |
| DOKS | DigitalOcean | 無料 | シンプル、低コスト |
| LKE | Linode/Akamai | 無料 | シンプル、低コスト |

### マネージドK8sの選定基準

```
┌─────────────────────────────────────────────────────────┐
│  マネージドK8s選定の判断基準                              │
│                                                          │
│  既存のクラウド契約は？                                   │
│    ├── AWS → EKS                                        │
│    ├── GCP → GKE (Autopilot推奨)                        │
│    ├── Azure → AKS                                      │
│    └── なし → コスト重視ならDOKS、機能重視ならGKE        │
│                                                          │
│  自動スケーリングの要件は？                               │
│    ├── ノードもPodも自動 → GKE Autopilot                │
│    ├── Podのみ自動 → どのサービスでもHPA対応             │
│    └── 固定 → コスト最小のDOKSやLKE                     │
│                                                          │
│  サーバーレスコンテナは？                                 │
│    ├── AWS Fargate（EKS連携）                           │
│    ├── GKE Autopilot                                    │
│    └── Azure Container Apps                             │
└─────────────────────────────────────────────────────────┘
```

### ローカルKubernetes環境の構築

```bash
# === minikube（推奨: 初学者向け） ===
# インストール
brew install minikube

# クラスタ起動
minikube start --driver=docker --memory=4096 --cpus=2

# ダッシュボード
minikube dashboard

# === kind（Kubernetes in Docker） ===
# インストール
brew install kind

# クラスタ作成（マルチノード）
cat <<EOF | kind create cluster --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
  - role: worker
  - role: worker
EOF

# === k3d（k3s on Docker、最軽量） ===
# インストール
brew install k3d

# クラスタ作成
k3d cluster create mycluster --agents 2

# === Docker Desktop ===
# Settings > Kubernetes > Enable Kubernetes
```

| ツール | 特徴 | リソース消費 | マルチノード | 起動速度 |
|--------|------|-------------|------------|---------|
| minikube | 公式、多機能 | 中 | アドオン | 中 |
| kind | Docker内にK8sクラスタ | 低 | 対応 | 速い |
| k3d | k3s (軽量K8s) on Docker | 最低 | 対応 | 最速 |
| Docker Desktop | 組み込みK8s | 中 | 不可 | 遅い |

---

## 6. 実践: Docker Compose から Swarm への移行

### 移行手順

```bash
# Step 1: 既存の docker-compose.yml を確認
docker compose config

# Step 2: Swarm 非対応の設定を修正
# - build: は使えない → 事前にイメージをビルド＆プッシュ
# - depends_on: の condition は使えない → healthcheck で代替
# - volumes のバインドマウント → Named Volume に変更

# Step 3: deploy セクションを追加
# - replicas, resources, placement 等

# Step 4: Swarm 初期化
docker swarm init

# Step 5: シークレットの作成
cat .env | while IFS='=' read -r key value; do
  echo "$value" | docker secret create "$key" -
done

# Step 6: Stack デプロイ
docker stack deploy -c docker-compose.yml myapp

# Step 7: 動作確認
docker stack services myapp
docker stack ps myapp
```

### docker-compose.yml → docker-stack.yml の変換例

```yaml
# Before: docker-compose.yml
version: "3.9"
services:
  api:
    build: ./api           # ← Swarmでは不可
    ports:
      - "8080:8080"
    depends_on:
      db:
        condition: service_healthy  # ← Swarmでは不可
    environment:
      DB_PASSWORD: secret123       # ← シークレットに移行

# After: docker-stack.yml
version: "3.9"
services:
  api:
    image: registry.example.com/api:1.0.0  # ← ビルド済みイメージ
    ports:
      - "8080:8080"
    deploy:
      replicas: 2
      update_config:
        parallelism: 1
        delay: 10s
      resources:
        limits:
          memory: 512M
          cpus: "1.0"
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:8080/health"]
      interval: 10s
      timeout: 3s
      retries: 5
    secrets:
      - db_password
    environment:
      DB_PASSWORD_FILE: /run/secrets/db_password  # ← ファイルから読み取り

secrets:
  db_password:
    external: true
```

---

## アンチパターン

### アンチパターン1: 過剰なオーケストレーション導入

```
# NG: 3コンテナのアプリにKubernetesを導入
# → 運用コストがアプリケーション開発コストを上回る

# OK: 規模に合ったツール選定
# 3コンテナ → Docker Compose
# 10コンテナ、高可用性 → Docker Swarm
# 50コンテナ以上、マルチチーム → Kubernetes
```

**なぜ問題か**: Kubernetesの学習コスト・運用コストは非常に高い。小規模プロジェクトでは、オーケストレーターの管理自体がボトルネックになる。

### アンチパターン2: ステートフルワークロードの安易なコンテナ化

```yaml
# NG: データベースをオーケストレーション配下で管理（知識不足の状態で）
services:
  postgres:
    deploy:
      replicas: 3  # データベースは単純にレプリカを増やせない

# OK: まずはマネージドDBサービスを使用
# AWS RDS, Cloud SQL, Azure Database 等
# → コンテナDB運用は十分な知識を得てから
```

**なぜ問題か**: データベースはステートフルであり、レプリケーション、フェイルオーバー、バックアップの設計が必要。マネージドサービスに任せた方が安全。

### アンチパターン3: 単一マネージャーノードでの本番運用

```bash
# NG: マネージャー1台で本番運用
# → マネージャー障害でクラスタ全体が制御不能

# OK: 最低3台のマネージャーで HA 構成
docker swarm init --advertise-addr 192.168.1.10
# + 2台のマネージャーを追加

# Kubernetes の場合:
# → マネージドサービス（EKS/GKE/AKS）で Control Plane の HA は自動
```

**なぜ問題か**: 単一マネージャーが障害を起こすと、新しいタスクのスケジューリングやサービスの更新ができなくなる。既存のコンテナは動作し続けるが、自己修復機能が停止する。

---

## FAQ

### Q1: Docker Swarmは今後も使い続けて大丈夫か？

Docker Swarmは Docker Engine に統合されており、メンテナンスは継続されている。ただしKubernetesが業界標準となり、新機能の追加やエコシステムの拡大はKubernetesに集中している。小規模プロジェクトでの使用は問題ないが、長期的にはKubernetesへの移行を視野に入れるのが現実的。

### Q2: docker composeとオーケストレーションの境界はどこか？

判断基準:
- **単一ホスト + 再起動で許容できるダウンタイム** → Docker Compose
- **複数ホスト or ゼロダウンタイム必須** → オーケストレーション

Docker Compose v2 の `--profile` や `restart: unless-stopped` で単一ホストの運用をかなりカバーできるが、ホスト障害時の自動フェイルオーバーはオーケストレーションでしか実現できない。

### Q3: Kubernetes をローカル環境で試すには？

| ツール | 特徴 | リソース消費 |
|--------|------|-------------|
| minikube | 公式、多機能 | 中 |
| kind | Docker内にK8sクラスタ | 低 |
| k3d | k3s (軽量K8s) on Docker | 最低 |
| Docker Desktop | 組み込みK8s | 中 |

```bash
# minikube で開始
minikube start --driver=docker --memory=4096 --cpus=2
kubectl get nodes
```

### Q4: Swarm から Kubernetes への移行はどの程度大変か？

概念は似ているが、設定ファイルの形式が完全に異なる。主な移行作業:

1. docker-stack.yml → Kubernetes manifests (Deployment, Service, ConfigMap, Secret) への変換
2. Overlay Network → Kubernetes Network Policy への移行
3. Docker Secrets → Kubernetes Secrets への移行
4. ルーティングメッシュ → Ingress Controller の設定
5. 監視・ログ基盤のK8s対応

移行ツール（Kompose）で基本的な変換は自動化できるが、本番品質にするには手動調整が必要。

```bash
# Kompose でdocker-compose.yml をK8sマニフェストに変換
kompose convert -f docker-compose.yml
```

### Q5: オーケストレーションなしでゼロダウンタイムデプロイは可能か？

Docker Compose でも `docker compose up -d --no-deps --scale api=2` と healthcheck を組み合わせることで、疑似的なゼロダウンタイムデプロイが可能。ただし:

- リバースプロキシ（nginx/Traefik）が必要
- スクリプトによる手動制御が必要
- ホスト障害時の自動復旧は不可

完全なゼロダウンタイムが必須なら、オーケストレーションの導入を推奨する。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| オーケストレーション | 複数ホストでのコンテナ管理を自動化 |
| Docker Swarm | シンプル、低学習コスト、小〜中規模向け |
| Kubernetes | 業界標準、高機能、大規模向け。マネージドサービス推奨 |
| 宣言的管理 | 「あるべき状態」を宣言し、オーケストレーターが維持 |
| 自己修復 | 障害を検知して自動的に復旧 |
| ローリングアップデート | ダウンタイムなしでバージョン更新 |
| 選定基準 | 規模・チーム体制・将来の拡張性で判断 |
| HA構成 | Swarmは最低3マネージャー、K8sはマネージド推奨 |
| ネットワーク | Swarm: Overlay、K8s: CNI Plugin |

---

## 次に読むべきガイド

- [Kubernetes基礎](./01-kubernetes-basics.md) -- Pod/Service/Deploymentの基本操作
- [Kubernetes応用](./02-kubernetes-advanced.md) -- Helm、Ingress、HPA等の実践
- [Docker CI/CD](../04-production/02-ci-cd-docker.md) -- オーケストレーションへのデプロイパイプライン

---

## 参考文献

1. Docker公式ドキュメント "Swarm mode overview" -- https://docs.docker.com/engine/swarm/
2. Kubernetes公式ドキュメント -- https://kubernetes.io/docs/
3. Nigel Poulton (2023) *The Kubernetes Book*, Chapter 1: Kubernetes Primer
4. CNCF "Cloud Native Landscape" -- https://landscape.cncf.io/
5. Kelsey Hightower, Brendan Burns, Joe Beda (2022) *Kubernetes: Up and Running*, 3rd Edition, O'Reilly
6. Docker公式ドキュメント "Deploy to Swarm" -- https://docs.docker.com/engine/swarm/stack-deploy/
7. Kompose (Kubernetes + Compose) -- https://kompose.io/
