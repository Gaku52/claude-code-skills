# オーケストレーション概要

> 複数ホストにまたがるコンテナ群を自動管理するオーケストレーションの基本概念と、Docker Swarm / Kubernetes の選定基準を理解する。

---

## この章で学ぶこと

1. **コンテナオーケストレーションが解決する課題**と基本概念を理解する
2. **Docker SwarmとKubernetesの違い**を比較し、適切な選定基準を持つ
3. **オーケストレーションの主要コンポーネント**（スケジューリング、自己修復、スケーリング）を把握する
4. **Docker Swarm の実践的なクラスタ構築**とサービス管理を習得する
5. **マネージドKubernetesサービス**の特徴と選定指針を理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

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

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## よくある誤解と注意点

### 誤解1: 「完璧な設計を最初から作るべき」

**現実:** 完璧な設計は存在しません。要件の変化に応じて設計も進化させるべきです。最初から完璧を目指すと、過度に複雑な設計になりがちです。

> "Make it work, make it right, make it fast" — Kent Beck

### 誤解2: 「最新の技術を使えば自動的に良くなる」

**現実:** 技術選択はプロジェクトの要件に基づいて行うべきです。最新の技術が必ずしもプロジェクトに最適とは限りません。チームの習熟度、エコシステムの成熟度、サポートの持続性も考慮しましょう。

### 誤解3: 「テストは開発速度を落とす」

**現実:** 短期的にはテストの作成に時間がかかりますが、中長期的にはバグの早期発見、リファクタリングの安全性確保、ドキュメントとしての役割により、開発速度の向上に貢献します。

```python
# テストの ROI（投資対効果）を示す例
class TestROICalculator:
    """テスト投資対効果の計算"""

    def __init__(self):
        self.test_writing_hours = 0
        self.bugs_prevented = 0
        self.debug_hours_saved = 0

    def add_test_investment(self, hours: float):
        """テスト作成にかかった時間"""
        self.test_writing_hours += hours

    def add_bug_prevention(self, count: int, avg_debug_hours: float = 2.0):
        """テストにより防いだバグ"""
        self.bugs_prevented += count
        self.debug_hours_saved += count * avg_debug_hours

    def calculate_roi(self) -> dict:
        """ROIの計算"""
        net_benefit = self.debug_hours_saved - self.test_writing_hours
        roi_percent = (net_benefit / self.test_writing_hours * 100
                      if self.test_writing_hours > 0 else 0)
        return {
            'test_hours': self.test_writing_hours,
            'bugs_prevented': self.bugs_prevented,
            'hours_saved': self.debug_hours_saved,
            'net_benefit_hours': net_benefit,
            'roi_percent': f'{roi_percent:.1f}%'
        }
```

### 誤解4: 「ドキュメントは後から書けばいい」

**現実:** コードの意図や設計判断は、書いた直後が最も正確に記録できます。後回しにするほど、正確な情報を失います。

### 誤解5: 「パフォーマンスは常に最優先」

**現実:** 可読性と保守性を犠牲にした最適化は、長期的にはコストが高くつきます。「推測するな、計測せよ」の原則に従い、ボトルネックを特定してから最適化しましょう。
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
