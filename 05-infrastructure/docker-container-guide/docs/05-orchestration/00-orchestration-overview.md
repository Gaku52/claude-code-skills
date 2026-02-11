# オーケストレーション概要

> 複数ホストにまたがるコンテナ群を自動管理するオーケストレーションの基本概念と、Docker Swarm / Kubernetes の選定基準を理解する。

---

## この章で学ぶこと

1. **コンテナオーケストレーションが解決する課題**と基本概念を理解する
2. **Docker SwarmとKubernetesの違い**を比較し、適切な選定基準を持つ
3. **オーケストレーションの主要コンポーネント**（スケジューリング、自己修復、スケーリング）を把握する

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
      rollback_config:
        parallelism: 1
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      placement:
        constraints:
          - node.role == worker
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    ports:
      - "80:80"
    networks:
      - frontend

  api:
    image: my-api:latest
    deploy:
      replicas: 2
      placement:
        constraints:
          - node.labels.tier == app
    networks:
      - frontend
      - backend
    secrets:
      - db_password

  db:
    image: postgres:16-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.tier == data
    volumes:
      - pgdata:/var/lib/postgresql/data
    networks:
      - backend
    secrets:
      - db_password

networks:
  frontend:
    driver: overlay
  backend:
    driver: overlay
    internal: true

volumes:
  pgdata:
    driver: local

secrets:
  db_password:
    external: true
```

```bash
# Stack のデプロイ
docker stack deploy -c docker-stack.yml myapp

# Stack の状態確認
docker stack ls
docker stack services myapp
docker stack ps myapp

# Stack の削除
docker stack rm myapp
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
    │       ├── 運用チームが小さい ──► Docker Swarm
    │       │
    │       └── 将来の拡張を見据える ──► Kubernetes (マネージド)
    │
    └── 大規模（100+コンテナ、20+ノード）
            │
            └── Kubernetes (EKS/GKE/AKS) 一択
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
  my-service
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

---

## 5. マネージドKubernetesサービス

### クラウドサービス比較表

| サービス | プロバイダー | Control Plane費用 | 特徴 |
|---------|------------|-------------------|------|
| EKS | AWS | $73/月 | AWS統合、Fargate対応 |
| GKE | Google Cloud | 無料（Autopilot） | 最も成熟、Autopilotモード |
| AKS | Azure | 無料 | Azure統合、Windows対応 |
| DOKS | DigitalOcean | 無料 | シンプル、低コスト |

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
