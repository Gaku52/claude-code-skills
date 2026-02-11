# コンテナ技術

> コンテナは「アプリケーションとその依存関係をパッケージ化し、どこでも同じように動作させる」技術。

## この章で学ぶこと

- [ ] コンテナの技術的仕組みを理解する
- [ ] Docker / OCI の基本を知る
- [ ] コンテナオーケストレーションの概要を知る

---

## 1. コンテナの仕組み

```
コンテナ = Namespaces + cgroups + Union FS + seccomp

  Union FS（OverlayFS）:
  複数のレイヤーを重ねて1つのFSに見せる

  ┌──────────────────────┐
  │ Container Layer (RW)  │  ← 書き込み可能（薄い）
  ├──────────────────────┤
  │ App Layer (RO)        │  ← npm install の結果
  ├──────────────────────┤
  │ Runtime Layer (RO)    │  ← Node.js
  ├──────────────────────┤
  │ Base Layer (RO)       │  ← Ubuntu 22.04
  └──────────────────────┘

  → 読み取り専用レイヤーは複数コンテナで共有
  → ディスク容量の大幅な節約
  → Copy-on-Write: 変更時のみコピー

OCI（Open Container Initiative）:
  コンテナの標準仕様
  - Runtime Spec: コンテナの実行方法
  - Image Spec: イメージのフォーマット
  → Docker以外のランタイム（containerd, CRI-O）でも互換

コンテナランタイムの階層:
  Docker CLI
     ↓
  Docker Engine (dockerd)
     ↓
  containerd（高レベルランタイム）
     ↓
  runc（低レベルランタイム / OCI Runtime）
     ↓
  Linux Kernel (namespaces + cgroups)
```

---

## 2. コンテナの実践

```
Dockerfile の基本:

  FROM node:20-slim          # ベースイメージ
  WORKDIR /app               # 作業ディレクトリ
  COPY package*.json ./      # 依存定義コピー
  RUN npm ci --production    # 依存インストール
  COPY . .                   # ソースコピー
  EXPOSE 3000                # ポート宣言
  CMD ["node", "server.js"]  # 実行コマンド

  ベストプラクティス:
  - マルチステージビルド（ビルド環境を最終イメージに含めない）
  - .dockerignore でnode_modules等を除外
  - rootユーザーで実行しない（USER命令）
  - レイヤーキャッシュを活用（変更頻度順に配置）

セキュリティ:
  - イメージスキャン（Trivy, Snyk）
  - distrolessイメージ（シェルなし）
  - read-onlyファイルシステム
  - seccompプロファイル
  - rootless コンテナ（Podman）
```

---

## 3. コンテナオーケストレーション

```
Kubernetes（K8s）:
  大量のコンテナを管理・運用するプラットフォーム

  基本概念:
  ┌──────────────────────────────────────┐
  │ Cluster                              │
  │ ┌──────────┐  ┌──────────────────┐  │
  │ │Control   │  │ Worker Node      │  │
  │ │ Plane    │  │ ┌──────────────┐ │  │
  │ │ API Server│  │ │Pod           │ │  │
  │ │ Scheduler│  │ │ ┌──────────┐ │ │  │
  │ │ etcd     │  │ │ │Container │ │ │  │
  │ └──────────┘  │ │ └──────────┘ │ │  │
  │               │ └──────────────┘ │  │
  │               └──────────────────┘  │
  └──────────────────────────────────────┘

  Pod: 1つ以上のコンテナのグループ（最小デプロイ単位）
  Service: Podへのネットワークアクセス
  Deployment: Podのレプリカ管理
  Ingress: 外部からのHTTPルーティング

軽量代替:
  Docker Compose: 単一ホストの複数コンテナ管理
  Docker Swarm: シンプルなオーケストレーション
  K3s: 軽量Kubernetes
  Nomad (HashiCorp): 汎用ワークロードスケジューラ
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| コンテナ | Namespace+cgroup+UnionFS。軽量隔離 |
| OCI | コンテナの標準仕様。互換性保証 |
| Docker | イメージビルド+実行の事実上の標準 |
| Kubernetes | コンテナオーケストレーション。大規模運用 |

---

## 次に読むべきガイド
→ [[../07-modern-os/00-mobile-os.md]] — モバイルOS

---

## 参考文献
1. Lukša, M. "Kubernetes in Action." 2nd Ed, Manning, 2022.
2. Kane, S. et al. "Docker: Up & Running." 3rd Ed, O'Reilly, 2023.
