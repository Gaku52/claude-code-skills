# GitOps

> Gitリポジトリを唯一の信頼源(Single Source of Truth)とし、インフラとアプリケーションの宣言的な状態をプルベースで自動同期する運用モデル

## この章で学ぶこと

1. GitOpsの4原則とプッシュ型/プル型デプロイの違いを理解する
2. ArgoCD、Fluxの仕組みと基本的な設定方法を習得する
3. イミュータブルインフラストラクチャとGitOpsの関係を把握する
4. GitOpsにおけるシークレット管理とマルチクラスタ運用を実践できる
5. GitOpsのトラブルシューティングと運用ベストプラクティスを理解する

---

## 1. GitOps とは

### 1.1 GitOps の4原則

```
GitOps 4原則 (OpenGitOps):

1. 宣言的 (Declarative)
   システムのあるべき状態をコードで宣言する
   例: Kubernetes マニフェスト、Helm チャート、Kustomize

2. バージョン管理・イミュータブル (Versioned and Immutable)
   あるべき状態は Git で管理し、変更履歴を保持する
   例: 全ての変更が Git コミットとして記録される

3. 自動プル (Pulled Automatically)
   承認された変更はシステムに自動的に適用される
   例: ArgoCD/Flux が Git の変更を検知して自動デプロイ

4. 継続的リコンサイル (Continuously Reconciled)
   実際の状態とあるべき状態の差分を検知し、自動修復する
   例: 手動変更(kubectl)が自動で元に戻される
```

### 1.2 GitOps の歴史と背景

GitOpsは2017年にWeaveWorksのAlexis Richardsonが提唱した概念である。Kubernetesの宣言的APIとGitの不変な変更履歴を組み合わせることで、信頼性の高いデプロイフローを実現する。

```
GitOps の進化:

2014年  Kubernetes の登場 → 宣言的インフラ管理の基盤
2015年  Helm の登場 → パッケージ管理の標準化
2017年  Weaveworks が "GitOps" を提唱
2018年  Flux v1 リリース (初のGitOpsツール)
2019年  Argo CD v1.0 リリース
2020年  CNCF に Flux/ArgoCD が Sandbox プロジェクトとして参加
2021年  Flux v2 リリース (完全リアーキテクチャ)
2022年  OpenGitOps プロジェクト発足 (原則の標準化)
2023年  Argo CD が CNCF Graduated プロジェクトに
2024年  Flux が CNCF Graduated プロジェクトに
```

### 1.3 プッシュ型 vs プル型デプロイ

```
プッシュ型 (従来の CI/CD):
  ┌─────┐     ┌──────┐     ┌───────────┐
  │ Git │ ──→ │ CI/CD │ ──→ │ Kubernetes │
  │     │     │Server│     │ Cluster    │
  └─────┘     └──────┘     └───────────┘
                  │
           CI がクラスタに
           直接デプロイ
           (push)

  問題:
  - CI に本番環境への認証情報が必要
  - CI サーバーが SPOF (単一障害点)
  - ドリフト(実態の乖離)を検知できない
  - CI の設定が複雑になりやすい
  - 監査証跡がCIログに依存

プル型 (GitOps):
  ┌─────┐                  ┌───────────┐
  │ Git │ ←── 監視 ←────── │ Agent     │
  │     │                  │ (ArgoCD)  │
  └─────┘                  │ in Cluster│
                           └───────────┘
                                │
                          差分を検知して
                          自動適用 (pull)

  利点:
  - クラスタ外に認証情報を露出しない
  - エージェントが常に差分を監視・修復
  - Git の履歴 = デプロイの履歴
  - CI とデプロイの責務が明確に分離
  - 監査証跡が Git コミットに残る
```

### 1.4 GitOps vs 従来の CI/CD

| 項目 | 従来の CI/CD | GitOps |
|---|---|---|
| デプロイ方式 | プッシュ型 | プル型 |
| 信頼源 | CI サーバー | Git リポジトリ |
| ドリフト検知 | なし | 継続的リコンサイル |
| ロールバック | 再デプロイ | Git revert |
| 認証情報 | CI に保存 | クラスタ内に閉じる |
| 監査証跡 | CI ログ | Git コミット履歴 |
| 再現性 | CI 設定に依存 | Git の状態から完全再現 |
| 障害復旧 | 手順書に依存 | Git から自動復元 |

---

## 2. ArgoCD

### 2.1 ArgoCD の仕組み

```
┌──────────────────────────────────────────────┐
│              Kubernetes Cluster               │
│                                               │
│  ┌─────────────┐     ┌──────────────────┐    │
│  │  ArgoCD     │     │  Application     │    │
│  │  Controller │ ──→ │  Resources       │    │
│  │             │     │  (Deployment,    │    │
│  │  ・Git監視   │     │   Service, etc.) │    │
│  │  ・差分検知   │     └──────────────────┘    │
│  │  ・自動同期   │                             │
│  └──────┬──────┘                              │
│         │ pull (3分ごと)                       │
└─────────┼────────────────────────────────────┘
          │
          ↓
  ┌───────────────┐
  │ Git Repository │
  │ (manifests)    │
  │                │
  │ ├── base/      │
  │ ├── overlays/  │
  │ └── apps/      │
  └───────────────┘
```

### 2.2 ArgoCD のインストールとセットアップ

```bash
# ArgoCD のインストール (Kubernetes クラスタが必要)
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# ArgoCD CLI のインストール
brew install argocd  # macOS
# または
curl -sSL -o argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
chmod +x argocd && sudo mv argocd /usr/local/bin/

# 初期パスワードの取得
argocd admin initial-password -n argocd

# ログイン
argocd login localhost:8080

# パスワード変更
argocd account update-password
```

```yaml
# ArgoCD を Helm でインストール (推奨: 本番環境向け)
# values.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: argocd

---
# Helm でのインストール
# helm repo add argo https://argoproj.github.io/argo-helm
# helm install argocd argo/argo-cd -n argocd -f values.yaml

# values.yaml
server:
  replicas: 2
  ingress:
    enabled: true
    ingressClassName: nginx
    hosts:
      - argocd.example.com
    tls:
      - secretName: argocd-tls
        hosts:
          - argocd.example.com

controller:
  replicas: 1
  resources:
    requests:
      cpu: 500m
      memory: 512Mi
    limits:
      cpu: 1000m
      memory: 1Gi

repoServer:
  replicas: 2
  resources:
    requests:
      cpu: 250m
      memory: 256Mi

redis:
  resources:
    requests:
      cpu: 100m
      memory: 128Mi

configs:
  params:
    server.insecure: false
  rbac:
    policy.csv: |
      g, dev-team, role:readonly
      g, ops-team, role:admin
```

### 2.3 ArgoCD Application 定義

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io  # アプリ削除時にリソースも削除
spec:
  project: default

  source:
    repoURL: https://github.com/myorg/k8s-manifests.git
    targetRevision: main
    path: overlays/production

  destination:
    server: https://kubernetes.default.svc
    namespace: production

  syncPolicy:
    automated:
      prune: true          # Git から削除されたリソースを自動削除
      selfHeal: true       # 手動変更を自動で元に戻す
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true      # 他リソース同期後に削除
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m

  # ヘルスチェック設定
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # HPA と競合防止
```

### 2.4 ApplicationSet (マルチ環境・マルチクラスタ)

```yaml
# ApplicationSet: 複数環境に一括デプロイ
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: my-app
  namespace: argocd
spec:
  generators:
    # リスト型: 明示的に環境を定義
    - list:
        elements:
          - cluster: dev
            url: https://dev-cluster.example.com
            revision: develop
            replicas: "1"
          - cluster: staging
            url: https://staging-cluster.example.com
            revision: main
            replicas: "2"
          - cluster: production
            url: https://prod-cluster.example.com
            revision: main
            replicas: "3"

  template:
    metadata:
      name: 'my-app-{{cluster}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/k8s-manifests.git
        targetRevision: '{{revision}}'
        path: 'overlays/{{cluster}}'
      destination:
        server: '{{url}}'
        namespace: 'my-app-{{cluster}}'
      syncPolicy:
        automated:
          prune: true
          selfHeal: true
```

```yaml
# Git Generator: ディレクトリ構造から自動生成
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: cluster-apps
  namespace: argocd
spec:
  generators:
    - git:
        repoURL: https://github.com/myorg/k8s-manifests.git
        revision: main
        directories:
          - path: 'apps/*'
          - path: 'apps/excluded-app'
            exclude: true

  template:
    metadata:
      name: '{{path.basename}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/myorg/k8s-manifests.git
        targetRevision: main
        path: '{{path}}'
      destination:
        server: https://kubernetes.default.svc
        namespace: '{{path.basename}}'
```

### 2.5 Kustomize を使ったマニフェスト管理

```yaml
# base/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: my-app:latest
          ports:
            - containerPort: 3000
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
          livenessProbe:
            httpGet:
              path: /health
              port: 3000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
---
# base/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: my-app
spec:
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 3000
---
# base/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - deployment.yaml
  - service.yaml
commonLabels:
  managed-by: argocd
---
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
resources:
  - ../../base
patchesStrategicMerge:
  - deployment-patch.yaml
  - hpa.yaml
---
# overlays/production/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: app
          image: my-app:v1.2.3  # 本番用のタグを固定
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: 1000m
              memory: 1Gi
---
# overlays/production/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 3
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### 2.6 Helm + ArgoCD

```yaml
# Helm チャートを ArgoCD で管理
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: nginx-ingress
  namespace: argocd
spec:
  project: infrastructure
  source:
    repoURL: https://kubernetes.github.io/ingress-nginx
    chart: ingress-nginx
    targetRevision: 4.8.3
    helm:
      releaseName: nginx-ingress
      values: |
        controller:
          replicaCount: 2
          service:
            type: LoadBalancer
            annotations:
              service.beta.kubernetes.io/aws-load-balancer-type: nlb
          resources:
            requests:
              cpu: 200m
              memory: 256Mi
          metrics:
            enabled: true
  destination:
    server: https://kubernetes.default.svc
    namespace: ingress-nginx
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

---

## 3. Flux

### 3.1 Flux の構成

```yaml
# flux-system/gotk-components.yaml (Flux Bootstrap で自動生成)
# Flux は以下のコントローラーで構成される:

# 1. Source Controller: Git リポジトリを監視
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/myorg/k8s-manifests
  ref:
    branch: main
  secretRef:
    name: git-credentials  # 認証情報

# 2. Kustomize Controller: マニフェストを適用
---
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 5m
  path: ./overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: my-app
  healthChecks:
    - apiVersion: apps/v1
      kind: Deployment
      name: my-app
      namespace: production
  timeout: 3m
  retryInterval: 2m
```

### 3.2 Flux Bootstrap

```bash
# Flux のブートストラップ (GitHub)
flux bootstrap github \
  --owner=myorg \
  --repository=k8s-manifests \
  --branch=main \
  --path=clusters/production \
  --personal

# ブートストラップ後のディレクトリ構造
k8s-manifests/
├── clusters/
│   └── production/
│       └── flux-system/
│           ├── gotk-components.yaml  # Flux コンポーネント
│           ├── gotk-sync.yaml        # 自己同期設定
│           └── kustomization.yaml
├── apps/
│   └── my-app/
│       ├── base/
│       └── overlays/
└── infrastructure/
    ├── cert-manager/
    └── ingress-nginx/
```

### 3.3 Flux のイメージ自動更新

```yaml
# イメージリポジトリの監視
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageRepository
metadata:
  name: my-app
  namespace: flux-system
spec:
  image: ghcr.io/myorg/my-app
  interval: 5m
  secretRef:
    name: ghcr-credentials

---
# イメージポリシー: セマンティックバージョニングで最新を選択
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImagePolicy
metadata:
  name: my-app
  namespace: flux-system
spec:
  imageRepositoryRef:
    name: my-app
  policy:
    semver:
      range: ">=1.0.0"

---
# 自動更新設定: Git にコミットを自動作成
apiVersion: image.toolkit.fluxcd.io/v1beta2
kind: ImageUpdateAutomation
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 5m
  sourceRef:
    kind: GitRepository
    name: my-app
  git:
    checkout:
      ref:
        branch: main
    commit:
      author:
        name: flux-bot
        email: flux@myorg.com
      messageTemplate: 'chore: update {{.AutomationObject}} images'
    push:
      branch: main
  update:
    path: ./overlays/production
    strategy: Setters
```

```yaml
# マニフェストにマーカーを設定 (イメージ自動更新用)
# overlays/production/deployment-patch.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  template:
    spec:
      containers:
        - name: app
          image: ghcr.io/myorg/my-app:v1.2.3  # {"$imagepolicy": "flux-system:my-app"}
```

### 3.4 Flux の通知設定

```yaml
# Slack 通知
apiVersion: notification.toolkit.fluxcd.io/v1beta3
kind: Provider
metadata:
  name: slack
  namespace: flux-system
spec:
  type: slack
  channel: deployments
  secretRef:
    name: slack-webhook

---
apiVersion: notification.toolkit.fluxcd.io/v1beta3
kind: Alert
metadata:
  name: deployment-alerts
  namespace: flux-system
spec:
  providerRef:
    name: slack
  eventSeverity: info
  eventSources:
    - kind: Kustomization
      name: '*'
    - kind: HelmRelease
      name: '*'
  exclusionList:
    - ".*upgrade.*"  # アップグレード通知を除外
```

---

## 4. ArgoCD vs Flux 比較

| 項目 | ArgoCD | Flux |
|---|---|---|
| UI | リッチなWeb UI内蔵 | 別途 Weave GitOps UI |
| マルチクラスタ | ApplicationSet で対応 | Kustomization のリモート参照 |
| Helm 対応 | ネイティブ対応 | HelmRelease CRD |
| SSO/RBAC | 内蔵 (OIDC, LDAP) | Kubernetes RBAC に委任 |
| イメージ自動更新 | Argo Image Updater | Image Automation Controller |
| アーキテクチャ | 中央集権型 (Server + UI) | 分散型 (CRD ベース) |
| 学習コスト | 低 (UIが直感的) | 中 (CRDの理解が必要) |
| 適用場面 | チームがUIを重視 | 軽量・CRDベースを好む |
| リソース消費 | 中〜大 (UI+Server) | 小〜中 (Controller のみ) |
| CNCF ステータス | Graduated | Graduated |
| Webhook 対応 | あり (即座同期トリガー) | あり (Receiver Controller) |

### GitOps ツールのデプロイフロー比較

| ステップ | ArgoCD | Flux |
|---|---|---|
| リポジトリ監視 | Application CRD | GitRepository CRD |
| 差分検知 | 3分間隔 (デフォルト) | 1分間隔 (設定可能) |
| マニフェスト生成 | Kustomize/Helm/jsonnet | Kustomize/Helm |
| 同期 | Sync (自動/手動) | Reconcile (自動) |
| ヘルスチェック | 内蔵 (多数のリソース対応) | healthChecks フィールド |
| ロールバック | UI からワンクリック | Git revert → 自動同期 |

### 4.1 選択ガイド

```
GitOps ツール選定:

チームが Kubernetes の CRD に慣れている？
├── Yes → Flux を検討
│         ├── 軽量で運用コストを抑えたい → Flux
│         ├── イメージ自動更新が必要 → Flux (Image Automation)
│         └── 複数のソースからマニフェストを取得 → Flux
└── No → ArgoCD を検討
          ├── Web UI でデプロイ状況を可視化したい → ArgoCD
          ├── マルチクラスタを一元管理したい → ArgoCD (ApplicationSet)
          ├── SSO/RBAC が必要 → ArgoCD
          └── 非技術者にもデプロイ状況を共有したい → ArgoCD
```

---

## 5. イミュータブルインフラストラクチャ

### 5.1 ミュータブル vs イミュータブル

```
ミュータブル (従来型):
  サーバーに SSH → パッチ適用 → 設定変更 → 再起動
  問題: 設定のドリフト、スノーフレークサーバー

  Server v1 → patch → Server v1.1 → patch → Server v1.2
  (同じサーバーを変更し続ける)

イミュータブル (GitOps):
  新しいイメージをビルド → 古いコンテナを新しいコンテナに置換
  利点: 再現性、ロールバック容易、ドリフトなし

  Container v1 → 破棄
  Container v2 → 新規作成
  Container v3 → 新規作成
  (変更するのではなく、置き換える)
```

### 5.2 GitOps + イミュータブルインフラの完全フロー

```
GitOps + イミュータブルインフラの流れ:
  ┌────────┐    ┌────────┐    ┌────────────┐    ┌──────────┐
  │コード変更│ →  │CI: ビルド│ →  │イメージPush │ →  │マニフェスト│
  │PR merge │    │テスト   │    │ghcr.io/... │    │のタグ更新 │
  └────────┘    └────────┘    └────────────┘    └────┬─────┘
                                                      │
                                                      ↓
                                                ┌──────────┐
                                                │Git commit │
                                                │(自動/手動) │
                                                └────┬─────┘
                                                      │
                                                      ↓
                                                ┌──────────┐
                                                │ArgoCD/Flux│
                                                │が同期     │
                                                └────┬─────┘
                                                      │
                                                      ↓
                                                ┌──────────────┐
                                                │ Rolling Update│
                                                │ 旧Pod → 新Pod │
                                                └──────────────┘
```

### 5.3 イメージタグ戦略

```
推奨: タグの不変性を保証する

✗ 悪い例: latest タグを上書き
  image: my-app:latest  # どのバージョンか不明、ロールバック不可

✗ 微妙な例: ブランチ名タグ
  image: my-app:main    # 上書きされる可能性

○ 良い例: Git SHA タグ
  image: my-app:abc1234  # 不変、追跡可能

◎ 最良: セマンティックバージョン + SHA
  image: my-app:v1.2.3-abc1234  # バージョン + 追跡性

  タグ付けの CI 設定:
    tags:
      - ghcr.io/myorg/my-app:${{ github.sha }}
      - ghcr.io/myorg/my-app:v${{ steps.version.outputs.value }}
```

---

## 6. GitOps のリポジトリ戦略

### 6.1 モノレポ vs 分離レポ

```yaml
# 戦略1: モノレポ (アプリ + マニフェスト同一リポジトリ)
my-app/
├── src/                    # アプリケーションコード
├── Dockerfile
├── k8s/                    # Kubernetes マニフェスト
│   ├── base/
│   └── overlays/
└── .github/workflows/      # CI/CD

# メリット: シンプル、アプリとインフラの変更を一緒にレビュー
# デメリット: CI が頻繁にマニフェスト変更で発火、権限分離が困難

# 戦略2: 分離レポ (推奨)
my-app/                     # アプリリポジトリ
├── src/
├── Dockerfile
└── .github/workflows/

k8s-manifests/              # マニフェストリポジトリ
├── apps/
│   ├── my-app/
│   │   ├── base/
│   │   └── overlays/
│   └── my-api/
├── infrastructure/
│   ├── cert-manager/
│   └── ingress-nginx/
└── clusters/
    ├── dev/
    ├── staging/
    └── production/

# メリット: 関心の分離、アクセス制御、CI/CD の独立
# デメリット: リポジトリ間の連携が必要
```

### 6.2 マニフェストリポジトリの推奨構成

```
k8s-manifests/
├── apps/                          # アプリケーション
│   ├── frontend/
│   │   ├── base/
│   │   │   ├── deployment.yaml
│   │   │   ├── service.yaml
│   │   │   ├── ingress.yaml
│   │   │   └── kustomization.yaml
│   │   └── overlays/
│   │       ├── dev/
│   │       │   ├── kustomization.yaml
│   │       │   └── patch.yaml
│   │       ├── staging/
│   │       └── production/
│   ├── backend/
│   │   ├── base/
│   │   └── overlays/
│   └── worker/
│       ├── base/
│       └── overlays/
├── infrastructure/                # インフラコンポーネント
│   ├── cert-manager/
│   │   ├── namespace.yaml
│   │   ├── helmrelease.yaml       # Helm リリース定義
│   │   └── kustomization.yaml
│   ├── ingress-nginx/
│   ├── external-secrets/
│   ├── prometheus-stack/
│   └── kustomization.yaml
├── clusters/                      # クラスタ別設定
│   ├── dev/
│   │   ├── apps.yaml             # どのアプリをデプロイするか
│   │   └── infrastructure.yaml   # どのインフラをデプロイするか
│   ├── staging/
│   └── production/
└── README.md
```

---

## 7. シークレット管理

GitOps では全ての状態を Git に保存するが、シークレット(パスワード、APIキー等)をそのままGitにコミットすることはセキュリティ上許容されない。以下の3つのアプローチが主流である。

### 7.1 Sealed Secrets

```yaml
# Sealed Secrets: クラスタの公開鍵で暗号化して Git に保存
# 暗号化されたシークレットは Git にコミット可能

# 1. kubeseal でシークレットを暗号化
# kubeseal < secret.yaml > sealed-secret.yaml

apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: db-credentials
  namespace: production
spec:
  encryptedData:
    password: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...  # 暗号化済み
    username: AgBy3i4OJSWK+PiTySYZZA9rO43cGDEq...  # 暗号化済み
  template:
    metadata:
      name: db-credentials
      namespace: production
    type: Opaque
```

### 7.2 External Secrets Operator

```yaml
# External Secrets Operator: 外部シークレットストアから取得
# AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager 等と連携

# ClusterSecretStore の定義
apiVersion: external-secrets.io/v1beta1
kind: ClusterSecretStore
metadata:
  name: aws-secrets-manager
spec:
  provider:
    aws:
      service: SecretsManager
      region: ap-northeast-1
      auth:
        jwt:
          serviceAccountRef:
            name: external-secrets
            namespace: external-secrets

---
# ExternalSecret: AWS Secrets Manager からシークレットを取得
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
  namespace: production
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: db-credentials
    creationPolicy: Owner
  data:
    - secretKey: password
      remoteRef:
        key: prod/db/password
    - secretKey: username
      remoteRef:
        key: prod/db/username
```

### 7.3 SOPS (Secrets OPerationS)

```yaml
# SOPS: Mozilla が開発した暗号化ツール
# AGE, PGP, AWS KMS, GCP KMS, Azure Key Vault で暗号化

# .sops.yaml (暗号化ルール)
creation_rules:
  - path_regex: .*secrets.*\.yaml$
    age: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  - path_regex: .*prod.*secrets.*\.yaml$
    kms: arn:aws:kms:ap-northeast-1:123456789012:key/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# 暗号化されたファイル例
apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  password: ENC[AES256_GCM,data:xxxxx,iv:xxxxx,tag:xxxxx,type:str]
  username: ENC[AES256_GCM,data:xxxxx,iv:xxxxx,tag:xxxxx,type:str]
sops:
  kms: []
  age:
    - recipient: age1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      enc: |
        -----BEGIN AGE ENCRYPTED FILE-----
        ...
        -----END AGE ENCRYPTED FILE-----
  lastmodified: "2024-01-15T10:00:00Z"
  version: 3.8.1
```

```yaml
# Flux での SOPS 統合
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: my-app
  namespace: flux-system
spec:
  decryption:
    provider: sops
    secretRef:
      name: sops-age  # AGE キーを含む Secret
  interval: 5m
  path: ./overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: my-app
```

### 7.4 シークレット管理の比較

| 項目 | Sealed Secrets | External Secrets | SOPS |
|---|---|---|---|
| 暗号化場所 | クラスタ内 | 外部サービス | ローカル/CI |
| 復号場所 | クラスタ内 | クラスタ内 | クラスタ内 |
| Git に保存 | 暗号化済みOK | 参照のみ | 暗号化済みOK |
| ローテーション | 手動 | 外部サービスで自動 | 手動 |
| 複数クラスタ | クラスタごとに再暗号化 | 共通の外部ストア | 共通キーで暗号化 |
| 依存関係 | クラスタ公開鍵 | 外部サービス | 暗号化キー |
| 推奨場面 | シンプルな構成 | エンタープライズ | マルチ環境 |

---

## 8. CI パイプラインとGitOpsの連携

### 8.1 典型的な連携フロー

```yaml
# アプリリポジトリの CI (GitOps 連携)
name: CI + GitOps Deploy
on:
  push:
    branches: [main]

jobs:
  ci:
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test && npm run build

      - name: Build and push Docker image
        id: meta
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}:latest

  update-manifests:
    needs: ci
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          repository: myorg/k8s-manifests
          token: ${{ secrets.MANIFEST_REPO_TOKEN }}
          ref: main

      - name: Update image tag in manifests
        run: |
          cd overlays/production
          kustomize edit set image \
            my-app=ghcr.io/${{ github.repository }}:${{ github.sha }}

      - name: Commit and push
        run: |
          git config user.name "ci-bot"
          git config user.email "ci@example.com"
          git add .
          git commit -m "chore: update my-app image to ${{ github.sha }}"
          git push

      # この後、ArgoCD/Flux が変更を検知して自動デプロイ
```

### 8.2 PR ベースのデプロイフロー

```yaml
# 本番デプロイを PR 経由で行う (より安全なアプローチ)
name: Create Deploy PR
on:
  push:
    branches: [main]

jobs:
  create-deploy-pr:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          repository: myorg/k8s-manifests
          token: ${{ secrets.MANIFEST_REPO_TOKEN }}

      - name: Create feature branch
        run: |
          BRANCH="deploy/my-app-${{ github.sha }}"
          git checkout -b $BRANCH

      - name: Update image tag
        run: |
          cd overlays/production
          kustomize edit set image \
            my-app=ghcr.io/myorg/my-app:${{ github.sha }}

      - name: Create PR
        run: |
          git add .
          git commit -m "chore: deploy my-app ${{ github.sha }}"
          git push origin $BRANCH
          gh pr create \
            --title "Deploy my-app ${{ github.sha }}" \
            --body "Auto-generated deploy PR for my-app" \
            --base main
        env:
          GH_TOKEN: ${{ secrets.MANIFEST_REPO_TOKEN }}
```

---

## 9. 運用とトラブルシューティング

### 9.1 ArgoCD のトラブルシューティング

```bash
# 同期状態の確認
argocd app get my-app

# 同期の詳細ログ
argocd app sync my-app --dry-run
argocd app sync my-app --force

# 差分の確認
argocd app diff my-app

# リソースの状態確認
argocd app resources my-app

# 同期履歴
argocd app history my-app

# ロールバック
argocd app rollback my-app <HISTORY_ID>

# Webhook の手動トリガー
argocd app sync my-app --revision HEAD
```

### 9.2 よくある問題と対処法

```
問題1: OutOfSync が解消しない
原因: ignoreDifferences の設定漏れ
対処:
  - HPA が replicas を変更 → ignoreDifferences に /spec/replicas を追加
  - Webhook が Last Applied を変更 → metadata.annotations を除外
  - Operator が Status を更新 → status フィールドを無視

問題2: Sync Failed - ComparisonError
原因: マニフェストの構文エラー、CRD 未インストール
対処:
  - kustomize build でローカル検証
  - CRD が先にインストールされているか確認
  - sync-wave アノテーションで順序制御

問題3: ImagePullBackOff
原因: イメージタグの不一致、レジストリ認証エラー
対処:
  - イメージタグが正しいか確認
  - imagePullSecrets の設定確認
  - ECR/GHCR のトークン有効期限確認
```

### 9.3 Sync Wave (同期順序の制御)

```yaml
# CRD を先にインストールし、次にアプリケーションを同期
apiVersion: v1
kind: Namespace
metadata:
  name: cert-manager
  annotations:
    argocd.argoproj.io/sync-wave: "-2"  # 最初に作成

---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: cert-manager
  annotations:
    argocd.argoproj.io/sync-wave: "-1"  # 名前空間の後
spec:
  source:
    repoURL: https://charts.jetstack.io
    chart: cert-manager
    targetRevision: v1.13.3

---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  annotations:
    argocd.argoproj.io/sync-wave: "0"   # cert-manager の後
spec:
  source:
    repoURL: https://github.com/myorg/k8s-manifests.git
    path: overlays/production
```

---

## 10. アンチパターン

### アンチパターン1: Git を経由しない手動変更

```
問題:
  kubectl apply -f ... で直接クラスタに変更を適用する

  Git の状態: replicas: 3
  クラスタの状態: replicas: 5 (kubectl scale で手動変更)

  → ArgoCD が差分を検知し、replicas: 3 に戻してしまう
  → または selfHeal が無効の場合、ドリフトが放置される

改善:
  - selfHeal: true を有効にする
  - kubectl での直接変更を禁止する RBAC 設定
  - 全ての変更は Git PR 経由で行うルールを徹底
  - 緊急時の手順書にも「PR 経由」を明記
  - kubectl の write 権限を限定的な緊急用 ServiceAccount のみに付与
```

### アンチパターン2: シークレットを Git にコミット

```
問題:
  データベースパスワードや API キーを平文で Git にコミット
  → セキュリティ事故

改善手段:
  1. Sealed Secrets: クラスタの公開鍵で暗号化して Git に保存
  2. External Secrets Operator: AWS Secrets Manager 等から取得
  3. SOPS (Mozilla): 暗号化して Git に保存、復号は CI/CD で
  4. Vault Agent: HashiCorp Vault からランタイムで取得
```

### アンチパターン3: latest タグの使用

```
問題:
  image: my-app:latest で GitOps を運用
  → どのバージョンがデプロイされているか不明
  → ロールバック不可
  → ArgoCD が差分を検知できない(タグは同じだが中身が違う)

改善:
  - 必ず不変のタグ(Git SHA、セマンティックバージョン)を使用
  - imagePullPolicy: Always は使わない
  - イメージタグの更新は Git コミットで管理
```

### アンチパターン4: 全環境を1つの Application で管理

```
問題:
  dev/staging/production を1つの ArgoCD Application で管理
  → 環境ごとの独立したデプロイができない
  → dev の変更が production に影響

改善:
  - 環境ごとに別々の Application を定義
  - ApplicationSet で環境パラメータを変数化
  - overlays で環境ごとの差分を管理
```

---

## 11. FAQ

### Q1: GitOps は Kubernetes 以外でも使えるか？

原理的にはイエス。GitOps の本質は「Git をSingle Source of Truth として宣言的状態を自動同期する」ことであり、Kubernetes に限定されない。Terraform + Atlantis / Spacelift でインフラに GitOps を適用したり、Crossplane で任意のクラウドリソースを Kubernetes CRD として管理する方法もある。ただし、ツールの成熟度は Kubernetes エコシステムが最も高い。

### Q2: GitOps でのロールバックはどう行うか？

Git revert で前のバージョンのコミットを作成し、PR をマージする。ArgoCD/Flux がその変更を検知して自動同期する。これにより「ロールバックもGitの履歴に残る」という利点がある。ArgoCD のUIからは過去の同期ポイントに直接ロールバックすることも可能。ただし、データベースマイグレーションを伴う場合は単純な revert では対応できないため、別途手順が必要。

### Q3: CI パイプラインと GitOps はどう連携するか？

典型的なフローは (1) アプリリポジトリで CI がビルド・テスト・イメージプッシュ、(2) CI がマニフェストリポジトリの PR を自動作成(イメージタグ更新)、(3) PR レビュー・マージ、(4) ArgoCD/Flux が自動同期。CI はイメージ作成まで、デプロイは GitOps エージェントが担当する責務分離が重要。

### Q4: GitOps で DB マイグレーションはどう扱うか？

DB マイグレーションは宣言的ではなく命令的な操作であるため、GitOps の原則に完全には適合しない。一般的なアプローチは、(1) Init Container でマイグレーションを実行、(2) Kubernetes Job としてマイグレーションを実行、(3) ArgoCD の PreSync Hook でマイグレーションを実行。いずれの場合もロールバック戦略を事前に定義しておくことが重要。

### Q5: マルチクラスタの GitOps はどう設計するか？

ArgoCD の場合は ApplicationSet を使い、クラスタごとに異なるパラメータ(URL, 環境名, レプリカ数等)を指定する。Hub-and-Spoke モデル(中央の管理クラスタから各クラスタを制御)が一般的。Flux の場合は各クラスタに Flux をインストールし、同じ Git リポジトリの異なるパス(clusters/dev/, clusters/prod/)を参照させる。

### Q6: GitOps の導入を段階的に進めるには？

Phase 1: 開発環境のみ GitOps 化(ArgoCD/Flux のインストール、1アプリの同期)、Phase 2: ステージング環境に拡大(ApplicationSet でマルチ環境)、Phase 3: 本番環境に適用(Protection Rules, シークレット管理)、Phase 4: インフラコンポーネントも GitOps 化(cert-manager, ingress 等)。各フェーズで1-2ヶ月の安定稼働を確認してから次に進む。

---

## まとめ

| 項目 | 要点 |
|---|---|
| GitOps の本質 | Git が唯一の信頼源、宣言的、プルベース、自動リコンサイル |
| プル型の利点 | 認証情報の局所化、ドリフト自動修復、監査証跡 |
| ArgoCD | リッチUI、マルチクラスタ、ApplicationSet、直感的 |
| Flux | 軽量、CRDベース、イメージ自動更新、SOPS統合 |
| イミュータブル | 変更するのではなく置き換える、不変タグ必須 |
| シークレット管理 | Sealed Secrets / External Secrets / SOPS |
| リポジトリ戦略 | アプリとマニフェストを分離(推奨) |
| CI連携 | CIはイメージ作成まで、デプロイはGitOpsエージェント |
| トラブルシューティング | argocd app diff/sync、ログ確認、sync-wave |

---

## 次に読むべきガイド

- [コンテナデプロイ](../02-deployment/02-container-deployment.md) -- Kubernetes でのデプロイ実践
- [デプロイ戦略](../02-deployment/00-deployment-strategies.md) -- Canary、Blue-Green との組合せ
- [Actions セキュリティ](../01-github-actions/04-security-actions.md) -- OIDC連携とセキュアなCI
- [Infrastructure as Code](./02-infrastructure-as-code.md) -- IaCとGitOpsの連携

---

## 参考文献

1. OpenGitOps. "GitOps Principles." https://opengitops.dev/
2. Argo Project. "Argo CD - Declarative GitOps CD for Kubernetes." https://argo-cd.readthedocs.io/
3. Fluxcd. "Flux - the GitOps family of projects." https://fluxcd.io/docs/
4. Weaveworks. "Guide To GitOps." https://www.weave.works/technologies/gitops/
5. Cornelia Davis. *Cloud Native Patterns*. Manning Publications, 2019.
6. Bilgin Ibryam, Roland Hub. *Kubernetes Patterns*, 2nd Edition. O'Reilly Media, 2023.
7. External Secrets Operator. "ESO Documentation." https://external-secrets.io/
8. Bitnami Labs. "Sealed Secrets." https://github.com/bitnami-labs/sealed-secrets
