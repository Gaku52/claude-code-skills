# GitOps

> Gitリポジトリを唯一の信頼源(Single Source of Truth)とし、インフラとアプリケーションの宣言的な状態をプルベースで自動同期する運用モデル

## この章で学ぶこと

1. GitOpsの4原則とプッシュ型/プル型デプロイの違いを理解する
2. ArgoCD、Fluxの仕組みと基本的な設定方法を習得する
3. イミュータブルインフラストラクチャとGitOpsの関係を把握する

---

## 1. GitOps とは

### 1.1 GitOps の4原則

```
GitOps 4原則 (OpenGitOps):

1. 宣言的 (Declarative)
   システムのあるべき状態をコードで宣言する

2. バージョン管理・イミュータブル (Versioned and Immutable)
   あるべき状態は Git で管理し、変更履歴を保持する

3. 自動プル (Pulled Automatically)
   承認された変更はシステムに自動的に適用される

4. 継続的リコンサイル (Continuously Reconciled)
   実際の状態とあるべき状態の差分を検知し、自動修復する
```

### 1.2 プッシュ型 vs プル型デプロイ

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
```

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

### 2.2 ArgoCD Application 定義

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
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
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
```

### 2.3 Kustomize を使ったマニフェスト管理

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
---
# overlays/production/kustomization.yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namespace: production
bases:
  - ../../base
patchesStrategicMerge:
  - deployment-patch.yaml
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
```

### 3.2 Flux のイメージ自動更新

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

### GitOps ツールのデプロイフロー比較

| ステップ | ArgoCD | Flux |
|---|---|---|
| リポジトリ監視 | Application CRD | GitRepository CRD |
| 差分検知 | 3分間隔 (デフォルト) | 1分間隔 (設定可能) |
| マニフェスト生成 | Kustomize/Helm/jsonnet | Kustomize/Helm |
| 同期 | Sync (自動/手動) | Reconcile (自動) |
| ヘルスチェック | 内蔵 (多数のリソース対応) | healthChecks フィールド |
| ロールバック | UI からワンクリック | Git revert → 自動同期 |

---

## 5. イミュータブルインフラストラクチャ

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
                                                └──────────┘
```

---

## 6. GitOps のリポジトリ戦略

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
# デメリット: CI が頻繁にマニフェスト変更で発火

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

---

## 7. アンチパターン

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
```

### アンチパターン2: シークレットを Git にコミット

```
問題:
  データベースパスワードや API キーを Git にコミット
  → セキュリティ事故

改善手段:
  1. Sealed Secrets: クラスタの公開鍵で暗号化して Git に保存
  2. External Secrets Operator: AWS Secrets Manager 等から取得
  3. SOPS (Mozilla): 暗号化して Git に保存、復号は CI/CD で

# External Secrets Operator の例
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: aws-secrets-manager
    kind: ClusterSecretStore
  target:
    name: db-credentials
  data:
    - secretKey: password
      remoteRef:
        key: prod/db/password
```

---

## 8. FAQ

### Q1: GitOps は Kubernetes 以外でも使えるか？

原理的にはイエス。GitOps の本質は「Git をSingle Source of Truth として宣言的状態を自動同期する」ことであり、Kubernetes に限定されない。Terraform + Atlantis / Spacelift でインフラに GitOps を適用したり、Crossplane で任意のクラウドリソースを Kubernetes CRD として管理する方法もある。

### Q2: GitOps でのロールバックはどう行うか？

Git revert で前のバージョンのコミットを作成し、PR をマージする。ArgoCD/Flux がその変更を検知して自動同期する。これにより「ロールバックもGitの履歴に残る」という利点がある。ArgoCD のUIからは過去の同期ポイントに直接ロールバックすることも可能。

### Q3: CI パイプラインと GitOps はどう連携するか？

典型的なフローは (1) アプリリポジトリで CI がビルド・テスト・イメージプッシュ、(2) CI がマニフェストリポジトリの PR を自動作成(イメージタグ更新)、(3) PR レビュー・マージ、(4) ArgoCD/Flux が自動同期。CI はイメージ作成まで、デプロイは GitOps エージェントが担当する責務分離が重要。

---

## まとめ

| 項目 | 要点 |
|---|---|
| GitOps の本質 | Git が唯一の信頼源、宣言的、プルベース、自動リコンサイル |
| プル型の利点 | 認証情報の局所化、ドリフト自動修復、監査証跡 |
| ArgoCD | リッチUI、マルチクラスタ、直感的 |
| Flux | 軽量、CRDベース、イメージ自動更新 |
| イミュータブル | 変更するのではなく置き換える |
| シークレット管理 | Sealed Secrets / External Secrets / SOPS |

---

## 次に読むべきガイド

- [コンテナデプロイ](../02-deployment/02-container-deployment.md) -- Kubernetes でのデプロイ実践
- [デプロイ戦略](../02-deployment/00-deployment-strategies.md) -- Canary、Blue-Green との組合せ
- [Actions セキュリティ](../01-github-actions/04-security-actions.md) -- OIDC連携とセキュアなCI

---

## 参考文献

1. OpenGitOps. "GitOps Principles." https://opengitops.dev/
2. Argo Project. "Argo CD - Declarative GitOps CD for Kubernetes." https://argo-cd.readthedocs.io/
3. Fluxcd. "Flux - the GitOps family of projects." https://fluxcd.io/docs/
4. Weaveworks. "Guide To GitOps." https://www.weave.works/technologies/gitops/
