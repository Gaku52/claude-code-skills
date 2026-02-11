# Amazon EKS 概要

> Amazon Elastic Kubernetes Service (EKS) のクラスター作成、ノードグループ、Fargate プロファイル、Helm、IRSA (IAM Roles for Service Accounts) までを体系的に学ぶ。

---

## この章で学ぶこと

1. **EKS クラスターの構成と作成** -- コントロールプレーンとデータプレーンの関係、eksctl によるクラスター構築を理解する
2. **ノードグループと Fargate プロファイル** -- マネージドノードグループ、セルフマネージドノード、Fargate の使い分けを習得する
3. **Helm と IRSA の活用** -- パッケージマネージャによるアプリケーションデプロイと、Pod レベルの IAM 権限制御を身につける

---

## 1. EKS のアーキテクチャ

### 1.1 全体構成

```
+----------------------------------------------------------+
|  Amazon EKS クラスター                                     |
|                                                          |
|  コントロールプレーン (AWS 管理)                              |
|  +------------------------------------------------------+|
|  | +--------+ +----------+ +----------+ +-----------+   ||
|  | | kube-  | | kube-    | | kube-    | | etcd      |   ||
|  | | api-   | | scheduler| | controller| | (3 AZ)   |   ||
|  | | server | |          | | manager  | |           |   ||
|  | +--------+ +----------+ +----------+ +-----------+   ||
|  +------------------------------------------------------+|
|           |                                              |
|           | (ENI / API エンドポイント)                      |
|           |                                              |
|  データプレーン (ユーザー管理)                                |
|  +------------------------------------------------------+|
|  | マネージドノードグループ     Fargate                     ||
|  | +----------+ +----------+ +-----------+              ||
|  | | EC2 Node | | EC2 Node | | Fargate   |              ||
|  | | +------+ | | +------+ | | Pod       |              ||
|  | | | Pod  | | | | Pod  | | |           |              ||
|  | | | Pod  | | | | Pod  | | +-----------+              ||
|  | | +------+ | | +------+ | +-----------+              ||
|  | | kubelet  | | kubelet  | | Fargate   |              ||
|  | +----------+ +----------+ | Pod       |              ||
|  |                           +-----------+              ||
|  +------------------------------------------------------+|
+----------------------------------------------------------+
```

### 1.2 EKS vs ECS 比較

| 特性 | EKS | ECS |
|------|-----|-----|
| オーケストレータ | Kubernetes | AWS 独自 |
| 学習コスト | 高い | 低い |
| 可搬性 | 高い (K8s 互換) | AWS 固有 |
| エコシステム | 非常に広い (CNCF) | AWS サービスと密連携 |
| コントロールプレーン費用 | $0.10/時 (~$73/月) | 無料 |
| 設定の柔軟性 | 非常に高い | 適度 |
| 運用複雑性 | 高い | 低い |
| 向いている組織 | K8s 経験あり/マルチクラウド | AWS 中心/小規模チーム |

---

## 2. クラスターの作成

### 2.1 eksctl によるクラスター作成

```yaml
# cluster-config.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: my-cluster
  region: ap-northeast-1
  version: "1.29"

vpc:
  cidr: "10.0.0.0/16"
  nat:
    gateway: HighlyAvailable  # 各AZにNAT GW

managedNodeGroups:
  - name: general-purpose
    instanceType: m5.large
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 50
    volumeType: gp3
    labels:
      role: general
    tags:
      Environment: production
    iam:
      withAddonPolicies:
        albIngress: true
        cloudWatch: true

  - name: spot-workers
    instanceTypes:
      - m5.large
      - m5a.large
      - m4.large
    spot: true
    desiredCapacity: 5
    minSize: 0
    maxSize: 20
    labels:
      role: spot-worker

fargateProfiles:
  - name: default
    selectors:
      - namespace: fargate-tasks
        labels:
          compute: fargate

cloudWatch:
  clusterLogging:
    enableTypes:
      - api
      - audit
      - authenticator
      - controllerManager
      - scheduler
```

```bash
# クラスター作成
eksctl create cluster -f cluster-config.yaml

# kubeconfig の設定
aws eks update-kubeconfig --name my-cluster --region ap-northeast-1

# クラスター情報の確認
kubectl cluster-info
kubectl get nodes
```

### 2.2 AWS CLI によるクラスター作成

```bash
# 1. クラスターロールの作成
aws iam create-role \
  --role-name eksClusterRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "eks.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name eksClusterRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

# 2. クラスターの作成
aws eks create-cluster \
  --name my-cluster \
  --role-arn arn:aws:iam::123456789012:role/eksClusterRole \
  --resources-vpc-config \
    subnetIds=subnet-111,subnet-222,subnet-333,\
securityGroupIds=sg-12345678 \
  --kubernetes-version 1.29
```

---

## 3. ノードグループ

### 3.1 ノードタイプの比較

```
ノードの選択肢:

+-------------------+     +-------------------+     +-------------------+
| マネージドノード   |     | セルフマネージド    |     | Fargate           |
| グループ          |     | ノード            |     |                   |
+-------------------+     +-------------------+     +-------------------+
| EC2 の管理を      |     | EC2 を完全に      |     | サーバーレス       |
| AWS が支援        |     | ユーザーが管理    |     | Pod 単位の実行     |
|                   |     |                   |     |                   |
| AMI 更新: 半自動  |     | AMI 更新: 手動    |     | AMI 管理: 不要    |
| ASG 管理: 自動    |     | ASG 管理: 手動    |     | スケール: 自動    |
| ドレイン: 自動    |     | ドレイン: 手動    |     | DaemonSet: 不可   |
+-------------------+     +-------------------+     +-------------------+
```

| 項目 | マネージドノードグループ | セルフマネージド | Fargate |
|------|----------------------|----------------|---------|
| インフラ管理 | 低 | 高 | なし |
| カスタマイズ性 | 中 | 高 | 低 |
| GPU サポート | あり | あり | なし |
| DaemonSet | あり | あり | なし |
| 起動速度 | 中 (EC2起動) | 中 | やや遅い |
| コスト | EC2 料金 | EC2 料金 | vCPU+メモリ課金 |

### 3.2 マネージドノードグループの作成

```bash
# ノードロールの作成
aws iam create-role \
  --role-name eksNodeRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "ec2.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# 必要なポリシーのアタッチ
aws iam attach-role-policy --role-name eksNodeRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
aws iam attach-role-policy --role-name eksNodeRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
aws iam attach-role-policy --role-name eksNodeRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly

# ノードグループの作成
aws eks create-nodegroup \
  --cluster-name my-cluster \
  --nodegroup-name general-ng \
  --node-role arn:aws:iam::123456789012:role/eksNodeRole \
  --subnets subnet-111 subnet-222 subnet-333 \
  --instance-types m5.large \
  --scaling-config minSize=2,maxSize=10,desiredSize=3 \
  --disk-size 50 \
  --capacity-type ON_DEMAND
```

---

## 4. Fargate プロファイル

### 4.1 Fargate プロファイルの仕組み

```
Fargate Pod スケジューリング:

Pod 作成リクエスト
    |
    v
+----------------------------+
| EKS スケジューラ            |
| Fargate プロファイルを確認   |
+----------------------------+
    |
    | namespace + labels がマッチ？
    |
 +--+--+
 |     |
マッチ  不一致
 |     |
 v     v
Fargate  EC2 ノード
で実行   で実行

Fargate プロファイル:
  namespace: "batch-jobs"
  labels:
    compute: "fargate"
```

### 4.2 Fargate プロファイルの作成

```bash
# Fargate Pod 実行ロール
aws iam create-role \
  --role-name eksFargatePodRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "eks-fargate-pods.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy --role-name eksFargatePodRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonEKSFargatePodExecutionRolePolicy

# Fargate プロファイルの作成
aws eks create-fargate-profile \
  --cluster-name my-cluster \
  --fargate-profile-name batch-profile \
  --pod-execution-role-arn arn:aws:iam::123456789012:role/eksFargatePodRole \
  --subnets subnet-111 subnet-222 \
  --selectors '[
    {
      "namespace": "batch-jobs",
      "labels": {"compute": "fargate"}
    }
  ]'
```

---

## 5. Helm

### 5.1 Helm の概念

```
Helm の構成要素:

Chart (パッケージ):
  my-chart/
  ├── Chart.yaml          # チャートのメタデータ
  ├── values.yaml         # デフォルト設定値
  ├── templates/          # Kubernetes マニフェストテンプレート
  │   ├── deployment.yaml
  │   ├── service.yaml
  │   ├── ingress.yaml
  │   └── _helpers.tpl    # テンプレートヘルパー
  └── charts/             # 依存チャート

Release (インストール済みインスタンス):
  helm install my-release my-chart --values custom-values.yaml
```

### 5.2 Helm によるアプリケーションデプロイ

```bash
# Helm リポジトリの追加
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# チャートの検索
helm search repo nginx

# インストール (NGINX Ingress Controller の例)
helm install ingress-nginx ingress-nginx/ingress-nginx \
  --namespace ingress-nginx \
  --create-namespace \
  --set controller.service.type=LoadBalancer \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/aws-load-balancer-type"=nlb

# カスタム values ファイルでインストール
helm install my-app ./my-chart \
  --namespace production \
  --create-namespace \
  --values production-values.yaml

# アップグレード
helm upgrade my-app ./my-chart \
  --namespace production \
  --values production-values.yaml

# ロールバック
helm rollback my-app 1 --namespace production

# 一覧確認
helm list --all-namespaces
```

### 5.3 values.yaml の例

```yaml
# production-values.yaml
replicaCount: 3

image:
  repository: 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app
  tag: "v1.2.3"
  pullPolicy: IfNotPresent

resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70

ingress:
  enabled: true
  className: alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
  hosts:
    - host: api.example.com
      paths:
        - path: /
          pathType: Prefix

serviceAccount:
  create: true
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/my-app-role
```

---

## 6. IRSA (IAM Roles for Service Accounts)

### 6.1 IRSA の仕組み

```
IRSA の認証フロー:

1. Pod 起動時、ServiceAccount にアノテーションされた
   IAM ロール ARN を取得

2. EKS が OIDC プロバイダ経由で STS と連携

3. Pod 内のアプリケーションが AWS SDK で
   自動的に一時認証情報を取得

+--------+     +--------+     +---------+     +-----+
| Pod    | --> | OIDC   | --> | AWS STS | --> | IAM |
| (SDK)  |     |Provider|     |AssumeRole|     |Role |
+--------+     +--------+     +---------+     +-----+
    |                                            |
    | AWS_WEB_IDENTITY_TOKEN_FILE                 |
    | AWS_ROLE_ARN                               |
    v                                            v
+--------+                                  +--------+
|一時認証 | <------------------------------- |権限付与 |
|情報取得 |                                  |        |
+--------+                                  +--------+
```

### 6.2 IRSA の設定手順

```bash
# 1. OIDC プロバイダの作成 (クラスター作成時に1回)
eksctl utils associate-iam-oidc-provider \
  --cluster my-cluster \
  --approve

# 2. IAM ポリシーの作成
aws iam create-policy \
  --policy-name my-app-s3-policy \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::my-bucket/*"
    }]
  }'

# 3. ServiceAccount と IAM ロールの紐付け
eksctl create iamserviceaccount \
  --cluster my-cluster \
  --namespace production \
  --name my-app-sa \
  --attach-policy-arn arn:aws:iam::123456789012:policy/my-app-s3-policy \
  --approve

# 4. Pod で ServiceAccount を指定
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  namespace: production
spec:
  serviceAccountName: my-app-sa
  containers:
    - name: app
      image: 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:v1.0
EOF
```

### 6.3 IRSA の IAM ロール信頼ポリシー

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::123456789012:oidc-provider/oidc.eks.ap-northeast-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B71EXAMPLE"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.ap-northeast-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B71EXAMPLE:sub": "system:serviceaccount:production:my-app-sa",
          "oidc.eks.ap-northeast-1.amazonaws.com/id/EXAMPLED539D4633E53DE1B71EXAMPLE:aud": "sts.amazonaws.com"
        }
      }
    }
  ]
}
```

---

## 7. アンチパターン

### 7.1 ノードの IAM ロールに広い権限を付与

```
[悪い例]
ノードロール --> AdministratorAccess
  → 全Podが管理者権限を持つ

[良い例]
ノードロール --> 最小限 (EKS Worker Policy, CNI Policy, ECR ReadOnly)
Pod レベル --> IRSA で個別に必要な権限を付与
  → 各Podが必要最小限の権限のみ持つ
```

**問題点**: ノードの IAM ロールに広い権限を付与すると、そのノード上の全 Pod がその権限を利用できてしまう。

**改善**: IRSA を使って Pod/ServiceAccount 単位で最小権限を付与する。

### 7.2 EKS アドオンを手動管理

**問題点**: CoreDNS、kube-proxy、VPC CNI などのクリティカルコンポーネントを手動でインストール・更新すると、バージョンの不整合やセキュリティパッチの遅れが発生する。

**改善**: EKS マネージドアドオンを使用し、AWS が推奨するバージョンの自動更新を活用する。

---

## 8. FAQ

### Q1. EKS と ECS のどちらを選ぶべきですか？

Kubernetes の経験がある、マルチクラウド/ハイブリッドクラウド戦略がある、CNCF エコシステム(Istio, ArgoCD 等)を活用したい場合は EKS が適している。AWS 中心のシンプルなコンテナワークロード、小規模チーム、運用負荷を最小限にしたい場合は ECS が適している。

### Q2. EKS のバージョンアップグレードはどう行うべきですか？

EKS は Kubernetes のバージョンサポート期間が限られている(約14ヶ月)。コントロールプレーンのアップグレードは `aws eks update-cluster-version` で実行し、その後マネージドノードグループの更新を行う。アドオンの互換性確認、アプリケーションの互換性テストを事前に実施し、ステージング環境で検証してから本番に適用する。

### Q3. EKS の費用はどのくらいですか？

コントロールプレーン料金は $0.10/時($73/月)。これに加えてデータプレーン(EC2 インスタンスまたは Fargate)の料金がかかる。EKS 自体よりもデータプレーンのコストが大きくなるため、ノードの適正サイズ選定やスポットインスタンス活用がコスト最適化の鍵となる。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| EKS アーキテクチャ | コントロールプレーン (AWS管理) + データプレーン (ユーザー管理) |
| ノードグループ | マネージド(推奨)、セルフマネージド、Fargate から選択 |
| Fargate プロファイル | namespace + labels でマッチする Pod をサーバーレス実行 |
| Helm | Kubernetes のパッケージマネージャ。Chart でアプリケーションを管理 |
| IRSA | Pod 単位で IAM ロールを付与。最小権限の実現に必須 |
| コスト | コントロールプレーン $73/月 + データプレーン料金 |

---

## 次に読むべきガイド

- [ECS 基礎](./00-ecs-basics.md) -- ECS との比較検討に
- [ECR](./01-ecr.md) -- コンテナイメージの管理
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- IRSA の IAM 設計を深める

---

## 参考文献

1. AWS 公式ドキュメント「Amazon EKS ユーザーガイド」 https://docs.aws.amazon.com/eks/latest/userguide/
2. AWS 公式「Amazon EKS ベストプラクティスガイド」 https://aws.github.io/aws-eks-best-practices/
3. eksctl 公式ドキュメント https://eksctl.io/
4. Helm 公式ドキュメント https://helm.sh/docs/
