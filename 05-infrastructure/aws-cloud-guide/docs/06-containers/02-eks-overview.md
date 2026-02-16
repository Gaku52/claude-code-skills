# Amazon EKS 概要

> Amazon Elastic Kubernetes Service (EKS) のクラスター作成、ノードグループ、Fargate プロファイル、Helm、IRSA (IAM Roles for Service Accounts) までを体系的に学ぶ。EKS アドオン管理、Cluster Autoscaler / Karpenter、ネットワークポリシー、可観測性、セキュリティ、GitOps まで含めた実践的な運用知識を網羅する。

---

## この章で学ぶこと

1. **EKS クラスターの構成と作成** -- コントロールプレーンとデータプレーンの関係、eksctl によるクラスター構築を理解する
2. **ノードグループと Fargate プロファイル** -- マネージドノードグループ、セルフマネージドノード、Fargate の使い分けを習得する
3. **Helm と IRSA の活用** -- パッケージマネージャによるアプリケーションデプロイと、Pod レベルの IAM 権限制御を身につける
4. **EKS アドオンとオートスケーリング** -- マネージドアドオン、Cluster Autoscaler、Karpenter によるクラスター運用の自動化を学ぶ
5. **セキュリティとネットワーク** -- Pod Security Standards、ネットワークポリシー、Secrets 管理の実践を理解する
6. **可観測性と GitOps** -- Container Insights、Prometheus、ArgoCD/Flux を用いた運用パターンを習得する

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

### 1.3 EKS のネットワークアーキテクチャ

```
EKS ネットワーク構成:

+------------------------------------------------------------------+
|  VPC (10.0.0.0/16)                                                |
|                                                                  |
|  パブリックサブネット                                               |
|  +---------------------+  +---------------------+                 |
|  | 10.0.1.0/24 (AZ-a)  |  | 10.0.2.0/24 (AZ-c)  |                |
|  | +------+  +------+  |  | +------+  +------+  |                |
|  | | NLB  |  | NAT  |  |  | | NLB  |  | NAT  |  |                |
|  | | (K8s |  | GW   |  |  | | (K8s |  | GW   |  |                |
|  | | Svc) |  |      |  |  | | Svc) |  |      |  |                |
|  | +------+  +------+  |  | +------+  +------+  |                |
|  +---------------------+  +---------------------+                 |
|                                                                  |
|  プライベートサブネット                                              |
|  +---------------------+  +---------------------+                 |
|  | 10.0.10.0/24 (AZ-a) |  | 10.0.20.0/24 (AZ-c) |                |
|  | +------+ +------+   |  | +------+ +------+   |                |
|  | | Node | | Node |   |  | | Node | | Node |   |                |
|  | | +--+ | | +--+ |   |  | | +--+ | | +--+ |   |                |
|  | | |P | | | |P | |   |  | | |P | | | |P | |   |                |
|  | | +--+ | | +--+ |   |  | | +--+ | | +--+ |   |                |
|  | +------+ +------+   |  | +------+ +------+   |                |
|  +---------------------+  +---------------------+                 |
|                                                                  |
|  ENI (コントロールプレーン通信用)                                    |
|  +---------------------+  +---------------------+                 |
|  | 10.0.100.0/24       |  | 10.0.200.0/24       |                |
|  +---------------------+  +---------------------+                 |
+------------------------------------------------------------------+
```

### 1.4 EKS API エンドポイントのアクセス制御

```bash
# パブリック + プライベートエンドポイント (推奨)
aws eks update-cluster-config \
  --name my-cluster \
  --resources-vpc-config \
    endpointPublicAccess=true,\
    endpointPrivateAccess=true,\
    publicAccessCidrs='["203.0.113.0/24","198.51.100.0/24"]'

# プライベートエンドポイントのみ (最もセキュア)
aws eks update-cluster-config \
  --name my-cluster \
  --resources-vpc-config \
    endpointPublicAccess=false,\
    endpointPrivateAccess=true
```

```
API エンドポイントの選択:

パブリックのみ:
  kubectl → インターネット → EKS API
  ⚠ セキュリティリスク

パブリック + プライベート (推奨):
  kubectl (社外) → インターネット → EKS API (CIDR制限)
  kubectl (VPC内) → ENI → EKS API (プライベート)

プライベートのみ:
  kubectl → VPN/DirectConnect → VPC → ENI → EKS API
  ✓ 最もセキュア / VPN必須
```

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

### 2.3 eksctl 高度な設定例

```yaml
# production-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: production-cluster
  region: ap-northeast-1
  version: "1.29"
  tags:
    Environment: production
    Team: platform

# KMS によるシークレット暗号化
secretsEncryption:
  keyARN: arn:aws:kms:ap-northeast-1:123456789012:key/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

# VPC 設定
vpc:
  cidr: "10.0.0.0/16"
  nat:
    gateway: HighlyAvailable
  clusterEndpoints:
    publicAccess: true
    privateAccess: true
  publicAccessCIDRs:
    - "203.0.113.0/24"

# IAM OIDC プロバイダ (IRSA用)
iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: aws-load-balancer-controller
        namespace: kube-system
      wellKnownPolicies:
        awsLoadBalancerController: true
    - metadata:
        name: external-dns
        namespace: kube-system
      wellKnownPolicies:
        externalDNS: true
    - metadata:
        name: cluster-autoscaler
        namespace: kube-system
      wellKnownPolicies:
        autoScaler: true

# マネージドアドオン
addons:
  - name: vpc-cni
    version: latest
    configurationValues: '{"env":{"ENABLE_PREFIX_DELEGATION":"true"}}'
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest
    serviceAccountRoleARN: arn:aws:iam::123456789012:role/ebs-csi-role

# ノードグループ
managedNodeGroups:
  - name: system
    instanceType: m5.large
    desiredCapacity: 3
    minSize: 3
    maxSize: 6
    volumeSize: 100
    volumeType: gp3
    volumeEncrypted: true
    labels:
      role: system
    taints:
      - key: CriticalAddonsOnly
        value: "true"
        effect: PreferNoSchedule
    privateNetworking: true
    iam:
      withAddonPolicies:
        albIngress: true
        cloudWatch: true
        ebs: true

  - name: app-on-demand
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 20
    volumeSize: 100
    volumeType: gp3
    volumeEncrypted: true
    labels:
      role: application
      lifecycle: on-demand
    privateNetworking: true
    ssh:
      allow: false

  - name: app-spot
    instanceTypes:
      - m5.xlarge
      - m5a.xlarge
      - m5d.xlarge
      - m4.xlarge
    spot: true
    desiredCapacity: 5
    minSize: 0
    maxSize: 30
    volumeSize: 100
    labels:
      role: application
      lifecycle: spot
    taints:
      - key: spot
        value: "true"
        effect: NoSchedule
    privateNetworking: true

# ログ設定
cloudWatch:
  clusterLogging:
    enableTypes:
      - api
      - audit
      - authenticator
      - controllerManager
      - scheduler
    logRetentionInDays: 90
```

### 2.4 クラスターのバージョンアップグレード

```bash
# 現在のバージョン確認
aws eks describe-cluster \
  --name my-cluster \
  --query 'cluster.version'

# コントロールプレーンのアップグレード
aws eks update-cluster-version \
  --name my-cluster \
  --kubernetes-version 1.30

# アップグレード状況の確認
aws eks describe-update \
  --name my-cluster \
  --update-id <update-id>

# アドオンの互換性確認
aws eks describe-addon-versions \
  --kubernetes-version 1.30 \
  --addon-name vpc-cni

# アドオンのアップグレード
aws eks update-addon \
  --cluster-name my-cluster \
  --addon-name vpc-cni \
  --addon-version v1.16.0-eksbuild.1 \
  --resolve-conflicts OVERWRITE

# マネージドノードグループのアップグレード
aws eks update-nodegroup-version \
  --cluster-name my-cluster \
  --nodegroup-name general-ng \
  --kubernetes-version 1.30

# ノードグループのアップグレード状況
aws eks describe-nodegroup \
  --cluster-name my-cluster \
  --nodegroup-name general-ng \
  --query 'nodegroup.{version:version,status:status}'
```

```
EKS バージョンアップグレード手順:

1. リリースノート確認
   ↓  非推奨 API、Breaking Changes の確認
2. ステージング環境でテスト
   ↓  アプリケーション互換性テスト
3. アドオン互換性確認
   ↓  vpc-cni, coredns, kube-proxy 対応バージョン
4. コントロールプレーンアップグレード
   ↓  約 20-30 分 (ダウンタイムなし)
5. アドオンアップグレード
   ↓  各アドオンを順次更新
6. ノードグループアップグレード
   ↓  ローリングアップデート (Pod のドレイン → 新ノード起動)
7. アプリケーション動作確認

⚠ 1バージョンずつ上げること (1.28 → 1.29 → 1.30)
⚠ コントロールプレーンとデータプレーンは2マイナーバージョン差まで
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

### 3.3 カスタム Launch Template の活用

```bash
# Launch Template を使ったノードグループ作成
# カスタム AMI、UserData、セキュリティグループなどを細かく制御可能

aws ec2 create-launch-template \
  --launch-template-name eks-custom-lt \
  --launch-template-data '{
    "BlockDeviceMappings": [
      {
        "DeviceName": "/dev/xvda",
        "Ebs": {
          "VolumeSize": 100,
          "VolumeType": "gp3",
          "Iops": 3000,
          "Throughput": 125,
          "Encrypted": true,
          "KmsKeyId": "arn:aws:kms:ap-northeast-1:123456789012:key/xxx"
        }
      }
    ],
    "MetadataOptions": {
      "HttpTokens": "required",
      "HttpPutResponseHopLimit": 2,
      "HttpEndpoint": "enabled"
    },
    "TagSpecifications": [
      {
        "ResourceType": "instance",
        "Tags": [
          {"Key": "Environment", "Value": "production"}
        ]
      }
    ]
  }'

# Launch Template を使用してノードグループ作成
aws eks create-nodegroup \
  --cluster-name my-cluster \
  --nodegroup-name custom-ng \
  --node-role arn:aws:iam::123456789012:role/eksNodeRole \
  --subnets subnet-111 subnet-222 \
  --launch-template name=eks-custom-lt,version=1 \
  --scaling-config minSize=2,maxSize=10,desiredSize=3
```

### 3.4 Bottlerocket ノード

```yaml
# eksctl での Bottlerocket ノードグループ
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: my-cluster
  region: ap-northeast-1

managedNodeGroups:
  - name: bottlerocket-ng
    amiFamily: Bottlerocket
    instanceType: m5.large
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 50
    bottlerocket:
      settings:
        kubernetes:
          maxPods: 58
        host-containers:
          admin:
            enabled: true
          control:
            enabled: true
```

```
Bottlerocket vs Amazon Linux 2:

Bottlerocket:
  ✓ コンテナ実行に特化した軽量 OS
  ✓ 不変インフラ (immutable)
  ✓ 自動セキュリティアップデート
  ✓ SSH 不要 (SSM 経由の admin container)
  ✓ 攻撃対象面が小さい
  ✗ 汎用的なパッケージ追加は不可
  ✗ 一部のカスタム設定に制限

Amazon Linux 2:
  ✓ 汎用的で柔軟
  ✓ yum でパッケージ追加可能
  ✓ 既存の運用ツールが使える
  ✗ OS パッチ管理が必要
  ✗ セキュリティ管理の負担が大きい
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

### 4.3 Fargate の制限事項と対策

```
Fargate の制限事項:

1. DaemonSet が使えない
   → サイドカーコンテナで代替
   → Fluent Bit サイドカーでログ転送

2. HostPort / HostNetwork が使えない
   → Service (LoadBalancer/ClusterIP) で対応

3. EBS ボリュームが使えない
   → EFS (Elastic File System) を使用
   → emptyDir は一時的に利用可能

4. GPU が使えない
   → GPU ワークロードは EC2 ノードへ

5. Privileged Container が使えない
   → セキュリティコンテキストの制限

6. 起動が遅い (30秒〜2分)
   → レイテンシに敏感なワークロードは EC2 ノードへ
```

```yaml
# Fargate での Fluent Bit サイドカー例
apiVersion: v1
kind: Pod
metadata:
  name: app-with-logging
  namespace: batch-jobs
  labels:
    compute: fargate
spec:
  containers:
    - name: app
      image: 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:v1.0
      volumeMounts:
        - name: log-volume
          mountPath: /var/log/app

    - name: fluent-bit
      image: public.ecr.aws/aws-observability/aws-for-fluent-bit:stable
      env:
        - name: AWS_REGION
          value: ap-northeast-1
      volumeMounts:
        - name: log-volume
          mountPath: /var/log/app
          readOnly: true
        - name: fluent-bit-config
          mountPath: /fluent-bit/etc/

  volumes:
    - name: log-volume
      emptyDir: {}
    - name: fluent-bit-config
      configMap:
        name: fluent-bit-config
```

### 4.4 Fargate でのロギング (FireLens)

```yaml
# Fargate Pod のログを CloudWatch に送信する ConfigMap
apiVersion: v1
kind: ConfigMap
metadata:
  name: aws-logging
  namespace: aws-observability
data:
  output.conf: |
    [OUTPUT]
        Name  cloudwatch_logs
        Match *
        region ap-northeast-1
        log_group_name /eks/fargate/my-cluster
        log_stream_prefix fargate-
        auto_create_group true
        log_retention_days 30

  parsers.conf: |
    [PARSER]
        Name json
        Format json
        Time_Key time
        Time_Format %Y-%m-%dT%H:%M:%S.%LZ

  filters.conf: |
    [FILTER]
        Name parser
        Match *
        Key_Name log
        Parser json
        Reserve_Data On
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

### 5.4 Helm Chart の作成

```bash
# 新しい Chart のスキャフォールド
helm create my-web-app

# 生成される構造
# my-web-app/
# ├── Chart.yaml
# ├── values.yaml
# ├── charts/
# ├── templates/
# │   ├── NOTES.txt
# │   ├── _helpers.tpl
# │   ├── deployment.yaml
# │   ├── hpa.yaml
# │   ├── ingress.yaml
# │   ├── service.yaml
# │   ├── serviceaccount.yaml
# │   └── tests/
# │       └── test-connection.yaml
# └── .helmignore
```

```yaml
# Chart.yaml
apiVersion: v2
name: my-web-app
description: A Helm chart for my web application
type: application
version: 0.1.0
appVersion: "1.0.0"
dependencies:
  - name: redis
    version: "17.x.x"
    repository: https://charts.bitnami.com/bitnami
    condition: redis.enabled
```

```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "my-web-app.fullname" . }}
  labels:
    {{- include "my-web-app.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "my-web-app.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      annotations:
        checksum/config: {{ include (print $.Template.BasePath "/configmap.yaml") . | sha256sum }}
      labels:
        {{- include "my-web-app.selectorLabels" . | nindent 8 }}
    spec:
      serviceAccountName: {{ include "my-web-app.serviceAccountName" . }}
      securityContext:
        runAsNonRoot: true
        fsGroup: 1000
      containers:
        - name: {{ .Chart.Name }}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          ports:
            - name: http
              containerPort: {{ .Values.containerPort | default 8080 }}
              protocol: TCP
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
            initialDelaySeconds: 15
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
          envFrom:
            - configMapRef:
                name: {{ include "my-web-app.fullname" . }}-config
          volumeMounts:
            - name: tmp
              mountPath: /tmp
      volumes:
        - name: tmp
          emptyDir: {}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

```bash
# Chart のテスト
helm template my-app ./my-web-app --values production-values.yaml

# Lint チェック
helm lint ./my-web-app

# Chart パッケージ
helm package ./my-web-app

# OCI レジストリ (ECR) への Chart プッシュ
aws ecr create-repository --repository-name helm-charts/my-web-app
helm push my-web-app-0.1.0.tgz oci://123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/helm-charts
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

### 6.4 EKS Pod Identity (IRSA の後継)

```bash
# EKS Pod Identity Association の作成 (IRSA より簡単)
# OIDC プロバイダの設定が不要

# 1. Pod Identity Agent アドオンのインストール
aws eks create-addon \
  --cluster-name my-cluster \
  --addon-name eks-pod-identity-agent

# 2. IAM ロールの信頼ポリシー (Pod Identity 用)
aws iam create-role \
  --role-name my-app-pod-identity-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {
        "Service": "pods.eks.amazonaws.com"
      },
      "Action": [
        "sts:AssumeRole",
        "sts:TagSession"
      ]
    }]
  }'

# 3. Pod Identity Association の作成
aws eks create-pod-identity-association \
  --cluster-name my-cluster \
  --namespace production \
  --service-account my-app-sa \
  --role-arn arn:aws:iam::123456789012:role/my-app-pod-identity-role

# 4. ServiceAccount の作成 (アノテーション不要！)
kubectl create serviceaccount my-app-sa -n production
```

```
IRSA vs EKS Pod Identity:

IRSA:
  ✓ 成熟した仕組み、広くドキュメント化
  ✗ クラスターごとに OIDC プロバイダ設定が必要
  ✗ IAM ロール信頼ポリシーにクラスター固有の OIDC URL が必要
  ✗ クラスター間の移行が煩雑

EKS Pod Identity (推奨):
  ✓ OIDC プロバイダ不要
  ✓ IAM ロールの信頼ポリシーがシンプル
  ✓ クラスター間の移行が容易
  ✓ ServiceAccount にアノテーション不要
  ✗ 比較的新しい機能 (2023年11月〜)
  ✗ Fargate 非対応 (IRSA を使用)
```

---

## 7. EKS マネージドアドオン

### 7.1 主要なマネージドアドオン

```bash
# 利用可能なアドオン一覧
aws eks describe-addon-versions \
  --kubernetes-version 1.29 \
  --query 'addons[].{name:addonName,versions:addonVersions[0].addonVersion}' \
  --output table

# アドオンのインストール
aws eks create-addon \
  --cluster-name my-cluster \
  --addon-name vpc-cni \
  --addon-version v1.16.0-eksbuild.1 \
  --service-account-role-arn arn:aws:iam::123456789012:role/vpc-cni-role

# アドオンの状態確認
aws eks describe-addon \
  --cluster-name my-cluster \
  --addon-name vpc-cni

# アドオンの更新
aws eks update-addon \
  --cluster-name my-cluster \
  --addon-name vpc-cni \
  --addon-version v1.17.0-eksbuild.1 \
  --resolve-conflicts OVERWRITE
```

```
主要アドオン一覧:

必須アドオン:
  vpc-cni          Pod のネットワーキング (ENI ベース)
  coredns          クラスター内 DNS
  kube-proxy       ネットワークプロキシ

ストレージ:
  aws-ebs-csi-driver    EBS ボリューム
  aws-efs-csi-driver    EFS ボリューム

セキュリティ:
  eks-pod-identity-agent  Pod Identity
  aws-guardduty-agent     脅威検知

ネットワーク:
  aws-load-balancer-controller  ALB/NLB 管理 (※Helmで導入)

可観測性:
  amazon-cloudwatch-observability  Container Insights
  adot                             AWS Distro for OpenTelemetry
```

### 7.2 VPC CNI の高度な設定

```bash
# Prefix Delegation の有効化 (Pod 密度の向上)
# 通常: ENI ごとに IP を 1 つずつ割り当て
# Prefix: ENI ごとに /28 プレフィックス (16 IP) を割り当て
kubectl set env daemonset aws-node \
  -n kube-system \
  ENABLE_PREFIX_DELEGATION=true \
  WARM_PREFIX_TARGET=1

# カスタムネットワーキング (Pod を別サブネットで実行)
kubectl set env daemonset aws-node \
  -n kube-system \
  AWS_VPC_K8S_CNI_CUSTOM_NETWORK_CFG=true

# ENIConfig リソースの作成
kubectl apply -f - <<EOF
apiVersion: crd.k8s.amazonaws.com/v1alpha1
kind: ENIConfig
metadata:
  name: ap-northeast-1a
spec:
  subnet: subnet-pod-az-a
  securityGroups:
    - sg-pod-sg
EOF
```

```
VPC CNI の IP アドレス管理:

通常モード (Secondary IP):
  m5.large: ENI 3個 × IP 10個 = 最大 29 Pod

Prefix Delegation モード:
  m5.large: ENI 3個 × /28 プレフィックス = 最大 110 Pod
  → Pod 密度が約 4 倍に向上

カスタムネットワーキング:
  ノード: 10.0.0.0/16 (VPC CIDR)
  Pod:  100.64.0.0/16 (別 CIDR)
  → VPC の IP アドレス枯渇を回避
```

---

## 8. オートスケーリング

### 8.1 Cluster Autoscaler

```yaml
# Cluster Autoscaler のデプロイ (Helm)
# values.yaml
autoDiscovery:
  clusterName: my-cluster

awsRegion: ap-northeast-1

rbac:
  serviceAccount:
    create: true
    annotations:
      eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/cluster-autoscaler-role

extraArgs:
  balance-similar-node-groups: true
  skip-nodes-with-system-pods: false
  expander: least-waste
  scale-down-delay-after-add: 10m
  scale-down-unneeded-time: 10m
  max-graceful-termination-sec: 600
```

```bash
# Helm でインストール
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --values ca-values.yaml

# 動作確認
kubectl logs -f deployment/cluster-autoscaler -n kube-system
```

### 8.2 Karpenter (推奨)

```bash
# Karpenter のインストール
export KARPENTER_VERSION="v0.33.0"
export CLUSTER_NAME="my-cluster"
export AWS_ACCOUNT_ID="123456789012"
export TEMPOUT=$(mktemp)

helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter \
  --version "${KARPENTER_VERSION}" \
  --namespace karpenter \
  --create-namespace \
  --set "settings.clusterName=${CLUSTER_NAME}" \
  --set "settings.interruptionQueue=${CLUSTER_NAME}" \
  --set controller.resources.requests.cpu=1 \
  --set controller.resources.requests.memory=1Gi \
  --set controller.resources.limits.cpu=1 \
  --set controller.resources.limits.memory=1Gi \
  --wait
```

```yaml
# NodePool 定義 (旧 Provisioner)
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: general-purpose
spec:
  template:
    metadata:
      labels:
        role: application
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand", "spot"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["m", "c", "r"]
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values: ["4"]
        - key: karpenter.k8s.aws/instance-size
          operator: In
          values: ["large", "xlarge", "2xlarge"]
      nodeClassRef:
        name: default
  limits:
    cpu: "100"
    memory: 400Gi
  disruption:
    consolidationPolicy: WhenUnderutilized
    expireAfter: 720h  # 30日でノード更新
---
# EC2NodeClass
apiVersion: karpenter.k8s.aws/v1beta1
kind: EC2NodeClass
metadata:
  name: default
spec:
  amiFamily: AL2
  role: KarpenterNodeRole-my-cluster
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: my-cluster
        kubernetes.io/role/internal-elb: "1"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: my-cluster
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 100Gi
        volumeType: gp3
        iops: 3000
        throughput: 125
        encrypted: true
  metadataOptions:
    httpEndpoint: enabled
    httpProtocolIPv6: disabled
    httpPutResponseHopLimit: 2
    httpTokens: required
  tags:
    Environment: production
```

```
Cluster Autoscaler vs Karpenter:

Cluster Autoscaler:
  ✓ 安定した成熟プロジェクト
  ✓ 設定がシンプル
  ✗ ASG ベース → インスタンスタイプが固定
  ✗ スケールアウトが遅い (1-3分)
  ✗ ビンパッキングが非効率

Karpenter (推奨):
  ✓ ASG 不要 → 直接 EC2 API を呼び出し
  ✓ スケールアウトが速い (数秒〜)
  ✓ Pod の要件に最適なインスタンスタイプを自動選択
  ✓ 効率的なビンパッキングとコンソリデーション
  ✓ Spot の中断を自動ハンドリング
  ✗ EKS 固有 (他の K8s ディストリビューションには未対応)
  ✗ 比較的新しいプロジェクト
```

### 8.3 Horizontal Pod Autoscaler (HPA)

```yaml
# HPA の定義
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    # カスタムメトリクス (Prometheus Adapter 経由)
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 100
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### 8.4 KEDA (Kubernetes Event-Driven Autoscaling)

```yaml
# KEDA ScaledObject (SQS キューベースのスケーリング)
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sqs-consumer-scaler
  namespace: production
spec:
  scaleTargetRef:
    name: sqs-consumer
  pollingInterval: 15
  cooldownPeriod: 60
  minReplicaCount: 1
  maxReplicaCount: 50
  triggers:
    - type: aws-sqs-queue
      authenticationRef:
        name: keda-aws-credentials
      metadata:
        queueURL: https://sqs.ap-northeast-1.amazonaws.com/123456789012/my-queue
        queueLength: "5"    # メッセージ5件につき1 Pod
        awsRegion: ap-northeast-1
        identityOwner: operator
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: keda-aws-credentials
  namespace: production
spec:
  podIdentity:
    provider: aws-eks
```

---

## 9. ネットワークとセキュリティ

### 9.1 AWS Load Balancer Controller

```bash
# AWS Load Balancer Controller のインストール
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  --namespace kube-system \
  --set clusterName=my-cluster \
  --set serviceAccount.create=true \
  --set serviceAccount.annotations."eks\.amazonaws\.com/role-arn"=arn:aws:iam::123456789012:role/aws-lbc-role
```

```yaml
# Ingress リソース (ALB)
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-app-ingress
  namespace: production
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:ap-northeast-1:123456789012:certificate/xxx
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTPS":443}]'
    alb.ingress.kubernetes.io/ssl-redirect: "443"
    alb.ingress.kubernetes.io/healthcheck-path: /healthz
    alb.ingress.kubernetes.io/healthcheck-interval-seconds: "15"
    alb.ingress.kubernetes.io/healthy-threshold-count: "2"
    alb.ingress.kubernetes.io/unhealthy-threshold-count: "3"
    alb.ingress.kubernetes.io/wafv2-acl-arn: arn:aws:wafv2:ap-northeast-1:123456789012:regional/webacl/my-acl/xxx
    alb.ingress.kubernetes.io/shield-advanced-protection: "true"
    alb.ingress.kubernetes.io/group.name: my-app
spec:
  rules:
    - host: api.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: my-app-svc
                port:
                  number: 80
          - path: /admin
            pathType: Prefix
            backend:
              service:
                name: admin-svc
                port:
                  number: 80
```

### 9.2 ネットワークポリシー

```bash
# Calico のインストール (ネットワークポリシーエンジン)
helm repo add projectcalico https://docs.projectcalico.org/charts
helm install calico projectcalico/tigera-operator \
  --namespace tigera-operator \
  --create-namespace
```

```yaml
# デフォルト拒否ポリシー (namespace 内の全通信を拒否)
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny-all
  namespace: production
spec:
  podSelector: {}
  policyTypes:
    - Ingress
    - Egress
---
# フロントエンド → バックエンド通信の許可
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-frontend-to-backend
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Ingress
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: frontend
      ports:
        - protocol: TCP
          port: 8080
---
# バックエンド → データベース通信の許可
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-backend-to-db
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: backend
  policyTypes:
    - Egress
  egress:
    - to:
        - podSelector:
            matchLabels:
              app: database
      ports:
        - protocol: TCP
          port: 5432
    # DNS 解決を許可
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
```

### 9.3 Pod Security Standards

```yaml
# Pod Security Admission (PSA) の設定
# namespace レベルでセキュリティ基準を適用

# Restricted レベル (最も厳格)
apiVersion: v1
kind: Namespace
metadata:
  name: production
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/enforce-version: latest
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
# Baseline レベル (開発環境向け)
apiVersion: v1
kind: Namespace
metadata:
  name: development
  labels:
    pod-security.kubernetes.io/enforce: baseline
    pod-security.kubernetes.io/warn: restricted
```

```yaml
# Restricted レベルに準拠した Pod 定義
apiVersion: v1
kind: Pod
metadata:
  name: secure-app
  namespace: production
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
  containers:
    - name: app
      image: 123456789012.dkr.ecr.ap-northeast-1.amazonaws.com/my-app:v1.0
      securityContext:
        allowPrivilegeEscalation: false
        readOnlyRootFilesystem: true
        capabilities:
          drop: ["ALL"]
      volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /var/cache
  volumes:
    - name: tmp
      emptyDir: {}
    - name: cache
      emptyDir: {}
```

### 9.4 Secrets 管理

```bash
# AWS Secrets Manager CSI Driver のインストール
helm repo add secrets-store-csi-driver \
  https://kubernetes-sigs.github.io/secrets-store-csi-driver/charts
helm install csi-secrets-store secrets-store-csi-driver/secrets-store-csi-driver \
  --namespace kube-system \
  --set syncSecret.enabled=true

# AWS Provider のインストール
kubectl apply -f https://raw.githubusercontent.com/aws/secrets-store-csi-driver-provider-aws/main/deployment/aws-provider-installer.yaml
```

```yaml
# SecretProviderClass
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: aws-secrets
  namespace: production
spec:
  provider: aws
  parameters:
    objects: |
      - objectName: "prod/my-app/database"
        objectType: "secretsmanager"
        jmesPath:
          - path: username
            objectAlias: db-username
          - path: password
            objectAlias: db-password
      - objectName: "/prod/my-app/api-key"
        objectType: "ssmparameter"
  secretObjects:
    - secretName: db-credentials
      type: Opaque
      data:
        - objectName: db-username
          key: username
        - objectName: db-password
          key: password
---
# Pod で Secret をマウント
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  namespace: production
spec:
  serviceAccountName: my-app-sa
  containers:
    - name: app
      image: my-app:v1.0
      env:
        - name: DB_USERNAME
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: username
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
      volumeMounts:
        - name: secrets
          mountPath: /mnt/secrets
          readOnly: true
  volumes:
    - name: secrets
      csi:
        driver: secrets-store.csi.k8s.io
        readOnly: true
        volumeAttributes:
          secretProviderClass: aws-secrets
```

---

## 10. 可観測性 (Observability)

### 10.1 Amazon CloudWatch Container Insights

```bash
# Container Insights アドオンのインストール
aws eks create-addon \
  --cluster-name my-cluster \
  --addon-name amazon-cloudwatch-observability \
  --addon-version v1.5.0-eksbuild.1 \
  --service-account-role-arn arn:aws:iam::123456789012:role/cloudwatch-agent-role
```

```yaml
# CloudWatch Agent の高度な設定
apiVersion: v1
kind: ConfigMap
metadata:
  name: cloudwatch-agent-config
  namespace: amazon-cloudwatch
data:
  cwagentconfig.json: |
    {
      "logs": {
        "metrics_collected": {
          "kubernetes": {
            "cluster_name": "my-cluster",
            "metrics_collection_interval": 60
          },
          "app_signals": {
            "enabled": true
          }
        },
        "force_flush_interval": 5
      },
      "traces": {
        "traces_collected": {
          "xray": {
            "tcp_proxy": {
              "bind_address": "0.0.0.0:2000"
            }
          },
          "otlp": {
            "grpc_endpoint": "0.0.0.0:4317",
            "http_endpoint": "0.0.0.0:4318"
          }
        }
      },
      "metrics": {
        "metrics_collected": {
          "prometheus": {
            "cluster_name": "my-cluster",
            "log_group_name": "/aws/containerinsights/my-cluster/prometheus"
          }
        }
      }
    }
```

### 10.2 Prometheus + Grafana

```bash
# Amazon Managed Prometheus (AMP) ワークスペースの作成
aws amp create-workspace \
  --alias my-cluster-metrics \
  --tags Environment=production

# Prometheus の Helm インストール (AMP リモートライト)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --values prometheus-values.yaml
```

```yaml
# prometheus-values.yaml
prometheus:
  prometheusSpec:
    remoteWrite:
      - url: https://aps-workspaces.ap-northeast-1.amazonaws.com/workspaces/ws-xxxxxxxx/api/v1/remote_write
        sigv4:
          region: ap-northeast-1
        queueConfig:
          maxSamplesPerSend: 1000
          maxShards: 200
          capacity: 2500
    retention: 2h  # ローカル保持は短く
    resources:
      requests:
        cpu: 500m
        memory: 2Gi
      limits:
        cpu: "2"
        memory: 8Gi
    storageSpec:
      volumeClaimTemplate:
        spec:
          storageClassName: gp3
          resources:
            requests:
              storage: 50Gi
    serviceMonitorSelectorNilUsesHelmValues: false

grafana:
  enabled: true
  adminPassword: "changeme"
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
        - name: AMP
          type: prometheus
          url: https://aps-workspaces.ap-northeast-1.amazonaws.com/workspaces/ws-xxxxxxxx
          jsonData:
            sigV4Auth: true
            sigV4AuthType: default
            sigV4Region: ap-northeast-1

alertmanager:
  config:
    global:
      resolve_timeout: 5m
    route:
      receiver: slack
      group_by: ['alertname', 'namespace']
      group_wait: 30s
      group_interval: 5m
      repeat_interval: 4h
    receivers:
      - name: slack
        slack_configs:
          - api_url: 'https://hooks.slack.com/services/xxx/yyy/zzz'
            channel: '#alerts'
            title: '{{ .GroupLabels.alertname }}'
            text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

```yaml
# ServiceMonitor (アプリケーションメトリクスの収集)
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: my-app-monitor
  namespace: production
spec:
  selector:
    matchLabels:
      app: my-app
  endpoints:
    - port: metrics
      interval: 15s
      path: /metrics
  namespaceSelector:
    matchNames:
      - production
```

### 10.3 AWS Distro for OpenTelemetry (ADOT)

```yaml
# ADOT Collector の設定
apiVersion: opentelemetry.io/v1alpha1
kind: OpenTelemetryCollector
metadata:
  name: adot-collector
  namespace: monitoring
spec:
  mode: deployment
  serviceAccount: adot-collector
  config: |
    receivers:
      otlp:
        protocols:
          grpc:
            endpoint: 0.0.0.0:4317
          http:
            endpoint: 0.0.0.0:4318

    processors:
      batch:
        timeout: 30s
        send_batch_size: 8192
      memory_limiter:
        check_interval: 5s
        limit_mib: 1500
        spike_limit_mib: 512

    exporters:
      awsxray:
        region: ap-northeast-1
      awsemf:
        region: ap-northeast-1
        namespace: MyApp
        log_group_name: /aws/containerinsights/my-cluster/application
      prometheusremotewrite:
        endpoint: https://aps-workspaces.ap-northeast-1.amazonaws.com/workspaces/ws-xxx/api/v1/remote_write
        auth:
          authenticator: sigv4auth

    extensions:
      sigv4auth:
        region: ap-northeast-1
        service: aps

    service:
      extensions: [sigv4auth]
      pipelines:
        traces:
          receivers: [otlp]
          processors: [batch, memory_limiter]
          exporters: [awsxray]
        metrics:
          receivers: [otlp]
          processors: [batch, memory_limiter]
          exporters: [awsemf, prometheusremotewrite]
```

---

## 11. GitOps

### 11.1 ArgoCD

```bash
# ArgoCD のインストール
helm repo add argo https://argoproj.github.io/argo-helm
helm install argocd argo/argo-cd \
  --namespace argocd \
  --create-namespace \
  --set server.service.type=LoadBalancer \
  --set server.extraArgs="{--insecure}" \
  --set configs.params."server\.insecure"=true
```

```yaml
# ArgoCD Application 定義
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: my-app
  namespace: argocd
  finalizers:
    - resources-finalizer.argocd.argoproj.io
spec:
  project: default
  source:
    repoURL: https://github.com/my-org/k8s-manifests.git
    targetRevision: main
    path: apps/my-app/overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
      allowEmpty: false
    syncOptions:
      - CreateNamespace=true
      - PrunePropagationPolicy=foreground
      - PruneLast=true
    retry:
      limit: 5
      backoff:
        duration: 5s
        factor: 2
        maxDuration: 3m
  ignoreDifferences:
    - group: apps
      kind: Deployment
      jsonPointers:
        - /spec/replicas  # HPA が管理するため無視
---
# ApplicationSet (複数環境の一括管理)
apiVersion: argoproj.io/v1alpha1
kind: ApplicationSet
metadata:
  name: my-app-set
  namespace: argocd
spec:
  generators:
    - list:
        elements:
          - cluster: dev
            url: https://kubernetes.default.svc
            namespace: dev
            values_file: dev
          - cluster: staging
            url: https://kubernetes.default.svc
            namespace: staging
            values_file: staging
          - cluster: production
            url: https://kubernetes.default.svc
            namespace: production
            values_file: production
  template:
    metadata:
      name: 'my-app-{{cluster}}'
    spec:
      project: default
      source:
        repoURL: https://github.com/my-org/k8s-manifests.git
        targetRevision: main
        path: apps/my-app
        helm:
          valueFiles:
            - 'values-{{values_file}}.yaml'
      destination:
        server: '{{url}}'
        namespace: '{{namespace}}'
```

### 11.2 Flux CD

```bash
# Flux CLI のインストール
curl -s https://fluxcd.io/install.sh | sudo bash

# Flux のブートストラップ
flux bootstrap github \
  --owner=my-org \
  --repository=fleet-infra \
  --branch=main \
  --path=clusters/production \
  --personal
```

```yaml
# GitRepository
apiVersion: source.toolkit.fluxcd.io/v1
kind: GitRepository
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 1m
  url: https://github.com/my-org/k8s-manifests
  ref:
    branch: main
  secretRef:
    name: github-token
---
# Kustomization
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: my-app
  namespace: flux-system
spec:
  interval: 5m
  path: ./apps/my-app/overlays/production
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
  retryInterval: 1m
```

```
ArgoCD vs Flux:

ArgoCD:
  ✓ リッチな Web UI
  ✓ マルチクラスター管理が容易
  ✓ ApplicationSet で大規模管理
  ✓ RBAC が充実
  ✗ リソース消費がやや大きい
  ✗ 学習コストがやや高い

Flux:
  ✓ 軽量・シンプル
  ✓ Git 中心の設計思想
  ✓ Helm Controller 内蔵
  ✓ CNCF Graduated プロジェクト
  ✗ Web UI が別途必要
  ✗ マルチクラスター管理がやや煩雑
```

---

## 12. コスト最適化

### 12.1 コスト構造の理解

```
EKS のコスト構成:

+--------------------------------------------+
| コントロールプレーン: $0.10/時 ($73/月)       |
+--------------------------------------------+
|                                            |
| データプレーン (コストの大部分):               |
| +----------------------------------------+ |
| | EC2 (On-Demand)    $$$$               | |
| | EC2 (Spot)         $$                  | |
| | EC2 (Reserved/SP)  $$$                | |
| | Fargate            $$$                | |
| +----------------------------------------+ |
|                                            |
| ネットワーク:                                |
| +----------------------------------------+ |
| | NAT Gateway        $$                  | |
| | ALB/NLB            $                   | |
| | データ転送          $                   | |
| +----------------------------------------+ |
|                                            |
| ストレージ:                                  |
| +----------------------------------------+ |
| | EBS (gp3)          $                   | |
| | EFS                $$                  | |
| +----------------------------------------+ |
|                                            |
| ログ・モニタリング:                           |
| +----------------------------------------+ |
| | CloudWatch Logs    $                   | |
| | Container Insights $                   | |
| +----------------------------------------+ |
+--------------------------------------------+
```

### 12.2 コスト最適化のベストプラクティス

```yaml
# Spot インスタンスの活用 (Karpenter)
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: spot-pool
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["m", "c", "r"]
        - key: karpenter.k8s.aws/instance-generation
          operator: Gt
          values: ["4"]
      nodeClassRef:
        name: default
  disruption:
    consolidationPolicy: WhenUnderutilized
  weight: 80  # Spot を優先
---
apiVersion: karpenter.sh/v1beta1
kind: NodePool
metadata:
  name: on-demand-pool
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: karpenter.k8s.aws/instance-category
          operator: In
          values: ["m"]
      nodeClassRef:
        name: default
  weight: 20  # フォールバック
```

```yaml
# リソースリクエスト/リミットの適正化
# VPA (Vertical Pod Autoscaler) で推奨値を取得
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: my-app-vpa
  namespace: production
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  updatePolicy:
    updateMode: "Off"  # まずは推奨値の確認のみ
  resourcePolicy:
    containerPolicies:
      - containerName: app
        minAllowed:
          cpu: 50m
          memory: 64Mi
        maxAllowed:
          cpu: "2"
          memory: 4Gi
```

```bash
# VPA の推奨値確認
kubectl describe vpa my-app-vpa -n production

# Kubecost のインストール (コスト可視化)
helm repo add kubecost https://kubecost.github.io/cost-analyzer/
helm install kubecost kubecost/cost-analyzer \
  --namespace kubecost \
  --create-namespace \
  --set kubecostToken="xxxxx"
```

---

## 13. アンチパターン

### 13.1 ノードの IAM ロールに広い権限を付与

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

**改善**: IRSA または EKS Pod Identity を使って Pod/ServiceAccount 単位で最小権限を付与する。

### 13.2 EKS アドオンを手動管理

**問題点**: CoreDNS、kube-proxy、VPC CNI などのクリティカルコンポーネントを手動でインストール・更新すると、バージョンの不整合やセキュリティパッチの遅れが発生する。

**改善**: EKS マネージドアドオンを使用し、AWS が推奨するバージョンの自動更新を活用する。

### 13.3 リソースリクエスト/リミット未設定

```
[悪い例]
containers:
  - name: app
    image: my-app:v1
    # resources 未設定
    # → ノードリソースを無制限に消費
    # → OOMKill や CPU スロットリングの原因

[良い例]
containers:
  - name: app
    image: my-app:v1
    resources:
      requests:
        cpu: 250m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
    # → スケジューラが適切にスケジュール
    # → HPA/VPA が正しく機能
```

### 13.4 PodDisruptionBudget 未設定

```yaml
# PDB を設定しないと、ノード更新時に全 Pod が同時停止する可能性

# 推奨: PDB の設定
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: my-app-pdb
  namespace: production
spec:
  minAvailable: 2
  # または maxUnavailable: 1
  selector:
    matchLabels:
      app: my-app
```

### 13.5 単一 AZ へのデプロイ

```yaml
# Pod Anti-Affinity で AZ 分散を強制
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
spec:
  replicas: 3
  template:
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: my-app
```

---

## 14. CloudFormation テンプレート

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'EKS クラスターの CloudFormation テンプレート'

Parameters:
  ClusterName:
    Type: String
    Default: my-cluster

  KubernetesVersion:
    Type: String
    Default: "1.29"

  NodeGroupInstanceType:
    Type: String
    Default: m5.large

  NodeGroupDesiredSize:
    Type: Number
    Default: 3

  NodeGroupMinSize:
    Type: Number
    Default: 2

  NodeGroupMaxSize:
    Type: Number
    Default: 10

Resources:
  # EKS クラスターロール
  EKSClusterRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${ClusterName}-cluster-role'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: eks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSClusterPolicy
        - arn:aws:iam::aws:policy/AmazonEKSVPCResourceController

  # ノードロール
  EKSNodeRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${ClusterName}-node-role'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: ec2.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        - arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        - arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        - arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore

  # EKS クラスター
  EKSCluster:
    Type: AWS::EKS::Cluster
    Properties:
      Name: !Ref ClusterName
      Version: !Ref KubernetesVersion
      RoleArn: !GetAtt EKSClusterRole.Arn
      ResourcesVpcConfig:
        SubnetIds:
          - !ImportValue 'network-PrivateSubnet1Id'
          - !ImportValue 'network-PrivateSubnet2Id'
          - !ImportValue 'network-PublicSubnet1Id'
          - !ImportValue 'network-PublicSubnet2Id'
        SecurityGroupIds:
          - !Ref ClusterSecurityGroup
        EndpointPublicAccess: true
        EndpointPrivateAccess: true
      Logging:
        ClusterLogging:
          EnabledTypes:
            - Type: api
            - Type: audit
            - Type: authenticator
            - Type: controllerManager
            - Type: scheduler

  # クラスターセキュリティグループ
  ClusterSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: EKS Cluster Security Group
      VpcId: !ImportValue 'network-VpcId'
      Tags:
        - Key: Name
          Value: !Sub '${ClusterName}-cluster-sg'

  # マネージドノードグループ
  EKSNodeGroup:
    Type: AWS::EKS::Nodegroup
    DependsOn: EKSCluster
    Properties:
      ClusterName: !Ref ClusterName
      NodegroupName: !Sub '${ClusterName}-general-ng'
      NodeRole: !GetAtt EKSNodeRole.Arn
      Subnets:
        - !ImportValue 'network-PrivateSubnet1Id'
        - !ImportValue 'network-PrivateSubnet2Id'
      InstanceTypes:
        - !Ref NodeGroupInstanceType
      ScalingConfig:
        DesiredSize: !Ref NodeGroupDesiredSize
        MinSize: !Ref NodeGroupMinSize
        MaxSize: !Ref NodeGroupMaxSize
      DiskSize: 100
      CapacityType: ON_DEMAND
      Labels:
        role: general
      Tags:
        Environment: production

  # OIDC プロバイダ (IRSA 用)
  OIDCProvider:
    Type: AWS::IAM::OIDCProvider
    DependsOn: EKSCluster
    Properties:
      Url: !GetAtt EKSCluster.OpenIdConnectIssuerUrl
      ClientIdList:
        - sts.amazonaws.com
      ThumbprintList:
        - "9e99a48a9960b14926bb7f3b02e22da2b0ab7280"

Outputs:
  ClusterName:
    Value: !Ref EKSCluster
    Export:
      Name: !Sub '${ClusterName}-ClusterName'

  ClusterEndpoint:
    Value: !GetAtt EKSCluster.Endpoint
    Export:
      Name: !Sub '${ClusterName}-ClusterEndpoint'

  ClusterArn:
    Value: !GetAtt EKSCluster.Arn

  OIDCProviderArn:
    Value: !GetAtt OIDCProvider.Arn
    Export:
      Name: !Sub '${ClusterName}-OIDCProviderArn'
```

---

## 15. FAQ

### Q1. EKS と ECS のどちらを選ぶべきですか？

Kubernetes の経験がある、マルチクラウド/ハイブリッドクラウド戦略がある、CNCF エコシステム(Istio, ArgoCD 等)を活用したい場合は EKS が適している。AWS 中心のシンプルなコンテナワークロード、小規模チーム、運用負荷を最小限にしたい場合は ECS が適している。

### Q2. EKS のバージョンアップグレードはどう行うべきですか？

EKS は Kubernetes のバージョンサポート期間が限られている(約14ヶ月)。コントロールプレーンのアップグレードは `aws eks update-cluster-version` で実行し、その後マネージドノードグループの更新を行う。アドオンの互換性確認、アプリケーションの互換性テストを事前に実施し、ステージング環境で検証してから本番に適用する。

### Q3. EKS の費用はどのくらいですか？

コントロールプレーン料金は $0.10/時($73/月)。これに加えてデータプレーン(EC2 インスタンスまたは Fargate)の料金がかかる。EKS 自体よりもデータプレーンのコストが大きくなるため、ノードの適正サイズ選定やスポットインスタンス活用がコスト最適化の鍵となる。

### Q4. Cluster Autoscaler と Karpenter のどちらを使うべきですか？

新規クラスターでは Karpenter を推奨する。Karpenter は ASG を使わず直接 EC2 API を呼び出すため、インスタンスタイプの柔軟な選択、高速なスケールアウト、効率的なビンパッキングが可能である。既存の ASG ベースの運用がある場合は Cluster Autoscaler を継続利用しつつ、段階的に Karpenter へ移行する戦略が現実的である。

### Q5. IRSA と EKS Pod Identity のどちらを使うべきですか？

新規セットアップでは EKS Pod Identity を推奨する。OIDC プロバイダの設定が不要で、IAM ロールの信頼ポリシーもシンプルになる。ただし Fargate Pod には対応していないため、Fargate を使う場合は IRSA を引き続き使用する。既存の IRSA 設定は移行の必要性は低く、新規の ServiceAccount から Pod Identity を使い始めるのが合理的である。

### Q6. EKS で複数チームがクラスターを共有する際のベストプラクティスは？

Namespace による論理的な分離、RBAC による権限制御、ResourceQuota による リソース制限、NetworkPolicy によるネットワーク分離、Pod Security Standards によるセキュリティ基準の適用を組み合わせる。大規模な組織では、チームごとにクラスターを分離し、GitOps ツール (ArgoCD ApplicationSet) で統一管理するアプローチも有効である。

### Q7. EKS で GPU ワークロードを実行するには？

GPU ノードグループ (p3, p4, g4dn, g5 インスタンス) を作成し、NVIDIA Device Plugin をインストールする。EKS 最適化 GPU AMI を使用すれば、ドライバのインストールは不要である。GPU リソースは `nvidia.com/gpu` としてリクエストし、Taint/Toleration でGPU ノードに専用 Pod のみスケジュールされるよう制御する。Karpenter を使えば、GPU インスタンスのオンデマンドな確保も自動化できる。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| EKS アーキテクチャ | コントロールプレーン (AWS管理) + データプレーン (ユーザー管理) |
| ノードグループ | マネージド(推奨)、セルフマネージド、Fargate から選択 |
| Fargate プロファイル | namespace + labels でマッチする Pod をサーバーレス実行 |
| Helm | Kubernetes のパッケージマネージャ。Chart でアプリケーションを管理 |
| IRSA / Pod Identity | Pod 単位で IAM ロールを付与。最小権限の実現に必須 |
| EKS アドオン | vpc-cni, coredns, kube-proxy 等をマネージドで管理 |
| オートスケーリング | Karpenter 推奨。HPA/KEDA で Pod レベルもスケール |
| セキュリティ | PSA, NetworkPolicy, Secrets CSI Driver で多層防御 |
| 可観測性 | Container Insights, Prometheus/Grafana, ADOT で統合監視 |
| GitOps | ArgoCD / Flux で宣言的なデプロイメント管理 |
| コスト最適化 | Spot + Karpenter, VPA, Kubecost で継続的に最適化 |

---

## 次に読むべきガイド

- [ECS 基礎](./00-ecs-basics.md) -- ECS との比較検討に
- [ECR](./01-ecr.md) -- コンテナイメージの管理
- [IAM 詳解](../08-security/00-iam-deep-dive.md) -- IRSA の IAM 設計を深める
- [CloudFormation](../07-devops/00-cloudformation.md) -- EKS クラスターの IaC 管理

---

## 参考文献

1. AWS 公式ドキュメント「Amazon EKS ユーザーガイド」 https://docs.aws.amazon.com/eks/latest/userguide/
2. AWS 公式「Amazon EKS ベストプラクティスガイド」 https://aws.github.io/aws-eks-best-practices/
3. eksctl 公式ドキュメント https://eksctl.io/
4. Helm 公式ドキュメント https://helm.sh/docs/
5. Karpenter 公式ドキュメント https://karpenter.sh/docs/
6. ArgoCD 公式ドキュメント https://argo-cd.readthedocs.io/
7. Flux CD 公式ドキュメント https://fluxcd.io/docs/
8. AWS 公式「EKS Pod Identity」 https://docs.aws.amazon.com/eks/latest/userguide/pod-identities.html
