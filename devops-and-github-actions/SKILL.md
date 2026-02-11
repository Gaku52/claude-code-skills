# DevOps と GitHub Actions

> DevOps はソフトウェアのビルド・テスト・デプロイを自動化し、開発と運用の壁を取り払う。CI/CD の基礎、GitHub Actions の実践、デプロイ戦略、監視・アラートまで、モダン DevOps の全体像を解説する。

## このSkillの対象者

- CI/CD パイプラインを構築したいエンジニア
- GitHub Actions を本格活用したい方
- デプロイ自動化・監視体制を整備したい方

## 前提知識

- Git の基本操作
- Docker の基礎知識
- YAML の記法

## 学習ガイド

### 00-devops-basics — DevOps の基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-devops-basics/00-devops-culture.md]] | DevOps 文化、CI/CD の概念、DORA メトリクス |
| 01 | [[docs/00-devops-basics/01-ci-cd-fundamentals.md]] | CI/CD パイプライン設計、ブランチ戦略連携、品質ゲート |
| 02 | [[docs/00-devops-basics/02-infrastructure-as-code.md]] | IaC 概念、Terraform 基礎、CDK、Pulumi |
| 03 | [[docs/00-devops-basics/03-gitops.md]] | GitOps 原則、ArgoCD、Flux、環境管理 |

### 01-github-actions — GitHub Actions

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-github-actions/00-actions-basics.md]] | ワークフロー構文、トリガー、ジョブ、ステップ、ランナー |
| 01 | [[docs/01-github-actions/01-actions-advanced.md]] | マトリクスビルド、キャッシュ、Artifacts、環境、シークレット |
| 02 | [[docs/01-github-actions/02-reusable-workflows.md]] | 再利用可能ワークフロー、Composite Actions、カスタムアクション |
| 03 | [[docs/01-github-actions/03-security-and-optimization.md]] | OIDC 認証、Dependabot、セキュリティベストプラクティス、コスト最適化 |
| 04 | [[docs/01-github-actions/04-ci-recipes.md]] | Node.js/Python/Go/Rust/Docker のCIレシピ集 |

### 02-deployment — デプロイメント

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-deployment/00-deployment-strategies.md]] | Blue-Green、Canary、Rolling、Feature Flag |
| 01 | [[docs/02-deployment/01-container-deployment.md]] | ECS/EKS デプロイ、Fargate、Helm Chart |
| 02 | [[docs/02-deployment/02-serverless-deployment.md]] | Lambda デプロイ、Vercel/Netlify、SST/Serverless Framework |
| 03 | [[docs/02-deployment/03-database-migrations.md]] | DBマイグレーション戦略、ゼロダウンタイム、Prisma migrate |

### 03-monitoring — 監視とオブザーバビリティ

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-monitoring/00-observability-basics.md]] | オブザーバビリティの3本柱（ログ・メトリクス・トレース） |
| 01 | [[docs/03-monitoring/01-logging.md]] | 構造化ログ、ログ集約、Datadog/CloudWatch Logs |
| 02 | [[docs/03-monitoring/02-metrics-and-alerting.md]] | SLI/SLO/SLA、Prometheus、Grafana、PagerDuty |
| 03 | [[docs/03-monitoring/03-distributed-tracing.md]] | OpenTelemetry、Jaeger、X-Ray、トレース設計 |

## クイックリファレンス

```
GitHub Actions 構文早見表:

  トリガー:
    on: push / pull_request / schedule / workflow_dispatch
    branches: [main] / paths: ['src/**']

  ジョブ:
    runs-on: ubuntu-latest
    strategy: matrix (node-version: [18, 20, 22])
    needs: [build, test]

  よく使うアクション:
    actions/checkout@v4
    actions/setup-node@v4
    actions/cache@v4
    docker/build-push-action@v5
    aws-actions/configure-aws-credentials@v4

  デプロイ戦略選定:
    低リスク小規模 → Rolling Update
    中リスク → Blue-Green
    高リスク大規模 → Canary
    実験的機能 → Feature Flag
```

## 参考文献

1. GitHub. "Actions Documentation." docs.github.com/actions, 2024.
2. Forsgren, N. et al. "Accelerate." IT Revolution Press, 2018.
3. HashiCorp. "Terraform Documentation." terraform.io/docs, 2024.
