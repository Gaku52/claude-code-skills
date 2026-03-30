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

### 01-github-actions — GitHub Actions

| # | ファイル | 内容 |
|---|---------|------|

### 02-deployment — デプロイメント

| # | ファイル | 内容 |
|---|---------|------|

### 03-monitoring — 監視とオブザーバビリティ

| # | ファイル | 内容 |
|---|---------|------|

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
