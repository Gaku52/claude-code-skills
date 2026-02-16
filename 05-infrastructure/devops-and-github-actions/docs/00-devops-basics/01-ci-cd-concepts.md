# CI/CD概念

> 継続的インテグレーション(CI)、継続的デリバリー(CD)、継続的デプロイメントの3段階を理解し、信頼性の高いパイプラインを設計する

## この章で学ぶこと

1. CI/CD/CDeploy の違いと段階的な導入アプローチを理解する
2. パイプライン設計の原則とステージ構成を習得する
3. ブランチ戦略とCI/CDの統合パターンを把握する
4. テスト戦略とCI/CDの連携方法を実践できる
5. モノレポ・マイクロサービスにおけるCI/CD設計パターンを理解する

---

## 1. CI/CD の3段階

### 1.1 全体像

```
コード変更 → CI → CD (デリバリー) → CD (デプロイメント)

+--------+    +------------------+    +-------------------+    +------------------+
| 開発者  | → | CI               | → | CD (デリバリー)     | → | CD (デプロイ)     |
| コード  |    | ビルド・テスト    |    | ステージング反映    |    | 本番自動反映      |
| 変更    |    | 自動実行          |    | 承認待ち           |    | 自動実行          |
+--------+    +------------------+    +-------------------+    +------------------+
                                                  ^
                                            手動承認ゲート
                                           (デリバリーの場合)
```

### 1.2 継続的インテグレーション (CI)

開発者がコードを頻繁に(1日数回)メインブランチに統合し、その都度自動でビルドとテストを実行するプラクティス。Martin Fowler が2006年に体系化した概念で、「統合の地獄(Integration Hell)」を回避することが最大の目的である。

**CIの核心原則:**

- コードは1日に1回以上メインブランチに統合する
- 全てのコミットに対してビルドとテストが自動実行される
- ビルドが壊れたら最優先で修復する(10分ルール)
- テストは高速で信頼性が高くなければならない

```yaml
# CI パイプラインの基本例 (GitHub Actions)
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run type-check

      - name: Unit tests
        run: npm test -- --coverage

      - name: Build
        run: npm run build
```

**CIの導入チェックリスト:**

```
□ ソースコードが単一のリポジトリで管理されている
□ ビルドが自動化されている(1コマンドでビルド可能)
□ テストスイートが存在し、自動実行される
□ コミットごとにCIが実行される設定がある
□ CIの結果が開発者に即座に通知される
□ ビルド失敗時の修復フローが確立されている
□ 10分以内にCIが完了する
□ メインブランチは常にグリーン(ビルド成功)状態を維持する
```

### 1.3 継続的デリバリー (Continuous Delivery)

CIに加え、リリース可能なアーティファクトを自動生成し、ステージング環境にデプロイする。本番デプロイは手動承認を経て実行。Jez Humble と David Farley が2010年に著書で定義した概念で、「いつでも安全にリリースできる状態」を常に維持することが目標である。

**継続的デリバリーの原則:**

- ソフトウェアは常にリリース可能な状態を維持する
- リリースはビジネス判断であり、技術的判断ではない
- デプロイプロセスは完全に自動化されている
- 本番デプロイと同一のプロセスが全環境で使用される

```yaml
# 継続的デリバリーの例
name: Continuous Delivery
on:
  push:
    branches: [main]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test && npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/

  deploy-staging:
    needs: build-and-test
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
      - run: ./scripts/deploy.sh staging

  # ステージング環境でのスモークテスト
  smoke-test-staging:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run smoke tests against staging
        run: |
          npm ci
          ENVIRONMENT=staging npm run test:smoke
        env:
          BASE_URL: https://staging.example.com

  deploy-production:
    needs: smoke-test-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      # 手動承認が必要（GitHub の Environment Protection Rules）
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
      - run: ./scripts/deploy.sh production
```

### 1.4 継続的デプロイメント (Continuous Deployment)

全てのテストに通過した変更が、人間の介入なしに本番環境へ自動デプロイされる。これは継続的デリバリーの延長であり、最も成熟したCI/CDの形態である。Facebook、Netflix、Etsy、GitHub自身がこの手法を採用している。

**継続的デプロイメントの前提条件:**

- 高いテストカバレッジ(ライン・ブランチともに80%以上)
- 包括的な自動テスト(ユニット、統合、E2E、パフォーマンス)
- 自動化されたロールバック機構
- カナリーデプロイまたはBlue-Greenデプロイの導入
- Feature Flag による機能の段階的公開
- 充実した監視・アラート(SLO/SLI ベース)
- 組織全体のリスク許容度と文化的成熟度

```
継続的デプロイメントのフロー:

Push → Build → Unit Test → Integration Test → E2E Test
  → Security Scan → Stage Deploy → Smoke Test
  → Canary Deploy (5%) → メトリクス監視 (15分)
  → Progressive Rollout (25% → 50% → 100%)
  → 異常検知時は自動ロールバック
```

```yaml
# 継続的デプロイメントの実装例 (Kubernetes + ArgoCD 連携)
name: Continuous Deployment
on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:all -- --coverage
      - name: Check coverage threshold
        run: |
          COVERAGE=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "Coverage $COVERAGE% is below 80% threshold"
            exit 1
          fi

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'

  build-and-push:
    needs: [test, security-scan]
    runs-on: ubuntu-latest
    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker image
        id: meta
        uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  update-manifests:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          repository: myorg/k8s-manifests
          token: ${{ secrets.MANIFEST_REPO_TOKEN }}
      - name: Update image tag
        run: |
          cd overlays/production
          kustomize edit set image my-app=ghcr.io/${{ github.repository }}:${{ github.sha }}
      - name: Commit and push
        run: |
          git config user.name "ci-bot"
          git config user.email "ci@example.com"
          git add .
          git commit -m "chore: update my-app to ${{ github.sha }}"
          git push
      # ArgoCD が変更を検知して自動同期 → カナリーデプロイ
```

### 1.5 3段階の段階的導入戦略

多くの組織は一足飛びに継続的デプロイメントに到達できない。以下の段階的アプローチが現実的である。

```
Phase 1: CI の確立 (1-3ヶ月)
  ├── 自動ビルドの設定
  ├── ユニットテストの整備 (カバレッジ 60% 目標)
  ├── Lint / 静的解析の導入
  └── PR レビュープロセスの確立

Phase 2: 継続的デリバリーの確立 (3-6ヶ月)
  ├── ステージング環境の自動デプロイ
  ├── 統合テスト・E2Eテストの整備 (カバレッジ 80% 目標)
  ├── Environment Protection Rules の設定
  ├── デプロイ手順の自動化
  └── ロールバック手順の確立

Phase 3: 継続的デプロイメントの確立 (6-12ヶ月)
  ├── カナリーデプロイの導入
  ├── 自動ロールバック機構の実装
  ├── SLO/SLI ベースの監視
  ├── Feature Flag の導入
  └── 文化的な変革 (失敗を許容する文化)
```

---

## 2. CI/CDの違い比較

### 2.1 3段階の比較表

| 項目 | CI | CD (デリバリー) | CD (デプロイメント) |
|---|---|---|---|
| 自動ビルド | はい | はい | はい |
| 自動テスト | はい | はい | はい |
| ステージングデプロイ | 任意 | 自動 | 自動 |
| 本番デプロイ | 手動 | 手動承認後自動 | 完全自動 |
| リスク | 低 | 中 | 高(要成熟度) |
| 前提条件 | テスト基盤 | CI + 環境管理 | CD + 高いテストカバレッジ |
| 適するチーム | 全チーム | CI成熟チーム | CD成熟チーム |
| リリース頻度 | - | 週1-2回 | 日に数回〜数十回 |
| フィードバック速度 | 分 | 時間 | 分 |
| 導入期間 | 1-3ヶ月 | 3-6ヶ月 | 6-12ヶ月 |

### 2.2 パイプラインツール比較

| ツール | ホスティング | 設定形式 | 特徴 | 適用場面 |
|---|---|---|---|---|
| GitHub Actions | SaaS (GitHub) | YAML | GitHub統合、マーケットプレイス | GitHub利用プロジェクト |
| GitLab CI | SaaS / Self-hosted | YAML | GitLab統合、Auto DevOps | GitLab利用プロジェクト |
| CircleCI | SaaS | YAML | 高速、Docker最適化 | パフォーマンス重視 |
| Jenkins | Self-hosted | Groovy/YAML | 高い拡張性、プラグイン | エンタープライズ |
| Dagger | ローカル / CI | CUE/Go/Python | ポータブル、ローカル再現 | マルチCI環境 |
| AWS CodePipeline | SaaS (AWS) | JSON/YAML | AWS統合、CodeBuild連携 | AWS中心のプロジェクト |
| Azure DevOps | SaaS / Self-hosted | YAML | Azure統合、Boards連携 | Microsoft エコシステム |
| Buildkite | SaaS + Self-hosted | YAML | 高スケーラビリティ | 大規模組織 |

### 2.3 ツール選定のディシジョンツリー

```
CI/CD ツール選定:

GitHub を使っている？
├── Yes → GitHub Actions (第一選択)
│         ├── セルフホステッドランナーが必要？ → Actions + Self-hosted Runner
│         └── AWS デプロイが中心？ → Actions + aws-actions/*
└── No → GitLab を使っている？
          ├── Yes → GitLab CI/CD
          └── No → エンタープライズ要件？
                    ├── Yes → Jenkins / Azure DevOps
                    └── No → マルチCI環境が必要？
                              ├── Yes → Dagger
                              └── No → CircleCI / Buildkite
```

---

## 3. パイプライン設計原則

### 3.1 パイプラインステージの標準構成

```
+-------+    +------+    +------+    +--------+    +--------+    +--------+
| Lint  | → | Build | → | Test  | → | Scan   | → | Stage  | → | Prod   |
| 静的  |    | 構築  |    | 検証  |    | 脆弱性 |    | 環境   |    | 環境   |
| 解析  |    |       |    |       |    | 検査   |    | デプロイ|    | デプロイ|
+-------+    +------+    +------+    +--------+    +--------+    +--------+
  ~10s        ~30s        ~2min       ~1min         ~3min         ~3min

                    ← 高速フィードバック →        ← 信頼性確保 →
              (失敗は早く検知して即座に修正)    (段階的に本番に近づける)
```

### 3.2 フィードバック速度の原則

```python
# パイプライン最適化: 高速フィードバックファースト
pipeline_stages = {
    "lint":           {"time": "10s",  "fail_rate": "20%", "order": 1},
    "type_check":     {"time": "15s",  "fail_rate": "10%", "order": 2},
    "unit_test":      {"time": "60s",  "fail_rate": "15%", "order": 3},
    "build":          {"time": "30s",  "fail_rate": "5%",  "order": 4},
    "integration_test":{"time": "180s", "fail_rate": "8%",  "order": 5},
    "e2e_test":       {"time": "300s", "fail_rate": "5%",  "order": 6},
    "security_scan":  {"time": "60s",  "fail_rate": "3%",  "order": 7},
}

# 原則: 失敗率の高いステージを先に実行
# 原則: 実行時間の短いステージを先に実行
# 原則: 独立したステージは並列実行
```

**フィードバック速度の目標値:**

| フェーズ | 目標時間 | 含まれる処理 |
|---|---|---|
| ローカルチェック | 30秒以内 | Lint、フォーマット |
| PR CI | 10分以内 | ビルド、テスト、静的解析 |
| ステージングデプロイ | 15分以内 | ビルド、テスト、デプロイ |
| 本番デプロイ | 30分以内 | 全テスト、承認、デプロイ、検証 |

### 3.3 並列化とキャッシュ

```yaml
# 並列化によるパイプライン高速化
name: Optimized CI
on: [push]

jobs:
  # Phase 1: 独立したジョブを並列実行
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run lint

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run type-check

  unit-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm test -- --coverage
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage
          path: coverage/

  # Phase 2: Phase 1 全てが成功したら実行
  build:
    needs: [lint, type-check, unit-test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci && npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/

  # Phase 3: E2E テスト (マトリクスで並列化)
  e2e-test:
    needs: build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npx playwright install --with-deps
      - run: npx playwright test --shard=${{ matrix.shard }}/4
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: playwright-report-${{ matrix.shard }}
          path: playwright-report/

  # Phase 4: セキュリティスキャン (Phase 2 と並列実行可能)
  security-scan:
    needs: [lint, type-check, unit-test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Snyk
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
```

### 3.4 パイプラインのメトリクス

パイプラインの健全性を測定するために以下のメトリクスを追跡する。

```
CI/CD メトリクス (DORA メトリクス):

1. デプロイ頻度 (Deployment Frequency)
   エリート: オンデマンド(1日複数回)
   高: 1日1回〜週1回
   中: 週1回〜月1回
   低: 月1回未満

2. リードタイム (Lead Time for Changes)
   エリート: 1時間未満
   高: 1日〜1週間
   中: 1週間〜1ヶ月
   低: 1ヶ月超

3. 変更失敗率 (Change Failure Rate)
   エリート: 0-15%
   高: 16-30%
   中: 16-30%
   低: 46-60%

4. 復旧時間 (Time to Restore Service)
   エリート: 1時間未満
   高: 1日未満
   中: 1日〜1週間
   低: 1週間超
```

```yaml
# CI メトリクスの収集例
name: CI Metrics
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - name: Calculate CI duration
        uses: actions/github-script@v7
        with:
          script: |
            const run = context.payload.workflow_run;
            const duration = (new Date(run.updated_at) - new Date(run.created_at)) / 1000;
            const status = run.conclusion;

            // メトリクスをDatadog等に送信
            console.log(`CI Duration: ${duration}s, Status: ${status}`);
            console.log(`Branch: ${run.head_branch}`);
            console.log(`Commit: ${run.head_sha}`);
```

---

## 4. ブランチ戦略とCI/CD

### 4.1 トランクベース開発

```
main ─────●────●────●────●────●────●── (常にデプロイ可能)
          │    │    │    │    │    │
          └─●──┘    └─●──┘    └─●──┘
         短命ブランチ  短命ブランチ  短命ブランチ
         (数時間〜1日) (数時間〜1日) (数時間〜1日)

特徴:
- main ブランチが常にデプロイ可能
- フィーチャーブランチは短命(最大1-2日)
- Feature Flag で未完成機能を隠す
- 継続的デプロイメントと相性が良い
```

**トランクベース開発のCI/CD設定:**

```yaml
# トランクベース開発向けCI/CD
name: Trunk-Based CD
on:
  push:
    branches: [main]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run lint
      - run: npm test -- --coverage
      - run: npm run build

  deploy:
    needs: ci
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Deploy with feature flags
        run: |
          # Feature Flag の状態を反映してデプロイ
          ./scripts/deploy.sh \
            --image ghcr.io/myorg/app:${{ github.sha }} \
            --feature-flags-config flags.json
        env:
          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
```

### 4.2 GitHub Flow

```
main ─────●─────────●─────────●──────── (保護ブランチ)
          │         ↑         │
          └──●──●──PR──merge──┘
          feature/xxx
          (PR レビュー必須)

特徴:
- シンプルなブランチモデル
- PR ベースのコードレビュー
- CI が PR 上で実行される
- main マージ後にデプロイ
```

### 4.3 Git Flow

```
main ───────────────●──────────────●──── (リリースタグ)
                    ↑              ↑
develop ──●──●──●───┤──●──●──●────┤──── (統合ブランチ)
          │  │  │   │  │  │  │    │
          └──┘  │   │  └──┘  │    │
        feature │   │  feature│    │
                │   │         │    │
            release  │     release │
              │      │         │   │
              └──→───┘         └──→┘

特徴:
- 複雑だがリリース管理に厳密
- main, develop, feature, release, hotfix ブランチ
- 長期メンテナンスのソフトウェアに適する
- CI/CD との統合は複雑になりがち
```

```yaml
# Git Flow 向けCI/CD
name: Git Flow CI/CD
on:
  push:
    branches: [main, develop, 'release/**', 'hotfix/**']
  pull_request:
    branches: [main, develop]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test && npm run build

  deploy-dev:
    if: github.ref == 'refs/heads/develop'
    needs: ci
    runs-on: ubuntu-latest
    environment: development
    steps:
      - run: ./deploy.sh development

  deploy-staging:
    if: startsWith(github.ref, 'refs/heads/release/')
    needs: ci
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - run: ./deploy.sh staging

  deploy-production:
    if: github.ref == 'refs/heads/main'
    needs: ci
    runs-on: ubuntu-latest
    environment: production
    steps:
      - run: ./deploy.sh production
```

### 4.4 ブランチ戦略の比較

| 項目 | トランクベース | GitHub Flow | Git Flow |
|---|---|---|---|
| 複雑さ | 低 | 低 | 高 |
| ブランチ数 | 最小 | 少 | 多 |
| リリース頻度 | 非常に高い | 高い | 中〜低 |
| Feature Flag 必要度 | 高 | 中 | 低 |
| CI/CD との相性 | 最高 | 良好 | 複雑 |
| 適用場面 | SaaS、Web | 一般的なプロジェクト | パッケージ、ライブラリ |
| チーム規模 | 小〜大 | 小〜中 | 中〜大 |

---

## 5. テスト戦略

### 5.1 テストピラミッド

```
            /\
           /  \
          / E2E \          少数・高コスト・遅い
         /------\
        /  統合   \        中程度
       /テスト     \
      /------------\
     /   ユニット    \     多数・低コスト・速い
    / テスト          \
   /------------------\

推奨比率:
  Unit     : Integration : E2E = 70 : 20 : 10
```

### 5.2 各テストレベルの特徴

| テストレベル | 実行時間 | カバレッジ目標 | テスト対象 | ツール例 |
|---|---|---|---|---|
| Unit | ミリ秒〜秒 | 80%+ | 関数、クラス | Jest, Vitest, pytest |
| Integration | 秒〜分 | 60%+ | API、DB連携 | Supertest, TestContainers |
| E2E | 分〜十分 | 主要フロー | ユーザーシナリオ | Playwright, Cypress |
| Performance | 分〜時間 | SLO目標 | レスポンスタイム | k6, Artillery |
| Security | 分 | 脆弱性ゼロ | 依存関係、コード | Snyk, Trivy, tfsec |

### 5.3 CI でのテスト実行例

```yaml
# テストピラミッドに基づくCI設定
name: Test Pipeline
on: [push]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:unit -- --coverage --coverageThreshold='{"global":{"branches":80}}'
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          fail_ci_if_error: true

  integration-tests:
    needs: unit-tests
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: testdb
          POSTGRES_PASSWORD: test
        ports: ['5432:5432']
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        ports: ['6379:6379']
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/testdb
          REDIS_URL: redis://localhost:6379

  e2e-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3]
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npx playwright install --with-deps
      - run: npm run test:e2e -- --shard=${{ matrix.shard }}/3
      - uses: actions/upload-artifact@v4
        if: failure()
        with:
          name: e2e-results-${{ matrix.shard }}
          path: |
            playwright-report/
            test-results/

  performance-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Run k6 load tests
        uses: grafana/k6-action@v0.3.1
        with:
          filename: tests/performance/load-test.js
        env:
          K6_CLOUD_TOKEN: ${{ secrets.K6_CLOUD_TOKEN }}
```

### 5.4 テスト品質の指標

```yaml
# Mutation Testing でテストの品質を測定
name: Mutation Testing
on:
  schedule:
    - cron: '0 3 * * 1'  # 毎週月曜 AM3時

jobs:
  mutation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - name: Run Stryker Mutation Testing
        run: npx stryker run
      - name: Check mutation score
        run: |
          SCORE=$(cat reports/mutation/mutation.json | jq '.schemaVersion' )
          echo "Mutation Score: $SCORE%"
          # 70% 以上を要求
```

---

## 6. モノレポにおけるCI/CD

### 6.1 モノレポのCI/CD課題

```
モノレポの構成例:
monorepo/
├── packages/
│   ├── web/          # フロントエンド
│   ├── api/          # バックエンド
│   ├── shared/       # 共通ライブラリ
│   └── mobile/       # モバイルアプリ
├── package.json
└── turbo.json

課題:
- 全パッケージのCI実行は時間がかかりすぎる
- 変更されていないパッケージのテストは無駄
- パッケージ間の依存関係を考慮する必要がある
```

### 6.2 Affected 戦略

```yaml
# Turborepo を使った affected ビルド
name: Monorepo CI
on: [push, pull_request]

jobs:
  determine-affected:
    runs-on: ubuntu-latest
    outputs:
      web: ${{ steps.filter.outputs.web }}
      api: ${{ steps.filter.outputs.api }}
      shared: ${{ steps.filter.outputs.shared }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            web:
              - 'packages/web/**'
              - 'packages/shared/**'
            api:
              - 'packages/api/**'
              - 'packages/shared/**'
            shared:
              - 'packages/shared/**'

  test-web:
    needs: determine-affected
    if: needs.determine-affected.outputs.web == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npx turbo test --filter=web...

  test-api:
    needs: determine-affected
    if: needs.determine-affected.outputs.api == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npx turbo test --filter=api...

  # Turborepo を使った一括実行(affected のみ)
  turbo-ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - run: npm ci
      - name: Build and test affected packages
        run: npx turbo build test lint --filter='...[HEAD~1]'
        env:
          TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
          TURBO_TEAM: ${{ vars.TURBO_TEAM }}
```

### 6.3 Nx を使ったモノレポCI

```yaml
# Nx の affected コマンドを使用
name: Nx Monorepo CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: nrwl/nx-set-shas@v4

      - run: npm ci

      - name: Lint affected
        run: npx nx affected -t lint --parallel=3

      - name: Test affected
        run: npx nx affected -t test --parallel=3 --ci

      - name: Build affected
        run: npx nx affected -t build --parallel=3

      - name: E2E affected
        run: npx nx affected -t e2e --parallel=1
```

---

## 7. マイクロサービスのCI/CD

### 7.1 サービスごとの独立パイプライン

```yaml
# サービスごとのCI/CDパイプライン
name: User Service CI/CD
on:
  push:
    branches: [main]
    paths:
      - 'services/user-service/**'
      - 'libs/common/**'

jobs:
  ci:
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/user-service
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test
      - run: npm run build

  build-image:
    needs: ci
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/build-push-action@v5
        with:
          context: services/user-service
          push: true
          tags: |
            ghcr.io/myorg/user-service:${{ github.sha }}
            ghcr.io/myorg/user-service:latest

  deploy:
    needs: build-image
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy user-service
        run: |
          kubectl set image deployment/user-service \
            app=ghcr.io/myorg/user-service:${{ github.sha }}
```

### 7.2 契約テスト (Contract Testing)

マイクロサービス間のAPI互換性を保証するため、契約テストをCIに組み込む。

```yaml
# Pact による契約テスト
name: Contract Tests
on: [push]

jobs:
  consumer-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - name: Run consumer contract tests
        run: npm run test:contract:consumer
      - name: Publish pacts
        run: |
          npx pact-broker publish pacts/ \
            --broker-base-url=${{ secrets.PACT_BROKER_URL }} \
            --consumer-app-version=${{ github.sha }}

  provider-verification:
    needs: consumer-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - name: Verify provider contracts
        run: npm run test:contract:provider
        env:
          PACT_BROKER_URL: ${{ secrets.PACT_BROKER_URL }}
          PACT_PROVIDER_VERSION: ${{ github.sha }}
```

---

## 8. デプロイ戦略との連携

### 8.1 Blue-Green デプロイ

```yaml
# Blue-Green デプロイの CI/CD 連携
name: Blue-Green Deploy
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Determine current environment
        id: current
        run: |
          CURRENT=$(aws elbv2 describe-target-groups \
            --names prod-active | jq -r '.TargetGroups[0].Tags[] | select(.Key=="color") | .Value')
          if [ "$CURRENT" = "blue" ]; then
            echo "deploy_to=green" >> "$GITHUB_OUTPUT"
          else
            echo "deploy_to=blue" >> "$GITHUB_OUTPUT"
          fi

      - name: Deploy to inactive environment
        run: ./deploy.sh ${{ steps.current.outputs.deploy_to }}

      - name: Run smoke tests
        run: ./smoke-test.sh ${{ steps.current.outputs.deploy_to }}

      - name: Switch traffic
        run: |
          aws elbv2 modify-listener \
            --listener-arn ${{ secrets.ALB_LISTENER_ARN }} \
            --default-actions Type=forward,TargetGroupArn=${{ steps.current.outputs.deploy_to }}-tg-arn
```

### 8.2 カナリーデプロイ

```yaml
# カナリーデプロイの CI/CD 連携
name: Canary Deploy
on:
  push:
    branches: [main]

jobs:
  canary:
    runs-on: ubuntu-latest
    environment: production
    steps:
      - uses: actions/checkout@v4

      - name: Deploy canary (5%)
        run: |
          kubectl apply -f k8s/canary-deployment.yaml
          kubectl set image deployment/app-canary \
            app=ghcr.io/myorg/app:${{ github.sha }}

      - name: Wait and monitor (15 minutes)
        run: |
          for i in $(seq 1 15); do
            ERROR_RATE=$(curl -s "$PROMETHEUS_URL/api/v1/query?query=rate(http_requests_total{status=~'5..'}[5m])" | jq '.data.result[0].value[1]')
            echo "Minute $i: Error rate = $ERROR_RATE"
            if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
              echo "Error rate exceeded threshold, rolling back"
              kubectl rollout undo deployment/app-canary
              exit 1
            fi
            sleep 60
          done

      - name: Progressive rollout
        run: |
          for WEIGHT in 25 50 75 100; do
            echo "Rolling out to ${WEIGHT}%"
            kubectl patch deployment/app-production \
              -p "{\"spec\":{\"replicas\":$((WEIGHT * TOTAL_REPLICAS / 100))}}"
            sleep 300  # 5分間監視
          done
```

---

## 9. アンチパターン

### アンチパターン1: CI シアター

```
問題:
  CI パイプラインは存在するが、形骸化している。
  - テストカバレッジが低く、テストが通っても品質が保証されない
  - 失敗したテストを skip や ignore で無視
  - "CI が通った = 安全" という誤った安心感

症状:
  ✗ テストカバレッジ 20% で "CI 通りました"
  ✗ @skip アノテーションが 50 個以上
  ✗ CI は通るが本番障害が頻発
  ✗ フレーキーテスト(不安定なテスト)を放置

改善:
  1. カバレッジ閾値を設定(最低80%)
  2. @skip テストの定期棚卸し
  3. 本番障害ごとに回帰テスト追加
  4. Mutation Testing の導入
  5. フレーキーテストの専用トラッカーで管理
```

### アンチパターン2: 巨大モノリシックパイプライン

```
問題:
  1つのパイプラインに全てを詰め込み、実行時間が30分〜1時間超。
  フィードバックが遅く、開発者がCIを無視するようになる。

  Push → Lint → Build → Unit → Integration → E2E → Deploy
  |                    30分〜1時間                        |

改善:
  1. ステージの並列化
  2. 変更されたパッケージのみテスト (affected)
  3. テストの分割実行 (sharding)
  4. キャッシュの活用
  5. 差分ビルド

  目標: PR の CI は 10 分以内に完了
```

### アンチパターン3: 環境差異の放置

```
問題:
  ローカル、CI、ステージング、本番で環境が異なる。
  "ローカルでは通るのにCIでは失敗する" が頻発。

症状:
  ✗ ローカルは Node 18、CI は Node 20
  ✗ CI にはデータベースがなく、統合テストをスキップ
  ✗ ステージングと本番でインフラ構成が異なる

改善:
  1. Docker / Dev Containers で開発環境を統一
  2. .node-version, .tool-versions で言語バージョンを固定
  3. CI に services: でデータベース等を起動
  4. IaC で全環境のインフラを管理
  5. 環境変数の管理を一元化
```

### アンチパターン4: 手動リリースプロセス

```
問題:
  CI は自動化されているが、リリースは手動。
  "リリース手順書" が存在し、手順を間違えるリスクがある。

症状:
  ✗ 30ステップのリリースチェックリスト
  ✗ リリースのたびに "今回のリリース担当" を決める
  ✗ リリース日は残業確定

改善:
  1. リリースプロセスをワークフローに定義
  2. semantic-release / Release Please で自動バージョニング
  3. Environment Protection Rules で承認フローを自動化
  4. ロールバック手順もワークフローで定義
```

---

## 10. セキュリティとCI/CD

### 10.1 サプライチェーンセキュリティ

```yaml
# 依存関係の脆弱性スキャン
name: Security Scan
on:
  push:
    branches: [main]
  schedule:
    - cron: '0 9 * * 1'  # 毎週月曜

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Dependency Review (PR のみ)
        if: github.event_name == 'pull_request'
        uses: actions/dependency-review-action@v4
        with:
          fail-on-severity: high

      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          severity: 'CRITICAL,HIGH'
          format: 'sarif'
          output: 'trivy-results.sarif'

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: 'trivy-results.sarif'

  sast:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: CodeQL Analysis
        uses: github/codeql-action/init@v3
        with:
          languages: javascript
      - uses: github/codeql-action/analyze@v3

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t app:scan .
      - name: Scan container image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'app:scan'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'
```

### 10.2 CI/CDパイプラインのセキュリティベストプラクティス

```
1. 最小権限の原則
   - GITHUB_TOKEN の permissions を明示的に設定
   - 各ジョブに必要最小限の権限のみ付与
   - サードパーティアクションはSHAでピン留め

2. シークレット管理
   - Environment Secrets で環境ごとに分離
   - OIDC でクラウドプロバイダーに認証(長期キー不要)
   - シークレットのローテーションを自動化

3. サプライチェーン保護
   - Dependabot / Renovate で依存関係を自動更新
   - ロックファイル(package-lock.json)をコミット
   - npm audit / pip audit をCIに組み込み
   - SLSA / Sigstore でビルドの証明

4. ブランチ保護
   - main ブランチへの直接プッシュを禁止
   - PR レビュー必須(最低2名)
   - CI 成功必須でマージ
   - 署名付きコミット推奨
```

---

## 11. FAQ

### Q1: CI/CD パイプラインの理想的な実行時間は？

PR 上の CI は 10 分以内が目標。開発者の集中力が途切れる前にフィードバックを返す必要がある。10分を超える場合は並列化、キャッシュ、テスト分割、差分ビルドを検討する。デプロイパイプラインは 15 分以内が目安。Google の研究(DORA)によると、エリートチームのリードタイムは1時間未満である。

### Q2: 継続的デプロイメントを導入する前提条件は？

(1) 高いテストカバレッジ(80%以上)、(2) 自動化されたロールバック機構、(3) カナリーデプロイ/Feature Flag、(4) 充実した監視・アラート、(5) 組織の信頼とリスク許容度。これら全てが揃わないまま導入するとインシデントが頻発する。段階的に導入し、まず継続的デリバリーを確立することを推奨する。

### Q3: モノレポの CI/CD はどう設計するか？

変更されたパッケージのみを対象にテスト・ビルドする「affected」戦略が基本。Nx、Turborepo、Bazel などのモノレポツールが差分検知を提供する。加えて、パッケージ間の依存関係グラフに基づいて、影響を受ける下流パッケージも含めてテストする。Remote Cache(Turborepo Cloud, Nx Cloud)を活用するとCIの高速化に大きく貢献する。

### Q4: フレーキーテスト(不安定なテスト)にどう対処するか？

まず不安定なテストを特定し、専用のトラッカー(Issue, スプレッドシート等)で管理する。根本原因は多くの場合、(1) テスト間の依存関係、(2) 非同期処理の待機不足、(3) 外部サービスへの依存、(4) 時間依存のロジックである。短期的には `retry` オプションでリトライし、中長期的には根本原因を修正する。GitHub Actions では `retry-on-error` アクションを使用できる。

### Q5: CI/CD のコストを最適化するには？

(1) キャッシュの活用(依存関係、ビルド成果物)、(2) 差分テスト(affected のみ)、(3) テストのシャーディング(並列化)、(4) 不要なワークフローの paths フィルタ、(5) スケジュール実行の見直し、(6) セルフホステッドランナーの検討(大量実行時)。GitHub Actions の場合、パブリックリポジトリは無料、プライベートは月間2,000分(Free)から50,000分(Enterprise)。

### Q6: Feature Flag とCI/CDの関係は？

Feature Flag はトランクベース開発と継続的デプロイメントの実現に不可欠な技術。未完成の機能をフラグで隠しながらメインブランチにマージし、本番にデプロイできる。これにより長寿命フィーチャーブランチを避け、マージコンフリクトのリスクを大幅に削減する。LaunchDarkly、Unleash、Flagsmith、AWS AppConfig 等のサービスが利用可能。

---

## まとめ

| 項目 | 要点 |
|---|---|
| CI | 頻繁な統合 + 自動ビルド・テスト |
| CD (デリバリー) | リリース可能状態を常に維持 + 手動承認デプロイ |
| CD (デプロイメント) | 全自動本番デプロイ(高成熟度が前提) |
| パイプライン設計 | 高速フィードバック、並列化、キャッシュ |
| テスト戦略 | テストピラミッド(Unit 70 : Integration 20 : E2E 10) |
| ブランチ戦略 | トランクベース(推奨)、GitHub Flow、Git Flow |
| モノレポ | affected 戦略 + Remote Cache |
| セキュリティ | 依存関係スキャン、SAST、最小権限 |
| メトリクス | DORA メトリクスで改善を定量化 |
| 目標時間 | CI 10分以内、デプロイ15分以内 |

---

## 次に読むべきガイド

- [GitHub Actions 基礎](../01-github-actions/00-actions-basics.md) -- CI/CD を実現する具体的ツール
- [デプロイ戦略](../02-deployment/00-deployment-strategies.md) -- Blue-Green、Canary 等の戦略
- [Infrastructure as Code](./02-infrastructure-as-code.md) -- インフラ自動化の基礎
- [GitOps](./03-gitops.md) -- Gitを中心としたデプロイモデル

---

## 参考文献

1. Jez Humble, David Farley. *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*. Addison-Wesley, 2010.
2. Martin Fowler. "Continuous Integration." https://martinfowler.com/articles/continuousIntegration.html
3. Google. "Trunk-Based Development." https://trunkbaseddevelopment.com/
4. Charity Majors. "Test in Production." https://increment.com/testing/i-test-in-production/
5. DORA Team. "Accelerate: State of DevOps Report." https://dora.dev/
6. Nicole Forsgren, Jez Humble, Gene Kim. *Accelerate: The Science of Lean Software and DevOps*. IT Revolution Press, 2018.
7. Sam Newman. *Building Microservices*, 2nd Edition. O'Reilly Media, 2021.
8. GitHub. "GitHub Actions Documentation." https://docs.github.com/en/actions
