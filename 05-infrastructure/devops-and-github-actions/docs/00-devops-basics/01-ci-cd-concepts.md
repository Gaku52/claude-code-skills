# CI/CD概念

> 継続的インテグレーション(CI)、継続的デリバリー(CD)、継続的デプロイメントの3段階を理解し、信頼性の高いパイプラインを設計する

## この章で学ぶこと

1. CI/CD/CDeploy の違いと段階的な導入アプローチを理解する
2. パイプライン設計の原則とステージ構成を習得する
3. ブランチ戦略とCI/CDの統合パターンを把握する

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

開発者がコードを頻繁に(1日数回)メインブランチに統合し、その都度自動でビルドとテストを実行するプラクティス。

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

### 1.3 継続的デリバリー (Continuous Delivery)

CIに加え、リリース可能なアーティファクトを自動生成し、ステージング環境にデプロイする。本番デプロイは手動承認を経て実行。

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

  deploy-production:
    needs: deploy-staging
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

全てのテストに通過した変更が、人間の介入なしに本番環境へ自動デプロイされる。

```
継続的デプロイメントのフロー:

Push → Build → Unit Test → Integration Test → E2E Test
  → Security Scan → Stage Deploy → Smoke Test
  → Canary Deploy (5%) → メトリクス監視 (15分)
  → Progressive Rollout (25% → 50% → 100%)
  → 異常検知時は自動ロールバック
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

### 2.2 パイプラインツール比較

| ツール | ホスティング | 設定形式 | 特徴 | 適用場面 |
|---|---|---|---|---|
| GitHub Actions | SaaS (GitHub) | YAML | GitHub統合、マーケットプレイス | GitHub利用プロジェクト |
| GitLab CI | SaaS / Self-hosted | YAML | GitLab統合、Auto DevOps | GitLab利用プロジェクト |
| CircleCI | SaaS | YAML | 高速、Docker最適化 | パフォーマンス重視 |
| Jenkins | Self-hosted | Groovy/YAML | 高い拡張性、プラグイン | エンタープライズ |
| Dagger | ローカル / CI | CUE/Go/Python | ポータブル、ローカル再現 | マルチCI環境 |

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
      - run: npm ci --cache .npm
      - run: npm run lint

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci --cache .npm
      - run: npm run type-check

  unit-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci --cache .npm
      - run: npm test

  # Phase 2: Phase 1 全てが成功したら実行
  build:
    needs: [lint, type-check, unit-test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm run build

  # Phase 3: E2E テスト (マトリクスで並列化)
  e2e-test:
    needs: build
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - run: npx playwright test --shard=${{ matrix.shard }}/4
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

### 5.2 CI でのテスト実行例

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
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm run test:integration
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/testdb

  e2e-tests:
    needs: integration-tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npx playwright install --with-deps
      - run: npm run test:e2e
```

---

## 6. アンチパターン

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

改善:
  1. カバレッジ閾値を設定(最低80%)
  2. @skip テストの定期棚卸し
  3. 本番障害ごとに回帰テスト追加
  4. Mutation Testing の導入
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

---

## 7. FAQ

### Q1: CI/CD パイプラインの理想的な実行時間は？

PR 上の CI は 10 分以内が目標。開発者の集中力が途切れる前にフィードバックを返す必要がある。10分を超える場合は並列化、キャッシュ、テスト分割、差分ビルドを検討する。デプロイパイプラインは 15 分以内が目安。

### Q2: 継続的デプロイメントを導入する前提条件は？

(1) 高いテストカバレッジ(80%以上)、(2) 自動化されたロールバック機構、(3) カナリーデプロイ/Feature Flag、(4) 充実した監視・アラート、(5) 組織の信頼とリスク許容度。これら全てが揃わないまま導入するとインシデントが頻発する。

### Q3: モノレポの CI/CD はどう設計するか？

変更されたパッケージのみを対象にテスト・ビルドする「affected」戦略が基本。Nx、Turborepo、Bazel などのモノレポツールが差分検知を提供する。加えて、パッケージ間の依存関係グラフに基づいて、影響を受ける下流パッケージも含めてテストする。

---

## まとめ

| 項目 | 要点 |
|---|---|
| CI | 頻繁な統合 + 自動ビルド・テスト |
| CD (デリバリー) | リリース可能状態を常に維持 + 手動承認デプロイ |
| CD (デプロイメント) | 全自動本番デプロイ(高成熟度が前提) |
| パイプライン設計 | 高速フィードバック、並列化、キャッシュ |
| テスト戦略 | テストピラミッド(Unit 70 : Integration 20 : E2E 10) |
| 目標時間 | CI 10分以内、デプロイ15分以内 |

---

## 次に読むべきガイド

- [GitHub Actions 基礎](../01-github-actions/00-actions-basics.md) -- CI/CD を実現する具体的ツール
- [デプロイ戦略](../02-deployment/00-deployment-strategies.md) -- Blue-Green、Canary 等の戦略
- [Infrastructure as Code](./02-infrastructure-as-code.md) -- インフラ自動化の基礎

---

## 参考文献

1. Jez Humble, David Farley. *Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation*. Addison-Wesley, 2010.
2. Martin Fowler. "Continuous Integration." https://martinfowler.com/articles/continuousIntegration.html
3. Google. "Trunk-Based Development." https://trunkbaseddevelopment.com/
4. Charity Majors. "Test in Production." https://increment.com/testing/i-test-in-production/
