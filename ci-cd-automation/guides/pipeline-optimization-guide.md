# パイプライン最適化 完全ガイド

**作成日**: 2026年1月
**対象**: GitHub Actions, Fastlane, CI/CD全般
**レベル**: 中級〜上級

---

## 目次

1. [最適化の基本原則](#最適化の基本原則)
2. [ビルド時間の分析](#ビルド時間の分析)
3. [並列化戦略](#並列化戦略)
4. [キャッシュ最適化](#キャッシュ最適化)
5. [依存関係の最適化](#依存関係の最適化)
6. [テスト最適化](#テスト最適化)
7. [Docker最適化](#docker最適化)
8. [コスト最適化](#コスト最適化)
9. [モニタリングとメトリクス](#モニタリングとメトリクス)
10. [実践的な最適化事例](#実践的な最適化事例)

---

## 最適化の基本原則

### パフォーマンス最適化のピラミッド

```
        ┌─────────────────┐
        │  並列化・分散   │  最大の効果
        ├─────────────────┤
        │  キャッシング   │
        ├─────────────────┤
        │  不要な処理削減 │
        ├─────────────────┤
        │ アルゴリズム改善│
        └─────────────────┘  基本
```

### 測定→分析→最適化のサイクル

```yaml
# 1. 測定: 現状を把握
- name: ビルド時間を記録
  run: |
    START_TIME=$(date +%s)
    npm run build
    END_TIME=$(date +%s)
    echo "Build time: $((END_TIME - START_TIME))s"

# 2. 分析: ボトルネックを特定
- name: プロファイリング
  run: npm run build -- --profile

# 3. 最適化: 改善を実施
- name: 最適化されたビルド
  run: npm run build -- --max-workers=4
```

### 最適化の優先順位

1. **並列化** (効果: 大、難易度: 中)
   - 独立したジョブを同時実行
   - テストのシャーディング

2. **キャッシング** (効果: 大、難易度: 低)
   - 依存関係のキャッシュ
   - ビルド成果物のキャッシュ

3. **不要な処理の削減** (効果: 中、難易度: 低)
   - 条件付き実行
   - パスフィルタリング

4. **リソース増強** (効果: 中、難易度: 低、コスト: 高)
   - より高速なランナー使用
   - メモリ増量

---

## ビルド時間の分析

### 1. 詳細なタイミング計測

```yaml
# .github/workflows/build-profiling.yml
name: Build Profiling

on: [push]

jobs:
  profile:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: 各ステップの時間計測
        run: |
          # タイムスタンプ関数
          timestamp() {
            date +%s
          }

          # チェックアウト時間
          START=$(timestamp)
          echo "::group::Checkout completed"
          echo "Time: $(($(timestamp) - START))s"
          echo "::endgroup::"

          # 依存関係インストール
          START=$(timestamp)
          npm ci
          echo "::group::Dependencies installed"
          echo "Time: $(($(timestamp) - START))s"
          echo "::endgroup::"

          # ビルド
          START=$(timestamp)
          npm run build
          echo "::group::Build completed"
          echo "Time: $(($(timestamp) - START))s"
          echo "::endgroup::"

          # テスト
          START=$(timestamp)
          npm test
          echo "::group::Tests completed"
          echo "Time: $(($(timestamp) - START))s"
          echo "::endgroup::"
```

### 2. ビルド時間の可視化

```yaml
- name: ビルド時間レポート生成
  if: always()
  run: |
    cat > build-report.md << 'EOF'
    # ビルド時間レポート

    | ステップ | 時間 | 割合 |
    |---------|------|------|
    | Checkout | 5s | 2% |
    | Dependencies | 120s | 48% |
    | Build | 90s | 36% |
    | Test | 35s | 14% |
    | **Total** | **250s** | **100%** |

    ## 改善提案
    - Dependencies: キャッシュ導入で90s削減可能
    - Build: 並列化で45s削減可能
    EOF

- name: レポートをPRにコメント
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const report = fs.readFileSync('build-report.md', 'utf8');
      github.rest.issues.createComment({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        body: report
      });
```

### 3. GitHub Actionsのタイミングデータを活用

```bash
# GitHub CLI でワークフロー実行時間を取得
gh run list --workflow=ci.yml --json databaseId,conclusion,createdAt,updatedAt | \
  jq '.[] | {
    id: .databaseId,
    duration: (.updatedAt | fromdateiso8601) - (.createdAt | fromdateiso8601),
    conclusion: .conclusion
  }'

# 平均実行時間を計算
gh run list --workflow=ci.yml --limit 100 --json createdAt,updatedAt | \
  jq '[.[] | ((.updatedAt | fromdateiso8601) - (.createdAt | fromdateiso8601))] | add / length'
```

---

## 並列化戦略

### 1. ジョブレベルの並列化

```yaml
# ❌ 悪い例: 直列実行（25分）
jobs:
  lint-build-test:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint      # 5分
      - run: npm run build     # 10分
      - run: npm test          # 10分

# ✅ 良い例: 並列実行（10分）
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint      # 5分（並列）

  build:
    runs-on: ubuntu-latest
    steps:
      - run: npm run build     # 10分（並列）

  test:
    runs-on: ubuntu-latest
    steps:
      - run: npm test          # 10分（並列）
```

### 2. マトリックス戦略

```yaml
# テストを4つのシャードに分割
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - name: テスト実行（シャード ${{ matrix.shard }}/4）
        run: |
          # Jestでシャーディング
          npx jest --shard=${{ matrix.shard }}/4

          # または環境変数で分割
          SHARD=${{ matrix.shard }} TOTAL_SHARDS=4 npm test
```

### 3. 動的並列化

```yaml
jobs:
  # 変更されたパッケージを検出
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      packages: ${{ steps.changes.outputs.packages }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - id: changes
        name: 変更されたパッケージを検出
        run: |
          # package.jsonが変更されたディレクトリを検出
          CHANGED=$(git diff --name-only ${{ github.event.before }} ${{ github.sha }} | \
            grep 'packages/.*/package.json' | \
            sed 's|packages/\(.*\)/package.json|\1|' | \
            jq -R -s -c 'split("\n")[:-1]')

          echo "packages=$CHANGED" >> $GITHUB_OUTPUT

  # 変更されたパッケージのみテスト
  test-changed:
    needs: detect-changes
    if: needs.detect-changes.outputs.packages != '[]'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        package: ${{ fromJson(needs.detect-changes.outputs.packages) }}
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - run: npm test --workspace=packages/${{ matrix.package }}
```

### 4. 並列化の制限

```yaml
strategy:
  max-parallel: 5  # 最大5つまで同時実行
  fail-fast: false  # 1つ失敗しても他を継続
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    node: [18, 20, 21]
```

---

## キャッシュ最適化

### 1. npm/yarn キャッシュ

```yaml
# ❌ キャッシュなし（3分）
- run: npm ci

# ✅ actions/setup-node のビルトインキャッシュ（30秒）
- uses: actions/setup-node@v4
  with:
    node-version: '20'
    cache: 'npm'

# ✅ 手動キャッシュ（細かい制御が可能）
- name: npm キャッシュ
  uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-

- run: npm ci
```

### 2. マルチレベルキャッシュ

```yaml
# レベル1: 依存関係キャッシュ
- name: 依存関係キャッシュ
  uses: actions/cache@v4
  with:
    path: |
      ~/.npm
      ~/.cache
    key: deps-${{ hashFiles('**/package-lock.json') }}
    restore-keys: deps-

# レベル2: ビルドキャッシュ
- name: ビルドキャッシュ
  uses: actions/cache@v4
  with:
    path: |
      .next/cache
      node_modules/.cache
    key: build-${{ hashFiles('**/*.ts', '**/*.tsx', '**/*.js') }}
    restore-keys: build-

# レベル3: テストキャッシュ
- name: Jestキャッシュ
  uses: actions/cache@v4
  with:
    path: .jest-cache
    key: jest-${{ hashFiles('**/*.test.ts') }}
    restore-keys: jest-
```

### 3. 条件付きキャッシュクリア

```yaml
- name: キャッシュキー生成
  id: cache-key
  run: |
    # PRラベルで強制的にキャッシュクリア
    if [[ "${{ contains(github.event.pull_request.labels.*.name, 'clear-cache') }}" == "true" ]]; then
      echo "key=build-${{ github.run_id }}" >> $GITHUB_OUTPUT
    else
      echo "key=build-${{ hashFiles('src/**') }}" >> $GITHUB_OUTPUT
    fi

- uses: actions/cache@v4
  with:
    path: dist/
    key: ${{ steps.cache-key.outputs.key }}
```

### 4. キャッシュサイズの最適化

```yaml
- name: 不要ファイルを除外してキャッシュ
  run: |
    # node_modulesから不要なファイルを削除
    find node_modules -name "*.md" -delete
    find node_modules -name "*.ts" -not -path "*/node_modules/*/@types/*" -delete
    find node_modules -name "test" -type d -exec rm -rf {} +
    find node_modules -name "tests" -type d -exec rm -rf {} +

- uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-modules-optimized-${{ hashFiles('**/package-lock.json') }}
```

### 5. キャッシュヒット率のモニタリング

```yaml
- name: キャッシュ統計
  run: |
    if [ -d ~/.npm ]; then
      echo "✅ npmキャッシュ: $(du -sh ~/.npm | cut -f1)"
      echo "cache-hit=true" >> $GITHUB_OUTPUT
    else
      echo "❌ npmキャッシュなし"
      echo "cache-hit=false" >> $GITHUB_OUTPUT
    fi
```

---

## 依存関係の最適化

### 1. 依存関係の削減

```json
// package.json の最適化

// ❌ 悪い例: 多すぎる依存関係
{
  "dependencies": {
    "lodash": "^4.17.21",
    "moment": "^2.29.4",
    "axios": "^1.6.0",
    "request": "^2.88.2"  // 非推奨
  }
}

// ✅ 良い例: 最小限の依存関係
{
  "dependencies": {
    "lodash-es": "^4.17.21",  // tree-shakable
    "date-fns": "^2.30.0",    // momentの代替（軽量）
    "axios": "^1.6.0"
    // request削除（axiosで代替）
  }
}
```

### 2. 開発依存関係の分離

```yaml
# 本番ビルドでは devDependencies をインストールしない
- name: 本番用依存関係のみインストール
  run: npm ci --omit=dev

# または
- run: npm ci --production
```

### 3. 依存関係の並列インストール

```yaml
# pnpmで高速化
- uses: pnpm/action-setup@v2
  with:
    version: 8

- uses: actions/setup-node@v4
  with:
    node-version: '20'
    cache: 'pnpm'

- run: pnpm install --frozen-lockfile

# または yarn berry
- run: yarn install --immutable --immutable-cache
```

### 4. lockfile の最適化

```bash
# package-lock.json のクリーンアップ
npm install
npm prune
npm dedupe

# または
npx npm-check-updates -u  # 依存関係を最新化
npm install
```

---

## テスト最適化

### 1. テストの選択的実行

```yaml
# 変更されたファイルに関連するテストのみ実行
- name: 関連テストのみ実行
  run: |
    # Gitで変更されたファイルを検出
    CHANGED_FILES=$(git diff --name-only ${{ github.event.before }} ${{ github.sha }})

    # 変更されたファイルに関連するテストを実行
    npx jest --findRelatedTests $CHANGED_FILES --coverage

# または Jestの--onlyChangedオプション
- run: npx jest --onlyChanged --coverage
```

### 2. テストのシャーディング

```yaml
# Playwrightでのシャーディング
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - run: npx playwright test --shard=${{ matrix.shard }}/4

# Jestでのシャーディング
- run: npx jest --shard=${{ matrix.shard }}/4 --maxWorkers=2
```

### 3. テストの並列実行制御

```json
// jest.config.js
module.exports = {
  // ワーカー数を最適化
  maxWorkers: process.env.CI ? 2 : '50%',

  // テストタイムアウト
  testTimeout: 10000,

  // カバレッジ収集の最適化
  collectCoverageFrom: [
    'src/**/*.{js,jsx,ts,tsx}',
    '!src/**/*.test.{js,jsx,ts,tsx}',
    '!src/**/*.stories.{js,jsx,ts,tsx}'
  ],

  // キャッシュ有効化
  cache: true,
  cacheDirectory: '.jest-cache'
}
```

### 4. 並列テスト実行

```yaml
# テストスイートを3つに分割
jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:unit

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:integration

  e2e-tests:
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:e2e
```

---

## Docker最適化

### 1. マルチステージビルド

```dockerfile
# ❌ 悪い例: 1.2GB
FROM node:20
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
CMD ["npm", "start"]

# ✅ 良い例: 150MB
# ステージ1: ビルド
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --production=false
COPY . .
RUN npm run build

# ステージ2: 実行
FROM node:20-alpine
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package*.json ./
CMD ["node", "dist/main.js"]
```

### 2. レイヤーキャッシュの活用

```dockerfile
# ✅ 依存関係を先にコピー（キャッシュ効率化）
FROM node:20-alpine
WORKDIR /app

# package.jsonのみ先にコピー
COPY package*.json ./
RUN npm ci --production

# ソースコードはその後
COPY . .
RUN npm run build

# ❌ 悪い例: 全てコピーしてからインストール
# COPY . .
# RUN npm ci  # ソースコード変更毎にキャッシュが無効化
```

### 3. BuildKitの活用

```yaml
# .github/workflows/docker-build.yml
- name: Dockerビルド（BuildKit有効）
  env:
    DOCKER_BUILDKIT: 1
  run: |
    docker build \
      --build-arg BUILDKIT_INLINE_CACHE=1 \
      --cache-from ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest \
      --tag ${{ env.IMAGE_NAME }}:${{ github.sha }} \
      .

# または docker buildx
- name: Docker Buildx セットアップ
  uses: docker/setup-buildx-action@v3

- name: ビルドとプッシュ
  uses: docker/build-push-action@v5
  with:
    context: .
    push: true
    tags: ${{ env.IMAGE_NAME }}:latest
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### 4. .dockerignoreの活用

```
# .dockerignore
node_modules
npm-debug.log
.git
.github
.gitignore
README.md
.env
.env.*
!.env.example
coverage
.next
dist
*.test.ts
*.spec.ts
```

---

## コスト最適化

### 1. 実行時間の最適化

```yaml
# GitHub Actions無料枠
# - Public: 無制限
# - Private: 2,000分/月（Freeプラン）

# ✅ 不要な実行を避ける
on:
  push:
    branches: [main, develop]
    paths-ignore:
      - '**.md'
      - 'docs/**'
      - '.github/ISSUE_TEMPLATE/**'
  pull_request:
    branches: [main]

# ✅ タイムアウト設定
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 10  # デフォルト360分から短縮
```

### 2. ランナーの選択

```yaml
# コストと速度のバランス

# ✅ 通常のジョブ: ubuntu-latest（安価）
jobs:
  test:
    runs-on: ubuntu-latest

# ✅ macOS必須の場合のみ
  ios-build:
    runs-on: macos-latest  # Linuxの10倍のコスト

# ✅ 並列化でトータルコストを削減
  test-sharded:
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    runs-on: ubuntu-latest
    # 4並列で実行時間1/4 = コスト同じ、スループット4倍
```

### 3. セルフホストランナーの活用

```yaml
# 頻繁に実行されるジョブはセルフホスト
jobs:
  lint:
    runs-on: self-hosted
    steps:
      - run: npm run lint

# 重いジョブやiOSビルドはGitHub-hosted
  ios-build:
    runs-on: macos-latest
    steps:
      - run: fastlane build
```

### 4. コスト可視化

```yaml
# ワークフロー実行時間を記録
- name: コスト計算
  run: |
    DURATION=$(($(date +%s) - ${{ github.event.workflow_run.created_at }}))
    MINUTES=$((DURATION / 60))

    # ランナー種別ごとのコスト（例）
    COST_PER_MIN=0.008  # ubuntu-latest
    TOTAL_COST=$(echo "$MINUTES * $COST_PER_MIN" | bc)

    echo "実行時間: ${MINUTES}分"
    echo "概算コスト: $${TOTAL_COST}"
```

---

## モニタリングとメトリクス

### 1. ビルド時間のトレンド分析

```yaml
# .github/workflows/metrics.yml
name: CI Metrics

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - name: メトリクス収集
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          # 過去100回の実行時間を取得
          gh run list \
            --workflow=ci.yml \
            --limit 100 \
            --json databaseId,conclusion,createdAt,updatedAt \
            > metrics.json

          # 統計情報を計算
          cat metrics.json | jq -r '
            map({
              duration: ((.updatedAt | fromdateiso8601) - (.createdAt | fromdateiso8601)),
              conclusion: .conclusion
            }) |
            {
              avg_duration: (map(.duration) | add / length),
              max_duration: (map(.duration) | max),
              min_duration: (map(.duration) | min),
              success_rate: ((map(select(.conclusion == "success")) | length) / length * 100)
            }
          '

      - name: Issueに投稿
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const metrics = JSON.parse(fs.readFileSync('metrics.json'));
            // Issueやコメントとして投稿
```

### 2. カスタムメトリクスの送信

```yaml
# Datadogへメトリクスを送信
- name: メトリクス送信
  run: |
    curl -X POST "https://api.datadoghq.com/api/v1/series" \
      -H "Content-Type: application/json" \
      -H "DD-API-KEY: ${{ secrets.DATADOG_API_KEY }}" \
      -d @- << EOF
    {
      "series": [{
        "metric": "ci.build.duration",
        "points": [[$(($(date +%s))), 250]],
        "type": "gauge",
        "tags": ["workflow:ci", "branch:main"]
      }]
    }
    EOF
```

### 3. ワークフローバッジの活用

```markdown
# README.md
![CI Status](https://github.com/owner/repo/workflows/CI/badge.svg)
![Build Time](https://img.shields.io/badge/build%20time-3m%2042s-brightgreen)
```

---

## 実践的な最適化事例

### ケーススタディ1: Next.jsアプリケーション（大規模）

**最適化前:**
```yaml
# 実行時間: 18分
jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm install          # 5分
      - run: npm run lint         # 2分
      - run: npm run type-check   # 3分
      - run: npm run build        # 6分
      - run: npm test             # 2分
```

**最適化後:**
```yaml
# 実行時間: 6分（67%削減）
jobs:
  setup:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'              # キャッシュ有効化

      - run: npm ci                 # 1分（キャッシュヒット時30秒）

      - uses: actions/cache@v4      # node_modulesをキャッシュ
        with:
          path: node_modules
          key: ${{ runner.os }}-modules-${{ hashFiles('package-lock.json') }}

  lint:                             # 並列実行
    needs: setup
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run lint           # 2分

  type-check:                       # 並列実行
    needs: setup
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run type-check     # 3分

  build:                            # 並列実行
    needs: setup
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v4      # ビルドキャッシュ
        with:
          path: .next/cache
          key: ${{ runner.os }}-nextjs-${{ hashFiles('src/**') }}
      - run: npm run build          # 4分（キャッシュヒット時2分）

  test:                             # 並列実行
    needs: setup
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1, 2]               # テストを2分割
    steps:
      - uses: actions/checkout@v4
      - run: npx jest --shard=${{ matrix.shard }}/2  # 1分
```

**成果:**
- 実行時間: 18分 → 6分（67%削減）
- 並列化により最長ジョブの時間が基準
- キャッシュヒット時は4分まで短縮可能

---

### ケーススタディ2: Monorepo（30パッケージ）

**課題:**
- 全パッケージのテストに35分かかる
- 変更されていないパッケージもテスト実行

**最適化戦略:**
```yaml
# .github/workflows/monorepo-ci.yml
jobs:
  # 変更検出
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      packages: ${{ steps.filter.outputs.changes }}
    steps:
      - uses: actions/checkout@v4

      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            pkg-auth: 'packages/auth/**'
            pkg-api: 'packages/api/**'
            pkg-ui: 'packages/ui/**'
            # ... 30パッケージ分

  # 変更されたパッケージのみテスト
  test:
    needs: detect-changes
    if: needs.detect-changes.outputs.packages != '[]'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        package: ${{ fromJson(needs.detect-changes.outputs.packages) }}
      max-parallel: 10  # 最大10並列
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci

      # Turborepoでキャッシュ活用
      - run: npx turbo run test --filter=${{ matrix.package }} --cache-dir=.turbo

      - uses: actions/cache@v4
        with:
          path: .turbo
          key: turbo-${{ github.sha }}
          restore-keys: turbo-
```

**成果:**
- 平均実行時間: 35分 → 8分（77%削減）
- 1パッケージのみ変更時: 2分
- Turborepoキャッシュで重複ビルドを回避

---

### ケーススタディ3: iOS アプリ（Fastlane）

**最適化前:**
```ruby
# fastlane/Fastfile
lane :test do
  scan(scheme: "MyApp")                  # 15分
end

lane :beta do
  match(type: "appstore")                # 2分
  increment_build_number                 # 1分
  build_app(scheme: "MyApp")             # 20分
  upload_to_testflight                   # 5分
end
```

**最適化後:**
```ruby
# fastlane/Fastfile
lane :test do
  # 並列テスト実行
  scan(
    scheme: "MyApp",
    parallel_testing: true,
    concurrent_workers: 4                # 15分 → 5分
  )
end

lane :beta do
  # 証明書キャッシュ
  match(
    type: "appstore",
    readonly: true,                      # 読み取り専用（高速）
    clone_branch_directly: true
  )

  # ビルド番号はGitHub Actionsで設定
  # increment_build_number（削除）

  # ビルド最適化
  build_app(
    scheme: "MyApp",
    export_method: "app-store",
    export_options: {
      compileBitcode: false,             # Bitcodeを無効化
      uploadSymbols: false,              # シンボルは後でアップ
      uploadBitcode: false
    }
  )

  # TestFlightアップロード（スキップ可能な処理を省略）
  upload_to_testflight(
    skip_waiting_for_build_processing: true,  # 処理待ちをスキップ
    skip_submission: true
  )
end
```

**GitHub Actions連携:**
```yaml
# .github/workflows/ios-ci.yml
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4

      # Xcodeキャッシュ
      - uses: actions/cache@v4
        with:
          path: |
            ~/Library/Developer/Xcode/DerivedData
            ~/.fastlane
          key: xcode-${{ hashFiles('Podfile.lock') }}

      # CocoaPodsキャッシュ
      - uses: actions/cache@v4
        with:
          path: Pods
          key: pods-${{ hashFiles('Podfile.lock') }}

      - run: bundle exec fastlane test
```

**成果:**
- テスト: 15分 → 5分（67%削減）
- ベータビルド: 28分 → 18分（36%削減）
- キャッシュヒット時はさらに3分短縮

---

### ケーススタディ4: Docker イメージビルド

**最適化前:**
```dockerfile
FROM node:20
WORKDIR /app
COPY . .
RUN npm install
RUN npm run build
# イメージサイズ: 1.2GB、ビルド時間: 8分
```

**最適化後:**
```dockerfile
# マルチステージビルド
FROM node:20-alpine AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci --production

FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV production

COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

USER node
CMD ["node", "dist/main.js"]
# イメージサイズ: 180MB、ビルド時間: 3分
```

**GitHub Actions:**
```yaml
- uses: docker/build-push-action@v5
  with:
    context: .
    cache-from: type=gha
    cache-to: type=gha,mode=max
    build-args: |
      BUILDKIT_INLINE_CACHE=1
```

**成果:**
- イメージサイズ: 1.2GB → 180MB（85%削減）
- ビルド時間: 8分 → 3分（キャッシュヒット時1分）

---

## まとめ

### 最適化チェックリスト

#### 即効性のある最適化（難易度: 低）
- [ ] actions/setup-nodeのキャッシュを有効化
- [ ] 並列実行可能なジョブを分離
- [ ] paths-ignoreで不要な実行を削減
- [ ] タイムアウト設定を追加
- [ ] .dockerignoreを作成

#### 中期的な最適化（難易度: 中）
- [ ] テストのシャーディング導入
- [ ] ビルドキャッシュの導入
- [ ] Docker マルチステージビルド
- [ ] 変更検出による選択的実行
- [ ] 依存関係の最適化

#### 長期的な最適化（難易度: 高）
- [ ] Monorepoツール導入（Turborepo/Nx）
- [ ] セルフホストランナーの導入
- [ ] カスタムビルドツールの開発
- [ ] マイクロサービス化によるビルド分離

### パフォーマンス目標

| プロジェクト規模 | 目標CI時間 | 推奨並列度 |
|----------------|-----------|----------|
| 小規模（~10ファイル） | 2-5分 | 2-3並列 |
| 中規模（~100ファイル） | 5-10分 | 4-6並列 |
| 大規模（~1000ファイル） | 10-15分 | 8-12並列 |
| 超大規模（Monorepo） | 15-20分 | 15-30並列 |

### 継続的改善

```bash
# 週次でメトリクス確認
gh run list --workflow=ci.yml --limit 50 --json conclusion,createdAt,updatedAt

# 月次で最適化効果を測定
# - 平均実行時間の推移
# - コスト削減額
# - 開発者体験の改善
```

---

**参考リンク:**
- [GitHub Actions: Best Practices](https://docs.github.com/en/actions/learn-github-actions/best-practices-for-github-actions)
- [Fastlane: Performance Optimization](https://docs.fastlane.tools/best-practices/)
- [Docker: Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)

**最終更新**: 2026年1月
