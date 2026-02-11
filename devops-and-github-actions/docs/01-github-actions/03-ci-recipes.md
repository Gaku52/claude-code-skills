# CI レシピ集

> Node.js、Python、Go、Rust、Docker の実践的なCI設定を網羅し、テスト・リント・ビルドの定番パターンを提供する

## この章で学ぶこと

1. 主要言語・フレームワーク別のCI設定パターンを把握する
2. テスト、リント、型チェック、セキュリティスキャンの統合方法を習得する
3. Docker イメージのビルド・プッシュの自動化を実装できる

---

## 1. Node.js / TypeScript CI

### 1.1 フルスタック Node.js CI

```yaml
name: Node.js CI
on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read
  pull-requests: write

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci

      - name: Lint (ESLint + Prettier)
        run: |
          npm run lint
          npm run format:check

      - name: Type check
        run: npx tsc --noEmit

      - name: Unit tests
        run: npm test -- --coverage --coverageReporters=json-summary

      - name: Build
        run: npm run build

      - name: E2E tests (Playwright)
        if: github.event_name == 'push'
        run: |
          npx playwright install --with-deps chromium
          npm run test:e2e
```

### 1.2 モノレポ (Turborepo) CI

```yaml
name: Monorepo CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2  # 差分検知に必要

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci

      # Turborepo のリモートキャッシュ
      - name: Run affected checks
        run: npx turbo run lint typecheck test build --filter='...[HEAD~1]'
        env:
          TURBO_TOKEN: ${{ secrets.TURBO_TOKEN }}
          TURBO_TEAM: ${{ vars.TURBO_TEAM }}
```

---

## 2. Python CI

### 2.1 Python プロジェクト CI

```yaml
name: Python CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint (Ruff)
        run: |
          ruff check .
          ruff format --check .

      - name: Type check (mypy)
        run: mypy src/

      - name: Test (pytest)
        run: pytest --cov=src --cov-report=xml -v

      - name: Security check (bandit)
        run: bandit -r src/ -c pyproject.toml
```

### 2.2 Poetry を使った Python CI

```yaml
name: Python CI (Poetry)
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Poetry
        run: pipx install poetry

      - name: Configure Poetry
        run: poetry config virtualenvs.in-project true

      - uses: actions/cache@v4
        with:
          path: .venv
          key: ${{ runner.os }}-poetry-${{ hashFiles('poetry.lock') }}

      - run: poetry install --no-interaction
      - run: poetry run ruff check .
      - run: poetry run mypy src/
      - run: poetry run pytest --cov
```

---

## 3. Go CI

### 3.1 Go プロジェクト CI

```yaml
name: Go CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: Lint (golangci-lint)
        uses: golangci/golangci-lint-action@v4
        with:
          version: latest

      - name: Test
        run: go test -v -race -coverprofile=coverage.out ./...

      - name: Build
        run: go build -v ./...

      - name: Security (govulncheck)
        run: |
          go install golang.org/x/vuln/cmd/govulncheck@latest
          govulncheck ./...
```

---

## 4. Rust CI

### 4.1 Rust プロジェクト CI

```yaml
name: Rust CI
on: [push, pull_request]

env:
  CARGO_TERM_COLOR: always

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy

      - uses: actions/cache@v4
        with:
          path: |
            ~/.cargo/bin/
            ~/.cargo/registry/
            ~/.cargo/git/
            target/
          key: ${{ runner.os }}-cargo-${{ hashFiles('Cargo.lock') }}

      - name: Format check
        run: cargo fmt --all -- --check

      - name: Clippy (lint)
        run: cargo clippy --all-targets --all-features -- -D warnings

      - name: Test
        run: cargo test --all-features

      - name: Build (release)
        run: cargo build --release

      - name: Security audit
        run: |
          cargo install cargo-audit
          cargo audit
```

---

## 5. Docker CI

### 5.1 Docker ビルド・プッシュ

```yaml
name: Docker Build
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:

permissions:
  contents: read
  packages: write

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - uses: docker/login-action@v3
        if: github.event_name != 'pull_request'
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - uses: docker/build-push-action@v5
        with:
          context: .
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

### 5.2 マルチステージ Dockerfile

```dockerfile
# ビルドステージ
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --production=false
COPY . .
RUN npm run build

# 実行ステージ
FROM node:20-alpine AS runner
WORKDIR /app
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001
COPY --from=builder --chown=nextjs:nodejs /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
USER nextjs
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

---

## 6. CI パイプラインの構成比較

```
言語別パイプラインステージ:

Node.js:  Lint → TypeCheck → UnitTest → Build → E2E
Python:   Lint → TypeCheck → UnitTest → Security
Go:       Lint → Test(race) → Build → Vulncheck
Rust:     Fmt → Clippy → Test → Build → Audit
Docker:   Lint(hadolint) → Build → Scan(trivy) → Push
```

### 6.1 言語別ツール比較

| 目的 | Node.js | Python | Go | Rust |
|---|---|---|---|---|
| リンター | ESLint | Ruff | golangci-lint | Clippy |
| フォーマッタ | Prettier | Ruff/Black | gofmt | rustfmt |
| 型チェック | TypeScript | mypy/pyright | (組込み) | (組込み) |
| テスト | Jest/Vitest | pytest | go test | cargo test |
| カバレッジ | c8/istanbul | coverage.py | go test -cover | cargo-tarpaulin |
| セキュリティ | npm audit | bandit/safety | govulncheck | cargo-audit |

### 6.2 CI 速度の目安

| 言語 | Lint | テスト | ビルド | 合計目標 |
|---|---|---|---|---|
| Node.js (中規模) | ~15s | ~60s | ~30s | < 3分 |
| Python (中規模) | ~10s | ~45s | N/A | < 2分 |
| Go (中規模) | ~20s | ~30s | ~15s | < 2分 |
| Rust (中規模) | ~30s | ~120s | ~180s | < 6分 |
| Docker ビルド | ~5s | N/A | ~120s | < 3分 |

---

## 7. アンチパターン

### アンチパターン1: テストなしのCI

```yaml
# 悪い例: ビルドだけで "CI通りました"
jobs:
  ci:
    steps:
      - run: npm run build
      # テストなし → ビルドが通れば OK ではない

# 改善: テストピラミッドに基づくステージ構成
jobs:
  ci:
    steps:
      - run: npm run lint
      - run: npm run type-check
      - run: npm test -- --coverage
      - run: npm run build
      # lint → type → test → build の順で高速フェイル
```

### アンチパターン2: 遅いCIの放置

```
問題:
  CI が 15分以上かかり、開発者が CI の結果を待たずにマージしてしまう。

改善チェックリスト:
  [ ] 依存関係のキャッシュを設定しているか
  [ ] テストを並列実行しているか (--shard, -j)
  [ ] 不要なステップを削除したか
  [ ] lint / type-check を最初に実行しているか
  [ ] Docker レイヤーキャッシュを使っているか
  [ ] 変更されたファイルのみテストしているか (affected)
```

---

## 8. FAQ

### Q1: PR の CI と main ブランチの CI で異なる処理を実行するには？

`github.event_name` で分岐する。PR では `lint + test + build` まで、main push では追加で `e2e + docker build + deploy` を実行する。環境 (environment) を使って main ブランチのみデプロイを許可する設定も有効。

### Q2: テストの並列実行はどう設定するか？

Jest は `--shard` オプション、Playwright は `--shard` オプション、pytest は `pytest-xdist` の `-n auto` で並列化できる。CI ではマトリクス戦略と組み合わせて複数ジョブに分散させるのが効果的。

### Q3: セキュリティスキャンはCIに組み込むべきか？

はい。`npm audit`、`govulncheck`、`cargo audit`、`trivy` (Docker)、`Dependabot` は最低限導入すべき。ただし全てをブロッキングにすると開発速度が落ちるため、Critical/High のみブロック、Medium 以下は警告とする段階的アプローチを推奨する。

---

## まとめ

| 項目 | 要点 |
|---|---|
| Node.js | ESLint + Prettier + TypeScript + Jest/Vitest |
| Python | Ruff + mypy + pytest + bandit |
| Go | golangci-lint + go test -race + govulncheck |
| Rust | clippy + rustfmt + cargo test + cargo audit |
| Docker | Buildx + GHA キャッシュ + マルチプラットフォーム |
| 共通原則 | Lint先行、キャッシュ活用、10分以内完了 |

---

## 次に読むべきガイド

- [Actions セキュリティ](./04-security-actions.md) -- サプライチェーン保護
- [デプロイ戦略](../02-deployment/00-deployment-strategies.md) -- CIの次はCD
- [Actions 応用](./01-actions-advanced.md) -- マトリクス、キャッシュの詳細

---

## 参考文献

1. GitHub. "Building and testing Node.js." https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-nodejs
2. GitHub. "Building and testing Python." https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python
3. GitHub. "Publishing Docker images." https://docs.github.com/en/actions/publishing-packages/publishing-docker-images
4. Docker. "Build with GitHub Actions." https://docs.docker.com/build/ci/github-actions/
