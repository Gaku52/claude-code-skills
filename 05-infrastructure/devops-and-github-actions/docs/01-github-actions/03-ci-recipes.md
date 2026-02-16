# CI レシピ集

> Node.js、Python、Go、Rust、Docker の実践的なCI設定を網羅し、テスト・リント・ビルドの定番パターンを提供する

## この章で学ぶこと

1. 主要言語・フレームワーク別のCI設定パターンを把握する
2. テスト、リント、型チェック、セキュリティスキャンの統合方法を習得する
3. Docker イメージのビルド・プッシュの自動化を実装できる
4. モノレポ環境での効率的なCI構成を理解する
5. CI パイプラインの高速化テクニックを実践できる

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

### 1.3 pnpm を使ったモノレポ CI

```yaml
name: pnpm Monorepo CI
on: [push, pull_request]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2

      - uses: pnpm/action-setup@v4
        with:
          version: 9

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'pnpm'

      - run: pnpm install --frozen-lockfile

      - name: Lint
        run: pnpm run -r lint

      - name: Type check
        run: pnpm run -r typecheck

      - name: Test
        run: pnpm run -r test -- --coverage

      - name: Build
        run: pnpm run -r build
```

### 1.4 Next.js 専用 CI

```yaml
name: Next.js CI
on: [push, pull_request]

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

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npx tsc --noEmit

      - name: Unit tests
        run: npm test -- --coverage

      - name: Build
        run: npm run build
        env:
          NEXT_TELEMETRY_DISABLED: 1

      # Next.js のビルドキャッシュ
      - uses: actions/cache@v4
        with:
          path: .next/cache
          key: ${{ runner.os }}-nextjs-${{ hashFiles('**/package-lock.json') }}-${{ hashFiles('**/*.js', '**/*.jsx', '**/*.ts', '**/*.tsx') }}
          restore-keys: |
            ${{ runner.os }}-nextjs-${{ hashFiles('**/package-lock.json') }}-

      # Lighthouse CI
      - name: Lighthouse CI
        if: github.event_name == 'pull_request'
        run: |
          npm install -g @lhci/cli
          lhci autorun
        env:
          LHCI_GITHUB_APP_TOKEN: ${{ secrets.LHCI_GITHUB_APP_TOKEN }}
```

### 1.5 Vitest + React Testing Library CI

```yaml
name: Frontend CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci

      - name: Unit tests with coverage
        run: npx vitest run --coverage --reporter=json --outputFile=test-results.json

      - name: Upload coverage to Codecov
        if: github.event_name == 'push'
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: ./coverage/lcov.info

      - name: Comment PR with coverage
        if: github.event_name == 'pull_request'
        uses: davelosert/vitest-coverage-report-action@v2
```

### 1.6 Playwright E2E テスト CI

```yaml
name: E2E Tests
on:
  push:
    branches: [main]
  pull_request:

jobs:
  e2e:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'

      - run: npm ci

      - name: Install Playwright Browsers
        run: npx playwright install --with-deps

      - name: Build application
        run: npm run build

      - name: Run E2E tests
        run: npx playwright test
        env:
          CI: true
          BASE_URL: http://localhost:3000

      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-report
          path: playwright-report/
          retention-days: 7

      - name: Upload traces on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: playwright-traces
          path: test-results/
          retention-days: 7

  # テストのシャーディング（大規模プロジェクト向け）
  e2e-sharded:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        shard: [1/4, 2/4, 3/4, 4/4]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npx playwright install --with-deps
      - run: npm run build
      - name: Run E2E tests (shard ${{ matrix.shard }})
        run: npx playwright test --shard=${{ matrix.shard }}
```

### 1.7 npm パッケージ公開 CI

```yaml
name: Publish Package
on:
  release:
    types: [published]

permissions:
  contents: read
  id-token: write  # npm provenance に必要

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'

      - run: npm ci
      - run: npm test
      - run: npm run build

      - name: Publish to npm
        run: npm publish --provenance --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
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

### 2.3 uv を使った高速 Python CI

```yaml
name: Python CI (uv)
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-extras --dev

      - name: Lint
        run: |
          uv run ruff check .
          uv run ruff format --check .

      - name: Type check
        run: uv run mypy src/

      - name: Test
        run: uv run pytest --cov=src --cov-report=xml -v

      - name: Security check
        run: uv run bandit -r src/ -c pyproject.toml
```

### 2.4 Django プロジェクト CI

```yaml
name: Django CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: testuser
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: testdb
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        ports:
          - 6379:6379

    env:
      DATABASE_URL: postgres://testuser:testpass@localhost:5432/testdb
      REDIS_URL: redis://localhost:6379

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - run: pip install -r requirements.txt

      - name: Lint
        run: |
          ruff check .
          ruff format --check .

      - name: Type check
        run: mypy .

      - name: Run migrations
        run: python manage.py migrate

      - name: Run tests
        run: python manage.py test --parallel --verbosity=2

      - name: Check for missing migrations
        run: python manage.py makemigrations --check --dry-run
```

### 2.5 FastAPI プロジェクト CI

```yaml
name: FastAPI CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt

      - name: Lint and format
        run: |
          ruff check .
          ruff format --check .

      - name: Type check
        run: mypy app/

      - name: Test
        run: pytest --cov=app --cov-report=xml -v --tb=short
        env:
          DATABASE_URL: postgresql://postgres:test@localhost:5432/postgres
          TESTING: "1"

      - name: OpenAPI schema validation
        run: |
          python -c "
          from app.main import app
          import json
          schema = app.openapi()
          with open('openapi.json', 'w') as f:
              json.dump(schema, f, indent=2)
          print('OpenAPI schema generated successfully')
          "
```

### 2.6 PyPI パッケージ公開 CI

```yaml
name: Publish to PyPI
on:
  release:
    types: [published]

permissions:
  contents: read
  id-token: write  # Trusted Publisher に必要

jobs:
  publish:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/my-package
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install build tools
        run: pip install build

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        # OIDC (Trusted Publisher) なのでトークン不要
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

### 3.2 Go マルチプラットフォームビルド

```yaml
name: Go Release
on:
  push:
    tags: ['v*']

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        include:
          - goos: linux
            goarch: amd64
          - goos: linux
            goarch: arm64
          - goos: darwin
            goarch: amd64
          - goos: darwin
            goarch: arm64
          - goos: windows
            goarch: amd64
            ext: .exe

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: Build
        run: |
          GOOS=${{ matrix.goos }} GOARCH=${{ matrix.goarch }} \
          go build -ldflags "-s -w -X main.version=${{ github.ref_name }}" \
          -o myapp-${{ matrix.goos }}-${{ matrix.goarch }}${{ matrix.ext }} \
          ./cmd/myapp/

      - name: Upload release asset
        uses: softprops/action-gh-release@v2
        with:
          files: myapp-*
```

### 3.3 Go + Protocol Buffers CI

```yaml
name: Go + Protobuf CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-go@v5
        with:
          go-version: '1.22'

      - name: Install protoc
        uses: arduino/setup-protoc@v3
        with:
          version: '25.x'

      - name: Install protoc-gen-go
        run: |
          go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
          go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

      - name: Generate protobuf code
        run: |
          protoc --go_out=. --go_opt=paths=source_relative \
                 --go-grpc_out=. --go-grpc_opt=paths=source_relative \
                 proto/*.proto

      - name: Check generated code is up to date
        run: |
          git diff --exit-code || \
            (echo "Generated code is out of date. Run 'make proto' and commit." && exit 1)

      - name: Lint
        uses: golangci/golangci-lint-action@v4

      - name: Test
        run: go test -v -race ./...
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

### 4.2 Rust マルチプラットフォームリリース

```yaml
name: Rust Release
on:
  push:
    tags: ['v*']

permissions:
  contents: write

jobs:
  build:
    strategy:
      matrix:
        include:
          - target: x86_64-unknown-linux-gnu
            os: ubuntu-latest
          - target: aarch64-unknown-linux-gnu
            os: ubuntu-latest
          - target: x86_64-apple-darwin
            os: macos-latest
          - target: aarch64-apple-darwin
            os: macos-latest
          - target: x86_64-pc-windows-msvc
            os: windows-latest

    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}

      - name: Install cross-compilation tools
        if: matrix.target == 'aarch64-unknown-linux-gnu'
        run: |
          sudo apt-get update
          sudo apt-get install -y gcc-aarch64-linux-gnu

      - name: Build
        run: cargo build --release --target ${{ matrix.target }}

      - name: Package (Unix)
        if: runner.os != 'Windows'
        run: |
          cd target/${{ matrix.target }}/release
          tar -czf ../../../myapp-${{ matrix.target }}.tar.gz myapp

      - name: Package (Windows)
        if: runner.os == 'Windows'
        run: |
          cd target/${{ matrix.target }}/release
          7z a ../../../myapp-${{ matrix.target }}.zip myapp.exe

      - name: Upload release asset
        uses: softprops/action-gh-release@v2
        with:
          files: |
            myapp-*.tar.gz
            myapp-*.zip
```

### 4.3 Rust + WebAssembly CI

```yaml
name: Rust WASM CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: wasm32-unknown-unknown
          components: rustfmt, clippy

      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

      - name: Lint
        run: |
          cargo fmt --all -- --check
          cargo clippy --target wasm32-unknown-unknown -- -D warnings

      - name: Test (native)
        run: cargo test

      - name: Test (wasm)
        run: wasm-pack test --headless --chrome

      - name: Build WASM package
        run: wasm-pack build --target web --release

      - name: Upload WASM artifact
        uses: actions/upload-artifact@v4
        with:
          name: wasm-package
          path: pkg/
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

### 5.3 Dockerfile Lint + セキュリティスキャン

```yaml
name: Docker Security
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Hadolint (Dockerfile lint)
        uses: hadolint/hadolint-action@v3.1.0
        with:
          dockerfile: Dockerfile
          failure-threshold: warning

  scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build image for scanning
        run: docker build -t myapp:scan .

      - name: Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'myapp:scan'
          format: 'sarif'
          output: 'trivy-results.sarif'
          severity: 'CRITICAL,HIGH'

      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v3
        if: always()
        with:
          sarif_file: 'trivy-results.sarif'

      # Grype によるセカンドオピニオン
      - name: Grype vulnerability scanner
        uses: anchore/scan-action@v4
        with:
          image: 'myapp:scan'
          severity-cutoff: high
          fail-build: true
```

### 5.4 Docker Compose を使った統合テスト

```yaml
name: Integration Tests
on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start services
        run: docker compose -f docker-compose.test.yml up -d --wait

      - name: Run integration tests
        run: |
          docker compose -f docker-compose.test.yml exec -T app \
            npm run test:integration

      - name: Collect logs on failure
        if: failure()
        run: docker compose -f docker-compose.test.yml logs > docker-logs.txt

      - name: Upload logs
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: docker-logs
          path: docker-logs.txt

      - name: Cleanup
        if: always()
        run: docker compose -f docker-compose.test.yml down -v
```

---

## 6. 追加言語・フレームワーク CI

### 6.1 Java (Gradle) CI

```yaml
name: Java CI
on: [push, pull_request]

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-java@v4
        with:
          distribution: 'temurin'
          java-version: '21'
          cache: 'gradle'

      - name: Lint (Checkstyle)
        run: ./gradlew checkstyleMain checkstyleTest

      - name: Test
        run: ./gradlew test

      - name: Build
        run: ./gradlew build

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: build/reports/tests/
```

### 6.2 Terraform CI

```yaml
name: Terraform CI
on:
  pull_request:
    paths:
      - 'terraform/**'
      - '.github/workflows/terraform.yml'

permissions:
  contents: read
  pull-requests: write
  id-token: write

jobs:
  plan:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        environment: [staging, production]
    defaults:
      run:
        working-directory: terraform/environments/${{ matrix.environment }}
    steps:
      - uses: actions/checkout@v4

      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: '1.7'

      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets[format('{0}_AWS_ROLE_ARN', matrix.environment)] }}
          aws-region: ap-northeast-1

      - name: Terraform Format
        run: terraform fmt -check -recursive

      - name: Terraform Init
        run: terraform init

      - name: Terraform Validate
        run: terraform validate

      - name: Terraform Plan
        id: plan
        run: terraform plan -no-color -out=tfplan
        continue-on-error: true

      - name: Comment PR with plan
        uses: peter-evans/create-or-update-comment@v4
        with:
          issue-number: ${{ github.event.pull_request.number }}
          body: |
            ### Terraform Plan: `${{ matrix.environment }}`
            ```
            ${{ steps.plan.outputs.stdout }}
            ```

      - name: Terraform Plan Status
        if: steps.plan.outcome == 'failure'
        run: exit 1

      # tfsec セキュリティスキャン
      - name: tfsec security scan
        uses: aquasecurity/tfsec-action@v1.0.0
        with:
          working_directory: terraform/environments/${{ matrix.environment }}
```

### 6.3 Helm Chart CI

```yaml
name: Helm CI
on:
  push:
    paths:
      - 'charts/**'
  pull_request:
    paths:
      - 'charts/**'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: azure/setup-helm@v4
        with:
          version: 'v3.14.0'

      - name: Helm lint
        run: |
          for chart in charts/*/; do
            echo "Linting $chart"
            helm lint "$chart" --strict
          done

      - name: Template validation
        run: |
          for chart in charts/*/; do
            echo "Templating $chart"
            helm template test "$chart" --debug
          done

      - name: Kubeval validation
        run: |
          wget -q https://github.com/instrumenta/kubeval/releases/latest/download/kubeval-linux-amd64.tar.gz
          tar xf kubeval-linux-amd64.tar.gz
          for chart in charts/*/; do
            helm template test "$chart" | ./kubeval --strict
          done
```

---

## 7. CI パイプラインの構成比較

```
言語別パイプラインステージ:

Node.js:  Lint → TypeCheck → UnitTest → Build → E2E
Python:   Lint → TypeCheck → UnitTest → Security
Go:       Lint → Test(race) → Build → Vulncheck
Rust:     Fmt → Clippy → Test → Build → Audit
Docker:   Lint(hadolint) → Build → Scan(trivy) → Push
Java:     Lint → Test → Build → Publish
Terraform: Fmt → Validate → Plan → tfsec
```

### 7.1 言語別ツール比較

| 目的 | Node.js | Python | Go | Rust | Java |
|---|---|---|---|---|---|
| リンター | ESLint | Ruff | golangci-lint | Clippy | Checkstyle |
| フォーマッタ | Prettier | Ruff/Black | gofmt | rustfmt | Spotless |
| 型チェック | TypeScript | mypy/pyright | (組込み) | (組込み) | (組込み) |
| テスト | Jest/Vitest | pytest | go test | cargo test | JUnit |
| カバレッジ | c8/istanbul | coverage.py | go test -cover | cargo-tarpaulin | JaCoCo |
| セキュリティ | npm audit | bandit/safety | govulncheck | cargo-audit | SpotBugs |
| パッケージ管理 | npm/pnpm | pip/uv/poetry | go mod | cargo | Gradle/Maven |

### 7.2 CI 速度の目安

| 言語 | Lint | テスト | ビルド | 合計目標 |
|---|---|---|---|---|
| Node.js (中規模) | ~15s | ~60s | ~30s | < 3分 |
| Python (中規模) | ~10s | ~45s | N/A | < 2分 |
| Go (中規模) | ~20s | ~30s | ~15s | < 2分 |
| Rust (中規模) | ~30s | ~120s | ~180s | < 6分 |
| Docker ビルド | ~5s | N/A | ~120s | < 3分 |
| Java (中規模) | ~15s | ~60s | ~30s | < 3分 |

### 7.3 CI 高速化テクニック一覧

```
1. 依存関係キャッシュ
   - actions/cache または各セットアップアクションの cache オプション
   - キーには lockfile のハッシュを使用

2. 並列実行
   - ジョブを分割して並列実行（lint / test / build を別ジョブに）
   - テストのシャーディング（--shard オプション）
   - matrix strategy でマルチバージョンテスト

3. 差分検知
   - dorny/paths-filter で変更ファイルを検知
   - Turborepo / Nx の affected 機能
   - git diff による変更パッケージの特定

4. 早期失敗
   - Lint と型チェックを最初に実行（高速かつ問題を早期検出）
   - fail-fast: true（デフォルト）でマトリクスの早期打ち切り

5. ビルドキャッシュ
   - Docker: GHA キャッシュ（type=gha）
   - Next.js: .next/cache のキャッシュ
   - Rust: target/ ディレクトリのキャッシュ
   - Go: GOMODCACHE と GOCACHE のキャッシュ

6. concurrency 制御
   - 同一ブランチの古い実行をキャンセル
   - cancel-in-progress: true

7. 条件分岐
   - PR では E2E テストをスキップ
   - main push でのみ Docker ビルド・デプロイ
```

---

## 8. アンチパターン

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
  [ ] concurrency で古い実行をキャンセルしているか
```

### アンチパターン3: キャッシュキーの設計ミス

```yaml
# 悪い例: キャッシュキーが固定でヒットしない
- uses: actions/cache@v4
  with:
    path: node_modules
    key: node-modules-cache  # 常に同じキーなので更新されない

# 悪い例: キャッシュキーが細かすぎて再利用されない
- uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-node-${{ github.sha }}  # コミットごとに新規キャッシュ

# 良い例: lockfile ベースのキャッシュキー
- uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-node-${{ hashFiles('package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

### アンチパターン4: 秘密情報のCIログ露出

```yaml
# 悪い例: 環境変数の全出力
- run: env | sort  # シークレットがログに表示される可能性

# 悪い例: デバッグ出力
- run: echo "Token is ${{ secrets.API_TOKEN }}"  # マスクされるが避けるべき

# 良い例: 必要な情報のみ出力
- run: echo "Using API endpoint: ${{ vars.API_URL }}"
  # シークレットではなく Variables を使用
```

---

## 9. FAQ

### Q1: PR の CI と main ブランチの CI で異なる処理を実行するには？

`github.event_name` で分岐する。PR では `lint + test + build` まで、main push では追加で `e2e + docker build + deploy` を実行する。環境 (environment) を使って main ブランチのみデプロイを許可する設定も有効。

### Q2: テストの並列実行はどう設定するか？

Jest は `--shard` オプション、Playwright は `--shard` オプション、pytest は `pytest-xdist` の `-n auto` で並列化できる。CI ではマトリクス戦略と組み合わせて複数ジョブに分散させるのが効果的。

### Q3: セキュリティスキャンはCIに組み込むべきか？

はい。`npm audit`、`govulncheck`、`cargo audit`、`trivy` (Docker)、`Dependabot` は最低限導入すべき。ただし全てをブロッキングにすると開発速度が落ちるため、Critical/High のみブロック、Medium 以下は警告とする段階的アプローチを推奨する。

### Q4: CI の実行時間が10分を超える場合の対処法は？

まず最も時間がかかっているステップを特定する。一般的な対処法は、(1) 依存関係のキャッシュ見直し、(2) テストの並列化（シャーディング）、(3) 不要なステップの削除、(4) lint/type-check の先行実行による早期失敗、(5) Docker ビルドのレイヤーキャッシュ最適化。それでも改善しない場合は Larger Runner の利用も検討する。

### Q5: 複数の言語を使うプロジェクトのCIはどう構成するか？

言語ごとにジョブを分割し、paths フィルターで変更があった部分のみ実行する。共通のセットアップ処理は Composite Action に切り出す。全体の依存関係（フロントエンドのビルドがバックエンドのテストに必要、など）がある場合は `needs` で制御する。

### Q6: CI でデータベースを使うテストの実行方法は？

GitHub Actions の `services` 機能を使って、PostgreSQL、MySQL、Redis などのコンテナをサイドカーとして起動する。`options` で `--health-cmd` を設定し、データベースが起動完了してからテストを実行するようにする。Django の CI レシピ（セクション 2.4）を参照。

### Q7: Dependabot / Renovate の更新 PR に対するCIはどう設定するか？

通常のPR と同じ CI を実行するのが基本。加えて、`dependabot` ラベルが付いた PR に対して自動マージを設定すると運用が楽になる。

```yaml
name: Auto-merge Dependabot
on:
  pull_request:

permissions:
  contents: write
  pull-requests: write

jobs:
  auto-merge:
    if: github.actor == 'dependabot[bot]'
    runs-on: ubuntu-latest
    steps:
      - uses: dependabot/fetch-metadata@v2
        id: metadata

      # patch/minor のみ自動マージ
      - if: steps.metadata.outputs.update-type != 'version-update:semver-major'
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 10. CI メトリクスと可視化

### 10.1 CI パフォーマンスのトラッキング

```yaml
# .github/workflows/ci-metrics.yml — CI メトリクス収集
name: CI Metrics Collection

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    permissions:
      actions: read
    steps:
      - name: Collect workflow metrics
        uses: actions/github-script@v7
        with:
          script: |
            const run = context.payload.workflow_run;
            const jobs = await github.rest.actions.listJobsForWorkflowRun({
              owner: context.repo.owner,
              repo: context.repo.repo,
              run_id: run.id,
            });

            const metrics = {
              workflow_name: run.name,
              run_id: run.id,
              conclusion: run.conclusion,
              duration_seconds: Math.round(
                (new Date(run.updated_at) - new Date(run.run_started_at)) / 1000
              ),
              branch: run.head_branch,
              commit_sha: run.head_sha,
              jobs: jobs.data.jobs.map(job => ({
                name: job.name,
                conclusion: job.conclusion,
                duration_seconds: Math.round(
                  (new Date(job.completed_at) - new Date(job.started_at)) / 1000
                ),
                steps: job.steps?.map(step => ({
                  name: step.name,
                  conclusion: step.conclusion,
                  duration_seconds: step.completed_at && step.started_at
                    ? Math.round(
                        (new Date(step.completed_at) - new Date(step.started_at)) / 1000
                      )
                    : 0,
                })),
              })),
            };

            console.log(JSON.stringify(metrics, null, 2));

            // CloudWatch / Datadog / Grafana などに送信
            // await fetch('https://metrics.example.com/ci', {
            //   method: 'POST',
            //   body: JSON.stringify(metrics),
            // });
```

### 10.2 テストカバレッジの PR コメント

```yaml
# テストカバレッジをPRコメントに投稿
- name: Run Tests with Coverage
  run: npx vitest run --coverage --reporter=json --outputFile=coverage.json

- name: Post Coverage Comment
  if: github.event_name == 'pull_request'
  uses: actions/github-script@v7
  with:
    script: |
      const fs = require('fs');
      const coverage = JSON.parse(fs.readFileSync('coverage.json', 'utf8'));
      const summary = coverage.total;

      const table = [
        '| Metric | Coverage |',
        '|--------|----------|',
        `| Statements | ${summary.statements.pct}% |`,
        `| Branches | ${summary.branches.pct}% |`,
        `| Functions | ${summary.functions.pct}% |`,
        `| Lines | ${summary.lines.pct}% |`,
      ].join('\n');

      const body = `## Test Coverage Report\n\n${table}\n\n` +
        `${summary.lines.pct >= 80 ? '✅' : '⚠️'} ` +
        `Line coverage: ${summary.lines.pct}% (threshold: 80%)`;

      const { data: comments } = await github.rest.issues.listComments({
        owner: context.repo.owner,
        repo: context.repo.repo,
        issue_number: context.issue.number,
      });

      const existing = comments.find(c => c.body.includes('Test Coverage Report'));
      if (existing) {
        await github.rest.issues.updateComment({
          owner: context.repo.owner,
          repo: context.repo.repo,
          comment_id: existing.id,
          body,
        });
      } else {
        await github.rest.issues.createComment({
          owner: context.repo.owner,
          repo: context.repo.repo,
          issue_number: context.issue.number,
          body,
        });
      }
```

### 10.3 CI 失敗時の自動通知

```yaml
# CI 失敗時に Slack 通知
- name: Notify CI Failure
  if: failure() && github.ref == 'refs/heads/main'
  uses: slackapi/slack-github-action@v2.0.0
  with:
    webhook: ${{ secrets.SLACK_CI_WEBHOOK }}
    webhook-type: incoming-webhook
    payload: |
      {
        "text": "CI Failed on main",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*CI Failed* :red_circle:\n*Branch*: `${{ github.ref_name }}`\n*Commit*: `${{ github.sha }}`\n*Author*: ${{ github.actor }}\n*<${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}|View Run>*"
            }
          }
        ]
      }
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Node.js | ESLint + Prettier + TypeScript + Jest/Vitest |
| Python | Ruff + mypy + pytest + bandit |
| Go | golangci-lint + go test -race + govulncheck |
| Rust | clippy + rustfmt + cargo test + cargo audit |
| Docker | Buildx + GHA キャッシュ + マルチプラットフォーム |
| Java | Checkstyle + JUnit + Gradle |
| Terraform | fmt + validate + plan + tfsec |
| 共通原則 | Lint先行、キャッシュ活用、10分以内完了 |
| 高速化 | キャッシュ + 並列化 + 差分検知 + 早期失敗 |
| セキュリティ | Critical/High のみブロック、Medium 以下は警告 |
| メトリクス | CI の実行時間・成功率をトラッキングし改善を継続 |

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
5. Playwright. "CI integration." https://playwright.dev/docs/ci
6. Rust. "CI with GitHub Actions." https://doc.rust-lang.org/cargo/guide/continuous-integration.html
