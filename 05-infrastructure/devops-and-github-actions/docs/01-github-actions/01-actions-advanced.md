# GitHub Actions 応用

> マトリクスビルド、キャッシュ戦略、アーティファクト管理、シークレット管理、環境(Environments)を駆使して高度なCI/CDパイプラインを構築する

## この章で学ぶこと

1. マトリクス戦略によるクロスプラットフォーム・マルチバージョンテストを設計できる
2. キャッシュとアーティファクトを適切に使い分けてパイプラインを高速化できる
3. シークレットと環境(Environments)による安全なデプロイ制御を実装できる
4. 再利用可能なワークフローとComposite Actionsでパイプラインを構造化できる
5. OIDC認証によるシークレットレスなクラウド連携を実装できる
6. Self-hosted runnerの構築・運用・セキュリティ対策を理解できる

---

## 1. マトリクス戦略

### 1.1 基本的なマトリクス

```yaml
name: Matrix CI
on: [push]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [18, 20, 22]
      fail-fast: false  # 1つ失敗しても他を継続
      max-parallel: 4   # 最大並列数

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
```

```
マトリクス展開の視覚化:

             node-18    node-20    node-22
ubuntu     [  Job 1  ] [  Job 2  ] [  Job 3  ]
macos      [  Job 4  ] [  Job 5  ] [  Job 6  ]
windows    [  Job 7  ] [  Job 8  ] [  Job 9  ]

合計: 3 x 3 = 9 ジョブが並列実行
```

### 1.2 高度なマトリクス: include / exclude

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        node-version: [18, 20]
        # 特定の組み合わせを追加
        include:
          - os: ubuntu-latest
            node-version: 22
            experimental: true
          - os: windows-latest
            node-version: 20
        # 特定の組み合わせを除外
        exclude:
          - os: macos-latest
            node-version: 18

    continue-on-error: ${{ matrix.experimental == true }}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test
```

### 1.3 動的マトリクス

```yaml
jobs:
  # マトリクスの値を動的に生成
  prepare:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
      - id: set-matrix
        run: |
          # 変更されたパッケージだけをテスト対象にする
          PACKAGES=$(find packages -name "package.json" -exec dirname {} \; | jq -R -s -c 'split("\n")[:-1]')
          echo "matrix={\"package\":$PACKAGES}" >> "$GITHUB_OUTPUT"

  test:
    needs: prepare
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.prepare.outputs.matrix) }}
    steps:
      - uses: actions/checkout@v4
      - run: cd ${{ matrix.package }} && npm test
```

### 1.4 変更検知ベースの動的マトリクス

モノレポ環境では変更されたパッケージのみをテスト対象にすることでCI時間を大幅に短縮できる。

```yaml
name: Monorepo Dynamic Matrix
on:
  pull_request:
    branches: [main]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.changes.outputs.matrix }}
      has_changes: ${{ steps.changes.outputs.has_changes }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全履歴取得（diff に必要）

      - id: changes
        run: |
          # main ブランチとの差分からパッケージを特定
          CHANGED_FILES=$(git diff --name-only origin/main...HEAD)
          echo "Changed files:"
          echo "$CHANGED_FILES"

          # 変更されたパッケージディレクトリを抽出
          PACKAGES=()
          for dir in packages/*/; do
            PKG_NAME=$(basename "$dir")
            if echo "$CHANGED_FILES" | grep -q "^packages/$PKG_NAME/"; then
              PACKAGES+=("$PKG_NAME")
            fi
          done

          # 共通ファイルの変更は全パッケージに影響
          if echo "$CHANGED_FILES" | grep -qE "^(package\.json|tsconfig\.base\.json|\.eslintrc)"; then
            PACKAGES=($(ls -d packages/*/ | xargs -n1 basename))
          fi

          if [ ${#PACKAGES[@]} -eq 0 ]; then
            echo "has_changes=false" >> "$GITHUB_OUTPUT"
            echo "matrix={\"package\":[]}" >> "$GITHUB_OUTPUT"
          else
            JSON=$(printf '%s\n' "${PACKAGES[@]}" | jq -R . | jq -s -c .)
            echo "has_changes=true" >> "$GITHUB_OUTPUT"
            echo "matrix={\"package\":$JSON}" >> "$GITHUB_OUTPUT"
          fi

  test:
    needs: detect-changes
    if: needs.detect-changes.outputs.has_changes == 'true'
    runs-on: ubuntu-latest
    strategy:
      matrix: ${{ fromJson(needs.detect-changes.outputs.matrix) }}
      fail-fast: false
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test --workspace=packages/${{ matrix.package }}
```

### 1.5 マトリクスの条件分岐パターン

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        node-version: [18, 20, 22]
        include:
          # Ubuntu + Node 22 のみでカバレッジを計測
          - os: ubuntu-latest
            node-version: 22
            coverage: true
          # Windows ではタイムアウトを長く
          - os: windows-latest
            node-version: 20
            timeout: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
      - run: npm ci
      - run: npm test

      # カバレッジは特定の組み合わせでのみ実行
      - name: Coverage
        if: matrix.coverage == true
        run: npm run test:coverage

      - name: Upload coverage
        if: matrix.coverage == true
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage/

      # OS固有のステップ
      - name: Windows specific cleanup
        if: runner.os == 'Windows'
        run: |
          # Windows固有のクリーンアップ
          Remove-Item -Recurse -Force node_modules\.cache -ErrorAction SilentlyContinue
        shell: pwsh
```

### 1.6 マトリクスの最大ジョブ数と制限

```
マトリクス制限:
┌─────────────────────────────────────────────────────┐
│ 最大ジョブ数: 256ジョブ/ワークフロー                      │
│ 最大マトリクスサイズ: 256組み合わせ                       │
│ 同時実行ランナー(Free): 20 (ubuntu) / 5 (macOS)       │
│ 同時実行ランナー(Team): 60 (ubuntu) / 5 (macOS)       │
│ 同時実行ランナー(Ent): 500 (ubuntu) / 50 (macOS)      │
│                                                       │
│ max-parallel の推奨値:                                 │
│   Free   → 3-5 (他のワークフローとの共有を考慮)          │
│   Team   → 10-20                                      │
│   Ent    → 50-100                                     │
└─────────────────────────────────────────────────────┘

コスト計算:
  マトリクス 3 OS x 3 Node x 10分/job = 90分の消費
  macOS は Linux の 10倍コスト
  → macOS のマトリクスを最小限に抑えることが重要
```

---

## 2. キャッシュ戦略

### 2.1 依存関係のキャッシュ

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # 方法1: setup-node の組み込みキャッシュ (推奨)
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'  # package-lock.json のハッシュがキー

      # 方法2: 明示的なキャッシュ (より細かい制御)
      - uses: actions/cache@v4
        id: npm-cache
        with:
          path: node_modules
          key: ${{ runner.os }}-node-${{ hashFiles('package-lock.json') }}
          restore-keys: |
            ${{ runner.os }}-node-

      - name: Install (キャッシュミス時のみ)
        if: steps.npm-cache.outputs.cache-hit != 'true'
        run: npm ci

      - run: npm test
```

### 2.2 キャッシュの仕組み

```
キャッシュのライフサイクル:

  1回目の実行 (キャッシュミス):
  ┌─────────┐    ┌──────────────┐    ┌──────────┐
  │ npm ci   │ →  │ node_modules │ →  │ キャッシュ │
  │ (90秒)   │    │  生成        │    │ に保存    │
  └─────────┘    └──────────────┘    └──────────┘

  2回目以降 (キャッシュヒット):
  ┌──────────┐    ┌──────────────┐
  │ キャッシュ │ →  │ node_modules │    npm ci スキップ!
  │ から復元  │    │ (3秒)        │    87秒の短縮
  └──────────┘    └──────────────┘

キーの仕組み:
  key: Linux-node-abc123def456
                   └── hashFiles('package-lock.json')

  package-lock.json が変わらない限り同じキーでヒット
  変わった場合は restore-keys で部分マッチ → npm ci 実行
```

### 2.3 言語別キャッシュ設定

```yaml
# Python (pip)
- uses: actions/setup-python@v5
  with:
    python-version: '3.12'
    cache: 'pip'

# Go
- uses: actions/setup-go@v5
  with:
    go-version: '1.22'
    cache: true

# Rust (手動キャッシュ)
- uses: actions/cache@v4
  with:
    path: |
      ~/.cargo/bin/
      ~/.cargo/registry/index/
      ~/.cargo/registry/cache/
      target/
    key: ${{ runner.os }}-cargo-${{ hashFiles('Cargo.lock') }}

# Docker レイヤーキャッシュ
- uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
```

### 2.4 高度なキャッシュパターン

```yaml
# パターン1: 複合キーによる段階的フォールバック
- uses: actions/cache@v4
  with:
    path: |
      node_modules
      ~/.npm
    key: ${{ runner.os }}-node-${{ matrix.node-version }}-${{ hashFiles('package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-${{ matrix.node-version }}-
      ${{ runner.os }}-node-
      ${{ runner.os }}-

# パターン2: ビルドキャッシュ (Next.js)
- uses: actions/cache@v4
  with:
    path: |
      ${{ github.workspace }}/.next/cache
    key: ${{ runner.os }}-nextjs-${{ hashFiles('package-lock.json') }}-${{ hashFiles('**/*.js', '**/*.jsx', '**/*.ts', '**/*.tsx') }}
    restore-keys: |
      ${{ runner.os }}-nextjs-${{ hashFiles('package-lock.json') }}-

# パターン3: Gradle ビルドキャッシュ
- uses: actions/cache@v4
  with:
    path: |
      ~/.gradle/caches
      ~/.gradle/wrapper
    key: ${{ runner.os }}-gradle-${{ hashFiles('**/*.gradle*', '**/gradle-wrapper.properties') }}
    restore-keys: |
      ${{ runner.os }}-gradle-
```

### 2.5 キャッシュのスコープとライフサイクル

```
キャッシュスコープの階層:

  ブランチ feature/xyz
       ↓ 検索
  ┌──────────────────────┐
  │ feature/xyz のキャッシュ │ ← まずここを検索
  └──────────┬───────────┘
             │ ミス
             ↓
  ┌──────────────────────┐
  │ main のキャッシュ       │ ← デフォルトブランチにフォールバック
  └──────────┬───────────┘
             │ ミス
             ↓
  ┌──────────────────────┐
  │ キャッシュなし → npm ci │ ← フル実行
  └──────────────────────┘

キャッシュのライフサイクル管理:
  - 最大サイズ: 10GB/リポジトリ
  - 未アクセス 7日で自動削除
  - FIFO (古いキャッシュから削除)
  - 手動クリア: GitHub UI または gh CLI
    gh cache list
    gh cache delete <key>
    gh actions-cache delete --all  # 全キャッシュ削除
```

### 2.6 キャッシュ効率の計測と最適化

```yaml
name: Cache Efficiency Report
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

jobs:
  report:
    runs-on: ubuntu-latest
    steps:
      - name: Check cache stats
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          echo "## Cache Usage Report"
          echo ""

          # キャッシュ一覧を取得
          CACHES=$(gh cache list --repo ${{ github.repository }} --json key,sizeInBytes,lastAccessedAt)

          # 合計サイズ
          TOTAL=$(echo "$CACHES" | jq '[.[].sizeInBytes] | add // 0')
          echo "Total cache size: $(numfmt --to=iec $TOTAL)"

          # 7日以上アクセスのないキャッシュ
          STALE=$(echo "$CACHES" | jq '[.[] | select(
            (.lastAccessedAt | fromdateiso8601) < (now - 604800)
          )] | length')
          echo "Stale caches (>7 days): $STALE"

          # サイズが大きいキャッシュ Top 5
          echo ""
          echo "### Top 5 largest caches:"
          echo "$CACHES" | jq -r 'sort_by(-.sizeInBytes) | .[0:5] | .[] |
            "\(.key): \(.sizeInBytes / 1048576 | floor)MB"'
```

---

## 3. アーティファクト管理

### 3.1 アーティファクトの用途

```
キャッシュ vs アーティファクト:

  キャッシュ (actions/cache):
  ┌──────────────────────────────────┐
  │ 用途: ビルド高速化 (依存関係等)    │
  │ 保持: ブランチスコープ、7日で削除   │
  │ 共有: 同一ワークフローの別ジョブ間  │
  │ 例: node_modules, .cache         │
  └──────────────────────────────────┘

  アーティファクト (actions/upload-artifact):
  ┌──────────────────────────────────┐
  │ 用途: ビルド成果物の保存・受渡し   │
  │ 保持: 設定可能 (1-90日)           │
  │ 共有: ジョブ間 + UIからダウンロード │
  │ 例: dist/, coverage/, logs/      │
  └──────────────────────────────────┘
```

### 3.2 ジョブ間でのアーティファクト受け渡し

```yaml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm run build

      - uses: actions/upload-artifact@v4
        with:
          name: build-output
          path: dist/
          retention-days: 1       # 短期保存
          if-no-files-found: error # ファイルがなければエラー

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
          path: dist/

      - run: ls -la dist/  # ビルド成果物を確認
      - run: ./deploy.sh dist/

  test-report:
    needs: build
    if: always()  # ビルドが失敗してもレポートは取得
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: build-output
          path: dist/
```

### 3.3 複数アーティファクトの統合

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm test

      # OS ごとにユニークな名前で保存
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.os }}
          path: |
            test-results/
            coverage/
          retention-days: 5

  merge-reports:
    needs: test
    if: always()
    runs-on: ubuntu-latest
    steps:
      # 全アーティファクトを一括ダウンロード
      - uses: actions/download-artifact@v4
        with:
          pattern: test-results-*
          merge-multiple: true   # v4 の新機能: 複数を統合
          path: all-results/

      - name: Generate combined report
        run: |
          echo "## Test Results Summary"
          for dir in all-results/*/; do
            OS_NAME=$(basename "$dir")
            PASS=$(grep -c "PASS" "$dir/results.txt" || true)
            FAIL=$(grep -c "FAIL" "$dir/results.txt" || true)
            echo "- $OS_NAME: $PASS passed, $FAIL failed"
          done

      - uses: actions/upload-artifact@v4
        with:
          name: combined-report
          path: combined-report.html
          retention-days: 30
```

### 3.4 アーティファクトのサイズ最適化

```yaml
steps:
  - name: Build
    run: npm run build

  # 悪い例: 不要なファイルを含む
  # - uses: actions/upload-artifact@v4
  #   with:
  #     name: build
  #     path: .   # リポジトリ全体！

  # 良い例: 必要なファイルのみ
  - uses: actions/upload-artifact@v4
    with:
      name: build
      path: |
        dist/
        !dist/**/*.map       # Source map を除外
        !dist/**/*.test.*    # テストファイルを除外
      compression-level: 9   # 最大圧縮
      retention-days: 1      # 短期保存

  # テスト結果: 要約のみアップロード
  - name: Minimize test output
    if: always()
    run: |
      # 大きなスナップショットファイルを除外
      find test-results -name "*.snap" -delete
      # HTMLレポートを圧縮
      tar czf test-report.tar.gz test-results/

  - uses: actions/upload-artifact@v4
    if: always()
    with:
      name: test-report
      path: test-report.tar.gz
      retention-days: 7
```

---

## 4. シークレット管理

### 4.1 シークレットの種類

```yaml
# Repository Secrets: リポジトリ単位
# Organization Secrets: 組織内の複数リポジトリで共有
# Environment Secrets: 環境(staging/prod)ごとに異なる値

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: production  # Environment Secrets を使用
    steps:
      # シークレットの参照
      - name: Deploy
        run: ./deploy.sh
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          API_KEY: ${{ secrets.API_KEY }}

      # 注意: シークレットはログにマスクされる
      - run: echo "${{ secrets.API_KEY }}"
        # 出力: ***
```

### 4.2 GITHUB_TOKEN

```yaml
jobs:
  pr-comment:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write  # 必要な権限のみ付与
    steps:
      - uses: actions/github-script@v7
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          script: |
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: 'CI passed! Ready for review.'
            })
```

### 4.3 GITHUB_TOKEN の権限モデル

```yaml
# GITHUB_TOKEN のデフォルト権限 (リポジトリ設定で制御)
# "Read repository contents and packages permissions" (推奨)
# または "Read and write permissions" (レガシー)

# ワークフローレベルで明示的に権限を指定
permissions:
  contents: read          # リポジトリのコンテンツ読み取り
  pull-requests: write    # PRへのコメント
  issues: write           # Issue操作
  packages: write         # パッケージ公開
  deployments: write      # デプロイメントステータス更新
  statuses: write         # コミットステータス更新
  checks: write           # チェック作成・更新
  id-token: write         # OIDC トークン取得

# ジョブレベルの権限指定（ワークフローレベルより優先）
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v4

  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      id-token: write     # AWS OIDC 認証に必要
    steps:
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/deploy-role
          aws-region: ap-northeast-1
```

### 4.4 OIDC (OpenID Connect) によるシークレットレス認証

```yaml
# 従来: 長期間有効なアクセスキーをシークレットに保存
# OIDC: 短期トークンを動的に取得（キーの管理不要）

name: Deploy with OIDC
on:
  push:
    branches: [main]

permissions:
  id-token: write   # OIDC トークンの取得に必須
  contents: read

jobs:
  deploy-aws:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # AWS OIDC 認証
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: arn:aws:iam::123456789012:role/github-actions-deploy
          aws-region: ap-northeast-1
          # audience: sts.amazonaws.com  # デフォルト

      - run: aws s3 sync ./dist s3://my-bucket/

  deploy-gcp:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # GCP OIDC 認証
      - uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: 'projects/123456/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
          service_account: 'deploy@my-project.iam.gserviceaccount.com'

      - uses: google-github-actions/deploy-cloudrun@v2
        with:
          service: my-service
          region: asia-northeast1
          image: gcr.io/my-project/my-app:${{ github.sha }}

  deploy-azure:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Azure OIDC 認証
      - uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - uses: azure/webapps-deploy@v3
        with:
          app-name: my-web-app
          package: ./dist
```

```
OIDC 認証のフロー:

  GitHub Actions Runner        GitHub OIDC Provider        Cloud Provider (AWS/GCP/Azure)
       │                              │                              │
       │  1. OIDC トークン要求          │                              │
       │─────────────────────────────→│                              │
       │                              │                              │
       │  2. JWT トークン発行           │                              │
       │  (sub: repo, ref等の情報)     │                              │
       │←─────────────────────────────│                              │
       │                              │                              │
       │  3. JWT で一時認証情報を要求                                    │
       │──────────────────────────────────────────────────────────────→│
       │                              │                              │
       │                              │  4. JWT を検証                │
       │                              │←─────────────────────────────│
       │                              │                              │
       │  5. 一時的なアクセストークンを返却                               │
       │←──────────────────────────────────────────────────────────────│
       │                              │                              │
       │  6. 一時トークンでAPIを呼び出し                                 │
       │──────────────────────────────────────────────────────────────→│

メリット:
  - 長期間有効なシークレットが不要
  - トークンは短時間（デフォルト1時間）で失効
  - ブランチやタグで認証範囲を制限可能
  - シークレットのローテーション不要
```

### 4.5 外部シークレットマネージャーとの連携

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      # HashiCorp Vault からシークレットを取得
      - uses: hashicorp/vault-action@v3
        with:
          url: https://vault.example.com
          method: jwt
          role: github-actions
          jwtGithubAudience: https://vault.example.com
          secrets: |
            secret/data/prod/db DB_PASSWORD | DB_PASSWORD ;
            secret/data/prod/api API_KEY | API_KEY

      - run: ./deploy.sh
        env:
          DB_PASSWORD: ${{ env.DB_PASSWORD }}
          API_KEY: ${{ env.API_KEY }}

      # AWS Secrets Manager からシークレットを取得
      - uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ap-northeast-1

      - name: Get secrets from AWS Secrets Manager
        run: |
          SECRET_JSON=$(aws secretsmanager get-secret-value \
            --secret-id prod/app/config \
            --query SecretString \
            --output text)
          echo "DB_HOST=$(echo $SECRET_JSON | jq -r .db_host)" >> "$GITHUB_ENV"
          # シークレット値をマスク
          echo "::add-mask::$(echo $SECRET_JSON | jq -r .db_password)"
          echo "DB_PASSWORD=$(echo $SECRET_JSON | jq -r .db_password)" >> "$GITHUB_ENV"
```

### 4.6 シークレットのセキュリティベストプラクティス

```
シークレット管理のベストプラクティス:

  1. 最小権限の原則
     ┌───────────────────────────────────┐
     │ - Repository > Organization の優先  │
     │ - Environment で環境ごとに分離       │
     │ - GITHUB_TOKEN は permissions で制限 │
     └───────────────────────────────────┘

  2. ローテーション
     ┌───────────────────────────────────┐
     │ - 90日以内のローテーションを推奨      │
     │ - OIDC で長期キーを排除              │
     │ - Vault/SecretsManager で自動化     │
     └───────────────────────────────────┘

  3. 監査
     ┌───────────────────────────────────┐
     │ - Audit log でシークレットアクセス確認│
     │ - GitHub Secret scanning を有効化   │
     │ - PRでシークレットが漏れないか確認    │
     └───────────────────────────────────┘

  4. フォーク対策
     ┌───────────────────────────────────┐
     │ - フォークからのPRにはシークレット    │
     │   が渡されない（pull_request）       │
     │ - pull_request_target は慎重に使用  │
     │ - approved フォークのみビルド許可     │
     └───────────────────────────────────┘
```

---

## 5. 環境 (Environments)

### 5.1 環境の設定

```yaml
jobs:
  deploy-staging:
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - run: ./deploy.sh staging

  deploy-production:
    needs: deploy-staging
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com
    # production 環境に以下を設定:
    # - Required reviewers: 2人の承認が必要
    # - Wait timer: 5分の待機時間
    # - Deployment branches: main のみ
    steps:
      - run: ./deploy.sh production
```

```
環境のデプロイフロー:

  PR merge to main
       │
       ↓
  ┌──────────┐
  │ CI Tests  │
  └────┬─────┘
       │
       ↓
  ┌──────────────────────┐
  │ Deploy to Staging     │ ← environment: staging
  │ (自動)                │
  └────┬─────────────────┘
       │
       ↓
  ┌──────────────────────┐
  │ 手動承認待ち           │ ← Required reviewers
  │ (Slack 通知)          │
  └────┬─────────────────┘
       │ 承認
       ↓
  ┌──────────────────────┐
  │ Wait Timer (5分)      │ ← 最終確認の猶予
  └────┬─────────────────┘
       │
       ↓
  ┌──────────────────────┐
  │ Deploy to Production  │ ← environment: production
  │ (承認後自動)           │
  └──────────────────────┘
```

### 5.2 マルチステージデプロイパイプライン

```yaml
name: Multi-stage Deploy
on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4

      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: type=sha,prefix=

      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-dev:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: development
      url: https://dev.example.com
    steps:
      - uses: actions/checkout@v4
      - run: |
          helm upgrade --install my-app ./charts/my-app \
            --set image.tag=${{ needs.build.outputs.image_tag }} \
            --namespace dev

  integration-tests:
    needs: deploy-dev
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run test:integration
        env:
          BASE_URL: https://dev.example.com

  deploy-staging:
    needs: integration-tests
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - uses: actions/checkout@v4
      - run: |
          helm upgrade --install my-app ./charts/my-app \
            --set image.tag=${{ needs.build.outputs.image_tag }} \
            --namespace staging

  smoke-tests:
    needs: deploy-staging
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          # ヘルスチェック
          for i in {1..30}; do
            STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://staging.example.com/health)
            if [ "$STATUS" = "200" ]; then
              echo "Health check passed"
              exit 0
            fi
            echo "Waiting... ($i/30)"
            sleep 10
          done
          echo "Health check failed"
          exit 1

      - run: npm run test:smoke
        env:
          BASE_URL: https://staging.example.com

  deploy-production:
    needs: smoke-tests
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://example.com
    steps:
      - uses: actions/checkout@v4
      - run: |
          helm upgrade --install my-app ./charts/my-app \
            --set image.tag=${{ needs.build.outputs.image_tag }} \
            --namespace production

  post-deploy-verification:
    needs: deploy-production
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npm run test:smoke
        env:
          BASE_URL: https://example.com

      # デプロイ通知
      - uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "Production deploy completed: ${{ needs.build.outputs.image_tag }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 5.3 環境変数と環境ごとの設定

```yaml
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: ${{ github.event.inputs.environment || 'staging' }}
    steps:
      - uses: actions/checkout@v4

      # 環境ごとの設定ファイルを使用
      - name: Load environment config
        run: |
          ENV_NAME=${{ github.event.inputs.environment || 'staging' }}
          if [ -f "config/${ENV_NAME}.env" ]; then
            # .env ファイルから環境変数を読み込み
            while IFS= read -r line; do
              # コメントと空行をスキップ
              [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
              echo "$line" >> "$GITHUB_ENV"
            done < "config/${ENV_NAME}.env"
          fi

      - name: Deploy with environment-specific config
        run: |
          echo "Deploying to: $DEPLOY_TARGET"
          echo "Replicas: $REPLICAS"
          echo "Region: $AWS_REGION"
          ./deploy.sh
```

### 5.4 カスタムデプロイ保護ルール

```
環境保護ルール一覧:

  ┌────────────────────────────────────────────────────┐
  │ Required reviewers (必須承認者)                      │
  │ - 最大6人まで指定                                    │
  │ - 1人以上の承認で続行                                 │
  │ - Teams の指定も可能                                 │
  ├────────────────────────────────────────────────────┤
  │ Wait timer (待機タイマー)                             │
  │ - 0-43200分（最大30日）                               │
  │ - 承認後に追加の待機時間                               │
  │ - キャンセル可能                                      │
  ├────────────────────────────────────────────────────┤
  │ Deployment branches (デプロイブランチ)                 │
  │ - All branches: 制限なし                              │
  │ - Protected branches: 保護ブランチのみ                │
  │ - Selected branches: パターンで指定                   │
  │   例: main, release/*                               │
  ├────────────────────────────────────────────────────┤
  │ Custom deployment protection rules (カスタムルール)    │
  │ - GitHub App による外部チェック                        │
  │ - 例: Datadog モニタリング確認                         │
  │ - 例: PagerDuty インシデント状態確認                   │
  │ - 例: 承認ワークフロー (ServiceNow等)                  │
  └────────────────────────────────────────────────────┘
```

---

## 6. 再利用可能なワークフロー

### 6.1 Reusable Workflow の定義

```yaml
# .github/workflows/reusable-build.yml
name: Reusable Build Workflow
on:
  workflow_call:
    inputs:
      node-version:
        description: 'Node.js version'
        required: false
        type: string
        default: '20'
      environment:
        description: 'Target environment'
        required: true
        type: string
      deploy:
        description: 'Whether to deploy'
        required: false
        type: boolean
        default: false
    secrets:
      npm-token:
        description: 'NPM authentication token'
        required: false
      deploy-key:
        description: 'Deployment SSH key'
        required: false
    outputs:
      artifact-name:
        description: 'Name of the uploaded artifact'
        value: ${{ jobs.build.outputs.artifact-name }}

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      artifact-name: ${{ steps.upload.outputs.artifact-name }}
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'npm'
          registry-url: 'https://registry.npmjs.org'

      - run: npm ci
        env:
          NODE_AUTH_TOKEN: ${{ secrets.npm-token }}

      - run: npm run build -- --mode ${{ inputs.environment }}

      - run: npm test

      - id: upload
        uses: actions/upload-artifact@v4
        with:
          name: build-${{ inputs.environment }}-${{ github.sha }}
          path: dist/
          retention-days: 7

      - name: Deploy
        if: inputs.deploy
        run: ./deploy.sh ${{ inputs.environment }}
        env:
          DEPLOY_KEY: ${{ secrets.deploy-key }}
```

### 6.2 Reusable Workflow の呼び出し

```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on:
  push:
    branches: [main]
  pull_request:

jobs:
  # PR時はビルドのみ
  build-pr:
    if: github.event_name == 'pull_request'
    uses: ./.github/workflows/reusable-build.yml
    with:
      environment: development
      deploy: false
    secrets:
      npm-token: ${{ secrets.NPM_TOKEN }}

  # main マージ時はステージングにデプロイ
  build-staging:
    if: github.ref == 'refs/heads/main'
    uses: ./.github/workflows/reusable-build.yml
    with:
      environment: staging
      deploy: true
    secrets:
      npm-token: ${{ secrets.NPM_TOKEN }}
      deploy-key: ${{ secrets.STAGING_DEPLOY_KEY }}

  # ステージング成功後、本番デプロイ
  build-production:
    needs: build-staging
    if: github.ref == 'refs/heads/main'
    uses: ./.github/workflows/reusable-build.yml
    with:
      environment: production
      deploy: true
    secrets:
      npm-token: ${{ secrets.NPM_TOKEN }}
      deploy-key: ${{ secrets.PRODUCTION_DEPLOY_KEY }}

  # 他のリポジトリのワークフローを呼び出し
  shared-security-scan:
    uses: my-org/shared-workflows/.github/workflows/security-scan.yml@v2
    with:
      scan-level: full
    secrets: inherit  # 全シークレットを継承
```

### 6.3 Composite Actions

```yaml
# .github/actions/setup-project/action.yml
name: 'Setup Project'
description: 'Setup Node.js, install dependencies, and prepare the project'
inputs:
  node-version:
    description: 'Node.js version'
    required: false
    default: '20'
  install-playwright:
    description: 'Install Playwright browsers'
    required: false
    default: 'false'
runs:
  using: 'composite'
  steps:
    - uses: actions/setup-node@v4
      with:
        node-version: ${{ inputs.node-version }}
        cache: 'npm'

    - name: Install dependencies
      shell: bash
      run: npm ci

    - name: Install Playwright
      if: inputs.install-playwright == 'true'
      shell: bash
      run: npx playwright install --with-deps chromium

    - name: Cache Playwright browsers
      if: inputs.install-playwright == 'true'
      uses: actions/cache@v4
      with:
        path: ~/.cache/ms-playwright
        key: playwright-${{ runner.os }}-${{ hashFiles('package-lock.json') }}
```

```yaml
# 使用例
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./.github/actions/setup-project
        with:
          install-playwright: 'true'
      - run: npm test
      - run: npx playwright test
```

### 6.4 Reusable Workflow vs Composite Actions 比較

```
Reusable Workflow vs Composite Actions:

┌─────────────────┬────────────────────────┬────────────────────────┐
│ 項目             │ Reusable Workflow      │ Composite Action       │
├─────────────────┼────────────────────────┼────────────────────────┤
│ 再利用の単位     │ ワークフロー全体         │ ステップの集合           │
│ ジョブの定義     │ 可能（複数ジョブ可）     │ 不可（ステップのみ）     │
│ シークレット     │ 明示的に渡す/inherit    │ 呼び出し元から自動継承   │
│ 環境の指定       │ ワークフロー内で指定     │ 呼び出し元ジョブで指定   │
│ ネスト           │ 最大4レベル             │ 最大10レベル            │
│ 条件分岐         │ ジョブ/ステップレベル    │ ステップレベル           │
│ 呼び出し構文     │ uses: org/repo/...@ref │ uses: ./.github/actions │
│ 典型的な用途     │ CI/CDパイプライン全体   │ セットアップ、共通処理   │
└─────────────────┴────────────────────────┴────────────────────────┘

使い分けの指針:
  - 共通のセットアップ手順 → Composite Action
  - 共通のCI/CDパイプライン → Reusable Workflow
  - 複数リポジトリで共有 → Reusable Workflow (別リポジトリ参照)
  - ローカルの繰り返し → Composite Action
```

---

## 7. Self-hosted Runner

### 7.1 Self-hosted Runner の構成

```
Runner アーキテクチャ:

  GitHub.com                           組織のインフラ
  ┌────────────┐                      ┌─────────────────────┐
  │ ワークフロー  │ ← Long Poll →      │ Self-hosted Runner  │
  │ キュー       │                      │                     │
  │             │  ジョブ割り当て         │ ┌─────────────────┐ │
  │             │─────────────────────→│ │ Runner Agent     │ │
  │             │                      │ │ (常時稼働)        │ │
  │             │  結果レポート          │ └────────┬────────┘ │
  │             │←─────────────────────│          │           │
  └────────────┘                      │ ┌────────↓────────┐ │
                                      │ │ ジョブ実行環境    │ │
                                      │ │ (Docker/VM)      │ │
                                      │ └─────────────────┘ │
                                      └─────────────────────┘
```

### 7.2 Kubernetes 上の Self-hosted Runner (ARC)

```yaml
# Actions Runner Controller (ARC) のデプロイ
# Helm で ARC をインストール
# helm install arc \
#   --namespace arc-systems \
#   oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller

# RunnerScaleSet の定義
apiVersion: actions.github.com/v1alpha1
kind: AutoscalingRunnerSet
metadata:
  name: my-runners
  namespace: arc-runners
spec:
  githubConfigUrl: "https://github.com/my-org"
  githubConfigSecret: github-config-secret
  maxRunners: 20
  minRunners: 2
  template:
    spec:
      containers:
        - name: runner
          image: ghcr.io/actions/actions-runner:latest
          resources:
            requests:
              cpu: "2"
              memory: "4Gi"
            limits:
              cpu: "4"
              memory: "8Gi"
          volumeMounts:
            - name: work
              mountPath: /home/runner/_work
            - name: docker-sock
              mountPath: /var/run/docker.sock
      volumes:
        - name: work
          emptyDir: {}
        - name: docker-sock
          hostPath:
            path: /var/run/docker.sock
```

```yaml
# ワークフローで Self-hosted Runner を使用
jobs:
  build:
    runs-on: arc-runner-set  # ARC で定義したランナーセット名
    steps:
      - uses: actions/checkout@v4
      - run: npm ci && npm run build

  # ラベルで特定のランナーを指定
  gpu-test:
    runs-on: [self-hosted, linux, gpu, a100]
    steps:
      - uses: actions/checkout@v4
      - run: python train.py --test

  # GitHub-hosted と Self-hosted の使い分け
  lint:
    runs-on: ubuntu-latest  # 軽量タスクは GitHub-hosted
    steps:
      - uses: actions/checkout@v4
      - run: npm run lint

  heavy-build:
    runs-on: [self-hosted, linux, x64, large]  # 重いタスクは Self-hosted
    steps:
      - uses: actions/checkout@v4
      - run: npm run build:all
```

### 7.3 Self-hosted Runner のセキュリティ

```
Self-hosted Runner のセキュリティ考慮事項:

  1. ランナーの分離
     ┌──────────────────────────────────────────┐
     │ - パブリックリポジトリでは使用しない         │
     │ - 一時的な環境（Ephemeral）を推奨           │
     │ - Docker-in-Docker で実行環境を分離         │
     │ - ジョブごとにクリーンな環境を保証            │
     └──────────────────────────────────────────┘

  2. エフェメラルランナー
     ┌──────────────────────────────────────────┐
     │ - --ephemeral フラグで1ジョブ後に終了      │
     │ - 前のジョブの残留物がない                   │
     │ - ARC の場合はデフォルトでエフェメラル        │
     └──────────────────────────────────────────┘

  3. ネットワーク分離
     ┌──────────────────────────────────────────┐
     │ - ランナーからの外部アクセスを制限            │
     │ - 必要なエンドポイントのみホワイトリスト       │
     │   - github.com                             │
     │   - api.github.com                         │
     │   - *.actions.githubusercontent.com         │
     │   - ghcr.io (コンテナレジストリ)             │
     │ - 内部ネットワークへのアクセスはVPCで制御     │
     └──────────────────────────────────────────┘

  4. 権限管理
     ┌──────────────────────────────────────────┐
     │ - ランナーは非root ユーザーで実行            │
     │ - sudo 権限は最小限に                       │
     │ - ファイルシステムの書き込み権限を制限         │
     │ - ネットワーク名前空間で分離                  │
     └──────────────────────────────────────────┘
```

---

## 8. ワークフローのデバッグとトラブルシューティング

### 8.1 デバッグログの有効化

```yaml
# 方法1: リポジトリのシークレットで設定
# ACTIONS_RUNNER_DEBUG = true
# ACTIONS_STEP_DEBUG = true

# 方法2: ワークフロー内でデバッグ情報を出力
steps:
  - name: Debug context
    run: |
      echo "Event name: ${{ github.event_name }}"
      echo "Ref: ${{ github.ref }}"
      echo "SHA: ${{ github.sha }}"
      echo "Actor: ${{ github.actor }}"
      echo "Workflow: ${{ github.workflow }}"
      echo "Run ID: ${{ github.run_id }}"
      echo "Run number: ${{ github.run_number }}"
      echo "Run attempt: ${{ github.run_attempt }}"

  - name: Dump contexts
    env:
      GITHUB_CONTEXT: ${{ toJson(github) }}
      JOB_CONTEXT: ${{ toJson(job) }}
      STEPS_CONTEXT: ${{ toJson(steps) }}
    run: |
      echo "::group::GitHub Context"
      echo "$GITHUB_CONTEXT"
      echo "::endgroup::"
      echo "::group::Job Context"
      echo "$JOB_CONTEXT"
      echo "::endgroup::"
      echo "::group::Steps Context"
      echo "$STEPS_CONTEXT"
      echo "::endgroup::"
```

### 8.2 よくあるエラーと解決策

```yaml
# エラー1: "Resource not accessible by integration"
# 原因: GITHUB_TOKEN の権限不足
# 解決:
permissions:
  contents: read
  pull-requests: write  # PR操作に必要な権限を追加

# エラー2: "No space left on device"
# 原因: ランナーのディスク容量不足
# 解決:
steps:
  - name: Free disk space
    run: |
      # 不要なプリインストールソフトウェアを削除
      sudo rm -rf /usr/share/dotnet
      sudo rm -rf /opt/ghc
      sudo rm -rf /usr/local/share/boost
      sudo rm -rf "$AGENT_TOOLSDIRECTORY"
      df -h

# エラー3: "The job was not started because recent account payments have failed"
# 原因: GitHub Actions の利用料金未払い
# 解決: 支払い情報を更新

# エラー4: キャッシュが復元されない
# 原因: キーの不一致、ブランチスコープの問題
# 解決:
steps:
  - name: Debug cache
    run: |
      echo "Expected key: ${{ runner.os }}-node-$(sha256sum package-lock.json | cut -d ' ' -f1)"
      # キャッシュの一覧を確認
      gh cache list --limit 20
    env:
      GH_TOKEN: ${{ github.token }}

# エラー5: "Error: Process completed with exit code 1"
# 原因: スクリプトの実行エラー
# 解決: set -euo pipefail とエラーハンドリング
steps:
  - name: Run with error handling
    run: |
      set -euo pipefail
      # エラー時の情報を出力
      trap 'echo "Error on line $LINENO. Exit code: $?"' ERR
      # 実際のコマンド
      npm run build 2>&1 | tee build.log
    shell: bash

# エラー6: タイムアウト
# 原因: ジョブの実行時間超過（デフォルト6時間）
# 解決:
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 30  # ジョブレベルのタイムアウト
    steps:
      - name: Long running task
        timeout-minutes: 10  # ステップレベルのタイムアウト
        run: npm run build
```

### 8.3 ワークフローの再実行とリトライ

```yaml
# 自動リトライの実装
jobs:
  flaky-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci

      # リトライ付きテスト実行
      - name: Run tests with retry
        uses: nick-fields/retry@v3
        with:
          timeout_minutes: 10
          max_attempts: 3
          retry_wait_seconds: 30
          command: npm test

      # シェルスクリプトでリトライ
      - name: Deploy with retry
        run: |
          MAX_RETRIES=3
          RETRY_DELAY=10
          for i in $(seq 1 $MAX_RETRIES); do
            echo "Attempt $i of $MAX_RETRIES"
            if ./deploy.sh; then
              echo "Deploy succeeded on attempt $i"
              exit 0
            fi
            if [ $i -lt $MAX_RETRIES ]; then
              echo "Retrying in ${RETRY_DELAY}s..."
              sleep $RETRY_DELAY
              RETRY_DELAY=$((RETRY_DELAY * 2))  # 指数バックオフ
            fi
          done
          echo "All retries failed"
          exit 1
```

---

## 9. パフォーマンス最適化

### 9.1 ワークフロー実行時間の分析

```
パフォーマンス最適化のチェックリスト:

  1. ボトルネック分析
     ┌─────────────────────────────────────────┐
     │ - Actions タブで各ステップの実行時間確認   │
     │ - 最も遅いステップを特定                   │
     │ - キャッシュヒット率を監視                  │
     └─────────────────────────────────────────┘

  2. 並列化
     ┌─────────────────────────────────────────┐
     │ - 独立したジョブを並列実行                  │
     │ - マトリクスで複数環境を並列テスト           │
     │ - ビルドとリントを並列化                    │
     └─────────────────────────────────────────┘

  3. キャッシュ活用
     ┌─────────────────────────────────────────┐
     │ - 依存関係キャッシュ                       │
     │ - ビルドキャッシュ (Next.js, Webpack等)   │
     │ - Docker レイヤーキャッシュ                │
     │ - テストのスナップショットキャッシュ          │
     └─────────────────────────────────────────┘

  4. 不要な処理の削減
     ┌─────────────────────────────────────────┐
     │ - パスフィルターで変更ファイルに基づき実行   │
     │ - 変更検知でスキップ判定                    │
     │ - 早期失敗 (lint → build → test)          │
     └─────────────────────────────────────────┘
```

### 9.2 効率的なパイプライン設計

```yaml
name: Optimized Pipeline
on:
  pull_request:
    branches: [main]
    paths-ignore:
      - '**/*.md'
      - 'docs/**'
      - '.github/ISSUE_TEMPLATE/**'

# 同一PRの同時実行を防止
concurrency:
  group: ${{ github.workflow }}-${{ github.head_ref || github.ref }}
  cancel-in-progress: true

jobs:
  # 最初にリント（最速で失敗を検知）
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
      - run: npm run typecheck

  # リント通過後にビルド
  build:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/
          retention-days: 1

  # ビルド成果物を使ってテスト（並列）
  unit-test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:unit -- --shard=1/2

  unit-test-2:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run test:unit -- --shard=2/2

  e2e-test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/
      - run: npx playwright install --with-deps chromium
      - run: npm run test:e2e
```

### 9.3 テストの並列分割(Sharding)

```yaml
jobs:
  test:
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

      # Jest のシャーディング
      - run: npx jest --shard=${{ matrix.shard }}/4

      # Playwright のシャーディング
      - run: npx playwright test --shard=${{ matrix.shard }}/4

      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-shard-${{ matrix.shard }}
          path: test-results/

  merge-results:
    needs: test
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: test-results-shard-*
          merge-multiple: true
          path: all-results/

      - name: Generate report
        run: |
          # テスト結果を統合してレポート生成
          npx jest-html-reporter --input all-results/ --output report.html
```

---

## 10. リリース自動化

### 10.1 セマンティックバージョニングとリリース

```yaml
name: Release
on:
  push:
    tags:
      - 'v*.*.*'

permissions:
  contents: write
  packages: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # 全タグ取得

      - name: Get version from tag
        id: version
        run: |
          TAG=${GITHUB_REF#refs/tags/}
          VERSION=${TAG#v}
          echo "tag=$TAG" >> "$GITHUB_OUTPUT"
          echo "version=$VERSION" >> "$GITHUB_OUTPUT"

          # 前のタグを取得（リリースノート生成用）
          PREV_TAG=$(git describe --tags --abbrev=0 HEAD^ 2>/dev/null || echo "")
          echo "prev_tag=$PREV_TAG" >> "$GITHUB_OUTPUT"

      - name: Generate changelog
        id: changelog
        run: |
          if [ -n "${{ steps.version.outputs.prev_tag }}" ]; then
            CHANGES=$(git log ${{ steps.version.outputs.prev_tag }}..HEAD \
              --pretty=format:"- %s (%h)" --no-merges)
          else
            CHANGES=$(git log --pretty=format:"- %s (%h)" --no-merges)
          fi

          # カテゴリ分け
          FEATURES=$(echo "$CHANGES" | grep -i "^- feat" || true)
          FIXES=$(echo "$CHANGES" | grep -i "^- fix" || true)
          OTHERS=$(echo "$CHANGES" | grep -iv "^- feat\|^- fix" || true)

          {
            echo "changelog<<EOF"
            [ -n "$FEATURES" ] && echo "### Features" && echo "$FEATURES" && echo ""
            [ -n "$FIXES" ] && echo "### Bug Fixes" && echo "$FIXES" && echo ""
            [ -n "$OTHERS" ] && echo "### Other Changes" && echo "$OTHERS" && echo ""
            echo "EOF"
          } >> "$GITHUB_OUTPUT"

      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npm run build

      # GitHub Release を作成
      - uses: softprops/action-gh-release@v2
        with:
          tag_name: ${{ steps.version.outputs.tag }}
          name: Release ${{ steps.version.outputs.version }}
          body: ${{ steps.changelog.outputs.changelog }}
          draft: false
          prerelease: ${{ contains(steps.version.outputs.tag, '-') }}
          files: |
            dist/*.tar.gz
            dist/*.zip

      # npm パッケージの公開
      - run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}

      # Docker イメージの公開
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ steps.version.outputs.version }}
            ghcr.io/${{ github.repository }}:latest
```

### 10.2 自動バージョンバンプとリリース

```yaml
name: Auto Release
on:
  push:
    branches: [main]

jobs:
  release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      # Conventional Commits からバージョンを自動決定
      - uses: googleapis/release-please-action@v4
        id: release
        with:
          release-type: node
          # package-name: my-package
          # changelog-types を指定してカスタマイズ

      # リリースが作成された場合のみ後続処理
      - name: Build and publish
        if: steps.release.outputs.release_created
        run: |
          echo "New version: ${{ steps.release.outputs.major }}.${{ steps.release.outputs.minor }}.${{ steps.release.outputs.patch }}"
          npm ci
          npm run build
          npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## 11. 高度なワークフローパターン

### 11.1 承認フロー付きデプロイ

```yaml
name: Deploy with Approval
on:
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deploy target'
        required: true
        type: choice
        options:
          - staging
          - production
      version:
        description: 'Version to deploy'
        required: true
        type: string
      dry_run:
        description: 'Dry run (no actual deploy)'
        required: false
        type: boolean
        default: false

jobs:
  validate:
    runs-on: ubuntu-latest
    outputs:
      image_exists: ${{ steps.check.outputs.exists }}
    steps:
      - name: Validate version
        id: check
        run: |
          # イメージの存在確認
          if docker manifest inspect ghcr.io/${{ github.repository }}:${{ inputs.version }} > /dev/null 2>&1; then
            echo "exists=true" >> "$GITHUB_OUTPUT"
          else
            echo "exists=false" >> "$GITHUB_OUTPUT"
            echo "::error::Image version ${{ inputs.version }} not found"
            exit 1
          fi

  deploy:
    needs: validate
    runs-on: ubuntu-latest
    environment:
      name: ${{ inputs.environment }}
      url: https://${{ inputs.environment == 'production' && '' || 'staging.' }}example.com
    steps:
      - uses: actions/checkout@v4

      - name: Deploy
        if: inputs.dry_run == false
        run: |
          echo "Deploying version ${{ inputs.version }} to ${{ inputs.environment }}"
          helm upgrade --install my-app ./charts/my-app \
            --set image.tag=${{ inputs.version }} \
            --namespace ${{ inputs.environment }}

      - name: Dry run
        if: inputs.dry_run == true
        run: |
          echo "DRY RUN: Would deploy ${{ inputs.version }} to ${{ inputs.environment }}"
          helm upgrade --install my-app ./charts/my-app \
            --set image.tag=${{ inputs.version }} \
            --namespace ${{ inputs.environment }} \
            --dry-run

      - name: Notify
        if: inputs.dry_run == false
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "${{ inputs.environment }} deployed: v${{ inputs.version }} by ${{ github.actor }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 11.2 条件付きワークフロー実行

```yaml
name: Conditional Workflow
on:
  pull_request:
    branches: [main]

jobs:
  # 変更されたファイルを検出
  changes:
    runs-on: ubuntu-latest
    outputs:
      frontend: ${{ steps.filter.outputs.frontend }}
      backend: ${{ steps.filter.outputs.backend }}
      infra: ${{ steps.filter.outputs.infra }}
      docs: ${{ steps.filter.outputs.docs }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            frontend:
              - 'packages/frontend/**'
              - 'packages/shared/**'
            backend:
              - 'packages/backend/**'
              - 'packages/shared/**'
            infra:
              - 'terraform/**'
              - 'k8s/**'
            docs:
              - 'docs/**'
              - '**/*.md'

  frontend-ci:
    needs: changes
    if: needs.changes.outputs.frontend == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd packages/frontend && npm ci && npm test

  backend-ci:
    needs: changes
    if: needs.changes.outputs.backend == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: cd packages/backend && npm ci && npm test

  infra-plan:
    needs: changes
    if: needs.changes.outputs.infra == 'true'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: hashicorp/setup-terraform@v3
      - run: cd terraform && terraform init && terraform plan

  # 全ジョブの結果を集約（required status check 用）
  ci-status:
    needs: [frontend-ci, backend-ci, infra-plan]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Check results
        run: |
          # スキップされたジョブは成功扱い
          RESULTS=("${{ needs.frontend-ci.result }}" "${{ needs.backend-ci.result }}" "${{ needs.infra-plan.result }}")
          for result in "${RESULTS[@]}"; do
            if [[ "$result" == "failure" || "$result" == "cancelled" ]]; then
              echo "CI failed: $result"
              exit 1
            fi
          done
          echo "All checks passed or were skipped"
```

### 11.3 ワークフロー間のトリガー連携

```yaml
# workflow_run: 他のワークフロー完了時にトリガー
name: Post-CI Actions
on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]
    branches: [main]

jobs:
  on-success:
    if: github.event.workflow_run.conclusion == 'success'
    runs-on: ubuntu-latest
    steps:
      - name: Download artifact from triggering workflow
        uses: actions/download-artifact@v4
        with:
          name: build-output
          github-token: ${{ secrets.GITHUB_TOKEN }}
          run-id: ${{ github.event.workflow_run.id }}

      - name: Deploy
        run: ./deploy.sh

  on-failure:
    if: github.event.workflow_run.conclusion == 'failure'
    runs-on: ubuntu-latest
    steps:
      - name: Notify failure
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "CI failed on main: ${{ github.event.workflow_run.html_url }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### 11.4 スケジュール実行とメンテナンスジョブ

```yaml
name: Scheduled Maintenance
on:
  schedule:
    # 毎日 AM 3:00 JST (UTC 18:00)
    - cron: '0 18 * * *'
  workflow_dispatch:  # 手動実行も可能

jobs:
  stale-cache-cleanup:
    runs-on: ubuntu-latest
    steps:
      - name: Clean up stale caches
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          echo "Cleaning up stale caches..."
          # マージ済みブランチのキャッシュを削除
          gh cache list --json key,ref | jq -r '.[] |
            select(.ref != "refs/heads/main") | .key' | while read key; do
            echo "Deleting cache: $key"
            gh cache delete "$key" || true
          done

  dependency-update-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm ci
      - name: Check for vulnerabilities
        run: |
          npm audit --audit-level=high || {
            echo "::warning::High severity vulnerabilities found"
          }
      - name: Check outdated packages
        run: npm outdated || true

  stale-branches:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Find stale branches
        run: |
          echo "## Stale Branches (>30 days)" >> $GITHUB_STEP_SUMMARY
          git for-each-ref --sort=committerdate refs/remotes/origin \
            --format='%(committerdate:iso8601) %(refname:short)' | \
            while read date branch; do
              AGE=$(( ($(date +%s) - $(date -d "$date" +%s)) / 86400 ))
              if [ $AGE -gt 30 ] && [ "$branch" != "origin/main" ]; then
                echo "- $branch ($AGE days old)" >> $GITHUB_STEP_SUMMARY
              fi
            done
```

---

## 12. 比較表

### 12.1 キャッシュ vs アーティファクト

| 項目 | キャッシュ | アーティファクト |
|---|---|---|
| 主な用途 | ビルド高速化 | 成果物の保存・受渡し |
| 保持期間 | 7日(アクセスなし) | 1-90日(設定可能) |
| スコープ | ブランチ(フォールバックあり) | ワークフロー実行単位 |
| UIダウンロード | 不可 | 可能 |
| 最大サイズ | 10GB/リポジトリ | 500MB/アーティファクト |
| 典型例 | node_modules, pip cache | dist/, coverage/, logs |

### 12.2 シークレットスコープ比較

| スコープ | 設定場所 | 共有範囲 | 用途 |
|---|---|---|---|
| Repository | リポジトリ設定 | 1リポジトリ | API キー、トークン |
| Environment | 環境設定 | 環境指定ジョブ | 環境別の認証情報 |
| Organization | 組織設定 | 指定リポジトリ群 | 共通認証情報 |
| GITHUB_TOKEN | 自動生成 | 当該ワークフロー | GitHub API 操作 |

### 12.3 認証方式比較

| 方式 | セキュリティ | 管理コスト | 適用場面 |
|---|---|---|---|
| Static Secrets | 低 | 高（ローテーション必要） | レガシーシステム |
| OIDC | 高 | 低 | AWS/GCP/Azure連携 |
| Vault連携 | 最高 | 中 | マルチクラウド・厳格な環境 |
| GITHUB_TOKEN | 中 | 最低 | GitHub API操作 |

### 12.4 GitHub-hosted vs Self-hosted Runner

| 項目 | GitHub-hosted | Self-hosted |
|---|---|---|
| セットアップ | 不要 | 必要 |
| メンテナンス | GitHub管理 | 自己管理 |
| コスト | 従量課金 | インフラ費用 |
| カスタマイズ | 限定的 | 自由 |
| セキュリティ | 分離済み | 自己管理 |
| パフォーマンス | 標準 | ハードウェア依存 |
| GPU対応 | なし(Larger Runnersで一部) | あり |
| 内部ネットワーク | 不可 | 可能 |
| 典型的用途 | 一般的なCI/CD | 特殊要件、大規模ビルド |

---

## 13. アンチパターン

### アンチパターン1: シークレットの間接的な漏洩

```yaml
# 悪い例: シークレットを環境変数経由でログに出力
steps:
  - run: |
      echo "Deploying with config:"
      env  # 全環境変数を出力 → シークレット漏洩!
    env:
      API_KEY: ${{ secrets.API_KEY }}

  - run: |
      curl -v https://api.example.com/deploy  # -v で認証ヘッダーが出力
    env:
      AUTH_TOKEN: ${{ secrets.AUTH_TOKEN }}

# 改善: ログへの出力を最小限に
steps:
  - run: |
      # シークレットは直接参照、ログには出さない
      ./deploy.sh  # スクリプト内でシークレットを使用
    env:
      API_KEY: ${{ secrets.API_KEY }}
```

### アンチパターン2: キャッシュの過信

```yaml
# 悪い例: キャッシュが壊れている可能性を考慮しない
steps:
  - uses: actions/cache@v4
    with:
      path: node_modules
      key: ${{ runner.os }}-node-${{ hashFiles('package-lock.json') }}
  # キャッシュヒットでも npm ci をスキップ
  # → 壊れたキャッシュで意味不明なエラー

# 改善: キャッシュミス時のフォールバックを用意
steps:
  - uses: actions/cache@v4
    id: cache
    with:
      path: node_modules
      key: ${{ runner.os }}-node-${{ hashFiles('package-lock.json') }}
  - if: steps.cache.outputs.cache-hit != 'true'
    run: npm ci
  # さらに、定期的にキャッシュをクリアする仕組みを用意
```

### アンチパターン3: ワークフローのモノリス化

```yaml
# 悪い例: 1つのワークフローに全てを詰め込む
name: Everything
on: [push]
jobs:
  do-everything:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint
      - run: npm run build
      - run: npm test
      - run: npm run e2e
      - run: ./deploy.sh staging
      - run: ./deploy.sh production
      # → 1ステップの失敗で全体が止まる
      # → 並列化できない
      # → 再実行は全体をやり直し

# 改善: ジョブを分割して並列化・依存関係を明確化
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: npm run lint

  build:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - run: npm run build

  test:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - run: npm test

  deploy-staging:
    needs: test
    environment: staging
    runs-on: ubuntu-latest
    steps:
      - run: ./deploy.sh staging
```

### アンチパターン4: 不必要な全体テスト

```yaml
# 悪い例: 全変更で全テストを実行
on:
  push:
    # READMEの変更でもCIが走る

# 改善: パスフィルターで不要な実行を防ぐ
on:
  push:
    paths-ignore:
      - '**/*.md'
      - 'docs/**'
      - '.github/ISSUE_TEMPLATE/**'
      - 'LICENSE'
      - '.gitignore'
```

### アンチパターン5: 長時間ジョブのタイムアウト未設定

```yaml
# 悪い例: タイムアウトなし（デフォルト6時間）
jobs:
  build:
    runs-on: ubuntu-latest  # 6時間分の料金が発生する可能性

# 改善: 適切なタイムアウトを設定
jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - run: npm run build
        timeout-minutes: 10
```

---

## 14. FAQ

### Q1: マトリクスで特定の組み合わせだけ異なる設定をするには？

`include` を使って特定の組み合わせに追加プロパティを設定する。例えば `include: [{os: ubuntu-latest, node-version: 22, coverage: true}]` とすれば、その組み合わせでのみ `matrix.coverage` が `true` になり、条件分岐に使える。

### Q2: キャッシュのキーが衝突したらどうなるか？

同じキーのキャッシュは上書きされない(イミュータブル)。異なるブランチのキャッシュは `restore-keys` のプレフィックスマッチでフォールバックする。デフォルトブランチのキャッシュは全ブランチからアクセス可能。キャッシュを強制更新したい場合はキーにランダム値やバージョン番号を含める。

### Q3: Environment の承認を自動化できるか？

GitHub API 経由でデプロイレビューを承認できるが、セキュリティ上の理由から完全自動化は推奨されない。代替手段として、ステージング環境でのスモークテスト成功を条件にした自動デプロイ(カナリー)を設計し、問題なければ承認者が1クリックで本番デプロイを承認するフローが現実的。

### Q4: OIDC とシークレットベースの認証、どちらを使うべきか？

可能な限り OIDC を推奨する。OIDC は長期間有効なシークレットを保存する必要がなく、短命トークンで認証するためセキュリティリスクが大幅に低減する。ただし、対応していないサービス(一部のSaaSプロバイダーなど)に対しては従来のシークレット方式を使う必要がある。OIDC に対応しているクラウドプロバイダー: AWS、GCP、Azure、HashiCorp Vault、Terraform Cloud。

### Q5: Self-hosted Runner をパブリックリポジトリで使っても安全か？

基本的に推奨されない。パブリックリポジトリではフォークからのPRが自由に作成でき、悪意あるコードがランナー上で実行される可能性がある。どうしても使用する場合は、エフェメラルランナー(ジョブ終了後に破棄)を使い、ランナーを完全に分離された環境で実行し、`pull_request` イベントでは自動実行しない設定にする。

### Q6: ワークフローの実行時間を効果的に短縮するには？

最も効果的な方法を優先度順に: (1) パスフィルターで不要な実行を防ぐ、(2) 依存関係のキャッシュを設定する、(3) ジョブを並列化する、(4) テストをシャーディングで分割する、(5) Larger Runner を使う、(6) ビルドキャッシュ(Next.js、Docker レイヤー等)を活用する。実際のワークフロー実行時間は Actions タブで確認でき、各ステップの時間を分析してボトルネックを特定する。

### Q7: Reusable Workflow の制限事項は？

主な制限: (1) ネストは最大4レベル、(2) 呼び出し元のワークフローから env コンテキストが継承されない、(3) strategy.matrix と reusable workflow は同じジョブで使えない(呼び出し元で matrix を定義する)、(4) `secrets: inherit` を使わない場合、シークレットは明示的に渡す必要がある、(5) 同一ワークフローファイルから複数回呼び出す場合は異なるジョブ名が必要。

### Q8: workflow_run と workflow_dispatch の違いは？

`workflow_run` は別のワークフローの完了を検知して自動的にトリガーされるイベント。`workflow_dispatch` はUI/API/gh CLIからの手動トリガー。`workflow_run` は CI 成功後のデプロイなどチェーン実行に使い、`workflow_dispatch` はオンデマンドのデプロイやメンテナンスタスクに使う。

### Q9: 大規模モノレポでの CI 最適化の定石は？

(1) `dorny/paths-filter` で変更検知し影響範囲のみテスト、(2) 動的マトリクスで変更されたパッケージだけをビルド対象にする、(3) Turborepo/Nx のリモートキャッシュを活用、(4) ビルド成果物をアーティファクトとしてジョブ間で共有、(5) `concurrency` で同一PRの重複実行を防止。これらを組み合わせることで CI 時間を 70-90% 削減した事例がある。

### Q10: GitHub Actions のコストを抑えるには？

(1) `concurrency` + `cancel-in-progress` で同一PRの古い実行をキャンセル、(2) パスフィルターで不要な実行を排除、(3) macOS ランナーの使用を最小限にする(Linuxの10倍のコスト)、(4) Self-hosted Runner で大量のビルドを処理、(5) キャッシュを効果的に使いビルド時間を短縮、(6) `timeout-minutes` を適切に設定してハングしたジョブの料金を防ぐ。GitHub Free プランは月2,000分(Linux)が無料で、これを超えるとPay-as-you-goになる。

---

## まとめ

| 項目 | 要点 |
|---|---|
| マトリクス | クロスプラットフォーム・マルチバージョンの並列テスト |
| キャッシュ | 依存関係を保存してビルド高速化(node_modules等) |
| アーティファクト | ビルド成果物のジョブ間受渡しとUI公開 |
| シークレット | 暗号化された機密情報、スコープで管理 |
| 環境 | デプロイ先の承認ルール・保護設定 |
| GITHUB_TOKEN | 自動生成、最小権限で permissions 指定 |
| OIDC | シークレットレス認証、短命トークンで安全 |
| Reusable Workflow | DRY原則のワークフロー再利用 |
| Composite Actions | 共通ステップのパッケージ化 |
| Self-hosted Runner | カスタムハードウェア、内部ネットワーク接続 |
| パフォーマンス | 並列化、キャッシュ、シャーディング、パスフィルター |
| リリース自動化 | タグベース、Conventional Commits、release-please |

---

## 次に読むべきガイド

- [再利用ワークフロー](./02-reusable-workflows.md) -- DRY原則に基づくワークフロー設計
- [CIレシピ集](./03-ci-recipes.md) -- 言語別の実践的CI設定
- [Actions セキュリティ](./04-security-actions.md) -- OIDC、依存ピン留め

---

## 参考文献

1. GitHub. "Using a matrix for your jobs." https://docs.github.com/en/actions/using-jobs/using-a-matrix-for-your-jobs
2. GitHub. "Caching dependencies to speed up workflows." https://docs.github.com/en/actions/using-workflows/caching-dependencies-to-speed-up-workflows
3. GitHub. "Using environments for deployment." https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment
4. GitHub. "Reusing workflows." https://docs.github.com/en/actions/using-workflows/reusing-workflows
5. GitHub. "Creating a composite action." https://docs.github.com/en/actions/creating-actions/creating-a-composite-action
6. GitHub. "About security hardening with OpenID Connect." https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect
7. GitHub. "About self-hosted runners." https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners
8. GitHub. "Usage limits, billing, and administration." https://docs.github.com/en/actions/learn-github-actions/usage-limits-billing-and-administration
9. Google Cloud. "Enabling keyless authentication from GitHub Actions." https://cloud.google.com/iam/docs/workload-identity-federation-with-deployment-pipelines
10. AWS. "Configuring OpenID Connect in Amazon Web Services." https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services
