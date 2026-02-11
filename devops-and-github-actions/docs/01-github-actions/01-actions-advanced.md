# GitHub Actions 応用

> マトリクスビルド、キャッシュ戦略、アーティファクト管理、シークレット管理、環境(Environments)を駆使して高度なCI/CDパイプラインを構築する

## この章で学ぶこと

1. マトリクス戦略によるクロスプラットフォーム・マルチバージョンテストを設計できる
2. キャッシュとアーティファクトを適切に使い分けてパイプラインを高速化できる
3. シークレットと環境(Environments)による安全なデプロイ制御を実装できる

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

---

## 6. 比較表

### 6.1 キャッシュ vs アーティファクト

| 項目 | キャッシュ | アーティファクト |
|---|---|---|
| 主な用途 | ビルド高速化 | 成果物の保存・受渡し |
| 保持期間 | 7日(アクセスなし) | 1-90日(設定可能) |
| スコープ | ブランチ(フォールバックあり) | ワークフロー実行単位 |
| UIダウンロード | 不可 | 可能 |
| 最大サイズ | 10GB/リポジトリ | 500MB/アーティファクト |
| 典型例 | node_modules, pip cache | dist/, coverage/, logs |

### 6.2 シークレットスコープ比較

| スコープ | 設定場所 | 共有範囲 | 用途 |
|---|---|---|---|
| Repository | リポジトリ設定 | 1リポジトリ | API キー、トークン |
| Environment | 環境設定 | 環境指定ジョブ | 環境別の認証情報 |
| Organization | 組織設定 | 指定リポジトリ群 | 共通認証情報 |
| GITHUB_TOKEN | 自動生成 | 当該ワークフロー | GitHub API 操作 |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: マトリクスで特定の組み合わせだけ異なる設定をするには？

`include` を使って特定の組み合わせに追加プロパティを設定する。例えば `include: [{os: ubuntu-latest, node-version: 22, coverage: true}]` とすれば、その組み合わせでのみ `matrix.coverage` が `true` になり、条件分岐に使える。

### Q2: キャッシュのキーが衝突したらどうなるか？

同じキーのキャッシュは上書きされない(イミュータブル)。異なるブランチのキャッシュは `restore-keys` のプレフィックスマッチでフォールバックする。デフォルトブランチのキャッシュは全ブランチからアクセス可能。キャッシュを強制更新したい場合はキーにランダム値やバージョン番号を含める。

### Q3: Environment の承認を自動化できるか？

GitHub API 経由でデプロイレビューを承認できるが、セキュリティ上の理由から完全自動化は推奨されない。代替手段として、ステージング環境でのスモークテスト成功を条件にした自動デプロイ(カナリー)を設計し、問題なければ承認者が1クリックで本番デプロイを承認するフローが現実的。

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
