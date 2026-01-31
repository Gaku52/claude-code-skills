# CI/CD ベストプラクティス集

## GitHub Actions ベストプラクティス

### 1. ワークフロー設計

#### ✅ DO: 関心の分離
```yaml
# ❌ 悪い例: 全てを1つのジョブに詰め込む
jobs:
  all-in-one:
    steps:
      - run: npm run lint
      - run: npm test
      - run: npm run build
      - run: npm run deploy

# ✅ 良い例: 独立したジョブに分割
jobs:
  lint:
    steps:
      - run: npm run lint

  test:
    steps:
      - run: npm test

  build:
    needs: [lint, test]
    steps:
      - run: npm run build

  deploy:
    needs: build
    steps:
      - run: npm run deploy
```

**理由:**
- 並列実行で時間短縮
- 失敗箇所の特定が容易
- 部分的な再実行が可能

#### ✅ DO: 適切なトリガー設定
```yaml
# ❌ 悪い例: 全てのpushで実行
on: push

# ✅ 良い例: 必要な時だけ実行
on:
  push:
    branches: [main, develop]
    paths-ignore:
      - '**.md'
      - 'docs/**'
  pull_request:
    branches: [main]
```

#### ✅ DO: タイムアウト設定
```yaml
jobs:
  test:
    timeout-minutes: 10  # デフォルト360分は長すぎる
    steps:
      - name: Run tests
        timeout-minutes: 5  # ステップレベルでも設定可能
        run: npm test
```

### 2. セキュリティ

#### ✅ DO: Secrets の適切な管理
```yaml
# ❌ 悪い例: ハードコード
- run: echo "API_KEY=sk-1234567890" >> .env

# ✅ 良い例: Secrets使用
- run: echo "API_KEY=${{ secrets.API_KEY }}" >> .env

# ✅ さらに良い例: Secretsを環境変数経由
- name: Build
  env:
    API_KEY: ${{ secrets.API_KEY }}
  run: npm run build
```

#### ✅ DO: 権限の最小化
```yaml
# ワークフロー全体で権限を制限
permissions:
  contents: read
  pull-requests: write

jobs:
  deploy:
    # ジョブごとに必要な権限のみ付与
    permissions:
      contents: write
      id-token: write
```

#### ✅ DO: サードパーティActionのバージョン固定
```yaml
# ❌ 悪い例: 最新版を使用
- uses: actions/checkout@v4

# ✅ 良い例: コミットSHAで固定
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11  # v4.1.1

# ✅ ベスト: タグ + コメントでSHA記載
- uses: actions/checkout@v4.1.1  # b4ffde65
```

### 3. パフォーマンス

#### ✅ DO: キャッシュの活用
```yaml
# npmキャッシュ（自動）
- uses: actions/setup-node@v4
  with:
    node-version: '20'
    cache: 'npm'

# ビルドキャッシュ
- uses: actions/cache@v4
  with:
    path: |
      .next/cache
      node_modules/.cache
    key: build-${{ hashFiles('src/**') }}
```

#### ✅ DO: 並列実行
```yaml
# マトリックス戦略で並列化
jobs:
  test:
    strategy:
      matrix:
        node-version: [18, 20, 21]
        os: [ubuntu-latest, windows-latest]
      max-parallel: 6
      fail-fast: false
```

#### ✅ DO: 条件付き実行で無駄を削減
```yaml
jobs:
  deploy:
    # mainブランチのみ
    if: github.ref == 'refs/heads/main'

  test-e2e:
    # PRでlabelがある場合のみ
    if: |
      github.event_name == 'pull_request' &&
      contains(github.event.pull_request.labels.*.name, 'run-e2e')
```

### 4. 可読性・保守性

#### ✅ DO: 分かりやすい名前
```yaml
# ❌ 悪い例
jobs:
  job1:
    name: j1
    steps:
      - name: s1
        run: npm test

# ✅ 良い例
jobs:
  unit-tests:
    name: Run Unit Tests
    steps:
      - name: Run Jest tests with coverage
        run: npm test -- --coverage
```

#### ✅ DO: 再利用可能ワークフロー
```yaml
# .github/workflows/reusable-test.yml
on:
  workflow_call:
    inputs:
      node-version:
        required: true
        type: string

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
      - run: npm test

# .github/workflows/ci.yml
jobs:
  test-node-20:
    uses: ./.github/workflows/reusable-test.yml
    with:
      node-version: '20'
```

#### ✅ DO: 環境変数の整理
```yaml
# グローバル環境変数
env:
  NODE_VERSION: '20'
  CACHE_KEY: v1

jobs:
  test:
    env:
      # ジョブレベル環境変数
      TEST_ENV: ci
    steps:
      - name: Test
        env:
          # ステップレベル環境変数
          SPECIFIC_VAR: value
        run: npm test
```

---

## Fastlane ベストプラクティス

### 1. Lane 設計

#### ✅ DO: 単一責任の原則
```ruby
# ❌ 悪い例: 1つのlaneで全てを行う
lane :deploy do
  run_tests
  build_app
  upload_to_testflight
  upload_to_app_store
  slack(message: "Deployed!")
end

# ✅ 良い例: 責任を分離
lane :test do
  run_tests(scheme: "MyApp")
end

lane :beta do
  test  # 既存のlaneを呼び出し
  build_app(scheme: "MyApp")
  upload_to_testflight
  notify_slack(message: "Beta deployed")
end

lane :release do
  test
  build_app(scheme: "MyApp", configuration: "Release")
  upload_to_app_store
  notify_slack(message: "Production deployed")
end

private_lane :notify_slack do |options|
  slack(
    message: options[:message],
    channel: "#releases"
  )
end
```

#### ✅ DO: エラーハンドリング
```ruby
lane :beta do
  begin
    # メイン処理
    build_app(scheme: "MyApp")
    upload_to_testflight

  rescue => exception
    # エラー時の処理
    slack(
      message: "❌ Beta build failed: #{exception.message}",
      success: false
    )

    # エラーを再スロー
    raise exception

  else
    # 成功時の処理
    slack(
      message: "✅ Beta build uploaded",
      success: true
    )

  ensure
    # 必ず実行される処理
    clean_build_artifacts
  end
end
```

### 2. 証明書管理（Match）

#### ✅ DO: Matchを使用
```ruby
# ❌ 悪い例: 手動で証明書管理
# → チーム全員が個別に証明書を持つ
# → 期限切れ・競合が頻発

# ✅ 良い例: Matchで一元管理
lane :certificates do
  match(
    type: "development",
    readonly: true,  # CIでは読み取り専用
    app_identifier: "com.example.app"
  )
end

lane :certificates_update do
  match(
    type: "appstore",
    force_for_new_devices: true  # 新しいデバイス追加時
  )
end
```

#### ✅ DO: 環境変数で管理
```ruby
# fastlane/Matchfile
git_url(ENV["MATCH_GIT_URL"])
storage_mode("git")
type("appstore")

git_basic_authorization(ENV["MATCH_GIT_BASIC_AUTHORIZATION"])
```

### 3. ビルド最適化

#### ✅ DO: キャッシュの活用
```ruby
lane :build_fast do
  # CocoaPodsキャッシュ
  cocoapods(
    repo_update: ENV["CI"] ? false : true  # CIではrepo更新しない
  )

  # ビルド
  build_app(
    scheme: "MyApp",
    skip_codesigning: true,  # テストビルドでは署名スキップ
    skip_archive: true,
    skip_package_ipa: true
  )
end
```

#### ✅ DO: 並列テスト実行
```ruby
lane :test do
  run_tests(
    scheme: "MyApp",
    devices: ["iPhone 15", "iPad Pro"],
    parallel_testing: true,
    concurrent_workers: 4,  # 4並列
    skip_slack: true
  )
end
```

---

## デプロイメント ベストプラクティス

### 1. 環境管理

#### ✅ DO: 環境ごとの設定分離
```yaml
# .github/workflows/deploy.yml
jobs:
  deploy-dev:
    environment:
      name: development
      url: https://dev.example.com
    steps:
      - run: npm run deploy:dev

  deploy-staging:
    environment:
      name: staging
      url: https://staging.example.com
    needs: [test]
    steps:
      - run: npm run deploy:staging

  deploy-prod:
    environment:
      name: production
      url: https://example.com
    needs: [test, deploy-staging]
    steps:
      - run: npm run deploy:prod
```

**Environment 保護設定:**
- Development: 制限なし
- Staging: テスト成功後、自動デプロイ
- Production: レビュアー承認必須、mainブランチのみ

#### ✅ DO: デプロイ前チェック
```yaml
- name: Health check before deploy
  run: curl -f https://example.com/health || exit 1

- name: Database backup
  run: npm run db:backup

- name: Deploy
  run: npm run deploy

- name: Health check after deploy
  run: |
    for i in {1..10}; do
      curl -f https://example.com/health && break
      sleep 10
    done
```

### 2. ロールバック戦略

#### ✅ DO: 前バージョンの保持
```yaml
# タグでバージョン管理
- name: Create backup tag
  run: |
    CURRENT_VERSION=$(git describe --tags --abbrev=0)
    git tag backup-$CURRENT_VERSION-$(date +%Y%m%d-%H%M%S)
    git push origin --tags

# または Blue-Green デプロイメント
- name: Deploy to green
  run: kubectl set image deployment/app app=myapp:${{ github.sha }}

- name: Switch traffic
  if: success()
  run: kubectl patch service app -p '{"spec":{"selector":{"version":"green"}}}'

- name: Rollback on failure
  if: failure()
  run: kubectl patch service app -p '{"spec":{"selector":{"version":"blue"}}}'
```

#### ✅ DO: 自動ロールバック
```yaml
- name: Deploy
  id: deploy
  run: npm run deploy

- name: Verify deployment
  id: verify
  run: |
    sleep 30
    ERROR_RATE=$(curl -s https://api.example.com/metrics/error_rate)
    if [ $ERROR_RATE -gt 5 ]; then
      echo "Error rate too high: $ERROR_RATE%"
      exit 1
    fi

- name: Rollback on failure
  if: failure() && steps.deploy.outcome == 'success'
  run: npm run deploy:rollback
```

### 3. モニタリング

#### ✅ DO: デプロイメトリクスの記録
```yaml
- name: Record deployment
  if: success()
  run: |
    curl -X POST https://api.example.com/deployments \
      -H "Content-Type: application/json" \
      -d '{
        "version": "${{ github.sha }}",
        "environment": "production",
        "deployer": "${{ github.actor }}",
        "timestamp": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
      }'
```

#### ✅ DO: アラート設定
```yaml
- name: Monitor for 5 minutes
  run: |
    for i in {1..10}; do
      ERROR_COUNT=$(curl -s https://api.example.com/metrics/errors/5m)
      if [ $ERROR_COUNT -gt 100 ]; then
        echo "::error::High error count: $ERROR_COUNT"
        # Slackに通知
        curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
          -d '{"text":"🚨 High error rate after deployment"}'
        exit 1
      fi
      sleep 30
    done
```

---

## テスト ベストプラクティス

### 1. テスト戦略

#### ✅ DO: テストピラミッド
```
       /\
      /E2E\     10% - フルフローテスト
     /------\
    /  統合  \   20% - API・統合テスト
   /----------\
  /ユニット   \  70% - 関数・コンポーネントテスト
 /--------------\
```

```yaml
jobs:
  unit-tests:
    # 高速、頻繁に実行
    runs-on: ubuntu-latest
    steps:
      - run: npm test -- --testPathPattern=unit

  integration-tests:
    # 中速、PR時に実行
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - run: npm test -- --testPathPattern=integration

  e2e-tests:
    # 低速、mainマージ時のみ
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: npm run test:e2e
```

#### ✅ DO: テストの並列化
```yaml
# シャーディングで高速化
jobs:
  test:
    strategy:
      matrix:
        shard: [1, 2, 3, 4]
    steps:
      - run: npx jest --shard=${{ matrix.shard }}/4
```

### 2. カバレッジ管理

#### ✅ DO: カバレッジ閾値の設定
```json
// jest.config.js
{
  "coverageThreshold": {
    "global": {
      "branches": 80,
      "functions": 80,
      "lines": 80,
      "statements": 80
    }
  }
}
```

```yaml
# CI/CDで強制
- name: Run tests with coverage
  run: npm test -- --coverage --coverageThreshold='{"global":{"lines":80}}'
```

#### ❌ DON'T: 100% カバレッジを目指す
- 100%は現実的でない
- テストの品質 > カバレッジ率
- 80-90%が適切

---

## コスト最適化

### 1. 実行時間の削減

#### ✅ DO: 不要な実行を避ける
```yaml
# パスフィルタリング
on:
  push:
    paths:
      - 'src/**'
      - 'package.json'
    paths-ignore:
      - '**.md'

# 並列ワークフローのキャンセル
concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
```

#### ✅ DO: セルフホストランナーの検討
```yaml
# 頻繁に実行されるジョブ
jobs:
  lint:
    runs-on: self-hosted  # 無料
    steps:
      - run: npm run lint

# 重いジョブはGitHub-hosted
  ios-build:
    runs-on: macos-latest  # 従量課金
    steps:
      - run: fastlane build
```

### 2. ストレージ最適化

#### ✅ DO: アーティファクトの保持期間短縮
```yaml
- uses: actions/upload-artifact@v4
  with:
    name: build
    path: dist/
    retention-days: 1  # デフォルト90日から短縮
```

#### ✅ DO: キャッシュサイズの最適化
```yaml
# 不要なファイルを除外
- run: |
    find node_modules -name "*.md" -delete
    find node_modules -name "test" -type d -exec rm -rf {} +

- uses: actions/cache@v4
  with:
    path: node_modules
    key: ${{ runner.os }}-optimized-${{ hashFiles('package-lock.json') }}
```

---

## トラブルシューティング

### 1. デバッグ方法

#### ✅ DO: デバッグモードの活用
```yaml
# ワークフロー実行時にデバッグログを有効化
# Settings → Secrets → ACTIONS_STEP_DEBUG = true
# Settings → Secrets → ACTIONS_RUNNER_DEBUG = true

- name: Debug info
  run: |
    echo "Runner OS: ${{ runner.os }}"
    echo "Node version: $(node -v)"
    echo "npm version: $(npm -v)"
    printenv | sort
```

#### ✅ DO: Tmate でリモートデバッグ
```yaml
# デバッグ用ステップ（失敗時のみ）
- name: Setup tmate session
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 30
```

### 2. よくある問題

#### ✅ DO: キャッシュミスの診断
```yaml
- name: Cache diagnosis
  run: |
    echo "Cache key: ${{ runner.os }}-${{ hashFiles('package-lock.json') }}"
    ls -la ~/.npm || echo "No npm cache"
    ls -la node_modules || echo "No node_modules"
```

#### ✅ DO: タイムアウト対策
```yaml
# ステップごとのタイムアウト
- name: Long running task
  timeout-minutes: 10
  run: npm run heavy-task

# 無限ループ防止
jobs:
  test:
    timeout-minutes: 30  # ジョブ全体のタイムアウト
```

---

## まとめ

### チェックリスト

**セキュリティ**
- [ ] Secretsを使用（ハードコードなし）
- [ ] 権限を最小化
- [ ] サードパーティActionのバージョン固定
- [ ] Dependabot有効化

**パフォーマンス**
- [ ] キャッシュ有効化
- [ ] 並列実行
- [ ] 条件付き実行
- [ ] タイムアウト設定

**品質**
- [ ] テストの自動実行
- [ ] Lintチェック
- [ ] カバレッジ測定
- [ ] コードレビュー自動化

**運用**
- [ ] デプロイ前チェック
- [ ] ロールバック手順
- [ ] モニタリング
- [ ] 通知設定

**コスト**
- [ ] 不要な実行を削減
- [ ] アーティファクト保持期間最適化
- [ ] セルフホストランナー検討

---

**最終更新**: 2026年1月
