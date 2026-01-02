# CI/CD トラブルシューティングガイド

## GitHub Actions トラブルシューティング

### 問題1: ワークフローが実行されない

**症状:**
```
プッシュしてもワークフローが実行されない
Actions タブに何も表示されない
```

**診断手順:**

1. **ファイル配置の確認**
```bash
# ❌ 間違った場所
workflows/ci.yml
.github/ci.yml

# ✅ 正しい場所
.github/workflows/ci.yml
```

2. **YAML構文エラーの確認**
```bash
# yamllintでチェック
brew install yamllint
yamllint .github/workflows/*.yml

# またはオンラインツール
# https://www.yamllint.com/
```

3. **トリガー設定の確認**
```yaml
# ❌ mainブランチのみ（他のブランチで実行されない）
on:
  push:
    branches: [main]

# ✅ 全ブランチで実行
on: [push, pull_request]
```

4. **パスフィルタの確認**
```yaml
# .mdファイルのみの変更では実行されない
on:
  push:
    paths-ignore:
      - '**.md'

# 確認方法
git diff --name-only HEAD~1 HEAD
```

**解決方法:**
```bash
# 1. ファイルパス確認
ls -la .github/workflows/

# 2. YAML検証
cat .github/workflows/ci.yml | yamllint -

# 3. 強制実行（手動トリガー）
# workflow_dispatchを追加
on:
  workflow_dispatch:
  push:
    branches: [main]
```

---

### 問題2: npm ci が失敗する

**症状:**
```
npm ERR! `npm ci` can only install packages when your package.json
and package-lock.json are in sync.
```

**原因:**
- package.json と package-lock.json の不整合
- ローカルで npm install 実行後、package-lock.json をコミットし忘れ

**診断手順:**
```bash
# 1. ローカルで確認
npm ci

# 2. package-lock.jsonの状態確認
git status package-lock.json

# 3. package.jsonとの差分確認
npm install --package-lock-only
git diff package-lock.json
```

**解決方法:**

**方法1: 同期させる（推奨）**
```bash
# ローカルで実行
npm install
git add package-lock.json
git commit -m "chore: sync package-lock.json"
git push
```

**方法2: CI/CDで npm install を使用**
```yaml
# ❌ 悪い例（遅い）
- run: npm install

# ✅ 良い例（高速）
- run: npm ci

# 🔧 緊急回避（同期まで）
- run: |
    rm -rf node_modules package-lock.json
    npm install
    npm test
```

**方法3: package-lock.json を再生成**
```bash
rm package-lock.json
npm install
git add package-lock.json
git commit -m "chore: regenerate package-lock.json"
```

---

### 問題3: キャッシュが効かない

**症状:**
```
毎回 npm ci に3分かかる
Cache restore が成功しているように見えるが効果なし
```

**診断手順:**

1. **キャッシュキーの確認**
```yaml
# ログで確認
Cache not found for input keys: linux-node-abc123...
```

2. **キャッシュヒット率の測定**
```yaml
- name: Check cache status
  id: cache-check
  run: |
    if [ -d ~/.npm ]; then
      echo "✅ Cache exists: $(du -sh ~/.npm)"
      echo "hit=true" >> $GITHUB_OUTPUT
    else
      echo "❌ No cache found"
      echo "hit=false" >> $GITHUB_OUTPUT
    fi

- name: Cache hit rate
  run: echo "Cache hit: ${{ steps.cache-check.outputs.hit }}"
```

**原因と解決方法:**

**原因1: キャッシュキーが毎回変わる**
```yaml
# ❌ 悪い例
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-${{ github.run_id }}  # 毎回異なる

# ✅ 良い例
- uses: actions/cache@v4
  with:
    path: ~/.npm
    key: ${{ runner.os }}-node-${{ hashFiles('**/package-lock.json') }}
    restore-keys: |
      ${{ runner.os }}-node-
```

**原因2: パスが間違っている**
```yaml
# npm のキャッシュ場所を確認
- run: npm config get cache
# 出力例: /home/runner/.npm

# ✅ 正しいパス指定
- uses: actions/cache@v4
  with:
    path: ~/.npm  # または /home/runner/.npm
```

**原因3: キャッシュが古すぎて削除された**
```
キャッシュは7日間アクセスされないと削除される
```

**解決方法: キャッシュのウォームアップ**
```yaml
# 週次でキャッシュを更新
on:
  schedule:
    - cron: '0 0 * * 0'  # 毎週日曜

jobs:
  warm-cache:
    steps:
      - run: npm ci
```

---

### 問題4: タイムアウトエラー

**症状:**
```
Error: The operation was canceled.
ジョブが6時間後に強制終了される
```

**診断手順:**
```bash
# どのステップで時間がかかっているか確認
# Actions の詳細ログを確認

# デバッグログ有効化
# Settings → Secrets
# ACTIONS_STEP_DEBUG = true
```

**解決方法:**

**1. タイムアウト時間を適切に設定**
```yaml
jobs:
  test:
    timeout-minutes: 15  # デフォルト360分を短縮

    steps:
      - name: Run tests
        timeout-minutes: 10  # ステップ単位でも設定
        run: npm test
```

**2. 無限ループの検出**
```yaml
# 失敗時にtmateで接続
- name: Debug with tmate
  if: failure()
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 15  # 最大15分で切断
```

**3. 並列化で高速化**
```yaml
# ❌ 悪い例: 直列実行（遅い）
steps:
  - run: npm run test:unit     # 10分
  - run: npm run test:e2e      # 15分
  # 合計25分

# ✅ 良い例: 並列実行（速い）
jobs:
  test-unit:
    steps:
      - run: npm run test:unit  # 10分
  test-e2e:
    steps:
      - run: npm run test:e2e   # 15分
  # 合計15分（並列）
```

---

### 問題5: 環境変数が読めない

**症状:**
```
Error: API_URL is not defined
Secretsが空文字列になる
```

**診断手順:**

1. **Secretsの設定確認**
```
Settings → Secrets and variables → Actions
必要なSecretsが全て登録されているか確認
```

2. **環境による違い**
```yaml
# ❌ 環境が違う
- name: Build
  run: npm run build
  # ここではSecretsにアクセスできない

# ✅ 環境変数経由でアクセス
- name: Build
  env:
    API_URL: ${{ secrets.API_URL }}
  run: npm run build
```

**解決方法:**

**1. 環境変数として設定**
```yaml
- name: Build
  env:
    API_URL: ${{ secrets.API_URL }}
    DB_URL: ${{ secrets.DB_URL }}
  run: npm run build
```

**2. .env ファイル生成**
```yaml
- name: Create .env file
  run: |
    cat > .env.production << EOF
    API_URL=${{ secrets.API_URL }}
    DATABASE_URL=${{ secrets.DATABASE_URL }}
    STRIPE_KEY=${{ secrets.STRIPE_KEY }}
    EOF

- run: npm run build
```

**3. デバッグ（値の確認）**
```yaml
- name: Debug environment variables
  run: |
    echo "NODE_ENV: $NODE_ENV"
    # Secretsの先頭のみ表示（セキュリティ）
    echo "API_URL: ${API_URL:0:10}..."
  env:
    API_URL: ${{ secrets.API_URL }}
```

**4. Organization Secrets vs Repository Secrets**
```
Organization Secrets: 組織全体で共有
Repository Secrets: リポジトリ固有

両方ある場合、Repository Secrets が優先される
```

---

### 問題6: 権限エラー

**症状:**
```
Error: Resource not accessible by integration
Permission denied
```

**原因:**
- GITHUB_TOKEN の権限不足
- リポジトリ設定で Actions の書き込み権限が無効

**診断手順:**
```yaml
# 現在の権限を確認
- name: Check permissions
  run: |
    echo "Actor: ${{ github.actor }}"
    echo "Token permissions: ${{ toJson(github.permissions) }}"
```

**解決方法:**

**1. リポジトリ設定を変更**
```
Settings → Actions → General → Workflow permissions
✅ Read and write permissions
✅ Allow GitHub Actions to create and approve pull requests
```

**2. ワークフロー内で権限を明示**
```yaml
# ワークフロー全体の権限
permissions:
  contents: write
  pull-requests: write
  issues: write

jobs:
  deploy:
    # ジョブごとの権限
    permissions:
      contents: write
      id-token: write  # OIDC用
```

**3. Personal Access Token を使用**
```yaml
# GitHub Tokenの代わりにPATを使用
- uses: actions/checkout@v4
  with:
    token: ${{ secrets.PAT }}  # より強い権限
```

---

### 問題7: マトリックスビルドでの部分的失敗

**症状:**
```
4つのマトリックスジョブのうち1つが失敗
他の3つは成功しているが、全体が失敗扱い
```

**診断手順:**
```yaml
# どのマトリックスが失敗したか確認
strategy:
  matrix:
    node: [18, 20, 21]
    os: [ubuntu, windows, macos]
```

**解決方法:**

**1. fail-fast を無効化**
```yaml
strategy:
  fail-fast: false  # 1つ失敗しても全て実行
  matrix:
    node: [18, 20, 21]
```

**2. 特定の組み合わせを除外**
```yaml
strategy:
  matrix:
    os: [ubuntu, windows, macos]
    node: [18, 20]
    exclude:
      # Windows + Node 18 を除外
      - os: windows
        node: 18
```

**3. 条件付き実行**
```yaml
jobs:
  test:
    strategy:
      matrix:
        os: [ubuntu, windows, macos]
    steps:
      - name: Windows only step
        if: matrix.os == 'windows'
        run: echo "Windows specific"
```

---

## Fastlane トラブルシューティング

### 問題8: 証明書エラー

**症状:**
```
Code signing error
Provisioning profile doesn't match
No signing certificate found
```

**診断手順:**
```bash
# 1. 証明書の確認
security find-identity -v -p codesigning

# 2. プロビジョニングプロファイルの確認
ls ~/Library/MobileDevice/Provisioning\ Profiles/

# 3. Match の状態確認
bundle exec fastlane match development --readonly
```

**解決方法:**

**1. Match で証明書を同期**
```bash
# 最新の証明書を取得
bundle exec fastlane match appstore --readonly

# 環境変数を確認
echo $MATCH_PASSWORD
echo $MATCH_GIT_BASIC_AUTHORIZATION
```

**2. CI/CD での設定**
```yaml
- name: Setup certificates
  run: bundle exec fastlane match appstore --readonly
  env:
    MATCH_PASSWORD: ${{ secrets.MATCH_PASSWORD }}
    MATCH_GIT_BASIC_AUTHORIZATION: ${{ secrets.MATCH_GIT_BASIC_AUTHORIZATION }}
```

**3. 証明書の再生成（最終手段）**
```bash
# 既存の証明書を削除して再生成
bundle exec fastlane match nuke development
bundle exec fastlane match nuke appstore

# 新しい証明書を生成
bundle exec fastlane match appstore
```

---

### 問題9: TestFlight アップロード失敗

**症状:**
```
Error uploading to TestFlight
iTunes Transporter failed
```

**診断手順:**
```bash
# 1. App Store Connect API Key の確認
# 環境変数が設定されているか確認

# 2. ビルドの検証
xcrun altool --validate-app -f YourApp.ipa \
  --type ios \
  --apiKey $API_KEY_ID \
  --apiIssuer $API_ISSUER_ID

# 3. 手動アップロード（テスト）
xcrun altool --upload-app -f YourApp.ipa \
  --type ios \
  --apiKey $API_KEY_ID \
  --apiIssuer $API_ISSUER_ID
```

**解決方法:**

**1. API Key 認証を使用（推奨）**
```ruby
# Fastfile
lane :beta do
  # App Store Connect API Key
  api_key = app_store_connect_api_key(
    key_id: ENV["APP_STORE_CONNECT_API_KEY_KEY_ID"],
    issuer_id: ENV["APP_STORE_CONNECT_API_KEY_ISSUER_ID"],
    key_content: ENV["APP_STORE_CONNECT_API_KEY_KEY"],
    is_key_content_base64: true
  )

  upload_to_testflight(
    api_key: api_key,
    skip_waiting_for_build_processing: true
  )
end
```

**2. リトライロジック追加**
```ruby
lane :beta do
  retry_count = 0
  begin
    upload_to_testflight
  rescue => exception
    retry_count += 1
    if retry_count < 3
      sleep(60)  # 60秒待機
      retry
    else
      raise exception
    end
  end
end
```

**3. ネットワークタイムアウト延長**
```ruby
lane :beta do
  upload_to_testflight(
    api_key: api_key,
    timeout: 3600  # 1時間
  )
end
```

---

### 問題10: ビルドが遅い

**症状:**
```
Fastlane でのビルドに20分以上かかる
CIパイプライン全体が30分超え
```

**診断手順:**
```bash
# 1. 各ステップの時間を計測
time bundle exec fastlane test
time bundle exec fastlane build

# 2. Xcodeビルド時間の確認
xcodebuild -showBuildSettings | grep BUILD_TIME
```

**解決方法:**

**1. 並列テスト実行**
```ruby
lane :test do
  run_tests(
    scheme: "MyApp",
    parallel_testing: true,
    concurrent_workers: 4  # 4並列
  )
end
```

**2. キャッシュの活用**
```ruby
lane :build do
  # CocoaPods repo update をスキップ
  cocoapods(
    repo_update: false  # CIでは更新しない
  )

  build_app(
    scheme: "MyApp",
    clean: false  # クリーンビルドしない
  )
end
```

**3. 不要な処理をスキップ**
```ruby
lane :beta do
  build_app(
    scheme: "MyApp",
    export_options: {
      compileBitcode: false,  # Bitcodeを無効化
      uploadSymbols: false     # シンボルは後でアップ
    }
  )

  upload_to_testflight(
    skip_waiting_for_build_processing: true  # 処理待ちしない
  )
end
```

---

## デバッグテクニック

### 1. デバッグログの有効化

```yaml
# GitHub Actions
# Settings → Secrets
# ACTIONS_STEP_DEBUG = true
# ACTIONS_RUNNER_DEBUG = true

jobs:
  debug:
    steps:
      - name: Debug info
        run: |
          echo "::debug::This is a debug message"
          echo "::warning::This is a warning"
          echo "::error::This is an error"
```

### 2. Tmate でリモート接続

```yaml
- name: Setup tmate session
  if: failure()  # 失敗時のみ
  uses: mxschmitt/action-tmate@v3
  timeout-minutes: 15
```

**使い方:**
```bash
# Actions のログに表示されるSSHコマンドを実行
ssh xxxxx@nyc1.tmate.io

# リモート環境で調査
ls -la
printenv
npm test
```

### 3. ステップごとのログ保存

```yaml
- name: Run tests
  run: npm test 2>&1 | tee test.log

- name: Upload logs
  if: always()
  uses: actions/upload-artifact@v4
  with:
    name: logs
    path: "*.log"
```

### 4. Slack通知でエラー詳細送信

```yaml
- name: Notify on failure
  if: failure()
  run: |
    ERROR_LOG=$(tail -n 50 test.log)
    curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
      -H 'Content-Type: application/json' \
      -d "{
        \"text\": \"❌ Build failed\",
        \"attachments\": [{
          \"color\": \"danger\",
          \"text\": \"$ERROR_LOG\"
        }]
      }"
```

---

## よくある質問（FAQ）

### Q1: Secretsが更新されない
**A:** Secrets は更新後、すぐに反映されます。キャッシュはありません。古い値が使われる場合は:
```bash
# 1. Secretsの名前を確認（タイポがないか）
# 2. Environment Secrets を確認（Environment別）
# 3. ワークフローを再実行
```

### Q2: GitHub Actionsの無料枠を使い切った
**A:**
```
Public リポジトリ: 無制限
Private リポジトリ:
  - Free: 2,000分/月
  - Pro: 3,000分/月
  - Team: 10,000分/月

対策:
1. セルフホストランナー使用
2. 並列度を下げる
3. 不要な実行を削減（paths-ignore）
```

### Q3: ワークフローが pending のまま動かない
**A:**
```
原因:
1. ランナーが不足（同時実行数上限）
2. セルフホストランナーがオフライン
3. ジョブの依存関係でブロック

確認:
- Actions の Usage タブで同時実行数を確認
- Settings → Actions → Runners でランナー状態確認
```

---

**最終更新**: 2026年1月
