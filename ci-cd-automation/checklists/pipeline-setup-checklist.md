# CI/CDパイプライン構築チェックリスト

## 事前準備

### プロジェクト分析
- [ ] プロジェクトの規模を把握（ファイル数、依存関係数）
- [ ] 既存のビルドプロセスを文書化
- [ ] テスト実行時間を計測（ローカル環境）
- [ ] デプロイ先環境を確認（Vercel, AWS, App Store等）
- [ ] チームのCI/CD経験レベルを確認

### リポジトリ設定
- [ ] `.github/workflows/` ディレクトリを作成
- [ ] GitHub Actions権限を確認（Settings → Actions → General）
- [ ] 必要なSecretsを登録（API keys, tokens等）
- [ ] Environment保護ルールを設定（本番環境）
- [ ] ブランチ保護ルールを設定（main, develop）

---

## 基本パイプライン構築

### Lintチェック
- [ ] ESLint/Prettierの設定ファイル確認
- [ ] Lintワークフロー作成（`.github/workflows/lint.yml`）
- [ ] エラー時のCI失敗設定
- [ ] 自動修正の検討（Prettier）
- [ ] ワークフロー動作確認

**サンプルワークフロー:**
```yaml
name: Lint
on: [push, pull_request]
jobs:
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
```

### ユニットテスト
- [ ] テストフレームワーク確認（Jest, Vitest等）
- [ ] テストワークフロー作成
- [ ] カバレッジ閾値設定
- [ ] カバレッジレポート保存
- [ ] テスト失敗時の通知設定

**必須設定:**
```yaml
- run: npm test -- --coverage --coverageThreshold='{"global":{"lines":80}}'
- uses: codecov/codecov-action@v4
  if: always()
```

### ビルドチェック
- [ ] ビルドコマンド確認（`npm run build`）
- [ ] ビルドワークフロー作成
- [ ] ビルド成果物の保存（アーティファクト）
- [ ] ビルドエラー時の通知
- [ ] ビルド時間の計測・記録

### 型チェック（TypeScript）
- [ ] tsconfig.jsonの確認
- [ ] 型チェックワークフロー追加
- [ ] strict モードの有効化検討
- [ ] 型エラーの修正期限設定

---

## 並列化・最適化

### ジョブの並列化
- [ ] 独立したジョブを分離（lint, test, build）
- [ ] 並列実行数の最適化
- [ ] 依存関係の設定（`needs`）
- [ ] 並列実行時間の測定

**推奨構成:**
```yaml
jobs:
  lint:
    # 並列実行
  test:
    # 並列実行
  build:
    needs: [lint, test]  # lint, test成功後に実行
```

### キャッシュ戦略
- [ ] npm/yarnキャッシュの有効化
- [ ] ビルドキャッシュの設定
- [ ] キャッシュキーの最適化
- [ ] キャッシュヒット率の確認
- [ ] キャッシュサイズのモニタリング

**チェック項目:**
```yaml
- uses: actions/setup-node@v4
  with:
    cache: 'npm'  # ✅ 有効化確認
```

### テストの最適化
- [ ] テストシャーディング検討（大規模プロジェクト）
- [ ] 並列テスト実行の設定
- [ ] 変更検出による選択的テスト実行
- [ ] E2Eテストの分離
- [ ] テストタイムアウトの設定

---

## デプロイメント設定

### 環境設定
- [ ] Development環境の設定
- [ ] Staging環境の設定
- [ ] Production環境の設定
- [ ] 環境変数の管理（Secrets）
- [ ] 環境保護ルールの設定

**環境保護設定:**
- [ ] Production: レビュアー必須
- [ ] Production: mainブランチのみ
- [ ] Staging: 自動デプロイ
- [ ] Development: PRプレビュー

### デプロイワークフロー
- [ ] デプロイトリガーの設定（push, tag等）
- [ ] デプロイ前チェックの実装
- [ ] デプロイ後の動作確認（ヘルスチェック）
- [ ] ロールバック手順の文書化
- [ ] デプロイ通知の設定（Slack等）

### ロールバック準備
- [ ] 前バージョンのタグ保持
- [ ] ロールバックワークフロー作成
- [ ] ロールバック手順書作成
- [ ] ロールバック訓練の実施

---

## モニタリング・通知

### ビルド状況の可視化
- [ ] README.mdにバッジ追加
- [ ] ビルド時間のトラッキング
- [ ] 成功率のモニタリング
- [ ] 定期的なメトリクスレビュー

**バッジ例:**
```markdown
![CI](https://github.com/owner/repo/workflows/CI/badge.svg)
```

### 通知設定
- [ ] ビルド失敗時のSlack通知
- [ ] デプロイ成功時の通知
- [ ] PRへのコメント（テスト結果）
- [ ] エラーログの自動Issue化

**Slack通知例:**
```yaml
- name: Slack通知
  if: failure()
  env:
    SLACK_WEBHOOK: ${{ secrets.SLACK_WEBHOOK }}
  run: |
    curl -X POST $SLACK_WEBHOOK \
      -H 'Content-Type: application/json' \
      -d '{"text":"❌ ビルド失敗: ${{ github.workflow }}"}'
```

---

## セキュリティ

### Secrets管理
- [ ] 全てのSecretsを暗号化
- [ ] Secretsのローテーション計画
- [ ] 最小権限の原則適用
- [ ] Secretsの定期監査

**必須Secrets:**
- [ ] `GITHUB_TOKEN`（自動提供）
- [ ] デプロイ先のAPI key
- [ ] 通知サービスのWebhook URL
- [ ] コード署名証明書（iOS/Android）

### 権限設定
- [ ] ワークフロー権限の最小化
- [ ] `permissions:` セクションの明示
- [ ] サードパーティActionのバージョン固定
- [ ] Dependabotの有効化

**推奨設定:**
```yaml
permissions:
  contents: read
  pull-requests: write
```

### 依存関係のセキュリティ
- [ ] Dependabotアラートの有効化
- [ ] 自動セキュリティアップデート
- [ ] 定期的な脆弱性スキャン
- [ ] ライセンス確認

---

## iOS/Android固有

### iOS（Fastlane）
- [ ] Fastlaneのインストール確認
- [ ] Fastfile作成
- [ ] Matchで証明書管理
- [ ] App Store Connect API key設定
- [ ] TestFlight自動配布設定

**必須ファイル:**
- [ ] `fastlane/Fastfile`
- [ ] `fastlane/Appfile`
- [ ] `fastlane/Matchfile`

### Android
- [ ] Gradleビルド設定
- [ ] 署名鍵の管理（Secrets）
- [ ] Google Play API設定
- [ ] アプリバンドル生成設定

---

## ドキュメント

### ワークフロー文書化
- [ ] 各ワークフローの目的を文書化
- [ ] トリガー条件の明記
- [ ] 実行時間の目安を記載
- [ ] トラブルシューティングガイド作成

**README.mdに追加:**
```markdown
## CI/CD

### ワークフロー
- **CI**: プッシュ時に自動実行（5-10分）
- **Deploy**: mainマージ時に本番デプロイ（3-5分）
- **Release**: タグプッシュ時にリリース（10-15分）

### トラブルシューティング
- ビルド失敗時: [troubleshooting.md](docs/troubleshooting.md)
```

### チーム向けガイド
- [ ] 新メンバー向けのCI/CD説明
- [ ] ローカルでのテスト実行方法
- [ ] デプロイ手順
- [ ] 緊急時の対応手順

---

## テストと検証

### ワークフローテスト
- [ ] PRでのテスト実行確認
- [ ] mainブランチでのデプロイ確認
- [ ] ロールバック手順の検証
- [ ] 手動トリガーの動作確認

### パフォーマンステスト
- [ ] ビルド時間の計測
- [ ] キャッシュヒット率の確認
- [ ] 並列実行の効果測定
- [ ] コスト試算

**目標値:**
- PR チェック: 5分以内
- デプロイ: 10分以内
- キャッシュヒット率: 80%以上

---

## 継続的改善

### 定期レビュー
- [ ] 月次でビルド時間をレビュー
- [ ] 失敗率の分析
- [ ] コストの確認
- [ ] チームフィードバックの収集

### 最適化計画
- [ ] ボトルネックの特定
- [ ] 改善施策の優先順位付け
- [ ] 実施と効果測定
- [ ] 知見の共有

**レビュー項目:**
```bash
# 過去30日の平均ビルド時間
gh run list --workflow=ci.yml --limit 100 --json createdAt,updatedAt

# 成功率
gh run list --workflow=ci.yml --limit 100 --json conclusion
```

---

## チェックリスト完了確認

### 最小限の構成（必須）
- [ ] Lintチェックワークフロー
- [ ] ユニットテストワークフロー
- [ ] ビルドチェックワークフロー
- [ ] キャッシュ設定
- [ ] ビルド失敗時の通知

### 推奨構成
- [ ] 並列実行
- [ ] テストシャーディング
- [ ] 環境別デプロイ
- [ ] ロールバック機能
- [ ] モニタリング

### 理想的な構成
- [ ] 完全自動デプロイ
- [ ] カナリアリリース
- [ ] 自動ロールバック
- [ ] パフォーマンス監視
- [ ] コスト最適化

---

## よくある問題と対処法

| 問題 | チェック項目 | 対処法 |
|------|------------|--------|
| ビルドが遅い | キャッシュ設定 | `actions/setup-node`の`cache`を有効化 |
| テストが失敗 | テスト環境 | 環境変数、データベース設定を確認 |
| デプロイ失敗 | Secrets設定 | 必要なSecretsが全て設定されているか確認 |
| 権限エラー | Permissions | `permissions:`セクションを追加 |

---

**使い方:**
1. プロジェクト開始時にこのチェックリストをコピー
2. Issue or Projectで進捗管理
3. 完了したら次のステップへ
4. 定期的に見直し・更新

**最終更新**: 2026年1月
