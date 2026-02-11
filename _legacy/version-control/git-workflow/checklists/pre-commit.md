# コミット前チェックリスト

## 基本チェック

### コード品質
- [ ] コードがビルドできる
- [ ] 全テストが通る（`npm test` / `pytest` / `swift test`）
- [ ] Lintエラーがない（`npm run lint` / `flake8` / `swiftlint`）
- [ ] フォーマットが適用されている（Prettier / Black / SwiftFormat）
- [ ] デバッグコード・console.logを削除
- [ ] コメントアウトされたコードを削除

### ファイルチェック
- [ ] 不要なファイルが含まれていない
- [ ] 一時ファイル（*.tmp, *.swp等）が含まれていない
- [ ] ビルド成果物（dist/, build/等）が含まれていない
- [ ] IDEの設定ファイル（.vscode/, .idea/等）が含まれていない
- [ ] OS固有ファイル（.DS_Store, Thumbs.db等）が含まれていない

### セキュリティチェック
- [ ] APIキー・パスワードがハードコードされていない
- [ ] .envファイルが.gitignoreに含まれている
- [ ] 機密情報（credentials, secrets）がコミットされていない
- [ ] テストデータに実際のユーザー情報が含まれていない

## コミットメッセージチェック

### 形式チェック
- [ ] Conventional Commits形式に従っている（`<type>(<scope>): <subject>`）
- [ ] Type が適切（feat/fix/docs/style/refactor/perf/test/chore/ci）
- [ ] Scope が設定されている（オプションだが推奨）
- [ ] Subject が50文字以内
- [ ] Subject が小文字で始まる
- [ ] Subject がピリオドで終わらない
- [ ] Subject が命令形（動詞の原形）

### 内容チェック
- [ ] 変更内容が明確に説明されている
- [ ] 複雑な変更にはBodyが記載されている
- [ ] 関連IssueがFooterに記載されている（Closes #123）
- [ ] Breaking Changeがある場合、適切にマークされている

## 変更内容チェック

### 論理的整合性
- [ ] 1コミット = 1つの論理的な変更
- [ ] 無関係な変更が混ざっていない
- [ ] 変更の粒度が適切（大きすぎない、小さすぎない）

### テストチェック
- [ ] 新機能にユニットテストを追加
- [ ] バグ修正に再現テストを追加
- [ ] エッジケースのテストを追加
- [ ] テストカバレッジが下がっていない

### ドキュメントチェック
- [ ] 複雑なロジックにコメントを追加
- [ ] 公開APIにドキュメントコメントを追加（JSDoc/Docstring等）
- [ ] README更新が必要な場合は更新
- [ ] API変更がある場合、ドキュメントを更新

## レビュー準備

### セルフレビュー
- [ ] `git diff` で変更内容を確認
- [ ] 意図しない変更が含まれていないか確認
- [ ] コードが読みやすいか確認
- [ ] 変数・関数名が明確か確認

### PR準備
- [ ] ブランチ名が命名規則に従っている
- [ ] mainブランチから分岐している（または最新を取り込んでいる）
- [ ] コンフリクトが発生していない

## プロジェクト固有チェック

### iOS開発
- [ ] SwiftLintエラーがない
- [ ] ビルドワーニングがない
- [ ] Asset Catalogが整理されている
- [ ] Info.plistに不要なキーがない

### React/TypeScript
- [ ] TypeScriptエラーがない
- [ ] 未使用のimportがない
- [ ] propTypesが定義されている（必要な場合）
- [ ] アクセシビリティ対応がされている

### Python
- [ ] type hintsが適切に設定されている
- [ ] docstringが記載されている
- [ ] requirements.txtが更新されている（依存追加時）
- [ ] mypyエラーがない

## 最終確認

- [ ] すべての変更を理解している
- [ ] 変更の理由を説明できる
- [ ] チームメンバーがレビューできる状態
- [ ] CI/CDが通る自信がある

---

## クイックコマンド

```bash
# 変更確認
git status
git diff

# ステージング
git add <file>
git add -p  # 部分的にステージング

# コミット
git commit -m "feat(scope): description"

# 直前のコミットを修正
git commit --amend

# テスト実行
npm test
pytest
swift test

# Lint実行
npm run lint
flake8 .
swiftlint

# ビルド確認
npm run build
python setup.py build
xcodebuild
```

---

**このチェックリストを習慣化することで、高品質なコミットが実現します！**
