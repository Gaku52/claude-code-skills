# コントリビューションガイド

Claude Code Skills への貢献方法

## 🎯 貢献の種類

1. **新しいSkillの追加**
2. **既存Skillの改善・拡充**
3. **バグ修正・誤字脱字修正**
4. **実際の失敗事例の追加**
5. **ドキュメントの改善**

## 📦 標準Git ワークフロー（必須）

**このプロジェクトでは、コンフリクトを防止するため、すべてのコミット&プッシュに `safe-commit-push.sh` の使用を規定としています。**

### なぜこのワークフローが必要か

```
問題: 通常のgit pushでは、リモートに新しいコミットがあるとpushが失敗する
  ↓
手動でpull → コンフリクト解決 → 再度push → また失敗...
  ↓
時間の無駄 & ストレス & ミスの原因
```

**解決策: 自動化された安全なワークフロー**

### 使い方

```bash
# 1. スクリプトを実行（コミットメッセージを引数として渡す）
./scripts/safe-commit-push.sh "feat: add new iOS security guide"

# これだけ！以下が自動的に実行されます:
# ✅ 最新の変更を取得 (pull --rebase)
# ✅ 全ての変更をステージング (git add -A)
# ✅ コミット作成
# ✅ プッシュ前に最終pull (コンフリクト防止)
# ✅ リモートにpush
```

### スクリプトが実行する処理

```bash
#!/bin/bash
# scripts/safe-commit-push.sh

# Step 1: 最新の変更を取得
git pull --rebase origin main

# Step 2: 変更があるか確認
if [[ -z $(git status -s) ]]; then
    echo "変更なし"
    exit 0
fi

# Step 3: すべての変更をステージング
git add -A

# Step 4: コミット作成
git commit -m "$1"

# Step 5: プッシュ前に最終pull（念のため）
git pull --rebase origin main

# Step 6: プッシュ
git push origin main
```

### メリット

✅ **コンフリクトゼロ**: 常に最新の状態でpush
✅ **ミス防止**: 手動操作のミスを排除
✅ **時間節約**: 5分 → 30秒
✅ **シンプル**: コマンド1つだけ

### NG例（使用禁止）

```bash
# ❌ 通常のgit pushは使用しない
git add .
git commit -m "update"
git push  # ← コンフリクトのリスク

# ✅ 必ずスクリプトを使用
./scripts/safe-commit-push.sh "update"
```

### トラブルシューティング

#### エラー: `permission denied`

```bash
# 実行権限を付与
chmod +x scripts/safe-commit-push.sh
```

#### エラー: `Conflict detected`

```bash
# rebase中にコンフリクトが発生した場合
# 1. コンフリクトを手動で解決
# 2. 以下を実行:
git add .
git rebase --continue
git push origin main
```

---

## 📝 新しいSkillの追加

### 1. Skillが必要か判断

以下の基準を満たすか確認：

- [ ] 既存のSkillでカバーされていない
- [ ] 独立したトピックとして成立する
- [ ] 複数の場面で再利用される
- [ ] Agent連携の可能性がある

### 2. Skillの構造を作成

テンプレートからコピー：

```bash
# git-workflowをテンプレートとして使用
cp -r git-workflow new-skill-name

# 既存ファイルをクリーンアップ
cd new-skill-name
rm -rf incidents/*
rm -rf guides/*
# 必要なファイルのみ残す
```

### 3. SKILL.mdを作成

必須セクション：

```markdown
---
name: skill-name
description: 1-2文の説明（最大1024文字）
---

# Skill Name

## 📋 目次
## 概要
## いつ使うか
## 詳細ガイド
## Agent連携
## クイックリファレンス
```

### 4. 詳細ドキュメントを追加

#### guides/ （詳細ガイド）

- 各トピック1000-2000行程度
- 具体例を豊富に
- コードスニペット付き

#### checklists/ （チェックリスト）

- 実行前・実行中・実行後
- 各10-30項目程度（多すぎない）

#### templates/ （テンプレート）

- すぐにコピーして使える
- プレースホルダーは `<>` で

#### references/ （リファレンス）

- ベストプラクティス集
- アンチパターン集
- トラブルシューティング

### 5. READMEの進捗表を更新

```markdown
| X | `new-skill-name` | ✅ 完成 | 説明 |
```

### 6. PRを作成

テンプレートに従ってPRを作成

## 🔧 既存Skillの改善

### 実際の開発で気づいた点を追加

1. **Issueを作成**

```markdown
タイトル: [git-workflow] コンフリクト解決ガイドに〇〇のケースが不足

## 現状
guides/07-conflict-resolution.md に〇〇のケースの説明がない

## 提案
〇〇のケースの解決手順を追加する

## 背景
実際の開発で〇〇が発生し、解決に時間がかかった
```

2. **ブランチ作成**

```bash
git checkout -b improve/git-workflow-conflict-resolution
```

3. **ドキュメント更新**

```bash
# ガイドに追記
vim git-workflow/guides/07-conflict-resolution.md

# incidents/ に事例追加（任意）
vim git-workflow/incidents/2024/001-xxx.md
```

4. **PRを作成**

## 🐛 失敗事例の追加（最重要！）

### 実際に発生した問題を記録

これがこのプロジェクトの最大の価値です。

#### 1. incident-loggerを使う

問題発生時、すぐに記録：

```markdown
# incidents/2024/XXX-title.md

## 発生日時
2024-12-24 15:30

## 概要
Xcodeビルドが突然失敗するようになった

## 症状
- エラーメッセージ: "Command PhaseScriptExecution failed"
- 再現率: 100%
- 影響範囲: 全開発者

## 原因
CocoaPodsのバージョン不整合

## 解決方法
1. `pod deintegrate`
2. `pod install`
3. Derived Dataクリア

## 予防策
- Podfile.lockをコミット
- CIで同じPodバージョンを使用

## 所要時間
- 調査: 2時間
- 解決: 5分

## 参考資料
- https://github.com/CocoaPods/CocoaPods/issues/XXX

## タグ
#build #cocoapods #xcode
```

#### 2. 該当Skillに反映

```bash
# チェックリストに追加
# references/common-pitfalls.md に追加
# guides/ の関連箇所に注意書き追加
```

#### 3. PRを作成

タイトル: `incident: add CocoaPods version mismatch case`

## 📋 チェックリスト

### PR作成前

- [ ] SKILL.mdの`---`フロントマターは正しいか
- [ ] リンク切れはないか（`guides/`, `checklists/`等）
- [ ] マークダウンフォーマットは正しいか
- [ ] コードブロックにはシンタックスハイライトがあるか
- [ ] 例は具体的で実用的か
- [ ] Agent連携セクションは記載したか（該当する場合）

### コミット前

- [ ] Conventional Commits形式に従っているか
- [ ] READMEの進捗表を更新したか（新Skill追加時）

## 🎨 スタイルガイド

### マークダウン

```markdown
# H1 - Skillタイトルのみ
## H2 - メインセクション
### H3 - サブセクション
#### H4 - 詳細項目

- 箇条書き
  - ネスト

**太字** - 重要なキーワード
`code` - コマンド、ファイル名、短いコード

\`\`\`language
コードブロック
\`\`\`

> 引用・注意事項

| テーブル | ヘッダー |
|---------|---------|
| データ  | データ  |
```

### 絵文字の使用

適度に使用してわかりやすく：

```markdown
✅ 完了・推奨
❌ 非推奨・NG
⚠️ 注意
📋 リスト・チェックリスト
🔧 設定・ツール
📝 ドキュメント
🚀 リリース・デプロイ
🐛 バグ
```

### コード例

#### ✅ Good

```swift
// 具体的で実用的
func fetchUserProfile(userId: String) async throws -> UserProfile {
    let endpoint = "/api/users/\(userId)"
    let response = try await apiClient.get(endpoint)
    return try JSONDecoder().decode(UserProfile.self, from: response)
}
```

#### ❌ Bad

```swift
// 抽象的すぎる
func doSomething() {
    // TODO
}
```

### ファイル名

```
小文字、ハイフン区切り

✅ conflict-resolution.md
✅ best-practices.md
✅ 01-getting-started.md

❌ Conflict_Resolution.md
❌ bestPractices.md
```

## 🔍 レビュー基準

PRは以下の観点でレビューされます：

### 内容

- [ ] 正確性: 技術的に正しいか
- [ ] 網羅性: 必要な情報が揃っているか
- [ ] 実用性: 実際の開発で使えるか
- [ ] 具体性: 抽象的すぎないか

### 構成

- [ ] 論理的: 順序立てて説明されているか
- [ ] 一貫性: 他のSkillsとスタイルが統一されているか
- [ ] リンク: 関連ドキュメントへのリンクがあるか

### 品質

- [ ] 誤字脱字なし
- [ ] マークダウン正しい
- [ ] コード例が動作する

## 🚀 リリースプロセス

1. **main ブランチ**: 常に安定版
2. **develop ブランチ**: 開発中の変更
3. **feature/* ブランチ**: 新機能・改善
4. **hotfix/* ブランチ**: 緊急修正

### バージョニング

Semantic Versioning に従う：

- `MAJOR.MINOR.PATCH`
- `0.1.0` - Alpha（現在）
- `0.5.0` - Beta
- `1.0.0` - 正式リリース

## 💡 Tips

### 良いドキュメントの特徴

1. **段階的**: 基礎→応用→上級
2. **具体的**: 実例が豊富
3. **検索可能**: 適切な見出し、キーワード
4. **最新**: 古い情報は更新・削除

### インシデント記録のコツ

- 発生直後に記録（記憶が新鮮）
- 感情より事実を記録
- 再現手順を明確に
- 解決方法だけでなく「なぜそうなったか」も

### Agent連携の設計

- どのタイミングで起動すべきか
- 並行実行可能か
- どのSkillを参照するか
- パラメータは何が必要か

## ❓ 質問・相談

- Issue: バグ報告、機能提案
- Discussion: 質問、アイデア

---

貢献ありがとうございます！
