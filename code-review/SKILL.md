---
name: code-review
description: 効果的なコードレビューの実施方法、レビュー観点チェックリスト、建設的なフィードバック技術、セルフレビュー手法、自動化ツール活用まで、品質とチームコラボレーションを向上させる包括的ガイド。
---

# Code Review Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [レビュー観点](#レビュー観点)
4. [レビュープロセス](#レビュープロセス)
5. [コミュニケーション](#コミュニケーション)
6. [自動化](#自動化)
7. [よくある問題](#よくある問題)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、効果的なコードレビューの全てをカバーします：

- ✅ レビュー観点チェックリスト
- ✅ セルフレビュー手法
- ✅ 建設的なフィードバック技術
- ✅ レビュープロセス設計
- ✅ 自動レビューツール活用
- ✅ チームコラボレーション向上
- ✅ よくある指摘事項パターン
- ✅ レビュー効率化テクニック

## 📚 公式ドキュメント・参考リソース

**このガイドで学べること**: レビュープロセス設計、効果的なフィードバック方法、自動化ツール活用、チーム文化構築
**公式で確認すべきこと**: 最新のレビューツール機能、GitHub/GitLab新機能、ベストプラクティス

### 主要な公式ドキュメント

- **[GitHub Pull Request Documentation](https://docs.github.com/en/pull-requests)** - PR運用ガイド
  - [Reviewing Changes](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests)
  - [Code Review](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/about-pull-request-reviews)

- **[Google Engineering Practices](https://google.github.io/eng-practices/review/)** - Googleコードレビューガイド
  - [Reviewer Guide](https://google.github.io/eng-practices/review/reviewer/)
  - [Author Guide](https://google.github.io/eng-practices/review/developer/)

- **[GitLab Code Review Guidelines](https://docs.gitlab.com/ee/development/code_review.html)** - GitLabレビューガイド

### 関連リソース

- **[Code Review Best Practices](https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/)** - SmartBearガイド
- **[The Art of Code Review](https://medium.com/@palantir/code-review-best-practices-19e02780015f)** - Palantirベストプラクティス
- **[Conventional Comments](https://conventionalcomments.org/)** - レビューコメント規約

---

## いつ使うか

### 自動的に参照されるケース

- PRを作成する時（セルフレビュー）
- PRをレビューする時
- レビューコメントに返信する時

### 手動で参照すべきケース

- チームのレビュー基準を決定する時
- レビュープロセスを改善する時
- 新メンバーへのレビュー教育時
- レビューで議論が発生した時

---

## レビュー観点

### 7つの主要観点

| 観点 | 確認内容 | 詳細ガイド |
|------|---------|-----------|
| **1. 機能性** | 要件を満たしているか | [guides/01-functionality.md](guides/01-functionality.md) |
| **2. 設計** | アーキテクチャに従っているか | [guides/02-design.md](guides/02-design.md) |
| **3. 可読性** | 理解しやすいか | [guides/03-readability.md](guides/03-readability.md) |
| **4. テスト** | 十分なテストがあるか | [guides/04-testing.md](guides/04-testing.md) |
| **5. セキュリティ** | 脆弱性はないか | [guides/05-security.md](guides/05-security.md) |
| **6. パフォーマンス** | 効率的か | [guides/06-performance.md](guides/06-performance.md) |
| **7. 保守性** | 将来の変更が容易か | [guides/07-maintainability.md](guides/07-maintainability.md) |

完全チェックリスト: [checklists/review-checklist.md](checklists/review-checklist.md)

---

## レビュープロセス

### 1. セルフレビュー（PR作成前）

**必須ステップ**:

```
1. 変更をdiffで確認
2. セルフレビューチェックリスト実行
3. デバッグコード・コメント削除
4. テスト実行
5. ビルド確認
```

詳細: [guides/08-self-review.md](guides/08-self-review.md)
チェックリスト: [checklists/self-review.md](checklists/self-review.md)

### 2. レビュー実施

**レビュワーのステップ**:

```
1. PRの目的・背景を理解
2. テスト・ビルドの通過確認
3. コード全体の構造把握
4. 詳細レビュー（7つの観点）
5. コメント作成
6. 総合判断（Approve/Request Changes）
```

詳細: [guides/09-reviewing.md](guides/09-reviewing.md)

### 3. フィードバック対応

**作成者のステップ**:

```
1. コメントを全て読む
2. 質問に回答
3. 修正が必要なものを修正
4. 議論が必要なものを議論
5. Re-review依頼
```

詳細: [guides/10-feedback-response.md](guides/10-feedback-response.md)

---

## コミュニケーション

### 建設的なフィードバック

#### ✅ Good

```
「ここでnil checkを追加すると、クラッシュを防げます。
Optional bindingを使うのはどうでしょうか？

if let user = user {
    // ...
}
」
```

#### ❌ Bad

```
「これだとクラッシュするよ」
```

### コメントの種類

| プレフィックス | 意味 | 例 |
|--------------|------|-----|
| **[必須]** | 修正必須 | `[必須] メモリリークの可能性` |
| **[推奨]** | 修正推奨 | `[推奨] この関数は分割できます` |
| **[質問]** | 質問・確認 | `[質問] この仕様で良いですか？` |
| **[提案]** | 代替案提示 | `[提案] Combineを使うと簡潔に` |
| **[学習]** | 学習機会 | `[学習] こういう書き方もあります` |
| **[賞賛]** | 良い点 | `[賞賛] いいリファクタですね！` |

詳細: [guides/11-communication.md](guides/11-communication.md)

---

## 自動化

### 自動レビューツール

| ツール | 用途 | 設定ガイド |
|--------|------|-----------|
| **SwiftLint** | コーディング規約 | [scripts/swiftlint-setup.sh](scripts/swiftlint-setup.sh) |
| **SwiftFormat** | フォーマット | [scripts/swiftformat-setup.sh](scripts/swiftformat-setup.sh) |
| **Danger** | PR自動チェック | [scripts/danger-setup.sh](scripts/danger-setup.sh) |
| **SonarQube** | 静的解析 | [scripts/sonarqube-setup.sh](scripts/sonarqube-setup.sh) |

詳細: [guides/12-automation.md](guides/12-automation.md)

### GitHub Actions統合

→ [references/github-actions-integration.md](references/github-actions-integration.md)

---

## よくある問題

### よくある指摘事項

#### 1. 強制アンラップ

```swift
❌ let name = user!.name
✅ guard let user = user else { return }
   let name = user.name
```

#### 2. Magic Number

```swift
❌ if items.count > 10 { ... }
✅ private let maxItemsCount = 10
   if items.count > maxItemsCount { ... }
```

#### 3. 長すぎる関数

```swift
❌ func processUser() { // 100行 }
✅ func processUser() {
      validateUser()
      saveUser()
      notifyUser()
   }
```

全パターン: [references/common-issues.md](references/common-issues.md)

### レビューが遅い

→ [references/review-efficiency.md](references/review-efficiency.md)

### レビューで対立

→ [references/conflict-resolution.md](references/conflict-resolution.md)

---

## Agent連携

### このSkillを使用するAgents

1. **code-review-agent**
   - PR自動レビュー
   - 7つの観点でチェック
   - Thoroughness: `thorough`

2. **style-checker-agent**
   - コーディングスタイル確認
   - Thoroughness: `quick`

3. **security-scanner-agent**
   - セキュリティ脆弱性スキャン
   - Thoroughness: `thorough`

4. **test-coverage-agent**
   - テストカバレッジ確認
   - Thoroughness: `medium`

### 推奨Agentワークフロー

#### PR作成時（並行実行）

```
code-review-agent (包括的レビュー) +
style-checker-agent (スタイルチェック) +
security-scanner-agent (セキュリティ) +
test-coverage-agent (カバレッジ)
→ 結果統合 → PRに自動コメント
```

#### セルフレビュー時（順次実行）

```
self-review-agent (セルフレビュー支援)
→ 指摘事項を確認・修正
→ 再度確認
```

---

## クイックリファレンス

### レビューコメントテンプレート

#### 機能性の問題

```
[必須] エッジケースの考慮不足

現在の実装だと、空配列の場合にクラッシュします。

提案:
guard !items.isEmpty else { return }

テストケース:
- 空配列
- 要素1つ
- 大量の要素
```

#### 設計の改善提案

```
[推奨] 責務の分離

現在ViewControllerに全てのロジックがありますが、
ViewModelに移動することで、テスタビリティが向上します。

メリット:
- テストが容易
- 再利用性向上
- 関心の分離
```

#### 賞賛

```
[賞賛] エラーハンドリング

きちんとエラーケースを考慮されていて素晴らしいです！
ユーザーにわかりやすいメッセージも表示されていますね。
```

---

## 詳細ドキュメント

### Guides（詳細ガイド）

**包括的ガイド（新規追加）**
1. **[ベストプラクティス完全ガイド](guides/best-practices-complete.md)** 🌟
   - レビューの原則と目的
   - レビュワー/作成者/メンテナーの視点
   - 建設的なフィードバック技術
   - セルフレビュー戦略
   - レビューツール活用
   - 言語別ベストプラクティス（TypeScript, Python, Swift, Go）
   - チーム文化の構築
   - ケーススタディ

2. **[チェックリスト完全ガイド](guides/checklist-complete.md)** ✅
   - 総合チェックリスト
   - TypeScript/JavaScriptチェックリスト
   - Pythonチェックリスト
   - Swiftチェックリスト
   - Goチェックリスト
   - セキュリティチェックリスト（OWASP Top 10）
   - パフォーマンスチェックリスト
   - テストチェックリスト
   - アーキテクチャチェックリスト
   - ドキュメントチェックリスト

3. **[レビュープロセス完全ガイド](guides/review-process-complete.md)** 📋
   - コードレビューの基礎
   - レビュープロセス詳細
   - レビュー観点とチェックリスト
   - 効果的なフィードバック
   - レビュー時間管理
   - チーム文化
   - メトリクス測定
   - ベストプラクティス

4. **[自動化完全ガイド](guides/review-automation-complete.md)** 🤖
   - 自動化の基礎
   - Danger.js実装
   - ReviewDog設定
   - 自動ラベリング
   - 自動レビュワー割り当て
   - AI支援レビュー
   - メトリクス自動収集
   - 統合ワークフロー

5. **[レビューテクニック完全ガイド](guides/review-techniques-complete.md)** 🔍
   - レビューテクニック基礎
   - 静的解析活用
   - セキュリティレビュー
   - パフォーマンスレビュー
   - アーキテクチャレビュー
   - テストレビュー
   - ドキュメントレビュー
   - ペアレビュー

### Checklists（チェックリスト）

- [セルフレビュー](checklists/self-review.md)
- [レビュー観点](checklists/review-checklist.md)
- [PR作成前](checklists/pre-pr.md)
- [レビュワー観点](checklists/reviewer-checklist.md)

### Templates（テンプレート）

- **[PRテンプレート](templates/pr-template.md)** - 包括的なPull Requestテンプレート
  - 概要、変更内容、テスト
  - Breaking Changes、マイグレーション
  - セキュリティ、アクセシビリティ
  - セルフレビューチェックリスト

- **[Dangerfile](templates/dangerfile.ts)** - 自動PRチェック
  - PRサイズチェック
  - Conventional Commits検証
  - カバレッジチェック
  - デバッグコード検出
  - セキュリティチェック
  - 影響範囲分析

### Workflows（ワークフロー）

- **[完全レビューワークフロー](workflows/complete-review-workflow.yml)** - GitHub Actions完全自動化
  - 基本チェック（PRサイズ、デバッグコード）
  - Linting & Formatting（ESLint, Prettier, TypeScript）
  - テスト & カバレッジ
  - セキュリティスキャン（npm audit, Snyk, CodeQL）
  - 依存関係分析
  - コード品質分析
  - Danger.js自動レビュー
  - ReviewDog統合
  - パフォーマンステスト
  - 自動ラベリング
  - レビュワー自動割り当て
  - メトリクス収集
  - 通知（Slack）

### References（リファレンス）

- [ベストプラクティス集](references/best-practices.md)
- [よくある指摘事項](references/common-issues.md)
- [レビュー効率化](references/review-efficiency.md)
- [対立解決](references/conflict-resolution.md)
- [GitHub Actions統合](references/github-actions-integration.md)

### Incidents（過去の問題事例）

- [見逃されたバグ](incidents/missed-bugs/)
- [レビュー遅延事例](incidents/review-delays/)
- [コミュニケーション問題](incidents/communication-issues/)

---

## 学習リソース

- 📚 [Google's Code Review Guidelines](https://google.github.io/eng-practices/review/)
- 📖 [The Art of Readable Code](https://www.amazon.com/dp/0596802293)
- 🎥 [Effective Code Reviews](https://www.youtube.com/...)
- 📘 [SwiftLint Rules](https://realm.github.io/SwiftLint/rule-directory.html)

---

## 関連Skills

- `git-workflow` - PR作成プロセス
- `testing-strategy` - テストレビュー観点
- `ios-development` - iOS固有の観点
- `ci-cd-automation` - 自動レビュー統合

---

## 更新履歴

このSkill自体の変更履歴は [CHANGELOG.md](CHANGELOG.md) を参照
