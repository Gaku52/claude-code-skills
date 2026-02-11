# クリーンコード原則

> クリーンコードは読みやすく、変更しやすく、テストしやすいコード。命名規則、関数設計、SOLID 原則、リファクタリングテクニック、コードレビュー文化まで、コード品質の全てを解説する。

## このSkillの対象者

- コード品質を向上させたいエンジニア
- チームのコーディング規約を整備したいリード
- リファクタリングの技法を学びたい方

## 前提知識

- 何らかのプログラミング言語の実務経験
- 基本的な設計パターンの知識

## 学習ガイド

### 00-fundamentals — 基礎原則

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-fundamentals/00-naming.md]] | 命名規則、意図を明確にする名前、一貫性 |
| 01 | [[docs/00-fundamentals/01-functions.md]] | 関数設計、単一責任、引数設計、副作用管理 |
| 02 | [[docs/00-fundamentals/02-solid.md]] | SOLID 原則（SRP/OCP/LSP/ISP/DIP）実践解説 |
| 03 | [[docs/00-fundamentals/03-dry-kiss-yagni.md]] | DRY/KISS/YAGNI、過剰抽象化の罠、適切な複雑さ |

### 01-practices — 実践テクニック

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-practices/00-error-handling.md]] | エラー処理設計、例外 vs Result 型、ログ設計 |
| 01 | [[docs/01-practices/01-testing-for-quality.md]] | テスト駆動設計、テスタビリティ、テストダブル |
| 02 | [[docs/01-practices/02-comments-and-documentation.md]] | コメントの書き方、自己文書化コード、JSDoc/TSDoc |
| 03 | [[docs/01-practices/03-code-organization.md]] | ファイル構成、モジュール設計、依存関係管理 |

### 02-refactoring — リファクタリング

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-refactoring/00-refactoring-catalog.md]] | リファクタリングカタログ（Extract/Inline/Move/Rename） |
| 01 | [[docs/02-refactoring/01-code-smells.md]] | コードスメル一覧、検出方法、修正パターン |
| 02 | [[docs/02-refactoring/02-legacy-code.md]] | レガシーコード改善、安全なリファクタリング、テスト追加戦略 |
| 03 | [[docs/02-refactoring/03-migration-strategies.md]] | 大規模リファクタリング、Strangler Fig、段階的移行 |

### 03-team — チーム品質

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-team/00-coding-standards.md]] | コーディング規約策定、ESLint/Prettier 統一、自動化 |
| 01 | [[docs/03-team/01-code-review-culture.md]] | コードレビュー文化、建設的フィードバック、PR 設計 |
| 02 | [[docs/03-team/02-technical-debt.md]] | 技術的負債の管理、優先順位付け、返済計画 |
| 03 | [[docs/03-team/03-metrics.md]] | コード品質メトリクス、循環的複雑度、SonarQube |

## クイックリファレンス

```
クリーンコードチェックリスト:
  ✓ 意図が明確な命名
  ✓ 関数は20行以内、引数は3個以下
  ✓ 1関数1責任
  ✓ 早期リターンでネスト削減
  ✓ マジックナンバーは定数化
  ✓ コメントより自己文書化コード
```

## 参考文献

1. Martin, R. "Clean Code." Prentice Hall, 2008.
2. Fowler, M. "Refactoring." Addison-Wesley, 2018.
3. Feathers, M. "Working Effectively with Legacy Code." Prentice Hall, 2004.
