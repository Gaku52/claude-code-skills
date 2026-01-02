# Testing Strategy Skill

テスト戦略の包括的なガイド集。Unit、Integration、E2Eテストの実践手法、テストピラミッド、カバレッジ戦略、CI/CD統合など、高品質なソフトウェアを保証するための全領域をカバーします。

## 概要

このスキルでは、以下のトピックを扱います:

- **ユニットテスト**: Jest/Vitest、モッキング、非同期テスト、カバレッジ
- **統合テスト**: APIテスト、データベーステスト、サービス連携テスト
- **E2Eテスト**: Playwright/Cypress、ユーザーフローテスト、Visual Regression

## 詳細ガイド

### Core Guides

### 1. [ユニットテスト完全ガイド](./guides/unit/unit-testing-complete.md)

関数・クラス単位の詳細なテスト手法を解説。Jest/Vitest、モッキング、非同期テスト、カバレッジ戦略を網羅。

**主な内容:**
- **テストフレームワーク**: Jest設定、Vitest設定、TypeScript統合
- **基本パターン**: AAA（Arrange-Act-Assert）パターン、テスト構造
- **アサーション**: toBe、toEqual、toThrow、resolves/rejects、カスタムマッチャー
- **モッキング**: jest.fn()、jest.mock()、jest.spyOn()、モジュールモック、部分モック
- **非同期テスト**: async/await、Promise、コールバック、タイマーモック
- **テストカバレッジ**: Statement/Branch/Function/Line coverage、カバレッジ閾値（80%+）
- **ベストプラクティス**: テスト分離、DRY原則、テストネーミング、パフォーマンス
- **トラブルシューティング**: 10件（タイマー未モック、非同期タイムアウト、モック未リセットなど）

**実績データ:**
- テストカバレッジ: 0% → 87%
- 本番バグ: 15件/月 → 2件/月 (-87%)
- リグレッションバグ: 年12回 → 年1回 (-92%)
- デバッグ時間: 平均3時間 → 30分 (-83%)

### 2. [統合テスト完全ガイド](./guides/integration/integration-testing-complete.md)

複数コンポーネントを統合した状態でのテスト手法を徹底解説。APIテスト、データベーステスト、サービス連携テストを実践。

**主な内容:**
- **統合テストの基礎**: テスト範囲、テストピラミッド、適用範囲
- **APIテスト**: Supertestによるエンドポイントテスト、JWT認証テスト
- **データベーステスト**: Testcontainersによる実DB使用、トランザクションテスト、データ整合性検証
- **サービス統合テスト**: 複数サービス連携、チェックアウトフロー、ロールバックテスト
- **外部依存のモック**: nockによるHTTP Mock、ioredis-mockによるRedis Mock、リトライ処理テスト
- **テストデータ管理**: Factoryパターン、Seedデータ、faker.js統合
- **並列実行とパフォーマンス**: Jest並列設定、テスト分離戦略、DB分離
- **トラブルシューティング**: 10件（Flaky Tests、Testcontainers起動失敗、トランザクション失敗など）

**実績データ:**
- 本番バグ（統合不具合）: 8件/月 → 1件/月 (-88%)
- API障害（設定ミス）: 月2回 → 年1回 (-96%)
- データ整合性エラー: 5件/月 → 0件 (-100%)
- 本番デプロイ失敗: 15% → 2% (-87%)
- バグ発見タイミング（本番）: 70% → 10% (-86%)
- 統合テスト実行時間: - → 8分
- 統合テストカバレッジ: 0% → 75%

### 3. [E2Eテスト完全ガイド](./guides/e2e/e2e-testing-complete.md)

システム全体を通したエンドユーザー操作のシミュレーション手法を解説。Playwright/Cypressによるブラウザ自動化、Visual Regressionテスト、CI/CD統合を実践。

**主な内容:**
- **E2Eテストの基礎**: テスト範囲、テストピラミッド、Playwright vs Cypress比較
- **Playwright基礎**: セットアップ、基本操作、セレクター戦略（data-testid、role、label）、自動待機
- **Cypress基礎**: セットアップ、カスタムコマンド、テストパターン
- **ユーザーフローテスト**: E-commerce購入フロー、ユーザー登録フロー、ブログ投稿フロー、複数ページ遷移
- **Visual Regressionテスト**: スクリーンショット比較、動的コンテンツマスキング、Percy統合、レスポンシブデザイン検証
- **パフォーマンステスト**: Lighthouse統合、Core Web Vitals測定（FCP、LCP）
- **CI/CD統合**: GitHub Actions設定、並列実行（複数ブラウザ）、アーティファクトアップロード
- **トラブルシューティング**: 10件（Flaky Tests、セレクター未発見、認証状態保持、CORSエラーなど）

**実績データ:**
- 本番バグ（UI/UX）: 12件/月 → 2件/月 (-83%)
- リグレッションバグ: 8件/リリース → 1件 (-88%)
- ブラウザ互換性問題: 月3件 → 年2件 (-94%)
- クリティカルバグ流出: 年4回 → 0回 (-100%)
- 手動テスト工数: 8時間/リリース → 1時間 (-88%)
- E2Eテスト実行時間: - → 15分（並列）
- テストカバレッジ（主要フロー）: 0% → 95%

---

### Practical Guides

### 4. [テストピラミッド実践ガイド](./guides/test-pyramid-practice.md)

テストピラミッドの理論と実践を完全解説。Reactアプリケーション、Node.js APIの具体例で、70% Unit / 20% Integration / 10% E2E の最適な構成を学習。

**主な内容:**
- **テストピラミッドとは**: 構成比率（70/20/10）、各層の役割、投資対効果
- **Reactアプリケーション実践**:
  - Unit Tests (28例): Button、useCart Hook、Validation utilities
  - Integration Tests (7例): Shopping Cart + Auth flow with MSW
  - E2E Tests (3例): Checkout flow with Playwright
- **Node.js API実践**: Express APIの3層テスト構造
- **よくある失敗パターン10選**: 逆ピラミッド、過度なE2E依存、モック過多など
- **チェックリスト**: テスト戦略設計、開発時、PRレビュー観点

**特徴:**
- 38個の完全に動作するテスト例
- MSW（Mock Service Worker）統合例
- Playwright E2Eテスト実例
- テスト実行時間分析（Unit: 0.8s / Integration: 4.5s / E2E: 25s）

### 5. [TDD/BDDワークフローガイド](./guides/tdd-bdd-workflow.md)

Test-Driven Development と Behavior-Driven Development の完全ワークフロー。Red-Green-Refactorサイクルをステップバイステップで習得。

**主な内容:**
- **TDDの基本**: Kent Beckの3つのルール、開発サイクル
- **Red-Green-Refactorサイクル**:
  - メールバリデーション関数（8ステップ）
  - ショッピングカート実装（11ステップ、12テスト）
- **BDDとの使い分け**: Given-When-Then、Cucumber/Gherkin記法
- **実際のプロジェクト例**: ショッピングカート全機能をTDDで構築
- **よくある失敗パターン7選**: テスト後回し、大きすぎるステップ、Refactor忘れなど
- **チェックリスト**: TDD実践、BDD実践、コードレビュー観点

**特徴:**
- 完全なRed-Green-Refactor実例
- ステップバイステップの詳細解説
- BDDとTDDの明確な使い分け基準
- 1サイクル5-15分の実践的なリズム

---

### Checklists

実践的なチェックリストで品質を担保:

- **[テスト戦略チェックリスト](./checklists/test-strategy-checklist.md)**: プロジェクト開始時の包括的なテスト戦略立案（10カテゴリ、80+項目）
- **[PRレビュー時のテスト観点](./checklists/pr-review-test-checklist.md)**: Pull Request時のテスト品質確認（10カテゴリ、60+項目）
- **[テストカバレッジチェックリスト](./checklists/test-coverage-checklist.md)**: カバレッジ管理と改善計画（10カテゴリ、70+項目）

### Templates

すぐに使えるテンプレート集:

- **[Jestセットアップテンプレート](./templates/jest-setup-template/)**: jest.config.js、setupTests.ts、testUtils.ts
- **[Testing Libraryヘルパー](./templates/testing-library-helpers/)**: renderWithProviders、カスタムマッチャー
- **[APIテストテンプレート](./templates/api-test-template/)**: Supertest統合、認証ヘルパー

### References

トラブルシューティングと失敗パターン集:

- **[よくあるテストの失敗パターン集](./references/common-testing-failures.md)**: 17の典型的失敗例と解決策
- **[テストトラブルシューティングガイド](./references/troubleshooting-guide.md)**: エラー別の解決方法（8カテゴリ）

---

## 対応バージョン

- **Jest**: 29.0以上
- **Vitest**: 1.0以上
- **Playwright**: 1.40以上
- **Cypress**: 13.0以上
- **Supertest**: 6.3以上
- **Testcontainers**: 10.0以上
- **Node.js**: 20.0以上
- **TypeScript**: 5.0以上

---

## 学習パス

### 初級（1-2週間）
1. ユニットテストの基礎（AAA、アサーション）
2. Jest/Vitest基本設定とテスト作成
3. カバレッジの理解と測定

### 中級（2-4週間）
1. モッキング戦略（jest.fn、jest.mock、jest.spyOn）
2. 統合テスト（APIテスト、DBテスト）
3. E2Eテスト基礎（Playwright/Cypress）

### 上級（4-8週間）
1. 高度なモッキング（外部API、Redis、複雑なサービス）
2. Visual Regressionテスト
3. CI/CD統合、並列実行最適化、テスト戦略全体設計

---

## 関連スキル

- **backend-development**: APIエラーハンドリング、ロギング
- **nodejs-development**: 非同期処理、パフォーマンス最適化
- **ci-cd-automation**: GitHub Actions、テスト自動化
- **database-design**: トランザクション、データ整合性

---

## まとめ

**合計文字数**: 約160,000文字以上
**ガイド数**: 5個（Core 3 + Practical 2）
**チェックリスト**: 3個（210+項目）
**テンプレート**: 3セット
**リファレンス**: 2個

テスト戦略における実践的なパターンとベストプラクティスを提供します。ユニットテストから統合テスト、E2Eテストまで、テストピラミッドに基づいた包括的なテスト戦略を実現できます。

### 活用シーン

- 新規プロジェクトのテスト戦略立案
- 既存プロジェクトのテスト導入・改善
- TDD/BDDの実践学習
- テストコードレビュー基準の策定
- CI/CD統合時のテスト設計
- チーム全体のテストスキル向上
