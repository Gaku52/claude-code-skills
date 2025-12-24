---
name: testing-strategy
description: Unit、UI、Integration、Snapshotテストの包括的な戦略。テストピラミッド、TDD、BDD、テストカバレッジ目標、モック・スタブ戦略、CI/CD統合まで、品質保証の全てをカバーします。
---

# Testing Strategy Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [テストピラミッド](#テストピラミッド)
4. [テストの種類](#テストの種類)
5. [テスト戦略](#テスト戦略)
6. [ベストプラクティス](#ベストプラクティス)
7. [よくある問題](#よくある問題)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、iOS開発における全てのテスト戦略をカバーします：

- ✅ テストピラミッド（Unit, Integration, UI）
- ✅ XCTest完全ガイド
- ✅ TDD（Test-Driven Development）実践
- ✅ BDD（Behavior-Driven Development）
- ✅ モック・スタブ・フェイク戦略
- ✅ テストカバレッジ目標と測定
- ✅ Snapshotテスト（ビジュアルリグレッション）
- ✅ パフォーマンステスト
- ✅ CI/CD統合
- ✅ テスタビリティ設計

---

## いつ使うか

### 自動的に参照されるケース

- 新しいテストを書く時
- テストが失敗した時
- テスト戦略を決定する時
- コードをテスタブルに設計する時

### 手動で参照すべきケース

- プロジェクト開始時のテスト戦略決定
- テストカバレッジ目標設定
- テスト自動化の導入
- チームメンバーへのテスト教育

---

## テストピラミッド

```
        ┌─────────────┐
        │   UI Tests  │  10% - 遅い、壊れやすい
        ├─────────────┤
        │ Integration │  20% - 中速、重要な統合
        │    Tests    │
        ├─────────────┤
        │             │
        │ Unit Tests  │  70% - 高速、安定
        │             │
        └─────────────┘
```

詳細: [guides/01-test-pyramid.md](guides/01-test-pyramid.md)

---

## テストの種類

### 1. Unit Tests（単体テスト）

**対象**: 個別のクラス・関数
**目的**: ロジックの正確性
**実行時間**: 数ミリ秒

```swift
func testUserProfileViewModel_fetchUser_success() async {
    // Given
    let mockRepository = MockUserRepository()
    mockRepository.userToReturn = User(id: "1", name: "Test")
    let viewModel = UserProfileViewModel(repository: mockRepository)

    // When
    await viewModel.fetchUser(id: "1")

    // Then
    XCTAssertEqual(viewModel.user?.name, "Test")
    XCTAssertFalse(viewModel.isLoading)
}
```

詳細: [guides/02-unit-testing.md](guides/02-unit-testing.md)

### 2. Integration Tests（統合テスト）

**対象**: 複数コンポーネント間の連携
**目的**: 統合動作の確認
**実行時間**: 数秒

詳細: [guides/03-integration-testing.md](guides/03-integration-testing.md)

### 3. UI Tests（UIテスト）

**対象**: ユーザーインタラクション
**目的**: エンドツーエンドシナリオ
**実行時間**: 数十秒〜数分

詳細: [guides/04-ui-testing.md](guides/04-ui-testing.md)

### 4. Snapshot Tests（スナップショットテスト）

**対象**: UI外観
**目的**: ビジュアルリグレッション防止

詳細: [guides/05-snapshot-testing.md](guides/05-snapshot-testing.md)

### 5. Performance Tests（パフォーマンステスト）

**対象**: 実行時間、メモリ使用量
**目的**: パフォーマンス劣化検知

詳細: [guides/06-performance-testing.md](guides/06-performance-testing.md)

---

## テスト戦略

### TDD（Test-Driven Development）

```
1. Red   - 失敗するテストを書く
2. Green - 最小限の実装でテストを通す
3. Refactor - リファクタリング
```

詳細: [guides/07-tdd-practice.md](guides/07-tdd-practice.md)

### BDD（Behavior-Driven Development）

```swift
// Given-When-Then パターン
func testUserLogin() {
    // Given: 初期状態
    // When: アクション
    // Then: 期待結果
}
```

詳細: [guides/08-bdd-practice.md](guides/08-bdd-practice.md)

### テストカバレッジ目標

| コンポーネント | 目標カバレッジ |
|---------------|--------------|
| ビジネスロジック | 90%+ |
| ViewModel | 80%+ |
| Repository | 70%+ |
| UI | 50%+ |
| 全体 | 70%+ |

詳細: [guides/09-coverage-strategy.md](guides/09-coverage-strategy.md)

---

## ベストプラクティス

### モック・スタブ戦略

→ [references/mocking-strategy.md](references/mocking-strategy.md)

### テストデータ管理

→ [references/test-data-management.md](references/test-data-management.md)

### テスタビリティ設計

→ [references/testability-design.md](references/testability-design.md)

### CI/CD統合

→ [references/ci-cd-integration.md](references/ci-cd-integration.md)

---

## よくある問題

### テストが遅い

| 原因 | 解決策 |
|------|--------|
| ネットワーク呼び出し | モック使用 |
| データベースアクセス | インメモリDB |
| UI Tests多すぎ | Unit Testsに置き換え |

詳細: [references/troubleshooting.md](references/troubleshooting.md)

### テストが不安定（Flaky）

→ [incidents/flaky-tests/](incidents/flaky-tests/)

### テストが書きづらい

→ [references/testability-design.md](references/testability-design.md)

---

## Agent連携

### このSkillを使用するAgents

1. **test-generator-agent**
   - 実装からテストコード自動生成
   - Thoroughness: `medium`

2. **test-runner-agent**
   - 全テストスイート実行
   - Thoroughness: `quick`

3. **coverage-analyzer-agent**
   - カバレッジ分析、不足箇所特定
   - Thoroughness: `thorough`

4. **test-refactoring-agent**
   - テストコードのリファクタリング
   - Thoroughness: `medium`

### 推奨Agentワークフロー

#### PR作成時（並行実行）

```
test-runner-agent (全テスト実行) +
coverage-analyzer-agent (カバレッジ確認) +
test-quality-checker-agent (テスト品質評価)
→ 結果統合 → PRコメント
```

#### 新機能実装時（順次実行）

```
test-generator-agent (テスト生成)
→ 手動レビュー・調整
→ test-runner-agent (実行確認)
```

---

## クイックリファレンス

### Unit Test基本形

```swift
import XCTest
@testable import YourApp

final class CalculatorTests: XCTestCase {
    var sut: Calculator!

    override func setUp() {
        super.setUp()
        sut = Calculator()
    }

    override func tearDown() {
        sut = nil
        super.tearDown()
    }

    func test_add_twoPositiveNumbers_returnsSum() {
        // Given
        let a = 5
        let b = 3

        // When
        let result = sut.add(a, b)

        // Then
        XCTAssertEqual(result, 8)
    }
}
```

### 非同期テスト

```swift
func test_fetchData_success() async throws {
    // Given
    let repository = DataRepository()

    // When
    let data = try await repository.fetchData()

    // Then
    XCTAssertNotNil(data)
}
```

### モック例

```swift
protocol UserRepositoryProtocol {
    func fetchUser(id: String) async throws -> User
}

class MockUserRepository: UserRepositoryProtocol {
    var userToReturn: User?
    var errorToThrow: Error?
    var fetchUserCalled = false

    func fetchUser(id: String) async throws -> User {
        fetchUserCalled = true
        if let error = errorToThrow {
            throw error
        }
        return userToReturn!
    }
}
```

---

## 詳細ドキュメント

### Guides（詳細ガイド）

1. [テストピラミッド](guides/01-test-pyramid.md)
2. [Unit Testing完全ガイド](guides/02-unit-testing.md)
3. [Integration Testing](guides/03-integration-testing.md)
4. [UI Testing](guides/04-ui-testing.md)
5. [Snapshot Testing](guides/05-snapshot-testing.md)
6. [Performance Testing](guides/06-performance-testing.md)
7. [TDD実践](guides/07-tdd-practice.md)
8. [BDD実践](guides/08-bdd-practice.md)
9. [カバレッジ戦略](guides/09-coverage-strategy.md)
10. [Quick/Nimble活用](guides/10-quick-nimble.md)

### Checklists（チェックリスト）

- [テスト作成前](checklists/before-writing-tests.md)
- [テストレビュー観点](checklists/test-review.md)
- [リリース前テスト](checklists/pre-release-testing.md)

### Templates（テンプレート）

- [Unit Test Template](templates/unit-test-template.swift)
- [UI Test Template](templates/ui-test-template.swift)
- [Mock Template](templates/mock-template.swift)
- [Test Plan](templates/test-plan.xctestplan)

### References（リファレンス）

- [ベストプラクティス集](references/best-practices.md)
- [モック戦略](references/mocking-strategy.md)
- [テストデータ管理](references/test-data-management.md)
- [テスタビリティ設計](references/testability-design.md)
- [CI/CD統合](references/ci-cd-integration.md)
- [トラブルシューティング](references/troubleshooting.md)

### Incidents（過去の問題事例）

- [Flaky Tests](incidents/flaky-tests/)
- [テスト失敗事例](incidents/test-failures/)
- [カバレッジ低下事例](incidents/coverage-issues/)

---

## 学習リソース

- 📚 [XCTest Documentation](https://developer.apple.com/documentation/xctest)
- 📖 [Test-Driven Development by Example](https://www.amazon.com/dp/0321146530)
- 🎥 [WWDC Testing Sessions](https://developer.apple.com/videos/testing)
- 📘 [Quick/Nimble](https://github.com/Quick/Quick)

---

## 関連Skills

- `code-review` - レビュー時のテスト確認
- `ci-cd-automation` - テスト自動化
- `quality-assurance` - QA全般
- `ios-development` - テスタブルな設計

---

## 更新履歴

このSkill自体の変更履歴は [CHANGELOG.md](CHANGELOG.md) を参照
