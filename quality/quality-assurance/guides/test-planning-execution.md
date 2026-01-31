# テスト計画・実行ガイド - 完全版

効果的なテスト計画の策定から実行、結果分析までを完全解説。プロジェクトの品質を確保するための実践的なガイドです。

## 目次

1. [テスト計画の基礎](#テスト計画の基礎)
2. [テスト戦略策定](#テスト戦略策定)
3. [テストケース設計](#テストケース設計)
4. [テスト実行管理](#テスト実行管理)
5. [テストデータ管理](#テストデータ管理)
6. [リグレッションテスト](#リグレッションテスト)
7. [テスト環境管理](#テスト環境管理)
8. [実践例とケーススタディ](#実践例とケーススタディ)
9. [トラブルシューティング](#トラブルシューティング)

---

## テスト計画の基礎

### テスト計画とは

**定義:**
テストプロジェクトの範囲、アプローチ、リソース、スケジュールを文書化したもの

**目的:**
- テストの範囲を明確化
- リスクの特定と軽減策の定義
- リソースと時間の最適配分
- ステークホルダーとの期待値の調整

### テスト計画の構成要素

```markdown
## テスト計画書の必須要素

### 1. プロジェクト情報
- プロジェクト名
- バージョン/リリース番号
- 作成日・更新日
- 作成者・承認者

### 2. テスト対象
- 対象機能一覧
- 対象外機能（スコープ外）
- 対象プラットフォーム/デバイス
- 対象OS/ブラウザバージョン

### 3. テスト戦略
- テストレベル（Unit/Integration/E2E）
- テストタイプ（機能/非機能）
- テストアプローチ（手動/自動）
- 優先順位付け基準

### 4. リソース計画
- テストチーム構成
- 役割と責任
- テスト環境
- ツール・ライセンス

### 5. スケジュール
- マイルストーン
- テストフェーズ
- 開始/終了条件
- 依存関係

### 6. 品質基準
- 合格基準（Entry/Exit Criteria）
- カバレッジ目標
- 品質メトリクス
- リリース判定基準

### 7. リスク管理
- 特定されたリスク
- 影響度評価
- 軽減策
- コンティンジェンシープラン

### 8. 成果物
- テストケース
- テスト結果レポート
- バグレポート
- メトリクスダッシュボード
```

---

## テスト戦略策定

### リスクベーステスト

**リスク評価マトリクス:**

```swift
struct RiskAssessment {
    enum Probability: Int {
        case veryLow = 1    // 発生可能性 < 10%
        case low = 2        // 10-30%
        case medium = 3     // 30-50%
        case high = 4       // 50-70%
        case veryHigh = 5   // > 70%
    }

    enum Impact: Int {
        case negligible = 1  // 影響なし
        case minor = 2       // 軽微
        case moderate = 3    // 中程度
        case major = 4       // 大きい
        case critical = 5    // 致命的
    }

    struct Risk {
        let id: String
        let description: String
        let probability: Probability
        let impact: Impact
        let mitigation: String

        var riskScore: Int {
            probability.rawValue * impact.rawValue
        }

        var priority: Priority {
            switch riskScore {
            case 1...4:   return .low
            case 5...9:   return .medium
            case 10...16: return .high
            case 17...25: return .critical
            default:      return .low
            }
        }

        enum Priority: String {
            case low = "Low (1-4)"
            case medium = "Medium (5-9)"
            case high = "High (10-16)"
            case critical = "Critical (17-25)"
        }
    }

    // リスク分析
    func analyzeRisks(_ risks: [Risk]) -> String {
        let critical = risks.filter { $0.priority == .critical }
        let high = risks.filter { $0.priority == .high }
        let medium = risks.filter { $0.priority == .medium }
        let low = risks.filter { $0.priority == .low }

        return """
        リスク分析結果:
        ===============
        Critical: \(critical.count)件 ⚠️
        High:     \(high.count)件
        Medium:   \(medium.count)件
        Low:      \(low.count)件

        即座に対応が必要なリスク:
        \(critical.map { "• [\($0.id)] \($0.description)" }.joined(separator: "\n"))

        優先対応リスク:
        \(high.map { "• [\($0.id)] \($0.description)" }.joined(separator: "\n"))
        """
    }
}

// 使用例
let risks = [
    RiskAssessment.Risk(
        id: "RISK-001",
        description: "決済APIの統合が未完了",
        probability: .high,
        impact: .critical,
        mitigation: "モックAPIでのテスト + 統合テストの優先実施"
    ),
    RiskAssessment.Risk(
        id: "RISK-002",
        description: "iOS 18での動作未確認",
        probability: .medium,
        impact: .major,
        mitigation: "Beta版デバイスでの事前テスト"
    ),
    RiskAssessment.Risk(
        id: "RISK-003",
        description: "大量データ処理時のパフォーマンス",
        probability: .medium,
        impact: .moderate,
        mitigation: "負荷テストの実施"
    ),
]

let assessment = RiskAssessment()
print(assessment.analyzeRisks(risks))
```

### テストレベル別戦略

**1. ユニットテスト戦略:**

```typescript
interface UnitTestStrategy {
  coverage: {
    target: number;           // 目標カバレッジ
    threshold: number;        // 最低ライン
    excludePatterns: string[]; // 除外パターン
  };
  frameworks: string[];       // Jest, Vitest, etc.
  mockingStrategy: 'minimal' | 'moderate' | 'extensive';
  executionFrequency: 'every-commit' | 'every-push' | 'daily';
}

const unitTestStrategy: UnitTestStrategy = {
  coverage: {
    target: 85,
    threshold: 80,
    excludePatterns: [
      '**/*.test.ts',
      '**/*.spec.ts',
      '**/mocks/**',
      '**/fixtures/**',
    ],
  },
  frameworks: ['Vitest', 'Testing Library'],
  mockingStrategy: 'minimal', // 実装依存を最小化
  executionFrequency: 'every-commit',
};

// テスト命名規則
const testNamingConvention = {
  pattern: 'describe-it',
  example: `
    describe('UserService', () => {
      describe('createUser', () => {
        it('should create a new user with valid data', () => {
          // ...
        });

        it('should throw error when email is invalid', () => {
          // ...
        });
      });
    });
  `,
};
```

**2. 統合テスト戦略:**

```swift
struct IntegrationTestStrategy {
    // テスト範囲
    enum Scope {
        case apiIntegration      // API統合
        case databaseIntegration // DB統合
        case serviceIntegration  // サービス間連携
        case thirdPartyIntegration // 外部サービス統合
    }

    // テスト環境
    struct Environment {
        let useRealDatabase: Bool
        let useTestContainers: Bool
        let mockExternalAPIs: Bool
        let isolationLevel: IsolationLevel

        enum IsolationLevel {
            case shared      // 共有環境
            case isolated    // 分離環境
            case perTest     // テスト毎に独立
        }
    }

    // 実行戦略
    struct ExecutionStrategy {
        let parallelization: Bool
        let transactionRollback: Bool
        let dataCleanup: DataCleanupStrategy

        enum DataCleanupStrategy {
            case beforeEach  // 各テスト前
            case afterEach   // 各テスト後
            case beforeSuite // スイート前
            case afterSuite  // スイート後
        }
    }

    let scopes: [Scope]
    let environment: Environment
    let execution: ExecutionStrategy
}

// 使用例
let strategy = IntegrationTestStrategy(
    scopes: [.apiIntegration, .databaseIntegration],
    environment: IntegrationTestStrategy.Environment(
        useRealDatabase: false,
        useTestContainers: true,
        mockExternalAPIs: true,
        isolationLevel: .perTest
    ),
    execution: IntegrationTestStrategy.ExecutionStrategy(
        parallelization: true,
        transactionRollback: true,
        dataCleanup: .afterEach
    )
)
```

**3. E2Eテスト戦略:**

```typescript
interface E2ETestStrategy {
  // 対象フロー
  criticalUserFlows: string[];
  secondaryFlows: string[];

  // ブラウザマトリクス
  browsers: {
    name: string;
    versions: string[];
    priority: 'high' | 'medium' | 'low';
  }[];

  // デバイスマトリクス
  devices: {
    type: 'desktop' | 'tablet' | 'mobile';
    viewportSizes: { width: number; height: number }[];
  }[];

  // 実行戦略
  execution: {
    frequency: 'every-pr' | 'nightly' | 'pre-release';
    parallelization: number;
    retryOnFailure: number;
    headless: boolean;
  };

  // データ戦略
  dataStrategy: {
    useProductionCopy: boolean;
    seedData: boolean;
    cleanup: boolean;
  };
}

const e2eStrategy: E2ETestStrategy = {
  criticalUserFlows: [
    'User Registration',
    'Login Flow',
    'Checkout Process',
    'Payment Flow',
  ],
  secondaryFlows: [
    'Profile Update',
    'Search Functionality',
    'Product Filtering',
  ],
  browsers: [
    { name: 'Chrome', versions: ['latest', 'latest-1'], priority: 'high' },
    { name: 'Safari', versions: ['latest'], priority: 'high' },
    { name: 'Firefox', versions: ['latest'], priority: 'medium' },
  ],
  devices: [
    {
      type: 'desktop',
      viewportSizes: [{ width: 1920, height: 1080 }],
    },
    {
      type: 'mobile',
      viewportSizes: [
        { width: 375, height: 667 }, // iPhone SE
        { width: 390, height: 844 }, // iPhone 14
      ],
    },
  ],
  execution: {
    frequency: 'every-pr',
    parallelization: 4,
    retryOnFailure: 2,
    headless: true,
  },
  dataStrategy: {
    useProductionCopy: false,
    seedData: true,
    cleanup: true,
  },
};
```

---

## テストケース設計

### 境界値分析（Boundary Value Analysis）

```swift
struct BoundaryValueTester {
    // 境界値テストケース生成
    func generateBoundaryTests(
        min: Int,
        max: Int,
        fieldName: String
    ) -> [TestCase] {
        var tests: [TestCase] = []

        // 境界値
        let values = [
            min - 1,        // 最小値未満（無効）
            min,            // 最小値（有効）
            min + 1,        // 最小値超（有効）
            (min + max) / 2, // 中間値（有効）
            max - 1,        // 最大値未満（有効）
            max,            // 最大値（有効）
            max + 1         // 最大値超（無効）
        ]

        for value in values {
            let isValid = value >= min && value <= max
            tests.append(TestCase(
                id: "TC-\(fieldName)-\(value)",
                input: value,
                expected: isValid ? .valid : .invalid,
                description: boundaryDescription(value, min, max)
            ))
        }

        return tests
    }

    private func boundaryDescription(_ value: Int, _ min: Int, _ max: Int) -> String {
        if value < min {
            return "最小値未満: \(value) < \(min) → エラー期待"
        } else if value == min {
            return "最小値: \(value) = \(min) → 正常"
        } else if value == min + 1 {
            return "最小値超: \(value) = \(min) + 1 → 正常"
        } else if value == max - 1 {
            return "最大値未満: \(value) = \(max) - 1 → 正常"
        } else if value == max {
            return "最大値: \(value) = \(max) → 正常"
        } else if value > max {
            return "最大値超: \(value) > \(max) → エラー期待"
        } else {
            return "中間値: \(value) → 正常"
        }
    }

    struct TestCase {
        let id: String
        let input: Int
        let expected: ExpectedResult
        let description: String

        enum ExpectedResult {
            case valid
            case invalid
        }
    }
}

// 使用例
let tester = BoundaryValueTester()
let ageTests = tester.generateBoundaryTests(
    min: 18,
    max: 120,
    fieldName: "Age"
)

for test in ageTests {
    print("\(test.id): \(test.description)")
}
```

### 同値分割（Equivalence Partitioning）

```typescript
interface EquivalenceClass {
  id: string;
  description: string;
  validValues: any[];
  invalidValues: any[];
}

class EquivalencePartitionTester {
  // メールアドレスの同値分割
  generateEmailTests(): EquivalenceClass[] {
    return [
      {
        id: 'EMAIL-VALID',
        description: '有効なメールアドレス',
        validValues: [
          'user@example.com',
          'test.user@example.co.jp',
          'user+tag@example.com',
          'user123@sub.example.com',
        ],
        invalidValues: [],
      },
      {
        id: 'EMAIL-INVALID-FORMAT',
        description: '無効な形式',
        invalidValues: [
          'invalid',
          'user@',
          '@example.com',
          'user @example.com',
          'user@example',
        ],
        validValues: [],
      },
      {
        id: 'EMAIL-EMPTY',
        description: '空文字',
        invalidValues: ['', null, undefined],
        validValues: [],
      },
    ];
  }

  // パスワード強度の同値分割
  generatePasswordTests(): EquivalenceClass[] {
    return [
      {
        id: 'PWD-STRONG',
        description: '強いパスワード（8文字以上、大小英数字記号含む）',
        validValues: [
          'Abcd123!',
          'P@ssw0rd',
          'MyP@ss123',
        ],
        invalidValues: [],
      },
      {
        id: 'PWD-WEAK-LENGTH',
        description: '短すぎる（8文字未満）',
        invalidValues: [
          'Abc123!',
          'Pass1!',
        ],
        validValues: [],
      },
      {
        id: 'PWD-WEAK-NO-UPPERCASE',
        description: '大文字なし',
        invalidValues: [
          'abcd123!',
          'password123!',
        ],
        validValues: [],
      },
      {
        id: 'PWD-WEAK-NO-LOWERCASE',
        description: '小文字なし',
        invalidValues: [
          'ABCD123!',
          'PASSWORD123!',
        ],
        validValues: [],
      },
      {
        id: 'PWD-WEAK-NO-NUMBER',
        description: '数字なし',
        invalidValues: [
          'Abcdefgh!',
          'Password!',
        ],
        validValues: [],
      },
      {
        id: 'PWD-WEAK-NO-SPECIAL',
        description: '記号なし',
        invalidValues: [
          'Abcd1234',
          'Password123',
        ],
        validValues: [],
      },
    ];
  }

  // テストケース生成
  generateTestCases(classes: EquivalenceClass[]): TestCase[] {
    const testCases: TestCase[] = [];
    let id = 1;

    for (const eqClass of classes) {
      // 有効値のテストケース
      for (const value of eqClass.validValues) {
        testCases.push({
          id: `TC-${String(id++).padStart(3, '0')}`,
          equivalenceClass: eqClass.id,
          input: value,
          expected: 'valid',
          description: `${eqClass.description} - 有効: ${value}`,
        });
      }

      // 無効値のテストケース
      for (const value of eqClass.invalidValues) {
        testCases.push({
          id: `TC-${String(id++).padStart(3, '0')}`,
          equivalenceClass: eqClass.id,
          input: value,
          expected: 'invalid',
          description: `${eqClass.description} - 無効: ${value}`,
        });
      }
    }

    return testCases;
  }
}

interface TestCase {
  id: string;
  equivalenceClass: string;
  input: any;
  expected: 'valid' | 'invalid';
  description: string;
}
```

### デシジョンテーブル

```markdown
## デシジョンテーブル例: ログイン機能

| Rule # | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 |
|--------|----|----|----|----|----|----|----|----|
| **Conditions** |
| 登録ユーザー | Y | Y | Y | Y | N | N | N | N |
| パスワード正しい | Y | Y | N | N | - | - | - | - |
| アカウント有効 | Y | N | Y | N | - | - | - | - |
| **Actions** |
| ログイン成功 | X | - | - | - | - | - | - | - |
| アカウント無効エラー | - | X | - | - | - | - | - | - |
| パスワードエラー | - | - | X | X | - | - | - | - |
| ユーザー不存在エラー | - | - | - | - | X | X | X | X |
```

**実装例:**

```swift
struct LoginDecisionTable {
    struct Condition {
        let isRegisteredUser: Bool
        let isPasswordCorrect: Bool?
        let isAccountActive: Bool?
    }

    enum LoginResult {
        case success
        case accountInactive
        case wrongPassword
        case userNotFound
    }

    func evaluate(condition: Condition) -> LoginResult {
        // Rule R5-R8: ユーザーが登録されていない
        guard condition.isRegisteredUser else {
            return .userNotFound
        }

        // Rule R3-R4: パスワードが間違っている
        guard condition.isPasswordCorrect == true else {
            return .wrongPassword
        }

        // Rule R2: アカウントが無効
        guard condition.isAccountActive == true else {
            return .accountInactive
        }

        // Rule R1: すべて満たす
        return .success
    }

    // テストケース生成
    func generateTestCases() -> [(Condition, LoginResult)] {
        return [
            // R1: 成功
            (Condition(isRegisteredUser: true, isPasswordCorrect: true, isAccountActive: true),
             .success),

            // R2: アカウント無効
            (Condition(isRegisteredUser: true, isPasswordCorrect: true, isAccountActive: false),
             .accountInactive),

            // R3: パスワード間違い（アカウント有効）
            (Condition(isRegisteredUser: true, isPasswordCorrect: false, isAccountActive: true),
             .wrongPassword),

            // R4: パスワード間違い（アカウント無効）
            (Condition(isRegisteredUser: true, isPasswordCorrect: false, isAccountActive: false),
             .wrongPassword),

            // R5-R8: ユーザー不存在
            (Condition(isRegisteredUser: false, isPasswordCorrect: nil, isAccountActive: nil),
             .userNotFound),
        ]
    }
}
```

---

## テスト実行管理

### テスト実行トラッキング

```typescript
interface TestExecution {
  id: string;
  testCaseId: string;
  executedBy: string;
  executedAt: Date;
  status: 'passed' | 'failed' | 'blocked' | 'skipped';
  duration: number; // ミリ秒
  environment: string;
  buildVersion: string;
  comments?: string;
  attachments?: string[];
  defects?: string[]; // 関連バグID
}

class TestExecutionManager {
  private executions: TestExecution[] = [];

  // テスト実行記録
  recordExecution(execution: TestExecution): void {
    this.executions.push(execution);
    this.updateMetrics();
    this.checkForAlerts(execution);
  }

  // 実行統計
  getExecutionStats(timeRange?: { start: Date; end: Date }): ExecutionStats {
    let filtered = this.executions;

    if (timeRange) {
      filtered = filtered.filter(
        e => e.executedAt >= timeRange.start && e.executedAt <= timeRange.end
      );
    }

    const total = filtered.length;
    const passed = filtered.filter(e => e.status === 'passed').length;
    const failed = filtered.filter(e => e.status === 'failed').length;
    const blocked = filtered.filter(e => e.status === 'blocked').length;
    const skipped = filtered.filter(e => e.status === 'skipped').length;

    const totalDuration = filtered.reduce((sum, e) => sum + e.duration, 0);
    const avgDuration = total > 0 ? totalDuration / total : 0;

    return {
      total,
      passed,
      failed,
      blocked,
      skipped,
      passRate: total > 0 ? (passed / total) * 100 : 0,
      failRate: total > 0 ? (failed / total) * 100 : 0,
      avgDuration,
      totalDuration,
    };
  }

  // 失敗テストの分析
  analyzeFailures(): FailureAnalysis {
    const failures = this.executions.filter(e => e.status === 'failed');

    // テストケース毎の失敗回数
    const failuresByTestCase = new Map<string, number>();
    for (const failure of failures) {
      const count = failuresByTestCase.get(failure.testCaseId) || 0;
      failuresByTestCase.set(failure.testCaseId, count + 1);
    }

    // 最も失敗しているテストケース
    const topFailingTests = Array.from(failuresByTestCase.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10)
      .map(([testCaseId, count]) => ({ testCaseId, count }));

    // 環境別失敗率
    const failuresByEnv = new Map<string, number>();
    for (const failure of failures) {
      const count = failuresByEnv.get(failure.environment) || 0;
      failuresByEnv.set(failure.environment, count + 1);
    }

    return {
      totalFailures: failures.length,
      topFailingTests,
      failuresByEnvironment: Array.from(failuresByEnv.entries()).map(
        ([env, count]) => ({ environment: env, count })
      ),
    };
  }

  // アラートチェック
  private checkForAlerts(execution: TestExecution): void {
    // 連続失敗の検出
    const recentExecutions = this.executions
      .filter(e => e.testCaseId === execution.testCaseId)
      .slice(-3);

    if (recentExecutions.every(e => e.status === 'failed')) {
      this.sendAlert(`Test ${execution.testCaseId} has failed 3 times in a row`);
    }

    // 実行時間の異常
    const historicalDurations = this.executions
      .filter(e => e.testCaseId === execution.testCaseId)
      .map(e => e.duration);

    if (historicalDurations.length > 0) {
      const avgDuration = historicalDurations.reduce((a, b) => a + b, 0) / historicalDurations.length;

      if (execution.duration > avgDuration * 2) {
        this.sendAlert(`Test ${execution.testCaseId} took ${execution.duration}ms (avg: ${avgDuration}ms)`);
      }
    }
  }

  private updateMetrics(): void {
    // メトリクス更新ロジック
  }

  private sendAlert(message: string): void {
    console.log(`🚨 ALERT: ${message}`);
    // Slack/Email通知
  }
}

interface ExecutionStats {
  total: number;
  passed: number;
  failed: number;
  blocked: number;
  skipped: number;
  passRate: number;
  failRate: number;
  avgDuration: number;
  totalDuration: number;
}

interface FailureAnalysis {
  totalFailures: number;
  topFailingTests: { testCaseId: string; count: number }[];
  failuresByEnvironment: { environment: string; count: number }[];
}
```

### テスト結果レポート生成

```swift
struct TestResultReport {
    let summary: Summary
    let details: [TestCaseResult]
    let metrics: Metrics
    let generatedAt: Date

    struct Summary {
        let totalTests: Int
        let passed: Int
        let failed: Int
        let skipped: Int
        let duration: TimeInterval

        var passRate: Double {
            guard totalTests > 0 else { return 0 }
            return (Double(passed) / Double(totalTests)) * 100
        }
    }

    struct TestCaseResult {
        let id: String
        let name: String
        let status: Status
        let duration: TimeInterval
        let errorMessage: String?
        let stackTrace: String?

        enum Status: String {
            case passed = "✅ Passed"
            case failed = "❌ Failed"
            case skipped = "⏭️ Skipped"
        }
    }

    struct Metrics {
        let coveragePercentage: Double
        let newBugsFound: Int
        let environmentDetails: String
    }

    // HTML レポート生成
    func generateHTML() -> String {
        """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - \(formatDate(generatedAt))</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f0f0f0; padding: 20px; border-radius: 8px; }
                .passed { color: green; }
                .failed { color: red; }
                .skipped { color: gray; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>Test Execution Report</h1>
            <p>Generated: \(formatDate(generatedAt))</p>

            <div class="summary">
                <h2>Summary</h2>
                <p>Total Tests: <strong>\(summary.totalTests)</strong></p>
                <p class="passed">Passed: <strong>\(summary.passed)</strong></p>
                <p class="failed">Failed: <strong>\(summary.failed)</strong></p>
                <p class="skipped">Skipped: <strong>\(summary.skipped)</strong></p>
                <p>Pass Rate: <strong>\(String(format: "%.1f", summary.passRate))%</strong></p>
                <p>Duration: <strong>\(formatDuration(summary.duration))</strong></p>
            </div>

            <h2>Metrics</h2>
            <ul>
                <li>Code Coverage: \(String(format: "%.1f", metrics.coveragePercentage))%</li>
                <li>New Bugs Found: \(metrics.newBugsFound)</li>
                <li>Environment: \(metrics.environmentDetails)</li>
            </ul>

            <h2>Test Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Test Case</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    \(generateTableRows())
                </tbody>
            </table>
        </body>
        </html>
        """
    }

    private func generateTableRows() -> String {
        details.map { result in
            """
            <tr>
                <td>\(result.name)</td>
                <td class="\(result.status.rawValue.lowercased())">\(result.status.rawValue)</td>
                <td>\(formatDuration(result.duration))</td>
                <td>\(result.errorMessage ?? "-")</td>
            </tr>
            """
        }.joined(separator: "\n")
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        String(format: "%.2fs", duration)
    }
}
```

---

## テストデータ管理

### テストデータ戦略

```typescript
interface TestDataStrategy {
  // データソース
  sources: {
    production: boolean;      // 本番データのコピー
    synthetic: boolean;       // 合成データ
    fixtures: boolean;        // 固定データ
    userGenerated: boolean;   // ユーザー生成データ
  };

  // データ管理
  management: {
    versionControl: boolean;  // バージョン管理
    encryption: boolean;      // 暗号化
    anonymization: boolean;   // 匿名化
    cleanup: 'manual' | 'auto' | 'scheduled';
  };

  // データリフレッシュ
  refresh: {
    frequency: 'daily' | 'weekly' | 'monthly';
    automated: boolean;
    validationRequired: boolean;
  };
}

// テストデータファクトリー
class TestDataFactory {
  // ユーザーデータ生成
  createUser(overrides?: Partial<User>): User {
    const faker = require('@faker-js/faker').faker;

    return {
      id: faker.string.uuid(),
      email: faker.internet.email(),
      firstName: faker.person.firstName(),
      lastName: faker.person.lastName(),
      age: faker.number.int({ min: 18, max: 80 }),
      address: {
        street: faker.location.streetAddress(),
        city: faker.location.city(),
        country: faker.location.country(),
        zipCode: faker.location.zipCode(),
      },
      createdAt: faker.date.past(),
      ...overrides,
    };
  }

  // プロダクトデータ生成
  createProduct(overrides?: Partial<Product>): Product {
    const faker = require('@faker-js/faker').faker;

    return {
      id: faker.string.uuid(),
      name: faker.commerce.productName(),
      description: faker.commerce.productDescription(),
      price: parseFloat(faker.commerce.price()),
      category: faker.commerce.department(),
      inStock: faker.datatype.boolean(),
      sku: faker.string.alphanumeric(10).toUpperCase(),
      ...overrides,
    };
  }

  // 注文データ生成
  createOrder(userId: string, productIds: string[]): Order {
    const faker = require('@faker-js/faker').faker;

    return {
      id: faker.string.uuid(),
      userId,
      productIds,
      status: faker.helpers.arrayElement(['pending', 'processing', 'shipped', 'delivered']),
      total: parseFloat(faker.commerce.price({ min: 10, max: 1000 })),
      createdAt: faker.date.recent(),
      shippingAddress: {
        street: faker.location.streetAddress(),
        city: faker.location.city(),
        country: faker.location.country(),
        zipCode: faker.location.zipCode(),
      },
    };
  }

  // 一括データ生成
  createBulkUsers(count: number): User[] {
    return Array.from({ length: count }, () => this.createUser());
  }

  createBulkProducts(count: number): Product[] {
    return Array.from({ length: count }, () => this.createProduct());
  }
}

// 使用例
const factory = new TestDataFactory();

// 単一ユーザー生成
const user = factory.createUser({
  email: 'test@example.com', // 特定の値を上書き
});

// 100人のユーザー生成
const users = factory.createBulkUsers(100);

// 注文生成
const order = factory.createOrder(user.id, [
  'product-id-1',
  'product-id-2',
]);
```

### データベースシーディング

```typescript
import { PrismaClient } from '@prisma/client';
import { TestDataFactory } from './test-data-factory';

class DatabaseSeeder {
  private prisma = new PrismaClient();
  private factory = new TestDataFactory();

  async seedAll(): Promise<void> {
    await this.cleanDatabase();
    await this.seedUsers();
    await this.seedProducts();
    await this.seedOrders();
  }

  async cleanDatabase(): Promise<void> {
    // トランザクションで全データ削除
    await this.prisma.$transaction([
      this.prisma.order.deleteMany(),
      this.prisma.product.deleteMany(),
      this.prisma.user.deleteMany(),
    ]);
  }

  async seedUsers(): Promise<void> {
    const users = this.factory.createBulkUsers(50);

    for (const user of users) {
      await this.prisma.user.create({
        data: user,
      });
    }

    console.log('✅ Seeded 50 users');
  }

  async seedProducts(): Promise<void> {
    const products = this.factory.createBulkProducts(100);

    await this.prisma.product.createMany({
      data: products,
    });

    console.log('✅ Seeded 100 products');
  }

  async seedOrders(): Promise<void> {
    const users = await this.prisma.user.findMany();
    const products = await this.prisma.product.findMany();

    for (let i = 0; i < 200; i++) {
      const randomUser = users[Math.floor(Math.random() * users.length)];
      const randomProducts = products
        .sort(() => 0.5 - Math.random())
        .slice(0, Math.floor(Math.random() * 5) + 1);

      const order = this.factory.createOrder(
        randomUser.id,
        randomProducts.map(p => p.id)
      );

      await this.prisma.order.create({
        data: order,
      });
    }

    console.log('✅ Seeded 200 orders');
  }

  async disconnect(): Promise<void> {
    await this.prisma.$disconnect();
  }
}

// 実行スクリプト
async function main() {
  const seeder = new DatabaseSeeder();

  try {
    console.log('🌱 Starting database seeding...');
    await seeder.seedAll();
    console.log('✅ Database seeding completed');
  } catch (error) {
    console.error('❌ Seeding failed:', error);
    process.exit(1);
  } finally {
    await seeder.disconnect();
  }
}

main();
```

---

## リグレッションテスト

### リグレッションテスト戦略

```swift
struct RegressionTestStrategy {
    // テスト選択戦略
    enum SelectionStrategy {
        case fullRegression      // 全テスト実行
        case impactBased        // 影響範囲ベース
        case riskBased          // リスクベース
        case timeBased          // 時間制約ベース
    }

    // 実行頻度
    enum ExecutionFrequency {
        case everyCommit        // コミット毎
        case everyPR            // PR毎
        case nightly            // 夜間バッチ
        case weekly             // 週次
        case preRelease         // リリース前
    }

    // 優先度
    enum Priority {
        case critical           // 最重要（常に実行）
        case high              // 重要（週1回以上）
        case medium            // 中程度（月1回以上）
        case low               // 低（リリース前のみ）
    }

    struct TestCase {
        let id: String
        let name: String
        let priority: Priority
        let executionTime: TimeInterval
        let lastExecuted: Date?
        let failureHistory: [Date]

        var failureRate: Double {
            // 過去30日の失敗率
            let thirtyDaysAgo = Date().addingTimeInterval(-30 * 86400)
            let recentFailures = failureHistory.filter { $0 > thirtyDaysAgo }
            // 簡易計算（実際は実行回数も考慮）
            return Double(recentFailures.count) / 30.0
        }

        var shouldRunInQuickRegression: Bool {
            // クイックリグレッションに含めるべきか
            return priority == .critical ||
                   failureRate > 0.1 ||
                   executionTime < 30
        }
    }

    // 影響範囲分析
    func analyzeImpact(changedFiles: [String], allTests: [TestCase]) -> [TestCase] {
        // 変更されたファイルに関連するテストを抽出
        // 実際はコードカバレッジデータやAST解析を使用
        var impactedTests: [TestCase] = []

        for file in changedFiles {
            if file.contains("User") {
                impactedTests.append(contentsOf: allTests.filter { $0.name.contains("User") })
            }
            if file.contains("Payment") {
                impactedTests.append(contentsOf: allTests.filter { $0.name.contains("Payment") })
            }
        }

        return Array(Set(impactedTests))
    }

    // テスト選択
    func selectTests(
        strategy: SelectionStrategy,
        allTests: [TestCase],
        timeLimit: TimeInterval? = nil,
        changedFiles: [String] = []
    ) -> [TestCase] {
        switch strategy {
        case .fullRegression:
            return allTests

        case .impactBased:
            return analyzeImpact(changedFiles: changedFiles, allTests: allTests)

        case .riskBased:
            return allTests
                .sorted { $0.failureRate > $1.failureRate }
                .prefix(100)
                .map { $0 }

        case .timeBased:
            guard let limit = timeLimit else { return allTests }

            var selected: [TestCase] = []
            var totalTime: TimeInterval = 0

            // 優先度順、実行時間の短い順に選択
            let sorted = allTests.sorted {
                if $0.priority != $1.priority {
                    return $0.priority.rawValue < $1.priority.rawValue
                }
                return $0.executionTime < $1.executionTime
            }

            for test in sorted {
                if totalTime + test.executionTime <= limit {
                    selected.append(test)
                    totalTime += test.executionTime
                }
            }

            return selected
        }
    }
}
```

### リグレッションスイート管理

```typescript
class RegressionSuiteManager {
  private suites: Map<string, TestSuite> = new Map();

  // スイート定義
  defineSuites(): void {
    // クイックリグレッション（5分以内）
    this.suites.set('quick', {
      name: 'Quick Regression',
      maxDuration: 300, // 5分
      tests: [
        // Critical smoke tests
        'auth.login',
        'auth.logout',
        'payment.checkout',
        'product.search',
      ],
    });

    // 標準リグレッション（30分以内）
    this.suites.set('standard', {
      name: 'Standard Regression',
      maxDuration: 1800, // 30分
      tests: [
        ...this.suites.get('quick')!.tests,
        'user.registration',
        'user.profile',
        'product.filter',
        'product.sort',
        'cart.add',
        'cart.remove',
        'order.history',
      ],
    });

    // 完全リグレッション（2時間以内）
    this.suites.set('full', {
      name: 'Full Regression',
      maxDuration: 7200, // 2時間
      tests: ['**/*'], // すべて
    });
  }

  // スイート実行
  async runSuite(suiteName: string): Promise<SuiteResult> {
    const suite = this.suites.get(suiteName);
    if (!suite) {
      throw new Error(`Suite '${suiteName}' not found`);
    }

    console.log(`Running ${suite.name}...`);
    const startTime = Date.now();

    const results: TestResult[] = [];

    for (const testPattern of suite.tests) {
      // テスト実行（実際の実装）
      const result = await this.executeTest(testPattern);
      results.push(result);

      // 時間制限チェック
      const elapsed = (Date.now() - startTime) / 1000;
      if (elapsed > suite.maxDuration) {
        console.warn(`⚠️ Suite exceeded time limit: ${elapsed}s > ${suite.maxDuration}s`);
        break;
      }
    }

    const duration = (Date.now() - startTime) / 1000;
    const passed = results.filter(r => r.status === 'passed').length;
    const failed = results.filter(r => r.status === 'failed').length;

    return {
      suiteName: suite.name,
      duration,
      totalTests: results.length,
      passed,
      failed,
      passRate: (passed / results.length) * 100,
    };
  }

  private async executeTest(pattern: string): Promise<TestResult> {
    // テスト実行ロジック
    return {
      testId: pattern,
      status: 'passed',
      duration: 1.5,
    };
  }
}

interface TestSuite {
  name: string;
  maxDuration: number; // 秒
  tests: string[];
}

interface TestResult {
  testId: string;
  status: 'passed' | 'failed' | 'skipped';
  duration: number;
}

interface SuiteResult {
  suiteName: string;
  duration: number;
  totalTests: number;
  passed: number;
  failed: number;
  passRate: number;
}
```

---

## テスト環境管理

### 環境構成管理

```yaml
# environments.yml
environments:
  development:
    api_url: "http://localhost:3000"
    database:
      host: "localhost"
      port: 5432
      name: "app_dev"
    features:
      - feature_flags: true
      - analytics: false
    credentials:
      use_test_accounts: true

  staging:
    api_url: "https://staging-api.example.com"
    database:
      host: "staging-db.example.com"
      port: 5432
      name: "app_staging"
    features:
      - feature_flags: true
      - analytics: true
    credentials:
      use_test_accounts: true

  production:
    api_url: "https://api.example.com"
    database:
      host: "prod-db.example.com"
      port: 5432
      name: "app_prod"
    features:
      - feature_flags: false
      - analytics: true
    credentials:
      use_test_accounts: false
```

**環境切り替え:**

```typescript
import { config } from 'dotenv';
import { readFileSync } from 'fs';
import * as yaml from 'js-yaml';

class EnvironmentManager {
  private environments: Map<string, EnvironmentConfig>;
  private currentEnv: string;

  constructor() {
    this.environments = this.loadEnvironments();
    this.currentEnv = process.env.TEST_ENV || 'development';
  }

  private loadEnvironments(): Map<string, EnvironmentConfig> {
    const file = readFileSync('environments.yml', 'utf8');
    const data = yaml.load(file) as { environments: Record<string, EnvironmentConfig> };

    return new Map(Object.entries(data.environments));
  }

  getConfig(): EnvironmentConfig {
    const config = this.environments.get(this.currentEnv);
    if (!config) {
      throw new Error(`Environment '${this.currentEnv}' not found`);
    }
    return config;
  }

  switchEnvironment(env: string): void {
    if (!this.environments.has(env)) {
      throw new Error(`Environment '${env}' not found`);
    }
    this.currentEnv = env;
    console.log(`Switched to ${env} environment`);
  }

  isProduction(): boolean {
    return this.currentEnv === 'production';
  }
}

interface EnvironmentConfig {
  api_url: string;
  database: {
    host: string;
    port: number;
    name: string;
  };
  features: Record<string, boolean>[];
  credentials: {
    use_test_accounts: boolean;
  };
}

// グローバルインスタンス
export const envManager = new EnvironmentManager();

// 使用例
const config = envManager.getConfig();
console.log(`API URL: ${config.api_url}`);
```

---

## 実践例とケーススタディ

### ケーススタディ: モバイルアプリのテスト計画

**プロジェクト概要:**
- iOS/Androidアプリ
- 主要機能: SNS、メッセージング、決済
- リリースサイクル: 2週間

**テスト計画:**

```markdown
# モバイルアプリ テスト計画書 v2.0

## 1. テスト対象

### 対象機能
✅ ユーザー認証（登録、ログイン、パスワードリセット）
✅ プロフィール管理
✅ 投稿機能（作成、編集、削除）
✅ メッセージング（1対1、グループ）
✅ 決済機能（課金、購入履歴）
✅ プッシュ通知

### 対象外
❌ 管理画面（別テスト計画）
❌ 分析ダッシュボード（内部ツール）

## 2. テスト環境

### デバイスマトリクス

| OS | デバイス | バージョン | 優先度 |
|----|---------|-----------|--------|
| iOS | iPhone 14 Pro | 17.0 | High |
| iOS | iPhone SE | 16.0 | Medium |
| iOS | iPad Pro 12.9" | 17.0 | Medium |
| Android | Pixel 7 | Android 14 | High |
| Android | Galaxy S23 | Android 14 | Medium |
| Android | Galaxy Tab | Android 13 | Low |

### ネットワーク条件
- WiFi（高速）
- 4G（標準）
- 3G（低速）
- オフライン

## 3. テスト戦略

### ユニットテスト（70%）
- 目標カバレッジ: 85%
- フレームワーク: XCTest (iOS), JUnit (Android)
- 実行: コミット毎

### 統合テスト（20%）
- API統合テスト
- データベーステスト
- 実行: PR毎

### E2Eテスト（10%）
- クリティカルユーザーフロー
- フレームワーク: Detox (React Native)
- 実行: Nightly

## 4. リスク管理

| リスク | 確率 | 影響 | 対策 |
|--------|------|------|------|
| 決済API障害 | Medium | Critical | モックAPIでのテスト + Sandbox環境テスト |
| プッシュ通知遅延 | High | Medium | 遅延テストシナリオ追加 |
| データ同期失敗 | Medium | High | オフラインモードテスト強化 |

## 5. スケジュール

| Phase | 期間 | 担当 |
|-------|------|------|
| テストケース作成 | 3日 | QAチーム |
| 機能テスト | 5日 | QA + Dev |
| リグレッションテスト | 2日 | QAチーム |
| 最終確認 | 1日 | 全員 |

## 6. 品質基準

### リリース可能基準
- [ ] Criticalバグ: 0件
- [ ] Majorバグ: 3件以下
- [ ] テストPass率: 95%以上
- [ ] クラッシュ率: 0.1%以下
- [ ] コードカバレッジ: 80%以上

## 7. 成果物
- テスト結果レポート (HTML)
- バグレポート (Jira)
- カバレッジレポート (Codecov)
- パフォーマンスレポート (Firebase)
```

---

## トラブルシューティング

### よくある問題と解決策

**1. テスト実行時間が長すぎる**

```markdown
## 問題: テスト実行時間が長い

### 症状
- フルテストスイートが2時間以上かかる
- 開発者がテストを待てない
- CI/CDパイプラインがボトルネック

### 原因
❌ すべてのテストを逐次実行
❌ 重複したセットアップ/ティアダウン
❌ 不要なE2Eテストが多い
❌ テストデータ生成が遅い

### 解決策
✅ 並列実行を有効化（Jest --maxWorkers=4）
✅ テストピラミッドの見直し（E2E削減）
✅ セットアップの共有化
✅ テストデータのキャッシング
✅ クイックスイートの作成（5分以内）
```

**2. テストが不安定（Flaky Tests）**

```markdown
## 問題: テストが不安定

### 症状
- 同じテストが時々失敗する
- ローカルでは成功、CIで失敗
- タイミングに依存した失敗

### 原因
❌ 非同期処理の待機不足
❌ テスト間の依存関係
❌ 共有状態の汚染
❌ ランダムデータの使用
❌ 外部依存のモック不足

### 解決策
✅ 適切な待機処理（waitFor, until）
✅ テスト分離の徹底
✅ beforeEach/afterEachでクリーンアップ
✅ 決定論的なテストデータ
✅ 外部APIのモック化
✅ リトライロジックの削除
```

---

## まとめ

### テスト計画成功の鍵

```markdown
## 成功の5原則

1. **明確なスコープ定義**
   - テスト対象を明確に
   - 対象外も明示
   - リスクベースで優先順位付け

2. **現実的なスケジュール**
   - バッファを確保
   - 依存関係を考慮
   - マイルストーンを設定

3. **適切なリソース配分**
   - スキルセットを考慮
   - ツールへの投資
   - 自動化の推進

4. **継続的な改善**
   - メトリクスの測定
   - 振り返りの実施
   - プロセスの最適化

5. **ステークホルダーとの連携**
   - 期待値の調整
   - 進捗の可視化
   - リスクの共有
```

---

**関連ガイド:**
- [QA Metrics & KPI Dashboard](./qa-metrics-kpi-dashboard.md)
- [Bug Management Complete](./bug-management-complete.md)
- [Release Management & Criteria](./release-management-criteria.md)
