# QA Metrics & KPI Dashboard - 完全ガイド

品質保証における測定指標とKPIダッシュボードの構築・運用を完全解説。データドリブンな品質管理を実現するための実践的なガイドです。

## 目次

1. [QAメトリクスの基礎](#qaメトリクスの基礎)
2. [主要KPI一覧](#主要kpi一覧)
3. [バグメトリクス](#バグメトリクス)
4. [テストメトリクス](#テストメトリクス)
5. [パフォーマンスメトリクス](#パフォーマンスメトリクス)
6. [品質ダッシュボード構築](#品質ダッシュボード構築)
7. [レポーティング戦略](#レポーティング戦略)
8. [実践例とケーススタディ](#実践例とケーススタディ)
9. [トラブルシューティング](#トラブルシューティング)

---

## QAメトリクスの基礎

### メトリクスとは

**定義:**
- 品質を数値化して測定可能にする指標
- 改善活動の効果を定量的に評価
- データに基づいた意思決定を支援

### メトリクスの種類

**1. プロダクトメトリクス（Product Metrics）:**
```
製品の品質を直接測定
- バグ数、バグ密度
- クラッシュ率
- パフォーマンス指標
- ユーザー満足度
```

**2. プロセスメトリクス（Process Metrics）:**
```
QAプロセスの効率を測定
- テストカバレッジ
- テスト実行時間
- バグ発見率
- バグ修正時間
```

**3. プロジェクトメトリクス（Project Metrics）:**
```
プロジェクト全体の健全性を測定
- スケジュール遵守率
- リリース品質
- チーム生産性
```

### SMART原則

**効果的なメトリクスの条件:**

```markdown
## SMART Metrics

✅ **Specific（具体的）**
   - 何を測定するか明確
   - 例: "テストケース実行率 95%以上"

✅ **Measurable（測定可能）**
   - 数値化できる
   - 例: "クラッシュ率 0.1%以下"

✅ **Achievable（達成可能）**
   - 現実的な目標
   - 例: "コードカバレッジ 80%以上（現在60%）"

✅ **Relevant（関連性がある）**
   - ビジネス目標と整合
   - 例: "ユーザー満足度スコア 4.5/5.0以上"

✅ **Time-bound（期限がある）**
   - 測定期間が明確
   - 例: "次回リリース（2週間後）までに"
```

---

## 主要KPI一覧

### 品質KPI体系

```swift
// 品質KPI管理システム
struct QualityKPIs {
    // 1. バグ関連KPI
    struct BugKPIs {
        let totalBugs: Int                    // 総バグ数
        let criticalBugs: Int                 // Criticalバグ数
        let bugDensity: Double                // バグ密度 (件/KLOC)
        let bugResolutionTime: TimeInterval   // 平均修正時間
        let bugReopenRate: Double             // 再発率 (%)
        let bugEscapeRate: Double             // 流出率 (%)

        // 目標値
        static let targets = (
            criticalBugs: 0,
            bugDensity: 1.0,        // 1件/1000行以下
            resolutionTime: 86400,  // 24時間以内
            reopenRate: 5.0,        // 5%以下
            escapeRate: 10.0        // 10%以下
        )
    }

    // 2. テスト関連KPI
    struct TestKPIs {
        let testCoverage: Double              // テストカバレッジ (%)
        let testExecutionRate: Double         // テスト実行率 (%)
        let testPassRate: Double              // テスト成功率 (%)
        let automationRate: Double            // 自動化率 (%)
        let testExecutionTime: TimeInterval   // テスト実行時間

        static let targets = (
            coverage: 80.0,         // 80%以上
            executionRate: 95.0,    // 95%以上
            passRate: 95.0,         // 95%以上
            automationRate: 70.0,   // 70%以上
            executionTime: 1800     // 30分以内
        )
    }

    // 3. パフォーマンスKPI
    struct PerformanceKPIs {
        let crashRate: Double                 // クラッシュ率 (%)
        let anrRate: Double                   // ANR率 (%)
        let appLaunchTime: TimeInterval       // 起動時間 (秒)
        let apiResponseTime: TimeInterval     // API応答時間 (ミリ秒)
        let memoryUsage: Double               // メモリ使用量 (MB)

        static let targets = (
            crashRate: 0.1,         // 0.1%以下
            anrRate: 0.05,          // 0.05%以下
            launchTime: 2.0,        // 2秒以内
            apiResponse: 1000,      // 1秒以内
            memory: 200.0           // 200MB以下
        )
    }

    // 4. ユーザー満足度KPI
    struct UserSatisfactionKPIs {
        let appStoreRating: Double            // ストア評価
        let nps: Double                       // Net Promoter Score
        let userRetentionRate: Double         // ユーザー継続率 (%)
        let supportTickets: Int               // サポートチケット数

        static let targets = (
            rating: 4.5,            // 4.5/5.0以上
            nps: 50.0,              // 50以上
            retention: 80.0,        // 80%以上
            tickets: 100            // 月100件以下
        )
    }
}
```

### KPI測定頻度

| KPI | 測定頻度 | 報告先 | アクショントリガー |
|-----|---------|--------|------------------|
| Criticalバグ数 | リアルタイム | 全員 | 1件発生時即対応 |
| クラッシュ率 | 日次 | QA/Dev | 0.15%超過時 |
| テストカバレッジ | コミット毎 | Dev | 80%未満時警告 |
| バグ密度 | 週次 | QA/PM | 2.0超過時改善計画 |
| テスト実行率 | 日次 | QA | 90%未満時エスカレーション |
| ユーザー満足度 | 週次 | 経営層 | 4.0未満時改善会議 |

---

## バグメトリクス

### バグ密度（Bug Density）

**定義:**
```
バグ密度 = 総バグ数 ÷ (コード行数 ÷ 1000)
単位: 件/KLOC（1000行あたり）
```

**実装例:**

```swift
struct BugDensityCalculator {
    func calculateBugDensity(
        totalBugs: Int,
        linesOfCode: Int
    ) -> Double {
        let kloc = Double(linesOfCode) / 1000.0
        return Double(totalBugs) / kloc
    }

    func assessQuality(density: Double) -> QualityLevel {
        switch density {
        case ..<0.5:
            return .excellent  // 優秀
        case 0.5..<1.0:
            return .good       // 良好
        case 1.0..<2.0:
            return .fair       // 許容範囲
        default:
            return .poor       // 改善必要
        }
    }

    enum QualityLevel: String {
        case excellent = "優秀 (0.5未満)"
        case good = "良好 (0.5-1.0)"
        case fair = "許容範囲 (1.0-2.0)"
        case poor = "改善必要 (2.0以上)"
    }
}

// 使用例
let calculator = BugDensityCalculator()
let density = calculator.calculateBugDensity(
    totalBugs: 15,
    linesOfCode: 10000
)
// density = 1.5 (件/KLOC)

let quality = calculator.assessQuality(density: density)
// quality = .fair
```

### バグ流出率（Bug Escape Rate）

**定義:**
```
バグ流出率 = (本番で発見されたバグ数 ÷ 総バグ数) × 100
目標: 10%以下
```

**実装例:**

```swift
struct BugEscapeRateTracker {
    struct BugStats {
        let foundInDev: Int      // 開発中発見
        let foundInQA: Int       // QA中発見
        let foundInStaging: Int  // Staging発見
        let foundInProduction: Int // 本番発見

        var total: Int {
            foundInDev + foundInQA + foundInStaging + foundInProduction
        }

        var escapeRate: Double {
            guard total > 0 else { return 0 }
            return (Double(foundInProduction) / Double(total)) * 100
        }

        var qaEffectiveness: Double {
            guard total > 0 else { return 0 }
            let caughtBeforeProduction = foundInDev + foundInQA + foundInStaging
            return (Double(caughtBeforeProduction) / Double(total)) * 100
        }
    }

    // バグフェーズ分析
    func analyzeBugPhases(stats: BugStats) -> String {
        """
        バグ検出フェーズ分析:
        - 開発中: \(stats.foundInDev)件 (\(percentage(stats.foundInDev, total: stats.total))%)
        - QA中: \(stats.foundInQA)件 (\(percentage(stats.foundInQA, total: stats.total))%)
        - Staging: \(stats.foundInStaging)件 (\(percentage(stats.foundInStaging, total: stats.total))%)
        - 本番: \(stats.foundInProduction)件 (\(percentage(stats.foundInProduction, total: stats.total))%)

        バグ流出率: \(String(format: "%.1f", stats.escapeRate))%
        QA有効性: \(String(format: "%.1f", stats.qaEffectiveness))%
        """
    }

    private func percentage(_ value: Int, total: Int) -> String {
        guard total > 0 else { return "0.0" }
        return String(format: "%.1f", (Double(value) / Double(total)) * 100)
    }
}

// 使用例
let stats = BugEscapeRateTracker.BugStats(
    foundInDev: 45,
    foundInQA: 32,
    foundInStaging: 8,
    foundInProduction: 5
)

print("バグ流出率: \(stats.escapeRate)%")  // 5.6%
print("QA有効性: \(stats.qaEffectiveness)%")  // 94.4%
```

### バグ再発率（Bug Reopen Rate）

**定義:**
```
バグ再発率 = (再オープンされたバグ数 ÷ クローズされたバグ数) × 100
目標: 5%以下
```

**実装例:**

```typescript
interface BugLifecycle {
  id: string;
  openedAt: Date;
  closedAt?: Date;
  reopenedAt?: Date;
  status: 'open' | 'closed' | 'reopened';
}

class BugReopenRateAnalyzer {
  calculateReopenRate(bugs: BugLifecycle[]): number {
    const closedBugs = bugs.filter(b => b.status === 'closed' || b.reopenedAt);
    const reopenedBugs = bugs.filter(b => b.reopenedAt);

    if (closedBugs.length === 0) return 0;

    return (reopenedBugs.length / closedBugs.length) * 100;
  }

  analyzeReopenReasons(bugs: BugLifecycle[]): Map<string, number> {
    const reasons = new Map<string, number>();

    // 再オープン理由を分類
    // 実際にはバグデータに理由フィールドがあると仮定

    return reasons;
  }

  generateReport(bugs: BugLifecycle[]): string {
    const reopenRate = this.calculateReopenRate(bugs);
    const totalClosed = bugs.filter(b => b.status === 'closed' || b.reopenedAt).length;
    const reopened = bugs.filter(b => b.reopenedAt).length;

    return `
バグ再発率レポート
==================
クローズ済みバグ: ${totalClosed}件
再オープンバグ: ${reopened}件
再発率: ${reopenRate.toFixed(1)}%

評価: ${reopenRate < 5 ? '✅ 目標達成' : '⚠️ 改善必要'}
    `.trim();
  }
}
```

### バグエイジング（Bug Aging）

**バグの滞留時間を追跡:**

```swift
struct BugAgingAnalyzer {
    struct BugAge {
        let bugId: String
        let priority: Priority
        let openedDate: Date
        let currentDate: Date

        enum Priority: String {
            case critical = "Critical"
            case major = "Major"
            case minor = "Minor"
            case trivial = "Trivial"

            var sla: TimeInterval {
                switch self {
                case .critical: return 86400        // 1日
                case .major: return 86400 * 3       // 3日
                case .minor: return 86400 * 7       // 7日
                case .trivial: return 86400 * 30    // 30日
                }
            }
        }

        var age: TimeInterval {
            currentDate.timeIntervalSince(openedDate)
        }

        var ageDays: Int {
            Int(age / 86400)
        }

        var isOverdue: Bool {
            age > priority.sla
        }

        var overdueBy: TimeInterval {
            max(0, age - priority.sla)
        }
    }

    func analyzeBugAging(bugs: [BugAge]) -> String {
        let overdueBugs = bugs.filter { $0.isOverdue }
        let criticalOverdue = overdueBugs.filter { $0.priority == .critical }

        let avgAge = bugs.reduce(0.0) { $0 + $1.age } / Double(bugs.count)
        let avgAgeDays = Int(avgAge / 86400)

        return """
        バグエイジング分析:
        - 総バグ数: \(bugs.count)件
        - 期限超過: \(overdueBugs.count)件 (\(percentage(overdueBugs.count, bugs.count))%)
        - Critical期限超過: \(criticalOverdue.count)件 ⚠️
        - 平均滞留日数: \(avgAgeDays)日

        最古のバグ:
        \(oldestBugs(bugs).map { "  - \($0.bugId): \($0.ageDays)日経過" }.joined(separator: "\n"))
        """
    }

    private func percentage(_ value: Int, _ total: Int) -> String {
        guard total > 0 else { return "0" }
        return String(format: "%.0f", Double(value) / Double(total) * 100)
    }

    private func oldestBugs(_ bugs: [BugAge]) -> [BugAge] {
        Array(bugs.sorted { $0.age > $1.age }.prefix(5))
    }
}
```

---

## テストメトリクス

### テストカバレッジ

**4つのカバレッジ指標:**

```swift
struct CodeCoverage {
    let statementCoverage: Double   // ステートメントカバレッジ
    let branchCoverage: Double      // 分岐カバレッジ
    let functionCoverage: Double    // 関数カバレッジ
    let lineCoverage: Double        // 行カバレッジ

    // 総合評価
    var overallScore: Double {
        (statementCoverage + branchCoverage + functionCoverage + lineCoverage) / 4.0
    }

    var grade: String {
        switch overallScore {
        case 90...100: return "A (優秀)"
        case 80..<90:  return "B (良好)"
        case 70..<80:  return "C (許容)"
        case 60..<70:  return "D (要改善)"
        default:       return "F (不十分)"
        }
    }

    func report() -> String {
        """
        コードカバレッジレポート
        ========================
        ステートメントカバレッジ: \(format(statementCoverage))%
        分岐カバレッジ:           \(format(branchCoverage))%
        関数カバレッジ:           \(format(functionCoverage))%
        行カバレッジ:             \(format(lineCoverage))%

        総合スコア: \(format(overallScore))%
        評価: \(grade)

        目標達成状況: \(overallScore >= 80 ? "✅" : "❌")
        """
    }

    private func format(_ value: Double) -> String {
        String(format: "%.1f", value)
    }
}

// Xcodeでのカバレッジ取得
class XcodeCoverageExtractor {
    func extractCoverage(from xcresultPath: String) -> CodeCoverage? {
        // xccov を使用してカバレッジデータを抽出
        let command = """
        xcrun xccov view --report --json \(xcresultPath)
        """

        // 実行とパース処理
        // ...

        return CodeCoverage(
            statementCoverage: 85.3,
            branchCoverage: 78.9,
            functionCoverage: 92.1,
            lineCoverage: 84.7
        )
    }
}
```

### テスト実行メトリクス

**テスト実行の効率性を測定:**

```typescript
interface TestExecutionMetrics {
  totalTests: number;
  executedTests: number;
  passedTests: number;
  failedTests: number;
  skippedTests: number;
  executionTime: number; // ミリ秒

  // 計算プロパティ
  executionRate: number;   // 実行率
  passRate: number;        // 成功率
  failRate: number;        // 失敗率
  averageTestTime: number; // 平均テスト時間
}

class TestMetricsCalculator {
  calculate(results: TestResult[]): TestExecutionMetrics {
    const total = results.length;
    const executed = results.filter(r => r.status !== 'skipped').length;
    const passed = results.filter(r => r.status === 'passed').length;
    const failed = results.filter(r => r.status === 'failed').length;
    const skipped = results.filter(r => r.status === 'skipped').length;
    const totalTime = results.reduce((sum, r) => sum + r.duration, 0);

    return {
      totalTests: total,
      executedTests: executed,
      passedTests: passed,
      failedTests: failed,
      skippedTests: skipped,
      executionTime: totalTime,
      executionRate: (executed / total) * 100,
      passRate: executed > 0 ? (passed / executed) * 100 : 0,
      failRate: executed > 0 ? (failed / executed) * 100 : 0,
      averageTestTime: executed > 0 ? totalTime / executed : 0,
    };
  }

  generateReport(metrics: TestExecutionMetrics): string {
    return `
テスト実行メトリクス
====================
総テスト数:     ${metrics.totalTests}件
実行済み:       ${metrics.executedTests}件 (${metrics.executionRate.toFixed(1)}%)
成功:           ${metrics.passedTests}件 (${metrics.passRate.toFixed(1)}%)
失敗:           ${metrics.failedTests}件 (${metrics.failRate.toFixed(1)}%)
スキップ:       ${metrics.skippedTests}件

実行時間:       ${(metrics.executionTime / 1000).toFixed(1)}秒
平均テスト時間: ${metrics.averageTestTime.toFixed(0)}ms

評価: ${this.evaluateMetrics(metrics)}
    `.trim();
  }

  private evaluateMetrics(metrics: TestExecutionMetrics): string {
    const issues: string[] = [];

    if (metrics.executionRate < 95) {
      issues.push('⚠️ テスト実行率が低い（目標95%以上）');
    }
    if (metrics.passRate < 95) {
      issues.push('❌ テスト成功率が低い（目標95%以上）');
    }
    if (metrics.executionTime > 1800000) {
      issues.push('⏱️ テスト実行時間が長い（目標30分以内）');
    }

    return issues.length === 0
      ? '✅ すべての目標を達成'
      : issues.join('\n');
  }
}
```

### テスト自動化率

```swift
struct TestAutomationMetrics {
    let totalTestCases: Int
    let automatedTestCases: Int
    let manualTestCases: Int

    var automationRate: Double {
        guard totalTestCases > 0 else { return 0 }
        return (Double(automatedTestCases) / Double(totalTestCases)) * 100
    }

    var manualEffort: TimeInterval {
        // 手動テスト1件あたり平均10分と仮定
        Double(manualTestCases) * 600
    }

    var automationROI: String {
        let savedTime = manualEffort
        let savedHours = savedTime / 3600

        return """
        自動化ROI分析:
        - 自動化率: \(String(format: "%.1f", automationRate))%
        - 手動テスト件数: \(manualTestCases)件
        - 手動テスト工数: \(String(format: "%.1f", savedHours))時間/回
        - 週次実行で節約: \(String(format: "%.1f", savedHours * 5))時間/週
        - 月次実行で節約: \(String(format: "%.1f", savedHours * 20))時間/月
        """
    }

    func automationPriority() -> [String] {
        var priorities: [String] = []

        if automationRate < 50 {
            priorities.append("🔴 Critical: 自動化率50%未満")
        } else if automationRate < 70 {
            priorities.append("🟡 High: 自動化率70%未満")
        } else {
            priorities.append("🟢 Good: 自動化率70%以上")
        }

        return priorities
    }
}
```

---

## パフォーマンスメトリクス

### クラッシュ率の測定

**Firebase Crashlytics統合:**

```swift
import FirebaseCrashlytics

struct CrashMetrics {
    let totalSessions: Int
    let crashedSessions: Int
    let crashFreeUsers: Double

    var crashRate: Double {
        guard totalSessions > 0 else { return 0 }
        return (Double(crashedSessions) / Double(totalSessions)) * 100
    }

    var isHealthy: Bool {
        crashRate <= 0.1 && crashFreeUsers >= 99.9
    }
}

class CrashMetricsTracker {
    func fetchMetrics(timeRange: TimeRange) async -> CrashMetrics {
        // Firebase Analyticsから取得
        let sessions = await fetchTotalSessions(timeRange)
        let crashes = await fetchCrashedSessions(timeRange)
        let crashFreeUserRate = await fetchCrashFreeUsers(timeRange)

        return CrashMetrics(
            totalSessions: sessions,
            crashedSessions: crashes,
            crashFreeUsers: crashFreeUserRate
        )
    }

    func generateAlert(metrics: CrashMetrics) -> String? {
        if metrics.crashRate > 0.15 {
            return """
            🚨 クラッシュ率アラート
            現在のクラッシュ率: \(String(format: "%.2f", metrics.crashRate))%
            目標: 0.1%以下
            即座に対応が必要です。
            """
        }
        return nil
    }
}
```

### アプリ起動時間

```swift
import os.signpost

class AppLaunchMetrics {
    private let log = OSLog(subsystem: "com.app", category: .pointsOfInterest)
    private var launchStart: Date?

    func trackLaunchStart() {
        launchStart = Date()
        os_signpost(.begin, log: log, name: "App Launch")
    }

    func trackLaunchEnd() {
        guard let start = launchStart else { return }

        let duration = Date().timeIntervalSince(start)
        os_signpost(.end, log: log, name: "App Launch")

        // メトリクス記録
        recordLaunchTime(duration)

        // アラート判定
        if duration > 2.0 {
            print("⚠️ 起動時間が目標を超過: \(duration)秒")
        }
    }

    private func recordLaunchTime(_ duration: TimeInterval) {
        // Analytics に送信
        Analytics.logEvent("app_launch_time", parameters: [
            "duration_ms": Int(duration * 1000)
        ])
    }
}

// 使用例
// AppDelegate.swift
func application(_ application: UIApplication,
                 didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
    AppLaunchMetrics.shared.trackLaunchStart()

    // 初期化処理...

    DispatchQueue.main.async {
        AppLaunchMetrics.shared.trackLaunchEnd()
    }

    return true
}
```

### API応答時間

```swift
struct APIPerformanceMetrics {
    let endpoint: String
    let responseTime: TimeInterval
    let statusCode: Int
    let timestamp: Date

    var isHealthy: Bool {
        responseTime < 1.0 && (200..<300).contains(statusCode)
    }
}

class APIMetricsCollector {
    private var metrics: [APIPerformanceMetrics] = []

    func track(endpoint: String, responseTime: TimeInterval, statusCode: Int) {
        let metric = APIPerformanceMetrics(
            endpoint: endpoint,
            responseTime: responseTime,
            statusCode: statusCode,
            timestamp: Date()
        )

        metrics.append(metric)

        // リアルタイムアラート
        if responseTime > 2.0 {
            sendSlowAPIAlert(endpoint: endpoint, time: responseTime)
        }
    }

    func generateReport(timeRange: TimeRange) -> String {
        let filtered = metrics.filter { timeRange.contains($0.timestamp) }

        let avgResponseTime = filtered.reduce(0.0) { $0 + $1.responseTime } / Double(filtered.count)
        let slowRequests = filtered.filter { $0.responseTime > 1.0 }

        return """
        API パフォーマンスレポート
        ==========================
        総リクエスト数: \(filtered.count)
        平均応答時間: \(String(format: "%.0f", avgResponseTime * 1000))ms
        遅延リクエスト: \(slowRequests.count)件 (\(String(format: "%.1f", Double(slowRequests.count) / Double(filtered.count) * 100))%)

        最遅エンドポイント:
        \(slowestEndpoints(filtered).map { "  - \($0.endpoint): \(String(format: "%.0f", $0.responseTime * 1000))ms" }.joined(separator: "\n"))
        """
    }

    private func slowestEndpoints(_ metrics: [APIPerformanceMetrics]) -> [APIPerformanceMetrics] {
        Array(metrics.sorted { $0.responseTime > $1.responseTime }.prefix(5))
    }

    private func sendSlowAPIAlert(endpoint: String, time: TimeInterval) {
        print("⚠️ 遅延API検出: \(endpoint) - \(String(format: "%.0f", time * 1000))ms")
    }
}
```

---

## 品質ダッシュボード構築

### リアルタイムダッシュボード設計

**ダッシュボード要件:**

```markdown
## 品質ダッシュボード要件

### 1. リアルタイム性
- データ更新頻度: 5分毎
- クリティカルメトリクス: リアルタイム
- アラート通知: 即時

### 2. 視覚化
- トレンドグラフ（時系列）
- ステータスインジケーター（Red/Yellow/Green）
- 比較チャート（目標 vs 実績）

### 3. アクセス性
- Webダッシュボード
- モバイル対応
- Slack統合（アラート）
```

### Grafana + Prometheus実装例

**docker-compose.yml:**

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana-dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

**prometheus.yml:**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'qa-metrics'
    static_configs:
      - targets: ['metrics-exporter:8080']
    metrics_path: '/metrics'
```

**メトリクスエクスポーター（Node.js）:**

```typescript
import express from 'express';
import client from 'prom-client';

const app = express();
const register = new client.Registry();

// メトリクス定義
const bugCountGauge = new client.Gauge({
  name: 'qa_bug_count_total',
  help: '総バグ数',
  labelNames: ['severity', 'status'],
  registers: [register],
});

const testCoverageGauge = new client.Gauge({
  name: 'qa_test_coverage_percentage',
  help: 'テストカバレッジ (%)',
  labelNames: ['type'],
  registers: [register],
});

const crashRateGauge = new client.Gauge({
  name: 'qa_crash_rate_percentage',
  help: 'クラッシュ率 (%)',
  registers: [register],
});

const testExecutionCounter = new client.Counter({
  name: 'qa_test_executions_total',
  help: 'テスト実行回数',
  labelNames: ['status'],
  registers: [register],
});

// メトリクス更新関数
async function updateMetrics() {
  // バグカウント更新
  const bugs = await fetchBugsFromJira();
  bugCountGauge.labels('critical', 'open').set(bugs.critical.open);
  bugCountGauge.labels('major', 'open').set(bugs.major.open);
  bugCountGauge.labels('minor', 'open').set(bugs.minor.open);

  // カバレッジ更新
  const coverage = await fetchCoverageFromCI();
  testCoverageGauge.labels('statement').set(coverage.statement);
  testCoverageGauge.labels('branch').set(coverage.branch);
  testCoverageGauge.labels('function').set(coverage.function);

  // クラッシュ率更新
  const crashRate = await fetchCrashRateFromFirebase();
  crashRateGauge.set(crashRate);
}

// 定期更新（5分毎）
setInterval(updateMetrics, 5 * 60 * 1000);
updateMetrics(); // 初回実行

// メトリクスエンドポイント
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});

app.listen(8080, () => {
  console.log('Metrics exporter running on port 8080');
});
```

### Grafanaダッシュボード定義

**qa-dashboard.json:**

```json
{
  "dashboard": {
    "title": "QA Metrics Dashboard",
    "panels": [
      {
        "id": 1,
        "title": "Bug Count by Severity",
        "type": "graph",
        "targets": [
          {
            "expr": "sum by (severity) (qa_bug_count_total{status=\"open\"})"
          }
        ],
        "gridPos": { "x": 0, "y": 0, "w": 12, "h": 8 }
      },
      {
        "id": 2,
        "title": "Test Coverage",
        "type": "gauge",
        "targets": [
          {
            "expr": "qa_test_coverage_percentage{type=\"statement\"}"
          }
        ],
        "gridPos": { "x": 12, "y": 0, "w": 6, "h": 8 },
        "fieldConfig": {
          "defaults": {
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                { "value": 0, "color": "red" },
                { "value": 70, "color": "yellow" },
                { "value": 80, "color": "green" }
              ]
            }
          }
        }
      },
      {
        "id": 3,
        "title": "Crash Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "qa_crash_rate_percentage"
          }
        ],
        "gridPos": { "x": 18, "y": 0, "w": 6, "h": 8 },
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 0.1, "color": "yellow" },
                { "value": 0.2, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "id": 4,
        "title": "Test Execution Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(qa_test_executions_total[5m])"
          }
        ],
        "gridPos": { "x": 0, "y": 8, "w": 24, "h": 8 }
      }
    ],
    "refresh": "5m",
    "time": { "from": "now-24h", "to": "now" }
  }
}
```

---

## レポーティング戦略

### 日次レポート自動生成

```typescript
import { createObjectCsvWriter } from 'csv-writer';
import nodemailer from 'nodemailer';

interface DailyQAReport {
  date: string;
  newBugs: number;
  fixedBugs: number;
  openBugs: number;
  testsPassed: number;
  testsFailed: number;
  testCoverage: number;
  crashRate: number;
}

class DailyReportGenerator {
  async generate(date: Date): Promise<DailyQAReport> {
    const [bugs, tests, coverage, crashes] = await Promise.all([
      this.fetchBugMetrics(date),
      this.fetchTestMetrics(date),
      this.fetchCoverageMetrics(date),
      this.fetchCrashMetrics(date),
    ]);

    return {
      date: date.toISOString().split('T')[0],
      newBugs: bugs.new,
      fixedBugs: bugs.fixed,
      openBugs: bugs.open,
      testsPassed: tests.passed,
      testsFailed: tests.failed,
      testCoverage: coverage.overall,
      crashRate: crashes.rate,
    };
  }

  async sendEmail(report: DailyQAReport) {
    const html = this.generateHTML(report);

    const transporter = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port: parseInt(process.env.SMTP_PORT || '587'),
      secure: false,
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS,
      },
    });

    await transporter.sendMail({
      from: 'qa-bot@example.com',
      to: 'qa-team@example.com',
      subject: `QA Daily Report - ${report.date}`,
      html,
    });
  }

  private generateHTML(report: DailyQAReport): string {
    const status = this.getStatus(report);

    return `
<!DOCTYPE html>
<html>
<head>
  <style>
    body { font-family: Arial, sans-serif; }
    .metric { margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }
    .good { border-color: #4CAF50; }
    .warning { border-color: #FF9800; }
    .bad { border-color: #F44336; }
  </style>
</head>
<body>
  <h1>QA Daily Report - ${report.date}</h1>

  <div class="metric ${status.bugs}">
    <h3>🐛 バグ状況</h3>
    <ul>
      <li>新規バグ: ${report.newBugs}件</li>
      <li>修正済み: ${report.fixedBugs}件</li>
      <li>未解決: ${report.openBugs}件</li>
    </ul>
  </div>

  <div class="metric ${status.tests}">
    <h3>✅ テスト結果</h3>
    <ul>
      <li>成功: ${report.testsPassed}件</li>
      <li>失敗: ${report.testsFailed}件</li>
      <li>成功率: ${((report.testsPassed / (report.testsPassed + report.testsFailed)) * 100).toFixed(1)}%</li>
    </ul>
  </div>

  <div class="metric ${status.coverage}">
    <h3>📊 カバレッジ</h3>
    <p>${report.testCoverage.toFixed(1)}%</p>
  </div>

  <div class="metric ${status.crashes}">
    <h3>💥 クラッシュ率</h3>
    <p>${report.crashRate.toFixed(3)}%</p>
  </div>
</body>
</html>
    `;
  }

  private getStatus(report: DailyQAReport) {
    return {
      bugs: report.openBugs > 50 ? 'bad' : report.openBugs > 30 ? 'warning' : 'good',
      tests: report.testsFailed > 5 ? 'bad' : report.testsFailed > 2 ? 'warning' : 'good',
      coverage: report.testCoverage < 70 ? 'bad' : report.testCoverage < 80 ? 'warning' : 'good',
      crashes: report.crashRate > 0.2 ? 'bad' : report.crashRate > 0.1 ? 'warning' : 'good',
    };
  }

  private async fetchBugMetrics(date: Date) {
    // JIRA APIから取得
    return { new: 5, fixed: 8, open: 35 };
  }

  private async fetchTestMetrics(date: Date) {
    // CI/CDから取得
    return { passed: 450, failed: 2 };
  }

  private async fetchCoverageMetrics(date: Date) {
    // Codecovから取得
    return { overall: 82.5 };
  }

  private async fetchCrashMetrics(date: Date) {
    // Firebase Crashlyticsから取得
    return { rate: 0.08 };
  }
}

// スケジューラー
import cron from 'node-cron';

// 毎日午前9時に実行
cron.schedule('0 9 * * *', async () => {
  const generator = new DailyReportGenerator();
  const report = await generator.generate(new Date());
  await generator.sendEmail(report);
  console.log(`Daily report sent: ${report.date}`);
});
```

### 週次トレンドレポート

```swift
struct WeeklyTrendReport {
    let weekNumber: Int
    let startDate: Date
    let endDate: Date

    struct Trend {
        let current: Double
        let previous: Double
        let change: Double
        let changePercentage: Double

        var direction: String {
            if change > 0 { return "📈 増加" }
            if change < 0 { return "📉 減少" }
            return "➡️ 変化なし"
        }

        var isImproving: Bool {
            // メトリクスによって改善方向は異なる
            // バグ数・クラッシュ率は減少が改善
            // カバレッジは増加が改善
            return change < 0
        }
    }

    let bugTrend: Trend
    let coverageTrend: Trend
    let crashRateTrend: Trend
    let testPassRateTrend: Trend

    func generateMarkdown() -> String {
        """
        # 週次QAトレンドレポート - Week \(weekNumber)

        **期間**: \(formatDate(startDate)) 〜 \(formatDate(endDate))

        ## 📊 主要メトリクストレンド

        ### バグ数
        - 今週: \(Int(bugTrend.current))件
        - 先週: \(Int(bugTrend.previous))件
        - 変化: \(bugTrend.direction) (\(formatPercentage(bugTrend.changePercentage)))
        - 評価: \(bugTrend.isImproving ? "✅ 改善" : "⚠️ 悪化")

        ### テストカバレッジ
        - 今週: \(formatPercentage(coverageTrend.current))
        - 先週: \(formatPercentage(coverageTrend.previous))
        - 変化: \(coverageTrend.direction) (\(formatPercentage(coverageTrend.changePercentage)))
        - 評価: \(!coverageTrend.isImproving ? "✅ 改善" : "⚠️ 悪化")

        ### クラッシュ率
        - 今週: \(String(format: "%.3f", crashRateTrend.current))%
        - 先週: \(String(format: "%.3f", crashRateTrend.previous))%
        - 変化: \(crashRateTrend.direction) (\(formatPercentage(crashRateTrend.changePercentage)))
        - 評価: \(crashRateTrend.isImproving ? "✅ 改善" : "⚠️ 悪化")

        ### テスト成功率
        - 今週: \(formatPercentage(testPassRateTrend.current))
        - 先週: \(formatPercentage(testPassRateTrend.previous))
        - 変化: \(testPassRateTrend.direction) (\(formatPercentage(testPassRateTrend.changePercentage)))
        - 評価: \(!testPassRateTrend.isImproving ? "✅ 改善" : "⚠️ 悪化")

        ## 🎯 改善アクション

        \(generateActionItems())

        ## 📈 次週の目標

        \(generateNextWeekGoals())
        """
    }

    private func generateActionItems() -> String {
        var items: [String] = []

        if !bugTrend.isImproving && bugTrend.current > 30 {
            items.append("- [ ] バグ数が増加傾向。トリアージ会議を実施")
        }

        if coverageTrend.current < 80 {
            items.append("- [ ] テストカバレッジが目標未達。カバレッジ改善週間を設定")
        }

        if !crashRateTrend.isImproving {
            items.append("- [ ] クラッシュ率が悪化。Top 5クラッシュの優先修正")
        }

        return items.isEmpty ? "特になし（すべて順調）" : items.joined(separator: "\n")
    }

    private func generateNextWeekGoals() -> String {
        """
        - バグ数: \(Int(bugTrend.current * 0.9))件以下（10%削減）
        - カバレッジ: \(formatPercentage(min(coverageTrend.current + 2, 100)))以上
        - クラッシュ率: \(String(format: "%.3f", crashRateTrend.current * 0.9))%以下
        - テスト成功率: 95%以上維持
        """
    }

    private func formatDate(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }

    private func formatPercentage(_ value: Double) -> String {
        String(format: "%.1f%%", value)
    }
}
```

---

## 実践例とケーススタディ

### ケーススタディ1: E-commerceアプリの品質改善

**背景:**
- 毎月平均20件のバグが本番で発見される
- クラッシュ率 0.3%
- テストカバレッジ 45%
- ユーザー評価 3.8/5.0

**実施した施策:**

```markdown
## 改善プログラム（3ヶ月）

### Month 1: 測定基盤構築
✅ Grafana + Prometheusダッシュボード構築
✅ Firebase Crashlytics統合
✅ CI/CDにテストカバレッジレポート追加
✅ 日次QAレポート自動送信開始

### Month 2: プロセス改善
✅ テストピラミッド導入（70/20/10）
✅ PRレビュー時のカバレッジチェック必須化
✅ バグトリアージ会議を週2回実施
✅ 探索的テストセッションを週1回実施

### Month 3: 自動化推進
✅ E2Eテスト（Playwright）50シナリオ追加
✅ Visual Regressionテスト導入
✅ リリース前チェックリスト自動化
✅ パフォーマンステスト自動化
```

**成果:**

| メトリクス | 導入前 | 3ヶ月後 | 改善率 |
|-----------|--------|---------|--------|
| 本番バグ数 | 20件/月 | 3件/月 | **-85%** |
| クラッシュ率 | 0.3% | 0.07% | **-77%** |
| テストカバレッジ | 45% | 83% | **+84%** |
| テスト自動化率 | 20% | 75% | **+275%** |
| ユーザー評価 | 3.8 | 4.6 | **+21%** |
| リリースサイクル | 4週間 | 2週間 | **-50%** |

**ROI計算:**

```swift
struct QAImprovementROI {
    // コスト
    let toolCosts = 500_000      // ツール導入費（年間）
    let trainingCosts = 300_000   // 研修費
    let timeCosts = 2_000_000     // 改善活動工数（3ヶ月）

    var totalCosts: Int {
        toolCosts + trainingCosts + timeCosts
    }

    // 効果
    let reducedBugFixingCost = 5_100_000  // バグ修正工数削減（-17件/月 × 30万円/件）
    let reducedSupportCost = 1_800_000    // サポート対応削減
    let increasedRevenue = 3_000_000      // ユーザー評価向上による売上増

    var totalBenefits: Int {
        reducedBugFixingCost + reducedSupportCost + increasedRevenue
    }

    var roi: Double {
        (Double(totalBenefits - totalCosts) / Double(totalCosts)) * 100
    }

    func report() -> String {
        """
        QA改善プログラム ROI分析
        ========================

        投資額:
        - ツール導入: ¥\(toolCosts.formatted())
        - 研修費: ¥\(trainingCosts.formatted())
        - 改善活動工数: ¥\(timeCosts.formatted())
        - 合計: ¥\(totalCosts.formatted())

        効果（年間換算）:
        - バグ修正コスト削減: ¥\(reducedBugFixingCost.formatted())
        - サポートコスト削減: ¥\(reducedSupportCost.formatted())
        - 売上増加: ¥\(increasedRevenue.formatted())
        - 合計: ¥\(totalBenefits.formatted())

        ROI: \(String(format: "%.0f", roi))%
        回収期間: \(String(format: "%.1f", Double(totalCosts) / Double(totalBenefits) * 12))ヶ月

        結論: ✅ 投資対効果が高く、継続推奨
        """
    }
}
```

### ケーススタディ2: SaaSプラットフォームのQA体制構築

**課題:**
- QAエンジニア不在（開発者がテスト兼務）
- リリース毎に重大バグが発生
- 顧客からのクレームが増加

**ソリューション:**

```markdown
## QA体制構築ロードマップ

### Phase 1: 緊急対応（Week 1-2）
- [x] リリース判定基準の明文化
- [x] バグ優先度定義の統一
- [x] クリティカルバグのホットフィックスプロセス確立

### Phase 2: 基盤整備（Week 3-6）
- [x] QAエンジニア採用（1名）
- [x] テスト計画テンプレート作成
- [x] バグトラッキングシステム導入（Jira）
- [x] CI/CDパイプライン構築

### Phase 3: プロセス標準化（Week 7-12）
- [x] QA観点チェックリスト作成
- [x] テストケース管理開始（TestRail）
- [x] 探索的テストガイドライン策定
- [x] リグレッションテスト自動化（50%）

### Phase 4: 継続的改善（Month 4-6）
- [x] 品質メトリクスダッシュボード構築
- [x] 月次品質レビュー会議開始
- [x] QA研修プログラム実施
- [x] テスト自動化率 70% 達成
```

**成果:**

```
Before（導入前）:
- リリース失敗率: 40%
- 本番重大バグ: 8件/リリース
- 顧客クレーム: 15件/月
- ホットフィックス: 3回/月

After（6ヶ月後）:
- リリース失敗率: 5% (-88%)
- 本番重大バグ: 0.5件/リリース (-94%)
- 顧客クレーム: 2件/月 (-87%)
- ホットフィックス: 0.3回/月 (-90%)

顧客満足度: 65% → 92% (+42%)
```

---

## トラブルシューティング

### よくある問題と解決策

**1. メトリクスが改善しない**

```markdown
## 問題: メトリクスが改善しない

### 症状
- 3ヶ月経過してもバグ数が減らない
- テストカバレッジが向上しない
- クラッシュ率が横ばい

### 原因
❌ メトリクスを測定するだけで改善アクションがない
❌ 目標が現実的でない
❌ チームの理解・協力が得られていない

### 解決策
✅ 測定 → 分析 → アクション → 振り返りのサイクルを回す
✅ SMART原則に基づく現実的な目標設定
✅ 週次レビューミーティングで進捗確認
✅ 成功事例の共有とモチベーション向上
✅ 改善活動への時間を公式に確保（20%ルール）
```

**2. ダッシュボードが誰も見ない**

```markdown
## 問題: ダッシュボードが誰も見ない

### 症状
- せっかく作ったダッシュボードのアクセスがほぼゼロ
- メトリクスがアクションに繋がらない

### 原因
❌ ダッシュボードが複雑すぎる
❌ 重要な情報が埋もれている
❌ アクセスが不便
❌ 見る習慣がない

### 解決策
✅ シンプルなデザイン（1画面に主要メトリクスのみ）
✅ Red/Yellow/Green の視覚的なステータス表示
✅ SlackやTeamsに自動投稿
✅ 朝会での確認を習慣化
✅ アラート機能で異常時のみ通知
✅ モバイル対応
```

**3. データの信頼性が低い**

```markdown
## 問題: データの信頼性が低い

### 症状
- メトリクスの数値が実態と乖離
- 手動集計とダッシュボードで値が違う
- データ欠損が多い

### 原因
❌ データソースの統合不足
❌ 手動入力に依存
❌ バグステータスの更新漏れ
❌ テスト実行結果の記録漏れ

### 解決策
✅ 自動化可能なものは全て自動化
✅ Single Source of Truth を確立
✅ データ検証スクリプトの導入
✅ 定期的なデータ監査
✅ チーム全体でのデータ入力ルール統一
```

**4. メトリクスに振り回される**

```markdown
## 問題: メトリクスに振り回される

### 症状
- カバレッジを上げるだけの意味のないテストが増える
- 数値改善のために本質的でない作業に時間を使う
- チームのモチベーション低下

### 原因
❌ メトリクスが目的化している
❌ 品質の本質を見失っている
❌ 短期的な数値改善を優先

### 解決策
✅ メトリクスはあくまで手段と認識
✅ 質的な評価も併用（コードレビュー、ユーザーフィードバック）
✅ ビジネスゴールとの整合性を常に確認
✅ チームでメトリクスの意義を定期的に議論
✅ 「良い品質」の定義を明確化
```

---

## まとめ

### QAメトリクス成功の鍵

```markdown
## 成功の5原則

1. **測定可能にする**
   - すべてを数値化
   - 自動収集を優先
   - リアルタイム性を確保

2. **可視化する**
   - シンプルなダッシュボード
   - 誰でも理解できる表現
   - トレンドを把握しやすく

3. **アクションに繋げる**
   - メトリクスから改善アクション導出
   - 振り返りサイクルの確立
   - 継続的な改善文化

4. **チーム全体で共有**
   - 透明性の確保
   - 定期的なレビュー
   - 成功体験の共有

5. **本質を見失わない**
   - ビジネス価値との整合
   - ユーザー視点を忘れない
   - 数値だけに依存しない
```

### 次のステップ

```markdown
## 推奨アクション

### すぐできること（今日から）
- [ ] 主要メトリクスを3つ選定
- [ ] 現状値をExcelやスプレッドシートで記録開始
- [ ] 週次レビューミーティングを設定

### 1週間以内
- [ ] ダッシュボードツール選定（Grafana/Google Data Studio/Looker）
- [ ] データ収集の自動化方法を検討
- [ ] チームにメトリクス運用を説明

### 1ヶ月以内
- [ ] ダッシュボード構築
- [ ] 日次/週次レポート自動化
- [ ] 目標値設定と達成計画策定

### 3ヶ月以内
- [ ] メトリクスに基づく改善活動実施
- [ ] ROI測定
- [ ] チーム全体への展開
```

---

**関連ガイド:**
- [テスト計画・実行ガイド](./test-planning-execution.md)
- [バグ管理完全ガイド](./bug-management-complete.md)
- [リリース管理とクライテリア](./release-management-criteria.md)
- [QA自動化とツール統合](./qa-automation-tools.md)
