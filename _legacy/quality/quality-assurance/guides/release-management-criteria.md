# リリース管理とクライテリア - 完全ガイド

リリース判定基準の策定からリリース後のモニタリングまで、安全で確実なリリースプロセスを完全解説します。

## 目次

1. [リリースクライテリアの基礎](#リリースクライテリアの基礎)
2. [Entry/Exit Criteria](#entryexit-criteria)
3. [リリース判定プロセス](#リリース判定プロセス)
4. [リリースチェックリスト](#リリースチェックリスト)
5. [段階的ロールアウト](#段階的ロールアウト)
6. [ロールバック戦略](#ロールバック戦略)
7. [リリース後モニタリング](#リリース後モニタリング)
8. [実践例とケーススタディ](#実践例とケーススタディ)
9. [トラブルシューティング](#トラブルシューティング)

---

## リリースクライテリアの基礎

### リリースクライテリアとは

**定義:**
ソフトウェアを本番環境にリリースするための明確な基準と条件

**目的:**
- 品質の客観的な評価
- リリース可否の判断基準の統一
- ステークホルダーへの説明責任
- リスクの最小化

### クライテリアの種類

```swift
struct ReleaseCriteria {
    // 1. 機能完了基準
    struct FunctionalCompleteness {
        let plannedFeatures: Int
        let completedFeatures: Int
        let criticalFeaturesComplete: Bool

        var completionRate: Double {
            guard plannedFeatures > 0 else { return 0 }
            return (Double(completedFeatures) / Double(plannedFeatures)) * 100
        }

        var meetsRequirement: Bool {
            // すべてのCritical機能 + 90%以上の計画機能
            return criticalFeaturesComplete && completionRate >= 90
        }
    }

    // 2. 品質基準
    struct QualityStandards {
        let criticalBugs: Int
        let majorBugs: Int
        let minorBugs: Int
        let testCoverage: Double
        let testPassRate: Double

        var meetsRequirement: Bool {
            return criticalBugs == 0 &&
                   majorBugs <= 5 &&
                   testCoverage >= 80 &&
                   testPassRate >= 95
        }

        var blockingIssues: [String] {
            var issues: [String] = []

            if criticalBugs > 0 {
                issues.append("Critical bugs: \(criticalBugs)件")
            }
            if majorBugs > 5 {
                issues.append("Major bugs: \(majorBugs)件 (上限5件)")
            }
            if testCoverage < 80 {
                issues.append("Test coverage: \(String(format: "%.1f", testCoverage))% (目標80%)")
            }
            if testPassRate < 95 {
                issues.append("Test pass rate: \(String(format: "%.1f", testPassRate))% (目標95%)")
            }

            return issues
        }
    }

    // 3. パフォーマンス基準
    struct PerformanceStandards {
        let crashRate: Double
        let anrRate: Double
        let appLaunchTime: TimeInterval
        let apiP95ResponseTime: TimeInterval

        var meetsRequirement: Bool {
            return crashRate <= 0.1 &&
                   anrRate <= 0.05 &&
                   appLaunchTime <= 2.0 &&
                   apiP95ResponseTime <= 1.0
        }
    }

    // 4. セキュリティ基準
    struct SecurityStandards {
        let vulnerabilityScanPassed: Bool
        let dependenciesUpToDate: Bool
        let securityReviewCompleted: Bool
        let criticalVulnerabilities: Int

        var meetsRequirement: Bool {
            return vulnerabilityScanPassed &&
                   dependenciesUpToDate &&
                   securityReviewCompleted &&
                   criticalVulnerabilities == 0
        }
    }

    // 5. ドキュメント基準
    struct DocumentationStandards {
        let releaseNotesReady: Bool
        let apiDocumentationUpdated: Bool
        let userGuideUpdated: Bool
        let changelogUpdated: Bool

        var meetsRequirement: Bool {
            return releaseNotesReady &&
                   apiDocumentationUpdated &&
                   userGuideUpdated &&
                   changelogUpdated
        }
    }

    let functional: FunctionalCompleteness
    let quality: QualityStandards
    let performance: PerformanceStandards
    let security: SecurityStandards
    let documentation: DocumentationStandards

    // 総合判定
    func canRelease() -> ReleaseDecision {
        var blockingIssues: [String] = []
        var warnings: [String] = []

        // 必須条件チェック
        if !functional.meetsRequirement {
            blockingIssues.append("機能完成度が基準未達: \(String(format: "%.1f", functional.completionRate))%")
        }

        if !quality.meetsRequirement {
            blockingIssues.append(contentsOf: quality.blockingIssues)
        }

        if !security.meetsRequirement {
            blockingIssues.append("セキュリティ基準未達")
        }

        // 警告条件チェック
        if !performance.meetsRequirement {
            warnings.append("パフォーマンス基準未達（リリース可能だが要改善）")
        }

        if !documentation.meetsRequirement {
            warnings.append("ドキュメント未完成（リリース後対応可）")
        }

        if blockingIssues.isEmpty {
            return .approved(warnings: warnings)
        } else {
            return .rejected(reasons: blockingIssues)
        }
    }

    enum ReleaseDecision {
        case approved(warnings: [String])
        case rejected(reasons: [String])

        var canRelease: Bool {
            if case .approved = self {
                return true
            }
            return false
        }
    }
}
```

---

## Entry/Exit Criteria

### Entry Criteria（開始基準）

**テストフェーズ開始前の条件:**

```markdown
## テストフェーズ Entry Criteria

### 必須条件（Must Have）
- [ ] すべての計画機能の開発完了
- [ ] コードレビュー完了率 100%
- [ ] ユニットテストPass率 95%以上
- [ ] ビルドが成功している
- [ ] テスト環境が利用可能
- [ ] テストデータが準備済み

### 推奨条件（Should Have）
- [ ] 統合テストPass率 90%以上
- [ ] コードカバレッジ 80%以上
- [ ] 既知のCriticalバグが0件
- [ ] テストケースレビュー完了

### 任意条件（Nice to Have）
- [ ] パフォーマンステスト完了
- [ ] セキュリティスキャン実施済み
```

**実装例:**

```typescript
interface EntryCriteria {
  developmentComplete: boolean;
  codeReviewComplete: boolean;
  unitTestPassRate: number;
  buildSuccess: boolean;
  testEnvironmentReady: boolean;
  testDataReady: boolean;
}

class EntryCriteriaChecker {
  check(criteria: EntryCriteria): CheckResult {
    const failures: string[] = [];

    if (!criteria.developmentComplete) {
      failures.push('開発が完了していません');
    }
    if (!criteria.codeReviewComplete) {
      failures.push('コードレビューが完了していません');
    }
    if (criteria.unitTestPassRate < 95) {
      failures.push(`ユニットテストPass率が不足: ${criteria.unitTestPassRate}% (目標95%)`);
    }
    if (!criteria.buildSuccess) {
      failures.push('ビルドが失敗しています');
    }
    if (!criteria.testEnvironmentReady) {
      failures.push('テスト環境が準備できていません');
    }
    if (!criteria.testDataReady) {
      failures.push('テストデータが準備できていません');
    }

    return {
      passed: failures.length === 0,
      failures,
      message: failures.length === 0
        ? '✅ Entry Criteriaを満たしています。テストフェーズを開始できます。'
        : `❌ Entry Criteriaを満たしていません:\n${failures.map(f => `  • ${f}`).join('\n')}`,
    };
  }
}

interface CheckResult {
  passed: boolean;
  failures: string[];
  message: string;
}

// 使用例
const checker = new EntryCriteriaChecker();
const result = checker.check({
  developmentComplete: true,
  codeReviewComplete: true,
  unitTestPassRate: 96,
  buildSuccess: true,
  testEnvironmentReady: true,
  testDataReady: true,
});

console.log(result.message);
```

### Exit Criteria（終了基準）

**テストフェーズ完了の条件:**

```markdown
## テストフェーズ Exit Criteria

### 必須条件（Must Have）
- [ ] 計画されたテストケースの実行率 95%以上
- [ ] テストPass率 95%以上
- [ ] Criticalバグ 0件
- [ ] Majorバグ 5件以下
- [ ] リグレッションテスト完了
- [ ] バグ修正の再テスト完了

### 推奨条件（Should Have）
- [ ] 探索的テスト実施済み
- [ ] パフォーマンステスト完了
- [ ] セキュリティテスト完了
- [ ] ユーザビリティテスト完了

### 任意条件（Nice to Have）
- [ ] Beta版フィードバック収集
- [ ] 負荷テスト実施
- [ ] アクセシビリティテスト完了
```

**実装例:**

```swift
struct ExitCriteria {
    let testExecutionRate: Double
    let testPassRate: Double
    let criticalBugs: Int
    let majorBugs: Int
    let regressionTestComplete: Bool
    let bugRetestComplete: Bool

    // オプション条件
    let exploratoryTestComplete: Bool
    let performanceTestComplete: Bool
    let securityTestComplete: Bool

    func check() -> CheckResult {
        var failures: [String] = []
        var warnings: [String] = []

        // 必須条件チェック
        if testExecutionRate < 95 {
            failures.append("テスト実行率不足: \(String(format: "%.1f", testExecutionRate))% (目標95%)")
        }

        if testPassRate < 95 {
            failures.append("テストPass率不足: \(String(format: "%.1f", testPassRate))% (目標95%)")
        }

        if criticalBugs > 0 {
            failures.append("Criticalバグが\(criticalBugs)件残っています")
        }

        if majorBugs > 5 {
            failures.append("Majorバグが\(majorBugs)件（上限5件）")
        }

        if !regressionTestComplete {
            failures.append("リグレッションテストが未完了")
        }

        if !bugRetestComplete {
            failures.append("バグ修正の再テストが未完了")
        }

        // 推奨条件チェック（警告）
        if !performanceTestComplete {
            warnings.append("パフォーマンステストが未完了")
        }

        if !securityTestComplete {
            warnings.append("セキュリティテストが未完了")
        }

        return CheckResult(
            passed: failures.isEmpty,
            failures: failures,
            warnings: warnings
        )
    }

    struct CheckResult {
        let passed: Bool
        let failures: [String]
        let warnings: [String]

        var message: String {
            var msg = ""

            if passed {
                msg += "✅ Exit Criteriaを満たしています。リリース判定に進めます。\n"

                if !warnings.isEmpty {
                    msg += "\n⚠️ 警告:\n"
                    msg += warnings.map { "  • \($0)" }.joined(separator: "\n")
                }
            } else {
                msg += "❌ Exit Criteriaを満たしていません:\n"
                msg += failures.map { "  • \($0)" }.joined(separator: "\n")

                if !warnings.isEmpty {
                    msg += "\n\n⚠️ 警告:\n"
                    msg += warnings.map { "  • \($0)" }.joined(separator: "\n")
                }
            }

            return msg
        }
    }
}
```

---

## リリース判定プロセス

### リリース判定会議

**会議構成:**

```markdown
## リリース判定会議（Go/No-Go Meeting）

### 参加者
- プロダクトマネージャー（意思決定者）
- QAリード
- 開発リード
- DevOpsエンジニア
- UXデザイナー（必要に応じて）

### アジェンダ
1. **品質メトリクスレビュー**（10分）
   - バグ統計
   - テスト結果
   - カバレッジ
   - パフォーマンス指標

2. **リスク評価**（10分）
   - 特定されたリスク
   - 既知の問題
   - 影響範囲分析

3. **クライテリアチェック**（10分）
   - Entry/Exit Criteria確認
   - リリース基準の達成状況

4. **Go/No-Go判定**（5分）
   - 最終判断
   - 条件付き承認の場合の条件明示

5. **次のステップ**（5分）
   - リリース日時確認
   - ロールアウト計画
   - モニタリング体制
```

**判定フレームワーク:**

```typescript
interface GoNoGoDecision {
  // 品質メトリクス
  metrics: {
    bugCount: { critical: number; major: number; minor: number };
    testResults: { total: number; passed: number; failed: number };
    coverage: number;
    performance: { crashRate: number; responseTime: number };
  };

  // リスク評価
  risks: Risk[];

  // クライテリア達成状況
  criteriaStatus: {
    functional: boolean;
    quality: boolean;
    performance: boolean;
    security: boolean;
    documentation: boolean;
  };

  // ステークホルダー承認
  approvals: {
    productManager: boolean;
    qaLead: boolean;
    devLead: boolean;
    securityTeam: boolean;
  };
}

interface Risk {
  id: string;
  description: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  mitigation: string;
  accepted: boolean;
}

class GoNoGoDecisionMaker {
  evaluate(decision: GoNoGoDecision): DecisionResult {
    const blockers: string[] = [];
    const warnings: string[] = [];

    // Critical blockers
    if (decision.metrics.bugCount.critical > 0) {
      blockers.push(`Critical bugs: ${decision.metrics.bugCount.critical}件`);
    }

    if (!decision.criteriaStatus.quality) {
      blockers.push('品質基準未達');
    }

    if (!decision.criteriaStatus.security) {
      blockers.push('セキュリティ基準未達');
    }

    // Critical risks
    const criticalRisks = decision.risks.filter(
      r => r.severity === 'critical' && !r.accepted
    );
    if (criticalRisks.length > 0) {
      blockers.push(`未承認のCriticalリスク: ${criticalRisks.length}件`);
    }

    // Approvals
    if (!decision.approvals.productManager) {
      blockers.push('PM承認待ち');
    }
    if (!decision.approvals.qaLead) {
      blockers.push('QAリード承認待ち');
    }

    // Warnings
    if (decision.metrics.bugCount.major > 5) {
      warnings.push(`Major bugs: ${decision.metrics.bugCount.major}件（推奨5件以下）`);
    }

    if (decision.metrics.coverage < 80) {
      warnings.push(`カバレッジ: ${decision.metrics.coverage}%（推奨80%以上）`);
    }

    // Final decision
    if (blockers.length === 0) {
      return {
        decision: 'GO',
        confidence: this.calculateConfidence(decision, warnings),
        blockers: [],
        warnings,
        recommendation: this.generateRecommendation(decision, warnings),
      };
    } else {
      return {
        decision: 'NO-GO',
        confidence: 0,
        blockers,
        warnings,
        recommendation: 'リリース基準を満たしていません。ブロッカーの解消後に再評価してください。',
      };
    }
  }

  private calculateConfidence(
    decision: GoNoGoDecision,
    warnings: string[]
  ): number {
    let score = 100;

    // Warnings reduce confidence
    score -= warnings.length * 5;

    // High-risk items reduce confidence
    const highRisks = decision.risks.filter(r => r.severity === 'high');
    score -= highRisks.length * 10;

    // Major bugs reduce confidence
    score -= decision.metrics.bugCount.major * 2;

    return Math.max(0, Math.min(100, score));
  }

  private generateRecommendation(
    decision: GoNoGoDecision,
    warnings: string[]
  ): string {
    if (warnings.length === 0) {
      return '✅ リリース推奨。品質基準を満たしています。';
    } else if (warnings.length <= 2) {
      return '✅ リリース可能。ただし以下の点に注意してモニタリングしてください。';
    } else {
      return '⚠️ リリース可能ですが、複数の警告があります。慎重にロールアウトしてください。';
    }
  }
}

interface DecisionResult {
  decision: 'GO' | 'NO-GO';
  confidence: number; // 0-100
  blockers: string[];
  warnings: string[];
  recommendation: string;
}
```

---

## リリースチェックリスト

### 包括的チェックリスト

```markdown
# リリース前チェックリスト

## 開発完了確認
- [ ] すべての計画機能が実装済み
- [ ] すべてのPRがマージ済み
- [ ] コードレビュー完了
- [ ] リファクタリング・技術的負債対応完了
- [ ] 不要なコメント・デバッグコード削除

## テスト完了確認
- [ ] ユニットテスト実行・Pass
- [ ] 統合テスト実行・Pass
- [ ] E2Eテスト実行・Pass
- [ ] リグレッションテスト完了
- [ ] 探索的テスト実施
- [ ] パフォーマンステスト実施
- [ ] セキュリティテスト実施
- [ ] クロスブラウザテスト完了（該当する場合）
- [ ] デバイスマトリクステスト完了

## バグ管理
- [ ] Criticalバグ 0件
- [ ] Majorバグ 5件以下
- [ ] すべてのバグが適切にトリアージ済み
- [ ] リリース後対応バグをバックログに登録
- [ ] 既知の問題をドキュメント化

## コード品質
- [ ] コードカバレッジ 80%以上
- [ ] Lintエラー 0件
- [ ] 型チェックエラー 0件
- [ ] セキュリティ脆弱性スキャン実施
- [ ] 依存関係の脆弱性チェック完了
- [ ] 未使用の依存関係削除

## ドキュメント
- [ ] README更新
- [ ] CHANGELOG更新
- [ ] リリースノート作成
- [ ] APIドキュメント更新（該当する場合）
- [ ] ユーザーガイド更新
- [ ] 移行ガイド作成（破壊的変更がある場合）

## インフラ・環境
- [ ] 本番環境の準備完了
- [ ] データベースマイグレーション準備
- [ ] 環境変数設定確認
- [ ] SSL証明書有効期限確認
- [ ] ドメイン・DNS設定確認
- [ ] CDN・キャッシュ設定確認

## ビルド・デプロイ
- [ ] 本番ビルド成功
- [ ] バージョン番号更新
- [ ] タグ作成（Git tag）
- [ ] アーティファクト署名（該当する場合）
- [ ] デプロイスクリプト検証

## モニタリング・アラート
- [ ] モニタリングダッシュボード準備
- [ ] アラート設定確認
- [ ] ログ収集設定確認
- [ ] エラートラッキング有効化（Sentry等）
- [ ] パフォーマンスモニタリング有効化

## ロールバック準備
- [ ] ロールバック手順確認
- [ ] 前バージョンのバックアップ確認
- [ ] データベースロールバック手順確認
- [ ] ロールバック担当者確定

## コミュニケーション
- [ ] ステークホルダーへリリース通知
- [ ] カスタマーサポートチームへ情報共有
- [ ] マーケティングチームと調整（必要に応じて）
- [ ] ダウンタイムの告知（必要な場合）

## 最終確認
- [ ] リリース判定会議実施
- [ ] Go/No-Go判定完了
- [ ] すべての承認取得
- [ ] リリース日時最終確認
- [ ] 緊急連絡体制確認
```

**チェックリスト自動化:**

```typescript
interface ChecklistItem {
  id: string;
  category: string;
  description: string;
  required: boolean; // 必須項目か
  automated: boolean; // 自動チェック可能か
  status: 'pending' | 'passed' | 'failed' | 'skipped';
  checkedBy?: string;
  checkedAt?: Date;
  notes?: string;
}

class ReleaseChecklistManager {
  private items: ChecklistItem[] = [];

  constructor() {
    this.initializeChecklist();
  }

  private initializeChecklist(): void {
    this.items = [
      {
        id: 'dev-001',
        category: '開発完了確認',
        description: 'すべてのPRがマージ済み',
        required: true,
        automated: true,
        status: 'pending',
      },
      {
        id: 'test-001',
        category: 'テスト完了確認',
        description: 'ユニットテスト実行・Pass',
        required: true,
        automated: true,
        status: 'pending',
      },
      {
        id: 'bug-001',
        category: 'バグ管理',
        description: 'Criticalバグ 0件',
        required: true,
        automated: true,
        status: 'pending',
      },
      // ... more items
    ];
  }

  async runAutomatedChecks(): Promise<void> {
    for (const item of this.items.filter(i => i.automated)) {
      item.status = await this.checkItem(item);
      item.checkedAt = new Date();
      item.checkedBy = 'automation';
    }
  }

  private async checkItem(item: ChecklistItem): Promise<'passed' | 'failed'> {
    // 各項目の自動チェックロジック
    switch (item.id) {
      case 'dev-001':
        return await this.checkAllPRsMerged() ? 'passed' : 'failed';
      case 'test-001':
        return await this.checkUnitTests() ? 'passed' : 'failed';
      case 'bug-001':
        return await this.checkCriticalBugs() ? 'passed' : 'failed';
      default:
        return 'passed';
    }
  }

  private async checkAllPRsMerged(): Promise<boolean> {
    // GitHub APIで未マージPRをチェック
    return true; // 簡略化
  }

  private async checkUnitTests(): Promise<boolean> {
    // CI/CDから最新のテスト結果を取得
    return true; // 簡略化
  }

  private async checkCriticalBugs(): Promise<boolean> {
    // Jira APIでCriticalバグをチェック
    return true; // 簡略化
  }

  getProgress(): ChecklistProgress {
    const total = this.items.length;
    const required = this.items.filter(i => i.required).length;
    const completed = this.items.filter(i => i.status === 'passed').length;
    const failed = this.items.filter(i => i.status === 'failed').length;
    const requiredCompleted = this.items.filter(
      i => i.required && i.status === 'passed'
    ).length;

    return {
      total,
      required,
      completed,
      failed,
      requiredCompleted,
      completionRate: (completed / total) * 100,
      requiredCompletionRate: (requiredCompleted / required) * 100,
      canRelease: requiredCompleted === required && failed === 0,
    };
  }

  generateReport(): string {
    const progress = this.getProgress();

    const groupedItems = this.items.reduce((acc, item) => {
      if (!acc[item.category]) {
        acc[item.category] = [];
      }
      acc[item.category].push(item);
      return acc;
    }, {} as Record<string, ChecklistItem[]>);

    let report = `
# リリースチェックリスト レポート

## 進捗状況
- 総項目数: ${progress.total}
- 完了: ${progress.completed} (${progress.completionRate.toFixed(1)}%)
- 失敗: ${progress.failed}
- 必須項目完了率: ${progress.requiredCompletionRate.toFixed(1)}%
- リリース可否: ${progress.canRelease ? '✅ 可能' : '❌ 不可'}

---

`;

    for (const [category, items] of Object.entries(groupedItems)) {
      report += `## ${category}\n\n`;

      for (const item of items) {
        const icon = this.getStatusIcon(item.status);
        const required = item.required ? '[必須]' : '[任意]';
        report += `${icon} ${required} ${item.description}\n`;

        if (item.notes) {
          report += `   備考: ${item.notes}\n`;
        }
      }

      report += '\n';
    }

    return report;
  }

  private getStatusIcon(status: ChecklistItem['status']): string {
    switch (status) {
      case 'passed':
        return '✅';
      case 'failed':
        return '❌';
      case 'skipped':
        return '⏭️';
      default:
        return '⏸️';
    }
  }
}

interface ChecklistProgress {
  total: number;
  required: number;
  completed: number;
  failed: number;
  requiredCompleted: number;
  completionRate: number;
  requiredCompletionRate: number;
  canRelease: boolean;
}
```

---

## 段階的ロールアウト

### カナリアリリース

```typescript
interface CanaryDeployment {
  // 段階的なトラフィック配分
  stages: DeploymentStage[];

  // モニタリング指標
  healthMetrics: {
    errorRate: number;
    latency: number;
    throughput: number;
    customMetrics: Record<string, number>;
  };

  // 自動ロールバックトリガー
  autoRollbackTriggers: {
    errorRateThreshold: number;
    latencyThreshold: number;
    customThresholds: Record<string, number>;
  };
}

interface DeploymentStage {
  name: string;
  trafficPercentage: number;
  duration: number; // 分
  successCriteria: SuccessCriteria;
}

interface SuccessCriteria {
  maxErrorRate: number;
  maxLatencyP95: number;
  minSuccessRate: number;
}

class CanaryDeploymentManager {
  private currentStage = 0;
  private stages: DeploymentStage[] = [
    {
      name: 'Initial Canary',
      trafficPercentage: 5,
      duration: 30,
      successCriteria: {
        maxErrorRate: 0.1,
        maxLatencyP95: 1000,
        minSuccessRate: 99.9,
      },
    },
    {
      name: 'Expanded Canary',
      trafficPercentage: 25,
      duration: 60,
      successCriteria: {
        maxErrorRate: 0.1,
        maxLatencyP95: 1000,
        minSuccessRate: 99.9,
      },
    },
    {
      name: 'Half Traffic',
      trafficPercentage: 50,
      duration: 120,
      successCriteria: {
        maxErrorRate: 0.1,
        maxLatencyP95: 1000,
        minSuccessRate: 99.9,
      },
    },
    {
      name: 'Full Rollout',
      trafficPercentage: 100,
      duration: 0,
      successCriteria: {
        maxErrorRate: 0.1,
        maxLatencyP95: 1000,
        minSuccessRate: 99.9,
      },
    },
  ];

  async startDeployment(): Promise<void> {
    console.log('🚀 Starting canary deployment...');

    for (let i = 0; i < this.stages.length; i++) {
      this.currentStage = i;
      const stage = this.stages[i];

      console.log(`\nStage ${i + 1}/${this.stages.length}: ${stage.name}`);
      console.log(`Traffic: ${stage.trafficPercentage}%`);

      // トラフィック配分を更新
      await this.updateTrafficSplit(stage.trafficPercentage);

      // モニタリング期間
      if (stage.duration > 0) {
        console.log(`Monitoring for ${stage.duration} minutes...`);
        await this.monitorStage(stage);
      }

      console.log(`✅ Stage ${i + 1} completed successfully`);
    }

    console.log('\n🎉 Deployment completed successfully!');
  }

  private async updateTrafficSplit(percentage: number): Promise<void> {
    // Kubernetes/Istio/AWS Load Balancer等でトラフィック配分を更新
    console.log(`Updating traffic split to ${percentage}%...`);
    // 実装は省略
  }

  private async monitorStage(stage: DeploymentStage): Promise<void> {
    const startTime = Date.now();
    const endTime = startTime + stage.duration * 60 * 1000;

    while (Date.now() < endTime) {
      const metrics = await this.collectMetrics();

      // 成功基準チェック
      if (!this.checkSuccessCriteria(metrics, stage.successCriteria)) {
        console.error('❌ Success criteria not met. Rolling back...');
        await this.rollback();
        throw new Error('Deployment failed: success criteria not met');
      }

      // 1分毎にチェック
      await this.sleep(60000);
    }
  }

  private async collectMetrics(): Promise<HealthMetrics> {
    // Prometheus/DataDog/CloudWatch等からメトリクス収集
    return {
      errorRate: 0.05,
      latencyP95: 850,
      successRate: 99.95,
    };
  }

  private checkSuccessCriteria(
    metrics: HealthMetrics,
    criteria: SuccessCriteria
  ): boolean {
    if (metrics.errorRate > criteria.maxErrorRate) {
      console.warn(`Error rate too high: ${metrics.errorRate}%`);
      return false;
    }

    if (metrics.latencyP95 > criteria.maxLatencyP95) {
      console.warn(`Latency too high: ${metrics.latencyP95}ms`);
      return false;
    }

    if (metrics.successRate < criteria.minSuccessRate) {
      console.warn(`Success rate too low: ${metrics.successRate}%`);
      return false;
    }

    return true;
  }

  private async rollback(): Promise<void> {
    console.log('Rolling back to previous version...');
    await this.updateTrafficSplit(0); // 新バージョンへのトラフィックを0に
    // 実装は省略
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

interface HealthMetrics {
  errorRate: number;
  latencyP95: number;
  successRate: number;
}
```

### フィーチャーフラグ活用

```swift
class FeatureFlagManager {
    enum RolloutStrategy {
        case percentage(Double)          // パーセンテージベース
        case userList([String])          // ユーザーリストベース
        case gradual(GradualRollout)     // 段階的ロールアウト
    }

    struct GradualRollout {
        let stages: [Stage]

        struct Stage {
            let percentage: Double
            let duration: TimeInterval
            let startDate: Date
        }
    }

    struct FeatureFlag {
        let name: String
        let enabled: Bool
        let strategy: RolloutStrategy
        let metadata: [String: Any]

        func isEnabledForUser(_ userId: String) -> Bool {
            guard enabled else { return false }

            switch strategy {
            case .percentage(let pct):
                return isUserInPercentage(userId, percentage: pct)

            case .userList(let users):
                return users.contains(userId)

            case .gradual(let rollout):
                let currentStage = rollout.getCurrentStage()
                return isUserInPercentage(userId, percentage: currentStage?.percentage ?? 0)
            }
        }

        private func isUserInPercentage(_ userId: String, percentage: Double) -> Bool {
            // 一貫性のあるハッシュベースの判定
            let hash = abs(userId.hashValue) % 100
            return Double(hash) < percentage
        }
    }
}

extension FeatureFlagManager.GradualRollout {
    func getCurrentStage() -> Stage? {
        let now = Date()

        for stage in stages {
            let endDate = stage.startDate.addingTimeInterval(stage.duration)
            if now >= stage.startDate && now < endDate {
                return stage
            }
        }

        // すべてのステージが完了している場合、最後のステージ
        return stages.last
    }
}

// 使用例
let newCheckoutFlag = FeatureFlagManager.FeatureFlag(
    name: "new_checkout_flow",
    enabled: true,
    strategy: .gradual(
        FeatureFlagManager.GradualRollout(
            stages: [
                .init(percentage: 5, duration: 86400, startDate: Date()), // 1日目: 5%
                .init(percentage: 25, duration: 86400, startDate: Date().addingTimeInterval(86400)), // 2日目: 25%
                .init(percentage: 50, duration: 86400, startDate: Date().addingTimeInterval(172800)), // 3日目: 50%
                .init(percentage: 100, duration: .infinity, startDate: Date().addingTimeInterval(259200)), // 4日目: 100%
            ]
        )
    ),
    metadata: [:]
)

// ユーザーにとって有効か確認
let userId = "user-12345"
if newCheckoutFlag.isEnabledForUser(userId) {
    // 新しいチェックアウトフローを表示
} else {
    // 旧チェックアウトフローを表示
}
```

---

## ロールバック戦略

### 自動ロールバック

```typescript
interface RollbackStrategy {
  // 自動ロールバック条件
  triggers: {
    errorRateThreshold: number;
    crashRateThreshold: number;
    latencyThreshold: number;
    customMetricThresholds: Record<string, number>;
  };

  // ロールバック手順
  procedure: RollbackProcedure;

  // 通知設定
  notifications: {
    slack: boolean;
    email: boolean;
    pagerduty: boolean;
  };
}

interface RollbackProcedure {
  steps: RollbackStep[];
  verificationSteps: string[];
}

interface RollbackStep {
  name: string;
  command: string;
  timeout: number;
  retryable: boolean;
}

class AutoRollbackMonitor {
  private metrics: MetricsCollector;
  private config: RollbackStrategy;

  constructor(config: RollbackStrategy) {
    this.config = config;
    this.metrics = new MetricsCollector();
  }

  async startMonitoring(): Promise<void> {
    console.log('🔍 Starting rollback monitoring...');

    setInterval(async () => {
      const currentMetrics = await this.metrics.collect();

      if (this.shouldRollback(currentMetrics)) {
        console.error('🚨 Rollback triggered!');
        await this.executeRollback(currentMetrics);
      }
    }, 60000); // 1分毎にチェック
  }

  private shouldRollback(metrics: CollectedMetrics): boolean {
    const { triggers } = this.config;

    if (metrics.errorRate > triggers.errorRateThreshold) {
      console.error(`Error rate exceeded: ${metrics.errorRate}% > ${triggers.errorRateThreshold}%`);
      return true;
    }

    if (metrics.crashRate > triggers.crashRateThreshold) {
      console.error(`Crash rate exceeded: ${metrics.crashRate}% > ${triggers.crashRateThreshold}%`);
      return true;
    }

    if (metrics.latencyP95 > triggers.latencyThreshold) {
      console.error(`Latency exceeded: ${metrics.latencyP95}ms > ${triggers.latencyThreshold}ms`);
      return true;
    }

    return false;
  }

  private async executeRollback(metrics: CollectedMetrics): Promise<void> {
    // 通知送信
    await this.sendNotifications(metrics);

    // ロールバック実行
    console.log('Executing rollback procedure...');

    for (const step of this.config.procedure.steps) {
      console.log(`Step: ${step.name}`);

      try {
        await this.executeStep(step);
        console.log(`✅ ${step.name} completed`);
      } catch (error) {
        console.error(`❌ ${step.name} failed:`, error);

        if (!step.retryable) {
          throw error;
        }

        // リトライ
        console.log(`Retrying ${step.name}...`);
        await this.executeStep(step);
      }
    }

    // 検証
    console.log('Verifying rollback...');
    await this.verifyRollback();

    console.log('✅ Rollback completed successfully');
  }

  private async executeStep(step: RollbackStep): Promise<void> {
    // コマンド実行（シェルコマンド、Kubernetes API等）
    // 実装は省略
  }

  private async verifyRollback(): Promise<void> {
    for (const verification of this.config.procedure.verificationSteps) {
      console.log(`Verifying: ${verification}`);
      // 検証ロジック
    }
  }

  private async sendNotifications(metrics: CollectedMetrics): Promise<void> {
    const message = `
🚨 自動ロールバック実行

理由:
- エラー率: ${metrics.errorRate}%
- クラッシュ率: ${metrics.crashRate}%
- レイテンシ: ${metrics.latencyP95}ms

ロールバックを開始しています...
    `;

    if (this.config.notifications.slack) {
      await this.sendSlackNotification(message);
    }

    if (this.config.notifications.email) {
      await this.sendEmailNotification(message);
    }

    if (this.config.notifications.pagerduty) {
      await this.triggerPagerDuty(message);
    }
  }

  private async sendSlackNotification(message: string): Promise<void> {
    // Slack Webhook実装
  }

  private async sendEmailNotification(message: string): Promise<void> {
    // Email送信実装
  }

  private async triggerPagerDuty(message: string): Promise<void> {
    // PagerDuty API実装
  }
}

class MetricsCollector {
  async collect(): Promise<CollectedMetrics> {
    // Prometheus/CloudWatch等からメトリクス収集
    return {
      errorRate: 0.05,
      crashRate: 0.01,
      latencyP95: 850,
      timestamp: new Date(),
    };
  }
}

interface CollectedMetrics {
  errorRate: number;
  crashRate: number;
  latencyP95: number;
  timestamp: Date;
}
```

---

## リリース後モニタリング

### リリース後24時間モニタリング計画

```markdown
# リリース後モニタリング計画

## 即時（0-1時間）

### モニタリング項目
- [ ] デプロイ成功確認
- [ ] ヘルスチェックエンドポイント応答確認
- [ ] エラー率（目標: < 0.1%）
- [ ] クラッシュ率（目標: < 0.05%）
- [ ] API応答時間（目標: P95 < 1秒）

### アクション
- リアルタイムダッシュボード監視
- エラーログ確認
- 主要機能の手動確認

### アラート閾値
- エラー率 > 0.2%
- クラッシュ率 > 0.1%
- API応答時間 P95 > 2秒

---

## 短期（1-6時間）

### モニタリング項目
- [ ] ユーザーアクティビティ（ログイン数、セッション数）
- [ ] 主要機能の利用状況
- [ ] データベースパフォーマンス
- [ ] キャッシュヒット率
- [ ] ユーザーフィードバック（レビュー、サポート問い合わせ）

### アクション
- メトリクスダッシュボード確認（30分毎）
- ユーザーレビュー監視
- サポートチケット確認

---

## 中期（6-24時間）

### モニタリング項目
- [ ] KPI達成状況
- [ ] コンバージョン率
- [ ] ユーザー継続率
- [ ] 新規バグ報告
- [ ] パフォーマンストレンド

### アクション
- 6時間毎のレポート確認
- 異常値の調査
- 改善機会の特定

---

## アラート対応フロー

```
アラート発生
    ↓
即座に確認（5分以内）
    ↓
Critical? ── Yes → 緊急対応チーム招集
    ↓              ↓
   No             ロールバック判断
    ↓              ↓
調査開始         実施 or ホットフィックス
    ↓
原因特定
    ↓
対策実施
    ↓
再発防止策策定
```
```

**モニタリング自動化:**

```swift
struct PostReleaseMonitor {
    struct MonitoringPlan {
        let releaseVersion: String
        let releaseTime: Date
        let checkpoints: [Checkpoint]

        struct Checkpoint {
            let timeOffset: TimeInterval // リリースからの経過時間
            let checks: [Check]

            struct Check {
                let name: String
                let metric: Metric
                let threshold: Threshold

                enum Metric {
                    case errorRate
                    case crashRate
                    case latency
                    case activeUsers
                    case customMetric(String)
                }

                struct Threshold {
                    let max: Double?
                    let min: Double?
                }
            }
        }
    }

    func execute(plan: MonitoringPlan) async {
        print("🔍 Post-release monitoring started for version \(plan.releaseVersion)")

        for checkpoint in plan.checkpoints {
            // 次のチェックポイントまで待機
            let waitTime = checkpoint.timeOffset
            try? await Task.sleep(nanoseconds: UInt64(waitTime * 1_000_000_000))

            print("\n⏰ Checkpoint at +\(formatDuration(checkpoint.timeOffset))")

            // チェック実行
            var allPassed = true

            for check in checkpoint.checks {
                let result = await performCheck(check)

                if result.passed {
                    print("✅ \(check.name): \(result.value)")
                } else {
                    print("❌ \(check.name): \(result.value) (閾値超過)")
                    allPassed = false

                    // アラート送信
                    await sendAlert(check: check, result: result)
                }
            }

            if allPassed {
                print("✅ All checks passed at this checkpoint")
            } else {
                print("⚠️ Some checks failed - review required")
            }
        }

        print("\n🎉 Post-release monitoring completed")
    }

    private func performCheck(_ check: MonitoringPlan.Checkpoint.Check) async -> CheckResult {
        // メトリクス取得
        let value = await fetchMetric(check.metric)

        // 閾値チェック
        let passed = checkThreshold(value: value, threshold: check.threshold)

        return CheckResult(
            checkName: check.name,
            value: value,
            passed: passed
        )
    }

    private func fetchMetric(_ metric: MonitoringPlan.Checkpoint.Check.Metric) async -> Double {
        // Prometheus/CloudWatch等からメトリクス取得
        // 簡略化のため固定値を返す
        switch metric {
        case .errorRate:
            return 0.05
        case .crashRate:
            return 0.02
        case .latency:
            return 850
        case .activeUsers:
            return 10000
        case .customMetric:
            return 0
        }
    }

    private func checkThreshold(value: Double, threshold: MonitoringPlan.Checkpoint.Check.Threshold) -> Bool {
        if let max = threshold.max, value > max {
            return false
        }

        if let min = threshold.min, value < min {
            return false
        }

        return true
    }

    private func sendAlert(check: MonitoringPlan.Checkpoint.Check, result: CheckResult) async {
        let message = """
        🚨 Post-Release Alert

        Check: \(result.checkName)
        Value: \(result.value)
        Status: Failed

        Please investigate immediately.
        """

        print(message)
        // Slack/Email/PagerDuty通知
    }

    private func formatDuration(_ duration: TimeInterval) -> String {
        let hours = Int(duration / 3600)
        let minutes = Int((duration.truncatingRemainder(dividingBy: 3600)) / 60)

        if hours > 0 {
            return "\(hours)h \(minutes)m"
        } else {
            return "\(minutes)m"
        }
    }

    struct CheckResult {
        let checkName: String
        let value: Double
        let passed: Bool
    }
}

// 使用例
let monitor = PostReleaseMonitor()

let plan = PostReleaseMonitor.MonitoringPlan(
    releaseVersion: "2.5.0",
    releaseTime: Date(),
    checkpoints: [
        // 15分後
        .init(timeOffset: 900, checks: [
            .init(
                name: "Error Rate",
                metric: .errorRate,
                threshold: .init(max: 0.1, min: nil)
            ),
            .init(
                name: "Crash Rate",
                metric: .crashRate,
                threshold: .init(max: 0.05, min: nil)
            ),
        ]),
        // 1時間後
        .init(timeOffset: 3600, checks: [
            .init(
                name: "Active Users",
                metric: .activeUsers,
                threshold: .init(max: nil, min: 5000)
            ),
        ]),
        // 6時間後
        .init(timeOffset: 21600, checks: [
            .init(
                name: "Latency P95",
                metric: .latency,
                threshold: .init(max: 1000, min: nil)
            ),
        ]),
    ]
)

Task {
    await monitor.execute(plan: plan)
}
```

---

## 実践例とケーススタディ

### ケーススタディ: モバイルアプリのリリース失敗と学び

**背景:**
- iOS/Androidアプリ v3.0のリリース
- 主要な新機能: ビデオ通話機能
- リリース判定会議で「GO」判断

**リリース後の問題:**
```markdown
## タイムライン

### T+0（リリース直後）
- App Store/Google Play公開完了
- 初期モニタリング: 正常

### T+30分
- クラッシュ率が急上昇: 0.05% → 1.2%
- ユーザーレビューに低評価が増加
- サポートチケット急増

### T+1時間
- 原因特定: 特定のAndroidデバイス（Xiaomi）でビデオ通話時にクラッシュ
- 影響範囲: Android全ユーザーの約15%

### T+2時間
- 緊急会議招集
- ロールバック判断: 実施

### T+3時間
- ホットフィックス対応開始
- Google Playからv3.0を削除、v2.9に戻す

### T+6時間
- 修正版v3.0.1リリース
- Xiaomiデバイスでのテスト完了

### T+12時間
- v3.0.1段階的ロールアウト開始（5% → 25% → 100%）
- 問題なく完了
```

**根本原因:**

```markdown
## 根本原因分析

### 直接原因
- XiaomiデバイスのカメラAPIの挙動が他のメーカーと異なる
- 権限ダイアログの表示タイミングでクラッシュ

### 間接原因
1. デバイスマトリクステストにXiaomiが含まれていなかった
2. ビデオ通話機能のE2Eテストが不十分
3. カナリアリリースを実施しなかった

### 改善策
✅ デバイスマトリクスにXiaomi追加（シェア15%）
✅ ビデオ通話E2Eテストを20シナリオ追加
✅ すべてのメジャーリリースでカナリア実施を必須化
✅ クラッシュ率アラート閾値を0.2% → 0.15%に引き下げ
✅ 自動ロールバック機能を実装
```

**教訓:**

```markdown
## Lessons Learned

### テスト観点
1. **市場シェアベースのデバイス選定**
   - 上位80%カバレッジを確保
   - マイナーメーカーでもシェア10%以上なら必須テスト

2. **主要機能の徹底テスト**
   - Critical機能は最低20シナリオ
   - 異常系テストも必須

### リリース戦略
3. **段階的ロールアウトの徹底**
   - すべてのメジャーリリースでカナリア実施
   - 5% → 25% → 50% → 100%の4段階

4. **自動ロールバックの実装**
   - クラッシュ率0.15%でアラート
   - 0.3%で自動ロールバック

### モニタリング
5. **リアルタイム監視の強化**
   - リリース後6時間は専任担当者配置
   - 15分毎のメトリクス確認

6. **ユーザーフィードバックの即時確認**
   - App Store/Google Playレビュー監視
   - サポートチケット集約ダッシュボード
```

---

## トラブルシューティング

### よくある問題と解決策

**1. リリース判定基準があいまい**

```markdown
## 問題: リリース判定基準があいまい

### 症状
- 毎回のリリース判定会議で議論が紛糾
- 判断基準が人によって異なる
- リリース延期の判断ができない

### 原因
❌ 定量的な基準がない
❌ ステークホルダー間で期待値が不一致
❌ リスク評価が主観的

### 解決策
✅ 定量的なクライテリアを文書化
✅ Must/Should/Nice to Haveの明確化
✅ リスク評価フレームワークの導入
✅ 過去の判定事例をナレッジベース化
```

**2. ロールバックに時間がかかる**

```markdown
## 問題: ロールバックに時間がかかる

### 症状
- 問題発生からロールバック完了まで3時間以上
- 手順が不明確で混乱
- ロールバック後も問題が残る

### 原因
❌ ロールバック手順が文書化されていない
❌ 自動化されていない
❌ テストされていない

### 解決策
✅ ロールバック手順の自動化
✅ 月1回のロールバック訓練実施
✅ ワンクリックロールバックの実装
✅ データベースロールバック戦略の確立
```

---

## まとめ

### リリース成功の鍵

```markdown
## 成功の5原則

1. **明確な基準**
   - 定量的なクライテリア
   - Must/Should/Nice to Haveの明確化
   - ステークホルダー間の合意

2. **段階的なアプローチ**
   - カナリアリリース
   - フィーチャーフラグ活用
   - リスクの最小化

3. **徹底したモニタリング**
   - リアルタイム監視
   - 自動アラート
   - 迅速な対応体制

4. **確実なロールバック**
   - 自動化された手順
   - 定期的な訓練
   - データの保護

5. **継続的な改善**
   - ポストモーテム実施
   - 教訓の共有
   - プロセスの最適化
```

---

**関連ガイド:**
- [QA Metrics & KPI Dashboard](./qa-metrics-kpi-dashboard.md)
- [Test Planning & Execution](./test-planning-execution.md)
- [QA Automation & Tools](./qa-automation-tools.md)
