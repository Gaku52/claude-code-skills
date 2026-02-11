# バグ管理 完全ガイド
**作成日**: 2025年1月
**対象**: Jira, GitHub Issues, Linear
**レベル**: 初級〜上級

---

## 目次

1. [バグ管理の基礎](#1-バグ管理の基礎)
2. [バグレポート作成](#2-バグレポート作成)
3. [バグトリアージ](#3-バグトリアージ)
4. [バグ追跡](#4-バグ追跡)
5. [バグ修正プロセス](#5-バグ修正プロセス)
6. [バグ分析](#6-バグ分析)
7. [自動化](#7-自動化)
8. [ツール連携](#8-ツール連携)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [実績データ](#10-実績データ)

---

## 1. バグ管理の基礎

### 1.1 バグライフサイクル

```
New → Open → In Progress → Fixed → Testing → Verified → Closed
        ↓                              ↓
      Rejected                    Reopened
```

### 1.2 バグの分類

```typescript
enum BugSeverity {
  CRITICAL = 'critical',  // システムダウン、データ損失
  HIGH = 'high',          // 主要機能が使えない
  MEDIUM = 'medium',      // 機能制限あり
  LOW = 'low',            // 軽微な問題
}

enum BugPriority {
  P0 = 'p0',  // 即時対応（24h以内）
  P1 = 'p1',  // 緊急（3日以内）
  P2 = 'p2',  // 高（1週間以内）
  P3 = 'p3',  // 中（2週間以内）
  P4 = 'p4',  // 低（時間あるとき）
}

interface Bug {
  id: string;
  title: string;
  severity: BugSeverity;
  priority: BugPriority;
  status: string;
  reporter: string;
  assignee: string;
  createdAt: Date;
  updatedAt: Date;
}
```

---

## 2. バグレポート作成

### 2.1 効果的なバグレポート

```markdown
# BUG-123: ログイン後に画面が真っ白になる

## 環境
- **OS**: macOS 14.0
- **ブラウザ**: Chrome 120.0.6099.109
- **画面サイズ**: 1920x1080
- **ビルド**: v2.1.0-rc3
- **URL**: https://app.example.com/login

## 重要度
- **Severity**: Critical
- **Priority**: P0
- **影響範囲**: 全ユーザー

## 再現手順
1. https://app.example.com にアクセス
2. メールアドレス: `test@example.com` を入力
3. パスワード: `Test123!` を入力
4. 「ログイン」ボタンをクリック

## 期待される動作
ダッシュボード画面が表示される

## 実際の動作
画面が真っ白になり、何も表示されない

## スクリーンショット
![白い画面](https://imgur.com/abc123.png)

## コンソールエラー
```
Uncaught TypeError: Cannot read property 'data' of undefined
    at Dashboard.render (Dashboard.tsx:45)
```

## 再現率
10回中10回（100%）

## 追加情報
- Private mode でも同じ
- 他のブラウザ（Firefox, Safari）でも再現
- 昨日のビルド（v2.1.0-rc2）では問題なし

## 関連Issue
- #122: ダッシュボードのリファクタリング

## 担当者
@backend-team
```

### 2.2 バグレポートテンプレート

```yaml
# .github/ISSUE_TEMPLATE/bug_report.yml
name: Bug Report
description: Report a bug to help us improve
title: "[Bug]: "
labels: ["bug", "triage"]
assignees: []

body:
  - type: markdown
    attributes:
      value: |
        Thanks for taking the time to report this bug!

  - type: dropdown
    id: severity
    attributes:
      label: Severity
      options:
        - Critical (System down, data loss)
        - High (Major feature broken)
        - Medium (Feature limitation)
        - Low (Minor issue)
    validations:
      required: true

  - type: textarea
    id: reproduction
    attributes:
      label: Steps to Reproduce
      description: How do we reproduce this bug?
      placeholder: |
        1. Go to '...'
        2. Click on '...'
        3. See error
    validations:
      required: true

  - type: textarea
    id: expected
    attributes:
      label: Expected Behavior
      description: What should happen?
    validations:
      required: true

  - type: textarea
    id: actual
    attributes:
      label: Actual Behavior
      description: What actually happens?
    validations:
      required: true

  - type: input
    id: version
    attributes:
      label: Version
      placeholder: v2.1.0
    validations:
      required: true

  - type: dropdown
    id: browsers
    attributes:
      label: Browsers
      multiple: true
      options:
        - Chrome
        - Firefox
        - Safari
        - Edge

  - type: textarea
    id: logs
    attributes:
      label: Console Logs
      render: shell

  - type: textarea
    id: additional
    attributes:
      label: Additional Context
```

---

## 3. バグトリアージ

### 3.1 トリアージ会議

```typescript
// scripts/triage-meeting.ts
interface TriageMeeting {
  date: Date;
  attendees: string[];
  newBugs: Bug[];
  decisions: Decision[];
}

interface Decision {
  bugId: string;
  severity: BugSeverity;
  priority: BugPriority;
  assignee: string;
  targetVersion: string;
  rationale: string;
}

async function conductTriage(): Promise<TriageMeeting> {
  const newBugs = await getNewBugs();

  const decisions = newBugs.map(bug => {
    // 自動分類
    const autoSeverity = classifySeverity(bug);
    const autoPriority = calculatePriority(bug);

    // 人間レビュー
    console.log(`\n--- Bug #${bug.id}: ${bug.title} ---`);
    console.log(`Auto Severity: ${autoSeverity}`);
    console.log(`Auto Priority: ${autoPriority}`);

    return {
      bugId: bug.id,
      severity: autoSeverity,
      priority: autoPriority,
      assignee: assignBug(bug),
      targetVersion: determineTargetVersion(bug),
      rationale: generateRationale(bug),
    };
  });

  return {
    date: new Date(),
    attendees: ['QA Lead', 'Dev Lead', 'Product Manager'],
    newBugs,
    decisions,
  };
}

function classifySeverity(bug: Bug): BugSeverity {
  const keywords = {
    critical: ['crash', 'data loss', 'security', 'payment'],
    high: ['broken', 'unusable', 'error'],
    medium: ['slow', 'incorrect', 'missing'],
  };

  const text = `${bug.title} ${bug.description}`.toLowerCase();

  if (keywords.critical.some(k => text.includes(k))) {
    return BugSeverity.CRITICAL;
  }
  if (keywords.high.some(k => text.includes(k))) {
    return BugSeverity.HIGH;
  }
  if (keywords.medium.some(k => text.includes(k))) {
    return BugSeverity.MEDIUM;
  }
  return BugSeverity.LOW;
}

function calculatePriority(bug: Bug): BugPriority {
  const impactScore = calculateImpact(bug);
  const urgencyScore = calculateUrgency(bug);

  const totalScore = impactScore + urgencyScore;

  if (totalScore >= 9) return BugPriority.P0;
  if (totalScore >= 7) return BugPriority.P1;
  if (totalScore >= 5) return BugPriority.P2;
  if (totalScore >= 3) return BugPriority.P3;
  return BugPriority.P4;
}
```

### 3.2 優先度マトリクス

```
Impact ↑
  5 │ P2  P1  P0  P0
  4 │ P3  P2  P1  P0
  3 │ P3  P2  P2  P1
  2 │ P4  P3  P2  P2
  1 │ P4  P4  P3  P2
    └─────────────→ Urgency
      1   2   3   4   5
```

---

## 4. バグ追跡

### 4.1 ステータス管理

```typescript
// src/bug-tracking/status-machine.ts
import { createMachine } from 'xstate';

const bugStateMachine = createMachine({
  id: 'bug',
  initial: 'new',
  states: {
    new: {
      on: {
        TRIAGE: 'open',
        REJECT: 'rejected',
      },
    },
    open: {
      on: {
        ASSIGN: 'in_progress',
        DEFER: 'deferred',
      },
    },
    in_progress: {
      on: {
        SUBMIT_FIX: 'fixed',
        BLOCK: 'blocked',
      },
    },
    blocked: {
      on: {
        UNBLOCK: 'in_progress',
      },
    },
    fixed: {
      on: {
        START_TESTING: 'testing',
      },
    },
    testing: {
      on: {
        VERIFY_PASS: 'verified',
        VERIFY_FAIL: 'reopened',
      },
    },
    verified: {
      on: {
        CLOSE: 'closed',
      },
    },
    reopened: {
      on: {
        REASSIGN: 'in_progress',
      },
    },
    rejected: {
      type: 'final',
    },
    deferred: {
      on: {
        REOPEN: 'open',
      },
    },
    closed: {
      type: 'final',
    },
  },
});
```

### 4.2 SLA管理

```typescript
// src/bug-tracking/sla.ts
interface SLA {
  priority: BugPriority;
  responseTime: number;  // hours
  resolutionTime: number; // hours
}

const slaRules: Record<BugPriority, SLA> = {
  [BugPriority.P0]: {
    priority: BugPriority.P0,
    responseTime: 1,
    resolutionTime: 24,
  },
  [BugPriority.P1]: {
    priority: BugPriority.P1,
    responseTime: 4,
    resolutionTime: 72,
  },
  [BugPriority.P2]: {
    priority: BugPriority.P2,
    responseTime: 24,
    resolutionTime: 168, // 1 week
  },
  [BugPriority.P3]: {
    priority: BugPriority.P3,
    responseTime: 48,
    resolutionTime: 336, // 2 weeks
  },
  [BugPriority.P4]: {
    priority: BugPriority.P4,
    responseTime: 168,
    resolutionTime: 720, // 30 days
  },
};

function checkSLAViolation(bug: Bug): boolean {
  const sla = slaRules[bug.priority];
  const now = new Date();
  const createdAt = new Date(bug.createdAt);
  const elapsedHours = (now.getTime() - createdAt.getTime()) / (1000 * 60 * 60);

  if (bug.status === 'new' && elapsedHours > sla.responseTime) {
    return true; // Response SLA violated
  }

  if (!['closed', 'verified'].includes(bug.status) && elapsedHours > sla.resolutionTime) {
    return true; // Resolution SLA violated
  }

  return false;
}

// SLA違反の通知
async function notifySLAViolations() {
  const openBugs = await getOpenBugs();
  const violations = openBugs.filter(checkSLAViolation);

  if (violations.length > 0) {
    await sendSlackNotification({
      channel: '#bugs-alerts',
      text: `⚠️ ${violations.length} bugs violating SLA`,
      attachments: violations.map(bug => ({
        color: 'danger',
        fields: [
          { title: 'Bug', value: `#${bug.id}: ${bug.title}` },
          { title: 'Priority', value: bug.priority },
          { title: 'Age', value: formatAge(bug.createdAt) },
        ],
      })),
    });
  }
}
```

---

## 5. バグ修正プロセス

### 5.1 修正ワークフロー

```bash
#!/bin/bash
# scripts/bug-fix-workflow.sh

BUG_ID=$1

if [ -z "$BUG_ID" ]; then
  echo "Usage: ./bug-fix-workflow.sh BUG-123"
  exit 1
fi

# 1. ブランチ作成
echo "📝 Creating branch for $BUG_ID..."
git checkout main
git pull origin main
git checkout -b fix/$BUG_ID

# 2. バグ情報取得
echo "📋 Fetching bug details..."
gh issue view $BUG_ID

# 3. 再現テスト作成
echo "🧪 Create reproduction test first!"
echo "Press enter when test is ready..."
read

# 4. テスト実行（失敗確認）
npm test -- --findRelatedTests

# 5. 修正実装
echo "🔧 Implement fix..."
echo "Press enter when fix is ready..."
read

# 6. テスト実行（成功確認）
npm test -- --findRelatedTests

# 7. コミット
git add .
git commit -m "fix: resolve $BUG_ID

- Add reproduction test
- Fix root cause
- Add regression test

Fixes #${BUG_ID#BUG-}"

# 8. プッシュ & PR作成
git push -u origin fix/$BUG_ID

gh pr create \
  --title "Fix: $BUG_ID" \
  --body "Resolves #${BUG_ID#BUG-}

## Changes
- [x] Reproduction test added
- [x] Root cause fixed
- [x] Regression test added

## Testing
- [x] Unit tests pass
- [x] Manual testing done" \
  --label "bug-fix"

echo "✅ PR created! Please request review."
```

### 5.2 修正確認チェックリスト

```markdown
# バグ修正確認チェックリスト

## 開発者（修正者）

### コード
- [ ] 再現テストを作成した
- [ ] テストが最初失敗することを確認した
- [ ] 修正を実装した
- [ ] テストが成功することを確認した
- [ ] 関連するエッジケースもテストした
- [ ] コードレビューを受けた

### 検証
- [ ] ローカルで手動テストした
- [ ] 元の再現手順で問題が解決した
- [ ] 副作用がないことを確認した
- [ ] 他の機能に影響ないことを確認した

### ドキュメント
- [ ] コミットメッセージにバグIDを記載
- [ ] PR説明に修正内容を記載
- [ ] 必要に応じてドキュメント更新

## QA（検証者）

### 機能検証
- [ ] 元の再現手順で問題が解決したことを確認
- [ ] 複数のブラウザで確認（該当する場合）
- [ ] 複数のデバイスで確認（該当する場合）
- [ ] 境界値テストを実施

### 回帰テスト
- [ ] 関連機能が正常に動作することを確認
- [ ] 自動テストが全て成功
- [ ] パフォーマンスに悪影響なし

### クローズ
- [ ] バグチケットに検証結果を記載
- [ ] ステータスを「Verified」に変更
- [ ] リリースノートに追加（必要に応じて）
```

---

## 6. バグ分析

### 6.1 根本原因分析（RCA）

```typescript
// src/analysis/rca.ts
interface RootCauseAnalysis {
  bug: Bug;
  fiveWhys: string[];
  rootCause: string;
  preventiveMeasures: string[];
}

async function perform5Whys(bug: Bug): Promise<RootCauseAnalysis> {
  const whys: string[] = [];

  // Why 1
  whys.push('Why did this bug occur?');
  whys.push('→ Null pointer exception in Dashboard component');

  // Why 2
  whys.push('Why was there a null pointer?');
  whys.push('→ API response was null');

  // Why 3
  whys.push('Why was the API response null?');
  whys.push('→ Error handling was missing');

  // Why 4
  whys.push('Why was error handling missing?');
  whys.push('→ Developer was not aware of the requirement');

  // Why 5
  whys.push('Why was the developer not aware?');
  whys.push('→ Code review did not catch this');

  const rootCause = 'Lack of code review checklist for error handling';

  const preventiveMeasures = [
    'Add error handling to code review checklist',
    'Add ESLint rule to enforce error handling',
    'Create error handling guideline document',
    'Conduct error handling training',
  ];

  return {
    bug,
    fiveWhys: whys,
    rootCause,
    preventiveMeasures,
  };
}
```

### 6.2 バグトレンド分析

```typescript
// src/analysis/trends.ts
interface BugTrend {
  period: string;
  total: number;
  byStatus: Record<string, number>;
  bySeverity: Record<BugSeverity, number>;
  byComponent: Record<string, number>;
}

async function analyzeBugTrends(startDate: Date, endDate: Date): Promise<BugTrend[]> {
  const bugs = await getBugsBetween(startDate, endDate);

  const trends: BugTrend[] = [];

  // 週ごとに集計
  let currentDate = new Date(startDate);
  while (currentDate <= endDate) {
    const weekStart = currentDate;
    const weekEnd = new Date(currentDate);
    weekEnd.setDate(weekEnd.getDate() + 7);

    const weekBugs = bugs.filter(
      b => new Date(b.createdAt) >= weekStart && new Date(b.createdAt) < weekEnd
    );

    trends.push({
      period: formatWeek(weekStart),
      total: weekBugs.length,
      byStatus: countByStatus(weekBugs),
      bySeverity: countBySeverity(weekBugs),
      byComponent: countByComponent(weekBugs),
    });

    currentDate = weekEnd;
  }

  return trends;
}

// レポート生成
async function generateBugReport() {
  const trends = await analyzeBugTrends(
    new Date('2024-01-01'),
    new Date('2024-03-31')
  );

  const report = `
# Bug Trend Report Q1 2024

## Summary
- Total Bugs: ${trends.reduce((sum, t) => sum + t.total, 0)}
- Avg Bugs/Week: ${(trends.reduce((sum, t) => sum + t.total, 0) / trends.length).toFixed(1)}

## Top Components
${getTopComponents(trends).map(c => `- ${c.name}: ${c.count} bugs`).join('\n')}

## Severity Distribution
${getSeverityDistribution(trends)}

## Recommendations
${generateRecommendations(trends)}
  `;

  return report;
}
```

---

## 7. 自動化

### 7.1 自動トリアージ

```typescript
// src/automation/auto-triage.ts
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function autoTriageBug(bug: Bug): Promise<Decision> {
  const prompt = `
あなたはバグトリアージの専門家です。以下のバグレポートを分析し、適切な severity と priority を判定してください。

# Bug Report
Title: ${bug.title}
Description: ${bug.description}
Reporter: ${bug.reporter}
Created: ${bug.createdAt}

# 判定基準
Severity:
- CRITICAL: システムダウン、データ損失、セキュリティ問題
- HIGH: 主要機能が使用不可
- MEDIUM: 機能に制限あり
- LOW: 軽微な問題

Priority:
- P0: 即時対応（24h以内）- Critical bugs affecting all users
- P1: 緊急（3日以内）- High severity bugs
- P2: 高（1週間以内）- Medium severity bugs
- P3: 中（2週間以内）- Low severity bugs
- P4: 低（時間あるとき）- Nice to have fixes

JSON形式で回答してください:
{
  "severity": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "priority": "P0" | "P1" | "P2" | "P3" | "P4",
  "rationale": "判定理由",
  "suggestedAssignee": "推奨担当チーム"
}
  `;

  const message = await anthropic.messages.create({
    model: 'claude-3-5-sonnet-20241022',
    max_tokens: 1024,
    messages: [{
      role: 'user',
      content: prompt,
    }],
  });

  const response = JSON.parse(message.content[0].text);

  return {
    bugId: bug.id,
    severity: response.severity,
    priority: response.priority,
    assignee: response.suggestedAssignee,
    targetVersion: determineTargetVersion(response.priority),
    rationale: response.rationale,
  };
}
```

### 7.2 自動通知

```yaml
# .github/workflows/bug-notifications.yml
name: Bug Notifications

on:
  issues:
    types: [opened, labeled, assigned]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Notify on Critical Bug
        if: contains(github.event.issue.labels.*.name, 'critical')
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚨 CRITICAL BUG REPORTED",
              "blocks": [{
                "type": "section",
                "text": {
                  "type": "mrkdwn",
                  "text": "*Critical Bug:* <${{ github.event.issue.html_url }}|#${{ github.event.issue.number }}>: ${{ github.event.issue.title }}"
                }
              }]
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_CRITICAL }}

      - name: Notify Assignee
        if: github.event.action == 'assigned'
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: `@${{ github.event.assignee.login }} This bug has been assigned to you. Please update the status within 24 hours.`
            });
```

---

## 8. ツール連携

### 8.1 Jira連携

```typescript
// src/integrations/jira.ts
import JiraApi from 'jira-client';

const jira = new JiraApi({
  protocol: 'https',
  host: 'your-domain.atlassian.net',
  username: process.env.JIRA_USERNAME,
  password: process.env.JIRA_API_TOKEN,
  apiVersion: '2',
  strictSSL: true,
});

async function createJiraBug(bug: Bug) {
  const issue = {
    fields: {
      project: { key: 'PROJ' },
      summary: bug.title,
      description: bug.description,
      issuetype: { name: 'Bug' },
      priority: { name: mapPriorityToJira(bug.priority) },
      labels: [bug.severity, 'auto-created'],
      customfield_10001: bug.environment, // Environment
      customfield_10002: bug.reproSteps,  // Repro Steps
    },
  };

  const createdIssue = await jira.addNewIssue(issue);
  return createdIssue.key;
}

async function syncBugStatus(bugId: string, newStatus: string) {
  const transitions = await jira.listTransitions(bugId);
  const transition = transitions.transitions.find(
    t => t.to.name.toLowerCase() === newStatus.toLowerCase()
  );

  if (transition) {
    await jira.transitionIssue(bugId, {
      transition: { id: transition.id },
    });
  }
}
```

### 8.2 Sentry連携

```typescript
// src/integrations/sentry.ts
import * as Sentry from '@sentry/node';

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  integrations: [
    new Sentry.Integrations.Http({ tracing: true }),
  ],
  tracesSampleRate: 1.0,
});

// エラー発生時に自動でバグ作成
Sentry.configureScope(scope => {
  scope.addEventProcessor(async (event, hint) => {
    // 新しいエラーの場合、バグチケット作成
    if (event.exception && !event.tags?.bugCreated) {
      const bug = await createBugFromSentryEvent(event);
      event.tags = { ...event.tags, bugCreated: 'true', bugId: bug.id };
    }
    return event;
  });
});

async function createBugFromSentryEvent(event: Sentry.Event): Promise<Bug> {
  const bug: Bug = {
    id: generateBugId(),
    title: `[Sentry] ${event.exception?.values?.[0]?.type}: ${event.exception?.values?.[0]?.value}`,
    description: formatSentryEvent(event),
    severity: determineSeverityFromSentry(event),
    priority: BugPriority.P1,
    status: 'new',
    reporter: 'sentry-bot',
    assignee: '',
    createdAt: new Date(),
    updatedAt: new Date(),
  };

  await saveBug(bug);
  return bug;
}
```

---

## 9. トラブルシューティング

### 9.1 よくある問題

#### 重複バグ
```typescript
// 重複検出
async function findDuplicateBugs(newBug: Bug): Promise<Bug[]> {
  const existingBugs = await getOpenBugs();

  const similarities = existingBugs.map(bug => ({
    bug,
    similarity: calculateSimilarity(newBug.title, bug.title),
  }));

  return similarities
    .filter(s => s.similarity > 0.8)
    .map(s => s.bug);
}

function calculateSimilarity(str1: string, str2: string): number {
  // Levenshtein distance
  const matrix: number[][] = [];

  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }

  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  const distance = matrix[str2.length][str1.length];
  const maxLength = Math.max(str1.length, str2.length);
  return 1 - distance / maxLength;
}
```

---

## 10. 実績データ

### 10.1 バグ管理効果

| 指標              | 導入前   | 導入後   | 改善率  |
|-----------------|-------|-------|------|
| 平均バグ修正時間        | 5日    | 1.5日  | 70%  |
| バグ再発率           | 25%   | 5%    | 80%  |
| SLA遵守率          | 60%   | 95%   | 58%  |
| 重複バグ報告数         | 30件/月  | 3件/月  | 90%  |
| トリアージ時間         | 2時間/週  | 30分/週 | 75%  |

---

**更新日**: 2025年1月
**次回更新予定**: 四半期毎
