# コードレビュープロセス 完全ガイド
**作成日**: 2025年1月
**対象**: GitHub, GitLab, Bitbucket
**レベル**: 中級〜上級

---

## 目次

1. [コードレビューの基礎](#1-コードレビューの基礎)
2. [レビュープロセス](#2-レビュープロセス)
3. [レビュー観点](#3-レビュー観点)
4. [効果的なフィードバック](#4-効果的なフィードバック)
5. [レビュー時間管理](#5-レビュー時間管理)
6. [チーム文化](#6-チーム文化)
7. [メトリクス測定](#7-メトリクス測定)
8. [ベストプラクティス](#8-ベストプラクティス)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [実績データ](#10-実績データ)

---

## 1. コードレビューの基礎

### 1.1 レビューの目的

```typescript
enum ReviewPurpose {
  BUG_DETECTION = 'バグの早期発見',
  CODE_QUALITY = 'コード品質の向上',
  KNOWLEDGE_SHARING = 'ナレッジ共有',
  CONSISTENCY = 'コーディング規約の統一',
  MENTORING = 'メンタリング・育成',
  SECURITY = 'セキュリティ問題の検出',
}

interface ReviewBenefits {
  immediate: string[];
  longTerm: string[];
}

const benefits: ReviewBenefits = {
  immediate: [
    'バグの早期発見（デプロイ前）',
    'コード品質の向上',
    'セキュリティ脆弱性の検出',
  ],
  longTerm: [
    'チーム全体のスキル向上',
    'コードベースの保守性向上',
    'バス因子の低減',
    'オンボーディング時間の短縮',
  ],
};
```

### 1.2 レビューの種類

```markdown
## 同期レビュー
- **ペアプログラミング**: リアルタイムでコード作成
- **モブプログラミング**: チーム全員で協力
- **同期レビュー会議**: 画面共有でレビュー

## 非同期レビュー
- **Pull Request**: GitHubでの非同期レビュー（最も一般的）
- **コードレビューツール**: Gerrit, Crucible等
- **メール/チャット**: 軽微な変更
```

---

## 2. レビュープロセス

### 2.1 標準的なフロー

```
開発完了 → セルフレビュー → PR作成 → 自動チェック → レビュー依頼
                                           ↓
                            フィードバック ← レビュー実施
                                           ↓
                            修正対応 → 再レビュー → 承認 → マージ
```

### 2.2 セルフレビュー

```typescript
// scripts/self-review-checklist.ts
interface SelfReviewChecklist {
  category: string;
  items: string[];
}

const selfReviewChecklist: SelfReviewChecklist[] = [
  {
    category: 'コード品質',
    items: [
      '命名が適切か',
      '関数が単一責任を持っているか',
      '重複コードがないか',
      'マジックナンバーを避けているか',
      'コメントが適切か',
    ],
  },
  {
    category: 'テスト',
    items: [
      'ユニットテストを追加したか',
      'エッジケースをテストしているか',
      '全てのテストが成功するか',
      'カバレッジが低下していないか',
    ],
  },
  {
    category: 'パフォーマンス',
    items: [
      'N+1クエリがないか',
      '不要なループがないか',
      'メモリリークの可能性がないか',
    ],
  },
  {
    category: 'セキュリティ',
    items: [
      'SQLインジェクション対策したか',
      'XSS対策したか',
      '機密情報をハードコードしていないか',
      '入力バリデーションを実装したか',
    ],
  },
  {
    category: 'ドキュメント',
    items: [
      'READMEを更新したか',
      'APIドキュメントを更新したか',
      'CHANGELOG を更新したか',
    ],
  },
];

async function performSelfReview(): Promise<void> {
  console.log('📋 Self Review Checklist\n');

  for (const section of selfReviewChecklist) {
    console.log(`\n## ${section.category}`);
    for (const item of section.items) {
      const answer = await prompt(`  - ${item} (y/n): `);
      if (answer.toLowerCase() !== 'y') {
        console.log(`    ⚠️  Please address: ${item}`);
      }
    }
  }

  console.log('\n✅ Self review completed!');
}
```

### 2.3 PR作成

```markdown
<!-- .github/pull_request_template.md -->
## 概要
<!-- このPRの目的を簡潔に説明 -->

## 変更内容
<!-- 主な変更点をリスト形式で -->
-
-
-

## 種類
<!-- 該当するものにチェック -->
- [ ] 新機能
- [ ] バグ修正
- [ ] リファクタリング
- [ ] パフォーマンス改善
- [ ] ドキュメント更新
- [ ] テスト追加

## テスト
<!-- テスト方法と結果 -->
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## セルフレビューチェックリスト
- [ ] コードを自分で確認した
- [ ] 命名が適切
- [ ] テストを追加した
- [ ] ドキュメントを更新した
- [ ] Breaking changesがある場合は明記した

## スクリーンショット
<!-- UI変更がある場合 -->

## 関連Issue
Closes #

## レビュー観点
<!-- レビュワーに特に見てほしいポイント -->
```

---

## 3. レビュー観点

### 3.1 レビューチェックリスト

```typescript
// src/review/checklist.ts
interface ReviewCategory {
  name: string;
  priority: 'critical' | 'high' | 'medium' | 'low';
  items: ReviewItem[];
}

interface ReviewItem {
  question: string;
  examples?: { good?: string; bad?: string };
}

const reviewChecklist: ReviewCategory[] = [
  {
    name: '機能性',
    priority: 'critical',
    items: [
      {
        question: '要件を満たしているか',
      },
      {
        question: 'エッジケースが考慮されているか',
        examples: {
          bad: 'if (users.length > 0) { ... }  // 空配列のみ考慮',
          good: 'if (users?.length) { ... }  // null/undefined も考慮',
        },
      },
      {
        question: 'エラーハンドリングが適切か',
      },
    ],
  },
  {
    name: 'コード品質',
    priority: 'high',
    items: [
      {
        question: '関数が小さく、単一責任か',
        examples: {
          bad: `
// 100行の巨大関数
function processUser(user) {
  // バリデーション
  // データ変換
  // API呼び出し
  // エラー処理
  // ログ記録
  // ...
}`,
          good: `
function processUser(user) {
  validateUser(user);
  const transformed = transformUserData(user);
  return saveUser(transformed);
}`,
        },
      },
      {
        question: '命名が適切か',
        examples: {
          bad: 'const d = new Date();  // 何のデータ？',
          good: 'const createdAt = new Date();',
        },
      },
      {
        question: '重複コードがないか',
      },
    ],
  },
  {
    name: 'パフォーマンス',
    priority: 'high',
    items: [
      {
        question: 'N+1クエリがないか',
        examples: {
          bad: `
// ユーザー1人ごとにクエリ
for (const user of users) {
  const posts = await db.posts.find({ userId: user.id });
}`,
          good: `
// 一括取得
const userIds = users.map(u => u.id);
const posts = await db.posts.find({ userId: { $in: userIds } });`,
        },
      },
      {
        question: '不要な計算がループ内にないか',
      },
    ],
  },
  {
    name: 'セキュリティ',
    priority: 'critical',
    items: [
      {
        question: 'SQLインジェクション対策されているか',
        examples: {
          bad: `query = "SELECT * FROM users WHERE id = " + userId;`,
          good: `query = "SELECT * FROM users WHERE id = ?", [userId];`,
        },
      },
      {
        question: 'XSS対策されているか',
      },
      {
        question: '機密情報がハードコードされていないか',
      },
    ],
  },
  {
    name: 'テスト',
    priority: 'high',
    items: [
      {
        question: 'テストが追加されているか',
      },
      {
        question: 'テストカバレッジが十分か',
      },
      {
        question: 'エッジケースのテストがあるか',
      },
    ],
  },
];
```

### 3.2 重要度別レビュー観点

```typescript
// レビュー時間配分
const reviewTimeFocus = {
  critical: {
    percentage: 50,
    items: [
      'セキュリティ',
      'データ整合性',
      'バグの可能性',
    ],
  },
  high: {
    percentage: 30,
    items: [
      'パフォーマンス',
      'エラーハンドリング',
      'テスト',
    ],
  },
  medium: {
    percentage: 15,
    items: [
      'コード品質',
      '可読性',
      '保守性',
    ],
  },
  low: {
    percentage: 5,
    items: [
      'コメント',
      'フォーマット',
      'タイポ',
    ],
  },
};
```

---

## 4. 効果的なフィードバック

### 4.1 建設的なコメント

```typescript
// good-feedback-examples.ts
const feedbackExamples = {
  good: [
    {
      type: '提案型',
      example: `
💡 **提案**: この処理は \`useMemo\` でメモ化すると、再レンダリング時のパフォーマンスが向上します。

\`\`\`typescript
const sortedUsers = useMemo(() =>
  users.sort((a, b) => a.name.localeCompare(b.name)),
  [users]
);
\`\`\`

**理由**: \`users\` が変更されない限り、毎回ソートする必要がないため。
      `,
    },
    {
      type: '質問型',
      example: `
❓ **質問**: この関数で \`null\` が返ることはありますか？もしそうなら、呼び出し側での null チェックが必要そうです。
      `,
    },
    {
      type: '称賛+提案',
      example: `
👍 **Good**: エラーハンドリングがしっかりしていて素晴らしいです！

💡 **Suggestion**: エラーメッセージをもう少しユーザーフレンドリーにできそうです。

\`\`\`typescript
// Before
throw new Error('Invalid input');

// After
throw new Error('メールアドレスの形式が正しくありません');
\`\`\`
      `,
    },
  ],
  bad: [
    {
      type: '曖昧で攻撃的',
      example: 'これは良くないです。修正してください。',
      why: '何が問題か不明確、命令的',
    },
    {
      type: '個人攻撃',
      example: '何でこんなコード書いたの？',
      why: '人格攻撃、建設的でない',
    },
    {
      type: '解決策なし',
      example: 'この実装は間違っています。',
      why: '問題指摘のみで改善案なし',
    },
  ],
};
```

### 4.2 コメントテンプレート

```markdown
## 🐛 バグ指摘
**問題**: [具体的な問題]
**影響**: [どんな問題が起こるか]
**修正案**: [コード例]

## 💡 改善提案
**現状**: [現在の実装]
**提案**: [改善案]
**理由**: [なぜ改善すべきか]
**参考**: [ドキュメントやベストプラクティスのリンク]

## ❓ 質問
**質問**: [理解したい点]
**背景**: [なぜ質問するか]

## 👍 称賛
**Good**: [良かった点]
**理由**: [なぜ良いか]

## ⚠️ 懸念点
**懸念**: [気になる点]
**シナリオ**: [問題が起きうる状況]
**提案**: [対策案]

## 🔍 Nitpick（小さな指摘）
**Nit**: [細かい指摘]
※ブロッカーではない
```

---

## 5. レビュー時間管理

### 5.1 レビュー時間の目安

```typescript
interface ReviewTimeEstimate {
  prSize: string;
  linesChanged: number;
  estimatedTime: number; // minutes
  maxReviewTime: number; // minutes
}

const reviewTimeGuide: ReviewTimeEstimate[] = [
  { prSize: 'XS', linesChanged: 10, estimatedTime: 5, maxReviewTime: 10 },
  { prSize: 'S', linesChanged: 100, estimatedTime: 15, maxReviewTime: 30 },
  { prSize: 'M', linesChanged: 500, estimatedTime: 30, maxReviewTime: 60 },
  { prSize: 'L', linesChanged: 1000, estimatedTime: 60, maxReviewTime: 90 },
  { prSize: 'XL', linesChanged: 1000, estimatedTime: -1, maxReviewTime: -1 },
  // XL: 分割推奨
];

function estimateReviewTime(linesChanged: number): number {
  const guide = reviewTimeGuide.find(
    g => linesChanged <= g.linesChanged
  ) || reviewTimeGuide[reviewTimeGuide.length - 1];

  if (guide.estimatedTime === -1) {
    console.warn('⚠️  PRが大きすぎます。分割を検討してください。');
    return -1;
  }

  return guide.estimatedTime;
}
```

### 5.2 レビュー効率化

```typescript
// scripts/review-efficiency.ts
interface ReviewSession {
  startTime: Date;
  endTime?: Date;
  prId: string;
  linesReviewed: number;
  commentsAdded: number;
}

class ReviewTimer {
  private sessions: ReviewSession[] = [];
  private currentSession?: ReviewSession;

  startReview(prId: string) {
    this.currentSession = {
      startTime: new Date(),
      prId,
      linesReviewed: 0,
      commentsAdded: 0,
    };
  }

  endReview(linesReviewed: number, commentsAdded: number) {
    if (!this.currentSession) return;

    this.currentSession.endTime = new Date();
    this.currentSession.linesReviewed = linesReviewed;
    this.currentSession.commentsAdded = commentsAdded;

    this.sessions.push(this.currentSession);
    this.currentSession = undefined;

    this.analyzeEfficiency();
  }

  private analyzeEfficiency() {
    const session = this.sessions[this.sessions.length - 1];
    if (!session.endTime) return;

    const durationMinutes =
      (session.endTime.getTime() - session.startTime.getTime()) / 1000 / 60;

    const linesPerMinute = session.linesReviewed / durationMinutes;

    console.log('\n📊 Review Metrics:');
    console.log(`  Duration: ${durationMinutes.toFixed(1)} min`);
    console.log(`  Lines reviewed: ${session.linesReviewed}`);
    console.log(`  Comments: ${session.commentsAdded}`);
    console.log(`  Speed: ${linesPerMinute.toFixed(0)} lines/min`);

    if (linesPerMinute > 500) {
      console.warn('  ⚠️  Review might be too fast. Consider slowing down.');
    }
    if (durationMinutes > 60) {
      console.warn('  ⚠️  Review took too long. Consider taking a break.');
    }
  }
}
```

---

## 6. チーム文化

### 6.1 レビュー文化の醸成

```markdown
# コードレビューガイドライン

## レビュワーの心得

### Do ✅
- コードではなく問題を指摘する
- 具体的で建設的なフィードバックを提供
- 良いコードを称賛する
- 質問する姿勢を持つ
- 学びの機会と捉える

### Don't ❌
- 人格攻撃をしない
- 完璧を求めすぎない
- 自分の好みを押し付けない
- レビューを遅延させない
- 小さな指摘で会話をブロックしない

## 作者の心得

### Do ✅
- フィードバックを謙虚に受け入れる
- 不明点は質問する
- 迅速に対応する
- 議論を建設的に進める
- レビュワーに感謝する

### Don't ❌
- フィードバックを個人攻撃と受け取らない
- 防衛的にならない
- 無視しない
- 同じ指摘を繰り返されないようにする
```

### 6.2 レビュー会議

```typescript
// scripts/review-meeting.ts
interface ReviewMeeting {
  date: Date;
  attendees: string[];
  prsReviewed: string[];
  discussions: Discussion[];
  decisions: Decision[];
}

interface Discussion {
  topic: string;
  participants: string[];
  conclusion: string;
}

async function conductReviewMeeting(): Promise<ReviewMeeting> {
  const pendingPRs = await getPendingPRs();

  console.log('🔍 Code Review Meeting\n');

  const meeting: ReviewMeeting = {
    date: new Date(),
    attendees: [],
    prsReviewed: [],
    discussions: [],
    decisions: [],
  };

  for (const pr of pendingPRs.slice(0, 5)) {
    console.log(`\n--- PR #${pr.number}: ${pr.title} ---`);
    console.log(`Author: ${pr.author}`);
    console.log(`Size: ${pr.additions + pr.deletions} lines`);

    // 主要な変更点を表示
    const keyChanges = await getKeyChanges(pr);
    console.log('\nKey Changes:');
    keyChanges.forEach(change => console.log(`  - ${change}`));

    // ディスカッション
    const discussion = await discuss(pr);
    meeting.discussions.push(discussion);

    // 決定
    const decision = await decide(pr);
    meeting.decisions.push(decision);

    meeting.prsReviewed.push(pr.number);
  }

  return meeting;
}
```

---

## 7. メトリクス測定

### 7.1 レビューメトリクス

```typescript
// src/metrics/review-metrics.ts
interface ReviewMetrics {
  prCount: number;
  avgReviewTime: number; // hours
  avgTimeToFirstReview: number; // hours
  avgCommentsPerPR: number;
  approvalRate: number; // %
  changesRequestedRate: number; // %
  avgCycleTime: number; // hours (PR作成からマージまで)
}

async function collectReviewMetrics(
  startDate: Date,
  endDate: Date
): Promise<ReviewMetrics> {
  const prs = await getPRsBetween(startDate, endDate);

  const reviewTimes = prs.map(pr => calculateReviewTime(pr));
  const timeToFirstReview = prs.map(pr => calculateTimeToFirstReview(pr));
  const comments = prs.map(pr => pr.comments.length);

  const approved = prs.filter(pr => pr.state === 'approved').length;
  const changesRequested = prs.filter(pr => pr.state === 'changes_requested').length;

  return {
    prCount: prs.length,
    avgReviewTime: average(reviewTimes),
    avgTimeToFirstReview: average(timeToFirstReview),
    avgCommentsPerPR: average(comments),
    approvalRate: (approved / prs.length) * 100,
    changesRequestedRate: (changesRequested / prs.length) * 100,
    avgCycleTime: average(prs.map(pr => calculateCycleTime(pr))),
  };
}

function calculateReviewTime(pr: PullRequest): number {
  if (!pr.reviewStartedAt || !pr.reviewCompletedAt) return 0;

  return (
    (pr.reviewCompletedAt.getTime() - pr.reviewStartedAt.getTime()) /
    1000 /
    60 /
    60
  );
}
```

### 7.2 ダッシュボード

```typescript
// dashboard/review-dashboard.tsx
export function ReviewDashboard() {
  const metrics = useReviewMetrics();

  return (
    <div className="dashboard">
      <MetricCard
        title="平均レビュー時間"
        value={`${metrics.avgReviewTime.toFixed(1)}h`}
        target="< 4h"
        status={metrics.avgReviewTime < 4 ? 'good' : 'warning'}
      />

      <MetricCard
        title="初回レビュ ーまでの時間"
        value={`${metrics.avgTimeToFirstReview.toFixed(1)}h`}
        target="< 2h"
        status={metrics.avgTimeToFirstReview < 2 ? 'good' : 'warning'}
      />

      <MetricCard
        title="承認率"
        value={`${metrics.approvalRate.toFixed(0)}%`}
        trend="+5%"
      />

      <Chart
        type="line"
        data={metrics.history}
        title="レビュー時間推移"
      />
    </div>
  );
}
```

---

## 8. ベストプラクティス

### 8.1 小さなPR

```typescript
// 理想的なPRサイズ
const idealPRSize = {
  linesChanged: 200,  // 理想は200行以下
  maxLinesChanged: 400, // 最大400行
  filesChanged: 5,    // 理想は5ファイル以下
};

// PR分割の例
const features = [
  'ユーザー登録機能',
];

// ❌ 悪い例: 全てを1つのPRに
const badPR = {
  title: 'ユーザー登録機能追加',
  changes: [
    'モデル追加',
    'API追加',
    'UI追加',
    'バリデーション追加',
    'テスト追加',
  ],
  linesChanged: 1500,
};

// ✅ 良い例: 機能ごとに分割
const goodPRs = [
  {
    title: 'feat: ユーザーモデル追加',
    linesChanged: 150,
  },
  {
    title: 'feat: ユーザー登録API追加',
    linesChanged: 200,
  },
  {
    title: 'feat: ユーザー登録UI追加',
    linesChanged: 250,
  },
  {
    title: 'test: ユーザー登録機能のテスト追加',
    linesChanged: 300,
  },
];
```

### 8.2 レビュー自動化

```yaml
# .github/workflows/pr-checks.yml
name: PR Checks

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  size-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Check PR size
        uses: actions/github-script@v7
        with:
          script: |
            const pr = context.payload.pull_request;
            const changes = pr.additions + pr.deletions;

            if (changes > 400) {
              core.setFailed(`PR is too large (${changes} lines). Please split into smaller PRs.`);
            } else if (changes > 200) {
              core.warning(`PR is large (${changes} lines). Consider splitting.`);
            }

  auto-label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v5
        with:
          repo-token: ${{ secrets.GITHUB_TOKEN }}

  auto-assign-reviewers:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v1.2.5
        with:
          configuration-path: '.github/auto-assign.yml'
```

---

## 9. トラブルシューティング

### 9.1 よくある問題

#### レビューが遅い
```typescript
// 解決策: SLA設定
const reviewSLA = {
  firstReviewTime: 4, // hours
  approvalTime: 24, // hours
};

// 自動リマインダー
async function sendReviewReminder() {
  const pendingPRs = await getPendingPRs();

  for (const pr of pendingPRs) {
    const hoursSinceCreated = getHoursSince(pr.createdAt);

    if (hoursSinceCreated > reviewSLA.firstReviewTime && !pr.hasReviews) {
      await sendSlackMessage(pr.requestedReviewers, {
        text: `⏰ PR #${pr.number} needs first review`,
        link: pr.url,
      });
    }
  }
}
```

#### コメントが多すぎて収拾つかない
```markdown
## 解決策
1. **重要度ラベル**: `[Critical]`, `[Nit]` を使用
2. **会話を収束**: 3往復以上は同期ミーティング
3. **ブロッキング明示**: "This blocks approval" を明記
```

---

## 10. 実績データ

### 10.1 レビュー効果

| 指標           | レビューなし | レビューあり | 改善率  |
|--------------|--------|--------|------|
| バグ密度（bugs/KLOC） | 25     | 5      | 80%  |
| 本番バグ率        | 40%    | 8%     | 80%  |
| コード品質スコア     | 60     | 85     | 42%  |
| 開発者スキル向上     | -      | +30%   | -    |

### 10.2 ROI

```
コスト:
- レビュー時間: 30分/PR × 100 PR/月 = 50h/月
- コスト: 50h × $50/h = $2,500/月

効果:
- バグ修正削減: 20 bugs/月 × 4h/bug × $50/h = $4,000/月
- ROI: ($4,000 - $2,500) / $2,500 = 60%

結論: 60%のROI、導入推奨
```

---

**更新日**: 2025年1月
**次回更新予定**: 四半期毎
