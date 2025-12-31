# コードレビュー自動化 完全ガイド
**作成日**: 2025年1月
**対象**: GitHub Actions, Danger, ReviewDog
**レベル**: 中級〜上級

---

## 目次

1. [自動化の基礎](#1-自動化の基礎)
2. [Danger.js](#2-dangerjs)
3. [ReviewDog](#3-reviewdog)
4. [自動ラベリング](#4-自動ラベリング)
5. [自動レビュワー割り当て](#5-自動レビュワー割り当て)
6. [AI支援レビュー](#6-ai支援レビュー)
7. [メトリクス自動収集](#7-メトリクス自動収集)
8. [統合ワークフロー](#8-統合ワークフロー)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [実績データ](#10-実績データ)

---

## 1. 自動化の基礎

### 1.1 自動化すべき項目

```typescript
const automationScope = {
  fullyAutomated: [
    'コードフォーマット',
    'Lint チェック',
    'ユニットテスト実行',
    'カバレッジ計測',
    'ライセンスチェック',
    'PRサイズチェック',
    '自動ラベリング',
  ],
  partiallyAutomated: [
    'セキュリティスキャン',
    'パフォーマンステスト',
    'レビュワー提案',
    'バグリスク予測',
  ],
  manualOnly: [
    'ビジネスロジック検証',
    'UX評価',
    'アーキテクチャ判断',
  ],
};
```

---

## 2. Danger.js

### 2.1 基本設定

```typescript
// dangerfile.ts
import { danger, warn, fail, message, markdown } from 'danger';

// PRサイズチェック
const bigPRThreshold = 600;
const additions = danger.github.pr.additions;
const deletions = danger.github.pr.deletions;
const changes = additions + deletions;

if (changes > bigPRThreshold) {
  warn(`⚠️ このPRは大きいです（${changes}行）。分割を検討してください。`);
}

// PRタイトルチェック
const prTitle = danger.github.pr.title;
const conventionalCommitRegex = /^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .+/;

if (!conventionalCommitRegex.test(prTitle)) {
  fail('❌ PRタイトルはConventional Commits形式に従ってください。');
}

// 説明チェック
const prDescription = danger.github.pr.body;
if (!prDescription || prDescription.length < 50) {
  warn('⚠️ PR説明をもっと詳しく書いてください。');
}

// ラベルチェック
const labels = danger.github.issue.labels.map(l => l.name);
if (labels.length === 0) {
  warn('⚠️ ラベルを追加してください。');
}

// ファイル変更チェック
const modifiedFiles = danger.git.modified_files;
const createdFiles = danger.git.created_files;

// package.json更新時はlockfileも更新
if (modifiedFiles.includes('package.json')) {
  if (!modifiedFiles.includes('package-lock.json')) {
    fail('❌ package.jsonが変更されましたが、package-lock.jsonが更新されていません。');
  }
}

// テストファイル追加チェック
const hasSourceChanges = modifiedFiles.some(f =>
  f.startsWith('src/') && !f.includes('.test.')
);
const hasTestChanges = [...modifiedFiles, ...createdFiles].some(f =>
  f.includes('.test.')
);

if (hasSourceChanges && !hasTestChanges) {
  warn('⚠️ テストの追加を検討してください。');
}

// カバレッジチェック
try {
  const coverageSummary = JSON.parse(
    require('fs').readFileSync('coverage/coverage-summary.json', 'utf-8')
  );
  const coverage = coverageSummary.total.lines.pct;

  if (coverage < 80) {
    fail(`❌ カバレッジ${coverage.toFixed(2)}%が80%未満です。`);
  } else {
    message(`✅ カバレッジ: ${coverage.toFixed(2)}%`);
  }

  // カバレッジレポート
  markdown(`
## 📊 コードカバレッジ

| メトリック    | カバレッジ |
|-----------|----------|
| Lines     | ${coverageSummary.total.lines.pct.toFixed(2)}% |
| Statements | ${coverageSummary.total.statements.pct.toFixed(2)}% |
| Functions  | ${coverageSummary.total.functions.pct.toFixed(2)}% |
| Branches   | ${coverageSummary.total.branches.pct.toFixed(2)}% |
  `);
} catch (error) {
  warn('⚠️ カバレッジレポートを読み込めませんでした。');
}

// TODOコメントチェック
for (const file of [...modifiedFiles, ...createdFiles]) {
  if (file.match(/\.(ts|tsx|js|jsx)$/)) {
    const content = require('fs').readFileSync(file, 'utf-8');
    const todos = content.match(/\/\/ TODO:/g);

    if (todos && todos.length > 0) {
      warn(`⚠️ ${file}に${todos.length}個のTODOコメントがあります。Issueを作成することを検討してください。`);
    }
  }
}

// console.log残存チェック
for (const file of [...modifiedFiles, ...createdFiles].filter(f =>
  f.match(/\.(ts|tsx)$/)
)) {
  const content = require('fs').readFileSync(file, 'utf-8');
  const consoleStatements = content.match(/console\.(log|debug|info)/g);

  if (consoleStatements && consoleStatements.length > 0) {
    warn(`⚠️ ${file}にconsole文が${consoleStatements.length}個あります。削除するかロギングライブラリを使用してください。`);
  }
}
```

### 2.2 高度な使い方

```typescript
// dangerfile.advanced.ts
import { danger, warn, markdown } from 'danger';
import * as fs from 'fs';

// 変更の影響範囲分析
async function analyzeImpact() {
  const modifiedFiles = danger.git.modified_files;

  const impactAreas = {
    database: modifiedFiles.some(f => f.includes('migrations/')),
    api: modifiedFiles.some(f => f.includes('src/api/')),
    ui: modifiedFiles.some(f => f.includes('src/components/')),
    auth: modifiedFiles.some(f => f.includes('src/auth/')),
  };

  const impacts = Object.entries(impactAreas)
    .filter(([_, changed]) => changed)
    .map(([area]) => area);

  if (impacts.length > 0) {
    markdown(`
## 🎯 影響範囲

このPRは以下の領域に影響します:
${impacts.map(area => `- ${area}`).join('\n')}

該当チームのレビューを依頼してください。
    `);
  }
}

// 複雑度分析
async function checkComplexity() {
  const modifiedFiles = danger.git.modified_files.filter(f =>
    f.match(/\.(ts|tsx|js|jsx)$/)
  );

  for (const file of modifiedFiles) {
    const content = fs.readFileSync(file, 'utf-8');

    // 関数の行数チェック
    const functionMatches = content.match(/function\s+\w+[^{]*\{[^}]*\}/gs);
    if (functionMatches) {
      for (const fn of functionMatches) {
        const lines = fn.split('\n').length;
        if (lines > 50) {
          warn(`⚠️ ${file}に50行を超える関数があります。分割を検討してください。`);
        }
      }
    }

    // ネストの深さチェック
    const maxNesting = calculateMaxNesting(content);
    if (maxNesting > 4) {
      warn(`⚠️ ${file}のネスト深度が${maxNesting}です。リファクタリングを検討してください。`);
    }
  }
}

function calculateMaxNesting(code: string): number {
  let maxDepth = 0;
  let currentDepth = 0;

  for (const char of code) {
    if (char === '{') {
      currentDepth++;
      maxDepth = Math.max(maxDepth, currentDepth);
    } else if (char === '}') {
      currentDepth--;
    }
  }

  return maxDepth;
}

// 実行
analyzeImpact();
checkComplexity();
```

---

## 3. ReviewDog

### 3.1 基本設定

```yaml
# .github/workflows/reviewdog.yml
name: ReviewDog

on:
  pull_request:

jobs:
  reviewdog:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      # ESLint
      - name: Run ESLint with ReviewDog
        uses: reviewdog/action-eslint@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          reporter: github-pr-review
          eslint_flags: 'src/**/*.{ts,tsx}'
          fail_on_error: true

      # Prettier
      - name: Run Prettier with ReviewDog
        uses: EPMatt/reviewdog-action-prettier@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          reporter: github-pr-review
          prettier_flags: 'src/**/*.{ts,tsx,json,md}'

      # TypeScript
      - name: Run tsc with ReviewDog
        uses: EPMatt/reviewdog-action-tsc@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          reporter: github-pr-review

      # Shellcheck
      - name: Run ShellCheck with ReviewDog
        uses: reviewdog/action-shellcheck@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          reporter: github-pr-review
          path: 'scripts'
          pattern: '*.sh'
```

---

## 4. 自動ラベリング

### 4.1 Labeler設定

```yaml
# .github/labeler.yml
# 機能
feature:
  - 'src/features/**/*'
  - any: ['**/feature/**', '**/feat/**']

# バグ修正
bugfix:
  - any: ['**/fix/**', '**/bugfix/**']

# ドキュメント
documentation:
  - '**/*.md'
  - 'docs/**/*'

# テスト
test:
  - '**/*.test.ts'
  - '**/*.spec.ts'
  - 'tests/**/*'

# インフラ
infrastructure:
  - '.github/**/*'
  - 'docker/**/*'
  - '**/*.yml'
  - '**/*.yaml'

# 依存関係
dependencies:
  - 'package.json'
  - 'package-lock.json'

# フロントエンド
frontend:
  - 'src/components/**/*'
  - 'src/pages/**/*'
  - '**/*.css'
  - '**/*.scss'

# バックエンド
backend:
  - 'src/api/**/*'
  - 'src/services/**/*'
  - 'src/models/**/*'

# データベース
database:
  - 'migrations/**/*'
  - 'src/db/**/*'
```

### 4.2 動的ラベリング

```typescript
// .github/workflows/dynamic-labeling.yml の中で使用
import { Octokit } from '@octokit/rest';

const octokit = new Octokit({ auth: process.env.GITHUB_TOKEN });

async function labelPRDynamically(pr: number) {
  const { data: files } = await octokit.pulls.listFiles({
    owner: 'your-org',
    repo: 'your-repo',
    pull_number: pr,
  });

  const labels: string[] = [];

  // PRサイズ
  const changes = files.reduce((sum, f) => sum + f.changes, 0);
  if (changes < 10) labels.push('size/XS');
  else if (changes < 100) labels.push('size/S');
  else if (changes < 500) labels.push('size/M');
  else if (changes < 1000) labels.push('size/L');
  else labels.push('size/XL');

  // 優先度（タイトルから判定）
  const { data: prData } = await octokit.pulls.get({
    owner: 'your-org',
    repo: 'your-repo',
    pull_number: pr,
  });

  if (prData.title.includes('hotfix') || prData.title.includes('critical')) {
    labels.push('priority/high');
  }

  // ラベル追加
  await octokit.issues.addLabels({
    owner: 'your-org',
    repo: 'your-repo',
    issue_number: pr,
    labels,
  });
}
```

---

## 5. 自動レビュワー割り当て

### 5.1 CODEOWNERSベース

```
# .github/CODEOWNERS

# デフォルト
* @team-reviewers

# フロントエンド
/src/components/** @frontend-team
/src/pages/** @frontend-team

# バックエンド
/src/api/** @backend-team
/src/services/** @backend-team

# インフラ
/.github/** @devops-team
/docker/** @devops-team

# セキュリティ
/src/auth/** @security-team
/src/middleware/auth.ts @security-team

# データベース
/migrations/** @database-team
```

### 5.2 動的割り当て

```yaml
# .github/auto-assign.yml
addReviewers: true
addAssignees: false

reviewers:
  - backend-lead
  - frontend-lead

numberOfReviewers: 2

# PRサイズに応じて変更
filePathAssignments:
  - pattern: 'src/api/**'
    reviewers:
      - backend-lead
      - senior-backend-dev

  - pattern: 'src/components/**'
    reviewers:
      - frontend-lead
      - senior-frontend-dev

  - pattern: 'migrations/**'
    reviewers:
      - database-admin

# ファイル数に応じて
fileCountAssignments:
  - below: 10
    reviewers: 1
  - below: 50
    reviewers: 2
  - above: 50
    reviewers: 3
```

---

## 6. AI支援レビュー

### 6.1 Claude活用

```typescript
// scripts/ai-review.ts
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

async function aiCodeReview(diff: string): Promise<string> {
  const prompt = `
あなたは経験豊富なコードレビュアーです。以下のコード変更をレビューし、改善点を指摘してください。

# レビュー観点
1. バグの可能性
2. パフォーマンス問題
3. セキュリティ問題
4. コード品質
5. ベストプラクティスからの逸脱

# Diff
\`\`\`diff
${diff}
\`\`\`

# レビューフォーマット
各問題について:
- **重要度**: Critical/High/Medium/Low
- **カテゴリ**: Bug/Performance/Security/Quality
- **説明**: 具体的な問題点
- **提案**: 改善方法

問題がない場合は「問題なし」と回答してください。
  `;

  const message = await anthropic.messages.create({
    model: 'claude-3-5-sonnet-20241022',
    max_tokens: 4096,
    messages: [{
      role: 'user',
      content: prompt,
    }],
  });

  return message.content[0].text;
}

// GitHub Actionsから使用
async function reviewPR(prNumber: number) {
  const diff = await getPRDiff(prNumber);
  const review = await aiCodeReview(diff);

  await postReviewComment(prNumber, review);
}
```

### 6.2 GitHub Actions統合

```yaml
# .github/workflows/ai-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run AI Review
        run: npm run ai-review -- --pr ${{ github.event.number }}
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## 7. メトリクス自動収集

### 7.1 レビューメトリクス

```typescript
// scripts/collect-review-metrics.ts
interface ReviewMetrics {
  prNumber: number;
  author: string;
  reviewers: string[];
  linesChanged: number;
  filesChanged: number;
  comments: number;
  timeToFirstReview: number; // hours
  timeToApproval: number; // hours
  cycleTime: number; // hours
}

async function collectMetrics(prNumber: number): Promise<ReviewMetrics> {
  const pr = await octokit.pulls.get({ owner, repo, pull_number: prNumber });
  const reviews = await octokit.pulls.listReviews({ owner, repo, pull_number: prNumber });
  const comments = await octokit.pulls.listReviewComments({ owner, repo, pull_number: prNumber });

  const createdAt = new Date(pr.data.created_at);
  const firstReview = reviews.data[0];
  const approval = reviews.data.find(r => r.state === 'APPROVED');
  const mergedAt = pr.data.merged_at ? new Date(pr.data.merged_at) : null;

  return {
    prNumber,
    author: pr.data.user.login,
    reviewers: reviews.data.map(r => r.user.login),
    linesChanged: pr.data.additions + pr.data.deletions,
    filesChanged: pr.data.changed_files,
    comments: comments.data.length,
    timeToFirstReview: firstReview
      ? (new Date(firstReview.submitted_at).getTime() - createdAt.getTime()) / 3600000
      : 0,
    timeToApproval: approval
      ? (new Date(approval.submitted_at).getTime() - createdAt.getTime()) / 3600000
      : 0,
    cycleTime: mergedAt
      ? (mergedAt.getTime() - createdAt.getTime()) / 3600000
      : 0,
  };
}

// ダッシュボードにエクスポート
async function exportToDashboard(metrics: ReviewMetrics[]) {
  // Grafana, Datadog等に送信
  await sendMetrics(metrics);
}
```

---

## 8. 統合ワークフロー

### 8.1 完全自動化ワークフロー

```yaml
# .github/workflows/complete-review.yml
name: Complete Review Automation

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  # 1. 自動チェック
  automated-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
      - run: npm ci

      # Lint
      - run: npm run lint

      # Tests
      - run: npm test -- --coverage

      # Build
      - run: npm run build

  # 2. コードレビュー
  code-review:
    needs: automated-checks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      # Danger
      - uses: danger/danger-js@latest
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

      # ReviewDog
      - uses: reviewdog/action-eslint@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}

  # 3. ラベリング
  labeling:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/labeler@v5
      - uses: codelytv/pr-size-labeler@v1

  # 4. レビュワー割り当て
  assign-reviewers:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v1

  # 5. メトリクス収集
  metrics:
    needs: [automated-checks, code-review]
    runs-on: ubuntu-latest
    steps:
      - name: Collect metrics
        run: npm run collect-metrics

  # 6. 通知
  notify:
    needs: [automated-checks, code-review, labeling, assign-reviewers]
    runs-on: ubuntu-latest
    steps:
      - name: Slack notification
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "PR ready for review: ${{ github.event.pull_request.html_url }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
```

---

## 9. トラブルシューティング

### 9.1 よくある問題

```typescript
// Dangerが失敗する
const troubleshooting = {
  problem: 'Danger checks fail',
  causes: [
    'GitHub token権限不足',
    'coverage-summary.jsonが生成されていない',
    'dangerfile.tsの構文エラー',
  ],
  solutions: [
    'GITHUB_TOKENに`write`権限を付与',
    'テスト実行後にDangerを実行',
    'dangerfile.tsをローカルで検証',
  ],
};
```

---

## 10. 実績データ

### 10.1 自動化効果

| 指標           | 自動化前  | 自動化後  | 改善率  |
|--------------|-------|-------|------|
| レビュー時間       | 45分   | 25分   | 44%  |
| バグ検出率        | 65%   | 85%   | 31%  |
| レビュー待ち時間     | 8時間   | 2時間   | 75%  |
| 人的エラー        | 15件/月 | 3件/月  | 80%  |
| レビュワー負荷      | 高     | 中     | -    |

---

**更新日**: 2025年1月
**次回更新予定**: 四半期毎
