# 自動化ガイド

## 概要

コードレビューの効率化のための自動化ガイドです。CI/CD統合、自動レビューツール、GitHub Actions設定について解説します。

## 目次

1. [CI/CD統合](#cicd統合)
2. [自動レビューツール](#自動レビューツール)
3. [GitHub Actions設定](#github-actions設定)
4. [カスタムルール](#カスタムルール)
5. [メトリクス収集](#メトリクス収集)

---

## CI/CD統合

### 基本的なCI/CDパイプライン

```yaml
# .github/workflows/ci.yml
name: CI

on:
  pull_request:
    branches: [main, develop]
  push:
    branches: [main, develop]

jobs:
  build-and-test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run type-check

      - name: Test
        run: npm test -- --coverage

      - name: Build
        run: npm run build

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/coverage-final.json
```

### ステータスチェック必須化

```yaml
# リポジトリ設定 > Branches > Branch protection rules

✅ Require status checks to pass before merging
  ✅ Require branches to be up to date before merging
  ✅ Status checks that are required:
    - build-and-test
    - lint
    - type-check
    - test
    - security-scan

✅ Require pull request reviews before merging
  - Required approving reviews: 1
  - Dismiss stale pull request approvals when new commits are pushed

✅ Require conversation resolution before merging

✅ Do not allow bypassing the above settings
```

---

## 自動レビューツール

### 1. ESLint（JavaScript/TypeScript）

```javascript
// .eslintrc.js
module.exports = {
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
  ],
  rules: {
    // コード品質
    'no-console': 'error',
    'no-debugger': 'error',
    'no-alert': 'error',
    'no-var': 'error',
    'prefer-const': 'error',
    'no-unused-vars': 'error',

    // 複雑度
    'complexity': ['error', 10],
    'max-depth': ['error', 3],
    'max-lines-per-function': ['error', 50],
    'max-params': ['error', 3],

    // セキュリティ
    'no-eval': 'error',
    'no-implied-eval': 'error',
    'no-new-func': 'error',

    // TypeScript
    '@typescript-eslint/no-explicit-any': 'error',
    '@typescript-eslint/explicit-function-return-type': 'warn',
  },
};

// package.json
{
  "scripts": {
    "lint": "eslint 'src/**/*.{ts,tsx}'",
    "lint:fix": "eslint 'src/**/*.{ts,tsx}' --fix"
  }
}
```

### 2. Prettier（フォーマット）

```javascript
// .prettierrc
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 100,
  "arrowParens": "avoid"
}

// .github/workflows/format-check.yml
name: Format Check

on: [pull_request]

jobs:
  format:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run format:check

      - name: Comment on PR
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: '⚠️ Code formatting issues detected. Please run `npm run format` locally.'
            })
```

### 3. SonarQube（静的解析）

```yaml
# .github/workflows/sonarqube.yml
name: SonarQube Analysis

on:
  pull_request:
    branches: [main]

jobs:
  sonarqube:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: SonarQube Scan
        uses: sonarsource/sonarqube-scan-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          SONAR_HOST_URL: ${{ secrets.SONAR_HOST_URL }}

      - name: SonarQube Quality Gate
        uses: sonarsource/sonarqube-quality-gate-action@master
        env:
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

```properties
# sonar-project.properties
sonar.projectKey=my-project
sonar.organization=my-org

sonar.sources=src
sonar.tests=src
sonar.test.inclusions=**/*.test.ts,**/*.test.tsx

sonar.javascript.lcov.reportPaths=coverage/lcov.info
sonar.typescript.lcov.reportPaths=coverage/lcov.info

sonar.coverage.exclusions=**/*.test.ts,**/*.test.tsx

# 品質ゲート
sonar.qualitygate.wait=true
```

### 4. Danger.js（PR自動チェック）

```typescript
// dangerfile.ts
import { danger, warn, fail, message, markdown } from 'danger';

// PRサイズチェック
const bigPRThreshold = 500;
const changes = danger.github.pr.additions + danger.github.pr.deletions;

if (changes > bigPRThreshold) {
  warn(
    `⚠️ このPRは${changes}行の変更があります。レビューしやすいように分割を検討してください。`
  );
}

// PR説明チェック
if (!danger.github.pr.body || danger.github.pr.body.length < 10) {
  fail('❌ PR説明を追加してください');
}

// コミットメッセージチェック
const commits = danger.git.commits;
const conventionalCommitRegex = /^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?: .+/;

commits.forEach(commit => {
  if (!conventionalCommitRegex.test(commit.message)) {
    warn(
      `⚠️ コミットメッセージがConventional Commits形式ではありません: "${commit.message}"`
    );
  }
});

// デバッグコードチェック
const modifiedFiles = danger.git.modified_files;
const createdFiles = danger.git.created_files;
const allFiles = [...modifiedFiles, ...createdFiles];

allFiles.forEach(async file => {
  if (file.endsWith('.ts') || file.endsWith('.tsx')) {
    const content = await danger.github.utils.fileContents(file);

    if (content.includes('console.log')) {
      fail(`❌ console.log が見つかりました: ${file}`);
    }

    if (content.includes('debugger')) {
      fail(`❌ debugger が見つかりました: ${file}`);
    }

    if (content.includes('.only(')) {
      fail(`❌ .only() が見つかりました（テスト限定実行）: ${file}`);
    }
  }
});

// テストファイルチェック
const hasAppChanges = allFiles.some(
  f => f.startsWith('src/') && !f.includes('.test.')
);
const hasTestChanges = allFiles.some(f => f.includes('.test.'));

if (hasAppChanges && !hasTestChanges) {
  warn('⚠️ アプリケーションコードが変更されていますが、テストが追加されていません');
}

// カバレッジチェック
const coverage = require('./coverage/coverage-summary.json');
const totalCoverage = coverage.total.lines.pct;

if (totalCoverage < 80) {
  warn(`⚠️ カバレッジが80%未満です: ${totalCoverage.toFixed(2)}%`);
}

// Breaking Changesチェック
if (danger.github.pr.title.includes('BREAKING CHANGE')) {
  message('🚨 このPRには破壊的変更が含まれています');
}

// レビュワー割り当てチェック
if (!danger.github.pr.assignees || danger.github.pr.assignees.length === 0) {
  warn('⚠️ レビュワーがアサインされていません');
}

// 変更ファイル数のレポート
markdown(`
## 📊 変更サマリー

- 追加: ${danger.github.pr.additions} 行
- 削除: ${danger.github.pr.deletions} 行
- 変更ファイル数: ${allFiles.length}

### 変更されたファイル

${allFiles.map(f => `- \`${f}\``).join('\n')}
`);
```

```yaml
# .github/workflows/danger.yml
name: Danger

on: [pull_request]

jobs:
  danger:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm test -- --coverage
      - run: npx danger ci
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

---

## GitHub Actions設定

### 完全な自動レビューワークフロー

```yaml
# .github/workflows/pr-review.yml
name: PR Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  # 1. 基本チェック
  basic-checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: PR Size Check
        run: |
          ADDITIONS=$(jq -r '.pull_request.additions' $GITHUB_EVENT_PATH)
          DELETIONS=$(jq -r '.pull_request.deletions' $GITHUB_EVENT_PATH)
          TOTAL=$((ADDITIONS + DELETIONS))

          if [ $TOTAL -gt 500 ]; then
            echo "::warning::PR is too large ($TOTAL lines). Consider splitting it."
          fi

      - name: PR Description Check
        run: |
          BODY=$(jq -r '.pull_request.body' $GITHUB_EVENT_PATH)
          LENGTH=${#BODY}

          if [ $LENGTH -lt 50 ]; then
            echo "::error::PR description is too short. Please provide more context."
            exit 1
          fi

  # 2. Lint & Format
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
          cache: 'npm'

      - run: npm ci
      - run: npm run lint
      - run: npm run format:check

  # 3. Type Check
  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run type-check

  # 4. Test & Coverage
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm test -- --coverage

      - name: Coverage Check
        run: |
          COVERAGE=$(node -p "require('./coverage/coverage-summary.json').total.lines.pct")
          echo "Coverage: $COVERAGE%"

          if (( $(echo "$COVERAGE < 80" | bc -l) )); then
            echo "::warning::Coverage is below 80%: $COVERAGE%"
          fi

      - uses: codecov/codecov-action@v3

  # 5. Security Scan
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: npm audit
        run: npm audit --audit-level=moderate

      - name: Snyk Security Scan
        uses: snyk/actions/node@master
        env:
          SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

      - name: CodeQL Analysis
        uses: github/codeql-action/analyze@v2

  # 6. Build
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm run build

      - name: Check bundle size
        run: |
          SIZE=$(du -sk dist | cut -f1)
          echo "Bundle size: ${SIZE}KB"

          if [ $SIZE -gt 1000 ]; then
            echo "::warning::Bundle size is large: ${SIZE}KB"
          fi

  # 7. Danger.js
  danger:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - run: npm ci
      - run: npm test -- --coverage
      - run: npx danger ci
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # 8. Auto Label
  auto-label:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Label by file changes
        uses: actions/github-script@v6
        with:
          script: |
            const files = await github.rest.pulls.listFiles({
              owner: context.repo.owner,
              repo: context.repo.repo,
              pull_number: context.issue.number,
            });

            const labels = [];

            const hasTest = files.data.some(f => f.filename.includes('.test.'));
            const hasDocs = files.data.some(f => f.filename.includes('.md'));
            const hasConfig = files.data.some(f =>
              f.filename.includes('package.json') ||
              f.filename.includes('.yml')
            );

            if (hasTest) labels.push('test');
            if (hasDocs) labels.push('documentation');
            if (hasConfig) labels.push('configuration');

            if (labels.length > 0) {
              github.rest.issues.addLabels({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                labels,
              });
            }

  # 9. Auto Assign Reviewers
  auto-assign:
    runs-on: ubuntu-latest
    steps:
      - uses: kentaro-m/auto-assign-action@v1.2.1
        with:
          configuration-path: '.github/auto-assign.yml'
```

### 自動レビュワー割り当て

```yaml
# .github/auto-assign.yml
addReviewers: true
addAssignees: false

reviewers:
  - team-lead
  - senior-dev-1
  - senior-dev-2

numberOfReviewers: 2

skipKeywords:
  - wip
  - draft
```

---

## カスタムルール

### カスタムESLintルール

```javascript
// .eslint/rules/no-direct-db-access.js
module.exports = {
  meta: {
    type: 'problem',
    docs: {
      description: 'Disallow direct database access outside repositories',
    },
  },
  create(context) {
    return {
      ImportDeclaration(node) {
        const importPath = node.source.value;

        // Repositoryファイル以外でDBライブラリをimportしているか
        const filename = context.getFilename();
        const isRepository = filename.includes('repository');
        const isDBImport = ['pg', 'mysql2', 'mongodb'].includes(importPath);

        if (!isRepository && isDBImport) {
          context.report({
            node,
            message: 'Direct database access is only allowed in repositories',
          });
        }
      },
    };
  },
};

// .eslintrc.js
module.exports = {
  plugins: ['local'],
  rules: {
    'local/no-direct-db-access': 'error',
  },
};
```

### カスタムGit hooks

```bash
#!/bin/bash
# .git/hooks/pre-push

echo "Running pre-push checks..."

# 1. Tests
echo "Running tests..."
npm test
if [ $? -ne 0 ]; then
  echo "❌ Tests failed"
  exit 1
fi

# 2. Lint
echo "Running lint..."
npm run lint
if [ $? -ne 0 ]; then
  echo "❌ Lint failed"
  exit 1
fi

# 3. Type check
echo "Running type check..."
npm run type-check
if [ $? -ne 0 ]; then
  echo "❌ Type check failed"
  exit 1
fi

# 4. ブランチ名チェック
BRANCH=$(git rev-parse --abbrev-ref HEAD)
PATTERN="^(feature|bugfix|hotfix|refactor)\/[a-z0-9-]+$"

if ! echo "$BRANCH" | grep -qE "$PATTERN"; then
  echo "❌ Invalid branch name: $BRANCH"
  echo "Expected format: feature/xxx, bugfix/xxx, etc."
  exit 1
fi

echo "✅ All pre-push checks passed"
```

---

## メトリクス収集

### PR メトリクス

```yaml
# .github/workflows/pr-metrics.yml
name: PR Metrics

on:
  pull_request:
    types: [opened, closed]

jobs:
  metrics:
    runs-on: ubuntu-latest
    steps:
      - name: Collect PR Metrics
        uses: actions/github-script@v6
        with:
          script: |
            const pr = context.payload.pull_request;

            const metrics = {
              pr_number: pr.number,
              title: pr.title,
              author: pr.user.login,
              created_at: pr.created_at,
              merged_at: pr.merged_at,
              additions: pr.additions,
              deletions: pr.deletions,
              changed_files: pr.changed_files,
              comments: pr.comments,
              review_comments: pr.review_comments,
              commits: pr.commits,
            };

            // 時間計算
            if (pr.merged_at) {
              const created = new Date(pr.created_at);
              const merged = new Date(pr.merged_at);
              const hoursToMerge = (merged - created) / (1000 * 60 * 60);
              metrics.hours_to_merge = hoursToMerge.toFixed(2);
            }

            console.log('PR Metrics:', JSON.stringify(metrics, null, 2));

            // データベースやAnalyticsサービスに送信
            // await sendToAnalytics(metrics);
```

### レビューメトリクスダッシュボード

```markdown
## メトリクス項目

### 効率性
- 平均レビュー時間
- PR から merge までの時間
- 平均 Re-review 回数

### 品質
- 平均コメント数
- バグ発見率
- テストカバレッジ推移

### スケール
- 週あたりPR数
- 平均PRサイズ
- レビュワーあたりPR数

### エンゲージメント
- レビュー参加率
- コメント返信率
- 賞賛コメント率
```

---

## ツール統合例

### VS Code 拡張機能

```json
// .vscode/extensions.json
{
  "recommendations": [
    "dbaeumer.vscode-eslint",
    "esbenp.prettier-vscode",
    "ms-vscode.vscode-typescript-next",
    "streetsidesoftware.code-spell-checker",
    "eamodio.gitlens",
    "github.vscode-pull-request-github"
  ]
}

// .vscode/settings.json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "eslint.validate": [
    "javascript",
    "javascriptreact",
    "typescript",
    "typescriptreact"
  ]
}
```

---

## まとめ

自動化により、レビュワーは本質的な問題に集中できます。

### 重要ポイント

1. **CI/CDパイプライン構築**
2. **自動チェックツール導入**
3. **カスタムルール作成**
4. **メトリクス測定**
5. **継続的改善**

### 自動化の優先順位

```markdown
## Phase 1: 基本（すぐに導入）
- [ ] Linter (ESLint)
- [ ] Formatter (Prettier)
- [ ] テスト自動実行
- [ ] ビルド確認

## Phase 2: 品質（1ヶ月以内）
- [ ] カバレッジチェック
- [ ] 型チェック
- [ ] セキュリティスキャン
- [ ] Danger.js

## Phase 3: 最適化（3ヶ月以内）
- [ ] カスタムルール
- [ ] メトリクス収集
- [ ] 自動レビュワー割り当て
- [ ] ダッシュボード

## Phase 4: 高度化（6ヶ月以内）
- [ ] AI支援レビュー
- [ ] パフォーマンステスト自動化
- [ ] 自動リファクタリング提案
```

### 次のステップ

- [ベストプラクティス完全ガイド](best-practices-complete.md)
- [チェックリスト完全ガイド](checklist-complete.md)
- [レビュープロセス完全ガイド](review-process-complete.md)
