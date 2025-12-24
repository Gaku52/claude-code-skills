# Phase 2: Sub Agents 詳細設計

## 🎯 明日のゴール
最初のAgent（code-reviewer-agent）の基盤を構築

## 📁 ディレクトリ構造（完成形）

```
~/.claude/
├── skills/                           ← Phase 1 ✅
│   ├── code-review/SKILL.md
│   ├── testing-strategy/SKILL.md
│   └── ...（26個）
│
├── agents/                           ← Phase 2 🚀
│   ├── package.json                  ← Agentsプロジェクトのルート
│   ├── tsconfig.json
│   ├── .gitignore
│   ├── README.md
│   │
│   ├── lib/                          ← 共通ライブラリ
│   │   ├── skill-loader.ts          ← SKILL.mdを読み込む
│   │   ├── github-api.ts            ← GitHub API wrapper
│   │   ├── logger.ts                ← ログ出力
│   │   └── types.ts                 ← 共通型定義
│   │
│   ├── code-reviewer/               ← 最初のAgent
│   │   ├── index.ts                 ← エントリーポイント
│   │   ├── reviewer.ts              ← レビューロジック
│   │   ├── config.ts                ← 設定管理
│   │   ├── types.ts                 ← 型定義
│   │   ├── README.md                ← 使い方
│   │   └── __tests__/               ← テスト
│   │       └── reviewer.test.ts
│   │
│   ├── test-runner/                 ← 2番目のAgent（後で）
│   │   └── ...
│   │
│   └── deployment/                  ← 3番目のAgent（後で）
│       └── ...
```

## 🔧 技術スタック

### 言語・ランタイム
- **TypeScript** 5.3+
- **Node.js** 18+

### 必須ライブラリ
```json
{
  "dependencies": {
    "@octokit/rest": "^20.0.0",     // GitHub API
    "commander": "^11.0.0",          // CLI
    "chalk": "^5.3.0",               // ターミナル装飾
    "dotenv": "^16.3.0",             // 環境変数
    "yaml": "^2.3.0",                // YAML parser（SKILL.md front matter用）
    "gray-matter": "^4.0.3"          // Markdown front matter parser
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "typescript": "^5.3.0",
    "ts-node": "^10.9.0",
    "tsx": "^4.7.0",                 // TypeScript実行（高速）
    "vitest": "^1.0.0",              // テスト
    "@typescript-eslint/eslint-plugin": "^6.0.0",
    "@typescript-eslint/parser": "^6.0.0",
    "eslint": "^8.0.0",
    "prettier": "^3.0.0"
  }
}
```

## 🏗️ 実装設計

### 1. Skill Loader（共通ライブラリ）

```typescript
// lib/skill-loader.ts

import fs from 'fs/promises'
import path from 'path'
import matter from 'gray-matter'

interface SkillMetadata {
  name: string
  description: string
}

interface Skill {
  metadata: SkillMetadata
  content: string
}

/**
 * SKILL.mdを読み込んでパースする
 */
export async function loadSkill(skillName: string): Promise<Skill> {
  const skillPath = path.join(
    process.env.HOME || '',
    '.claude',
    'skills',
    skillName,
    'SKILL.md'
  )
  
  const fileContent = await fs.readFile(skillPath, 'utf-8')
  const { data, content } = matter(fileContent)
  
  return {
    metadata: data as SkillMetadata,
    content
  }
}

/**
 * SKILL.mdから特定のセクションを抽出
 */
export function extractSection(content: string, sectionTitle: string): string {
  const regex = new RegExp(
    `## ${sectionTitle}\\s+([\\s\\S]*?)(?=##|$)`,
    'i'
  )
  const match = content.match(regex)
  return match ? match[1].trim() : ''
}

/**
 * SKILL.mdからチェックリストを抽出
 */
export function extractChecklist(content: string): string[] {
  const checklistSection = extractSection(content, 'チェックリスト')
  const items = checklistSection.match(/- \[([ x])\] (.+)/g) || []
  return items.map(item => item.replace(/- \[[ x]\] /, ''))
}
```

### 2. Code Reviewer Agent

```typescript
// code-reviewer/index.ts

import { Command } from 'commander'
import chalk from 'chalk'
import { reviewPullRequest } from './reviewer'

const program = new Command()

program
  .name('code-reviewer')
  .description('Automated code review agent using SKILL.md knowledge')
  .version('1.0.0')

program
  .command('review')
  .description('Review a pull request')
  .requiredOption('-p, --pr <number>', 'Pull request number')
  .option('-r, --repo <name>', 'Repository name (owner/repo)')
  .action(async (options) => {
    console.log(chalk.blue('🔍 Starting code review...'))
    
    try {
      const result = await reviewPullRequest({
        prNumber: parseInt(options.pr),
        repo: options.repo
      })
      
      console.log(chalk.green('✅ Review completed!'))
      console.log(result)
    } catch (error) {
      console.error(chalk.red('❌ Review failed:'), error)
      process.exit(1)
    }
  })

program.parse()
```

```typescript
// code-reviewer/reviewer.ts

import { Octokit } from '@octokit/rest'
import { loadSkill, extractChecklist } from '../lib/skill-loader'

interface ReviewOptions {
  prNumber: number
  repo: string
}

interface ReviewComment {
  path: string
  line: number
  body: string
  severity: 'error' | 'warning' | 'info'
}

/**
 * PRをレビューする
 */
export async function reviewPullRequest(options: ReviewOptions): Promise<void> {
  // 1. SKILL.mdから知識を読み込む
  const codeReviewSkill = await loadSkill('code-review')
  const checklist = extractChecklist(codeReviewSkill.content)
  
  console.log('📋 Review checklist:', checklist)
  
  // 2. GitHub APIでPRの情報を取得
  const octokit = new Octokit({
    auth: process.env.GITHUB_TOKEN
  })
  
  const [owner, repoName] = options.repo.split('/')
  
  const { data: pr } = await octokit.pulls.get({
    owner,
    repo: repoName,
    pull_number: options.prNumber
  })
  
  // 3. PRのファイルを取得
  const { data: files } = await octokit.pulls.listFiles({
    owner,
    repo: repoName,
    pull_number: options.prNumber
  })
  
  // 4. 各ファイルをチェック
  const comments: ReviewComment[] = []
  
  for (const file of files) {
    // ファイルタイプに応じて適切なSkillを読み込む
    let additionalSkill = null
    
    if (file.filename.endsWith('.ts') || file.filename.endsWith('.tsx')) {
      additionalSkill = await loadSkill('react-development')
    } else if (file.filename.endsWith('.swift')) {
      additionalSkill = await loadSkill('ios-development')
    }
    
    // ファイルを解析してコメント生成
    const fileComments = await analyzeFile(file, checklist, additionalSkill)
    comments.push(...fileComments)
  }
  
  // 5. PRにコメントを投稿
  if (comments.length > 0) {
    await postReviewComments(octokit, options, comments)
  }
}

async function analyzeFile(
  file: any,
  checklist: string[],
  additionalSkill: any
): Promise<ReviewComment[]> {
  const comments: ReviewComment[] = []
  
  // TODO: 実際の解析ロジック
  // - 命名規約チェック
  // - テストの有無チェック
  // - コード品質チェック
  // - セキュリティチェック
  
  return comments
}

async function postReviewComments(
  octokit: Octokit,
  options: ReviewOptions,
  comments: ReviewComment[]
): Promise<void> {
  // TODO: GitHub APIでコメント投稿
}
```

### 3. GitHub API Wrapper

```typescript
// lib/github-api.ts

import { Octokit } from '@octokit/rest'

export class GitHubAPI {
  private octokit: Octokit
  
  constructor(token?: string) {
    this.octokit = new Octokit({
      auth: token || process.env.GITHUB_TOKEN
    })
  }
  
  async getPullRequest(owner: string, repo: string, prNumber: number) {
    const { data } = await this.octokit.pulls.get({
      owner,
      repo,
      pull_number: prNumber
    })
    return data
  }
  
  async getPullRequestFiles(owner: string, repo: string, prNumber: number) {
    const { data } = await this.octokit.pulls.listFiles({
      owner,
      repo,
      pull_number: prNumber
    })
    return data
  }
  
  async createReviewComment(
    owner: string,
    repo: string,
    prNumber: number,
    comment: {
      body: string
      path: string
      line: number
    }
  ) {
    await this.octokit.pulls.createReviewComment({
      owner,
      repo,
      pull_number: prNumber,
      ...comment
    })
  }
}
```

## 📝 明日の実装手順

### Step 1: プロジェクト初期化（30分）
```bash
cd ~/.claude
mkdir agents
cd agents

# package.json作成
npm init -y

# TypeScript設定
npm install -D typescript @types/node ts-node tsx
npx tsc --init

# 必須パッケージインストール
npm install @octokit/rest commander chalk dotenv gray-matter
npm install -D vitest @typescript-eslint/eslint-plugin prettier
```

### Step 2: 共通ライブラリ実装（1時間）
- lib/skill-loader.ts
- lib/types.ts
- lib/logger.ts

### Step 3: Code Reviewer Agent実装（2時間）
- code-reviewer/index.ts
- code-reviewer/reviewer.ts
- code-reviewer/types.ts

### Step 4: テスト・動作確認（30分）
```bash
# テスト実行
npm run test

# 実際に動かす
tsx code-reviewer/index.ts review --pr 123 --repo owner/repo
```

### Step 5: ドキュメント作成（30分）
- README.md
- code-reviewer/README.md

## 🎯 成功基準

明日終わりに以下ができていればOK：

✅ Agentsプロジェクトの基盤構築
✅ skill-loader動作確認（SKILL.mdを読み込める）
✅ code-reviewer-agentの基本実装
✅ GitHub API連携確認
✅ 実際のPRでレビュー実行できる

## 💡 Tips

### TypeScript学習ポイント
- インターフェース定義
- async/await
- Promise
- ジェネリクス（後で）
- 型ガード

### 開発効率化
- Cursorで開発
- Claude Codeでレビュー
- GitHub Copilotも併用可

### トラブルシューティング準備
- GITHUB_TOKEN環境変数設定
- ~/.claude/skillsへのアクセス権限
- Node.jsバージョン確認

## 📚 参考資料

- Octokit Documentation: https://octokit.github.io/rest.js/
- Commander.js: https://github.com/tj/commander.js
- TypeScript Handbook: https://www.typescriptlang.org/docs/

---

**準備完了！明日が楽しみです！** 🚀
