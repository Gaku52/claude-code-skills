# Phase 2 クイックスタートガイド

## 🚀 明日の朝、最初にやること

### 1. 環境確認（5分）

```bash
# Node.jsバージョン確認（18以上必須）
node --version

# npmバージョン確認
npm --version

# GitHubトークン確認
echo $GITHUB_TOKEN
# なければ作成: https://github.com/settings/tokens
```

### 2. プロジェクト作成（10分）

```bash
# Agentsプロジェクト作成
cd ~/.claude
mkdir -p agents/lib agents/code-reviewer
cd agents

# package.json作成
cat > package.json << 'JSON'
{
  "name": "claude-code-agents",
  "version": "1.0.0",
  "description": "Automated agents powered by Claude Code Skills",
  "type": "module",
  "scripts": {
    "dev": "tsx",
    "test": "vitest",
    "build": "tsc",
    "lint": "eslint .",
    "format": "prettier --write ."
  },
  "keywords": ["claude", "agents", "automation"],
  "author": "Gaku",
  "license": "MIT"
}
JSON

# TypeScript設定
cat > tsconfig.json << 'JSON'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022"],
    "moduleResolution": "node",
    "esModuleInterop": true,
    "resolveJsonModule": true,
    "strict": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": ".",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["**/*.ts"],
  "exclude": ["node_modules", "dist"]
}
JSON

# .gitignore作成
cat > .gitignore << 'IGNORE'
node_modules/
dist/
.env
*.log
.DS_Store
IGNORE

# 依存関係インストール
npm install @octokit/rest commander chalk dotenv gray-matter
npm install -D typescript @types/node tsx vitest prettier eslint
```

### 3. 共通ライブラリ作成（30分）

**lib/skill-loader.ts をコピペ:**
- PHASE2_DESIGN.mdの「Skill Loader」セクションをコピー
- lib/skill-loader.ts に貼り付け

**lib/types.ts を作成:**
```typescript
// lib/types.ts
export interface SkillMetadata {
  name: string
  description: string
}

export interface Skill {
  metadata: SkillMetadata
  content: string
}

export interface ReviewComment {
  path: string
  line: number
  body: string
  severity: 'error' | 'warning' | 'info'
}
```

**lib/logger.ts を作成:**
```typescript
// lib/logger.ts
import chalk from 'chalk'

export const logger = {
  info: (message: string) => console.log(chalk.blue('ℹ'), message),
  success: (message: string) => console.log(chalk.green('✓'), message),
  error: (message: string) => console.error(chalk.red('✗'), message),
  warn: (message: string) => console.warn(chalk.yellow('⚠'), message)
}
```

### 4. テスト実行（10分）

**lib/skill-loader.test.ts を作成:**
```typescript
// lib/skill-loader.test.ts
import { describe, it, expect } from 'vitest'
import { loadSkill, extractSection } from './skill-loader'

describe('Skill Loader', () => {
  it('should load a skill', async () => {
    const skill = await loadSkill('code-review')
    expect(skill).toBeDefined()
    expect(skill.metadata.name).toBe('code-review')
  })
  
  it('should extract section', () => {
    const content = '## Test\nContent here\n## Next'
    const section = extractSection(content, 'Test')
    expect(section).toBe('Content here')
  })
})
```

**実行:**
```bash
npm run test
```

### 5. 動作確認スクリプト（5分）

```bash
# test.ts作成
cat > test.ts << 'TS'
import { loadSkill } from './lib/skill-loader.js'

async function test() {
  console.log('Loading code-review skill...')
  const skill = await loadSkill('code-review')
  console.log('✅ Skill loaded:', skill.metadata.name)
  console.log('Description:', skill.metadata.description)
}

test()
TS

# 実行
tsx test.ts
```

## ✅ チェックリスト

明日の午前中にここまで完了すれば完璧：

- [ ] Node.js/npm確認
- [ ] GitHubトークン設定
- [ ] プロジェクト初期化
- [ ] package.json作成
- [ ] tsconfig.json作成
- [ ] 依存関係インストール
- [ ] lib/skill-loader.ts作成
- [ ] lib/types.ts作成
- [ ] lib/logger.ts作成
- [ ] テスト作成・実行
- [ ] skill-loader動作確認

## 🎯 午後の目標

午前中に基盤ができたら、午後はCode Reviewer Agent本体を実装：

- [ ] code-reviewer/index.ts作成
- [ ] code-reviewer/reviewer.ts作成
- [ ] GitHub API連携
- [ ] 実際のPRでテスト

## 💡 つまづきそうなポイント

### 問題1: GITHUB_TOKEN未設定
```bash
# GitHub Personal Access Tokenを作成
# https://github.com/settings/tokens
# scope: repo (全て)

# .envファイルに設定
echo "GITHUB_TOKEN=ghp_xxxxx" > .env
```

### 問題2: ESM vs CommonJS
```bash
# package.jsonに "type": "module" を追加済み
# import/export使える
```

### 問題3: TypeScriptエラー
```bash
# tsxで実行（型チェックスキップ）
tsx test.ts

# ビルドして実行
npm run build
node dist/test.js
```

## 📚 便利コマンド

```bash
# 開発サーバー（ファイル監視）
tsx watch test.ts

# Prettier実行
npm run format

# Lint実行
npm run lint

# ビルド
npm run build
```

---

**この通りに進めれば、午前中で基盤完成です！** 🚀
**午後にはCode Reviewerが動き始めます！** 🎉
