# Monorepo構成（推奨案）

## ディレクトリ構造

```
~/.claude/skills/               ← Gitリポジトリルート
├── .git/
├── .gitignore
├── README.md                   ← 全体の説明
├── CONTRIBUTING.md
├── LICENSE
│
├── skills/                     ← Phase 1: Skills
│   ├── README.md              ← Skills専用README
│   ├── ios-development/
│   ├── react-development/
│   └── ...（26個）
│
├── agents/                     ← Phase 2: Agents
│   ├── README.md              ← Agents専用README
│   ├── package.json           ← Agents用
│   ├── tsconfig.json
│   ├── .gitignore
│   ├── lib/
│   │   ├── skill-loader.ts
│   │   ├── github-api.ts
│   │   └── types.ts
│   ├── code-reviewer/
│   │   ├── index.ts
│   │   └── ...
│   └── test-runner/
│       └── ...
│
└── docs/                       ← ドキュメント（将来）
    ├── getting-started.md
    ├── skills-guide.md
    └── agents-guide.md
```

## package.json構成

### ルートのpackage.json（オプション）
```json
{
  "name": "claude-code-framework",
  "version": "1.0.0",
  "private": true,
  "workspaces": [
    "agents"
  ],
  "scripts": {
    "install:agents": "cd agents && npm install",
    "test": "cd agents && npm test",
    "build": "cd agents && npm run build"
  }
}
```

### agents/package.json
```json
{
  "name": "@your-name/claude-agents",
  "version": "1.0.0",
  "description": "Automated agents powered by Claude Code Skills",
  "main": "dist/index.js",
  "types": "dist/index.d.ts",
  "scripts": {
    "dev": "tsx",
    "test": "vitest",
    "build": "tsc"
  },
  "dependencies": {
    "@octokit/rest": "^20.0.0",
    "commander": "^11.0.0",
    "chalk": "^5.3.0"
  }
}
```

## .gitignore（ルート）

```gitignore
# Node.js
agents/node_modules/
agents/dist/
agents/.env

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
npm-debug.log*

# Temporary
.tmp/
temp/
```

## Skills ⇔ Agents 連携

**Agentsから相対パスでSkillsを読み込む:**

```typescript
// agents/lib/skill-loader.ts

import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export async function loadSkill(skillName: string) {
  // agents/ から ../skills/ を参照
  const skillPath = path.join(
    __dirname,
    '..',      // agents/lib から agents/ へ
    '..',      // agents/ から ルート へ
    'skills',  // skills/
    skillName,
    'SKILL.md'
  )
  
  // 読み込み処理...
}
```

## Git運用

### コミット例
```bash
# Skills更新
git commit -m "feat(skills): add ios-security best practices"

# Agents更新
git commit -m "feat(agents): implement code-reviewer-agent"

# 両方更新
git commit -m "feat: integrate code-review skill with reviewer agent"
```

### ブランチ戦略
```
main                    ← 安定版
├── develop            ← 開発版
│   ├── feature/skills-xxx
│   └── feature/agents-xxx
```

## npmパッケージ公開（将来）

**agents/を独立したパッケージとして公開:**

```bash
cd agents
npm publish --access public
```

**ユーザーは:**
```bash
# Agentsだけインストール
npm install -g @your-name/claude-agents

# Skillsは別途git clone
git clone https://github.com/your-name/claude-code-skills.git ~/.claude/skills
```

---

**メリット:**
- 今は1つのリポジトリで楽に管理
- 将来、Agentsだけnpmパッケージ化できる
- Skillsは引き続きGitで直接配布
