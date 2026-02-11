# 🚀 Node.js CLI 実装ガイド

> **目的**: Commander、Inquirer、chalk などを使った実践的な Node.js CLI ツール開発の手法を習得する

## 📚 目次

1. [プロジェクトセットアップ](#プロジェクトセットアップ)
2. [Commander（引数パース）](#commander引数パース)
3. [Inquirer（インタラクティブUI）](#inquirerインタラクティブui)
4. [出力とスタイリング](#出力とスタイリング)
5. [ファイル操作](#ファイル操作)
6. [実践例](#実践例)

---

## プロジェクトセットアップ

### 初期化

```bash
# プロジェクト作成
mkdir my-cli-tool
cd my-cli-tool

# package.json 作成
npm init -y

# TypeScript プロジェクト化
npm install -D typescript @types/node ts-node
npx tsc --init
```

**package.json**:
```json
{
  "name": "my-cli-tool",
  "version": "1.0.0",
  "description": "A sample CLI tool",
  "main": "dist/index.js",
  "bin": {
    "my-cli": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "ts-node src/index.ts",
    "start": "node dist/index.js"
  },
  "keywords": ["cli", "tool"],
  "author": "",
  "license": "MIT"
}
```

**tsconfig.json**:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### 依存関係インストール

```bash
# CLI 開発用ライブラリ
npm install commander inquirer chalk ora
npm install -D @types/inquirer

# ファイル操作
npm install fs-extra
npm install -D @types/fs-extra

# その他便利ライブラリ
npm install execa dotenv
```

### ディレクトリ構造

```
my-cli-tool/
├── src/
│   ├── index.ts           # エントリーポイント
│   ├── commands/          # コマンド定義
│   │   ├── create.ts
│   │   ├── list.ts
│   │   └── delete.ts
│   ├── utils/             # ユーティリティ
│   │   ├── logger.ts
│   │   ├── file.ts
│   │   └── prompt.ts
│   └── types/             # 型定義
│       └── index.ts
├── templates/             # テンプレートファイル
├── dist/                  # ビルド出力
├── package.json
├── tsconfig.json
└── README.md
```

---

## Commander（引数パース）

### 基本的な使い方

**src/index.ts**:
```typescript
#!/usr/bin/env node

import { Command } from 'commander'
import { version } from '../package.json'

const program = new Command()

program
  .name('my-cli')
  .description('A sample CLI tool')
  .version(version)

// グローバルオプション
program
  .option('-v, --verbose', 'Enable verbose logging')
  .option('--no-color', 'Disable color output')

program.parse()
```

### コマンド定義

**基本的なコマンド**:
```typescript
program
  .command('create <name>')
  .description('Create a new project')
  .option('-t, --template <template>', 'Template to use', 'default')
  .option('-d, --dir <directory>', 'Output directory', '.')
  .action((name, options) => {
    console.log(`Creating project: ${name}`)
    console.log(`Template: ${options.template}`)
    console.log(`Directory: ${options.dir}`)
  })
```

**別ファイルにコマンド分割**:

**src/commands/create.ts**:
```typescript
import { Command } from 'commander'
import chalk from 'chalk'

export function createCommand() {
  return new Command('create')
    .description('Create a new project')
    .argument('<name>', 'Project name')
    .option('-t, --template <template>', 'Template to use', 'default')
    .option('-d, --dir <directory>', 'Output directory', '.')
    .action(async (name, options) => {
      console.log(chalk.blue('Creating project...'))
      console.log(`Name: ${name}`)
      console.log(`Template: ${options.template}`)
      console.log(`Directory: ${options.dir}`)

      // 実装
      await createProject(name, options)

      console.log(chalk.green('✓ Project created successfully!'))
    })
}

async function createProject(name: string, options: any) {
  // プロジェクト作成ロジック
}
```

**src/index.ts**:
```typescript
#!/usr/bin/env node

import { Command } from 'commander'
import { createCommand } from './commands/create'
import { listCommand } from './commands/list'
import { deleteCommand } from './commands/delete'

const program = new Command()

program
  .name('my-cli')
  .description('A sample CLI tool')
  .version('1.0.0')

// コマンド登録
program.addCommand(createCommand())
program.addCommand(listCommand())
program.addCommand(deleteCommand())

program.parse()
```

### サブコマンド

```typescript
// サブコマンド: my-cli project create <name>
const projectCommand = new Command('project')
  .description('Manage projects')

projectCommand
  .command('create <name>')
  .description('Create a new project')
  .action((name) => {
    console.log(`Creating project: ${name}`)
  })

projectCommand
  .command('list')
  .description('List all projects')
  .action(() => {
    console.log('Listing projects...')
  })

program.addCommand(projectCommand)
```

### バリデーション

```typescript
program
  .command('create <name>')
  .option('-p, --port <port>', 'Port number')
  .action((name, options) => {
    // 名前のバリデーション
    if (!/^[a-z0-9-]+$/.test(name)) {
      console.error(chalk.red('Error: Project name must contain only lowercase letters, numbers, and hyphens'))
      process.exit(1)
    }

    // ポート番号のバリデーション
    if (options.port) {
      const port = parseInt(options.port)
      if (isNaN(port) || port < 1 || port > 65535) {
        console.error(chalk.red('Error: Port must be a number between 1 and 65535'))
        process.exit(1)
      }
    }

    // 処理実行
    createProject(name, options)
  })
```

---

## Inquirer（インタラクティブUI）

### プロンプトの種類

**input（テキスト入力）**:
```typescript
import inquirer from 'inquirer'

const answers = await inquirer.prompt([
  {
    type: 'input',
    name: 'projectName',
    message: 'Project name:',
    default: 'my-project',
    validate: (input) => {
      if (input.length === 0) {
        return 'Project name is required'
      }
      if (!/^[a-z0-9-]+$/.test(input)) {
        return 'Project name must contain only lowercase letters, numbers, and hyphens'
      }
      return true
    }
  }
])

console.log(answers.projectName)
```

**list（単一選択）**:
```typescript
const answers = await inquirer.prompt([
  {
    type: 'list',
    name: 'framework',
    message: 'Select a framework:',
    choices: [
      'React',
      'Vue',
      'Next.js',
      'Vite',
      new inquirer.Separator(),  // 区切り線
      'Other'
    ],
    default: 'React'
  }
])

console.log(answers.framework)
```

**checkbox（複数選択）**:
```typescript
const answers = await inquirer.prompt([
  {
    type: 'checkbox',
    name: 'features',
    message: 'Select features:',
    choices: [
      { name: 'ESLint', checked: true },
      { name: 'Prettier', checked: true },
      { name: 'Tailwind CSS', checked: false },
      { name: 'Vitest', checked: false }
    ]
  }
])

console.log(answers.features)  // ['ESLint', 'Prettier']
```

**confirm（Yes/No）**:
```typescript
const answers = await inquirer.prompt([
  {
    type: 'confirm',
    name: 'useTypeScript',
    message: 'Use TypeScript?',
    default: true
  }
])

console.log(answers.useTypeScript)  // true or false
```

**password（パスワード入力）**:
```typescript
const answers = await inquirer.prompt([
  {
    type: 'password',
    name: 'password',
    message: 'Enter password:',
    mask: '*',
    validate: (input) => {
      if (input.length < 8) {
        return 'Password must be at least 8 characters'
      }
      return true
    }
  }
])
```

### 条件分岐（when）

```typescript
const answers = await inquirer.prompt([
  {
    type: 'confirm',
    name: 'useDatabase',
    message: 'Use database?',
    default: false
  },
  {
    type: 'list',
    name: 'database',
    message: 'Select a database:',
    choices: ['PostgreSQL', 'MySQL', 'SQLite'],
    when: (answers) => answers.useDatabase  // useDatabase が true の時のみ表示
  }
])
```

### プロンプトの再利用

**src/utils/prompt.ts**:
```typescript
import inquirer from 'inquirer'

export async function promptProjectName(defaultName = 'my-project'): Promise<string> {
  const { name } = await inquirer.prompt([
    {
      type: 'input',
      name: 'name',
      message: 'Project name:',
      default: defaultName,
      validate: (input) => {
        if (input.length === 0) return 'Project name is required'
        if (!/^[a-z0-9-]+$/.test(input)) {
          return 'Project name must contain only lowercase letters, numbers, and hyphens'
        }
        return true
      }
    }
  ])
  return name
}

export async function promptTemplate(): Promise<string> {
  const { template } = await inquirer.prompt([
    {
      type: 'list',
      name: 'template',
      message: 'Select a template:',
      choices: ['React', 'Vue', 'Next.js', 'Vite']
    }
  ])
  return template
}

export async function confirmAction(message: string): Promise<boolean> {
  const { confirmed } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'confirmed',
      message,
      default: false
    }
  ])
  return confirmed
}
```

**使用例**:
```typescript
import { promptProjectName, promptTemplate, confirmAction } from './utils/prompt'

async function createProject() {
  const name = await promptProjectName()
  const template = await promptTemplate()

  const confirmed = await confirmAction(`Create project '${name}' with ${template}?`)
  if (!confirmed) {
    console.log('Cancelled')
    return
  }

  // プロジェクト作成
}
```

---

## 出力とスタイリング

### chalk（カラー出力）

```typescript
import chalk from 'chalk'

// 基本的な色
console.log(chalk.green('Success!'))
console.log(chalk.red('Error!'))
console.log(chalk.yellow('Warning'))
console.log(chalk.blue('Info'))

// スタイル
console.log(chalk.bold('Bold'))
console.log(chalk.italic('Italic'))
console.log(chalk.underline('Underlined'))

// 背景色
console.log(chalk.bgGreen.black(' SUCCESS '))
console.log(chalk.bgRed.white(' ERROR '))

// 組み合わせ
console.log(chalk.bold.green('✓ Success!'))
console.log(chalk.bold.red('✗ Error!'))

// RGB カラー
console.log(chalk.rgb(123, 45, 67).underline('Custom color'))
console.log(chalk.hex('#8b5cf6')('Purple'))
```

### ora（スピナー・ローディング）

```typescript
import ora from 'ora'

async function installDependencies() {
  const spinner = ora('Installing dependencies...').start()

  try {
    // 処理実行
    await new Promise(resolve => setTimeout(resolve, 3000))

    spinner.succeed('Dependencies installed!')
  } catch (error) {
    spinner.fail('Failed to install dependencies')
    throw error
  }
}

// 複数ステップ
async function build() {
  const spinner = ora()

  spinner.start('Compiling TypeScript...')
  await compile()
  spinner.succeed('TypeScript compiled')

  spinner.start('Bundling assets...')
  await bundle()
  spinner.succeed('Assets bundled')

  spinner.start('Optimizing...')
  await optimize()
  spinner.succeed('Optimized')

  console.log(chalk.green('\n✓ Build complete!'))
}
```

### ロガーの作成

**src/utils/logger.ts**:
```typescript
import chalk from 'chalk'

export class Logger {
  private verbose: boolean

  constructor(verbose = false) {
    this.verbose = verbose
  }

  success(message: string) {
    console.log(chalk.green(`✓ ${message}`))
  }

  error(message: string) {
    console.error(chalk.red(`✗ ${message}`))
  }

  warn(message: string) {
    console.warn(chalk.yellow(`⚠ ${message}`))
  }

  info(message: string) {
    console.log(chalk.blue(`ℹ ${message}`))
  }

  debug(message: string) {
    if (this.verbose) {
      console.log(chalk.gray(`[DEBUG] ${message}`))
    }
  }

  log(message: string) {
    console.log(message)
  }
}

// グローバルロガー
export const logger = new Logger()

// 使用例
export function setVerbose(verbose: boolean) {
  logger['verbose'] = verbose
}
```

**使用例**:
```typescript
import { logger, setVerbose } from './utils/logger'

program
  .option('-v, --verbose', 'Enable verbose logging')
  .hook('preAction', (thisCommand) => {
    const options = thisCommand.opts()
    if (options.verbose) {
      setVerbose(true)
    }
  })

// コマンド内で使用
logger.info('Creating project...')
logger.debug('Using template: react')
logger.success('Project created!')
```

---

## ファイル操作

### fs-extra

```bash
npm install fs-extra
npm install -D @types/fs-extra
```

**基本操作**:
```typescript
import fs from 'fs-extra'
import path from 'path'

// ディレクトリ作成
await fs.ensureDir('./my-project')

// ファイル書き込み
await fs.writeFile('./my-project/package.json', JSON.stringify({
  name: 'my-project',
  version: '1.0.0'
}, null, 2))

// ファイル読み込み
const content = await fs.readFile('./my-project/package.json', 'utf-8')

// ファイルコピー
await fs.copy('./templates/react', './my-project')

// ファイル存在確認
const exists = await fs.pathExists('./my-project')

// ディレクトリ削除
await fs.remove('./my-project')
```

### テンプレートファイルのコピー

**templates/react/package.json**:
```json
{
  "name": "{{projectName}}",
  "version": "1.0.0",
  "scripts": {
    "dev": "vite",
    "build": "vite build"
  }
}
```

**src/utils/template.ts**:
```typescript
import fs from 'fs-extra'
import path from 'path'

export async function copyTemplate(
  templateName: string,
  destination: string,
  variables: Record<string, string>
) {
  const templateDir = path.join(__dirname, '../../templates', templateName)

  // テンプレートディレクトリ確認
  const exists = await fs.pathExists(templateDir)
  if (!exists) {
    throw new Error(`Template '${templateName}' not found`)
  }

  // コピー
  await fs.copy(templateDir, destination)

  // 変数置換
  await replaceVariables(destination, variables)
}

async function replaceVariables(
  dir: string,
  variables: Record<string, string>
) {
  const files = await fs.readdir(dir)

  for (const file of files) {
    const filePath = path.join(dir, file)
    const stat = await fs.stat(filePath)

    if (stat.isDirectory()) {
      await replaceVariables(filePath, variables)
    } else {
      let content = await fs.readFile(filePath, 'utf-8')

      // {{variable}} を置換
      for (const [key, value] of Object.entries(variables)) {
        const regex = new RegExp(`{{${key}}}`, 'g')
        content = content.replace(regex, value)
      }

      await fs.writeFile(filePath, content)
    }
  }
}
```

**使用例**:
```typescript
import { copyTemplate } from './utils/template'

await copyTemplate('react', './my-project', {
  projectName: 'my-project',
  author: 'John Doe'
})
```

### package.json の操作

```typescript
import fs from 'fs-extra'
import path from 'path'

export async function updatePackageJson(
  projectDir: string,
  updates: Record<string, any>
) {
  const pkgPath = path.join(projectDir, 'package.json')
  const pkg = await fs.readJSON(pkgPath)

  // マージ
  const newPkg = {
    ...pkg,
    ...updates
  }

  // 書き込み
  await fs.writeJSON(pkgPath, newPkg, { spaces: 2 })
}

// 使用例
await updatePackageJson('./my-project', {
  description: 'My awesome project',
  dependencies: {
    react: '^18.2.0',
    'react-dom': '^18.2.0'
  }
})
```

---

## 実践例

### プロジェクトジェネレーター

**src/commands/create.ts**:
```typescript
import { Command } from 'commander'
import inquirer from 'inquirer'
import chalk from 'chalk'
import ora from 'ora'
import fs from 'fs-extra'
import path from 'path'
import { execa } from 'execa'
import { copyTemplate } from '../utils/template'
import { logger } from '../utils/logger'

interface CreateOptions {
  template?: string
  skipInstall?: boolean
}

export function createCommand() {
  return new Command('create')
    .description('Create a new project')
    .argument('[name]', 'Project name')
    .option('-t, --template <template>', 'Template to use')
    .option('--skip-install', 'Skip npm install')
    .action(async (name, options: CreateOptions) => {
      try {
        await createProject(name, options)
      } catch (error) {
        logger.error('Failed to create project')
        console.error(error)
        process.exit(1)
      }
    })
}

async function createProject(name: string | undefined, options: CreateOptions) {
  // プロジェクト名の取得
  let projectName = name
  if (!projectName) {
    const { name } = await inquirer.prompt([
      {
        type: 'input',
        name: 'name',
        message: 'Project name:',
        default: 'my-project',
        validate: (input) => {
          if (!/^[a-z0-9-]+$/.test(input)) {
            return 'Project name must contain only lowercase letters, numbers, and hyphens'
          }
          return true
        }
      }
    ])
    projectName = name
  }

  // テンプレート選択
  let template = options.template
  if (!template) {
    const { selectedTemplate } = await inquirer.prompt([
      {
        type: 'list',
        name: 'selectedTemplate',
        message: 'Select a template:',
        choices: [
          { name: 'React', value: 'react' },
          { name: 'Vue', value: 'vue' },
          { name: 'Next.js', value: 'nextjs' },
          { name: 'Vite', value: 'vite' }
        ]
      }
    ])
    template = selectedTemplate
  }

  // 追加設定
  const config = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'useTypeScript',
      message: 'Use TypeScript?',
      default: true
    },
    {
      type: 'checkbox',
      name: 'features',
      message: 'Select features:',
      choices: [
        { name: 'ESLint', value: 'eslint', checked: true },
        { name: 'Prettier', value: 'prettier', checked: true },
        { name: 'Tailwind CSS', value: 'tailwind', checked: false },
        { name: 'Vitest', value: 'vitest', checked: false }
      ]
    }
  ])

  // 確認
  console.log(chalk.cyan('\nProject configuration:'))
  console.log(`  Name: ${projectName}`)
  console.log(`  Template: ${template}`)
  console.log(`  TypeScript: ${config.useTypeScript ? 'Yes' : 'No'}`)
  console.log(`  Features: ${config.features.join(', ') || 'None'}`)

  const { confirmed } = await inquirer.prompt([
    {
      type: 'confirm',
      name: 'confirmed',
      message: 'Create project?',
      default: true
    }
  ])

  if (!confirmed) {
    logger.warn('Cancelled')
    return
  }

  // プロジェクト作成
  const projectDir = path.join(process.cwd(), projectName)

  // ディレクトリ存在確認
  const exists = await fs.pathExists(projectDir)
  if (exists) {
    logger.error(`Directory '${projectName}' already exists`)
    process.exit(1)
  }

  // テンプレートコピー
  const spinner = ora('Creating project...').start()

  try {
    const templateName = config.useTypeScript ? `${template}-ts` : template
    await copyTemplate(templateName, projectDir, {
      projectName,
      features: config.features.join(',')
    })

    spinner.succeed('Project created')

    // 依存関係インストール
    if (!options.skipInstall) {
      spinner.start('Installing dependencies...')
      await execa('npm', ['install'], { cwd: projectDir })
      spinner.succeed('Dependencies installed')
    }

    // 完了メッセージ
    console.log(chalk.green('\n✓ Project created successfully!\n'))
    console.log(chalk.cyan('Next steps:'))
    console.log(`  cd ${projectName}`)
    if (options.skipInstall) {
      console.log('  npm install')
    }
    console.log('  npm run dev')

  } catch (error) {
    spinner.fail('Failed to create project')
    throw error
  }
}
```

### データ処理 CLI

**src/commands/process.ts**:
```typescript
import { Command } from 'commander'
import fs from 'fs-extra'
import Papa from 'papaparse'
import chalk from 'chalk'
import { logger } from '../utils/logger'

interface ProcessOptions {
  filter?: string
  output?: string
  format?: 'csv' | 'json'
}

export function processCommand() {
  return new Command('process')
    .description('Process CSV file')
    .argument('<input>', 'Input CSV file')
    .option('-f, --filter <filter>', 'Filter expression (e.g., "age > 20")')
    .option('-o, --output <output>', 'Output file')
    .option('--format <format>', 'Output format (csv, json)', 'csv')
    .action(async (input, options: ProcessOptions) => {
      try {
        await processFile(input, options)
      } catch (error) {
        logger.error('Failed to process file')
        console.error(error)
        process.exit(1)
      }
    })
}

async function processFile(inputPath: string, options: ProcessOptions) {
  // ファイル読み込み
  const exists = await fs.pathExists(inputPath)
  if (!exists) {
    logger.error(`File not found: ${inputPath}`)
    process.exit(1)
  }

  const content = await fs.readFile(inputPath, 'utf-8')
  const { data } = Papa.parse(content, { header: true })

  logger.info(`Loaded ${data.length} rows`)

  // フィルタリング
  let filtered = data
  if (options.filter) {
    // 簡易的なフィルタ実装
    filtered = data.filter((row: any) => {
      // 例: "age > 20"
      const [field, op, value] = options.filter!.split(' ')
      const rowValue = row[field]

      switch (op) {
        case '>': return parseFloat(rowValue) > parseFloat(value)
        case '<': return parseFloat(rowValue) < parseFloat(value)
        case '==': return rowValue === value
        default: return true
      }
    })

    logger.info(`Filtered to ${filtered.length} rows`)
  }

  // 出力
  let output: string
  if (options.format === 'json') {
    output = JSON.stringify(filtered, null, 2)
  } else {
    output = Papa.unparse(filtered)
  }

  if (options.output) {
    await fs.writeFile(options.output, output)
    logger.success(`Saved to ${options.output}`)
  } else {
    console.log(output)
  }
}
```

---

## まとめ

### Node.js CLI 開発チェックリスト

**プロジェクトセットアップ**:
- [ ] TypeScript 設定
- [ ] package.json の bin フィールド
- [ ] Shebang (`#!/usr/bin/env node`)

**Commander**:
- [ ] コマンド定義
- [ ] オプション定義
- [ ] バリデーション

**Inquirer**:
- [ ] インタラクティブプロンプト
- [ ] 条件分岐（when）
- [ ] バリデーション

**出力**:
- [ ] chalk でカラー出力
- [ ] ora でスピナー
- [ ] ロガー実装

**ファイル操作**:
- [ ] fs-extra でファイル操作
- [ ] テンプレートコピー
- [ ] 変数置換

---

## 次のステップ

1. **03-distribution.md**: CLI 配布・パッケージングガイド

---

*使いやすい CLI ツールで開発を効率化しましょう。*
