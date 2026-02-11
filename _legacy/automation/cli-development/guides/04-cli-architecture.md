# 🏗️ CLI Architecture & Design Patterns Guide

> **目的**: CLIアプリケーションの堅牢なアーキテクチャ設計、引数パース戦略、設定管理、出力フォーマッティング、エラーハンドリング、テスト手法を習得する

## 📚 目次

1. [CLIアーキテクチャパターン](#cliアーキテクチャパターン)
2. [引数パース戦略](#引数パース戦略)
3. [設定管理システム](#設定管理システム)
4. [出力フォーマッティング](#出力フォーマッティング)
5. [エラーハンドリング](#エラーハンドリング)
6. [CLIテスト戦略](#cliテスト戦略)
7. [インタラクティブプロンプト](#インタラクティブプロンプト)
8. [実践的なアーキテクチャ例](#実践的なアーキテクチャ例)

---

## CLIアーキテクチャパターン

### レイヤードアーキテクチャ

**構造**:
```
cli-tool/
├── src/
│   ├── cli/                  # CLI レイヤー（UI）
│   │   ├── index.ts         # エントリーポイント
│   │   ├── commands/        # コマンド定義
│   │   │   ├── create.ts
│   │   │   ├── list.ts
│   │   │   └── delete.ts
│   │   └── middleware/      # ミドルウェア
│   │       ├── auth.ts
│   │       ├── logger.ts
│   │       └── validator.ts
│   ├── core/                # ビジネスロジック
│   │   ├── services/
│   │   │   ├── ProjectService.ts
│   │   │   └── TemplateService.ts
│   │   └── models/
│   │       └── Project.ts
│   ├── infrastructure/      # インフラストラクチャ
│   │   ├── filesystem/
│   │   │   └── FileSystem.ts
│   │   ├── network/
│   │   │   └── ApiClient.ts
│   │   └── storage/
│   │       └── ConfigStore.ts
│   └── utils/               # ユーティリティ
│       ├── logger.ts
│       ├── formatter.ts
│       └── validator.ts
└── templates/               # テンプレートファイル
```

**実装例**:

**CLI レイヤー（src/cli/commands/create.ts）**:
```typescript
import { Command } from 'commander'
import { ProjectService } from '../../core/services/ProjectService'
import { logger } from '../../utils/logger'
import { validateProjectName } from '../middleware/validator'

export function createCommand(projectService: ProjectService) {
  return new Command('create')
    .description('Create a new project')
    .argument('<name>', 'Project name')
    .option('-t, --template <template>', 'Template to use', 'default')
    .hook('preAction', validateProjectName)
    .action(async (name, options) => {
      try {
        await projectService.create(name, options)
        logger.success(`Project '${name}' created successfully`)
      } catch (error) {
        logger.error('Failed to create project')
        throw error
      }
    })
}
```

**ビジネスロジック（src/core/services/ProjectService.ts）**:
```typescript
import { Project } from '../models/Project'
import { FileSystem } from '../../infrastructure/filesystem/FileSystem'
import { TemplateService } from './TemplateService'

export class ProjectService {
  constructor(
    private fileSystem: FileSystem,
    private templateService: TemplateService
  ) {}

  async create(name: string, options: CreateOptions): Promise<Project> {
    // ビジネスロジック
    const project = new Project(name, options.template)

    // プロジェクトディレクトリ作成
    await this.fileSystem.createDirectory(project.path)

    // テンプレート適用
    await this.templateService.apply(project, options.template)

    // 初期化
    await project.initialize()

    return project
  }

  async list(): Promise<Project[]> {
    return this.fileSystem.listProjects()
  }

  async delete(name: string): Promise<void> {
    const project = await this.find(name)
    await this.fileSystem.deleteDirectory(project.path)
  }

  private async find(name: string): Promise<Project> {
    const projects = await this.list()
    const project = projects.find(p => p.name === name)
    if (!project) {
      throw new Error(`Project '${name}' not found`)
    }
    return project
  }
}
```

**インフラストラクチャ（src/infrastructure/filesystem/FileSystem.ts）**:
```typescript
import fs from 'fs-extra'
import path from 'path'
import { Project } from '../../core/models/Project'

export class FileSystem {
  constructor(private baseDir: string = process.cwd()) {}

  async createDirectory(dirPath: string): Promise<void> {
    await fs.ensureDir(path.join(this.baseDir, dirPath))
  }

  async deleteDirectory(dirPath: string): Promise<void> {
    await fs.remove(path.join(this.baseDir, dirPath))
  }

  async listProjects(): Promise<Project[]> {
    const dirs = await fs.readdir(this.baseDir)
    const projects: Project[] = []

    for (const dir of dirs) {
      const pkgPath = path.join(this.baseDir, dir, 'package.json')
      if (await fs.pathExists(pkgPath)) {
        const pkg = await fs.readJSON(pkgPath)
        projects.push(new Project(pkg.name, dir))
      }
    }

    return projects
  }

  async copyTemplate(source: string, dest: string): Promise<void> {
    await fs.copy(source, path.join(this.baseDir, dest))
  }

  async writeFile(filePath: string, content: string): Promise<void> {
    await fs.writeFile(path.join(this.baseDir, filePath), content)
  }

  async readFile(filePath: string): Promise<string> {
    return fs.readFile(path.join(this.baseDir, filePath), 'utf-8')
  }
}
```

**依存性注入（src/cli/index.ts）**:
```typescript
#!/usr/bin/env node

import { Command } from 'commander'
import { FileSystem } from '../infrastructure/filesystem/FileSystem'
import { TemplateService } from '../core/services/TemplateService'
import { ProjectService } from '../core/services/ProjectService'
import { createCommand } from './commands/create'
import { listCommand } from './commands/list'
import { deleteCommand } from './commands/delete'

// 依存性の初期化
const fileSystem = new FileSystem()
const templateService = new TemplateService(fileSystem)
const projectService = new ProjectService(fileSystem, templateService)

// CLI プログラム
const program = new Command()

program
  .name('my-cli')
  .description('A sample CLI tool')
  .version('1.0.0')

// コマンド登録（依存性を注入）
program.addCommand(createCommand(projectService))
program.addCommand(listCommand(projectService))
program.addCommand(deleteCommand(projectService))

program.parse()
```

### プラグインアーキテクチャ

**プラグインインターフェース（src/core/Plugin.ts）**:
```typescript
export interface Plugin {
  name: string
  version: string
  commands?: Command[]
  hooks?: PluginHooks
  initialize?(context: PluginContext): Promise<void>
}

export interface PluginHooks {
  beforeCommand?: (context: CommandContext) => Promise<void>
  afterCommand?: (context: CommandContext) => Promise<void>
  onError?: (error: Error, context: CommandContext) => Promise<void>
}

export interface PluginContext {
  config: Config
  logger: Logger
  fileSystem: FileSystem
}

export interface CommandContext {
  command: string
  args: string[]
  options: Record<string, any>
}
```

**プラグインマネージャー（src/core/PluginManager.ts）**:
```typescript
import { Plugin, PluginContext } from './Plugin'

export class PluginManager {
  private plugins: Map<string, Plugin> = new Map()
  private context: PluginContext

  constructor(context: PluginContext) {
    this.context = context
  }

  async register(plugin: Plugin): Promise<void> {
    // プラグイン初期化
    if (plugin.initialize) {
      await plugin.initialize(this.context)
    }

    this.plugins.set(plugin.name, plugin)
    this.context.logger.info(`Plugin '${plugin.name}' registered`)
  }

  async load(pluginPath: string): Promise<void> {
    const plugin = await import(pluginPath)
    await this.register(plugin.default)
  }

  async executeHook(
    hookName: keyof PluginHooks,
    context: any
  ): Promise<void> {
    for (const plugin of this.plugins.values()) {
      const hook = plugin.hooks?.[hookName]
      if (hook) {
        await hook(context)
      }
    }
  }

  getPlugins(): Plugin[] {
    return Array.from(this.plugins.values())
  }

  getPlugin(name: string): Plugin | undefined {
    return this.plugins.get(name)
  }
}
```

**プラグインの作成例**:
```typescript
// plugins/analytics/index.ts
import { Plugin, CommandContext } from '../../core/Plugin'

const analyticsPlugin: Plugin = {
  name: 'analytics',
  version: '1.0.0',

  async initialize(context) {
    context.logger.info('Analytics plugin initialized')
  },

  hooks: {
    async beforeCommand(context: CommandContext) {
      // コマンド実行前の処理
      console.log(`[Analytics] Command: ${context.command}`)
    },

    async afterCommand(context: CommandContext) {
      // コマンド実行後の処理
      console.log(`[Analytics] Command completed: ${context.command}`)
    }
  }
}

export default analyticsPlugin
```

**プラグインの使用**:
```typescript
import { PluginManager } from './core/PluginManager'
import analyticsPlugin from './plugins/analytics'

const pluginManager = new PluginManager({
  config,
  logger,
  fileSystem
})

// プラグイン登録
await pluginManager.register(analyticsPlugin)

// フック実行
await pluginManager.executeHook('beforeCommand', {
  command: 'create',
  args: ['myapp'],
  options: {}
})
```

---

## 引数パース戦略

### Commander（Node.js）

**基本パターン**:
```typescript
import { Command } from 'commander'

const program = new Command()

// 1. 位置引数
program
  .command('create <name>')
  .description('Create a new project')
  .action((name) => {
    console.log(`Creating: ${name}`)
  })

// 2. オプション引数
program
  .command('build')
  .option('-w, --watch', 'Watch mode')
  .option('-m, --minify', 'Minify output')
  .option('-o, --output <dir>', 'Output directory', 'dist')
  .action((options) => {
    console.log('Options:', options)
  })

// 3. 可変長引数
program
  .command('install [packages...]')
  .description('Install packages')
  .action((packages) => {
    console.log('Installing:', packages)
  })

// 4. サブコマンド
const dockerCmd = program
  .command('docker')
  .description('Docker commands')

dockerCmd
  .command('build <image>')
  .action((image) => {
    console.log(`Building: ${image}`)
  })

program.parse()
```

**高度なバリデーション**:
```typescript
import { Command, Option } from 'commander'

program
  .command('deploy')
  .addOption(
    new Option('-e, --env <environment>', 'Environment')
      .choices(['dev', 'staging', 'production'])
      .default('dev')
  )
  .addOption(
    new Option('-p, --port <port>', 'Port number')
      .argParser(parseInt)
      .env('PORT')
      .default(3000)
  )
  .addOption(
    new Option('--verbose', 'Verbose logging')
      .conflicts('quiet')
  )
  .addOption(
    new Option('--quiet', 'Quiet mode')
      .conflicts('verbose')
  )
  .action((options) => {
    // バリデーション
    if (options.port < 1 || options.port > 65535) {
      console.error('Error: Port must be between 1 and 65535')
      process.exit(1)
    }

    console.log('Deploy options:', options)
  })
```

### Click（Python）

**基本パターン**:
```python
import click

@click.group()
@click.version_option()
def cli():
    """My CLI Tool"""
    pass

# 1. 位置引数
@cli.command()
@click.argument('name')
def create(name):
    """Create a new project"""
    click.echo(f'Creating: {name}')

# 2. オプション引数
@cli.command()
@click.option('--watch', '-w', is_flag=True, help='Watch mode')
@click.option('--minify', '-m', is_flag=True, help='Minify output')
@click.option('--output', '-o', default='dist', help='Output directory')
def build(watch, minify, output):
    """Build the project"""
    click.echo(f'Building to {output}')
    if watch:
        click.echo('Watch mode enabled')

# 3. 可変長引数
@cli.command()
@click.argument('packages', nargs=-1)
def install(packages):
    """Install packages"""
    for pkg in packages:
        click.echo(f'Installing: {pkg}')

# 4. サブコマンド
@cli.group()
def docker():
    """Docker commands"""
    pass

@docker.command()
@click.argument('image')
def build_image(image):
    """Build a Docker image"""
    click.echo(f'Building: {image}')

if __name__ == '__main__':
    cli()
```

**高度なバリデーション**:
```python
import click

def validate_port(ctx, param, value):
    if value < 1 or value > 65535:
        raise click.BadParameter('Port must be between 1 and 65535')
    return value

def validate_email(ctx, param, value):
    import re
    if not re.match(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', value):
        raise click.BadParameter('Invalid email address')
    return value

@cli.command()
@click.option(
    '--env',
    type=click.Choice(['dev', 'staging', 'production'], case_sensitive=False),
    default='dev',
    help='Environment'
)
@click.option(
    '--port',
    type=int,
    default=3000,
    callback=validate_port,
    help='Port number'
)
@click.option(
    '--email',
    callback=validate_email,
    help='Email address'
)
@click.option('--verbose', '-v', count=True, help='Verbose level')
def deploy(env, port, email, verbose):
    """Deploy the application"""
    click.echo(f'Deploying to {env} on port {port}')
    if email:
        click.echo(f'Notification email: {email}')

    # Verbose レベル
    if verbose >= 2:
        click.echo('Debug mode')
    elif verbose == 1:
        click.echo('Verbose mode')
```

### Typer（Python、推奨）

**基本パターン**:
```python
import typer
from typing import Optional, List
from enum import Enum

app = typer.Typer()

class Environment(str, Enum):
    dev = "dev"
    staging = "staging"
    production = "production"

# 1. 位置引数
@app.command()
def create(name: str):
    """Create a new project"""
    typer.echo(f'Creating: {name}')

# 2. オプション引数（型ヒント）
@app.command()
def build(
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch mode"),
    minify: bool = typer.Option(False, "--minify", "-m", help="Minify output"),
    output: str = typer.Option("dist", "--output", "-o", help="Output directory")
):
    """Build the project"""
    typer.echo(f'Building to {output}')
    if watch:
        typer.echo('Watch mode enabled')

# 3. 可変長引数
@app.command()
def install(packages: List[str]):
    """Install packages"""
    for pkg in packages:
        typer.echo(f'Installing: {pkg}')

# 4. サブコマンド
docker_app = typer.Typer()
app.add_typer(docker_app, name="docker", help="Docker commands")

@docker_app.command("build")
def build_image(image: str):
    """Build a Docker image"""
    typer.echo(f'Building: {image}')

# 5. Enum を使った選択肢
@app.command()
def deploy(
    env: Environment = typer.Option(Environment.dev, help="Environment"),
    port: int = typer.Option(3000, min=1, max=65535, help="Port number"),
    email: Optional[str] = typer.Option(None, help="Email address"),
    verbose: int = typer.Option(0, "-v", count=True, help="Verbose level")
):
    """Deploy the application"""
    typer.echo(f'Deploying to {env.value} on port {port}')
    if email:
        typer.echo(f'Notification email: {email}')

    if verbose >= 2:
        typer.echo('Debug mode')
    elif verbose == 1:
        typer.echo('Verbose mode')

if __name__ == '__main__':
    app()
```

**カスタムバリデーション**:
```python
import typer
import re
from typing import Optional

def validate_email(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None

    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(pattern, value):
        raise typer.BadParameter("Invalid email address")
    return value

def validate_project_name(value: str) -> str:
    if not re.match(r'^[a-z0-9-]+$', value):
        raise typer.BadParameter(
            "Project name must contain only lowercase letters, numbers, and hyphens"
        )
    return value

@app.command()
def create(
    name: str = typer.Argument(..., callback=validate_project_name),
    email: Optional[str] = typer.Option(None, callback=validate_email)
):
    """Create a new project"""
    typer.echo(f'Creating: {name}')
    if email:
        typer.echo(f'Notification email: {email}')
```

### Cobra（Go）

**基本パターン**:
```go
package main

import (
    "fmt"
    "github.com/spf13/cobra"
    "os"
)

var (
    watch   bool
    minify  bool
    output  string
    verbose int
)

var rootCmd = &cobra.Command{
    Use:   "mycli",
    Short: "A sample CLI tool",
    Long:  "A comprehensive CLI tool for project management",
}

var createCmd = &cobra.Command{
    Use:   "create [name]",
    Short: "Create a new project",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        name := args[0]
        fmt.Printf("Creating: %s\n", name)
    },
}

var buildCmd = &cobra.Command{
    Use:   "build",
    Short: "Build the project",
    Run: func(cmd *cobra.Command, args []string) {
        fmt.Printf("Building to %s\n", output)
        if watch {
            fmt.Println("Watch mode enabled")
        }
        if minify {
            fmt.Println("Minify enabled")
        }
    },
}

var installCmd = &cobra.Command{
    Use:   "install [packages...]",
    Short: "Install packages",
    Args:  cobra.MinimumNArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        for _, pkg := range args {
            fmt.Printf("Installing: %s\n", pkg)
        }
    },
}

func init() {
    // Build コマンドのフラグ
    buildCmd.Flags().BoolVarP(&watch, "watch", "w", false, "Watch mode")
    buildCmd.Flags().BoolVarP(&minify, "minify", "m", false, "Minify output")
    buildCmd.Flags().StringVarP(&output, "output", "o", "dist", "Output directory")

    // グローバルフラグ
    rootCmd.PersistentFlags().CountVarP(&verbose, "verbose", "v", "Verbose level")

    // コマンド追加
    rootCmd.AddCommand(createCmd)
    rootCmd.AddCommand(buildCmd)
    rootCmd.AddCommand(installCmd)
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Fprintf(os.Stderr, "Error: %v\n", err)
        os.Exit(1)
    }
}
```

---

## 設定管理システム

### 設定ファイルの階層

**優先順位**:
```
1. コマンドライン引数（最優先）
2. 環境変数
3. プロジェクト設定ファイル（./myconfig.json）
4. ユーザー設定ファイル（~/.myconfig）
5. グローバル設定ファイル（/etc/myconfig）
6. デフォルト値（最低優先）
```

### cosmiconfig を使った設定管理

**インストール**:
```bash
npm install cosmiconfig
```

**実装**:
```typescript
import { cosmiconfigSync } from 'cosmiconfig'
import { Config } from './types'

const moduleName = 'mycli'

export class ConfigManager {
  private explorer = cosmiconfigSync(moduleName)

  load(): Config {
    // 設定ファイル検索
    const result = this.explorer.search()

    if (result) {
      return this.mergeWithDefaults(result.config)
    }

    return this.getDefaults()
  }

  loadFrom(filepath: string): Config {
    const result = this.explorer.load(filepath)
    if (!result) {
      throw new Error(`Config file not found: ${filepath}`)
    }
    return this.mergeWithDefaults(result.config)
  }

  private mergeWithDefaults(config: Partial<Config>): Config {
    return {
      ...this.getDefaults(),
      ...config
    }
  }

  private getDefaults(): Config {
    return {
      template: 'default',
      port: 3000,
      verbose: false,
      features: []
    }
  }
}

// 使用例
const configManager = new ConfigManager()
const config = configManager.load()
```

**対応する設定ファイル形式**:
```
.myconfig
.myconfig.json
.myconfig.yaml
.myconfig.yml
.myconfig.js
.myconfig.cjs
myconfig.config.js
myconfig.config.cjs
package.json (mycli フィールド)
```

### 環境変数の管理

**dotenv を使った環境変数**:
```typescript
import dotenv from 'dotenv'
import path from 'path'

export class EnvManager {
  load(envFile?: string): void {
    // .env ファイル読み込み
    const envPath = envFile || path.join(process.cwd(), '.env')
    dotenv.config({ path: envPath })
  }

  get(key: string, defaultValue?: string): string | undefined {
    return process.env[key] || defaultValue
  }

  getRequired(key: string): string {
    const value = process.env[key]
    if (!value) {
      throw new Error(`Required environment variable not set: ${key}`)
    }
    return value
  }

  getInt(key: string, defaultValue: number): number {
    const value = process.env[key]
    if (!value) return defaultValue

    const parsed = parseInt(value, 10)
    if (isNaN(parsed)) {
      throw new Error(`Invalid integer value for ${key}: ${value}`)
    }
    return parsed
  }

  getBool(key: string, defaultValue: boolean): boolean {
    const value = process.env[key]
    if (!value) return defaultValue
    return value.toLowerCase() === 'true'
  }
}

// 使用例
const env = new EnvManager()
env.load()

const apiKey = env.getRequired('API_KEY')
const port = env.getInt('PORT', 3000)
const debug = env.getBool('DEBUG', false)
```

### 統合設定システム

**全ソースを統合**:
```typescript
import { ConfigManager } from './ConfigManager'
import { EnvManager } from './EnvManager'
import { Config } from './types'

export class ConfigLoader {
  private configManager = new ConfigManager()
  private envManager = new EnvManager()

  load(cliOptions: Partial<Config> = {}): Config {
    // 1. 環境変数読み込み
    this.envManager.load()

    // 2. 設定ファイル読み込み
    const fileConfig = this.configManager.load()

    // 3. 環境変数から設定を取得
    const envConfig: Partial<Config> = {
      port: this.envManager.getInt('MYCLI_PORT', fileConfig.port),
      verbose: this.envManager.getBool('MYCLI_VERBOSE', fileConfig.verbose),
      template: this.envManager.get('MYCLI_TEMPLATE', fileConfig.template)
    }

    // 4. マージ（優先順位: CLI > 環境変数 > ファイル > デフォルト）
    return {
      ...fileConfig,
      ...envConfig,
      ...cliOptions
    }
  }

  validate(config: Config): void {
    // バリデーション
    if (config.port < 1 || config.port > 65535) {
      throw new Error('Port must be between 1 and 65535')
    }

    if (!config.template) {
      throw new Error('Template is required')
    }
  }
}

// 使用例
import { Command } from 'commander'

const program = new Command()

program
  .command('create <name>')
  .option('-t, --template <template>', 'Template')
  .option('-p, --port <port>', 'Port', parseInt)
  .action((name, options) => {
    const loader = new ConfigLoader()
    const config = loader.load(options)
    loader.validate(config)

    console.log('Final config:', config)
  })
```

---

## 出力フォーマッティング

### テーブル出力（Node.js）

**cli-table3 を使った実装**:
```bash
npm install cli-table3
```

```typescript
import Table from 'cli-table3'
import chalk from 'chalk'

interface Project {
  name: string
  template: string
  created: Date
  size: string
}

export function formatProjectTable(projects: Project[]): string {
  const table = new Table({
    head: [
      chalk.cyan('Name'),
      chalk.cyan('Template'),
      chalk.cyan('Created'),
      chalk.cyan('Size')
    ],
    style: {
      head: [],
      border: ['gray']
    }
  })

  for (const project of projects) {
    table.push([
      chalk.bold(project.name),
      project.template,
      project.created.toLocaleDateString(),
      project.size
    ])
  }

  return table.toString()
}

// 使用例
const projects: Project[] = [
  { name: 'myapp', template: 'react', created: new Date(), size: '10 MB' },
  { name: 'api', template: 'nodejs', created: new Date(), size: '5 MB' }
]

console.log(formatProjectTable(projects))
```

**出力例**:
```
┌─────────┬──────────┬────────────┬────────┐
│ Name    │ Template │ Created    │ Size   │
├─────────┼──────────┼────────────┼────────┤
│ myapp   │ react    │ 1/3/2026   │ 10 MB  │
│ api     │ nodejs   │ 1/3/2026   │ 5 MB   │
└─────────┴──────────┴────────────┴────────┘
```

### プログレスバー

**cli-progress を使った実装**:
```bash
npm install cli-progress
```

```typescript
import cliProgress from 'cli-progress'
import chalk from 'chalk'

export async function downloadWithProgress(url: string): Promise<void> {
  const progressBar = new cliProgress.SingleBar({
    format: `Downloading ${chalk.cyan('{filename}')} |` +
            chalk.cyan('{bar}') + '| {percentage}% || {value}/{total} MB',
    barCompleteChar: '\u2588',
    barIncompleteChar: '\u2591',
    hideCursor: true
  })

  // プログレスバー開始
  progressBar.start(100, 0, { filename: 'package.zip' })

  // ダウンロードシミュレーション
  for (let i = 0; i <= 100; i++) {
    await new Promise(resolve => setTimeout(resolve, 50))
    progressBar.update(i)
  }

  progressBar.stop()
  console.log(chalk.green('✓ Download complete'))
}

// マルチバー
export async function buildWithMultiProgress(): Promise<void> {
  const multiBar = new cliProgress.MultiBar({
    clearOnComplete: false,
    hideCursor: true,
    format: ' {task} |{bar}| {percentage}%'
  })

  const compileBar = multiBar.create(100, 0, { task: 'Compiling' })
  const bundleBar = multiBar.create(100, 0, { task: 'Bundling ' })
  const minifyBar = multiBar.create(100, 0, { task: 'Minifying' })

  // 並行処理シミュレーション
  const tasks = [
    updateBar(compileBar, 100),
    updateBar(bundleBar, 100),
    updateBar(minifyBar, 100)
  ]

  await Promise.all(tasks)
  multiBar.stop()
  console.log(chalk.green('\n✓ Build complete'))
}

async function updateBar(bar: any, total: number): Promise<void> {
  for (let i = 0; i <= total; i++) {
    await new Promise(resolve => setTimeout(resolve, Math.random() * 100))
    bar.update(i)
  }
}
```

### JSON/CSV/YAML 出力

**フォーマット切り替え**:
```typescript
import yaml from 'js-yaml'
import Papa from 'papaparse'

export enum OutputFormat {
  JSON = 'json',
  CSV = 'csv',
  YAML = 'yaml',
  TABLE = 'table'
}

export class OutputFormatter {
  format(data: any[], format: OutputFormat): string {
    switch (format) {
      case OutputFormat.JSON:
        return this.formatJSON(data)
      case OutputFormat.CSV:
        return this.formatCSV(data)
      case OutputFormat.YAML:
        return this.formatYAML(data)
      case OutputFormat.TABLE:
        return this.formatTable(data)
      default:
        throw new Error(`Unknown format: ${format}`)
    }
  }

  private formatJSON(data: any[]): string {
    return JSON.stringify(data, null, 2)
  }

  private formatCSV(data: any[]): string {
    return Papa.unparse(data)
  }

  private formatYAML(data: any[]): string {
    return yaml.dump(data)
  }

  private formatTable(data: any[]): string {
    // cli-table3 を使用
    return formatProjectTable(data)
  }
}

// 使用例
program
  .command('list')
  .option('--format <format>', 'Output format', 'table')
  .action((options) => {
    const projects = getProjects()
    const formatter = new OutputFormatter()

    const output = formatter.format(projects, options.format as OutputFormat)
    console.log(output)
  })
```

### カラーテーマ

**テーマシステム**:
```typescript
import chalk from 'chalk'

export interface Theme {
  success: chalk.Chalk
  error: chalk.Chalk
  warning: chalk.Chalk
  info: chalk.Chalk
  highlight: chalk.Chalk
  muted: chalk.Chalk
}

export const defaultTheme: Theme = {
  success: chalk.green,
  error: chalk.red,
  warning: chalk.yellow,
  info: chalk.blue,
  highlight: chalk.cyan,
  muted: chalk.gray
}

export const darkTheme: Theme = {
  success: chalk.greenBright,
  error: chalk.redBright,
  warning: chalk.yellowBright,
  info: chalk.blueBright,
  highlight: chalk.cyanBright,
  muted: chalk.gray
}

export class ThemedLogger {
  constructor(private theme: Theme = defaultTheme) {}

  success(message: string): void {
    console.log(this.theme.success(`✓ ${message}`))
  }

  error(message: string): void {
    console.error(this.theme.error(`✗ ${message}`))
  }

  warning(message: string): void {
    console.warn(this.theme.warning(`⚠ ${message}`))
  }

  info(message: string): void {
    console.log(this.theme.info(`ℹ ${message}`))
  }

  highlight(message: string): void {
    console.log(this.theme.highlight(message))
  }

  muted(message: string): void {
    console.log(this.theme.muted(message))
  }
}

// 使用例
const logger = new ThemedLogger(darkTheme)
logger.success('Project created')
logger.error('Failed to build')
logger.warning('Deprecated feature')
logger.info('Installing dependencies')
```

---

## エラーハンドリング

### エラークラス階層

```typescript
export abstract class CLIError extends Error {
  abstract exitCode: number

  constructor(
    message: string,
    public suggestion?: string
  ) {
    super(message)
    this.name = this.constructor.name
  }
}

// ユーザーエラー（入力ミス、設定ミス）
export class UserError extends CLIError {
  exitCode = 1
}

export class InvalidArgumentError extends UserError {
  constructor(argument: string, expected: string) {
    super(
      `Invalid argument: ${argument}`,
      `Expected: ${expected}`
    )
  }
}

export class ConfigError extends UserError {
  constructor(message: string, configPath?: string) {
    super(
      message,
      configPath ? `Check config file: ${configPath}` : undefined
    )
  }
}

// システムエラー（権限、ネットワーク、ファイルシステム）
export class SystemError extends CLIError {
  exitCode = 2
}

export class FileSystemError extends SystemError {
  constructor(operation: string, path: string, cause?: Error) {
    super(
      `Failed to ${operation}: ${path}`,
      cause ? `Reason: ${cause.message}` : undefined
    )
  }
}

export class NetworkError extends SystemError {
  constructor(url: string, cause?: Error) {
    super(
      `Network request failed: ${url}`,
      'Check your internet connection'
    )
  }
}

// アプリケーションエラー（予期しないエラー）
export class ApplicationError extends CLIError {
  exitCode = 3
}
```

### グローバルエラーハンドラー

```typescript
import chalk from 'chalk'
import { CLIError } from './errors'

export function setupErrorHandler(): void {
  process.on('uncaughtException', (error) => {
    handleError(error)
    process.exit(3)
  })

  process.on('unhandledRejection', (reason) => {
    handleError(reason as Error)
    process.exit(3)
  })
}

export function handleError(error: Error): void {
  if (error instanceof CLIError) {
    console.error(chalk.red(`\n✗ ${error.message}`))

    if (error.suggestion) {
      console.error(chalk.yellow(`\nSuggestion: ${error.suggestion}`))
    }

    if (process.env.DEBUG) {
      console.error(chalk.gray('\nStack trace:'))
      console.error(chalk.gray(error.stack))
    }

    process.exit(error.exitCode)
  } else {
    // 予期しないエラー
    console.error(chalk.red('\n✗ An unexpected error occurred:'))
    console.error(error.message)
    console.error(chalk.gray('\nStack trace:'))
    console.error(error.stack)
    process.exit(3)
  }
}

// 使用例
setupErrorHandler()

program
  .command('create <name>')
  .action(async (name) => {
    try {
      if (!/^[a-z0-9-]+$/.test(name)) {
        throw new InvalidArgumentError(
          name,
          'lowercase letters, numbers, and hyphens only'
        )
      }

      await createProject(name)
    } catch (error) {
      handleError(error as Error)
    }
  })
```

### リトライ機構

```typescript
export interface RetryOptions {
  maxAttempts: number
  delay: number
  backoff?: 'linear' | 'exponential'
  onRetry?: (attempt: number, error: Error) => void
}

export async function retry<T>(
  fn: () => Promise<T>,
  options: RetryOptions
): Promise<T> {
  const { maxAttempts, delay, backoff = 'linear', onRetry } = options

  let lastError: Error

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      return await fn()
    } catch (error) {
      lastError = error as Error

      if (attempt === maxAttempts) {
        break
      }

      if (onRetry) {
        onRetry(attempt, lastError)
      }

      // 待機時間計算
      const waitTime = backoff === 'exponential'
        ? delay * Math.pow(2, attempt - 1)
        : delay * attempt

      await new Promise(resolve => setTimeout(resolve, waitTime))
    }
  }

  throw lastError!
}

// 使用例
async function downloadFile(url: string): Promise<void> {
  await retry(
    async () => {
      const response = await fetch(url)
      if (!response.ok) {
        throw new NetworkError(url)
      }
      return response
    },
    {
      maxAttempts: 3,
      delay: 1000,
      backoff: 'exponential',
      onRetry: (attempt, error) => {
        console.log(chalk.yellow(`Retry ${attempt}/3: ${error.message}`))
      }
    }
  )
}
```

---

## CLIテスト戦略

### ユニットテスト（Jest）

**コマンドのテスト**:
```typescript
// src/commands/create.test.ts
import { createCommand } from './create'
import { ProjectService } from '../core/services/ProjectService'

describe('create command', () => {
  let projectService: jest.Mocked<ProjectService>

  beforeEach(() => {
    projectService = {
      create: jest.fn(),
      list: jest.fn(),
      delete: jest.fn()
    } as any
  })

  it('should create a project with default template', async () => {
    const command = createCommand(projectService)

    await command.parseAsync(['node', 'test', 'myapp'])

    expect(projectService.create).toHaveBeenCalledWith('myapp', {
      template: 'default'
    })
  })

  it('should create a project with custom template', async () => {
    const command = createCommand(projectService)

    await command.parseAsync(['node', 'test', 'myapp', '--template', 'react'])

    expect(projectService.create).toHaveBeenCalledWith('myapp', {
      template: 'react'
    })
  })

  it('should throw error for invalid project name', async () => {
    const command = createCommand(projectService)

    await expect(
      command.parseAsync(['node', 'test', 'My App'])
    ).rejects.toThrow()
  })
})
```

### 統合テスト

**CLIの統合テスト**:
```typescript
// tests/integration/cli.test.ts
import { exec } from 'child_process'
import { promisify } from 'util'
import fs from 'fs-extra'
import path from 'path'

const execAsync = promisify(exec)
const CLI_PATH = path.join(__dirname, '../../dist/index.js')
const TEST_DIR = path.join(__dirname, '../../test-projects')

describe('CLI integration tests', () => {
  beforeEach(async () => {
    await fs.ensureDir(TEST_DIR)
    process.chdir(TEST_DIR)
  })

  afterEach(async () => {
    await fs.remove(TEST_DIR)
  })

  it('should create a project', async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} create myapp --skip-install`)

    expect(stdout).toContain('Project created successfully')
    expect(await fs.pathExists('./myapp')).toBe(true)
    expect(await fs.pathExists('./myapp/package.json')).toBe(true)
  })

  it('should list projects', async () => {
    await execAsync(`node ${CLI_PATH} create app1 --skip-install`)
    await execAsync(`node ${CLI_PATH} create app2 --skip-install`)

    const { stdout } = await execAsync(`node ${CLI_PATH} list`)

    expect(stdout).toContain('app1')
    expect(stdout).toContain('app2')
  })

  it('should show help', async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} --help`)

    expect(stdout).toContain('Usage:')
    expect(stdout).toContain('Commands:')
  })

  it('should show version', async () => {
    const { stdout } = await execAsync(`node ${CLI_PATH} --version`)

    expect(stdout).toMatch(/\d+\.\d+\.\d+/)
  })

  it('should handle errors gracefully', async () => {
    try {
      await execAsync(`node ${CLI_PATH} create "Invalid Name"`)
      fail('Should have thrown an error')
    } catch (error: any) {
      expect(error.stderr).toContain('Project name must contain')
    }
  })
})
```

### スナップショットテスト

**ヘルプ出力のスナップショット**:
```typescript
// tests/snapshots/help.test.ts
import { Command } from 'commander'
import { createCommand } from '../../src/commands/create'

describe('help output snapshots', () => {
  it('should match create command help', () => {
    const program = new Command()
    program.addCommand(createCommand(mockProjectService))

    const helpOutput = program.helpInformation()
    expect(helpOutput).toMatchSnapshot()
  })
})
```

### E2Eテスト（Playwright）

**対話的プロンプトのテスト**:
```typescript
import { test, expect } from '@playwright/test'
import { spawn } from 'child_process'

test('interactive project creation', async () => {
  const cli = spawn('node', ['dist/index.js', 'create'], {
    stdio: ['pipe', 'pipe', 'pipe']
  })

  let output = ''
  cli.stdout.on('data', (data) => {
    output += data.toString()
  })

  // プロンプトに応答
  setTimeout(() => cli.stdin.write('myapp\n'), 100)
  setTimeout(() => cli.stdin.write('\n'), 200)  // デフォルト選択
  setTimeout(() => cli.stdin.write('y\n'), 300)

  await new Promise((resolve) => cli.on('close', resolve))

  expect(output).toContain('Project name:')
  expect(output).toContain('Project created successfully')
})
```

---

## インタラクティブプロンプト

### Inquirer の高度な使用法

**動的な質問生成**:
```typescript
import inquirer from 'inquirer'

export async function configureProject(): Promise<ProjectConfig> {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'name',
      message: 'Project name:',
      validate: (input) => {
        if (!input) return 'Project name is required'
        if (!/^[a-z0-9-]+$/.test(input)) {
          return 'Must contain only lowercase letters, numbers, and hyphens'
        }
        return true
      }
    },
    {
      type: 'list',
      name: 'type',
      message: 'Project type:',
      choices: ['web', 'mobile', 'desktop', 'cli']
    },
    {
      type: 'checkbox',
      name: 'languages',
      message: 'Programming languages:',
      choices: ['JavaScript', 'TypeScript', 'Python', 'Go', 'Rust'],
      when: (answers) => answers.type !== 'mobile',  // モバイル以外
      validate: (input) => {
        if (input.length === 0) return 'Select at least one language'
        return true
      }
    },
    {
      type: 'list',
      name: 'mobileLanguage',
      message: 'Mobile language:',
      choices: ['Swift', 'Kotlin', 'React Native', 'Flutter'],
      when: (answers) => answers.type === 'mobile'
    },
    {
      type: 'confirm',
      name: 'useDocker',
      message: 'Use Docker?',
      default: false
    },
    {
      type: 'input',
      name: 'dockerImage',
      message: 'Docker image:',
      when: (answers) => answers.useDocker,
      default: 'node:18-alpine'
    }
  ])

  return answers as ProjectConfig
}
```

**カスタムプロンプト**:
```typescript
import inquirer from 'inquirer'

// 複数行入力
inquirer.registerPrompt('editor', require('inquirer-editor'))

// ファイルパス選択
inquirer.registerPrompt('file-tree', require('inquirer-file-tree-selection'))

// オートコンプリート
inquirer.registerPrompt('autocomplete', require('inquirer-autocomplete-prompt'))

export async function advancedPrompts(): Promise<void> {
  const answers = await inquirer.prompt([
    {
      type: 'editor',
      name: 'description',
      message: 'Project description:',
      default: '# Description\n\nWrite your project description here...'
    },
    {
      type: 'file-tree',
      name: 'directory',
      message: 'Select output directory:',
      root: process.cwd()
    },
    {
      type: 'autocomplete',
      name: 'framework',
      message: 'Select framework:',
      source: async (answersSoFar: any, input: string) => {
        const frameworks = ['React', 'Vue', 'Angular', 'Svelte', 'Next.js']
        if (!input) return frameworks

        return frameworks.filter(f =>
          f.toLowerCase().includes(input.toLowerCase())
        )
      }
    }
  ])
}
```

---

## 実践的なアーキテクチャ例

### プロジェクトジェネレーターCLI

**完全な実装例**:

```typescript
// src/cli/index.ts
#!/usr/bin/env node

import { Command } from 'commander'
import { ConfigLoader } from '../infrastructure/config/ConfigLoader'
import { FileSystem } from '../infrastructure/filesystem/FileSystem'
import { TemplateService } from '../core/services/TemplateService'
import { ProjectService } from '../core/services/ProjectService'
import { ThemedLogger, defaultTheme } from '../utils/logger'
import { setupErrorHandler } from '../utils/errorHandler'
import { createCommand } from './commands/create'
import { listCommand } from './commands/list'
import { deleteCommand } from './commands/delete'
import { updateCommand } from './commands/update'

// エラーハンドラー設定
setupErrorHandler()

// 依存性の初期化
const logger = new ThemedLogger(defaultTheme)
const configLoader = new ConfigLoader()
const fileSystem = new FileSystem()
const templateService = new TemplateService(fileSystem)
const projectService = new ProjectService(fileSystem, templateService)

// CLI プログラム
const program = new Command()

program
  .name('create-app')
  .description('A powerful project generator')
  .version('1.0.0')
  .option('-v, --verbose', 'Verbose output')
  .option('--no-color', 'Disable color output')
  .hook('preAction', (thisCommand) => {
    const options = thisCommand.opts()
    if (options.verbose) {
      process.env.DEBUG = 'true'
    }
  })

// コマンド登録
program.addCommand(createCommand(projectService, logger))
program.addCommand(listCommand(projectService, logger))
program.addCommand(deleteCommand(projectService, logger))
program.addCommand(updateCommand(logger))

// ヘルプのカスタマイズ
program.addHelpText('after', `

Examples:
  $ create-app create myapp
  $ create-app create myapp --template react
  $ create-app list
  $ create-app delete myapp

Documentation: https://github.com/username/create-app
`)

program.parse()
```

この包括的なガイドにより、CLIアプリケーションの設計から実装、テストまでの全体像を理解できます。次のステップとして、Python CLIの詳細な実装ガイドに進みましょう。

---

## まとめ

### CLIアーキテクチャチェックリスト

**アーキテクチャ**:
- [ ] レイヤード構造（CLI / Core / Infrastructure）
- [ ] 依存性注入
- [ ] プラグインシステム（必要に応じて）

**引数パース**:
- [ ] 適切なライブラリ選択（Commander / Click / Typer / Cobra）
- [ ] バリデーション実装
- [ ] ヘルプメッセージ

**設定管理**:
- [ ] 設定ファイルサポート
- [ ] 環境変数サポート
- [ ] 優先順位の明確化

**出力**:
- [ ] テーブル / JSON / CSV / YAML サポート
- [ ] カラーテーマ
- [ ] プログレス表示

**エラーハンドリング**:
- [ ] エラークラス階層
- [ ] グローバルエラーハンドラー
- [ ] 適切な終了コード

**テスト**:
- [ ] ユニットテスト
- [ ] 統合テスト
- [ ] スナップショットテスト

---

## 次のステップ

1. **05-python-cli.md**: Python CLI 開発ガイド（Click、Typer、Rich）
2. **templates/**: CLI プロジェクトテンプレート集

---

*堅牢なアーキテクチャで、保守性の高い CLI ツールを構築しましょう。*
