---
name: cli-development
description: CLIツール開発ガイド。Node.js（Commander、Inquirer）、Python（Click、Typer）、Go、引数パース、インタラクティブUI、配布方法など、プロフェッショナルなCLIツール開発のベストプラクティス。
---

# CLI Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [Node.js CLI](#nodejscli)
4. [Python CLI](#pythoncli)
5. [インタラクティブUI](#インタラクティブui)
6. [配布方法](#配布方法)
7. [実践例](#実践例)
8. [Agent連携](#agent連携)

---

## 概要

このSkillは、CLIツール開発をカバーします：

- **Node.js CLI** - Commander、Inquirer
- **Python CLI** - Click、Typer
- **引数パース** - オプション、サブコマンド
- **インタラクティブUI** - プロンプト、選択肢
- **カラー出力** - chalk、colorama
- **配布** - npm、PyPI、Homebrew

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 開発ツール作成時
- [ ] 自動化ツール作成時
- [ ] データ処理ツール作成時
- [ ] プロジェクトジェネレーター作成時

---

## Node.js CLI

### プロジェクトセットアップ

```bash
mkdir my-cli
cd my-cli
pnpm init
pnpm add commander inquirer chalk ora
pnpm add -D @types/node @types/inquirer typescript ts-node
```

```json
// package.json
{
  "name": "my-cli",
  "version": "1.0.0",
  "bin": {
    "my-cli": "./dist/index.js"
  },
  "scripts": {
    "build": "tsc",
    "dev": "ts-node src/index.ts"
  }
}
```

### Commander（引数パース）

```typescript
#!/usr/bin/env node

import { Command } from 'commander'

const program = new Command()

program
  .name('my-cli')
  .description('A sample CLI tool')
  .version('1.0.0')

// コマンド: my-cli create <name>
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

// コマンド: my-cli list
program
  .command('list')
  .description('List all projects')
  .option('-a, --all', 'Show all projects')
  .action((options) => {
    console.log('Listing projects...')
    if (options.all) {
      console.log('Showing all projects')
    }
  })

program.parse()
```

### Inquirer（インタラクティブプロンプト）

```typescript
import inquirer from 'inquirer'

async function createProject() {
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
        return true
      }
    },
    {
      type: 'list',
      name: 'template',
      message: 'Select a template:',
      choices: ['React', 'Vue', 'Next.js', 'Vite']
    },
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
      choices: ['ESLint', 'Prettier', 'Tailwind CSS', 'Vitest']
    }
  ])

  console.log('Creating project with:')
  console.log(answers)
}

createProject()
```

### Chalk（カラー出力）

```typescript
import chalk from 'chalk'

console.log(chalk.green('✅ Success!'))
console.log(chalk.red('❌ Error!'))
console.log(chalk.yellow('⚠️  Warning'))
console.log(chalk.blue('ℹ️  Info'))

console.log(chalk.bold('Bold text'))
console.log(chalk.italic('Italic text'))
console.log(chalk.underline('Underlined text'))

console.log(chalk.bgGreen.black(' SUCCESS '))
```

### Ora（スピナー）

```typescript
import ora from 'ora'

async function install() {
  const spinner = ora('Installing packages...').start()

  // 非同期処理
  await new Promise(resolve => setTimeout(resolve, 3000))

  spinner.succeed('Packages installed!')
}

install()
```

---

## Python CLI

### Click

```python
# cli.py
import click

@click.group()
@click.version_option()
def cli():
    """My CLI Tool"""
    pass

@cli.command()
@click.argument('name')
@click.option('--template', '-t', default='default', help='Template to use')
@click.option('--dir', '-d', default='.', help='Output directory')
def create(name, template, dir):
    """Create a new project"""
    click.echo(f'Creating project: {name}')
    click.echo(f'Template: {template}')
    click.echo(f'Directory: {dir}')

@cli.command()
@click.option('--all', '-a', is_flag=True, help='Show all projects')
def list(all):
    """List all projects"""
    click.echo('Listing projects...')
    if all:
        click.echo('Showing all projects')

if __name__ == '__main__':
    cli()

# 使用例:
# python cli.py create my-project --template react
# python cli.py list --all
```

### Typer（推奨）

```python
# cli.py
import typer
from typing import Optional
from enum import Enum

app = typer.Typer()

class Template(str, Enum):
    react = "react"
    vue = "vue"
    nextjs = "nextjs"

@app.command()
def create(
    name: str,
    template: Template = typer.Option(Template.react, help="Template to use"),
    dir: str = typer.Option(".", help="Output directory")
):
    """Create a new project"""
    typer.echo(f'Creating project: {name}')
    typer.echo(f'Template: {template.value}')
    typer.echo(f'Directory: {dir}')

@app.command()
def list(all: bool = typer.Option(False, "--all", "-a", help="Show all projects")):
    """List all projects"""
    typer.echo('Listing projects...')
    if all:
        typer.echo('Showing all projects')

if __name__ == '__main__':
    app()
```

### Rich（カラー・テーブル出力）

```python
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time

console = Console()

# カラー出力
console.print('[green]✅ Success![/green]')
console.print('[red]❌ Error![/red]')
console.print('[yellow]⚠️  Warning[/yellow]')

# テーブル
table = Table(title="Users")
table.add_column("ID", style="cyan")
table.add_column("Name", style="magenta")
table.add_column("Email", style="green")

table.add_row("1", "John Doe", "john@example.com")
table.add_row("2", "Jane Smith", "jane@example.com")

console.print(table)

# プログレスバー
for i in track(range(100), description="Processing..."):
    time.sleep(0.01)
```

---

## インタラクティブUI

### Node.js（Inquirer）

```typescript
import inquirer from 'inquirer'
import chalk from 'chalk'

async function setupProject() {
  console.log(chalk.bold.blue('\n🚀 Project Setup\n'))

  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'name',
      message: 'Project name:',
      default: 'my-project'
    },
    {
      type: 'list',
      name: 'framework',
      message: 'Select a framework:',
      choices: ['React', 'Vue', 'Next.js', 'Vite']
    },
    {
      type: 'confirm',
      name: 'typescript',
      message: 'Use TypeScript?',
      default: true
    },
    {
      type: 'checkbox',
      name: 'tools',
      message: 'Additional tools:',
      choices: [
        { name: 'ESLint', checked: true },
        { name: 'Prettier', checked: true },
        { name: 'Tailwind CSS', checked: false },
        { name: 'Vitest', checked: false }
      ]
    }
  ])

  console.log(chalk.green('\n✅ Setup complete!\n'))
  console.log(chalk.gray('Configuration:'))
  console.log(answers)
}

setupProject()
```

### Python（InquirerPy）

```python
from InquirerPy import inquirer
from InquirerPy.base.control import Choice

def setup_project():
    name = inquirer.text(
        message="Project name:",
        default="my-project"
    ).execute()

    framework = inquirer.select(
        message="Select a framework:",
        choices=["React", "Vue", "Next.js", "Vite"]
    ).execute()

    typescript = inquirer.confirm(
        message="Use TypeScript?",
        default=True
    ).execute()

    tools = inquirer.checkbox(
        message="Additional tools:",
        choices=[
            Choice("ESLint", enabled=True),
            Choice("Prettier", enabled=True),
            Choice("Tailwind CSS"),
            Choice("Vitest")
        ]
    ).execute()

    print(f"\n✅ Creating {name} with {framework}")
    print(f"TypeScript: {typescript}")
    print(f"Tools: {', '.join(tools)}")

setup_project()
```

---

## 配布方法

### npm パッケージ（Node.js）

```json
// package.json
{
  "name": "my-cli-tool",
  "version": "1.0.0",
  "bin": {
    "my-cli": "./dist/index.js"
  },
  "files": [
    "dist"
  ],
  "scripts": {
    "build": "tsc",
    "prepublishOnly": "pnpm build"
  }
}
```

```bash
# ビルド
pnpm build

# npmに公開
npm login
npm publish

# インストール
npm install -g my-cli-tool

# 実行
my-cli --help
```

### PyPI パッケージ（Python）

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='my-cli-tool',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'click>=8.0.0',
        'rich>=13.0.0'
    ],
    entry_points={
        'console_scripts': [
            'my-cli=my_cli.cli:main'
        ]
    }
)
```

```bash
# ビルド
python setup.py sdist bdist_wheel

# PyPIに公開
pip install twine
twine upload dist/*

# インストール
pip install my-cli-tool

# 実行
my-cli --help
```

---

## 実践例

### Example 1: プロジェクトジェネレーター（Node.js）

```typescript
#!/usr/bin/env node

import { Command } from 'commander'
import inquirer from 'inquirer'
import chalk from 'chalk'
import ora from 'ora'
import fs from 'fs/promises'
import path from 'path'

const program = new Command()

program
  .name('create-app')
  .description('Create a new app')
  .version('1.0.0')

program
  .argument('[name]', 'Project name')
  .action(async (name) => {
    let projectName = name

    if (!projectName) {
      const answers = await inquirer.prompt([
        {
          type: 'input',
          name: 'projectName',
          message: 'Project name:',
          default: 'my-app'
        }
      ])
      projectName = answers.projectName
    }

    const config = await inquirer.prompt([
      {
        type: 'list',
        name: 'template',
        message: 'Select a template:',
        choices: ['React', 'Vue', 'Next.js']
      },
      {
        type: 'confirm',
        name: 'typescript',
        message: 'Use TypeScript?',
        default: true
      }
    ])

    const spinner = ora('Creating project...').start()

    try {
      // プロジェクトディレクトリ作成
      const projectDir = path.join(process.cwd(), projectName)
      await fs.mkdir(projectDir, { recursive: true })

      // package.json作成
      const packageJson = {
        name: projectName,
        version: '0.1.0',
        private: true
      }
      await fs.writeFile(
        path.join(projectDir, 'package.json'),
        JSON.stringify(packageJson, null, 2)
      )

      spinner.succeed(chalk.green('Project created!'))

      console.log(chalk.cyan('\nNext steps:'))
      console.log(`  cd ${projectName}`)
      console.log('  npm install')
      console.log('  npm run dev')
    } catch (error) {
      spinner.fail(chalk.red('Failed to create project'))
      console.error(error)
      process.exit(1)
    }
  })

program.parse()
```

---

## Agent連携

### 📖 Agentへの指示例

**Node.js CLI作成**
```
以下の機能を持つNode.js CLIツールを作成してください：
- create <name>コマンド（プロジェクト作成）
- list コマンド（プロジェクト一覧）
- インタラクティブプロンプト（Inquirer）
- カラー出力（chalk）
```

**Python CLI作成**
```
Typerを使って、以下のPython CLIツールを作成してください：
- データ処理コマンド
- CSVファイルを読み込み、フィルタリング
- Richでテーブル出力
```

---

## まとめ

### CLI開発のベストプラクティス

1. **引数パース** - Commander（Node.js）、Typer（Python）
2. **インタラクティブUI** - Inquirer、InquirerPy
3. **カラー出力** - chalk、Rich
4. **エラーハンドリング** - 適切なエラーメッセージ

---

_Last updated: 2025-12-24_
