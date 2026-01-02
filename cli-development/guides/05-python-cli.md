# 🐍 Python CLI Development Guide

> **目的**: Click、Typer、Rich を使った実践的な Python CLI ツール開発の手法を習得する

## 📚 目次

1. [Python CLI フレームワーク比較](#python-cliフレームワーク比較)
2. [Click 完全ガイド](#click完全ガイド)
3. [Typer 完全ガイド](#typer完全ガイド)
4. [Rich による美しい出力](#richによる美しい出力)
5. [設定管理とプラグイン](#設定管理とプラグイン)
6. [テストとデバッグ](#テストとデバッグ)
7. [パッケージングと配布](#パッケージングと配布)
8. [実践的なプロジェクト例](#実践的なプロジェクト例)

---

## Python CLIフレームワーク比較

### 主要フレームワーク

| フレームワーク | 特徴 | 学習コスト | 推奨度 |
|---------------|------|-----------|--------|
| **argparse** | 標準ライブラリ | 低 | ⭐⭐ |
| **Click** | デコレータベース、柔軟 | 中 | ⭐⭐⭐⭐ |
| **Typer** | 型ヒント、モダン | 低 | ⭐⭐⭐⭐⭐ |
| **Fire** | 自動CLI生成 | 低 | ⭐⭐⭐ |

### argparse（標準ライブラリ）

**基本例**:
```python
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='A sample CLI tool',
        epilog='For more information, visit https://example.com'
    )

    # 位置引数
    parser.add_argument('name', help='Project name')

    # オプション引数
    parser.add_argument(
        '-t', '--template',
        default='default',
        help='Template to use'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    # サブコマンド
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # create コマンド
    create_parser = subparsers.add_parser('create', help='Create a project')
    create_parser.add_argument('name', help='Project name')

    # list コマンド
    list_parser = subparsers.add_parser('list', help='List projects')

    args = parser.parse_args()

    if args.command == 'create':
        print(f'Creating project: {args.name}')
    elif args.command == 'list':
        print('Listing projects...')

if __name__ == '__main__':
    main()
```

**利点**:
- 標準ライブラリ（追加インストール不要）
- 豊富な機能
- 公式ドキュメントが充実

**欠点**:
- 冗長なコード
- デコレータがない
- 型ヒントのサポートが弱い

---

## Click完全ガイド

### プロジェクトセットアップ

```bash
# プロジェクト作成
mkdir my-cli-tool
cd my-cli-tool

# 仮想環境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 依存関係インストール
pip install click rich
```

**ディレクトリ構造**:
```
my-cli-tool/
├── src/
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── commands/
│   │       ├── __init__.py
│   │       ├── create.py
│   │       ├── list.py
│   │       └── delete.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── project.py
│   │   └── template.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       └── config.py
├── tests/
│   ├── __init__.py
│   ├── test_commands.py
│   └── test_core.py
├── setup.py
├── pyproject.toml
└── README.md
```

### Click 基本概念

**グループとコマンド**:
```python
# src/cli/main.py
import click

@click.group()
@click.version_option(version='1.0.0')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, verbose):
    """My CLI Tool - A powerful project generator"""
    # コンテキストオブジェクトにデータを保存
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose

@cli.command()
@click.argument('name')
@click.option('--template', '-t', default='default', help='Template to use')
@click.pass_context
def create(ctx, name, template):
    """Create a new project"""
    verbose = ctx.obj['VERBOSE']

    if verbose:
        click.echo(f'Creating project: {name}')
        click.echo(f'Template: {template}')

    # プロジェクト作成ロジック
    create_project(name, template)

    click.secho('✓ Project created successfully!', fg='green')

@cli.command()
@click.option('--all', '-a', is_flag=True, help='Show all projects')
def list(all):
    """List all projects"""
    projects = get_projects(show_all=all)

    for project in projects:
        click.echo(f'  {project}')

if __name__ == '__main__':
    cli()
```

### Click の引数とオプション

**引数（Argument）**:
```python
import click

# 1つの引数
@click.command()
@click.argument('name')
def create(name):
    click.echo(f'Creating: {name}')

# 複数の引数
@click.command()
@click.argument('source')
@click.argument('destination')
def copy(source, destination):
    click.echo(f'Copying {source} to {destination}')

# 可変長引数
@click.command()
@click.argument('files', nargs=-1)
def process(files):
    for file in files:
        click.echo(f'Processing: {file}')

# ファイルパス引数
@click.command()
@click.argument('input', type=click.File('r'))
@click.argument('output', type=click.File('w'))
def convert(input, output):
    content = input.read()
    output.write(content.upper())
```

**オプション（Option）**:
```python
import click

# 真偽値フラグ
@click.command()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def build(verbose):
    if verbose:
        click.echo('Verbose mode enabled')

# 値を取るオプション
@click.command()
@click.option('--port', '-p', type=int, default=3000, help='Port number')
@click.option('--host', '-h', default='localhost', help='Host address')
def serve(port, host):
    click.echo(f'Serving on {host}:{port}')

# 選択肢
@click.command()
@click.option(
    '--env',
    type=click.Choice(['dev', 'staging', 'production'], case_sensitive=False),
    default='dev',
    help='Environment'
)
def deploy(env):
    click.echo(f'Deploying to {env}')

# 複数値
@click.command()
@click.option('--tag', '-t', multiple=True, help='Tags')
def create(tag):
    click.echo(f'Tags: {", ".join(tag)}')
    # Usage: mycli create -t python -t cli -t tool

# カウント
@click.command()
@click.option('--verbose', '-v', count=True, help='Verbose level')
def run(verbose):
    if verbose >= 2:
        click.echo('Debug mode')
    elif verbose == 1:
        click.echo('Verbose mode')

# パスワード
@click.command()
@click.option('--password', prompt=True, hide_input=True, confirmation_prompt=True)
def login(password):
    click.echo('Logging in...')

# 環境変数から取得
@click.command()
@click.option('--api-key', envvar='API_KEY', required=True, help='API key')
def api_call(api_key):
    click.echo(f'Using API key: {api_key[:4]}...')
```

### Click のバリデーション

**カスタムバリデーション**:
```python
import click
import re

def validate_email(ctx, param, value):
    """メールアドレスのバリデーション"""
    if value is None:
        return None

    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(pattern, value):
        raise click.BadParameter('Invalid email address')
    return value

def validate_port(ctx, param, value):
    """ポート番号のバリデーション"""
    if value < 1 or value > 65535:
        raise click.BadParameter('Port must be between 1 and 65535')
    return value

def validate_project_name(ctx, param, value):
    """プロジェクト名のバリデーション"""
    if not re.match(r'^[a-z0-9-]+$', value):
        raise click.BadParameter(
            'Project name must contain only lowercase letters, numbers, and hyphens'
        )
    return value

@click.command()
@click.argument('name', callback=validate_project_name)
@click.option('--email', callback=validate_email)
@click.option('--port', type=int, default=3000, callback=validate_port)
def create(name, email, port):
    """Create a new project"""
    click.echo(f'Creating project: {name}')
    if email:
        click.echo(f'Notification email: {email}')
    click.echo(f'Port: {port}')
```

**カスタム型**:
```python
import click
from pathlib import Path

class PathType(click.Path):
    """Path オブジェクトを返す型"""
    def convert(self, value, param, ctx):
        return Path(super().convert(value, param, ctx))

class PortRangeType(click.ParamType):
    """ポート範囲の型"""
    name = 'port_range'

    def convert(self, value, param, ctx):
        try:
            start, end = map(int, value.split('-'))
            if start < 1 or end > 65535 or start > end:
                self.fail(f'{value} is not a valid port range', param, ctx)
            return (start, end)
        except ValueError:
            self.fail(f'{value} is not a valid port range', param, ctx)

@click.command()
@click.argument('config', type=PathType(exists=True, dir_okay=False))
@click.option('--ports', type=PortRangeType(), default='3000-3100')
def configure(config, ports):
    """Configure the application"""
    click.echo(f'Config file: {config}')
    click.echo(f'Port range: {ports[0]}-{ports[1]}')
```

### Click のプロンプト

**インタラクティブプロンプト**:
```python
import click

@click.command()
def configure():
    """Interactive configuration"""

    # テキスト入力
    name = click.prompt('Project name', default='my-project')

    # パスワード入力
    password = click.prompt('Password', hide_input=True, confirmation_prompt=True)

    # 数値入力
    port = click.prompt('Port', type=int, default=3000)

    # 選択
    template = click.prompt(
        'Template',
        type=click.Choice(['react', 'vue', 'nextjs']),
        default='react'
    )

    # 確認
    if click.confirm('Use TypeScript?', default=True):
        click.echo('TypeScript enabled')

    # ファイル選択
    config_file = click.prompt(
        'Config file',
        type=click.Path(exists=True),
        default='config.json'
    )

    click.echo(f'\nProject: {name}')
    click.echo(f'Template: {template}')
    click.echo(f'Port: {port}')
```

### Click のコンテキスト

**グローバル設定の共有**:
```python
import click

class Config:
    def __init__(self):
        self.verbose = False
        self.debug = False
        self.config_file = None

pass_config = click.make_pass_decorator(Config, ensure=True)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--debug', is_flag=True, help='Debug mode')
@click.option('--config', type=click.Path(exists=True), help='Config file')
@pass_config
def cli(config, verbose, debug, config_file):
    """My CLI Tool"""
    config.verbose = verbose
    config.debug = debug
    config.config_file = config_file

@cli.command()
@pass_config
def create(config):
    """Create a new project"""
    if config.verbose:
        click.echo('Verbose mode enabled')
    if config.debug:
        click.echo('Debug mode enabled')

    click.echo('Creating project...')

@cli.command()
@pass_config
def list(config):
    """List all projects"""
    if config.verbose:
        click.echo('Listing projects...')

    projects = get_projects()
    for project in projects:
        click.echo(f'  {project}')
```

---

## Typer完全ガイド

### なぜ Typer か

**Typer の利点**:
- 型ヒントベース（自動バリデーション）
- シンプルな API
- Click ベース（Click の機能も使える）
- IDE サポートが優れている
- ドキュメントの自動生成

**基本例の比較**:

**Click**:
```python
import click

@click.command()
@click.argument('name')
@click.option('--age', type=int, required=True)
@click.option('--email', required=True)
def create(name, age, email):
    click.echo(f'{name}, {age}, {email}')
```

**Typer**:
```python
import typer

def create(name: str, age: int, email: str):
    typer.echo(f'{name}, {age}, {email}')

if __name__ == '__main__':
    typer.run(create)
```

### Typer の基本

**シンプルなCLI**:
```python
import typer

def main(
    name: str,
    verbose: bool = False,
    count: int = 1
):
    """
    A simple greeting application.

    Args:
        name: Your name
        verbose: Enable verbose output
        count: Number of times to greet
    """
    for _ in range(count):
        if verbose:
            typer.echo(f'Hello {name}! (verbose mode)')
        else:
            typer.echo(f'Hello {name}!')

if __name__ == '__main__':
    typer.run(main)
```

**複数コマンド**:
```python
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def create(
    name: str,
    template: str = typer.Option('default', help='Template to use'),
    typescript: bool = typer.Option(False, '--typescript/--no-typescript'),
    port: int = typer.Option(3000, min=1, max=65535)
):
    """Create a new project"""
    typer.echo(f'Creating project: {name}')
    typer.echo(f'Template: {template}')
    typer.echo(f'TypeScript: {typescript}')
    typer.echo(f'Port: {port}')

@app.command()
def list_projects(
    all: bool = typer.Option(False, '--all', '-a', help='Show all')
):
    """List all projects"""
    typer.echo('Listing projects...')

@app.command()
def delete(
    name: str,
    force: bool = typer.Option(False, '--force', '-f', help='Force delete')
):
    """Delete a project"""
    if not force:
        confirmed = typer.confirm(f'Delete {name}?')
        if not confirmed:
            typer.echo('Cancelled')
            raise typer.Abort()

    typer.echo(f'Deleting {name}...')

if __name__ == '__main__':
    app()
```

### Typer の型ヒント

**Enum を使った選択肢**:
```python
import typer
from enum import Enum

class Environment(str, Enum):
    dev = 'dev'
    staging = 'staging'
    production = 'production'

class Template(str, Enum):
    react = 'react'
    vue = 'vue'
    nextjs = 'nextjs'
    vite = 'vite'

app = typer.Typer()

@app.command()
def deploy(
    env: Environment = typer.Option(Environment.dev, help='Environment'),
    template: Template = typer.Option(Template.react, help='Template')
):
    """Deploy the application"""
    typer.echo(f'Deploying to {env.value} with {template.value}')

if __name__ == '__main__':
    app()
```

**Optional とデフォルト値**:
```python
import typer
from typing import Optional, List
from pathlib import Path

app = typer.Typer()

@app.command()
def create(
    name: str,                                    # 必須の位置引数
    template: str = 'default',                    # デフォルト値あり
    email: Optional[str] = None,                  # オプショナル
    tags: Optional[List[str]] = None,             # リスト
    config: Optional[Path] = None,                # Path 型
    port: int = typer.Option(3000, min=1, max=65535),  # 範囲指定
    verbose: int = typer.Option(0, '-v', count=True)   # カウント
):
    """Create a new project"""
    typer.echo(f'Project: {name}')
    typer.echo(f'Template: {template}')

    if email:
        typer.echo(f'Email: {email}')

    if tags:
        typer.echo(f'Tags: {", ".join(tags)}')

    if config:
        typer.echo(f'Config: {config}')

    typer.echo(f'Port: {port}')

    if verbose >= 2:
        typer.echo('Debug mode')
    elif verbose == 1:
        typer.echo('Verbose mode')

if __name__ == '__main__':
    app()
```

### Typer のバリデーション

**Annotated を使ったバリデーション**:
```python
import typer
from typing import Annotated
import re

def validate_email(value: str) -> str:
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(pattern, value):
        raise typer.BadParameter('Invalid email address')
    return value

def validate_project_name(value: str) -> str:
    if not re.match(r'^[a-z0-9-]+$', value):
        raise typer.BadParameter(
            'Project name must contain only lowercase letters, numbers, and hyphens'
        )
    return value

app = typer.Typer()

@app.command()
def create(
    name: Annotated[str, typer.Argument(callback=validate_project_name)],
    email: Annotated[
        str,
        typer.Option(callback=validate_email, help='Email address')
    ] = None
):
    """Create a new project"""
    typer.echo(f'Creating: {name}')
    if email:
        typer.echo(f'Email: {email}')

if __name__ == '__main__':
    app()
```

### Typer のプロンプト

**インタラクティブプロンプト**:
```python
import typer
from typing import Optional

app = typer.Typer()

@app.command()
def configure(
    name: str = typer.Option(..., prompt=True),
    email: str = typer.Option(..., prompt=True),
    password: str = typer.Option(
        ...,
        prompt=True,
        hide_input=True,
        confirmation_prompt=True
    ),
    port: int = typer.Option(3000, prompt=True),
    typescript: bool = typer.Option(True, prompt='Use TypeScript?')
):
    """Interactive configuration"""
    typer.echo(f'Name: {name}')
    typer.echo(f'Email: {email}')
    typer.echo(f'Port: {port}')
    typer.echo(f'TypeScript: {typescript}')

if __name__ == '__main__':
    app()
```

### Typer のサブコマンド

**グループ化されたコマンド**:
```python
import typer

app = typer.Typer()

# プロジェクト管理コマンド
project_app = typer.Typer()
app.add_typer(project_app, name='project', help='Project management')

@project_app.command('create')
def project_create(name: str):
    """Create a new project"""
    typer.echo(f'Creating project: {name}')

@project_app.command('list')
def project_list():
    """List all projects"""
    typer.echo('Listing projects...')

@project_app.command('delete')
def project_delete(name: str):
    """Delete a project"""
    typer.echo(f'Deleting project: {name}')

# Docker 管理コマンド
docker_app = typer.Typer()
app.add_typer(docker_app, name='docker', help='Docker management')

@docker_app.command('build')
def docker_build(image: str):
    """Build a Docker image"""
    typer.echo(f'Building image: {image}')

@docker_app.command('run')
def docker_run(container: str):
    """Run a Docker container"""
    typer.echo(f'Running container: {container}')

if __name__ == '__main__':
    app()

# 使用例:
# python cli.py project create myapp
# python cli.py project list
# python cli.py docker build myimage
```

### Typer のコールバック

**グローバル設定**:
```python
import typer
from pathlib import Path

app = typer.Typer()

# グローバルステート
class State:
    def __init__(self):
        self.verbose = False
        self.config_path: Path = None

state = State()

@app.callback()
def main(
    verbose: bool = typer.Option(False, '--verbose', '-v'),
    config: Path = typer.Option(None, '--config', '-c', exists=True)
):
    """
    My CLI Tool - A powerful project generator

    Use --verbose for detailed output.
    """
    state.verbose = verbose
    state.config_path = config

    if verbose:
        typer.echo('Verbose mode enabled')
    if config:
        typer.echo(f'Using config: {config}')

@app.command()
def create(name: str):
    """Create a new project"""
    if state.verbose:
        typer.echo(f'Creating project: {name}')

    # プロジェクト作成ロジック
    typer.secho('✓ Project created!', fg=typer.colors.GREEN)

@app.command()
def list_projects():
    """List all projects"""
    if state.verbose:
        typer.echo('Fetching projects...')

    projects = ['app1', 'app2', 'app3']
    for project in projects:
        typer.echo(f'  {project}')

if __name__ == '__main__':
    app()
```

---

## Richによる美しい出力

### Rich のインストール

```bash
pip install rich
```

### コンソール出力

**基本的な出力**:
```python
from rich.console import Console

console = Console()

# カラー出力
console.print('[green]Success![/green]')
console.print('[red]Error![/red]')
console.print('[yellow]Warning[/yellow]')
console.print('[blue]Info[/blue]')

# スタイル
console.print('[bold]Bold text[/bold]')
console.print('[italic]Italic text[/italic]')
console.print('[underline]Underlined text[/underline]')

# 組み合わせ
console.print('[bold green]✓ Success![/bold green]')
console.print('[bold red]✗ Error![/bold red]')

# RGB カラー
console.print('[rgb(123,45,67)]Custom color[/rgb(123,45,67)]')

# 背景色
console.print('[white on blue] INFO [/white on blue]')
console.print('[white on red] ERROR [/white on red]')
```

### Rich テーブル

**美しいテーブル出力**:
```python
from rich.console import Console
from rich.table import Table

console = Console()

def show_projects():
    table = Table(
        title='Projects',
        show_header=True,
        header_style='bold magenta'
    )

    table.add_column('Name', style='cyan', no_wrap=True)
    table.add_column('Template', style='green')
    table.add_column('Created', style='yellow')
    table.add_column('Size', justify='right', style='blue')

    table.add_row('myapp', 'React', '2026-01-03', '10 MB')
    table.add_row('api', 'Node.js', '2026-01-02', '5 MB')
    table.add_row('dashboard', 'Vue', '2026-01-01', '8 MB')

    console.print(table)

show_projects()
```

**動的テーブル**:
```python
from rich.console import Console
from rich.table import Table
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Project:
    name: str
    template: str
    created: datetime
    size: int  # in MB

console = Console()

def format_size(size_mb: int) -> str:
    if size_mb >= 1024:
        return f'{size_mb / 1024:.1f} GB'
    return f'{size_mb} MB'

def show_project_table(projects: list[Project]):
    table = Table(title='Projects')

    table.add_column('Name', style='cyan')
    table.add_column('Template', style='green')
    table.add_column('Created', style='yellow')
    table.add_column('Size', justify='right', style='blue')
    table.add_column('Status', justify='center')

    for project in projects:
        status = '✓' if project.size < 100 else '⚠'
        status_style = 'green' if project.size < 100 else 'yellow'

        table.add_row(
            project.name,
            project.template,
            project.created.strftime('%Y-%m-%d'),
            format_size(project.size),
            f'[{status_style}]{status}[/{status_style}]'
        )

    console.print(table)

# 使用例
projects = [
    Project('myapp', 'React', datetime.now(), 10),
    Project('api', 'Node.js', datetime.now(), 5),
    Project('dashboard', 'Vue', datetime.now(), 150)
]

show_project_table(projects)
```

### Rich プログレスバー

**プログレス表示**:
```python
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import time

def download_files():
    with Progress() as progress:
        task1 = progress.add_task('[cyan]Downloading...', total=100)
        task2 = progress.add_task('[green]Processing...', total=100)
        task3 = progress.add_task('[red]Uploading...', total=100)

        while not progress.finished:
            progress.update(task1, advance=0.9)
            progress.update(task2, advance=0.6)
            progress.update(task3, advance=0.3)
            time.sleep(0.02)

# カスタムプログレスバー
def custom_progress():
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task('[cyan]Installing dependencies...', total=100)

        for i in range(100):
            time.sleep(0.05)
            progress.update(task, advance=1)

download_files()
```

### Rich パネルとレイアウト

**情報パネル**:
```python
from rich.console import Console
from rich.panel import Panel
from rich.columns import Columns

console = Console()

# シンプルなパネル
console.print(Panel('Hello, World!', title='Greeting'))

# スタイル付きパネル
console.print(
    Panel(
        '[green]Project created successfully![/green]',
        title='Success',
        border_style='green'
    )
)

# 複数パネルを並べて表示
panels = [
    Panel('[cyan]React[/cyan]\n18.2.0', title='Framework', border_style='cyan'),
    Panel('[yellow]TypeScript[/yellow]\n5.0.0', title='Language', border_style='yellow'),
    Panel('[green]Vite[/green]\n4.0.0', title='Build Tool', border_style='green')
]

console.print(Columns(panels))
```

### Rich ツリー

**ディレクトリツリー**:
```python
from rich.tree import Tree
from rich.console import Console

console = Console()

def show_directory_tree():
    tree = Tree('📁 my-project', guide_style='bold bright_blue')

    src = tree.add('📁 src', guide_style='cyan')
    src.add('📄 index.ts')
    src.add('📄 App.tsx')

    components = src.add('📁 components', guide_style='cyan')
    components.add('📄 Header.tsx')
    components.add('📄 Footer.tsx')

    tree.add('📄 package.json')
    tree.add('📄 tsconfig.json')
    tree.add('📄 README.md')

    console.print(tree)

show_directory_tree()
```

### Rich マークダウン

**マークダウンレンダリング**:
```python
from rich.console import Console
from rich.markdown import Markdown

console = Console()

markdown_text = """
# Project Created Successfully!

## Next Steps

1. Install dependencies: `npm install`
2. Start dev server: `npm run dev`
3. Build for production: `npm run build`

## Features

- ✅ TypeScript support
- ✅ Hot Module Replacement
- ✅ ESLint & Prettier
- ✅ Tailwind CSS

Visit **https://example.com** for documentation.
"""

markdown = Markdown(markdown_text)
console.print(markdown)
```

### Typer + Rich の統合

**統合例**:
```python
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time

app = typer.Typer()
console = Console()

@app.command()
def create(name: str, template: str = 'react'):
    """Create a new project"""
    console.print(f'[cyan]Creating project: {name}[/cyan]')

    # プログレス付き処理
    for _ in track(range(100), description='Setting up...'):
        time.sleep(0.01)

    console.print('[green]✓ Project created successfully![/green]')

@app.command()
def list_projects():
    """List all projects"""
    table = Table(title='Projects')
    table.add_column('Name', style='cyan')
    table.add_column('Template', style='green')
    table.add_column('Status', style='yellow')

    table.add_row('myapp', 'React', '✓ Active')
    table.add_row('api', 'Node.js', '✓ Active')

    console.print(table)

if __name__ == '__main__':
    app()
```

---

## 設定管理とプラグイン

### TOML 設定ファイル

**pyproject.toml を使った設定**:
```python
# config.py
import tomli
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Config:
    name: str
    version: str
    template: str
    features: List[str]
    port: int
    verbose: bool

class ConfigLoader:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path('pyproject.toml')

    def load(self) -> Config:
        if not self.config_path.exists():
            return self.get_defaults()

        with open(self.config_path, 'rb') as f:
            data = tomli.load(f)

        tool_config = data.get('tool', {}).get('mycli', {})

        return Config(
            name=tool_config.get('name', 'my-project'),
            version=tool_config.get('version', '1.0.0'),
            template=tool_config.get('template', 'default'),
            features=tool_config.get('features', []),
            port=tool_config.get('port', 3000),
            verbose=tool_config.get('verbose', False)
        )

    def get_defaults(self) -> Config:
        return Config(
            name='my-project',
            version='1.0.0',
            template='default',
            features=[],
            port=3000,
            verbose=False
        )

# 使用例
loader = ConfigLoader()
config = loader.load()
```

**pyproject.toml 例**:
```toml
[tool.mycli]
name = "my-awesome-project"
version = "1.0.0"
template = "react"
features = ["eslint", "prettier", "tailwind"]
port = 3000
verbose = false
```

### プラグインシステム

**プラグインインターフェース**:
```python
# plugins/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict

class Plugin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name"""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version"""
        pass

    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize plugin"""
        pass

    def before_command(self, command: str, args: Dict[str, Any]) -> None:
        """Hook before command execution"""
        pass

    def after_command(self, command: str, result: Any) -> None:
        """Hook after command execution"""
        pass

    def on_error(self, error: Exception) -> None:
        """Hook on error"""
        pass
```

**プラグインマネージャー**:
```python
# plugins/manager.py
from typing import List, Dict, Any
from .base import Plugin

class PluginManager:
    def __init__(self):
        self.plugins: List[Plugin] = []

    def register(self, plugin: Plugin) -> None:
        """Register a plugin"""
        plugin.initialize({})
        self.plugins.append(plugin)
        print(f'Plugin registered: {plugin.name} v{plugin.version}')

    def execute_before_hooks(self, command: str, args: Dict[str, Any]) -> None:
        """Execute before hooks"""
        for plugin in self.plugins:
            plugin.before_command(command, args)

    def execute_after_hooks(self, command: str, result: Any) -> None:
        """Execute after hooks"""
        for plugin in self.plugins:
            plugin.after_command(command, result)

    def execute_error_hooks(self, error: Exception) -> None:
        """Execute error hooks"""
        for plugin in self.plugins:
            plugin.on_error(error)
```

**プラグイン例**:
```python
# plugins/analytics.py
from .base import Plugin
from typing import Any, Dict

class AnalyticsPlugin(Plugin):
    @property
    def name(self) -> str:
        return 'analytics'

    @property
    def version(self) -> str:
        return '1.0.0'

    def initialize(self, context: Dict[str, Any]) -> None:
        print('Analytics plugin initialized')
        self.command_count = 0

    def before_command(self, command: str, args: Dict[str, Any]) -> None:
        self.command_count += 1
        print(f'[Analytics] Command: {command} (Total: {self.command_count})')

    def after_command(self, command: str, result: Any) -> None:
        print(f'[Analytics] Command completed: {command}')

    def on_error(self, error: Exception) -> None:
        print(f'[Analytics] Error occurred: {error}')
```

**プラグインの使用**:
```python
import typer
from plugins.manager import PluginManager
from plugins.analytics import AnalyticsPlugin

app = typer.Typer()
plugin_manager = PluginManager()

# プラグイン登録
plugin_manager.register(AnalyticsPlugin())

@app.command()
def create(name: str):
    """Create a new project"""
    plugin_manager.execute_before_hooks('create', {'name': name})

    try:
        # プロジェクト作成ロジック
        result = f'Project {name} created'
        typer.echo(result)

        plugin_manager.execute_after_hooks('create', result)
    except Exception as e:
        plugin_manager.execute_error_hooks(e)
        raise

if __name__ == '__main__':
    app()
```

---

## テストとデバッグ

### pytest によるテスト

**インストール**:
```bash
pip install pytest pytest-cov
```

**コマンドのユニットテスト**:
```python
# tests/test_commands.py
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()

def test_create_command():
    result = runner.invoke(app, ['create', 'myapp'])
    assert result.exit_code == 0
    assert 'Creating project: myapp' in result.stdout

def test_create_with_template():
    result = runner.invoke(app, ['create', 'myapp', '--template', 'react'])
    assert result.exit_code == 0
    assert 'react' in result.stdout

def test_invalid_project_name():
    result = runner.invoke(app, ['create', 'My App'])
    assert result.exit_code != 0
    assert 'Invalid' in result.stdout

def test_list_command():
    result = runner.invoke(app, ['list'])
    assert result.exit_code == 0

def test_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'Usage:' in result.stdout
```

**モックを使ったテスト**:
```python
# tests/test_with_mock.py
from unittest.mock import patch, MagicMock
from cli.main import create_project

def test_create_project_with_mock():
    with patch('cli.main.FileSystem') as mock_fs:
        mock_fs.return_value.create_directory = MagicMock()
        mock_fs.return_value.write_file = MagicMock()

        create_project('myapp', 'react')

        mock_fs.return_value.create_directory.assert_called_once()
        mock_fs.return_value.write_file.assert_called()
```

### デバッグ

**ログ出力**:
```python
import typer
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def create(
    name: str,
    verbose: bool = typer.Option(False, '--verbose', '-v')
):
    """Create a new project"""
    if verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(f'Creating project: {name}')
    logger.debug(f'Project path: /path/to/{name}')

    try:
        # プロジェクト作成
        logger.info('Project created successfully')
    except Exception as e:
        logger.error(f'Failed to create project: {e}')
        raise

if __name__ == '__main__':
    app()
```

---

## パッケージングと配布

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name='my-cli-tool',
    version='1.0.0',
    description='A powerful CLI tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/username/my-cli-tool',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'typer>=0.9.0',
        'rich>=13.0.0',
        'tomli>=2.0.0'
    ],
    entry_points={
        'console_scripts': [
            'mycli=cli.main:app'
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)
```

### pyproject.toml（推奨）

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "my-cli-tool"
version = "1.0.0"
description = "A powerful CLI tool"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "tomli>=2.0.0"
]

[project.scripts]
mycli = "cli.main:app"

[project.urls]
Homepage = "https://github.com/username/my-cli-tool"
Documentation = "https://my-cli-tool.readthedocs.io"
Repository = "https://github.com/username/my-cli-tool.git"
```

### PyPI へ公開

```bash
# ビルド
pip install build
python -m build

# 公開
pip install twine
twine upload dist/*

# インストール
pip install my-cli-tool

# 実行
mycli --help
```

---

## 実践的なプロジェクト例

### フル機能のプロジェクトジェネレーター

```python
# src/cli/main.py
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from typing import Optional
from pathlib import Path
import time

app = typer.Typer()
console = Console()

@app.command()
def create(
    name: str = typer.Argument(..., help='Project name'),
    template: str = typer.Option('react', help='Template to use'),
    typescript: bool = typer.Option(True, '--typescript/--no-typescript'),
    install: bool = typer.Option(True, '--install/--no-install'),
    git: bool = typer.Option(True, '--git/--no-git')
):
    """
    Create a new project from a template.

    Examples:
        mycli create myapp
        mycli create myapp --template vue --no-typescript
    """
    console.print(Panel.fit(
        f'[cyan]Creating project: {name}[/cyan]',
        title='Project Generator',
        border_style='cyan'
    ))

    # プロジェクト作成
    console.print(f'Template: [green]{template}[/green]')
    console.print(f'TypeScript: [green]{typescript}[/green]')

    # プログレスバー付き処理
    tasks = [
        'Creating directory structure',
        'Copying template files',
        'Configuring project',
        'Installing dependencies' if install else 'Skipping dependencies',
        'Initializing git' if git else 'Skipping git'
    ]

    for task in track(tasks, description='Setting up project'):
        time.sleep(0.5)

    console.print('\n[bold green]✓ Project created successfully![/bold green]\n')
    console.print('[cyan]Next steps:[/cyan]')
    console.print(f'  cd {name}')
    if not install:
        console.print('  npm install')
    console.print('  npm run dev')

@app.command('list')
def list_projects(
    all: bool = typer.Option(False, '--all', '-a', help='Show all projects')
):
    """List all projects"""
    table = Table(title='Projects')

    table.add_column('Name', style='cyan', no_wrap=True)
    table.add_column('Template', style='green')
    table.add_column('Created', style='yellow')
    table.add_column('Size', justify='right', style='blue')

    table.add_row('myapp', 'React', '2026-01-03', '10 MB')
    table.add_row('api', 'Node.js', '2026-01-02', '5 MB')
    table.add_row('dashboard', 'Vue', '2026-01-01', '8 MB')

    console.print(table)

@app.command()
def delete(
    name: str = typer.Argument(..., help='Project name'),
    force: bool = typer.Option(False, '--force', '-f', help='Force delete')
):
    """Delete a project"""
    if not force:
        confirmed = typer.confirm(f'Delete project "{name}"?')
        if not confirmed:
            console.print('[yellow]Cancelled[/yellow]')
            raise typer.Abort()

    console.print(f'[red]Deleting project: {name}[/red]')
    time.sleep(1)
    console.print('[green]✓ Project deleted[/green]')

if __name__ == '__main__':
    app()
```

この包括的なガイドにより、Python を使った CLI ツール開発の全体像を理解できます。Click と Typer の違い、Rich による美しい出力、テスト手法、そして実践的な例まで網羅しています。

---

## まとめ

### Python CLI 開発チェックリスト

**フレームワーク選択**:
- [ ] Typer（推奨）: 型ヒント、モダン、シンプル
- [ ] Click: 柔軟、成熟、大規模プロジェクト向け

**出力**:
- [ ] Rich でカラフルな出力
- [ ] テーブル / プログレスバー / パネル
- [ ] マークダウンレンダリング

**設定管理**:
- [ ] TOML 設定ファイル
- [ ] 環境変数サポート
- [ ] プラグインシステム（必要に応じて）

**テスト**:
- [ ] pytest によるユニットテスト
- [ ] CliRunner による統合テスト
- [ ] モックを使ったテスト

**パッケージング**:
- [ ] pyproject.toml 設定
- [ ] PyPI へ公開
- [ ] entry_points 設定

---

## 次のステップ

1. **templates/**: Python CLI プロジェクトテンプレート
2. **examples/**: 実践的な CLI ツール例集

---

*Python で美しく使いやすい CLI ツールを構築しましょう。*
