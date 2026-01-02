---
name: cli-development
description: CLIツール開発ガイド。Node.js（Commander、Inquirer）、Python（Click、Typer）、Go（Cobra）、引数パース、インタラクティブUI、アーキテクチャ設計、テスト、配布方法など、プロフェッショナルなCLIツール開発のベストプラクティス。
---

# CLI Development Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [クイックスタート](#クイックスタート)
4. [詳細ガイド](#詳細ガイド)
5. [プロジェクトテンプレート](#プロジェクトテンプレート)
6. [ベストプラクティス](#ベストプラクティス)
7. [Agent連携](#agent連携)

---

## 概要

このSkillは、プロフェッショナルなCLIツール開発をカバーします：

### フレームワーク
- **Node.js** - Commander、Inquirer、chalk、ora
- **Python** - Click、Typer、Rich
- **Go** - Cobra、Viper

### カバー範囲
- **アーキテクチャ** - レイヤード、プラグイン
- **引数パース** - オプション、サブコマンド、バリデーション
- **出力** - カラー、テーブル、プログレスバー
- **設定管理** - 設定ファイル、環境変数
- **テスト** - ユニット、統合、E2E
- **配布** - npm、PyPI、Homebrew、バイナリ

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 開発ツール作成時
- [ ] 自動化ツール作成時
- [ ] データ処理ツール作成時
- [ ] プロジェクトジェネレーター作成時
- [ ] DevOps ツール作成時
- [ ] CI/CD パイプライン作成時

---

## クイックスタート

### Node.js CLI（Commander）

```bash
# プロジェクト作成
mkdir my-cli && cd my-cli
npm init -y

# 依存関係インストール
npm install commander inquirer chalk ora
npm install -D typescript @types/node ts-node
```

**src/index.ts**:
```typescript
#!/usr/bin/env node
import { Command } from 'commander'

const program = new Command()

program
  .name('my-cli')
  .description('A sample CLI tool')
  .version('1.0.0')

program
  .command('create <name>')
  .description('Create a new project')
  .option('-t, --template <template>', 'Template to use', 'default')
  .action((name, options) => {
    console.log(`Creating project: ${name}`)
    console.log(`Template: ${options.template}`)
  })

program.parse()
```

### Python CLI（Typer）

```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate

# 依存関係インストール
pip install "typer[all]" rich
```

**main.py**:
```python
import typer
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def create(
    name: str,
    template: str = typer.Option("default", help="Template to use")
):
    """Create a new project"""
    console.print(f"[cyan]Creating project: {name}[/cyan]")
    console.print(f"Template: [green]{template}[/green]")

if __name__ == "__main__":
    app()
```

### Go CLI（Cobra）

```bash
# プロジェクト作成
mkdir my-cli && cd my-cli
go mod init github.com/username/my-cli

# 依存関係インストール
go get github.com/spf13/cobra@latest
```

**main.go**:
```go
package main

import (
    "fmt"
    "github.com/spf13/cobra"
)

var rootCmd = &cobra.Command{
    Use:   "my-cli",
    Short: "A sample CLI tool",
}

var createCmd = &cobra.Command{
    Use:   "create [name]",
    Short: "Create a new project",
    Args:  cobra.ExactArgs(1),
    Run: func(cmd *cobra.Command, args []string) {
        name := args[0]
        template, _ := cmd.Flags().GetString("template")
        fmt.Printf("Creating project: %s\n", name)
        fmt.Printf("Template: %s\n", template)
    },
}

func init() {
    createCmd.Flags().StringP("template", "t", "default", "Template to use")
    rootCmd.AddCommand(createCmd)
}

func main() {
    if err := rootCmd.Execute(); err != nil {
        fmt.Println(err)
        os.Exit(1)
    }
}
```

---

## 詳細ガイド

### 📚 包括的なガイド

1. **[CLI設計原則ガイド](./guides/01-cli-design.md)**
   - CLI設計哲学（UNIX哲学、ユーザビリティ）
   - コマンド設計パターン
   - 引数とオプションの設計
   - 出力設計とエラーハンドリング

2. **[Node.js CLI実装ガイド](./guides/02-nodejs-cli.md)**
   - Commander による引数パース
   - Inquirer によるインタラクティブUI
   - chalk、ora による出力
   - ファイル操作と実践例

3. **[CLI配布・パッケージングガイド](./guides/03-distribution.md)**
   - npm パッケージ配布
   - Homebrew 配布
   - バイナリ配布（pkg）
   - GitHub Releases、自動更新

4. **[CLIアーキテクチャ & デザインパターンガイド](./guides/04-cli-architecture.md)** 🆕
   - レイヤードアーキテクチャ
   - プラグインアーキテクチャ
   - 引数パース戦略（Commander、Click、Typer、Cobra）
   - 設定管理システム
   - 出力フォーマッティング
   - エラーハンドリング
   - CLIテスト戦略

5. **[Python CLI開発ガイド](./guides/05-python-cli.md)** 🆕
   - Python CLIフレームワーク比較
   - Click 完全ガイド
   - Typer 完全ガイド（推奨）
   - Rich による美しい出力
   - 設定管理とプラグイン
   - テストとデバッグ
   - PyPI パッケージング

---

## プロジェクトテンプレート

### 🚀 すぐに使えるテンプレート

#### 1. [Python CLI Template (Typer)](./templates/python-typer/)
- Typer + Rich
- pytest によるテスト
- TOML 設定ファイル
- プラグインシステム
- ロギング、エラーハンドリング

#### 2. [Node.js CLI Template (Commander)](./templates/nodejs-commander/)
- TypeScript サポート
- Commander + Inquirer + chalk
- Jest によるテスト
- ESLint + Prettier

#### 3. [Go CLI Template (Cobra)](./templates/go-cobra/)
- Cobra + Viper
- カラフルな出力
- テスト
- クロスコンパイル対応

### テンプレート使用方法

**Python Typer**:
```bash
# テンプレートをコピー
cp -r templates/python-typer my-cli
cd my-cli

# 仮想環境作成
python -m venv venv
source venv/bin/activate

# 依存関係インストール
pip install -e ".[dev]"

# CLI 実行
mycli --help
```

**Node.js Commander**:
```bash
# テンプレートをコピー
cp -r templates/nodejs-commander my-cli
cd my-cli

# 依存関係インストール
npm install

# ビルド
npm run build

# CLI 実行
npm start -- --help
```

**Go Cobra**:
```bash
# テンプレートをコピー
cp -r templates/go-cobra my-cli
cd my-cli

# 依存関係インストール
go mod download

# ビルド
go build -o mycli

# CLI 実行
./mycli --help
```

---

## ベストプラクティス

### 📋 チェックリスト

**[CLI開発チェックリスト](./best-practices/CLI_CHECKLIST.md)** で品質を確保：

- **設計段階**: コマンド設計、引数設計、出力設計
- **実装段階**: アーキテクチャ、エラーハンドリング、設定管理
- **テスト段階**: ユニット、統合、E2Eテスト
- **配布段階**: パッケージング、ドキュメント、バージョン管理
- **保守段階**: 更新機構、ログ、セキュリティ

### 🧪 テスト

**[テストガイド](./best-practices/TESTING_GUIDE.md)** で堅牢性を確保：

- ユニットテスト（Jest、pytest、Go testing）
- 統合テスト
- E2Eテスト（インタラクティブプロンプト）
- スナップショットテスト
- カバレッジレポート

### 📦 配布

**[配布ガイド](./best-practices/DISTRIBUTION_GUIDE.md)** で幅広いユーザーに届ける：

- npm / PyPI 公開
- Homebrew Formula
- バイナリ配布（pkg）
- Docker 配布
- GitHub Actions 自動化
- 自動更新機構

---

## 実践例

### Example 1: プロジェクトジェネレーター

**機能**:
- インタラクティブプロンプトでプロジェクト設定
- テンプレートからプロジェクト作成
- 依存関係の自動インストール
- Git 初期化

**実装**: `templates/nodejs-commander/` または `templates/python-typer/` を参照

### Example 2: データ処理CLI

**機能**:
- CSV/JSON ファイルの読み込み
- フィルタリング、変換
- 複数形式での出力（CSV、JSON、YAML）
- プログレスバー表示

**Python 実装例**:
```python
import typer
from rich.console import Console
from rich.table import Table
import pandas as pd

app = typer.Typer()
console = Console()

@app.command()
def process(
    input_file: str,
    output: str = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("csv", help="Output format (csv, json, yaml)")
):
    """Process CSV file"""
    # データ読み込み
    df = pd.read_csv(input_file)

    # テーブル表示
    table = Table(title="Data")
    for col in df.columns:
        table.add_column(col)

    for _, row in df.head().iterrows():
        table.add_row(*[str(val) for val in row])

    console.print(table)

    # 出力
    if output:
        if format == "json":
            df.to_json(output, orient="records")
        elif format == "yaml":
            import yaml
            with open(output, 'w') as f:
                yaml.dump(df.to_dict(orient='records'), f)
        else:
            df.to_csv(output, index=False)

        console.print(f"[green]Saved to {output}[/green]")
```

### Example 3: 開発ツールCLI

**機能**:
- プロジェクトのビルド、テスト、デプロイ
- 設定ファイルの読み込み
- 環境変数の管理
- ログ出力

**アーキテクチャ**: `guides/04-cli-architecture.md` のレイヤードアーキテクチャを参照

---

## Agent連携

### 📖 Agentへの指示例

**基本的なCLI作成**:
```
Node.js CLIツールを作成してください：
- create <name>コマンド（プロジェクト作成）
- list コマンド（プロジェクト一覧）
- delete <name>コマンド（プロジェクト削除）
- Commanderで引数パース
- Inquirerでインタラクティブプロンプト
- chalkでカラー出力
- Jestでテスト
```

**アーキテクチャ重視のCLI作成**:
```
Python CLIツールを作成してください：
- Typer + Rich を使用
- レイヤードアーキテクチャ（CLI / Core / Infrastructure）
- 設定ファイルサポート（TOML）
- プラグインシステム
- 包括的なテスト（pytest）
- PyPI パッケージング
```

**データ処理CLI作成**:
```
CSV処理CLIツールを作成してください：
- CSVファイルを読み込み
- フィルタリング、ソート、集計
- 複数形式での出力（CSV、JSON、YAML）
- Richでテーブル表示
- プログレスバー付き
```

### 🎯 推奨フレームワーク選択

| 用途 | 推奨フレームワーク | 理由 |
|------|------------------|------|
| 小規模CLI | Typer (Python) | シンプル、型安全 |
| 中規模CLI | Commander (Node.js) | 柔軟、エコシステム豊富 |
| 大規模CLI | Cobra (Go) | 高速、バイナリ配布 |
| データ処理 | Typer + Rich | 美しい出力、テーブル |
| DevOps ツール | Cobra (Go) | クロスプラットフォーム |

---

## まとめ

### CLI開発の成功要因

1. **優れた設計**
   - UNIX哲学に従う
   - ユーザビリティを重視
   - 一貫性のあるコマンド構造

2. **堅牢な実装**
   - レイヤードアーキテクチャ
   - 適切なエラーハンドリング
   - 包括的なテスト

3. **優れたUX**
   - 美しい出力（カラー、テーブル）
   - インタラクティブプロンプト
   - 分かりやすいエラーメッセージ

4. **適切な配布**
   - npm / PyPI への公開
   - バイナリ配布（必要に応じて）
   - 自動更新機構

### リソース

- **ガイド**: `guides/` - 詳細な実装ガイド
- **テンプレート**: `templates/` - すぐに使えるテンプレート
- **ベストプラクティス**: `best-practices/` - チェックリスト、テスト、配布

---

*プロフェッショナルなCLIツールで、開発者体験を向上させましょう。*

_Last updated: 2026-01-03_
