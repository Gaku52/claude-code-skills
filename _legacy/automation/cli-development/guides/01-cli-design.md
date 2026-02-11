# 🎨 CLI 設計原則ガイド

> **目的**: 使いやすく、保守性の高い CLI ツールを設計するための原則とベストプラクティスを習得する

## 📚 目次

1. [CLI 設計哲学](#cli-設計哲学)
2. [コマンド設計](#コマンド設計)
3. [引数とオプション](#引数とオプション)
4. [出力設計](#出力設計)
5. [エラーハンドリング](#エラーハンドリング)
6. [設定管理](#設定管理)

---

## CLI 設計哲学

### UNIX 哲学

**1. Do One Thing Well**
```bash
# ❌ 悪い例: すべてを1つのコマンドで
mytool --create --name myapp --template react --install --deploy

# ✅ 良い例: 機能ごとにコマンド分割
mytool create myapp --template react
mytool install
mytool deploy
```

**2. Composable（組み合わせ可能）**
```bash
# パイプで連携できる設計
mytool list --format json | jq '.[] | select(.active == true)'
mytool search "query" | grep -i "keyword"
mytool export --format csv > data.csv
```

**3. Silent Success（成功時は静か）**
```bash
# ❌ 不必要な出力
$ mytool update
Updating...
Processing item 1...
Processing item 2...
...
Update complete!
Success!

# ✅ 必要最小限
$ mytool update
# 何も出力しない（成功）

# 詳細が必要なら --verbose
$ mytool update --verbose
Updating 42 items...
✓ Complete
```

### CLI のユーザビリティ原則

**1. 発見可能性（Discoverability）**
```bash
# ヘルプを常に提供
mytool --help
mytool create --help

# サブコマンドの一覧
mytool
# Available commands:
#   create    Create a new project
#   list      List all projects
#   delete    Delete a project

# typo に対するサジェスト
$ mytool crate myapp
Unknown command: crate
Did you mean: create?
```

**2. 一貫性（Consistency）**
```bash
# ❌ 一貫性がない
mytool create --name myapp
mytool remove myapp        # --name がない
mytool list-all --verbose  # ハイフンの有無

# ✅ 一貫性がある
mytool create myapp --template react
mytool delete myapp
mytool list --all --verbose
```

**3. 安全性（Safety）**
```bash
# 破壊的操作には確認
$ mytool delete myapp
Are you sure you want to delete 'myapp'? (y/N)

# --force で確認スキップ可能に
$ mytool delete myapp --force

# Dry Run オプション
$ mytool deploy --dry-run
Would deploy:
  - app.js
  - index.html
  - styles.css
```

---

## コマンド設計

### コマンド構造

**基本形**:
```
<tool> <command> <subcommand> [arguments] [options]
```

**例**:
```bash
# シンプル
git commit -m "message"

# サブコマンド
docker container ls
npm run build

# 複数階層
kubectl get pods --namespace production
aws s3 cp file.txt s3://bucket/
```

### コマンド命名規則

**動詞ベース（CRUD操作）**:
```bash
mytool create <name>    # Create
mytool list             # Read
mytool update <name>    # Update
mytool delete <name>    # Delete

# その他の動詞
mytool start <service>
mytool stop <service>
mytool restart <service>
mytool deploy <target>
mytool migrate
mytool rollback
```

**名詞ベース（リソース管理）**:
```bash
# Docker スタイル
mytool container ls
mytool container rm <id>
mytool image pull <name>

# kubectl スタイル
mytool get pods
mytool describe pod <name>
mytool delete deployment <name>
```

### サブコマンド vs オプション

**サブコマンド**: 異なる機能
```bash
git commit      # コミット作成
git push        # リモートに送信
git pull        # リモートから取得

npm install     # パッケージインストール
npm run         # スクリプト実行
npm test        # テスト実行
```

**オプション**: 振る舞いの変更
```bash
git commit -m "message"          # メッセージ指定
git commit --amend               # 最後のコミット修正
git log --oneline --graph        # 出力形式変更

npm install --save-dev           # devDependencies に追加
npm install --global             # グローバルインストール
```

**判断基準**:
| 用途 | サブコマンド | オプション |
|------|-------------|-----------|
| 新しい機能 | ✅ | ❌ |
| 振る舞いの変更 | ❌ | ✅ |
| 出力形式の変更 | ❌ | ✅ |
| リソースタイプ | ✅ | ❌ |

---

## 引数とオプション

### 位置引数（Positional Arguments）

**必須引数**:
```bash
# 1つの引数
mytool create <name>

# 複数の引数
mytool copy <source> <destination>

# 可変長引数
mytool delete <file1> [file2] [file3...]
```

**引数の順序**:
```bash
# ❌ 分かりにくい
mytool copy --recursive source dest

# ✅ 分かりやすい
mytool copy source dest --recursive
```

### オプション（Options/Flags）

**短縮形と長形式**:
```bash
# 両方提供するのがベストプラクティス
mytool create myapp -t react        # 短縮形（よく使う）
mytool create myapp --template react # 長形式（分かりやすい）

# 一般的な短縮形
-h, --help       # ヘルプ
-v, --version    # バージョン
-V, --verbose    # 詳細出力
-q, --quiet      # 静かに実行
-f, --force      # 強制実行
-y, --yes        # 確認スキップ
-n, --dry-run    # ドライラン
-o, --output     # 出力先
```

**真偽値フラグ**:
```bash
# フラグの有無で判断
mytool build --watch        # watch有効
mytool build                # watch無効

# 明示的に無効化
mytool build --no-minify    # minify無効
```

**値を取るオプション**:
```bash
# 値が必須
mytool create myapp --template <template>
mytool create myapp -t <template>

# デフォルト値あり
mytool create myapp --port 3000  # デフォルトは8080

# 複数値
mytool lint --ignore node_modules --ignore dist
mytool lint --ignore node_modules,dist
```

### オプションのグループ化

**排他的オプション**:
```bash
# 同時に指定できない
mytool deploy --staging      # ステージング環境
mytool deploy --production   # 本番環境

# エラー
$ mytool deploy --staging --production
Error: Cannot specify both --staging and --production
```

**必須オプション**:
```bash
# 必須オプション（引数にすることを検討）
mytool login --username <user> --password <pass>

# ❌ 必須オプションが多すぎる
mytool create --name myapp --template react --dir ./projects --git-init

# ✅ 引数 + オプション
mytool create myapp --template react
```

---

## 出力設計

### 標準出力と標準エラー出力

**stdout（標準出力）**: データ出力
```bash
# パイプで処理できるデータ
mytool list --format json | jq '.[] | .name'
mytool export > data.csv
```

**stderr（標準エラー出力）**: メッセージ・エラー
```bash
# エラーメッセージ
$ mytool create myapp 2> errors.log
Creating myapp...  # stderr
✓ Complete        # stderr

# データは stdout へ
$ mytool list > projects.txt
```

**実装例（Node.js）**:
```typescript
// データ出力: stdout
console.log(JSON.stringify(data))

// メッセージ: stderr
console.error('Creating project...')
console.error('✓ Complete')
```

**実装例（Python）**:
```python
import sys

# データ出力: stdout
print(json.dumps(data))

# メッセージ: stderr
print('Creating project...', file=sys.stderr)
print('✓ Complete', file=sys.stderr)
```

### 出力形式

**人間向け（デフォルト）**:
```bash
$ mytool list
Projects:
  myapp      (React, TypeScript)
  dashboard  (Vue, JavaScript)
  api        (Node.js, TypeScript)
```

**機械向け（--format オプション）**:
```bash
# JSON
$ mytool list --format json
[
  {"name": "myapp", "framework": "React", "lang": "TypeScript"},
  {"name": "dashboard", "framework": "Vue", "lang": "JavaScript"}
]

# CSV
$ mytool list --format csv
name,framework,lang
myapp,React,TypeScript
dashboard,Vue,JavaScript

# YAML
$ mytool list --format yaml
- name: myapp
  framework: React
  lang: TypeScript
```

### カラー出力

**カラー使用の原則**:
```bash
# 成功: 緑
✅ Project created successfully

# エラー: 赤
❌ Failed to create project

# 警告: 黄
⚠️  Warning: No .gitignore found

# 情報: 青
ℹ️  Installing dependencies...

# 重要: 太字
Project: myapp
```

**TTY 判定**:
```typescript
import chalk from 'chalk'

// TTY（ターミナル）の場合のみカラー有効
const isInteractive = process.stdout.isTTY

if (isInteractive) {
  console.log(chalk.green('Success!'))
} else {
  console.log('Success!')  // パイプ時はプレーンテキスト
}

// または --no-color オプション
program.option('--no-color', 'Disable color output')
```

### プログレス表示

**スピナー（短い処理）**:
```bash
⠋ Installing dependencies...
```

**プログレスバー（長い処理）**:
```bash
Downloading [████████████░░░░░░░░] 60% (120/200 MB)
```

**実装例（Node.js）**:
```typescript
import ora from 'ora'

const spinner = ora('Installing dependencies...').start()

// 処理実行
await install()

spinner.succeed('Dependencies installed!')
```

**実装例（Python）**:
```python
from rich.progress import track
import time

for i in track(range(100), description="Processing..."):
    time.sleep(0.01)
```

---

## エラーハンドリング

### エラーメッセージの設計

**良いエラーメッセージの要素**:
1. 何が起きたか（What）
2. なぜ起きたか（Why）
3. どう直すか（How）

**❌ 悪い例**:
```bash
$ mytool deploy
Error
```

**✅ 良い例**:
```bash
$ mytool deploy
❌ Deployment failed

Reason: No build files found in ./dist

Suggestion: Run 'mytool build' before deploying
```

### エラーの種類

**ユーザーエラー（終了コード 1）**:
```bash
# 引数不足
$ mytool create
Error: Missing required argument: <name>
Usage: mytool create <name> [options]

# 無効な値
$ mytool create myapp --port invalid
Error: Invalid value for --port: 'invalid'
Expected: number between 1 and 65535

# ファイルが見つからない
$ mytool build
Error: Configuration file not found: mytool.config.js
Did you mean: mytool init
```

**システムエラー（終了コード 2）**:
```bash
# 権限エラー
$ mytool install
Error: Permission denied: /usr/local/bin
Suggestion: Run with sudo or choose a different directory

# ディスク容量不足
$ mytool build
Error: Not enough disk space
Required: 500 MB, Available: 100 MB
```

**実装例**:
```typescript
import chalk from 'chalk'

class CLIError extends Error {
  constructor(
    message: string,
    public suggestion?: string,
    public exitCode: number = 1
  ) {
    super(message)
    this.name = 'CLIError'
  }
}

function handleError(error: Error) {
  if (error instanceof CLIError) {
    console.error(chalk.red(`❌ ${error.message}`))
    if (error.suggestion) {
      console.error(chalk.yellow(`\nSuggestion: ${error.suggestion}`))
    }
    process.exit(error.exitCode)
  } else {
    console.error(chalk.red('Unexpected error:'))
    console.error(error)
    process.exit(2)
  }
}

// 使用例
try {
  if (!projectName) {
    throw new CLIError(
      'Missing required argument: <name>',
      "Usage: mytool create <name>"
    )
  }
} catch (error) {
  handleError(error)
}
```

### 終了コード

| コード | 意味 |
|--------|------|
| 0 | 成功 |
| 1 | ユーザーエラー（引数不正、設定ミスなど） |
| 2 | システムエラー（権限、ネットワークなど） |
| 130 | Ctrl+C で中断 |

```typescript
// 成功
process.exit(0)

// ユーザーエラー
process.exit(1)

// システムエラー
process.exit(2)
```

---

## 設定管理

### 設定ファイルの優先順位

```
1. コマンドライン引数（最優先）
2. 環境変数
3. プロジェクト設定ファイル（./mytool.config.js）
4. ユーザー設定ファイル（~/.mytoolrc）
5. デフォルト値
```

**実装例**:
```typescript
import { cosmiconfigSync } from 'cosmiconfig'

const explorer = cosmiconfigSync('mytool')

function loadConfig(options: any) {
  // 1. コマンドライン引数
  const cliConfig = options

  // 2. 環境変数
  const envConfig = {
    port: process.env.MYTOOL_PORT,
    apiKey: process.env.MYTOOL_API_KEY
  }

  // 3. 設定ファイル
  const fileConfig = explorer.search()?.config || {}

  // 4. デフォルト
  const defaultConfig = {
    port: 8080,
    verbose: false
  }

  // マージ（優先順位順）
  return {
    ...defaultConfig,
    ...fileConfig,
    ...envConfig,
    ...cliConfig
  }
}
```

### 設定ファイル形式

**JSON（シンプル）**:
```json
{
  "template": "react",
  "port": 3000,
  "features": ["eslint", "prettier"]
}
```

**YAML（読みやすい）**:
```yaml
template: react
port: 3000
features:
  - eslint
  - prettier
```

**JavaScript/TypeScript（プログラマブル）**:
```typescript
// mytool.config.ts
import { defineConfig } from 'mytool'

export default defineConfig({
  template: 'react',
  port: 3000,
  features: ['eslint', 'prettier'],
  // 関数も使える
  beforeBuild: () => {
    console.log('Building...')
  }
})
```

### 環境変数

**命名規則**:
```bash
# プレフィックス + アンダースコア + 大文字
MYTOOL_API_KEY="xxx"
MYTOOL_PORT="3000"
MYTOOL_VERBOSE="true"

# 使用例
$ MYTOOL_PORT=4000 mytool start
```

**実装例**:
```typescript
import dotenv from 'dotenv'

// .env ファイル読み込み
dotenv.config()

const config = {
  apiKey: process.env.MYTOOL_API_KEY,
  port: parseInt(process.env.MYTOOL_PORT || '8080'),
  verbose: process.env.MYTOOL_VERBOSE === 'true'
}
```

---

## まとめ

### CLI 設計チェックリスト

**基本原則**:
- [ ] Do One Thing Well
- [ ] Composable（パイプで連携可能）
- [ ] Silent Success（成功時は静か）

**コマンド設計**:
- [ ] 動詞ベースの命名
- [ ] サブコマンドで機能分割
- [ ] 一貫性のある構造

**引数とオプション**:
- [ ] 位置引数は必須のものだけ
- [ ] 短縮形と長形式の両方提供
- [ ] デフォルト値の提供

**出力設計**:
- [ ] stdout にデータ、stderr にメッセージ
- [ ] --format オプションで機械可読形式
- [ ] TTY 判定でカラー制御

**エラーハンドリング**:
- [ ] What, Why, How を含むエラーメッセージ
- [ ] 適切な終了コード
- [ ] サジェスト機能

**設定管理**:
- [ ] 設定ファイルのサポート
- [ ] 環境変数のサポート
- [ ] 明確な優先順位

---

## 参考: 優れた CLI の例

### Git
```bash
# シンプルで一貫性がある
git add .
git commit -m "message"
git push origin main

# サブコマンドで機能分割
git branch
git remote
git log

# 豊富なオプション
git log --oneline --graph --all
```

### Docker
```bash
# 名詞ベースのサブコマンド
docker container ls
docker image pull nginx
docker network create mynet

# 一貫性のあるオプション
docker run --name myapp --port 3000:3000 nginx
```

### npm
```bash
# 動詞ベースのコマンド
npm install
npm run build
npm test

# グローバル vs ローカル
npm install --global
npm install --save-dev
```

---

## 次のステップ

1. **02-nodejs-cli.md**: Node.js CLI 実装ガイド（Commander、Inquirer）
2. **03-distribution.md**: CLI 配布・パッケージングガイド

---

*使いやすい CLI ツールで開発者体験を向上させましょう。*
