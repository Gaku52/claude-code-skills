# Claude Code ── CLI、エージェント、MCP

> Anthropic公式のCLI型AIコーディングツール「Claude Code」の全機能を理解し、エージェントモードとMCPプロトコルを活用した高度な開発ワークフローを構築する。

---

## この章で学ぶこと

1. **Claude Codeの基本操作** ── インストールからCLI操作、CLAUDE.mdによるプロジェクト設定までを習得する
2. **エージェントモードの活用** ── 自律的にタスクを完遂するエージェントの設計と運用を学ぶ
3. **MCPによるツール拡張** ── Model Context Protocolでカスタムツールを接続し、開発環境を拡張する
4. **実運用パターン** ── CI/CDパイプライン統合、チーム運用、大規模プロジェクト対応の実践知を身につける

---

## 1. Claude Codeの基本アーキテクチャ

### 1.1 システム構成

```
┌──────────────────────────────────────────────────────┐
│                  Claude Code アーキテクチャ            │
│                                                      │
│  ┌──────────┐     ┌───────────────────────────────┐  │
│  │ ターミナル │     │      Claude Code CLI          │  │
│  │ (ユーザー) │────►│                               │  │
│  └──────────┘     │  ┌─────────┐  ┌────────────┐ │  │
│                   │  │プロンプト│  │ コンテキスト│ │  │
│                   │  │解析     │  │ 管理       │ │  │
│                   │  └────┬────┘  └─────┬──────┘ │  │
│                   │       │             │        │  │
│                   │  ┌────▼─────────────▼──────┐ │  │
│                   │  │    Tool Use Engine       │ │  │
│                   │  │  ┌─────┐┌─────┐┌──────┐│ │  │
│                   │  │  │Read ││Write││Bash  ││ │  │
│                   │  │  │File ││File ││Exec  ││ │  │
│                   │  │  └─────┘└─────┘└──────┘│ │  │
│                   │  │  ┌─────┐┌─────┐┌──────┐│ │  │
│                   │  │  │Grep ││Glob ││MCP   ││ │  │
│                   │  │  │     ││     ││Tools ││ │  │
│                   │  │  └─────┘└─────┘└──────┘│ │  │
│                   │  └────────────┬────────────┘ │  │
│                   └──────────────┼───────────────┘  │
│                                  │ API呼出          │
│                                  ▼                  │
│                   ┌───────────────────────────────┐  │
│                   │   Anthropic API (Claude)       │  │
│                   │   Claude Sonnet / Opus         │  │
│                   └───────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 1.2 主要機能マップ

```
Claude Code 機能マップ
├── 対話モード
│   ├── 通常チャット（質問・相談）
│   ├── コード生成（仕様→実装）
│   └── デバッグ支援（エラー→修正）
├── エージェントモード
│   ├── 自律タスク実行
│   ├── マルチファイル編集
│   └── テスト実行→修正ループ
├── ツール連携
│   ├── File System（Read/Write/Glob/Grep）
│   ├── Bash（コマンド実行）
│   └── MCP（外部ツール）
├── プロジェクト設定
│   ├── CLAUDE.md（指示ファイル）
│   ├── .claude/settings.json
│   └── 権限管理
└── ワークフロー統合
    ├── Git連携
    ├── GitHub PR/Issue
    └── CI/CDパイプライン
```

### 1.3 Claude Codeの内部処理フロー

```
ユーザー入力
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. プロンプト解析                         │
│    - CLAUDE.md の読み込みと適用           │
│    - 会話履歴のコンテキスト構築           │
│    - システムプロンプトのマージ           │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 2. API リクエスト構築                     │
│    - トークン数の計算                     │
│    - ツール定義の付与                     │
│    - モデル選択（Sonnet/Opus）            │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 3. レスポンス処理                         │
│    ├── テキスト応答 → 表示               │
│    ├── Tool Use → ツール実行             │
│    │   ├── 権限チェック                  │
│    │   ├── 実行・結果取得               │
│    │   └── 結果をAPIに再送信            │
│    └── 終了判定                          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 4. ループ制御                             │
│    - ツール実行後に再度API呼び出し       │
│    - 最大反復回数のチェック              │
│    - ユーザー承認が必要な操作の確認      │
└─────────────────────────────────────────┘
```

---

## 2. 基本操作

### コード例1: インストールと初期設定

```bash
# インストール（npm経由）
npm install -g @anthropic-ai/claude-code

# 初回起動（認証が必要）
claude

# バージョン確認
claude --version

# ヘルプ表示
claude --help

# 非対話モードで実行
claude -p "package.jsonの依存関係を一覧にして"

# ファイルをパイプで渡す
cat error.log | claude -p "このエラーログを分析して原因を特定して"

# 特定ディレクトリで実行
claude --cwd /path/to/project "テストを実行して失敗しているものを修正して"

# JSON形式で出力（スクリプト連携用）
claude -p "package.jsonのdependenciesを一覧にして" --output-format json

# 特定のモデルを指定して実行
claude --model claude-sonnet-4-20250514 -p "コードレビューして"

# 会話の継続（resume）
claude --resume  # 最後の会話を継続
claude --resume <session-id>  # 特定セッションを継続
```

### コード例2: CLAUDE.mdによるプロジェクト設定

```markdown
# CLAUDE.md - プロジェクト固有のAI指示

## プロジェクト概要
ECプラットフォームのバックエンドAPI。Python 3.12 + FastAPI。

## コーディング規約
- 型ヒントは必須。Any型は禁止
- docstringはGoogle style
- テストはpytestで書く。カバレッジ80%以上
- import順: stdlib → third-party → local（isortで管理）
- エラーハンドリングはResult型（returns ライブラリ）

## アーキテクチャ
- Clean Architecture: domain/ → usecase/ → infra/ → presentation/
- ドメイン層は外部依存なし
- DI（依存注入）はdependency-injectorを使用

## データベース
- PostgreSQL 16 + SQLAlchemy 2.0
- マイグレーションはAlembic
- テスト用DBはSQLite in-memory

## テスト実行
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

## やってはいけないこと
- .envファイルを読んだり変更したりしない
- main ブランチに直接コミットしない
- 既存のテストを削除しない
```

### コード例3: CLAUDE.mdの階層構成

```
プロジェクトルート/
├── CLAUDE.md                    # プロジェクト全体の規約
├── src/
│   ├── CLAUDE.md                # ソースコード固有のルール
│   ├── domain/
│   │   └── CLAUDE.md            # ドメイン層の制約
│   └── presentation/
│       └── CLAUDE.md            # API層の規約
├── tests/
│   └── CLAUDE.md                # テスト固有のルール
└── .claude/
    └── settings.json            # ツール権限設定
```

```markdown
# src/domain/CLAUDE.md

## ドメイン層の厳格なルール
- 外部ライブラリのimportは絶対に禁止
- フレームワーク依存のコードを書かない
- データベースやHTTPの概念を持ち込まない
- 全ての値オブジェクトは不変（frozen=True）
- ドメインイベントで副作用を表現する
- 例外は domain/exceptions.py に定義されたもののみ使用
```

### コード例4: 日常的な対話操作

```bash
# バグ修正
claude "tests/test_order.py の test_cancel_shipped_order が
       失敗している。原因を調べて修正して"

# リファクタリング
claude "src/services/payment.py が300行を超えている。
       単一責任の原則に従って分割して"

# 新機能実装
claude "以下の仕様でクーポン機能を実装して:
       - クーポンコード: 英数字8桁
       - 割引タイプ: 定額 or 定率
       - 有効期限あり
       - 1ユーザー1回のみ使用可能
       テストも含めてお願い"

# コードレビュー
claude "git diff main...HEAD の変更をレビューして。
       セキュリティ、パフォーマンス、保守性の観点で"

# ドキュメント生成
claude "src/api/ 配下のエンドポイントを分析して
       OpenAPI仕様書を生成して。既存のdocs/api.yamlを更新"

# 依存関係の分析
claude "このプロジェクトの依存関係グラフを分析して
       循環依存がないか確認し、あれば解消案を提示して"
```

### コード例5: エージェントモードの活用

```bash
# 複雑なタスクを自律的に実行
claude "以下のステップでAPIのバージョンアップを行って:
1. OpenAPI仕様書(docs/api.yaml)を読んで現状を把握
2. 全エンドポイントにv2プレフィックスを追加
3. v1は後方互換のため残す（v2にリダイレクト）
4. 全テストを更新して通ることを確認
5. CHANGELOG.mdを更新"

# Claude Codeは以下を自律的に実行:
# - ファイルを読んで現状を分析
# - 必要な変更を計画
# - コードを修正
# - テストを実行
# - 失敗したら修正を繰り返す
# - 最終確認してレポート
```

### コード例6: サブエージェント（Task）の活用

```bash
# Task ツールを活用した並列調査
claude "以下の3つを並行して調査して:
1. src/services/ の全サービスクラスのメソッド一覧
2. tests/ のカバレッジが低いモジュール上位5つ
3. package.json の脆弱性のある依存パッケージ
それぞれの結果をまとめて報告して"

# Claude Codeは内部でTaskサブエージェントを生成し
# 並行して調査を実行する
```

### コード例7: MCP（Model Context Protocol）の設定

```jsonc
// .claude/settings.json - MCP サーバー設定
{
  "mcpServers": {
    // PostgreSQLへの直接アクセス
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/mydb"
      }
    },

    // GitHubリポジトリ操作
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },

    // Slack通知
    "slack": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-slack"],
      "env": {
        "SLACK_BOT_TOKEN": "${SLACK_BOT_TOKEN}"
      }
    },

    // Playwright（ブラウザテスト）
    "playwright": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-playwright"]
    },

    // Sentry（エラー監視）
    "sentry": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-sentry"],
      "env": {
        "SENTRY_AUTH_TOKEN": "${SENTRY_AUTH_TOKEN}",
        "SENTRY_ORG": "my-org"
      }
    }
  }
}
```

---

## 3. 高度な活用パターン

### 3.1 MCPツール連携のフロー

```
┌──────────────────────────────────────────────────────────┐
│              MCP連携ワークフロー例                         │
│                                                          │
│  ユーザー: "Issue #42を修正してPRを作成して"                │
│                                                          │
│  ┌─────────┐  MCP:GitHub   ┌─────────┐                  │
│  │Claude   │──────────────►│GitHub   │ Issueの内容を取得 │
│  │Code     │◄──────────────│API      │                  │
│  │         │               └─────────┘                  │
│  │         │  Tool:Read                                  │
│  │         │──────────────► ソースコードを読む              │
│  │         │                                             │
│  │         │  Tool:Write                                  │
│  │         │──────────────► 修正コードを書き込む            │
│  │         │                                             │
│  │         │  Tool:Bash                                   │
│  │         │──────────────► テスト実行                     │
│  │         │                                             │
│  │         │  MCP:GitHub   ┌─────────┐                  │
│  │         │──────────────►│GitHub   │ PR作成            │
│  │         │◄──────────────│API      │                  │
│  └─────────┘               └─────────┘                  │
│                                                          │
│  結果: "PR #123を作成しました: https://..."                │
└──────────────────────────────────────────────────────────┘
```

### 3.2 カスタムMCPサーバーの構築

```typescript
// custom-mcp-server.ts
// 社内システムと連携するカスタムMCPサーバーの例

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server(
  { name: "internal-tools", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// 社内チケットシステムとの連携
server.setRequestHandler("tools/list", async () => ({
  tools: [
    {
      name: "get_ticket",
      description: "社内チケットシステムからチケット情報を取得",
      inputSchema: {
        type: "object",
        properties: {
          ticket_id: { type: "string", description: "チケットID" }
        },
        required: ["ticket_id"]
      }
    },
    {
      name: "update_ticket_status",
      description: "チケットのステータスを更新",
      inputSchema: {
        type: "object",
        properties: {
          ticket_id: { type: "string" },
          status: {
            type: "string",
            enum: ["open", "in_progress", "review", "done"]
          },
          comment: { type: "string" }
        },
        required: ["ticket_id", "status"]
      }
    },
    {
      name: "search_wiki",
      description: "社内Wikiを検索",
      inputSchema: {
        type: "object",
        properties: {
          query: { type: "string", description: "検索クエリ" },
          category: { type: "string", description: "カテゴリ" }
        },
        required: ["query"]
      }
    }
  ]
}));

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "get_ticket":
      const ticket = await fetchFromInternalAPI(
        `/tickets/${args.ticket_id}`
      );
      return {
        content: [{ type: "text", text: JSON.stringify(ticket) }]
      };

    case "update_ticket_status":
      await updateInternalAPI(
        `/tickets/${args.ticket_id}`,
        { status: args.status, comment: args.comment }
      );
      return {
        content: [{ type: "text", text: "ステータスを更新しました" }]
      };

    case "search_wiki":
      const results = await searchInternalWiki(args.query, args.category);
      return {
        content: [{ type: "text", text: JSON.stringify(results) }]
      };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 3.3 Copilot vs Claude Code 比較

| 機能 | GitHub Copilot | Claude Code |
|------|---------------|-------------|
| 動作形式 | エディタ内補完 | CLI / エージェント |
| コンテキスト | 開いているファイル | プロジェクト全体 |
| 操作範囲 | 1ファイル内 | マルチファイル |
| ツール連携 | エディタ機能のみ | MCP / Bash / Git |
| テスト実行 | 不可 | 自律的に実行・修正 |
| Git操作 | 不可 | コミット・PR作成 |
| 最適用途 | リアルタイム補完 | 複雑なタスク自動化 |
| 料金 | $10-39/月 | API使用量ベース |

### 3.4 Copilot + Claude Code の併用戦略

| 場面 | 推奨ツール | 理由 |
|------|-----------|------|
| 関数内のロジック記述 | Copilot | リアルタイム補完が快適 |
| 新機能の設計と実装 | Claude Code | マルチファイル対応が必要 |
| バグ調査 | Claude Code | ログ分析・grep・テスト実行が必要 |
| テスト追加 | 両方 | Copilotで補完しつつ、Claude Codeで検証 |
| ドキュメント生成 | Claude Code | プロジェクト全体の理解が必要 |
| リファクタリング | Claude Code | 影響範囲分析とテスト確認が必要 |

---

## 4. CI/CD統合パターン

### 4.1 GitHub Actionsでのコードレビュー自動化

```yaml
# .github/workflows/claude-review.yml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  claude-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install Claude Code
        run: npm install -g @anthropic-ai/claude-code

      - name: Run AI Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          # 差分を取得してClaude Codeにレビューを依頼
          DIFF=$(git diff origin/main...HEAD)
          claude -p "以下のコード差分をレビューしてください。
          セキュリティ、パフォーマンス、保守性の観点で問題を指摘し、
          改善案を提示してください。マークダウン形式で出力してください。

          $DIFF" > review.md

      - name: Post Review Comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = fs.readFileSync('review.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## AI Code Review\n\n${review}`
            });
```

### 4.2 テスト生成の自動化

```yaml
# .github/workflows/claude-test-gen.yml
name: AI Test Generation

on:
  pull_request:
    paths:
      - 'src/**/*.ts'
      - 'src/**/*.py'

jobs:
  generate-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Identify Changed Files
        id: changed
        run: |
          FILES=$(git diff --name-only origin/main...HEAD | grep -E '\.(ts|py)$' | grep -v test)
          echo "files=$FILES" >> $GITHUB_OUTPUT

      - name: Generate Tests
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          claude -p "以下の変更ファイルに対してテストが不足している箇所を
          特定し、テストコードを生成してください。
          既存のテストスタイルに合わせてください。

          変更ファイル: ${{ steps.changed.outputs.files }}"
```

### 4.3 コミットメッセージの自動生成

```bash
#!/bin/bash
# scripts/ai-commit.sh
# AIによるコミットメッセージの自動生成

# ステージされた変更の差分を取得
STAGED_DIFF=$(git diff --cached)

if [ -z "$STAGED_DIFF" ]; then
    echo "ステージされた変更がありません"
    exit 1
fi

# Claude Codeにコミットメッセージを生成させる
MESSAGE=$(claude -p "以下のgit diffに基づいて、
Conventional Commitsの形式でコミットメッセージを生成してください。
1行目: type(scope): 簡潔な説明（50文字以内）
3行目以降: 変更の詳細（任意）

diff:
$STAGED_DIFF" --output-format text)

echo "生成されたコミットメッセージ:"
echo "$MESSAGE"
echo ""
read -p "このメッセージでコミットしますか？ (y/n): " CONFIRM

if [ "$CONFIRM" = "y" ]; then
    git commit -m "$MESSAGE"
    echo "コミットしました"
else
    echo "キャンセルしました"
fi
```

---

## 5. 権限管理とセキュリティ

### 5.1 権限設定の詳細

```jsonc
// .claude/settings.json - 権限管理
{
  "permissions": {
    // 許可するツールと範囲
    "allow": [
      "Read",                              // 全ファイル読み取り可
      "Write(src/**,tests/**,docs/**)",    // 特定ディレクトリのみ書き込み可
      "Bash(npm test,npm run *,pytest *)", // 特定コマンドのみ実行可
      "Grep",                              // 検索は無制限
      "Glob"                               // ファイル一覧も無制限
    ],
    // 明示的に拒否するツールと範囲
    "deny": [
      "Bash(rm -rf *)",          // 全削除禁止
      "Bash(git push --force*)", // 強制プッシュ禁止
      "Write(.env*)",            // 環境変数ファイル変更禁止
      "Write(*.pem)",            // 証明書ファイル変更禁止
      "Write(*.key)",            // 秘密鍵ファイル変更禁止
      "Read(.env*)",             // 環境変数ファイル読み取り禁止
      "Bash(curl *)",            // 外部通信禁止
      "Bash(wget *)"             // 外部ダウンロード禁止
    ]
  }
}
```

### 5.2 セキュリティベストプラクティス

```
┌──────────────────────────────────────────────────────┐
│           Claude Code セキュリティ対策                  │
│                                                      │
│  1. 最小権限の原則                                     │
│     ├── 必要なディレクトリのみ書き込み許可             │
│     ├── 実行可能なコマンドをホワイトリスト管理         │
│     └── .env, secrets は読み書き禁止                  │
│                                                      │
│  2. レビュー必須のフロー                               │
│     ├── AIの変更は必ず git diff で確認                 │
│     ├── 破壊的操作はユーザー承認を要求                 │
│     └── 本番環境への直接操作は禁止                    │
│                                                      │
│  3. 監査ログ                                          │
│     ├── Claude Codeの操作ログを保存                   │
│     ├── ~/.claude/logs/ にセッション履歴              │
│     └── チーム共有の操作ポリシーを策定                │
│                                                      │
│  4. ネットワークセキュリティ                           │
│     ├── API通信はHTTPS（TLS 1.3）                     │
│     ├── プロキシ環境にも対応                          │
│     └── VPN経由での利用を推奨                        │
└──────────────────────────────────────────────────────┘
```

### 5.3 チーム向けセキュリティポリシーテンプレート

```markdown
# Claude Code チームセキュリティポリシー

## 1. API キー管理
- 個人のAPI キーは環境変数で管理（.envに直書き禁止）
- チーム共有キーは AWS Secrets Manager 等で管理
- 月次でキーをローテーション

## 2. 操作制限
- 本番DBへの直接接続は禁止
- 本番環境のファイルシステムへのアクセスは禁止
- パッケージのインストール（npm install, pip install）は事前承認制

## 3. コードレビュー
- AI生成コードは通常のコードと同じレビュープロセスを適用
- セキュリティ関連コード（認証、暗号化等）は2名以上のレビュー必須
- AI生成コードであることをPRに明記

## 4. データ保護
- 個人情報を含むデータをプロンプトに含めない
- テスト用データは匿名化されたものを使用
- APIに送信されるコードの範囲を理解し管理する
```

---

## 6. 大規模プロジェクトでの運用

### 6.1 コンテキスト管理戦略

```
┌──────────────────────────────────────────────────────┐
│       大規模プロジェクトのコンテキスト管理              │
│                                                      │
│  課題: 100万行超のコードベースを扱う場合               │
│                                                      │
│  戦略1: CLAUDE.mdの階層化                             │
│  ┌─────────────────────────────────────┐             │
│  │ root/CLAUDE.md        (全体方針)    │             │
│  │ └── src/CLAUDE.md     (開発規約)    │             │
│  │     └── api/CLAUDE.md (API固有)    │             │
│  └─────────────────────────────────────┘             │
│                                                      │
│  戦略2: タスクの分割と委譲                            │
│  ┌─────────────────────────────────────┐             │
│  │ メインAgent: 全体計画策定           │             │
│  │ ├── SubAgent 1: フロントエンド修正  │             │
│  │ ├── SubAgent 2: バックエンド修正    │             │
│  │ └── SubAgent 3: テスト更新         │             │
│  └─────────────────────────────────────┘             │
│                                                      │
│  戦略3: /compact による会話圧縮                       │
│  ┌─────────────────────────────────────┐             │
│  │ 長い会話 → /compact → 要約された   │             │
│  │ コンテキストで続行                  │             │
│  └─────────────────────────────────────┘             │
└──────────────────────────────────────────────────────┘
```

### 6.2 モノレポでの活用

```bash
# モノレポ構成でのClaude Code活用例
# packages/
# ├── frontend/     (React)
# ├── backend/      (FastAPI)
# ├── shared/       (共通型定義)
# └── infra/        (Terraform)

# 特定パッケージに対してタスクを実行
claude --cwd packages/backend "新しいAPIエンドポイントを追加して"

# パッケージ間の整合性チェック
claude "packages/shared/types.tsの型定義と
       packages/backend/schemas.pyのPydanticモデルが
       整合しているか確認して。不整合があれば修正して"

# インフラとアプリケーションの一貫した変更
claude "新しいマイクロサービスをデプロイするために:
1. packages/infra/にECSタスク定義を追加
2. packages/backend/に新しいサービスのスケルトンを生成
3. packages/shared/にサービス間通信の型を追加
4. docker-compose.ymlにサービスを追加"
```

### 6.3 エージェント運用のベストプラクティス

```
エージェント運用の最適化指針

1. タスク粒度の設計
   ┌──────────────────────────────────────┐
   │ 大きすぎるタスク → コンテキスト溢れ   │
   │   "アプリ全体をリファクタリングして"   │
   │                                      │
   │ 小さすぎるタスク → 往復コスト増大     │
   │   "この変数名を変えて"               │
   │                                      │
   │ 最適な粒度 → 1機能・1モジュール単位   │
   │   "注文キャンセル機能を実装して       │
   │    テストも含めて"                    │
   └──────────────────────────────────────┘

2. コンテキストウィンドウの管理
   - 1セッションあたり10-15ファイルが最適
   - 50ファイル以上を扱うと途中停止のリスク
   - /compact を定期的に使用してコンテキストを圧縮

3. 並列Agent実行
   - 独立したタスクは並列Agentで同時実行
   - 同時Agent数は8-10が最適（15以上はリソース過多）
   - 依存関係のあるタスクは直列に実行

4. エラーリカバリ
   - Agentが停止した場合は --resume で再開
   - レート制限時は非生成タスク（分析・レビュー）に切り替え
   - 再試行は同じプロンプトではなく、追加コンテキストを付与
```

---

## 7. 実践ユースケース集

### 7.1 レガシーコードのモダナイゼーション

```bash
# Step 1: 現状分析
claude "src/legacy/ ディレクトリのコードを分析して:
1. 使用されている技術・パターンを一覧化
2. テストカバレッジを確認
3. 依存関係グラフを作成
4. リスクの高いモジュール（循環依存、巨大ファイル等）を特定"

# Step 2: 移行計画の策定
claude "分析結果に基づいて、以下の条件で移行計画を立てて:
- 段階的に移行（ビッグバンは不可）
- 各フェーズでテストが通ること
- 移行中も既存機能が動作すること
- Strangler Fig パターンを適用"

# Step 3: 段階的な実行
claude "移行計画のPhase 1を実行して:
- UserService クラスをClean Architectureに分割
- 既存のテストを新構造に適応
- 新旧のコードが共存できるアダプターを作成"
```

### 7.2 データベースマイグレーションの支援

```bash
# スキーマ変更の自動生成
claude "以下の要件でAlembicマイグレーションを作成して:
1. usersテーブルにemail_verifiedカラムを追加（boolean, default=false）
2. user_preferencesテーブルを新規作成
3. 既存データの移行スクリプトも含める
4. ロールバック手順も定義して"

# データ整合性チェック
claude "現在のSQLAlchemyモデル定義とDBスキーマの差分を検出して。
       models/ ディレクトリのモデルと
       alembic/versions/ の最新マイグレーションを比較して"
```

### 7.3 パフォーマンスチューニング

```bash
# ボトルネック調査
claude "以下のエンドポイントが遅い:
GET /api/v1/products?category=electronics&sort=price

1. src/api/products.py のクエリを分析
2. SQLAlchemyのクエリログから問題のSQLを特定
3. N+1クエリがないか確認
4. インデックスの提案
5. クエリ最適化の実装"

# ロードテスト結果の分析
claude "locustのテスト結果（results/load_test.csv）を分析して:
1. レスポンスタイムのp50, p95, p99を計算
2. スループットのボトルネックを特定
3. メモリリークの兆候がないか確認
4. 改善提案を優先度順にリストアップ"
```

### 7.4 セキュリティ監査

```bash
# コードベースのセキュリティスキャン
claude "セキュリティの観点でコードベースを監査して:
1. ハードコードされたシークレットの検出
2. SQLインジェクションの可能性がある箇所
3. XSS脆弱性の可能性がある箇所
4. 認証・認可のバイパスが可能な箇所
5. 依存パッケージの既知の脆弱性
各問題の深刻度（Critical/High/Medium/Low）と修正方法も示して"
```

### 7.5 APIドキュメントの自動メンテナンス

```bash
# 実装とドキュメントの同期
claude "src/api/ のエンドポイント実装と docs/api.yaml のOpenAPI仕様を比較して:
1. ドキュメントに記載がないエンドポイント
2. 実装と異なるパラメータ定義
3. レスポンスコードの不一致
不整合を全て修正して docs/api.yaml を更新して"
```

---

## 8. トラブルシューティング

### 8.1 よくあるエラーと対処法

```
┌───────────────────────────────────────────────────────┐
│          Claude Code トラブルシューティング              │
│                                                       │
│  エラー1: "Rate limit exceeded"                        │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因: APIレート制限に到達                     │      │
│  │ 対処:                                        │      │
│  │   - 数分待ってリトライ                        │      │
│  │   - Claude Max契約で制限緩和                  │      │
│  │   - 非生成タスク（分析・レビュー）に切替      │      │
│  │   - --model でより低コストモデルに変更         │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  エラー2: "Context window exceeded"                    │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因: 会話コンテキストがトークン上限を超過     │      │
│  │ 対処:                                        │      │
│  │   - /compact で会話を圧縮                     │      │
│  │   - 新しいセッションを開始                    │      │
│  │   - タスクを小さい単位に分割                  │      │
│  │   - Read時にoffset/limitで部分読み込み        │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  エラー3: "Permission denied"                          │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因: settings.jsonの権限設定                 │      │
│  │ 対処:                                        │      │
│  │   - .claude/settings.json を確認             │      │
│  │   - 必要な権限をallowに追加                   │      │
│  │   - denyルールと競合していないか確認          │      │
│  │   - グローバル設定とプロジェクト設定の優先順位 │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  エラー4: "MCP server connection failed"               │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因: MCPサーバーの起動/接続エラー            │      │
│  │ 対処:                                        │      │
│  │   - npxコマンドが正しいか確認                 │      │
│  │   - 環境変数が設定されているか確認            │      │
│  │   - MCPサーバーを単体で起動テスト             │      │
│  │   - ポート競合がないか確認                    │      │
│  └─────────────────────────────────────────────┘      │
│                                                       │
│  エラー5: "Tool execution timeout"                     │
│  ┌─────────────────────────────────────────────┐      │
│  │ 原因: Bashコマンド等のタイムアウト            │      │
│  │ 対処:                                        │      │
│  │   - タイムアウト設定を延長                    │      │
│  │   - 重いテストスイートは対象を限定            │      │
│  │   - ビルドはバックグラウンド実行を検討        │      │
│  └─────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────┘
```

### 8.2 パフォーマンス最適化

```
Claude Code 応答速度の最適化

1. プロンプトの最適化
   ├── 具体的な指示ほど処理が速い
   ├── 不必要なコンテキストを含めない
   └── 段階的な指示で1回あたりの処理量を抑える

2. モデル選択の最適化
   ├── 単純なタスク → Sonnet（高速・低コスト）
   ├── 複雑なタスク → Opus（高品質・高コスト）
   └── コード補完 → Haiku（最速・最低コスト）

3. ツール実行の最適化
   ├── Glob/Grepで事前絞り込み → Read
   ├── 大きなファイルはoffset/limitで部分読み込み
   └── 不要なBash実行を避ける

4. セッション管理
   ├── 定期的に /compact で圧縮
   ├── 長大なセッションより短いセッションの連続
   └── 関連性の薄いタスクは別セッションで実行
```

### 8.3 デバッグ手法

```bash
# Claude Codeのデバッグログを有効化
CLAUDE_CODE_DEBUG=1 claude "タスクを実行して"

# APIリクエストの詳細を確認
ANTHROPIC_LOG=debug claude -p "テスト"

# セッションログの確認
ls -la ~/.claude/logs/
# 各セッションのログが保存されている

# MCPサーバーの接続テスト
npx -y @modelcontextprotocol/server-postgres 2>&1
# エラーメッセージを確認
```

---

## アンチパターン

### アンチパターン 1: 権限設定の放置

```jsonc
// BAD: 全ツールを無制限に許可
{
  "permissions": {
    "allow": ["*"]  // 危険！何でも実行できる
  }
}

// GOOD: 必要最小限の権限を設定
{
  "permissions": {
    "allow": [
      "Read",
      "Write(src/**,tests/**)",   // src/とtests/のみ書き込み可
      "Bash(npm test,npm run *)", // 特定コマンドのみ
      "Grep",
      "Glob"
    ],
    "deny": [
      "Bash(rm -rf *)",
      "Write(.env*)",
      "Write(*.pem)"
    ]
  }
}
```

### アンチパターン 2: CLAUDE.mdの未設定

```
❌ CLAUDE.mdなしで使う問題:
   - AIがプロジェクトの規約を知らない
   - 毎回同じ説明を繰り返す必要がある
   - チームメンバー間でAIの振る舞いが不統一
   - 機密ファイルにアクセスするリスク

✅ CLAUDE.mdを適切に設定する効果:
   - 一貫したコーディングスタイル
   - プロジェクト固有の制約を自動適用
   - 「やってはいけないこと」を明示
   - チーム全体で同じAI体験を共有
```

### アンチパターン 3: 巨大タスクの一括投入

```
❌ BAD: 一度に全てを依頼
   "プロジェクト全体をTypeScript化して、テストも全部書き直して、
    ドキュメントも更新して"
   → コンテキスト溢れ、品質低下、途中停止

✅ GOOD: 段階的に実行
   Phase 1: "src/utils/ をTypeScript化して"
   Phase 2: "src/utils/ のテストを更新して"
   Phase 3: "src/services/ をTypeScript化して"
   → 各フェーズで品質確認し、問題があれば即座に対処
```

### アンチパターン 4: コンテキストの汚染

```
❌ BAD: 無関係な情報をセッションに蓄積
   - 複数の無関係なタスクを同一セッションで実行
   - 大量のログやエラーメッセージをペースト
   - 試行錯誤の失敗結果をそのまま会話に残す

✅ GOOD: クリーンなコンテキスト管理
   - タスクごとに新しいセッションを使用
   - /compact で不要な履歴を圧縮
   - エラー情報は要約して提供
   - 必要な情報だけをコンテキストに含める
```

---

## FAQ

### Q1: Claude Codeの料金体系はどうなっているか？

Claude CodeはAnthropicのAPI使用量に基づく従量課金。Claude Sonnetは入力$3/100万トークン、出力$15/100万トークン。1日の開発で平均$5-20程度（使用量による）。Claude Max契約（月額$100/$200）を利用すると、一定量のClaude Code利用が含まれる。

### Q2: オフライン環境でClaude Codeは使えるか？

Claude Code自体はAnthropicのAPIに接続する必要があるため、完全なオフライン使用は不可。ただし、MCPサーバーをローカルに立てれば、ファイル操作やコマンド実行はローカルで完結する。API通信のみがインターネット接続を必要とする。VPN経由での利用は問題ない。

### Q3: Claude CodeとCursorのどちらをメインツールにすべきか？

両者は補完的な関係にある。Cursorは「GUIでの対話的開発」に優れ、コードの視覚的な確認やリアルタイム補完が得意。Claude Codeは「CLIでの自動化・エージェント実行」に優れ、複雑なマルチステップタスクやCI/CD統合に向く。理想はCursorで日常コーディング、Claude Codeで複雑タスクの自動化という使い分け。

### Q4: CLAUDE.mdはチームで共有すべきか？

CLAUDE.mdはリポジトリにコミットしてチーム全体で共有すべき。これにより全メンバーが同じAI体験を得られ、コーディング規約やアーキテクチャの制約が自動的に適用される。ただし、個人の好みに関する設定（エディタ設定等）は `~/.claude/CLAUDE.md` に記述してグローバルに適用するとよい。

### Q5: MCPサーバーのセキュリティリスクはあるか？

MCPサーバーはローカルプロセスとして動作するため、ネットワーク露出のリスクは低い。ただし、DBやAPIへのアクセストークンを環境変数で渡す際は、`.env` ファイルの管理に注意が必要。本番環境のクレデンシャルは使用せず、開発用のアクセス権限のみを付与する。MCPサーバーのコード自体も信頼できるソースのみを使用する。

### Q6: Claude Codeのコスト最適化のコツは？

以下の戦略でコストを抑えられる。(1) 単純なタスクはSonnetモデルを指定（Opusの約1/5のコスト）、(2) プロンプトを具体的にして往復回数を減らす、(3) /compact で会話コンテキストを圧縮して入力トークンを削減、(4) CLAUDE.mdでプロジェクト情報を事前に提供してAIの問い合わせを減らす、(5) 大量のタスクはバッチ化して非対話モード（`claude -p`）で実行する。

### Q7: Claude Codeで生成したコードの著作権はどうなるか？

AnthropicのTerms of Serviceに基づき、Claude Codeで生成したコードの著作権はユーザーに帰属する。ただし、AIが学習データから既存のオープンソースコードを再現する可能性があるため、ライセンス互換性の確認は開発者の責任で行う必要がある。セキュリティ上重要なコードや特許に関わるコードは、生成後に人間が必ずレビューすべきである。

### Q8: 大規模チーム（50人以上）でClaude Codeを導入する際の注意点は？

大規模チームでは以下に注意する。(1) API使用量の予算管理（チーム/個人の上限設定）、(2) CLAUDE.mdのガバナンス（変更はチームリードの承認が必要）、(3) セキュリティポリシーの統一（.claude/settings.jsonのテンプレート化）、(4) ナレッジ共有（効果的なプロンプトパターンのチームWiki化）、(5) オンボーディングプロセスの整備（新メンバー向けClaude Code研修）。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 基本形式 | CLI型のAIエージェント、プロジェクト全体を操作可能 |
| CLAUDE.md | プロジェクト規約・制約をAIに伝える設定ファイル（階層化可能） |
| エージェント | 複雑なタスクを自律的にステップ実行（Task で並列化も可能） |
| MCP | 外部ツール（DB、GitHub、Slack等）との連携プロトコル |
| 権限管理 | 必要最小限の権限設定が重要（allow/deny の明示） |
| CI/CD統合 | GitHub ActionsでのPRレビュー・テスト生成の自動化 |
| 併用戦略 | Copilotでリアルタイム補完、Claude Codeで複雑タスク |
| 大規模対応 | コンテキスト管理、タスク分割、/compact の活用が鍵 |

---

## 次に読むべきガイド

- [02-cursor-and-windsurf.md](./02-cursor-and-windsurf.md) ── Cursor/WindsurfとのAI IDE比較
- [03-ai-coding-best-practices.md](./03-ai-coding-best-practices.md) ── AIコーディングの品質保証
- [../02-workflow/03-ai-debugging.md](../02-workflow/03-ai-debugging.md) ── Claude Codeを使ったデバッグ

---

## 参考文献

1. Anthropic, "Claude Code Documentation," 2025. https://docs.anthropic.com/en/docs/claude-code
2. Anthropic, "Model Context Protocol (MCP) Specification," 2024. https://modelcontextprotocol.io/
3. Anthropic, "Building effective agents," 2024. https://www.anthropic.com/research/building-effective-agents
4. Simon Willison, "Claude Code review," simonwillison.net, 2025. https://simonwillison.net/
5. Anthropic, "Claude Code best practices," 2025. https://docs.anthropic.com/en/docs/claude-code/best-practices
6. MCP Community, "MCP Server Registry," 2025. https://github.com/modelcontextprotocol/servers
