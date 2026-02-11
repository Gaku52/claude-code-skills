# Claude Code ── CLI、エージェント、MCP

> Anthropic公式のCLI型AIコーディングツール「Claude Code」の全機能を理解し、エージェントモードとMCPプロトコルを活用した高度な開発ワークフローを構築する。

---

## この章で学ぶこと

1. **Claude Codeの基本操作** ── インストールからCLI操作、CLAUDE.mdによるプロジェクト設定までを習得する
2. **エージェントモードの活用** ── 自律的にタスクを完遂するエージェントの設計と運用を学ぶ
3. **MCPによるツール拡張** ── Model Context Protocolでカスタムツールを接続し、開発環境を拡張する

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

### コード例3: 日常的な対話操作

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
```

### コード例4: エージェントモードの活用

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

### コード例5: MCP（Model Context Protocol）の設定

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

### 3.2 Copilot vs Claude Code 比較

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

### 3.3 Copilot + Claude Code の併用戦略

| 場面 | 推奨ツール | 理由 |
|------|-----------|------|
| 関数内のロジック記述 | Copilot | リアルタイム補完が快適 |
| 新機能の設計と実装 | Claude Code | マルチファイル対応が必要 |
| バグ調査 | Claude Code | ログ分析・grep・テスト実行が必要 |
| テスト追加 | 両方 | Copilotで補完しつつ、Claude Codeで検証 |
| ドキュメント生成 | Claude Code | プロジェクト全体の理解が必要 |
| リファクタリング | Claude Code | 影響範囲分析とテスト確認が必要 |

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

---

## FAQ

### Q1: Claude Codeの料金体系はどうなっているか？

Claude CodeはAnthropicのAPI使用量に基づく従量課金。Claude Sonnetは入力$3/100万トークン、出力$15/100万トークン。1日の開発で平均$5-20程度（使用量による）。Claude Max契約（月額$100/$200）を利用すると、一定量のClaude Code利用が含まれる。

### Q2: オフライン環境でClaude Codeは使えるか？

Claude Code自体はAnthropicのAPIに接続する必要があるため、完全なオフライン使用は不可。ただし、MCPサーバーをローカルに立てれば、ファイル操作やコマンド実行はローカルで完結する。API通信のみがインターネット接続を必要とする。VPN経由での利用は問題ない。

### Q3: Claude CodeとCursorのどちらをメインツールにすべきか？

両者は補完的な関係にある。Cursorは「GUIでの対話的開発」に優れ、コードの視覚的な確認やリアルタイム補完が得意。Claude Codeは「CLIでの自動化・エージェント実行」に優れ、複雑なマルチステップタスクやCI/CD統合に向く。理想はCursorで日常コーディング、Claude Codeで複雑タスクの自動化という使い分け。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 基本形式 | CLI型のAIエージェント、プロジェクト全体を操作可能 |
| CLAUDE.md | プロジェクト規約・制約をAIに伝える設定ファイル |
| エージェント | 複雑なタスクを自律的にステップ実行 |
| MCP | 外部ツール（DB、GitHub、Slack等）との連携プロトコル |
| 権限管理 | 必要最小限の権限設定が重要 |
| 併用戦略 | Copilotでリアルタイム補完、Claude Codeで複雑タスク |

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
