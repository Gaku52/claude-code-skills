# MCPエージェント

> Model Context Protocol――AIアプリケーションとツール間の標準プロトコルを使ったエージェント構築。サーバー/クライアント実装、ツール定義、リソース管理を解説する。

## この章で学ぶこと

1. MCPの設計思想とアーキテクチャ（クライアント/サーバーモデル）
2. MCPサーバーの実装（ツール・リソース・プロンプトの提供）
3. MCPクライアントの構築とエージェントへの統合パターン

---

## 1. MCPの全体像

### 1.1 MCPとは

MCP（Model Context Protocol）はAnthropicが提唱した **AIアプリケーションとツール/データソースを接続するためのオープン標準プロトコル** 。USBのように「一度実装すればどのAIアプリからも使える」標準化を目指す。

```
MCPなし (N x M 問題):
  App1 ──カスタム統合──→ Tool1
  App1 ──カスタム統合──→ Tool2
  App2 ──カスタム統合──→ Tool1  ← 全組み合わせを個別実装
  App2 ──カスタム統合──→ Tool2

MCPあり (N + M):
  App1 ──MCP──→ +---------+ ──MCP──→ Tool1
  App2 ──MCP──→ | MCP     | ──MCP──→ Tool2
                | Protocol|
                +---------+
  標準プロトコルで統一 → 実装コスト激減
```

### 1.2 アーキテクチャ

```
MCP アーキテクチャ

+------------------+                   +------------------+
|   MCP Host       |                   |   MCP Server     |
|  (AIアプリ)      |                   |  (ツールプロバイダ)|
|                  |     JSON-RPC      |                  |
|  +-----------+   |   over stdio/SSE  |  +-----------+   |
|  | MCP       |<========================>| MCP       |   |
|  | Client    |   |                   |  | Server    |   |
|  +-----------+   |                   |  +-----------+   |
|                  |                   |                  |
|  +-----------+   |   機能:           |  +-----------+   |
|  | LLM       |   |   - Tools        |  | ツール    |   |
|  +-----------+   |   - Resources     |  | 実装     |   |
|                  |   - Prompts       |  +-----------+   |
|  +-----------+   |   - Sampling      |  +-----------+   |
|  | Agent     |   |                   |  | データ    |   |
|  | Logic     |   |                   |  | ソース   |   |
|  +-----------+   |                   |  +-----------+   |
+------------------+                   +------------------+

ホスト: Claude Desktop, Claude Code, Cursor, Cline...
サーバー: DB接続, API連携, ファイル操作, Git操作...
```

---

## 2. MCPサーバーの実装

### 2.1 基本的なMCPサーバー

```python
# MCPサーバーの基本実装（Python SDK）
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
from mcp.server.stdio import stdio_server
import json
import sqlite3
import asyncio

# サーバーインスタンス
app = Server("company-tools")

# === ツール定義 ===
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="query_employees",
            description="社員データベースを検索する。名前、部署、役職で検索可能。",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "社員名（部分一致）"
                    },
                    "department": {
                        "type": "string",
                        "enum": ["engineering", "sales", "hr", "marketing"],
                        "description": "部署"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "最大件数（デフォルト: 10）",
                        "default": 10
                    }
                }
            }
        ),
        Tool(
            name="create_ticket",
            description="JIRAチケットを作成する。作成後のチケットIDを返す。",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "チケットタイトル"},
                    "description": {"type": "string", "description": "詳細説明"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "medium", "low"]
                    }
                },
                "required": ["title", "priority"]
            }
        )
    ]

# === ツール実行 ===
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_employees":
        conn = sqlite3.connect("/data/employees.db")
        cursor = conn.cursor()
        query = "SELECT * FROM employees WHERE 1=1"
        params = []

        if "name" in arguments:
            query += " AND name LIKE ?"
            params.append(f"%{arguments['name']}%")
        if "department" in arguments:
            query += " AND department = ?"
            params.append(arguments["department"])

        query += f" LIMIT {arguments.get('limit', 10)}"
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        return [TextContent(
            type="text",
            text=json.dumps(results, ensure_ascii=False)
        )]

    elif name == "create_ticket":
        ticket_id = create_jira_ticket(
            title=arguments["title"],
            description=arguments.get("description", ""),
            priority=arguments["priority"]
        )
        return [TextContent(
            type="text",
            text=f"チケット作成完了: {ticket_id}"
        )]

    return [TextContent(type="text", text=f"不明なツール: {name}")]

# サーバー起動
async def main():
    async with stdio_server() as (read, write):
        await app.run(read, write)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2.2 リソースの提供

```python
# MCPリソース: AIに文脈として提供するデータ
@app.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="company://docs/api-guide",
            name="API仕様書",
            description="社内APIの仕様書（OpenAPI形式）",
            mimeType="application/json"
        ),
        Resource(
            uri="company://docs/coding-standards",
            name="コーディング規約",
            description="社内Pythonコーディング規約",
            mimeType="text/markdown"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    if uri == "company://docs/api-guide":
        with open("/docs/api-spec.json") as f:
            return f.read()
    elif uri == "company://docs/coding-standards":
        with open("/docs/coding-standards.md") as f:
            return f.read()
    raise ValueError(f"不明なリソース: {uri}")
```

### 2.3 プロンプトテンプレート

```python
# MCPプロンプト: 再利用可能なプロンプトテンプレート
from mcp.types import Prompt, PromptArgument, PromptMessage

@app.list_prompts()
async def list_prompts() -> list[Prompt]:
    return [
        Prompt(
            name="code_review",
            description="コードレビューを実行する",
            arguments=[
                PromptArgument(
                    name="code",
                    description="レビュー対象のコード",
                    required=True
                ),
                PromptArgument(
                    name="language",
                    description="プログラミング言語",
                    required=False
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict) -> list[PromptMessage]:
    if name == "code_review":
        return [
            PromptMessage(
                role="user",
                content=f"""以下の{arguments.get('language', '')}コードをレビューしてください。

セキュリティ、パフォーマンス、可読性の観点で評価し、
改善点があれば具体的なコード例を提示してください。

```
{arguments['code']}
```"""
            )
        ]
```

---

## 3. MCPクライアントの実装

```python
# MCPクライアントの実装（エージェント側）
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic

async def run_mcp_agent():
    # MCPサーバーに接続
    server_params = StdioServerParameters(
        command="python",
        args=["company_tools_server.py"],
        env={"DATABASE_PATH": "/data/employees.db"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 利用可能なツールを取得
            tools_response = await session.list_tools()
            print(f"利用可能ツール: {[t.name for t in tools_response.tools]}")

            # Anthropic APIのツール形式に変換
            anthropic_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                }
                for t in tools_response.tools
            ]

            # エージェントループ
            client = anthropic.Anthropic()
            messages = [{"role": "user", "content": "エンジニアリング部門の社員を検索して"}]

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    tools=anthropic_tools,
                    messages=messages
                )

                if response.stop_reason == "end_turn":
                    print(response.content[0].text)
                    break

                # MCPサーバー経由でツール実行
                for block in response.content:
                    if block.type == "tool_use":
                        result = await session.call_tool(
                            block.name, block.input
                        )
                        # 結果をメッセージに追加
                        messages.append({
                            "role": "assistant",
                            "content": response.content
                        })
                        messages.append({
                            "role": "user",
                            "content": [{
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result.content[0].text
                            }]
                        })
```

---

## 4. MCP設定ファイル

### 4.1 Claude Desktopの設定

```json
{
  "mcpServers": {
    "company-tools": {
      "command": "python",
      "args": ["/path/to/company_tools_server.py"],
      "env": {
        "DATABASE_PATH": "/data/employees.db",
        "JIRA_API_TOKEN": "..."
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/gaku/projects"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxx"
      }
    }
  }
}
```

### 4.2 通信プロトコル

```
MCP 通信フロー (JSON-RPC 2.0)

Client → Server: initialize
Server → Client: capabilities (対応機能一覧)

Client → Server: tools/list
Server → Client: [Tool1, Tool2, ...]

Client → Server: tools/call {name: "query", arguments: {...}}
Server → Client: {content: [{type: "text", text: "結果"}]}

Client → Server: resources/list
Server → Client: [Resource1, Resource2, ...]

Client → Server: resources/read {uri: "company://docs/api"}
Server → Client: {content: "APIドキュメント内容..."}
```

---

## 5. 比較表

### 5.1 MCP vs REST API vs GraphQL

| 観点 | MCP | REST API | GraphQL |
|------|-----|----------|---------|
| 目的 | AI-ツール接続 | 一般的なWeb API | 柔軟なデータ取得 |
| プロトコル | JSON-RPC 2.0 | HTTP | HTTP |
| 通信方式 | stdio / SSE | HTTP | HTTP |
| スキーマ | JSON Schema | OpenAPI | GraphQL SDL |
| AI最適化 | ネイティブ | 別途ラッパー必要 | 別途ラッパー必要 |
| ツール発見 | list_tools | 手動 | Introspection |
| 状態管理 | セッション | ステートレス | ステートレス |

### 5.2 MCPサーバー実装言語比較

| 言語 | SDK | 成熟度 | エコシステム | おすすめ場面 |
|------|-----|--------|------------|------------|
| Python | mcp (公式) | 高 | 最大 | データ処理、ML |
| TypeScript | @modelcontextprotocol/sdk | 高 | 大 | Web統合 |
| Rust | mcp-rust | 中 | 中 | 高性能要件 |
| Go | mcp-go | 中 | 中 | インフラツール |

---

## 6. 既存MCPサーバーの活用

```
公式・コミュニティ MCPサーバー一覧

ファイルシステム:
  @modelcontextprotocol/server-filesystem
  ファイルの読み書き、ディレクトリ操作

GitHub:
  @modelcontextprotocol/server-github
  リポジトリ、Issue、PR操作

PostgreSQL:
  @modelcontextprotocol/server-postgres
  データベースクエリ

Slack:
  @modelcontextprotocol/server-slack
  メッセージ送受信、チャンネル操作

Google Drive:
  @modelcontextprotocol/server-gdrive
  ドキュメントの読み取り

Puppeteer:
  @modelcontextprotocol/server-puppeteer
  Webブラウジング、スクリーンショット
```

---

## 7. アンチパターン

### アンチパターン1: セキュリティの軽視

```python
# NG: ユーザー入力をそのままSQLに埋め込み
@app.call_tool()
async def call_tool(name, arguments):
    query = f"SELECT * FROM users WHERE name = '{arguments['name']}'"
    # SQLインジェクション脆弱性!

# OK: パラメータ化クエリを使用
@app.call_tool()
async def call_tool(name, arguments):
    query = "SELECT * FROM users WHERE name = ?"
    cursor.execute(query, (arguments["name"],))
```

### アンチパターン2: エラーを握りつぶす

```python
# NG: エラーを無視して空結果を返す
@app.call_tool()
async def call_tool(name, arguments):
    try:
        result = do_something(arguments)
        return [TextContent(type="text", text=result)]
    except Exception:
        return [TextContent(type="text", text="")]  # LLMに何が起きたか伝わらない

# OK: エラー情報を明示的に返す
@app.call_tool()
async def call_tool(name, arguments):
    try:
        result = do_something(arguments)
        return [TextContent(type="text", text=result)]
    except ValueError as e:
        return [TextContent(type="text",
                text=f"入力エラー: {e}。パラメータを確認してください。")]
    except ConnectionError:
        return [TextContent(type="text",
                text="データベース接続エラー。しばらく後に再試行してください。")]
```

---

## 8. FAQ

### Q1: MCPサーバーのデバッグ方法は？

- **MCP Inspector**: `npx @modelcontextprotocol/inspector` で対話的にテスト
- **ログ出力**: `stderr` にログを出力（stdioは通信に使われるため）
- **単体テスト**: サーバーのハンドラ関数を直接テスト

### Q2: 1つのMCPサーバーに何ツールまで載せてよいか？

推奨は **10-20ツール** まで。それ以上は複数サーバーに分割する。カテゴリ別（DB操作サーバー、メール操作サーバー等）に分けると管理しやすい。

### Q3: MCPとFunction Callingの使い分けは？

- **MCP**: 複数のAIアプリケーションでツールを共有したい場合、ツールをプロセス分離したい場合
- **Function Calling**: 単一アプリケーション内で完結する場合、最もシンプルに実装したい場合

両者は排他的ではなく、MCPサーバーのツールをFunction Callingの形式に変換して使うことが一般的。

---

## まとめ

| 項目 | 内容 |
|------|------|
| MCP | AIとツール間の標準プロトコル |
| 3つの機能 | Tools, Resources, Prompts |
| 通信方式 | stdio（ローカル）/ SSE（リモート） |
| サーバー実装 | Python/TypeScript SDK で構築 |
| クライアント | Claude Desktop, Code, Cursor等で利用 |
| 原則 | セキュリティ重視、エラー情報を正確に伝達 |

## 次に読むべきガイド

- [03-claude-agent-sdk.md](./03-claude-agent-sdk.md) — Claude Agent SDKでのMCP統合
- [../00-fundamentals/02-tool-use.md](../00-fundamentals/02-tool-use.md) — ツール使用の基礎
- [../04-production/00-deployment.md](../04-production/00-deployment.md) — MCPサーバーのデプロイ

## 参考文献

1. Model Context Protocol Specification — https://modelcontextprotocol.io/
2. MCP GitHub Organization — https://github.com/modelcontextprotocol
3. Anthropic, "Introducing the Model Context Protocol" (2024) — https://www.anthropic.com/news/model-context-protocol
