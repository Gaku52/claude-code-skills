# MCPエージェント

> Model Context Protocol――AIアプリケーションとツール間の標準プロトコルを使ったエージェント構築。サーバー/クライアント実装、ツール定義、リソース管理を解説する。

## この章で学ぶこと

1. MCPの設計思想とアーキテクチャ（クライアント/サーバーモデル）
2. MCPサーバーの実装（ツール・リソース・プロンプトの提供）
3. MCPクライアントの構築とエージェントへの統合パターン
4. 高度なMCPサーバー設計（認証、ロギング、ミドルウェア）
5. 複数MCPサーバーの統合と動的ツール管理
6. SSEトランスポートによるリモートMCPサーバーの構築
7. 本番運用に向けたセキュリティ・テスト・デプロイ戦略

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

### 1.3 MCPの4つの機能カテゴリ

```
MCP が提供する4つの機能

1. Tools（ツール）
   - LLMが呼び出せるアクション
   - 例: DB検索、API呼び出し、ファイル操作
   - LLMが「いつ使うか」を判断

2. Resources（リソース）
   - 文脈として提供するデータ
   - 例: ドキュメント、設定ファイル、データベーススキーマ
   - ユーザーまたはアプリが選択

3. Prompts（プロンプト）
   - 再利用可能なプロンプトテンプレート
   - 例: コードレビュー、要約、翻訳テンプレート
   - ユーザーが選択して使用

4. Sampling（サンプリング）
   - サーバーからLLMへの呼び出し要求
   - 例: サーバー側での再帰的処理
   - サーバーがクライアントのLLMを利用
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

### 2.3 動的リソースとリソーステンプレート

```python
from mcp.types import Resource, ResourceTemplate

# 動的リソース: URLパターンに基づいてリソースを生成
@app.list_resource_templates()
async def list_resource_templates() -> list[ResourceTemplate]:
    return [
        ResourceTemplate(
            uriTemplate="company://employees/{employee_id}/profile",
            name="社員プロフィール",
            description="指定された社員IDのプロフィール情報"
        ),
        ResourceTemplate(
            uriTemplate="company://projects/{project_id}/summary",
            name="プロジェクト概要",
            description="指定されたプロジェクトの概要情報"
        ),
        ResourceTemplate(
            uriTemplate="company://metrics/{date}/dashboard",
            name="日次メトリクス",
            description="指定日のダッシュボードデータ"
        )
    ]

@app.read_resource()
async def read_resource(uri: str) -> str:
    import re

    # 社員プロフィール
    match = re.match(r"company://employees/(\w+)/profile", uri)
    if match:
        employee_id = match.group(1)
        conn = sqlite3.connect("/data/employees.db")
        cursor = conn.execute(
            "SELECT * FROM employees WHERE id = ?", (employee_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return json.dumps({
                "id": row[0], "name": row[1],
                "department": row[2], "role": row[3]
            }, ensure_ascii=False)
        raise ValueError(f"社員ID {employee_id} が見つかりません")

    # プロジェクト概要
    match = re.match(r"company://projects/(\w+)/summary", uri)
    if match:
        project_id = match.group(1)
        # プロジェクト情報を取得
        return json.dumps({
            "project_id": project_id,
            "status": "active",
            "members": 12,
            "progress": "65%"
        }, ensure_ascii=False)

    raise ValueError(f"不明なリソースURI: {uri}")
```

### 2.4 プロンプトテンプレート

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
        ),
        Prompt(
            name="bug_report",
            description="バグレポートのテンプレートを生成する",
            arguments=[
                PromptArgument(
                    name="title",
                    description="バグの概要",
                    required=True
                ),
                PromptArgument(
                    name="steps",
                    description="再現手順",
                    required=True
                ),
                PromptArgument(
                    name="severity",
                    description="重要度（critical/high/medium/low）",
                    required=False
                )
            ]
        ),
        Prompt(
            name="sql_query_helper",
            description="自然言語からSQLクエリを生成する",
            arguments=[
                PromptArgument(
                    name="description",
                    description="取得したいデータの説明",
                    required=True
                ),
                PromptArgument(
                    name="tables",
                    description="利用可能なテーブル名（カンマ区切り）",
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

    elif name == "bug_report":
        severity = arguments.get("severity", "medium")
        return [
            PromptMessage(
                role="user",
                content=f"""以下の情報からバグレポートを作成してください。

## バグ概要
{arguments['title']}

## 重要度
{severity}

## 再現手順
{arguments['steps']}

以下のフォーマットで出力してください:
1. タイトル
2. 環境情報
3. 再現手順（番号付き）
4. 期待される動作
5. 実際の動作
6. 影響範囲
7. 推奨される対応"""
            )
        ]

    elif name == "sql_query_helper":
        tables_info = arguments.get("tables", "不明")
        return [
            PromptMessage(
                role="user",
                content=f"""以下の要件に合うSQLクエリを生成してください。

要件: {arguments['description']}
利用可能なテーブル: {tables_info}

クエリの説明、パフォーマンスに関する注意点も添えてください。"""
            )
        ]

    raise ValueError(f"不明なプロンプト: {name}")
```

### 2.5 サーバーのライフサイクル管理

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger("mcp-server")

# リソース管理付きサーバー
class DatabaseMCPServer:
    """データベース接続を管理するMCPサーバー"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        self.server = Server("database-tools")
        self._setup_handlers()

    def _setup_handlers(self):
        """ハンドラの登録"""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name="execute_query",
                    description="読み取り専用SQLクエリを実行する（SELECT文のみ）",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "実行するSQLクエリ（SELECT文のみ）"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="list_tables",
                    description="データベース内のテーブル一覧を取得する",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="describe_table",
                    description="テーブルのスキーマ情報を取得する",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "テーブル名"
                            }
                        },
                        "required": ["table_name"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            if name == "execute_query":
                return await self._execute_query(arguments["query"])
            elif name == "list_tables":
                return await self._list_tables()
            elif name == "describe_table":
                return await self._describe_table(arguments["table_name"])
            return [TextContent(type="text", text=f"不明なツール: {name}")]

    async def _execute_query(self, query: str) -> list[TextContent]:
        """SQLクエリの実行（SELECT文のみ）"""
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return [TextContent(
                type="text",
                text="エラー: SELECT文のみ実行可能です。"
            )]

        # 危険なキーワードのチェック
        dangerous = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE"]
        for keyword in dangerous:
            if keyword in query_upper:
                return [TextContent(
                    type="text",
                    text=f"エラー: '{keyword}' を含むクエリは実行できません。"
                )]

        try:
            cursor = self.conn.execute(query)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            result = {
                "columns": columns,
                "rows": [list(row) for row in rows[:100]],  # 最大100行
                "total_rows": len(rows)
            }
            return [TextContent(
                type="text",
                text=json.dumps(result, ensure_ascii=False, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"クエリ実行エラー: {str(e)}"
            )]

    async def _list_tables(self) -> list[TextContent]:
        """テーブル一覧の取得"""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        return [TextContent(
            type="text",
            text=json.dumps({"tables": tables}, ensure_ascii=False)
        )]

    async def _describe_table(self, table_name: str) -> list[TextContent]:
        """テーブルスキーマの取得"""
        cursor = self.conn.execute(f"PRAGMA table_info({table_name})")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row[1],
                "type": row[2],
                "not_null": bool(row[3]),
                "primary_key": bool(row[5])
            })
        return [TextContent(
            type="text",
            text=json.dumps({"table": table_name, "columns": columns}, ensure_ascii=False, indent=2)
        )]

    async def run(self):
        """サーバーの起動"""
        import sqlite3
        self.conn = sqlite3.connect(self.db_path)
        logger.info(f"データベース接続: {self.db_path}")

        try:
            async with stdio_server() as (read, write):
                await self.server.run(read, write)
        finally:
            if self.conn:
                self.conn.close()
                logger.info("データベース切断")

# 起動
if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data.db"
    server = DatabaseMCPServer(db_path)
    asyncio.run(server.run())
```

---

## 3. MCPクライアントの実装

### 3.1 基本クライアント

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

### 3.2 複数MCPサーバーの統合クライアント

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import anthropic
import asyncio
from dataclasses import dataclass

@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] = None

class MultiServerMCPAgent:
    """複数のMCPサーバーを統合するエージェント"""

    def __init__(self, server_configs: list[MCPServerConfig]):
        self.server_configs = server_configs
        self.sessions: dict[str, ClientSession] = {}
        self.tool_to_server: dict[str, str] = {}
        self.all_tools: list[dict] = []

    async def connect_all(self):
        """全MCPサーバーに接続"""
        for config in self.server_configs:
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env
            )

            read_stream, write_stream = await self._create_connection(params)
            session = ClientSession(read_stream, write_stream)
            await session.initialize()

            self.sessions[config.name] = session

            # ツール一覧を取得
            tools_response = await session.list_tools()
            for tool in tools_response.tools:
                self.tool_to_server[tool.name] = config.name
                self.all_tools.append({
                    "name": tool.name,
                    "description": f"[{config.name}] {tool.description}",
                    "input_schema": tool.inputSchema
                })

        print(f"接続完了: {len(self.sessions)} サーバー, {len(self.all_tools)} ツール")

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """適切なMCPサーバーでツールを実行"""
        server_name = self.tool_to_server.get(tool_name)
        if not server_name:
            return f"エラー: ツール '{tool_name}' が見つかりません"

        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result.content[0].text

    async def run_agent_loop(self, user_message: str) -> str:
        """エージェントループの実行"""
        client = anthropic.Anthropic()
        messages = [{"role": "user", "content": user_message}]

        max_iterations = 10
        for _ in range(max_iterations):
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=self.all_tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            # ツール実行
            messages.append({
                "role": "assistant",
                "content": response.content
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = await self.call_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            if tool_results:
                messages.append({
                    "role": "user",
                    "content": tool_results
                })

        return "最大イテレーション数に達しました"

    async def _create_connection(self, params):
        """サーバー接続を作成（簡略化）"""
        # 実際の実装ではstdio_clientのコンテキストマネージャーを使用
        pass

# 使用例
async def main():
    configs = [
        MCPServerConfig(
            name="database",
            command="python",
            args=["db_server.py"],
            env={"DB_PATH": "/data/app.db"}
        ),
        MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={"GITHUB_TOKEN": "ghp_xxx"}
        ),
        MCPServerConfig(
            name="slack",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-slack"],
            env={"SLACK_TOKEN": "xoxb-xxx"}
        )
    ]

    agent = MultiServerMCPAgent(configs)
    await agent.connect_all()
    result = await agent.run_agent_loop(
        "GitHubの最新PRとSlackの未読メッセージをまとめて"
    )
    print(result)
```

### 3.3 リソースとプロンプトの活用

```python
async def use_resources_and_prompts(session: ClientSession):
    """リソースとプロンプトの活用例"""

    # リソース一覧の取得
    resources = await session.list_resources()
    for resource in resources.resources:
        print(f"リソース: {resource.name} ({resource.uri})")

    # リソースの読み取り
    api_docs = await session.read_resource("company://docs/api-guide")
    print(f"API仕様書: {api_docs.contents[0].text[:200]}...")

    # プロンプト一覧の取得
    prompts = await session.list_prompts()
    for prompt in prompts.prompts:
        print(f"プロンプト: {prompt.name} - {prompt.description}")

    # プロンプトの使用
    review_prompt = await session.get_prompt(
        "code_review",
        arguments={
            "code": "def hello(): print('world')",
            "language": "Python"
        }
    )
    print(f"生成されたプロンプト: {review_prompt.messages[0].content}")

    # リソースをLLMコンテキストに含める
    client = anthropic.Anthropic()
    coding_standards = await session.read_resource("company://docs/coding-standards")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=f"""以下のコーディング規約に従ってレビューしてください:

{coding_standards.contents[0].text}""",
        messages=[{
            "role": "user",
            "content": review_prompt.messages[0].content
        }]
    )
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

### 4.2 Claude Code の設定

```json
{
  "mcpServers": {
    "database": {
      "command": "python",
      "args": ["/path/to/db_mcp_server.py", "--db", "/data/analytics.db"],
      "env": {}
    },
    "playwright": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-playwright"],
      "env": {}
    },
    "custom-api": {
      "command": "node",
      "args": ["/path/to/api_server.js"],
      "env": {
        "API_BASE_URL": "https://api.example.com",
        "API_KEY": "sk-xxx"
      }
    }
  }
}
```

### 4.3 通信プロトコル

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

Client → Server: prompts/list
Server → Client: [Prompt1, Prompt2, ...]

Client → Server: prompts/get {name: "code_review", arguments: {...}}
Server → Client: {messages: [{role: "user", content: "..."}]}
```

### 4.4 JSON-RPCメッセージ例

```json
// ツール一覧リクエスト
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list"
}

// ツール一覧レスポンス
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "query_employees",
        "description": "社員データベースを検索する",
        "inputSchema": {
          "type": "object",
          "properties": {
            "name": {"type": "string", "description": "社員名"}
          }
        }
      }
    ]
  }
}

// ツール実行リクエスト
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "query_employees",
    "arguments": {"name": "田中", "department": "engineering"}
  }
}

// ツール実行レスポンス
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "[{\"id\": 1, \"name\": \"田中太郎\", \"dept\": \"engineering\"}]"
      }
    ]
  }
}
```

---

## 5. SSEトランスポート（リモートMCPサーバー）

### 5.1 SSEサーバーの実装

```python
# SSE（Server-Sent Events）トランスポートによるリモートMCPサーバー
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import JSONResponse
import uvicorn

app = Server("remote-tools")

# ツール定義（同じインターフェース）
@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="web_search",
            description="Web検索を実行する",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "web_search":
        # Web検索の実行
        results = await perform_web_search(
            arguments["query"],
            arguments.get("max_results", 5)
        )
        return [TextContent(type="text", text=json.dumps(results, ensure_ascii=False))]
    return [TextContent(type="text", text=f"不明なツール: {name}")]

# SSEトランスポートの設定
sse = SseServerTransport("/messages")

async def handle_sse(request):
    """SSE接続のハンドリング"""
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await app.run(
            streams[0], streams[1],
            app.create_initialization_options()
        )

async def handle_messages(request):
    """メッセージの受信"""
    await sse.handle_post_message(request.scope, request.receive, request._send)

# Starletteアプリ
starlette_app = Starlette(
    routes=[
        Route("/sse", handle_sse),
        Route("/messages", handle_messages, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8080)
```

### 5.2 SSEクライアントの接続

```python
from mcp.client.sse import sse_client

async def connect_remote_server():
    """リモートMCPサーバーへの接続"""
    async with sse_client("http://remote-server:8080/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 通常のMCPクライアントと同じインターフェース
            tools = await session.list_tools()
            print(f"リモートツール: {[t.name for t in tools.tools]}")

            result = await session.call_tool(
                "web_search",
                {"query": "MCP protocol", "max_results": 3}
            )
            print(result.content[0].text)
```

---

## 6. TypeScript MCPサーバー

### 6.1 TypeScript実装

```typescript
// TypeScriptでのMCPサーバー実装
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  ListToolsRequestSchema,
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "analytics-server", version: "1.0.0" },
  { capabilities: { tools: {}, resources: {} } }
);

// ツール定義
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "analyze_data",
      description: "データセットの統計分析を実行する",
      inputSchema: {
        type: "object" as const,
        properties: {
          dataset: {
            type: "string",
            description: "データセット名",
          },
          metrics: {
            type: "array",
            items: { type: "string" },
            description: "計算するメトリクス（mean, median, std, etc.）",
          },
        },
        required: ["dataset"],
      },
    },
    {
      name: "generate_chart",
      description: "データからグラフを生成する",
      inputSchema: {
        type: "object" as const,
        properties: {
          chart_type: {
            type: "string",
            enum: ["bar", "line", "pie", "scatter"],
          },
          data: {
            type: "object",
            description: "グラフデータ（x軸、y軸の配列）",
          },
          title: { type: "string" },
        },
        required: ["chart_type", "data"],
      },
    },
  ],
}));

// ツール実行
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  if (name === "analyze_data") {
    // 統計分析の実行
    const results = await performAnalysis(
      args.dataset as string,
      (args.metrics as string[]) || ["mean", "median", "std"]
    );
    return {
      content: [{ type: "text", text: JSON.stringify(results, null, 2) }],
    };
  }

  if (name === "generate_chart") {
    const chartUrl = await createChart(
      args.chart_type as string,
      args.data as Record<string, unknown>,
      (args.title as string) || "Chart"
    );
    return {
      content: [
        { type: "text", text: `グラフを生成しました: ${chartUrl}` },
      ],
    };
  }

  return {
    content: [{ type: "text", text: `不明なツール: ${name}` }],
    isError: true,
  };
});

// サーバー起動
const transport = new StdioServerTransport();
await server.connect(transport);
```

---

## 7. テスト

### 7.1 MCPサーバーのユニットテスト

```python
import pytest
import json
from unittest.mock import patch, MagicMock

# ツールハンドラの直接テスト
class TestQueryEmployeesTool:
    """query_employees ツールのテスト"""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """テスト用データベース"""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT,
                department TEXT,
                role TEXT
            )
        """)
        conn.execute(
            "INSERT INTO employees VALUES (1, '田中太郎', 'engineering', 'Senior Engineer')"
        )
        conn.execute(
            "INSERT INTO employees VALUES (2, '鈴木花子', 'engineering', 'Manager')"
        )
        conn.execute(
            "INSERT INTO employees VALUES (3, '佐藤一郎', 'sales', 'Sales Rep')"
        )
        conn.commit()
        conn.close()
        return str(db_path)

    @pytest.mark.asyncio
    async def test_search_by_name(self, mock_db):
        """名前で検索できる"""
        with patch("__main__.sqlite3.connect") as mock_connect:
            mock_connect.return_value = sqlite3.connect(mock_db)
            result = await call_tool("query_employees", {"name": "田中"})
            data = json.loads(result[0].text)
            assert len(data) == 1
            assert "田中" in str(data)

    @pytest.mark.asyncio
    async def test_search_by_department(self, mock_db):
        """部署で検索できる"""
        with patch("__main__.sqlite3.connect") as mock_connect:
            mock_connect.return_value = sqlite3.connect(mock_db)
            result = await call_tool("query_employees", {"department": "engineering"})
            data = json.loads(result[0].text)
            assert len(data) == 2

    @pytest.mark.asyncio
    async def test_limit(self, mock_db):
        """件数制限が機能する"""
        with patch("__main__.sqlite3.connect") as mock_connect:
            mock_connect.return_value = sqlite3.connect(mock_db)
            result = await call_tool("query_employees", {"limit": 1})
            data = json.loads(result[0].text)
            assert len(data) <= 1

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """不明なツールの場合"""
        result = await call_tool("nonexistent_tool", {})
        assert "不明なツール" in result[0].text

class TestCreateTicketTool:
    """create_ticket ツールのテスト"""

    @pytest.mark.asyncio
    async def test_create_ticket(self):
        """チケットが作成される"""
        with patch("__main__.create_jira_ticket") as mock_jira:
            mock_jira.return_value = "PROJ-123"
            result = await call_tool("create_ticket", {
                "title": "テストチケット",
                "priority": "high",
                "description": "テスト用"
            })
            assert "PROJ-123" in result[0].text
            mock_jira.assert_called_once()
```

### 7.2 MCP Inspectorによるインタラクティブテスト

```bash
# MCP Inspector のインストールと使用
npx @modelcontextprotocol/inspector

# 特定のサーバーに接続してテスト
npx @modelcontextprotocol/inspector python company_tools_server.py

# Inspector の機能:
# - ツール一覧の確認
# - ツールの対話的実行
# - リソースの読み取り
# - プロンプトの取得
# - サーバーのログ確認
```

### 7.3 統合テスト

```python
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class TestMCPServerIntegration:
    """MCPサーバーの統合テスト"""

    @pytest.fixture
    async def session(self):
        """テスト用MCPセッション"""
        params = StdioServerParameters(
            command="python",
            args=["company_tools_server.py"],
            env={"DATABASE_PATH": ":memory:"}
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @pytest.mark.asyncio
    async def test_list_tools(self, session):
        """ツール一覧が取得できる"""
        result = await session.list_tools()
        tool_names = [t.name for t in result.tools]
        assert "query_employees" in tool_names
        assert "create_ticket" in tool_names

    @pytest.mark.asyncio
    async def test_tool_schema_valid(self, session):
        """ツールスキーマが有効"""
        result = await session.list_tools()
        for tool in result.tools:
            assert tool.inputSchema is not None
            assert "type" in tool.inputSchema
            assert tool.inputSchema["type"] == "object"

    @pytest.mark.asyncio
    async def test_list_resources(self, session):
        """リソース一覧が取得できる"""
        result = await session.list_resources()
        assert len(result.resources) > 0

    @pytest.mark.asyncio
    async def test_list_prompts(self, session):
        """プロンプト一覧が取得できる"""
        result = await session.list_prompts()
        assert len(result.prompts) > 0
```

---

## 8. セキュリティ

### 8.1 入力バリデーション

```python
from pydantic import BaseModel, Field, validator
from typing import Optional
import re

class EmployeeQueryInput(BaseModel):
    """社員検索の入力バリデーション"""
    name: Optional[str] = Field(None, max_length=100)
    department: Optional[str] = Field(None)
    limit: int = Field(default=10, ge=1, le=100)

    @validator("name")
    def validate_name(cls, v):
        if v and not re.match(r"^[\w\s\-]+$", v):
            raise ValueError("不正な文字が含まれています")
        return v

    @validator("department")
    def validate_department(cls, v):
        valid_depts = {"engineering", "sales", "hr", "marketing"}
        if v and v not in valid_depts:
            raise ValueError(f"不正な部署: {v}")
        return v

# バリデーション付きツール実行
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "query_employees":
        try:
            validated = EmployeeQueryInput(**arguments)
        except ValueError as e:
            return [TextContent(
                type="text",
                text=f"入力エラー: {str(e)}"
            )]

        # バリデーション済みの値でクエリ実行
        return await execute_employee_query(validated)
```

### 8.2 認証とアクセス制御

```python
import os
import hashlib
import hmac
from datetime import datetime

class AuthenticatedMCPServer:
    """認証付きMCPサーバー"""

    def __init__(self):
        self.api_key = os.environ.get("MCP_API_KEY")
        self.allowed_tools: dict[str, list[str]] = {
            "read": ["query_employees", "list_tables", "describe_table"],
            "write": ["create_ticket", "update_employee"],
            "admin": ["execute_raw_query", "delete_record"]
        }

    def verify_request(self, request_meta: dict) -> str:
        """リクエストの認証とロール判定"""
        token = request_meta.get("auth_token")
        if not token:
            raise PermissionError("認証トークンがありません")

        # トークンからロールを判定（実際にはJWTなどを使用）
        role = self._decode_token(token)
        return role

    def check_permission(self, role: str, tool_name: str) -> bool:
        """ツールへのアクセス権限を確認"""
        for permission_level, tools in self.allowed_tools.items():
            if tool_name in tools:
                if permission_level == "read":
                    return True  # 全ロールが読み取り可能
                elif permission_level == "write":
                    return role in ["write", "admin"]
                elif permission_level == "admin":
                    return role == "admin"
        return False

    def _decode_token(self, token: str) -> str:
        """トークンのデコード（簡略化）"""
        # 実際にはJWTデコードなどを実装
        if token.startswith("admin_"):
            return "admin"
        elif token.startswith("write_"):
            return "write"
        return "read"
```

### 8.3 レート制限

```python
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    """ツール呼び出しのレート制限"""

    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls = max_calls_per_minute
        self.call_history: dict[str, list[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def check_rate_limit(self, tool_name: str) -> bool:
        """レート制限チェック"""
        async with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)

            # 古い履歴を削除
            self.call_history[tool_name] = [
                t for t in self.call_history[tool_name] if t > cutoff
            ]

            if len(self.call_history[tool_name]) >= self.max_calls:
                return False

            self.call_history[tool_name].append(now)
            return True

rate_limiter = RateLimiter(max_calls_per_minute=30)

# レート制限付きツール実行
@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if not await rate_limiter.check_rate_limit(name):
        return [TextContent(
            type="text",
            text=f"レート制限超過: ツール '{name}' は1分間に{rate_limiter.max_calls}回まで"
        )]

    # 通常のツール実行
    return await execute_tool(name, arguments)
```

---

## 9. 既存MCPサーバーの活用

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

Brave Search:
  @modelcontextprotocol/server-brave-search
  Web検索

Memory:
  @modelcontextprotocol/server-memory
  永続的なキーバリューストア

Fetch:
  @modelcontextprotocol/server-fetch
  HTTPリクエストの実行
```

### 9.1 公式サーバーの設定例

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem",
               "/Users/gaku/projects", "/Users/gaku/documents"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxx"
      }
    },
    "postgres": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres",
               "postgresql://user:pass@localhost:5432/mydb"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "BSA_xxxx"
      }
    }
  }
}
```

---

## 10. 比較表

### 10.1 MCP vs REST API vs GraphQL

| 観点 | MCP | REST API | GraphQL |
|------|-----|----------|---------|
| 目的 | AI-ツール接続 | 一般的なWeb API | 柔軟なデータ取得 |
| プロトコル | JSON-RPC 2.0 | HTTP | HTTP |
| 通信方式 | stdio / SSE | HTTP | HTTP |
| スキーマ | JSON Schema | OpenAPI | GraphQL SDL |
| AI最適化 | ネイティブ | 別途ラッパー必要 | 別途ラッパー必要 |
| ツール発見 | list_tools | 手動 | Introspection |
| 状態管理 | セッション | ステートレス | ステートレス |

### 10.2 MCPサーバー実装言語比較

| 言語 | SDK | 成熟度 | エコシステム | おすすめ場面 |
|------|-----|--------|------------|------------|
| Python | mcp (公式) | 高 | 最大 | データ処理、ML |
| TypeScript | @modelcontextprotocol/sdk | 高 | 大 | Web統合 |
| Rust | mcp-rust | 中 | 中 | 高性能要件 |
| Go | mcp-go | 中 | 中 | インフラツール |

### 10.3 トランスポート方式の比較

| 方式 | 通信形態 | レイテンシ | セキュリティ | 推奨用途 |
|------|---------|----------|------------|---------|
| stdio | ローカルプロセス | 最低 | プロセス分離 | ローカルツール |
| SSE | HTTP/リモート | 中 | HTTPS対応 | リモートサーバー |
| WebSocket | 双方向リアルタイム | 低 | WSS対応 | リアルタイム要件 |

### 10.4 MCPホスト対応状況

| ホスト | MCP対応 | stdio | SSE | 備考 |
|--------|---------|-------|-----|------|
| Claude Desktop | 公式 | 対応 | 対応 | 最も完全な対応 |
| Claude Code | 公式 | 対応 | 対応 | CLI環境 |
| Cursor | 対応 | 対応 | 一部 | IDE統合 |
| Cline | 対応 | 対応 | 一部 | VS Code拡張 |
| Continue | 対応 | 対応 | 計画中 | オープンソースIDE |

---

## 11. アンチパターン

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

### アンチパターン3: ツール説明の不足

```python
# NG: LLMが適切に使えない曖昧な説明
Tool(
    name="search",
    description="検索する",  # 何を？どうやって？
    inputSchema={"type": "object", "properties": {"q": {"type": "string"}}}
)

# OK: 具体的で明確な説明
Tool(
    name="search_employees",
    description="社員データベースを名前・部署・役職で検索する。部分一致検索に対応。結果は最大件数まで返される。",
    inputSchema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "社員名で検索（部分一致）。例: '田中'"
            },
            "department": {
                "type": "string",
                "enum": ["engineering", "sales", "hr"],
                "description": "部署で絞り込み"
            },
            "limit": {
                "type": "integer",
                "description": "最大結果数（1-100、デフォルト10）",
                "default": 10,
                "minimum": 1,
                "maximum": 100
            }
        }
    }
)
```

### アンチパターン4: 巨大なレスポンス

```python
# NG: 全データをそのまま返す
@app.call_tool()
async def call_tool(name, arguments):
    cursor.execute("SELECT * FROM huge_table")
    results = cursor.fetchall()  # 100万行
    return [TextContent(type="text", text=json.dumps(results))]
    # トークン数が膨大になりLLMのコンテキストを溢れさせる

# OK: 結果を制限して要約付きで返す
@app.call_tool()
async def call_tool(name, arguments):
    limit = min(arguments.get("limit", 50), 100)
    cursor.execute(f"SELECT * FROM huge_table LIMIT {limit}")
    results = cursor.fetchall()

    # 全体の件数も返す
    cursor.execute("SELECT COUNT(*) FROM huge_table")
    total_count = cursor.fetchone()[0]

    return [TextContent(type="text", text=json.dumps({
        "results": results,
        "returned": len(results),
        "total": total_count,
        "note": f"全{total_count}件中、上位{len(results)}件を表示"
    }, ensure_ascii=False))]
```

### アンチパターン5: stdoutへのログ出力

```python
# NG: stdoutにログを出力（MCPの通信を妨害）
print("Debug: processing request...")  # stdio通信を壊す

# OK: stderrにログを出力
import sys
print("Debug: processing request...", file=sys.stderr)

# OK: loggingモジュールを使用（stderrにリダイレクト）
import logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
logger = logging.getLogger("mcp-server")
logger.info("Processing request...")
```

---

## 12. FAQ

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

### Q4: MCPサーバーのパフォーマンス最適化は？

- **接続プーリング**: データベース接続を再利用する
- **キャッシュ**: 頻繁にアクセスするデータをメモリキャッシュ
- **非同期I/O**: asyncioを活用してI/O待ちを最小化
- **レスポンスサイズ制限**: 返却データの上限を設定
- **バッチ処理**: 複数リクエストをまとめて処理

### Q5: MCPサーバーをDockerで運用するには？

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "mcp_server.py"]
```

```yaml
# docker-compose.yml
services:
  mcp-db-server:
    build: ./mcp-db-server
    volumes:
      - ./data:/data
    environment:
      - DATABASE_PATH=/data/app.db
  mcp-api-server:
    build: ./mcp-api-server
    environment:
      - API_KEY=${API_KEY}
```

### Q6: MCPサーバーの監視方法は？

```python
# Prometheus メトリクス付きMCPサーバー
from prometheus_client import Counter, Histogram, start_http_server
import time

tool_calls = Counter(
    "mcp_tool_calls_total",
    "Total tool calls",
    ["tool_name", "status"]
)

tool_latency = Histogram(
    "mcp_tool_latency_seconds",
    "Tool execution latency",
    ["tool_name"]
)

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    start = time.time()
    try:
        result = await execute_tool(name, arguments)
        tool_calls.labels(tool_name=name, status="success").inc()
        return result
    except Exception as e:
        tool_calls.labels(tool_name=name, status="error").inc()
        raise
    finally:
        tool_latency.labels(tool_name=name).observe(time.time() - start)

# メトリクスエンドポイントを起動（別ポート）
start_http_server(9090)
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| MCP | AIとツール間の標準プロトコル |
| 4つの機能 | Tools, Resources, Prompts, Sampling |
| 通信方式 | stdio（ローカル）/ SSE（リモート） |
| サーバー実装 | Python/TypeScript SDK で構築 |
| クライアント | Claude Desktop, Code, Cursor等で利用 |
| セキュリティ | 入力バリデーション、認証、レート制限 |
| テスト | ユニットテスト + MCP Inspector + 統合テスト |
| 原則 | セキュリティ重視、エラー情報を正確に伝達 |

## 次に読むべきガイド

- [03-claude-agent-sdk.md](./03-claude-agent-sdk.md) -- Claude Agent SDKでのMCP統合
- [../00-fundamentals/02-tool-use.md](../00-fundamentals/02-tool-use.md) -- ツール使用の基礎
- [../04-production/00-deployment.md](../04-production/00-deployment.md) -- MCPサーバーのデプロイ

## 参考文献

1. Model Context Protocol Specification -- https://modelcontextprotocol.io/
2. MCP GitHub Organization -- https://github.com/modelcontextprotocol
3. Anthropic, "Introducing the Model Context Protocol" (2024) -- https://www.anthropic.com/news/model-context-protocol
4. MCP Python SDK -- https://github.com/modelcontextprotocol/python-sdk
5. MCP TypeScript SDK -- https://github.com/modelcontextprotocol/typescript-sdk
