# ツール使用（Tool Use）

> Function Calling、MCP、ツール定義――LLMに「手足」を与えるための仕組みを理解し、安全かつ効果的なツール統合を実装する。

## この章で学ぶこと

1. Function Callingの仕組みとLLMがツールを選択する原理
2. MCP（Model Context Protocol）によるツールサーバーの構築と接続
3. ツール定義のベストプラクティスと安全性の確保方法
4. 高度なツール連携パターンと実践的なトラブルシューティング
5. プロダクション環境でのツール運用とモニタリング

---

## 1. ツール使用の全体像

### 1.1 なぜツールが必要か

LLM単体では以下ができない:
- **最新情報の取得**（学習データのカットオフ）
- **正確な計算**（浮動小数点演算の不確実性）
- **外部システムとの連携**（DB操作、API呼び出し）
- **ファイル操作**（読み書き、作成、削除）

ツール使用はこれらの制約を突破し、LLMを **「知っている」から「できる」** へ拡張する。

```
ツール使用の流れ
+--------+    質問     +---------+   ツール呼出   +--------+
| ユーザー|----------->|  LLM    |--------------->| ツール |
|        |            |         |<---------------|        |
|        |    回答     |         |   実行結果     |        |
|        |<-----------|         |                |        |
+--------+            +---------+                +--------+
                           |
                      ツールを使うか
                      どうかはLLMが判断
```

### 1.2 ツール使用の歴史的背景

```
ツール使用の進化
2022年以前: LLM = テキスト生成のみ
     ↓
2023年初: ChatGPT Plugins / Function Calling 登場
     ↓
2023年中: Anthropic Tool Use / Google Function Calling
     ↓
2024年: MCP（Model Context Protocol）の登場
     ↓
2025年: 標準化の進展、ツールエコシステムの成熟
     ↓
2026年: エージェンティックツール使用の高度化
         - 並列ツール実行の最適化
         - 自律的なツール発見と登録
         - セキュリティフレームワークの標準化
```

### 1.3 ツール使用のカテゴリ分類

```
ツールの分類体系

+-------------------+-------------------+-------------------+
|   情報取得系       |   操作実行系       |   生成系           |
+-------------------+-------------------+-------------------+
| - Web検索         | - ファイル操作     | - 画像生成         |
| - DB検索          | - API呼び出し     | - コード生成       |
| - ドキュメント検索  | - メール送信      | - ドキュメント生成  |
| - データ取得       | - デプロイ        | - チャート生成     |
+-------------------+-------------------+-------------------+
| リスク: 低         | リスク: 中-高     | リスク: 低-中      |
| 副作用: なし       | 副作用: あり      | 副作用: リソース消費|
+-------------------+-------------------+-------------------+
```

---

## 2. Function Calling

### 2.1 仕組み

```
Function Calling のライフサイクル
+------------------------------------------------------------------+
| 1. ツール定義をLLMに渡す                                           |
|    tools: [{name, description, parameters}]                       |
|                                                                    |
| 2. ユーザーメッセージを送信                                        |
|    "東京の天気を教えて"                                            |
|                                                                    |
| 3. LLMがツール呼び出しを生成（JSONで返却）                         |
|    {tool: "get_weather", input: {city: "東京"}}                   |
|                                                                    |
| 4. アプリケーションがツールを実行                                  |
|    result = get_weather("東京")                                    |
|                                                                    |
| 5. 結果をLLMに返却                                                |
|    tool_result: "東京: 晴れ, 22°C"                                |
|                                                                    |
| 6. LLMが最終回答を生成                                            |
|    "東京は現在晴れで、気温は22°Cです"                              |
+------------------------------------------------------------------+
```

### 2.2 Anthropic API でのFunction Calling

```python
import anthropic
import json

client = anthropic.Anthropic()

# ツール定義
tools = [
    {
        "name": "get_weather",
        "description": "指定された都市の現在の天気情報を取得する",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "天気を取得する都市名（例: 東京、大阪）"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度の単位（デフォルト: celsius）"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "search_restaurant",
        "description": "指定された条件でレストランを検索する",
        "input_schema": {
            "type": "object",
            "properties": {
                "area": {"type": "string", "description": "エリア名"},
                "cuisine": {"type": "string", "description": "料理ジャンル"},
                "budget": {"type": "integer", "description": "予算上限（円）"}
            },
            "required": ["area"]
        }
    }
]

# ツール実行関数
def execute_tool(name: str, input_data: dict) -> str:
    if name == "get_weather":
        # 実際にはWeather APIを呼び出す
        return json.dumps({
            "city": input_data["city"],
            "weather": "晴れ",
            "temperature": 22,
            "humidity": 45
        })
    elif name == "search_restaurant":
        return json.dumps({
            "results": [
                {"name": "寿司太郎", "rating": 4.5, "budget": 5000},
                {"name": "天ぷら花", "rating": 4.2, "budget": 3000}
            ]
        })
    return "ツールが見つかりません"

# エージェントループ
def agent_chat(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        # 最終回答の場合
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

        # ツール呼び出しの場合
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  ツール実行: {block.name}({block.input})")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

# 使用例
answer = agent_chat("渋谷で予算5000円以内のおすすめレストランと、今の天気を教えて")
print(answer)
```

### 2.3 OpenAI API でのFunction Calling

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "都市の天気を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "東京の天気は？"}],
    tools=tools,
    tool_choice="auto"  # LLMが自動判断
)

# tool_choice の選択肢
# "auto"     : LLMが判断（デフォルト）
# "required" : 必ずツールを使用
# "none"     : ツールを使わない
# {"type": "function", "function": {"name": "get_weather"}} : 特定ツール強制
```

### 2.4 Google Gemini API でのFunction Calling

```python
import google.generativeai as genai

# ツール定義（Gemini形式）
weather_tool = genai.protos.Tool(
    function_declarations=[
        genai.protos.FunctionDeclaration(
            name="get_weather",
            description="指定された都市の天気情報を取得する",
            parameters=genai.protos.Schema(
                type=genai.protos.Type.OBJECT,
                properties={
                    "city": genai.protos.Schema(
                        type=genai.protos.Type.STRING,
                        description="都市名"
                    )
                },
                required=["city"]
            )
        )
    ]
)

model = genai.GenerativeModel(
    "gemini-2.0-flash",
    tools=[weather_tool]
)

chat = model.start_chat()
response = chat.send_message("東京の天気を教えて")

# Function Callの処理
for part in response.parts:
    if fn := part.function_call:
        print(f"関数呼び出し: {fn.name}({dict(fn.args)})")
        # 結果を返す
        result = get_weather(dict(fn.args))
        response = chat.send_message(
            genai.protos.Content(
                parts=[genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fn.name,
                        response={"result": result}
                    )
                )]
            )
        )
```

### 2.5 並列ツール呼び出し

```python
# 複数ツールの並列呼び出し対応
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelToolExecutor:
    """複数のツール呼び出しを並列実行する"""

    def __init__(self, tool_handlers: dict, max_workers: int = 5):
        self.handlers = tool_handlers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def execute_parallel(self, tool_calls: list) -> list:
        """複数のツール呼び出しを並列実行"""
        loop = asyncio.get_event_loop()
        tasks = []

        for call in tool_calls:
            handler = self.handlers.get(call["name"])
            if handler:
                task = loop.run_in_executor(
                    self.executor,
                    lambda h=handler, a=call["input"]: h(**a)
                )
                tasks.append((call["id"], task))

        results = []
        for call_id, task in tasks:
            try:
                result = await task
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": str(result)
                })
            except Exception as e:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": f"エラー: {e}",
                    "is_error": True
                })

        return results

# 使用例
executor = ParallelToolExecutor({
    "get_weather": get_weather,
    "search_restaurant": search_restaurant,
    "get_exchange_rate": get_exchange_rate
})

# LLMが複数ツールを同時に呼び出した場合
parallel_calls = [
    {"id": "call_1", "name": "get_weather", "input": {"city": "東京"}},
    {"id": "call_2", "name": "search_restaurant", "input": {"area": "渋谷"}},
    {"id": "call_3", "name": "get_exchange_rate", "input": {"from": "USD", "to": "JPY"}}
]

results = asyncio.run(executor.execute_parallel(parallel_calls))
```

### 2.6 ストリーミングでのツール使用

```python
# ストリーミング応答でのツール呼び出し処理
import anthropic

client = anthropic.Anthropic()

def stream_with_tools(user_message: str, tools: list):
    """ストリーミングでツール呼び出しを処理する"""
    messages = [{"role": "user", "content": user_message}]

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        tools=tools,
        messages=messages
    ) as stream:
        current_tool = None
        tool_input_json = ""

        for event in stream:
            if event.type == "content_block_start":
                if hasattr(event.content_block, "type"):
                    if event.content_block.type == "tool_use":
                        current_tool = {
                            "id": event.content_block.id,
                            "name": event.content_block.name
                        }
                        tool_input_json = ""
                    elif event.content_block.type == "text":
                        current_tool = None

            elif event.type == "content_block_delta":
                if hasattr(event.delta, "text"):
                    # テキスト応答をリアルタイム出力
                    print(event.delta.text, end="", flush=True)
                elif hasattr(event.delta, "partial_json"):
                    # ツール入力のJSON断片を蓄積
                    tool_input_json += event.delta.partial_json

            elif event.type == "content_block_stop":
                if current_tool:
                    # ツール呼び出し完了
                    tool_input = json.loads(tool_input_json)
                    print(f"\n[ツール実行: {current_tool['name']}]")
                    result = execute_tool(current_tool["name"], tool_input)
                    # 結果をLLMに返して続行...
```

---

## 3. MCP（Model Context Protocol）

### 3.1 MCPとは

```
MCP アーキテクチャ
+------------------+     stdio/SSE     +------------------+
|   MCP Client     |<=================>|   MCP Server     |
|  (Claude Code,   |                   |  (ツールサーバー)  |
|   Cursor, etc.)  |   JSON-RPC 2.0   |                   |
+------------------+                   +------------------+
                                       |  +-----------+   |
                                       |  | Tools     |   |
                                       |  +-----------+   |
                                       |  | Resources |   |
                                       |  +-----------+   |
                                       |  | Prompts   |   |
                                       |  +-----------+   |
                                       +------------------+
```

MCPは **AIアプリケーションとツール間の標準プロトコル** 。Anthropicが提唱し、以下の3つの機能を提供する:

| 機能 | 説明 | 例 |
|------|------|-----|
| Tools | LLMが呼び出す関数 | ファイル操作、DB検索 |
| Resources | コンテキストとして提供するデータ | ファイル内容、API応答 |
| Prompts | 再利用可能なプロンプトテンプレート | コードレビュー用プロンプト |

### 3.2 MCPサーバーの実装

```python
# MCP サーバーの実装例（Python SDK）
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

app = Server("my-tools")

# ツール一覧の定義
@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="read_database",
            description="SQLiteデータベースを検索して結果を返す",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL SELECT文"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="send_email",
            description="指定されたアドレスにメールを送信する",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        )
    ]

# ツール実行
@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "read_database":
        results = execute_sql(arguments["query"])
        return [TextContent(type="text", text=json.dumps(results))]

    elif name == "send_email":
        send_email(
            to=arguments["to"],
            subject=arguments["subject"],
            body=arguments["body"]
        )
        return [TextContent(type="text", text="メール送信完了")]

# サーバー起動
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    async def main():
        async with stdio_server() as (read, write):
            await app.run(read, write)

    asyncio.run(main())
```

### 3.3 MCP設定（Claude Code）

```json
// .claude/settings.json
{
  "mcpServers": {
    "my-tools": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "DATABASE_PATH": "/path/to/db.sqlite"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-filesystem", "/allowed/path"]
    }
  }
}
```

### 3.4 MCPリソースの実装

```python
# MCPリソース: コンテキストデータの提供
from mcp.server import Server
from mcp.types import Resource, TextContent
import json

app = Server("resource-server")

@app.list_resources()
async def list_resources():
    return [
        Resource(
            uri="project://config",
            name="プロジェクト設定",
            description="現在のプロジェクトの設定情報",
            mimeType="application/json"
        ),
        Resource(
            uri="project://readme",
            name="README",
            description="プロジェクトのREADMEファイル",
            mimeType="text/markdown"
        ),
        Resource(
            uri="metrics://daily",
            name="日次メトリクス",
            description="本日のアプリケーションメトリクス",
            mimeType="application/json"
        )
    ]

@app.read_resource()
async def read_resource(uri: str):
    if uri == "project://config":
        config = load_project_config()
        return TextContent(
            type="text",
            text=json.dumps(config, indent=2, ensure_ascii=False)
        )
    elif uri == "project://readme":
        with open("README.md", "r") as f:
            return TextContent(type="text", text=f.read())
    elif uri == "metrics://daily":
        metrics = fetch_daily_metrics()
        return TextContent(
            type="text",
            text=json.dumps(metrics, indent=2)
        )
    raise ValueError(f"Unknown resource: {uri}")
```

### 3.5 MCPプロンプトの実装

```python
# MCPプロンプト: 再利用可能なプロンプトテンプレート
from mcp.types import Prompt, PromptArgument, PromptMessage, TextContent

@app.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code_review",
            description="コードレビュー用のプロンプト",
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
            description="バグレポート作成用のプロンプト",
            arguments=[
                PromptArgument(
                    name="error_message",
                    description="エラーメッセージ",
                    required=True
                ),
                PromptArgument(
                    name="context",
                    description="エラー発生時のコンテキスト",
                    required=False
                )
            ]
        )
    ]

@app.get_prompt()
async def get_prompt(name: str, arguments: dict):
    if name == "code_review":
        language = arguments.get("language", "不明")
        return [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""以下の{language}コードをレビューしてください。

確認項目:
1. バグや論理エラー
2. セキュリティ上の問題
3. パフォーマンスの改善点
4. コードの可読性
5. ベストプラクティスへの準拠

コード:
```
{arguments['code']}
```"""
                )
            )
        ]
```

### 3.6 SSEトランスポートでのMCPサーバー

```python
# HTTP SSE トランスポートを使ったMCPサーバー
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn

app = Server("sse-tools")

# ツール定義（省略）

# SSEトランスポートの設定
sse = SseServerTransport("/messages/")

async def handle_sse(request):
    """SSE接続のハンドリング"""
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send
    ) as streams:
        await app.run(
            streams[0],
            streams[1],
            app.create_initialization_options()
        )

# Starletteアプリケーション
starlette_app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=8000)
```

### 3.7 MCPクライアントの実装

```python
# MCPクライアント: サーバーに接続してツールを使用
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def use_mcp_tools():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env={"DATABASE_PATH": "/path/to/db.sqlite"}
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初期化
            await session.initialize()

            # 利用可能なツール一覧を取得
            tools = await session.list_tools()
            print("利用可能なツール:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")

            # ツール実行
            result = await session.call_tool(
                "read_database",
                arguments={"query": "SELECT * FROM users LIMIT 5"}
            )
            print(f"結果: {result.content[0].text}")

            # リソース取得
            resources = await session.list_resources()
            for resource in resources.resources:
                content = await session.read_resource(resource.uri)
                print(f"リソース [{resource.name}]: {content}")
```

---

## 4. ツール定義のベストプラクティス

### 4.1 良いツール定義と悪いツール定義

```
ツール定義の品質ピラミッド

        /\
       /  \    名前: 動詞_名詞 形式
      / 命名 \   例: search_web, read_file
     /--------\
    /          \   説明: 何をするか + いつ使うか
   /   説明    \   + 入出力の形式
  /------------\
 /              \   パラメータ: 型 + 制約 + デフォルト
/  パラメータ    \   + 具体的な例
/----------------\
```

### 4.2 ツール設計パターン

```python
# パターン1: 検索系ツール（読み取り専用）
search_tool = {
    "name": "search_documents",
    "description": (
        "社内ドキュメントを全文検索する。"
        "キーワードまたは自然言語のクエリを受け付ける。"
        "最大20件の結果を関連度順に返す。"
        "各結果にはタイトル、スニペット、URLが含まれる。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索クエリ（例: 'リモートワーク 規定'）"
            },
            "max_results": {
                "type": "integer",
                "description": "最大結果数（1-50、デフォルト: 10）",
                "default": 10,
                "minimum": 1,
                "maximum": 50
            },
            "department": {
                "type": "string",
                "enum": ["engineering", "sales", "hr", "all"],
                "description": "検索対象の部署（デフォルト: all）",
                "default": "all"
            }
        },
        "required": ["query"]
    }
}

# パターン2: 書き込み系ツール（副作用あり）
write_tool = {
    "name": "create_ticket",
    "description": (
        "JIRAにチケットを作成する。副作用あり: 実際にチケットが作成される。"
        "作成後のチケットIDとURLを返す。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "チケットタイトル（100文字以内）"},
            "description": {"type": "string", "description": "詳細説明（Markdown可）"},
            "priority": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low"],
                "description": "優先度"
            },
            "assignee": {"type": "string", "description": "担当者のメールアドレス"}
        },
        "required": ["title", "description", "priority"]
    }
}
```

### 4.3 高度なツール定義テクニック

```python
# テクニック1: 条件付きパラメータ
conditional_tool = {
    "name": "send_notification",
    "description": (
        "通知を送信する。channelがemailの場合はemail_addressが必須、"
        "channelがslackの場合はslack_channelが必須。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {
                "type": "string",
                "enum": ["email", "slack", "sms"],
                "description": "通知チャネル"
            },
            "message": {"type": "string", "description": "通知メッセージ"},
            "email_address": {
                "type": "string",
                "description": "メールアドレス（channel=emailの場合に必須）"
            },
            "slack_channel": {
                "type": "string",
                "description": "Slackチャネル名（channel=slackの場合に必須）"
            },
            "phone_number": {
                "type": "string",
                "description": "電話番号（channel=smsの場合に必須）"
            }
        },
        "required": ["channel", "message"]
    }
}

# テクニック2: ネストされたオブジェクト
nested_tool = {
    "name": "create_order",
    "description": "注文を作成する",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["name", "email"]
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "quantity": {"type": "integer", "minimum": 1}
                    },
                    "required": ["product_id", "quantity"]
                },
                "minItems": 1,
                "description": "注文商品リスト（1件以上必須）"
            },
            "shipping_address": {
                "type": "object",
                "properties": {
                    "zip": {"type": "string"},
                    "prefecture": {"type": "string"},
                    "city": {"type": "string"},
                    "address": {"type": "string"}
                },
                "required": ["zip", "prefecture", "city", "address"]
            }
        },
        "required": ["customer", "items", "shipping_address"]
    }
}

# テクニック3: ツールの説明に使用例を含める
example_tool = {
    "name": "query_analytics",
    "description": (
        "アナリティクスデータを自然言語でクエリする。\n\n"
        "使用例:\n"
        "- 'yesterday page views' → 昨日のページビュー数を返す\n"
        "- 'top 10 pages this week' → 今週のトップ10ページを返す\n"
        "- 'conversion rate last 30 days' → 直近30日のコンバージョン率を返す\n\n"
        "注意: 最大取得期間は90日。それ以上はバッチエクスポートを使用。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "自然言語のクエリ"},
            "format": {
                "type": "string",
                "enum": ["table", "chart", "raw"],
                "default": "table",
                "description": "出力形式"
            }
        },
        "required": ["query"]
    }
}
```

### 4.4 ツール結果のフォーマット最適化

```python
# ツール結果の構造化パターン

class ToolResultFormatter:
    """ツール結果をLLMが理解しやすい形式にフォーマットする"""

    @staticmethod
    def format_search_results(results: list, query: str) -> str:
        """検索結果のフォーマット"""
        if not results:
            return f"「{query}」に一致する結果はありませんでした。"

        formatted = f"「{query}」の検索結果（{len(results)}件）:\n\n"
        for i, r in enumerate(results, 1):
            formatted += f"{i}. **{r['title']}**\n"
            formatted += f"   URL: {r['url']}\n"
            formatted += f"   抜粋: {r['snippet'][:200]}...\n"
            formatted += f"   関連度: {r['score']:.2f}\n\n"
        return formatted

    @staticmethod
    def format_error(error: Exception, context: dict = None) -> str:
        """エラーのフォーマット（LLMが再試行できるように）"""
        result = {
            "status": "error",
            "error_type": type(error).__name__,
            "message": str(error),
            "suggestions": []
        }

        if isinstance(error, TimeoutError):
            result["suggestions"] = [
                "タイムアウトが発生しました",
                "クエリを簡略化するか、結果数を減らして再試行してください"
            ]
        elif isinstance(error, PermissionError):
            result["suggestions"] = [
                "アクセス権限がありません",
                "別のリソースを使用するか、管理者に連絡してください"
            ]
        elif isinstance(error, ValueError):
            result["suggestions"] = [
                "入力値が不正です",
                f"詳細: {error}",
                "パラメータを確認して再試行してください"
            ]

        if context:
            result["context"] = context

        return json.dumps(result, ensure_ascii=False, indent=2)

    @staticmethod
    def format_large_result(data: dict, max_items: int = 10) -> str:
        """大量の結果を要約する"""
        total = len(data.get("items", []))
        truncated = data["items"][:max_items] if total > max_items else data["items"]

        return json.dumps({
            "total_count": total,
            "showing": len(truncated),
            "items": truncated,
            "note": f"{total}件中{len(truncated)}件を表示。残りは'offset'パラメータで取得可能。"
        }, ensure_ascii=False, indent=2)
```

---

## 5. ツール使用の比較

### 5.1 Function Calling vs MCP

| 観点 | Function Calling | MCP |
|------|-----------------|-----|
| 標準化 | 各社独自仕様 | オープン標準 |
| 通信方式 | HTTP API内に埋め込み | stdio / SSE |
| サーバー管理 | アプリ内で実行 | 独立プロセス |
| ツール共有 | アプリ固有 | 複数クライアントで共有可能 |
| エコシステム | 各社SDK | 共通MCPサーバー |
| 適用場面 | 単体アプリ | 複数AI製品での共有 |
| セキュリティ | アプリレベル | プロセス分離 |

### 5.2 プロバイダ別 Function Calling 比較

| 項目 | Anthropic (Claude) | OpenAI (GPT) | Google (Gemini) |
|------|-------------------|--------------|-----------------|
| 呼び出し形式 | tool_use ブロック | function_call | functionCall |
| 並列呼び出し | 対応 | 対応 | 対応 |
| ストリーミング | 対応 | 対応 | 対応 |
| tool_choice | auto/any/tool指定 | auto/required/none/指定 | auto/any/none |
| 結果返却 | tool_result | tool message | functionResponse |

### 5.3 MCPサーバーエコシステム

```
利用可能なMCPサーバー（2025-2026年）

公式サーバー:
├── @anthropic/mcp-filesystem    - ファイル操作
├── @anthropic/mcp-git           - Git操作
├── @anthropic/mcp-github        - GitHub API
├── @anthropic/mcp-postgres      - PostgreSQL
├── @anthropic/mcp-sqlite        - SQLite
├── @anthropic/mcp-puppeteer     - Webブラウジング
└── @anthropic/mcp-memory        - 知識グラフメモリ

コミュニティサーバー:
├── mcp-server-slack             - Slack統合
├── mcp-server-notion            - Notion統合
├── mcp-server-jira              - JIRA統合
├── mcp-server-kubernetes        - K8s管理
├── mcp-server-aws               - AWS操作
├── mcp-server-gcp               - GCP操作
├── mcp-server-docker            - Docker管理
├── mcp-server-redis             - Redis操作
├── mcp-server-elasticsearch     - Elasticsearch検索
└── mcp-server-stripe            - Stripe決済
```

---

## 6. 高度なツール使用パターン

### 6.1 ツールチェイニング

```python
# 複数ツールを連鎖的に使用するパターン
class ToolChain:
    """ツールの出力を次のツールの入力にパイプする"""

    def __init__(self, tools: dict):
        self.tools = tools

    def chain(self, steps: list[dict]) -> dict:
        """
        steps = [
            {"tool": "search", "input": {"query": "..."}, "output_key": "results"},
            {"tool": "summarize", "input_from": "results", "output_key": "summary"},
            {"tool": "translate", "input_from": "summary", "output_key": "translation"}
        ]
        """
        context = {}

        for step in steps:
            tool_name = step["tool"]

            # 前のステップの出力を入力に使用
            if "input_from" in step:
                tool_input = {"text": context[step["input_from"]]}
            else:
                tool_input = step["input"]

            # ツール実行
            result = self.tools[tool_name](**tool_input)
            context[step["output_key"]] = result

        return context

# 使用例: 検索→要約→翻訳のチェイン
chain = ToolChain(tools={
    "search": search_web,
    "summarize": summarize_text,
    "translate": translate_text
})

result = chain.chain([
    {"tool": "search", "input": {"query": "quantum computing 2025"}, "output_key": "results"},
    {"tool": "summarize", "input_from": "results", "output_key": "summary"},
    {"tool": "translate", "input_from": "summary", "output_key": "japanese"}
])
print(result["japanese"])
```

### 6.2 ツールのフォールバック

```python
# プライマリツールが失敗した場合のフォールバック
class ToolWithFallback:
    """プライマリ→セカンダリの順でツールを試行"""

    def __init__(self):
        self.fallback_chains = {}

    def register(self, name: str, primary, *fallbacks):
        self.fallback_chains[name] = [primary] + list(fallbacks)

    def execute(self, name: str, **kwargs) -> dict:
        chain = self.fallback_chains.get(name, [])
        errors = []

        for i, tool in enumerate(chain):
            try:
                result = tool(**kwargs)
                return {
                    "status": "success",
                    "data": result,
                    "tool_index": i,
                    "fallback_used": i > 0
                }
            except Exception as e:
                errors.append(f"Tool {i}: {type(e).__name__}: {e}")
                continue

        return {
            "status": "all_failed",
            "errors": errors
        }

# 使用例
tools = ToolWithFallback()
tools.register(
    "search",
    google_search,      # プライマリ: Google検索
    bing_search,         # フォールバック1: Bing検索
    duckduckgo_search    # フォールバック2: DuckDuckGo検索
)

result = tools.execute("search", query="AI agents 2025")
```

### 6.3 ツールの動的登録と発見

```python
# ツールレジストリ: 実行時にツールを動的に追加・削除
class ToolRegistry:
    """動的ツール登録・発見システム"""

    def __init__(self):
        self.tools = {}
        self.metadata = {}

    def register(self, name: str, handler, schema: dict,
                 tags: list[str] = None, version: str = "1.0"):
        """ツールを登録"""
        self.tools[name] = handler
        self.metadata[name] = {
            "schema": schema,
            "tags": tags or [],
            "version": version,
            "registered_at": time.time()
        }

    def unregister(self, name: str):
        """ツールを削除"""
        self.tools.pop(name, None)
        self.metadata.pop(name, None)

    def discover(self, tags: list[str] = None,
                 query: str = None) -> list[dict]:
        """条件に合うツールを発見"""
        results = []
        for name, meta in self.metadata.items():
            if tags and not set(tags).intersection(set(meta["tags"])):
                continue
            if query and query.lower() not in meta["schema"]["description"].lower():
                continue
            results.append({
                "name": name,
                "description": meta["schema"]["description"],
                "tags": meta["tags"],
                "version": meta["version"]
            })
        return results

    def get_tools_for_llm(self, tags: list[str] = None) -> list[dict]:
        """LLMに渡すツール定義を生成"""
        discovered = self.discover(tags=tags)
        return [
            self.metadata[t["name"]]["schema"]
            for t in discovered
        ]

# 使用例
registry = ToolRegistry()

registry.register("search_web", search_web, {
    "name": "search_web",
    "description": "Webを検索する",
    "input_schema": {...}
}, tags=["search", "web"])

registry.register("query_db", query_db, {
    "name": "query_db",
    "description": "データベースを検索する",
    "input_schema": {...}
}, tags=["search", "database"])

# 検索系ツールのみ取得
search_tools = registry.get_tools_for_llm(tags=["search"])
```

### 6.4 ツール使用の監査ログ

```python
# 全ツール呼び出しの監査ログ
import logging
from datetime import datetime
from functools import wraps

class ToolAuditLogger:
    """ツール使用の監査ログを記録"""

    def __init__(self, log_file: str = "tool_audit.log"):
        self.logger = logging.getLogger("tool_audit")
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def wrap(self, tool_name: str, handler):
        """ツールハンドラをラップして監査ログを追加"""
        @wraps(handler)
        def wrapped(**kwargs):
            start_time = datetime.now()
            request_id = str(uuid.uuid4())[:8]

            # 入力ログ
            self.logger.info(
                f"CALL | {request_id} | {tool_name} | "
                f"input={json.dumps(kwargs, ensure_ascii=False, default=str)}"
            )

            try:
                result = handler(**kwargs)
                duration = (datetime.now() - start_time).total_seconds()

                # 成功ログ
                result_preview = str(result)[:500]
                self.logger.info(
                    f"OK   | {request_id} | {tool_name} | "
                    f"duration={duration:.3f}s | result_preview={result_preview}"
                )
                return result

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()

                # エラーログ
                self.logger.error(
                    f"FAIL | {request_id} | {tool_name} | "
                    f"duration={duration:.3f}s | error={type(e).__name__}: {e}"
                )
                raise

        return wrapped

# 使用例
audit = ToolAuditLogger()
search_web = audit.wrap("search_web", search_web)
query_db = audit.wrap("query_db", query_db)
```

---

## 7. セキュリティ

### 7.1 ツールのセキュリティモデル

```
ツールセキュリティの層

+-----------------------------------------------+
| Layer 4: 人間の承認 (Human Approval)            |
|   破壊的操作の前にユーザー確認を挟む              |
+-----------------------------------------------+
| Layer 3: 監査ログ (Audit Logging)               |
|   全ツール呼び出しを記録・検知                   |
+-----------------------------------------------+
| Layer 2: 入力バリデーション (Input Validation)   |
|   SQLインジェクション、パストラバーサル防止       |
+-----------------------------------------------+
| Layer 1: 最小権限 (Least Privilege)             |
|   ツールに必要最小限のアクセス権限のみ付与        |
+-----------------------------------------------+
```

### 7.2 入力バリデーションの実装

```python
# ツール入力のセキュリティバリデーション
import re
from pathlib import Path

class ToolInputValidator:
    """ツール入力のセキュリティバリデーション"""

    @staticmethod
    def validate_sql(query: str) -> tuple[bool, str]:
        """SQLクエリのバリデーション"""
        # SELECTのみ許可
        query_upper = query.strip().upper()
        if not query_upper.startswith("SELECT"):
            return False, "SELECT文のみ許可されています"

        # 危険なキーワードの検出
        dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
                     "CREATE", "TRUNCATE", "EXEC", "EXECUTE", "UNION"]
        for keyword in dangerous:
            if re.search(rf'\b{keyword}\b', query_upper):
                return False, f"'{keyword}'は使用できません"

        # コメントの検出（インジェクション対策）
        if "--" in query or "/*" in query:
            return False, "SQLコメントは使用できません"

        return True, "OK"

    @staticmethod
    def validate_file_path(path: str, allowed_dirs: list[str]) -> tuple[bool, str]:
        """ファイルパスのバリデーション"""
        resolved = Path(path).resolve()

        # パストラバーサルの検出
        if ".." in str(resolved):
            return False, "パストラバーサルが検出されました"

        # 許可ディレクトリ内かチェック
        for allowed in allowed_dirs:
            if str(resolved).startswith(str(Path(allowed).resolve())):
                return True, "OK"

        return False, f"パス '{path}' は許可されたディレクトリ外です"

    @staticmethod
    def validate_url(url: str) -> tuple[bool, str]:
        """URLのバリデーション"""
        from urllib.parse import urlparse

        parsed = urlparse(url)

        if parsed.scheme not in ("http", "https"):
            return False, "HTTPまたはHTTPSのみ許可されています"

        # 内部ネットワークへのアクセスを防止
        internal_patterns = [
            r"localhost", r"127\.0\.0\.1", r"0\.0\.0\.0",
            r"10\.\d+\.\d+\.\d+", r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+",
            r"192\.168\.\d+\.\d+", r"169\.254\.\d+\.\d+"
        ]
        for pattern in internal_patterns:
            if re.search(pattern, parsed.hostname or ""):
                return False, "内部ネットワークへのアクセスは禁止されています"

        return True, "OK"

# ツールにバリデーションを組み込む
validator = ToolInputValidator()

def safe_query_db(query: str) -> str:
    valid, message = validator.validate_sql(query)
    if not valid:
        return json.dumps({"error": message})
    return execute_sql(query)
```

### 7.3 人間の承認フロー

```python
# Human-in-the-Loop: 危険な操作の確認
class HumanApprovalGate:
    """破壊的操作の前に人間の承認を求める"""

    # 承認が必要な操作のカテゴリ
    REQUIRES_APPROVAL = {
        "destructive": ["delete_file", "drop_table", "remove_user"],
        "external": ["send_email", "post_to_slack", "deploy"],
        "financial": ["create_payment", "refund", "update_pricing"],
        "sensitive": ["access_pii", "export_data", "change_permissions"]
    }

    def __init__(self, approval_callback=None):
        self.callback = approval_callback or self._cli_approval

    def check(self, tool_name: str, args: dict) -> bool:
        """操作に承認が必要か確認し、必要なら承認を求める"""
        for category, tools in self.REQUIRES_APPROVAL.items():
            if tool_name in tools:
                return self.callback(
                    tool_name=tool_name,
                    category=category,
                    args=args
                )
        return True  # 承認不要

    def _cli_approval(self, tool_name: str, category: str, args: dict) -> bool:
        """CLI上で承認を求める"""
        print(f"\n{'='*60}")
        print(f"[承認要求] {category}カテゴリの操作")
        print(f"ツール: {tool_name}")
        print(f"引数: {json.dumps(args, ensure_ascii=False, indent=2)}")
        print(f"{'='*60}")

        response = input("実行を許可しますか？ (yes/no): ").strip().lower()
        return response == "yes"

# エージェントに組み込む
gate = HumanApprovalGate()

def guarded_tool_execution(tool_name: str, args: dict):
    if not gate.check(tool_name, args):
        return {"status": "rejected", "message": "ユーザーが操作を拒否しました"}
    return execute_tool(tool_name, args)
```

### 7.4 レート制限とクォータ管理

```python
# ツール使用のレート制限
import time
from collections import defaultdict

class ToolRateLimiter:
    """ツール呼び出しのレート制限"""

    def __init__(self):
        self.limits = {}
        self.call_history = defaultdict(list)

    def set_limit(self, tool_name: str, max_calls: int,
                  window_seconds: int = 60):
        """レート制限を設定"""
        self.limits[tool_name] = {
            "max_calls": max_calls,
            "window": window_seconds
        }

    def check(self, tool_name: str) -> tuple[bool, str]:
        """レート制限をチェック"""
        if tool_name not in self.limits:
            return True, "OK"

        limit = self.limits[tool_name]
        now = time.time()
        window_start = now - limit["window"]

        # ウィンドウ内の呼び出し回数をカウント
        recent = [t for t in self.call_history[tool_name] if t > window_start]
        self.call_history[tool_name] = recent  # クリーンアップ

        if len(recent) >= limit["max_calls"]:
            wait_time = recent[0] - window_start
            return False, (
                f"レート制限超過: {tool_name}は{limit['window']}秒間に"
                f"{limit['max_calls']}回まで。{wait_time:.1f}秒後に再試行。"
            )

        self.call_history[tool_name].append(now)
        return True, "OK"

# 使用例
limiter = ToolRateLimiter()
limiter.set_limit("search_web", max_calls=10, window_seconds=60)
limiter.set_limit("send_email", max_calls=5, window_seconds=300)
limiter.set_limit("query_db", max_calls=30, window_seconds=60)
```

---

## 8. アンチパターン

### アンチパターン1: ツールの過剰提供

```python
# NG: 50個のツールを一度に渡す
tools = [tool1, tool2, ..., tool50]
# LLMは選択に迷い、精度が下がる

# OK: タスクに関連するツールのみ提供（5-10個が理想）
def select_tools(task_type: str) -> list:
    tool_sets = {
        "research": [search_web, read_page, summarize],
        "coding": [read_file, write_file, run_code],
        "data": [query_db, create_chart, export_csv]
    }
    return tool_sets.get(task_type, [])
```

### アンチパターン2: エラーハンドリングの欠如

```python
# NG: ツール実行結果をそのまま返す
def call_tool(name, args):
    return tools[name](**args)  # 例外時にクラッシュ

# OK: 構造化されたエラーレスポンス
def call_tool(name, args):
    try:
        result = tools[name](**args)
        return {"status": "success", "data": result}
    except KeyError:
        return {"status": "error", "message": f"ツール '{name}' は存在しません"}
    except ValidationError as e:
        return {"status": "error", "message": f"引数エラー: {e}"}
    except TimeoutError:
        return {"status": "error", "message": "タイムアウト。再試行してください"}
    except Exception as e:
        return {"status": "error", "message": f"予期せぬエラー: {type(e).__name__}"}
```

### アンチパターン3: ツール結果の無加工返却

```python
# NG: 生のAPIレスポンスをそのままLLMに渡す
def search_tool(query):
    response = requests.get(f"https://api.example.com/search?q={query}")
    return response.text  # HTML、巨大JSON、不要なメタデータが含まれる

# OK: LLMが処理しやすい形式に整形
def search_tool(query):
    response = requests.get(f"https://api.example.com/search?q={query}")
    data = response.json()

    # 必要な情報のみ抽出
    results = []
    for item in data["results"][:10]:
        results.append({
            "title": item["title"],
            "snippet": item["snippet"][:300],
            "url": item["url"]
        })

    return json.dumps({
        "query": query,
        "total_results": data["total"],
        "showing": len(results),
        "results": results
    }, ensure_ascii=False)
```

### アンチパターン4: ツール説明の曖昧さ

```python
# NG: 曖昧な説明
bad_tool = {
    "name": "process",  # 何を処理するか不明
    "description": "データを処理する",  # 具体性なし
    "input_schema": {
        "type": "object",
        "properties": {
            "input": {"type": "string"}  # 何の入力か不明
        }
    }
}

# OK: 具体的な説明
good_tool = {
    "name": "analyze_sentiment",
    "description": (
        "テキストの感情分析を行い、positive/negative/neutralのラベルと"
        "信頼度スコア(0.0-1.0)を返す。日本語と英語に対応。"
        "最大5000文字まで。それ以上の場合は分割して送信すること。"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "分析対象のテキスト（最大5000文字）",
                "maxLength": 5000
            },
            "language": {
                "type": "string",
                "enum": ["ja", "en", "auto"],
                "description": "テキストの言語（autoで自動検出）",
                "default": "auto"
            }
        },
        "required": ["text"]
    }
}
```

---

## 9. ツール使用のモニタリングと最適化

### 9.1 ツール使用メトリクス

```python
# ツール使用のメトリクス収集
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class ToolMetrics:
    """ツール使用のメトリクスを収集・分析"""
    call_counts: dict = field(default_factory=lambda: defaultdict(int))
    error_counts: dict = field(default_factory=lambda: defaultdict(int))
    latencies: dict = field(default_factory=lambda: defaultdict(list))
    token_costs: dict = field(default_factory=lambda: defaultdict(int))

    def record_call(self, tool_name: str, duration: float,
                    success: bool, tokens_used: int = 0):
        self.call_counts[tool_name] += 1
        self.latencies[tool_name].append(duration)
        self.token_costs[tool_name] += tokens_used
        if not success:
            self.error_counts[tool_name] += 1

    def get_summary(self) -> dict:
        summary = {}
        for name in self.call_counts:
            lats = self.latencies[name]
            summary[name] = {
                "total_calls": self.call_counts[name],
                "error_rate": (
                    self.error_counts[name] / self.call_counts[name] * 100
                    if self.call_counts[name] > 0 else 0
                ),
                "avg_latency_ms": sum(lats) / len(lats) * 1000 if lats else 0,
                "p95_latency_ms": sorted(lats)[int(len(lats) * 0.95)] * 1000 if lats else 0,
                "total_tokens": self.token_costs[name]
            }
        return summary

    def get_recommendations(self) -> list[str]:
        """最適化の推奨事項を生成"""
        recommendations = []
        summary = self.get_summary()

        for name, stats in summary.items():
            if stats["error_rate"] > 20:
                recommendations.append(
                    f"[{name}] エラー率が{stats['error_rate']:.1f}%と高い。"
                    f"入力バリデーションの強化を検討。"
                )
            if stats["avg_latency_ms"] > 5000:
                recommendations.append(
                    f"[{name}] 平均レイテンシが{stats['avg_latency_ms']:.0f}msと高い。"
                    f"キャッシュの導入を検討。"
                )
            if stats["total_calls"] > 100 and stats["total_calls"] > sum(
                s["total_calls"] for s in summary.values()
            ) * 0.5:
                recommendations.append(
                    f"[{name}] 全呼び出しの50%以上を占めている。"
                    f"バッチ処理の導入を検討。"
                )

        return recommendations
```

### 9.2 ツールキャッシュ

```python
# ツール結果のキャッシュ
import hashlib
from functools import lru_cache

class ToolCache:
    """ツール結果をキャッシュして重複呼び出しを削減"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds

    def _make_key(self, tool_name: str, args: dict) -> str:
        args_str = json.dumps(args, sort_keys=True)
        return hashlib.sha256(f"{tool_name}:{args_str}".encode()).hexdigest()

    def get(self, tool_name: str, args: dict):
        key = self._make_key(tool_name, args)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["result"]
            del self.cache[key]
        return None

    def set(self, tool_name: str, args: dict, result):
        key = self._make_key(tool_name, args)
        self.cache[key] = {
            "result": result,
            "timestamp": time.time()
        }

    def wrap(self, tool_name: str, handler, cacheable: bool = True):
        """ツールハンドラにキャッシュを追加"""
        if not cacheable:
            return handler

        def cached_handler(**kwargs):
            # キャッシュヒットチェック
            cached = self.get(tool_name, kwargs)
            if cached is not None:
                return cached

            # ツール実行してキャッシュ
            result = handler(**kwargs)
            self.set(tool_name, kwargs, result)
            return result

        return cached_handler

# 使用例
cache = ToolCache(ttl_seconds=600)

# 読み取り系ツールにキャッシュを適用
search_web = cache.wrap("search_web", search_web, cacheable=True)
# 書き込み系ツールにはキャッシュを適用しない
send_email = cache.wrap("send_email", send_email, cacheable=False)
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| ツールが呼ばれない | 説明が不明確 | description を具体的に改善 |
| 間違ったツールが呼ばれる | 類似ツールの区別不足 | ツール名・説明の差別化 |
| パラメータが不正 | スキーマの制約不足 | enum、min/max、デフォルト値を追加 |
| 無限ループ | 同じツールを繰り返し呼ぶ | ループ検出・最大ステップ数の設定 |
| レスポンスが遅い | ツール実行のレイテンシ | キャッシュ、並列実行、タイムアウト設定 |
| トークン超過 | ツール結果が大きすぎる | 結果の要約・ページネーション |
| 権限エラー | 不適切なアクセス制御 | 最小権限の原則を適用 |

### 10.2 デバッグテクニック

```python
# ツール使用のデバッグ支援
class ToolDebugger:
    """ツール使用のデバッグを支援するクラス"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.trace = []

    def log_tool_selection(self, available_tools: list,
                           selected_tool: str, reasoning: str):
        """ツール選択の過程を記録"""
        entry = {
            "event": "tool_selection",
            "available": [t["name"] for t in available_tools],
            "selected": selected_tool,
            "reasoning": reasoning,
            "timestamp": time.time()
        }
        self.trace.append(entry)
        if self.verbose:
            print(f"[DEBUG] ツール選択: {selected_tool}")
            print(f"  理由: {reasoning}")
            print(f"  候補: {entry['available']}")

    def log_tool_call(self, tool_name: str, input_data: dict,
                      result, duration: float):
        """ツール呼び出しの詳細を記録"""
        entry = {
            "event": "tool_call",
            "tool": tool_name,
            "input": input_data,
            "result_preview": str(result)[:200],
            "duration_ms": duration * 1000,
            "timestamp": time.time()
        }
        self.trace.append(entry)
        if self.verbose:
            print(f"[DEBUG] {tool_name} 実行完了 ({duration*1000:.1f}ms)")
            print(f"  入力: {json.dumps(input_data, ensure_ascii=False)[:200]}")
            print(f"  結果: {str(result)[:200]}")

    def export_trace(self, filepath: str):
        """トレースをファイルに出力"""
        with open(filepath, "w") as f:
            json.dump(self.trace, f, ensure_ascii=False, indent=2, default=str)

    def analyze_trace(self) -> dict:
        """トレースを分析してボトルネックを検出"""
        tool_calls = [e for e in self.trace if e["event"] == "tool_call"]

        analysis = {
            "total_calls": len(tool_calls),
            "total_duration_ms": sum(e["duration_ms"] for e in tool_calls),
            "slowest_call": max(tool_calls, key=lambda e: e["duration_ms"]) if tool_calls else None,
            "tool_frequency": defaultdict(int)
        }

        for call in tool_calls:
            analysis["tool_frequency"][call["tool"]] += 1

        # 重複呼び出しの検出
        seen_inputs = {}
        duplicates = []
        for call in tool_calls:
            key = f"{call['tool']}:{json.dumps(call['input'], sort_keys=True)}"
            if key in seen_inputs:
                duplicates.append(call["tool"])
            seen_inputs[key] = True

        analysis["duplicate_calls"] = duplicates
        return analysis
```

### 10.3 ツール定義の検証

```python
# ツール定義のバリデーション
class ToolDefinitionValidator:
    """ツール定義の品質を自動チェック"""

    def validate(self, tool_def: dict) -> list[str]:
        warnings = []

        # 名前のチェック
        name = tool_def.get("name", "")
        if not re.match(r'^[a-z][a-z0-9_]*$', name):
            warnings.append(f"名前 '{name}' はsnake_case形式にしてください")

        # 説明のチェック
        desc = tool_def.get("description", "")
        if len(desc) < 20:
            warnings.append("説明が短すぎます（20文字以上推奨）")
        if len(desc) > 500:
            warnings.append("説明が長すぎます（500文字以内推奨）")
        if "。" not in desc and "." not in desc:
            warnings.append("説明に文の終止が含まれていません")

        # スキーマのチェック
        schema = tool_def.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_def in props.items():
            if "description" not in prop_def:
                warnings.append(f"パラメータ '{prop_name}' にdescriptionがありません")
            if prop_def.get("type") == "string" and "enum" not in prop_def:
                if prop_name not in required:
                    if "default" not in prop_def:
                        warnings.append(
                            f"オプションパラメータ '{prop_name}' にデフォルト値がありません"
                        )

        return warnings

# 使用例
validator = ToolDefinitionValidator()
for tool in tools:
    warnings = validator.validate(tool)
    if warnings:
        print(f"\n[{tool['name']}] の問題点:")
        for w in warnings:
            print(f"  - {w}")
```

---

## 11. FAQ

### Q1: ツールの数が多い場合のパフォーマンスへの影響は？

ツール定義はすべてプロンプトのトークンとしてカウントされる。ツールが増えるほど:
- **コストが増加**（入力トークン増）
- **選択精度が低下**（類似ツール間の混同）
- **レイテンシが増加**

推奨は **1リクエストあたり5-15個** 。それ以上はカテゴリ分けして動的に選択する。

### Q2: ツールの結果が大きすぎる場合は？

LLMのコンテキストウィンドウを圧迫するため、以下の対策を取る:
- **要約して返す**（上位N件のみ）
- **ページネーション**（offset/limitパラメータ）
- **フィルタリング**（必要なフィールドのみ）

### Q3: 機密データを扱うツールのセキュリティは？

- **最小権限の原則**: ツールに必要最小限の権限のみ付与
- **入力バリデーション**: SQLインジェクション等を防止
- **監査ログ**: すべてのツール呼び出しを記録
- **人間の承認**: 破壊的操作（削除、送信）は確認を挟む

### Q4: MCPサーバーの選択基準は？

| 基準 | stdio | SSE (HTTP) |
|------|-------|------------|
| デプロイ | ローカルプロセス | リモートサーバー |
| セキュリティ | プロセス分離 | ネットワーク分離 |
| スケーラビリティ | 単一マシン | 水平スケール可能 |
| レイテンシ | 非常に低い | ネットワーク遅延あり |
| 適用場面 | ローカルツール | 共有サービス |

**ローカルの開発ツール** にはstdio、**チーム共有のサービス** にはSSEが適切。

### Q5: ツールの応答時間が遅い場合の対策は？

1. **タイムアウト設定**: 各ツールに適切なタイムアウトを設定（デフォルト30秒）
2. **キャッシュ**: 読み取り系ツールの結果をキャッシュ（TTL: 5-10分）
3. **並列実行**: 独立した複数のツール呼び出しを並列化
4. **非同期処理**: 長時間実行ツールは非同期にして進捗を返す
5. **結果の分割**: 大きな結果はストリーミングまたはページネーションで返す

### Q6: LLMがツールを誤用する場合の対策は？

```python
# ツール使用の制約を明示するシステムプロンプト
system_prompt = """
ツール使用のルール:
1. delete_file は確認後のみ使用（まず対象ファイルの内容を確認）
2. send_email は下書き作成後にユーザー確認を経て送信
3. query_db は SELECT 文のみ使用可能
4. 同じツールを3回連続で失敗した場合は代替手段を検討
5. ファイル操作は /workspace/ 配下のみ
"""
```

### Q7: カスタムMCPサーバーのテスト方法は？

```python
# MCPサーバーのユニットテスト
import pytest
from mcp.test import create_test_client

@pytest.fixture
async def mcp_client():
    """テスト用MCPクライアントを作成"""
    client = await create_test_client("python", ["mcp_server.py"])
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_list_tools(mcp_client):
    """ツール一覧のテスト"""
    tools = await mcp_client.list_tools()
    assert len(tools.tools) > 0
    tool_names = [t.name for t in tools.tools]
    assert "read_database" in tool_names

@pytest.mark.asyncio
async def test_read_database(mcp_client):
    """DB検索ツールのテスト"""
    result = await mcp_client.call_tool(
        "read_database",
        {"query": "SELECT 1 AS test"}
    )
    assert result.content[0].text is not None
    data = json.loads(result.content[0].text)
    assert data[0]["test"] == 1

@pytest.mark.asyncio
async def test_invalid_sql(mcp_client):
    """不正SQLの拒否テスト"""
    result = await mcp_client.call_tool(
        "read_database",
        {"query": "DROP TABLE users"}
    )
    data = json.loads(result.content[0].text)
    assert "error" in data
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| Function Calling | LLMがJSON形式でツール呼び出しを生成する仕組み |
| MCP | AIとツール間の標準プロトコル（Anthropic提唱） |
| ツール定義 | 名前+説明+パラメータ。説明の品質が精度を決める |
| ベストプラクティス | 5-15個、明確な説明、エラーハンドリング |
| セキュリティ | 最小権限、入力検証、監査ログ、人間承認 |
| 高度なパターン | チェイニング、フォールバック、動的登録、キャッシュ |
| モニタリング | メトリクス収集、トレース分析、ボトルネック検出 |

## 次に読むべきガイド

- [03-memory-systems.md](./03-memory-systems.md) — メモリシステムの設計
- [../02-implementation/02-mcp-agents.md](../02-implementation/02-mcp-agents.md) — MCPエージェントの実装
- [../02-implementation/03-claude-agent-sdk.md](../02-implementation/03-claude-agent-sdk.md) — Claude Agent SDKの詳細

## 参考文献

1. Anthropic, "Tool use (function calling)" — https://docs.anthropic.com/en/docs/build-with-claude/tool-use
2. Anthropic, "Model Context Protocol" — https://modelcontextprotocol.io/
3. OpenAI, "Function calling" — https://platform.openai.com/docs/guides/function-calling
4. Schick, T. et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023) — https://arxiv.org/abs/2302.04761
5. Google, "Function calling with Gemini" — https://ai.google.dev/docs/function_calling
6. Qin, Y. et al., "Tool Learning with Foundation Models" (2023) — https://arxiv.org/abs/2304.08354
