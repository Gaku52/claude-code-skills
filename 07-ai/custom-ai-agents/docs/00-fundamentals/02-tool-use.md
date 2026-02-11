# ツール使用（Tool Use）

> Function Calling、MCP、ツール定義――LLMに「手足」を与えるための仕組みを理解し、安全かつ効果的なツール統合を実装する。

## この章で学ぶこと

1. Function Callingの仕組みとLLMがツールを選択する原理
2. MCP（Model Context Protocol）によるツールサーバーの構築と接続
3. ツール定義のベストプラクティスと安全性の確保方法

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

---

## 6. アンチパターン

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

---

## 7. FAQ

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

---

## まとめ

| 項目 | 内容 |
|------|------|
| Function Calling | LLMがJSON形式でツール呼び出しを生成する仕組み |
| MCP | AIとツール間の標準プロトコル（Anthropic提唱） |
| ツール定義 | 名前+説明+パラメータ。説明の品質が精度を決める |
| ベストプラクティス | 5-15個、明確な説明、エラーハンドリング |
| セキュリティ | 最小権限、入力検証、監査ログ、人間承認 |

## 次に読むべきガイド

- [03-memory-systems.md](./03-memory-systems.md) — メモリシステムの設計
- [../02-implementation/02-mcp-agents.md](../02-implementation/02-mcp-agents.md) — MCPエージェントの実装
- [../02-implementation/03-claude-agent-sdk.md](../02-implementation/03-claude-agent-sdk.md) — Claude Agent SDKの詳細

## 参考文献

1. Anthropic, "Tool use (function calling)" — https://docs.anthropic.com/en/docs/build-with-claude/tool-use
2. Anthropic, "Model Context Protocol" — https://modelcontextprotocol.io/
3. OpenAI, "Function calling" — https://platform.openai.com/docs/guides/function-calling
4. Schick, T. et al., "Toolformer: Language Models Can Teach Themselves to Use Tools" (2023) — https://arxiv.org/abs/2302.04761
