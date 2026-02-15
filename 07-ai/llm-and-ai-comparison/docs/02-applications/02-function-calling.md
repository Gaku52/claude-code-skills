# Function Calling — ツール使用・スキーマ定義・エラーハンドリング

> Function Calling は LLM が外部ツール (API、データベース、計算機等) を構造化された形式で呼び出す仕組みであり、LLM を「考えるだけ」の存在から「行動できる」エージェントへ進化させる中核技術である。

## この章で学ぶこと

1. **Function Calling の仕組みと設計原理** — LLM がどのように関数呼び出しを判断し、引数を生成するか
2. **スキーマ定義のベストプラクティス** — JSON Schema による関数定義、パラメータ設計、説明文の書き方
3. **実践的なエラーハンドリングとセキュリティ** — 障害時のフォールバック、入力検証、権限管理
4. **マルチツールオーケストレーション** — ツールチェイン、並列実行、動的ツール選択
5. **本番環境での運用パターン** — モニタリング、コスト最適化、テスト戦略

---

## 1. Function Calling の仕組み

```
┌──────────────────────────────────────────────────────────┐
│           Function Calling の実行フロー                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. ユーザー: 「東京の明日の天気を教えて」                │
│     │                                                    │
│     ▼                                                    │
│  2. LLM: 関数呼び出しが必要と判断                         │
│     → get_weather(city="Tokyo", date="2025-03-15")       │
│     │                                                    │
│     ▼                                                    │
│  3. アプリ: 関数を実行し結果を取得                        │
│     → {"temp": 18, "condition": "晴れ", ...}             │
│     │                                                    │
│     ▼                                                    │
│  4. LLM: 結果を自然言語で回答                             │
│     → 「明日の東京は晴れで、気温は18度の予想です。」      │
│                                                          │
│  ┌──────────────────────────────────────────────┐        │
│  │  重要: LLM は関数を直接実行しない             │        │
│  │  LLM は「どの関数を、どの引数で呼ぶか」を     │        │
│  │  JSON で出力するだけ。実行はアプリ側の責任。   │        │
│  └──────────────────────────────────────────────┘        │
└──────────────────────────────────────────────────────────┘
```

### 1.1 Function Calling の内部メカニズム

```
┌─────────────────────────────────────────────────────────────────┐
│           LLM 内部での Function Calling 判断プロセス              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ユーザー入力 + ツール定義 (システムプロンプトに埋め込まれる)      │
│       │                                                        │
│       ▼                                                        │
│  ┌──────────────────────┐                                      │
│  │  意図分析              │                                      │
│  │  - ツール呼び出しが    │                                      │
│  │    必要か判定          │                                      │
│  │  - どのツールが適切か  │                                      │
│  └──────┬───────────────┘                                      │
│         │                                                      │
│    ┌────┴────┐                                                 │
│    ▼         ▼                                                 │
│  [直接回答]  [ツール呼び出し]                                    │
│              │                                                  │
│              ▼                                                  │
│  ┌──────────────────────┐                                      │
│  │  引数生成              │                                      │
│  │  - JSON Schema に従い  │                                      │
│  │    引数を構造化出力    │                                      │
│  │  - required の検証     │                                      │
│  │  - enum の制約チェック │                                      │
│  └──────────────────────┘                                      │
│                                                                 │
│  ポイント:                                                      │
│  - ツール定義の description が判断の最大の手がかり               │
│  - パラメータの description も引数生成精度に直結                 │
│  - ツール数が増えると選択精度が低下 (20個以下推奨)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Function Calling vs 他のアプローチ

| アプローチ | 仕組み | 精度 | 柔軟性 | 導入コスト |
|-----------|--------|------|--------|----------|
| Function Calling (ネイティブ) | LLM API の組み込み機能 | 高 | 中 | 低 |
| ReAct プロンプティング | プロンプトでツール使用を指示 | 中 | 高 | 低 |
| Code Interpreter | LLM がコードを生成して実行 | 高 | 最高 | 中 |
| プラグインシステム | LLM にプラグイン一覧を提供 | 中〜高 | 高 | 高 |
| MCP (Model Context Protocol) | 標準化されたツール接続 | 高 | 最高 | 中 |

---

## 2. プロバイダ別の実装

### 2.1 OpenAI Function Calling

```python
from openai import OpenAI
import json

client = OpenAI()

# ツール定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "指定された都市の天気予報を取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "都市名 (例: 'Tokyo', 'Osaka')",
                    },
                    "date": {
                        "type": "string",
                        "description": "日付 (YYYY-MM-DD形式)",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度の単位",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# LLM呼び出し
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "東京の明日の天気は？"}],
    tools=tools,
    tool_choice="auto",  # auto / required / none / {"type":"function","function":{"name":"..."}}
)

# Function Call の処理
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"関数: {function_name}, 引数: {arguments}")

        # 実際の関数を実行
        result = execute_function(function_name, arguments)

        # 結果を LLM に返す
        follow_up = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": "東京の明日の天気は？"},
                message,  # LLMの関数呼び出し要求
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                },
            ],
        )
        print(follow_up.choices[0].message.content)
```

### 2.2 OpenAI Structured Output + Function Calling

```python
from openai import OpenAI
from pydantic import BaseModel

client = OpenAI()

# Pydantic モデルでスキーマを定義 (Structured Output)
class FlightSearch(BaseModel):
    origin: str
    destination: str
    date: str
    passengers: int = 1
    cabin_class: str = "economy"

class FlightSearchResult(BaseModel):
    flights: list[dict]
    total_count: int
    cheapest_price: float

# Structured Output を使ったツール定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "フライトを検索します",
            "parameters": FlightSearch.model_json_schema(),
            "strict": True,  # Structured Output モード
        },
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "来週の月曜に東京から大阪へのフライトを探して"}],
    tools=tools,
)

# strict: True により、スキーマに完全に準拠した引数が保証される
if response.choices[0].message.tool_calls:
    args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    validated = FlightSearch(**args)  # Pydantic で追加検証
    print(validated)
```

### 2.3 Anthropic Tool Use

```python
from anthropic import Anthropic

client = Anthropic()

# ツール定義
tools = [
    {
        "name": "get_weather",
        "description": "指定された都市の天気予報を取得します",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "都市名",
                },
                "date": {
                    "type": "string",
                    "description": "日付 (YYYY-MM-DD)",
                },
            },
            "required": ["city"],
        },
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "東京の天気を教えて"}],
)

# tool_use ブロックを処理
for block in response.content:
    if block.type == "tool_use":
        print(f"ツール: {block.name}, 入力: {block.input}")

        # 結果を返す
        result = execute_function(block.name, block.input)

        follow_up = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": "東京の天気を教えて"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result),
                        }
                    ],
                },
            ],
        )
```

### 2.4 Anthropic Tool Use with Streaming

```python
from anthropic import Anthropic
import json

client = Anthropic()

def stream_with_tools(user_message: str, tools: list):
    """ストリーミング対応のツール使用"""

    messages = [{"role": "user", "content": user_message}]

    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    ) as stream:
        tool_use_blocks = []
        text_content = ""

        for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    tool_use_blocks.append({
                        "id": event.content_block.id,
                        "name": event.content_block.name,
                        "input_json": "",
                    })
            elif event.type == "content_block_delta":
                if hasattr(event.delta, "partial_json"):
                    tool_use_blocks[-1]["input_json"] += event.delta.partial_json
                elif hasattr(event.delta, "text"):
                    text_content += event.delta.text
                    print(event.delta.text, end="", flush=True)

        # ツール呼び出しがある場合は実行
        for tool_block in tool_use_blocks:
            input_data = json.loads(tool_block["input_json"])
            result = execute_function(tool_block["name"], input_data)
            print(f"\n[ツール実行] {tool_block['name']}: {result}")
```

### 2.5 Gemini Function Calling

```python
import google.generativeai as genai

genai.configure(api_key="YOUR_API_KEY")

# 関数宣言
get_weather = genai.protos.FunctionDeclaration(
    name="get_weather",
    description="指定された都市の天気予報を取得します",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "city": genai.protos.Schema(type=genai.protos.Type.STRING),
            "date": genai.protos.Schema(type=genai.protos.Type.STRING),
        },
        required=["city"],
    ),
)

tool = genai.protos.Tool(function_declarations=[get_weather])
model = genai.GenerativeModel("gemini-1.5-pro", tools=[tool])

response = model.generate_content("東京の明日の天気は？")

# function_call を処理
for part in response.parts:
    if fn := part.function_call:
        print(f"関数: {fn.name}, 引数: {dict(fn.args)}")
```

### 2.6 Gemini 自動関数実行モード

```python
import google.generativeai as genai

def get_weather_impl(city: str, date: str = None) -> dict:
    """実際の天気取得関数"""
    # API コール等の実装
    return {"city": city, "temp": 22, "condition": "晴れ"}

# 自動関数実行モード: LLM が関数呼び出しを出力すると自動で実行
model = genai.GenerativeModel(
    "gemini-1.5-pro",
    tools=[get_weather_impl],  # Python 関数を直接渡す
)

# enable_automatic_function_calling で自動実行
chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message("東京の明日の天気は？")

# → 関数が自動実行され、最終回答が直接返される
print(response.text)
# "東京の明日の天気は晴れで、気温は22度の予想です。"
```

---

## 3. スキーマ設計のベストプラクティス

### 3.1 良いスキーマの書き方

```python
# OK: 詳細な説明と制約を持つスキーマ
good_schema = {
    "name": "search_products",
    "description": (
        "ECサイトの商品を検索します。"
        "キーワード、カテゴリ、価格帯で絞り込みが可能です。"
        "結果は関連度順で最大20件返されます。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索キーワード (例: 'ワイヤレスイヤホン ノイキャン')",
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "fashion", "books", "food", "sports"],
                "description": "商品カテゴリ。省略時は全カテゴリを検索",
            },
            "price_min": {
                "type": "integer",
                "minimum": 0,
                "description": "最低価格 (円)。省略時は下限なし",
            },
            "price_max": {
                "type": "integer",
                "minimum": 0,
                "description": "最高価格 (円)。省略時は上限なし",
            },
            "sort_by": {
                "type": "string",
                "enum": ["relevance", "price_asc", "price_desc", "rating", "newest"],
                "description": "ソート順。デフォルトは relevance",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "取得件数。デフォルトは10",
            },
        },
        "required": ["query"],
    },
}
```

### 3.2 スキーマ設計チェックリスト

```python
def validate_tool_schema(schema: dict) -> list[str]:
    """ツールスキーマの品質チェック"""

    warnings = []

    # 1. 関数名のチェック
    name = schema.get("name", "")
    if not name:
        warnings.append("ERROR: name が未定義")
    elif "_" not in name and len(name) > 15:
        warnings.append("WARNING: 関数名が長すぎます。snake_case で簡潔に")

    # 2. description のチェック
    desc = schema.get("description", "")
    if len(desc) < 20:
        warnings.append("WARNING: description が短すぎます (20文字以上推奨)")
    if "例" not in desc and "example" not in desc.lower():
        warnings.append("HINT: description に使用例を含めると精度向上")

    # 3. パラメータのチェック
    params = schema.get("parameters", {}).get("properties", {})
    for param_name, param_def in params.items():
        param_desc = param_def.get("description", "")
        if not param_desc:
            warnings.append(f"WARNING: パラメータ '{param_name}' に description がありません")
        if param_def.get("type") == "string" and "enum" not in param_def:
            if "format" not in param_def:
                warnings.append(
                    f"HINT: パラメータ '{param_name}' に enum または format を追加すると精度向上"
                )

    # 4. required のチェック
    required = schema.get("parameters", {}).get("required", [])
    if not required:
        warnings.append("HINT: required パラメータを明示すると LLM の判断が改善")

    return warnings


# 使用例
schema = {
    "name": "query_db",
    "description": "DBを検索",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"},
        },
    },
}

issues = validate_tool_schema(schema)
for issue in issues:
    print(issue)
# WARNING: description が短すぎます (20文字以上推奨)
# HINT: description に使用例を含めると精度向上
# WARNING: パラメータ 'q' に description がありません
# HINT: パラメータ 'q' に enum または format を追加すると精度向上
# HINT: required パラメータを明示すると LLM の判断が改善
```

### 3.3 複雑なスキーマのパターン

```python
# パターン1: ネストされたオブジェクト
nested_schema = {
    "name": "create_order",
    "description": "注文を作成します",
    "parameters": {
        "type": "object",
        "properties": {
            "customer": {
                "type": "object",
                "description": "顧客情報",
                "properties": {
                    "name": {"type": "string", "description": "顧客名"},
                    "email": {"type": "string", "format": "email", "description": "メールアドレス"},
                    "phone": {"type": "string", "description": "電話番号 (ハイフン付き)"},
                },
                "required": ["name", "email"],
            },
            "items": {
                "type": "array",
                "description": "注文商品リスト",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string", "description": "商品ID"},
                        "quantity": {"type": "integer", "minimum": 1, "description": "数量"},
                    },
                    "required": ["product_id", "quantity"],
                },
                "minItems": 1,
            },
            "shipping_address": {
                "type": "object",
                "description": "配送先住所",
                "properties": {
                    "postal_code": {"type": "string", "pattern": "^\\d{3}-\\d{4}$"},
                    "prefecture": {"type": "string"},
                    "city": {"type": "string"},
                    "street": {"type": "string"},
                },
                "required": ["postal_code", "prefecture", "city", "street"],
            },
        },
        "required": ["customer", "items", "shipping_address"],
    },
}

# パターン2: 条件分岐のあるスキーマ (oneOf)
conditional_schema = {
    "name": "process_payment",
    "description": "支払いを処理します。クレジットカードまたは銀行振込を選択",
    "parameters": {
        "type": "object",
        "properties": {
            "amount": {"type": "number", "description": "金額 (円)"},
            "method": {
                "type": "string",
                "enum": ["credit_card", "bank_transfer"],
                "description": "支払い方法",
            },
            "credit_card": {
                "type": "object",
                "description": "クレジットカード情報 (method=credit_card の場合必須)",
                "properties": {
                    "number": {"type": "string", "description": "カード番号 (16桁)"},
                    "expiry": {"type": "string", "description": "有効期限 (MM/YY)"},
                    "cvv": {"type": "string", "description": "セキュリティコード (3桁)"},
                },
            },
            "bank_account": {
                "type": "object",
                "description": "銀行口座情報 (method=bank_transfer の場合必須)",
                "properties": {
                    "bank_name": {"type": "string"},
                    "account_number": {"type": "string"},
                },
            },
        },
        "required": ["amount", "method"],
    },
}
```

### 3.4 プロバイダ別のスキーマ形式

```
┌──────────────────────────────────────────────────────────┐
│        プロバイダ別 Function Calling 仕様比較              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  OpenAI                                                  │
│  ├── tools[].function.parameters (JSON Schema)           │
│  ├── tool_choice: auto / required / none / specific      │
│  ├── parallel_tool_calls: true/false                     │
│  └── Structured Output (response_format) と併用可        │
│                                                          │
│  Anthropic                                               │
│  ├── tools[].input_schema (JSON Schema)                  │
│  ├── tool_choice: auto / any / tool (specific)           │
│  ├── 並列ツール呼び出し対応                               │
│  └── <thinking> タグで推論過程を表示可能                  │
│                                                          │
│  Google Gemini                                           │
│  ├── FunctionDeclaration (protobuf形式)                  │
│  ├── function_calling_config: AUTO / ANY / NONE          │
│  ├── allowed_function_names で制限可能                    │
│  └── 自動関数実行モード (automatic_function_calling)     │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 複数ツールのオーケストレーション

### 4.1 ツールチェイン

```python
import json
from openai import OpenAI

client = OpenAI()

# 複数のツール定義
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "フライトを検索します",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string", "description": "出発地 (空港コード)"},
                    "destination": {"type": "string", "description": "目的地 (空港コード)"},
                    "date": {"type": "string", "description": "出発日 (YYYY-MM-DD)"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "フライトを予約します",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_id": {"type": "string"},
                    "passenger_name": {"type": "string"},
                },
                "required": ["flight_id", "passenger_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "ホテルを検索します",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "check_in": {"type": "string"},
                    "check_out": {"type": "string"},
                },
                "required": ["city", "check_in", "check_out"],
            },
        },
    },
]

def agent_loop(user_message: str):
    """マルチツール対応のエージェントループ"""
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content  # 最終回答

        # 全ツール呼び出しを処理
        for tool_call in message.tool_calls:
            result = execute_function(
                tool_call.function.name,
                json.loads(tool_call.function.arguments),
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

# 使用例: 複数ツールが連鎖的に呼ばれる
answer = agent_loop("来週の月曜日に東京から大阪への出張を手配して。ホテルも1泊必要です。")
```

### 4.2 並列ツール実行

```python
import asyncio
import json
from openai import OpenAI

client = OpenAI()

async def execute_tools_parallel(tool_calls: list) -> list[dict]:
    """複数のツール呼び出しを並列実行"""

    async def execute_one(tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # 各ツールを非同期で実行
        result = await asyncio.to_thread(execute_function, name, args)
        return {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False),
        }

    # 全ツールを並列実行
    results = await asyncio.gather(
        *[execute_one(tc) for tc in tool_calls],
        return_exceptions=True,
    )

    # エラーハンドリング
    tool_results = []
    for result, tool_call in zip(results, tool_calls):
        if isinstance(result, Exception):
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({"error": str(result)}),
            })
        else:
            tool_results.append(result)

    return tool_results


async def agent_loop_async(user_message: str, max_iterations: int = 10):
    """非同期エージェントループ (並列ツール実行対応)"""
    messages = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            parallel_tool_calls=True,  # 並列呼び出しを有効化
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        # 並列実行
        tool_results = await execute_tools_parallel(message.tool_calls)
        messages.extend(tool_results)

    return "処理が上限に達しました。"
```

### 4.3 動的ツール選択 (ツールルーティング)

```python
from openai import OpenAI

client = OpenAI()

# 全ツール定義 (大量にある場合)
ALL_TOOLS = {
    "weather": [weather_tool_1, weather_tool_2],
    "travel": [flight_tool, hotel_tool, car_tool],
    "finance": [stock_tool, exchange_tool, portfolio_tool],
    "hr": [employee_tool, leave_tool, payroll_tool],
    "it": [ticket_tool, deploy_tool, monitor_tool],
}

def select_relevant_tools(user_message: str, max_tools: int = 10) -> list:
    """ユーザーメッセージに基づいてツールを動的に選択"""

    # 1. LLM でカテゴリを判定
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""以下のユーザーメッセージに関連するツールカテゴリを選択してください。
カテゴリ: {list(ALL_TOOLS.keys())}
カンマ区切りで返してください。

メッセージ: {user_message}""",
        }],
    )

    categories = [c.strip() for c in response.choices[0].message.content.split(",")]

    # 2. 該当カテゴリのツールを集約
    selected = []
    for cat in categories:
        if cat in ALL_TOOLS:
            selected.extend(ALL_TOOLS[cat])

    # 3. ツール数が多すぎる場合は制限
    return selected[:max_tools]


def agent_with_dynamic_tools(user_message: str):
    """動的ツール選択付きエージェント"""
    # Step 1: 関連ツールを選択
    relevant_tools = select_relevant_tools(user_message)

    # Step 2: 選択されたツールでエージェントループ
    return agent_loop(user_message, tools=relevant_tools)
```

### 4.4 ツール結果の圧縮

```python
def compress_tool_result(result: dict, max_length: int = 2000) -> str:
    """ツール実行結果を LLM フレンドリーなサイズに圧縮"""

    result_str = json.dumps(result, ensure_ascii=False)

    if len(result_str) <= max_length:
        return result_str

    # 方法1: キーの優先度による切り詰め
    if isinstance(result, dict):
        priority_keys = ["summary", "title", "name", "status", "error", "count"]
        compressed = {}
        remaining_budget = max_length

        # 優先キーを先に追加
        for key in priority_keys:
            if key in result:
                value = result[key]
                entry = json.dumps({key: value}, ensure_ascii=False)
                if len(entry) < remaining_budget:
                    compressed[key] = value
                    remaining_budget -= len(entry)

        # 残りのキーを追加
        for key, value in result.items():
            if key not in compressed:
                entry = json.dumps({key: value}, ensure_ascii=False)
                if len(entry) < remaining_budget:
                    compressed[key] = value
                    remaining_budget -= len(entry)

        return json.dumps(compressed, ensure_ascii=False)

    # 方法2: リスト結果の件数制限
    if isinstance(result, list):
        truncated = result[:10]  # 最初の10件のみ
        return json.dumps({
            "items": truncated,
            "total_count": len(result),
            "truncated": len(result) > 10,
        }, ensure_ascii=False)

    # 方法3: 文字列の切り詰め
    return result_str[:max_length] + "... (truncated)"
```

---

## 5. エラーハンドリング

### 5.1 堅牢なエラーハンドリング

```python
import json
from typing import Any
from enum import Enum

class FunctionCallStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    UNAUTHORIZED = "unauthorized"
    RATE_LIMITED = "rate_limited"

class FunctionCallError(Exception):
    pass

def safe_execute_function(name: str, arguments: dict) -> dict[str, Any]:
    """安全な関数実行ラッパー"""

    # 1. 関数の存在確認
    registry = {
        "get_weather": get_weather,
        "search_products": search_products,
        "create_order": create_order,
    }

    if name not in registry:
        return {
            "error": f"Unknown function: {name}",
            "status": FunctionCallStatus.ERROR.value,
            "suggestion": f"Available functions: {list(registry.keys())}",
        }

    # 2. 引数の検証
    try:
        validated_args = validate_arguments(name, arguments)
    except ValueError as e:
        return {
            "error": f"Invalid arguments: {e}",
            "status": FunctionCallStatus.ERROR.value,
        }

    # 3. 権限チェック
    if not check_permission(name, current_user):
        return {
            "error": f"Permission denied for function: {name}",
            "status": FunctionCallStatus.UNAUTHORIZED.value,
        }

    # 4. 実行 (タイムアウト付き)
    try:
        import asyncio
        result = asyncio.wait_for(
            registry[name](**validated_args),
            timeout=10.0,  # 10秒タイムアウト
        )
        return {"result": result, "status": FunctionCallStatus.SUCCESS.value}
    except asyncio.TimeoutError:
        return {
            "error": "Function execution timed out (10s limit)",
            "status": FunctionCallStatus.TIMEOUT.value,
        }
    except Exception as e:
        return {
            "error": str(e),
            "status": FunctionCallStatus.ERROR.value,
            "traceback": traceback.format_exc() if DEBUG else None,
        }

def validate_arguments(function_name: str, args: dict) -> dict:
    """引数の検証とサニタイズ"""
    # SQLインジェクション等の防止
    for key, value in args.items():
        if isinstance(value, str):
            if any(dangerous in value.lower() for dangerous in
                   ["drop table", "delete from", "; --", "' or '1'='1"]):
                raise ValueError(f"Potentially dangerous input in {key}")

            # XSS 対策
            if "<script" in value.lower():
                raise ValueError(f"Script tags not allowed in {key}")

    return args
```

### 5.2 リトライとフォールバック

```python
import time
from functools import wraps

class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0

def with_retry(config: RetryConfig = RetryConfig()):
    """リトライデコレータ"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < config.max_retries:
                        delay = min(
                            config.base_delay * (config.exponential_base ** attempt),
                            config.max_delay,
                        )
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator


class ToolExecutor:
    """フォールバック付きツール実行"""

    def __init__(self):
        self.fallbacks: dict[str, list[callable]] = {}

    def register_fallback(self, tool_name: str, fallback_fn: callable):
        """ツールにフォールバック関数を登録"""
        if tool_name not in self.fallbacks:
            self.fallbacks[tool_name] = []
        self.fallbacks[tool_name].append(fallback_fn)

    async def execute(self, name: str, args: dict) -> dict:
        """フォールバックチェイン付き実行"""

        # プライマリ実行
        result = await safe_execute_function(name, args)
        if result["status"] == "success":
            return result

        # フォールバック実行
        if name in self.fallbacks:
            for fallback_fn in self.fallbacks[name]:
                try:
                    fb_result = await fallback_fn(**args)
                    return {
                        "result": fb_result,
                        "status": "success",
                        "fallback_used": True,
                    }
                except Exception:
                    continue

        return result  # 全フォールバック失敗時


# 使用例
executor = ToolExecutor()

# プライマリ: OpenWeather API
# フォールバック1: WeatherAPI
# フォールバック2: キャッシュから返す
executor.register_fallback("get_weather", get_weather_from_backup_api)
executor.register_fallback("get_weather", get_weather_from_cache)
```

### 5.3 LLM への適切なエラー通知

```python
def format_error_for_llm(error_result: dict) -> str:
    """エラーを LLM が理解しやすい形式にフォーマット"""

    status = error_result.get("status", "error")
    error_msg = error_result.get("error", "Unknown error")

    if status == "timeout":
        return json.dumps({
            "error": True,
            "message": "この操作はタイムアウトしました。",
            "suggestion": "条件を絞り込むか、時間をおいて再試行してください。",
        }, ensure_ascii=False)

    elif status == "unauthorized":
        return json.dumps({
            "error": True,
            "message": "この操作を実行する権限がありません。",
            "suggestion": "ユーザーに権限が必要であることを伝えてください。",
        }, ensure_ascii=False)

    elif status == "rate_limited":
        return json.dumps({
            "error": True,
            "message": "API のレート制限に達しました。",
            "suggestion": "しばらく待ってから再試行するか、別の方法を提案してください。",
        }, ensure_ascii=False)

    else:
        return json.dumps({
            "error": True,
            "message": f"エラーが発生しました: {error_msg}",
            "suggestion": "ユーザーにエラーが発生したことを伝え、代替手段を提案してください。",
        }, ensure_ascii=False)
```

---

## 6. セキュリティ

### 6.1 権限管理フレームワーク

```python
from enum import Enum
from dataclasses import dataclass

class PermissionLevel(Enum):
    PUBLIC = "public"           # 誰でも使用可能
    AUTHENTICATED = "authenticated"  # ログインユーザーのみ
    ADMIN = "admin"             # 管理者のみ
    SYSTEM = "system"           # システム内部のみ

@dataclass
class ToolPermission:
    tool_name: str
    required_level: PermissionLevel
    allowed_roles: list[str] | None = None
    rate_limit: int | None = None  # 1分あたりの呼び出し上限
    requires_confirmation: bool = False  # ユーザー確認が必要

# ツール権限設定
TOOL_PERMISSIONS = {
    "search_products": ToolPermission("search_products", PermissionLevel.PUBLIC),
    "get_user_profile": ToolPermission("get_user_profile", PermissionLevel.AUTHENTICATED),
    "create_order": ToolPermission(
        "create_order",
        PermissionLevel.AUTHENTICATED,
        requires_confirmation=True,  # 注文前に確認
    ),
    "delete_user": ToolPermission(
        "delete_user",
        PermissionLevel.ADMIN,
        allowed_roles=["super_admin"],
        requires_confirmation=True,
    ),
    "execute_sql": ToolPermission(
        "execute_sql",
        PermissionLevel.SYSTEM,  # API 経由では呼び出し不可
    ),
}

class PermissionChecker:
    def check(self, tool_name: str, user: dict) -> tuple[bool, str]:
        """ツール使用権限をチェック"""
        perm = TOOL_PERMISSIONS.get(tool_name)
        if not perm:
            return False, f"Unknown tool: {tool_name}"

        user_level = user.get("permission_level", "public")
        user_roles = user.get("roles", [])

        # レベルチェック
        level_order = [e.value for e in PermissionLevel]
        if level_order.index(user_level) < level_order.index(perm.required_level.value):
            return False, f"Insufficient permission level: requires {perm.required_level.value}"

        # ロールチェック
        if perm.allowed_roles:
            if not any(role in perm.allowed_roles for role in user_roles):
                return False, f"Required roles: {perm.allowed_roles}"

        # レートリミットチェック
        if perm.rate_limit:
            current_count = get_rate_limit_count(tool_name, user["id"])
            if current_count >= perm.rate_limit:
                return False, "Rate limit exceeded"

        return True, "OK"
```

### 6.2 入力サニタイゼーション

```python
import re
from typing import Any

class InputSanitizer:
    """LLM が生成した引数のサニタイゼーション"""

    DANGEROUS_PATTERNS = [
        r";\s*--",               # SQL コメント
        r"'\s*OR\s*'1'\s*=\s*'1", # SQL インジェクション
        r"DROP\s+TABLE",          # SQL 破壊コマンド
        r"<script[^>]*>",        # XSS
        r"\{\{.*\}\}",           # テンプレートインジェクション
        r"\$\{.*\}",             # テンプレートリテラル
        r"__import__",           # Python コード注入
        r"eval\s*\(",            # eval 実行
        r"exec\s*\(",            # exec 実行
    ]

    @classmethod
    def sanitize(cls, args: dict, schema: dict) -> dict:
        """スキーマに基づいて引数をサニタイズ"""

        sanitized = {}
        properties = schema.get("parameters", {}).get("properties", {})

        for key, value in args.items():
            if key not in properties:
                continue  # スキーマに定義されていないパラメータは除去

            prop_def = properties[key]

            # 型チェック
            expected_type = prop_def.get("type")
            value = cls._coerce_type(value, expected_type)

            # 文字列のサニタイズ
            if isinstance(value, str):
                value = cls._sanitize_string(value, prop_def)

            # 数値の範囲チェック
            if isinstance(value, (int, float)):
                value = cls._clamp_number(value, prop_def)

            sanitized[key] = value

        return sanitized

    @classmethod
    def _sanitize_string(cls, value: str, prop_def: dict) -> str:
        """文字列のサニタイズ"""
        # 危険なパターンのチェック
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(f"Dangerous input detected: matches pattern {pattern}")

        # enum チェック
        if "enum" in prop_def and value not in prop_def["enum"]:
            raise ValueError(f"Value '{value}' not in allowed enum: {prop_def['enum']}")

        # 最大長チェック
        max_length = prop_def.get("maxLength", 10000)
        if len(value) > max_length:
            value = value[:max_length]

        return value

    @classmethod
    def _clamp_number(cls, value: float, prop_def: dict) -> float:
        """数値の範囲制限"""
        if "minimum" in prop_def:
            value = max(value, prop_def["minimum"])
        if "maximum" in prop_def:
            value = min(value, prop_def["maximum"])
        return value

    @classmethod
    def _coerce_type(cls, value: Any, expected_type: str) -> Any:
        """型の変換"""
        try:
            if expected_type == "integer":
                return int(value)
            elif expected_type == "number":
                return float(value)
            elif expected_type == "boolean":
                return bool(value)
            elif expected_type == "string":
                return str(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {value} to {expected_type}")
        return value
```

### 6.3 確認フロー

```python
class ConfirmationManager:
    """破壊的操作の確認フロー"""

    DESTRUCTIVE_ACTIONS = {
        "delete_user": "ユーザー '{name}' を完全に削除します。この操作は取り消せません。",
        "cancel_order": "注文 #{order_id} をキャンセルします。返金処理が開始されます。",
        "create_order": "以下の注文を確定します:\n{items}\n合計: {total}円",
        "deploy_service": "サービス '{service}' を {environment} にデプロイします。",
    }

    def needs_confirmation(self, tool_name: str) -> bool:
        """確認が必要か判定"""
        return tool_name in self.DESTRUCTIVE_ACTIONS

    def generate_confirmation_message(self, tool_name: str, args: dict) -> str:
        """確認メッセージを生成"""
        template = self.DESTRUCTIVE_ACTIONS.get(tool_name, "この操作を実行しますか？")
        try:
            return template.format(**args)
        except KeyError:
            return template

    def create_pending_action(self, tool_name: str, args: dict) -> dict:
        """保留中のアクションを作成"""
        import uuid
        action_id = str(uuid.uuid4())
        return {
            "action_id": action_id,
            "tool_name": tool_name,
            "args": args,
            "confirmation_message": self.generate_confirmation_message(tool_name, args),
            "status": "pending_confirmation",
            "requires_user_approval": True,
        }
```

---

## 7. 比較表

### 7.1 プロバイダ別 Function Calling 機能比較

| 機能 | OpenAI | Anthropic | Google Gemini |
|------|--------|-----------|--------------|
| 並列呼び出し | 対応 | 対応 | 対応 |
| ストリーミング | 対応 | 対応 | 対応 |
| 強制呼び出し | tool_choice | tool_choice | function_calling_config |
| 最大ツール数 | 128 | 制限なし(推奨20) | 制限なし |
| ネストJSON | 対応 | 対応 | 対応 |
| Structured Output | 対応 | JSON Mode | 対応 |
| 自動実行 | なし | なし | あり (opt-in) |
| MCP 対応 | 対応 | 対応 (ネイティブ) | 対応 |

### 7.2 ユースケース別のツール設計

| ユースケース | ツール数 | 設計パターン | 注意点 |
|-------------|---------|------------|--------|
| 天気Bot | 1-2 | 単一ツール | シンプルに保つ |
| ECアシスタント | 5-10 | ツールチェイン | 状態管理が重要 |
| 社内業務Bot | 10-20 | ルーター型 | 権限管理必須 |
| 自律エージェント | 20+ | ReAct パターン | ループ上限設定 |
| マルチモーダル | 5-15 | パイプライン | 入出力型の整合性 |

---

## 8. MCP (Model Context Protocol) との連携

### 8.1 MCP 概要

```
┌─────────────────────────────────────────────────────────────────┐
│              MCP (Model Context Protocol) アーキテクチャ          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐      ┌──────────┐      ┌──────────────────┐     │
│  │  LLM     │      │  MCP     │      │  MCP Server      │     │
│  │  Client  │◀────▶│  Host    │◀────▶│  (ツール提供)     │     │
│  │  (Claude │      │ (アプリ)  │      │  - DB検索        │     │
│  │   GPT等)  │      │          │      │  - API呼び出し   │     │
│  └──────────┘      └──────────┘      │  - ファイル操作   │     │
│                                       └──────────────────┘     │
│                                                                 │
│  利点:                                                          │
│  - ツール定義の標準化 (プロバイダ間で共通)                        │
│  - ツールの再利用性向上                                          │
│  - セキュリティの一元管理                                        │
│  - ツールの動的発見と登録                                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 MCP Server の実装

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("weather-server")

@server.tool("get_weather")
async def get_weather(city: str, date: str = None) -> list[TextContent]:
    """指定された都市の天気予報を取得します"""

    # 実際の API 呼び出し
    weather_data = await fetch_weather_api(city, date)

    return [TextContent(
        type="text",
        text=json.dumps({
            "city": city,
            "temperature": weather_data["temp"],
            "condition": weather_data["condition"],
            "humidity": weather_data["humidity"],
        }, ensure_ascii=False),
    )]


@server.tool("get_forecast")
async def get_forecast(city: str, days: int = 7) -> list[TextContent]:
    """指定された都市の週間天気予報を取得します"""

    forecast_data = await fetch_forecast_api(city, days)

    return [TextContent(
        type="text",
        text=json.dumps(forecast_data, ensure_ascii=False),
    )]


# サーバー起動
if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server

    asyncio.run(stdio_server(server))
```

---

## 9. 実務ユースケース

### 9.1 カスタマーサポート Bot

```python
# カスタマーサポート向けツール定義
support_tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": (
                "注文番号で注文情報を検索します。"
                "注文状況、配送状況、商品詳細が確認できます。"
                "注文番号は 'ORD-' で始まる文字列です。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "pattern": "^ORD-\\d{8}$",
                        "description": "注文番号 (例: ORD-20250315)",
                    },
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": (
                "商品の返品可否を確認します。"
                "購入日から30日以内かつ未使用の場合のみ返品可能です。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "item_id": {"type": "string"},
                    "reason": {
                        "type": "string",
                        "enum": ["defective", "wrong_item", "not_as_described", "change_of_mind", "other"],
                        "description": "返品理由",
                    },
                },
                "required": ["order_id", "item_id", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "サポートチケットを作成します。"
                "エスカレーションが必要な場合や、自動対応できない問題の場合に使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "チケットの件名"},
                    "description": {"type": "string", "description": "問題の詳細"},
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "urgent"],
                    },
                    "category": {
                        "type": "string",
                        "enum": ["order", "payment", "shipping", "product", "account", "other"],
                    },
                },
                "required": ["subject", "description", "priority", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_faq",
            "description": (
                "よくある質問 (FAQ) データベースを検索します。"
                "一般的な質問にはまずこの関数で回答を検索してください。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "検索クエリ"},
                },
                "required": ["query"],
            },
        },
    },
]


class CustomerSupportAgent:
    """カスタマーサポートエージェント"""

    def __init__(self, client: OpenAI):
        self.client = client
        self.system_prompt = """あなたは親切で丁寧なカスタマーサポートアシスタントです。

ルール:
1. まず FAQ を検索し、一般的な質問には FAQ の回答を使用してください
2. 注文に関する質問では、必ず注文番号を確認してください
3. 返品希望の場合は、返品資格を確認してから手続きを案内してください
4. 自動対応できない場合は、サポートチケットを作成してください
5. 個人情報 (クレジットカード番号等) は決して求めないでください
6. 常に丁寧語で対応してください"""

    async def handle(self, user_message: str, conversation_history: list) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *conversation_history,
            {"role": "user", "content": user_message},
        ]

        return await agent_loop_async(messages, tools=support_tools, max_iterations=5)
```

### 9.2 データ分析アシスタント

```python
analysis_tools = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "SQLクエリを実行してデータを取得します。"
                "SELECT文のみ実行可能です (INSERT/UPDATE/DELETE は不可)。"
                "テーブル: users, orders, products, sessions"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "実行する SELECT SQL クエリ",
                    },
                    "limit": {
                        "type": "integer",
                        "maximum": 1000,
                        "description": "最大取得行数 (デフォルト: 100)",
                    },
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": "データからグラフを生成します",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "line", "pie", "scatter", "heatmap"],
                    },
                    "title": {"type": "string"},
                    "data": {
                        "type": "object",
                        "description": "x: ラベル配列, y: 値配列",
                        "properties": {
                            "x": {"type": "array", "items": {"type": "string"}},
                            "y": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                },
                "required": ["chart_type", "title", "data"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_statistics",
            "description": "数値配列の統計量を計算します (平均、中央値、標準偏差等)",
            "parameters": {
                "type": "object",
                "properties": {
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "数値の配列",
                    },
                    "metrics": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["mean", "median", "std", "min", "max", "percentiles"],
                        },
                        "description": "計算する統計量",
                    },
                },
                "required": ["values"],
            },
        },
    },
]
```

### 9.3 DevOps 自動化エージェント

```python
devops_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": "サービスの稼働状態を確認します",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {
                        "type": "string",
                        "enum": ["api-gateway", "user-service", "order-service", "payment-service"],
                    },
                    "environment": {
                        "type": "string",
                        "enum": ["production", "staging", "development"],
                    },
                },
                "required": ["service_name", "environment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metrics",
            "description": "サービスのメトリクスを取得します (CPU, メモリ, レスポンスタイム等)",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "metric_type": {
                        "type": "string",
                        "enum": ["cpu", "memory", "latency", "error_rate", "throughput"],
                    },
                    "time_range": {
                        "type": "string",
                        "enum": ["1h", "6h", "24h", "7d", "30d"],
                        "description": "集計期間",
                    },
                },
                "required": ["service_name", "metric_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scale_service",
            "description": "サービスのレプリカ数をスケールします。本番環境は確認が必要です。",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "environment": {"type": "string"},
                    "replicas": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["service_name", "environment", "replicas"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_logs",
            "description": "サービスのログを取得します",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                    "environment": {"type": "string"},
                    "level": {
                        "type": "string",
                        "enum": ["error", "warn", "info", "debug"],
                    },
                    "time_range": {"type": "string"},
                    "search_query": {"type": "string", "description": "ログ内検索キーワード"},
                },
                "required": ["service_name", "environment"],
            },
        },
    },
]
```

---

## 10. テスト戦略

### 10.1 ツールスキーマのテスト

```python
import pytest
import json
from jsonschema import validate, ValidationError

class TestToolSchemas:
    """ツールスキーマのバリデーションテスト"""

    def test_schema_is_valid_json_schema(self):
        """各ツールのスキーマが有効な JSON Schema であること"""
        for tool in tools:
            schema = tool["function"]["parameters"]
            # JSON Schema Draft 7 に準拠しているか
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_required_fields_exist_in_properties(self):
        """required に指定されたフィールドが properties に存在すること"""
        for tool in tools:
            schema = tool["function"]["parameters"]
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            for field in required:
                assert field in properties, f"Required field '{field}' not in properties"

    def test_enum_values_are_valid(self):
        """enum 値が空でないこと"""
        for tool in tools:
            for prop_name, prop_def in tool["function"]["parameters"].get("properties", {}).items():
                if "enum" in prop_def:
                    assert len(prop_def["enum"]) > 0, f"Empty enum in {prop_name}"

    def test_all_tools_have_description(self):
        """全ツールに description があること"""
        for tool in tools:
            assert tool["function"].get("description"), f"Missing description for {tool['function']['name']}"

    def test_all_parameters_have_description(self):
        """全パラメータに description があること"""
        for tool in tools:
            for prop_name, prop_def in tool["function"]["parameters"].get("properties", {}).items():
                assert prop_def.get("description"), f"Missing description for {prop_name}"


class TestToolExecution:
    """ツール実行のテスト"""

    @pytest.mark.asyncio
    async def test_valid_arguments_succeed(self):
        """有効な引数で正常終了すること"""
        result = await safe_execute_function(
            "get_weather",
            {"city": "Tokyo", "date": "2025-03-15"},
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_invalid_arguments_return_error(self):
        """無効な引数でエラーが返ること"""
        result = await safe_execute_function(
            "get_weather",
            {"city": "'; DROP TABLE users; --"},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_unknown_function_returns_error(self):
        """存在しない関数でエラーが返ること"""
        result = await safe_execute_function(
            "nonexistent_function",
            {},
        )
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """タイムアウトが正しく処理されること"""
        result = await safe_execute_function(
            "slow_function",
            {"delay": 30},  # 30秒のスリープ
        )
        assert result["status"] == "timeout"
```

### 10.2 LLM との統合テスト

```python
class TestLLMFunctionCalling:
    """LLM の関数呼び出し判断のテスト"""

    @pytest.mark.asyncio
    async def test_correct_function_selection(self):
        """適切な関数が選択されること"""
        test_cases = [
            {
                "input": "東京の天気を教えて",
                "expected_function": "get_weather",
                "expected_args": {"city": "Tokyo"},
            },
            {
                "input": "注文 ORD-20250315 の状況を確認したい",
                "expected_function": "lookup_order",
                "expected_args": {"order_id": "ORD-20250315"},
            },
        ]

        for case in test_cases:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": case["input"]}],
                tools=tools,
            )

            if response.choices[0].message.tool_calls:
                actual_fn = response.choices[0].message.tool_calls[0].function.name
                assert actual_fn == case["expected_function"], \
                    f"Expected {case['expected_function']}, got {actual_fn}"

    @pytest.mark.asyncio
    async def test_no_function_call_when_unnecessary(self):
        """関数呼び出しが不要な場合は直接回答すること"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "こんにちは、元気ですか？"}],
            tools=tools,
        )

        assert not response.choices[0].message.tool_calls, \
            "Should not call any function for a greeting"
```

---

## 11. モニタリングとコスト最適化

### 11.1 Function Calling のメトリクス

```python
import time
from dataclasses import dataclass, field

@dataclass
class FunctionCallMetrics:
    """Function Calling のメトリクス"""
    function_name: str
    arguments: dict
    execution_time_ms: float = 0
    status: str = ""
    error: str | None = None
    llm_model: str = ""
    token_usage: dict = field(default_factory=dict)
    retry_count: int = 0

class FunctionCallMonitor:
    """Function Calling のモニタリング"""

    def __init__(self, metrics_backend):
        self.backend = metrics_backend

    def track(self, metrics: FunctionCallMetrics):
        """メトリクスを記録"""
        self.backend.histogram(
            "function_call.execution_time",
            metrics.execution_time_ms,
            tags={"function": metrics.function_name},
        )

        self.backend.counter(
            "function_call.total",
            1,
            tags={
                "function": metrics.function_name,
                "status": metrics.status,
            },
        )

        if metrics.error:
            self.backend.counter(
                "function_call.errors",
                1,
                tags={
                    "function": metrics.function_name,
                    "error_type": type(metrics.error).__name__,
                },
            )

        # トークン使用量
        if metrics.token_usage:
            self.backend.histogram(
                "function_call.tokens",
                metrics.token_usage.get("total_tokens", 0),
                tags={"model": metrics.llm_model},
            )
```

### 11.2 コスト最適化

```python
class CostOptimizer:
    """Function Calling のコスト最適化"""

    def __init__(self):
        self.tool_usage_stats: dict[str, int] = {}

    def optimize_tool_set(self, tools: list, user_context: dict) -> list:
        """ユーザーコンテキストに基づいてツールセットを最適化"""

        # 1. 使用頻度の低いツールを除外
        min_usage = 10  # 過去30日で10回未満は除外
        frequently_used = [
            t for t in tools
            if self.tool_usage_stats.get(t["function"]["name"], 0) >= min_usage
        ]

        # 2. ユーザーの権限に基づいてフィルタ
        authorized = [
            t for t in frequently_used
            if check_permission(t["function"]["name"], user_context)
        ]

        # 3. ツール数を制限 (トークンコスト削減)
        max_tools = 15
        if len(authorized) > max_tools:
            # 使用頻度順でソートして上位のみ
            authorized.sort(
                key=lambda t: self.tool_usage_stats.get(t["function"]["name"], 0),
                reverse=True,
            )
            authorized = authorized[:max_tools]

        return authorized

    def estimate_cost(self, tools: list, model: str = "gpt-4o") -> dict:
        """ツール定義のトークンコストを推定"""
        tools_json = json.dumps(tools, ensure_ascii=False)
        estimated_tokens = len(tools_json) // 4  # 概算

        cost_per_1k_tokens = {
            "gpt-4o": 0.005,
            "gpt-4o-mini": 0.00015,
            "claude-3-5-sonnet": 0.003,
        }

        rate = cost_per_1k_tokens.get(model, 0.005)

        return {
            "estimated_tokens": estimated_tokens,
            "cost_per_request": estimated_tokens / 1000 * rate,
            "cost_per_1000_requests": estimated_tokens * rate,
        }
```

---

## 12. アンチパターン

### アンチパターン 1: 無制限のツール実行ループ

```python
# NG: ツール呼び出しの回数制限なし
while True:
    response = call_llm(messages, tools)
    if not response.tool_calls:
        break
    # → LLM が無限にツールを呼び続ける可能性

# OK: 明示的なループ上限
MAX_ITERATIONS = 10
for i in range(MAX_ITERATIONS):
    response = call_llm(messages, tools)
    if not response.tool_calls:
        break
    # ツール実行...
else:
    return "処理が複雑すぎるため、上限に達しました。質問を分割してください。"
```

### アンチパターン 2: 関数説明の不足

```python
# NG: 説明が不十分
bad_tool = {
    "name": "query_db",
    "description": "DBを検索",  # 何のDB？何を検索？
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"},  # 何を入れる？
        },
    },
}

# OK: 具体的で明確な説明
good_tool = {
    "name": "search_employee_database",
    "description": (
        "社内の従業員データベースを検索します。"
        "名前、部署、スキルで検索できます。"
        "結果には氏名、部署、役職、メールアドレスが含まれます。"
        "個人情報のため、正当な業務目的でのみ使用してください。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "従業員の氏名 (部分一致検索)。例: '田中'",
            },
            "department": {
                "type": "string",
                "enum": ["engineering", "sales", "hr", "finance"],
                "description": "部署コード",
            },
        },
        "required": [],
    },
}
```

### アンチパターン 3: ツール結果の肥大化

```python
# NG: 巨大な結果をそのまま返す
result = database.query("SELECT * FROM products")  # 10,000 行
return json.dumps(result)  # 数MB のJSON → コンテキストを浪費

# OK: 必要な情報のみ返す
result = database.query("SELECT id, name, price FROM products LIMIT 20")
return json.dumps({
    "items": result,
    "total_count": total_count,
    "showing": "1-20",
    "has_more": total_count > 20,
}, ensure_ascii=False)
```

### アンチパターン 4: 権限チェックの欠如

```python
# NG: LLM の指示をそのまま実行
def execute_sql(query: str):
    return db.execute(query)  # DELETE や DROP も実行される

# OK: 権限チェックとクエリの制限
def execute_sql(query: str):
    # SELECT のみ許可
    if not query.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed"}

    # テーブルのホワイトリスト
    allowed_tables = {"products", "orders", "categories"}
    # ... テーブル名のチェック

    # LIMIT の強制
    if "LIMIT" not in query.upper():
        query += " LIMIT 100"

    return db.execute(query)
```

---

## 13. FAQ

### Q1: Function Calling と Structured Output の違いは?

Function Calling は LLM に外部ツールの呼び出し意図を表明させる仕組み。実際の実行はアプリ側。
Structured Output は LLM の出力を JSON Schema に厳密に従わせる仕組み。ツール呼び出しではなく出力フォーマットの制御。
両者は組み合わせ可能で、「ツール呼び出し結果を構造化 JSON で整形して返す」といった使い方ができる。

### Q2: LLM が間違った関数や引数を選んだ場合の対処は?

まずスキーマの description を改善する (最も効果的)。
enum で選択肢を明示する、examples を含める、否定形の指示 (「この関数はXXには使わないでください」) を追加する。
それでも改善しない場合は、アプリ側でバリデーション + LLM への再試行リクエストで対処する。

### Q3: Function Calling のコストへの影響は?

ツール定義はシステムプロンプトの一部としてトークンにカウントされる。
10個のツール定義で約 500-1000 トークン追加されるのが一般的。
ツール数が多い場合は、ユーザーの意図に応じて渡すツールセットを動的にフィルタリングするとコスト削減できる。

### Q4: 並列ツール呼び出しはどう制御する?

OpenAI: `parallel_tool_calls=True/False` で制御。Anthropic: デフォルトで並列対応、ツール間の依存関係は LLM が自動判断。順序保証が必要な場合は、ツールの description に「この関数は search_flights の結果を受けて実行してください」と明記する。

### Q5: ストリーミングと Function Calling は併用できる?

全主要プロバイダで対応している。OpenAI の場合、ストリーミング中に `tool_calls` チャンクが部分的に送信されるため、`function.arguments` を蓄積してから JSON パースする必要がある。Anthropic は `content_block_delta` イベントで `partial_json` を受信し、完了時にパースする。

### Q6: MCP と従来の Function Calling の使い分けは?

MCP はツール提供側を標準化するプロトコル。複数の LLM プロバイダで同じツールを使いたい場合、ツールをマイクロサービスとして独立させたい場合に有効。小規模なプロジェクトや単一プロバイダ利用の場合は従来の Function Calling で十分。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 本質 | LLM が「どの関数を、どの引数で呼ぶか」を構造化出力 |
| 実行責任 | アプリケーション側 (LLM は実行しない) |
| スキーマ | JSON Schema で定義、description が精度に直結 |
| エラー対策 | タイムアウト、ループ上限、入力検証、権限チェック |
| 並列呼び出し | 主要プロバイダ全て対応 |
| セキュリティ | 入力サニタイゼーション、権限管理、確認フロー |
| テスト | スキーマ検証、統合テスト、LLM 判断テスト |
| 発展形 | AI Agent (ReAct、Plan-and-Execute)、MCP |

---

## 次に読むべきガイド

- [03-embeddings.md](./03-embeddings.md) — Function Calling の結果をベクトル化して活用
- [01-rag.md](./01-rag.md) — RAG パイプラインでの Function Calling 統合
- [../03-infrastructure/00-api-integration.md](../03-infrastructure/00-api-integration.md) — API 統合の実践

---

## 参考文献

1. OpenAI, "Function Calling Guide," https://platform.openai.com/docs/guides/function-calling
2. Anthropic, "Tool Use Documentation," https://docs.anthropic.com/claude/docs/tool-use
3. Google, "Gemini Function Calling," https://ai.google.dev/docs/function_calling
4. Schick et al., "Toolformer: Language Models Can Teach Themselves to Use Tools," NeurIPS 2023
5. Anthropic, "Model Context Protocol," https://modelcontextprotocol.io/
6. Qin et al., "ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs," ICLR 2024
