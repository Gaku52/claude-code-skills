# Function Calling — ツール使用・スキーマ定義・エラーハンドリング

> Function Calling は LLM が外部ツール (API、データベース、計算機等) を構造化された形式で呼び出す仕組みであり、LLM を「考えるだけ」の存在から「行動できる」エージェントへ進化させる中核技術である。

## この章で学ぶこと

1. **Function Calling の仕組みと設計原理** — LLM がどのように関数呼び出しを判断し、引数を生成するか
2. **スキーマ定義のベストプラクティス** — JSON Schema による関数定義、パラメータ設計、説明文の書き方
3. **実践的なエラーハンドリングとセキュリティ** — 障害時のフォールバック、入力検証、権限管理

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

### 2.2 Anthropic Tool Use

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
    model="claude-3-5-sonnet-20241022",
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
            model="claude-3-5-sonnet-20241022",
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

### 2.3 Gemini Function Calling

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

### 3.2 プロバイダ別のスキーマ形式

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

---

## 5. エラーハンドリング

### 5.1 堅牢なエラーハンドリング

```python
import json
from typing import Any

class FunctionCallError(Exception):
    pass

def safe_execute_function(name: str, arguments: dict) -> dict[str, Any]:
    """安全な関数実行ラッパー"""

    # 1. 関数の存在確認
    registry = {
        "get_weather": get_weather,
        "search_products": search_products,
    }

    if name not in registry:
        return {"error": f"Unknown function: {name}", "status": "error"}

    # 2. 引数の検証
    try:
        validated_args = validate_arguments(name, arguments)
    except ValueError as e:
        return {"error": f"Invalid arguments: {e}", "status": "error"}

    # 3. 実行 (タイムアウト付き)
    try:
        import asyncio
        result = asyncio.wait_for(
            registry[name](**validated_args),
            timeout=10.0,  # 10秒タイムアウト
        )
        return {"result": result, "status": "success"}
    except asyncio.TimeoutError:
        return {"error": "Function timed out", "status": "timeout"}
    except Exception as e:
        return {"error": str(e), "status": "error"}

def validate_arguments(function_name: str, args: dict) -> dict:
    """引数の検証とサニタイズ"""
    # SQLインジェクション等の防止
    for key, value in args.items():
        if isinstance(value, str):
            if any(dangerous in value.lower() for dangerous in
                   ["drop table", "delete from", "; --", "' or '1'='1"]):
                raise ValueError(f"Potentially dangerous input in {key}")
    return args
```

---

## 6. 比較表

### 6.1 プロバイダ別 Function Calling 機能比較

| 機能 | OpenAI | Anthropic | Google Gemini |
|------|--------|-----------|--------------|
| 並列呼び出し | 対応 | 対応 | 対応 |
| ストリーミング | 対応 | 対応 | 対応 |
| 強制呼び出し | tool_choice | tool_choice | function_calling_config |
| 最大ツール数 | 128 | 制限なし(推奨20) | 制限なし |
| ネストJSON | 対応 | 対応 | 対応 |
| Structured Output | 対応 | JSON Mode | 対応 |
| 自動実行 | なし | なし | あり (opt-in) |

### 6.2 ユースケース別のツール設計

| ユースケース | ツール数 | 設計パターン | 注意点 |
|-------------|---------|------------|--------|
| 天気Bot | 1-2 | 単一ツール | シンプルに保つ |
| ECアシスタント | 5-10 | ツールチェイン | 状態管理が重要 |
| 社内業務Bot | 10-20 | ルーター型 | 権限管理必須 |
| 自律エージェント | 20+ | ReAct パターン | ループ上限設定 |

---

## 7. アンチパターン

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

---

## 8. FAQ

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

---

## まとめ

| 項目 | 内容 |
|------|------|
| 本質 | LLM が「どの関数を、どの引数で呼ぶか」を構造化出力 |
| 実行責任 | アプリケーション側 (LLM は実行しない) |
| スキーマ | JSON Schema で定義、description が精度に直結 |
| エラー対策 | タイムアウト、ループ上限、入力検証、権限チェック |
| 並列呼び出し | 主要プロバイダ全て対応 |
| 発展形 | AI Agent (ReAct、Plan-and-Execute) |

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
