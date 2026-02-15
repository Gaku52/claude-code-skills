# Claude Agent SDK

> Anthropic公式のエージェント構築ツール――最小限のコードでツール使用エージェントを構築し、MCPサーバーとネイティブ統合する方法を解説する。

## この章で学ぶこと

1. Claude Messages API を使ったエージェントループの構築パターン
2. ツール定義・並列ツール呼び出し・ストリーミングの実装
3. MCPとの統合およびプロダクション向け設計パターン
4. マルチエージェントオーケストレーションの実装
5. エラーハンドリング・リトライ・ガードレールの設計
6. コンテキスト管理と会話メモリの最適化
7. 本番環境でのパフォーマンスチューニングとモニタリング

---

## 1. Claude Agent SDKの位置づけ

```
エージェント構築の選択肢

 抽象度 高  +---------+
            | CrewAI  |  高レベルフレームワーク
            +---------+
            | LangChain|  汎用フレームワーク
            +---------+
            | Claude   |  公式SDK（直接API）
            | Agent SDK|
            +---------+  ← この章の範囲
 抽象度 低  | Raw HTTP |  生のAPI呼び出し
            +---------+

Claude Agent SDK の利点:
- 最小の依存関係（anthropic パッケージのみ）
- APIの全機能に直接アクセス
- MCPとのネイティブ統合
- 抽象化レイヤーによる「魔法」がない
```

### 1.1 SDKのインストールと初期設定

```bash
# 基本インストール
pip install anthropic

# ストリーミング・非同期サポート付き
pip install "anthropic[bedrock,vertex]"

# 開発環境での推奨セットアップ
pip install anthropic python-dotenv pydantic
```

```python
# 環境変数の設定
import os
from dotenv import load_dotenv

load_dotenv()

# 方法1: 環境変数から自動読み込み（推奨）
# ANTHROPIC_API_KEY を環境変数に設定しておく
import anthropic
client = anthropic.Anthropic()  # 自動的に ANTHROPIC_API_KEY を参照

# 方法2: 明示的にAPIキーを指定
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# 方法3: AWS Bedrock経由
bedrock_client = anthropic.AnthropicBedrock(
    aws_region="us-east-1",
    aws_access_key=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

# 方法4: Google Vertex AI経由
vertex_client = anthropic.AnthropicVertex(
    project_id="my-project",
    region="us-east5",
)
```

### 1.2 他フレームワークとの詳細比較

```
Claude Agent SDK vs 他フレームワーク 機能比較

+------------------+------------+-----------+----------+----------+
|                  | Claude SDK | LangChain | CrewAI   | AutoGen  |
+------------------+------------+-----------+----------+----------+
| 依存パッケージ数  |    1       |   50+     |   30+    |   20+    |
| 学習コスト        |    低      |   高      |   中     |   中     |
| 型安全性          |    高      |   低      |   中     |   中     |
| デバッグ容易性    |    高      |   低      |   中     |   中     |
| カスタマイズ性    |    最高    |   高      |   中     |   高     |
| MCP統合          |    ネイティブ|  プラグイン|  なし    |  なし    |
| マルチモデル      |    Claude  |  任意     |  任意    |  任意    |
| プロダクション対応 |    高      |   中      |   低     |   中     |
| コミュニティ規模  |    中      |   最大    |   中     |   中     |
+------------------+------------+-----------+----------+----------+
```

### 1.3 APIバージョニングとモデルIDの管理

```python
# モデルIDの管理パターン
from enum import Enum

class ClaudeModel(str, Enum):
    """利用可能なClaudeモデル一覧"""
    HAIKU = "claude-haiku-4-20250514"
    SONNET = "claude-sonnet-4-20250514"
    OPUS = "claude-opus-4-20250514"

    @property
    def cost_per_1k_input(self) -> float:
        """入力1Kトークンあたりのコスト（USD）"""
        costs = {
            self.HAIKU: 0.00025,
            self.SONNET: 0.003,
            self.OPUS: 0.015,
        }
        return costs[self]

    @property
    def cost_per_1k_output(self) -> float:
        """出力1Kトークンあたりのコスト（USD）"""
        costs = {
            self.HAIKU: 0.00125,
            self.SONNET: 0.015,
            self.OPUS: 0.075,
        }
        return costs[self]

    @property
    def max_context_window(self) -> int:
        """最大コンテキストウィンドウサイズ"""
        return 200_000  # 全モデル共通

# 使用例
model = ClaudeModel.SONNET
print(f"モデル: {model.value}")
print(f"入力コスト: ${model.cost_per_1k_input}/1Kトークン")
print(f"コンテキスト: {model.max_context_window:,}トークン")
```

---

## 2. 基本的なエージェントループ

### 2.1 最小構成

```python
# Claude Agent SDK: 最小構成のエージェント
import anthropic

client = anthropic.Anthropic()

def simple_agent(user_message: str) -> str:
    """最もシンプルなエージェント"""
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=messages
    )

    return response.content[0].text
```

### 2.2 ツール使用エージェント

```python
# ツール使用付きの完全なエージェントループ
import anthropic
import json
from typing import Any

client = anthropic.Anthropic()

# ツール定義
TOOLS = [
    {
        "name": "read_file",
        "description": "指定されたファイルの内容を読み取って返す",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "読み取るファイルのパス"
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "指定されたファイルに内容を書き込む",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "ファイルパス"},
                "content": {"type": "string", "description": "書き込む内容"}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "run_command",
        "description": "シェルコマンドを実行して結果を返す",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "実行するコマンド"}
            },
            "required": ["command"]
        }
    }
]

# ツール実行ハンドラ
def execute_tool(name: str, input_data: dict) -> str:
    try:
        if name == "read_file":
            with open(input_data["path"]) as f:
                return f.read()
        elif name == "write_file":
            with open(input_data["path"], "w") as f:
                f.write(input_data["content"])
            return f"ファイル書き込み完了: {input_data['path']}"
        elif name == "run_command":
            import subprocess
            result = subprocess.run(
                input_data["command"],
                shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout + result.stderr
        else:
            return f"不明なツール: {name}"
    except Exception as e:
        return f"エラー: {type(e).__name__}: {e}"

# エージェントループ
def agent_loop(
    user_message: str,
    system_prompt: str = "",
    max_steps: int = 20
) -> str:
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_steps):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        # 最終回答の場合
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # ツール呼び出しの場合
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [{step}] Tool: {block.name}({json.dumps(block.input, ensure_ascii=False)[:80]})")
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result[:10000]  # 結果が大きすぎる場合を制限
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "最大ステップ数に達しました。"

# 実行
result = agent_loop(
    "setup.pyを読んで、テストを実行して結果を報告して",
    system_prompt="あなたはPython開発アシスタントです。"
)
print(result)
```

### 2.3 レスポンスの詳細解析

```python
# レスポンスオブジェクトの構造を理解する
from anthropic.types import Message, ContentBlock, TextBlock, ToolUseBlock

def analyze_response(response: Message) -> dict:
    """レスポンスの詳細を解析してログに記録"""
    analysis = {
        "id": response.id,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "content_blocks": [],
    }

    for block in response.content:
        if isinstance(block, TextBlock):
            analysis["content_blocks"].append({
                "type": "text",
                "length": len(block.text),
                "preview": block.text[:100],
            })
        elif isinstance(block, ToolUseBlock):
            analysis["content_blocks"].append({
                "type": "tool_use",
                "name": block.name,
                "id": block.id,
                "input_keys": list(block.input.keys()),
            })

    # コスト計算
    model = response.model
    input_cost = response.usage.input_tokens * 0.003 / 1000  # Sonnet想定
    output_cost = response.usage.output_tokens * 0.015 / 1000
    analysis["estimated_cost_usd"] = round(input_cost + output_cost, 6)

    return analysis

# 使用例
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}]
)
info = analyze_response(response)
print(json.dumps(info, indent=2, ensure_ascii=False))
```

### 2.4 stop_reason の完全ガイド

```python
# stop_reasonの種類と対処法
STOP_REASON_HANDLERS = {
    "end_turn": "モデルが自然に回答を終了。最終テキストを取得。",
    "tool_use": "ツール呼び出しが必要。ツールを実行して結果を返す。",
    "max_tokens": "出力トークン上限に達した。max_tokensを増やすか分割処理。",
    "stop_sequence": "指定したstop_sequenceにマッチ。カスタム終了条件。",
}

def handle_stop_reason(response) -> str:
    """stop_reasonに基づいて適切な処理を実行"""
    reason = response.stop_reason

    if reason == "end_turn":
        return extract_text(response)

    elif reason == "tool_use":
        # ツール呼び出し処理（エージェントループで継続）
        return "CONTINUE_LOOP"

    elif reason == "max_tokens":
        # 出力が途中で切れた場合の処理
        partial_text = extract_text(response)
        # 続きを要求
        return partial_text + "\n[出力が途中で切れました。続きを生成中...]"

    elif reason == "stop_sequence":
        # カスタム終了条件
        return extract_text(response)

    else:
        raise ValueError(f"未知のstop_reason: {reason}")
```

---

## 3. 高度な機能

### 3.1 ストリーミング

```python
# ストリーミングでリアルタイム出力
def streaming_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        tool_use_blocks = []
        current_text = ""

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        print(event.delta.text, end="", flush=True)
                        current_text += event.delta.text

            response = stream.get_final_message()

        if response.stop_reason == "end_turn":
            return current_text

        # ツール呼び出し処理
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

### 3.2 高度なストリーミングイベント処理

```python
# ストリーミングイベントの全種類を処理するハンドラ
from dataclasses import dataclass, field
from typing import Optional
import json

@dataclass
class StreamState:
    """ストリーミング中の状態管理"""
    current_block_type: Optional[str] = None
    current_tool_name: Optional[str] = None
    current_tool_id: Optional[str] = None
    accumulated_text: str = ""
    accumulated_json: str = ""
    tool_calls: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0

def advanced_streaming_agent(user_message: str):
    """全イベントを処理する高度なストリーミングエージェント"""
    messages = [{"role": "user", "content": user_message}]

    while True:
        state = StreamState()

        with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages
        ) as stream:
            for event in stream:
                # メッセージ開始
                if event.type == "message_start":
                    state.input_tokens = event.message.usage.input_tokens
                    print(f"\n[入力トークン: {state.input_tokens}]")

                # コンテンツブロック開始
                elif event.type == "content_block_start":
                    if event.content_block.type == "text":
                        state.current_block_type = "text"
                    elif event.content_block.type == "tool_use":
                        state.current_block_type = "tool_use"
                        state.current_tool_name = event.content_block.name
                        state.current_tool_id = event.content_block.id
                        state.accumulated_json = ""
                        print(f"\n[ツール呼び出し: {event.content_block.name}]")

                # デルタ（差分データ）
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        print(event.delta.text, end="", flush=True)
                        state.accumulated_text += event.delta.text
                    elif hasattr(event.delta, "partial_json"):
                        state.accumulated_json += event.delta.partial_json

                # コンテンツブロック終了
                elif event.type == "content_block_stop":
                    if state.current_block_type == "tool_use":
                        try:
                            tool_input = json.loads(state.accumulated_json)
                        except json.JSONDecodeError:
                            tool_input = {}
                        state.tool_calls.append({
                            "name": state.current_tool_name,
                            "id": state.current_tool_id,
                            "input": tool_input,
                        })
                    state.current_block_type = None

                # メッセージデルタ（使用量情報）
                elif event.type == "message_delta":
                    state.output_tokens = event.usage.output_tokens

            response = stream.get_final_message()

        print(f"\n[出力トークン: {state.output_tokens}]")

        if response.stop_reason == "end_turn":
            return state.accumulated_text

        # ツール呼び出し処理
        tool_results = []
        for call in state.tool_calls:
            result = execute_tool(call["name"], call["input"])
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": result[:10000],
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

### 3.3 並列ツール呼び出し

```python
# Claudeは1回のレスポンスで複数のツールを同時に呼び出せる
# 例: "東京と大阪の天気を教えて" → 2つのget_weatherを同時呼び出し

def handle_parallel_tool_calls(response) -> list:
    """並列ツール呼び出しを処理"""
    tool_results = []

    for block in response.content:
        if block.type == "tool_use":
            # 各ツール呼び出しを処理
            result = execute_tool(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result
            })

    return tool_results

# 非同期版（本当の並列実行）
async def handle_parallel_tool_calls_async(response) -> list:
    import asyncio

    async def execute_single(block):
        result = await async_execute_tool(block.name, block.input)
        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": result
        }

    tool_blocks = [b for b in response.content if b.type == "tool_use"]
    results = await asyncio.gather(*[execute_single(b) for b in tool_blocks])
    return list(results)
```

### 3.4 並列ツール呼び出しの実務パターン

```python
# 実務的な並列ツール呼び出しの例: 複数API同時呼び出し
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

class ParallelToolExecutor:
    """ツール呼び出しを並列実行するエグゼキューター"""

    def __init__(self, max_workers: int = 5, timeout: float = 30.0):
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def execute_parallel(self, tool_calls: list) -> list:
        """複数のツール呼び出しを並列に実行"""
        tasks = []
        for call in tool_calls:
            task = asyncio.create_task(
                self._execute_with_timeout(call)
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        tool_results = []
        for call, result in zip(tool_calls, results):
            if isinstance(result, Exception):
                content = f"エラー: {type(result).__name__}: {result}"
                is_error = True
            else:
                content = str(result)
                is_error = False

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": call["id"],
                "content": content,
                "is_error": is_error,
            })

        return tool_results

    async def _execute_with_timeout(self, call: dict):
        """タイムアウト付きでツールを実行"""
        return await asyncio.wait_for(
            self._execute_tool_async(call["name"], call["input"]),
            timeout=self.timeout,
        )

    async def _execute_tool_async(self, name: str, input_data: dict):
        """非同期ツール実行"""
        if name == "fetch_url":
            async with aiohttp.ClientSession() as session:
                async with session.get(input_data["url"]) as resp:
                    return await resp.text()
        elif name == "query_database":
            # DB接続は同期処理をスレッドプールで実行
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                lambda: self._sync_db_query(input_data["query"])
            )
        else:
            raise ValueError(f"不明なツール: {name}")

    def _sync_db_query(self, query: str) -> str:
        """同期的なDB問い合わせ"""
        import sqlite3
        conn = sqlite3.connect("app.db")
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return json.dumps(rows, ensure_ascii=False)

# 使用例
executor = ParallelToolExecutor(max_workers=10, timeout=15.0)

async def agent_with_parallel_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            return extract_text(response)

        # 並列実行
        tool_calls = [
            {"name": b.name, "id": b.id, "input": b.input}
            for b in response.content if b.type == "tool_use"
        ]
        tool_results = await executor.execute_parallel(tool_calls)

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})
```

### 3.5 システムプロンプト設計

```python
# エージェント向けシステムプロンプト
CODING_AGENT_PROMPT = """あなたはシニアソフトウェアエンジニアとして振る舞うコーディングエージェントです。

## 行動規範
1. コードを変更する前に、必ず既存のコードを読んで理解する
2. テストを書いてから実装する（TDD）
3. 変更は最小限に留める
4. エラーが発生したら原因を特定してから修正する

## ツール使用ガイドライン
- read_file: コードの構造を理解するために最初に使用
- write_file: テスト→実装の順で使用
- run_command: テスト実行、lint、型チェックに使用

## 出力形式
- 作業内容を簡潔に説明してから実行する
- 完了後に変更の要約を提供する
"""
```

### 3.6 高度なシステムプロンプト設計パターン

```python
# 役割ベースの動的システムプロンプト生成
from string import Template
from datetime import datetime

class SystemPromptBuilder:
    """構造化されたシステムプロンプトを動的に構築"""

    def __init__(self):
        self.sections: dict[str, str] = {}

    def set_role(self, role: str) -> "SystemPromptBuilder":
        self.sections["role"] = f"## 役割\n{role}"
        return self

    def set_rules(self, rules: list[str]) -> "SystemPromptBuilder":
        rules_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))
        self.sections["rules"] = f"## 行動規範\n{rules_text}"
        return self

    def set_tool_guidelines(self, guidelines: dict[str, str]) -> "SystemPromptBuilder":
        lines = [f"- {tool}: {desc}" for tool, desc in guidelines.items()]
        self.sections["tools"] = f"## ツール使用ガイドライン\n" + "\n".join(lines)
        return self

    def set_output_format(self, format_desc: str) -> "SystemPromptBuilder":
        self.sections["output"] = f"## 出力形式\n{format_desc}"
        return self

    def set_constraints(self, constraints: list[str]) -> "SystemPromptBuilder":
        lines = [f"- {c}" for c in constraints]
        self.sections["constraints"] = f"## 制約事項\n" + "\n".join(lines)
        return self

    def add_context(self, key: str, value: str) -> "SystemPromptBuilder":
        self.sections[f"context_{key}"] = f"## {key}\n{value}"
        return self

    def build(self) -> str:
        parts = []
        # 順序を保証
        order = ["role", "rules", "tools", "output", "constraints"]
        for key in order:
            if key in self.sections:
                parts.append(self.sections[key])

        # その他のセクション
        for key, value in self.sections.items():
            if key not in order:
                parts.append(value)

        # メタ情報
        parts.append(f"\n## メタ情報\n- 現在日時: {datetime.now().isoformat()}")

        return "\n\n".join(parts)

# 使用例: コードレビューエージェント
review_prompt = (
    SystemPromptBuilder()
    .set_role("あなたはシニアソフトウェアエンジニアとしてコードレビューを行います。")
    .set_rules([
        "セキュリティ上の問題を最優先で指摘する",
        "パフォーマンスへの影響を評価する",
        "テストの網羅性を確認する",
        "コード規約への準拠を検証する",
        "建設的なフィードバックを心がける",
    ])
    .set_tool_guidelines({
        "read_file": "レビュー対象のファイルを読み込む",
        "run_command": "テスト実行・静的解析に使用",
        "search_code": "関連コードの検索に使用",
    })
    .set_output_format(
        "レビュー結果はMarkdown形式で出力。"
        "重要度（Critical/Warning/Info）を付与。"
    )
    .set_constraints([
        "承認・却下の判断は最終的に人間が行う",
        "自動修正は提案のみ、実行しない",
        "個人攻撃的なコメントは絶対にしない",
    ])
    .build()
)
```

### 3.7 Extended Thinking（拡張思考）の活用

```python
# Extended Thinkingを使ったエージェント
def agent_with_thinking(user_message: str, budget_tokens: int = 8000):
    """拡張思考を活用するエージェント"""
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": budget_tokens,  # 思考に使うトークン数
        },
        messages=messages,
    )

    # 思考プロセスとレスポンスを分離
    thinking_text = ""
    response_text = ""

    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            response_text = block.text

    return {
        "thinking": thinking_text,  # デバッグ用に思考プロセスを保持
        "response": response_text,
    }

# Extended Thinkingとツール使用の組み合わせ
def planning_agent(user_message: str):
    """計画フェーズでExtended Thinking、実行フェーズで通常モードを使用"""

    # Phase 1: 計画（Extended Thinking有効）
    plan_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        thinking={"type": "enabled", "budget_tokens": 5000},
        messages=[{
            "role": "user",
            "content": f"以下のタスクの実行計画を立ててください:\n{user_message}"
        }],
    )

    plan_text = ""
    for block in plan_response.content:
        if block.type == "text":
            plan_text = block.text

    # Phase 2: 実行（通常モード + ツール使用）
    result = agent_loop(
        f"以下の計画に従って実行してください:\n\n{plan_text}",
        system_prompt="計画に忠実に従い、各ステップを実行してください。"
    )

    return {
        "plan": plan_text,
        "result": result,
    }
```

---

## 4. エージェントの構造化

### 4.1 クラスベースのエージェント設計

```python
# クラスベースのエージェント設計
from dataclasses import dataclass, field
from typing import Callable
import time

@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    max_steps: int = 20
    temperature: float = 0.0
    system_prompt: str = ""
    timeout: float = 300.0  # 秒

class ClaudeAgent:
    def __init__(self, config: AgentConfig, tools: list, handlers: dict):
        self.config = config
        self.tools = tools
        self.handlers = handlers  # {tool_name: handler_function}
        self.client = anthropic.Anthropic()
        self.conversation_history = []

    def run(self, user_message: str) -> str:
        self.conversation_history.append({
            "role": "user", "content": user_message
        })
        start_time = time.time()

        for step in range(self.config.max_steps):
            if time.time() - start_time > self.config.timeout:
                return "タイムアウトしました"

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.config.system_prompt,
                tools=self.tools,
                messages=self.conversation_history
            )

            if response.stop_reason == "end_turn":
                text = self._extract_text(response)
                self.conversation_history.append({
                    "role": "assistant", "content": response.content
                })
                return text

            # ツール処理
            tool_results = self._process_tools(response)
            self.conversation_history.append({
                "role": "assistant", "content": response.content
            })
            self.conversation_history.append({
                "role": "user", "content": tool_results
            })

        return "最大ステップ数に達しました"

    def _process_tools(self, response) -> list:
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = self.handlers.get(block.name)
                if handler:
                    try:
                        result = handler(**block.input)
                    except Exception as e:
                        result = f"エラー: {e}"
                else:
                    result = f"ハンドラ未登録: {block.name}"
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })
        return results

    def _extract_text(self, response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def reset(self):
        self.conversation_history = []
```

### 4.2 プラグイン型ツールシステム

```python
# デコレータベースのツール登録システム
from typing import Callable, Any, get_type_hints
import inspect
import json

class ToolRegistry:
    """ツールの登録・管理・実行を統合的に行うレジストリ"""

    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def tool(self, description: str = ""):
        """デコレータでツールを登録"""
        def decorator(func: Callable) -> Callable:
            name = func.__name__
            schema = self._generate_schema(func, description)
            self._tools[name] = schema
            self._handlers[name] = func
            return func
        return decorator

    def _generate_schema(self, func: Callable, description: str) -> dict:
        """関数シグネチャからJSONスキーマを自動生成"""
        hints = get_type_hints(func)
        sig = inspect.signature(func)
        doc = description or func.__doc__ or f"{func.__name__}を実行"

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = hints.get(param_name, str)
            json_type = self._python_type_to_json(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"{param_name}パラメータ",
            }

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "name": func.__name__,
            "description": doc,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }

    def _python_type_to_json(self, python_type) -> str:
        """Pythonの型をJSONスキーマの型に変換"""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(python_type, "string")

    def get_tool_definitions(self) -> list[dict]:
        """APIに渡すツール定義のリストを取得"""
        return list(self._tools.values())

    def execute(self, name: str, input_data: dict) -> str:
        """ツールを名前で実行"""
        handler = self._handlers.get(name)
        if not handler:
            return f"エラー: 未登録のツール '{name}'"
        try:
            result = handler(**input_data)
            return str(result)
        except Exception as e:
            return f"エラー: {type(e).__name__}: {e}"

# 使用例
registry = ToolRegistry()

@registry.tool("ファイルの内容を読み取る")
def read_file(path: str, encoding: str = "utf-8") -> str:
    with open(path, encoding=encoding) as f:
        return f.read()

@registry.tool("指定ディレクトリ内のファイル一覧を取得する")
def list_files(directory: str, pattern: str = "*") -> str:
    import glob
    files = glob.glob(f"{directory}/{pattern}")
    return json.dumps(files, ensure_ascii=False)

@registry.tool("HTTPリクエストを実行する")
def http_request(url: str, method: str = "GET") -> str:
    import urllib.request
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode()

# エージェントで使用
agent = ClaudeAgent(
    config=AgentConfig(system_prompt="開発アシスタント"),
    tools=registry.get_tool_definitions(),
    handlers=registry._handlers,
)
```

### 4.3 ミドルウェアパターン

```python
# ツール呼び出しの前後にフックを挟むミドルウェア
from typing import Callable, Optional
import time
import logging

logger = logging.getLogger(__name__)

class ToolMiddleware:
    """ツール実行のミドルウェアチェーン"""

    def __init__(self):
        self._before_hooks: list[Callable] = []
        self._after_hooks: list[Callable] = []
        self._error_hooks: list[Callable] = []

    def before(self, hook: Callable) -> "ToolMiddleware":
        """ツール実行前フック"""
        self._before_hooks.append(hook)
        return self

    def after(self, hook: Callable) -> "ToolMiddleware":
        """ツール実行後フック"""
        self._after_hooks.append(hook)
        return self

    def on_error(self, hook: Callable) -> "ToolMiddleware":
        """エラー時フック"""
        self._error_hooks.append(hook)
        return self

    def wrap(self, handler: Callable) -> Callable:
        """ハンドラをミドルウェアでラップ"""
        before_hooks = self._before_hooks
        after_hooks = self._after_hooks
        error_hooks = self._error_hooks

        def wrapped(**kwargs):
            context = {"name": handler.__name__, "input": kwargs, "start": time.time()}

            # Before hooks
            for hook in before_hooks:
                hook(context)

            try:
                result = handler(**kwargs)
                context["result"] = result
                context["duration"] = time.time() - context["start"]

                # After hooks
                for hook in after_hooks:
                    hook(context)

                return result

            except Exception as e:
                context["error"] = e
                context["duration"] = time.time() - context["start"]

                for hook in error_hooks:
                    hook(context)

                raise

        return wrapped

# 実用的なミドルウェア例
def logging_hook(context: dict):
    """ツール呼び出しをログに記録"""
    if "result" in context:
        logger.info(
            f"Tool {context['name']} completed in {context['duration']:.2f}s"
        )
    elif "error" in context:
        logger.error(
            f"Tool {context['name']} failed: {context['error']}"
        )
    else:
        logger.info(f"Tool {context['name']} starting with {list(context['input'].keys())}")

def rate_limit_hook(context: dict):
    """レート制限チェック"""
    # 短時間に多くのツールが呼ばれないようにする
    time.sleep(0.1)

def sanitize_hook(context: dict):
    """入力のサニタイズ"""
    for key, value in context["input"].items():
        if isinstance(value, str) and any(
            dangerous in value for dangerous in ["rm -rf", "DROP TABLE", "eval("]
        ):
            raise ValueError(f"危険な入力が検出されました: {key}")

# ミドルウェアの適用
middleware = (
    ToolMiddleware()
    .before(sanitize_hook)
    .before(logging_hook)
    .after(logging_hook)
    .on_error(logging_hook)
)

# ハンドラをラップ
safe_read_file = middleware.wrap(read_file)
```

---

## 5. MCP（Model Context Protocol）統合

### 5.1 MCPクライアントの基本

```python
# MCP統合の基本パターン
import subprocess
import json
from typing import Optional

class MCPClient:
    """MCPサーバーとの通信を管理するクライアント"""

    def __init__(self, server_command: list[str]):
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._request_id = 0

    def _send_request(self, method: str, params: dict = None) -> dict:
        """JSON-RPCリクエストを送信"""
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def initialize(self) -> dict:
        """MCPサーバーを初期化"""
        return self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "claude-agent", "version": "1.0"},
        })

    def list_tools(self) -> list[dict]:
        """利用可能なツール一覧を取得"""
        response = self._send_request("tools/list")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """ツールを呼び出す"""
        response = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        result = response.get("result", {})
        content = result.get("content", [])
        return "\n".join(
            item.get("text", "") for item in content if item.get("type") == "text"
        )

    def close(self):
        """MCPサーバーを終了"""
        self.process.terminate()
        self.process.wait()
```

### 5.2 MCPツールとClaudeツールの統合

```python
# MCPサーバーのツールをClaudeのtool_useと統合
class MCPIntegratedAgent:
    """MCPサーバーとネイティブツールを統合するエージェント"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = anthropic.Anthropic()
        self.mcp_clients: dict[str, MCPClient] = {}
        self.native_tools: dict[str, dict] = {}
        self.native_handlers: dict[str, Callable] = {}

    def add_mcp_server(self, name: str, command: list[str]):
        """MCPサーバーを追加"""
        mcp = MCPClient(command)
        mcp.initialize()
        self.mcp_clients[name] = mcp

    def add_native_tool(self, tool_def: dict, handler: Callable):
        """ネイティブツールを追加"""
        self.native_tools[tool_def["name"]] = tool_def
        self.native_handlers[tool_def["name"]] = handler

    def get_all_tools(self) -> list[dict]:
        """全ツール定義を取得（MCP + ネイティブ）"""
        tools = list(self.native_tools.values())

        for server_name, mcp in self.mcp_clients.items():
            mcp_tools = mcp.list_tools()
            for tool in mcp_tools:
                # MCPツール定義をClaude API形式に変換
                tools.append({
                    "name": f"{server_name}__{tool['name']}",
                    "description": tool.get("description", ""),
                    "input_schema": tool.get("inputSchema", {
                        "type": "object", "properties": {}
                    }),
                })

        return tools

    def execute_tool(self, name: str, input_data: dict) -> str:
        """ツール名に基づいて適切なハンドラに振り分け"""
        # ネイティブツール
        if name in self.native_handlers:
            try:
                return str(self.native_handlers[name](**input_data))
            except Exception as e:
                return f"エラー: {e}"

        # MCPツール（server_name__tool_name形式）
        if "__" in name:
            server_name, tool_name = name.split("__", 1)
            mcp = self.mcp_clients.get(server_name)
            if mcp:
                return mcp.call_tool(tool_name, input_data)

        return f"不明なツール: {name}"

    def run(self, user_message: str) -> str:
        """エージェントループを実行"""
        messages = [{"role": "user", "content": user_message}]
        tools = self.get_all_tools()

        for step in range(self.config.max_steps):
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:10000],
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "最大ステップ数に達しました"

    def close(self):
        """全MCPサーバーを終了"""
        for mcp in self.mcp_clients.values():
            mcp.close()

# 使用例
agent = MCPIntegratedAgent(AgentConfig(
    system_prompt="ファイルシステムとデータベースを操作できるアシスタントです。"
))

# MCPサーバーを追加
agent.add_mcp_server("filesystem", ["npx", "@modelcontextprotocol/server-filesystem", "/tmp"])
agent.add_mcp_server("sqlite", ["npx", "@modelcontextprotocol/server-sqlite", "app.db"])

# ネイティブツールを追加
agent.add_native_tool(
    {"name": "calculate", "description": "計算を実行", "input_schema": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"]
    }},
    handler=lambda expression: eval(expression)  # 実運用ではsafe_evalを使う
)

result = agent.run("データベースのユーザー数を取得して、/tmp/report.txtに書き込んで")
print(result)
agent.close()
```

---

## 6. マルチエージェントオーケストレーション

### 6.1 オーケストレーター/ワーカーパターン

```python
# マルチエージェント: オーケストレーター + ワーカー
from enum import Enum
from typing import Optional
import json

class AgentRole(str, Enum):
    ORCHESTRATOR = "orchestrator"
    CODER = "coder"
    REVIEWER = "reviewer"
    TESTER = "tester"

class MultiAgentSystem:
    """複数のClaudeエージェントを協調動作させるシステム"""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.agents: dict[AgentRole, AgentConfig] = {}
        self.shared_context: dict = {}

    def register_agent(self, role: AgentRole, config: AgentConfig):
        self.agents[role] = config

    def run_agent(self, role: AgentRole, message: str, tools: list = None) -> str:
        """特定の役割のエージェントを実行"""
        config = self.agents[role]
        messages = [{"role": "user", "content": message}]

        response = self.client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            system=config.system_prompt,
            tools=tools or [],
            messages=messages,
        )

        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

    def orchestrate(self, task: str) -> dict:
        """タスクを分解して各エージェントに割り当て"""

        # Step 1: オーケストレーターがタスクを分解
        plan = self.run_agent(
            AgentRole.ORCHESTRATOR,
            f"""以下のタスクを分解してください。
各サブタスクにはcoder/reviewer/testerの役割を割り当ててください。
JSON形式で出力: [{{"role": "coder", "task": "..."}}]

タスク: {task}"""
        )

        try:
            subtasks = json.loads(plan)
        except json.JSONDecodeError:
            return {"error": "タスク分解に失敗", "raw": plan}

        # Step 2: 各サブタスクを実行
        results = []
        for subtask in subtasks:
            role = AgentRole(subtask["role"])
            result = self.run_agent(role, subtask["task"])
            results.append({
                "role": subtask["role"],
                "task": subtask["task"],
                "result": result,
            })

        # Step 3: 結果を統合
        summary = self.run_agent(
            AgentRole.ORCHESTRATOR,
            f"以下の作業結果を統合してレポートを作成:\n{json.dumps(results, ensure_ascii=False)}"
        )

        return {"subtasks": results, "summary": summary}

# システムの構築
system = MultiAgentSystem()

system.register_agent(AgentRole.ORCHESTRATOR, AgentConfig(
    model="claude-sonnet-4-20250514",
    system_prompt="タスクを分解し、適切な担当者に割り当てるプロジェクトマネージャーです。"
))

system.register_agent(AgentRole.CODER, AgentConfig(
    model="claude-sonnet-4-20250514",
    system_prompt="高品質なコードを書くシニアエンジニアです。テスト可能な設計を心がけます。"
))

system.register_agent(AgentRole.REVIEWER, AgentConfig(
    model="claude-sonnet-4-20250514",
    system_prompt="コードの品質・セキュリティ・パフォーマンスを評価するレビュアーです。"
))

system.register_agent(AgentRole.TESTER, AgentConfig(
    model="claude-sonnet-4-20250514",
    system_prompt="テストケースの設計と実装を行うQAエンジニアです。"
))

result = system.orchestrate("ユーザー認証APIをFastAPIで実装する")
```

### 6.2 パイプラインパターン

```python
# パイプライン型マルチエージェント: 出力が次のエージェントの入力になる
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class PipelineStage:
    """パイプラインの各ステージ"""
    name: str
    model: str
    system_prompt: str
    transform_output: Optional[Callable[[str], str]] = None

class AgentPipeline:
    """エージェントを直列に接続するパイプライン"""

    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        self.client = anthropic.Anthropic()
        self.stage_results: list[dict] = []

    def run(self, initial_input: str) -> dict:
        """パイプラインを実行"""
        current_input = initial_input
        self.stage_results = []

        for i, stage in enumerate(self.stages):
            print(f"[Stage {i+1}/{len(self.stages)}] {stage.name}")

            response = self.client.messages.create(
                model=stage.model,
                max_tokens=4096,
                system=stage.system_prompt,
                messages=[{"role": "user", "content": current_input}],
            )

            output = ""
            for block in response.content:
                if hasattr(block, "text"):
                    output = block.text

            if stage.transform_output:
                output = stage.transform_output(output)

            self.stage_results.append({
                "stage": stage.name,
                "input_preview": current_input[:200],
                "output_preview": output[:200],
                "tokens": {
                    "input": response.usage.input_tokens,
                    "output": response.usage.output_tokens,
                },
            })

            current_input = output

        return {
            "final_output": current_input,
            "stages": self.stage_results,
        }

# 使用例: 技術文書生成パイプライン
pipeline = AgentPipeline([
    PipelineStage(
        name="要件分析",
        model="claude-sonnet-4-20250514",
        system_prompt="技術文書の要件を分析し、構成案をJSON形式で出力してください。",
    ),
    PipelineStage(
        name="ドラフト作成",
        model="claude-sonnet-4-20250514",
        system_prompt="与えられた構成案に基づいて技術文書のドラフトを作成してください。",
    ),
    PipelineStage(
        name="レビュー・改善",
        model="claude-sonnet-4-20250514",
        system_prompt="技術文書をレビューし、改善版を出力してください。正確性・明瞭性・完全性を評価。",
    ),
])

result = pipeline.run("Kubernetes上でのマイクロサービスデプロイガイドを作成して")
print(result["final_output"])
```

---

## 7. エラーハンドリングとガードレール

### 7.1 包括的エラーハンドリング

```python
# 堅牢なエラーハンドリング
from anthropic import (
    APIError,
    APIConnectionError,
    RateLimitError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
)
import time
import logging

logger = logging.getLogger(__name__)

class RobustAgent:
    """プロダクション品質のエラーハンドリングを備えたエージェント"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = anthropic.Anthropic()
        self.retry_config = {
            "max_retries": 3,
            "base_delay": 1.0,
            "max_delay": 60.0,
            "backoff_factor": 2.0,
        }

    def _call_api_with_retry(self, **kwargs) -> "Message":
        """リトライ付きAPI呼び出し"""
        last_error = None

        for attempt in range(self.retry_config["max_retries"]):
            try:
                return self.client.messages.create(**kwargs)

            except RateLimitError as e:
                delay = min(
                    self.retry_config["base_delay"] * (
                        self.retry_config["backoff_factor"] ** attempt
                    ),
                    self.retry_config["max_delay"],
                )
                # Retry-Afterヘッダがあればそちらを使用
                retry_after = getattr(e, "response", None)
                if retry_after:
                    headers = getattr(retry_after, "headers", {})
                    if "retry-after" in headers:
                        delay = float(headers["retry-after"])

                logger.warning(
                    f"レート制限 (attempt {attempt+1}): "
                    f"{delay:.1f}秒後にリトライ"
                )
                time.sleep(delay)
                last_error = e

            except APIConnectionError as e:
                delay = self.retry_config["base_delay"] * (
                    self.retry_config["backoff_factor"] ** attempt
                )
                logger.warning(f"接続エラー (attempt {attempt+1}): {e}")
                time.sleep(delay)
                last_error = e

            except AuthenticationError as e:
                logger.error(f"認証エラー: {e}")
                raise  # リトライしない

            except BadRequestError as e:
                logger.error(f"リクエストエラー: {e}")
                raise  # リトライしない

            except APIStatusError as e:
                if e.status_code >= 500:
                    delay = self.retry_config["base_delay"] * (
                        self.retry_config["backoff_factor"] ** attempt
                    )
                    logger.warning(f"サーバーエラー {e.status_code} (attempt {attempt+1})")
                    time.sleep(delay)
                    last_error = e
                else:
                    raise  # 4xx系はリトライしない

        raise last_error

    def _safe_execute_tool(self, name: str, input_data: dict) -> dict:
        """安全なツール実行（タイムアウト・サンドボックス付き）"""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("ツール実行がタイムアウトしました")

        # タイムアウト設定
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)  # 30秒タイムアウト

        try:
            result = execute_tool(name, input_data)
            return {"content": result[:10000], "is_error": False}
        except TimeoutError:
            return {"content": "ツール実行がタイムアウト (30秒)", "is_error": True}
        except Exception as e:
            return {"content": f"エラー: {type(e).__name__}: {e}", "is_error": True}
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
```

### 7.2 入力バリデーションとサンドボックス

```python
# ツール入力のバリデーション
from pydantic import BaseModel, Field, validator
from pathlib import Path
import re

class FileReadInput(BaseModel):
    """ファイル読み取りの入力バリデーション"""
    path: str = Field(..., description="ファイルパス")

    @validator("path")
    def validate_path(cls, v):
        # パストラバーサル攻撃の防止
        resolved = Path(v).resolve()
        allowed_dirs = [Path("/workspace"), Path("/tmp")]
        if not any(str(resolved).startswith(str(d)) for d in allowed_dirs):
            raise ValueError(f"アクセス禁止: {resolved}")
        return str(resolved)

class CommandInput(BaseModel):
    """コマンド実行の入力バリデーション"""
    command: str = Field(..., description="実行するコマンド")

    @validator("command")
    def validate_command(cls, v):
        # 危険なコマンドのブロック
        dangerous_patterns = [
            r"rm\s+-rf\s+/",
            r"mkfs\.",
            r"dd\s+if=",
            r":()\{",  # fork bomb
            r">\s*/dev/sd",
            r"curl.*\|\s*bash",
            r"wget.*\|\s*sh",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, v):
                raise ValueError(f"危険なコマンドパターンを検出: {pattern}")
        return v

# バリデーション付きツール実行
def validated_execute_tool(name: str, input_data: dict) -> str:
    """入力バリデーション付きのツール実行"""
    validators = {
        "read_file": FileReadInput,
        "run_command": CommandInput,
    }

    validator_cls = validators.get(name)
    if validator_cls:
        try:
            validated = validator_cls(**input_data)
            input_data = validated.dict()
        except Exception as e:
            return f"バリデーションエラー: {e}"

    return execute_tool(name, input_data)
```

### 7.3 ガードレールの実装

```python
# コンテンツガードレール
class ContentGuardrail:
    """エージェントの出力をチェックするガードレール"""

    def __init__(self):
        self.checks: list[Callable[[str], Optional[str]]] = []

    def add_check(self, check: Callable[[str], Optional[str]]):
        """チェック関数を追加。問題があればエラーメッセージを返す"""
        self.checks.append(check)

    def validate(self, content: str) -> tuple[bool, list[str]]:
        """コンテンツを検証"""
        errors = []
        for check in self.checks:
            error = check(content)
            if error:
                errors.append(error)
        return len(errors) == 0, errors

# ガードレールの定義
guardrail = ContentGuardrail()

def check_no_secrets(content: str) -> Optional[str]:
    """秘密情報の漏洩チェック"""
    patterns = [
        (r"(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}", "AWSアクセスキー"),
        (r"sk-[a-zA-Z0-9]{20,}", "APIキー"),
        (r"ghp_[a-zA-Z0-9]{36}", "GitHubトークン"),
        (r"-----BEGIN (?:RSA )?PRIVATE KEY-----", "秘密鍵"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, content):
            return f"秘密情報の漏洩検出: {name}"
    return None

def check_no_pii(content: str) -> Optional[str]:
    """個人情報のチェック"""
    patterns = [
        (r"\b\d{3}-\d{4}-\d{4}\b", "電話番号"),
        (r"\b\d{3}-\d{2}-\d{4}\b", "SSN"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "メールアドレス"),
    ]
    for pattern, name in patterns:
        if re.search(pattern, content):
            return f"PII検出: {name}"
    return None

def check_max_length(content: str) -> Optional[str]:
    """出力長チェック"""
    if len(content) > 50000:
        return f"出力が長すぎます: {len(content)}文字"
    return None

guardrail.add_check(check_no_secrets)
guardrail.add_check(check_no_pii)
guardrail.add_check(check_max_length)

# エージェントの出力をチェック
def guarded_agent(user_message: str) -> str:
    result = agent_loop(user_message)

    is_valid, errors = guardrail.validate(result)
    if not is_valid:
        logger.warning(f"ガードレール違反: {errors}")
        return "申し訳ございませんが、回答にセキュリティ上の問題が検出されました。"

    return result
```

---

## 8. コンテキスト管理と会話メモリ

### 8.1 トークンカウントと予算管理

```python
# トークン使用量の追跡と予算管理
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TokenBudget:
    """トークン使用量の予算管理"""
    max_input_tokens: int = 150_000  # 200Kの75%を上限に
    max_output_tokens_per_turn: int = 4096
    max_total_cost_usd: float = 1.0  # セッション全体のコスト上限

    # 累積値
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    api_calls: int = 0

    def record_usage(self, input_tokens: int, output_tokens: int, model: str):
        """使用量を記録"""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1

        # コスト計算（モデルごとの単価）
        cost_map = {
            "claude-haiku-4-20250514": (0.00025, 0.00125),
            "claude-sonnet-4-20250514": (0.003, 0.015),
            "claude-opus-4-20250514": (0.015, 0.075),
        }
        input_rate, output_rate = cost_map.get(model, (0.003, 0.015))
        cost = (input_tokens * input_rate + output_tokens * output_rate) / 1000
        self.total_cost_usd += cost

    def check_budget(self) -> tuple[bool, str]:
        """予算チェック"""
        if self.total_cost_usd >= self.max_total_cost_usd:
            return False, f"コスト上限超過: ${self.total_cost_usd:.4f} >= ${self.max_total_cost_usd}"
        return True, "OK"

    def get_report(self) -> dict:
        """使用レポート"""
        return {
            "api_calls": self.api_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "budget_remaining_usd": round(self.max_total_cost_usd - self.total_cost_usd, 6),
        }
```

### 8.2 会話履歴の圧縮

```python
# 会話履歴の圧縮戦略
class ConversationManager:
    """会話履歴を効率的に管理するマネージャー"""

    def __init__(self, max_messages: int = 50, summarize_threshold: int = 30):
        self.messages: list[dict] = []
        self.max_messages = max_messages
        self.summarize_threshold = summarize_threshold
        self.client = anthropic.Anthropic()
        self.summaries: list[str] = []

    def add_message(self, role: str, content):
        """メッセージを追加"""
        self.messages.append({"role": role, "content": content})

        # しきい値を超えたら圧縮
        if len(self.messages) >= self.summarize_threshold:
            self._compress()

    def _compress(self):
        """古い会話を要約して圧縮"""
        # 前半を要約
        half = len(self.messages) // 2
        old_messages = self.messages[:half]

        # 要約生成
        summary_text = self._summarize(old_messages)
        self.summaries.append(summary_text)

        # 要約で置き換え
        summary_message = {
            "role": "user",
            "content": f"[これまでの会話の要約]\n{summary_text}"
        }
        self.messages = [summary_message] + self.messages[half:]

    def _summarize(self, messages: list[dict]) -> str:
        """メッセージリストを要約"""
        # メッセージをテキスト化
        text_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                text_parts.append(f"{role}: {content[:500]}")
            elif isinstance(content, list):
                # ツール結果等
                text_parts.append(f"{role}: [ツール操作]")

        conversation_text = "\n".join(text_parts)

        response = self.client.messages.create(
            model="claude-haiku-4-20250514",  # 要約は安価なモデルで
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"以下の会話を200文字以内で要約:\n{conversation_text}"
            }]
        )

        return response.content[0].text

    def get_messages(self) -> list[dict]:
        """現在のメッセージリストを返す"""
        return self.messages.copy()
```

### 8.3 スライディングウィンドウ戦略

```python
# スライディングウィンドウによるコンテキスト管理
class SlidingWindowManager:
    """固定サイズのウィンドウでメッセージを管理"""

    def __init__(self, window_size: int = 20, keep_system: bool = True):
        self.window_size = window_size
        self.keep_system = keep_system
        self.all_messages: list[dict] = []
        self.pinned_messages: list[dict] = []  # 常に保持するメッセージ

    def add(self, message: dict):
        self.all_messages.append(message)

    def pin(self, message: dict):
        """常に保持するメッセージを追加"""
        self.pinned_messages.append(message)

    def get_window(self) -> list[dict]:
        """現在のウィンドウ内のメッセージを取得"""
        # ピン留めメッセージ + 最新N件
        recent = self.all_messages[-self.window_size:]

        # assistant/userの対が壊れないように調整
        if recent and recent[0]["role"] == "assistant":
            recent = recent[1:]  # assistantから始まる場合は除去

        return self.pinned_messages + recent

    def estimate_tokens(self) -> int:
        """現在のウィンドウのトークン数を推定"""
        total_chars = sum(
            len(str(m.get("content", ""))) for m in self.get_window()
        )
        return total_chars // 4  # 大まかな推定（日本語は約2文字/トークン）
```

---

## 9. 非同期エージェント

### 9.1 完全非同期実装

```python
# 完全非同期のエージェント実装
import anthropic
import asyncio
from typing import AsyncIterator

class AsyncClaudeAgent:
    """非同期版のClaudeエージェント"""

    def __init__(self, config: AgentConfig, tools: list, handlers: dict):
        self.config = config
        self.tools = tools
        self.handlers = handlers
        self.client = anthropic.AsyncAnthropic()

    async def run(self, user_message: str) -> str:
        """非同期でエージェントループを実行"""
        messages = [{"role": "user", "content": user_message}]

        for step in range(self.config.max_steps):
            response = await self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            tool_results = await self._process_tools_async(response)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "最大ステップ数に達しました"

    async def run_streaming(self, user_message: str) -> AsyncIterator[str]:
        """ストリーミング付き非同期実行"""
        messages = [{"role": "user", "content": user_message}]

        while True:
            async with self.client.messages.stream(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                tools=self.tools,
                messages=messages,
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield event.delta.text

                response = await stream.get_final_message()

            if response.stop_reason == "end_turn":
                return

            tool_results = await self._process_tools_async(response)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    async def _process_tools_async(self, response) -> list:
        """非同期ツール処理"""
        tasks = []
        for block in response.content:
            if block.type == "tool_use":
                tasks.append(self._execute_tool_async(block))

        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_tool_async(self, block) -> dict:
        """単一ツールの非同期実行"""
        handler = self.handlers.get(block.name)
        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(**block.input)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: handler(**block.input))
        except Exception as e:
            result = f"エラー: {e}"

        return {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": str(result)[:10000],
        }

    def _extract_text(self, response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""

# 使用例
async def main():
    agent = AsyncClaudeAgent(
        config=AgentConfig(system_prompt="非同期開発アシスタント"),
        tools=TOOLS,
        handlers={"read_file": read_file, "run_command": run_command},
    )

    # 通常の非同期実行
    result = await agent.run("プロジェクトの構造を教えて")
    print(result)

    # ストリーミング実行
    async for chunk in agent.run_streaming("テストを実行して"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

---

## 10. モニタリングとオブザーバビリティ

### 10.1 構造化ログとメトリクス

```python
# エージェントのモニタリング
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

@dataclass
class AgentMetrics:
    """エージェント実行のメトリクス"""
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    total_steps: int = 0
    api_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    total_cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "duration_seconds": round(self.end_time - self.start_time, 2),
            "total_steps": self.total_steps,
            "api_calls": self.api_calls,
            "tokens": {
                "input": self.total_input_tokens,
                "output": self.total_output_tokens,
                "total": self.total_input_tokens + self.total_output_tokens,
            },
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "cost_usd": round(self.total_cost_usd, 6),
        }

class MonitoredAgent:
    """メトリクス収集付きエージェント"""

    def __init__(self, config: AgentConfig, tools: list, handlers: dict):
        self.config = config
        self.tools = tools
        self.handlers = handlers
        self.client = anthropic.Anthropic()
        self.logger = logging.getLogger("agent")
        self.metrics: Optional[AgentMetrics] = None

    def run(self, user_message: str) -> tuple[str, AgentMetrics]:
        """メトリクス付きで実行"""
        import uuid
        self.metrics = AgentMetrics(
            session_id=str(uuid.uuid4())[:8],
            start_time=time.time(),
        )

        messages = [{"role": "user", "content": user_message}]

        try:
            result = self._agent_loop(messages)
        except Exception as e:
            self.metrics.errors.append({
                "type": type(e).__name__,
                "message": str(e),
                "step": self.metrics.total_steps,
            })
            result = f"エラーで中断: {e}"
        finally:
            self.metrics.end_time = time.time()

        self.logger.info(
            "Agent completed",
            extra={"metrics": self.metrics.to_dict()}
        )

        return result, self.metrics

    def _agent_loop(self, messages: list) -> str:
        for step in range(self.config.max_steps):
            self.metrics.total_steps = step + 1

            step_start = time.time()
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                system=self.config.system_prompt,
                tools=self.tools,
                messages=messages,
            )
            api_latency = time.time() - step_start

            # メトリクス記録
            self.metrics.api_calls += 1
            self.metrics.total_input_tokens += response.usage.input_tokens
            self.metrics.total_output_tokens += response.usage.output_tokens

            self.logger.debug(
                f"Step {step}: stop_reason={response.stop_reason}, "
                f"tokens={response.usage.input_tokens}+{response.usage.output_tokens}, "
                f"latency={api_latency:.2f}s"
            )

            if response.stop_reason == "end_turn":
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            # ツール処理
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_start = time.time()
                    result = execute_tool(block.name, block.input)
                    tool_duration = time.time() - tool_start

                    self.metrics.tool_calls.append({
                        "name": block.name,
                        "duration_seconds": round(tool_duration, 3),
                        "result_length": len(result),
                        "step": step,
                    })

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result[:10000],
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return "最大ステップ数に達しました"

# 使用例
agent = MonitoredAgent(
    config=AgentConfig(system_prompt="開発アシスタント"),
    tools=TOOLS,
    handlers={"read_file": read_file},
)
result, metrics = agent.run("setup.pyを読んで内容を説明して")
print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))
```

### 10.2 OpenTelemetryとの統合

```python
# OpenTelemetryを使ったトレーシング
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import StatusCode

# トレーサーの設定
provider = TracerProvider()
processor = SimpleSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("claude-agent")

class TracedAgent:
    """OpenTelemetryトレーシング付きエージェント"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = anthropic.Anthropic()

    def run(self, user_message: str) -> str:
        with tracer.start_as_current_span("agent.run") as span:
            span.set_attribute("agent.model", self.config.model)
            span.set_attribute("agent.max_steps", self.config.max_steps)
            span.set_attribute("input.length", len(user_message))

            messages = [{"role": "user", "content": user_message}]

            for step in range(self.config.max_steps):
                with tracer.start_as_current_span(f"agent.step.{step}") as step_span:
                    # API呼び出し
                    with tracer.start_as_current_span("api.messages.create") as api_span:
                        response = self.client.messages.create(
                            model=self.config.model,
                            max_tokens=self.config.max_tokens,
                            system=self.config.system_prompt,
                            tools=TOOLS,
                            messages=messages,
                        )
                        api_span.set_attribute("tokens.input", response.usage.input_tokens)
                        api_span.set_attribute("tokens.output", response.usage.output_tokens)
                        api_span.set_attribute("stop_reason", response.stop_reason)

                    if response.stop_reason == "end_turn":
                        result = self._extract_text(response)
                        span.set_attribute("output.length", len(result))
                        span.set_attribute("total_steps", step + 1)
                        return result

                    # ツール処理
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            with tracer.start_as_current_span(
                                f"tool.{block.name}"
                            ) as tool_span:
                                tool_span.set_attribute("tool.input", json.dumps(block.input)[:200])
                                try:
                                    result = execute_tool(block.name, block.input)
                                    tool_span.set_attribute("tool.result_length", len(result))
                                except Exception as e:
                                    tool_span.set_status(StatusCode.ERROR)
                                    tool_span.record_exception(e)
                                    result = f"エラー: {e}"

                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result[:10000],
                                })

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user", "content": tool_results})

            span.set_status(StatusCode.ERROR, "最大ステップ数超過")
            return "最大ステップ数に達しました"

    def _extract_text(self, response) -> str:
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
```

---

## 11. アーキテクチャ図

```
Claude Agent SDK ベースのエージェント構成

+------------------------------------------------------+
|                    Application                        |
|                                                        |
|  +--------------------------------------------------+|
|  |            ClaudeAgent                            ||
|  |                                                    ||
|  |  +--------+  +----------+  +---------+           ||
|  |  | Config |  | History  |  | Metrics |           ||
|  |  +--------+  +----------+  +---------+           ||
|  |       |                                           ||
|  |  +----v----+                                      ||
|  |  | Agent   |  messages.create()                   ||
|  |  | Loop    |<--------------------> Anthropic API  ||
|  |  +----+----+                                      ||
|  |       |                                           ||
|  |  +----v-----------+                               ||
|  |  | Tool Dispatcher |                              ||
|  |  +----+-----------+                               ||
|  |       |     |     |                               ||
|  |  +----v+ +--v--+ +v------+                        ||
|  |  |File | |Shell| |MCP    |                        ||
|  |  |Ops  | |Exec | |Client |                        ||
|  |  +-----+ +-----+ +---+---+                        ||
|  |                       |                            ||
|  +--------------------------------------------------+||
|                          |                             |
+------------------------------------------------------+
                           v
                    +------+------+
                    | MCP Servers |
                    +-------------+
```

```
マルチエージェントオーケストレーション構成

+----------------------------------------------------------+
|                    Orchestrator Agent                      |
|  (タスク分解・割り当て・結果統合)                          |
+----+-------------------+-------------------+--------------+
     |                   |                   |
     v                   v                   v
+----------+      +----------+        +----------+
| Coder    |      | Reviewer |        | Tester   |
| Agent    |      | Agent    |        | Agent    |
+----+-----+      +----+-----+        +----+-----+
     |                  |                   |
     v                  v                   v
+----------+      +----------+        +----------+
| File Ops |      | Code     |        | Test     |
| Shell    |      | Analysis |        | Runner   |
| MCP      |      | Tools    |        | Tools    |
+----------+      +----------+        +----------+
```

---

## 12. 比較表

### 12.1 Claude SDKの利用パターン

| パターン | コード量 | 柔軟性 | 複雑度 | 適用場面 |
|----------|---------|--------|--------|---------|
| 直接API呼び出し | 最少 | 最高 | 低 | 単純なツール使用 |
| クラスベース | 中 | 高 | 中 | 再利用可能なエージェント |
| MCP統合 | 中-多 | 高 | 中-高 | ツール共有 |
| マルチエージェント | 多 | 高 | 高 | 複雑なタスク |
| パイプライン | 中 | 中 | 低-中 | 段階的処理 |
| 非同期 | 中 | 高 | 中 | 高スループット要件 |

### 12.2 モデル選択ガイド

| モデル | コスト | 速度 | 推論力 | 適用場面 |
|--------|--------|------|--------|---------|
| Claude Haiku | 最低 | 最速 | 基本 | 分類、ルーティング、要約 |
| Claude Sonnet | 中 | 速い | 高 | 一般的なエージェント、コーディング |
| Claude Opus | 高 | 遅い | 最高 | 複雑な推論、設計、重要な判断 |

### 12.3 エラーハンドリング戦略

| エラー種別 | リトライ | 対処法 | 備考 |
|-----------|---------|--------|------|
| RateLimitError | する | 指数バックオフ | Retry-Afterヘッダ参照 |
| APIConnectionError | する | 指数バックオフ | ネットワーク一時障害 |
| AuthenticationError | しない | APIキー確認 | 即座にエラー返却 |
| BadRequestError | しない | 入力修正 | メッセージ形式の問題 |
| 500系サーバーエラー | する | 指数バックオフ | サーバー側の問題 |
| ToolTimeoutError | 条件付き | タイムアウト延長 or スキップ | ツール依存 |

---

## 13. プロダクションデプロイパターン

### 13.1 FastAPIとの統合

```python
# FastAPIでエージェントをAPIとして公開
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import uuid

app = FastAPI(title="Claude Agent API")

class AgentRequest(BaseModel):
    message: str
    system_prompt: str = "あなたは有能なAIアシスタントです。"
    max_steps: int = 10
    model: str = "claude-sonnet-4-20250514"

class AgentResponse(BaseModel):
    session_id: str
    result: str
    metrics: dict

# セッション管理
sessions: dict[str, dict] = {}

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    """同期的にエージェントを実行"""
    session_id = str(uuid.uuid4())[:8]

    config = AgentConfig(
        model=request.model,
        max_steps=request.max_steps,
        system_prompt=request.system_prompt,
    )

    agent = MonitoredAgent(config=config, tools=TOOLS, handlers={
        "read_file": read_file,
        "write_file": write_file,
        "run_command": run_command,
    })

    result, metrics = agent.run(request.message)

    sessions[session_id] = {
        "result": result,
        "metrics": metrics.to_dict(),
    }

    return AgentResponse(
        session_id=session_id,
        result=result,
        metrics=metrics.to_dict(),
    )

@app.post("/agent/stream")
async def stream_agent(request: AgentRequest):
    """ストリーミングでエージェントを実行"""
    async_agent = AsyncClaudeAgent(
        config=AgentConfig(
            model=request.model,
            max_steps=request.max_steps,
            system_prompt=request.system_prompt,
        ),
        tools=TOOLS,
        handlers={"read_file": read_file},
    )

    async def generate():
        async for chunk in async_agent.run_streaming(request.message):
            yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """セッション情報を取得"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="セッションが見つかりません")
    return sessions[session_id]
```

### 13.2 キューイングとバックグラウンド処理

```python
# Celeryを使ったバックグラウンドエージェント実行
from celery import Celery
import redis

celery_app = Celery("agent_tasks", broker="redis://localhost:6379/0")
redis_client = redis.Redis(host="localhost", port=6379, db=1)

@celery_app.task(bind=True, max_retries=2)
def run_agent_task(self, task_id: str, message: str, config_dict: dict):
    """バックグラウンドでエージェントタスクを実行"""
    try:
        # 進捗を通知
        redis_client.hset(f"task:{task_id}", "status", "running")

        config = AgentConfig(**config_dict)
        agent = MonitoredAgent(config=config, tools=TOOLS, handlers={
            "read_file": read_file,
            "write_file": write_file,
            "run_command": run_command,
        })

        result, metrics = agent.run(message)

        # 結果を保存
        redis_client.hset(f"task:{task_id}", mapping={
            "status": "completed",
            "result": result,
            "metrics": json.dumps(metrics.to_dict(), ensure_ascii=False),
        })
        redis_client.expire(f"task:{task_id}", 3600)  # 1時間保持

        return {"task_id": task_id, "status": "completed"}

    except Exception as e:
        redis_client.hset(f"task:{task_id}", mapping={
            "status": "failed",
            "error": str(e),
        })
        raise self.retry(exc=e, countdown=5)

# タスクの投入と状態確認
def submit_agent_task(message: str) -> str:
    task_id = str(uuid.uuid4())[:8]
    redis_client.hset(f"task:{task_id}", "status", "queued")

    run_agent_task.delay(
        task_id=task_id,
        message=message,
        config_dict={"model": "claude-sonnet-4-20250514", "max_steps": 20},
    )
    return task_id

def check_task_status(task_id: str) -> dict:
    data = redis_client.hgetall(f"task:{task_id}")
    return {k.decode(): v.decode() for k, v in data.items()}
```

---

## 14. アンチパターン

### アンチパターン1: 会話履歴の無制限蓄積

```python
# NG: 履歴が無限に増えてコンテキスト超過
messages = []
while True:
    messages.append(...)  # 増え続ける

# OK: 履歴の管理（要約 or スライディングウィンドウ）
MAX_HISTORY = 50

def manage_history(messages: list) -> list:
    if len(messages) > MAX_HISTORY:
        # 古い履歴を要約
        summary = summarize(messages[:MAX_HISTORY//2])
        return [
            {"role": "user", "content": f"これまでの要約: {summary}"},
            *messages[MAX_HISTORY//2:]
        ]
    return messages
```

### アンチパターン2: ツール結果のサイズ無制限

```python
# NG: 巨大なファイル内容をそのまま返す
def read_file(path):
    with open(path) as f:
        return f.read()  # 100MBのファイルかもしれない

# OK: サイズ制限と要約
def read_file(path, max_chars=10000):
    with open(path) as f:
        content = f.read()
    if len(content) > max_chars:
        return content[:max_chars] + f"\n... (残り {len(content)-max_chars} 文字省略)"
    return content
```

### アンチパターン3: エラー無視のツール実行

```python
# NG: エラーを握りつぶす
def execute_tool(name, input_data):
    try:
        return do_something(name, input_data)
    except:
        return ""  # 空文字を返して何もなかったことにする

# OK: エラーを明示的に返す
def execute_tool(name, input_data):
    try:
        return do_something(name, input_data)
    except Exception as e:
        return json.dumps({
            "error": True,
            "type": type(e).__name__,
            "message": str(e),
            "tool": name,
        }, ensure_ascii=False)
```

### アンチパターン4: 無制限のステップ数

```python
# NG: 無限ループの可能性
def agent_loop(message):
    while True:  # 永遠に続く可能性
        response = call_api(message)
        if response.stop_reason == "end_turn":
            return response

# OK: ステップ制限 + タイムアウト
def agent_loop(message, max_steps=20, timeout=300):
    start = time.time()
    for step in range(max_steps):
        if time.time() - start > timeout:
            return "タイムアウト"
        response = call_api(message)
        if response.stop_reason == "end_turn":
            return response
    return "最大ステップ数超過"
```

### アンチパターン5: 単一モデルでの全処理

```python
# NG: すべてのタスクに高価なモデルを使用
def process_all(tasks):
    for task in tasks:
        result = client.messages.create(
            model="claude-opus-4-20250514",  # 全部Opusは高コスト
            ...
        )

# OK: タスクに応じたモデル選択
MODEL_ROUTING = {
    "classify": "claude-haiku-4-20250514",     # 分類は安価に
    "summarize": "claude-haiku-4-20250514",    # 要約も安価に
    "generate": "claude-sonnet-4-20250514",    # 生成は中程度
    "reason": "claude-opus-4-20250514",        # 複雑な推論のみ高品質
}

def smart_process(task_type, content):
    model = MODEL_ROUTING.get(task_type, "claude-sonnet-4-20250514")
    return client.messages.create(model=model, ...)
```

---

## 15. テストパターン

### 15.1 エージェントのユニットテスト

```python
# エージェントのテスト
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

class TestClaudeAgent:
    """ClaudeAgentのユニットテスト"""

    def setup_method(self):
        self.config = AgentConfig(
            model="claude-sonnet-4-20250514",
            max_steps=5,
            system_prompt="テスト用プロンプト",
        )

    @patch("anthropic.Anthropic")
    def test_simple_response(self, mock_anthropic_cls):
        """ツール使用なしの単純なレスポンス"""
        # モックの設定
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [MagicMock(text="テスト回答", type="text")]
        mock_response.content[0].text = "テスト回答"
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        mock_client.messages.create.return_value = mock_response

        agent = ClaudeAgent(self.config, tools=[], handlers={})
        result = agent.run("テスト質問")

        assert result == "テスト回答"
        mock_client.messages.create.assert_called_once()

    @patch("anthropic.Anthropic")
    def test_tool_use_and_response(self, mock_anthropic_cls):
        """ツール使用後にレスポンスを返す"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # 1回目: ツール呼び出し
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "read_file"
        tool_block.id = "tool_123"
        tool_block.input = {"path": "test.py"}
        tool_response.content = [tool_block]
        tool_response.usage.input_tokens = 100
        tool_response.usage.output_tokens = 50

        # 2回目: 最終回答
        final_response = MagicMock()
        final_response.stop_reason = "end_turn"
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ファイル内容: hello"
        final_response.content = [text_block]
        final_response.usage.input_tokens = 200
        final_response.usage.output_tokens = 80

        mock_client.messages.create.side_effect = [tool_response, final_response]

        # ハンドラ
        handlers = {"read_file": lambda path: "hello"}

        agent = ClaudeAgent(self.config, tools=TOOLS, handlers=handlers)
        result = agent.run("test.pyを読んで")

        assert "hello" in result
        assert mock_client.messages.create.call_count == 2

    @patch("anthropic.Anthropic")
    def test_max_steps_exceeded(self, mock_anthropic_cls):
        """最大ステップ数超過"""
        mock_client = MagicMock()
        mock_anthropic_cls.return_value = mock_client

        # 常にツール呼び出しを返す（無限ループ）
        tool_response = MagicMock()
        tool_response.stop_reason = "tool_use"
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "read_file"
        tool_block.id = "tool_123"
        tool_block.input = {"path": "test.py"}
        tool_response.content = [tool_block]
        tool_response.usage.input_tokens = 100
        tool_response.usage.output_tokens = 50

        mock_client.messages.create.return_value = tool_response

        config = AgentConfig(max_steps=3)
        agent = ClaudeAgent(config, tools=TOOLS, handlers={
            "read_file": lambda path: "content"
        })
        result = agent.run("テスト")

        assert "最大ステップ" in result
        assert mock_client.messages.create.call_count == 3

class TestToolRegistry:
    """ToolRegistryのテスト"""

    def test_tool_registration(self):
        """ツール登録のテスト"""
        registry = ToolRegistry()

        @registry.tool("テスト用ツール")
        def my_tool(name: str, count: int = 1) -> str:
            return f"{name} x {count}"

        tools = registry.get_tool_definitions()
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"
        assert "name" in tools[0]["input_schema"]["properties"]
        assert "name" in tools[0]["input_schema"]["required"]
        assert "count" not in tools[0]["input_schema"]["required"]

    def test_tool_execution(self):
        """ツール実行のテスト"""
        registry = ToolRegistry()

        @registry.tool()
        def add(a: int, b: int) -> int:
            return a + b

        result = registry.execute("add", {"a": 3, "b": 5})
        assert result == "8"

    def test_unknown_tool(self):
        """未登録ツールの実行"""
        registry = ToolRegistry()
        result = registry.execute("unknown", {})
        assert "未登録" in result
```

### 15.2 インテグレーションテスト

```python
# エージェントのインテグレーションテスト
import pytest
import tempfile
import os

class TestAgentIntegration:
    """実際のAPIを使ったインテグレーションテスト"""

    @pytest.fixture
    def agent(self):
        config = AgentConfig(
            model="claude-haiku-4-20250514",  # テストは安価なモデルで
            max_steps=5,
            system_prompt="テスト用アシスタント。簡潔に回答してください。",
        )
        return ClaudeAgent(config, tools=TOOLS, handlers={
            "read_file": lambda path: open(path).read(),
            "write_file": lambda path, content: open(path, "w").write(content) or "OK",
        })

    @pytest.mark.integration
    def test_file_read_agent(self, agent):
        """ファイル読み取りエージェントのE2Eテスト"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("テストデータ: 42")
            temp_path = f.name

        try:
            result = agent.run(f"{temp_path}を読んで内容を教えて")
            assert "42" in result or "テストデータ" in result
        finally:
            os.unlink(temp_path)

    @pytest.mark.integration
    def test_agent_timeout(self, agent):
        """タイムアウトテスト"""
        agent.config.timeout = 0.001  # 極端に短いタイムアウト
        result = agent.run("複雑な計算をして")
        assert "タイムアウト" in result or len(result) > 0
```

---

## 16. FAQ

### Q1: extended thinkingをエージェントで使うべきか？

extended thinking（拡張思考）は複雑な推論タスクに有効だが、エージェントでは注意が必要:
- **利点**: 計画立案、複雑なバグ修正で品質向上
- **注意点**: レイテンシ増加、ツール使用との組み合わせに制約あり
- **推奨**: 計画フェーズのみextended thinkingを有効にし、ツール実行フェーズでは無効にする

### Q2: レート制限への対処方法は？

```python
import time
from anthropic import RateLimitError

def call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
    raise Exception("レート制限超過")
```

### Q3: 複数のClaudeモデルをエージェント内で使い分けるには？

```python
# ルーティング用は安価なモデル、生成は高品質モデル
ROUTING_MODEL = "claude-haiku-4-20250514"
GENERATION_MODEL = "claude-sonnet-4-20250514"

def smart_agent(query):
    # Step 1: 高速モデルで分類
    category = client.messages.create(
        model=ROUTING_MODEL, ...
    )
    # Step 2: 高品質モデルで回答
    answer = client.messages.create(
        model=GENERATION_MODEL, ...
    )
```

### Q4: コンテキストウィンドウを効率的に使うには？

```python
# コンテキスト最適化の3つの戦略

# 1. ツール結果の切り詰め
def truncate_result(result: str, max_chars: int = 5000) -> str:
    if len(result) <= max_chars:
        return result
    # 先頭と末尾を保持
    head = result[:max_chars // 2]
    tail = result[-(max_chars // 2):]
    return f"{head}\n\n... ({len(result) - max_chars}文字省略) ...\n\n{tail}"

# 2. 不要な中間結果の除去
def clean_history(messages: list) -> list:
    """古いツール結果を圧縮"""
    cleaned = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            # ツール結果を短縮
            shortened = []
            for item in msg["content"]:
                if item.get("type") == "tool_result":
                    content = item.get("content", "")
                    if len(content) > 500:
                        item = {**item, "content": content[:500] + "...(省略)"}
                shortened.append(item)
            cleaned.append({"role": "user", "content": shortened})
        else:
            cleaned.append(msg)
    return cleaned

# 3. セマンティックキャッシュ
class SemanticCache:
    """類似の質問に対するキャッシュ"""
    def __init__(self):
        self.cache: dict[str, str] = {}

    def get(self, query: str) -> str | None:
        # 簡易的なキーマッチング（実運用ではembeddingを使用）
        normalized = query.lower().strip()
        return self.cache.get(normalized)

    def set(self, query: str, response: str):
        normalized = query.lower().strip()
        self.cache[normalized] = response
```

### Q5: エージェントのデバッグ方法は？

```python
# デバッグモードのエージェント
import sys

class DebugAgent(ClaudeAgent):
    """デバッグ情報を出力するエージェント"""

    def __init__(self, *args, verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose

    def run(self, user_message: str) -> str:
        if self.verbose:
            print(f"=== Agent Start ===", file=sys.stderr)
            print(f"Model: {self.config.model}", file=sys.stderr)
            print(f"Message: {user_message[:100]}...", file=sys.stderr)
            print(f"Tools: {[t['name'] for t in self.tools]}", file=sys.stderr)

        result = super().run(user_message)

        if self.verbose:
            print(f"=== Agent End ===", file=sys.stderr)
            print(f"Steps: {len(self.conversation_history) // 2}", file=sys.stderr)
            print(f"Result length: {len(result)}", file=sys.stderr)

        return result
```

### Q6: Batching APIで大量リクエストを効率的に処理するには？

```python
# Batch APIを使った大量処理
import anthropic
import json

def create_batch_requests(tasks: list[dict]) -> list[dict]:
    """バッチリクエストを作成"""
    requests = []
    for i, task in enumerate(tasks):
        requests.append({
            "custom_id": f"task-{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": task["prompt"]}],
            }
        })
    return requests

def submit_batch(requests: list[dict]) -> str:
    """バッチを送信"""
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)
    return batch.id

def check_batch_status(batch_id: str) -> dict:
    """バッチの状態を確認"""
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    return {
        "id": batch.id,
        "status": batch.processing_status,
        "created_at": str(batch.created_at),
        "request_counts": {
            "processing": batch.request_counts.processing,
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
        },
    }
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| 基本ループ | messages.create → stop_reason判定 → ツール実行 → 繰り返し |
| ストリーミング | messages.stream でリアルタイム出力 |
| 並列ツール | 1レスポンスで複数tool_useブロック |
| MCP統合 | MCPサーバーのツールをそのまま利用 |
| マルチエージェント | オーケストレーター + ワーカー / パイプライン |
| エラーハンドリング | 指数バックオフ + バリデーション + ガードレール |
| モニタリング | 構造化ログ + メトリクス + OpenTelemetry |
| コンテキスト管理 | スライディングウィンドウ + 履歴圧縮 |
| テスト | ユニットテスト(モック) + インテグレーションテスト |
| 設計原則 | 最小限のコード、明示的な制御、安全なデフォルト |

## 次に読むべきガイド

- [04-evaluation.md](./04-evaluation.md) -- エージェントの評価手法
- [02-mcp-agents.md](./02-mcp-agents.md) -- MCP統合の詳細
- [../04-production/00-deployment.md](../04-production/00-deployment.md) -- デプロイとスケーリング

## 参考文献

1. Anthropic, "Claude API Reference" -- https://docs.anthropic.com/en/api/
2. Anthropic, "Building effective agents" -- https://docs.anthropic.com/en/docs/build-with-claude/agentic
3. anthropic-sdk-python GitHub -- https://github.com/anthropics/anthropic-sdk-python
4. Anthropic, "Tool use (function calling)" -- https://docs.anthropic.com/en/docs/build-with-claude/tool-use
5. Anthropic, "Extended thinking" -- https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
6. Anthropic, "Message Batches API" -- https://docs.anthropic.com/en/docs/build-with-claude/message-batches
7. Model Context Protocol Specification -- https://spec.modelcontextprotocol.io/
8. OpenTelemetry Python SDK -- https://opentelemetry.io/docs/languages/python/
