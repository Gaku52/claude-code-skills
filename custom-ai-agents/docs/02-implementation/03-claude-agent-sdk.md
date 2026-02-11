# Claude Agent SDK

> Anthropic公式のエージェント構築ツール――最小限のコードでツール使用エージェントを構築し、MCPサーバーとネイティブ統合する方法を解説する。

## この章で学ぶこと

1. Claude Messages API を使ったエージェントループの構築パターン
2. ツール定義・並列ツール呼び出し・ストリーミングの実装
3. MCPとの統合およびプロダクション向け設計パターン

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

### 3.2 並列ツール呼び出し

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

### 3.3 システムプロンプト設計

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

---

## 4. エージェントの構造化

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

---

## 5. アーキテクチャ図

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

---

## 6. 比較表

### 6.1 Claude SDKの利用パターン

| パターン | コード量 | 柔軟性 | 複雑度 | 適用場面 |
|----------|---------|--------|--------|---------|
| 直接API呼び出し | 最少 | 最高 | 低 | 単純なツール使用 |
| クラスベース | 中 | 高 | 中 | 再利用可能なエージェント |
| MCP統合 | 中-多 | 高 | 中-高 | ツール共有 |
| マルチエージェント | 多 | 高 | 高 | 複雑なタスク |

### 6.2 モデル選択ガイド

| モデル | コスト | 速度 | 推論力 | 適用場面 |
|--------|--------|------|--------|---------|
| Claude Haiku | 最低 | 最速 | 基本 | 分類、ルーティング |
| Claude Sonnet | 中 | 速い | 高 | 一般的なエージェント |
| Claude Opus | 高 | 遅い | 最高 | 複雑な推論、コーディング |

---

## 7. アンチパターン

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

---

## 8. FAQ

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

---

## まとめ

| 項目 | 内容 |
|------|------|
| 基本ループ | messages.create → stop_reason判定 → ツール実行 → 繰り返し |
| ストリーミング | messages.stream でリアルタイム出力 |
| 並列ツール | 1レスポンスで複数tool_useブロック |
| MCP統合 | MCPサーバーのツールをそのまま利用 |
| 設計原則 | 最小限のコード、明示的な制御、安全なデフォルト |

## 次に読むべきガイド

- [04-evaluation.md](./04-evaluation.md) — エージェントの評価手法
- [02-mcp-agents.md](./02-mcp-agents.md) — MCP統合の詳細
- [../04-production/00-deployment.md](../04-production/00-deployment.md) — デプロイとスケーリング

## 参考文献

1. Anthropic, "Claude API Reference" — https://docs.anthropic.com/en/api/
2. Anthropic, "Building effective agents" — https://docs.anthropic.com/en/docs/build-with-claude/agentic
3. anthropic-sdk-python GitHub — https://github.com/anthropics/anthropic-sdk-python
