# AIエージェント概要

> LLMを頭脳として持ち、ツールを手足として使い、自律的にタスクを遂行するソフトウェアシステム――AIエージェントの定義・種類・アーキテクチャを体系的に解説する。

## この章で学ぶこと

1. AIエージェントの定義と従来のチャットボットとの根本的な違い
2. エージェントの5つの主要アーキテクチャパターンと選択基準
3. エージェントループ（Perceive-Think-Act）の内部構造と実装原理

---

## 1. AIエージェントとは何か

### 1.1 定義

AIエージェントとは **目標を与えられると、環境を観察し、推論し、ツールを使って行動するシステム** である。従来のチャットボットが「質問→回答」の1ターンで終わるのに対し、エージェントは **複数ステップを自律的に計画・実行・評価** するループを持つ。

```
従来のチャットボット:
  ユーザー → [LLM] → 回答（1ターン完結）

AIエージェント:
  ユーザー → [計画] → [ツール実行] → [結果観察] → [再計画] → ... → 最終回答
```

### 1.2 エージェントの3要素

```
+---------------------------------------------------+
|                  AI エージェント                     |
|                                                     |
|  +-------------+  +-----------+  +---------------+  |
|  |   頭脳      |  |   記憶    |  |   手足        |  |
|  |  (LLM)      |  | (Memory)  |  |  (Tools)      |  |
|  |             |  |           |  |               |  |
|  | - 推論      |  | - 短期    |  | - Web検索     |  |
|  | - 計画      |  | - 長期    |  | - コード実行  |  |
|  | - 判断      |  | - 外部DB  |  | - API呼出     |  |
|  +-------------+  +-----------+  +---------------+  |
+---------------------------------------------------+
```

---

## 2. チャットボット vs エージェント

| 特性 | チャットボット | AIエージェント |
|------|---------------|---------------|
| ターン数 | 1ターン（Q&A） | 複数ターン（ループ） |
| ツール使用 | なし or 限定的 | 複数ツールを自律選択 |
| 計画能力 | なし | タスク分解・優先順位付け |
| 状態管理 | 会話履歴のみ | 短期/長期メモリ |
| 自律性 | ユーザー主導 | 目標駆動で自律行動 |
| エラー回復 | 不可 | 再試行・代替手段の選択 |
| 典型例 | FAQ対応 | コーディング・リサーチ |

---

## 3. エージェントの種類

### 3.1 分類体系

```
AIエージェントの分類
├── 反応型（Reactive）
│   └── 入力→即応答。内部状態なし
├── 熟慮型（Deliberative）
│   └── 計画→実行→評価のループ
├── ハイブリッド型
│   └── 反応＋熟慮の組み合わせ
├── マルチエージェント
│   └── 複数エージェントの協調
└── 自律型（Autonomous）
    └── 長時間の自律実行
```

### 3.2 種類比較表

| 種類 | 計画 | ツール | メモリ | 複雑度 | 代表例 |
|------|------|--------|--------|--------|--------|
| 反応型 | なし | 限定的 | なし | 低 | 簡易チャット |
| 熟慮型 | あり | 複数 | 短期 | 中 | ReActエージェント |
| ハイブリッド | あり | 複数 | 短期+長期 | 中-高 | LangChain Agent |
| マルチ | あり | 分散 | 共有 | 高 | CrewAI |
| 自律型 | 高度 | 広範 | 永続 | 最高 | Devin, Claude Code |

---

## 4. エージェントアーキテクチャ

### 4.1 基本ループ: Perceive-Think-Act

```python
# エージェントの基本ループ（概念コード）
class SimpleAgent:
    def __init__(self, llm, tools, memory):
        self.llm = llm
        self.tools = tools
        self.memory = memory

    def run(self, goal: str) -> str:
        """目標を受け取り、完了まで自律実行する"""
        self.memory.add("goal", goal)

        while not self.is_done():
            # 1. Perceive: 現在の状態を観察
            context = self.memory.get_context()

            # 2. Think: 次のアクションを推論
            action = self.llm.decide(context, self.tools)

            # 3. Act: ツールを実行
            if action.type == "tool_call":
                result = self.tools.execute(action)
                self.memory.add("observation", result)
            elif action.type == "final_answer":
                return action.content

        return self.memory.get_summary()
```

### 4.2 ReActパターン

```python
# ReAct (Reasoning + Acting) パターン
REACT_PROMPT = """
以下の形式で思考と行動を繰り返してください：

Thought: 現状の分析と次にすべきことの推論
Action: 使用するツール名[引数]
Observation: ツールの実行結果
... (繰り返し)
Thought: 最終的な結論
Final Answer: ユーザーへの回答
"""

class ReActAgent:
    def step(self, messages):
        response = self.llm.generate(
            system=REACT_PROMPT,
            messages=messages
        )

        if "Final Answer:" in response:
            return {"type": "answer", "content": response}

        # Action を解析してツール実行
        tool_name, args = self.parse_action(response)
        result = self.tools[tool_name](**args)

        return {"type": "observation", "content": result}
```

### 4.3 Function Calling パターン

```python
# OpenAI / Anthropic スタイルの Function Calling
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "web_search",
        "description": "Webを検索して情報を取得する",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "検索クエリ"}
            },
            "required": ["query"]
        }
    }
]

def agent_loop(user_message):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return response.content[0].text

        # ツール呼び出しを処理
        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result",
                                 "tool_use_id": block.id,
                                 "content": result}]
                })
```

### 4.4 アーキテクチャの全体図

```
+------------------------------------------------------------+
|                    Agent Runtime                            |
|                                                              |
|  +-----------+     +------------------+     +-----------+   |
|  |  Input    |---->|  Orchestrator    |---->|  Output   |   |
|  | Handler   |     |                  |     | Handler   |   |
|  +-----------+     |  +------------+  |     +-----------+   |
|                    |  | Planner    |  |                      |
|                    |  +-----+------+  |                      |
|                    |        |         |                      |
|                    |  +-----v------+  |     +-----------+   |
|                    |  | Executor   |------->| Tool      |   |
|                    |  +-----+------+  |     | Registry  |   |
|                    |        |         |     +-----------+   |
|                    |  +-----v------+  |                      |
|                    |  | Evaluator  |  |     +-----------+   |
|                    |  +------------+  |     | Memory    |   |
|                    |                  |<--->| Store     |   |
|                    +------------------+     +-----------+   |
+------------------------------------------------------------+
```

---

## 5. エージェントの構成要素

### 5.1 コンポーネント詳細

```python
# エージェントの主要コンポーネントの実装例
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Tool:
    """エージェントが使用するツール"""
    name: str
    description: str
    function: Callable
    parameters: dict

@dataclass
class Memory:
    """エージェントの記憶"""
    short_term: list = field(default_factory=list)   # 現在タスクの文脈
    long_term: dict = field(default_factory=dict)     # 永続的な知識
    working: dict = field(default_factory=dict)       # 一時的な作業領域

@dataclass
class AgentConfig:
    """エージェントの設定"""
    model: str = "claude-sonnet-4-20250514"
    max_steps: int = 20           # 最大ステップ数
    temperature: float = 0.0      # 決定論的な出力
    max_tokens: int = 4096
    tools: list = field(default_factory=list)
    system_prompt: str = ""
```

---

## 6. アンチパターン

### アンチパターン1: 無限ループエージェント

```python
# NG: 停止条件がないエージェント
class BadAgent:
    def run(self, goal):
        while True:  # 永遠に終わらない可能性
            action = self.think(goal)
            self.execute(action)

# OK: 最大ステップ数とタイムアウトを設定
class GoodAgent:
    def run(self, goal, max_steps=20, timeout=300):
        for step in range(max_steps):
            if time.time() - start > timeout:
                return self.summarize_progress()
            action = self.think(goal)
            if action.is_final:
                return action.result
            self.execute(action)
        return self.summarize_progress()  # 途中経過を返す
```

### アンチパターン2: ツール定義の曖昧さ

```python
# NG: 曖昧なツール説明
bad_tool = {
    "name": "search",
    "description": "検索する"  # 何を？ どう？
}

# OK: 具体的で明確なツール説明
good_tool = {
    "name": "web_search",
    "description": "指定されたクエリでGoogle検索を実行し、上位10件の結果（タイトル、URL、スニペット）を返す。事実確認や最新情報の取得に使用する。",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "検索クエリ（日本語 or 英語）"
            },
            "num_results": {
                "type": "integer",
                "description": "取得件数（デフォルト: 10）",
                "default": 10
            }
        },
        "required": ["query"]
    }
}
```

---

## 7. FAQ

### Q1: AIエージェントとRAGの違いは？

RAG（Retrieval-Augmented Generation）は **情報検索 + 生成** の仕組みであり、エージェントの一構成要素として使われることが多い。エージェントはRAGに加えてツール実行・計画・状態管理の能力を持つ。RAGは「知識の拡張」、エージェントは「行動の自律化」と考えるとわかりやすい。

### Q2: エージェントに適さないタスクは？

以下のタスクにはエージェントは過剰である:
- **単純なQ&A**: 1ターンで答えが出る質問
- **テンプレート的処理**: 入力→出力が固定のタスク
- **リアルタイム性が必要**: レイテンシが許容できない場合（エージェントは複数ステップを要するため遅い）

逆に、**調査・コーディング・データ分析** など試行錯誤が必要なタスクにはエージェントが有効。

### Q3: どのLLMがエージェントに最適か？

2025年時点では以下が有力:
- **Claude 3.5 Sonnet / Claude 4 系**: ツール使用が安定、コーディング能力が高い
- **GPT-4o / GPT-4 Turbo**: Function Callingの安定性
- **Gemini 1.5 Pro**: 長文コンテキスト（100万トークン）

選択基準は「ツール呼び出しの安定性」「指示追従性」「コスト」の3軸で評価する。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 定義 | 目標駆動で自律的に計画・実行するLLMシステム |
| 3要素 | 頭脳（LLM）・記憶（Memory）・手足（Tools） |
| 基本ループ | Perceive → Think → Act の反復 |
| 主要パターン | ReAct, Function Calling, Plan-and-Execute |
| 種類 | 反応型・熟慮型・ハイブリッド・マルチ・自律型 |
| 成功の鍵 | 明確なツール定義・停止条件・エラー処理 |

## 次に読むべきガイド

- [01-agent-frameworks.md](./01-agent-frameworks.md) — 主要フレームワークの詳細比較
- [02-tool-use.md](./02-tool-use.md) — ツール使用の実装パターン
- [03-memory-systems.md](./03-memory-systems.md) — メモリシステムの設計

## 参考文献

1. Anthropic, "Building effective agents" (2024) — https://docs.anthropic.com/en/docs/build-with-claude/agentic
2. Yao, S. et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2023) — https://arxiv.org/abs/2210.03629
3. Wang, L. et al., "A Survey on Large Language Model based Autonomous Agents" (2023) — https://arxiv.org/abs/2308.11432
4. LangChain Documentation, "Agents" — https://python.langchain.com/docs/concepts/agents/
