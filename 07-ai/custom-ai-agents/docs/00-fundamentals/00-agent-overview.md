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

### 1.3 エージェントの歴史と発展

AIエージェントの概念は1950年代のサイバネティクスまで遡るが、LLMベースのエージェントが実用化されたのは2023年以降である。

```
AIエージェントの発展タイムライン

1950s  サイバネティクス（フィードバックループ）
1980s  エキスパートシステム（ルールベース推論）
1990s  BDIアーキテクチャ（信念-欲求-意図モデル）
2000s  マルチエージェントシステム研究
2017   Transformer登場（Attention Is All You Need）
2022   ChatGPT発表 → LLMの実用化
2023   AutoGPT/BabyAGI → LLMエージェントブーム
       ReAct論文 → 推論+行動パターンの確立
       LangChain/LangGraph → フレームワーク成熟
2024   Claude Code, Devin → 実用レベルのコーディングエージェント
       MCP(Model Context Protocol) → ツール標準化
       マルチエージェントフレームワーク成熟
2025   Claude Agent SDK → 公式エージェント構築ツール
       企業での本番採用が加速
```

### 1.4 エージェントの構成要素詳細

```python
# エージェントの構成要素を詳細に定義する
from dataclasses import dataclass, field
from typing import Callable, Any, Protocol
from enum import Enum

class AgentCapability(Enum):
    """エージェントの能力分類"""
    REASONING = "reasoning"       # 推論能力
    PLANNING = "planning"         # 計画立案能力
    TOOL_USE = "tool_use"         # ツール使用能力
    MEMORY = "memory"             # 記憶能力
    LEARNING = "learning"         # 学習能力（メタ認知）
    COMMUNICATION = "communication"  # 通信能力（マルチエージェント用）

class ToolCategory(Enum):
    """ツールの分類"""
    INFORMATION = "information"   # 情報取得（検索、読み取り）
    COMPUTATION = "computation"   # 計算（数値計算、コード実行）
    COMMUNICATION = "communication"  # 通信（メール送信、API呼び出し）
    MANIPULATION = "manipulation"    # 操作（ファイル操作、DB操作）

@dataclass
class ToolDefinition:
    """ツールの定義"""
    name: str
    description: str
    category: ToolCategory
    parameters: dict
    function: Callable
    is_destructive: bool = False  # 破壊的操作かどうか
    requires_approval: bool = False  # 人間の承認が必要か
    rate_limit: int = 0  # 1分あたりの最大呼び出し回数（0=無制限）

@dataclass
class AgentProfile:
    """エージェントのプロファイル"""
    name: str
    role: str
    capabilities: list[AgentCapability]
    tools: list[ToolDefinition]
    model: str = "claude-sonnet-4-20250514"
    max_steps: int = 20
    temperature: float = 0.0
    system_prompt: str = ""

    def has_capability(self, cap: AgentCapability) -> bool:
        return cap in self.capabilities

    def get_tools_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        return [t for t in self.tools if t.category == category]

    def get_safe_tools(self) -> list[ToolDefinition]:
        """非破壊的なツールのみ取得"""
        return [t for t in self.tools if not t.is_destructive]
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

### 2.1 具体的な比較シナリオ

```
シナリオ: 「来週の出張の準備をして」

チャットボット:
  → "出張の準備には以下をお勧めします:
     1. 航空券の予約
     2. ホテルの予約
     3. 持ち物リストの作成
     ..."
  （情報提供のみ）

AIエージェント:
  Step 1: カレンダーを確認 → 出張日程を特定
  Step 2: 出張先と目的を確認（ユーザーに質問）
  Step 3: 航空券を検索・比較 → 最適な便を提案
  Step 4: ユーザーの承認を得て予約実行
  Step 5: 出張先のホテルを検索・予約
  Step 6: 経路・天気を調べて持ち物リスト作成
  Step 7: 社内システムで出張申請を提出
  Step 8: 完了報告
  （実際に行動を起こす）
```

### 2.2 エージェントの優位性が発揮される条件

```
エージェントが有効な条件チェックリスト:

[✓] 複数ステップが必要
[✓] 途中で判断が必要（条件分岐）
[✓] 外部データの取得が必要
[✓] 試行錯誤が発生しうる
[✓] ツール/APIの使用が必要
[✓] 中間結果に基づく次のアクションが変わる

エージェントが不要な条件:
[✗] 単純な質問応答
[✗] テンプレート的な処理
[✗] 即時応答が必須（レイテンシ制約）
[✗] 100%の正確性が必要（人間の確認なし）
```

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

### 3.3 各種類の詳細な特性

```python
# 反応型エージェントの実装例
class ReactiveAgent:
    """入力に対して即座に反応する最もシンプルなエージェント"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    def respond(self, user_input: str) -> str:
        """状態を持たず、入力に即応答"""
        # ツールが必要か判断
        tool_needed = self.llm.classify(user_input,
            categories=["direct_answer", "tool_needed"])

        if tool_needed == "direct_answer":
            return self.llm.generate(user_input)

        # 最も適切なツールを1つ選択して実行
        tool = self.llm.select_tool(user_input, self.tools)
        result = tool.execute(user_input)
        return self.llm.synthesize(user_input, result)

# 熟慮型エージェントの実装例
class DeliberativeAgent:
    """計画-実行-評価のサイクルを持つエージェント"""

    def __init__(self, llm, tools, max_iterations=10):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.scratchpad = []  # 思考の記録

    def run(self, goal: str) -> str:
        # Phase 1: 計画
        plan = self._plan(goal)

        for i in range(self.max_iterations):
            # Phase 2: 次のアクションを決定
            action = self._decide_next_action(plan, self.scratchpad)

            if action.is_final:
                return action.content

            # Phase 3: アクション実行
            result = self._execute(action)
            self.scratchpad.append({
                "thought": action.thought,
                "action": action.name,
                "result": result
            })

            # Phase 4: 計画の見直し
            if self._needs_replan(plan, self.scratchpad):
                plan = self._replan(goal, self.scratchpad)

        return self._summarize_progress()

# ハイブリッド型エージェントの実装例
class HybridAgent:
    """反応型と熟慮型を組み合わせたエージェント"""

    def __init__(self, reactive: ReactiveAgent, deliberative: DeliberativeAgent):
        self.reactive = reactive
        self.deliberative = deliberative

    def handle(self, user_input: str) -> str:
        # 入力の複雑さを評価
        complexity = self._assess_complexity(user_input)

        if complexity == "simple":
            # 単純な質問 → 反応型で即応答
            return self.reactive.respond(user_input)
        else:
            # 複雑なタスク → 熟慮型で計画的に実行
            return self.deliberative.run(user_input)

    def _assess_complexity(self, user_input: str) -> str:
        """入力の複雑さを判定"""
        # ステップ数、ツール必要性、曖昧さ等で判断
        indicators = {
            "multiple_steps": any(w in user_input for w in ["そして", "その後", "次に"]),
            "tool_needed": any(w in user_input for w in ["検索", "計算", "作成", "実行"]),
            "ambiguous": len(user_input) > 200 or "?" in user_input
        }
        complex_count = sum(indicators.values())
        return "complex" if complex_count >= 2 else "simple"
```

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

### 4.4 Plan-and-Execute パターン

```python
# Plan-and-Execute パターンの完全な実装
class PlanAndExecuteAgent:
    """先に全体計画を立ててから順に実行するパターン"""

    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, goal: str) -> str:
        # Phase 1: 計画立案
        plan = self._create_plan(goal)

        # Phase 2: 順次実行
        results = []
        for i, step in enumerate(plan):
            print(f"Step {i+1}/{len(plan)}: {step}")
            result = self._execute_step(step, results)
            results.append({"step": step, "result": result})

            # 計画の見直しが必要か確認
            if self._needs_replan(goal, plan, results):
                remaining_plan = self._replan(goal, results, plan[i+1:])
                plan = plan[:i+1] + remaining_plan

        # Phase 3: 結果の統合
        return self._synthesize(goal, results)

    def _create_plan(self, goal: str) -> list[str]:
        """目標を具体的なステップに分解"""
        response = self.planner.generate(f"""
目標: {goal}

利用可能なツール:
{self._format_tools()}

この目標を達成するための具体的なステップを列挙してください。
各ステップは1つのツール呼び出しまたは1つの思考で完結すること。

出力形式:
1. [ステップの説明]
2. [ステップの説明]
...
""")
        return self._parse_steps(response)

    def _execute_step(self, step: str, previous_results: list) -> str:
        """個別のステップを実行"""
        context = "\n".join([
            f"- {r['step']}: {r['result'][:200]}"
            for r in previous_results[-3:]  # 直近3件の結果のみ
        ])

        response = self.executor.generate(f"""
実行するステップ: {step}

これまでの結果:
{context}

利用可能なツール:
{self._format_tools()}

このステップを実行してください。
""")
        return response
```

### 4.5 アーキテクチャの全体図

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

### 4.6 アーキテクチャパターンの比較表

| パターン | 計画性 | 柔軟性 | 実装難度 | 適用場面 |
|----------|--------|--------|---------|---------|
| ReAct | 低（逐次的） | 高 | 低 | 汎用タスク |
| Function Calling | 低（逐次的） | 高 | 低 | API連携 |
| Plan-and-Execute | 高（事前計画） | 中 | 中 | 構造化タスク |
| Tree of Thoughts | 最高（探索的） | 最高 | 高 | 複雑な推論 |
| Reflexion | 中（振り返り） | 高 | 中 | 品質重視タスク |

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

### 5.2 プロンプトエンジニアリングの原則

エージェントのシステムプロンプトは通常のチャットとは異なり、行動規範を明確に定義する必要がある。

```python
# エージェント向けシステムプロンプトの設計パターン
class AgentPromptBuilder:
    """エージェント用のシステムプロンプトを構造的に構築する"""

    @staticmethod
    def build(role: str, tools: list, constraints: list,
              examples: list = None) -> str:
        prompt_parts = []

        # 1. 役割定義
        prompt_parts.append(f"## 役割\nあなたは{role}です。")

        # 2. 利用可能なツール
        tool_descriptions = "\n".join([
            f"- **{t.name}**: {t.description}" for t in tools
        ])
        prompt_parts.append(f"## 利用可能なツール\n{tool_descriptions}")

        # 3. 行動規範
        constraint_list = "\n".join([f"- {c}" for c in constraints])
        prompt_parts.append(f"## 行動規範\n{constraint_list}")

        # 4. 思考プロセス
        prompt_parts.append("""## 思考プロセス
1. まず目標を明確化する
2. 必要な情報を特定する
3. 最も効率的なツールを選択する
4. 実行結果を評価する
5. 目標が達成されたか判断する
6. 未達成なら次のアクションを計画する""")

        # 5. 出力形式
        prompt_parts.append("""## 出力形式
- 作業の意図を簡潔に説明してからツールを使用する
- 結果を分析して次のステップを判断する
- 最終回答は構造化して提供する""")

        # 6. 例示（Few-shot）
        if examples:
            example_text = "\n\n".join([
                f"### 例 {i+1}\n入力: {e['input']}\n出力: {e['output']}"
                for i, e in enumerate(examples)
            ])
            prompt_parts.append(f"## 例\n{example_text}")

        return "\n\n".join(prompt_parts)

# 使用例
system_prompt = AgentPromptBuilder.build(
    role="シニアソフトウェアエンジニア",
    tools=[read_file_tool, write_file_tool, run_tests_tool],
    constraints=[
        "コードを変更する前に必ず既存のコードを読んで理解する",
        "テストを書いてから実装する（TDD）",
        "破壊的な変更の前にユーザーの確認を求める",
        "エラーが発生したら原因を特定してから修正する"
    ]
)
```

### 5.3 ツール設計のベストプラクティス

```python
# ツール設計のガイドライン
class ToolDesignGuidelines:
    """
    良いツール設計の原則:

    1. 単一責任の原則: 1ツール = 1機能
    2. 明確な命名: 動詞_名詞 形式（search_web, read_file）
    3. 詳細な説明: 何をするか + いつ使うか + 入出力
    4. 適切な粒度: 粗すぎず細かすぎず
    5. エラーハンドリング: 失敗時の情報を充実させる
    6. 冪等性: 可能な限り同じ入力→同じ出力
    """

    @staticmethod
    def validate_tool_definition(tool: dict) -> list[str]:
        """ツール定義の品質をチェック"""
        issues = []

        # 名前のチェック
        if not tool.get("name"):
            issues.append("名前が未定義")
        elif "_" not in tool["name"]:
            issues.append("名前は動詞_名詞形式が推奨（例: search_web）")

        # 説明のチェック
        desc = tool.get("description", "")
        if len(desc) < 20:
            issues.append("説明が短すぎます（最低20文字）")
        if "使用" not in desc and "use" not in desc.lower():
            issues.append("いつ使うかの説明が推奨されます")

        # パラメータのチェック
        schema = tool.get("input_schema", {})
        props = schema.get("properties", {})
        for param_name, param_def in props.items():
            if "description" not in param_def:
                issues.append(f"パラメータ '{param_name}' に説明がありません")

        return issues
```

---

## 6. エージェントの実装パターン詳細

### 6.1 イベント駆動エージェント

```python
# イベント駆動型のエージェント実装
from typing import Protocol
from dataclasses import dataclass
import asyncio

class EventHandler(Protocol):
    async def handle(self, event: dict) -> dict: ...

@dataclass
class AgentEvent:
    type: str  # "user_input", "tool_result", "error", "timeout"
    data: dict
    timestamp: float

class EventDrivenAgent:
    """イベント駆動型のエージェント"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.event_queue = asyncio.Queue()
        self.handlers: dict[str, EventHandler] = {}
        self.state = {"status": "idle", "history": []}

    def register_handler(self, event_type: str, handler: EventHandler):
        self.handlers[event_type] = handler

    async def run(self):
        """イベントループ"""
        while True:
            event = await self.event_queue.get()
            handler = self.handlers.get(event.type)

            if handler:
                try:
                    result = await handler.handle(event.data)
                    self.state["history"].append({
                        "event": event.type,
                        "result": result
                    })
                except Exception as e:
                    await self.event_queue.put(AgentEvent(
                        type="error",
                        data={"error": str(e), "original_event": event},
                        timestamp=time.time()
                    ))

    async def submit(self, event_type: str, data: dict):
        """イベントを投入"""
        await self.event_queue.put(AgentEvent(
            type=event_type,
            data=data,
            timestamp=time.time()
        ))
```

### 6.2 ストリーミングエージェント

```python
# ストリーミング対応のエージェント
import anthropic
from typing import AsyncGenerator

class StreamingAgent:
    """リアルタイムにトークンをストリーミングするエージェント"""

    def __init__(self):
        self.client = anthropic.Anthropic()

    async def run_streaming(self, user_message: str,
                            tools: list) -> AsyncGenerator[dict, None]:
        """ストリーミングでエージェントの出力を返す"""
        messages = [{"role": "user", "content": user_message}]

        while True:
            with self.client.messages.stream(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                tools=tools,
                messages=messages
            ) as stream:
                current_text = ""
                for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            current_text += event.delta.text
                            yield {
                                "type": "text_delta",
                                "text": event.delta.text
                            }

                response = stream.get_final_message()

            if response.stop_reason == "end_turn":
                yield {"type": "complete", "text": current_text}
                return

            # ツール呼び出し処理
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    yield {
                        "type": "tool_call",
                        "tool": block.name,
                        "input": block.input
                    }
                    result = self._execute_tool(block.name, block.input)
                    yield {
                        "type": "tool_result",
                        "tool": block.name,
                        "result": str(result)[:200]
                    }
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
```

### 6.3 コンテキスト管理パターン

```python
# 効率的なコンテキスト管理
class ContextManager:
    """エージェントのコンテキストウィンドウを効率的に管理する"""

    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        self.messages: list[dict] = []
        self.system_prompt: str = ""
        self.pinned_context: list[str] = []  # 常に含めるコンテキスト

    def estimate_tokens(self, text: str) -> int:
        """トークン数を概算（1文字≒1.5トークンで日本語を概算）"""
        return int(len(text) * 1.5)

    def get_current_tokens(self) -> int:
        total = self.estimate_tokens(self.system_prompt)
        total += sum(self.estimate_tokens(str(m)) for m in self.messages)
        total += sum(self.estimate_tokens(c) for c in self.pinned_context)
        return total

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()

    def _trim_if_needed(self):
        """コンテキストが上限を超えたら古いメッセージを要約"""
        while self.get_current_tokens() > self.max_tokens * 0.8:
            if len(self.messages) <= 4:
                break  # 最低限のメッセージは保持

            # 古いメッセージを要約して圧縮
            old_messages = self.messages[:len(self.messages)//2]
            summary = self._summarize(old_messages)

            self.messages = [
                {"role": "system", "content": f"これまでの会話の要約: {summary}"}
            ] + self.messages[len(self.messages)//2:]

    def _summarize(self, messages: list) -> str:
        """メッセージ群を要約"""
        content = "\n".join([f"{m['role']}: {str(m['content'])[:200]}" for m in messages])
        return f"[要約] {content[:500]}"

    def pin_context(self, context: str):
        """常に含めるコンテキストを追加"""
        self.pinned_context.append(context)

    def get_messages_for_api(self) -> list:
        """API呼び出し用のメッセージリストを取得"""
        result = []
        if self.pinned_context:
            result.append({
                "role": "user",
                "content": "参考情報:\n" + "\n".join(self.pinned_context)
            })
        result.extend(self.messages)
        return result
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

### アンチパターン3: コンテキスト爆発

```python
# NG: ツール結果を全て保持してコンテキストが爆発
class ContextExplosionAgent:
    def run(self, goal):
        messages = [{"role": "user", "content": goal}]
        for _ in range(100):
            response = llm.generate(messages=messages)
            tool_result = execute(response)
            # 巨大な結果がそのまま蓄積
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": tool_result})
        # → 数万トークンに膨張、コスト爆発

# OK: コンテキストマネージャーで管理
class ManagedAgent:
    def __init__(self):
        self.context_manager = ContextManager(max_tokens=50000)

    def run(self, goal):
        self.context_manager.add_message("user", goal)
        for _ in range(20):
            messages = self.context_manager.get_messages_for_api()
            response = llm.generate(messages=messages)

            # 結果を要約してから追加
            result = execute(response)
            summarized = summarize_if_large(result, max_chars=2000)
            self.context_manager.add_message("assistant", str(response))
            self.context_manager.add_message("user", summarized)
```

### アンチパターン4: エラーハンドリングの欠如

```python
# NG: エラー時にクラッシュ
class FragileAgent:
    def run(self, goal):
        result = self.tools["search"](goal)  # ツールがない → KeyError
        return result  # ネットワークエラー → 未処理例外

# OK: 多層的なエラーハンドリング
class RobustAgent:
    def run(self, goal):
        try:
            for step in range(self.max_steps):
                action = self.decide_action(goal)

                if action.tool not in self.tools:
                    self.report_error(f"ツール '{action.tool}' は利用できません")
                    continue

                try:
                    result = self.tools[action.tool](**action.args)
                except TimeoutError:
                    result = "タイムアウト。再試行してください。"
                except Exception as e:
                    result = f"エラー: {type(e).__name__}: {e}"

                self.memory.add(action, result)

        except Exception as e:
            return f"エージェントでエラーが発生しました: {e}\n部分的な結果: {self.get_partial_results()}"
```

---

## 7. パフォーマンス最適化

### 7.1 レイテンシ最適化

```
エージェントのレイテンシ構成

典型的な1ステップ:
  [LLM推論]     1-5秒    ████████████████████
  [ツール実行]   0.1-2秒   ████
  [結果処理]     0.01秒    █

5ステップのタスク:
  合計: 5-35秒

最適化レバー:
1. モデル選択: Haiku(高速) vs Sonnet(バランス) vs Opus(高品質)
2. 並列ツール呼び出し: 独立したツールは同時実行
3. ストリーミング: 部分結果を即座に返す
4. キャッシュ: 同じ入力に対する結果をキャッシュ
5. プロンプト最適化: 不要なコンテキストを削減
```

```python
# パフォーマンス最適化の実装例
import asyncio
import hashlib
from functools import lru_cache

class OptimizedAgent:
    def __init__(self):
        self.cache = {}
        self.metrics = {"total_time": 0, "cache_hits": 0, "api_calls": 0}

    async def run_optimized(self, goal: str) -> str:
        """最適化されたエージェントループ"""
        start = time.time()
        messages = [{"role": "user", "content": goal}]

        for step in range(self.max_steps):
            # キャッシュチェック
            cache_key = self._make_cache_key(messages)
            if cache_key in self.cache:
                self.metrics["cache_hits"] += 1
                response = self.cache[cache_key]
            else:
                self.metrics["api_calls"] += 1
                response = await self._call_llm_async(messages)
                self.cache[cache_key] = response

            if response.stop_reason == "end_turn":
                self.metrics["total_time"] = time.time() - start
                return self._extract_text(response)

            # 並列ツール実行
            tool_calls = [b for b in response.content if b.type == "tool_use"]
            if len(tool_calls) > 1:
                results = await self._execute_tools_parallel(tool_calls)
            else:
                results = [await self._execute_tool_async(tool_calls[0])]

            # メッセージ更新
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": results})

        self.metrics["total_time"] = time.time() - start
        return "最大ステップ数到達"

    async def _execute_tools_parallel(self, tool_calls):
        """複数のツール呼び出しを並列に実行"""
        tasks = [
            self._execute_tool_async(tc) for tc in tool_calls
        ]
        return await asyncio.gather(*tasks)

    def _make_cache_key(self, messages: list) -> str:
        content = str(messages[-3:])  # 直近3メッセージでキーを生成
        return hashlib.md5(content.encode()).hexdigest()

    def get_performance_report(self) -> str:
        return (
            f"総実行時間: {self.metrics['total_time']:.1f}秒\n"
            f"API呼び出し: {self.metrics['api_calls']}回\n"
            f"キャッシュヒット: {self.metrics['cache_hits']}回\n"
            f"キャッシュ率: {self.metrics['cache_hits']/(self.metrics['api_calls']+self.metrics['cache_hits'])*100:.0f}%"
        )
```

### 7.2 コスト最適化

```python
# コスト最適化のためのモデルルーティング
class CostOptimizedAgent:
    """タスクの複雑さに応じてモデルを使い分ける"""

    MODELS = {
        "fast": "claude-haiku-4-20250514",     # 分類、ルーティング用
        "balanced": "claude-sonnet-4-20250514", # 一般的なタスク
        "powerful": "claude-opus-4-20250514",   # 複雑な推論
    }

    def select_model(self, task_type: str, complexity: str) -> str:
        """タスクとその複雑さに応じたモデル選択"""
        model_map = {
            ("classification", "any"): "fast",
            ("routing", "any"): "fast",
            ("generation", "low"): "balanced",
            ("generation", "high"): "powerful",
            ("reasoning", "low"): "balanced",
            ("reasoning", "high"): "powerful",
            ("coding", "any"): "balanced",
        }

        # 複雑さに関わらないタスク
        key = (task_type, "any")
        if key in model_map:
            return self.MODELS[model_map[key]]

        # 複雑さを考慮するタスク
        key = (task_type, complexity)
        tier = model_map.get(key, "balanced")
        return self.MODELS[tier]
```

---

## 8. トラブルシューティングガイド

### 8.1 よくある問題と解決策

| 症状 | 原因 | 解決策 |
|------|------|--------|
| 同じツールを何度も呼ぶ | ループ検出がない | 直近N回の呼び出しパターンを監視 |
| 途中で止まる | コンテキスト超過 | メッセージの圧縮・要約を実装 |
| 的外れなツール選択 | ツール説明が不明確 | 説明文の改善、使用例の追加 |
| コストが高すぎる | 不要なステップが多い | max_stepsの調整、モデルルーティング |
| 最終回答の品質が低い | コンテキスト汚染 | 関連情報のみを含めるフィルタリング |
| タイムアウト | ツール実行が遅い | タイムアウト設定、非同期実行 |

### 8.2 デバッグ手法

```python
# エージェントのデバッグツール
class AgentDebugger:
    """エージェントの実行をトレース・分析するデバッグツール"""

    def __init__(self):
        self.trace = []

    def log_step(self, step_num: int, thought: str, action: str,
                 result: str, tokens_used: int):
        self.trace.append({
            "step": step_num,
            "thought": thought[:200],
            "action": action,
            "result": result[:200],
            "tokens": tokens_used,
            "timestamp": time.time()
        })

    def print_trace(self):
        """実行トレースを可視化"""
        print("=" * 60)
        print("エージェント実行トレース")
        print("=" * 60)

        total_tokens = 0
        for entry in self.trace:
            print(f"\n--- Step {entry['step']} ---")
            print(f"  思考: {entry['thought']}")
            print(f"  行動: {entry['action']}")
            print(f"  結果: {entry['result']}")
            print(f"  トークン: {entry['tokens']:,}")
            total_tokens += entry['tokens']

        print(f"\n{'=' * 60}")
        print(f"合計ステップ: {len(self.trace)}")
        print(f"合計トークン: {total_tokens:,}")
        print(f"推定コスト: ${total_tokens * 3 / 1_000_000:.4f}")

    def detect_loops(self) -> list[str]:
        """ループパターンを検出"""
        issues = []
        actions = [t["action"] for t in self.trace]

        # 同じアクションの連続
        for i in range(len(actions) - 2):
            if actions[i] == actions[i+1] == actions[i+2]:
                issues.append(
                    f"Step {i}-{i+2}: '{actions[i]}' が3回連続呼び出し"
                )

        # 合計呼び出し回数
        from collections import Counter
        counts = Counter(actions)
        for action, count in counts.most_common(3):
            if count > 5:
                issues.append(
                    f"'{action}' が{count}回呼び出し（過剰の可能性）"
                )

        return issues
```

---

## 9. 設計チェックリスト

エージェントを設計する際の確認項目:

```
[ ] 目標の明確化
    [ ] タスクの範囲が定義されている
    [ ] 成功基準が明確である
    [ ] 想定ステップ数が見積もられている

[ ] ツール設計
    [ ] 各ツールの説明が具体的で明確
    [ ] 入力パラメータに制約が設定されている
    [ ] エラーレスポンスが構造化されている
    [ ] 破壊的操作にはガードレールがある

[ ] メモリ設計
    [ ] 短期記憶の容量制限がある
    [ ] コンテキスト圧縮の仕組みがある
    [ ] 必要に応じて長期記憶を利用

[ ] 安全性
    [ ] 最大ステップ数が設定されている
    [ ] タイムアウトが設定されている
    [ ] コスト上限が設定されている
    [ ] 破壊的操作の前に確認を求める

[ ] エラー処理
    [ ] ツール実行の再試行ロジックがある
    [ ] 代替手段のフォールバックがある
    [ ] 部分的な結果を返す仕組みがある

[ ] 監視
    [ ] 各ステップのログが記録される
    [ ] トークン使用量が追跡される
    [ ] 異常パターンの検出がある
```

---

## 10. FAQ

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

### Q4: エージェントの実行コストはどの程度か？

タスクの複雑さとモデルによるが、目安は以下の通り:

| タスク種別 | ステップ数 | 推定コスト（Claude Sonnet） |
|-----------|-----------|--------------------------|
| 単純な検索+回答 | 2-3 | $0.01-0.05 |
| ファイル操作 | 5-10 | $0.05-0.20 |
| コーディング | 10-20 | $0.20-1.00 |
| 複雑なリサーチ | 20-50 | $1.00-5.00 |
| プロジェクト全体 | 50-200 | $5.00-50.00 |

### Q5: エージェントの品質をどう保証するか？

3つのレイヤーで品質を保証する:

1. **設計時**: ツール定義の品質チェック、プロンプトのテスト
2. **実行時**: ガードレール、ループ検出、コスト制限
3. **事後評価**: 成功率の測定、LLM-as-Judge、人間レビュー

### Q6: マルチモーダルエージェントとは？

テキストだけでなく画像・音声・動画を入出力として扱えるエージェント。例えば:
- スクリーンショットを見て操作を行うUI操作エージェント
- 図面を読み取って設計レビューを行うエージェント
- 音声入力で指示を受けて作業するハンズフリーエージェント

Claude 3.5以降はマルチモーダル対応しており、画像入力を含むエージェントの構築が可能。

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
5. Shinn, N. et al., "Reflexion: Language Agents with Verbal Reinforcement Learning" (2023) — https://arxiv.org/abs/2303.11366
6. Wei, J. et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022) — https://arxiv.org/abs/2201.11903
7. Yao, S. et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (2023) — https://arxiv.org/abs/2305.10601
