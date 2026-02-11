# ワークフローエージェント

> DAG・条件分岐・並列実行――事前定義されたフローに従いつつ、LLMの判断で動的に経路を選択するワークフローエージェントの設計と実装。

## この章で学ぶこと

1. ワークフローエージェントとフリーフォームエージェントの違いと使い分け
2. DAG（有向非巡回グラフ）ベースのフロー設計と条件分岐の実装
3. LangGraphを用いた状態管理付きワークフローの構築パターン

---

## 1. ワークフローエージェントとは

### 1.1 フリーフォーム vs ワークフロー

```
フリーフォームエージェント:
  目標 → [LLMが自由に判断] → ... → 結果
  ・LLMが毎ステップ何をするか決定
  ・柔軟だが予測困難

ワークフローエージェント:
  目標 → [ノード1] → [条件] → [ノード2a or 2b] → [ノード3] → 結果
  ・事前定義されたフローに従う
  ・各ノードでLLMが処理を実行
  ・予測可能で制御しやすい
```

### 1.2 位置づけ

```
制御の度合いスペクトラム

 完全手動                                    完全自律
 +--------+-----------+-------------+--------+
 | 固定   | ワーク    | シングル     | 自律   |
 | パイプ | フロー    | エージェント | エージ |
 | ライン | エージェント| (ReAct)    | ェント |
 +--------+-----------+-------------+--------+
           ^^^^^^^^^^^^
           この章の範囲

 フロー: 開発者が設計
 ノード内処理: LLMが実行
```

---

## 2. DAGベースのフロー設計

### 2.1 DAGとは

```
DAG (Directed Acyclic Graph) 例: ドキュメント処理パイプライン

  [入力受付] ──→ [言語判定] ──→ [翻訳?] ──→ [要約] ──→ [出力]
                     |              ^
                     |  日本語       | 英語
                     +──(不要)──→───+
                     |
                     v 他言語
                  [翻訳実行] ─────→──+

※ サイクル（ループ）がない = 必ず終了する
```

### 2.2 基本的なワークフロー実装

```python
# DAGベースのワークフローエンジン
from dataclasses import dataclass, field
from typing import Callable, Any
from enum import Enum

class NodeType(Enum):
    LLM = "llm"           # LLM処理ノード
    TOOL = "tool"          # ツール実行ノード
    CONDITION = "condition" # 条件分岐ノード
    PARALLEL = "parallel"   # 並列実行ノード

@dataclass
class WorkflowNode:
    name: str
    type: NodeType
    handler: Callable
    next_nodes: list[str] = field(default_factory=list)
    condition: Callable = None  # 条件分岐用

class WorkflowEngine:
    def __init__(self):
        self.nodes: dict[str, WorkflowNode] = {}
        self.state: dict[str, Any] = {}

    def add_node(self, node: WorkflowNode):
        self.nodes[node.name] = node

    def run(self, start_node: str, initial_state: dict) -> dict:
        self.state = initial_state
        current = start_node

        while current:
            node = self.nodes[current]
            print(f"実行中: {node.name}")

            if node.type == NodeType.CONDITION:
                # 条件分岐: conditionの結果で次のノードを決定
                branch = node.condition(self.state)
                current = branch
            elif node.type == NodeType.PARALLEL:
                # 並列実行
                results = self._run_parallel(node.next_nodes)
                self.state["parallel_results"] = results
                current = node.next_nodes[-1] if node.next_nodes else None
            else:
                # LLM/ツール処理
                result = node.handler(self.state)
                self.state[f"{node.name}_result"] = result
                current = node.next_nodes[0] if node.next_nodes else None

        return self.state
```

### 2.3 条件分岐の実装

```python
# 条件分岐を含むワークフロー
def build_support_workflow():
    engine = WorkflowEngine()

    # ノード1: 問い合わせ分類
    engine.add_node(WorkflowNode(
        name="classify",
        type=NodeType.LLM,
        handler=lambda state: classify_inquiry(state["user_message"]),
        next_nodes=["route"]
    ))

    # ノード2: ルーティング（条件分岐）
    engine.add_node(WorkflowNode(
        name="route",
        type=NodeType.CONDITION,
        handler=None,
        condition=lambda state: {
            "billing": "handle_billing",
            "technical": "handle_technical",
            "general": "handle_general"
        }.get(state["classify_result"], "handle_general")
    ))

    # ノード3a: 請求対応
    engine.add_node(WorkflowNode(
        name="handle_billing",
        type=NodeType.LLM,
        handler=lambda state: handle_billing_inquiry(state),
        next_nodes=["respond"]
    ))

    # ノード3b: 技術対応
    engine.add_node(WorkflowNode(
        name="handle_technical",
        type=NodeType.LLM,
        handler=lambda state: handle_technical_inquiry(state),
        next_nodes=["respond"]
    ))

    # ノード3c: 一般対応
    engine.add_node(WorkflowNode(
        name="handle_general",
        type=NodeType.LLM,
        handler=lambda state: handle_general_inquiry(state),
        next_nodes=["respond"]
    ))

    # ノード4: 回答生成
    engine.add_node(WorkflowNode(
        name="respond",
        type=NodeType.LLM,
        handler=lambda state: generate_response(state),
        next_nodes=[]
    ))

    return engine

# 実行
workflow = build_support_workflow()
result = workflow.run("classify", {
    "user_message": "先月の請求額が間違っています"
})
```

---

## 3. LangGraphによるワークフロー

### 3.1 LangGraphの状態グラフ

```
LangGraph の状態グラフモデル

+--------+     +--------+     +----------+     +--------+
| START  |---->| Node A |---->| Condition|---->| Node B |
+--------+     +--------+     +-----+----+     +--------+
                                     |               |
                                     v               v
                               +--------+       +--------+
                               | Node C |------>|  END   |
                               +--------+       +--------+

各ノードは State を受け取り、更新された State を返す
条件エッジは State の内容に基づいて次のノードを決定
```

### 3.2 LangGraph実装例

```python
# LangGraphによるワークフローエージェント
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

# 状態の型定義
class AgentState(TypedDict):
    messages: list
    current_step: str
    classification: str
    draft: str
    review_result: str
    final_output: str

# ノード関数
def classify_request(state: AgentState) -> AgentState:
    """リクエストを分類する"""
    messages = state["messages"]
    result = llm.invoke(
        f"以下のリクエストを 'simple' / 'complex' に分類:\n{messages[-1]}"
    )
    return {"classification": result.content.strip()}

def handle_simple(state: AgentState) -> AgentState:
    """簡単なリクエストを処理"""
    draft = llm.invoke(f"簡潔に回答: {state['messages'][-1]}")
    return {"draft": draft.content}

def handle_complex(state: AgentState) -> AgentState:
    """複雑なリクエストを処理"""
    draft = llm.invoke(f"詳細に回答: {state['messages'][-1]}")
    return {"draft": draft.content}

def review(state: AgentState) -> AgentState:
    """回答をレビュー"""
    result = llm.invoke(
        f"回答の品質を評価。PASS/FAIL:\n{state['draft']}"
    )
    return {"review_result": result.content.strip()}

def finalize(state: AgentState) -> AgentState:
    """最終出力を生成"""
    return {"final_output": state["draft"]}

# 条件分岐関数
def route_by_classification(state: AgentState) -> Literal["simple", "complex"]:
    return "simple" if "simple" in state["classification"] else "complex"

def route_by_review(state: AgentState) -> Literal["revise", "finalize"]:
    return "finalize" if "PASS" in state["review_result"] else "revise"

# グラフ構築
workflow = StateGraph(AgentState)

# ノード追加
workflow.add_node("classify", classify_request)
workflow.add_node("handle_simple", handle_simple)
workflow.add_node("handle_complex", handle_complex)
workflow.add_node("review", review)
workflow.add_node("finalize", finalize)

# エッジ追加
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route_by_classification, {
    "simple": "handle_simple",
    "complex": "handle_complex"
})
workflow.add_edge("handle_simple", "review")
workflow.add_edge("handle_complex", "review")
workflow.add_conditional_edges("review", route_by_review, {
    "finalize": "finalize",
    "revise": "handle_complex"  # やり直し
})
workflow.add_edge("finalize", END)

# コンパイル・実行
app = workflow.compile()
result = app.invoke({
    "messages": ["マイクロサービスのベストプラクティスを教えて"],
    "current_step": "", "classification": "", "draft": "",
    "review_result": "", "final_output": ""
})
```

---

## 4. 並列実行パターン

```python
# 並列ノードの実装
import asyncio

class ParallelWorkflow:
    async def run_parallel_nodes(self, nodes: list, state: dict) -> dict:
        """複数のノードを並列に実行"""
        tasks = [
            asyncio.create_task(node.handler(state))
            for node in nodes
        ]
        results = await asyncio.gather(*tasks)

        for node, result in zip(nodes, results):
            state[f"{node.name}_result"] = result

        return state

# 使用例: 複数情報源からの並列収集
async def parallel_research(state):
    """複数ソースから並列にデータ収集"""
    tasks = [
        search_academic_papers(state["query"]),
        search_news_articles(state["query"]),
        search_github_repos(state["query"])
    ]
    papers, news, repos = await asyncio.gather(*tasks)
    return {
        "papers": papers,
        "news": news,
        "repos": repos
    }
```

---

## 5. ワークフローパターン比較

### 5.1 フロー形状別比較

| パターン | 形状 | 特徴 | 適用場面 |
|----------|------|------|---------|
| 直列 | A→B→C | 最もシンプル | パイプライン処理 |
| 条件分岐 | A→B or C | 入力に応じた経路 | ルーティング |
| 並列 | A→[B,C]→D | 独立タスクの同時実行 | データ収集 |
| ループ | A→B→A(条件付き) | 品質基準まで繰り返し | レビュー/改善 |
| サブワークフロー | A→[Sub]→B | 再利用可能な子フロー | 共通処理の部品化 |

### 5.2 ワークフロー vs 自律エージェント

| 観点 | ワークフロー | 自律エージェント |
|------|------------|----------------|
| 制御性 | 高（予測可能） | 低（非決定的） |
| 柔軟性 | 中（設計済みパス内） | 高（任意の行動可能） |
| デバッグ | 容易（各ノード個別） | 困難（自由行動） |
| コスト | 予測可能 | 予測困難 |
| 設計コスト | 高（事前設計必要） | 低（プロンプトのみ） |
| 信頼性 | 高 | 中 |
| 適用場面 | 業務プロセス自動化 | 探索的タスク |

---

## 6. 状態管理

```
状態管理のパターン

1. 受け渡し方式
   [Node A] --state--> [Node B] --state--> [Node C]
   各ノードが state を受け取り、更新して次に渡す

2. 中央集権方式
   [Node A] --update-->                <--read-- [Node B]
                        [State Store]
   [Node C] --update-->                <--read-- [Node D]

3. イベント方式
   [Node A] --event--> [Event Bus] --notify--> [Node B]
                                   --notify--> [Node C]
```

```python
# Pydanticを使った型安全な状態管理
from pydantic import BaseModel, Field
from typing import Optional

class WorkflowState(BaseModel):
    """ワークフローの状態（型安全）"""
    user_input: str
    step: int = 0
    classification: Optional[str] = None
    intermediate_results: list[str] = Field(default_factory=list)
    final_output: Optional[str] = None
    errors: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def advance(self) -> "WorkflowState":
        """ステップを進める"""
        return self.model_copy(update={"step": self.step + 1})

    def add_result(self, result: str) -> "WorkflowState":
        """中間結果を追加"""
        new_results = self.intermediate_results + [result]
        return self.model_copy(update={"intermediate_results": new_results})
```

---

## 7. アンチパターン

### アンチパターン1: 過度に複雑なDAG

```
# NG: 20ノード以上のモノリシックなDAG
[A]→[B]→[C]→[D]→[E]→[F]→[G]→[H]→[I]→[J]→...
 ↓   ↓   ↓   ↓   ↓   ↓   ↓
[K] [L] [M] [N] [O] [P] [Q]
 理解不能、メンテナンス不能

# OK: サブワークフローに分割
[メインフロー]
  [受付] → [サブフロー:分析] → [サブフロー:処理] → [出力]

各サブフローは5ノード以下で独立してテスト可能
```

### アンチパターン2: 状態の暗黙的依存

```python
# NG: グローバル変数で状態を共有
global_state = {}  # どのノードが何を書き込んだか追跡不能

def node_a(state):
    global_state["temp"] = "value"  # 副作用!

# OK: 明示的な状態の受け渡し
def node_a(state: WorkflowState) -> WorkflowState:
    return state.model_copy(update={"classification": "technical"})
```

---

## 8. FAQ

### Q1: ワークフローにサイクル（ループ）を含めてよいか？

含めてよい。ただし「DAG」は定義上サイクルを含まないため、サイクルがある場合は「状態グラフ」と呼ぶ。LangGraphはサイクルを明示的にサポートしている。重要なのは **最大反復回数の制限** を必ず設けること。

### Q2: ワークフローの各ノードに異なるLLMを使ってよいか？

推奨される。例えば:
- **分類ノード**: 高速・安価なモデル（Claude Haiku）
- **生成ノード**: 高品質なモデル（Claude Sonnet）
- **レビューノード**: 最高品質のモデル（Claude Opus）

コストと品質のバランスを各ノードで最適化できるのがワークフローの利点。

### Q3: エラーが発生したノードの再実行は？

**チェックポイント方式** を推奨する。各ノードの完了時に状態を永続化し、失敗時はその時点から再開する。LangGraphにはチェックポイント機能が組み込まれている。

---

## まとめ

| 項目 | 内容 |
|------|------|
| ワークフロー | 事前定義されたフローに従うエージェント |
| DAG | ノード+エッジの有向非巡回グラフ |
| 条件分岐 | LLMの判断で経路を選択 |
| 並列実行 | 独立ノードを同時に処理 |
| 状態管理 | 型安全な状態オブジェクトの受け渡し |
| 原則 | 制御性と柔軟性のバランスを取る |

## 次に読むべきガイド

- [03-autonomous-agents.md](./03-autonomous-agents.md) — 自律エージェントの設計
- [../02-implementation/01-langgraph.md](../02-implementation/01-langgraph.md) — LangGraphの詳細実装
- [../03-applications/02-customer-support.md](../03-applications/02-customer-support.md) — サポートワークフローの実例

## 参考文献

1. LangGraph Documentation — https://langchain-ai.github.io/langgraph/
2. Anthropic, "Building effective agents - Workflows" (2024) — https://docs.anthropic.com/en/docs/build-with-claude/agentic
3. AWS, "Step Functions" — https://docs.aws.amazon.com/step-functions/
