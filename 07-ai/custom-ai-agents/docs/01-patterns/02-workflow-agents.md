# ワークフローエージェント

> DAG・条件分岐・並列実行――事前定義されたフローに従いつつ、LLMの判断で動的に経路を選択するワークフローエージェントの設計と実装。

## この章で学ぶこと

1. ワークフローエージェントとフリーフォームエージェントの違いと使い分け
2. DAG（有向非巡回グラフ）ベースのフロー設計と条件分岐の実装
3. LangGraphを用いた状態管理付きワークフローの構築パターン
4. 並列実行・サブワークフロー・動的フロー生成の実践手法
5. エラーハンドリング・チェックポイント・リトライ戦略
6. 本番運用のためのモニタリング・コスト最適化・テスト手法

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

### 1.3 ワークフローエージェントの判断フローチャート

```
ワークフローエージェントを選ぶべきか？

Q1: タスクの手順は事前に定義できるか？
├─ YES → Q2へ
└─ NO  → フリーフォームエージェント（ReAct/自律型）を検討

Q2: 各ステップでLLMの判断が必要か？
├─ YES → Q3へ
└─ NO  → 固定パイプライン（LLM不要）で十分

Q3: 実行パスの分岐は予測可能か？
├─ YES → ワークフローエージェントが最適
└─ NO  → ハイブリッド（ワークフロー＋自律ノード）を検討

Q4: 並列実行で高速化できるステップがあるか？
├─ YES → 並列ワークフローパターンを採用
└─ NO  → 直列ワークフローで十分

Q5: 処理の再現性・監査が必要か？
├─ YES → ワークフロー＋チェックポイントが必須
└─ NO  → シンプルなワークフローで開始
```

### 1.4 典型的なユースケース

```
ユースケース別ワークフロー適合度

高い適合度:
  ・カスタマーサポートの問い合わせ処理
  ・コンテンツ生成パイプライン（記事作成→レビュー→公開）
  ・ドキュメント処理（解析→分類→要約→格納）
  ・コードレビュー自動化（解析→問題検出→修正提案→適用）
  ・データETLパイプライン（抽出→変換→品質チェック→ロード）

中程度:
  ・リサーチアシスタント（情報収集→分析→レポート）
  ・メール/メッセージの自動返信
  ・レポート自動生成

低い適合度:
  ・自由対話型チャットボット
  ・探索的なデータ分析
  ・創造的なブレインストーミング
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
import time
import logging

logger = logging.getLogger(__name__)

class NodeType(Enum):
    LLM = "llm"           # LLM処理ノード
    TOOL = "tool"          # ツール実行ノード
    CONDITION = "condition" # 条件分岐ノード
    PARALLEL = "parallel"   # 並列実行ノード
    SUBWORKFLOW = "subworkflow"  # サブワークフローノード

class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class NodeExecution:
    """ノード実行の記録"""
    node_name: str
    status: NodeStatus
    start_time: float
    end_time: float = 0.0
    result: Any = None
    error: str = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class WorkflowNode:
    name: str
    type: NodeType
    handler: Callable
    next_nodes: list[str] = field(default_factory=list)
    condition: Callable = None  # 条件分岐用
    retry_count: int = 0       # リトライ回数
    timeout: float = 30.0      # タイムアウト秒数
    description: str = ""      # ノードの説明

class WorkflowEngine:
    def __init__(self, name: str = "default"):
        self.name = name
        self.nodes: dict[str, WorkflowNode] = {}
        self.state: dict[str, Any] = {}
        self.execution_log: list[NodeExecution] = []
        self.hooks: dict[str, list[Callable]] = {
            "before_node": [],
            "after_node": [],
            "on_error": [],
        }

    def add_node(self, node: WorkflowNode):
        self.nodes[node.name] = node

    def add_hook(self, event: str, callback: Callable):
        """フックを追加"""
        if event in self.hooks:
            self.hooks[event].append(callback)

    def _execute_hooks(self, event: str, **kwargs):
        for hook in self.hooks.get(event, []):
            hook(**kwargs)

    def _execute_node(self, node: WorkflowNode) -> Any:
        """ノードを実行（リトライ付き）"""
        last_error = None
        for attempt in range(node.retry_count + 1):
            try:
                self._execute_hooks("before_node", node=node, state=self.state)
                result = node.handler(self.state)
                self._execute_hooks("after_node", node=node, state=self.state, result=result)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"ノード {node.name} 失敗 (試行 {attempt + 1}): {e}")
                if attempt < node.retry_count:
                    time.sleep(2 ** attempt)  # 指数バックオフ

        self._execute_hooks("on_error", node=node, error=last_error)
        raise last_error

    def run(self, start_node: str, initial_state: dict) -> dict:
        self.state = initial_state
        self.execution_log = []
        current = start_node

        while current:
            node = self.nodes[current]
            execution = NodeExecution(
                node_name=node.name,
                status=NodeStatus.RUNNING,
                start_time=time.time()
            )
            logger.info(f"実行中: {node.name} ({node.type.value})")

            try:
                if node.type == NodeType.CONDITION:
                    branch = node.condition(self.state)
                    current = branch
                    execution.status = NodeStatus.COMPLETED
                    execution.result = f"分岐先: {branch}"
                elif node.type == NodeType.PARALLEL:
                    results = self._run_parallel(node.next_nodes)
                    self.state["parallel_results"] = results
                    current = node.next_nodes[-1] if node.next_nodes else None
                    execution.status = NodeStatus.COMPLETED
                else:
                    result = self._execute_node(node)
                    self.state[f"{node.name}_result"] = result
                    current = node.next_nodes[0] if node.next_nodes else None
                    execution.status = NodeStatus.COMPLETED
                    execution.result = result
            except Exception as e:
                execution.status = NodeStatus.FAILED
                execution.error = str(e)
                logger.error(f"ノード {node.name} で致命的エラー: {e}")
                raise
            finally:
                execution.end_time = time.time()
                self.execution_log.append(execution)

        return self.state

    def get_execution_summary(self) -> str:
        """実行サマリーを取得"""
        lines = [f"ワークフロー '{self.name}' 実行結果:"]
        total_duration = 0
        for ex in self.execution_log:
            status_icon = "✓" if ex.status == NodeStatus.COMPLETED else "✗"
            lines.append(
                f"  {status_icon} {ex.node_name}: {ex.duration:.2f}s "
                f"({ex.status.value})"
            )
            total_duration += ex.duration
        lines.append(f"  合計時間: {total_duration:.2f}s")
        return "\n".join(lines)
```

### 2.3 条件分岐の実装

```python
# 条件分岐を含むワークフロー
import anthropic

client = anthropic.Anthropic()

def classify_inquiry(message: str) -> str:
    """問い合わせをLLMで分類"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": f"""以下の問い合わせを分類してください。
カテゴリ: billing, technical, general
1単語で回答: {message}"""
        }]
    )
    return response.content[0].text.strip().lower()

def handle_billing_inquiry(state: dict) -> str:
    """請求関連の対応"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        system="あなたは請求担当のサポートエージェントです。丁寧かつ正確に回答してください。",
        messages=[{
            "role": "user",
            "content": state["user_message"]
        }]
    )
    return response.content[0].text

def handle_technical_inquiry(state: dict) -> str:
    """技術関連の対応"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system="あなたは技術サポートエージェントです。技術的な問題を解決してください。",
        messages=[{
            "role": "user",
            "content": state["user_message"]
        }]
    )
    return response.content[0].text

def handle_general_inquiry(state: dict) -> str:
    """一般問い合わせの対応"""
    response = client.messages.create(
        model="claude-haiku-3-20240307",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": state["user_message"]
        }]
    )
    return response.content[0].text

def generate_response(state: dict) -> str:
    """最終回答を整形"""
    # 分類結果に応じた結果を取得
    classification = state.get("classify_result", "general")
    handler_key = f"handle_{classification}_result"
    raw_response = state.get(handler_key, "回答を生成できませんでした")

    return f"""
【カテゴリ】{classification}
【回答】
{raw_response}

何かご不明な点がございましたらお気軽にお問い合わせください。
"""

def build_support_workflow():
    engine = WorkflowEngine(name="カスタマーサポート")

    # ノード1: 問い合わせ分類
    engine.add_node(WorkflowNode(
        name="classify",
        type=NodeType.LLM,
        handler=lambda state: classify_inquiry(state["user_message"]),
        next_nodes=["route"],
        description="問い合わせ内容をLLMで分類"
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
        }.get(state["classify_result"], "handle_general"),
        description="分類結果に基づいてルーティング"
    ))

    # ノード3a: 請求対応
    engine.add_node(WorkflowNode(
        name="handle_billing",
        type=NodeType.LLM,
        handler=lambda state: handle_billing_inquiry(state),
        next_nodes=["respond"],
        retry_count=2,
        description="請求関連の問い合わせに回答"
    ))

    # ノード3b: 技術対応
    engine.add_node(WorkflowNode(
        name="handle_technical",
        type=NodeType.LLM,
        handler=lambda state: handle_technical_inquiry(state),
        next_nodes=["respond"],
        retry_count=2,
        description="技術的な問い合わせに回答"
    ))

    # ノード3c: 一般対応
    engine.add_node(WorkflowNode(
        name="handle_general",
        type=NodeType.LLM,
        handler=lambda state: handle_general_inquiry(state),
        next_nodes=["respond"],
        retry_count=1,
        description="一般的な問い合わせに回答"
    ))

    # ノード4: 回答生成
    engine.add_node(WorkflowNode(
        name="respond",
        type=NodeType.LLM,
        handler=lambda state: generate_response(state),
        next_nodes=[],
        description="最終回答を整形して返す"
    ))

    return engine

# 実行
workflow = build_support_workflow()
result = workflow.run("classify", {
    "user_message": "先月の請求額が間違っています"
})
print(workflow.get_execution_summary())
```

### 2.4 動的フロー生成

```python
# LLMがワークフロー自体を生成するパターン
import json

class DynamicWorkflowBuilder:
    """LLMの判断でワークフローを動的に構築"""

    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.available_handlers = {}

    def register_handler(self, name: str, handler: Callable, description: str):
        """利用可能なハンドラーを登録"""
        self.available_handlers[name] = {
            "handler": handler,
            "description": description
        }

    def build_workflow(self, task_description: str) -> WorkflowEngine:
        """タスク記述からワークフローを自動生成"""
        handler_descriptions = "\n".join(
            f"- {name}: {info['description']}"
            for name, info in self.available_handlers.items()
        )

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""以下のタスクを処理するワークフローをJSON形式で設計してください。

タスク: {task_description}

利用可能なハンドラー:
{handler_descriptions}

出力形式:
{{
    "nodes": [
        {{
            "name": "ノード名",
            "handler": "ハンドラー名",
            "next": ["次のノード名"],
            "type": "llm|condition|parallel"
        }}
    ],
    "start_node": "最初のノード名"
}}"""
            }]
        )

        # JSONを解析してワークフローを構築
        workflow_spec = json.loads(response.content[0].text)
        engine = WorkflowEngine(name=f"dynamic_{task_description[:20]}")

        for node_spec in workflow_spec["nodes"]:
            handler_name = node_spec["handler"]
            handler = self.available_handlers.get(handler_name, {}).get("handler")

            if handler:
                engine.add_node(WorkflowNode(
                    name=node_spec["name"],
                    type=NodeType(node_spec.get("type", "llm")),
                    handler=handler,
                    next_nodes=node_spec.get("next", [])
                ))

        return engine, workflow_spec["start_node"]

# 使用例
builder = DynamicWorkflowBuilder(client)
builder.register_handler(
    "extract_text", extract_text_from_pdf,
    "PDFからテキストを抽出"
)
builder.register_handler(
    "translate", translate_text,
    "テキストを翻訳"
)
builder.register_handler(
    "summarize", summarize_text,
    "テキストを要約"
)

workflow, start = builder.build_workflow(
    "英語のPDFを読み取り、日本語に翻訳して要約する"
)
result = workflow.run(start, {"pdf_path": "document.pdf"})
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
from typing import TypedDict, Literal, Annotated
from operator import add

# 状態の型定義
class AgentState(TypedDict):
    messages: list
    current_step: str
    classification: str
    draft: str
    review_result: str
    review_count: Annotated[int, "レビュー回数"]
    final_output: str
    token_usage: Annotated[list[dict], add]

# ノード関数
def classify_request(state: AgentState) -> AgentState:
    """リクエストを分類する"""
    messages = state["messages"]
    result = llm.invoke(
        f"以下のリクエストを 'simple' / 'complex' に分類:\n{messages[-1]}"
    )
    return {
        "classification": result.content.strip(),
        "token_usage": [{"node": "classify", "tokens": result.usage_metadata}]
    }

def handle_simple(state: AgentState) -> AgentState:
    """簡単なリクエストを処理"""
    draft = llm.invoke(f"簡潔に回答: {state['messages'][-1]}")
    return {
        "draft": draft.content,
        "token_usage": [{"node": "handle_simple", "tokens": draft.usage_metadata}]
    }

def handle_complex(state: AgentState) -> AgentState:
    """複雑なリクエストを処理"""
    context = ""
    if state.get("review_result") and "FAIL" in state["review_result"]:
        context = f"\n\n前回のフィードバック: {state['review_result']}"

    draft = llm.invoke(
        f"詳細に回答: {state['messages'][-1]}{context}"
    )
    return {
        "draft": draft.content,
        "token_usage": [{"node": "handle_complex", "tokens": draft.usage_metadata}]
    }

def review(state: AgentState) -> AgentState:
    """回答をレビュー"""
    result = llm.invoke(
        f"回答の品質を評価。PASS/FAIL（FAILの場合は理由も記載）:\n{state['draft']}"
    )
    return {
        "review_result": result.content.strip(),
        "review_count": state.get("review_count", 0) + 1,
        "token_usage": [{"node": "review", "tokens": result.usage_metadata}]
    }

def finalize(state: AgentState) -> AgentState:
    """最終出力を生成"""
    return {"final_output": state["draft"]}

# 条件分岐関数
def route_by_classification(state: AgentState) -> Literal["simple", "complex"]:
    return "simple" if "simple" in state["classification"] else "complex"

def route_by_review(state: AgentState) -> Literal["revise", "finalize"]:
    # 最大3回のレビューで打ち切り
    if state.get("review_count", 0) >= 3:
        return "finalize"
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
    "review_result": "", "review_count": 0,
    "final_output": "", "token_usage": []
})
```

### 3.3 LangGraphチェックポイント

```python
# チェックポイント機能付きワークフロー
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.postgres import PostgresSaver
import sqlite3

# SQLiteチェックポイント（開発用）
def create_workflow_with_checkpoint():
    """チェックポイント機能付きワークフロー"""
    conn = sqlite3.connect("workflow_checkpoints.db")
    memory = SqliteSaver(conn)

    workflow = StateGraph(AgentState)
    # ... ノードとエッジを追加 ...

    # チェックポイント付きでコンパイル
    app = workflow.compile(checkpointer=memory)

    # スレッドIDで実行状態を管理
    config = {"configurable": {"thread_id": "support-001"}}

    # 実行（途中で失敗しても再開可能）
    try:
        result = app.invoke(initial_state, config)
    except Exception as e:
        print(f"エラー発生: {e}")
        # 最後のチェックポイントから状態を取得
        state = app.get_state(config)
        print(f"最後の成功ノード: {state.values}")

        # 状態を修正して再開
        app.update_state(config, {"draft": "修正済みドラフト"})
        result = app.invoke(None, config)  # Noneで前回の状態から再開

    return result

# PostgreSQLチェックポイント（本番用）
def create_production_workflow():
    """本番環境向けチェックポイント"""
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        "postgresql://user:pass@localhost/workflows",
        max_size=20
    )
    memory = PostgresSaver(pool)
    memory.setup()  # テーブル作成

    workflow = StateGraph(AgentState)
    # ... ノード追加 ...
    app = workflow.compile(checkpointer=memory)

    return app
```

### 3.4 LangGraphストリーミング

```python
# ストリーミング実行
async def stream_workflow():
    """ワークフローの進捗をリアルタイムで取得"""
    app = create_workflow()

    # ノードごとの出力をストリーミング
    async for event in app.astream(
        initial_state,
        config={"configurable": {"thread_id": "stream-001"}}
    ):
        for node_name, output in event.items():
            print(f"[{node_name}] 完了:")
            print(f"  出力: {json.dumps(output, ensure_ascii=False, indent=2)}")

    # トークン単位のストリーミング
    async for event in app.astream_events(
        initial_state,
        config={"configurable": {"thread_id": "stream-002"}},
        version="v2"
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(chunk.content, end="", flush=True)
        elif event["event"] == "on_chain_end":
            print(f"\n--- {event['name']} 完了 ---")
```

---

## 4. 並列実行パターン

### 4.1 基本的な並列実行

```python
# 並列ノードの実装
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

class ParallelResult(NamedTuple):
    node_name: str
    result: Any
    duration: float
    success: bool
    error: str = ""

class ParallelWorkflow:
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers

    async def run_parallel_nodes(
        self,
        nodes: list[WorkflowNode],
        state: dict,
        timeout: float = 60.0
    ) -> list[ParallelResult]:
        """複数のノードを並列に実行（タイムアウト付き）"""

        async def execute_with_tracking(node: WorkflowNode) -> ParallelResult:
            start = time.time()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(node.handler, state.copy()),
                    timeout=node.timeout
                )
                return ParallelResult(
                    node_name=node.name,
                    result=result,
                    duration=time.time() - start,
                    success=True
                )
            except asyncio.TimeoutError:
                return ParallelResult(
                    node_name=node.name,
                    result=None,
                    duration=time.time() - start,
                    success=False,
                    error=f"タイムアウト ({node.timeout}s)"
                )
            except Exception as e:
                return ParallelResult(
                    node_name=node.name,
                    result=None,
                    duration=time.time() - start,
                    success=False,
                    error=str(e)
                )

        tasks = [execute_with_tracking(node) for node in nodes]
        results = await asyncio.gather(*tasks)

        # 結果をステートに統合
        for pr in results:
            if pr.success:
                state[f"{pr.node_name}_result"] = pr.result
            else:
                state[f"{pr.node_name}_error"] = pr.error

        return list(results)

# 使用例: 複数情報源からの並列収集
async def parallel_research(state: dict) -> dict:
    """複数ソースから並列にデータ収集"""
    query = state["query"]

    async def search_academic(q: str) -> list[dict]:
        # Semantic Scholar APIなど
        await asyncio.sleep(1)  # シミュレーション
        return [{"title": "Paper A", "source": "academic"}]

    async def search_news(q: str) -> list[dict]:
        await asyncio.sleep(0.5)
        return [{"title": "News B", "source": "news"}]

    async def search_github(q: str) -> list[dict]:
        await asyncio.sleep(0.8)
        return [{"title": "Repo C", "source": "github"}]

    papers, news, repos = await asyncio.gather(
        search_academic(query),
        search_news(query),
        search_github(query)
    )

    return {
        "papers": papers,
        "news": news,
        "repos": repos,
        "total_sources": len(papers) + len(news) + len(repos)
    }
```

### 4.2 Fan-Out / Fan-In パターン

```python
# Fan-Out / Fan-In: 入力を分割して並列処理し、結果を集約
class FanOutFanInWorkflow:
    """大量データを分割して並列処理するワークフロー"""

    def __init__(self, client: anthropic.Anthropic, max_concurrent: int = 5):
        self.client = client
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def fan_out(self, items: list, chunk_size: int = 10) -> list[list]:
        """入力をチャンクに分割"""
        return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

    async def process_chunk(self, chunk: list, prompt_template: str) -> list[dict]:
        """1チャンクをLLMで処理"""
        async with self.semaphore:
            items_text = "\n".join(str(item) for item in chunk)
            response = self.client.messages.create(
                model="claude-haiku-3-20240307",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": prompt_template.format(items=items_text)
                }]
            )
            return {"chunk_result": response.content[0].text}

    async def fan_in(self, results: list[dict]) -> dict:
        """並列処理の結果を集約"""
        all_results = []
        for r in results:
            all_results.append(r["chunk_result"])

        # 集約結果をLLMで統合
        combined = "\n---\n".join(all_results)
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"以下の分析結果を統合して総合レポートを作成:\n{combined}"
            }]
        )
        return {"summary": response.content[0].text}

    async def run(self, items: list, prompt_template: str) -> dict:
        """Fan-Out → 並列処理 → Fan-In の完全フロー"""
        # Fan-Out
        chunks = await self.fan_out(items)
        print(f"  {len(chunks)}チャンクに分割")

        # 並列処理
        tasks = [self.process_chunk(chunk, prompt_template) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        print(f"  {len(results)}チャンク処理完了")

        # Fan-In
        summary = await self.fan_in(list(results))
        return summary

# 使用例: 1000件のレビューを並列分析
async def analyze_reviews():
    workflow = FanOutFanInWorkflow(client, max_concurrent=10)

    reviews = load_reviews()  # 1000件のレビュー
    result = await workflow.run(
        items=reviews,
        prompt_template="以下のレビューを感情分析（positive/negative/neutral）:\n{items}"
    )
    print(result["summary"])
```

### 4.3 Map-Reduce ワークフロー

```python
# LangGraphでのMap-Reduce実装
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from operator import add

class MapReduceState(TypedDict):
    documents: list[str]
    summaries: Annotated[list[str], add]  # addで結果をマージ
    final_summary: str

def map_summarize(state: MapReduceState) -> MapReduceState:
    """各ドキュメントを個別に要約（Mapフェーズ）"""
    summaries = []
    for doc in state["documents"]:
        response = llm.invoke(f"以下を100文字で要約:\n{doc}")
        summaries.append(response.content)
    return {"summaries": summaries}

def reduce_combine(state: MapReduceState) -> MapReduceState:
    """要約を統合（Reduceフェーズ）"""
    all_summaries = "\n".join(
        f"{i+1}. {s}" for i, s in enumerate(state["summaries"])
    )
    response = llm.invoke(
        f"以下の要約を統合して総合的なサマリーを作成:\n{all_summaries}"
    )
    return {"final_summary": response.content}

# グラフ構築
map_reduce = StateGraph(MapReduceState)
map_reduce.add_node("map", map_summarize)
map_reduce.add_node("reduce", reduce_combine)
map_reduce.set_entry_point("map")
map_reduce.add_edge("map", "reduce")
map_reduce.add_edge("reduce", END)

app = map_reduce.compile()
```

---

## 5. ワークフローパターン比較

### 5.1 フロー形状別比較

| パターン | 形状 | 特徴 | 適用場面 | 実装難度 |
|----------|------|------|---------|---------|
| 直列 | A→B→C | 最もシンプル | パイプライン処理 | 低 |
| 条件分岐 | A→B or C | 入力に応じた経路 | ルーティング | 低 |
| 並列 | A→[B,C]→D | 独立タスクの同時実行 | データ収集 | 中 |
| ループ | A→B→A(条件付き) | 品質基準まで繰り返し | レビュー/改善 | 中 |
| Fan-Out/In | A→[B1..Bn]→C | 大量データの分散処理 | バッチ分析 | 高 |
| サブワークフロー | A→[Sub]→B | 再利用可能な子フロー | 共通処理の部品化 | 中 |
| 動的生成 | LLMがフロー設計 | 柔軟だが予測困難 | 汎用タスク | 高 |

### 5.2 ワークフロー vs 自律エージェント

| 観点 | ワークフロー | 自律エージェント |
|------|------------|----------------|
| 制御性 | 高（予測可能） | 低（非決定的） |
| 柔軟性 | 中（設計済みパス内） | 高（任意の行動可能） |
| デバッグ | 容易（各ノード個別） | 困難（自由行動） |
| コスト | 予測可能 | 予測困難 |
| 設計コスト | 高（事前設計必要） | 低（プロンプトのみ） |
| 信頼性 | 高 | 中 |
| レイテンシ | 最適化可能（並列化） | 最適化困難 |
| 監査性 | 高（実行ログ明確） | 低（行動が不定） |
| 適用場面 | 業務プロセス自動化 | 探索的タスク |

### 5.3 パターン選択の意思決定マトリクス

```
パターン選択ガイド

                    タスクの複雑さ
                    低い        高い
フローの    固定   │ 直列       │ サブワークフロー │
予測可能性         │ パイプライン │ 階層型フロー     │
            ───────┼────────────┼──────────────────┤
            変動   │ 条件分岐    │ 動的生成 +       │
                   │ ワークフロー│ ハイブリッド      │

データ量     少量  │ 直列/条件分岐│
            大量  │ Fan-Out/In  │ Map-Reduce      │

リアルタイム性
            必要  │ ストリーミング + 並列実行       │
            不要  │ バッチ処理 + チェックポイント    │
```

---

## 6. 状態管理

### 6.1 状態管理パターンの概要

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

### 6.2 型安全な状態管理

```python
# Pydanticを使った型安全な状態管理
from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
from enum import Enum

class WorkflowPhase(Enum):
    INTAKE = "intake"
    PROCESSING = "processing"
    REVIEW = "review"
    OUTPUT = "output"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowState(BaseModel):
    """ワークフローの状態（型安全）"""
    # 基本情報
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str
    step: int = 0
    phase: WorkflowPhase = WorkflowPhase.INTAKE

    # 処理結果
    classification: Optional[str] = None
    intermediate_results: list[str] = Field(default_factory=list)
    final_output: Optional[str] = None

    # エラー管理
    errors: list[str] = Field(default_factory=list)
    retry_counts: dict[str, int] = Field(default_factory=dict)

    # メタデータ
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    # トークン使用量追跡
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @validator("step")
    def step_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("step must be non-negative")
        return v

    def advance(self) -> "WorkflowState":
        """ステップを進める"""
        return self.model_copy(update={
            "step": self.step + 1,
            "updated_at": datetime.now()
        })

    def add_result(self, result: str) -> "WorkflowState":
        """中間結果を追加"""
        new_results = self.intermediate_results + [result]
        return self.model_copy(update={
            "intermediate_results": new_results,
            "updated_at": datetime.now()
        })

    def add_error(self, error: str, node_name: str = "") -> "WorkflowState":
        """エラーを記録"""
        new_errors = self.errors + [f"[{node_name}] {error}"]
        retry = self.retry_counts.copy()
        if node_name:
            retry[node_name] = retry.get(node_name, 0) + 1
        return self.model_copy(update={
            "errors": new_errors,
            "retry_counts": retry,
            "updated_at": datetime.now()
        })

    def track_tokens(self, input_tokens: int, output_tokens: int) -> "WorkflowState":
        """トークン使用量を追跡"""
        return self.model_copy(update={
            "total_input_tokens": self.total_input_tokens + input_tokens,
            "total_output_tokens": self.total_output_tokens + output_tokens,
        })

    def transition_to(self, phase: WorkflowPhase) -> "WorkflowState":
        """フェーズを遷移"""
        return self.model_copy(update={
            "phase": phase,
            "updated_at": datetime.now()
        })

    @property
    def estimated_cost(self) -> float:
        """推定コスト（USD）- Claude Sonnet基準"""
        input_cost = self.total_input_tokens * 3.0 / 1_000_000
        output_cost = self.total_output_tokens * 15.0 / 1_000_000
        return input_cost + output_cost
```

### 6.3 永続化可能な状態ストア

```python
# Redis/SQLiteバックエンドの状態ストア
import json
import sqlite3
from abc import ABC, abstractmethod

class StateStore(ABC):
    """状態ストアの抽象基底クラス"""

    @abstractmethod
    def save(self, workflow_id: str, state: WorkflowState) -> None:
        pass

    @abstractmethod
    def load(self, workflow_id: str) -> Optional[WorkflowState]:
        pass

    @abstractmethod
    def list_workflows(self, status: Optional[str] = None) -> list[str]:
        pass

class SQLiteStateStore(StateStore):
    """SQLiteベースの状態ストア"""

    def __init__(self, db_path: str = "workflow_states.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS workflow_states (
                workflow_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                phase TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def save(self, workflow_id: str, state: WorkflowState) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO workflow_states
               (workflow_id, state_json, phase, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                workflow_id,
                state.model_dump_json(),
                state.phase.value,
                state.created_at.isoformat(),
                state.updated_at.isoformat()
            )
        )
        self.conn.commit()

    def load(self, workflow_id: str) -> Optional[WorkflowState]:
        row = self.conn.execute(
            "SELECT state_json FROM workflow_states WHERE workflow_id = ?",
            (workflow_id,)
        ).fetchone()
        if row:
            return WorkflowState.model_validate_json(row[0])
        return None

    def list_workflows(self, status: Optional[str] = None) -> list[str]:
        if status:
            rows = self.conn.execute(
                "SELECT workflow_id FROM workflow_states WHERE phase = ?",
                (status,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT workflow_id FROM workflow_states"
            ).fetchall()
        return [row[0] for row in rows]

class RedisStateStore(StateStore):
    """Redisベースの状態ストア（高速アクセス）"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        import redis
        self.redis = redis.from_url(redis_url)
        self.prefix = "workflow:"

    def save(self, workflow_id: str, state: WorkflowState) -> None:
        key = f"{self.prefix}{workflow_id}"
        self.redis.set(key, state.model_dump_json())
        self.redis.sadd(f"{self.prefix}index:{state.phase.value}", workflow_id)
        # TTL: 30日
        self.redis.expire(key, 30 * 24 * 3600)

    def load(self, workflow_id: str) -> Optional[WorkflowState]:
        data = self.redis.get(f"{self.prefix}{workflow_id}")
        if data:
            return WorkflowState.model_validate_json(data)
        return None

    def list_workflows(self, status: Optional[str] = None) -> list[str]:
        if status:
            return [
                m.decode()
                for m in self.redis.smembers(f"{self.prefix}index:{status}")
            ]
        # 全件取得
        keys = self.redis.keys(f"{self.prefix}*")
        return [
            k.decode().replace(self.prefix, "")
            for k in keys
            if b"index:" not in k
        ]
```

---

## 7. エラーハンドリングとリトライ

### 7.1 ノードレベルのエラーハンドリング

```python
# 堅牢なエラーハンドリング付きワークフロー
from dataclasses import dataclass
from typing import Optional
import traceback

@dataclass
class NodeError:
    node_name: str
    error_type: str
    message: str
    traceback: str
    retry_attempt: int
    timestamp: float

class ResilientWorkflowEngine(WorkflowEngine):
    """障害耐性のあるワークフローエンジン"""

    def __init__(self, name: str, state_store: Optional[StateStore] = None):
        super().__init__(name)
        self.state_store = state_store
        self.error_handlers: dict[str, Callable] = {}
        self.fallback_handlers: dict[str, Callable] = {}
        self.errors: list[NodeError] = []

    def set_error_handler(self, node_name: str, handler: Callable):
        """ノード固有のエラーハンドラーを設定"""
        self.error_handlers[node_name] = handler

    def set_fallback(self, node_name: str, fallback: Callable):
        """フォールバック処理を設定"""
        self.fallback_handlers[node_name] = fallback

    def _execute_node_resilient(self, node: WorkflowNode) -> Any:
        """耐障害性のあるノード実行"""
        for attempt in range(node.retry_count + 1):
            try:
                return node.handler(self.state)
            except Exception as e:
                error = NodeError(
                    node_name=node.name,
                    error_type=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                    retry_attempt=attempt,
                    timestamp=time.time()
                )
                self.errors.append(error)

                # ノード固有のエラーハンドラー
                if node.name in self.error_handlers:
                    should_retry = self.error_handlers[node.name](
                        error, self.state
                    )
                    if not should_retry:
                        break

                if attempt < node.retry_count:
                    wait = min(2 ** attempt * 1.0, 30.0)  # 最大30秒
                    logger.warning(
                        f"リトライ {attempt+1}/{node.retry_count}: "
                        f"{node.name} ({wait:.1f}s待機)"
                    )
                    time.sleep(wait)

        # 全リトライ失敗 → フォールバック
        if node.name in self.fallback_handlers:
            logger.info(f"フォールバック実行: {node.name}")
            return self.fallback_handlers[node.name](self.state)

        raise RuntimeError(
            f"ノード '{node.name}' が {node.retry_count + 1}回の試行後に失敗"
        )

    def run(self, start_node: str, initial_state: dict) -> dict:
        """チェックポイント付き実行"""
        self.state = initial_state
        workflow_id = self.state.get("workflow_id", str(time.time()))
        current = start_node

        while current:
            node = self.nodes[current]

            # チェックポイント保存
            if self.state_store:
                ws = WorkflowState(
                    workflow_id=workflow_id,
                    user_input=str(initial_state),
                    step=len(self.execution_log),
                    phase=WorkflowPhase.PROCESSING
                )
                self.state_store.save(workflow_id, ws)

            try:
                if node.type == NodeType.CONDITION:
                    branch = node.condition(self.state)
                    current = branch
                else:
                    result = self._execute_node_resilient(node)
                    self.state[f"{node.name}_result"] = result
                    current = node.next_nodes[0] if node.next_nodes else None
            except Exception as e:
                self.state["workflow_error"] = str(e)
                self.state["failed_node"] = node.name
                logger.error(f"ワークフロー停止: {node.name} - {e}")
                break

        return self.state

# 使用例
engine = ResilientWorkflowEngine("resilient_support", SQLiteStateStore())

# エラーハンドラー設定
def handle_llm_error(error: NodeError, state: dict) -> bool:
    """LLMエラーのハンドリング"""
    if "rate_limit" in error.message.lower():
        time.sleep(60)  # レート制限時は60秒待機
        return True  # リトライする
    if "overloaded" in error.message.lower():
        return True  # リトライする
    return False  # リトライしない

engine.set_error_handler("handle_technical", handle_llm_error)

# フォールバック設定
engine.set_fallback(
    "handle_technical",
    lambda state: "申し訳ございません。技術チームに転送いたします。"
)
```

### 7.2 サーキットブレーカーパターン

```python
# ワークフローノード用サーキットブレーカー
from enum import Enum
import threading

class CircuitState(Enum):
    CLOSED = "closed"      # 正常
    OPEN = "open"          # 遮断中
    HALF_OPEN = "half_open"  # 試行中

class CircuitBreaker:
    """ワークフローノード用サーキットブレーカー"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = threading.Lock()

    def can_execute(self) -> bool:
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False
            else:  # HALF_OPEN
                return True

    def record_success(self):
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            self.failure_count = 0

    def record_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"サーキットブレーカー OPEN: "
                    f"{self.failure_count}回連続失敗"
                )

def with_circuit_breaker(breaker: CircuitBreaker):
    """サーキットブレーカーデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not breaker.can_execute():
                raise RuntimeError("サーキットブレーカーが開いています")
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        return wrapper
    return decorator

# 使用例
llm_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=120)

@with_circuit_breaker(llm_breaker)
def call_llm_with_breaker(state: dict) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{"role": "user", "content": state["prompt"]}]
    )
    return response.content[0].text
```

---

## 8. サブワークフロー

### 8.1 サブワークフローの設計

```python
# 再利用可能なサブワークフロー
class SubWorkflow:
    """独立してテスト・再利用できるサブワークフロー"""

    def __init__(self, name: str):
        self.name = name
        self.engine = WorkflowEngine(name=name)
        self.input_schema: dict = {}
        self.output_schema: dict = {}

    def define_interface(self, input_keys: list[str], output_keys: list[str]):
        """入出力インターフェースを定義"""
        self.input_schema = {k: True for k in input_keys}
        self.output_schema = {k: True for k in output_keys}

    def validate_input(self, state: dict) -> bool:
        """入力の検証"""
        missing = [k for k in self.input_schema if k not in state]
        if missing:
            raise ValueError(f"必須入力が不足: {missing}")
        return True

    def execute(self, input_state: dict, start_node: str) -> dict:
        """サブワークフローを実行"""
        self.validate_input(input_state)
        result = self.engine.run(start_node, input_state)

        # 出力のみを返す
        output = {k: result.get(k) for k in self.output_schema}
        return output

# サブワークフロー: テキスト品質チェック
def build_quality_check_subworkflow() -> SubWorkflow:
    """テキスト品質チェックのサブワークフロー"""
    sub = SubWorkflow("quality_check")
    sub.define_interface(
        input_keys=["text", "criteria"],
        output_keys=["quality_score", "issues", "improved_text"]
    )

    # 文法チェックノード
    sub.engine.add_node(WorkflowNode(
        name="grammar_check",
        type=NodeType.LLM,
        handler=lambda s: check_grammar(s["text"]),
        next_nodes=["style_check"]
    ))

    # スタイルチェックノード
    sub.engine.add_node(WorkflowNode(
        name="style_check",
        type=NodeType.LLM,
        handler=lambda s: check_style(s["text"], s["criteria"]),
        next_nodes=["score"]
    ))

    # スコアリングノード
    sub.engine.add_node(WorkflowNode(
        name="score",
        type=NodeType.LLM,
        handler=lambda s: calculate_score(s),
        next_nodes=[]
    ))

    return sub

# メインワークフローからサブワークフローを呼び出し
quality_checker = build_quality_check_subworkflow()

def content_pipeline():
    engine = WorkflowEngine("content_pipeline")

    # 記事生成ノード
    engine.add_node(WorkflowNode(
        name="generate",
        type=NodeType.LLM,
        handler=generate_article,
        next_nodes=["quality_check"]
    ))

    # サブワークフロー呼び出しノード
    engine.add_node(WorkflowNode(
        name="quality_check",
        type=NodeType.SUBWORKFLOW,
        handler=lambda state: quality_checker.execute(
            {"text": state["generate_result"], "criteria": "blog"},
            start_node="grammar_check"
        ),
        next_nodes=["publish"]
    ))

    # 公開ノード
    engine.add_node(WorkflowNode(
        name="publish",
        type=NodeType.TOOL,
        handler=publish_article,
        next_nodes=[]
    ))

    return engine
```

---

## 9. 実践的なワークフロー事例

### 9.1 コードレビューワークフロー

```python
# 自動コードレビューワークフロー
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class CodeReviewState(TypedDict):
    code: str
    language: str
    diff: str
    static_analysis: dict
    security_issues: list[dict]
    performance_issues: list[dict]
    style_issues: list[dict]
    review_summary: str
    approval_status: str  # "approved", "changes_requested", "blocked"

def detect_language(state: CodeReviewState) -> CodeReviewState:
    """プログラミング言語を検出"""
    response = client.messages.create(
        model="claude-haiku-3-20240307",
        max_tokens=20,
        messages=[{
            "role": "user",
            "content": f"このコードの言語を1単語で: {state['code'][:500]}"
        }]
    )
    return {"language": response.content[0].text.strip().lower()}

def run_static_analysis(state: CodeReviewState) -> CodeReviewState:
    """静的解析を実行"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""以下の{state['language']}コードを静的解析してください。
JSON形式で結果を返してください:
{{"complexity": "low/medium/high", "issues": [...], "metrics": {{}}}}

コード:
{state['code']}"""
        }]
    )
    return {"static_analysis": json.loads(response.content[0].text)}

def check_security(state: CodeReviewState) -> CodeReviewState:
    """セキュリティチェック"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system="セキュリティ専門のコードレビュアーとして脆弱性を分析してください。",
        messages=[{
            "role": "user",
            "content": f"""以下のコードのセキュリティ脆弱性を検出:
- SQLインジェクション
- XSS
- 認証/認可の不備
- 機密情報のハードコード

JSON配列で返してください。

コード:
{state['code']}"""
        }]
    )
    return {"security_issues": json.loads(response.content[0].text)}

def check_performance(state: CodeReviewState) -> CodeReviewState:
    """パフォーマンスチェック"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""パフォーマンス観点でコードレビュー:
- N+1クエリ
- メモリリーク
- 不要なループ
- キャッシュ未使用

JSON配列で: {state['code']}"""
        }]
    )
    return {"performance_issues": json.loads(response.content[0].text)}

def generate_review_summary(state: CodeReviewState) -> CodeReviewState:
    """レビューサマリーを生成"""
    total_issues = (
        len(state.get("security_issues", []))
        + len(state.get("performance_issues", []))
        + len(state.get("style_issues", []))
    )

    has_critical = any(
        issue.get("severity") == "critical"
        for issue in state.get("security_issues", [])
    )

    if has_critical:
        status = "blocked"
    elif total_issues > 5:
        status = "changes_requested"
    else:
        status = "approved"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""以下のレビュー結果を要約:
セキュリティ: {json.dumps(state.get('security_issues', []), ensure_ascii=False)}
パフォーマンス: {json.dumps(state.get('performance_issues', []), ensure_ascii=False)}
ステータス: {status}

Markdown形式でレビューサマリーを作成:"""
        }]
    )

    return {
        "review_summary": response.content[0].text,
        "approval_status": status
    }

# グラフ構築
review_flow = StateGraph(CodeReviewState)

review_flow.add_node("detect_lang", detect_language)
review_flow.add_node("static_analysis", run_static_analysis)
review_flow.add_node("security_check", check_security)
review_flow.add_node("performance_check", check_performance)
review_flow.add_node("summarize", generate_review_summary)

review_flow.set_entry_point("detect_lang")
review_flow.add_edge("detect_lang", "static_analysis")
# 静的解析後、セキュリティとパフォーマンスを並列実行
review_flow.add_edge("static_analysis", "security_check")
review_flow.add_edge("static_analysis", "performance_check")
review_flow.add_edge("security_check", "summarize")
review_flow.add_edge("performance_check", "summarize")
review_flow.add_edge("summarize", END)

code_reviewer = review_flow.compile()
```

### 9.2 コンテンツ生成パイプライン

```python
# ブログ記事自動生成ワークフロー
class ContentState(TypedDict):
    topic: str
    target_audience: str
    outline: str
    draft: str
    seo_keywords: list[str]
    review_feedback: str
    review_pass: bool
    final_article: str
    metadata: dict

def research_topic(state: ContentState) -> ContentState:
    """トピックをリサーチ"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""以下のトピックについてリサーチして、
記事のアウトラインを作成してください。

トピック: {state['topic']}
対象読者: {state['target_audience']}

アウトライン形式:
1. 導入部
2. 本文（3-5セクション）
3. まとめ
各セクションに含めるべきポイントを箇条書きで。"""
        }]
    )
    return {"outline": response.content[0].text}

def extract_seo_keywords(state: ContentState) -> ContentState:
    """SEOキーワードを抽出"""
    response = client.messages.create(
        model="claude-haiku-3-20240307",
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": f"トピック「{state['topic']}」のSEOキーワードを10個、JSON配列で:"
        }]
    )
    return {"seo_keywords": json.loads(response.content[0].text)}

def write_draft(state: ContentState) -> ContentState:
    """記事のドラフトを執筆"""
    feedback_context = ""
    if state.get("review_feedback"):
        feedback_context = f"\n\n前回のフィードバック:\n{state['review_feedback']}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"""以下のアウトラインに従って記事を執筆してください。

アウトライン:
{state['outline']}

SEOキーワード（自然に組み込む）:
{', '.join(state.get('seo_keywords', []))}
{feedback_context}

Markdown形式で2000-3000文字程度の記事を書いてください。"""
        }]
    )
    return {"draft": response.content[0].text}

def review_article(state: ContentState) -> ContentState:
    """記事をレビュー"""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""以下の記事を編集者としてレビューしてください。

記事:
{state['draft']}

評価基準:
1. 正確性
2. 読みやすさ
3. SEO最適化
4. 対象読者への適切さ

最初の行に PASS または FAIL を記載し、
FAILの場合は具体的な改善点を列挙してください。"""
        }]
    )
    result = response.content[0].text
    is_pass = result.strip().startswith("PASS")
    return {
        "review_feedback": result,
        "review_pass": is_pass
    }

def finalize_article(state: ContentState) -> ContentState:
    """記事を最終化"""
    return {
        "final_article": state["draft"],
        "metadata": {
            "topic": state["topic"],
            "keywords": state.get("seo_keywords", []),
            "word_count": len(state["draft"]),
        }
    }

def route_review(state: ContentState) -> Literal["revise", "finalize"]:
    return "finalize" if state.get("review_pass", False) else "revise"

# グラフ構築
content_flow = StateGraph(ContentState)
content_flow.add_node("research", research_topic)
content_flow.add_node("seo", extract_seo_keywords)
content_flow.add_node("write", write_draft)
content_flow.add_node("review", review_article)
content_flow.add_node("finalize", finalize_article)

content_flow.set_entry_point("research")
content_flow.add_edge("research", "seo")
content_flow.add_edge("seo", "write")
content_flow.add_edge("write", "review")
content_flow.add_conditional_edges("review", route_review, {
    "revise": "write",
    "finalize": "finalize"
})
content_flow.add_edge("finalize", END)

content_pipeline = content_flow.compile()
```

---

## 10. モニタリングとオブザーバビリティ

### 10.1 ワークフローメトリクス

```python
# ワークフロー実行のメトリクス収集
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

@dataclass
class WorkflowMetrics:
    """ワークフロー実行メトリクスの収集と分析"""

    executions: list[dict] = field(default_factory=list)
    node_durations: dict[str, list[float]] = field(
        default_factory=lambda: defaultdict(list)
    )
    node_failures: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    total_tokens: dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_execution(self, execution_log: list[NodeExecution]):
        """実行ログからメトリクスを記録"""
        execution_data = {
            "timestamp": time.time(),
            "nodes": [],
            "total_duration": 0,
            "success": True
        }

        for ex in execution_log:
            self.node_durations[ex.node_name].append(ex.duration)
            if ex.status == NodeStatus.FAILED:
                self.node_failures[ex.node_name] += 1
                execution_data["success"] = False

            execution_data["nodes"].append({
                "name": ex.node_name,
                "duration": ex.duration,
                "status": ex.status.value
            })
            execution_data["total_duration"] += ex.duration

        self.executions.append(execution_data)

    def get_bottleneck_nodes(self, top_n: int = 3) -> list[dict]:
        """ボトルネックノードを特定"""
        bottlenecks = []
        for node_name, durations in self.node_durations.items():
            if len(durations) >= 3:
                bottlenecks.append({
                    "node": node_name,
                    "avg_duration": statistics.mean(durations),
                    "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
                    "max_duration": max(durations),
                    "failure_rate": (
                        self.node_failures[node_name] / len(durations)
                    )
                })

        return sorted(
            bottlenecks,
            key=lambda x: x["avg_duration"],
            reverse=True
        )[:top_n]

    def generate_report(self) -> str:
        """メトリクスレポートを生成"""
        total = len(self.executions)
        success = sum(1 for e in self.executions if e["success"])

        report = [
            f"=== ワークフローメトリクスレポート ===",
            f"総実行回数: {total}",
            f"成功率: {success/total*100:.1f}%",
            f"",
            f"--- ノード別パフォーマンス ---"
        ]

        for b in self.get_bottleneck_nodes(10):
            report.append(
                f"  {b['node']}: "
                f"平均 {b['avg_duration']:.2f}s, "
                f"P95 {b['p95_duration']:.2f}s, "
                f"失敗率 {b['failure_rate']*100:.1f}%"
            )

        return "\n".join(report)
```

### 10.2 可視化ダッシュボード

```python
# Streamlitベースのワークフロー監視ダッシュボード
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def workflow_dashboard(metrics: WorkflowMetrics):
    """ワークフロー監視ダッシュボード"""
    st.title("ワークフローモニタリング")

    # KPIカード
    col1, col2, col3, col4 = st.columns(4)
    total = len(metrics.executions)
    success = sum(1 for e in metrics.executions if e["success"])

    col1.metric("総実行回数", total)
    col2.metric("成功率", f"{success/max(total,1)*100:.1f}%")
    col3.metric(
        "平均実行時間",
        f"{statistics.mean([e['total_duration'] for e in metrics.executions]):.1f}s"
    )
    col4.metric(
        "アクティブ実行",
        sum(1 for e in metrics.executions
            if time.time() - e["timestamp"] < 60)
    )

    # ノード別実行時間のヒートマップ
    st.subheader("ノード別パフォーマンス")
    bottlenecks = metrics.get_bottleneck_nodes(10)
    if bottlenecks:
        fig = px.bar(
            bottlenecks,
            x="node",
            y="avg_duration",
            color="failure_rate",
            color_continuous_scale="RdYlGn_r",
            title="ノード別平均実行時間と失敗率"
        )
        st.plotly_chart(fig)

    # 実行時間のトレンド
    st.subheader("実行時間トレンド")
    if metrics.executions:
        times = [e["timestamp"] for e in metrics.executions]
        durations = [e["total_duration"] for e in metrics.executions]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[datetime.fromtimestamp(t) for t in times],
            y=durations,
            mode="lines+markers",
            name="実行時間"
        ))
        st.plotly_chart(fig)

# Streamlitアプリとして実行: streamlit run dashboard.py
```

---

## 11. コスト最適化

### 11.1 ノード別モデル選択

```python
# コスト効率を最大化するモデル選択戦略
class ModelSelector:
    """ノードの特性に応じてモデルを選択"""

    MODELS = {
        "fast": {
            "name": "claude-haiku-3-20240307",
            "input_cost": 0.25,   # per 1M tokens
            "output_cost": 1.25,
            "speed": "fast",
            "quality": "good"
        },
        "balanced": {
            "name": "claude-sonnet-4-20250514",
            "input_cost": 3.0,
            "output_cost": 15.0,
            "speed": "medium",
            "quality": "excellent"
        },
        "best": {
            "name": "claude-opus-4-20250514",
            "input_cost": 15.0,
            "output_cost": 75.0,
            "speed": "slow",
            "quality": "best"
        }
    }

    @staticmethod
    def select_for_node(node_type: str, complexity: str = "medium") -> dict:
        """ノードタイプと複雑さに基づいてモデルを選択"""
        selection_matrix = {
            # (node_type, complexity) → model_tier
            ("classify", "low"): "fast",
            ("classify", "medium"): "fast",
            ("classify", "high"): "balanced",
            ("generate", "low"): "fast",
            ("generate", "medium"): "balanced",
            ("generate", "high"): "best",
            ("review", "low"): "fast",
            ("review", "medium"): "balanced",
            ("review", "high"): "best",
            ("summarize", "low"): "fast",
            ("summarize", "medium"): "fast",
            ("summarize", "high"): "balanced",
        }

        tier = selection_matrix.get(
            (node_type, complexity), "balanced"
        )
        return ModelSelector.MODELS[tier]

# コスト追跡付きワークフロー
class CostTracker:
    """ワークフロー全体のコストを追跡"""

    def __init__(self, budget_limit: float = 1.0):
        self.budget_limit = budget_limit  # USD
        self.node_costs: dict[str, float] = {}
        self.total_cost = 0.0

    def track(self, node_name: str, input_tokens: int,
              output_tokens: int, model: str) -> float:
        """コストを記録"""
        model_info = next(
            (m for m in ModelSelector.MODELS.values() if m["name"] == model),
            ModelSelector.MODELS["balanced"]
        )

        cost = (
            input_tokens * model_info["input_cost"] / 1_000_000
            + output_tokens * model_info["output_cost"] / 1_000_000
        )

        self.node_costs[node_name] = self.node_costs.get(node_name, 0) + cost
        self.total_cost += cost

        if self.total_cost > self.budget_limit * 0.8:
            logger.warning(
                f"コスト警告: ${self.total_cost:.4f} / ${self.budget_limit}"
            )

        return cost

    def get_summary(self) -> str:
        """コストサマリー"""
        lines = [f"総コスト: ${self.total_cost:.4f}"]
        for node, cost in sorted(
            self.node_costs.items(), key=lambda x: x[1], reverse=True
        ):
            pct = cost / max(self.total_cost, 0.0001) * 100
            lines.append(f"  {node}: ${cost:.4f} ({pct:.1f}%)")
        return "\n".join(lines)
```

### 11.2 キャッシュ戦略

```python
# ワークフローノードのキャッシュ
import hashlib
from functools import lru_cache

class WorkflowCache:
    """ワークフローノード結果のキャッシュ"""

    def __init__(self, backend: str = "memory", ttl: int = 3600):
        self.ttl = ttl
        self.cache: dict[str, dict] = {}

    def _make_key(self, node_name: str, input_data: str) -> str:
        """キャッシュキーを生成"""
        content = f"{node_name}:{input_data}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, node_name: str, input_data: str) -> Optional[Any]:
        """キャッシュから取得"""
        key = self._make_key(node_name, input_data)
        entry = self.cache.get(key)
        if entry and time.time() - entry["timestamp"] < self.ttl:
            logger.info(f"キャッシュヒット: {node_name}")
            return entry["result"]
        return None

    def set(self, node_name: str, input_data: str, result: Any):
        """キャッシュに保存"""
        key = self._make_key(node_name, input_data)
        self.cache[key] = {
            "result": result,
            "timestamp": time.time(),
            "node": node_name
        }

    def get_stats(self) -> dict:
        """キャッシュ統計"""
        valid = sum(
            1 for e in self.cache.values()
            if time.time() - e["timestamp"] < self.ttl
        )
        return {
            "total_entries": len(self.cache),
            "valid_entries": valid,
            "expired_entries": len(self.cache) - valid
        }

# キャッシュ付きノード実行
cache = WorkflowCache(ttl=1800)  # 30分

def cached_node_handler(node_name: str, handler: Callable) -> Callable:
    """キャッシュ付きのノードハンドラーを生成"""
    def wrapper(state: dict) -> Any:
        # 入力のハッシュをキーにする
        input_key = json.dumps(
            {k: v for k, v in state.items() if not k.endswith("_result")},
            sort_keys=True, default=str
        )

        cached = cache.get(node_name, input_key)
        if cached is not None:
            return cached

        result = handler(state)
        cache.set(node_name, input_key, result)
        return result

    return wrapper
```

---

## 12. テスト

### 12.1 ノード単体テスト

```python
# ワークフローノードのテスト
import pytest
from unittest.mock import patch, MagicMock

class TestWorkflowNodes:
    """ノード単体テスト"""

    def test_classify_billing(self):
        """請求分類のテスト"""
        with patch("anthropic.Anthropic") as mock_client:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="billing")]
            mock_client.return_value.messages.create.return_value = mock_response

            result = classify_inquiry("請求額が間違っています")
            assert result == "billing"

    def test_classify_technical(self):
        """技術分類のテスト"""
        with patch("anthropic.Anthropic") as mock_client:
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="technical")]
            mock_client.return_value.messages.create.return_value = mock_response

            result = classify_inquiry("APIがエラーを返します")
            assert result == "technical"

    def test_routing_condition(self):
        """ルーティング条件のテスト"""
        state = {"classify_result": "billing"}
        route_fn = lambda s: {
            "billing": "handle_billing",
            "technical": "handle_technical",
            "general": "handle_general"
        }.get(s["classify_result"], "handle_general")

        assert route_fn(state) == "handle_billing"

        state["classify_result"] = "unknown"
        assert route_fn(state) == "handle_general"

class TestWorkflowEngine:
    """ワークフローエンジンのテスト"""

    def test_linear_workflow(self):
        """直列ワークフローのテスト"""
        engine = WorkflowEngine("test")

        engine.add_node(WorkflowNode(
            name="step1",
            type=NodeType.LLM,
            handler=lambda s: "result1",
            next_nodes=["step2"]
        ))
        engine.add_node(WorkflowNode(
            name="step2",
            type=NodeType.LLM,
            handler=lambda s: f"result2_{s['step1_result']}",
            next_nodes=[]
        ))

        result = engine.run("step1", {})
        assert result["step1_result"] == "result1"
        assert result["step2_result"] == "result2_result1"

    def test_conditional_workflow(self):
        """条件分岐ワークフローのテスト"""
        engine = WorkflowEngine("test_conditional")

        engine.add_node(WorkflowNode(
            name="classify",
            type=NodeType.LLM,
            handler=lambda s: "A",
            next_nodes=["route"]
        ))
        engine.add_node(WorkflowNode(
            name="route",
            type=NodeType.CONDITION,
            handler=None,
            condition=lambda s: (
                "handle_a" if s["classify_result"] == "A" else "handle_b"
            )
        ))
        engine.add_node(WorkflowNode(
            name="handle_a",
            type=NodeType.LLM,
            handler=lambda s: "handled_A",
            next_nodes=[]
        ))
        engine.add_node(WorkflowNode(
            name="handle_b",
            type=NodeType.LLM,
            handler=lambda s: "handled_B",
            next_nodes=[]
        ))

        result = engine.run("classify", {})
        assert result["handle_a_result"] == "handled_A"
        assert "handle_b_result" not in result

    def test_retry_on_failure(self):
        """リトライのテスト"""
        call_count = 0

        def flaky_handler(state):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("一時的なエラー")
            return "success"

        engine = WorkflowEngine("test_retry")
        engine.add_node(WorkflowNode(
            name="flaky",
            type=NodeType.LLM,
            handler=flaky_handler,
            next_nodes=[],
            retry_count=3
        ))

        result = engine.run("flaky", {})
        assert result["flaky_result"] == "success"
        assert call_count == 3

class TestWorkflowState:
    """状態管理のテスト"""

    def test_state_advance(self):
        state = WorkflowState(user_input="test")
        new_state = state.advance()
        assert new_state.step == 1
        assert state.step == 0  # 元の状態は不変

    def test_state_add_result(self):
        state = WorkflowState(user_input="test")
        new_state = state.add_result("result1")
        assert len(new_state.intermediate_results) == 1
        assert new_state.intermediate_results[0] == "result1"

    def test_state_error_tracking(self):
        state = WorkflowState(user_input="test")
        new_state = state.add_error("connection timeout", "api_call")
        assert len(new_state.errors) == 1
        assert new_state.retry_counts["api_call"] == 1

    def test_cost_estimation(self):
        state = WorkflowState(user_input="test")
        state = state.track_tokens(1000, 500)
        assert state.estimated_cost > 0
```

### 12.2 統合テスト

```python
# ワークフロー統合テスト
class TestSupportWorkflowIntegration:
    """サポートワークフローの統合テスト"""

    @pytest.fixture
    def mock_llm(self):
        """LLMのモック"""
        with patch("anthropic.Anthropic") as mock:
            def create_response(text):
                resp = MagicMock()
                resp.content = [MagicMock(text=text)]
                return resp

            mock.return_value.messages.create.side_effect = [
                create_response("billing"),      # classify
                create_response("請求内容の確認"),  # handle_billing
                create_response("最終回答")        # respond
            ]
            yield mock

    def test_billing_flow(self, mock_llm):
        """請求フローの統合テスト"""
        workflow = build_support_workflow()
        result = workflow.run("classify", {
            "user_message": "先月の請求額が違います"
        })

        assert result["classify_result"] == "billing"
        assert "handle_billing_result" in result
        assert "respond_result" in result

    def test_execution_log(self, mock_llm):
        """実行ログの検証"""
        workflow = build_support_workflow()
        workflow.run("classify", {
            "user_message": "テスト問い合わせ"
        })

        log = workflow.execution_log
        assert len(log) >= 3  # classify, route, handle_*, respond
        assert all(ex.status == NodeStatus.COMPLETED for ex in log)
        assert all(ex.duration >= 0 for ex in log)
```

---

## 13. アンチパターン

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

### アンチパターン3: 無制限ループ

```python
# NG: ループ回数制限なし
def route_review(state):
    if state["review_result"] == "FAIL":
        return "revise"  # 永遠にループする可能性
    return "finalize"

# OK: 最大回数を制限
def route_review(state):
    if state.get("review_count", 0) >= 3:
        return "finalize"  # 3回で強制終了
    if state["review_result"] == "FAIL":
        return "revise"
    return "finalize"
```

### アンチパターン4: エラーハンドリングの欠如

```python
# NG: エラーを握りつぶす
def node_handler(state):
    try:
        return call_llm(state)
    except:
        return ""  # 空文字を返して続行

# OK: 適切なエラーハンドリング
def node_handler(state):
    try:
        return call_llm(state)
    except anthropic.RateLimitError:
        time.sleep(60)
        return call_llm(state)  # リトライ
    except anthropic.APIError as e:
        logger.error(f"API error: {e}")
        raise  # ワークフローエンジンに伝播
```

### アンチパターン5: 全ノードで同じモデル

```python
# NG: 全ノードで最高性能モデルを使用（コスト爆発）
def classify(state):
    return client.messages.create(
        model="claude-opus-4-20250514",  # 分類に高性能モデルは不要
        max_tokens=10,
        messages=[...]
    )

# OK: ノード特性に合わせたモデル選択
def classify(state):
    return client.messages.create(
        model="claude-haiku-3-20240307",  # 分類は高速・安価で十分
        max_tokens=10,
        messages=[...]
    )

def generate_detailed_report(state):
    return client.messages.create(
        model="claude-sonnet-4-20250514",  # 生成は高品質モデル
        max_tokens=4000,
        messages=[...]
    )
```

---

## 14. FAQ

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

### Q4: ワークフローの実行時間が長すぎる場合の対策は？

1. **並列化**: 独立したノードを並列実行する
2. **モデル最適化**: 軽量タスクにはHaikuを使用する
3. **キャッシュ**: 同一入力に対する結果をキャッシュする
4. **タイムアウト**: 各ノードにタイムアウトを設定する
5. **非同期実行**: 時間のかかるノードはバックグラウンドで実行する

### Q5: ワークフローのバージョン管理はどうすべきか？

ワークフロー定義をコードとして管理し、バージョン番号を付与する。実行中のワークフローは作成時のバージョンで動作を保証する。新バージョンへの移行はロールバック可能な形で段階的に行う。

### Q6: ワークフローエージェントとマイクロサービスの関係は？

ワークフローの各ノードは独立したマイクロサービスとして実装可能。ただし、LLMコールを含むノードはステートレスに設計し、状態は外部ストア（Redis/PostgreSQL）で管理するのがベストプラクティス。Kubernetes上ではArgo Workflowsと組み合わせることも有効。

---

## まとめ

| 項目 | 内容 |
|------|------|
| ワークフロー | 事前定義されたフローに従うエージェント |
| DAG | ノード+エッジの有向非巡回グラフ |
| 条件分岐 | LLMの判断で経路を選択 |
| 並列実行 | 独立ノードを同時に処理（Fan-Out/In、Map-Reduce） |
| 状態管理 | 型安全な状態オブジェクトの受け渡し |
| エラーハンドリング | リトライ、サーキットブレーカー、フォールバック |
| チェックポイント | 途中失敗からの再開を保証 |
| コスト最適化 | ノード別モデル選択とキャッシュ |
| モニタリング | 実行メトリクス収集とボトルネック分析 |
| 原則 | 制御性と柔軟性のバランスを取る |

## 次に読むべきガイド

- [03-autonomous-agents.md](./03-autonomous-agents.md) — 自律エージェントの設計
- [../02-implementation/01-langgraph.md](../02-implementation/01-langgraph.md) — LangGraphの詳細実装
- [../03-applications/02-customer-support.md](../03-applications/02-customer-support.md) — サポートワークフローの実例

## 参考文献

1. LangGraph Documentation — https://langchain-ai.github.io/langgraph/
2. Anthropic, "Building effective agents - Workflows" (2024) — https://docs.anthropic.com/en/docs/build-with-claude/agentic
3. AWS, "Step Functions" — https://docs.aws.amazon.com/step-functions/
4. Temporal.io — ワークフローオーケストレーション — https://temporal.io/
5. Prefect — データワークフロー管理 — https://www.prefect.io/
