# LangGraph

> 状態グラフ・サイクル・チェックポイント――LangGraphを使った状態管理付きエージェントワークフローの設計と実装。

## この章で学ぶこと

1. LangGraphの状態グラフモデルとLangChain AgentExecutorとの違い
2. サイクル（ループ）を含むグラフの設計と条件分岐の実装
3. チェックポイントによる永続化とヒューマン・イン・ザ・ループの組み込み

---

## 1. LangGraphとは

### 1.1 LangChain AgentExecutor vs LangGraph

```
AgentExecutor: ブラックボックスなループ
+-------------------------------------------+
|  [LLM] → [ツール] → [LLM] → ... → [回答]  |
|  内部のフローを制御できない                  |
+-------------------------------------------+

LangGraph: 明示的な状態グラフ
+-------------------------------------------+
|  [Node A] ──→ [Condition] ──→ [Node B]    |
|       ^              |                     |
|       |              v                     |
|       +────── [Node C] ──→ [END]          |
|  各ノード・エッジを個別に定義・制御          |
+-------------------------------------------+
```

### 1.2 コアコンセプト

```
LangGraph の3つのコアコンセプト

1. State (状態)
   - グラフ全体で共有されるデータ
   - TypedDictまたはPydanticで型定義
   - 各ノードが読み書き

2. Nodes (ノード)
   - 状態を受け取り、更新を返す関数
   - LLM呼び出し、ツール実行、データ変換

3. Edges (エッジ)
   - ノード間の接続
   - 通常エッジ: 無条件で次のノードへ
   - 条件エッジ: 状態に基づいて分岐
```

---

## 2. 基本的なグラフ構築

### 2.1 シンプルなチャットボット

```python
# LangGraph の基本構造
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langchain_anthropic import ChatAnthropic
import operator

# 状態の定義
class ChatState(TypedDict):
    messages: Annotated[list, operator.add]  # メッセージは累積
    response_count: int

# LLM
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# ノード関数
def chatbot(state: ChatState) -> dict:
    """LLMを呼び出してメッセージを追加"""
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "response_count": state["response_count"] + 1
    }

# グラフ構築
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)

# コンパイル
app = graph.compile()

# 実行
result = app.invoke({
    "messages": [("human", "LangGraphについて教えて")],
    "response_count": 0
})
```

### 2.2 ツール使用付きエージェント

```python
# ツール使用付きの完全なエージェント
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, Literal
import operator

# 状態
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]

# ツール
@tool
def get_weather(city: str) -> str:
    """都市の天気を取得する"""
    weather_data = {"東京": "晴れ 22°C", "大阪": "曇り 20°C"}
    return weather_data.get(city, f"{city}のデータなし")

@tool
def calculate(expression: str) -> str:
    """数式を計算する"""
    return str(eval(expression))

tools = [get_weather, calculate]
llm_with_tools = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)

# ノード
def agent_node(state: AgentState) -> dict:
    """LLMがツール使用を判断"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

# ルーティング
def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """ツール呼び出しが必要かを判定"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"

# グラフ構築
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "agent")  # ツール結果→LLMに戻る（サイクル）

app = workflow.compile()

# 実行
result = app.invoke({
    "messages": [HumanMessage(content="東京の天気と、25 * 4 の計算結果を教えて")]
})
```

---

## 3. サイクル（ループ）の設計

### 3.1 レビュー付きサイクル

```
レビュー付きサイクルのグラフ

  START → [生成] → [レビュー] → (合格?) → YES → END
                       ^            |
                       |           NO
                       +←──[修正]←──+
```

```python
# レビュー付きの改善サイクル
class ReviewState(TypedDict):
    task: str
    draft: str
    review: str
    revision_count: int
    is_approved: bool

def generate(state: ReviewState) -> dict:
    """初稿を生成"""
    response = llm.invoke(f"以下のタスクに取り組んでください: {state['task']}")
    return {"draft": response.content, "revision_count": 0}

def review(state: ReviewState) -> dict:
    """ドラフトをレビュー"""
    response = llm.invoke(
        f"以下の文書をレビューしてください。品質が十分なら'APPROVED'、"
        f"改善が必要なら具体的な改善点を指摘:\n\n{state['draft']}"
    )
    is_approved = "APPROVED" in response.content
    return {"review": response.content, "is_approved": is_approved}

def revise(state: ReviewState) -> dict:
    """レビューに基づいて修正"""
    response = llm.invoke(
        f"原文:\n{state['draft']}\n\n"
        f"レビュー:\n{state['review']}\n\n"
        f"レビューの指摘に基づいて修正してください。"
    )
    return {
        "draft": response.content,
        "revision_count": state["revision_count"] + 1
    }

def route_review(state: ReviewState) -> Literal["revise", "end"]:
    if state["is_approved"] or state["revision_count"] >= 3:
        return "end"
    return "revise"

# グラフ
graph = StateGraph(ReviewState)
graph.add_node("generate", generate)
graph.add_node("review", review)
graph.add_node("revise", revise)

graph.add_edge(START, "generate")
graph.add_edge("generate", "review")
graph.add_conditional_edges("review", route_review, {
    "revise": "revise",
    "end": END
})
graph.add_edge("revise", "review")  # サイクル

app = graph.compile()
```

---

## 4. チェックポイントと永続化

### 4.1 メモリベースのチェックポイント

```python
# チェックポイントによる状態の永続化
from langgraph.checkpoint.memory import MemorySaver

# チェックポインタ設定
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# スレッドIDで会話を管理
config = {"configurable": {"thread_id": "user-123"}}

# 1回目の実行
result1 = app.invoke(
    {"messages": [HumanMessage(content="私の名前は田中です")]},
    config=config
)

# 2回目の実行（前回の状態を保持）
result2 = app.invoke(
    {"messages": [HumanMessage(content="私の名前は何ですか？")]},
    config=config  # 同じthread_id → 状態が継続
)
```

### 4.2 SQLiteチェックポイント（永続化）

```python
# SQLiteによる永続チェックポイント
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
app = workflow.compile(checkpointer=checkpointer)

# サーバー再起動後も状態が復元される
```

---

## 5. ヒューマン・イン・ザ・ループ

```python
# 人間の承認を挟むパターン
from langgraph.graph import StateGraph, END, START

class ApprovalState(TypedDict):
    task: str
    plan: str
    approved: bool
    result: str

def create_plan(state: ApprovalState) -> dict:
    plan = llm.invoke(f"タスク: {state['task']}\n実行計画を作成:")
    return {"plan": plan.content}

def execute_plan(state: ApprovalState) -> dict:
    result = llm.invoke(f"計画に従って実行:\n{state['plan']}")
    return {"result": result.content}

def check_approval(state: ApprovalState) -> Literal["execute", "end"]:
    if state.get("approved"):
        return "execute"
    return "end"

graph = StateGraph(ApprovalState)
graph.add_node("plan", create_plan)
graph.add_node("execute", execute_plan)

graph.add_edge(START, "plan")
# interrupt_before で人間の介入ポイントを設定
graph.add_conditional_edges("plan", check_approval, {
    "execute": "execute",
    "end": END
})
graph.add_edge("execute", END)

app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["execute"]  # execute の前で中断
)

# 実行（planで中断）
config = {"configurable": {"thread_id": "approval-1"}}
result = app.invoke({"task": "本番環境にデプロイ"}, config=config)

# 人間が計画を確認 → 承認
app.update_state(config, {"approved": True})

# 再開
result = app.invoke(None, config=config)
```

---

## 6. 比較表

### 6.1 LangGraph vs 他のワークフローツール

| 機能 | LangGraph | Apache Airflow | Temporal | Prefect |
|------|-----------|---------------|----------|---------|
| 対象 | LLMエージェント | データパイプライン | マイクロサービス | データワークフロー |
| サイクル | ネイティブ | 非対応 | 対応 | 非対応 |
| LLM統合 | ネイティブ | プラグイン | 手動 | プラグイン |
| チェックポイント | 組み込み | 組み込み | 組み込み | 組み込み |
| HITL | 組み込み | 外部 | 外部 | 外部 |
| 学習コスト | 中 | 高 | 高 | 中 |

### 6.2 グラフ構造パターン

| パターン | 用途 | サイクル | 複雑度 |
|----------|------|---------|--------|
| 直列グラフ | パイプライン | なし | 低 |
| 分岐グラフ | ルーティング | なし | 中 |
| サイクルグラフ | 改善ループ | あり | 中 |
| サブグラフ | 再利用部品 | あり/なし | 中-高 |
| マルチエージェント | 協調 | あり | 高 |

---

## 7. マルチエージェントグラフ

```python
# LangGraph でのマルチエージェント
class MultiAgentState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str
    task_status: str

# エージェントノード
def researcher(state: MultiAgentState) -> dict:
    """リサーチエージェント"""
    response = research_llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "researcher"}

def coder(state: MultiAgentState) -> dict:
    """コーディングエージェント"""
    response = coding_llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "coder"}

def reviewer(state: MultiAgentState) -> dict:
    """レビューエージェント"""
    response = review_llm.invoke(state["messages"])
    return {"messages": [response], "current_agent": "reviewer"}

# ルーター
def route_to_agent(state: MultiAgentState) -> str:
    last = state["messages"][-1].content
    if "調査が必要" in last:
        return "researcher"
    elif "コードを書いて" in last:
        return "coder"
    elif "レビューして" in last:
        return "reviewer"
    return "end"
```

```
マルチエージェントグラフ

         +──────────→ [Researcher]──+
         |                          |
START → [Router] ──→ [Coder] ──────+──→ [Aggregator] → END
         |                          |
         +──────────→ [Reviewer]───+
```

---

## 8. アンチパターン

### アンチパターン1: 状態の肥大化

```python
# NG: 全データを状態に保持
class BadState(TypedDict):
    messages: list           # 全会話履歴
    all_search_results: list # 全検索結果（巨大）
    all_documents: list      # 全ドキュメント内容（巨大）

# OK: 必要最小限の状態 + 外部ストア参照
class GoodState(TypedDict):
    messages: list
    current_context: str     # 現在のステップに必要な情報のみ
    document_ids: list[str]  # IDのみ保持、内容は外部から取得
```

### アンチパターン2: 深すぎるサイクル

```python
# NG: 無制限のサイクル
def route(state):
    if not state["is_perfect"]:
        return "revise"  # 完璧になるまで無限ループ

# OK: 最大回数を制限
def route(state):
    if not state["is_approved"] and state["revision_count"] < 3:
        return "revise"
    return "end"  # 3回で打ち切り
```

---

## 9. FAQ

### Q1: LangGraphのデバッグ方法は？

- **LangSmith** との統合で実行トレースを可視化
- **verbose出力**: 各ノードの入出力をログ出力
- **ステップ実行**: `app.stream()` で1ステップずつ確認
- **状態スナップショット**: チェックポイントで任意の時点の状態を確認

### Q2: グラフのテスト方法は？

ノード単位でテスト → エッジ単位 → 全体テストの順で:
```python
def test_review_node():
    state = {"draft": "テスト文書", "review": "", "is_approved": False}
    result = review(state)
    assert "review" in result
```

### Q3: LangGraph Cloud とは？

LangGraphの本番デプロイメントサービス。グラフをAPIとして公開し、チェックポイント、スケーリング、モニタリングをマネージドで提供する。LangSmith と統合されている。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 状態グラフ | ノード+エッジ+状態の明示的なグラフ |
| サイクル | ループ（改善サイクル）をネイティブサポート |
| チェックポイント | 状態の永続化と中断・再開 |
| HITL | interrupt_before/after で人間介入 |
| マルチエージェント | ルーターで複数エージェントを協調 |
| 原則 | 状態を最小限に、サイクルに上限を設定 |

## 次に読むべきガイド

- [02-mcp-agents.md](./02-mcp-agents.md) — MCPエージェントの実装
- [03-claude-agent-sdk.md](./03-claude-agent-sdk.md) — Claude Agent SDKの詳細
- [../01-patterns/02-workflow-agents.md](../01-patterns/02-workflow-agents.md) — ワークフロー設計パターン

## 参考文献

1. LangGraph Documentation — https://langchain-ai.github.io/langgraph/
2. LangGraph GitHub — https://github.com/langchain-ai/langgraph
3. LangChain Blog, "LangGraph: Multi-Agent Workflows" — https://blog.langchain.dev/langgraph-multi-agent-workflows/
