# エージェントフレームワーク

> LangChain・CrewAI・AutoGen・LangGraph――主要AIエージェントフレームワークの設計思想・機能・トレードオフを比較し、プロジェクトに最適な選択を導く。

## この章で学ぶこと

1. 主要フレームワーク4種の設計思想と得意領域の違い
2. 各フレームワークの実装パターンとコード例
3. プロジェクト要件に基づくフレームワーク選定基準

---

## 1. フレームワーク全体像

```
AIエージェントフレームワーク 生態系 (2025)
+---------------------------------------------------------------+
|                        高レベル                                  |
|  +----------+  +----------+  +----------+  +-----------+       |
|  | CrewAI   |  | AutoGen  |  | Claude   |  | OpenAI    |       |
|  | (役割)   |  | (会話)   |  | Agent SDK|  | Assistants|       |
|  +----+-----+  +----+-----+  +----+-----+  +-----+-----+      |
|       |              |             |              |              |
|  +----v--------------v-------------v--------------v-----+       |
|  |              LangChain / LangGraph                    |       |
|  |              (基盤 + オーケストレーション)              |       |
|  +----+----------------------------------------------+--+       |
|       |                                              |           |
|  +----v-----+  +----------+  +----------+  +--------v--+       |
|  | LLM APIs |  | Vector   |  | Tool     |  | Memory    |       |
|  | (Claude, |  | Stores   |  | Servers  |  | Stores    |       |
|  |  GPT)    |  | (Pinecone|  | (MCP)    |  | (Redis)   |       |
|  +----------+  +----------+  +----------+  +-----------+       |
|                        低レベル                                  |
+---------------------------------------------------------------+
```

---

## 2. LangChain

### 2.1 設計思想

LangChainは **コンポーザブルなビルディングブロック** の思想で構築されている。LLM呼び出し、プロンプトテンプレート、ツール、メモリを個別のコンポーネントとして提供し、それらを自由に組み合わせる。

### 2.2 基本実装

```python
# LangChain でのエージェント構築
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 1. ツール定義
@tool
def calculate(expression: str) -> str:
    """数式を計算して結果を返す。例: '2 + 3 * 4'"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"計算エラー: {e}"

@tool
def search_web(query: str) -> str:
    """Webを検索して結果を返す"""
    # 実際にはSerpAPI等を使用
    return f"'{query}' の検索結果: ..."

# 2. LLM設定
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

# 3. プロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは有能なアシスタントです。ツールを使って正確に回答してください。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# 4. エージェント構築
tools = [calculate, search_web]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. 実行
result = executor.invoke({"input": "日本の人口は何人？ 世界人口の何%？"})
print(result["output"])
```

### 2.3 LangChainのアーキテクチャ

```
LangChain Architecture
+------------------------------------------+
|            AgentExecutor                  |
|  +--------------------------------------+|
|  |  Agent (推論エンジン)                 ||
|  |  +-----------+  +------------------+ ||
|  |  | LLM       |  | Prompt Template  | ||
|  |  +-----------+  +------------------+ ||
|  +--------------------------------------+|
|  +--------------------------------------+|
|  |  Tools (ツール群)                     ||
|  |  [Search] [Calculate] [Code] [DB]    ||
|  +--------------------------------------+|
|  +--------------------------------------+|
|  |  Memory (記憶)                        ||
|  |  [ConversationBuffer] [Summary]      ||
|  +--------------------------------------+|
+------------------------------------------+
```

---

## 3. CrewAI

### 3.1 設計思想

CrewAIは **役割ベースのマルチエージェント** フレームワーク。現実世界のチーム構成をメタファーとして、各エージェントに「役割」「目標」「バックストーリー」を与え、タスクを協調的に遂行する。

### 3.2 基本実装

```python
# CrewAI でのマルチエージェント構築
from crewai import Agent, Task, Crew, Process

# 1. エージェント定義（役割ベース）
researcher = Agent(
    role="シニアリサーチャー",
    goal="AIエージェントの最新トレンドを調査する",
    backstory="10年の研究経験を持つAI研究者。学術論文と産業応用の両方に精通している。",
    tools=[search_tool, scrape_tool],
    llm="claude-sonnet-4-20250514",
    verbose=True
)

writer = Agent(
    role="テクニカルライター",
    goal="調査結果を分かりやすい技術記事にまとめる",
    backstory="技術ブログの編集者として5年の経験がある。複雑な概念を平易に説明する能力に長けている。",
    llm="claude-sonnet-4-20250514",
    verbose=True
)

# 2. タスク定義
research_task = Task(
    description="2025年のAIエージェントの主要トレンドを5つ特定し、それぞれの概要をまとめよ",
    expected_output="トレンド5つのリスト（各200字程度の説明付き）",
    agent=researcher
)

writing_task = Task(
    description="調査結果に基づいて、2000字程度の技術記事を執筆せよ",
    expected_output="マークダウン形式の技術記事",
    agent=writer,
    context=[research_task]  # リサーチ結果を参照
)

# 3. クルー構築・実行
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # 順次実行
    verbose=True
)

result = crew.kickoff()
```

---

## 4. AutoGen

### 4.1 設計思想

AutoGen（Microsoft）は **会話ベースのマルチエージェント** フレームワーク。エージェント同士がチャットメッセージを交換しながらタスクを遂行する。「会話可能エージェント（ConversableAgent）」が基本単位。

### 4.2 基本実装

```python
# AutoGen でのマルチエージェント会話
from autogen import ConversableAgent

# 1. エージェント定義
coder = ConversableAgent(
    name="コーダー",
    system_message="""あなたはPythonの専門家です。
    要件に基づいてコードを書いてください。
    コードはそのまま実行可能な形式で提供してください。""",
    llm_config={"model": "claude-sonnet-4-20250514"}
)

reviewer = ConversableAgent(
    name="レビュアー",
    system_message="""あなたはコードレビューの専門家です。
    コードの品質、セキュリティ、パフォーマンスを評価してください。
    問題があれば具体的な改善案を提示してください。""",
    llm_config={"model": "claude-sonnet-4-20250514"}
)

# 2. 会話（自動的にメッセージを交換）
result = coder.initiate_chat(
    reviewer,
    message="ファイルを読み込んでCSVに変換するPythonスクリプトを書いてください",
    max_turns=4  # 最大4往復
)
```

---

## 5. Claude Agent SDK

### 5.1 設計思想

Anthropicの公式SDK。**シンプルなエージェントループ** を最小限のコードで構築でき、MCPツールとのネイティブ統合が特徴。

```python
# Claude Agent SDK でのエージェント構築
import anthropic

client = anthropic.Anthropic()

# ツール定義
tools = [
    {
        "name": "read_file",
        "description": "指定されたファイルの内容を読み取る",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "ファイルパス"}
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
                "path": {"type": "string"},
                "content": {"type": "string"}
            },
            "required": ["path", "content"]
        }
    }
]

def run_agent(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system="あなたはファイル操作エージェントです。",
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            return extract_text(response)

        # ツール呼び出しを処理
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

---

## 6. フレームワーク比較

### 6.1 機能比較表

| 機能 | LangChain | CrewAI | AutoGen | Claude SDK |
|------|-----------|--------|---------|------------|
| マルチエージェント | LangGraph経由 | ネイティブ | ネイティブ | 手動実装 |
| ツール統合 | 豊富 | LangChain互換 | 独自 | MCP/ネイティブ |
| メモリ管理 | 多種対応 | 基本的 | 会話履歴 | 手動実装 |
| 学習曲線 | 中 | 低 | 低-中 | 低 |
| カスタマイズ性 | 高 | 中 | 中 | 最高 |
| 本番運用実績 | 高 | 中 | 中 | 高 |
| ドキュメント | 充実 | 良好 | 良好 | 充実 |
| コミュニティ | 最大 | 成長中 | 成長中 | 成長中 |

### 6.2 選定フローチャート

```
プロジェクトに最適なフレームワークは？

Q1: 複数エージェントの協調が必要か？
├── YES → Q2: 役割ベースかメッセージベースか？
│   ├── 役割ベース → CrewAI
│   └── メッセージベース → AutoGen
└── NO → Q3: 高度なカスタマイズが必要か？
    ├── YES → Q4: 既存ツールが豊富に必要か？
    │   ├── YES → LangChain + LangGraph
    │   └── NO → Claude Agent SDK
    └── NO → LangChain (基本Agent)
```

---

## 7. アンチパターン

### アンチパターン1: フレームワーク過剰

```python
# NG: 単純なタスクにフレームワークを導入
from crewai import Agent, Task, Crew
# 1つのLLM呼び出しで済むタスクに5つのAgentを作成...

# OK: タスクの複雑さに合わせた選択
# 単純なQ&A → 直接LLM呼び出し
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Pythonのリスト内包表記とは？"}]
)
```

### アンチパターン2: フレームワークロックイン

```python
# NG: フレームワーク固有の機能に依存しすぎ
class MyAgent(LangChainSpecificBaseClass):
    # LangChainの内部APIに深く依存
    pass

# OK: 抽象層を設けて交換可能にする
class AgentInterface(ABC):
    @abstractmethod
    def run(self, goal: str) -> str: ...

class LangChainAgent(AgentInterface):
    def run(self, goal): ...

class ClaudeSDKAgent(AgentInterface):
    def run(self, goal): ...
```

---

## 8. FAQ

### Q1: 初心者にはどのフレームワークがおすすめ？

**Claude Agent SDK** または **CrewAI** がおすすめ。Claude Agent SDKは最小限のコードでエージェントが構築でき、APIの理解がそのまま活きる。CrewAIは直感的な「役割」「タスク」の概念で設計でき、学習曲線が緩やか。

### Q2: LangChainとLangGraphの違いは？

LangChainは **線形的なチェーン** の構築に適し、LangGraphは **状態を持つグラフ（サイクルあり）** の構築に適す。エージェントのようなループ構造にはLangGraphが必要。LangChainのAgentExecutorは内部的にループを実装しているが、複雑なワークフローにはLangGraphを使うべき。

### Q3: 複数のフレームワークを組み合わせてよいか？

可能だが注意が必要。例えば CrewAI の各エージェントが LangChain のツールを使う構成は公式にサポートされている。ただし依存関係が増えるため、デバッグの複雑さとメンテナンスコストは上がる。明確な理由がない限り1つのフレームワークに統一することを推奨する。

---

## まとめ

| 項目 | 内容 |
|------|------|
| LangChain | コンポーザブルなビルディングブロック。エコシステム最大 |
| CrewAI | 役割ベースのマルチエージェント。直感的な設計 |
| AutoGen | 会話ベースのマルチエージェント。自然な対話フロー |
| Claude SDK | 最小限のコードでエージェント構築。MCP統合 |
| 選定基準 | タスク複雑度・マルチエージェント要否・カスタマイズ性 |
| 原則 | タスクに合った最小限の抽象度を選ぶ |

## 次に読むべきガイド

- [02-tool-use.md](./02-tool-use.md) — ツール使用とFunction Callingの詳細
- [../01-patterns/00-single-agent.md](../01-patterns/00-single-agent.md) — シングルエージェントパターン
- [../02-implementation/00-langchain-agent.md](../02-implementation/00-langchain-agent.md) — LangChain実装の詳細

## 参考文献

1. LangChain Documentation — https://python.langchain.com/docs/
2. CrewAI Documentation — https://docs.crewai.com/
3. AutoGen Documentation — https://microsoft.github.io/autogen/
4. Anthropic, "Claude API Reference" — https://docs.anthropic.com/en/api/
