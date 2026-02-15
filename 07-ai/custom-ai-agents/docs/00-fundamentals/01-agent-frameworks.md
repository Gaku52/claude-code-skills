# エージェントフレームワーク

> LangChain・CrewAI・AutoGen・LangGraph――主要AIエージェントフレームワークの設計思想・機能・トレードオフを比較し、プロジェクトに最適な選択を導く。

## この章で学ぶこと

1. 主要フレームワーク4種の設計思想と得意領域の違い
2. 各フレームワークの実装パターンとコード例
3. プロジェクト要件に基づくフレームワーク選定基準
4. フレームワーク移行とベンダーロックイン回避の戦略
5. 各フレームワークのパフォーマンス特性とスケーラビリティ

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

### 1.1 フレームワークの進化の歴史

AIエージェントフレームワークは2023年以降急速に発展してきた。

```
フレームワーク進化タイムライン
2022 Q4  ├── LangChain 初期リリース（チェーン中心）
2023 Q1  ├── LangChain AgentExecutor 追加
2023 Q2  ├── AutoGen (Microsoft) 公開
2023 Q3  ├── CrewAI 初期リリース
2023 Q4  ├── LangGraph 公開（グラフベースオーケストレーション）
2024 Q1  ├── Claude Tool Use GA / OpenAI Assistants API v2
2024 Q2  ├── MCP (Model Context Protocol) 発表
2024 Q3  ├── LangGraph Studio / CrewAI 2.0
2024 Q4  ├── AutoGen 0.4 (大幅リアーキテクチャ)
2025 Q1  ├── Claude Agent SDK / OpenAI Agents SDK
2025 Q2  ├── 各フレームワーク成熟期
```

### 1.2 フレームワーク選択が重要な理由

フレームワーク選択は単なる技術決定ではなく、プロジェクトの成功に直結する戦略的判断である。

```
フレームワーク選択の影響範囲

+------------------+     +--------------------+     +------------------+
| 開発速度          |     | 運用コスト          |     | チーム学習曲線    |
| - プロトタイプ速度 |     | - LLM API費用      |     | - 習得時間        |
| - 機能追加速度    |     | - インフラコスト    |     | - ドキュメント品質 |
| - デバッグ効率    |     | - メンテナンス工数  |     | - コミュニティ規模 |
+--------+---------+     +--------+-----------+     +--------+---------+
         |                         |                          |
         +------------+------------+-----------+--------------+
                      |                        |
              +-------v--------+      +--------v-------+
              | プロジェクト成功 |      | 技術的負債     |
              +----------------+      +----------------+
```

---

## 2. LangChain

### 2.1 設計思想

LangChainは **コンポーザブルなビルディングブロック** の思想で構築されている。LLM呼び出し、プロンプトテンプレート、ツール、メモリを個別のコンポーネントとして提供し、それらを自由に組み合わせる。

LangChainの核心的な設計原則:
- **LCEL (LangChain Expression Language)**: パイプ演算子(`|`)によるチェーン構築
- **Runnable Protocol**: すべてのコンポーネントが `invoke`, `stream`, `batch` を持つ
- **コンポーネント分離**: LLM、プロンプト、ツール、メモリが独立
- **プロバイダー非依存**: 同じコードで Claude、GPT、Gemini を切り替え可能

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

### 2.4 LCEL（LangChain Expression Language）の詳細

LCELはLangChain v0.2以降の推奨パターンであり、宣言的にチェーンを構築できる。

```python
# LCEL によるチェーン構築
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 基本的なLCELチェーン
prompt = ChatPromptTemplate.from_template(
    "以下のトピックについて3行で説明してください: {topic}"
)
llm = ChatAnthropic(model="claude-sonnet-4-20250514")
parser = StrOutputParser()

# パイプ演算子でチェーン構築
chain = prompt | llm | parser
result = chain.invoke({"topic": "機械学習"})

# 複雑なチェーン: 並列実行 + 結合
from langchain_core.runnables import RunnableParallel

analysis_chain = RunnableParallel(
    summary=prompt_summary | llm | parser,
    keywords=prompt_keywords | llm | JsonOutputParser(),
    sentiment=prompt_sentiment | llm | parser
)

# 1回の呼び出しで3つの分析を並列実行
result = analysis_chain.invoke({"text": "分析対象のテキスト..."})
# result = {"summary": "...", "keywords": [...], "sentiment": "positive"}
```

```python
# LCEL でのカスタムロジック組み込み
from langchain_core.runnables import RunnableLambda

def preprocess(input_data: dict) -> dict:
    """前処理: 入力テキストのクリーニング"""
    text = input_data["text"]
    text = text.strip().lower()
    return {"cleaned_text": text, "original": input_data["text"]}

def postprocess(output: str) -> dict:
    """後処理: 出力のフォーマッティング"""
    return {
        "result": output,
        "word_count": len(output.split()),
        "char_count": len(output)
    }

# カスタム関数をチェーンに組み込み
pipeline = (
    RunnableLambda(preprocess)
    | prompt
    | llm
    | parser
    | RunnableLambda(postprocess)
)
```

### 2.5 LangChainのストリーミング対応

```python
# ストリーミング対応のエージェント
import asyncio
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(model="claude-sonnet-4-20250514", streaming=True)

# 同期ストリーミング
for chunk in chain.stream({"topic": "量子コンピュータ"}):
    print(chunk, end="", flush=True)

# 非同期ストリーミング
async def stream_response():
    async for chunk in chain.astream({"topic": "量子コンピュータ"}):
        print(chunk, end="", flush=True)

asyncio.run(stream_response())

# イベントストリーミング（ツール実行含む）
async def stream_agent_events():
    async for event in executor.astream_events(
        {"input": "東京の天気を調べて"},
        version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # LLMからのトークン
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n[ツール開始: {event['name']}]")
        elif kind == "on_tool_end":
            print(f"\n[ツール完了: {event['name']}]")
```

### 2.6 LangChainのメモリ統合

```python
# LangChain でのメモリ統合パターン
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder

# メモリ付きプロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは有能なアシスタントです。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ウィンドウメモリ（直近5往復を保持）
memory = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history",
    return_messages=True
)

# メモリ付きエージェント
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 連続会話
executor.invoke({"input": "私の名前は田中です"})
executor.invoke({"input": "私の名前を覚えていますか？"})
# → "はい、田中さんですね"
```

### 2.7 LangChainのデバッグとトレーシング

```python
# LangSmith によるトレーシング
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# これだけで全てのLLM呼び出し、ツール実行が自動追跡される
result = executor.invoke({"input": "データ分析をして"})

# カスタムコールバックでのデバッグ
from langchain_core.callbacks import BaseCallbackHandler

class DebugCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"[LLM開始] トークン数推定: {sum(len(p) // 4 for p in prompts)}")

    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[ツール開始] {serialized.get('name', 'unknown')}: {input_str[:100]}")

    def on_tool_end(self, output, **kwargs):
        print(f"[ツール完了] 結果長: {len(str(output))} chars")

    def on_llm_error(self, error, **kwargs):
        print(f"[LLMエラー] {type(error).__name__}: {error}")

    def on_chain_end(self, outputs, **kwargs):
        print(f"[チェーン完了] 出力キー: {list(outputs.keys())}")

# コールバック付き実行
result = executor.invoke(
    {"input": "分析して"},
    config={"callbacks": [DebugCallback()]}
)
```

---

## 3. CrewAI

### 3.1 設計思想

CrewAIは **役割ベースのマルチエージェント** フレームワーク。現実世界のチーム構成をメタファーとして、各エージェントに「役割」「目標」「バックストーリー」を与え、タスクを協調的に遂行する。

CrewAIの核心的な設計原則:
- **Role-Playing**: エージェントに「ペルソナ」を与えることで出力品質を向上
- **Task Delegation**: エージェント間のタスク委任を自然に記述
- **Sequential/Hierarchical Process**: チーム構造に応じた実行フロー
- **Memory Integration**: エージェント間の知識共有メカニズム

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

### 3.3 CrewAIの階層的プロセス

```python
# 階層的プロセス: マネージャーがタスクを委任
from crewai import Agent, Task, Crew, Process

# マネージャー（自動で追加される）
manager = Agent(
    role="プロジェクトマネージャー",
    goal="チームを効率的に管理し、高品質な成果物を生み出す",
    backstory="10年のPM経験を持ち、AI開発チームのマネジメントに精通",
    llm="claude-sonnet-4-20250514",
    allow_delegation=True  # 他のエージェントへの委任を許可
)

# チームメンバー
analyst = Agent(
    role="データアナリスト",
    goal="データを分析して洞察を導き出す",
    backstory="統計学の修士号を持ち、PythonとSQLに精通",
    tools=[query_db, create_chart],
    llm="claude-sonnet-4-20250514"
)

developer = Agent(
    role="バックエンド開発者",
    goal="分析結果をAPIとして実装する",
    backstory="FastAPIとPythonでの開発経験5年",
    tools=[read_file, write_file, run_tests],
    llm="claude-sonnet-4-20250514"
)

# 階層的プロセスで実行
crew = Crew(
    agents=[analyst, developer],
    tasks=[analysis_task, development_task, review_task],
    process=Process.hierarchical,  # マネージャーが判断
    manager_agent=manager,
    verbose=True
)

result = crew.kickoff()
```

### 3.4 CrewAIのカスタムツール定義

```python
# CrewAI用カスタムツール
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type

class SearchInput(BaseModel):
    query: str = Field(description="検索クエリ")
    max_results: int = Field(default=5, description="最大結果数")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Webを検索して最新情報を取得する"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, max_results: int = 5) -> str:
        # 実際のWeb検索実装
        import requests
        results = []
        # SerpAPI等を使用して検索
        response = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "num": max_results, "api_key": "..."}
        )
        for item in response.json().get("organic_results", [])[:max_results]:
            results.append(f"- {item['title']}: {item['snippet']}")
        return "\n".join(results) if results else "結果が見つかりませんでした"

# LangChain ツールとの互換性
from langchain.tools import tool as langchain_tool

@langchain_tool
def database_query(sql: str) -> str:
    """SQLクエリを実行してデータベースから結果を取得する"""
    # CrewAIはLangChainツールをそのまま使用可能
    import sqlite3
    conn = sqlite3.connect("data.db")
    result = conn.execute(sql).fetchall()
    conn.close()
    return str(result)
```

### 3.5 CrewAI のメモリシステム

```python
# CrewAI のメモリ設定
from crewai import Crew

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    memory=True,  # メモリを有効化
    # メモリの種類:
    # - Short-term: タスク実行中の会話記憶
    # - Long-term: 過去のタスク結果の記憶
    # - Entity: エンティティ（人名、組織名等）の記憶
    embedder={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    },
    verbose=True
)

# 2回目以降の実行では過去の結果が記憶として活用される
result1 = crew.kickoff(inputs={"topic": "AIエージェント"})
result2 = crew.kickoff(inputs={"topic": "AIエージェントの応用"})
# result2 では result1 の結果が長期記憶として参照される
```

---

## 4. AutoGen

### 4.1 設計思想

AutoGen（Microsoft）は **会話ベースのマルチエージェント** フレームワーク。エージェント同士がチャットメッセージを交換しながらタスクを遂行する。「会話可能エージェント（ConversableAgent）」が基本単位。

AutoGen v0.4 の核心的な設計変更:
- **Actor Model**: エージェントをアクターとして非同期メッセージパッシング
- **Runtime**: エージェントのライフサイクルを管理するランタイム
- **Handoff**: エージェント間の制御移譲パターン
- **Team**: エージェントをチームとして組織化

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

### 4.3 AutoGen v0.4 のアーキテクチャ

```python
# AutoGen v0.4 の新しいAPI（Actor Model ベース）
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

# モデルクライアントの設定
model_client = AnthropicChatCompletionClient(
    model="claude-sonnet-4-20250514"
)

# エージェント定義（v0.4スタイル）
planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    system_message="""あなたはプロジェクトプランナーです。
    要件を分析し、実装計画を立ててください。
    計画が完成したら 'HANDOFF_TO_CODER' と言ってください。"""
)

coder = AssistantAgent(
    name="coder",
    model_client=model_client,
    system_message="""あなたはPython開発者です。
    プランナーの計画に基づいてコードを実装してください。
    実装が完了したら 'HANDOFF_TO_REVIEWER' と言ってください。"""
)

reviewer_agent = AssistantAgent(
    name="reviewer",
    model_client=model_client,
    system_message="""あなたはコードレビュアーです。
    コードの品質を評価してください。
    問題がなければ 'APPROVE' と言ってください。"""
)

# チーム構成
termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat(
    participants=[planner, coder, reviewer_agent],
    termination_condition=termination,
    max_turns=10
)

# 実行
import asyncio

async def main():
    result = await team.run(
        task="CSVファイルを読み込んでデータを分析するスクリプトを作成してください"
    )
    print(result)

asyncio.run(main())
```

### 4.4 AutoGen のコード実行機能

```python
# AutoGen のサンドボックスコード実行
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.agents import CodeExecutorAgent

# Docker ベースのコード実行環境
code_executor = DockerCommandLineCodeExecutor(
    image="python:3.11-slim",
    timeout=60,
    work_dir="/workspace"
)

# コード実行エージェント
executor_agent = CodeExecutorAgent(
    name="executor",
    code_executor=code_executor
)

# コーダーが書いたコードを自動実行
team = RoundRobinGroupChat(
    participants=[coder, executor_agent, reviewer_agent],
    max_turns=8
)
```

### 4.5 AutoGen の Human-in-the-Loop

```python
# AutoGen での人間介入パターン
from autogen_agentchat.agents import UserProxyAgent

# 人間の代理エージェント
human_proxy = UserProxyAgent(
    name="human",
    # 自動で承認するかどうか
    human_input_mode="ALWAYS",  # 常に人間の入力を要求
    # "NEVER": 人間の入力なし
    # "TERMINATE": 終了時のみ
)

# 人間が会話に参加
result = coder.initiate_chat(
    human_proxy,
    message="要件を教えてください",
    max_turns=10
)
```

---

## 5. Claude Agent SDK

### 5.1 設計思想

Anthropicの公式SDK。**シンプルなエージェントループ** を最小限のコードで構築でき、MCPツールとのネイティブ統合が特徴。

Claude Agent SDKの核心的な設計原則:
- **Minimal Abstraction**: フレームワークの抽象度を最低限に保つ
- **Native Tool Use**: Claude APIのtool_useを直接活用
- **MCP First**: MCPプロトコルとのネイティブ統合
- **Full Control**: エージェントループの全ステップを開発者が制御可能

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

### 5.2 Claude Agent SDK の高度な実装パターン

```python
# 高度なエージェントループ（エラーハンドリング、リトライ、コスト追跡付き）
import anthropic
import time
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AgentMetrics:
    """エージェント実行のメトリクス"""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def estimated_cost_usd(self) -> float:
        # Claude Sonnet の概算料金
        input_cost = self.total_input_tokens * 3.0 / 1_000_000
        output_cost = self.total_output_tokens * 15.0 / 1_000_000
        return input_cost + output_cost

class ClaudeAgent:
    def __init__(
        self,
        system_prompt: str,
        tools: list[dict],
        model: str = "claude-sonnet-4-20250514",
        max_turns: int = 20,
        max_retries: int = 3
    ):
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.tools = tools
        self.model = model
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.tool_handlers = {}
        self.metrics = AgentMetrics()

    def register_tool(self, name: str, handler):
        """ツールハンドラーを登録"""
        self.tool_handlers[name] = handler

    def _execute_tool(self, name: str, input_data: dict) -> str:
        """ツールを安全に実行"""
        handler = self.tool_handlers.get(name)
        if not handler:
            return json.dumps({"error": f"ツール '{name}' は登録されていません"})
        try:
            result = handler(**input_data)
            self.metrics.tool_calls += 1
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            self.metrics.errors += 1
            return json.dumps({
                "error": f"{type(e).__name__}: {str(e)}",
                "tool": name,
                "input": input_data
            })

    def run(self, user_message: str) -> dict:
        """エージェントを実行"""
        messages = [{"role": "user", "content": user_message}]
        self.metrics = AgentMetrics()  # メトリクスリセット

        for turn in range(self.max_turns):
            # リトライ付きLLM呼び出し
            response = self._call_llm_with_retry(messages)
            if response is None:
                return {"error": "LLM呼び出しに失敗しました", "metrics": self.metrics}

            self.metrics.llm_calls += 1
            self.metrics.total_input_tokens += response.usage.input_tokens
            self.metrics.total_output_tokens += response.usage.output_tokens

            # 最終回答の場合
            if response.stop_reason == "end_turn":
                text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text += block.text
                return {
                    "output": text,
                    "metrics": self.metrics
                }

            # ツール呼び出しの処理
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self._execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        return {"error": "最大ターン数に達しました", "metrics": self.metrics}

    def _call_llm_with_retry(self, messages: list) -> Optional[object]:
        """リトライ付きLLM呼び出し"""
        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=self.system_prompt,
                    tools=self.tools,
                    messages=messages
                )
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 10
                print(f"レート制限。{wait}秒待機...")
                time.sleep(wait)
            except anthropic.APIError as e:
                print(f"APIエラー (試行 {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                time.sleep(2 ** attempt)
        return None

# 使用例
agent = ClaudeAgent(
    system_prompt="あなたはデータ分析エージェントです。",
    tools=[
        {
            "name": "query_database",
            "description": "SQLクエリを実行してデータを取得する",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": "SQL SELECT文"}
                },
                "required": ["sql"]
            }
        }
    ]
)

agent.register_tool("query_database", lambda sql: execute_sql(sql))
result = agent.run("売上データの月次推移を分析してください")
print(f"結果: {result['output']}")
print(f"コスト: ${result['metrics'].estimated_cost_usd:.4f}")
print(f"ツール呼び出し: {result['metrics'].tool_calls}回")
```

### 5.3 Claude Agent SDK + MCP 統合

```python
# Claude Agent SDK と MCP の統合
import anthropic
import subprocess
import json

class MCPClient:
    """MCPサーバーとのstdio通信クライアント"""

    def __init__(self, command: list[str]):
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self._request_id = 0

    def _send_request(self, method: str, params: dict = None) -> dict:
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {}
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response_line = self.process.stdout.readline()
        return json.loads(response_line)

    def list_tools(self) -> list[dict]:
        """MCPサーバーからツール一覧を取得"""
        response = self._send_request("tools/list")
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """MCPサーバーのツールを実行"""
        response = self._send_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        result = response.get("result", {})
        contents = result.get("content", [])
        return "\n".join(c.get("text", "") for c in contents)

    def close(self):
        self.process.terminate()

# MCP統合エージェント
def run_mcp_agent(user_message: str, mcp_servers: dict[str, list[str]]):
    """MCPサーバー群を使うエージェント"""
    client = anthropic.Anthropic()

    # MCPクライアントを起動
    mcp_clients = {}
    all_tools = []
    tool_to_server = {}

    for name, command in mcp_servers.items():
        mcp = MCPClient(command)
        mcp_clients[name] = mcp

        # ツール一覧を取得してClaude用フォーマットに変換
        for tool in mcp.list_tools():
            claude_tool = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {})
            }
            all_tools.append(claude_tool)
            tool_to_server[tool["name"]] = name

    try:
        messages = [{"role": "user", "content": user_message}]

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system="あなたはMCPツールを活用するエージェントです。",
                tools=all_tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                return extract_text(response)

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    server_name = tool_to_server[block.name]
                    result = mcp_clients[server_name].call_tool(
                        block.name, block.input
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
    finally:
        for mcp in mcp_clients.values():
            mcp.close()

# 使用例
result = run_mcp_agent(
    "プロジェクトのファイルを読み込んで分析して",
    mcp_servers={
        "filesystem": ["npx", "-y", "@anthropic/mcp-filesystem", "/project"],
        "database": ["python", "db_mcp_server.py"]
    }
)
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

### 6.2 パフォーマンス比較

```
各フレームワークのオーバーヘッド（概算）

タスク: 単一ツール実行
+-------------------+--------+----------+---------+
| フレームワーク      | 追加   | メモリ    | 依存    |
|                    | レイテンシ| 使用量   | パッケージ|
+-------------------+--------+----------+---------+
| Claude SDK (直接) | ~5ms   | ~20MB    | 1       |
| LangChain         | ~50ms  | ~100MB   | 20+     |
| CrewAI            | ~80ms  | ~150MB   | 30+     |
| AutoGen           | ~60ms  | ~120MB   | 15+     |
+-------------------+--------+----------+---------+

タスク: 3エージェント協調（5ターン）
+-------------------+--------+----------+---------+
| フレームワーク      | 追加   | メモリ    | LLM    |
|                    | レイテンシ| ピーク   | 呼出回数|
+-------------------+--------+----------+---------+
| Claude SDK (手動) | ~20ms  | ~50MB    | 最小限  |
| LangGraph         | ~100ms | ~200MB   | 最適化可|
| CrewAI            | ~200ms | ~300MB   | 固定    |
| AutoGen           | ~150ms | ~250MB   | 固定    |
+-------------------+--------+----------+---------+
```

### 6.3 選定フローチャート

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

### 6.4 詳細選定マトリクス

| ユースケース | 推奨フレームワーク | 理由 |
|-------------|-------------------|------|
| プロトタイプ / PoC | Claude SDK | 最小コード、最速起動 |
| RAG + チャットボット | LangChain | 豊富なRAGコンポーネント |
| 複数専門家の協調 | CrewAI | 役割定義が直感的 |
| コードレビュー自動化 | AutoGen | 対話的なレビューフロー |
| 複雑なワークフロー | LangGraph | 状態管理+条件分岐 |
| 高カスタマイズAPI | Claude SDK | 完全制御可能 |
| 既存LangChainプロジェクト | LangGraph | シームレスな移行 |
| Human-in-the-Loop | AutoGen / LangGraph | 承認フローの組み込み |

### 6.5 コスト比較（月間推定）

```
月間10,000リクエスト処理時のコスト概算

                    Claude SDK   LangChain   CrewAI   AutoGen
                    ──────────   ─────────   ──────   ───────
LLM API費用         $500        $500        $800     $700
(基本は同じだが、フレームワークの   (内部プロンプト  (エージェント
 オーバーヘッドで追加トークン発生)   が追加)       間通信で追加)

インフラ費用         $50         $100        $100     $100
(依存パッケージ、メモリ使用量差)

開発・保守工数       40h         30h         25h      30h
(フレームワーク機能で省力化)

初期構築工数         20h         10h         8h       12h
(フレームワークの学習+構築)
```

---

## 7. フレームワーク間の移行戦略

### 7.1 抽象層を設けたフレームワーク非依存設計

```python
# フレームワーク非依存のエージェントインターフェース
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import dataclass

@dataclass
class AgentResult:
    """フレームワーク非依存のエージェント結果"""
    output: str
    tool_calls: list[dict]
    metadata: dict

class AgentInterface(ABC):
    """フレームワーク非依存のエージェントインターフェース"""

    @abstractmethod
    def run(self, goal: str, context: dict = None) -> AgentResult:
        """タスクを実行して結果を返す"""
        ...

    @abstractmethod
    def add_tool(self, name: str, description: str, handler: callable):
        """ツールを追加する"""
        ...

    @abstractmethod
    def set_memory(self, memory_store: Any):
        """メモリストアを設定する"""
        ...

class ToolDefinition:
    """フレームワーク非依存のツール定義"""
    def __init__(self, name: str, description: str,
                 parameters: dict, handler: callable):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.handler = handler

    def to_langchain(self):
        """LangChain形式に変換"""
        from langchain.tools import StructuredTool
        return StructuredTool.from_function(
            func=self.handler,
            name=self.name,
            description=self.description
        )

    def to_anthropic(self) -> dict:
        """Anthropic API形式に変換"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters
        }

    def to_openai(self) -> dict:
        """OpenAI API形式に変換"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
```

### 7.2 具体的な移行実装

```python
# LangChain実装
class LangChainAgent(AgentInterface):
    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        from langchain_anthropic import ChatAnthropic
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_core.prompts import ChatPromptTemplate

        self.llm = ChatAnthropic(model=model_name)
        self.tools = []
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "あなたは有能なアシスタントです。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])

    def add_tool(self, name, description, handler):
        from langchain.tools import StructuredTool
        tool = StructuredTool.from_function(
            func=handler, name=name, description=description
        )
        self.tools.append(tool)

    def run(self, goal, context=None):
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        executor = AgentExecutor(agent=agent, tools=self.tools)
        result = executor.invoke({"input": goal})
        return AgentResult(output=result["output"], tool_calls=[], metadata={})

    def set_memory(self, memory_store):
        pass  # LangChain Memory を設定

# Claude SDK実装
class ClaudeSDKAgent(AgentInterface):
    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model_name
        self.tools = []
        self.tool_handlers = {}

    def add_tool(self, name, description, handler):
        # Anthropic API形式のツール定義を作成
        import inspect
        sig = inspect.signature(handler)
        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            properties[param_name] = {"type": "string", "description": param_name}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        self.tools.append({
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        })
        self.tool_handlers[name] = handler

    def run(self, goal, context=None):
        messages = [{"role": "user", "content": goal}]
        all_tool_calls = []

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=self.tools,
                messages=messages
            )

            if response.stop_reason == "end_turn":
                text = "".join(
                    b.text for b in response.content if hasattr(b, "text")
                )
                return AgentResult(
                    output=text,
                    tool_calls=all_tool_calls,
                    metadata={"usage": response.usage}
                )

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.tool_handlers[block.name](**block.input)
                    all_tool_calls.append({
                        "name": block.name,
                        "input": block.input,
                        "result": str(result)
                    })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    def set_memory(self, memory_store):
        pass

# ファクトリーパターンでフレームワークを切り替え
class AgentFactory:
    @staticmethod
    def create(framework: str, **kwargs) -> AgentInterface:
        if framework == "langchain":
            return LangChainAgent(**kwargs)
        elif framework == "claude_sdk":
            return ClaudeSDKAgent(**kwargs)
        else:
            raise ValueError(f"未知のフレームワーク: {framework}")

# 使用例: フレームワークを設定で切り替え
import os
framework = os.environ.get("AGENT_FRAMEWORK", "claude_sdk")
agent = AgentFactory.create(framework)
agent.add_tool("search", "Webを検索する", lambda query: f"検索結果: {query}")
result = agent.run("最新のAIニュースを教えて")
```

---

## 8. 各フレームワークの本番運用パターン

### 8.1 LangChainの本番構成

```python
# LangChain + LangSmith + LangServe の本番構成
from langserve import add_routes
from fastapi import FastAPI
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableWithFallbacks

# フォールバック付きLLM
primary_llm = ChatAnthropic(model="claude-sonnet-4-20250514")
fallback_llm = ChatAnthropic(model="claude-haiku-4-20250514")

llm_with_fallback = primary_llm.with_fallbacks([fallback_llm])

# FastAPIでエージェントをデプロイ
app = FastAPI(title="Agent API")

add_routes(
    app,
    chain,  # LCELチェーンをそのままデプロイ
    path="/agent",
    enable_feedback_endpoint=True,  # フィードバック収集
    enable_public_trace_link_endpoint=True  # トレース共有
)

# ヘルスチェック
@app.get("/health")
async def health():
    return {"status": "healthy", "model": "claude-sonnet-4-20250514"}
```

### 8.2 CrewAI の本番構成

```python
# CrewAI の本番構成
from crewai import Crew
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

# コールバックによる監視
class ProductionCallbacks:
    @staticmethod
    def on_task_start(task):
        logging.info(f"タスク開始: {task.description[:50]}...")

    @staticmethod
    def on_task_end(task, output):
        logging.info(f"タスク完了: {task.description[:50]}... 出力長: {len(str(output))}")

    @staticmethod
    def on_agent_action(agent, action):
        logging.info(f"エージェント '{agent.role}' アクション: {action}")

# タスクの出力をファイルに保存
from crewai import Task

report_task = Task(
    description="分析レポートを作成してください",
    expected_output="構造化されたレポート",
    agent=analyst,
    output_file="output/report.md"  # 出力先ファイル
)
```

### 8.3 エラーハンドリングパターン比較

```python
# 各フレームワークでのエラーハンドリング

# --- LangChain ---
from langchain.agents import AgentExecutor

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # パースエラーを自動リカバリ
    max_iterations=10,           # 最大イテレーション
    early_stopping_method="generate",  # 上限到達時にLLMに要約させる
    return_intermediate_steps=True     # 中間ステップを返す
)

# --- CrewAI ---
from crewai import Crew

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    max_rpm=10,  # 分あたりの最大リクエスト数（レート制限対策）
    share_crew=False,
    step_callback=lambda step: print(f"ステップ: {step}"),
    task_callback=lambda task: print(f"タスク完了: {task}")
)

# --- AutoGen ---
# v0.4 ではチームレベルでの制御
team = RoundRobinGroupChat(
    participants=[planner, coder, reviewer_agent],
    termination_condition=termination,
    max_turns=10  # 無限ループ防止
)

# --- Claude SDK ---
# 完全に手動制御
MAX_TURNS = 20
MAX_TOOL_ERRORS = 3
tool_error_count = 0

for turn in range(MAX_TURNS):
    response = client.messages.create(...)
    if response.stop_reason == "end_turn":
        break
    # ツールエラーのカウントと制御
    for block in response.content:
        if block.type == "tool_use":
            try:
                result = execute_tool(block.name, block.input)
            except Exception:
                tool_error_count += 1
                if tool_error_count >= MAX_TOOL_ERRORS:
                    # エラーが多すぎる場合は中断
                    break
```

---

## 9. 新興フレームワークの動向

### 9.1 注目すべき新興フレームワーク

```
2025年に注目すべきフレームワーク

+------------------+------------------+-----------------------+
| フレームワーク     | 特徴              | 適用場面               |
+------------------+------------------+-----------------------+
| DSPy             | プロンプト自動最適化 | RAG/パイプライン最適化  |
| Semantic Kernel  | エンタープライズ向け | C#/.NET環境           |
| Haystack         | ドキュメント処理特化 | 検索・RAGパイプライン   |
| LlamaIndex       | データ接続特化      | 構造化/非構造化データ   |
| Pydantic AI      | 型安全なエージェント | Python型システム活用   |
| Mastra           | TypeScript特化    | Node.jsエコシステム    |
+------------------+------------------+-----------------------+
```

### 9.2 Pydantic AI の例

```python
# Pydantic AI: 型安全なエージェント構築
from pydantic_ai import Agent
from pydantic import BaseModel

class WeatherResult(BaseModel):
    """天気情報の型定義"""
    city: str
    temperature: float
    condition: str
    humidity: int

# 型付きエージェント
weather_agent = Agent(
    model="anthropic:claude-sonnet-4-20250514",
    result_type=WeatherResult,  # 返り値の型を指定
    system_prompt="あなたは天気情報を提供するエージェントです。"
)

# 実行結果は型安全
result = weather_agent.run_sync("東京の天気を教えて")
print(result.data.city)         # str型が保証
print(result.data.temperature)  # float型が保証
```

---

## 10. アンチパターン

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

**なぜNGか**: フレームワークは抽象度を上げる代わりに、デバッグの複雑さ、依存パッケージの管理コスト、パフォーマンスオーバーヘッドを追加する。単純なタスクにはシンプルなAPI呼び出しで十分。

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

**なぜNGか**: フレームワークは頻繁にAPIが変更される（LangChain v0.1→v0.2の破壊的変更が顕著な例）。抽象層を設けることで、フレームワーク変更時の影響を最小化できる。

### アンチパターン3: フレームワークの機能を再実装

```python
# NG: フレームワークが提供する機能を手動で再実装
class MyCustomMemory:
    # LangChainが提供するメモリ機能と同等のものを自作
    pass

class MyCustomToolExecutor:
    # フレームワークのツール実行機能と同等のものを自作
    pass

# OK: フレームワークの機能を活用し、カスタマイズが必要な部分だけ拡張
from langchain.memory import ConversationBufferWindowMemory

class EnhancedMemory(ConversationBufferWindowMemory):
    """既存のメモリクラスを拡張"""
    def save_context(self, inputs, outputs):
        super().save_context(inputs, outputs)
        # カスタムロジック: 重要な情報を長期記憶にも保存
        self._save_to_long_term(inputs, outputs)
```

### アンチパターン4: 過度なマルチエージェント設計

```python
# NG: 全てをマルチエージェントで解決しようとする
researcher = Agent(role="リサーチャー", ...)
validator = Agent(role="バリデーター", ...)
formatter = Agent(role="フォーマッター", ...)
reviewer = Agent(role="レビュアー", ...)
editor = Agent(role="エディター", ...)
# 5エージェントで5回のLLM呼び出し → コスト5倍

# OK: 必要最小限のエージェント数
# 1つのエージェントが「調査→検証→整形」を一貫して行う方が効率的
agent = Agent(
    role="リサーチャー兼ライター",
    goal="調査から記事作成まで一貫して行う",
    tools=[search_tool, format_tool],
    ...
)
```

---

## 11. トラブルシューティング

### 11.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| LangChainのバージョン不一致 | langchain-core と langchain-community のバージョン差 | `pip install -U langchain langchain-core langchain-community` で統一 |
| CrewAI のエージェントが無限ループ | タスク完了条件が不明確 | `max_iter` パラメータを設定、expected_outputを具体的に |
| AutoGen のメッセージが長すぎる | 会話履歴が肥大化 | `max_turns` を制限、要約機能を有効化 |
| Claude SDK のレート制限 | API呼び出し頻度超過 | 指数バックオフ + キューイングを実装 |
| ツール呼び出しの精度低下 | ツール定義の説明が不十分 | 具体的な使用例、入出力例を説明に追加 |

### 11.2 デバッグチェックリスト

```
フレームワーク問題のデバッグ手順

[ ] 1. バージョン確認
    pip list | grep langchain  (or crewai, autogen)

[ ] 2. 最小再現コードの作成
    フレームワークを除外してAPI直接呼び出しで再現するか確認

[ ] 3. ログレベルの引き上げ
    verbose=True、logging.DEBUG を設定

[ ] 4. トークン使用量の確認
    入力トークンが想定以上に多い場合、
    フレームワークの内部プロンプトが原因の可能性

[ ] 5. ツール定義の検証
    ツールの説明文が曖昧でないか確認

[ ] 6. メモリの状態確認
    メモリが想定通りに更新されているか確認

[ ] 7. ネットワーク/APIの確認
    APIキーの有効性、レート制限の状態を確認
```

---

## 12. 演習

### 演習1（基礎）: フレームワーク評価

以下の要件に対して最適なフレームワークを選定し、理由を説明せよ:

1. 社内FAQ チャットボット（RAG + 単一エージェント）
2. コードレビュー自動化（3人のレビュアーの視点）
3. データパイプラインの監視・自動修復
4. カスタマーサポートの自動エスカレーション

### 演習2（応用）: 抽象層の設計

`AgentInterface` を拡張し、以下の機能を追加するコードを書け:
- ストリーミングサポート
- メトリクス収集
- ツールの動的追加・削除

### 演習3（発展）: フレームワーク移行

LangChain AgentExecutor で実装されたエージェントを、Claude Agent SDK ベースに移行せよ。以下の要件を満たすこと:
- 既存のツール定義を変換
- メモリ機能の移植
- テストの互換性維持

---

## 13. FAQ

### Q1: 初心者にはどのフレームワークがおすすめ？

**Claude Agent SDK** または **CrewAI** がおすすめ。Claude Agent SDKは最小限のコードでエージェントが構築でき、APIの理解がそのまま活きる。CrewAIは直感的な「役割」「タスク」の概念で設計でき、学習曲線が緩やか。

### Q2: LangChainとLangGraphの違いは？

LangChainは **線形的なチェーン** の構築に適し、LangGraphは **状態を持つグラフ（サイクルあり）** の構築に適す。エージェントのようなループ構造にはLangGraphが必要。LangChainのAgentExecutorは内部的にループを実装しているが、複雑なワークフローにはLangGraphを使うべき。

### Q3: 複数のフレームワークを組み合わせてよいか？

可能だが注意が必要。例えば CrewAI の各エージェントが LangChain のツールを使う構成は公式にサポートされている。ただし依存関係が増えるため、デバッグの複雑さとメンテナンスコストは上がる。明確な理由がない限り1つのフレームワークに統一することを推奨する。

### Q4: フレームワークのバージョンアップにどう対応すべきか？

- **依存バージョンの固定**: `requirements.txt` でバージョンを明示
- **抽象層の活用**: フレームワーク固有APIへの直接依存を最小化
- **テストの充実**: フレームワーク更新時にリグレッションを早期発見
- **Changelog監視**: 破壊的変更を事前に把握

### Q5: エンタープライズ環境ではどのフレームワークが適切か？

LangChain（+ LangSmith）が最も成熟している。理由:
- 監視・トレーシングツール（LangSmith）が充実
- デプロイツール（LangServe）が利用可能
- コミュニティとドキュメントが最大
- エンタープライズサポートが利用可能

ただし、Anthropicの Claude を中心にする場合は Claude Agent SDK + MCP が最も効率的。

### Q6: フレームワークなしで十分なケースは？

以下の条件を **すべて** 満たす場合、フレームワークは不要:
- 単一エージェント（マルチエージェント不要）
- ツール数が5個以下
- メモリ要件が単純（会話履歴のみ）
- チームにAPI直接利用の経験がある
- 高度なカスタマイズが必要

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

- [02-tool-use.md](./02-tool-use.md) -- ツール使用とFunction Callingの詳細
- [../01-patterns/00-single-agent.md](../01-patterns/00-single-agent.md) -- シングルエージェントパターン
- [../02-implementation/00-langchain-agent.md](../02-implementation/00-langchain-agent.md) -- LangChain実装の詳細

## 参考文献

1. LangChain Documentation -- https://python.langchain.com/docs/
2. CrewAI Documentation -- https://docs.crewai.com/
3. AutoGen Documentation -- https://microsoft.github.io/autogen/
4. Anthropic, "Claude API Reference" -- https://docs.anthropic.com/en/api/
5. Pydantic AI Documentation -- https://ai.pydantic.dev/
6. LangGraph Documentation -- https://langchain-ai.github.io/langgraph/
7. Model Context Protocol -- https://modelcontextprotocol.io/
