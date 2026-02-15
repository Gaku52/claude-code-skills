# LangChainエージェント

> チェーン・プロンプトテンプレート・ツール統合――LangChainを使ったエージェント構築の実践的な実装パターンとベストプラクティス。

## この章で学ぶこと

1. LangChainのコアコンセプト（チェーン、プロンプト、ツール）の理解
2. Tool Calling Agent の構築と AgentExecutor の活用法
3. カスタムツールと高度なプロンプト設計の実装パターン
4. LCEL（LangChain Expression Language）による柔軟なパイプライン構築
5. メモリ管理とコンテキスト制御の戦略
6. 本番運用に向けたエラーハンドリングとオブザーバビリティ
7. コスト最適化とパフォーマンスチューニングの実践手法

---

## 1. LangChainのコアコンポーネント

```
LangChain コンポーネント構成

+---------------------------------------------------+
|                  Application Layer                  |
+---------------------------------------------------+
|  +------------+  +-------------+  +-------------+ |
|  | Agent      |  | Chain       |  | Retriever   | |
|  | Executor   |  | (LCEL)      |  | (RAG)       | |
|  +-----+------+  +------+------+  +------+------+ |
|        |                |                |          |
+--------v----------------v----------------v----------+
|                  Core Components                     |
|  +--------+  +--------+  +--------+  +---------+   |
|  | LLM    |  | Prompt |  | Tools  |  | Memory  |   |
|  |        |  | Templ. |  |        |  |         |   |
|  +--------+  +--------+  +--------+  +---------+   |
+-----------------------------------------------------+
|                  Integrations                        |
|  [Anthropic] [OpenAI] [Chroma] [Pinecone] [...]    |
+-----------------------------------------------------+
```

### 1.1 コンポーネント間の関係

LangChainの設計思想は「コンポーザビリティ」にある。各コンポーネントは独立して機能しつつ、統一されたインターフェースで組み合わせることができる。

```python
# コンポーネント階層の理解
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

# すべてのコンポーネントは Runnable プロトコルを実装
# invoke(), ainvoke(), stream(), astream(), batch(), abatch()
# この統一インターフェースにより、任意の組み合わせが可能

# 例: 各コンポーネントの Runnable としての使い方
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm: Runnable = ChatAnthropic(model="claude-sonnet-4-20250514")
prompt: Runnable = ChatPromptTemplate.from_template("質問: {question}")
parser: Runnable = StrOutputParser()

# Runnable 同士はパイプ演算子で接続
chain: Runnable = prompt | llm | parser

# すべての Runnable メソッドが使える
result = chain.invoke({"question": "LangChainとは？"})
results = chain.batch([{"question": "Q1"}, {"question": "Q2"}])
```

### 1.2 パッケージ構成の理解

```
langchain パッケージエコシステム

langchain-core       ... コアインターフェース・抽象クラス（最も安定）
langchain            ... チェーン・エージェントの実装
langchain-community  ... サードパーティ統合（非公式）
langchain-anthropic  ... Anthropic公式統合
langchain-openai     ... OpenAI公式統合
langchain-chroma     ... Chroma公式統合
langgraph            ... ステートフルなグラフベースワークフロー
langsmith            ... テスト・デバッグ・モニタリング
```

```python
# 推奨パッケージ構成（pyproject.toml）
# [project]
# dependencies = [
#     "langchain-core>=0.3.0,<0.4",
#     "langchain>=0.3.0,<0.4",
#     "langchain-anthropic>=0.3.0,<0.4",
#     "langgraph>=0.2.0,<0.3",
# ]

# バージョン確認
import langchain_core
import langchain
import langchain_anthropic
print(f"langchain-core: {langchain_core.__version__}")
print(f"langchain: {langchain.__version__}")
print(f"langchain-anthropic: {langchain_anthropic.__version__}")
```

---

## 2. LCEL（LangChain Expression Language）

### 2.1 基本チェーン

```python
# LCEL によるチェーン構築
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# コンポーネント定義
llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "あなたは{role}です。{style}で回答してください。"),
    ("human", "{input}")
])

output_parser = StrOutputParser()

# パイプラインで接続
chain = prompt | llm | output_parser

# 実行
result = chain.invoke({
    "role": "Python専門家",
    "style": "簡潔",
    "input": "リスト内包表記の使い方を教えて"
})
print(result)
```

### 2.2 分岐チェーン

```python
# 条件分岐を含むチェーン
from langchain_core.runnables import RunnableBranch, RunnablePassthrough

# 分類器
classifier = (
    ChatPromptTemplate.from_template(
        "以下の質問を 'technical' / 'general' に分類: {input}"
    )
    | llm
    | StrOutputParser()
)

# 分岐
branch = RunnableBranch(
    (
        lambda x: "technical" in x["classification"],
        ChatPromptTemplate.from_template(
            "技術専門家として回答: {input}"
        ) | llm | StrOutputParser()
    ),
    # デフォルト（一般回答）
    ChatPromptTemplate.from_template(
        "一般的な知識で回答: {input}"
    ) | llm | StrOutputParser()
)

# 全体チェーン
full_chain = (
    RunnablePassthrough.assign(
        classification=lambda x: classifier.invoke(x)
    )
    | branch
)
```

### 2.3 並列チェーン（RunnableParallel）

```python
from langchain_core.runnables import RunnableParallel

# 複数の処理を並列実行
parallel_chain = RunnableParallel(
    summary=ChatPromptTemplate.from_template(
        "以下のテキストを3行で要約: {text}"
    ) | llm | StrOutputParser(),

    keywords=ChatPromptTemplate.from_template(
        "以下のテキストからキーワードを5つ抽出（カンマ区切り）: {text}"
    ) | llm | StrOutputParser(),

    sentiment=ChatPromptTemplate.from_template(
        "以下のテキストの感情をpositive/neutral/negativeで判定: {text}"
    ) | llm | StrOutputParser(),
)

# 1回の呼び出しで3つの結果を取得
result = parallel_chain.invoke({
    "text": "LangChainは素晴らしいフレームワークです。"
})
print(result["summary"])
print(result["keywords"])
print(result["sentiment"])
```

### 2.4 RunnableLambda と変換処理

```python
from langchain_core.runnables import RunnableLambda

# カスタム変換ステップ
def format_results(data: dict) -> str:
    """並列実行の結果をフォーマット"""
    return f"""
## 分析結果
**要約**: {data['summary']}
**キーワード**: {data['keywords']}
**感情**: {data['sentiment']}
    """.strip()

# チェーンに組み込み
analysis_pipeline = (
    parallel_chain
    | RunnableLambda(format_results)
)

# RunnableLambda のエラーハンドリング
def safe_parse(text: str) -> dict:
    """パース失敗時にデフォルト値を返す"""
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "パース失敗", "raw": text}

safe_parser = RunnableLambda(safe_parse)
```

### 2.5 フォールバックチェーン

```python
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# プライマリモデル
primary_llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_retries=2
)

# フォールバックモデル
fallback_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

# フォールバック付きLLM
resilient_llm = primary_llm.with_fallbacks([fallback_llm])

# チェーンに組み込み
resilient_chain = prompt | resilient_llm | output_parser

# 特定の例外のみフォールバック
from anthropic import RateLimitError
resilient_llm_selective = primary_llm.with_fallbacks(
    [fallback_llm],
    exceptions_to_handle=(RateLimitError,)
)
```

### 2.6 リトライとレート制限

```python
from langchain_core.runnables import RunnableConfig

# リトライ設定付きチェーン
chain_with_retry = chain.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True
)

# レート制限付き実行
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1.0,
    check_every_n_seconds=0.1,
    max_bucket_size=10
)

llm_with_rate_limit = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    rate_limiter=rate_limiter
)
```

---

## 3. ツール定義と統合

### 3.1 カスタムツールの作成

```python
# 方法1: @tool デコレータ
from langchain.tools import tool
from typing import Optional

@tool
def search_database(
    query: str,
    table: str = "products",
    limit: int = 10
) -> str:
    """SQLiteデータベースを検索する。

    Args:
        query: 検索キーワード
        table: 検索対象テーブル（products, users, orders）
        limit: 最大結果数
    """
    # 実際のDB検索処理
    import sqlite3
    conn = sqlite3.connect("app.db")
    cursor = conn.execute(
        f"SELECT * FROM {table} WHERE name LIKE ? LIMIT ?",
        (f"%{query}%", limit)
    )
    results = cursor.fetchall()
    conn.close()
    return str(results)

# 方法2: StructuredTool（より詳細な制御）
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

class EmailInput(BaseModel):
    to: str = Field(description="送信先メールアドレス")
    subject: str = Field(description="件名")
    body: str = Field(description="本文")

def send_email(to: str, subject: str, body: str) -> str:
    # メール送信処理
    return f"メール送信完了: {to}"

email_tool = StructuredTool.from_function(
    func=send_email,
    name="send_email",
    description="メールを送信する。重要な通知や報告に使用。",
    args_schema=EmailInput,
    return_direct=False
)

# 方法3: BaseTool 継承（最も柔軟）
from langchain.tools import BaseTool

class WebScraperTool(BaseTool):
    name: str = "web_scraper"
    description: str = "指定URLのWebページ内容を取得する"

    def _run(self, url: str) -> str:
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()[:2000]

    async def _arun(self, url: str) -> str:
        # 非同期版
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                text = await response.text()
                return text[:2000]
```

### 3.2 高度なツール定義パターン

```python
# エラーハンドリング付きツール
@tool(handle_tool_error=True)
def risky_operation(query: str) -> str:
    """外部APIに問い合わせる（エラー時は自動リカバリ）。

    Args:
        query: 問い合わせ内容
    """
    import requests
    try:
        response = requests.get(
            f"https://api.example.com/search?q={query}",
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        raise ToolException(f"API呼び出し失敗: {e}")

# カスタムエラーハンドラ
from langchain_core.tools import ToolException

def handle_error(error: ToolException) -> str:
    return f"エラーが発生しました。別の方法を試してください: {str(error)}"

@tool(handle_tool_error=handle_error)
def external_api_call(endpoint: str) -> str:
    """外部APIを呼び出す。

    Args:
        endpoint: APIエンドポイントのパス
    """
    pass

# 非同期専用ツール
from langchain_core.tools import StructuredTool
import asyncio

async def async_web_search(query: str, max_results: int = 5) -> str:
    """非同期でWeb検索を実行する"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.search.example.com/search",
            params={"q": query, "limit": max_results}
        ) as response:
            data = await response.json()
            return str(data["results"])

async_search_tool = StructuredTool.from_function(
    coroutine=async_web_search,
    name="async_web_search",
    description="非同期でWeb検索を実行する"
)

# ツールの動的生成
def create_database_tool(db_path: str, table_name: str) -> BaseTool:
    """データベーステーブルごとにツールを動的に生成"""
    @tool(name=f"query_{table_name}")
    def query_table(condition: str) -> str:
        f"""{table_name}テーブルを検索する。

        Args:
            condition: WHERE句の条件式
        """
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            f"SELECT * FROM {table_name} WHERE {condition} LIMIT 20"
        )
        results = cursor.fetchall()
        conn.close()
        return str(results)
    return query_table

# テーブルごとのツールを自動生成
tables = ["users", "products", "orders", "reviews"]
db_tools = [create_database_tool("app.db", t) for t in tables]
```

### 3.3 ツールの構成パターン

```
ツール構成パターン

1. フラット構成（全ツールを直接提供）
   Agent ── [Tool A, Tool B, Tool C, Tool D]

2. ツールキット構成（カテゴリ別にグループ化）
   Agent ── [DBツールキット] ── [query, insert, update]
        ── [Webツールキット] ── [search, scrape, download]

3. 動的構成（タスクに応じて選択）
   Agent ── TaskClassifier ── coding: [read, write, run]
                           ── research: [search, scrape, summarize]
```

### 3.4 ツールキットの実装

```python
# カスタムツールキットの作成
from langchain_core.tools import BaseToolkit

class DataAnalysisToolkit(BaseToolkit):
    """データ分析用ツールキット"""
    db_path: str

    def get_tools(self) -> list[BaseTool]:
        return [
            self._create_query_tool(),
            self._create_stats_tool(),
            self._create_plot_tool(),
        ]

    def _create_query_tool(self) -> BaseTool:
        db_path = self.db_path

        @tool
        def run_sql_query(query: str) -> str:
            """SQLクエリを実行してデータを取得する。

            Args:
                query: 実行するSQLクエリ（SELECT文のみ許可）
            """
            if not query.strip().upper().startswith("SELECT"):
                return "エラー: SELECT文のみ実行できます"
            import sqlite3
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.execute(query)
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return f"列: {columns}\nデータ: {rows[:50]}"
            except Exception as e:
                return f"クエリエラー: {e}"
            finally:
                conn.close()
        return run_sql_query

    def _create_stats_tool(self) -> BaseTool:
        @tool
        def calculate_statistics(data: str) -> str:
            """数値データの基本統計量を計算する。

            Args:
                data: カンマ区切りの数値データ
            """
            import statistics
            nums = [float(x.strip()) for x in data.split(",")]
            return f"""
平均: {statistics.mean(nums):.2f}
中央値: {statistics.median(nums):.2f}
標準偏差: {statistics.stdev(nums):.2f}
最小値: {min(nums)}
最大値: {max(nums)}
"""
        return calculate_statistics

    def _create_plot_tool(self) -> BaseTool:
        @tool
        def create_chart(
            chart_type: str,
            x_data: str,
            y_data: str,
            title: str = "Chart"
        ) -> str:
            """グラフを作成してファイルに保存する。

            Args:
                chart_type: グラフの種類（bar, line, scatter, pie）
                x_data: X軸データ（カンマ区切り）
                y_data: Y軸データ（カンマ区切り）
                title: グラフのタイトル
            """
            import matplotlib.pyplot as plt
            x = x_data.split(",")
            y = [float(v) for v in y_data.split(",")]

            fig, ax = plt.subplots()
            if chart_type == "bar":
                ax.bar(x, y)
            elif chart_type == "line":
                ax.plot(x, y, marker="o")
            elif chart_type == "scatter":
                ax.scatter(range(len(y)), y)
            elif chart_type == "pie":
                ax.pie(y, labels=x, autopct='%1.1f%%')
            ax.set_title(title)

            filepath = f"/tmp/{title.replace(' ', '_')}.png"
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            return f"グラフを保存: {filepath}"
        return create_chart

# ツールキットの使用
toolkit = DataAnalysisToolkit(db_path="analytics.db")
tools = toolkit.get_tools()
```

---

## 4. AgentExecutor

### 4.1 標準的なエージェント構築

```python
# Tool Calling Agent の構築
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=4096
)

# プロンプト
prompt = ChatPromptTemplate.from_messages([
    ("system", """あなたは有能なリサーチアシスタントです。
ユーザーの質問に対して、必要に応じてツールを使い、正確な情報を提供してください。
情報源がある場合は必ず引用してください。"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# ツール
tools = [search_database, email_tool, WebScraperTool()]

# エージェント作成
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor（実行エンジン）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 実行過程を表示
    max_iterations=15,     # 最大イテレーション
    max_execution_time=120, # タイムアウト（秒）
    handle_parsing_errors=True,  # パースエラーの自動処理
    return_intermediate_steps=True  # 中間ステップも返す
)

# 実行
result = executor.invoke({
    "input": "最新のAIエージェントフレームワークを調べて比較表を作って",
    "chat_history": []
})

print(result["output"])
for step in result["intermediate_steps"]:
    print(f"  Tool: {step[0].tool}, Result: {step[1][:100]}...")
```

### 4.2 ストリーミング実行

```python
# ストリーミングでエージェントの思考過程をリアルタイム表示
async def stream_agent():
    async for event in executor.astream_events(
        {"input": "Pythonの非同期処理について解説して"},
        version="v2"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            # LLMのトークン出力
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind == "on_tool_start":
            # ツール実行開始
            print(f"\n[ツール開始: {event['name']}]")
        elif kind == "on_tool_end":
            # ツール実行完了
            print(f"[ツール完了: {event['name']}]")
```

### 4.3 カスタムAgentの実装

```python
# 完全カスタムAgentの実装
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnableSerializable
from typing import Union

class CustomReasoningAgent(RunnableSerializable):
    """カスタム推論ロジックを持つAgent"""
    llm: ChatAnthropic
    tools: list[BaseTool]
    system_prompt: str
    max_reasoning_steps: int = 5

    def invoke(
        self,
        input: dict,
        config: RunnableConfig | None = None
    ) -> Union[AgentAction, AgentFinish]:
        messages = self._build_messages(input)
        response = self.llm.invoke(messages, config=config)
        return self._parse_response(response, input)

    def _build_messages(self, input: dict) -> list:
        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [SystemMessage(content=self.system_prompt)]

        # 会話履歴
        if "chat_history" in input:
            messages.extend(input["chat_history"])

        messages.append(HumanMessage(content=input["input"]))

        # 中間ステップ
        if "intermediate_steps" in input:
            for action, result in input["intermediate_steps"]:
                messages.append(HumanMessage(
                    content=f"ツール'{action.tool}'の結果: {result}"
                ))

        return messages

    def _parse_response(self, response, input: dict) -> Union[AgentAction, AgentFinish]:
        content = response.content

        # ツール呼び出しの検出
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            return AgentAction(
                tool=tool_call["name"],
                tool_input=tool_call["args"],
                log=f"ツール呼び出し: {tool_call['name']}"
            )

        # 最終回答
        return AgentFinish(
            return_values={"output": content},
            log="最終回答を生成"
        )
```

### 4.4 マルチツール実行パターン

```python
# 複数ツールを1ステップで呼び出す（Parallel Tool Calling）
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0
)

# Claude は1回のレスポンスで複数のtool_callsを返せる
# AgentExecutor はこれらを並列に実行する

@tool
def get_weather(city: str) -> str:
    """指定都市の天気を取得する。

    Args:
        city: 都市名
    """
    # 天気API呼び出し
    return f"{city}: 晴れ 25°C"

@tool
def get_exchange_rate(currency_pair: str) -> str:
    """為替レートを取得する。

    Args:
        currency_pair: 通貨ペア（例: USD/JPY）
    """
    rates = {"USD/JPY": "150.5", "EUR/JPY": "163.2", "GBP/JPY": "190.1"}
    return f"{currency_pair}: {rates.get(currency_pair, '不明')}"

@tool
def get_stock_price(symbol: str) -> str:
    """株価を取得する。

    Args:
        symbol: 銘柄コード
    """
    return f"{symbol}: ¥3,450 (+2.1%)"

tools = [get_weather, get_exchange_rate, get_stock_price]

# この入力に対して、Claudeは3つのツールを同時に呼び出す
# "東京の天気、USD/JPYの為替、7203の株価を教えて"
```

---

## 5. メモリの統合

### 5.1 基本的なメモリ設定

```python
# 会話メモリ付きエージェント
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=10  # 直近10件の会話を保持
)

executor_with_memory = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 連続的な対話
executor_with_memory.invoke({"input": "PythonのFastAPIについて教えて"})
executor_with_memory.invoke({"input": "それとFlaskの違いは？"})  # 文脈を保持
```

### 5.2 メモリ戦略の比較と実装

```python
# 1. ConversationBufferMemory: 全会話を保持（メモリ使用量に注意）
from langchain.memory import ConversationBufferMemory
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 2. ConversationBufferWindowMemory: 直近N件のみ保持
from langchain.memory import ConversationBufferWindowMemory
window_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=20  # 直近20ターン
)

# 3. ConversationSummaryMemory: 要約して保持（長期会話向け）
from langchain.memory import ConversationSummaryMemory
summary_memory = ConversationSummaryMemory(
    llm=ChatAnthropic(model="claude-haiku-4-20250514"),
    memory_key="chat_history",
    return_messages=True
)

# 4. ConversationSummaryBufferMemory: 要約+直近のバッファ
from langchain.memory import ConversationSummaryBufferMemory
summary_buffer_memory = ConversationSummaryBufferMemory(
    llm=ChatAnthropic(model="claude-haiku-4-20250514"),
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000  # このトークン数を超えると要約
)

# 5. 永続化メモリ（Redis使用）
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

redis_history = RedisChatMessageHistory(
    session_id="user-123-session-456",
    url="redis://localhost:6379",
    ttl=3600  # 1時間で期限切れ
)

persistent_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    chat_memory=redis_history,
    return_messages=True,
    k=50
)
```

### 5.3 セマンティックメモリ

```python
# ベクトルDBを使ったセマンティック検索メモリ
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_anthropic import ChatAnthropic

# ベクトルストアの設定
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

vectorstore = Chroma(
    collection_name="conversation_memory",
    embedding_function=embeddings,
    persist_directory="./memory_db"
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}  # 関連度上位5件を取得
)

semantic_memory = VectorStoreRetrieverMemory(
    retriever=retriever,
    memory_key="relevant_history"
)

# 過去の会話から関連する内容を自動的に検索して提供
```

---

## 6. 出力パーサー

### 6.1 構造化出力

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Optional

# 出力スキーマの定義
class TaskAnalysis(BaseModel):
    task_name: str = Field(description="タスクの名前")
    priority: str = Field(description="優先度（high/medium/low）")
    estimated_hours: float = Field(description="推定所要時間")
    dependencies: list[str] = Field(description="依存タスクのリスト")
    risks: Optional[list[str]] = Field(description="リスク要因", default=None)

# JSON出力パーサー
parser = JsonOutputParser(pydantic_object=TaskAnalysis)

prompt = ChatPromptTemplate.from_messages([
    ("system", "タスクを分析して以下のフォーマットで回答してください。\n{format_instructions}"),
    ("human", "タスク: {task_description}")
])

chain = prompt | llm | parser

result = chain.invoke({
    "task_description": "ECサイトの決済機能のリファクタリング",
    "format_instructions": parser.get_format_instructions()
})

print(f"タスク: {result['task_name']}")
print(f"優先度: {result['priority']}")
print(f"推定時間: {result['estimated_hours']}時間")
```

### 6.2 ストリーミング対応パーサー

```python
from langchain_core.output_parsers import JsonOutputParser

# ストリーミング中の部分的なJSONをパース
async def stream_structured_output():
    parser = JsonOutputParser()

    chain = prompt | llm | parser

    async for partial_result in chain.astream({
        "task_description": "API設計のレビュー"
    }):
        # 部分的な結果が徐々に構築される
        print(f"Current state: {partial_result}")
```

### 6.3 カスタム出力パーサー

```python
from langchain_core.output_parsers import BaseOutputParser
import re

class MarkdownTableParser(BaseOutputParser[list[dict]]):
    """Markdownテーブルをパースするカスタムパーサー"""

    def parse(self, text: str) -> list[dict]:
        lines = text.strip().split("\n")

        # ヘッダー行を探す
        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if "|" in line and "---" not in line:
                if header_line is None:
                    header_line = line
                    data_start = i + 2  # セパレータ行をスキップ
                    break

        if header_line is None:
            return []

        headers = [h.strip() for h in header_line.split("|") if h.strip()]

        results = []
        for line in lines[data_start:]:
            if "|" not in line:
                continue
            values = [v.strip() for v in line.split("|") if v.strip()]
            if len(values) == len(headers):
                results.append(dict(zip(headers, values)))

        return results

    @property
    def _type(self) -> str:
        return "markdown_table"

# 使用例
table_parser = MarkdownTableParser()
comparison_chain = (
    ChatPromptTemplate.from_template(
        "以下の項目をMarkdownテーブルで比較してください: {items}"
    )
    | llm
    | table_parser
)

result = comparison_chain.invoke({"items": "React, Vue, Angular"})
for row in result:
    print(row)
```

---

## 7. コールバックとオブザーバビリティ

### 7.1 カスタムコールバック

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from datetime import datetime
import json
import logging

logger = logging.getLogger("langchain_agent")

class ProductionCallbackHandler(BaseCallbackHandler):
    """本番環境用のコールバックハンドラ"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = None
        self.tool_calls = []
        self.token_usage = {"input": 0, "output": 0}
        self.errors = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.start_time = datetime.now()
        logger.info(f"[{self.session_id}] Chain started")

    def on_chain_end(self, outputs, **kwargs):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(
            f"[{self.session_id}] Chain completed in {elapsed:.2f}s | "
            f"Tool calls: {len(self.tool_calls)} | "
            f"Tokens: {self.token_usage}"
        )

    def on_agent_action(self, action: AgentAction, **kwargs):
        self.tool_calls.append({
            "tool": action.tool,
            "input": str(action.tool_input)[:200],
            "timestamp": datetime.now().isoformat()
        })
        logger.info(
            f"[{self.session_id}] Tool call: {action.tool}"
        )

    def on_tool_error(self, error: Exception, **kwargs):
        self.errors.append({
            "error": str(error),
            "timestamp": datetime.now().isoformat()
        })
        logger.error(
            f"[{self.session_id}] Tool error: {error}"
        )

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            self.token_usage["input"] += usage.get("input_tokens", 0)
            self.token_usage["output"] += usage.get("output_tokens", 0)

    def get_metrics(self) -> dict:
        """メトリクスを取得"""
        return {
            "session_id": self.session_id,
            "duration": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            "tool_calls": len(self.tool_calls),
            "errors": len(self.errors),
            "token_usage": self.token_usage,
            "tool_details": self.tool_calls
        }

# 使用
callback = ProductionCallbackHandler(session_id="sess-abc123")
result = executor.invoke(
    {"input": "売上データを分析して"},
    config={"callbacks": [callback]}
)
metrics = callback.get_metrics()
print(json.dumps(metrics, indent=2, ensure_ascii=False))
```

### 7.2 LangSmith統合

```python
# LangSmithによるトレーシング
import os

# 環境変数で設定
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls__..."
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# プログラマティックに設定
from langsmith import Client

client = Client()

# トレースのアノテーション
from langchain_core.tracers import LangChainTracer

tracer = LangChainTracer(
    project_name="production-agent",
    client=client
)

# 実行時にトレーサーを渡す
result = executor.invoke(
    {"input": "月次レポートを作成して"},
    config={"callbacks": [tracer]}
)

# カスタムメタデータの追加
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    metadata={
        "user_id": "user-123",
        "request_type": "report_generation",
        "environment": "production"
    },
    tags=["production", "report", "monthly"]
)

result = executor.invoke(
    {"input": "月次レポートを作成して"},
    config=config
)
```

### 7.3 構造化ログの実装

```python
import structlog
from langchain_core.callbacks import BaseCallbackHandler

# structlog の設定
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(ensure_ascii=False)
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)

log = structlog.get_logger()

class StructuredLoggingCallback(BaseCallbackHandler):
    """構造化ログを出力するコールバック"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.step_count = 0

    def on_agent_action(self, action, **kwargs):
        self.step_count += 1
        log.info(
            "agent_action",
            request_id=self.request_id,
            step=self.step_count,
            tool=action.tool,
            tool_input=str(action.tool_input)[:100]
        )

    def on_agent_finish(self, finish, **kwargs):
        log.info(
            "agent_finish",
            request_id=self.request_id,
            total_steps=self.step_count,
            output_length=len(finish.return_values.get("output", ""))
        )

    def on_tool_error(self, error, **kwargs):
        log.error(
            "tool_error",
            request_id=self.request_id,
            step=self.step_count,
            error=str(error)
        )
```

---

## 8. 本番運用パターン

### 8.1 FastAPI統合

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import uuid

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False

class ChatResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: list[dict]
    token_usage: dict

# セッション管理
sessions: dict[str, ConversationBufferWindowMemory] = {}

def get_or_create_session(session_id: str | None) -> tuple[str, ConversationBufferWindowMemory]:
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True,
        k=20
    )
    sessions[new_id] = memory
    return new_id, memory

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id, memory = get_or_create_session(request.session_id)
    callback = ProductionCallbackHandler(session_id=session_id)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        max_iterations=10,
        max_execution_time=60,
        handle_parsing_errors=True,
        return_intermediate_steps=True
    )

    try:
        result = await executor.ainvoke(
            {"input": request.message},
            config={"callbacks": [callback]}
        )

        metrics = callback.get_metrics()

        return ChatResponse(
            session_id=session_id,
            response=result["output"],
            tool_calls=metrics["tool_details"],
            token_usage=metrics["token_usage"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE（Server-Sent Events）でストリーミング応答"""
    session_id, memory = get_or_create_session(request.session_id)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        max_iterations=10,
        handle_parsing_errors=True
    )

    async def event_generator():
        async for event in executor.astream_events(
            {"input": request.message},
            version="v2"
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    yield f"data: {json.dumps({'type': 'token', 'content': content}, ensure_ascii=False)}\n\n"
            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'tool_start', 'tool': event['name']}, ensure_ascii=False)}\n\n"
            elif kind == "on_tool_end":
                yield f"data: {json.dumps({'type': 'tool_end', 'tool': event['name']}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "deleted"}
```

### 8.2 エラーハンドリング戦略

```python
from langchain_core.runnables import RunnableConfig
from tenacity import retry, stop_after_attempt, wait_exponential
import traceback

class RobustAgentExecutor:
    """堅牢なエージェント実行クラス"""

    def __init__(
        self,
        agent,
        tools: list,
        max_iterations: int = 10,
        max_execution_time: int = 120,
        fallback_response: str = "申し訳ございません。現在処理できません。"
    ):
        self.executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_iterations,
            max_execution_time=max_execution_time,
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )
        self.fallback_response = fallback_response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30)
    )
    async def _execute_with_retry(
        self,
        input_data: dict,
        config: RunnableConfig
    ) -> dict:
        return await self.executor.ainvoke(input_data, config=config)

    async def run(
        self,
        message: str,
        session_id: str = "default",
        metadata: dict = None
    ) -> dict:
        config = RunnableConfig(
            metadata=metadata or {},
            tags=[session_id]
        )

        try:
            result = await self._execute_with_retry(
                {"input": message},
                config=config
            )
            return {
                "success": True,
                "output": result["output"],
                "steps": len(result.get("intermediate_steps", [])),
            }
        except Exception as e:
            logger.error(
                "agent_execution_failed",
                session_id=session_id,
                error=str(e),
                traceback=traceback.format_exc()
            )
            return {
                "success": False,
                "output": self.fallback_response,
                "error": str(e)
            }
```

### 8.3 セキュリティ対策

```python
# ツール実行のサンドボックス化
import subprocess
import tempfile
import os

@tool
def safe_code_execution(code: str, language: str = "python") -> str:
    """コードを安全に実行する（サンドボックス環境）。

    Args:
        code: 実行するコード
        language: プログラミング言語（python, javascript）
    """
    # 危険なパターンのチェック
    dangerous_patterns = [
        "import os", "import subprocess", "import shutil",
        "__import__", "eval(", "exec(", "open(",
        "rm -rf", "sudo", "chmod"
    ]

    for pattern in dangerous_patterns:
        if pattern in code:
            return f"セキュリティエラー: '{pattern}' は許可されていません"

    # 一時ファイルに書き出して実行
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{language[:2]}", delete=False
    ) as f:
        f.write(code)
        f.flush()

        try:
            result = subprocess.run(
                ["python", f.name] if language == "python" else ["node", f.name],
                capture_output=True,
                text=True,
                timeout=10,  # 10秒タイムアウト
                cwd="/tmp",
                env={
                    "PATH": "/usr/bin:/usr/local/bin",
                    "HOME": "/tmp"
                }  # 最小限の環境変数
            )

            if result.returncode != 0:
                return f"実行エラー:\n{result.stderr[:500]}"
            return result.stdout[:2000]
        except subprocess.TimeoutExpired:
            return "タイムアウト: 実行時間が10秒を超えました"
        finally:
            os.unlink(f.name)

# 入力バリデーション
from pydantic import BaseModel, Field, validator

class SafeQueryInput(BaseModel):
    query: str = Field(max_length=1000, description="検索クエリ")

    @validator("query")
    def validate_query(cls, v):
        # SQLインジェクション対策
        injection_patterns = [
            "DROP", "DELETE", "INSERT", "UPDATE",
            "--", ";", "UNION", "OR 1=1"
        ]
        upper_v = v.upper()
        for pattern in injection_patterns:
            if pattern in upper_v:
                raise ValueError(f"不正な入力パターン: {pattern}")
        return v
```

---

## 9. キャッシュとパフォーマンス最適化

### 9.1 LLMキャッシュ

```python
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache, RedisCache

# SQLiteキャッシュ（開発環境向け）
set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))

# Redisキャッシュ（本番環境向け）
import redis
redis_client = redis.Redis(host="localhost", port=6379)
set_llm_cache(RedisCache(redis_=redis_client, ttl=3600))

# セマンティックキャッシュ（類似クエリのキャッシュ）
from langchain_community.cache import RedisSemanticCache
from langchain_community.embeddings import HuggingFaceEmbeddings

set_llm_cache(RedisSemanticCache(
    redis_url="redis://localhost:6379",
    embedding=HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base"
    ),
    score_threshold=0.95  # 類似度95%以上でキャッシュヒット
))

# 特定のチェーンでキャッシュを無効化
from langchain_core.runnables import RunnableConfig

result = chain.invoke(
    {"input": "最新ニュースを教えて"},
    config=RunnableConfig(
        metadata={"cache": False}  # キャッシュを使わない
    )
)
```

### 9.2 バッチ処理と並列実行

```python
import asyncio
from langchain_core.runnables import RunnableConfig

# バッチ処理
inputs = [
    {"input": f"質問{i}: {q}"}
    for i, q in enumerate([
        "Pythonのデコレータとは？",
        "asyncioの使い方は？",
        "型ヒントのベストプラクティスは？",
        "テストの書き方は？",
    ])
]

# 同期バッチ（内部で並列実行）
results = chain.batch(
    inputs,
    config=RunnableConfig(max_concurrency=3)  # 同時実行数を制限
)

# 非同期バッチ
async def process_batch():
    results = await chain.abatch(
        inputs,
        config=RunnableConfig(max_concurrency=5)
    )
    return results

# ストリーミングバッチ
async def stream_batch():
    """各入力の結果をストリーミングで取得"""
    tasks = [
        chain.astream(input_data)
        for input_data in inputs
    ]

    for i, stream in enumerate(asyncio.as_completed(tasks)):
        result = await stream
        print(f"Input {i} completed")
```

### 9.3 モデルルーティング

```python
from langchain_core.runnables import RunnableLambda, RunnableBranch

# タスクの複雑さに応じてモデルを切り替え
cheap_llm = ChatAnthropic(model="claude-haiku-4-20250514", temperature=0)
expensive_llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)

def estimate_complexity(input_data: dict) -> dict:
    """タスクの複雑さを推定"""
    text = input_data.get("input", "")
    word_count = len(text.split())

    # 単純なヒューリスティクス
    complex_keywords = ["分析", "比較", "設計", "アーキテクチャ", "最適化"]
    has_complex_keyword = any(kw in text for kw in complex_keywords)

    input_data["complexity"] = "high" if (
        word_count > 50 or has_complex_keyword
    ) else "low"
    return input_data

# モデルルーティングチェーン
routed_chain = (
    RunnableLambda(estimate_complexity)
    | RunnableBranch(
        (
            lambda x: x["complexity"] == "high",
            ChatPromptTemplate.from_template("{input}") | expensive_llm | StrOutputParser()
        ),
        ChatPromptTemplate.from_template("{input}") | cheap_llm | StrOutputParser()
    )
)

# コスト削減: 単純な質問はHaiku、複雑な質問はSonnet
```

---

## 10. テストとデバッグ

### 10.1 ユニットテスト

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

# ツールのテスト
class TestSearchDatabaseTool:
    def test_basic_search(self, tmp_path):
        """基本的な検索が動作する"""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE products (id INT, name TEXT)")
        conn.execute("INSERT INTO products VALUES (1, 'Widget')")
        conn.commit()
        conn.close()

        result = search_database.invoke({
            "query": "Widget",
            "table": "products",
            "limit": 5
        })
        assert "Widget" in result

    def test_empty_results(self, tmp_path):
        """結果が空の場合"""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE products (id INT, name TEXT)")
        conn.commit()
        conn.close()

        result = search_database.invoke({
            "query": "nonexistent",
            "table": "products"
        })
        assert result == "[]"

# エージェントのテスト（LLMをモック）
class TestAgent:
    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.invoke.return_value = AIMessage(content="テスト回答")
        return mock

    def test_agent_responds(self, mock_llm):
        """エージェントが応答を返す"""
        chain = (
            ChatPromptTemplate.from_template("{input}")
            | mock_llm
            | StrOutputParser()
        )
        result = chain.invoke({"input": "テスト質問"})
        assert result is not None

# チェーンの統合テスト
class TestChain:
    @pytest.mark.integration
    def test_full_chain_execution(self):
        """チェーン全体が正しく動作する"""
        llm = ChatAnthropic(
            model="claude-haiku-4-20250514",
            temperature=0
        )
        chain = (
            ChatPromptTemplate.from_template(
                "「{word}」を使った短い文を1つ作って"
            )
            | llm
            | StrOutputParser()
        )

        result = chain.invoke({"word": "テスト"})
        assert len(result) > 0
        assert "テスト" in result
```

### 10.2 LangSmithによるテスト自動化

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# テストデータセットの作成
dataset = client.create_dataset(
    "agent-evaluation",
    description="エージェントの品質評価用データセット"
)

# テストケースの追加
examples = [
    {
        "input": "東京の天気は？",
        "expected_output": "天気情報を返す",
        "expected_tools": ["get_weather"]
    },
    {
        "input": "1+1は？",
        "expected_output": "2",
        "expected_tools": []  # ツール不要
    },
]

for example in examples:
    client.create_example(
        inputs={"input": example["input"]},
        outputs={
            "output": example["expected_output"],
            "tools": example["expected_tools"]
        },
        dataset_id=dataset.id
    )

# 評価関数
def correct_tool_usage(run, example) -> dict:
    """正しいツールが使われたかを評価"""
    expected_tools = example.outputs.get("tools", [])

    # 中間ステップからツール呼び出しを抽出
    actual_tools = []
    if run.outputs and "intermediate_steps" in run.outputs:
        actual_tools = [
            step[0].tool
            for step in run.outputs["intermediate_steps"]
        ]

    return {
        "key": "correct_tools",
        "score": 1.0 if set(expected_tools) == set(actual_tools) else 0.0
    }

# 評価の実行
results = evaluate(
    lambda input: executor.invoke(input),
    data=dataset.name,
    evaluators=[correct_tool_usage],
    experiment_prefix="agent-v1"
)
```

### 10.3 デバッグテクニック

```python
# 1. 詳細ログの有効化
import langchain
langchain.debug = True  # 全ステップの詳細ログ

# 2. 特定ステップのデバッグ
from langchain_core.tracers import ConsoleCallbackHandler

# コンソールに全ステップを出力
result = chain.invoke(
    {"input": "テスト"},
    config={"callbacks": [ConsoleCallbackHandler()]}
)

# 3. チェーンの可視化
chain.get_graph().print_ascii()
# 出力例:
#     +--------+
#     | Prompt |
#     +---+----+
#         |
#     +---v---+
#     |  LLM  |
#     +---+---+
#         |
#     +---v----+
#     | Parser |
#     +--------+

# 4. 中間結果のインスペクション
from langchain_core.runnables import RunnableLambda

def inspect(state):
    """中間状態をログに出力"""
    print(f"=== Inspect ===")
    print(f"Type: {type(state)}")
    if isinstance(state, dict):
        for k, v in state.items():
            print(f"  {k}: {str(v)[:100]}")
    else:
        print(f"  Value: {str(state)[:200]}")
    print(f"================")
    return state

debug_chain = (
    prompt
    | RunnableLambda(inspect)  # プロンプト後
    | llm
    | RunnableLambda(inspect)  # LLM出力後
    | output_parser
)
```

---

## 11. 比較表

### 11.1 エージェント作成方法の比較

| 方法 | コード量 | 柔軟性 | ツール方式 | 推奨場面 |
|------|---------|--------|-----------|---------|
| create_tool_calling_agent | 少 | 中 | Function Calling | 一般的 |
| create_react_agent | 少 | 中 | テキストベース | レガシーモデル |
| カスタムAgent | 多 | 高 | 任意 | 特殊要件 |
| LangGraph | 中 | 最高 | 任意 | 複雑なフロー |

### 11.2 ツール定義方法の比較

| 方法 | 手軽さ | 型安全性 | 非同期 | バリデーション |
|------|--------|---------|--------|--------------|
| @tool デコレータ | 最高 | 中 | 可 | 基本 |
| StructuredTool | 中 | 高 | 可 | Pydantic |
| BaseTool 継承 | 低 | 高 | 可 | 完全制御 |
| Tool.from_function | 高 | 低 | 不可 | なし |

### 11.3 メモリ方式の比較

| 方式 | メモリ使用量 | 長期対話 | 精度 | コスト | 推奨用途 |
|------|------------|---------|------|--------|---------|
| ConversationBufferMemory | 高 | 不向き | 最高 | 高 | 短い会話 |
| ConversationBufferWindowMemory | 中 | 可 | 高 | 中 | 一般的な対話 |
| ConversationSummaryMemory | 低 | 最適 | 中 | LLM呼び出しあり | 長時間セッション |
| ConversationSummaryBufferMemory | 中 | 最適 | 高 | LLM呼び出しあり | バランス重視 |
| VectorStoreRetrieverMemory | 低 | 最適 | 検索依存 | 埋め込み計算 | ナレッジベース |

### 11.4 キャッシュ方式の比較

| 方式 | 速度 | 永続性 | スケーラビリティ | セマンティック | 推奨環境 |
|------|------|--------|----------------|--------------|---------|
| InMemoryCache | 最速 | なし | 単一プロセス | 不可 | 開発 |
| SQLiteCache | 速い | あり | 単一マシン | 不可 | 小規模本番 |
| RedisCache | 速い | あり | 分散 | 不可 | 本番 |
| RedisSemanticCache | 中 | あり | 分散 | 可能 | 高トラフィック |

---

## 12. 実践プロジェクト: カスタマーサポートエージェント

```python
"""
カスタマーサポートエージェントの完全実装例
- FAQデータベース検索
- 注文ステータス確認
- エスカレーション機能
- 対話ログの記録
"""
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationSummaryBufferMemory
from pydantic import BaseModel, Field
from datetime import datetime
import json

# --- ツール定義 ---

@tool
def search_faq(query: str, category: str = "all") -> str:
    """FAQデータベースを検索する。

    Args:
        query: 検索キーワード
        category: カテゴリ（shipping, payment, returns, account, all）
    """
    # 簡易FAQデータベース（実際はベクトルDBを使用）
    faqs = {
        "shipping": [
            {"q": "配送日数は？", "a": "通常2-3営業日です。離島は5-7営業日。"},
            {"q": "送料は？", "a": "5,000円以上のご注文で送料無料。それ以下は一律550円。"},
            {"q": "配送状況の確認", "a": "注文詳細ページから追跡番号でご確認いただけます。"},
        ],
        "payment": [
            {"q": "支払い方法", "a": "クレジットカード、銀行振込、コンビニ決済に対応。"},
            {"q": "分割払い", "a": "3回・6回・12回の分割払いが可能（手数料あり）。"},
        ],
        "returns": [
            {"q": "返品ポリシー", "a": "商品到着後14日以内であれば返品可能。未使用品に限ります。"},
            {"q": "返金", "a": "返品確認後、5営業日以内にご返金いたします。"},
        ],
    }

    results = []
    search_categories = [category] if category != "all" else faqs.keys()

    for cat in search_categories:
        if cat in faqs:
            for faq in faqs[cat]:
                if query.lower() in faq["q"].lower() or query.lower() in faq["a"].lower():
                    results.append(f"[{cat}] Q: {faq['q']} A: {faq['a']}")

    return "\n".join(results) if results else "該当するFAQが見つかりませんでした。"

@tool
def check_order_status(order_id: str) -> str:
    """注文のステータスを確認する。

    Args:
        order_id: 注文番号（例: ORD-2024-001）
    """
    # モックデータ
    orders = {
        "ORD-2024-001": {
            "status": "配送中",
            "items": "ワイヤレスヘッドフォン x1",
            "tracking": "JP123456789",
            "estimated_delivery": "2024-12-20"
        },
        "ORD-2024-002": {
            "status": "処理中",
            "items": "USBケーブル x3",
            "tracking": None,
            "estimated_delivery": "2024-12-22"
        }
    }

    if order_id in orders:
        order = orders[order_id]
        return json.dumps(order, ensure_ascii=False, indent=2)
    return f"注文番号 {order_id} が見つかりません。正しい注文番号をご確認ください。"

@tool
def escalate_to_human(
    reason: str,
    priority: str = "normal",
    customer_summary: str = ""
) -> str:
    """問題をヒューマンオペレーターにエスカレーションする。

    Args:
        reason: エスカレーションの理由
        priority: 優先度（low, normal, high, urgent）
        customer_summary: お客様の状況の要約
    """
    ticket_id = f"TICKET-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    return (
        f"エスカレーションチケット作成: {ticket_id}\n"
        f"優先度: {priority}\n"
        f"理由: {reason}\n"
        f"オペレーターに引き継ぎます。しばらくお待ちください。"
    )

# --- エージェント構築 ---

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    temperature=0,
    max_tokens=2048
)

system_prompt = """あなたは「テックストア」のカスタマーサポートAIアシスタントです。

## 対応方針
- 丁寧で親切な対応を心がける
- 不明な点は推測せず、FAQを検索するか確認する
- 注文に関する問い合わせは必ず注文番号を確認する
- 解決できない問題は速やかにヒューマンオペレーターにエスカレーションする

## エスカレーション基準
以下の場合はヒューマンオペレーターに引き継ぐ:
- お客様が明示的に人間との対話を希望した場合
- 返金・交換の具体的な処理が必要な場合
- クレームや苦情の対応
- システムに情報がない場合

## 禁止事項
- 存在しない情報を作り上げない
- 個人情報を尋ねすぎない
- 他社製品の批判をしない
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

tools = [search_faq, check_order_status, escalate_to_human]

agent = create_tool_calling_agent(llm, tools, prompt)

memory = ConversationSummaryBufferMemory(
    llm=ChatAnthropic(model="claude-haiku-4-20250514"),
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
)

support_agent = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,
    max_iterations=8,
    max_execution_time=60,
    handle_parsing_errors=True
)

# 使用例
# result = support_agent.invoke({"input": "注文ORD-2024-001の配送状況を教えてください"})
```

---

## 13. アンチパターン

### アンチパターン1: 過度なチェーン連結

```python
# NG: 読みにくい長大なチェーン
chain = (
    prompt1 | llm | parser1 | transform1 |
    prompt2 | llm | parser2 | transform2 |
    prompt3 | llm | parser3 | transform3 |
    prompt4 | llm | parser4
)  # デバッグ困難

# OK: 意味のある単位で分割
research_chain = prompt1 | llm | parser1
analysis_chain = prompt2 | llm | parser2
report_chain = prompt3 | llm | parser3

# 組み合わせ
full_chain = research_chain | analysis_chain | report_chain
```

### アンチパターン2: verbose=True を本番環境で使う

```python
# NG: 本番でverbose出力（セキュリティリスク+パフォーマンス低下）
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# OK: 環境に応じた設定
import os
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=os.getenv("ENV") == "development",
    callbacks=[production_logger] if os.getenv("ENV") == "production" else []
)
```

### アンチパターン3: ツール説明の不備

```python
# NG: 説明が不十分でLLMが適切にツールを選択できない
@tool
def search(q: str) -> str:
    """検索する"""  # 何を検索？どんな入力が期待される？
    pass

# OK: 具体的で明確な説明
@tool
def search_product_catalog(
    query: str,
    category: str = "all",
    price_range: str = "any"
) -> str:
    """商品カタログから製品を検索する。商品名、ブランド名、カテゴリで検索可能。

    Args:
        query: 検索キーワード（商品名やブランド名）
        category: 商品カテゴリ（electronics, clothing, books, all）
        price_range: 価格帯（budget: ~5000円, mid: 5000-20000円, premium: 20000円~, any）

    Returns:
        マッチした商品のリスト（名前、価格、在庫状況を含む）
    """
    pass
```

### アンチパターン4: メモリリークの放置

```python
# NG: メモリ管理なしの長時間セッション
memory = ConversationBufferMemory()  # 会話が増え続ける
executor = AgentExecutor(agent=agent, tools=tools, memory=memory)
# 何百ターンも続くと OOM の危険

# OK: ウィンドウ or サマリーメモリ + クリーンアップ
memory = ConversationSummaryBufferMemory(
    llm=ChatAnthropic(model="claude-haiku-4-20250514"),
    max_token_limit=2000,
    return_messages=True,
    memory_key="chat_history"
)

# 定期的なクリーンアップ
def cleanup_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    # Redis の場合は TTL で自動期限切れ
```

### アンチパターン5: 同期処理でのブロッキング

```python
# NG: FastAPIで同期エージェントを実行（リクエストをブロック）
@app.post("/chat")
def chat_sync(request: ChatRequest):
    result = executor.invoke({"input": request.message})
    return result

# OK: 非同期で実行
@app.post("/chat")
async def chat_async(request: ChatRequest):
    result = await executor.ainvoke({"input": request.message})
    return result
```

---

## 14. FAQ

### Q1: LangChainのバージョン管理が難しいのですが？

LangChainはAPIの変更が頻繁。対策:
- **langchain-core** を固定（最も安定）
- **requirements.txt** でバージョンを明示的に固定
- **LangSmith** でテストの自動化
- 破壊的変更がある場合は `langchain` の CHANGELOG を確認
- langchain-core と langchain のメジャーバージョンを揃える（0.3.x 系統）

### Q2: AgentExecutor と LangGraph のどちらを使うべき？

- **AgentExecutor**: 単純なツール使用エージェント（5ツール以下、直線的な処理）
- **LangGraph**: 条件分岐、ループ、状態管理、マルチエージェントが必要な場合

LangChain公式も複雑なケースでは LangGraph を推奨している。

### Q3: LangChainのコスト最適化方法は？

- **キャッシュ**: `langchain.cache` でLLM応答をキャッシュ
- **モデル切り替え**: 分類はHaiku、生成はSonnetなどノードごとに最適化
- **早期終了**: `max_iterations` を適切に設定
- **バッチ処理**: `chain.batch([input1, input2, ...])` で並列実行

### Q4: ツールが多すぎるとエージェントの精度が下がる？

ツールが増えるとLLMのツール選択精度が低下する。対策:
- **ツール数は10個以下** を目安にする
- ツールの説明を明確かつ差別化する
- 関連ツールをツールキットにグループ化する
- 動的ツール選択（タスクに応じてツールセットを切り替え）を実装する
- LangGraphに移行して、ノードごとに異なるツールセットを提供する

### Q5: LangChainとLlamaIndexの使い分けは？

- **LangChain**: 汎用エージェント、ツール統合、ワークフロー構築
- **LlamaIndex**: RAG（検索拡張生成）に特化、ドキュメントインデックス
- **併用**: LlamaIndexのRetrieverをLangChainのツールとして統合するのが一般的

```python
# LlamaIndex + LangChain の統合例
from llama_index.core import VectorStoreIndex
from langchain.tools import Tool

# LlamaIndex のインデックスを LangChain ツールに変換
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

llama_tool = Tool(
    name="document_search",
    func=query_engine.query,
    description="社内ドキュメントを検索する"
)

# LangChain エージェントのツールリストに追加
tools = [llama_tool, search_database, email_tool]
```

### Q6: 非同期処理でデッドロックが発生する場合の対処法は？

```python
# 問題: 同期コード内でasync関数を呼ぼうとしてデッドロック
# NG:
import asyncio
result = asyncio.run(executor.ainvoke({"input": "test"}))  # デッドロックの可能性

# OK: nest_asyncio を使用（Jupyter環境など）
import nest_asyncio
nest_asyncio.apply()

# OK: 専用のイベントループで実行
import asyncio
from concurrent.futures import ThreadPoolExecutor

def run_async_in_thread(coro):
    """別スレッドで非同期関数を実行"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

with ThreadPoolExecutor() as pool:
    future = pool.submit(
        run_async_in_thread,
        executor.ainvoke({"input": "test"})
    )
    result = future.result(timeout=60)
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| LCEL | パイプラインでコンポーネントを接続 |
| ツール | @tool / StructuredTool / BaseTool の3方式 |
| AgentExecutor | エージェントの実行エンジン |
| メモリ | ConversationBuffer系で会話を保持 |
| ストリーミング | astream_events でリアルタイム出力 |
| キャッシュ | SQLite / Redis / セマンティックの3方式 |
| テスト | ユニットテスト + LangSmith評価 |
| 本番運用 | FastAPI統合、エラーハンドリング、セキュリティ |
| 原則 | シンプルに始め、必要に応じてLangGraphに移行 |

## 次に読むべきガイド

- [01-langgraph.md](./01-langgraph.md) -- LangGraphによる高度なワークフロー
- [02-mcp-agents.md](./02-mcp-agents.md) -- MCPエージェントの実装
- [04-evaluation.md](./04-evaluation.md) -- エージェントの評価手法

## 参考文献

1. LangChain Documentation -- https://python.langchain.com/docs/
2. LangChain GitHub -- https://github.com/langchain-ai/langchain
3. Harrison Chase, "LangChain Expression Language (LCEL)" -- https://python.langchain.com/docs/concepts/lcel/
4. LangSmith Documentation -- https://docs.smith.langchain.com/
5. LangChain Cookbook -- https://github.com/langchain-ai/langchain/tree/master/cookbook
