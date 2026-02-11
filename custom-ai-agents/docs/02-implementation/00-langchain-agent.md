# LangChainエージェント

> チェーン・プロンプトテンプレート・ツール統合――LangChainを使ったエージェント構築の実践的な実装パターンとベストプラクティス。

## この章で学ぶこと

1. LangChainのコアコンセプト（チェーン、プロンプト、ツール）の理解
2. Tool Calling Agent の構築と AgentExecutor の活用法
3. カスタムツールと高度なプロンプト設計の実装パターン

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

### 3.2 ツールの構成パターン

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

---

## 5. メモリの統合

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

---

## 6. 比較表

### 6.1 エージェント作成方法の比較

| 方法 | コード量 | 柔軟性 | ツール方式 | 推奨場面 |
|------|---------|--------|-----------|---------|
| create_tool_calling_agent | 少 | 中 | Function Calling | 一般的 |
| create_react_agent | 少 | 中 | テキストベース | レガシーモデル |
| カスタムAgent | 多 | 高 | 任意 | 特殊要件 |
| LangGraph | 中 | 最高 | 任意 | 複雑なフロー |

### 6.2 ツール定義方法の比較

| 方法 | 手軽さ | 型安全性 | 非同期 | バリデーション |
|------|--------|---------|--------|--------------|
| @tool デコレータ | 最高 | 中 | 可 | 基本 |
| StructuredTool | 中 | 高 | 可 | Pydantic |
| BaseTool 継承 | 低 | 高 | 可 | 完全制御 |
| Tool.from_function | 高 | 低 | 不可 | なし |

---

## 7. アンチパターン

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

---

## 8. FAQ

### Q1: LangChainのバージョン管理が難しいのですが？

LangChainはAPIの変更が頻繁。対策:
- **langchain-core** を固定（最も安定）
- **requirements.txt** でバージョンを明示的に固定
- **LangSmith** でテストの自動化
- 破壊的変更がある場合は `langchain` の CHANGELOG を確認

### Q2: AgentExecutor と LangGraph のどちらを使うべき？

- **AgentExecutor**: 単純なツール使用エージェント（5ツール以下、直線的な処理）
- **LangGraph**: 条件分岐、ループ、状態管理、マルチエージェントが必要な場合

LangChain公式も複雑なケースでは LangGraph を推奨している。

### Q3: LangChainのコスト最適化方法は？

- **キャッシュ**: `langchain.cache` でLLM応答をキャッシュ
- **モデル切り替え**: 分類はHaiku、生成はSonnetなどノードごとに最適化
- **早期終了**: `max_iterations` を適切に設定
- **バッチ処理**: `chain.batch([input1, input2, ...])` で並列実行

---

## まとめ

| 項目 | 内容 |
|------|------|
| LCEL | パイプラインでコンポーネントを接続 |
| ツール | @tool / StructuredTool / BaseTool の3方式 |
| AgentExecutor | エージェントの実行エンジン |
| メモリ | ConversationBuffer系で会話を保持 |
| ストリーミング | astream_events でリアルタイム出力 |
| 原則 | シンプルに始め、必要に応じてLangGraphに移行 |

## 次に読むべきガイド

- [01-langgraph.md](./01-langgraph.md) — LangGraphによる高度なワークフロー
- [02-mcp-agents.md](./02-mcp-agents.md) — MCPエージェントの実装
- [04-evaluation.md](./04-evaluation.md) — エージェントの評価手法

## 参考文献

1. LangChain Documentation — https://python.langchain.com/docs/
2. LangChain GitHub — https://github.com/langchain-ai/langchain
3. Harrison Chase, "LangChain Expression Language (LCEL)" — https://python.langchain.com/docs/concepts/lcel/
