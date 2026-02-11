# メモリシステム

> 短期記憶・長期記憶・RAG――AIエージェントが文脈を保持し、過去の経験から学習するためのメモリアーキテクチャを設計・実装する。

## この章で学ぶこと

1. エージェントにおけるメモリの3層構造（短期・作業・長期）の役割と設計
2. RAG（Retrieval-Augmented Generation）によるスケーラブルな記憶の実装
3. メモリ戦略の選定基準と実装パターン

---

## 1. メモリの必要性

### 1.1 メモリがないエージェントの問題

```
メモリなしエージェント:
  Turn 1: "私の名前は田中です" → "こんにちは田中さん"
  Turn 2: "私の名前は？"      → "わかりません"  ← 忘れている!

メモリありエージェント:
  Turn 1: "私の名前は田中です" → "こんにちは田中さん" [記憶に保存]
  Turn 2: "私の名前は？"      → "田中さんですね"    ← 記憶を参照
```

### 1.2 メモリの3層構造

```
エージェントメモリの3層構造
+--------------------------------------------------------+
|                                                          |
|  +--------------------------------------------------+   |
|  |  短期記憶 (Short-Term Memory)                     |   |
|  |  - 現在の会話履歴                                  |   |
|  |  - 直近のツール実行結果                            |   |
|  |  - 寿命: 1セッション                              |   |
|  +--------------------------------------------------+   |
|                                                          |
|  +--------------------------------------------------+   |
|  |  作業記憶 (Working Memory)                        |   |
|  |  - 現在のタスクの計画                              |   |
|  |  - 中間結果のスクラッチパッド                      |   |
|  |  - 寿命: 1タスク                                  |   |
|  +--------------------------------------------------+   |
|                                                          |
|  +--------------------------------------------------+   |
|  |  長期記憶 (Long-Term Memory)                      |   |
|  |  - ユーザーの好み・プロフィール                    |   |
|  |  - 過去のタスク結果                                |   |
|  |  - 学習したパターン                                |   |
|  |  - 寿命: 永続                                     |   |
|  +--------------------------------------------------+   |
|                                                          |
+--------------------------------------------------------+
```

---

## 2. 短期記憶の実装

### 2.1 会話バッファ

```python
# 最もシンプルな短期記憶: 全履歴を保持
class ConversationBufferMemory:
    def __init__(self):
        self.messages = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def get_context(self) -> list:
        return self.messages.copy()

    def clear(self):
        self.messages = []
```

### 2.2 スライディングウィンドウ

```python
# 直近N件のメッセージのみ保持
class SlidingWindowMemory:
    def __init__(self, window_size: int = 20):
        self.messages = []
        self.window_size = window_size

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        # ウィンドウを超えた古いメッセージを削除
        if len(self.messages) > self.window_size:
            self.messages = self.messages[-self.window_size:]

    def get_context(self) -> list:
        return self.messages.copy()
```

### 2.3 要約メモリ

```python
# 古い履歴を要約して圧縮する
class SummaryMemory:
    def __init__(self, llm, max_tokens: int = 2000):
        self.llm = llm
        self.max_tokens = max_tokens
        self.summary = ""
        self.recent_messages = []

    def add(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})

        # トークン数が閾値を超えたら要約
        if self._count_tokens() > self.max_tokens:
            self._compress()

    def _compress(self):
        """古いメッセージを要約に統合"""
        old_messages = self.recent_messages[:-4]  # 直近4件は残す

        summary_prompt = f"""
以下の会話履歴を200字以内で要約してください。
重要な事実、ユーザーの要望、決定事項を保持してください。

既存の要約: {self.summary}

新しい会話:
{self._format_messages(old_messages)}
"""
        self.summary = self.llm.generate(summary_prompt)
        self.recent_messages = self.recent_messages[-4:]

    def get_context(self) -> list:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"これまでの会話の要約: {self.summary}"
            })
        context.extend(self.recent_messages)
        return context
```

---

## 3. 長期記憶とRAG

### 3.1 RAGアーキテクチャ

```
RAG (Retrieval-Augmented Generation) の流れ

1. インデックス構築 (オフライン)
+----------+    チャンク化    +---------+   埋め込み    +----------+
| ドキュメント|------------->| テキスト |------------>| ベクトル  |
|          |               | チャンク  |              | DB       |
+----------+               +---------+              +----------+

2. 検索と生成 (オンライン)
+--------+   クエリ   +---------+   検索   +----------+
| ユーザー|--------->|  埋め込み |-------->| ベクトル  |
|        |          |  モデル   |         | DB       |
+--------+          +---------+         +----+-----+
     ^                                       |
     |              +---------+              | 上位K件
     +------<-------|   LLM   |<------<------+
        回答        |  生成   |   文脈+質問
                    +---------+
```

### 3.2 RAGの実装

```python
# RAGによる長期記憶の実装
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

class RAGMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path="./memory_db")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, text: str, metadata: dict = None):
        """テキストをベクトル化して保存"""
        embedding = self.embedder.encode(text).tolist()
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """クエリに類似した記憶を検索"""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0]

    def retrieve_with_filter(self, query: str, filter_metadata: dict,
                              top_k: int = 5) -> list[str]:
        """メタデータでフィルタリングして検索"""
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        return results["documents"][0]

# 使用例
memory = RAGMemory()

# 過去のタスク結果を保存
memory.store(
    "ユーザーはPythonのFastAPIを好み、Flaskよりも優先する",
    metadata={"type": "preference", "user": "tanaka"}
)

memory.store(
    "プロジェクトXのデプロイはAWS ECS + Fargateで構成",
    metadata={"type": "fact", "project": "X"}
)

# 検索
relevant = memory.retrieve("このプロジェクトのインフラ構成は？")
print(relevant)
```

### 3.3 チャンキング戦略

```python
# テキストのチャンク化戦略
from typing import Generator

def chunk_by_tokens(text: str, chunk_size: int = 500,
                     overlap: int = 50) -> Generator[str, None, None]:
    """トークン数ベースのチャンク化（オーバーラップ付き）"""
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            yield chunk

def chunk_by_semantic(text: str) -> list[str]:
    """セマンティック（意味的）チャンク化"""
    # 段落、見出し、コードブロックなどの境界で分割
    import re
    sections = re.split(r'\n#{1,3}\s|\n\n\n', text)
    return [s.strip() for s in sections if s.strip()]

def chunk_by_recursive(text: str, max_size: int = 1000) -> list[str]:
    """再帰的チャンク化（LangChain方式）"""
    separators = ["\n\n", "\n", ". ", " ", ""]

    for sep in separators:
        if len(text) <= max_size:
            return [text]

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            if len(current) + len(part) + len(sep) <= max_size:
                current += (sep if current else "") + part
            else:
                if current:
                    chunks.append(current)
                current = part

        if current:
            chunks.append(current)

        if all(len(c) <= max_size for c in chunks):
            return chunks

    return [text[:max_size]]
```

---

## 4. メモリ戦略の比較

### 4.1 短期記憶パターン比較

| パターン | メモリ使用量 | 文脈保持 | コスト | 適用場面 |
|----------|-------------|---------|--------|---------|
| 全履歴バッファ | 高（線形増加） | 完全 | 高 | 短い会話 |
| スライディングウィンドウ | 固定 | 直近のみ | 中 | 一般的な対話 |
| 要約メモリ | 低 | 要約で圧縮 | 中（要約コスト） | 長い会話 |
| トークン制限バッファ | 固定 | 制限内 | 中 | API制限意識 |

### 4.2 長期記憶ストア比較

| ストア | 検索方式 | スケーラビリティ | コスト | 代表製品 |
|--------|---------|----------------|--------|---------|
| ベクトルDB | セマンティック | 高 | 中-高 | Pinecone, Chroma |
| キーバリュー | 完全一致 | 高 | 低 | Redis, DynamoDB |
| グラフDB | 関係性 | 中 | 中 | Neo4j |
| RDBMS | SQL | 高 | 低-中 | PostgreSQL + pgvector |
| ファイル | 全文検索 | 低 | 最低 | JSON, SQLite |

---

## 5. 統合メモリシステム

```python
# 短期 + 長期を統合したメモリシステム
class IntegratedMemory:
    def __init__(self, llm, rag_memory: RAGMemory):
        self.short_term = SummaryMemory(llm)
        self.working = {}  # タスク固有の作業領域
        self.long_term = rag_memory

    def add_conversation(self, role: str, content: str):
        """会話を短期記憶に追加"""
        self.short_term.add(role, content)

    def add_fact(self, fact: str, metadata: dict = None):
        """事実を長期記憶に保存"""
        self.long_term.store(fact, metadata)

    def set_working(self, key: str, value):
        """作業記憶に一時データを保存"""
        self.working[key] = value

    def get_context(self, current_query: str) -> dict:
        """現在の文脈を統合して返す"""
        return {
            "conversation": self.short_term.get_context(),
            "relevant_memories": self.long_term.retrieve(current_query, top_k=3),
            "working_data": self.working
        }

    def end_task(self, task_summary: str):
        """タスク終了時に結果を長期記憶に保存"""
        self.long_term.store(
            task_summary,
            metadata={"type": "task_result", "timestamp": time.time()}
        )
        self.working.clear()
```

---

## 6. メモリのライフサイクル

```
メモリのライフサイクル管理

セッション開始
     |
     v
+--------------------+
| 長期記憶からロード   |  ← ユーザー情報、過去のタスク
+----+---------------+
     |
     v
+--------------------+
| 短期記憶に会話追加   |  ← 各ターンで更新
+----+---------------+
     |
     v (閾値超過?)
+--------------------+
| 要約・圧縮          |  ← 古い履歴を要約
+----+---------------+
     |
     v
+--------------------+
| タスク完了          |
+----+---------------+
     |
     v
+--------------------+
| 重要情報を           |  ← 学習した事実、ユーザー好み
| 長期記憶に保存       |
+----+---------------+
     |
     v
セッション終了
```

---

## 7. アンチパターン

### アンチパターン1: 無制限の会話履歴

```python
# NG: 全履歴をそのままLLMに渡す
messages = load_all_history()  # 10万トークン超え
response = llm.generate(messages=messages)  # コンテキスト超過エラー

# OK: 適切な圧縮戦略を適用
memory = SummaryMemory(llm, max_tokens=4000)
for msg in load_all_history():
    memory.add(msg["role"], msg["content"])
response = llm.generate(messages=memory.get_context())
```

### アンチパターン2: 検索なしの長期記憶

```python
# NG: すべての長期記憶をプロンプトに含める
all_memories = database.get_all()  # 大量のデータ
prompt = f"知識: {all_memories}\n質問: {query}"

# OK: クエリに関連する記憶のみ検索して含める
relevant = rag_memory.retrieve(query, top_k=5)
prompt = f"関連知識:\n{chr(10).join(relevant)}\n\n質問: {query}"
```

---

## 8. FAQ

### Q1: どのベクトルDBを選ぶべきか？

- **プロトタイプ**: ChromaDB（組み込み型、セットアップ不要）
- **小〜中規模本番**: PostgreSQL + pgvector（既存DBを活用）
- **大規模本番**: Pinecone（マネージド、スケーラブル）
- **オンプレミス**: Weaviate, Milvus（セルフホスト可能）

### Q2: メモリに何を保存すべきか？

優先度順に:
1. **ユーザーの好み・設定**（「Pythonを好む」「簡潔な回答を希望」）
2. **プロジェクト固有の事実**（「DBはPostgreSQLを使用」）
3. **過去のタスク結果の要約**（「前回のレビューで指摘した3点」）
4. **エラーと解決策のペア**（「X のエラーは Y で解決」）

### Q3: RAGの精度を上げるには？

- **チャンクサイズの最適化**: 小さすぎると文脈不足、大きすぎるとノイズ混入。300-800トークンが一般的
- **ハイブリッド検索**: ベクトル検索 + キーワード検索（BM25）の組み合わせ
- **リランキング**: 検索結果をCross-Encoderで再ランク付け
- **メタデータフィルタリング**: 検索前にカテゴリ等で絞り込み

---

## まとめ

| 項目 | 内容 |
|------|------|
| 3層構造 | 短期（セッション）・作業（タスク）・長期（永続） |
| 短期記憶 | バッファ / スライディングウィンドウ / 要約 |
| 長期記憶 | ベクトルDB + RAGによるセマンティック検索 |
| チャンキング | トークン / セマンティック / 再帰的 |
| 統合設計 | 短期+長期を統合し、文脈に応じて使い分け |
| 核心原則 | 関連する記憶のみを効率的に取得する |

## 次に読むべきガイド

- [../01-patterns/00-single-agent.md](../01-patterns/00-single-agent.md) — シングルエージェントでのメモリ活用
- [../01-patterns/03-autonomous-agents.md](../01-patterns/03-autonomous-agents.md) — 自律エージェントの長期記憶
- [../02-implementation/00-langchain-agent.md](../02-implementation/00-langchain-agent.md) — LangChainでのメモリ実装

## 参考文献

1. Lewis, P. et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020) — https://arxiv.org/abs/2005.11401
2. LangChain, "Memory" — https://python.langchain.com/docs/concepts/memory/
3. ChromaDB Documentation — https://docs.trychroma.com/
4. Zhang, Z. et al., "A Survey on the Memory Mechanism of Large Language Model based Agents" (2024) — https://arxiv.org/abs/2404.13501
