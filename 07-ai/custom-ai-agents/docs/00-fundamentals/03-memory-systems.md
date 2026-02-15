# メモリシステム

> 短期記憶・長期記憶・RAG――AIエージェントが文脈を保持し、過去の経験から学習するためのメモリアーキテクチャを設計・実装する。

## この章で学ぶこと

1. エージェントにおけるメモリの3層構造（短期・作業・長期）の役割と設計
2. RAG（Retrieval-Augmented Generation）によるスケーラブルな記憶の実装
3. メモリ戦略の選定基準と実装パターン
4. 高度なメモリアーキテクチャ（知識グラフ、エピソード記憶、セマンティックメモリ）
5. プロダクション環境でのメモリシステムの運用とチューニング

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

### 1.3 人間の記憶モデルとの対応

```
人間の記憶                    AIエージェントのメモリ
+-----------------------+     +-----------------------+
| 感覚記憶（数秒）       | --> | 入力バッファ           |
| - 視覚・聴覚の一時保持  |     | - 生のリクエスト       |
+-----------------------+     +-----------------------+
| 短期記憶（数十秒）     | --> | コンテキストウィンドウ  |
| - 電話番号の暗記       |     | - 会話履歴             |
+-----------------------+     +-----------------------+
| 作業記憶（秒〜分）     | --> | スクラッチパッド       |
| - 暗算中の一時保持     |     | - タスク中間結果       |
+-----------------------+     +-----------------------+
| 長期記憶（永続）       | --> | ベクトルDB / ファイル   |
| - エピソード記憶       |     | - 過去の会話要約       |
| - 意味記憶            |     | - 知識ベース           |
| - 手続き記憶          |     | - 学習パターン         |
+-----------------------+     +-----------------------+
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

### 2.4 トークンベースバッファ

```python
# トークン数に基づいて管理する短期記憶
import tiktoken

class TokenBasedMemory:
    """トークン数の上限に基づいてメッセージを管理"""

    def __init__(self, max_tokens: int = 8000, model: str = "cl100k_base"):
        self.max_tokens = max_tokens
        self.messages = []
        self.encoder = tiktoken.get_encoding(model)

    def _count_tokens(self, messages: list) -> int:
        """メッセージリストのトークン数を計算"""
        total = 0
        for msg in messages:
            total += len(self.encoder.encode(msg["content"]))
            total += 4  # メッセージオーバーヘッド
        return total

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self):
        """トークン上限を超えた場合、古いメッセージから削除"""
        while (self._count_tokens(self.messages) > self.max_tokens
               and len(self.messages) > 2):  # 最低2件は保持
            # システムメッセージは保持
            if self.messages[0]["role"] == "system":
                self.messages.pop(1)
            else:
                self.messages.pop(0)

    def get_context(self) -> list:
        return self.messages.copy()

    def get_stats(self) -> dict:
        """メモリ使用状況を返す"""
        return {
            "message_count": len(self.messages),
            "total_tokens": self._count_tokens(self.messages),
            "max_tokens": self.max_tokens,
            "usage_percent": (
                self._count_tokens(self.messages) / self.max_tokens * 100
            )
        }
```

### 2.5 ハイブリッド短期記憶

```python
# 要約 + スライディングウィンドウのハイブリッド
class HybridShortTermMemory:
    """要約メモリとスライディングウィンドウを組み合わせる"""

    def __init__(self, llm, window_size: int = 10,
                 max_summary_length: int = 500):
        self.llm = llm
        self.window_size = window_size
        self.max_summary_length = max_summary_length
        self.summary = ""
        self.messages = []
        self.important_facts = []  # 重要な事実を別途保持

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

        # 重要な事実を自動抽出
        if role == "user":
            self._extract_facts(content)

        # ウィンドウ超過時に要約
        if len(self.messages) > self.window_size:
            overflow = self.messages[:-self.window_size]
            self._update_summary(overflow)
            self.messages = self.messages[-self.window_size:]

    def _extract_facts(self, content: str):
        """ユーザー入力から重要な事実を抽出"""
        fact_indicators = [
            "私の名前は", "私は", "好きな", "嫌いな",
            "使っている", "プロジェクト", "会社"
        ]
        if any(indicator in content for indicator in fact_indicators):
            self.important_facts.append(content)
            # 重複排除（最新10件まで）
            self.important_facts = list(set(self.important_facts))[-10:]

    def _update_summary(self, overflow_messages: list):
        """溢れたメッセージを要約に統合"""
        messages_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in overflow_messages
        )
        prompt = f"""既存の要約を更新してください。
既存の要約: {self.summary}
新しい会話:
{messages_text}
{self.max_summary_length}文字以内で要約:"""
        self.summary = self.llm.generate(prompt)

    def get_context(self) -> list:
        context = []
        if self.summary:
            context.append({
                "role": "system",
                "content": f"会話の要約: {self.summary}"
            })
        if self.important_facts:
            context.append({
                "role": "system",
                "content": f"重要な事実:\n" + "\n".join(
                    f"- {f}" for f in self.important_facts
                )
            })
        context.extend(self.messages)
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

### 3.4 ハイブリッド検索

```python
# ベクトル検索 + キーワード検索のハイブリッド
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """ベクトル検索とBM25のハイブリッド検索"""

    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store
        self.documents = []
        self.bm25 = None

    def add_documents(self, documents: list[str], metadatas: list[dict] = None):
        """ドキュメントを追加"""
        self.documents.extend(documents)

        # BM25インデックスを再構築
        tokenized = [doc.split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized)

        # ベクトルDBにも追加
        for i, doc in enumerate(documents):
            self.vector_store.store(
                doc,
                metadata=metadatas[i] if metadatas else {}
            )

    def search(self, query: str, top_k: int = 5,
               vector_weight: float = 0.7) -> list[dict]:
        """ハイブリッド検索（RRF融合）"""
        # ベクトル検索
        vector_results = self.vector_store.retrieve_with_scores(query, top_k=top_k * 2)

        # BM25検索
        tokenized_query = query.split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = np.argsort(bm25_scores)[-top_k * 2:][::-1]

        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60  # RRFパラメータ

        for rank, (doc, score) in enumerate(vector_results):
            rrf_scores[doc] = rrf_scores.get(doc, 0) + vector_weight / (k + rank + 1)

        for rank, idx in enumerate(bm25_top):
            doc = self.documents[idx]
            rrf_scores[doc] = rrf_scores.get(doc, 0) + (1 - vector_weight) / (k + rank + 1)

        # スコア順にソート
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [{"document": doc, "score": score} for doc, score in sorted_results[:top_k]]
```

### 3.5 リランキング

```python
# Cross-Encoderによるリランキング
from sentence_transformers import CrossEncoder

class Reranker:
    """検索結果をCross-Encoderで再ランク付け"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[str],
               top_k: int = 5) -> list[dict]:
        """ドキュメントを再ランク付け"""
        # クエリとドキュメントのペアを作成
        pairs = [(query, doc) for doc in documents]

        # スコアリング
        scores = self.model.predict(pairs)

        # スコア順にソート
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [
            {"document": doc, "score": float(score)}
            for doc, score in scored_docs[:top_k]
        ]

# 使用例: 検索→リランク
retriever = HybridRetriever(embedder, vector_store)
reranker = Reranker()

# 1段階目: ハイブリッド検索で候補を取得
candidates = retriever.search(query, top_k=20)
candidate_docs = [c["document"] for c in candidates]

# 2段階目: Cross-Encoderで精度の高いリランキング
final_results = reranker.rerank(query, candidate_docs, top_k=5)
```

---

## 4. 知識グラフメモリ

### 4.1 知識グラフの構造

```
知識グラフメモリ

  [田中] --所属--> [エンジニアリング部]
    |                    |
    |--使用言語-->  [Python]
    |                    |
    |--好む-->     [FastAPI] --カテゴリ--> [Webフレームワーク]
    |                                         |
    |--担当-->    [プロジェクトX] --使用--> [AWS ECS]
    |                    |
    v                    v
  [senior]        [2024年Q3開始]
```

### 4.2 知識グラフメモリの実装

```python
# 知識グラフベースのメモリシステム
from dataclasses import dataclass
from collections import defaultdict
import json

@dataclass
class Triple:
    """知識グラフのトリプル（主語-述語-目的語）"""
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    timestamp: float = 0.0

class KnowledgeGraphMemory:
    """知識グラフベースの長期記憶"""

    def __init__(self, llm=None):
        self.triples: list[Triple] = []
        self.entity_index = defaultdict(list)  # エンティティ→トリプルのインデックス
        self.llm = llm

    def add_triple(self, subject: str, predicate: str, obj: str,
                   confidence: float = 1.0):
        """トリプルを追加"""
        import time
        triple = Triple(
            subject=subject.lower(),
            predicate=predicate.lower(),
            object=obj.lower(),
            confidence=confidence,
            timestamp=time.time()
        )
        self.triples.append(triple)
        self.entity_index[subject.lower()].append(triple)
        self.entity_index[obj.lower()].append(triple)

    def extract_and_store(self, text: str):
        """テキストから知識を自動抽出して保存"""
        if not self.llm:
            raise ValueError("LLMが必要です")

        prompt = f"""以下のテキストから事実をトリプル形式で抽出してください。

テキスト: {text}

JSON形式で出力:
[{{"subject": "...", "predicate": "...", "object": "..."}}]

例:
"田中さんはPythonが得意です" → [{{"subject": "田中", "predicate": "得意", "object": "Python"}}]
"""
        response = self.llm.generate(prompt)
        triples = json.loads(response)
        for t in triples:
            self.add_triple(t["subject"], t["predicate"], t["object"])

    def query(self, entity: str) -> list[Triple]:
        """エンティティに関連するトリプルを検索"""
        return self.entity_index.get(entity.lower(), [])

    def query_relation(self, subject: str = None, predicate: str = None,
                       obj: str = None) -> list[Triple]:
        """条件に合うトリプルを検索"""
        results = self.triples
        if subject:
            results = [t for t in results if t.subject == subject.lower()]
        if predicate:
            results = [t for t in results if t.predicate == predicate.lower()]
        if obj:
            results = [t for t in results if t.object == obj.lower()]
        return results

    def get_subgraph(self, entity: str, depth: int = 2) -> list[Triple]:
        """エンティティを中心とした部分グラフを取得"""
        visited = set()
        result = []
        queue = [(entity.lower(), 0)]

        while queue:
            current, d = queue.pop(0)
            if current in visited or d > depth:
                continue
            visited.add(current)

            related = self.entity_index.get(current, [])
            result.extend(related)

            for triple in related:
                if triple.subject not in visited:
                    queue.append((triple.subject, d + 1))
                if triple.object not in visited:
                    queue.append((triple.object, d + 1))

        return result

    def to_context_string(self, entity: str, depth: int = 1) -> str:
        """エンティティの知識をコンテキスト文字列として出力"""
        triples = self.get_subgraph(entity, depth)
        if not triples:
            return f"{entity}に関する情報はありません。"

        lines = []
        for t in triples:
            lines.append(f"- {t.subject} は {t.predicate} {t.object}")
        return f"{entity}に関する知識:\n" + "\n".join(lines)

# 使用例
kg = KnowledgeGraphMemory(llm=llm)
kg.add_triple("田中", "所属", "エンジニアリング部")
kg.add_triple("田中", "使用言語", "Python")
kg.add_triple("田中", "好むフレームワーク", "FastAPI")
kg.add_triple("プロジェクトX", "使用インフラ", "AWS ECS")
kg.add_triple("田中", "担当", "プロジェクトX")

# 田中に関する情報を取得
context = kg.to_context_string("田中")
print(context)
# → 田中に関する知識:
#    - 田中 は 所属 エンジニアリング部
#    - 田中 は 使用言語 Python
#    - 田中 は 好むフレームワーク FastAPI
#    - 田中 は 担当 プロジェクトX
```

---

## 5. エピソード記憶

### 5.1 エピソード記憶の設計

```
エピソード記憶: 過去の「体験」を時系列で保存

Episode 1 (2024-01-15):
  タスク: "FastAPIでCRUD APIを作成"
  結果: 成功
  学んだこと: "SQLAlchemyとの組み合わせが効率的"
  困難だった点: "非同期セッション管理"

Episode 2 (2024-01-16):
  タスク: "APIにJWT認証を追加"
  結果: 成功（2回目の試行で）
  学んだこと: "python-joseよりPyJWTの方がシンプル"
  困難だった点: "トークンリフレッシュのロジック"

→ 新しいタスクで類似状況に遭遇したら、過去のエピソードを参照
```

### 5.2 エピソード記憶の実装

```python
# エピソード記憶: 過去の体験から学習
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class Episode:
    """1つのタスク実行エピソード"""
    task: str
    actions: list[str]
    result: str
    success: bool
    lessons_learned: list[str]
    difficulties: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    tags: list[str] = field(default_factory=list)

class EpisodicMemory:
    """エピソード記憶: 過去の体験を保存・検索"""

    def __init__(self, rag_memory: RAGMemory):
        self.episodes: list[Episode] = []
        self.rag = rag_memory

    def record_episode(self, episode: Episode):
        """エピソードを記録"""
        self.episodes.append(episode)

        # RAGにもインデックス
        episode_text = (
            f"タスク: {episode.task}\n"
            f"結果: {'成功' if episode.success else '失敗'}\n"
            f"学び: {', '.join(episode.lessons_learned)}\n"
            f"困難: {', '.join(episode.difficulties)}"
        )
        self.rag.store(episode_text, metadata={
            "type": "episode",
            "success": episode.success,
            "timestamp": episode.timestamp.isoformat(),
            "tags": ",".join(episode.tags)
        })

    def recall_similar(self, current_task: str,
                       top_k: int = 3) -> list[Episode]:
        """現在のタスクに類似した過去のエピソードを想起"""
        similar_texts = self.rag.retrieve(current_task, top_k=top_k)
        # テキストからエピソードを復元
        recalled = []
        for text in similar_texts:
            for episode in self.episodes:
                if episode.task in text:
                    recalled.append(episode)
                    break
        return recalled

    def get_lessons_for_task(self, task: str) -> str:
        """タスクに関連する過去の教訓をまとめて返す"""
        similar = self.recall_similar(task, top_k=5)
        if not similar:
            return "関連する過去のエピソードはありません。"

        lessons = []
        for ep in similar:
            if ep.success:
                lessons.append(f"[成功] {ep.task}: {', '.join(ep.lessons_learned)}")
            else:
                lessons.append(f"[失敗] {ep.task}: {', '.join(ep.difficulties)}")

        return "過去のエピソードからの教訓:\n" + "\n".join(lessons)

    def get_success_rate(self, tag: str = None) -> float:
        """成功率を計算"""
        episodes = self.episodes
        if tag:
            episodes = [e for e in episodes if tag in e.tags]
        if not episodes:
            return 0.0
        return sum(1 for e in episodes if e.success) / len(episodes)
```

---

## 6. メモリ戦略の比較

### 6.1 短期記憶パターン比較

| パターン | メモリ使用量 | 文脈保持 | コスト | 適用場面 |
|----------|-------------|---------|--------|---------|
| 全履歴バッファ | 高（線形増加） | 完全 | 高 | 短い会話 |
| スライディングウィンドウ | 固定 | 直近のみ | 中 | 一般的な対話 |
| 要約メモリ | 低 | 要約で圧縮 | 中（要約コスト） | 長い会話 |
| トークン制限バッファ | 固定 | 制限内 | 中 | API制限意識 |
| ハイブリッド | 中 | 要約+直近 | 中 | バランス重視 |

### 6.2 長期記憶ストア比較

| ストア | 検索方式 | スケーラビリティ | コスト | 代表製品 |
|--------|---------|----------------|--------|---------|
| ベクトルDB | セマンティック | 高 | 中-高 | Pinecone, Chroma |
| キーバリュー | 完全一致 | 高 | 低 | Redis, DynamoDB |
| グラフDB | 関係性 | 中 | 中 | Neo4j |
| RDBMS | SQL | 高 | 低-中 | PostgreSQL + pgvector |
| ファイル | 全文検索 | 低 | 最低 | JSON, SQLite |

### 6.3 メモリアーキテクチャの選定フローチャート

```
メモリアーキテクチャ選定

Q1: 会話は長時間続くか？
├── NO → 全履歴バッファで十分
└── YES
    Q2: 過去のセッションの情報が必要か？
    ├── NO → 要約メモリ or スライディングウィンドウ
    └── YES
        Q3: エンティティ間の関係性が重要か？
        ├── YES → 知識グラフ + ベクトルDB
        └── NO
            Q4: データ量は？
            ├── 少（〜1万件） → ChromaDB（ローカル）
            ├── 中（〜100万件） → PostgreSQL + pgvector
            └── 大（100万件〜） → Pinecone / Milvus
```

---

## 7. 統合メモリシステム

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

### 7.1 プロダクション統合メモリ

```python
# プロダクション対応の統合メモリシステム
import time
import logging
from typing import Optional

class ProductionMemorySystem:
    """プロダクション環境対応の統合メモリシステム"""

    def __init__(self, config: dict):
        self.logger = logging.getLogger("memory")

        # 短期記憶
        self.short_term = HybridShortTermMemory(
            llm=config["llm"],
            window_size=config.get("window_size", 15)
        )

        # 長期記憶（ベクトルDB）
        self.long_term = RAGMemory(
            collection_name=config.get("collection", "production_memory")
        )

        # 知識グラフ
        self.knowledge_graph = KnowledgeGraphMemory(llm=config["llm"])

        # エピソード記憶
        self.episodic = EpisodicMemory(rag_memory=self.long_term)

        # メトリクス
        self.metrics = {
            "store_count": 0,
            "retrieve_count": 0,
            "cache_hits": 0,
            "avg_retrieve_latency": 0.0
        }

        # 検索結果キャッシュ
        self._cache = {}
        self._cache_ttl = config.get("cache_ttl", 300)

    def store(self, content: str, memory_type: str = "fact",
              metadata: dict = None):
        """記憶を保存"""
        meta = metadata or {}
        meta["memory_type"] = memory_type
        meta["stored_at"] = time.time()

        self.long_term.store(content, metadata=meta)
        self.metrics["store_count"] += 1

        # 知識グラフにも抽出・保存
        try:
            self.knowledge_graph.extract_and_store(content)
        except Exception as e:
            self.logger.warning(f"知識グラフ抽出失敗: {e}")

    def retrieve(self, query: str, top_k: int = 5,
                 use_cache: bool = True) -> dict:
        """統合検索"""
        # キャッシュチェック
        cache_key = f"{query}:{top_k}"
        if use_cache and cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() - entry["timestamp"] < self._cache_ttl:
                self.metrics["cache_hits"] += 1
                return entry["result"]

        start = time.time()

        # 各メモリソースから検索
        result = {
            "conversation_context": self.short_term.get_context(),
            "semantic_matches": self.long_term.retrieve(query, top_k),
            "knowledge_graph": self.knowledge_graph.to_context_string(
                query.split()[0] if query else ""
            ),
            "past_episodes": self.episodic.get_lessons_for_task(query)
        }

        latency = time.time() - start
        self.metrics["retrieve_count"] += 1
        self._update_avg_latency(latency)

        # キャッシュに保存
        self._cache[cache_key] = {
            "result": result,
            "timestamp": time.time()
        }

        return result

    def build_prompt_context(self, query: str) -> str:
        """LLMに渡すコンテキスト文字列を構築"""
        retrieved = self.retrieve(query)

        parts = []

        # 会話コンテキスト
        conv = retrieved["conversation_context"]
        if conv:
            parts.append("=== 会話履歴 ===")
            for msg in conv[-5:]:
                parts.append(f"{msg['role']}: {msg['content']}")

        # セマンティック検索結果
        matches = retrieved["semantic_matches"]
        if matches:
            parts.append("\n=== 関連する記憶 ===")
            for i, match in enumerate(matches, 1):
                parts.append(f"{i}. {match}")

        # 知識グラフ
        kg = retrieved["knowledge_graph"]
        if kg and "情報はありません" not in kg:
            parts.append(f"\n=== 知識 ===\n{kg}")

        # エピソード記憶
        episodes = retrieved["past_episodes"]
        if episodes and "ありません" not in episodes:
            parts.append(f"\n=== 過去の経験 ===\n{episodes}")

        return "\n".join(parts)

    def _update_avg_latency(self, latency: float):
        n = self.metrics["retrieve_count"]
        old_avg = self.metrics["avg_retrieve_latency"]
        self.metrics["avg_retrieve_latency"] = (old_avg * (n-1) + latency) / n

    def get_metrics(self) -> dict:
        return self.metrics.copy()
```

---

## 8. メモリのライフサイクル

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

### 8.1 メモリのガベージコレクション

```python
# 不要なメモリの自動クリーンアップ
class MemoryGarbageCollector:
    """古い・重複するメモリの自動クリーンアップ"""

    def __init__(self, memory: RAGMemory, max_age_days: int = 90):
        self.memory = memory
        self.max_age_days = max_age_days

    def cleanup_old_entries(self):
        """古いエントリを削除"""
        cutoff = time.time() - (self.max_age_days * 86400)
        # メタデータでフィルタリングして古いエントリを特定・削除
        results = self.memory.collection.get(
            where={"stored_at": {"$lt": cutoff}}
        )
        if results["ids"]:
            self.memory.collection.delete(ids=results["ids"])
            return len(results["ids"])
        return 0

    def deduplicate(self, similarity_threshold: float = 0.95):
        """重複するメモリを統合"""
        all_docs = self.memory.collection.get()
        to_delete = []

        for i, doc_i in enumerate(all_docs["documents"]):
            for j, doc_j in enumerate(all_docs["documents"]):
                if i >= j:
                    continue
                # 類似度チェック
                similarity = self._compute_similarity(doc_i, doc_j)
                if similarity > similarity_threshold:
                    # 古い方を削除候補に
                    to_delete.append(all_docs["ids"][j])

        if to_delete:
            self.memory.collection.delete(ids=list(set(to_delete)))
            return len(set(to_delete))
        return 0

    def compact_summaries(self, llm, max_per_topic: int = 5):
        """同じトピックのメモリを要約して統合"""
        # トピック別にグループ化
        all_docs = self.memory.collection.get(include=["documents", "metadatas"])
        topics = defaultdict(list)

        for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
            topic = meta.get("topic", "general")
            topics[topic].append(doc)

        # トピックごとに要約
        for topic, docs in topics.items():
            if len(docs) > max_per_topic:
                combined = "\n".join(docs)
                summary = llm.generate(
                    f"以下の{len(docs)}件の情報を{max_per_topic}件に要約:\n{combined}"
                )
                # 古いエントリを削除し、要約を保存
                # (実際の実装ではIDの追跡が必要)
```

---

## 9. トラブルシューティング

### 9.1 よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| 記憶が見つからない | 検索クエリとチャンクのミスマッチ | チャンクサイズの調整、ハイブリッド検索の導入 |
| 不正確な記憶の想起 | 類似だが異なる文脈の記憶 | メタデータフィルタ、リランキングの追加 |
| メモリ肥大化 | ガベージコレクション不足 | 定期的なクリーンアップ、TTLの設定 |
| 会話の文脈喪失 | 要約が重要情報を落とす | 重要事実の別途保存、ハイブリッド短期記憶 |
| レイテンシ増大 | 検索対象の増大 | インデックス最適化、キャッシュ、バッチ検索 |
| コンテキスト超過 | 取得記憶が多すぎる | top_kの制限、結果の要約 |

### 9.2 メモリデバッグツール

```python
# メモリシステムのデバッグ支援
class MemoryDebugger:
    """メモリシステムのデバッグ・分析ツール"""

    def __init__(self, memory_system):
        self.memory = memory_system

    def inspect_retrieval(self, query: str, top_k: int = 10) -> dict:
        """検索結果の詳細分析"""
        results = self.memory.long_term.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances", "metadatas"]
        )

        analysis = {
            "query": query,
            "num_results": len(results["documents"][0]),
            "results": []
        }

        for i, (doc, dist, meta) in enumerate(zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0]
        )):
            analysis["results"].append({
                "rank": i + 1,
                "distance": float(dist),
                "similarity": 1 - float(dist),
                "document_preview": doc[:200],
                "metadata": meta
            })

        return analysis

    def check_memory_health(self) -> dict:
        """メモリシステムの健全性チェック"""
        collection = self.memory.long_term.collection
        count = collection.count()

        return {
            "total_entries": count,
            "short_term_size": len(self.memory.short_term.messages),
            "working_memory_keys": list(self.memory.working.keys()),
            "metrics": self.memory.get_metrics()
            if hasattr(self.memory, "get_metrics") else "N/A"
        }
```

---

## 10. アンチパターン

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

### アンチパターン3: メモリの冗長保存

```python
# NG: 同じ情報を何度も保存
for turn in conversation:
    memory.store(f"ユーザーの名前は{user_name}")  # 毎ターン保存

# OK: 変更があった場合のみ保存
def store_if_new(memory, key, value):
    existing = memory.retrieve(key, top_k=1)
    if not existing or existing[0] != value:
        memory.store(value, metadata={"key": key, "updated": time.time()})
```

### アンチパターン4: チャンクサイズの不適切な設定

```python
# NG: チャンクが小さすぎる
chunks = chunk_by_tokens(text, chunk_size=50)  # 文脈不足

# NG: チャンクが大きすぎる
chunks = chunk_by_tokens(text, chunk_size=5000)  # ノイズが多い

# OK: 適切なサイズ（300-800トークン）でオーバーラップ付き
chunks = chunk_by_tokens(text, chunk_size=500, overlap=50)
```

---

## 11. FAQ

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

### Q4: メモリの永続化方法は？

| 方法 | 特徴 | 適用場面 |
|------|------|---------|
| ファイル保存（JSON） | シンプル、バックアップ容易 | プロトタイプ |
| SQLite | 組み込み型、SQL対応 | 小規模本番 |
| PostgreSQL + pgvector | スケーラブル、ベクトル検索 | 本番環境 |
| Redis | 高速、揮発性 | キャッシュ層 |
| S3 + DynamoDB | AWS統合、コスト最適 | クラウドネイティブ |

### Q5: メモリの一貫性をどう保つか？

```python
# メモリの一貫性維持パターン
class ConsistentMemory:
    def update_fact(self, key: str, new_value: str):
        """事実の更新時に一貫性を保つ"""
        # 1. 古い事実を検索
        old = self.retrieve(key, top_k=1)

        # 2. 矛盾チェック
        if old and self._contradicts(old[0], new_value):
            # 3. 古い事実を無効化
            self.invalidate(old[0])
            self.store(
                f"[更新] {key}: {new_value}（以前: {old[0]}）",
                metadata={"type": "fact_update"}
            )
        else:
            self.store(new_value, metadata={"key": key})
```

### Q6: エンベディングモデルの選択基準は？

| モデル | 次元数 | 性能 | 速度 | コスト |
|--------|-------|------|------|--------|
| all-MiniLM-L6-v2 | 384 | 中 | 非常に速い | 無料 |
| all-mpnet-base-v2 | 768 | 高 | 速い | 無料 |
| text-embedding-3-small | 1536 | 高 | API依存 | 安い |
| text-embedding-3-large | 3072 | 最高 | API依存 | 中程度 |
| Cohere embed-v3 | 1024 | 高 | API依存 | 中程度 |

**推奨**: プロトタイプは all-MiniLM-L6-v2、本番は text-embedding-3-small 以上。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 3層構造 | 短期（セッション）・作業（タスク）・長期（永続） |
| 短期記憶 | バッファ / スライディングウィンドウ / 要約 / ハイブリッド |
| 長期記憶 | ベクトルDB + RAGによるセマンティック検索 |
| 知識グラフ | エンティティ間の関係性を構造化保存 |
| エピソード記憶 | 過去の体験（成功・失敗）から学習 |
| チャンキング | トークン / セマンティック / 再帰的 |
| 統合設計 | 短期+長期+知識グラフ+エピソードを統合 |
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
5. Anthropic, "MCP Memory Server" — https://github.com/anthropics/mcp-memory
6. Robertson, S. et al., "The Probabilistic Relevance Framework: BM25 and Beyond" — https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf
