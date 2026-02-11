# NLP — テキスト分類、固有表現抽出、感情分析

> 自然言語処理の主要タスクを実装し、テキストデータから価値ある情報を抽出する

## この章で学ぶこと

1. **テキスト前処理** — トークン化、ベクトル化、埋め込み表現の構築
2. **テキスト分類と感情分析** — 古典手法からTransformerファインチューニングまで
3. **固有表現抽出（NER）** — 系列ラベリングによる情報抽出

---

## 1. テキスト前処理

### NLPパイプライン

```
生テキスト → 前処理パイプライン → 特徴量 → モデル → 出力

┌──────────────┐
│ 生テキスト   │
│ "東京は晴れ" │
└──────┬───────┘
       │
       v
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ 正規化       │   │ トークン化   │   │ ベクトル化   │
│ ・小文字化   │──>│ ・形態素解析 │──>│ ・BoW        │
│ ・記号除去   │   │ ・BPE/WordPiece│ │ ・TF-IDF     │
│ ・Unicode正規│   │ ・サブワード │   │ ・Word2Vec   │
└──────────────┘   └──────────────┘   │ ・BERT埋め込み│
                                       └──────────────┘
                                              │
                                              v
                                       ┌──────────────┐
                                       │ モデル       │
                                       │ ・SVM        │
                                       │ ・BERT       │
                                       │ ・GPT        │
                                       └──────────────┘
```

### コード例1: テキスト前処理パイプライン

```python
import re
import unicodedata
from typing import List

class TextPreprocessor:
    """日本語/英語対応のテキスト前処理"""

    def __init__(self, language: str = "ja"):
        self.language = language

    def normalize(self, text: str) -> str:
        """Unicode正規化 + 基本的なクリーニング"""
        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"https?://\S+", "[URL]", text)
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_ja(self, text: str) -> List[str]:
        """日本語形態素解析（MeCab）"""
        import MeCab
        tagger = MeCab.Tagger("-Owakati")
        return tagger.parse(text).strip().split()

    def tokenize_en(self, text: str) -> List[str]:
        """英語トークン化"""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        return text.split()

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        if self.language == "ja":
            return self.tokenize_ja(text)
        return self.tokenize_en(text)

    def remove_stopwords(self, tokens: List[str],
                          stopwords: set = None) -> List[str]:
        if stopwords is None:
            stopwords = {"の", "に", "は", "を", "た", "が", "で",
                         "て", "と", "し", "れ", "さ", "ある", "いる",
                         "も", "する", "から", "な", "こと", "として"}
        return [t for t in tokens if t not in stopwords and len(t) > 1]

# 使用例
preprocessor = TextPreprocessor(language="ja")
text = "東京は今日も晴れです。  気温は２５度でした。"
tokens = preprocessor.tokenize(text)
clean_tokens = preprocessor.remove_stopwords(tokens)
print(f"トークン: {tokens}")
print(f"前処理後: {clean_tokens}")
```

---

## 2. テキスト分類

### 分類アプローチの進化

```
テキスト分類の発展:

  2010年以前         2013-2018          2018-現在
  ┌─────────┐     ┌───────────┐     ┌────────────┐
  │ BoW     │     │ Word2Vec  │     │ BERT       │
  │ TF-IDF  │ ──> │ + CNN/LSTM│ ──> │ GPT        │
  │ + SVM   │     │           │     │ Fine-tuning│
  │ + NB    │     │           │     │ Few-shot   │
  └─────────┘     └───────────┘     └────────────┘
  手作り特徴量     分散表現+DL        事前学習+転移
  精度: 中         精度: 高            精度: 最高
```

### コード例2: 古典的テキスト分類

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# サンプルデータ
texts = [
    "この映画は素晴らしい演技で感動した",
    "ストーリーが退屈で眠くなった",
    "映像美が際立つ傑作だ",
    "期待外れの駄作だった",
    "心温まる感動的な作品",
    "つまらない展開の連続で苦痛だった",
    "演技力に圧倒される名作",
    "時間の無駄だった",
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=ポジティブ, 0=ネガティブ

# TF-IDF + 各種分類器
models = {
    "LogisticRegression": make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
        LogisticRegression(max_iter=1000)
    ),
    "LinearSVC": make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
        LinearSVC()
    ),
    "MultinomialNB": make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
        MultinomialNB()
    ),
}

for name, model in models.items():
    model.fit(texts, labels)
    print(f"\n{name}:")

    # 新しいテキストで予測
    test_texts = ["感動的な映画だった", "退屈な映画だった"]
    preds = model.predict(test_texts)
    for t, p in zip(test_texts, preds):
        print(f"  '{t}' → {'ポジティブ' if p == 1 else 'ネガティブ'}")
```

### コード例3: BERTファインチューニング

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
import torch
import numpy as np

def fine_tune_bert(texts, labels, model_name="cl-tohoku/bert-base-japanese"):
    """日本語BERTのファインチューニング"""

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # データセット作成
    dataset = Dataset.from_dict({"text": texts, "label": labels})
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length",
                         truncation=True, max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True)

    # 学習設定
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model, tokenizer

# model, tokenizer = fine_tune_bert(texts, labels)
```

---

## 3. 固有表現抽出（NER）

### NERのタグ体系

```
BIO タグ体系:

  テキスト: "田中太郎は東京大学の教授です"

  トークン:  田中  太郎  は  東京  大学  の  教授  です
  タグ:      B-PER I-PER O  B-ORG I-ORG O  B-TTL O

  B = Begin（エンティティの開始）
  I = Inside（エンティティの内部）
  O = Outside（エンティティ外）

  エンティティ種別:
  ┌──────────────────────────────────┐
  │ PER (Person)    : 人名           │
  │ ORG (Organization): 組織名       │
  │ LOC (Location)  : 地名           │
  │ DATE            : 日付           │
  │ MONEY           : 金額           │
  │ TTL (Title)     : 肩書           │
  └──────────────────────────────────┘
```

### コード例4: spaCyによるNER

```python
# spaCy による固有表現抽出
import spacy

def extract_entities(text: str, model_name: str = "ja_core_news_lg"):
    """spaCyで固有表現を抽出"""
    nlp = spacy.load(model_name)
    doc = nlp(text)

    entities = []
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        })

    return entities

# Hugging Face Transformers によるNER
from transformers import pipeline

def ner_with_transformers(text: str,
                           model_name: str = "dslim/bert-base-NER"):
    """Transformerベースの固有表現抽出"""
    ner_pipeline = pipeline("ner", model=model_name,
                            aggregation_strategy="simple")
    results = ner_pipeline(text)

    entities = []
    for ent in results:
        entities.append({
            "text": ent["word"],
            "label": ent["entity_group"],
            "score": round(ent["score"], 4),
            "start": ent["start"],
            "end": ent["end"],
        })

    return entities

# 使用例
text = "Apple CEO Tim Cook announced new products in San Francisco."
entities = ner_with_transformers(text)
for ent in entities:
    print(f"  [{ent['label']}] {ent['text']} (信頼度: {ent['score']})")
```

### コード例5: 感情分析パイプライン

```python
from transformers import pipeline
import pandas as pd

class SentimentAnalyzer:
    """マルチ言語感情分析"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.pipe = pipeline("sentiment-analysis", model=model_name)

    def analyze(self, texts: list) -> pd.DataFrame:
        """テキストリストの感情分析"""
        results = self.pipe(texts, truncation=True, max_length=512)

        df = pd.DataFrame({
            "text": texts,
            "label": [r["label"] for r in results],
            "score": [round(r["score"], 4) for r in results],
        })
        return df

    def analyze_aspects(self, text: str, aspects: list) -> dict:
        """アスペクトベースの感情分析"""
        results = {}
        for aspect in aspects:
            prompt = f"{aspect}について: {text}"
            result = self.pipe(prompt, truncation=True)[0]
            results[aspect] = {
                "label": result["label"],
                "score": round(result["score"], 4),
            }
        return results

# 使用例
analyzer = SentimentAnalyzer()

reviews = [
    "This product is amazing! Best purchase ever.",
    "Terrible quality, broke after one day.",
    "It's okay, nothing special.",
]

df = analyzer.analyze(reviews)
print(df.to_string(index=False))

# アスペクトベース
# result = analyzer.analyze_aspects(
#     "料理は美味しかったが、サービスが遅かった",
#     aspects=["料理", "サービス", "雰囲気"]
# )
```

### コード例6: テキスト埋め込みと類似度検索

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    """文埋め込みによるセマンティック検索"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index(self, documents: list) -> None:
        """ドキュメントをインデックス化"""
        self.documents = documents
        self.embeddings = self.model.encode(documents, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list:
        """クエリに最も類似するドキュメントを検索"""
        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "similarity": float(similarities[idx]),
                "index": int(idx),
            })
        return results

# 使用例
search = SemanticSearch()
docs = [
    "Python is a programming language.",
    "Machine learning uses data to learn patterns.",
    "Tokyo is the capital of Japan.",
    "Neural networks are inspired by the brain.",
    "Deep learning requires large datasets.",
]
search.index(docs)

results = search.search("AI and data science", top_k=3)
for r in results:
    print(f"  [{r['similarity']:.3f}] {r['document']}")
```

---

## 比較表

### テキスト分類手法の比較

| 手法 | 精度 | 速度 | データ量要件 | 解釈性 | 多言語 |
|---|---|---|---|---|---|
| BoW + NaiveBayes | 中 | 極速 | 少量OK | 高い | △ |
| TF-IDF + SVM | 中〜高 | 速い | 中量 | 中程度 | △ |
| Word2Vec + LSTM | 高い | 中程度 | 中量 | 低い | △ |
| BERT Fine-tuning | 最高 | 遅い | 少量OK | 低い | ○ |
| GPT Few-shot | 高い | 遅い | 極少量 | 低い | ○ |
| GPT Zero-shot | 中〜高 | 遅い | 不要 | 低い | ○ |

### 日本語NLPライブラリの比較

| ライブラリ | 形態素解析 | NER | 分類 | 速度 | 精度 | 用途 |
|---|---|---|---|---|---|---|
| MeCab | ○ | × | × | 極速 | 高い | 前処理 |
| Janome | ○ | × | × | 速い | 中程度 | 軽量環境 |
| spaCy (ja) | ○ | ○ | ○ | 速い | 高い | パイプライン |
| GiNZA | ○ | ○ | △ | 中程度 | 高い | 詳細分析 |
| Transformers (BERT) | △ | ○ | ○ | 遅い | 最高 | 高精度タスク |

---

## アンチパターン

### アンチパターン1: テキスト長を考慮しないトークン化

```python
# BAD: BERTの512トークン制限を無視
inputs = tokenizer(long_text, return_tensors="pt")  # 切り詰められる!

# GOOD: 長文対応戦略
def handle_long_text(text, tokenizer, max_length=512, stride=128):
    """オーバーラップチャンク分割で長文を処理"""
    tokens = tokenizer(text, return_offsets_mapping=True)
    total_tokens = len(tokens["input_ids"])

    if total_tokens <= max_length:
        return [tokenizer(text, max_length=max_length,
                          truncation=True, return_tensors="pt")]

    # スライディングウィンドウで分割
    chunks = tokenizer(
        text, max_length=max_length, truncation=True,
        stride=stride, return_overflowing_tokens=True,
        return_tensors="pt"
    )
    return chunks
```

### アンチパターン2: 前処理の不統一

```python
# BAD: 学習時と推論時で前処理が異なる
# 学習時: 小文字化 + 記号除去
# 推論時: そのまま入力 → 性能低下

# GOOD: 前処理をパイプラインに組み込む
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# パイプラインに前処理を含める
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        preprocessor=lambda x: TextPreprocessor("en").normalize(x),
        analyzer="char_wb",
        ngram_range=(2, 4)
    )),
    ("classifier", LogisticRegression()),
])
# 学習時も推論時も同じ前処理が自動適用
```

---

## FAQ

### Q1: BERTファインチューニングに必要なデータ量は？

**A:** タスクによるが、1クラスあたり100〜500サンプルで有効。特にBERTは事前学習の知識があるため少量データでも高性能。10サンプル程度ならFew-shot学習（GPT系）の方が適する。1000サンプル以上あればBERTファインチューニングが安定する。

### Q2: 日本語NLPの特有の課題は？

**A:** (1) 単語の区切りがない（形態素解析が必要）、(2) 漢字・ひらがな・カタカナの混在、(3) 敬語による表現の多様性、(4) 学習データが英語に比べて少ない。BERTモデルは「cl-tohoku/bert-base-japanese」、「nlp-waseda/roberta-base-japanese」等の日本語特化モデルを使用する。

### Q3: 感情分析の精度を上げるには？

**A:** (1) ドメイン固有のラベル付きデータで追加学習、(2) アスペクトベース分析で側面ごとに評価、(3) 否定表現（「良くない」）やスラングへの対応、(4) 文脈を考慮（皮肉、比喩の検出）。LLMをアノテーターとして使い、高品質なラベルデータを効率的に作成する手法も有効。

---

## まとめ

| 項目 | 要点 |
|---|---|
| 前処理 | 正規化→トークン化→ベクトル化。一貫したパイプラインで管理 |
| テキスト分類 | ベースライン: TF-IDF+SVM。高精度: BERT Fine-tuning |
| NER | BIOタグで系列ラベリング。spaCy or Transformers |
| 感情分析 | 事前学習済みモデルで即座に利用可能。ドメイン適応で精度向上 |
| 埋め込み | SentenceTransformersで文ベクトル化。類似度検索に活用 |

---

## 次に読むべきガイド

- [01-computer-vision.md](./01-computer-vision.md) — コンピュータビジョンの応用
- [02-mlops.md](./02-mlops.md) — NLPモデルのデプロイと運用

---

## 参考文献

1. **Jacob Devlin et al.** "BERT: Pre-training of Deep Bidirectional Transformers" NAACL 2019
2. **Hugging Face** "Transformers Documentation" — https://huggingface.co/docs/transformers/
3. **Daniel Jurafsky, James H. Martin** "Speech and Language Processing" 3rd Edition (Draft) — https://web.stanford.edu/~jurafsky/slp3/
