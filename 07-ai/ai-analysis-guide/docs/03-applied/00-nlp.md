# NLP — テキスト分類、固有表現抽出、感情分析

> 自然言語処理の主要タスクを実装し、テキストデータから価値ある情報を抽出する

## この章で学ぶこと

1. **テキスト前処理** — トークン化、ベクトル化、埋め込み表現の構築
2. **テキスト分類と感情分析** — 古典手法からTransformerファインチューニングまで
3. **固有表現抽出（NER）** — 系列ラベリングによる情報抽出
4. **テキスト生成と要約** — 文章自動生成、抽出型・生成型要約の実装
5. **実践的パイプライン構築** — 前処理からデプロイまでの一気通貫設計

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
from typing import List, Optional, Dict
from dataclasses import dataclass

@dataclass
class TokenInfo:
    """トークンの詳細情報"""
    surface: str      # 表層形
    base: str = ""    # 原形
    pos: str = ""     # 品詞
    reading: str = "" # 読み

class TextPreprocessor:
    """日本語/英語対応の高度なテキスト前処理"""

    def __init__(self, language: str = "ja"):
        self.language = language
        self._stopwords_ja = {
            "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
            "れ", "さ", "ある", "いる", "も", "する", "から", "な",
            "こと", "として", "い", "や", "れる", "など", "なっ",
            "ない", "この", "ため", "その", "あっ", "よう", "また",
            "もの", "という", "あり", "まで", "られ", "なる", "へ",
            "か", "だ", "これ", "によって", "により", "おり", "より",
        }
        self._stopwords_en = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "out", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just",
            "it", "its", "he", "she", "they", "them", "his", "her",
            "this", "that", "these", "those", "i", "me", "my", "we",
        }

    def normalize(self, text: str) -> str:
        """Unicode正規化 + 基本的なクリーニング"""
        # NFKC正規化（全角→半角、異体字統一）
        text = unicodedata.normalize("NFKC", text)
        # URL/メール/ハッシュタグの置換
        text = re.sub(r"https?://\S+", "[URL]", text)
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)
        text = re.sub(r"#(\w+)", r"[HASHTAG:\1]", text)
        text = re.sub(r"@(\w+)", r"[MENTION:\1]", text)
        # 連続する空白の正規化
        text = re.sub(r"\s+", " ", text).strip()
        # HTML タグの除去
        text = re.sub(r"<[^>]+>", "", text)
        # 制御文字の除去
        text = "".join(c for c in text if unicodedata.category(c)[0] != "C" or c in "\n\t ")
        return text

    def normalize_neologd(self, text: str) -> str:
        """NEologd 風の正規化（日本語向け）"""
        text = self.normalize(text)
        # 長音記号の正規化
        text = re.sub(r"[〜～]", "ー", text)
        # 繰り返し記号の削減
        text = re.sub(r"([!?！？]){2,}", r"\1", text)
        text = re.sub(r"(ー){2,}", "ー", text)
        text = re.sub(r"(っ){2,}", "っ", text)
        text = re.sub(r"(。){2,}", "。", text)
        # 括弧の正規化
        text = re.sub(r"[（\(]", "(", text)
        text = re.sub(r"[）\)]", ")", text)
        return text

    def tokenize_ja(self, text: str, with_pos: bool = False) -> List:
        """日本語形態素解析（MeCab）"""
        import MeCab
        tagger = MeCab.Tagger()
        parsed = tagger.parse(text)

        tokens = []
        for line in parsed.split("\n"):
            if line == "EOS" or line == "":
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            surface = parts[0]
            features = parts[1].split(",") if len(parts) > 1 else []

            if with_pos:
                tokens.append(TokenInfo(
                    surface=surface,
                    pos=features[0] if len(features) > 0 else "",
                    base=features[6] if len(features) > 6 else surface,
                    reading=features[7] if len(features) > 7 else "",
                ))
            else:
                tokens.append(surface)

        return tokens

    def tokenize_en(self, text: str) -> List[str]:
        """英語トークン化"""
        text = text.lower()
        # 基本的なトークン化（句読点を分離）
        text = re.sub(r"([.!?,;:'\"-])", r" \1 ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text.split()

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        if self.language == "ja":
            return self.tokenize_ja(text)
        return self.tokenize_en(text)

    def remove_stopwords(self, tokens: List[str],
                          custom_stopwords: set = None) -> List[str]:
        stopwords = (custom_stopwords or
                     (self._stopwords_ja if self.language == "ja"
                      else self._stopwords_en))
        return [t for t in tokens if t not in stopwords and len(t) > 1]

    def extract_keywords(self, text: str, top_k: int = 10) -> List[Dict]:
        """TF-IDFベースのキーワード抽出"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np

        tokens = self.tokenize(text)
        clean_tokens = self.remove_stopwords(tokens)

        # 疑似TF-IDF（単文書の場合はTFのみ）
        word_freq = {}
        for token in clean_tokens:
            word_freq[token] = word_freq.get(token, 0) + 1

        total = sum(word_freq.values())
        keywords = sorted(
            [{"word": w, "score": c / total} for w, c in word_freq.items()],
            key=lambda x: x["score"], reverse=True
        )[:top_k]

        return keywords

# 使用例
preprocessor = TextPreprocessor(language="ja")
text = "東京は今日も晴れです。  気温は２５度でした。https://example.com"
normalized = preprocessor.normalize(text)
tokens = preprocessor.tokenize(normalized)
clean_tokens = preprocessor.remove_stopwords(tokens)
print(f"正規化: {normalized}")
print(f"トークン: {tokens}")
print(f"前処理後: {clean_tokens}")
```

### コード例2: サブワードトークナイザの比較

```python
from transformers import AutoTokenizer

def compare_tokenizers(text: str):
    """複数のトークナイザの挙動を比較する"""
    tokenizer_names = {
        "BERT (日本語)": "cl-tohoku/bert-base-japanese-v3",
        "GPT-2": "gpt2",
        "T5": "t5-small",
        "Llama2": "meta-llama/Llama-2-7b-hf",
    }

    print(f"テキスト: '{text}'")
    print("-" * 60)

    for name, model_name in tokenizer_names.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)

            print(f"\n{name}:")
            print(f"  トークン数: {len(tokens)}")
            print(f"  トークン: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
            print(f"  ID: {ids[:10]}{'...' if len(ids) > 10 else ''}")
            print(f"  復元: {decoded[:100]}")
        except Exception as e:
            print(f"\n{name}: スキップ ({e})")

# compare_tokenizers("自然言語処理は人工知能の重要な研究分野です。")
# compare_tokenizers("Natural language processing is important.")
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

### コード例3: 古典的テキスト分類（本格版）

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV

class TextClassificationPipeline:
    """テキスト分類の包括的パイプライン"""

    def __init__(self, language: str = "ja"):
        self.language = language
        self.preprocessor = TextPreprocessor(language)
        self.pipeline = None

    def build_pipeline(self, model_type: str = "svm") -> Pipeline:
        """分類パイプラインを構築する"""
        tfidf = TfidfVectorizer(
            analyzer="char_wb" if self.language == "ja" else "word",
            ngram_range=(2, 4) if self.language == "ja" else (1, 2),
            max_features=50000,
            sublinear_tf=True,     # TFのlog正規化
            min_df=2,              # 最低2文書に出現
            max_df=0.95,           # 95%以上の文書に出現する語は除外
        )

        models = {
            "lr": LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced"
            ),
            "svm": CalibratedClassifierCV(
                LinearSVC(C=1.0, class_weight="balanced", max_iter=5000),
                cv=3
            ),
            "nb": MultinomialNB(alpha=0.1),
            "ensemble": VotingClassifier(
                estimators=[
                    ("lr", LogisticRegression(max_iter=1000, C=1.0)),
                    ("svm", CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
                    ("nb", MultinomialNB(alpha=0.1)),
                ],
                voting="soft"
            ),
        }

        self.pipeline = make_pipeline(tfidf, models[model_type])
        return self.pipeline

    def evaluate(self, texts, labels, cv=5):
        """交差検証で評価する"""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(
            self.pipeline, texts, labels,
            cv=skf, scoring="f1_weighted", n_jobs=-1
        )
        print(f"F1 (weighted): {scores.mean():.4f} (+/- {scores.std():.4f})")
        return scores

    def train_and_report(self, X_train, y_train, X_test, y_test):
        """学習して詳細レポートを出力する"""
        self.pipeline.fit(X_train, y_train)
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        return y_pred

# 使用例
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
labels = [1, 0, 1, 0, 1, 0, 1, 0]

pipeline = TextClassificationPipeline(language="ja")
pipeline.build_pipeline("lr")
pipeline.pipeline.fit(texts, labels)

test_texts = ["感動的な映画だった", "退屈な映画だった"]
preds = pipeline.pipeline.predict(test_texts)
for t, p in zip(test_texts, preds):
    print(f"  '{t}' -> {'ポジティブ' if p == 1 else 'ネガティブ'}")
```

### コード例4: BERTファインチューニング（完全版）

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class BERTClassifier:
    """日本語BERTのファインチューニングパイプライン"""

    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-v3",
                 num_labels: int = 2, max_length: int = 128):
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def prepare_dataset(self, texts, labels, test_size=0.2):
        """データセットを準備する"""
        dataset = Dataset.from_dict({"text": texts, "label": labels})
        dataset = dataset.train_test_split(test_size=test_size, seed=42,
                                            stratify_by_column="label")

        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        tokenized = dataset.map(tokenize_fn, batched=True,
                                 remove_columns=["text"])
        tokenized.set_format("torch")
        return tokenized

    def compute_metrics(self, eval_pred):
        """評価メトリクスを計算する"""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted"),
            "precision": precision_score(labels, predictions, average="weighted"),
            "recall": recall_score(labels, predictions, average="weighted"),
        }

    def train(self, tokenized_dataset, output_dir="./results",
              epochs=5, batch_size=16, lr=2e-5):
        """学習を実行する"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            logging_steps=50,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=2,
            report_to="none",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        trainer.train()
        return trainer

    def predict(self, texts: list) -> list:
        """テキストを分類する"""
        self.model.eval()
        inputs = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        return predictions.cpu().tolist()

# 使用例
# classifier = BERTClassifier(num_labels=2)
# dataset = classifier.prepare_dataset(texts, labels)
# trainer = classifier.train(dataset)
# predictions = classifier.predict(["素晴らしい映画だった"])
```

### コード例5: LLM によるゼロショット/フューショット分類

```python
from transformers import pipeline
import json

class LLMClassifier:
    """LLMを使ったゼロショット/フューショット分類"""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model=model_name
        )

    def classify_zero_shot(self, text: str, labels: list,
                            multi_label: bool = False) -> dict:
        """ゼロショット分類"""
        result = self.zero_shot(
            text, labels,
            multi_label=multi_label
        )
        return {
            "text": text,
            "predictions": [
                {"label": label, "score": round(score, 4)}
                for label, score in zip(result["labels"], result["scores"])
            ],
            "top_label": result["labels"][0],
            "top_score": round(result["scores"][0], 4),
        }

    def classify_batch(self, texts: list, labels: list) -> list:
        """バッチゼロショット分類"""
        results = self.zero_shot(texts, labels)
        if not isinstance(results, list):
            results = [results]
        return [
            {
                "text": text,
                "label": r["labels"][0],
                "score": round(r["scores"][0], 4),
            }
            for text, r in zip(texts, results)
        ]

    @staticmethod
    def few_shot_prompt(text: str, examples: list,
                         labels: list) -> str:
        """フューショット学習用のプロンプトを構築する"""
        prompt = "以下のテキストを分類してください。\n\n"
        prompt += f"カテゴリ: {', '.join(labels)}\n\n"
        prompt += "例:\n"
        for ex in examples:
            prompt += f"テキスト: {ex['text']}\n"
            prompt += f"カテゴリ: {ex['label']}\n\n"
        prompt += f"テキスト: {text}\n"
        prompt += "カテゴリ: "
        return prompt

# 使用例
# classifier = LLMClassifier()
# result = classifier.classify_zero_shot(
#     "この商品は品質が良く、価格も手頃です",
#     ["ポジティブ", "ネガティブ", "ニュートラル"]
# )
# print(f"分類結果: {result['top_label']} ({result['top_score']})")
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
  ┌──────────────────────────────────────────┐
  │ PER (Person)        : 人名               │
  │ ORG (Organization)  : 組織名             │
  │ LOC (Location)      : 地名               │
  │ DATE                : 日付               │
  │ MONEY               : 金額               │
  │ TTL (Title)         : 肩書               │
  │ PRODUCT             : 製品名             │
  │ EVENT               : イベント名         │
  │ PERCENT             : パーセンテージ     │
  │ QUANTITY            : 数量               │
  └──────────────────────────────────────────┘

  BIOES (拡張):
  B = Begin, I = Inside, O = Outside
  E = End（エンティティの終端）
  S = Single（1トークンのエンティティ）
```

### コード例6: spaCy + Transformers による NER

```python
import spacy
from transformers import pipeline
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float = 1.0

class NERPipeline:
    """複数バックエンドに対応するNERパイプライン"""

    def __init__(self, backend: str = "spacy", model_name: str = None):
        self.backend = backend

        if backend == "spacy":
            model_name = model_name or "ja_core_news_lg"
            self.nlp = spacy.load(model_name)
        elif backend == "transformers":
            model_name = model_name or "dslim/bert-base-NER"
            self.ner_pipeline = pipeline(
                "ner", model=model_name,
                aggregation_strategy="simple"
            )

    def extract(self, text: str) -> List[Entity]:
        """固有表現を抽出する"""
        if self.backend == "spacy":
            return self._extract_spacy(text)
        else:
            return self._extract_transformers(text)

    def _extract_spacy(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        return [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            )
            for ent in doc.ents
        ]

    def _extract_transformers(self, text: str) -> List[Entity]:
        results = self.ner_pipeline(text)
        return [
            Entity(
                text=ent["word"],
                label=ent["entity_group"],
                start=ent["start"],
                end=ent["end"],
                score=round(ent["score"], 4),
            )
            for ent in results
        ]

    def extract_batch(self, texts: List[str]) -> List[List[Entity]]:
        """バッチでNERを実行する"""
        return [self.extract(text) for text in texts]

    def format_output(self, text: str, entities: List[Entity]) -> str:
        """エンティティをハイライトした文字列を生成する"""
        result = text
        # 末尾からの置換で位置がずれないようにする
        for ent in sorted(entities, key=lambda e: e.start, reverse=True):
            result = (
                result[:ent.start]
                + f"[{ent.text}]({ent.label})"
                + result[ent.end:]
            )
        return result

# 使用例
ner = NERPipeline(backend="transformers", model_name="dslim/bert-base-NER")
text = "Apple CEO Tim Cook announced new products in San Francisco."
entities = ner.extract(text)
for ent in entities:
    print(f"  [{ent.label}] {ent.text} (信頼度: {ent.score})")
print(f"\n  注釈付き: {ner.format_output(text, entities)}")
```

### コード例7: カスタム NER モデルの学習

```python
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np

class CustomNERTrainer:
    """カスタムNERモデルの学習パイプライン"""

    def __init__(self, model_name: str = "cl-tohoku/bert-base-japanese-v3",
                 label_list: list = None):
        self.model_name = model_name
        self.label_list = label_list or [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG",
            "B-LOC", "I-LOC", "B-DATE", "I-DATE",
        ]
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self.label_list),
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def tokenize_and_align(self, examples):
        """トークン化してラベルをアラインメントする"""
        tokenized = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=128,
        )

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # 特殊トークンは無視
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # サブワードの2番目以降: B- を I- に変換
                    lbl = label[word_idx]
                    if self.label_list[lbl].startswith("B-"):
                        lbl = self.label2id[
                            self.label_list[lbl].replace("B-", "I-")
                        ]
                    label_ids.append(lbl)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    def compute_metrics(self, eval_pred):
        """NER用のメトリクスを計算する"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        # -100を除外して評価
        true_labels = []
        true_predictions = []
        for pred, label in zip(predictions, labels):
            for p, l in zip(pred, label):
                if l != -100:
                    true_labels.append(self.label_list[l])
                    true_predictions.append(self.label_list[p])

        # エンティティレベルのF1
        from seqeval.metrics import f1_score, classification_report
        f1 = f1_score([true_labels], [true_predictions])
        return {"f1": f1}

    def train(self, train_dataset, eval_dataset, output_dir="./ner_model"):
        """NERモデルを学習する"""
        data_collator = DataCollatorForTokenClassification(
            self.tokenizer, pad_to_multiple_of=8
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            learning_rate=3e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        return trainer
```

---

## 4. 感情分析

### コード例8: 多機能感情分析パイプライン

```python
from transformers import pipeline
import pandas as pd
from typing import Dict, List
import numpy as np

class SentimentAnalyzer:
    """マルチ言語・マルチアスペクト感情分析"""

    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.pipe = pipeline("sentiment-analysis", model=model_name,
                              device=-1)

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

    def analyze_with_context(self, text: str,
                               context: str = None) -> Dict:
        """コンテキスト付き感情分析"""
        if context:
            input_text = f"[コンテキスト: {context}] {text}"
        else:
            input_text = text

        result = self.pipe(input_text, truncation=True)[0]
        return {
            "text": text,
            "context": context,
            "label": result["label"],
            "score": round(result["score"], 4),
        }

    def analyze_trends(self, texts: list,
                        timestamps: list = None) -> pd.DataFrame:
        """時系列での感情トレンド分析"""
        results = self.pipe(texts, truncation=True)

        df = pd.DataFrame({
            "text": texts,
            "label": [r["label"] for r in results],
            "score": [r["score"] for r in results],
        })

        if timestamps:
            df["timestamp"] = timestamps
            df = df.sort_values("timestamp")

        # 感情スコアの移動平均
        df["score_ma"] = df["score"].rolling(window=5, min_periods=1).mean()

        return df

# 使用例
analyzer = SentimentAnalyzer()

reviews = [
    "This product is amazing! Best purchase ever.",
    "Terrible quality, broke after one day.",
    "It's okay, nothing special.",
    "Exceeded my expectations, highly recommended!",
    "Not worth the price at all.",
]

df = analyzer.analyze(reviews)
print(df.to_string(index=False))

# アスペクトベース
# result = analyzer.analyze_aspects(
#     "料理は美味しかったが、サービスが遅かった",
#     aspects=["料理", "サービス", "雰囲気"]
# )
```

---

## 5. テキスト埋め込みと類似度検索

### コード例9: 高性能セマンティック検索

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Optional
import json

class SemanticSearch:
    """文埋め込みによる高度なセマンティック検索"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.metadata = []
        self.embeddings = None

    def index(self, documents: list, metadata: list = None) -> None:
        """ドキュメントをインデックス化"""
        self.documents = documents
        self.metadata = metadata or [{}] * len(documents)
        self.embeddings = self.model.encode(
            documents, normalize_embeddings=True,
            show_progress_bar=True, batch_size=32
        )

    def search(self, query: str, top_k: int = 5,
               threshold: float = 0.0) -> list:
        """クエリに最も類似するドキュメントを検索"""
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        )
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append({
                    "document": self.documents[idx],
                    "similarity": round(sim, 4),
                    "index": int(idx),
                    "metadata": self.metadata[idx],
                })
        return results

    def find_similar_pairs(self, threshold: float = 0.8) -> list:
        """類似度が高い文書ペアを検出する"""
        sim_matrix = np.dot(self.embeddings, self.embeddings.T)
        pairs = []
        n = len(self.documents)

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] >= threshold:
                    pairs.append({
                        "doc1": self.documents[i],
                        "doc2": self.documents[j],
                        "similarity": round(float(sim_matrix[i][j]), 4),
                    })

        return sorted(pairs, key=lambda x: x["similarity"], reverse=True)

    def cluster_documents(self, n_clusters: int = 5) -> dict:
        """ドキュメントをクラスタリングする"""
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                "document": self.documents[i],
                "index": i,
            })

        return clusters

# 使用例
search = SemanticSearch()
docs = [
    "Python is a programming language.",
    "Machine learning uses data to learn patterns.",
    "Tokyo is the capital of Japan.",
    "Neural networks are inspired by the brain.",
    "Deep learning requires large datasets.",
    "Japan is an island country in East Asia.",
    "Python supports object-oriented programming.",
    "Gradient descent optimizes neural network weights.",
]
search.index(docs)

results = search.search("AI and data science", top_k=3)
for r in results:
    print(f"  [{r['similarity']:.3f}] {r['document']}")

# 類似ペア検出
print("\n類似文書ペア:")
pairs = search.find_similar_pairs(threshold=0.5)
for p in pairs[:5]:
    print(f"  [{p['similarity']:.3f}] '{p['doc1']}' <-> '{p['doc2']}'")
```

### コード例10: RAG (Retrieval-Augmented Generation) パイプライン

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SimpleRAG:
    """シンプルなRAGパイプライン"""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(embedding_model)
        self.documents = []
        self.embeddings = None

    def add_documents(self, documents: list):
        """ドキュメントを追加する"""
        self.documents.extend(documents)
        self.embeddings = self.encoder.encode(
            self.documents, normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """関連ドキュメントを検索する"""
        query_emb = self.encoder.encode(
            [query], normalize_embeddings=True
        )
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            {
                "text": self.documents[idx],
                "score": float(similarities[idx]),
            }
            for idx in top_indices
        ]

    def build_prompt(self, query: str, contexts: list) -> str:
        """コンテキスト付きプロンプトを構築する"""
        context_text = "\n\n".join(
            f"[文書{i+1}] {ctx['text']}"
            for i, ctx in enumerate(contexts)
        )

        prompt = f"""以下の参考文書に基づいて質問に回答してください。
参考文書に情報がない場合は「情報がありません」と回答してください。

## 参考文書
{context_text}

## 質問
{query}

## 回答
"""
        return prompt

    def query(self, question: str, top_k: int = 3) -> dict:
        """質問に対して検索→回答生成を行う"""
        # 1. 検索
        contexts = self.retrieve(question, top_k=top_k)

        # 2. プロンプト構築
        prompt = self.build_prompt(question, contexts)

        # 3. LLMで生成（ここではプロンプトを返す）
        return {
            "question": question,
            "contexts": contexts,
            "prompt": prompt,
            # "answer": llm.generate(prompt)  # LLM呼び出し
        }

# 使用例
# rag = SimpleRAG()
# rag.add_documents([
#     "Pythonは1991年にGuido van Rossumによって作られた。",
#     "Pythonはインタプリタ型の高水準プログラミング言語。",
#     "Pythonはデータサイエンスや機械学習で広く使われている。",
# ])
# result = rag.query("Pythonは誰が作った？")
# print(result["prompt"])
```

---

## 比較表

### テキスト分類手法の比較

| 手法 | 精度 | 速度 | データ量要件 | 解釈性 | 多言語 | コスト |
|---|---|---|---|---|---|---|
| BoW + NaiveBayes | 中 | 極速 | 少量OK | 高い | 要対応 | 無料 |
| TF-IDF + SVM | 中〜高 | 速い | 中量 | 中程度 | 要対応 | 無料 |
| Word2Vec + LSTM | 高い | 中程度 | 中量 | 低い | 要対応 | GPU推奨 |
| BERT Fine-tuning | 最高 | 遅い | 少量OK | 低い | モデル依存 | GPU必要 |
| GPT Few-shot | 高い | 遅い | 極少量 | 低い | 高い | API課金 |
| GPT Zero-shot | 中〜高 | 遅い | 不要 | 低い | 高い | API課金 |

### 日本語NLPライブラリの比較

| ライブラリ | 形態素解析 | NER | 分類 | 速度 | 精度 | 用途 |
|---|---|---|---|---|---|---|
| MeCab | ○ | x | x | 極速 | 高い | 前処理 |
| Janome | ○ | x | x | 速い | 中程度 | 軽量環境 |
| spaCy (ja) | ○ | ○ | ○ | 速い | 高い | パイプライン |
| GiNZA | ○ | ○ | △ | 中程度 | 高い | 詳細分析 |
| SudachiPy | ○ | x | x | 速い | 高い | 正規化に強い |
| Transformers (BERT) | △ | ○ | ○ | 遅い | 最高 | 高精度タスク |

### 埋め込みモデルの比較

| モデル | 次元数 | 速度 | 品質 | 多言語 | サイズ |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | 高速 | 高い | 英語中心 | 80MB |
| multilingual-e5-large | 1024 | 中程度 | 最高 | ○ | 2.2GB |
| paraphrase-multilingual | 768 | 中程度 | 高い | ○ | 1.1GB |
| text-embedding-ada-002 | 1536 | API依存 | 最高 | ○ | API |
| text-embedding-3-small | 1536 | API依存 | 高い | ○ | API |

---

## アンチパターン

### アンチパターン1: テキスト長を考慮しないトークン化

```python
# BAD: BERTの512トークン制限を無視
inputs = tokenizer(long_text, return_tensors="pt")  # 切り詰められる!

# GOOD: 長文対応戦略
def handle_long_text(text, tokenizer, model, max_length=512, stride=128):
    """オーバーラップチャンク分割で長文を処理"""
    inputs = tokenizer(
        text, max_length=max_length, truncation=True,
        stride=stride, return_overflowing_tokens=True,
        return_tensors="pt", padding=True
    )

    all_logits = []
    for i in range(inputs["input_ids"].shape[0]):
        chunk_inputs = {
            k: v[i:i+1] for k, v in inputs.items()
            if k != "overflow_to_sample_mapping"
        }
        with torch.no_grad():
            outputs = model(**chunk_inputs)
            all_logits.append(outputs.logits)

    # チャンクの結果を統合（平均）
    avg_logits = torch.stack(all_logits).mean(dim=0)
    return avg_logits
```

### アンチパターン2: 前処理の不統一

```python
# BAD: 学習時と推論時で前処理が異なる
# 学習時: 小文字化 + 記号除去
# 推論時: そのまま入力 → 性能低下

# GOOD: 前処理をパイプラインに組み込む
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

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

### アンチパターン3: 日本語のBPEトークン化の落とし穴

```python
# BAD: 英語用BPEトークナイザを日本語に適用
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.tokenize("東京は日本の首都です")
# → 1文字ずつバイト分割され、大量のトークンを消費

# GOOD: 日本語対応モデルを使う
tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-v3")
tokens = tokenizer.tokenize("東京は日本の首都です")
# → 形態素に近い単位でトークン化される

# 多言語の場合: 多言語対応モデルを使用
tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
```

---

## FAQ

### Q1: BERTファインチューニングに必要なデータ量は？

**A:** タスクによるが、1クラスあたり100〜500サンプルで有効。特にBERTは事前学習の知識があるため少量データでも高性能。10サンプル程度ならFew-shot学習（GPT系）の方が適する。1000サンプル以上あればBERTファインチューニングが安定する。

**データ量の目安:**

| データ量 | 推奨アプローチ | 期待精度 |
|----------|-------------|---------|
| 0件 | GPT Zero-shot | 中〜高 |
| 5-20件 | GPT Few-shot | 高い |
| 100-500件 | BERT Fine-tuning | 高い |
| 1000件以上 | BERT Fine-tuning | 最高 |
| 10000件以上 | 古典ML or BERT | 最高 |

### Q2: 日本語NLPの特有の課題は？

**A:** (1) 単語の区切りがない（形態素解析が必要）、(2) 漢字・ひらがな・カタカナの混在、(3) 敬語による表現の多様性、(4) 学習データが英語に比べて少ない。BERTモデルは「cl-tohoku/bert-base-japanese-v3」、「nlp-waseda/roberta-base-japanese」等の日本語特化モデルを使用する。

### Q3: 感情分析の精度を上げるには？

**A:** (1) ドメイン固有のラベル付きデータで追加学習、(2) アスペクトベース分析で側面ごとに評価、(3) 否定表現（「良くない」）やスラングへの対応、(4) 文脈を考慮（皮肉、比喩の検出）。LLMをアノテーターとして使い、高品質なラベルデータを効率的に作成する手法も有効。

### Q4: RAG と Fine-tuning のどちらを使うべきですか？

**A:** 用途によります。

| 観点 | RAG | Fine-tuning |
|------|-----|-------------|
| データの鮮度 | リアルタイム更新可能 | 再学習が必要 |
| ハルシネーション | 根拠を提示可能 | 抑制しにくい |
| コスト | 検索インフラが必要 | 学習GPU が必要 |
| カスタマイズ | 外部知識の追加が容易 | モデルの挙動を変更 |
| 推奨場面 | FAQ、ドキュメント検索 | スタイル変更、専門タスク |

両者を組み合わせることも有効です（Fine-tuned モデル + RAG）。

### Q5: テキスト分類で不均衡データにどう対処しますか？

**A:** 以下の戦略を組み合わせます。

1. **データレベル**: オーバーサンプリング（SMOTE）、アンダーサンプリング
2. **損失関数**: Focal Loss、Class-weighted Cross Entropy
3. **評価指標**: Accuracy ではなく F1-score, AUPRC を使用
4. **データ拡張**: 同義語置換、バック翻訳、LLMによるパラフレーズ生成

---

## まとめ

| 項目 | 要点 |
|---|---|
| 前処理 | 正規化→トークン化→ベクトル化。一貫したパイプラインで管理 |
| テキスト分類 | ベースライン: TF-IDF+SVM。高精度: BERT Fine-tuning |
| NER | BIOタグで系列ラベリング。spaCy or Transformers |
| 感情分析 | 事前学習済みモデルで即座に利用可能。ドメイン適応で精度向上 |
| 埋め込み | SentenceTransformersで文ベクトル化。類似度検索に活用 |
| RAG | 検索+生成で知識ベースのQAを構築 |
| Zero/Few-shot | ラベルデータなしでGPT/BARTで分類可能 |

---

## 次に読むべきガイド

- [01-computer-vision.md](./01-computer-vision.md) — コンピュータビジョンの応用
- [02-mlops.md](./02-mlops.md) — NLPモデルのデプロイと運用

---

## 参考文献

1. **Jacob Devlin et al.** "BERT: Pre-training of Deep Bidirectional Transformers" NAACL 2019
2. **Hugging Face** "Transformers Documentation" — https://huggingface.co/docs/transformers/
3. **Daniel Jurafsky, James H. Martin** "Speech and Language Processing" 3rd Edition (Draft) — https://web.stanford.edu/~jurafsky/slp3/
4. **Reimers, N. & Gurevych, I.** "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" EMNLP 2019
5. **Lewis, P. et al.** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" NeurIPS 2020
