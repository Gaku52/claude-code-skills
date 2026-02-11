# 検索エンジン設計

> 大規模データセットに対する全文検索システムの設計原則を、転置インデックス・ランキングアルゴリズム・Elasticsearch の実装を通じて解説し、検索品質とパフォーマンスを両立させるアーキテクチャを構築する

## この章で学ぶこと

1. **検索エンジンの基本原理** — 転置インデックス、トークナイズ、TF-IDF/BM25 ランキング
2. **システムアーキテクチャ** — インデックス構築パイプライン、クエリ処理フロー、分散検索
3. **Elasticsearch を用いた実装** — マッピング設計、日本語解析、クエリ最適化

---

## 1. 検索エンジンの基本原理

### 1.1 転置インデックス

```
文書:
  Doc1: "東京の天気は晴れです"
  Doc2: "大阪の天気は雨です"
  Doc3: "東京タワーの観光ガイド"

転置インデックス:
  "東京"   → [Doc1, Doc3]
  "天気"   → [Doc1, Doc2]
  "晴れ"   → [Doc1]
  "大阪"   → [Doc2]
  "雨"     → [Doc2]
  "タワー" → [Doc3]
  "観光"   → [Doc3]

検索 "東京 天気":
  "東京" → {Doc1, Doc3}
  "天気" → {Doc1, Doc2}
  AND演算: {Doc1, Doc3} ∩ {Doc1, Doc2} = {Doc1}
```

### 1.2 検索パイプライン

```
インデックス構築 (Write Path)

  Raw Data --> [Crawler/Ingest] --> [Text Extraction] --> [Analyzer]
                                                            |
                                                     +------+------+
                                                     |             |
                                                [Tokenizer]  [Filter]
                                                     |             |
                                                "東京の天気"   小文字化
                                                     |        ストップワード除去
                                                ["東京","の","天気"] ステミング
                                                     |        同義語展開
                                                     v
                                              [Inverted Index]

クエリ処理 (Read Path)

  Query "東京 天気" --> [Query Parser] --> [Analyzer(同一処理)]
       --> [Index Lookup] --> [Scoring (BM25)] --> [Sort & Filter]
       --> [Highlight] --> [Results]
```

### 1.3 ランキング: BM25 アルゴリズム

```
BM25 スコア計算:

  score(D, Q) = SUM[ IDF(qi) * (f(qi,D) * (k1+1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl)) ]

  各要素:
  - IDF(qi):   逆文書頻度。珍しい語ほど高スコア
  - f(qi,D):   文書D内での語qiの出現頻度
  - |D|:       文書Dの長さ
  - avgdl:     全文書の平均長
  - k1 (=1.2): 頻度の飽和パラメータ
  - b  (=0.75): 文書長の正規化パラメータ

  直感的理解:
  +-----------+----------------------------------------+
  | 因子       | 効果                                   |
  +-----------+----------------------------------------+
  | TF (高)   | その語が文書内に多く出現 → スコア上昇     |
  | IDF (高)  | 全体で珍しい語 → スコア上昇              |
  | 文書長(短) | 短い文書に出現 → スコア上昇(密度が高い)   |
  +-----------+----------------------------------------+
```

---

## 2. システムアーキテクチャ

### 2.1 分散検索の構成

```
                  分散検索アーキテクチャ

  Client
    |
  [API Gateway / Load Balancer]
    |
  [Coordinator Node]
    |
    +---- Scatter (クエリを全シャードに送信)
    |
    +---> [Shard 0] --> Index (docs 0-999K)
    +---> [Shard 1] --> Index (docs 1M-1.999M)
    +---> [Shard 2] --> Index (docs 2M-2.999M)
    |
    +---- Gather (結果をマージ、スコアでソート)
    |
  [Top-K Results]
    |
  Client
```

### 2.2 インデックス更新パイプライン

```
  データソース                   インデックス更新
  +--------+                  +-------------------+
  | DB     |--CDC (Debezium)-->| Kafka Topic      |
  | (CRUD) |                  | "product-updates" |
  +--------+                  +--------+----------+
                                       |
                              +--------v----------+
                              | Index Worker       |
                              | - Transform        |
                              | - Enrich (カテゴリ) |
                              | - Index to ES      |
                              +--------+----------+
                                       |
                              +--------v----------+
                              | Elasticsearch      |
                              | Cluster            |
                              +-------------------+

  ★ DB更新 → CDC → Kafka → Worker → ES の非同期パイプライン
  ★ インデックス遅延: 通常1-5秒
```

---

## 3. Elasticsearch 実装

### 3.1 インデックスマッピング

```python
# Elasticsearch インデックス設定 (Python - elasticsearch-py)
from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])

# 日本語検索用のインデックス設定
index_settings = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "ja_analyzer": {
                    "type": "custom",
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": [
                        "kuromoji_baseform",      # 活用形 → 基本形
                        "kuromoji_part_of_speech", # 助詞等の除去
                        "cjk_width",              # 全角半角統一
                        "ja_stop",                # ストップワード
                        "lowercase",
                    ]
                },
                "ja_search_analyzer": {
                    "type": "custom",
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": [
                        "kuromoji_baseform",
                        "kuromoji_part_of_speech",
                        "cjk_width",
                        "ja_stop",
                        "lowercase",
                        "synonym_filter",          # 検索時のみ同義語展開
                    ]
                }
            },
            "filter": {
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms": [
                        "PC, パソコン, コンピュータ",
                        "スマホ, スマートフォン, 携帯電話",
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "ja_analyzer",
                "search_analyzer": "ja_search_analyzer",
                "fields": {
                    "keyword": {"type": "keyword"},  # ソート・集計用
                }
            },
            "description": {
                "type": "text",
                "analyzer": "ja_analyzer",
            },
            "category": {"type": "keyword"},
            "price": {"type": "integer"},
            "rating": {"type": "float"},
            "created_at": {"type": "date"},
            "suggest": {
                "type": "completion",       # オートコンプリート用
                "analyzer": "ja_analyzer",
            }
        }
    }
}

es.indices.create(index='products', body=index_settings)
```

### 3.2 検索クエリ

```python
# 商品検索: 複合クエリ
def search_products(query: str, category: str = None,
                    min_price: int = None, max_price: int = None,
                    page: int = 1, size: int = 20):
    must_clauses = [
        {
            "multi_match": {
                "query": query,
                "fields": ["title^3", "description"],  # タイトル3倍ブースト
                "type": "best_fields",
                "fuzziness": "AUTO",                    # タイポ許容
            }
        }
    ]

    filter_clauses = []
    if category:
        filter_clauses.append({"term": {"category": category}})
    if min_price or max_price:
        price_range = {}
        if min_price: price_range["gte"] = min_price
        if max_price: price_range["lte"] = max_price
        filter_clauses.append({"range": {"price": price_range}})

    body = {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
            }
        },
        "highlight": {
            "fields": {
                "title": {},
                "description": {"fragment_size": 150},
            },
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
        "aggs": {
            "categories": {
                "terms": {"field": "category", "size": 20}
            },
            "price_ranges": {
                "range": {
                    "field": "price",
                    "ranges": [
                        {"to": 1000},
                        {"from": 1000, "to": 5000},
                        {"from": 5000, "to": 10000},
                        {"from": 10000},
                    ]
                }
            }
        },
        "from": (page - 1) * size,
        "size": size,
        "sort": [
            {"_score": "desc"},
            {"rating": "desc"},
        ],
    }

    return es.search(index='products', body=body)
```

### 3.3 オートコンプリート

```python
# サジェスト（オートコンプリート）
def autocomplete(prefix: str, size: int = 5):
    body = {
        "suggest": {
            "product_suggest": {
                "prefix": prefix,
                "completion": {
                    "field": "suggest",
                    "size": size,
                    "fuzzy": {
                        "fuzziness": 1,    # 1文字のタイポを許容
                    }
                }
            }
        }
    }
    result = es.search(index='products', body=body)
    suggestions = result['suggest']['product_suggest'][0]['options']
    return [s['text'] for s in suggestions]
```

### 3.4 インデックス更新

```python
# バルクインデキシング
from elasticsearch.helpers import bulk

def index_products(products: list):
    """商品リストを一括インデックス"""
    actions = []
    for product in products:
        actions.append({
            "_index": "products",
            "_id": product['id'],
            "_source": {
                "title": product['name'],
                "description": product['description'],
                "category": product['category'],
                "price": product['price'],
                "rating": product['rating'],
                "created_at": product['created_at'],
                "suggest": {
                    "input": [product['name']] + product.get('keywords', []),
                    "weight": int(product['rating'] * 10),
                },
            }
        })

    success, errors = bulk(es, actions, chunk_size=500, request_timeout=60)
    print(f"インデックス完了: {success} 件, エラー: {len(errors)} 件")
```

---

## 4. 比較表

| 特性 | Elasticsearch | Apache Solr | Meilisearch | Typesense |
|------|:------------:|:-----------:|:-----------:|:---------:|
| ベースエンジン | Lucene | Lucene | 独自 (Rust) | 独自 (C++) |
| 日本語対応 | kuromoji プラグイン | kuromoji | Lindera | 基本的 |
| リアルタイム検索 | 1秒以内 | 1秒以内 | 即時 | 即時 |
| 分散スケール | ネイティブ対応 | SolrCloud | 限定的 | 限定的 |
| 運用の複雑さ | 高 | 高 | 低 | 低 |
| 最適用途 | 大規模全文検索・ログ分析 | エンタープライズ検索 | 小中規模・高速サジェスト | 小中規模・タイポ耐性 |

| 検索機能 | 実装方法 | 効果 |
|---------|---------|------|
| ファジー検索 | fuzziness: "AUTO" | タイポ許容 |
| 同義語展開 | synonym filter | 表記揺れ対応 |
| ブースト | fields: ["title^3"] | フィールド重み付け |
| ハイライト | highlight | 該当箇所の強調 |
| ファセット | aggregations | カテゴリ別件数 |
| オートコンプリート | completion suggester | 入力補完 |

---

## 5. アンチパターン

### アンチパターン 1: DB に全文検索を任せる

```sql
-- BAD: LIKE 検索はインデックスが効かない
SELECT * FROM products WHERE name LIKE '%東京%' OR description LIKE '%東京%';
-- → フルテーブルスキャン、100万行で数秒かかる

-- BAD: MySQL FULLTEXT もCJK(日本語)に弱い
SELECT * FROM products WHERE MATCH(name) AGAINST('東京の観光' IN BOOLEAN MODE);
-- → 形態素解析なし、精度が低い
```

```
GOOD: 専用の検索エンジンを使う
  DB (PostgreSQL) --- CDC ---> Elasticsearch
  → 形態素解析、BM25 ランキング、ファセット、サジェスト全対応
```

### アンチパターン 2: インデックス設計を後回しにする

```
BAD: デフォルトのダイナミックマッピングで運用
  → 全フィールドが text + keyword 双方にインデックス
  → ストレージ2倍、インデキシング遅延

GOOD: 明示的なマッピングを事前設計
  - 検索対象: text 型 + 適切なアナライザー
  - フィルター: keyword 型
  - 数値: integer / float 型
  - 不要フィールド: enabled: false
```

---

## 6. FAQ

### Q1. Elasticsearch のシャード数はどう決める？

**A.** 1シャードあたり10-50GB、ヒープ使用量が全体の50%以下が目安。例えば100GBのデータなら3-10シャード。シャードが多すぎるとオーバーヘッドが増加し、少なすぎるとスケールできない。初期は少なめに設定し、必要に応じて Reindex でシャード数を変更する。ロールオーバーポリシーで時系列インデックスを自動管理するのも有効。

### Q2. 検索の関連性（レリバンシー）を改善するには？

**A.** (1) アナライザーの最適化（同義語辞書、ユーザー辞書の追加）。(2) フィールドブースト（title に高いウェイト）。(3) Function Score Query で人気度・新着度をスコアに反映。(4) 検索ログを分析してクエリ書き換えルール（"iPhone" → "iPhone OR アイフォン"）を追加。(5) Learning to Rank でクリックスルーデータから機械学習モデルを構築。

### Q3. インデックスの再構築はどうやる？

**A.** ダウンタイムなしで再構築するには Alias を使う。(1) 新インデックス `products_v2` を作成。(2) 全データを新インデックスにバルクインデックス。(3) Alias `products` を `products_v1` → `products_v2` に切り替え。(4) 旧インデックスを削除。クライアントは常に Alias 名でアクセスするため、切り替えは透過的。

---

## まとめ

| 項目 | ポイント |
|------|---------|
| 転置インデックス | 検索エンジンの中核データ構造。語 → 文書リストのマッピング |
| アナライザー | トークナイズ → フィルタリング → インデックス。日本語には形態素解析が必須 |
| BM25 | TF-IDF の改良版。Elasticsearch のデフォルトランキング |
| 分散検索 | Scatter-Gather パターン。シャード数がスケールの鍵 |
| Elasticsearch | 大規模全文検索の標準。kuromoji で日本語対応 |
| インデックス更新 | CDC + Kafka パイプラインで非同期に同期 |
| 関連性改善 | 同義語辞書、ブースト、Function Score、Learning to Rank |

---

## 次に読むべきガイド

- [レートリミッター設計](./03-rate-limiter.md) — 検索 API のレート制限
- [CDN](../01-components/03-cdn.md) — 検索結果ページのキャッシュ戦略
- [DBスケーリング](../01-components/04-database-scaling.md) — データソースのスケーリング

---

## 参考文献

1. **Elasticsearch: The Definitive Guide** — Clinton Gormley & Zachary Tong (O'Reilly, 2015) — Elasticsearch の包括的リファレンス
2. **Information Retrieval** — Christopher Manning et al. (Cambridge University Press, 2008) — 情報検索の理論的基盤
3. **Relevant Search** — Doug Turnbull & John Berryman (Manning, 2016) — 検索の関連性改善の実践ガイド
4. **Elasticsearch 公式ドキュメント** — https://www.elastic.co/guide/
