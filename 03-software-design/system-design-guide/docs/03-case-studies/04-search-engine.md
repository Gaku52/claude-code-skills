# 検索エンジン設計

> 大規模データセットに対する全文検索システムの設計原則を、転置インデックス・ランキングアルゴリズム・分散アーキテクチャ・Elasticsearch の実装を通じて解説し、検索品質とパフォーマンスを両立させるアーキテクチャを構築する

---

## この章で学ぶこと

1. **検索エンジンの基本原理** -- 転置インデックスの内部構造、トークナイズ処理、TF-IDF/BM25 ランキングの数学的背景と直感的理解
2. **システムアーキテクチャ** -- インデックス構築パイプライン、クエリ処理フロー、分散検索の Scatter-Gather パターン、レプリケーション戦略
3. **Elasticsearch を用いた実装** -- マッピング設計、日本語形態素解析、クエリ最適化、オートコンプリート、集計
4. **検索品質の改善手法** -- 同義語辞書、Function Score Query、Learning to Rank、A/Bテスト
5. **運用と監視** -- クラスタ管理、インデックスライフサイクル、パフォーマンスチューニング、障害対応

---

## 前提知識

この章を理解するために、以下の知識を事前に身につけておくことを推奨する。

| 前提知識 | 参照先 |
|---------|--------|
| システム設計の基本概念(スケーラビリティ、可用性) | [スケーラビリティ](../00-fundamentals/01-scalability.md) |
| CAP 定理と分散システムのトレードオフ | [CAP 定理](../00-fundamentals/03-cap-theorem.md) |
| メッセージキューの基礎(Kafka など) | [メッセージキュー](../01-components/02-message-queue.md) |
| データベーススケーリングの基本 | [DB スケーリング](../01-components/04-database-scaling.md) |
| キャッシュ戦略 | [キャッシング](../01-components/01-caching.md) |
| Observer パターン(イベント駆動の理解) | [Observer パターン](../../../design-patterns-guide/docs/02-behavioral/00-observer.md) |

---

## 1. 検索エンジンの基本原理

### 1.1 なぜ検索エンジンが必要なのか (WHY)

リレーショナルデータベースの `LIKE '%keyword%'` クエリは、テーブルのフルスキャンを引き起こす。100万行のテーブルで `LIKE` 検索を実行すると数秒から数十秒かかり、1億行になるとタイムアウトする。B-Tree インデックスは前方一致(`LIKE 'keyword%'`)にしか効かず、中間一致・後方一致には無力である。

さらに、自然言語の検索には以下の課題がある:

- **形態素解析**: 「東京タワーの観光ガイド」を「東京」「タワー」「観光」「ガイド」に分割する必要がある
- **表記揺れ**: 「パソコン」「PC」「コンピュータ」を同一視する必要がある
- **ランキング**: 検索結果を関連性順に並べる必要がある
- **スケール**: 数十億ドキュメントに対してミリ秒単位で応答する必要がある

これらの課題を解決するのが、専用の検索エンジンである。検索エンジンは転置インデックスという特殊なデータ構造を使い、テキストの全文検索を O(1) に近い計算量で実現する。

```
なぜ専用検索エンジンが必要か:

  問題: RDB の LIKE 検索
  +--------------------------------------------------+
  | SELECT * FROM products                            |
  | WHERE name LIKE '%東京%'                           |
  |                                                    |
  | 実行計画: Full Table Scan                           |
  | 100万行 → 2-5秒                                    |
  | 1億行  → タイムアウト                               |
  | 形態素解析: なし                                     |
  | ランキング: なし                                     |
  | 同義語: 非対応                                      |
  +--------------------------------------------------+
         ↓ 解決策
  +--------------------------------------------------+
  | Elasticsearch (転置インデックス)                     |
  |                                                    |
  | 100万ドキュメント → 10-50ms                         |
  | 1億ドキュメント   → 50-200ms                        |
  | 形態素解析: kuromoji (日本語対応)                     |
  | ランキング: BM25 + カスタムスコアリング               |
  | 同義語: synonym filter 対応                         |
  +--------------------------------------------------+
```

### 1.2 転置インデックスの内部構造

転置インデックス (Inverted Index) は、検索エンジンの核となるデータ構造である。「語 (Term) → その語を含む文書リスト (Posting List)」のマッピングを保持する。これは書籍の「索引 (Index)」と同じ発想だ。書籍の索引では「キーワード → ページ番号」のマッピングがあり、目的のキーワードが書かれたページをすぐに見つけられる。

```
文書 (Document):
  Doc1: "東京の天気は晴れです"
  Doc2: "大阪の天気は雨です"
  Doc3: "東京タワーの観光ガイド"

Step 1: トークナイズ (形態素解析)
  Doc1 → ["東京", "の", "天気", "は", "晴れ", "です"]
  Doc2 → ["大阪", "の", "天気", "は", "雨", "です"]
  Doc3 → ["東京", "タワー", "の", "観光", "ガイド"]

Step 2: フィルタリング (ストップワード除去)
  Doc1 → ["東京", "天気", "晴れ"]
  Doc2 → ["大阪", "天気", "雨"]
  Doc3 → ["東京", "タワー", "観光", "ガイド"]

Step 3: 転置インデックス構築
  +----------+--------------------------------------------+
  | Term     | Posting List                               |
  |          | (DocID, TermFreq, [Positions])             |
  +----------+--------------------------------------------+
  | "東京"   | [(Doc1, 1, [0]), (Doc3, 1, [0])]           |
  | "天気"   | [(Doc1, 1, [1]), (Doc2, 1, [1])]           |
  | "晴れ"   | [(Doc1, 1, [2])]                           |
  | "大阪"   | [(Doc2, 1, [0])]                           |
  | "雨"     | [(Doc2, 1, [2])]                           |
  | "タワー" | [(Doc3, 1, [1])]                           |
  | "観光"   | [(Doc3, 1, [2])]                           |
  | "ガイド" | [(Doc3, 1, [3])]                           |
  +----------+--------------------------------------------+

Step 4: 検索実行 "東京 天気"
  "東京" → {Doc1, Doc3}
  "天気" → {Doc1, Doc2}
  AND 演算: {Doc1, Doc3} ∩ {Doc1, Doc2} = {Doc1}
  OR  演算: {Doc1, Doc3} ∪ {Doc1, Doc2} = {Doc1, Doc2, Doc3}
```

#### なぜ転置インデックスが高速なのか

転置インデックスの検索は、ハッシュマップのルックアップと同様に O(1) に近い計算量で実行される。具体的には以下のステップで動作する:

1. **Term Dictionary**: 全ての語をソート済みで保持し、二分探索で O(log N) で検索
2. **Posting List**: 見つかった語に対応する文書 ID リストを取得 (O(1))
3. **Boolean 演算**: 複数の Posting List をマージ (ソート済みリストのマージで O(M+N))

Lucene (Elasticsearch の内部エンジン) では、Term Dictionary に FST (Finite State Transducer) というデータ構造を使い、メモリ効率を高めながら高速な前方一致検索を実現している。

```
転置インデックスの内部データ構造 (Lucene):

  +-- Term Dictionary (FST: Finite State Transducer) --+
  | メモリ上に配置、前方一致で高速検索                      |
  |                                                      |
  | "大阪" → Block 0x3A (ディスク上の位置)                 |
  | "東京" → Block 0x1F                                   |
  | "天気" → Block 0x2B                                   |
  | ...                                                   |
  +------------------------------------------------------+
            |
            v
  +-- Term Index (Skip List) ----+
  | Term → Posting List の位置    |
  +------------------------------+
            |
            v
  +-- Posting List (.doc file) --+
  | DocID のソート済みリスト       |
  | 差分符号化で圧縮               |
  | [1, 3, 7, 15, 20, ...]       |
  | → 差分: [1, 2, 4, 8, 5, ...] |
  | → VByte 符号化でさらに圧縮    |
  +------------------------------+
            |
            v
  +-- Position (.pos file) ------+
  | 各 Doc 内での出現位置          |
  | フレーズ検索に使用             |
  +------------------------------+
            |
            v
  +-- Stored Fields (.fdt) ------+
  | 元の文書の内容(圧縮保存)       |
  | ハイライト表示に使用            |
  +------------------------------+
```

### 1.3 検索パイプライン

検索エンジンには「書き込みパス (Write Path)」と「読み取りパス (Read Path)」の2つの処理フローがある。

```
===================== Write Path (インデックス構築) =====================

  Raw Data ──→ [Crawler / Ingest API]
                     │
                     v
              [Text Extraction]       PDF/HTML/JSON からテキスト抽出
                     │
                     v
              [Character Filter]      HTML タグ除去、全角半角変換
                     │
                     v
              [Tokenizer]             形態素解析 (kuromoji)
                     │                "東京の天気" → ["東京","の","天気"]
                     v
              [Token Filter]          ストップワード除去、基本形変換
                     │                同義語展開、小文字化
                     v
              [Inverted Index]        転置インデックスに書き込み
                     │
                     v
              [Segment]               Lucene セグメントとしてディスク保存
                                      ※ immutable (一度書いたら変更不可)

===================== Read Path (クエリ処理) ============================

  Query "東京 天気" ──→ [Query Parser]     クエリ構文解析
                              │
                              v
                        [Analyzer]          Write Path と同じトークナイズ
                              │             + 検索時フィルタ (同義語展開)
                              v
                        [Index Lookup]      転置インデックスを参照
                              │
                              v
                        [Scoring (BM25)]    各文書のスコア計算
                              │
                              v
                        [Collector]         Top-K 文書を収集 (優先度キュー)
                              │
                              v
                        [Post Filter]       フィルタ条件適用 (price, category)
                              │
                              v
                        [Highlight]         マッチ箇所のハイライト生成
                              │
                              v
                        [Response]          JSON レスポンス構築
```

#### Write Path と Read Path の一貫性が重要な理由

Write Path と Read Path で**同じ Analyzer** を使わないと、検索が正しく動作しない。例えば、Write Path で「東京タワー」を「東京」「タワー」に分割しているのに、Read Path で「東京タワー」を1つのトークンとして検索すると、転置インデックスにマッチしない。

ただし、**同義語展開は検索時のみ**に行うのがベストプラクティスである。インデックス時に同義語を展開すると、同義語辞書を更新するたびに全文書の再インデックスが必要になるためだ。

### 1.4 ランキング: TF-IDF と BM25

#### TF-IDF (Term Frequency - Inverse Document Frequency)

TF-IDF は情報検索の古典的なランキング手法で、直感的には「その文書の中でよく出てくるが、他の文書ではあまり出てこない語ほど重要」という考え方に基づく。

```
TF-IDF の計算:

  TF(t, d) = 文書 d 内での語 t の出現回数 / 文書 d の総語数

  IDF(t) = log(全文書数 / 語 t を含む文書数)

  TF-IDF(t, d) = TF(t, d) × IDF(t)

  例: 全100文書、Doc1 (100語) に "東京" が5回出現、
      "東京" は10文書に出現

  TF("東京", Doc1) = 5 / 100 = 0.05
  IDF("東京")      = log(100 / 10) = log(10) ≈ 2.30
  TF-IDF           = 0.05 × 2.30 = 0.115

  "の" (全文書に出現する語) の場合:
  IDF("の") = log(100 / 100) = log(1) = 0
  TF-IDF    = 0.05 × 0 = 0  (助詞は重要度ゼロ)
```

#### TF-IDF の問題点

TF-IDF には2つの大きな問題がある:

1. **TF の飽和がない**: 語が10回出現する文書と100回出現する文書で、スコアが10倍になる。しかし実際には、ある語が10回出現すればその文書がその語に関連していることは十分に示されており、100回出現しても関連度が10倍になるわけではない
2. **文書長の正規化が不十分**: 長い文書は自然と語の出現回数が多くなるため、短い文書より不当に高いスコアを得てしまう

#### BM25 (Best Matching 25)

BM25 は TF-IDF の問題を解決した改良版で、Elasticsearch のデフォルトランキングアルゴリズムである。

```
BM25 スコア計算:

  score(D, Q) = Σ [ IDF(qi) × f(qi,D) × (k1 + 1)                    ]
                    [         ─────────────────────────────────────────]
                    [         f(qi,D) + k1 × (1 - b + b × |D| / avgdl)]

  各要素の意味:
  ┌──────────────┬────────────────────────────────────────────────┐
  │ 変数          │ 意味                                           │
  ├──────────────┼────────────────────────────────────────────────┤
  │ Q            │ 検索クエリ (複数の語 q1, q2, ... の集合)         │
  │ D            │ スコアを計算する対象の文書                        │
  │ qi           │ クエリ内の i 番目の語                            │
  │ f(qi, D)     │ 文書 D 内での語 qi の出現頻度 (Term Frequency)   │
  │ |D|          │ 文書 D の長さ (語数)                             │
  │ avgdl        │ 全文書の平均長 (Average Document Length)         │
  │ k1 (= 1.2)  │ TF の飽和パラメータ (大きいほど飽和が遅い)        │
  │ b (= 0.75)  │ 文書長の正規化パラメータ (0で無効、1で完全正規化)  │
  │ IDF(qi)      │ 逆文書頻度 (珍しい語ほど大きい)                  │
  └──────────────┴────────────────────────────────────────────────┘

  BM25 の IDF 計算 (Lucene 実装):
  IDF(qi) = log(1 + (N - n(qi) + 0.5) / (n(qi) + 0.5))
    N:      全文書数
    n(qi):  語 qi を含む文書数

  BM25 が TF-IDF より優れている点:
  ┌─────────────────────┬──────────────┬──────────────────────┐
  │ 特性                 │ TF-IDF       │ BM25                 │
  ├─────────────────────┼──────────────┼──────────────────────┤
  │ TF の飽和            │ なし (線形)   │ あり (k1 で制御)      │
  │ 文書長の正規化        │ 不十分       │ b パラメータで制御     │
  │ 高頻度語の抑制        │ 弱い         │ 飽和により自然に抑制   │
  │ パラメータ調整        │ なし         │ k1, b で調整可能      │
  └─────────────────────┴──────────────┴──────────────────────┘
```

#### BM25 の TF 飽和の直感的理解

```
TF 飽和のグラフ (k1=1.2):

  スコア寄与
  ^
  |                                    -------- TF-IDF (線形)
  |                               ----/
  |                          ----/
  |                     ----/
  |     -------========---------- BM25 (飽和あり)
  |   -/  ----/
  | -/ --/
  |/ /
  +----------------------------------------→ 出現回数
  0   1   2   3   5   10  20  50

  TF-IDF: 出現回数に比例してスコアが増加し続ける
  BM25:   数回の出現でスコアはほぼ飽和する
         → "東京" が10回出ても100回出てもスコアはほぼ同じ
         → より自然なランキングを実現
```

### 1.5 形態素解析と日本語処理

日本語の検索エンジンにおいて、最も重要な要素の一つが形態素解析 (Morphological Analysis) である。英語は空白で単語が区切られているが、日本語には空白区切りがないため、文を単語に分割する処理が必要になる。

```
日本語テキストのトークナイズ比較:

  入力: "東京スカイツリーの展望台"

  ┌─────────────────────┬─────────────────────────────────────┐
  │ 手法                 │ 結果                                │
  ├─────────────────────┼─────────────────────────────────────┤
  │ N-gram (bigram)      │ ["東京", "京ス", "スカ", "カイ",     │
  │                      │  "イツ", "ツリ", "リー", "ーの",     │
  │                      │  "の展", "展望", "望台"]             │
  │                      │ → 再現率は高いが適合率が低い          │
  │                      │ → "京ス" でも検索にヒットしてしまう   │
  ├─────────────────────┼─────────────────────────────────────┤
  │ kuromoji (形態素解析) │ ["東京", "スカイツリー", "展望台"]    │
  │                      │ → 意味のある単位で分割               │
  │                      │ → 適合率が高い                       │
  ├─────────────────────┼─────────────────────────────────────┤
  │ kuromoji + ユーザー辞書│ ["東京スカイツリー", "展望台"]       │
  │                      │ → 固有名詞を1トークンとして認識      │
  │                      │ → 検索精度がさらに向上               │
  └─────────────────────┴─────────────────────────────────────┘

  N-gram vs 形態素解析のトレードオフ:
  ┌──────────────┬──────────────────┬──────────────────────┐
  │ 指標          │ N-gram           │ 形態素解析            │
  ├──────────────┼──────────────────┼──────────────────────┤
  │ 再現率        │ 高 (漏れが少ない) │ 中 (辞書にない語は漏れ)│
  │ 適合率        │ 低 (ノイズが多い) │ 高 (意味単位で一致)    │
  │ インデックスサイズ│ 大             │ 小                    │
  │ 辞書依存      │ なし             │ あり                   │
  │ 新語対応      │ 自動             │ 辞書更新が必要         │
  │ 検索速度      │ 遅い (候補が多い) │ 速い                   │
  └──────────────┴──────────────────┴──────────────────────┘
```

---

## 2. システムアーキテクチャ

### 2.1 全体アーキテクチャ

大規模検索システムの全体像を示す。データの取り込みからユーザーへの結果返却まで、複数のコンポーネントが協調して動作する。

```
             大規模検索エンジン 全体アーキテクチャ

  ┌──────────────────────────────────────────────────────────────┐
  │                        Client Layer                          │
  │  [Web App] [Mobile App] [API Client]                         │
  └──────────┬───────────────────────────────────────────────────┘
             │
  ┌──────────v───────────────────────────────────────────────────┐
  │                        Gateway Layer                         │
  │  [CDN (静的コンテンツ)]  [API Gateway]  [Rate Limiter]        │
  │                              │                               │
  │                    [Authentication]                           │
  │                    [Query Rewriting]                          │
  │                    [Request Routing]                          │
  └──────────────────────┬──────────────────────────────────────-┘
                         │
  ┌──────────────────────v──────────────────────────────────────-┐
  │                     Search Service                           │
  │                                                              │
  │  [Query Parser] → [Spell Check] → [Synonym Expansion]       │
  │       → [Query Planner] → [Elasticsearch Client]            │
  │       → [Result Formatter] → [Personalization]              │
  │       → [Cache (Redis)]                                     │
  └──────────────────────┬──────────────────────────────────────-┘
                         │
  ┌──────────────────────v──────────────────────────────────────-┐
  │              Elasticsearch Cluster                           │
  │                                                              │
  │  [Master Node x3]   選出・クラスタ管理                        │
  │  [Data Node x6]     インデックス保持・検索実行                 │
  │  [Coordinator x2]   クエリルーティング・結果マージ              │
  │  [Ingest Node x2]   ドキュメント前処理パイプライン             │
  └──────────────────────────────────────────────────────────────┘
                         │
  ┌──────────────────────v──────────────────────────────────────-┐
  │              Data Ingestion Pipeline                         │
  │                                                              │
  │  [Source DB] → [CDC (Debezium)] → [Kafka] → [Index Worker]  │
  │  [Crawler]  → [Content Parser] → [Enrichment] → [ES Bulk]  │
  │  [File Store] → [Tika (抽出)] → [Kafka] → [Index Worker]   │
  └──────────────────────────────────────────────────────────────┘
                         │
  ┌──────────────────────v──────────────────────────────────────-┐
  │              Monitoring & Analytics                          │
  │                                                              │
  │  [Prometheus/Grafana]  クラスタメトリクス                      │
  │  [Search Analytics]    クエリログ・CTR 分析                    │
  │  [Alerting]            異常検知・アラート                      │
  └──────────────────────────────────────────────────────────────┘
```

### 2.2 分散検索の Scatter-Gather パターン

Elasticsearch はデータをシャード (Shard) に分割し、複数ノードに分散配置する。検索時はすべてのシャードに並列でクエリを投げ (Scatter)、結果をマージする (Gather)。

```
                  Scatter-Gather パターンの詳細

  Client
    │
    v
  [Coordinator Node]
    │
    ├── Phase 1: Query Phase (Scatter)
    │   │
    │   ├──→ [Shard 0, Node A] ── BM25計算 ──→ (DocID, Score) Top-K
    │   ├──→ [Shard 1, Node B] ── BM25計算 ──→ (DocID, Score) Top-K
    │   ├──→ [Shard 2, Node C] ── BM25計算 ──→ (DocID, Score) Top-K
    │   ├──→ [Shard 3, Node A] ── BM25計算 ──→ (DocID, Score) Top-K
    │   └──→ [Shard 4, Node B] ── BM25計算 ──→ (DocID, Score) Top-K
    │
    ├── Phase 2: Merge (Gather)
    │   │
    │   └── 全シャードの Top-K を Score でソート
    │       → Global Top-K を決定
    │       → 例: Top-10 を取得するなら、各シャードが Top-10 を返し、
    │         Coordinator が 50件中 Top-10 を選ぶ
    │
    └── Phase 3: Fetch Phase
        │
        ├──→ [Shard 0] ── DocID 3,7 の _source を取得
        ├──→ [Shard 2] ── DocID 15 の _source を取得
        └──→ [Shard 4] ── DocID 42,88 の _source を取得

        → 文書本文、ハイライト、集計結果を返却

  ※ Query Phase は DocID + Score のみを返すため軽量
  ※ Fetch Phase で実際の文書内容を取得する (2段階方式)
```

#### シャーディング戦略

```
シャーディングの設計指針:

  ┌──────────────────────┬────────────────────────────────────┐
  │ 考慮事項              │ ガイドライン                        │
  ├──────────────────────┼────────────────────────────────────┤
  │ シャードサイズ         │ 10-50 GB / シャード (推奨)          │
  │ シャード数の上限       │ ヒープ 1GB あたり 20 シャード以下    │
  │ シャード数の決定       │ データ総量 ÷ 目標シャードサイズ      │
  │ 例: 500GB データ      │ 500 ÷ 30 ≈ 17 シャード             │
  ├──────────────────────┼────────────────────────────────────┤
  │ レプリカ数            │ 最低 1 (可用性確保)                  │
  │                      │ 読み取り負荷が高い場合は 2-3          │
  ├──────────────────────┼────────────────────────────────────┤
  │ ルーティング          │ デフォルト: hash(_id) % num_shards  │
  │                      │ カスタム: user_id ベースルーティング  │
  └──────────────────────┴────────────────────────────────────┘

  時系列インデックスのパターン (ログ、イベント):

  logs-2024.01.01  (hot:  SSD, 1 primary + 1 replica)
  logs-2024.01.02  (hot:  SSD, 1 primary + 1 replica)
  ...
  logs-2023.12.01  (warm: HDD, 1 primary + 1 replica, force-merge済)
  logs-2023.11.01  (cold: S3, searchable snapshot)

  → ILM (Index Lifecycle Management) で自動管理
  → hot → warm → cold → delete のライフサイクル
```

### 2.3 インデックス更新パイプライン

プライマリデータストア (RDB) と検索インデックス (Elasticsearch) の同期は、CDC (Change Data Capture) パターンで実現する。

```
  CDC + Kafka パイプラインの詳細

  ┌──────────┐    CDC (WAL)    ┌──────────┐    Consumer    ┌────────────┐
  │ PostgreSQL│───────────────→│  Kafka    │──────────────→│Index Worker│
  │           │  Debezium      │  Topic    │               │            │
  │ products  │  Connector     │"product-  │  Consumer     │ Transform  │
  │ テーブル   │               │ updates"  │  Group        │ Enrich     │
  └──────────┘                └──────────┘               │ Validate   │
                                                          │ Bulk Index │
                                                          └─────┬──────┘
                                                                │
                                                          ┌─────v──────┐
                                                          │Elasticsearch│
                                                          │  Cluster   │
                                                          └────────────┘

  CDC イベントの構造:
  {
    "op": "u",              // c=create, u=update, d=delete
    "before": {...},        // 変更前の行データ
    "after": {              // 変更後の行データ
      "id": 12345,
      "name": "商品A 改訂版",
      "price": 2980,
      "updated_at": "2024-01-15T10:30:00Z"
    },
    "source": {
      "table": "products",
      "lsn": 123456789      // WAL のログシーケンス番号
    }
  }

  Index Worker の処理フロー:
  1. Kafka からイベントを消費 (バッチ: 100-500件ずつ)
  2. op に応じて処理を分岐:
     - "c" / "u": ドキュメントを変換 → ES に upsert
     - "d": ES からドキュメントを削除
  3. エンリッチメント: カテゴリ名の解決、画像URL の変換等
  4. バリデーション: 必須フィールドのチェック
  5. Bulk API で ES に一括書き込み
  6. offset をコミット

  遅延: DB更新 → ES反映まで 通常 1-5秒
  スループット: 1ワーカーで 5,000-10,000 docs/sec
```

### 2.4 キャッシュ戦略

検索システムでは、頻出クエリのキャッシュが性能に大きく影響する。

```
検索キャッシュの多層構造:

  Layer 1: CDN / Edge Cache
  ┌─────────────────────────────────────────┐
  │ 静的な検索結果ページ (SEO 用)             │
  │ TTL: 5-15分                              │
  │ Hit率: 10-20% (検索クエリの多様性が高い)   │
  └─────────────────────────────────────────┘

  Layer 2: Application Cache (Redis)
  ┌─────────────────────────────────────────┐
  │ クエリ結果のキャッシュ                     │
  │ Key: hash(query + filters + page)        │
  │ TTL: 1-5分                               │
  │ Hit率: 30-50% (人気クエリは繰り返される)   │
  │ 無効化: インデックス更新時に関連キャッシュを│
  │        パージ                             │
  └─────────────────────────────────────────┘

  Layer 3: Elasticsearch Request Cache
  ┌─────────────────────────────────────────┐
  │ シャードレベルの結果キャッシュ              │
  │ filter 句の結果をキャッシュ                │
  │ シャードの refresh で自動無効化            │
  │ Hit率: 高 (同一フィルタ条件が多い場合)     │
  └─────────────────────────────────────────┘

  Layer 4: Elasticsearch Field Data Cache
  ┌─────────────────────────────────────────┐
  │ ソート・集計に使うフィールドデータ          │
  │ doc_values で事前構築 (推奨)              │
  │ ヒープメモリ上に展開                      │
  └─────────────────────────────────────────┘
```

---

## 3. Elasticsearch 実装

### 3.1 インデックスマッピング設計

```python
# コード例 1: Elasticsearch インデックス設定 (Python - elasticsearch-py)
from elasticsearch import Elasticsearch

es = Elasticsearch(['http://localhost:9200'])

# 日本語検索用のインデックス設定
index_settings = {
    "settings": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "refresh_interval": "1s",    # インデックス更新間隔
        "analysis": {
            "char_filter": {
                "normalize_filter": {
                    "type": "icu_normalizer",      # Unicode 正規化
                    "name": "nfkc_cf",
                }
            },
            "analyzer": {
                "ja_analyzer": {
                    "type": "custom",
                    "char_filter": ["normalize_filter"],
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": [
                        "kuromoji_baseform",      # 活用形 → 基本形 (走った→走る)
                        "kuromoji_part_of_speech", # 助詞・助動詞の除去
                        "cjk_width",              # 全角半角統一
                        "ja_stop",                # 日本語ストップワード
                        "lowercase",              # 英字小文字化
                    ]
                },
                "ja_search_analyzer": {
                    "type": "custom",
                    "char_filter": ["normalize_filter"],
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": [
                        "kuromoji_baseform",
                        "kuromoji_part_of_speech",
                        "cjk_width",
                        "ja_stop",
                        "lowercase",
                        "synonym_filter",          # 検索時のみ同義語展開
                    ]
                },
                "ja_ngram_analyzer": {
                    "type": "custom",
                    "char_filter": ["normalize_filter"],
                    "tokenizer": "ja_ngram_tokenizer",
                    "filter": ["lowercase"]
                }
            },
            "tokenizer": {
                "ja_ngram_tokenizer": {
                    "type": "ngram",
                    "min_gram": 2,
                    "max_gram": 3,
                    "token_chars": ["letter", "digit"]
                }
            },
            "filter": {
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms": [
                        "PC, パソコン, コンピュータ",
                        "スマホ, スマートフォン, 携帯電話",
                        "テレビ, TV, ティーヴィー",
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
                    "keyword": {"type": "keyword"},  # 完全一致・ソート・集計用
                    "ngram": {                        # 部分一致検索用
                        "type": "text",
                        "analyzer": "ja_ngram_analyzer"
                    }
                }
            },
            "description": {
                "type": "text",
                "analyzer": "ja_analyzer",
                "search_analyzer": "ja_search_analyzer",
            },
            "category": {
                "type": "keyword",               # フィルタ・集計用
            },
            "tags": {
                "type": "keyword",               # 複数タグ
            },
            "price": {
                "type": "integer",               # 範囲フィルタ用
            },
            "rating": {
                "type": "float",                 # ソート・ブースト用
            },
            "review_count": {
                "type": "integer",               # 人気度スコアリング用
            },
            "created_at": {
                "type": "date",                  # 時系列フィルタ用
            },
            "updated_at": {
                "type": "date",
            },
            "location": {
                "type": "geo_point",             # 地理検索用
            },
            "suggest": {
                "type": "completion",            # オートコンプリート用
                "analyzer": "ja_analyzer",
            },
            "metadata": {
                "type": "object",
                "enabled": False,                # インデックスしない (保存のみ)
            }
        }
    }
}

# インデックス作成 (エイリアス経由)
index_name = "products_v1"
alias_name = "products"

es.indices.create(index=index_name, body=index_settings)
es.indices.put_alias(index=index_name, name=alias_name)
print(f"インデックス '{index_name}' を作成し、エイリアス '{alias_name}' を設定しました")
```

### 3.2 検索クエリの実装

```python
# コード例 2: 複合検索クエリ (商品検索API)
from typing import Optional, List, Dict, Any
from datetime import datetime


def search_products(
    query: str,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    min_price: Optional[int] = None,
    max_price: Optional[int] = None,
    min_rating: Optional[float] = None,
    sort_by: str = "relevance",
    page: int = 1,
    size: int = 20,
) -> Dict[str, Any]:
    """
    商品の全文検索を実行する。

    Args:
        query: 検索キーワード
        category: カテゴリフィルタ
        tags: タグフィルタ (AND 条件)
        min_price: 最低価格
        max_price: 最高価格
        min_rating: 最低レーティング
        sort_by: ソート順 ("relevance", "price_asc", "price_desc",
                 "rating", "newest")
        page: ページ番号 (1-indexed)
        size: 1ページあたりの件数

    Returns:
        検索結果 (ヒット件数、商品リスト、ファセット情報)
    """
    # --- must 句: 全文検索 ---
    must_clauses = []
    if query:
        must_clauses.append({
            "multi_match": {
                "query": query,
                "fields": [
                    "title^3",          # タイトルは3倍ブースト
                    "title.ngram^0.5",  # N-gram は低ブースト
                    "description",
                    "tags^2",
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",    # タイポ許容 (3-5文字→1編集距離)
                "minimum_should_match": "75%",
            }
        })

    # --- filter 句: 絞り込み (スコアに影響しない) ---
    filter_clauses = []
    if category:
        filter_clauses.append({"term": {"category": category}})
    if tags:
        for tag in tags:
            filter_clauses.append({"term": {"tags": tag}})
    if min_price is not None or max_price is not None:
        price_range = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        filter_clauses.append({"range": {"price": price_range}})
    if min_rating is not None:
        filter_clauses.append({"range": {"rating": {"gte": min_rating}}})

    # --- ソート順の決定 ---
    sort_options = {
        "relevance": [{"_score": "desc"}, {"rating": "desc"}],
        "price_asc": [{"price": "asc"}, {"_score": "desc"}],
        "price_desc": [{"price": "desc"}, {"_score": "desc"}],
        "rating": [{"rating": "desc"}, {"review_count": "desc"}],
        "newest": [{"created_at": "desc"}],
    }
    sort = sort_options.get(sort_by, sort_options["relevance"])

    # --- クエリ本体の構築 ---
    body = {
        "query": {
            "bool": {
                "must": must_clauses,
                "filter": filter_clauses,
            }
        },
        "highlight": {
            "fields": {
                "title": {"number_of_fragments": 0},  # 全文返却
                "description": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                },
            },
            "pre_tags": ["<mark>"],
            "post_tags": ["</mark>"],
        },
        "aggs": {
            "categories": {
                "terms": {"field": "category", "size": 20}
            },
            "price_stats": {
                "stats": {"field": "price"}
            },
            "price_ranges": {
                "range": {
                    "field": "price",
                    "ranges": [
                        {"key": "~1000円", "to": 1000},
                        {"key": "1000~5000円", "from": 1000, "to": 5000},
                        {"key": "5000~10000円", "from": 5000, "to": 10000},
                        {"key": "10000円~", "from": 10000},
                    ]
                }
            },
            "avg_rating": {
                "avg": {"field": "rating"}
            },
            "tags": {
                "terms": {"field": "tags", "size": 30}
            },
        },
        "from": (page - 1) * size,
        "size": size,
        "sort": sort,
        "_source": {
            "excludes": ["suggest", "metadata"]  # 不要フィールド除外
        },
    }

    result = es.search(index="products", body=body)

    # --- レスポンスの整形 ---
    return {
        "total": result["hits"]["total"]["value"],
        "page": page,
        "size": size,
        "items": [
            {
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"],
                "highlight": hit.get("highlight", {}),
            }
            for hit in result["hits"]["hits"]
        ],
        "facets": {
            "categories": [
                {"key": b["key"], "count": b["doc_count"]}
                for b in result["aggregations"]["categories"]["buckets"]
            ],
            "price_ranges": [
                {"key": b["key"], "count": b["doc_count"]}
                for b in result["aggregations"]["price_ranges"]["buckets"]
            ],
            "price_stats": result["aggregations"]["price_stats"],
            "avg_rating": result["aggregations"]["avg_rating"]["value"],
            "tags": [
                {"key": b["key"], "count": b["doc_count"]}
                for b in result["aggregations"]["tags"]["buckets"]
            ],
        },
    }
```

### 3.3 オートコンプリート (Completion Suggester)

```python
# コード例 3: サジェスト (オートコンプリート)
def autocomplete(prefix: str, size: int = 5, category: str = None) -> list:
    """
    入力中のキーワードに対してサジェスト候補を返す。

    Args:
        prefix: ユーザーの入力文字列
        size: 返すサジェスト数
        category: カテゴリで絞り込む場合

    Returns:
        サジェスト候補のリスト
    """
    contexts = {}
    if category:
        contexts["category"] = [category]

    body = {
        "suggest": {
            "product_suggest": {
                "prefix": prefix,
                "completion": {
                    "field": "suggest",
                    "size": size,
                    "fuzzy": {
                        "fuzziness": 1,       # 1文字のタイポを許容
                        "transpositions": True, # 文字の入れ替えを許容
                    },
                    "contexts": contexts if contexts else None,
                    "skip_duplicates": True,
                }
            }
        },
        "_source": ["title", "category", "price"],
    }

    result = es.search(index="products", body=body)
    suggestions = result["suggest"]["product_suggest"][0]["options"]

    return [
        {
            "text": s["text"],
            "title": s["_source"]["title"],
            "category": s["_source"]["category"],
            "price": s["_source"]["price"],
            "score": s["_score"],
        }
        for s in suggestions
    ]


# サジェストデータのインデックス時の設定
def build_suggest_input(product: dict) -> dict:
    """サジェスト用の入力データを構築する"""
    inputs = [product["name"]]

    # 読み仮名があれば追加 (ローマ字入力対応)
    if product.get("name_kana"):
        inputs.append(product["name_kana"])

    # ブランド名
    if product.get("brand"):
        inputs.append(product["brand"])

    # キーワード
    inputs.extend(product.get("keywords", []))

    return {
        "input": inputs,
        "weight": int(product.get("rating", 3.0) * product.get("review_count", 1)),
        "contexts": {
            "category": [product["category"]],
        }
    }
```

### 3.4 バルクインデキシング

```python
# コード例 4: バルクインデキシングとエラーハンドリング
from elasticsearch.helpers import bulk, BulkIndexError
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def index_products(products: list, chunk_size: int = 500) -> dict:
    """
    商品リストを一括でインデックスする。

    Args:
        products: 商品データのリスト
        chunk_size: 1回のバルクリクエストで送る件数

    Returns:
        成功件数とエラー件数
    """
    def generate_actions(products):
        """バルクアクション生成のジェネレータ"""
        for product in products:
            yield {
                "_index": "products",
                "_id": product["id"],
                "_source": {
                    "title": product["name"],
                    "description": product.get("description", ""),
                    "category": product["category"],
                    "tags": product.get("tags", []),
                    "price": product["price"],
                    "rating": product.get("rating", 0.0),
                    "review_count": product.get("review_count", 0),
                    "created_at": product.get("created_at",
                                              datetime.now().isoformat()),
                    "updated_at": datetime.now().isoformat(),
                    "suggest": build_suggest_input(product),
                },
            }

    try:
        success, errors = bulk(
            es,
            generate_actions(products),
            chunk_size=chunk_size,
            request_timeout=120,
            raise_on_error=False,     # エラーでも処理を続行
            raise_on_exception=False,
            max_retries=3,            # リトライ回数
            initial_backoff=1,        # リトライ間隔(秒)
            max_backoff=60,
        )
        if errors:
            logger.error(f"バルクインデックスエラー: {len(errors)} 件")
            for error in errors[:5]:  # 最初の5件のみログ
                logger.error(f"  {error}")
        logger.info(f"インデックス完了: 成功={success}, エラー={len(errors)}")
        return {"success": success, "errors": len(errors)}

    except BulkIndexError as e:
        logger.exception(f"バルクインデックス致命的エラー: {e}")
        raise


def reindex_with_zero_downtime(new_settings: dict,
                                alias: str = "products") -> str:
    """
    ダウンタイムなしでインデックスを再構築する。

    Args:
        new_settings: 新しいインデックス設定
        alias: エイリアス名

    Returns:
        新しいインデックス名
    """
    # Step 1: 現在のインデックス名を取得
    current_indices = list(es.indices.get_alias(name=alias).keys())
    current_index = current_indices[0] if current_indices else None

    # Step 2: 新しいインデックス名を生成
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    new_index = f"{alias}_v_{timestamp}"

    # Step 3: 新しいインデックスを作成
    es.indices.create(index=new_index, body=new_settings)
    logger.info(f"新インデックス '{new_index}' を作成しました")

    # Step 4: リインデックス
    es.reindex(
        body={
            "source": {"index": current_index},
            "dest": {"index": new_index},
        },
        wait_for_completion=True,
        request_timeout=3600,
    )
    logger.info(f"リインデックス完了: {current_index} → {new_index}")

    # Step 5: エイリアスを切り替え (アトミック操作)
    es.indices.update_aliases(body={
        "actions": [
            {"remove": {"index": current_index, "alias": alias}},
            {"add": {"index": new_index, "alias": alias}},
        ]
    })
    logger.info(f"エイリアス '{alias}' を '{new_index}' に切り替えました")

    # Step 6: 旧インデックスを削除 (任意)
    # es.indices.delete(index=current_index)

    return new_index
```

### 3.5 Function Score Query (スコアカスタマイズ)

```python
# コード例 5: Function Score Query による高度なランキング
def search_with_custom_scoring(
    query: str,
    user_location: tuple = None,   # (lat, lon)
    boost_new: bool = True,
    boost_popular: bool = True,
    page: int = 1,
    size: int = 20,
) -> dict:
    """
    BM25 スコアにビジネスロジックを組み合わせた検索。

    スコア = BM25 × popularity_boost × freshness_boost × distance_decay

    Args:
        query: 検索キーワード
        user_location: ユーザーの位置情報 (緯度, 経度)
        boost_new: 新しい商品をブーストするか
        boost_popular: 人気商品をブーストするか
        page: ページ番号
        size: 件数
    """
    functions = []

    # --- 人気度ブースト ---
    if boost_popular:
        functions.append({
            "field_value_factor": {
                "field": "review_count",
                "modifier": "log1p",    # log(1 + review_count)
                "factor": 0.5,
                "missing": 0,
            },
            "weight": 2,
        })
        functions.append({
            "field_value_factor": {
                "field": "rating",
                "modifier": "none",
                "factor": 1,
                "missing": 3.0,         # レビューなしは3.0扱い
            },
            "weight": 1.5,
        })

    # --- 新着ブースト (直近30日を優遇) ---
    if boost_new:
        functions.append({
            "gauss": {
                "created_at": {
                    "origin": "now",
                    "scale": "30d",     # 30日で半減
                    "offset": "7d",     # 7日以内は減衰なし
                    "decay": 0.5,
                }
            },
            "weight": 1.2,
        })

    # --- 距離ブースト (近い店舗を優遇) ---
    if user_location:
        functions.append({
            "gauss": {
                "location": {
                    "origin": {
                        "lat": user_location[0],
                        "lon": user_location[1],
                    },
                    "scale": "5km",     # 5km で半減
                    "offset": "1km",    # 1km 以内は減衰なし
                    "decay": 0.5,
                }
            },
            "weight": 1.5,
        })

    body = {
        "query": {
            "function_score": {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "description", "tags^2"],
                        "type": "best_fields",
                        "fuzziness": "AUTO",
                    }
                },
                "functions": functions,
                "score_mode": "multiply",   # 各関数の結果を掛け算
                "boost_mode": "multiply",   # BM25 と掛け算
                "max_boost": 10,            # ブースト上限
            }
        },
        "from": (page - 1) * size,
        "size": size,
    }

    return es.search(index="products", body=body)
```

### 3.6 検索ログの収集と分析

```python
# コード例 6: 検索ログの収集と分析クエリ
import json
from datetime import datetime


def log_search_event(
    query: str,
    user_id: str,
    results_count: int,
    clicked_ids: list = None,
    response_time_ms: int = 0,
):
    """検索イベントをログインデックスに記録する"""
    event = {
        "query": query,
        "user_id": user_id,
        "results_count": results_count,
        "clicked_ids": clicked_ids or [],
        "has_clicks": bool(clicked_ids),
        "response_time_ms": response_time_ms,
        "timestamp": datetime.now().isoformat(),
    }
    es.index(index="search_logs", body=event)


def analyze_zero_result_queries(days: int = 7) -> list:
    """
    ゼロ件ヒットのクエリを分析する。
    → 同義語辞書の追加候補を発見するのに有用。
    """
    body = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"timestamp": {"gte": f"now-{days}d"}}},
                    {"term": {"results_count": 0}},
                ]
            }
        },
        "aggs": {
            "zero_result_queries": {
                "terms": {
                    "field": "query.keyword",
                    "size": 50,
                    "order": {"_count": "desc"},
                }
            }
        },
        "size": 0,
    }
    result = es.search(index="search_logs", body=body)
    return [
        {"query": b["key"], "count": b["doc_count"]}
        for b in result["aggregations"]["zero_result_queries"]["buckets"]
    ]


def calculate_ctr(days: int = 7) -> list:
    """
    クエリごとの CTR (Click Through Rate) を計算する。
    → ランキング改善の指標として使用。
    """
    body = {
        "query": {
            "range": {"timestamp": {"gte": f"now-{days}d"}}
        },
        "aggs": {
            "queries": {
                "terms": {
                    "field": "query.keyword",
                    "size": 100,
                    "min_doc_count": 10,
                },
                "aggs": {
                    "click_rate": {
                        "avg": {
                            "script": {
                                "source": "doc['has_clicks'].value ? 1 : 0"
                            }
                        }
                    },
                    "avg_response_time": {
                        "avg": {"field": "response_time_ms"}
                    }
                }
            }
        },
        "size": 0,
    }
    result = es.search(index="search_logs", body=body)
    return [
        {
            "query": b["key"],
            "searches": b["doc_count"],
            "ctr": round(b["click_rate"]["value"] * 100, 1),
            "avg_response_ms": round(
                b["avg_response_time"]["value"], 0
            ),
        }
        for b in result["aggregations"]["queries"]["buckets"]
    ]
```

### 3.7 Kafka Consumer による非同期インデックス更新

```python
# コード例 7: Kafka Consumer (CDC → Elasticsearch)
from confluent_kafka import Consumer, KafkaError
import json
import signal
import sys


class SearchIndexConsumer:
    """
    Kafka から CDC イベントを消費し、Elasticsearch を更新する。
    """

    def __init__(self, kafka_config: dict, es_client, batch_size: int = 100):
        self.consumer = Consumer(kafka_config)
        self.es = es_client
        self.batch_size = batch_size
        self.running = True
        self.buffer = []

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        self.running = False

    def start(self, topics: list):
        """Consumer ループを開始する"""
        self.consumer.subscribe(topics)

        while self.running:
            msg = self.consumer.poll(timeout=1.0)
            if msg is None:
                # タイムアウト: バッファに溜まったものをフラッシュ
                if self.buffer:
                    self._flush_buffer()
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    print(f"Kafka エラー: {msg.error()}")
                    continue

            # メッセージを処理
            event = json.loads(msg.value().decode("utf-8"))
            action = self._build_action(event)
            if action:
                self.buffer.append(action)

            # バッチサイズに達したらフラッシュ
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
                self.consumer.commit()

        # 終了前にバッファをフラッシュ
        if self.buffer:
            self._flush_buffer()
        self.consumer.close()

    def _build_action(self, event: dict) -> dict:
        """CDC イベントからバルクアクションを構築する"""
        op = event.get("op")

        if op in ("c", "u", "r"):  # create, update, read (snapshot)
            after = event["after"]
            return {
                "_op_type": "index",
                "_index": "products",
                "_id": after["id"],
                "_source": {
                    "title": after["name"],
                    "description": after.get("description", ""),
                    "category": after["category"],
                    "price": after["price"],
                    "rating": after.get("rating", 0.0),
                    "updated_at": after.get("updated_at"),
                },
            }
        elif op == "d":  # delete
            before = event["before"]
            return {
                "_op_type": "delete",
                "_index": "products",
                "_id": before["id"],
            }
        return None

    def _flush_buffer(self):
        """バッファの内容を Elasticsearch に一括書き込み"""
        if not self.buffer:
            return

        from elasticsearch.helpers import bulk
        try:
            success, errors = bulk(
                self.es,
                self.buffer,
                raise_on_error=False,
            )
            if errors:
                print(f"バルクエラー: {len(errors)} 件")
            print(f"フラッシュ完了: {success} 件処理")
        except Exception as e:
            print(f"バルク書き込みエラー: {e}")
        finally:
            self.buffer = []


# 使用例
if __name__ == "__main__":
    kafka_config = {
        "bootstrap.servers": "kafka:9092",
        "group.id": "search-index-consumer",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    }
    es_client = Elasticsearch(["http://elasticsearch:9200"])
    consumer = SearchIndexConsumer(kafka_config, es_client)
    consumer.start(["dbserver1.public.products"])
```

---

## 4. 検索品質の改善手法

### 4.1 検索品質の評価指標

検索品質を客観的に評価するための主要な指標を理解する。

```
検索品質の評価指標:

  ┌────────────────┬──────────────────────────────────────────────┐
  │ 指標            │ 説明                                         │
  ├────────────────┼──────────────────────────────────────────────┤
  │ Precision@K     │ 上位K件中の関連文書の割合                      │
  │                │ = (K件中の関連文書数) / K                      │
  │                │ 例: Top-10中7件が関連 → P@10 = 0.7            │
  ├────────────────┼──────────────────────────────────────────────┤
  │ Recall@K       │ 全関連文書中、上位K件に含まれる割合             │
  │                │ = (K件中の関連文書数) / (全関連文書数)          │
  ├────────────────┼──────────────────────────────────────────────┤
  │ MRR            │ Mean Reciprocal Rank                         │
  │ (平均逆順位)    │ = 1/N × Σ(1/rank_i)                         │
  │                │ 最初の関連文書の順位を重視                      │
  │                │ 例: 3番目に関連文書 → RR = 1/3               │
  ├────────────────┼──────────────────────────────────────────────┤
  │ nDCG           │ Normalized Discounted Cumulative Gain        │
  │                │ 順位が下がるほど割引される累積ゲイン             │
  │                │ 多段階の関連度判定に対応                        │
  │                │ nDCG@10 >= 0.7 が一般的な目標                 │
  ├────────────────┼──────────────────────────────────────────────┤
  │ CTR            │ Click Through Rate                           │
  │ (クリック率)    │ = クリックされた検索数 / 総検索数               │
  │                │ オンライン指標、30-60% が健全               │
  ├────────────────┼──────────────────────────────────────────────┤
  │ Zero Result    │ 0件ヒットの検索クエリの割合                    │
  │ Rate           │ < 5% が目標                                  │
  └────────────────┴──────────────────────────────────────────────┘
```

### 4.2 クエリ書き換え (Query Rewriting)

ユーザーが入力したクエリをそのまま検索エンジンに渡すのではなく、前処理を行うことで検索品質を大幅に改善できる。

```python
# コード例 8: クエリ書き換えパイプライン
import re
from typing import List, Tuple


class QueryRewriter:
    """検索クエリの前処理と書き換えを行う"""

    def __init__(self):
        self.spelling_corrections = {
            "あいふぉん": "iPhone",
            "ぐーぐる": "Google",
            "あまぞん": "Amazon",
        }
        self.query_expansions = {
            "ノートPC": ["ノートPC", "ノートパソコン", "ラップトップ"],
            "イヤホン": ["イヤホン", "イヤフォン", "ヘッドホン"],
        }
        self.stop_patterns = [
            r"を?\s*探して(い?ます|ください)?",
            r"が?\s*欲しい(です)?",
            r"おすすめ",
        ]

    def rewrite(self, query: str) -> dict:
        """
        クエリを分析し、書き換えた結果を返す。

        Returns:
            {
                "original": 元のクエリ,
                "rewritten": 書き換え後のクエリ,
                "expansions": 展開されたクエリ,
                "corrections": 修正内容,
            }
        """
        corrections = []
        rewritten = query.strip()

        # Step 1: 自然言語パターンの除去
        for pattern in self.stop_patterns:
            cleaned = re.sub(pattern, "", rewritten).strip()
            if cleaned != rewritten:
                corrections.append(
                    f"パターン除去: '{rewritten}' → '{cleaned}'"
                )
                rewritten = cleaned

        # Step 2: スペル修正
        for wrong, correct in self.spelling_corrections.items():
            if wrong in rewritten.lower():
                rewritten = rewritten.replace(wrong, correct)
                corrections.append(f"スペル修正: {wrong} → {correct}")

        # Step 3: クエリ展開
        expansions = [rewritten]
        for term, expanded in self.query_expansions.items():
            if term in rewritten:
                expansions = expanded
                corrections.append(f"クエリ展開: {term} → {expanded}")

        return {
            "original": query,
            "rewritten": rewritten,
            "expansions": expansions,
            "corrections": corrections,
        }


# 使用例
rewriter = QueryRewriter()
result = rewriter.rewrite("ノートPCを探しています")
# → {
#     "original": "ノートPCを探しています",
#     "rewritten": "ノートPC",
#     "expansions": ["ノートPC", "ノートパソコン", "ラップトップ"],
#     "corrections": [
#         "パターン除去: 'ノートPCを探しています' → 'ノートPC'",
#         "クエリ展開: ノートPC → ['ノートPC', 'ノートパソコン', 'ラップトップ']"
#     ]
# }
```

### 4.3 Learning to Rank (LTR)

BM25 だけでは最適なランキングが得られない場合、機械学習を使ってランキングモデルを構築する手法が Learning to Rank (LTR) である。

```
Learning to Rank のアーキテクチャ:

  ┌── オフライン (モデル学習) ──────────────────────────┐
  │                                                    │
  │  [Search Logs] ─→ [Click Data] ─→ [Judgment List]  │
  │                                                    │
  │  Judgment List の例:                                │
  │  query="ノートPC", doc_id=123, grade=3 (Perfect)   │
  │  query="ノートPC", doc_id=456, grade=2 (Good)      │
  │  query="ノートPC", doc_id=789, grade=0 (Bad)       │
  │                                                    │
  │  [Feature Extraction]                              │
  │    - BM25 スコア                                   │
  │    - タイトル一致度                                  │
  │    - 商品レーティング                                │
  │    - レビュー数                                     │
  │    - 価格                                           │
  │    - 売上数                                         │
  │    - クリック数                                      │
  │                                                    │
  │  [LambdaMART / RankNet / LambdaRank]               │
  │       ↓                                            │
  │  [Trained Model]                                   │
  └────────┬───────────────────────────────────────────┘
           │
  ┌────────v── オンライン (推論) ───────────────────────┐
  │                                                    │
  │  Query → [BM25 で候補100件取得]                     │
  │       → [特徴量抽出]                                │
  │       → [LTR モデルでリスコア]                       │
  │       → [リランキング結果を返却]                     │
  │                                                    │
  │  Elasticsearch LTR Plugin:                         │
  │  POST products/_search                             │
  │  {                                                 │
  │    "query": { ... },                               │
  │    "rescore": {                                    │
  │      "window_size": 100,                           │
  │      "query": {                                    │
  │        "rescore_query": {                          │
  │          "sltr": {                                 │
  │            "model": "my_ltr_model",                │
  │            "params": { "query": "ノートPC" }        │
  │          }                                         │
  │        }                                           │
  │      }                                             │
  │    }                                               │
  │  }                                                 │
  └────────────────────────────────────────────────────┘
```

---

## 5. 比較表

### 5.1 検索エンジン比較

| 特性 | Elasticsearch | Apache Solr | Meilisearch | Typesense | OpenSearch |
|------|:------------:|:-----------:|:-----------:|:---------:|:----------:|
| ベースエンジン | Lucene | Lucene | 独自 (Rust) | 独自 (C++) | Lucene |
| ライセンス | SSPL / Elastic | Apache 2.0 | MIT | GPL v3 | Apache 2.0 |
| 日本語対応 | kuromoji | kuromoji | Lindera | 基本的 | kuromoji |
| リアルタイム検索 | 1秒以内 | 1秒以内 | 即時 | 即時 | 1秒以内 |
| 分散スケール | ネイティブ | SolrCloud | 限定的 | 限定的 | ネイティブ |
| 運用の複雑さ | 高 | 高 | 低 | 低 | 高 |
| エコシステム | Kibana, Logstash | Banana | Dashboard | Dashboard | OpenSearch Dashboards |
| マネージドサービス | Elastic Cloud, AWS | なし | Meilisearch Cloud | Typesense Cloud | AWS OpenSearch |
| 最適用途 | 大規模全文検索・ログ分析 | エンタープライズ | 小中規模・高速サジェスト | 小中規模・タイポ耐性 | 大規模 (OSS 要件) |

### 5.2 検索機能の実装方法比較

| 検索機能 | 実装方法 | 効果 | 実装難易度 |
|---------|---------|------|-----------|
| ファジー検索 | fuzziness: "AUTO" | タイポ許容 (1-2文字) | 低 |
| 同義語展開 | synonym filter | 表記揺れ対応 | 低 |
| フィールドブースト | fields: ["title^3"] | フィールド重み付け | 低 |
| ハイライト | highlight API | 該当箇所の強調表示 | 低 |
| ファセット検索 | aggregations | カテゴリ別件数表示 | 中 |
| オートコンプリート | completion suggester | 入力補完 | 中 |
| 地理検索 | geo_point + geo_distance | 距離ベースの検索 | 中 |
| Function Score | function_score query | ビジネスロジック反映 | 中 |
| クエリ書き換え | アプリ層で前処理 | 検索精度向上 | 高 |
| Learning to Rank | LTR プラグイン | ML ベースのランキング | 高 |

### 5.3 インデックス設計のフィールドタイプ選択

| ユースケース | フィールドタイプ | 理由 |
|------------|----------------|------|
| 全文検索対象 | text | トークナイズしてインデックス |
| フィルタ・ソート・集計 | keyword | 完全一致、高速 |
| 数値フィルタ | integer / float | 範囲検索、ソート |
| 日時フィルタ | date | 範囲検索、時系列 |
| 位置情報 | geo_point | 距離検索 |
| 入力補完 | completion | プレフィックス検索 |
| 保存のみ (検索不要) | object + enabled:false | ストレージ節約 |
| ネストされたオブジェクト | nested | オブジェクト内の独立検索 |

---

## 6. アンチパターン

### アンチパターン 1: RDB に全文検索を任せる

```sql
-- NG: LIKE 検索はインデックスが効かない
SELECT * FROM products
WHERE name LIKE '%東京%' OR description LIKE '%東京%';
-- → フルテーブルスキャン
-- → 100万行で2-5秒、スケールしない
-- → 形態素解析なし、ランキングなし

-- NG: MySQL FULLTEXT も CJK (日本語) に弱い
SELECT * FROM products
WHERE MATCH(name, description)
AGAINST('東京の観光' IN BOOLEAN MODE);
-- → 形態素解析なし (ngram のみ)
-- → 精度が低い、同義語非対応
-- → ファセット検索不可
```

```python
# OK: 専用の検索エンジンを使う
#
# アーキテクチャ:
#   DB (PostgreSQL) --- CDC (Debezium) ---> Kafka ---> Elasticsearch
#
# メリット:
#   - 形態素解析 (kuromoji) で日本語を正確にトークナイズ
#   - BM25 ランキングで関連性の高い結果を上位に
#   - ファセット検索、サジェスト、ハイライト全対応
#   - 水平スケーリング可能 (シャーディング)
#
# DB は SSOT (Single Source of Truth) として維持し、
# Elasticsearch は検索専用のリードモデルとして位置づける

result = es.search(
    index="products",
    body={
        "query": {
            "multi_match": {
                "query": "東京の観光",
                "fields": ["title^3", "description"],
                "analyzer": "ja_search_analyzer",
            }
        },
        "highlight": {"fields": {"title": {}, "description": {}}},
        "aggs": {"categories": {"terms": {"field": "category"}}},
    }
)
```

### アンチパターン 2: ダイナミックマッピングに頼る

```python
# NG: マッピングを定義せずにドキュメントをインデックス
es.index(index="products", body={
    "name": "テスト商品",        # → text + keyword (両方にインデックス)
    "price": 1000,              # → long (integer で十分)
    "description": "テスト説明",  # → text + keyword (keyword は不要)
    "internal_notes": "社内メモ", # → text + keyword (検索対象外なのにインデックス)
    "created_at": "2024-01-15",  # → date (正しいが偶然)
})
# 問題点:
#   - 全フィールドが text + keyword 双方にインデックスされる
#   - ストレージが2倍以上
#   - インデキシング速度が低下
#   - 不要なフィールドまで検索対象になる
```

```python
# OK: 明示的なマッピングを事前設計
# (3.1 節のインデックスマッピング設計を参照)
#
# 設計原則:
#   - 検索対象フィールド: text + 適切なアナライザー
#   - フィルタ/ソート/集計用: keyword
#   - 数値: integer / float (最小限の型を選ぶ)
#   - 検索不要フィールド: enabled: false
#   - dynamic: "strict" で未知フィールドを拒否

index_settings = {
    "mappings": {
        "dynamic": "strict",  # 未定義フィールドはエラーにする
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "ja_analyzer",
            },
            "price": {"type": "integer"},
            "metadata": {
                "type": "object",
                "enabled": False,   # インデックスしない
            },
        }
    }
}
```

### アンチパターン 3: 検索時の同義語展開をインデックス時に行う

```python
# NG: インデックス時に同義語展開
index_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "my_analyzer": {
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": ["synonym_filter"],  # インデックス時に同義語展開
                }
            },
            "filter": {
                "synonym_filter": {
                    "type": "synonym",
                    "synonyms": ["PC, パソコン, コンピュータ"]
                }
            }
        }
    }
}
# 問題点:
#   - 同義語辞書を更新するたびに全文書の再インデックスが必要
#   - インデックスサイズが膨張する
#   - 100万文書の再インデックスに数時間かかることも
```

```python
# OK: 検索時のみ同義語展開
index_settings = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ja_index_analyzer": {     # インデックス時
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": ["kuromoji_baseform", "lowercase"]
                    # ← 同義語フィルタなし
                },
                "ja_search_analyzer": {    # 検索時
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": [
                        "kuromoji_baseform",
                        "lowercase",
                        "synonym_filter",  # ← 検索時のみ同義語展開
                    ]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "ja_index_analyzer",       # インデックス時
                "search_analyzer": "ja_search_analyzer" # 検索時
            }
        }
    }
}
# メリット:
#   - 同義語辞書の更新は analyzer の reload のみ
#   - 再インデックス不要
#   - インデックスサイズが小さい
```

---

## 7. 運用と監視

### 7.1 Elasticsearch クラスタの監視

```
クラスタ監視で見るべきメトリクス:

  ┌────────────────────┬──────────────────┬──────────────────────┐
  │ メトリクス          │ 正常値           │ アラート閾値          │
  ├────────────────────┼──────────────────┼──────────────────────┤
  │ Cluster Status     │ green            │ yellow/red           │
  │ JVM Heap 使用率    │ < 75%            │ > 85%                │
  │ CPU 使用率         │ < 70%            │ > 85%                │
  │ Disk 使用率        │ < 80%            │ > 85% (watermark)    │
  │ Search Latency p99 │ < 500ms          │ > 1000ms             │
  │ Index Latency p99  │ < 200ms          │ > 500ms              │
  │ GC 頻度           │ < 5回/分          │ > 10回/分            │
  │ Circuit Breaker    │ trip = 0         │ trip > 0             │
  │ Pending Tasks      │ < 5              │ > 20                 │
  │ Rejected Threads   │ 0                │ > 0                  │
  └────────────────────┴──────────────────┴──────────────────────┘
```

### 7.2 パフォーマンスチューニングチェックリスト

```
パフォーマンスチューニング:

  インデキシング最適化:
  ┌────────────────────────────────────────────────────────────┐
  │ 1. refresh_interval: "30s" (バルクインデックス時は "-1")     │
  │ 2. number_of_replicas: 0 (バルク中は無効化)                 │
  │ 3. bulk API を使用 (1ドキュメントずつ index しない)          │
  │ 4. chunk_size: 500-1000 (ネットワーク状況に応じて調整)       │
  │ 5. translog.flush_threshold_size: "1gb" に増加             │
  └────────────────────────────────────────────────────────────┘

  検索最適化:
  ┌────────────────────────────────────────────────────────────┐
  │ 1. filter 句は bool/filter に配置 (キャッシュされる)         │
  │ 2. _source filtering で不要フィールドを除外                 │
  │ 3. size を最小限に (全件取得しない)                          │
  │ 4. scroll API → search_after に移行 (大量取得時)            │
  │ 5. doc_values: true (ソート・集計フィールド)                 │
  │ 6. fielddata は避ける (text フィールドのソートは keyword で)  │
  └────────────────────────────────────────────────────────────┘

  クラスタ最適化:
  ┌────────────────────────────────────────────────────────────┐
  │ 1. JVM Heap: 物理メモリの50%、ただし32GB以下                │
  │ 2. 残り50%はファイルシステムキャッシュに使用                  │
  │ 3. SSD を使用 (HDD 比で5-10倍高速)                         │
  │ 4. master / data / coordinator ノードを分離                 │
  │ 5. force_merge: 読み取り専用インデックスはセグメントを統合     │
  └────────────────────────────────────────────────────────────┘
```

---

## 8. 実践演習

### 演習 1: 基礎 -- 転置インデックスの手動構築

**課題**: 以下の3つの文書から、手動で転置インデックスを構築し、検索クエリの結果を求めよ。

```
文書:
  Doc1: "Python で Web アプリケーションを開発する"
  Doc2: "Python のデータ分析ライブラリ Pandas"
  Doc3: "Web フレームワーク Django で REST API を構築する"

問題:
(a) 各文書をトークナイズせよ (ストップワード "で","の","を" は除去)
(b) 転置インデックスを構築せよ
(c) 検索クエリ "Python Web" を AND 検索した場合の結果は？
(d) 検索クエリ "Python Web" を OR 検索した場合の結果は？
(e) 各文書の BM25 スコアを概算せよ
    (k1=1.2, b=0.75, avgdl=5 と仮定)
```

**期待される出力**:

```
(a) トークナイズ結果:
  Doc1: ["Python", "Web", "アプリケーション", "開発する"]
  Doc2: ["Python", "データ分析", "ライブラリ", "Pandas"]
  Doc3: ["Web", "フレームワーク", "Django", "REST", "API", "構築する"]

(b) 転置インデックス:
  "Python"           → [Doc1, Doc2]
  "Web"              → [Doc1, Doc3]
  "アプリケーション"   → [Doc1]
  "開発する"          → [Doc1]
  "データ分析"         → [Doc2]
  "ライブラリ"         → [Doc2]
  "Pandas"            → [Doc2]
  "フレームワーク"     → [Doc3]
  "Django"            → [Doc3]
  "REST"              → [Doc3]
  "API"               → [Doc3]
  "構築する"           → [Doc3]

(c) AND 検索 "Python Web":
  "Python" → {Doc1, Doc2}
  "Web"    → {Doc1, Doc3}
  AND: {Doc1, Doc2} ∩ {Doc1, Doc3} = {Doc1}
  → 結果: Doc1

(d) OR 検索 "Python Web":
  "Python" → {Doc1, Doc2}
  "Web"    → {Doc1, Doc3}
  OR: {Doc1, Doc2} ∪ {Doc1, Doc3} = {Doc1, Doc2, Doc3}
  → 結果: Doc1, Doc2, Doc3

(e) BM25 概算 (クエリ "Python Web"):
  N=3, avgdl=5

  IDF("Python") = log(1 + (3-2+0.5)/(2+0.5)) = log(1 + 0.6) ≈ 0.47
  IDF("Web")    = log(1 + (3-2+0.5)/(2+0.5)) = log(1 + 0.6) ≈ 0.47

  Doc1 (|D|=4):
    score = 0.47 × (1×2.2)/(1+1.2×(1-0.75+0.75×4/5))
          + 0.47 × (1×2.2)/(1+1.2×(1-0.75+0.75×4/5))
          ≈ 0.47 × 1.02 + 0.47 × 1.02 ≈ 0.96

  Doc2 (|D|=4, "Web"なし):
    score = 0.47 × 1.02 + 0 ≈ 0.48

  Doc3 (|D|=6, "Python"なし):
    score = 0 + 0.47 × (1×2.2)/(1+1.2×(1-0.75+0.75×6/5))
          ≈ 0 + 0.47 × 0.91 ≈ 0.43

  ランキング: Doc1 (0.96) > Doc2 (0.48) > Doc3 (0.43)
```

### 演習 2: 応用 -- Elasticsearch の検索品質改善

**課題**: 以下の Elasticsearch インデックスに対して、検索品質を改善するための設定変更を行え。

```python
"""
前提:
- 商品検索システム (ECサイト)
- 100万件の商品データ
- 検索ログから以下の問題が判明:
  1. "パソコン" で検索しても "PC" がヒットしない
  2. "iphon" (タイポ) で検索すると0件
  3. 長い商品説明の方が短いタイトル一致より上位に来る
  4. 新着商品が埋もれてしまう
  5. "東京スカイツリー" が "東京" と "スカイツリー" に分割される

課題:
(a) 問題1を解決する同義語設定を書け
(b) 問題2を解決するファジー検索設定を書け
(c) 問題3を解決するフィールドブースト設定を書け
(d) 問題4を解決する Function Score 設定を書け
(e) 問題5を解決するユーザー辞書設定を書け
"""
```

**期待される出力**:

```python
# (a) 同義語設定
synonym_settings = {
    "filter": {
        "synonym_filter": {
            "type": "synonym",
            "synonyms": [
                "PC, パソコン, コンピュータ, パーソナルコンピュータ",
                "スマホ, スマートフォン, 携帯電話",
                "テレビ, TV, ティーヴィー",
            ]
        }
    }
}

# (b) ファジー検索 (fuzziness: "AUTO" は長さに応じて自動調整)
fuzzy_query = {
    "multi_match": {
        "query": "iphon",
        "fields": ["title^3", "description"],
        "fuzziness": "AUTO",        # 3-5文字: 編集距離1, 6文字以上: 距離2
        "prefix_length": 1,         # 先頭1文字は一致必須
        "max_expansions": 50,
    }
}

# (c) フィールドブースト
boosted_query = {
    "multi_match": {
        "query": "ノートパソコン",
        "fields": [
            "title^5",          # タイトル完全一致を最重視
            "title.ngram^1",    # タイトル部分一致
            "description^1",    # 説明文は低ウェイト
        ],
        "type": "best_fields",
    }
}

# (d) 新着ブースト (Function Score)
freshness_query = {
    "function_score": {
        "query": {"match_all": {}},
        "functions": [
            {
                "gauss": {
                    "created_at": {
                        "origin": "now",
                        "scale": "14d",
                        "decay": 0.5,
                    }
                },
                "weight": 2,
            }
        ],
        "boost_mode": "multiply",
    }
}

# (e) ユーザー辞書 (kuromoji)
# userdict.txt に以下を追加:
# 東京スカイツリー,東京スカイツリー,トウキョウスカイツリー,カスタム名詞
user_dict_settings = {
    "tokenizer": {
        "kuromoji_user_dict": {
            "type": "kuromoji_tokenizer",
            "user_dictionary": "userdict.txt",
            "mode": "search",   # search モードで複合語を分割
        }
    }
}
```

### 演習 3: 発展 -- 分散検索システムの設計

**課題**: 以下の要件を満たす検索システムのアーキテクチャを設計せよ。

```
要件:
- 対象: ECサイトの商品検索
- 商品数: 5,000万件
- データサイズ: 1商品あたり平均 5KB → 合計 250GB
- 検索QPS: ピーク時 10,000 QPS
- レイテンシ: p99 < 200ms
- 可用性: 99.9%
- 日本語対応必須

設計すべき内容:
(a) シャード数とレプリカ数の決定
(b) ノード構成 (台数、スペック)
(c) インデックス更新のアーキテクチャ
(d) キャッシュ戦略
(e) 障害対応計画
```

**期待される出力**:

```
(a) シャード設計:
  データ量: 250GB
  目標シャードサイズ: 30GB
  プライマリシャード数: 250 / 30 ≈ 9 → 10 シャード
  レプリカ数: 2 (可用性 99.9% 確保)
  総シャード数: 10 × (1 + 2) = 30

(b) ノード構成:
  Master Node: 3台 (専用、小型インスタンス)
    - 4 vCPU, 8GB RAM
    - クラスタ管理のみ
  Data Node: 6台
    - 16 vCPU, 64GB RAM (Heap: 30GB)
    - SSD: 500GB
    - 1ノードあたり 5 シャード (30/6)
  Coordinator Node: 3台
    - 8 vCPU, 32GB RAM
    - Scatter-Gather の実行
  合計: 12台

(c) インデックス更新:
  Source DB (PostgreSQL)
    → Debezium CDC Connector
    → Kafka (3ブローカー、パーティション=10)
    → Index Worker (3インスタンス、Consumer Group)
    → Elasticsearch Bulk API

  更新遅延: 1-3秒
  スループット: 30,000 docs/sec (3ワーカー合計)

(d) キャッシュ戦略:
  L1: Redis (検索結果キャッシュ)
    - キー: hash(query + filters + sort + page)
    - TTL: 60秒
    - ヒット率目標: 40%
    - → 10,000 QPS × 0.4 = 4,000 QPS がキャッシュで処理
    - → ES への実際の QPS: 6,000
  L2: ES Request Cache
    - filter 句の結果をキャッシュ
    - refresh で自動無効化
  L3: ES Field Data Cache / doc_values
    - ソート・集計フィールド

(e) 障害対応:
  - ノード障害: レプリカ2なので1ノード障害は自動フェイルオーバー
  - AZ障害: 3 AZ にノードを分散配置
                AZ-a: Data×2, Master×1, Coord×1
                AZ-b: Data×2, Master×1, Coord×1
                AZ-c: Data×2, Master×1, Coord×1
  - インデックス破損: スナップショット (S3) から日次リストア
  - Kafka 障害: レプリケーション=3、ISR=2
  - Circuit Breaker: ES の memory circuit breaker で OOM 防止
```

---

## 9. FAQ

### Q1. Elasticsearch のシャード数はどう決める？

**A.** 1シャードあたり10-50GB (推奨30GB前後) が目安である。シャード数は後から変更できないため (Reindex が必要)、初期設計が重要だ。具体的な手順は以下のとおり:

1. 現在のデータ量と将来の増加率を見積もる
2. `データ量 ÷ 目標シャードサイズ` でプライマリシャード数を計算
3. ヒープ使用量が `ヒープサイズ × 20 シャード/GB` 以下になるよう調整
4. 例: 500GBのデータ、30GB/シャード → 17シャード → 切り上げて20シャード

注意点として、シャードが多すぎるとコーディネータの Scatter-Gather オーバーヘッドが増加し、少なすぎるとノード追加時にリバランスできない。時系列データの場合は ILM (Index Lifecycle Management) でロールオーバーポリシーを設定し、日次・週次でインデックスを自動分割するのが効果的である。

### Q2. 検索の関連性 (レリバンシー) を改善するには？

**A.** 段階的に改善するアプローチが有効である:

1. **Step 1 (即効性あり)**: アナライザーの最適化 -- 同義語辞書の追加、ユーザー辞書の追加 (固有名詞対応)、ストップワードの調整
2. **Step 2 (中期)**: フィールドブースト -- タイトルに高いウェイト (`title^3`)、カテゴリに中程度のウェイト (`tags^2`)
3. **Step 3 (中期)**: Function Score Query -- 人気度 (`log1p(review_count)`)、新着度 (`gauss(created_at)`)、距離 (`gauss(location)`) をスコアに反映
4. **Step 4 (長期)**: 検索ログ分析 -- ゼロ件ヒットクエリの分析 → 同義語辞書追加、CTR の低いクエリ → ランキング調整
5. **Step 5 (長期)**: Learning to Rank -- クリックスルーデータから機械学習モデルを構築し、BM25 + Function Score では捉えられない複雑な関連性を学習

### Q3. インデックスの再構築はどうやる？

**A.** ダウンタイムなしで再構築するには Alias パターンを使う:

1. 新インデックス `products_v2` を新しいマッピングで作成
2. Reindex API で `products_v1` → `products_v2` にデータをコピー
3. Alias `products` を `products_v1` → `products_v2` にアトミックに切り替え
4. 旧インデックス `products_v1` を削除

クライアントは常に Alias 名 (`products`) でアクセスするため、切り替えは透過的に行われる。大量データの場合は `slices` パラメータで並列リインデックスすると高速化できる。コード例は 3.4 節の `reindex_with_zero_downtime` 関数を参照。

### Q4. Elasticsearch と RDB のデータ不整合はどう防ぐ？

**A.** CDC (Change Data Capture) パターンを使えば、RDB のトランザクションログ (WAL) から変更を検出するため、アプリケーション層のバグによるデータ不整合を防げる。ただし、以下の対策も必要:

1. **整合性チェックジョブ**: 定期的に RDB と ES のドキュメント数・更新日時を比較
2. **デッドレターキュー**: インデキシング失敗したイベントを別キューに退避し、後で再処理
3. **フルリビルド**: 月次で全データの再インデックスを実行 (差分では補えない不整合を解消)
4. **べき等性**: Index Worker をべき等に実装 (同じイベントを2回処理しても同じ結果)

### Q5. 検索速度が遅い場合のデバッグ方法は？

**A.** 以下の手順で原因を特定する:

1. **Profile API** でクエリの各フェーズの実行時間を確認
   ```
   POST products/_search
   { "profile": true, "query": { ... } }
   ```
2. **Slow Log** を有効化して遅いクエリを特定
   ```
   PUT products/_settings
   { "index.search.slowlog.threshold.query.warn": "1s" }
   ```
3. **Hot Threads API** で CPU ボトルネックを特定
   ```
   GET _nodes/hot_threads
   ```
4. よくある原因と対策:
   - Wildcard クエリの先頭に `*` → 避ける
   - Script Score の重い計算 → Painless スクリプトの最適化
   - 大量の aggregation → 必要最小限に削減
   - 巨大な _source の返却 → `_source` フィルタリング

---

## 10. まとめ

| 項目 | ポイント |
|------|---------|
| 転置インデックス | 検索エンジンの中核データ構造。語 → 文書リストのマッピング。Lucene では FST + Skip List で高速化 |
| アナライザー | Character Filter → Tokenizer → Token Filter のパイプライン。日本語には kuromoji が必須 |
| BM25 | TF-IDF の改良版。TF の飽和と文書長の正規化により、より自然なランキングを実現 |
| 分散検索 | Scatter-Gather パターン。Query Phase + Fetch Phase の2段階。シャード数がスケールの鍵 |
| Elasticsearch | Lucene ベースの大規模全文検索エンジン。kuromoji で日本語対応、豊富な検索 DSL |
| インデックス更新 | CDC (Debezium) + Kafka パイプラインで DB 変更を非同期に ES へ反映。遅延1-5秒 |
| 検索品質改善 | 同義語辞書 → ブースト → Function Score → Query Rewriting → Learning to Rank の段階的改善 |
| 運用 | Alias パターンでゼロダウンタイム再構築、ILM で時系列管理、Profile API でデバッグ |

---

## 次に読むべきガイド

- [レートリミッター設計](./03-rate-limiter.md) -- 検索 API のレート制限設計
- [通知システム設計](./02-notification-system.md) -- 検索アラート通知の実装
- [CDN](../01-components/03-cdn.md) -- 検索結果ページのキャッシュ戦略
- [DB スケーリング](../01-components/04-database-scaling.md) -- データソース (RDB) のスケーリング
- [メッセージキュー](../01-components/02-message-queue.md) -- CDC パイプラインの Kafka 設計
- [キャッシング](../01-components/01-caching.md) -- Redis を使った検索結果キャッシュ
- [イベント駆動アーキテクチャ](../02-architecture/03-event-driven.md) -- CDC とイベントストリーミングの基盤
- [CQRS / Event Sourcing](../../../design-patterns-guide/docs/04-architectural/02-event-sourcing-cqrs.md) -- 検索インデックスをリードモデルとして捉える CQRS パターン

---

## 参考文献

1. **Elasticsearch: The Definitive Guide** -- Clinton Gormley & Zachary Tong (O'Reilly, 2015) -- Elasticsearch の包括的リファレンス
2. **Information Retrieval: Implementing and Evaluating Search Engines** -- Christopher Manning, Prabhakar Raghavan, Hinrich Schutze (Cambridge University Press, 2008) -- 情報検索の理論的基盤 (転置インデックス、BM25、評価指標)
3. **Relevant Search** -- Doug Turnbull & John Berryman (Manning, 2016) -- 検索の関連性改善の実践ガイド
4. **Elasticsearch 公式ドキュメント** -- https://www.elastic.co/guide/ -- マッピング、クエリ DSL、クラスタ管理のリファレンス
5. **Apache Lucene 公式サイト** -- https://lucene.apache.org/ -- Elasticsearch の内部エンジンの仕様
6. **Designing Data-Intensive Applications** -- Martin Kleppmann (O'Reilly, 2017) -- 分散システム、CDC、ストリーム処理の理論的基盤
