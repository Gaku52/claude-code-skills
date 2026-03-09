# データ構造の選び方

> 適切なデータ構造を選ぶことは、適切なアルゴリズムを選ぶことと同等に重要である。
> --- Niklaus Wirth, "Algorithms + Data Structures = Programs" (1976)

## この章で学ぶこと

- [ ] 要件に応じて最適なデータ構造を選択できる
- [ ] 各データ構造のトレードオフを定量的に理解する
- [ ] 選択基準を体系的に整理し、判断フレームワークを身につける
- [ ] 実務でよくある場面に対して即座に適切な構造を選べる
- [ ] アンチパターンを認識し、回避できる

## 前提知識


---

## 1. データ構造選択の重要性

### 1.1 なぜデータ構造の選択が決定的なのか

ソフトウェア工学において、データ構造の選択はシステム全体の性能・保守性・拡張性を根本的に左右する。Niklaus Wirth が著書のタイトル "Algorithms + Data Structures = Programs" で示したように、アルゴリズムとデータ構造は車の両輪であり、どちらか一方だけを最適化しても十分な成果は得られない。

データ構造の選択が与える影響を具体的に整理すると、以下の 4 つの軸に分類できる。

```
データ構造選択の影響範囲:

  ┌──────────────────────────────────────────────────────┐
  │              データ構造の選択                          │
  └──────────┬───────────┬───────────┬──────────┬────────┘
             │           │           │          │
             ▼           ▼           ▼          ▼
  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ 時間計算量   │ │空間計算量│ │ 可読性   │ │ 拡張性   │
  │              │ │          │ │          │ │          │
  │ 操作ごとの   │ │メモリ消費│ │コードの  │ │要件変更  │
  │ 実行速度     │ │ と局所性 │ │明快さ    │ │への対応力│
  └──────────────┘ └──────────┘ └──────────┘ └──────────┘
```

#### 時間計算量の観点

同じ「値の検索」という操作でも、使用するデータ構造によって計算量は劇的に異なる。未ソートの配列での線形探索は O(n) だが、ハッシュテーブルなら期待 O(1)、平衡 BST なら O(log n) で完了する。10 万件のデータに対して、n=100,000 回の検索を行う場面を想定すると、理論上の比較回数は以下のようになる。

| データ構造 | 1 回の検索 | 10 万回の検索 |
|---|---|---|
| 未ソート配列 (線形探索) | 平均 50,000 回 | 約 50 億回 |
| ソート済み配列 (二分探索) | 約 17 回 | 約 170 万回 |
| ハッシュテーブル | 期待 1 回 | 約 10 万回 |
| 平衡 BST | 約 17 回 | 約 170 万回 |

この差は、小規模データでは体感できないことが多いが、データ量が増えるにつれて指数的に顕在化する。

#### 空間計算量の観点

メモリ使用量も無視できない要素である。ハッシュテーブルは高速な検索を実現する代わりに、ハッシュ値格納やチェイン用ポインタ、負荷率維持のための未使用スロットなど、追加のメモリオーバーヘッドを伴う。一方、配列は要素を連続領域に格納するため、メモリ効率が高く CPU キャッシュの恩恵も受けやすい。

組み込みシステムやモバイルアプリケーションなど、メモリが制約されている環境では、空間効率の観点からデータ構造を選ぶ必要がある場面も少なくない。

#### 可読性の観点

データ構造の選択はコードの可読性にも直結する。Python の `dict` を使ったルックアップテーブルは、if-elif の連鎖よりも意図が明確であり、保守しやすい。同様に、`set` を使った重複排除は、リストに対する手動ループよりもコードの意図を端的に表現できる。

#### 拡張性の観点

初期設計で選んだデータ構造は、後から変更するコストが非常に高い。API のインターフェースやデータベースのスキーマに組み込まれてしまうと、内部構造の変更がシステム全体に波及する。したがって、将来の要件変更を見越した選択が重要になる。

### 1.2 選択を間違えるとどうなるか

データ構造の選択ミスは、以下のような形で顕在化する。

1. **性能の崩壊**: データ量が増えるにつれて急激にレスポンスが悪化する。典型例は、リスト内検索を多用するシステムが、ハッシュセットで解決できるケース。
2. **メモリの浪費**: 不要な冗長性を持つ構造の選択。例えば、少数の固定キーに対して巨大なハッシュテーブルを用意する。
3. **コードの複雑化**: 不適切なデータ構造を補うための「回避策コード」が増殖し、保守コストが上昇する。
4. **並行処理の困難**: スレッドセーフでないデータ構造を選んだ結果、後からロック機構を追加する必要に迫られる。

### 1.3 この章のアプローチ

本章では、データ構造の選択を「勘と経験」ではなく「体系的な判断プロセス」として整理する。具体的には以下のステップで進める。

1. 選択基準の軸を明確化する（セクション 2）
2. 主要データ構造の特性を定量的に比較する（セクション 3）
3. ユースケースごとの推奨を示す（セクション 4）
4. 実装例とベンチマークで裏付ける（セクション 5）
5. 判断フローチャートで体系化する（セクション 6）
6. 陥りやすいアンチパターンを提示する（セクション 7）

---

## 2. 選択基準

データ構造を選択する際には、複数の基準を同時に考慮する必要がある。ここでは主要な 6 つの基準軸を解説する。

### 2.1 操作頻度と操作種別

最も基本的な基準は「どの操作を最も頻繁に行うか」である。データ構造ごとに得意な操作と不得意な操作があるため、主要操作の計算量が支配的要因となる。

```
操作分類マトリクス:

  ┌────────────────────────────────────────────────────────┐
  │                    操作の種類                           │
  ├──────────────┬──────────────┬──────────────────────────┤
  │  読み取り系  │  書き込み系  │     特殊操作             │
  ├──────────────┼──────────────┼──────────────────────────┤
  │ ・インデックス│ ・挿入       │ ・ソート                │
  │   アクセス   │ ・削除       │ ・範囲検索              │
  │ ・検索       │ ・更新       │ ・前方一致/部分一致     │
  │ ・最小/最大  │ ・先頭追加   │ ・集合演算（和/積/差）  │
  │ ・順序走査   │ ・末尾追加   │ ・マージ               │
  │ ・ランダム   │ ・中間挿入   │ ・Top-K / 中央値       │
  │   アクセス   │              │ ・ランク（順位）       │
  └──────────────┴──────────────┴──────────────────────────┘
```

例えば「挿入は稀だが検索が頻繁」という要件であれば、挿入コストが高くても検索が高速なデータ構造（ソート済み配列 + 二分探索、ハッシュテーブルなど）を選ぶべきである。逆に「挿入・削除が頻繁だが検索はほぼ行わない」なら、連結リストやキューが候補になる。

以下のコード例は、操作パターンに応じた構造選択の基本的な考え方を示す。

```python
"""
コード例1: 操作パターンによるデータ構造選択の比較

要件: 大量の整数データに対して「存在確認」を頻繁に行う
"""
import time
from typing import List, Set


def benchmark_membership_test(data_list: List[int], data_set: Set[int],
                               queries: List[int]) -> None:
    """リストとセットでの存在確認の性能差を示す"""

    # --- リストでの存在確認: O(n) ---
    start = time.perf_counter()
    count_list = 0
    for q in queries:
        if q in data_list:
            count_list += 1
    elapsed_list = time.perf_counter() - start

    # --- セットでの存在確認: 期待 O(1) ---
    start = time.perf_counter()
    count_set = 0
    for q in queries:
        if q in data_set:
            count_set += 1
    elapsed_set = time.perf_counter() - start

    print(f"データ件数: {len(data_list):>10,}")
    print(f"クエリ件数: {len(queries):>10,}")
    print(f"リスト検索: {elapsed_list:.4f} 秒  (ヒット数: {count_list})")
    print(f"セット検索: {elapsed_set:.4f} 秒  (ヒット数: {count_set})")
    print(f"速度比:     {elapsed_list / max(elapsed_set, 1e-9):.1f} 倍")
    print()


if __name__ == "__main__":
    import random
    random.seed(42)

    for size in [1_000, 10_000, 100_000]:
        data = list(range(size))
        data_set = set(data)
        queries = [random.randint(0, size * 2) for _ in range(10_000)]
        benchmark_membership_test(data, data_set, queries)
```

このコードを実行すると、データ件数が増えるほどリストとセットの性能差が拡大することが確認できる。1,000 件程度では数倍の差だが、100,000 件では数百倍以上の差になる。

### 2.2 データサイズと成長パターン

データの規模と成長パターンは、選択に大きな影響を与える。

| データ規模 | 特徴 | 推奨戦略 |
|---|---|---|
| 小規模 (〜1,000) | 計算量の定数項が支配的 | 単純な構造（配列）で十分 |
| 中規模 (1,000〜100,000) | 計算量のオーダーが影響し始める | 要件に応じて適切な構造を選択 |
| 大規模 (100,000〜) | オーダーの差が支配的 | O(1) や O(log n) の構造が必須 |
| 超大規模 (数億〜) | メモリに乗りきらない可能性 | 外部記憶向け構造（B+木等）を検討 |

重要な注意点として、**小規模データでは単純なデータ構造のほうが高速な場合がある**。これは、計算量の定数項やキャッシュ効率の影響である。配列の線形探索は O(n) だが、n が小さければ CPU キャッシュに乗る連続メモリアクセスの恩恵で、ハッシュテーブルの O(1) 検索より速いこともある。

### 2.3 メモリ使用量と局所性

データ構造ごとにメモリの使い方は大きく異なる。以下に Python での代表的なメモリオーバーヘッドを整理する。

```
Python における各構造のメモリ特性（概算）:

  構造          要素あたりの追加オーバーヘッド    メモリ局所性
  ──────────    ──────────────────────────────    ──────────
  list          8 バイト（ポインタ配列）           高い
  tuple         8 バイト（ポインタ配列）           高い（不変）
  set           各要素にハッシュ値格納             中程度
  dict          キー + 値 + ハッシュ値             中程度
  deque         ブロック単位の連結                 中程度
  連結リスト    ノードオブジェクト + ポインタ      低い

  ※ Python のオブジェクトは全てヒープ上に確保されるため、
    C/C++ の配列ほどの局所性は得られない点に注意
```

現代のプロセッサは階層的なキャッシュ機構を持っており、メモリアクセスの局所性（locality）が性能に大きく影響する。配列は連続メモリ領域を使用するため、キャッシュラインを効率的に活用できる。一方、連結リストやツリー構造はノードが散在するため、キャッシュミスが頻発しやすい。

### 2.4 順序の保持

データの順序を保持する必要があるかどうかは、重要な選択基準である。

- **順序不要**: ハッシュテーブル、ハッシュセットが最適。挿入順さえ不要なら最も高速。
- **挿入順の保持**: Python 3.7+ の `dict` は挿入順を保持する。`collections.OrderedDict` も同等。
- **ソート順の維持**: `sortedcontainers.SortedList`、平衡 BST（`TreeMap` 等）、B+木。
- **カスタム順序**: ヒープ（優先度キュー）で優先度に基づく順序を管理。

### 2.5 並行性とスレッドセーフティ

マルチスレッド環境では、データ構造のスレッドセーフティが重要になる。

| 戦略 | 特徴 | 適用場面 |
|---|---|---|
| ロックなし（immutable） | 不変データ構造を使用 | 読み取り専用データの共有 |
| グローバルロック | 全操作にロックを取得 | 低競合の簡易な共有 |
| 細粒度ロック | 操作ごとに最小限のロック | 高競合の読み書き混在 |
| ロックフリー構造 | CAS 等で非ロック同期 | 超高並行性が求められる場面 |
| スレッドローカル | 各スレッドに独立コピー | 書き込みが主体の場面 |

Python では GIL（Global Interpreter Lock）の存在により、CPU バウンドな操作ではマルチスレッドの恩恵が限定的だが、I/O バウンドな処理や `multiprocessing` を使う場合には依然として重要な考慮事項である。

`queue.Queue`、`queue.PriorityQueue`、`collections.deque`（両端の append/pop はスレッドセーフ）などは、スレッドセーフなデータ構造として利用できる。

### 2.6 永続性と直列化

データをディスクに保存したり、ネットワーク越しに送受信する場合、直列化（シリアライゼーション）の容易さも選択基準になる。

- **配列・辞書**: JSON、MessagePack、Protocol Buffers 等で容易に直列化可能。
- **ツリー・グラフ**: ノード間の参照関係の表現に工夫が必要。
- **カスタム構造**: 独自の直列化ロジックが必要になる場合がある。

また、永続データ構造（persistent data structure）という概念も存在する。これは変更時に新しいバージョンを作成し、過去のバージョンも参照可能な構造である。関数型プログラミングでよく使われ、Git の内部データモデルもこの概念に基づいている。

---

## 3. 主要データ構造の特性比較

### 3.1 計算量の総合比較表

以下は、主要データ構造における各操作の計算量を網羅的にまとめた表である。選択の際の基本参照資料として活用できる。

```
主要データ構造の計算量一覧（平均 / 最悪）:

  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────┐
  │ データ構造        │ アクセス     │ 検索         │ 挿入         │ 削除         │ 空間     │
  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────┤
  │ 静的配列          │ O(1)/O(1)    │ O(n)/O(n)    │ O(n)/O(n)    │ O(n)/O(n)    │ O(n)     │
  │ 動的配列 (list)   │ O(1)/O(1)    │ O(n)/O(n)    │ O(1)*/O(n)   │ O(n)/O(n)    │ O(n)     │
  │ 単方向連結リスト  │ O(n)/O(n)    │ O(n)/O(n)    │ O(1)/O(1)    │ O(1)/O(1)    │ O(n)     │
  │ 双方向連結リスト  │ O(n)/O(n)    │ O(n)/O(n)    │ O(1)/O(1)    │ O(1)/O(1)    │ O(n)     │
  │ スタック          │ O(n)/O(n)    │ O(n)/O(n)    │ O(1)/O(1)    │ O(1)/O(1)    │ O(n)     │
  │ キュー            │ O(n)/O(n)    │ O(n)/O(n)    │ O(1)/O(1)    │ O(1)/O(1)    │ O(n)     │
  │ ハッシュテーブル  │ N/A          │ O(1)/O(n)    │ O(1)/O(n)    │ O(1)/O(n)    │ O(n)     │
  │ ハッシュセット    │ N/A          │ O(1)/O(n)    │ O(1)/O(n)    │ O(1)/O(n)    │ O(n)     │
  │ BST（非平衡）    │ O(log n)/O(n)│ O(log n)/O(n)│ O(log n)/O(n)│ O(log n)/O(n)│ O(n)     │
  │ 平衡BST           │ O(log n)     │ O(log n)     │ O(log n)     │ O(log n)     │ O(n)     │
  │ 二分ヒープ        │ O(1)†        │ O(n)/O(n)    │ O(log n)     │ O(log n)     │ O(n)     │
  │ Trie              │ N/A          │ O(m)/O(m)    │ O(m)/O(m)    │ O(m)/O(m)    │ O(SIGMA) │
  │ B木 / B+木        │ O(log n)     │ O(log n)     │ O(log n)     │ O(log n)     │ O(n)     │
  │ スキップリスト    │ O(log n)     │ O(log n)     │ O(log n)     │ O(log n)     │ O(n)     │
  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────┘

  * 末尾追加の償却計算量  † 最小または最大のみ O(1)
  m = キーの長さ  SIGMA = 全キーの文字数合計
```

### 3.2 操作別の最適構造

各操作において最適なデータ構造は以下の通りである。

**インデックスアクセス（i 番目の要素を取得）**
- 最適: 配列 / 動的配列 → O(1)
- 次点: スキップリスト → O(log n)（インデックス操作を拡張した場合）
- 不適: 連結リスト → O(n)

**値の検索（指定値の存在確認）**
- 最適: ハッシュテーブル / ハッシュセット → 期待 O(1)
- 次点: 平衡 BST / ソート済み配列 → O(log n)
- 不適: 未ソート配列 / 連結リスト → O(n)

**最小値 / 最大値の取得**
- 最適: 二分ヒープ → O(1)（片方のみ）
- 次点: 平衡 BST → O(log n)（両方取得可能）
- 不適: ハッシュテーブル → O(n)

**範囲検索（a 以上 b 以下の要素を全取得）**
- 最適: 平衡 BST / B+木 → O(log n + k)（k は結果の個数）
- 次点: ソート済み配列 + 二分探索 → O(log n + k)
- 不適: ハッシュテーブル → O(n)

**先頭への挿入 / 削除**
- 最適: 連結リスト / deque → O(1)
- 次点: なし
- 不適: 配列 → O(n)（全要素のシフトが発生）

**ソート順の動的維持**
- 最適: 平衡 BST / スキップリスト → O(log n)
- 次点: ソート済み配列（挿入は O(n) だが検索は O(log n)）
- 不適: ハッシュテーブル（順序を持たない）

### 3.3 Python 標準ライブラリにおける対応

Python の標準ライブラリは、多くのデータ構造を組み込み型やモジュールとして提供している。

```
Python 標準ライブラリのデータ構造対応:

  抽象的な構造              Python での実装
  ──────────────            ──────────────────────────────
  動的配列                  list
  不変配列                  tuple
  ハッシュテーブル          dict
  ハッシュセット            set / frozenset
  両端キュー (deque)        collections.deque
  ヒープ (優先度キュー)     heapq モジュール (list ベース)
  スタック                  list（末尾 append/pop）
  FIFO キュー               collections.deque（または queue.Queue）
  順序付き辞書              dict（3.7+）/ collections.OrderedDict
  デフォルト辞書            collections.defaultdict
  カウンタ                  collections.Counter
  名前付きタプル            collections.namedtuple / typing.NamedTuple
  不変集合                  frozenset
  ビット配列                int のビット演算 / array.array
  型付き配列                array.array / numpy.ndarray（外部）

  ※ 平衡 BST、Trie、スキップリスト等は標準ライブラリに含まれないため、
    sortedcontainers（外部）や自前実装が必要
```

### 3.4 言語間の対応関係

複数言語でプログラミングを行う場合、同等の構造が異なる名前で提供されている点に注意が必要である。

| 抽象構造 | Python | Java | C++ | Go | Rust |
|---|---|---|---|---|---|
| 動的配列 | `list` | `ArrayList` | `std::vector` | `slice` | `Vec<T>` |
| 連結リスト | (自前) | `LinkedList` | `std::list` | `list.List` | `LinkedList<T>` |
| ハッシュマップ | `dict` | `HashMap` | `std::unordered_map` | `map` | `HashMap<K,V>` |
| ハッシュセット | `set` | `HashSet` | `std::unordered_set` | (map利用) | `HashSet<T>` |
| ソート済みマップ | (外部) | `TreeMap` | `std::map` | (外部) | `BTreeMap<K,V>` |
| ヒープ | `heapq` | `PriorityQueue` | `std::priority_queue` | `heap` | `BinaryHeap<T>` |
| 両端キュー | `deque` | `ArrayDeque` | `std::deque` | (外部) | `VecDeque<T>` |
| スタック | `list` | `Stack`/`Deque` | `std::stack` | `slice` | `Vec<T>` |

---

## 4. ユースケース別ガイド

### 4.1 検索の最適化

検索は最も頻繁に行われる操作の一つであり、用途に応じて最適な構造が異なる。

#### 完全一致検索

要素が存在するかどうか、またはキーに対応する値を取得する操作。

- **推奨**: ハッシュテーブル（`dict`）/ ハッシュセット（`set`）
- **計算量**: 期待 O(1)
- **注意点**: ハッシュ関数の品質に依存。最悪 O(n) になる可能性があるが、Python の組み込みハッシュ関数は十分に高品質。

```python
"""
コード例2: 検索パターン別の最適構造
"""


# --- 完全一致検索: dict / set が最適 ---
def build_user_lookup(users: list[dict]) -> dict[int, dict]:
    """ユーザーリストから ID → ユーザー情報のルックアップテーブルを構築"""
    return {user["id"]: user for user in users}


# 使用例
users = [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"},
    {"id": 3, "name": "Charlie", "email": "charlie@example.com"},
]
lookup = build_user_lookup(users)
print(lookup[2])  # O(1) で取得: {'id': 2, 'name': 'Bob', ...}


# --- 複数条件での検索: 複合インデックスの構築 ---
from collections import defaultdict


def build_multi_index(users: list[dict]) -> dict:
    """複数のキーでインデックスを構築"""
    index = {
        "by_id": {},
        "by_email": {},
        "by_name_prefix": defaultdict(list),
    }
    for user in users:
        index["by_id"][user["id"]] = user
        index["by_email"][user["email"]] = user
        # 名前の各プレフィックスでインデックス（簡易 Trie 的アプローチ）
        name = user["name"].lower()
        for i in range(1, len(name) + 1):
            index["by_name_prefix"][name[:i]].append(user)
    return index


multi_idx = build_multi_index(users)
print(multi_idx["by_email"]["bob@example.com"])     # O(1)
print(multi_idx["by_name_prefix"]["ch"])             # O(1) + 結果数
```

#### 範囲検索

「10 以上 20 以下」「日付が先月の範囲内」のように、値の範囲に基づいて検索する操作。

- **推奨**: ソート済み配列 + `bisect` / 平衡 BST
- **計算量**: O(log n + k)（k は結果の個数）
- **ハッシュテーブルでは不可能**: ハッシュテーブルは順序を持たないため、範囲検索には対応できない。

```python
"""
コード例3: 範囲検索の実装（bisect モジュール活用）
"""
import bisect
from datetime import datetime, timedelta


class TimeSeriesIndex:
    """タイムスタンプベースの範囲検索を効率的に行うインデックス"""

    def __init__(self):
        self._timestamps: list[float] = []
        self._values: list[dict] = []

    def insert(self, timestamp: datetime, value: dict) -> None:
        """タイムスタンプ順を維持しながらデータを挿入"""
        ts = timestamp.timestamp()
        pos = bisect.bisect_left(self._timestamps, ts)
        self._timestamps.insert(pos, ts)
        self._values.insert(pos, value)

    def range_query(self, start: datetime, end: datetime) -> list[dict]:
        """指定した時間範囲内のデータを取得 - O(log n + k)"""
        start_ts = start.timestamp()
        end_ts = end.timestamp()
        left = bisect.bisect_left(self._timestamps, start_ts)
        right = bisect.bisect_right(self._timestamps, end_ts)
        return self._values[left:right]

    def __len__(self) -> int:
        return len(self._timestamps)


# 使用例
index = TimeSeriesIndex()
base_time = datetime(2025, 1, 1)

# データ挿入
for i in range(1000):
    t = base_time + timedelta(hours=i)
    index.insert(t, {"event_id": i, "time": t.isoformat()})

# 範囲検索: 最初の24時間のイベント
results = index.range_query(
    base_time,
    base_time + timedelta(hours=24)
)
print(f"最初の24時間のイベント数: {len(results)}")  # 25
print(f"最初のイベント: {results[0]}")
print(f"最後のイベント: {results[-1]}")
```

#### 前方一致検索（プレフィックス検索）

- **推奨**: Trie（トライ木）
- **計算量**: O(m)（m はプレフィックスの長さ）
- **用途**: オートコンプリート、辞書の前方一致、IP アドレスのルーティングテーブル

### 4.2 順序付きデータの管理

データを常にソートされた状態で保持し、動的に追加・削除する必要がある場面。

| 要件 | 推奨構造 | 理由 |
|---|---|---|
| 静的データのソート | 配列 + sort | O(n log n) でソート後、O(log n) で検索 |
| 動的にソート順維持 | 平衡 BST / SortedList | 挿入・削除が O(log n) |
| Top-K の動的管理 | ヒープ | 挿入 O(log K)、最小/最大取得 O(1) |
| ランク（順位）クエリ | 順序統計木 | k 番目の要素を O(log n) で取得 |

### 4.3 FIFO / LIFO パターン

#### スタック（LIFO: Last In, First Out）

典型的な用途:
- 関数呼び出しの管理
- Undo/Redo 機能
- 括弧の対応チェック
- DFS（深さ優先探索）
- 式の評価（逆ポーランド記法）

Python での実装: `list` の `append()` / `pop()` が最もシンプル。

#### キュー（FIFO: First In, First Out）

典型的な用途:
- タスクキュー / ジョブキュー
- BFS（幅優先探索）
- イベント処理
- バッファリング

Python での実装: `collections.deque` が最適。`list` の `pop(0)` は O(n) なので不適切。

#### 優先度キュー

典型的な用途:
- Dijkstra のアルゴリズム
- タスクスケジューリング（優先度付き）
- リアルタイム Top-K
- イベント駆動シミュレーション

Python での実装: `heapq` モジュール。

```python
"""
コード例4: 用途に応じたキュー選択
"""
import heapq
from collections import deque


# --- 通常のキュー (FIFO): deque ---
class TaskQueue:
    """単純な先入先出のタスクキュー"""

    def __init__(self):
        self._queue = deque()

    def enqueue(self, task: str) -> None:
        self._queue.append(task)

    def dequeue(self) -> str:
        if not self._queue:
            raise IndexError("キューが空です")
        return self._queue.popleft()  # O(1)

    def __len__(self) -> int:
        return len(self._queue)


# --- 優先度キュー: heapq ---
class PriorityTaskQueue:
    """優先度付きタスクキュー（数値が小さいほど高優先度）"""

    def __init__(self):
        self._heap: list[tuple[int, int, str]] = []
        self._counter = 0  # タイブレーカー（同一優先度の安定性）

    def enqueue(self, task: str, priority: int) -> None:
        heapq.heappush(self._heap, (priority, self._counter, task))
        self._counter += 1

    def dequeue(self) -> tuple[int, str]:
        if not self._heap:
            raise IndexError("キューが空です")
        priority, _, task = heapq.heappop(self._heap)  # O(log n)
        return priority, task

    def peek(self) -> tuple[int, str]:
        if not self._heap:
            raise IndexError("キューが空です")
        priority, _, task = self._heap[0]  # O(1)
        return priority, task

    def __len__(self) -> int:
        return len(self._heap)


# 使用例
print("=== 通常のキュー ===")
tq = TaskQueue()
for task in ["メール送信", "ログ出力", "DB更新"]:
    tq.enqueue(task)
while len(tq) > 0:
    print(f"  処理: {tq.dequeue()}")

print("\n=== 優先度キュー ===")
pq = PriorityTaskQueue()
pq.enqueue("バグ修正", priority=1)       # 最高優先度
pq.enqueue("新機能開発", priority=3)     # 低優先度
pq.enqueue("セキュリティ修正", priority=1)  # 最高優先度
pq.enqueue("ドキュメント更新", priority=5)  # 最低優先度
pq.enqueue("パフォーマンス改善", priority=2)

while len(pq) > 0:
    pri, task = pq.dequeue()
    print(f"  優先度 {pri}: {task}")
```

### 4.4 集合演算

重複排除、和集合、積集合、差集合などの集合演算が必要な場面。

- **推奨**: `set` / `frozenset`
- **用途**: タグのフィルタリング、権限管理、データの重複排除、共通要素の抽出

```python
# 集合演算の活用例
admin_permissions = {"read", "write", "delete", "admin"}
editor_permissions = {"read", "write"}
viewer_permissions = {"read"}

# 和集合: ユーザーが持つ全権限
user_roles = ["editor", "viewer"]
all_permissions: set[str] = set()
role_map = {
    "admin": admin_permissions,
    "editor": editor_permissions,
    "viewer": viewer_permissions,
}
for role in user_roles:
    all_permissions |= role_map[role]  # O(len(smaller_set))
print(f"全権限: {all_permissions}")  # {'read', 'write'}

# 積集合: 全ロールに共通の権限
common = admin_permissions & editor_permissions & viewer_permissions
print(f"共通権限: {common}")  # {'read'}

# 差集合: admin だけが持つ権限
admin_only = admin_permissions - editor_permissions
print(f"admin固有: {admin_only}")  # {'delete', 'admin'}
```

### 4.5 キャッシュ

アクセス頻度に基づいてデータを保持・排除する必要がある場面。

- **LRU キャッシュ**: `functools.lru_cache` または `collections.OrderedDict`
- **内部構造**: ハッシュテーブル + 双方向連結リスト
- **計算量**: get/put ともに O(1)

### 4.6 グラフの表現

エンティティ間の関係を表現する必要がある場面。

| 表現方法 | 適用場面 | 空間計算量 |
|---|---|---|
| 隣接行列 | 密グラフ、辺の存在確認が頻繁 | O(V^2) |
| 隣接リスト | 疎グラフ、隣接頂点の列挙が頻繁 | O(V + E) |
| 辺リスト | Kruskal のアルゴリズム等 | O(E) |
| 隣接マップ | 重み付きグラフ、辺の存在確認と列挙の両方 | O(V + E) |

多くの実用的なグラフ（ソーシャルネットワーク、Web のリンク構造等）は疎グラフであるため、隣接リストまたは隣接マップが第一選択になる。

---

## 5. 実装例とベンチマーク比較

### 5.1 同一問題に対する異なるデータ構造の適用

ここでは「重複排除」という具体的な問題に対して、異なるデータ構造を使った解法を比較する。

```python
"""
コード例5: 重複排除の実装比較とベンチマーク
"""
import time
import random
from typing import Callable


def deduplicate_with_list(data: list[int]) -> list[int]:
    """リストを使った重複排除 - O(n^2)"""
    result: list[int] = []
    for item in data:
        if item not in result:  # O(n) の検索が毎回発生
            result.append(item)
    return result


def deduplicate_with_set(data: list[int]) -> list[int]:
    """セットを使った重複排除（挿入順保持）- O(n)"""
    seen: set[int] = set()
    result: list[int] = []
    for item in data:
        if item not in seen:  # O(1) の検索
            seen.add(item)
            result.append(item)
    return result


def deduplicate_with_dict(data: list[int]) -> list[int]:
    """dictを使った重複排除（Python 3.7+ 挿入順保持）- O(n)"""
    return list(dict.fromkeys(data))


def benchmark(func: Callable, data: list[int], label: str) -> float:
    """関数の実行時間を計測"""
    start = time.perf_counter()
    result = func(data)
    elapsed = time.perf_counter() - start
    return elapsed


if __name__ == "__main__":
    random.seed(42)

    print("重複排除ベンチマーク")
    print("=" * 60)

    for size in [100, 1_000, 10_000]:
        # データの50%が重複するように生成
        data = [random.randint(0, size // 2) for _ in range(size)]

        print(f"\nデータ件数: {size:>10,}  (ユニーク: 約 {size // 2:,})")
        print("-" * 60)

        # リスト方式
        t_list = benchmark(deduplicate_with_list, data, "list")
        print(f"  list方式:     {t_list:.6f} 秒")

        # セット方式
        t_set = benchmark(deduplicate_with_set, data, "set")
        print(f"  set方式:      {t_set:.6f} 秒")

        # dict方式
        t_dict = benchmark(deduplicate_with_dict, data, "dict")
        print(f"  dict方式:     {t_dict:.6f} 秒")

        if t_set > 0:
            print(f"  list/set比:   {t_list / t_set:.1f} 倍")
```

この例から読み取れるポイント:

1. **小規模 (100 件)**: どの方式でも差はほとんど感じられない。
2. **中規模 (1,000 件)**: リスト方式が明らかに遅くなり始める。
3. **大規模 (10,000 件)**: リスト方式は O(n^2) のため壊滅的に遅く、セット/dict 方式と数百倍の差がつく。

### 5.2 LRU キャッシュの実装比較

キャッシュは多くのシステムで重要なコンポーネントである。ここでは LRU（Least Recently Used）キャッシュを異なるアプローチで実装し、性能を比較する。

```python
"""
コード例6: LRU キャッシュの実装
ハッシュテーブル + 双方向連結リスト による O(1) 実装
"""
from collections import OrderedDict
from typing import Optional, Hashable


class LRUCache:
    """
    OrderedDict を用いた LRU キャッシュ

    OrderedDict は内部的に双方向連結リストを持ち、
    要素の順序変更を O(1) で行える。
    これにハッシュテーブルの O(1) 検索を組み合わせることで、
    get/put ともに O(1) を実現する。
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("キャパシティは正の整数を指定してください")
        self._capacity = capacity
        self._cache: OrderedDict[Hashable, object] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: Hashable) -> Optional[object]:
        """
        キーに対応する値を取得（O(1)）
        アクセスされた要素は末尾（最新）に移動する
        """
        if key in self._cache:
            self._cache.move_to_end(key)  # O(1): 最新としてマーク
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: Hashable, value: object) -> None:
        """
        キーと値を格納（O(1)）
        容量超過時は最も古い要素を削除する
        """
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._capacity:
            self._cache.popitem(last=False)  # O(1): 最古の要素を削除

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return (f"LRUCache(capacity={self._capacity}, size={len(self)}, "
                f"hit_rate={self.hit_rate:.2%})")


# 使用例
cache = LRUCache(capacity=3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))   # 1（"a" が最新に）
cache.put("d", 4)       # 容量超過 → "b" が削除される
print(cache.get("b"))   # None（削除済み）
print(cache.get("c"))   # 3
print(cache)            # LRUCache(capacity=3, size=3, hit_rate=66.67%)
```

### 5.3 データ構造ごとの挿入・検索ベンチマーク

以下は、主要なデータ構造に対して挿入と検索の性能を比較するベンチマークである。

```python
"""
コード例7: 主要データ構造の挿入・検索ベンチマーク
"""
import time
import random
import bisect
from collections import deque


def benchmark_insert_search(n: int) -> dict:
    """各データ構造の挿入と検索の所要時間を計測"""
    random.seed(42)
    data = [random.randint(0, n * 10) for _ in range(n)]
    queries = [random.randint(0, n * 10) for _ in range(min(n, 10_000))]

    results = {}

    # --- list ---
    start = time.perf_counter()
    lst: list[int] = []
    for item in data:
        lst.append(item)
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        _ = q in lst
    search_time = time.perf_counter() - start
    results["list"] = {"insert": insert_time, "search": search_time}

    # --- set ---
    start = time.perf_counter()
    s: set[int] = set()
    for item in data:
        s.add(item)
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        _ = q in s
    search_time = time.perf_counter() - start
    results["set"] = {"insert": insert_time, "search": search_time}

    # --- dict ---
    start = time.perf_counter()
    d: dict[int, bool] = {}
    for item in data:
        d[item] = True
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        _ = q in d
    search_time = time.perf_counter() - start
    results["dict"] = {"insert": insert_time, "search": search_time}

    # --- ソート済みリスト + bisect ---
    start = time.perf_counter()
    sorted_lst: list[int] = []
    for item in data:
        bisect.insort(sorted_lst, item)
    insert_time = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries:
        idx = bisect.bisect_left(sorted_lst, q)
        _ = idx < len(sorted_lst) and sorted_lst[idx] == q
    search_time = time.perf_counter() - start
    results["sorted_list"] = {"insert": insert_time, "search": search_time}

    return results


if __name__ == "__main__":
    print("データ構造別 挿入・検索ベンチマーク")
    print("=" * 70)

    for n in [1_000, 10_000, 100_000]:
        print(f"\n--- n = {n:,} ---")
        results = benchmark_insert_search(n)
        print(f"  {'構造':<15} {'挿入(秒)':<15} {'検索(秒)':<15}")
        print(f"  {'-'*45}")
        for name, times in results.items():
            print(f"  {name:<15} {times['insert']:<15.6f} {times['search']:<15.6f}")
```

### 5.4 ベンチマーク結果の解釈指針

ベンチマーク結果を解釈する際には、以下の点に留意する。

1. **定数項の影響**: O(1) のハッシュテーブル検索でも、ハッシュ計算のコストがある。小さな n では O(n) の線形探索のほうが速い場合がある。
2. **キャッシュ効率**: 配列ベースの構造はメモリ局所性が高く、理論的な計算量以上に高速に動作することがある。
3. **Python のオーバーヘッド**: Python はインタプリタ言語であるため、各操作に一定のオーバーヘッドがかかる。C 拡張で実装された組み込み型（`list`、`dict`、`set`）は純 Python 実装より大幅に高速。
4. **ガベージコレクション**: Python の GC が計測中に動作すると、結果にブレが生じる。`gc.disable()` で一時停止するか、複数回計測して中央値を取ることが望ましい。

---

## 6. 選択フローチャート

### 6.1 基本フローチャート

以下のフローチャートは、データ構造選択の基本的な判断プロセスを視覚化したものである。

```
データ構造選択の判断フロー（詳細版）:

  START: どのような操作が支配的か？
  │
  ├─ キーによる検索が主体 ─────────────────────┐
  │                                             │
  │  Q: 順序は必要？                             │
  │  ├── No ──→ ハッシュテーブル (dict)          │
  │  │          期待 O(1) 検索/挿入/削除         │
  │  │                                           │
  │  └── Yes ─→ Q: 範囲検索も必要？             │
  │             ├── Yes ──→ 平衡 BST / B+木     │
  │             │           O(log n) + O(k)      │
  │             └── No ───→ ソート済み配列       │
  │                         + bisect             │
  │                         O(log n) 検索        │
  │                                              │
  ├─ 存在確認が主体 ─────────────────────────────┤
  │                                              │
  │  Q: 厳密な結果が必要？                       │
  │  ├── Yes ──→ ハッシュセット (set)            │
  │  │           期待 O(1)                       │
  │  └── No ───→ Q: 大量データ＋メモリ制約？     │
  │              ├── Yes ──→ Bloom Filter        │
  │              │           空間効率的、偽陽性あり│
  │              └── No ───→ ハッシュセット (set) │
  │                                              │
  ├─ 順序付き処理が主体 ─────────────────────────┤
  │                                              │
  │  Q: どのような順序？                         │
  │  ├── LIFO ──────→ スタック (list)            │
  │  ├── FIFO ──────→ キュー (deque)             │
  │  ├── 優先度付き ─→ ヒープ (heapq)            │
  │  └── 両端操作 ──→ 両端キュー (deque)         │
  │                                              │
  ├─ ソート順の動的維持が主体 ───────────────────┤
  │                                              │
  │  Q: 挿入頻度は？                             │
  │  ├── 高頻度 ──→ 平衡 BST / SortedList       │
  │  │              O(log n) 挿入/削除           │
  │  └── 低頻度 ──→ 配列 + 定期的ソート         │
  │                  挿入 O(n)、ソート O(n log n)│
  │                                              │
  ├─ 順次アクセス / ランダムアクセスが主体 ──────┤
  │                                              │
  │  Q: サイズは固定？                           │
  │  ├── Yes ──→ 配列 / tuple                    │
  │  │           O(1) アクセス、メモリ効率良好   │
  │  └── No ───→ 動的配列 (list)                 │
  │              O(1) 末尾追加（償却）            │
  │                                              │
  └─ 関係性の表現が主体 ─────────────────────────┘
     │
     Q: グラフの密度は？
     ├── 密 ──→ 隣接行列
     │          O(1) 辺の存在確認
     └── 疎 ──→ 隣接リスト / 隣接マップ
                O(V + E) 空間
```

### 6.2 ユースケース別クイックリファレンス

```
ユースケース別 推奨データ構造クイックリファレンス:

  ┌──────────────────────────────┬─────────────────────────────┐
  │ ユースケース                 │ 推奨データ構造               │
  ├──────────────────────────────┼─────────────────────────────┤
  │ ユーザー一覧の表示           │ list（動的配列）            │
  │ ID からユーザーを検索        │ dict（ハッシュテーブル）    │
  │ ユーザー名の重複チェック     │ set（ハッシュセット）       │
  │ ランキング（Top-K）          │ heapq（二分ヒープ）        │
  │ Undo / Redo                  │ list（スタック）× 2        │
  │ タスクキュー                 │ deque または heapq          │
  │ オートコンプリート           │ Trie                        │
  │ 範囲検索（日付、金額等）     │ SortedList / 平衡 BST      │
  │ LRU キャッシュ               │ OrderedDict                 │
  │ グラフの最短経路             │ 隣接リスト + heapq          │
  │ 大量データの存在確認         │ Bloom Filter                │
  │ 設定値の管理                 │ dict                        │
  │ イベントログ                 │ deque（maxlen 指定）        │
  │ 式の解析（構文木）           │ ツリー構造                  │
  │ データベースのインデックス   │ B+木                        │
  │ ネットワークルーティング     │ Trie / 基数木               │
  │ リアルタイム中央値           │ 2 つのヒープ（最大 + 最小） │
  │ 区間の重なり判定             │ 区間木                      │
  │ 文字列の部分一致             │ 接尾辞木 / 接尾辞配列      │
  └──────────────────────────────┴─────────────────────────────┘
```

### 6.3 判断に迷った場合の黄金ルール

データ構造の選択に迷った場合は、以下のルールに従う。

**ルール1: まず配列かハッシュテーブルを検討する**

実務の 90% 以上の場面は、配列（`list`）またはハッシュテーブル（`dict` / `set`）で十分に対応できる。特殊な構造が必要になるのは、明確な性能要件がある場合に限られる。

**ルール2: 計測なき最適化は避ける**

「リストでは遅いに違いない」という直感だけでデータ構造を変更してはならない。まず計測し、ボトルネックを特定してから最適化を行う。小規模データに対する過度な最適化は、可読性を犠牲にするだけで実益がない。

**ルール3: 将来の拡張を 1 段階だけ見据える**

現在の要件だけでなく、近い将来に追加されそうな要件を 1 段階だけ先読みする。ただし、YAGNI（You Ain't Gonna Need It）原則に従い、2 段階以上先の最適化は行わない。

**ルール4: 抽象化層を設ける**

データ構造を直接公開 API に露出させず、抽象化層（クラスやインターフェース）を挟む。これにより、後からデータ構造を変更する際の影響範囲を最小化できる。

```python
# 悪い例: データ構造を直接公開
class UserService:
    def __init__(self):
        self.users: list[dict] = []  # 内部構造が外部に露出

# 良い例: 抽象化層を設ける
class UserRepository:
    def __init__(self):
        self._users_by_id: dict[int, dict] = {}  # 内部構造は非公開

    def add(self, user: dict) -> None:
        self._users_by_id[user["id"]] = user

    def find_by_id(self, user_id: int) -> dict | None:
        return self._users_by_id.get(user_id)

    def find_all(self) -> list[dict]:
        return list(self._users_by_id.values())
```

---

## 7. アンチパターン

データ構造の選択において、頻繁に見られる誤りをアンチパターンとして整理する。これらを認識することで、設計段階での失敗を防ぐことができる。

### 7.1 アンチパターン1: 何でもリスト症候群

**症状**: あらゆる場面で `list` を使い、他のデータ構造を検討しない。

**根本原因**: リストは最も親しみやすいデータ構造であるため、無意識に選択してしまう。特に入門者に多い傾向がある。

**具体例と修正**:

```python
# ============================================================
# アンチパターン: 何でもリストで処理する
# ============================================================

# --- 問題のあるコード ---
def find_duplicates_bad(items: list[str]) -> list[str]:
    """重複する要素を見つける（アンチパターン）"""
    duplicates = []
    for i, item in enumerate(items):
        if item in items[i + 1:]:       # O(n) の検索が毎回
            if item not in duplicates:   # O(n) の検索が毎回
                duplicates.append(item)
    return duplicates
    # 全体の計算量: O(n^3) ← 壊滅的に遅い


# --- 修正後のコード ---
def find_duplicates_good(items: list[str]) -> list[str]:
    """重複する要素を見つける（改善版）"""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in items:
        if item in seen:          # O(1) の検索
            duplicates.add(item)  # O(1) の挿入
        else:
            seen.add(item)        # O(1) の挿入
    return list(duplicates)
    # 全体の計算量: O(n) ← 線形時間


# --- 性能差の検証 ---
import time
import random
import string

def generate_random_strings(n: int, length: int = 5) -> list[str]:
    """ランダムな文字列リストを生成"""
    return ["".join(random.choices(string.ascii_lowercase, k=length))
            for _ in range(n)]

if __name__ == "__main__":
    random.seed(42)
    for size in [100, 1_000, 5_000]:
        data = generate_random_strings(size, length=3)

        start = time.perf_counter()
        result_bad = find_duplicates_bad(data)
        t_bad = time.perf_counter() - start

        start = time.perf_counter()
        result_good = find_duplicates_good(data)
        t_good = time.perf_counter() - start

        print(f"n={size:>5,}: "
              f"list方式={t_bad:.4f}s, set方式={t_good:.6f}s, "
              f"比率={t_bad/max(t_good, 1e-9):.0f}倍")
```

**識別のポイント**:
- `if x in some_list` が頻繁に登場する
- リスト内のリニアサーチがループの中にネストされている
- `list.index()` や `list.count()` が頻繁に呼ばれる

**修正方針**: 存在確認には `set`、キーによる検索には `dict` を使う。

### 7.2 アンチパターン2: 過剰な最適化（Premature Optimization）

**症状**: データ件数が少ないにもかかわらず、複雑なデータ構造を導入する。

**根本原因**: 計算量の理論的な優位性にのみ注目し、実際のデータ規模や可読性への影響を考慮しない。

**具体例と修正**:

```python
# ============================================================
# アンチパターン: 過剰な最適化
# ============================================================

# --- 問題のあるコード: 設定値の管理に B+木を使う ---
# 設定項目は多くても数十件程度
# B+木の実装コストと保守コストが見合わない

# class ConfigStore:
#     def __init__(self):
#         self._btree = BPlusTree(order=4)  # 数十件のデータに B+木...
#
#     def get(self, key: str) -> str:
#         return self._btree.search(key)
#
#     def set(self, key: str, value: str) -> None:
#         self._btree.insert(key, value)


# --- 修正後のコード: 単純な dict で十分 ---
class ConfigStore:
    """アプリケーション設定の管理（数十件程度を想定）"""

    def __init__(self):
        self._config: dict[str, str] = {}

    def get(self, key: str, default: str = "") -> str:
        return self._config.get(key, default)

    def set(self, key: str, value: str) -> None:
        self._config[key] = value

    def get_all(self) -> dict[str, str]:
        return dict(self._config)
```

**識別のポイント**:
- データ件数が 100 件未満なのに、自前のツリー構造やスキップリストを実装している
- 「将来的に数百万件になるかもしれない」という根拠のない仮定に基づいている
- 可読性やテストの容易さが著しく低下している

**修正方針**: YAGNI 原則に従い、現在の要件と現実的な将来の規模に基づいて選択する。必要になったタイミングで最適化すればよい。

### 7.3 アンチパターン3: list.pop(0) の多用

**症状**: `list` を FIFO キューとして使い、`pop(0)` で先頭要素を取り出す。

**根本原因**: `list.pop()` が O(1) であることから、`list.pop(0)` も O(1) だと誤解している。

**問題**: `list.pop(0)` は全要素を 1 つずつ前にシフトする必要があるため、O(n) の計算量がかかる。n 回の操作で O(n^2) になる。

```python
# --- アンチパターン ---
queue_bad: list[int] = list(range(10_000))
while queue_bad:
    item = queue_bad.pop(0)  # O(n) × n 回 = O(n^2)

# --- 修正 ---
from collections import deque
queue_good: deque[int] = deque(range(10_000))
while queue_good:
    item = queue_good.popleft()  # O(1) × n 回 = O(n)
```

### 7.4 アンチパターン4: ネストされた dict の乱用

**症状**: 複雑なデータ関係を深くネストされた辞書で表現する。

**根本原因**: クラスやデータクラスの設計を避け、場当たり的に `dict` を重ねていく。

```python
# --- アンチパターン ---
# 何の構造なのか、型が不明、typo に気づけない
user = {
    "profile": {
        "name": "Alice",
        "address": {
            "city": "Tokyo",
            "zip": "100-0001"
        }
    },
    "settings": {
        "notifications": {
            "email": True,
            "push": False
        }
    }
}
# user["profile"]["adress"]["city"]  # typo に気づけない（KeyError）

# --- 修正: データクラスを使う ---
from dataclasses import dataclass

@dataclass
class Address:
    city: str
    zip_code: str

@dataclass
class NotificationSettings:
    email: bool = True
    push: bool = False

@dataclass
class UserProfile:
    name: str
    address: Address

@dataclass
class UserSettings:
    notifications: NotificationSettings

@dataclass
class User:
    profile: UserProfile
    settings: UserSettings

# 型チェッカーが typo を検出してくれる
user_obj = User(
    profile=UserProfile(
        name="Alice",
        address=Address(city="Tokyo", zip_code="100-0001")
    ),
    settings=UserSettings(
        notifications=NotificationSettings(email=True, push=False)
    )
)
print(user_obj.profile.address.city)  # IDE の補完が効く
```

### 7.5 アンチパターン5: 不変データに可変構造を使う

**症状**: 変更されないデータに `list` や `dict` を使い、意図しない変更のリスクを残す。

**修正**: `tuple`、`frozenset`、`types.MappingProxyType` などの不変構造を使う。

```python
# --- アンチパターン ---
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# 誰かが WEEKDAYS.append("Holiday") すると全体が壊れる

# --- 修正 ---
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
# tuple は不変なので append できない → 安全
```

### 7.6 アンチパターンの一覧表

| アンチパターン | 症状 | 計算量への影響 | 修正方針 |
|---|---|---|---|
| 何でもリスト | `in list` の多用 | O(n) → O(1) | `set` / `dict` に変更 |
| 過剰な最適化 | 小データに複雑構造 | 変わらないが保守性低下 | 単純な構造に戻す |
| list.pop(0) | FIFO にリスト使用 | O(n^2) → O(n) | `deque` に変更 |
| ネスト dict 乱用 | 深い辞書のネスト | 変わらないが型安全性低下 | データクラスに変更 |
| 可変構造の誤用 | 定数に list/dict | バグの温床 | 不変構造に変更 |

---

## 8. 演習問題

### 演習1: 要件分析（基礎）

以下の各要件に対して、最適なデータ構造を選び、選択理由を O 記法を用いて説明せよ。

**問題 1-1**: 直近 100 件のログを保持し、古いものから自動的に削除したい。

<details>
<summary>解答例</summary>

**推奨**: `collections.deque(maxlen=100)`

**理由**:
- `deque` は両端の追加・削除が O(1) で行える
- `maxlen` を指定すると、容量超過時に自動的に反対端の要素が削除される
- FIFO（先入先出）のセマンティクスが自然に表現される
- `list` でも実現可能だが、先頭の削除は O(n) となるため不適切

```python
from collections import deque

log_buffer = deque(maxlen=100)
for i in range(200):
    log_buffer.append(f"log entry {i}")
# len(log_buffer) == 100
# log_buffer[0] == "log entry 100"（最古）
# log_buffer[-1] == "log entry 199"（最新）
```

</details>

**問題 1-2**: 英単語辞書（約 30 万語）から、入力された文字列に前方一致する単語をすべて取得したい。

<details>
<summary>解答例</summary>

**推奨**: Trie（トライ木）

**理由**:
- Trie は前方一致検索に特化した構造で、プレフィックスの長さ m に対して O(m) で検索ノードに到達できる
- 到達後、その部分木を走査することで全一致単語を O(k)（k は結果数）で列挙できる
- ハッシュテーブルでは全キーに対する前方一致チェックが必要となり O(n) かかる
- ソート済み配列 + bisect でも O(log n + k) で実現可能だが、Trie のほうがプレフィックス操作に特化しており効率的

**補足**: `sortedcontainers.SortedList` を使えば、Trie を自前実装せずとも `irange` メソッドで近似的な前方一致検索が可能。実務では実装コストとのバランスも考慮する。

</details>

**問題 1-3**: ストリーミングデータに対して、リアルタイムで中央値を計算したい。

<details>
<summary>解答例</summary>

**推奨**: 2 つのヒープ（最大ヒープ + 最小ヒープ）

**理由**:
- 最大ヒープ（下位半分を管理）と最小ヒープ（上位半分を管理）を組み合わせる
- データ追加: O(log n)（ヒープへの挿入とバランス調整）
- 中央値取得: O(1)（両ヒープの先頭を参照するだけ）
- ソート済み配列だと挿入が O(n)、平衡 BST でも O(log n) だが実装が複雑

```python
import heapq

class MedianFinder:
    """2 つのヒープを使ったリアルタイム中央値計算"""

    def __init__(self):
        self._max_heap: list[int] = []  # 下位半分（符号反転で最大ヒープ）
        self._min_heap: list[int] = []  # 上位半分

    def add(self, num: int) -> None:
        # まず最大ヒープに追加
        heapq.heappush(self._max_heap, -num)
        # 最大ヒープの最大値を最小ヒープに移動
        heapq.heappush(self._min_heap, -heapq.heappop(self._max_heap))
        # サイズバランス: 最大ヒープのサイズ >= 最小ヒープのサイズ
        if len(self._min_heap) > len(self._max_heap):
            heapq.heappush(self._max_heap, -heapq.heappop(self._min_heap))

    def median(self) -> float:
        if len(self._max_heap) > len(self._min_heap):
            return float(-self._max_heap[0])
        return (-self._max_heap[0] + self._min_heap[0]) / 2.0


mf = MedianFinder()
for num in [5, 2, 8, 1, 9]:
    mf.add(num)
    print(f"追加: {num}, 中央値: {mf.median()}")
# 追加: 5, 中央値: 5.0
# 追加: 2, 中央値: 3.5
# 追加: 8, 中央値: 5.0
# 追加: 1, 中央値: 3.5
# 追加: 9, 中央値: 5.0
```

</details>

**問題 1-4**: ゲームのリーダーボード（ランキング）を管理し、スコア更新とランク取得を効率的に行いたい。

<details>
<summary>解答例</summary>

**推奨**: 平衡 BST（`sortedcontainers.SortedList`）またはスキップリスト

**理由**:
- スコア更新（削除 + 挿入）: O(log n)
- ランク取得（指定スコア以上の要素数）: O(log n)
- Top-K 取得: O(K)（末尾からの走査）
- ハッシュテーブルではランク計算が O(n)、ヒープでは任意のスコア更新が困難

</details>

### 演習2: 設計問題（応用）

**問題 2-1**: SNS のタイムライン機能を設計せよ。

以下の要件を満たすデータ構造を設計し、各操作の計算量を示すこと。

- ユーザーが投稿を作成できる（テキスト + タイムスタンプ）
- ユーザーは他のユーザーをフォローできる
- タイムライン取得: フォローしているユーザーの投稿を新しい順に最大 20 件取得
- 投稿数: 1 ユーザーあたり最大数千件、全体で数百万件を想定

<details>
<summary>解答例</summary>

```python
"""
SNS タイムライン設計

データ構造の選択根拠:
- ユーザー検索: dict（O(1)）
- フォロー関係: dict[int, set]（O(1) でフォロー/アンフォロー/確認）
- 投稿保存: dict[int, list]（ユーザー別にリストで時系列順保持）
- タイムライン取得: ヒープマージ（フォロー先の最新投稿を効率的にマージ）
"""
import heapq
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Post:
    post_id: int
    user_id: int
    text: str
    created_at: datetime

    def __lt__(self, other: "Post") -> bool:
        # ヒープ用: タイムスタンプの降順（新しい順）
        return self.created_at > other.created_at


@dataclass
class SocialNetwork:
    # ユーザー ID → ユーザー名: O(1) 検索
    users: dict[int, str] = field(default_factory=dict)

    # ユーザー ID → フォロー先 ID の集合: O(1) フォロー確認
    following: dict[int, set[int]] = field(default_factory=lambda: {})

    # ユーザー ID → 投稿リスト（時系列順）
    posts: dict[int, list[Post]] = field(default_factory=lambda: {})

    _post_counter: int = 0

    def create_post(self, user_id: int, text: str) -> Post:
        """投稿作成 - O(1)"""
        self._post_counter += 1
        post = Post(
            post_id=self._post_counter,
            user_id=user_id,
            text=text,
            created_at=datetime.now()
        )
        if user_id not in self.posts:
            self.posts[user_id] = []
        self.posts[user_id].append(post)
        return post

    def follow(self, user_id: int, target_id: int) -> None:
        """フォロー - O(1)"""
        if user_id not in self.following:
            self.following[user_id] = set()
        self.following[user_id].add(target_id)

    def unfollow(self, user_id: int, target_id: int) -> None:
        """アンフォロー - O(1)"""
        if user_id in self.following:
            self.following[user_id].discard(target_id)

    def get_timeline(self, user_id: int, limit: int = 20) -> list[Post]:
        """
        タイムライン取得 - O(F * log F + limit * log F)
        F = フォロー数

        K-way マージアルゴリズム:
        各フォロー先の最新投稿をヒープに入れ、
        最新のものから limit 件取り出す
        """
        follow_ids = self.following.get(user_id, set())
        if not follow_ids:
            return []

        # 各フォロー先の最新投稿のイテレータを用意
        # (投稿, ユーザーの投稿リスト, インデックス) のヒープ
        heap: list[tuple[Post, int, int]] = []
        for fid in follow_ids:
            user_posts = self.posts.get(fid, [])
            if user_posts:
                idx = len(user_posts) - 1
                heapq.heappush(heap, (user_posts[idx], fid, idx))

        timeline: list[Post] = []
        while heap and len(timeline) < limit:
            post, fid, idx = heapq.heappop(heap)
            timeline.append(post)
            if idx > 0:
                next_idx = idx - 1
                next_post = self.posts[fid][next_idx]
                heapq.heappush(heap, (next_post, fid, next_idx))

        return timeline
```

**計算量のまとめ**:
| 操作 | 計算量 | 使用構造 |
|---|---|---|
| 投稿作成 | O(1) | list (append) |
| フォロー/アンフォロー | O(1) | set (add/discard) |
| フォロー確認 | O(1) | set (in) |
| タイムライン取得 | O(F log F + L log F) | heapq (K-way merge) |

（F = フォロー数、L = limit）

</details>

**問題 2-2**: テキストエディタのバッファ構造を設計せよ。

以下の操作を効率的にサポートすること:
- カーソル位置への文字挿入
- カーソル位置の文字削除
- カーソルの移動（前後、行頭、行末）
- Undo / Redo

<details>
<summary>解答例</summary>

**推奨構造の組み合わせ**:
- **テキストバッファ**: ギャップバッファ（Gap Buffer）
  - カーソル位置に「ギャップ」（空白領域）を設けた配列
  - カーソル周辺の挿入・削除が O(1)
  - カーソル移動時にギャップを移動: O(移動距離)
  - 連続入力は局所的なため、実用上ほぼ O(1) で動作
- **Undo/Redo**: 2 つのスタック
  - 操作スタック（Undo 用）と Redo スタック
  - コマンドパターンで操作を表現

**代替案**: Rope（ロープ）構造
- 大きなテキスト（数 MB 以上）に適した平衡二分木ベースの文字列構造
- 挿入・削除・連結が O(log n)
- Visual Studio Code の内部で採用されている構造に類似

</details>

### 演習3: 最適化問題（発展）

**問題 3-1**: 以下のコードのボトルネックを特定し、データ構造の変更によって改善せよ。

```python
def count_common_elements(list_a: list[int], list_b: list[int]) -> int:
    """2 つのリストの共通要素数を数える（改善前）"""
    count = 0
    for item in list_a:
        if item in list_b:  # ← ボトルネック: O(n) × m 回
            count += 1
    return count
```

<details>
<summary>解答例</summary>

```python
def count_common_elements_optimized(list_a: list[int],
                                     list_b: list[int]) -> int:
    """2 つのリストの共通要素数を数える（改善後）"""
    set_b = set(list_b)  # O(n) で構築
    count = 0
    for item in list_a:
        if item in set_b:  # O(1) の検索
            count += 1
    return count
    # 全体: O(m + n)

# さらに簡潔に書く場合:
def count_common_elements_pythonic(list_a: list[int],
                                    list_b: list[int]) -> int:
    """集合の積集合を使った実装"""
    return len(set(list_a) & set(list_b))
```

**改善効果**: O(m * n) → O(m + n)

</details>

**問題 3-2**: 大量のセンサーデータ（1 秒あたり 1,000 件）をリアルタイムで受信し、以下のクエリに応答するシステムを設計せよ。

- 直近 5 分間のデータの平均値
- 直近 5 分間のデータの最大値・最小値
- 指定した時間範囲内のデータ一覧

<details>
<summary>解答例</summary>

**推奨構造の組み合わせ**:

1. **リングバッファ（`deque(maxlen=300_000)`）**: 直近 5 分間（300,000 件）のデータを保持
2. **累積和 / スライディングウィンドウ**: 平均値を O(1) で計算するための累計管理
3. **単調デキュー（Monotonic Deque）**: 最大値・最小値をウィンドウ内で O(1) で取得
4. **ソート済みインデックス**: 時間範囲クエリ用（bisect ベース）

```python
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SensorReading:
    timestamp: datetime
    value: float


class SensorAggregator:
    """センサーデータのリアルタイム集計"""

    def __init__(self, window_seconds: int = 300):
        self._window = timedelta(seconds=window_seconds)
        self._data: deque[SensorReading] = deque()
        self._sum: float = 0.0
        # 単調デキュー: 最大値用（インデックスを保持）
        self._max_deque: deque[int] = deque()
        # 単調デキュー: 最小値用（インデックスを保持）
        self._min_deque: deque[int] = deque()
        self._index: int = 0

    def _evict_old(self, now: datetime) -> None:
        """ウィンドウ外のデータを除去"""
        cutoff = now - self._window
        while self._data and self._data[0].timestamp < cutoff:
            old = self._data.popleft()
            self._sum -= old.value

    def add(self, reading: SensorReading) -> None:
        """データ追加 - 償却 O(1)"""
        self._evict_old(reading.timestamp)
        self._data.append(reading)
        self._sum += reading.value
        self._index += 1

    def average(self) -> float:
        """直近ウィンドウの平均値 - O(1)"""
        if not self._data:
            return 0.0
        return self._sum / len(self._data)

    def count(self) -> int:
        """直近ウィンドウのデータ数 - O(1)"""
        return len(self._data)
```

</details>

**問題 3-3**: 以下の要件を持つインメモリデータベースのインデックス構造を設計せよ。

- レコード数: 最大 100 万件
- プライマリキー（整数）による完全一致検索: O(1)
- 名前フィールドによる前方一致検索: 効率的に
- 作成日時による範囲検索: 効率的に
- 挿入・削除: O(log n) 以下

<details>
<summary>解答例</summary>

**複合インデックス戦略**:

| フィールド | インデックス構造 | 計算量 |
|---|---|---|
| プライマリキー | `dict` (ハッシュテーブル) | 検索 O(1)、挿入 O(1) |
| 名前 | Trie | 前方一致 O(m + k) |
| 作成日時 | `SortedList` (平衡 BST) | 範囲検索 O(log n + k)、挿入 O(log n) |

各インデックスはレコードへの参照（ID）を保持し、実データはプライマリキーの `dict` に格納する。インデックスの同期は挿入・削除時に全インデックスを更新することで維持する。

**トレードオフ**: インデックスを増やすほど検索は高速化するが、挿入・削除時の更新コストが増大する。読み取り頻度と書き込み頻度の比率を考慮して決定する。

</details>

---

## 9. よくある質問（FAQ）

### FAQ 1: dict と defaultdict はどう使い分ける？

**回答**: `dict` はキーが存在しない場合に `KeyError` を発生させる。一方、`defaultdict` はキーが存在しない場合に自動的にデフォルト値を生成する。

```python
from collections import defaultdict

# --- dict の場合: 明示的な初期化が必要 ---
word_count_dict: dict[str, int] = {}
for word in ["apple", "banana", "apple", "cherry", "banana", "apple"]:
    if word not in word_count_dict:
        word_count_dict[word] = 0
    word_count_dict[word] += 1
# または dict.get() を使う
# word_count_dict[word] = word_count_dict.get(word, 0) + 1

# --- defaultdict の場合: 初期化不要 ---
word_count_dd: defaultdict[str, int] = defaultdict(int)
for word in ["apple", "banana", "apple", "cherry", "banana", "apple"]:
    word_count_dd[word] += 1  # キーが存在しなければ int() = 0 が自動生成

# --- Counter の場合: さらに簡潔 ---
from collections import Counter
word_count_counter = Counter(["apple", "banana", "apple", "cherry",
                               "banana", "apple"])
print(word_count_counter.most_common(2))  # [('apple', 3), ('banana', 2)]
```

**使い分けの指針**:
- 単純なカウント → `Counter`
- グルーピング（キー → リスト）→ `defaultdict(list)`
- キー不在時にエラーにしたい → `dict`
- キー不在時にデフォルト値が欲しい → `defaultdict` または `dict.setdefault()`

### FAQ 2: list と tuple はどう使い分ける？

**回答**: 意味的な違いと技術的な違いの両面がある。

**意味的な違い**:
- `list`: 同種の要素の可変長コレクション（例: ユーザーの一覧）
- `tuple`: 異種の要素の固定長レコード（例: (x座標, y座標)、(名前, 年齢)）

**技術的な違い**:
- `tuple` は不変（immutable）であり、ハッシュ可能。`dict` のキーや `set` の要素に使える
- `tuple` はわずかにメモリ効率が良い（`list` はサイズ変更用のバッファを持つため）
- `tuple` の生成はわずかに高速（`list` は内部配列のアロケーションが必要）

```python
import sys

# メモリ比較
lst = [1, 2, 3, 4, 5]
tpl = (1, 2, 3, 4, 5)
print(f"list: {sys.getsizeof(lst)} bytes")  # list: 104 bytes (目安)
print(f"tuple: {sys.getsizeof(tpl)} bytes")  # tuple: 80 bytes (目安)
```

**指針**: データが変更されないなら `tuple`、変更される可能性があるなら `list`。

### FAQ 3: set の要素の順序は保証されるか？

**回答**: **保証されない**。`set` は内部的にハッシュテーブルで実装されており、要素の格納順序はハッシュ値に依存する。イテレーション時の順序は実装依存であり、Python のバージョンや実行環境によって異なる可能性がある。

```python
# 順序が保証されない例
s = {3, 1, 4, 1, 5, 9, 2, 6}
print(s)  # {1, 2, 3, 4, 5, 6, 9} ← この順序は保証されない

# 順序が必要な場合の選択肢:
# 1. ソート済みリストに変換
sorted_list = sorted(s)  # [1, 2, 3, 4, 5, 6, 9]

# 2. 挿入順を保持したい場合は dict.fromkeys()
ordered_unique = list(dict.fromkeys([3, 1, 4, 1, 5, 9, 2, 6]))
print(ordered_unique)  # [3, 1, 4, 5, 9, 2, 6]（挿入順を保持）
```

注意: Python 3.7+ の `dict` は挿入順序が保証されているが、これは `dict` の仕様であり、`set` には適用されない。

### FAQ 4: heapq はなぜ最小ヒープなのか？最大ヒープが必要な場合は？

**回答**: Python の `heapq` は最小ヒープ（min-heap）のみを提供する。最大ヒープが必要な場合は、値の符号を反転させるのが標準的な手法である。

```python
import heapq

# --- 最小ヒープ（そのまま） ---
min_heap: list[int] = []
for val in [5, 3, 8, 1, 9]:
    heapq.heappush(min_heap, val)
print(heapq.heappop(min_heap))  # 1（最小値）

# --- 最大ヒープ（符号反転） ---
max_heap: list[int] = []
for val in [5, 3, 8, 1, 9]:
    heapq.heappush(max_heap, -val)  # 符号反転して格納
print(-heapq.heappop(max_heap))  # 9（最大値）

# --- Top-K（最大の K 個を取得）---
data = [5, 3, 8, 1, 9, 2, 7, 4, 6]
top_3 = heapq.nlargest(3, data)    # [9, 8, 7]
bottom_3 = heapq.nsmallest(3, data)  # [1, 2, 3]
```

### FAQ 5: numpy の配列と Python のリストはどう使い分ける？

**回答**: 数値計算が主体なら `numpy.ndarray`、汎用データの管理なら `list` を使う。

| 観点 | `list` | `numpy.ndarray` |
|---|---|---|
| 要素の型 | 混在可能 | 同一型のみ |
| メモリ効率 | 低い（オブジェクトへのポインタ配列） | 高い（連続した値の配列） |
| 数値演算 | 遅い（Python ループ） | 高速（C 実装のベクトル演算） |
| 柔軟性 | 高い（append、extend 等） | 低い（サイズ変更はコストが高い） |
| 用途 | 汎用データ管理 | 数値計算、科学計算、機械学習 |

10 万件以上の数値データに対して一括演算を行う場合、`numpy` は `list` の数十〜数百倍高速になることがある。

### FAQ 6: データベースのインデックスと言語のデータ構造の関係は？

**回答**: リレーショナルデータベースのインデックスは、データ構造の応用である。

| データベース機能 | 内部で使われるデータ構造 |
|---|---|
| B-Tree インデックス | B+木 |
| ハッシュインデックス | ハッシュテーブル |
| 全文検索インデックス | 転置インデックス（ハッシュテーブル + ソート済みリスト） |
| 空間インデックス（GiST） | R 木 |
| カバリングインデックス | B+木（葉にデータを含む） |

データ構造の理論を理解することで、データベースのインデックス設計やクエリ最適化の背景を深く理解できるようになる。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 10. まとめ

### 10.1 選択の基本原則

本章で扱った内容を、選択の基本原則として整理する。

**原則1: 操作の頻度を基準にする**

最も頻繁に行う操作が最も高速になるデータ構造を選ぶ。全ての操作を同時に最適化することはできないため、トレードオフを受け入れる。

**原則2: まず単純な構造から始める**

実務の大半は `list`、`dict`、`set` の 3 つで解決できる。特殊なデータ構造は、計測によってボトルネックが確認された後に検討する。

**原則3: データの規模を見積もる**

n が 100 未満なら、どの構造を選んでも体感差はない。n が 10,000 を超えるあたりから、計算量のオーダーが効いてくる。n が 100 万を超えると、O(n) と O(log n) の差は致命的になる。

**原則4: 抽象化で変更に備える**

内部のデータ構造を直接外部に公開せず、インターフェースを通じてアクセスする設計にする。これにより、後からデータ構造を差し替える際のコストを最小化できる。

**原則5: 読みやすさを犠牲にしない**

コードは書く時間より読む時間のほうが長い。データ構造の選択が原因でコードの意図が不明瞭になるなら、多少の性能を犠牲にしてでも可読性を優先する。

### 10.2 要件別推奨の最終一覧

| 要件 | 推奨データ構造 | 主要操作の計算量 |
|---|---|---|
| 順次アクセス・末尾追加 | `list` | アクセス O(1)、末尾追加 O(1)* |
| キーによる高速検索 | `dict` | 検索 O(1)、挿入 O(1) |
| 重複排除・存在確認 | `set` | 検索 O(1)、挿入 O(1) |
| ソート順の動的維持 | `SortedList` / 平衡 BST | 挿入 O(log n)、検索 O(log n) |
| LIFO（Undo/Redo 等） | `list`（スタック） | push O(1)、pop O(1) |
| FIFO（タスクキュー等） | `deque` | enqueue O(1)、dequeue O(1) |
| 優先度付き処理 | `heapq` | 挿入 O(log n)、最小取得 O(1) |
| 前方一致検索 | Trie | 検索 O(m)（m = キー長） |
| 範囲検索 | ソート済み配列 / 平衡 BST | O(log n + k) |
| キャッシュ（LRU） | `OrderedDict` | get/put O(1) |
| 大量データの存在確認 | Bloom Filter | 検索 O(k)（k = ハッシュ数） |
| グラフ（疎） | `dict[int, set[int]]` | 辺の追加 O(1)、隣接 O(1) |
| グラフ（密） | 二次元配列 | 辺の確認 O(1) |
| リアルタイム中央値 | 2 つのヒープ | 追加 O(log n)、取得 O(1) |

### 10.3 学習のロードマップ

データ構造の選択力を向上させるためのロードマップを以下に示す。

```
データ構造 選択力向上のロードマップ:

  Level 1: 基礎（必須）
  ┌─────────────────────────────────────────────────┐
  │ ・list, dict, set の特性と計算量を完全に理解     │
  │ ・tuple, deque, heapq の使い分け                 │
  │ ・O 記法の直感的な理解（n=10万でどの程度か）     │
  └─────────────────────────────────────────────────┘
            │
            ▼
  Level 2: 応用（実務で頻出）
  ┌─────────────────────────────────────────────────┐
  │ ・sortedcontainers での範囲検索                   │
  │ ・複合インデックスの設計                          │
  │ ・LRU キャッシュの実装と活用                      │
  │ ・2 つのヒープによる中央値計算                    │
  │ ・グラフの表現方法の使い分け                      │
  └─────────────────────────────────────────────────┘
            │
            ▼
  Level 3: 発展（専門的な場面）
  ┌─────────────────────────────────────────────────┐
  │ ・Trie、接尾辞木、接尾辞配列                     │
  │ ・Bloom Filter、Count-Min Sketch                 │
  │ ・永続データ構造（Persistent Data Structures）    │
  │ ・並行データ構造（Lock-Free, Wait-Free）          │
  │ ・外部記憶アルゴリズム（B+木、LSM 木）           │
  └─────────────────────────────────────────────────┘
```

---

## 次に読むべきガイド

---

## 参考文献

1. Skiena, S. S. *The Algorithm Design Manual*, 3rd Edition, Springer, 2020.
   - 第 3 章「Data Structures」: データ構造の選択に関する実践的なガイドライン。第 12 章「Data Structures」のカタログでは、各構造のトレードオフが詳細に整理されている。

2. Kleppmann, M. *Designing Data-Intensive Applications*, O'Reilly Media, 2017.
   - 第 3 章「Storage and Retrieval」: B 木、LSM 木、ハッシュインデックスなど、データベース内部で使われるデータ構造の選択基準を解説。

3. Cormen, T. H., Leiserson, C. E., Rivest, R. L., and Stein, C. *Introduction to Algorithms*, 4th Edition, MIT Press, 2022.
   - 通称 CLRS。各データ構造の理論的な計算量分析の基礎を網羅。特に第 III 部「Data Structures」が本章の理論的背景。

4. Wirth, N. *Algorithms + Data Structures = Programs*, Prentice-Hall, 1976.
   - データ構造とアルゴリズムの不可分性を示した古典的名著。プログラムの設計においてデータ構造の選択がいかに重要かを論じている。

5. Knuth, D. E. *The Art of Computer Programming, Volume 3: Sorting and Searching*, 2nd Edition, Addison-Wesley, 1998.
   - 検索アルゴリズムとデータ構造の理論的基盤。ハッシュ法、木構造、ソートの詳細な分析。

6. Python 公式ドキュメント「Data Structures」
   - https://docs.python.org/3/tutorial/datastructures.html
   - Python 組み込みデータ構造の公式リファレンス。`list`、`dict`、`set`、`tuple` の使い方と性能特性。

7. Python 公式ドキュメント「collections --- Container datatypes」
   - https://docs.python.org/3/library/collections.html
   - `deque`、`defaultdict`、`Counter`、`OrderedDict` などの追加データ構造の公式リファレンス。

8. Python Wiki「TimeComplexity」
   - https://wiki.python.org/moin/TimeComplexity
   - Python 組み込み型の各操作の計算量を網羅した公式リファレンス。データ構造選択時の計算量確認に不可欠。
