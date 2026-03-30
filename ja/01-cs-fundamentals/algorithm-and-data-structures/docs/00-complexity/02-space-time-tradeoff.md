# 時間空間トレードオフ — メモ化・テーブル・キャッシュ・ブルームフィルタ

> メモリを追加で使用することで計算時間を削減する（または逆に空間を節約して時間を犠牲にする）手法を体系的に学ぶ。
> アルゴリズム設計における最も基本的な判断軸の一つであり、あらゆるソフトウェアの性能チューニングに直結する知識である。

---

## この章で学ぶこと

1. **トレードオフの理論的基盤** — なぜ時間と空間は交換可能なのか
2. **メモ化（Memoization）** — 再帰アルゴリズムの高速化とそのコスト分析
3. **キャッシュ戦略** — LRU、LFU、TTL によるメモリ管理と実践パターン
4. **ルックアップテーブル** — 事前計算による O(1) 参照の設計手法
5. **ブルームフィルタ** — 確率的データ構造による空間効率の劇的改善
6. **実践的な判断基準** — どの手法をいつ選ぶべきかの意思決定フレームワーク

### 前提知識

- 基本的な計算量表記（O, Theta, Omega）を理解していること
- Python の基本構文（再帰、辞書、リスト内包表記）を読めること
- ハッシュ関数の概念を知っていること（詳細は不要）

---

## 1. トレードオフの理論的基盤

### 1.1 なぜ時間と空間は交換可能なのか

計算の本質は「状態の変換」である。ある計算結果を再利用する場面を考えよう。選択肢は2つある。

1. **毎回計算し直す** — 空間は不要だが、計算に時間がかかる
2. **計算結果を保存して再利用する** — 時間は短縮されるが、保存のための空間が必要になる

この関係は数学的に表現できる。関数 f(x) を n 回評価する場面で、1 回の計算に時間 T がかかるとする。

- **保存なし**: 時間 = O(n * T)、空間 = O(1)
- **全保存**: 時間 = O(T + n)（初回計算 + n 回の参照）、空間 = O(結果のサイズ)

この2つの極端な選択肢の間に、無数の中間点が存在する。これが「トレードオフ」の意味である。

### 1.2 トレードオフの全体像

```
時間空間トレードオフの全体像:

  時間 ▲
  (計算量) │
       │  ● 素朴な再計算
       │     (時間大・空間小)
       │     例: 毎回フィボナッチを再帰計算
       │
       │     ● 部分キャッシュ
       │        (LRU で頻出のみ保存)
       │
       │        ● メモ化
       │           (計算済みを全保存)
       │
       │           ● ルックアップテーブル
       │              (事前に全パターン計算)
       │                 時間小・空間大
       ┼──────────────────────────────────► 空間
                                          (メモリ使用量)

  「パレートフロンティア」上のどの点を選ぶかが設計判断
```

### 1.3 Cobham の定理との関係

計算量理論において、**PSPACE** は多項式空間で解ける問題のクラスであり、**P** は多項式時間で解ける問題のクラスである。P ⊆ PSPACE が成り立つが、これは「時間で効率的に解ける問題は空間でも効率的に解ける」ことを意味する。逆は未解決（PSPACE = P かどうかは不明）であり、空間効率と時間効率は必ずしも等価ではない。

この理論的背景を踏まえると、実務では以下の指針が得られる。

| 状況 | 推奨方向 | 理由 |
|------|---------|------|
| メモリ潤沢・レイテンシ重要 | 時間優先（空間を消費） | ユーザー体験に直結するため |
| 組み込み・IoT | 空間優先（時間を犠牲） | 物理メモリの制約が厳しいため |
| バッチ処理 | 状況依存 | スループットとコストのバランスによる |
| クラウド環境 | コスト最適化 | メモリ課金と計算時間課金の比較による |

---

## 2. メモ化（Memoization）

### 2.1 メモ化の原理

メモ化とは、関数の計算結果を引数をキーとして保存し、同じ引数での再呼び出し時に保存済みの結果を返す手法である。この手法が有効になるための条件が2つある。

1. **重複部分問題の存在**: 同じ引数で関数が複数回呼ばれること
2. **参照透過性**: 同じ引数に対して常に同じ結果を返すこと（副作用がないこと）

条件 1 がなければキャッシュヒットが発生せず空間の無駄になり、条件 2 がなければキャッシュした値が不正になる。

### 2.2 素朴なフィボナッチ vs メモ化

以下は完全に動作するコードで、メモ化の効果を計測できる。

```python
"""
フィボナッチ数列: 素朴な再帰 vs メモ化 の比較
- 素朴版: 時間 O(2^n), 空間 O(n)（コールスタック）
- メモ化版: 時間 O(n), 空間 O(n)（キャッシュ + コールスタック）
"""

import time
import sys


def fib_naive(n: int) -> int:
    """素朴な再帰によるフィボナッチ数列。

    なぜ O(2^n) なのか:
    fib(n) は fib(n-1) と fib(n-2) を呼ぶ。
    この再帰木の高さは n であり、各レベルで分岐が最大2つ。
    よって呼び出し回数は最悪 2^n に近い。
    実際には fib(n) の呼び出し回数は fib(n+1) - 1 回であり、
    これは黄金比 φ = (1+√5)/2 ≈ 1.618 を用いて φ^n に比例する。
    """
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


def fib_memo(n: int, memo: dict = None) -> int:
    """メモ化によるフィボナッチ数列。

    なぜ O(n) なのか:
    各 fib(k) (k=0..n) は最大1回だけ「実際に計算」される。
    2回目以降は memo から O(1) で取得する。
    よって計算回数は n+1 回、各回 O(1) なので全体 O(n)。

    注意: デフォルト引数に mutable オブジェクトを使うと
    関数呼び出し間で共有される。ここでは意図的に None を
    デフォルトにし、初回呼び出しで辞書を生成している。
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]


def fib_bottom_up(n: int) -> int:
    """ボトムアップ DP によるフィボナッチ数列。

    なぜこの方法が存在するのか:
    メモ化（トップダウン）は再帰呼び出しを使うため、
    Python のデフォルト再帰上限（通常 1000）に引っかかる。
    ボトムアップはループで計算するため再帰上限の制約がない。

    時間 O(n), 空間 O(1)（直前の2値のみ保持）
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


def benchmark_fibonacci():
    """3手法の性能を比較する。"""
    print("=" * 60)
    print("フィボナッチ数列: 手法別ベンチマーク")
    print("=" * 60)

    # 素朴版は n=35 程度が限界（それ以上は待ち時間が長すぎる）
    for n in [10, 20, 30, 35]:
        start = time.perf_counter()
        result = fib_naive(n)
        elapsed = time.perf_counter() - start
        print(f"素朴版     fib({n:2d}) = {result:>15,d}  "
              f"時間: {elapsed:.6f}秒")

    print("-" * 60)

    # メモ化版とボトムアップ版は大きな n でも高速
    sys.setrecursionlimit(10000)
    for n in [10, 100, 1000, 5000]:
        memo = {}
        start = time.perf_counter()
        result_memo = fib_memo(n, memo)
        elapsed_memo = time.perf_counter() - start

        start = time.perf_counter()
        result_bu = fib_bottom_up(n)
        elapsed_bu = time.perf_counter() - start

        assert result_memo == result_bu, "結果が一致しません"
        digits = len(str(result_memo))
        print(f"メモ化版   fib({n:4d}): {digits:>5d}桁  "
              f"時間: {elapsed_memo:.6f}秒  "
              f"キャッシュサイズ: {len(memo)}")
        print(f"ボトムアップ fib({n:4d}): {digits:>5d}桁  "
              f"時間: {elapsed_bu:.6f}秒  "
              f"追加空間: O(1)")


if __name__ == "__main__":
    benchmark_fibonacci()
```

### 2.3 呼び出しの削減を図で理解

```
素朴なフィボナッチ fib(5) の再帰木 — 重複計算が多い:

               fib(5)
              /      \
          fib(4)      fib(3)        ← fib(3) が 2 回出現
          /    \       /   \
      fib(3)  fib(2) fib(2) fib(1)  ← fib(2) が 3 回出現
      /  \    /  \    /  \
   f(2) f(1) f(1) f(0) f(1) f(0)   ← fib(1) が 5 回出現
   / \
 f(1) f(0)

呼び出し回数: 15 回（n=5 で既に多い）
n=40 では約 3.3 億回に達する

───────────────────────────────────────────

メモ化版 fib(5) — 各値は1回だけ「真の計算」:

  呼び出し順序:
  fib(5) → fib(4) → fib(3) → fib(2) → fib(1): return 1  ← 計算
                                       fib(0): return 0  ← 計算
                              fib(2) = 1       ← 計算(1+0)
                     fib(3):  fib(1) → キャッシュヒット
                     fib(3) = 2                ← 計算(1+1)
            fib(4):  fib(2) → キャッシュヒット
            fib(4) = 3                         ← 計算(2+1)
   fib(5):  fib(3) → キャッシュヒット
   fib(5) = 5                                  ← 計算(3+2)

  実際の計算: 6 回（fib(0)..fib(5) 各1回）
  キャッシュ参照: 3 回

  memo の状態変化:
  {} → {1:1} → {1:1, 0:0} → {1:1, 0:0, 2:1}
     → {..., 3:2} → {..., 4:3} → {..., 5:5}
```

### 2.4 Python デコレータによるメモ化

`functools.lru_cache` は Python 標準ライブラリが提供するメモ化デコレータである。内部的には二重連結リストとハッシュマップを組み合わせた LRU（Least Recently Used）キャッシュを実装している。

```python
"""
functools.lru_cache の活用パターンと注意点

lru_cache が内部で行っていること:
1. 引数をタプルに変換してハッシュキーを生成
2. キーがキャッシュに存在すれば、そのエントリを「最近使用」に移動して返す
3. 存在しなければ関数を実行し、結果をキャッシュに追加
4. maxsize を超えたら「最も古い」エントリを削除

なぜ maxsize=None と maxsize=128 を使い分けるのか:
- maxsize=None: 無制限キャッシュ。全結果を保存。メモリ使用量が増え続ける。
  → 部分問題数が有限で予測可能な場合（DP など）に適する
- maxsize=128（デフォルト）: 最大128エントリ。LRU で古いものを追い出す。
  → 入力の種類が膨大だがアクセスに局所性がある場合に適する
"""

from functools import lru_cache
import time


@lru_cache(maxsize=None)
def fib_cached(n: int) -> int:
    """メモ化フィボナッチ — lru_cache 版

    なぜ maxsize=None なのか:
    fib(n) の部分問題数は n+1 個で有限。
    全てキャッシュしても空間 O(n) で収まる。
    LRU の追い出しが発生すると再計算が必要になり非効率。
    """
    if n <= 1:
        return n
    return fib_cached(n - 1) + fib_cached(n - 2)


@lru_cache(maxsize=256)
def expensive_api_simulation(user_id: int, query: str) -> dict:
    """高コスト処理のキャッシュ例

    なぜ maxsize=256 なのか:
    ユーザーID × クエリの組み合わせは膨大になりうる。
    全てキャッシュするとメモリが枯渇する。
    アクセスパターンには時間的局所性があるため、
    最近のエントリのみ保持すれば十分なヒット率が得られる。
    """
    # 重い処理をシミュレート
    time.sleep(0.001)
    return {"user_id": user_id, "query": query, "result": "data"}


def demonstrate_cache_info():
    """lru_cache の統計情報を確認する。"""
    # キャッシュをクリア
    fib_cached.cache_clear()

    # 計算実行
    result = fib_cached(100)
    info = fib_cached.cache_info()

    print(f"fib(100) = {result}")
    print(f"キャッシュ情報:")
    print(f"  ヒット数:   {info.hits}")
    print(f"  ミス数:     {info.misses}")
    print(f"  最大サイズ: {info.maxsize}")
    print(f"  現在サイズ: {info.currsize}")
    print(f"  ヒット率:   {info.hits / (info.hits + info.misses) * 100:.1f}%")


if __name__ == "__main__":
    demonstrate_cache_info()
```

### 2.5 メモ化の空間コスト分析

メモ化は「時間を空間で買う」手法であるため、空間コストの分析が重要になる。

```
メモ化の空間コスト内訳:

1. キャッシュ本体
   ┌────────────────────────────────────────┐
   │  辞書 (dict) のオーバーヘッド           │
   │  - ハッシュテーブルの空きスロット        │
   │  - 各エントリのキーと値                  │
   │  - Python オブジェクトのヘッダ           │
   │                                         │
   │  例: fib(1000) のメモ化                  │
   │  - エントリ数: 1001                      │
   │  - 1エントリあたり: ~100バイト（概算）    │
   │  - 合計: ~100KB                          │
   └────────────────────────────────────────┘

2. コールスタック（再帰の場合）
   ┌────────────────────────────────────────┐
   │  再帰呼び出しのスタックフレーム          │
   │  - 最大深さ: O(n)                        │
   │  - 1フレームあたり: ~数百バイト           │
   │                                         │
   │  Python デフォルト再帰上限: 1000          │
   │  → sys.setrecursionlimit() で変更可能    │
   │  → ただし変更はスタックオーバーフローの   │
   │    リスクを伴う                          │
   └────────────────────────────────────────┘

3. ボトムアップ DP との空間比較
   ┌────────────────────────────────────────┐
   │  問題          メモ化     ボトムアップ   │
   │  ────────────  ────────   ────────────  │
   │  フィボナッチ  O(n)       O(1) ※最適化後│
   │  最長共通部分  O(m*n)     O(min(m,n))   │
   │  ナップサック  O(n*W)     O(W) ※1行DP  │
   │                                         │
   │  ※ ボトムアップでは「もう使わない」行を  │
   │    捨てることで空間を削減できる場合がある │
   └────────────────────────────────────────┘
```

### 2.6 メモ化が有効なケースと無効なケース

| 条件 | メモ化が有効 | メモ化が無効 |
|------|------------|------------|
| 重複部分問題 | あり（フィボナッチ、最短経路） | なし（二分探索、マージソート） |
| 参照透過性 | あり（純粋関数） | なし（乱数、現在時刻に依存） |
| 部分問題数 | 多項式個（n^k） | 指数個（2^n）※キャッシュ自体が爆発 |
| 引数のハッシュ | 可能（整数、文字列） | 困難（リスト、可変オブジェクト） |

---

## 3. キャッシュ戦略の体系

### 3.1 キャッシュ追い出しポリシーの比較

メモ化は「全結果を保存する」単純な戦略であるが、実務ではメモリに制約があるため、どのエントリを残しどれを追い出すかという**キャッシュ追い出しポリシー**が重要になる。

```python
"""
主要なキャッシュ追い出しポリシーの実装と比較

なぜ複数のポリシーが存在するのか:
アクセスパターンによって最適なポリシーが異なるためである。
- LRU: 時間的局所性（最近アクセスしたものを再びアクセスしやすい）に強い
- LFU: 頻度的局所性（よく使うものは今後も使いやすい）に強い
- FIFO: 実装が最も単純。局所性を仮定しない場合の基本選択
"""

from collections import OrderedDict, defaultdict
import time


class LRUCache:
    """Least Recently Used キャッシュ

    なぜ OrderedDict を使うのか:
    LRU は「最も最近使われていないエントリ」を追い出す。
    これには各エントリの「最終アクセス時刻」の管理が必要。
    OrderedDict は挿入順序を保持し、move_to_end() で
    O(1) でエントリを末尾に移動できるため、LRU に最適。

    時間計算量: get/put ともに O(1) 平均
    空間計算量: O(capacity)
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 先頭（最古）を削除

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class LFUCache:
    """Least Frequently Used キャッシュ

    なぜ LRU ではなく LFU を選ぶ場合があるのか:
    アクセスパターンに「人気アイテム」が存在する場合、
    LFU は高頻度アイテムを保持し続けるため効率が良い。
    例: CDN でのコンテンツキャッシュ（人気動画は常にキャッシュ）

    LFU の弱点:
    - 過去に人気だったが今は不要なアイテムが残り続ける
    - 新しいアイテムがキャッシュに定着しにくい（頻度が低いため）
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}            # key -> value
        self.freq = defaultdict(int)  # key -> 使用頻度
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            self.freq[key] += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.cache:
            self.cache[key] = value
            self.freq[key] += 1
            return
        if len(self.cache) >= self.capacity:
            # 最低頻度のキーを見つけて削除
            min_freq_key = min(self.freq, key=lambda k: self.freq[k])
            del self.cache[min_freq_key]
            del self.freq[min_freq_key]
        self.cache[key] = value
        self.freq[key] = 1

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class TTLCache:
    """Time-To-Live キャッシュ

    なぜ TTL が必要なのか:
    外部データ（API レスポンス、DB クエリ結果）は時間とともに
    変化する可能性がある。TTL を設定することで、古くなった
    キャッシュエントリを自動的に無効化し、データの鮮度を保証する。

    LRU/LFU との違い:
    - LRU/LFU: 容量制約に基づく追い出し
    - TTL: 時間制約に基づく無効化
    - 実務では LRU + TTL を組み合わせることが多い
    """

    def __init__(self, capacity: int, ttl_seconds: float):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()  # key -> (value, timestamp)
        self.hits = 0
        self.misses = 0

    def _is_expired(self, key) -> bool:
        if key not in self.cache:
            return True
        _, timestamp = self.cache[key]
        return (time.time() - timestamp) > self.ttl

    def get(self, key):
        if key in self.cache and not self._is_expired(key):
            self.hits += 1
            value, _ = self.cache[key]
            # タイムスタンプを更新
            self.cache[key] = (value, time.time())
            self.cache.move_to_end(key)
            return value
        # 期限切れエントリがあれば削除
        if key in self.cache:
            del self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        self.cache[key] = (value, time.time())
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


def compare_cache_policies():
    """異なるキャッシュポリシーの比較デモ。

    シナリオ: キャッシュ容量5、アクセスパターンに偏りあり
    """
    import random
    random.seed(42)

    # Zipf 分布的なアクセスパターン: 少数のキーに集中
    # キー 0-2 が高頻度、キー 3-19 が低頻度
    access_pattern = []
    for _ in range(1000):
        if random.random() < 0.7:
            access_pattern.append(random.randint(0, 2))   # 70%: 人気キー
        else:
            access_pattern.append(random.randint(3, 19))   # 30%: その他

    lru = LRUCache(capacity=5)
    lfu = LFUCache(capacity=5)

    for key in access_pattern:
        # get して miss なら put
        for cache in [lru, lfu]:
            if cache.get(key) is None:
                cache.put(key, f"value_{key}")

    print("キャッシュポリシー比較 (容量=5, アクセス=1000回)")
    print(f"  LRU ヒット率: {lru.hit_rate():.1%}")
    print(f"  LFU ヒット率: {lfu.hit_rate():.1%}")


if __name__ == "__main__":
    compare_cache_policies()
```

### 3.2 キャッシュポリシーの選択基準

| ポリシー | 得意パターン | 苦手パターン | 実装の複雑さ |
|---------|------------|------------|------------|
| LRU | 時間的局所性が強い場合 | スキャン汚染（大量の1回限りアクセス） | O(1) 操作、中程度 |
| LFU | 人気アイテムが固定的 | 人気の移り変わりが激しい | O(1) は工夫が必要、やや高 |
| FIFO | 局所性が弱い・予測不能 | 局所性がある場合（LRU に劣る） | 最も単純 |
| TTL | 鮮度が重要なデータ | 鮮度不要で容量が問題の場合 | 時刻管理が必要、中程度 |
| Random | 計算コストを最小化したい | 局所性がある場合 | 最も単純 |

---

## 4. ルックアップテーブル

### 4.1 事前計算の原理

ルックアップテーブルは「入力空間が有限かつ小さい」場合に、全ての入力に対する出力を事前に計算してテーブルに格納する手法である。クエリ時は計算を一切行わず、テーブル参照のみで O(1) で結果を得る。

なぜこの手法が強力なのか: 元の関数の計算量が O(f(n)) であっても、テーブル参照は常に O(1) である。つまり、事前計算のコストを1回だけ支払えば、以降は無制限に O(1) クエリが可能になる。

### 4.2 popcount（ビットカウント）テーブル

```python
"""
popcount（整数のビット中の 1 の数）を3つの手法で実装し比較する。

なぜ popcount が重要なのか:
- ハミング距離の計算（誤り訂正符号）
- 集合演算のビットベクトル実装
- チェスエンジンのビットボード評価
- ブルームフィルタの実装
"""

import time


def popcount_naive(n: int) -> int:
    """ビットを1つずつ調べる素朴な方法。

    時間: O(log n) — n のビット数に比例
    空間: O(1)

    なぜ n & 1 でビットが分かるのか:
    n & 1 は n の最下位ビットを取り出す演算。
    n >>= 1 で右シフトすると次のビットが最下位に来る。
    これを n が 0 になるまで繰り返す。
    """
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count


def popcount_kernighan(n: int) -> int:
    """Brian Kernighan のアルゴリズム。

    時間: O(セットビット数) — 1 のビット数に比例
    空間: O(1)

    なぜ n & (n-1) で最下位の 1 ビットが消えるのか:
    n = ...10...0 のとき（最下位の1ビットとその右の0）
    n-1 = ...01...1 となる（その1ビットが0になり右が全て1）
    よって n & (n-1) は最下位の1ビットだけが消える。

    例: n = 12 = 1100
         n-1 = 11 = 1011
         n & (n-1) = 1000 = 8  → 最下位の 1(位置2) が消えた
    """
    count = 0
    while n:
        n &= n - 1
        count += 1
    return count


# 8ビットルックアップテーブルの構築
# なぜ 8 ビットなのか:
# - 256 エントリで全パターンを網羅でき、メモリは 256 バイトのみ
# - 16 ビット（65536 エントリ）でも良いが、L1 キャッシュに収まる
#   サイズが望ましい
# - 32 ビット（約40億エントリ）は実用的でない
POPCOUNT_TABLE_8BIT = [0] * 256
for i in range(256):
    # i のビット数 = 最下位ビット + (i >> 1) のビット数
    # この漸化式で 0 から順に構築できる
    POPCOUNT_TABLE_8BIT[i] = (i & 1) + POPCOUNT_TABLE_8BIT[i >> 1]


def popcount_table(n: int) -> int:
    """8ビットテーブル参照による popcount。

    時間: O(1) — 32ビット整数なら常に4回のテーブル参照
    空間: O(256) — テーブルのサイズ

    なぜ 8 ビットずつに分割するのか:
    32 ビット整数を 4 つの 8 ビットチャンクに分割し、
    各チャンクの popcount をテーブルから O(1) で取得して合計する。
    """
    return (POPCOUNT_TABLE_8BIT[n & 0xFF] +
            POPCOUNT_TABLE_8BIT[(n >> 8) & 0xFF] +
            POPCOUNT_TABLE_8BIT[(n >> 16) & 0xFF] +
            POPCOUNT_TABLE_8BIT[(n >> 24) & 0xFF])


def benchmark_popcount():
    """3手法の性能比較。"""
    import random
    random.seed(42)
    test_values = [random.randint(0, 2**32 - 1) for _ in range(100000)]

    # 正しさの検証
    for v in test_values[:100]:
        assert popcount_naive(v) == popcount_kernighan(v) == popcount_table(v), \
            f"不一致: {v}"

    methods = [
        ("素朴版(ビットスキャン)", popcount_naive),
        ("Kernighan法",          popcount_kernighan),
        ("テーブル参照(8bit)",    popcount_table),
    ]

    print("popcount ベンチマーク (100,000 回)")
    print("-" * 50)
    for name, func in methods:
        start = time.perf_counter()
        for v in test_values:
            func(v)
        elapsed = time.perf_counter() - start
        print(f"  {name:25s}: {elapsed:.4f}秒")


if __name__ == "__main__":
    benchmark_popcount()
```

### 4.3 三角関数テーブル — ゲーム開発での定番パターン

```python
"""
三角関数のルックアップテーブル

なぜゲーム開発で三角関数テーブルが使われるのか:
1. math.sin/cos は内部で Taylor 展開や CORDIC を使い、
   1回あたり数十〜数百ナノ秒かかる
2. ゲームでは毎フレーム（1/60秒）に数千回の三角関数計算が必要
3. テーブル参照なら配列アクセス1回（数ナノ秒）で済む
4. 角度の精度は 1度 や 0.1度 で十分な場合が多い

トレードオフ:
- 精度: テーブルの粒度（1度 vs 0.1度 vs 0.01度）で制御
- 空間: 粒度を細かくするほどテーブルが大きくなる
- 補間: テーブル間の値は線形補間で精度を上げられる
"""

import math
import time


class TrigTable:
    """三角関数ルックアップテーブル（線形補間付き）"""

    def __init__(self, resolution: float = 1.0):
        """
        resolution: 角度の刻み幅（度）。小さいほど精度が高いが空間が増える。

        なぜコンストラクタで全計算するのか:
        テーブルは不変であり、アプリケーション起動時に1回だけ構築すれば、
        以降は全スレッドで安全に共有できる。
        """
        self.resolution = resolution
        self.size = int(360 / resolution)
        self.sin_table = [0.0] * self.size
        self.cos_table = [0.0] * self.size

        for i in range(self.size):
            angle_rad = math.radians(i * resolution)
            self.sin_table[i] = math.sin(angle_rad)
            self.cos_table[i] = math.cos(angle_rad)

        # 空間使用量の計算
        self.memory_bytes = self.size * 8 * 2  # float64 × 2テーブル

    def sin(self, degrees: float) -> float:
        """テーブル参照 + 線形補間による sin 近似。"""
        degrees = degrees % 360
        idx_f = degrees / self.resolution
        idx = int(idx_f)
        frac = idx_f - idx  # 小数部分（補間用）

        if frac < 1e-9:
            return self.sin_table[idx % self.size]

        # 線形補間: f(x) ≈ f(a) + (f(b) - f(a)) * t
        val_a = self.sin_table[idx % self.size]
        val_b = self.sin_table[(idx + 1) % self.size]
        return val_a + (val_b - val_a) * frac

    def cos(self, degrees: float) -> float:
        """テーブル参照 + 線形補間による cos 近似。"""
        degrees = degrees % 360
        idx_f = degrees / self.resolution
        idx = int(idx_f)
        frac = idx_f - idx

        if frac < 1e-9:
            return self.cos_table[idx % self.size]

        val_a = self.cos_table[idx % self.size]
        val_b = self.cos_table[(idx + 1) % self.size]
        return val_a + (val_b - val_a) * frac


def benchmark_trig():
    """精度と速度の比較。"""
    import random
    random.seed(42)

    table_1deg = TrigTable(resolution=1.0)    # 360 エントリ
    table_01deg = TrigTable(resolution=0.1)   # 3600 エントリ

    test_angles = [random.uniform(0, 360) for _ in range(100000)]

    # 精度の比較
    max_error_1deg = 0
    max_error_01deg = 0
    for angle in test_angles[:1000]:
        exact = math.sin(math.radians(angle))
        err_1 = abs(table_1deg.sin(angle) - exact)
        err_01 = abs(table_01deg.sin(angle) - exact)
        max_error_1deg = max(max_error_1deg, err_1)
        max_error_01deg = max(max_error_01deg, err_01)

    print("三角関数テーブル: 精度と速度の比較")
    print("=" * 55)
    print(f"  1度刻み:   最大誤差 = {max_error_1deg:.8f}  "
          f"空間 = {table_1deg.memory_bytes:,} bytes")
    print(f"  0.1度刻み: 最大誤差 = {max_error_01deg:.8f}  "
          f"空間 = {table_01deg.memory_bytes:,} bytes")
    print()

    # 速度の比較
    methods = [
        ("math.sin",       lambda a: math.sin(math.radians(a))),
        ("テーブル(1度)",   table_1deg.sin),
        ("テーブル(0.1度)", table_01deg.sin),
    ]
    print("速度比較 (100,000 回)")
    print("-" * 45)
    for name, func in methods:
        start = time.perf_counter()
        for a in test_angles:
            func(a)
        elapsed = time.perf_counter() - start
        print(f"  {name:20s}: {elapsed:.4f}秒")


if __name__ == "__main__":
    benchmark_trig()
```

### 4.4 テーブルサイズと精度のトレードオフ

```
テーブルサイズと精度の関係（三角関数の例）:

  最大誤差 ▲
           │
  1e-2     │  ●  10度刻み (36エントリ, 576B)
           │
  1e-3     │     ●  1度刻み (360エントリ, 5.6KB)
           │
  1e-5     │        ●  0.1度刻み (3600エントリ, 56KB)
           │
  1e-7     │           ●  0.01度刻み (36000エントリ, 562KB)
           │
  1e-9     │              ●  0.001度刻み (360000エントリ, 5.6MB)
           │
           ┼────────────────────────────────────► テーブルサイズ
                                                 (メモリ使用量)

  精度が1桁上がるごとにテーブルサイズが10倍になる。
  ほとんどのアプリケーションでは 0.1度刻み（56KB）で十分。
  56KB は現代の CPU の L1 キャッシュ（通常 32-64KB）に
  ギリギリ収まるサイズであり、高速なアクセスが期待できる。
```

---

## 5. ブルームフィルタ

### 5.1 なぜブルームフィルタが必要なのか

巨大なデータセットに対する「存在判定」は多くのシステムで必要になる。

- Web ブラウザ: URL が悪意あるサイトのリストに含まれるか（Google Safe Browsing）
- データベース: あるキーが SSTable に存在するか（LevelDB、RocksDB、Cassandra）
- ネットワーク: パケットのフィンガープリントが既知のものか

これらの場面で「完全なハッシュセット」を使うと、データ量に比例したメモリが必要になる。例えば 1000 万 URL をハッシュセットで保持すると数百 MB になる。ブルームフィルタなら同じデータを数 MB で表現でき、しかも偽陰性（本当は存在するのに「存在しない」と答える）が発生しない。

### 5.2 仕組みの詳細

```
ブルームフィルタの動作原理:

【初期状態】m=12 ビットの配列、k=3 個のハッシュ関数

  位置:  0  1  2  3  4  5  6  7  8  9  10 11
  値:   [0][0][0][0][0][0][0][0][0][0][0][0]

【"apple" を追加】
  h1("apple") = 1
  h2("apple") = 5    → 位置 1, 5, 9 を 1 にする
  h3("apple") = 9

  位置:  0  1  2  3  4  5  6  7  8  9  10 11
  値:   [0][1][0][0][0][1][0][0][0][1][0][0]

【"banana" を追加】
  h1("banana") = 3
  h2("banana") = 5   → 位置 3, 5, 11 を 1 にする
  h3("banana") = 11     (位置 5 は既に 1)

  位置:  0  1  2  3  4  5  6  7  8  9  10 11
  値:   [0][1][0][1][0][1][0][0][0][1][0][1]

【"cherry" を検索 — 真の陰性】
  h1("cherry") = 2   → 位置 2: 0 → 即座に「存在しない」
  h2("cherry") = 7       (1つでも 0 なら確実に不在)
  h3("cherry") = 9

【"date" を検索 — 偽陽性!】
  h1("date") = 1    → 位置 1: 1 ✓
  h2("date") = 3    → 位置 3: 1 ✓  (banana が設定)
  h3("date") = 9    → 位置 9: 1 ✓  (apple が設定)

  全て 1 → 「たぶん存在する」と回答
  しかし "date" は追加していない → これが偽陽性

【なぜ偽陰性は発生しないのか】
  要素 x を追加すると、h1(x), h2(x), ..., hk(x) の全位置が 1 になる。
  ビットは 0→1 にしか変化せず、1→0 にはならない（削除がないため）。
  よって、追加済みの要素を検索すると必ず全ビットが 1 である。
```

### 5.3 数学的分析 — 偽陽性率の導出

ブルームフィルタの偽陽性率は以下のパラメータで決まる。

- **m**: ビット配列のサイズ
- **n**: 追加する要素数
- **k**: ハッシュ関数の数

n 個の要素を追加した後、あるビットが 0 のままである確率は:

```
P(bit=0) = (1 - 1/m)^(k*n) ≈ e^(-kn/m)
```

偽陽性が発生するのは k 個のハッシュ位置が全て 1 の場合なので:

```
偽陽性率 ≈ (1 - e^(-kn/m))^k
```

最適なハッシュ関数の数は:

```
k_opt = (m/n) * ln(2) ≈ 0.693 * (m/n)
```

このとき偽陽性率は:

```
FPR_opt = (1/2)^k = (0.6185)^(m/n)
```

### 5.4 完全な Python 実装

```python
"""
ブルームフィルタの完全実装 — パラメータ最適化と偽陽性率の検証付き

このコードは以下を含む:
1. 基本的なブルームフィルタクラス
2. 最適パラメータの自動計算
3. 偽陽性率の理論値と想定される実測値の比較
"""

import hashlib
import math


class BloomFilter:
    """ブルームフィルタの完全実装。

    なぜ md5 を使うのか:
    ブルームフィルタのハッシュ関数に求められるのは暗号学的安全性ではなく、
    出力の均一分布性である。md5 は暗号用途では破られているが、
    128ビットの出力が均一に分布するという性質は健在であり、
    ブルームフィルタには十分である。

    実務では mmh3（MurmurHash3）や xxHash のような
    非暗号学的ハッシュ関数がより高速で推奨される。
    """

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        """
        expected_items: 想定される追加要素数
        false_positive_rate: 許容する偽陽性率（0〜1）

        なぜこのコンストラクタで m と k を自動計算するのか:
        ユーザーが最適な m と k を手動で計算するのは面倒であり、
        間違いやすい。expected_items と false_positive_rate から
        自動的に最適値を導出する方が安全で使いやすい。
        """
        # 最適なビット配列サイズ: m = -n*ln(p) / (ln2)^2
        self.size = self._optimal_size(expected_items, false_positive_rate)
        # 最適なハッシュ関数数: k = (m/n) * ln2
        self.num_hashes = self._optimal_hash_count(
            self.size, expected_items
        )
        self.bit_array = bytearray(
            (self.size + 7) // 8  # ビット→バイトに変換（切り上げ）
        )
        self.count = 0  # 追加した要素数

        # パラメータの記録
        self.expected_items = expected_items
        self.target_fpr = false_positive_rate

    @staticmethod
    def _optimal_size(n: int, p: float) -> int:
        """最適なビット配列サイズを計算する。

        導出:
        偽陽性率 p = (1 - e^(-kn/m))^k を最小化する m を求める。
        k を最適値 k = (m/n)*ln2 で置換すると:
        m = -n * ln(p) / (ln2)^2
        """
        m = -n * math.log(p) / (math.log(2) ** 2)
        return int(math.ceil(m))

    @staticmethod
    def _optimal_hash_count(m: int, n: int) -> int:
        """最適なハッシュ関数数を計算する。"""
        k = (m / n) * math.log(2)
        return max(1, int(round(k)))

    def _get_bit(self, index: int) -> bool:
        """ビット配列の指定位置の値を取得する。"""
        byte_index = index // 8
        bit_offset = index % 8
        return bool(self.bit_array[byte_index] & (1 << bit_offset))

    def _set_bit(self, index: int):
        """ビット配列の指定位置を 1 にする。"""
        byte_index = index // 8
        bit_offset = index % 8
        self.bit_array[byte_index] |= (1 << bit_offset)

    def _hashes(self, item: str) -> list:
        """要素から k 個のハッシュ値を生成する。

        なぜ double hashing を使うのか:
        k 個の独立したハッシュ関数を用意するのは面倒である。
        代わりに、2つのハッシュ値 h1, h2 から
        gi(x) = h1(x) + i*h2(x) (mod m)  (i = 0, 1, ..., k-1)
        として k 個のハッシュ値を生成する。
        Kirsch & Mitzenmacher (2006) により、この方法は
        k 個の独立ハッシュ関数と同等の偽陽性率を達成することが
        証明されている。
        """
        h = hashlib.md5(str(item).encode()).hexdigest()
        h1 = int(h[:16], 16)
        h2 = int(h[16:], 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item: str):
        """要素を追加する。"""
        for pos in self._hashes(item):
            self._set_bit(pos)
        self.count += 1

    def might_contain(self, item: str) -> bool:
        """要素の存在を確認する。

        True: 「たぶん存在する」（偽陽性の可能性あり）
        False: 「確実に存在しない」（偽陰性は発生しない）
        """
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def theoretical_fpr(self) -> float:
        """現在の状態での理論的偽陽性率。"""
        if self.count == 0:
            return 0.0
        exponent = -self.num_hashes * self.count / self.size
        return (1 - math.exp(exponent)) ** self.num_hashes

    def memory_usage_bytes(self) -> int:
        """メモリ使用量（バイト）。"""
        return len(self.bit_array)

    def info(self) -> dict:
        """フィルタの状態情報を返す。"""
        return {
            "ビット配列サイズ(m)": self.size,
            "ハッシュ関数数(k)": self.num_hashes,
            "追加済み要素数": self.count,
            "メモリ使用量": f"{self.memory_usage_bytes():,} bytes",
            "理論的偽陽性率": f"{self.theoretical_fpr():.6f}",
            "目標偽陽性率": f"{self.target_fpr:.6f}",
        }


def verify_bloom_filter():
    """ブルームフィルタの動作検証。"""
    print("=" * 60)
    print("ブルームフィルタの動作検証")
    print("=" * 60)

    # 10000 要素、偽陽性率 1% で構築
    bf = BloomFilter(expected_items=10000, false_positive_rate=0.01)

    # フィルタ情報
    for key, val in bf.info().items():
        print(f"  {key}: {val}")
    print()

    # 10000 個の要素を追加
    added = set()
    for i in range(10000):
        word = f"word_{i}"
        bf.add(word)
        added.add(word)

    # 偽陰性の確認（追加した要素は必ず True）
    false_negatives = 0
    for word in added:
        if not bf.might_contain(word):
            false_negatives += 1
    print(f"偽陰性数: {false_negatives} (理論上 0)")

    # 偽陽性率の計測
    false_positives = 0
    test_count = 100000
    for i in range(test_count):
        word = f"test_{i}"
        if word not in added and bf.might_contain(word):
            false_positives += 1

    actual_fpr = false_positives / test_count
    print(f"想定される偽陽性率: {actual_fpr:.4f} "
          f"(理論値: {bf.theoretical_fpr():.4f})")

    # ハッシュセットとの空間比較
    import sys
    hash_set_size = sys.getsizeof(added)
    bloom_size = bf.memory_usage_bytes()
    print(f"\n空間比較:")
    print(f"  ハッシュセット: {hash_set_size:>10,} bytes")
    print(f"  ブルームフィルタ: {bloom_size:>10,} bytes")
    print(f"  削減率: {(1 - bloom_size / hash_set_size) * 100:.1f}%")


if __name__ == "__main__":
    verify_bloom_filter()
```

### 5.5 ブルームフィルタのバリエーション

```
ブルームフィルタのバリエーション比較:

┌────────────────────┬──────────┬──────────┬────────────────┐
│ バリエーション      │ 削除対応 │ カウント │ 主な用途        │
├────────────────────┼──────────┼──────────┼────────────────┤
│ 標準ブルーム        │ ×        │ ×        │ 存在判定全般    │
│ Counting Bloom     │ ○        │ ○        │ 動的な集合      │
│ Cuckoo Filter      │ ○        │ ×        │ 削除が必要な場合│
│ Quotient Filter    │ ○        │ ×        │ SSD 親和性      │
│ Scalable Bloom     │ ×        │ ×        │ 要素数が不明    │
└────────────────────┴──────────┴──────────┴────────────────┘

Counting Bloom Filter のビット配列:
  標準: 各位置 1ビット → [0][1][1][0][1][0]...
  Counting: 各位置 4ビット → [0][3][2][0][1][0]...
  → カウンタをデクリメントすることで削除が可能
  → ただし空間は 4 倍になる

Cuckoo Filter:
  ブルームフィルタより空間効率が良い場合がある。
  削除をサポートし、検索も高速。
  ただし挿入時にリロケーションが必要で最悪ケースが存在する。
```

---

## 6. 代表的なトレードオフパターン集

### 6.1 ハッシュテーブル vs 線形探索 — 重複検出

```python
"""
配列内の重複要素検出: 空間 vs 時間のトレードオフ

このパターンはコーディング面接で最頻出の一つである。
「空間を O(n) 使えば時間を O(n^2) から O(n) に改善できる」
という典型的なトレードオフを示す。
"""

import time
import random


def has_duplicate_brute_force(arr: list) -> bool:
    """全ペア比較による重複検出。

    時間: O(n^2) — 全ペアを比較
    空間: O(1)   — 追加メモリなし

    なぜ O(n^2) なのか:
    外側ループが n 回、内側ループが平均 n/2 回。
    合計で n*(n-1)/2 回の比較 = O(n^2)。
    """
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                return True
    return False


def has_duplicate_sort(arr: list) -> bool:
    """ソート後に隣接要素を比較する方法。

    時間: O(n log n) — ソートの時間
    空間: O(n)       — ソート用（Timsort の場合）

    なぜソートすると隣接比較だけで済むのか:
    ソート後は同じ値が隣り合う。よって隣接ペアのみ
    チェックすれば全重複を検出できる。
    """
    sorted_arr = sorted(arr)
    for i in range(len(sorted_arr) - 1):
        if sorted_arr[i] == sorted_arr[i + 1]:
            return True
    return False


def has_duplicate_hash_set(arr: list) -> bool:
    """ハッシュセットによる重複検出。

    時間: O(n) — 各要素を1回ずつ処理
    空間: O(n) — ハッシュセットのサイズ

    なぜ O(n) なのか:
    set の in 演算子と add 演算子はハッシュテーブルにより
    平均 O(1) で動作する。n 個の要素を1回ずつ処理するので
    全体で O(n)。
    """
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False


def benchmark_duplicate_detection():
    """3手法の性能比較。"""
    random.seed(42)

    print("重複検出ベンチマーク")
    print("=" * 65)

    for n in [1000, 5000, 10000]:
        arr = list(range(n))
        arr[-1] = 0  # 最後の要素を重複させる（最悪ケース）

        results = {}
        for name, func in [
            ("全ペア比較 O(n^2)", has_duplicate_brute_force),
            ("ソート後比較 O(n log n)", has_duplicate_sort),
            ("ハッシュセット O(n)", has_duplicate_hash_set),
        ]:
            start = time.perf_counter()
            result = func(arr[:])  # コピーして渡す
            elapsed = time.perf_counter() - start
            results[name] = elapsed
            assert result is True

        print(f"\nn = {n:,}")
        for name, elapsed in results.items():
            print(f"  {name:30s}: {elapsed:.6f}秒")


if __name__ == "__main__":
    benchmark_duplicate_detection()
```

### 6.2 Two Sum 問題 — 面接定番のトレードオフ例

```python
"""
Two Sum: 配列から合計が target になるペアを見つける

この問題は LeetCode #1 であり、面接で最も頻出する問題の一つ。
3つの手法が存在し、それぞれ異なるトレードオフを示す。
"""


def two_sum_brute_force(nums: list, target: int) -> tuple:
    """全ペア探索。
    時間: O(n^2), 空間: O(1)
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return (i, j)
    return None


def two_sum_sort(nums: list, target: int) -> tuple:
    """ソート + 二分探索 / 二ポインタ。
    時間: O(n log n), 空間: O(n)（インデックスの保持）

    注意: ソートするとインデックスが変わるため、
    元のインデックスを別途保持する必要がある。
    """
    indexed = sorted(enumerate(nums), key=lambda x: x[1])
    left, right = 0, len(indexed) - 1
    while left < right:
        current_sum = indexed[left][1] + indexed[right][1]
        if current_sum == target:
            return (indexed[left][0], indexed[right][0])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None


def two_sum_hash(nums: list, target: int) -> tuple:
    """ハッシュマップ。
    時間: O(n), 空間: O(n)

    なぜ 1-pass で済むのか:
    各要素 nums[i] を見るとき、target - nums[i] が
    既にマップに登録されていればペアが見つかる。
    まだなければ nums[i] を登録して次へ。
    1回の走査で完了する。
    """
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None


def test_two_sum():
    """3手法の正当性と性能を検証。"""
    test_cases = [
        ([2, 7, 11, 15], 9, {(0, 1)}),
        ([3, 2, 4], 6, {(1, 2)}),
        ([3, 3], 6, {(0, 1)}),
    ]

    for nums, target, expected in test_cases:
        for name, func in [
            ("brute_force", two_sum_brute_force),
            ("sort",        two_sum_sort),
            ("hash",        two_sum_hash),
        ]:
            result = func(nums, target)
            assert result is not None, f"{name}: 解が見つかりません"
            pair = tuple(sorted(result))
            assert pair in expected, \
                f"{name}: {pair} は期待値 {expected} に含まれません"

    print("全テストケース通過")


if __name__ == "__main__":
    test_two_sum()
```

---

## 7. 比較表

### 表1: トレードオフ手法の総合比較

| 手法 | 時間改善 | 空間コスト | 前提条件 | 適用場面 | リスク |
|------|---------|-----------|---------|---------|-------|
| メモ化 | 指数→多項式 | O(部分問題数) | 重複部分問題・参照透過性 | DP、再帰的な計算 | スタックオーバーフロー |
| LRU キャッシュ | クエリ O(1) | O(容量) | アクセスの時間的局所性 | API 応答、DB クエリ | キャッシュ汚染 |
| ルックアップテーブル | O(f)→O(1) | O(入力空間) | 入力空間が有限かつ小さい | 三角関数、ビット操作 | L1 キャッシュ溢れ |
| ハッシュセット | O(n^2)→O(n) | O(n) | ハッシュ可能な要素 | 重複検出、存在判定 | ハッシュ衝突 |
| ブルームフィルタ | 存在判定 O(k) | O(m), m << n | 偽陽性を許容できる | 大規模な存在判定 | 偽陽性、削除不可 |
| ソート前処理 | クエリ O(log n) | O(1)〜O(n) | 静的データ | 繰り返し検索 | ソートコスト O(n log n) |
| 逆引きインデックス | 検索 O(1) | O(n) | キーの抽出が可能 | 全文検索、DB インデックス | 更新コスト |

### 表2: ブルームフィルタ vs 他のデータ構造

| 特性 | ハッシュセット | ブルームフィルタ | Cuckoo Filter | ソート済み配列 |
|------|-------------|---------------|--------------|-------------|
| 空間計算量 | O(n) | O(m), m << n | O(n) だが定数小 | O(n) |
| 追加 | O(1) 平均 | O(k) | O(1) 平均 | O(n) |
| 検索 | O(1) 平均 | O(k) | O(1) 平均 | O(log n) |
| 削除 | O(1) 平均 | 不可（標準版） | O(1) 平均 | O(n) |
| 偽陽性 | なし | あり（制御可能） | あり（制御可能） | なし |
| 偽陰性 | なし | なし | なし | なし |
| 要素の取り出し | 可能 | 不可 | フィンガープリントのみ | 可能 |
| 要素数 10^7 時の空間 | ~400MB | ~12MB (FPR=1%) | ~80MB | ~80MB |

### 表3: キャッシュ戦略の選択ガイド

| 状況 | 推奨戦略 | 理由 |
|------|---------|------|
| 部分問題数が少なく全て必要 | メモ化（maxsize=None） | 全結果を保持しても空間が小さいため |
| 入力パターンが膨大だが局所性あり | LRU キャッシュ | 最近のアクセスに偏りがあるため |
| 人気アイテムが固定的 | LFU キャッシュ | 高頻度アイテムを優先保持するため |
| データの鮮度が重要 | TTL キャッシュ | 古いデータを自動的に無効化するため |
| 入力空間が小さく固定 | ルックアップテーブル | O(1) 参照が保証されるため |
| 大規模データの存在判定 | ブルームフィルタ | 空間効率が桁違いに良いため |

---

## 8. アンチパターン

### アンチパターン1: 重複部分問題がない再帰にメモ化を適用する

```python
"""
アンチパターン: 二分探索にメモ化を適用する

なぜこれが無意味なのか:
二分探索は各再帰呼び出しで探索範囲が異なる（lo, hi の組み合わせが一意）。
つまり同じ引数で2回呼ばれることがない。
キャッシュに格納してもヒットが0回であり、空間の無駄である。

一般原則: メモ化が有効なのは「同じ部分問題が複数回出現する」場合のみ。
再帰であっても、分割統治（二分探索、マージソート、クイックソート）
のように各部分問題が1回しか呼ばれない構造にはメモ化は不要。
"""

from functools import lru_cache


# BAD: 無意味なメモ化（全呼び出しがキャッシュミス）
@lru_cache(maxsize=None)
def binary_search_bad(arr_tuple: tuple, target: int,
                      lo: int, hi: int) -> int:
    """二分探索にメモ化を適用 — 無意味な例。

    問題点:
    1. (arr_tuple, target, lo, hi) の組み合わせは全て一意
       → キャッシュヒット率 = 0%
    2. arr_tuple のハッシュ計算に O(n) かかる
       → メモ化なしより遅くなる
    3. キャッシュが O(log n) エントリ分のメモリを無駄に消費する
    """
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if arr_tuple[mid] == target:
        return mid
    elif arr_tuple[mid] < target:
        return binary_search_bad(arr_tuple, target, mid + 1, hi)
    else:
        return binary_search_bad(arr_tuple, target, lo, mid - 1)


# GOOD: メモ化なしの素直な実装
def binary_search_good(arr: list, target: int) -> int:
    """素直な二分探索 — 再帰ではなくループ版。

    なぜループ版が良いのか:
    1. 再帰オーバーヘッドがない
    2. スタック空間 O(1)（再帰版は O(log n)）
    3. 末尾再帰最適化がない Python では特に重要
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def demonstrate_useless_memoization():
    """無意味なメモ化のコスト。"""
    import time

    arr = list(range(100000))
    arr_tuple = tuple(arr)
    target = 99999

    # キャッシュクリア
    binary_search_bad.cache_clear()

    start = time.perf_counter()
    for _ in range(100):
        binary_search_bad.cache_clear()
        binary_search_bad(arr_tuple, target, 0, len(arr) - 1)
    elapsed_bad = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(100):
        binary_search_good(arr, target)
    elapsed_good = time.perf_counter() - start

    info = binary_search_bad.cache_info()
    print("無意味なメモ化の実例")
    print(f"  メモ化版:  {elapsed_bad:.4f}秒")
    print(f"  素直な版:  {elapsed_good:.4f}秒")
    print(f"  キャッシュヒット: {info.hits}回 "
          f"(ミス: {info.misses}回)")
    print(f"  → ヒット率 0% であり、メモ化は完全に無駄")


if __name__ == "__main__":
    demonstrate_useless_memoization()
```

### アンチパターン2: テーブルサイズの見積もり不足

```python
"""
アンチパターン: 入力空間を過大評価したルックアップテーブル

なぜこれが危険なのか:
テーブルサイズが大きすぎると:
1. メモリ不足（OOM）でプロセスがクラッシュする
2. テーブルが L1/L2 キャッシュに収まらず、参照が遅くなる
   → テーブル参照のはずが、毎回メインメモリアクセスになる
   → 元の計算より遅くなる逆転現象が発生しうる
3. 構築時間が長くなり、アプリケーションの起動が遅延する
"""

import sys


def demonstrate_table_size_problem():
    """テーブルサイズの見積もり失敗例。"""
    print("テーブルサイズの見積もり")
    print("=" * 50)

    sizes = [
        ("8ビット",  2**8,   "popcount 等に最適"),
        ("16ビット", 2**16,  "まだ許容範囲"),
        ("20ビット", 2**20,  "L2 キャッシュに収まる可能性"),
        ("24ビット", 2**24,  "16MB — L3 キャッシュの限界"),
        ("32ビット", 2**32,  "16GB — 一般的なPCのRAMを超える"),
    ]

    for name, size, comment in sizes:
        # int のリストの場合（Python の int は 28 バイト以上）
        memory_mb = size * 28 / (1024 * 1024)
        feasible = "OK" if memory_mb < 100 else "危険" if memory_mb < 10000 else "不可能"
        print(f"  {name:10s}: {size:>15,}エントリ  "
              f"~{memory_mb:>10,.0f}MB  [{feasible}] {comment}")

    print()
    print("対策: 分割テーブル")
    print("  32ビット値の popcount を求めるとき:")
    print("  BAD:  table[2^32] = 16GB のテーブル")
    print("  GOOD: table[2^8] = 256 エントリ × 4回参照 = 実質 O(1)")


# BAD: メモリを大量消費するテーブル
def create_bad_table():
    """絶対に実行しないこと — メモリを 16GB 以上消費する。"""
    # table = [0] * (2**32)  # ~16GB — 実行するとクラッシュの恐れ
    print("このコードは危険なので実行をスキップします")


# GOOD: 分割テーブルで同じ結果を達成
SMALL_TABLE = [0] * 256
for i in range(256):
    SMALL_TABLE[i] = (i & 1) + SMALL_TABLE[i >> 1]


def popcount_safe(n: int) -> int:
    """安全な分割テーブルによる popcount。

    なぜ 4 回のテーブル参照が 1 回の巨大テーブル参照と同等なのか:
    32ビット整数は 4 つの 8 ビットチャンクに分割できる。
    各チャンクの popcount は独立に計算でき、合計すれば全体の
    popcount になる。4 回の O(1) 参照は依然として O(1) である。
    しかもテーブルは L1 キャッシュに確実に収まるため、
    巨大テーブルよりキャッシュ効率が良い。
    """
    return (SMALL_TABLE[n & 0xFF] +
            SMALL_TABLE[(n >> 8) & 0xFF] +
            SMALL_TABLE[(n >> 16) & 0xFF] +
            SMALL_TABLE[(n >> 24) & 0xFF])


if __name__ == "__main__":
    demonstrate_table_size_problem()
```

### アンチパターン3: キャッシュの無効化を忘れる

```python
"""
アンチパターン: 変化するデータに対してキャッシュを無効化しない

なぜこれが危険なのか:
キャッシュした値が古くなると「正しくない結果」を返すようになる。
これはサイレントバグ（エラーにならず静かに間違った結果を返す）であり、
デバッグが非常に困難である。

Phil Karlton の有名な格言:
"There are only two hard things in Computer Science:
 cache invalidation and naming things."
（コンピュータサイエンスで本当に難しいのは
 キャッシュの無効化と命名の2つだけだ）
"""

from functools import lru_cache
import time


# BAD: 外部状態に依存する関数をキャッシュ
EXCHANGE_RATE = {"USD_JPY": 150.0}  # 為替レートは常に変動する


@lru_cache(maxsize=128)
def convert_usd_to_jpy_bad(amount: float) -> float:
    """BAD: 為替レートが変わってもキャッシュが古い値を返す。"""
    return amount * EXCHANGE_RATE["USD_JPY"]


# GOOD: TTL 付きキャッシュで鮮度を保証
class CurrencyConverter:
    """為替変換器 — TTL 付きキャッシュ版。

    なぜ TTL が必要なのか:
    為替レートは秒単位で変動する。しかし、
    毎回 API を呼ぶのはコストが高い（レート制限、遅延）。
    TTL を設定することで「最大 N 秒前のレート」を使い、
    API 呼び出し回数を削減しつつ、ある程度の鮮度を保証する。
    """

    def __init__(self, ttl_seconds: float = 60.0):
        self.ttl = ttl_seconds
        self._cache = {}
        self._timestamps = {}

    def get_rate(self, pair: str) -> float:
        """為替レートを取得（キャッシュ付き）。"""
        now = time.time()
        if pair in self._cache:
            age = now - self._timestamps[pair]
            if age < self.ttl:
                return self._cache[pair]

        # キャッシュミスまたは期限切れ → API 呼び出し（シミュレート）
        rate = self._fetch_rate_from_api(pair)
        self._cache[pair] = rate
        self._timestamps[pair] = now
        return rate

    def _fetch_rate_from_api(self, pair: str) -> float:
        """API からレートを取得（シミュレート）。"""
        # 実際にはここで外部 API を呼ぶ
        return EXCHANGE_RATE.get(pair, 1.0)

    def convert(self, amount: float, pair: str = "USD_JPY") -> float:
        """通貨変換。"""
        return amount * self.get_rate(pair)


def demonstrate_cache_invalidation():
    """キャッシュ無効化の重要性を示す。"""
    print("キャッシュ無効化の問題")
    print("=" * 50)

    # BAD 版
    convert_usd_to_jpy_bad.cache_clear()
    result1 = convert_usd_to_jpy_bad(100.0)
    print(f"BAD版: $100 = ¥{result1:.0f} (レート: {EXCHANGE_RATE['USD_JPY']})")

    EXCHANGE_RATE["USD_JPY"] = 140.0  # レートが変動
    result2 = convert_usd_to_jpy_bad(100.0)
    print(f"BAD版: $100 = ¥{result2:.0f} (レート変動後も古い値が返る!)")

    # GOOD 版
    converter = CurrencyConverter(ttl_seconds=0.1)  # TTL 0.1秒
    result3 = converter.convert(100.0)
    print(f"GOOD版: $100 = ¥{result3:.0f}")

    EXCHANGE_RATE["USD_JPY"] = 130.0
    time.sleep(0.15)  # TTL を超えるまで待つ
    result4 = converter.convert(100.0)
    print(f"GOOD版: $100 = ¥{result4:.0f} (TTL経過後に新レートを取得)")


if __name__ == "__main__":
    demonstrate_cache_invalidation()
```

---

## 9. エッジケース分析

### エッジケース1: メモ化と Python の再帰制限

```python
"""
エッジケース: メモ化で再帰制限に到達する

Python のデフォルト再帰制限は 1000 である。
メモ化フィボナッチで fib(1000) を直接呼ぶと
RecursionError が発生する。

なぜこれが問題なのか:
メモ化は再帰を前提とするため、深い再帰が必要な問題では
Python の再帰制限がボトルネックになる。

解決策:
1. sys.setrecursionlimit() で上限を引き上げる（推奨しない）
   → スタックオーバーフローのリスクがある
2. ボトムアップ DP に書き換える（推奨）
   → ループなので再帰制限に依存しない
3. 反復的深化: 小さい値から段階的に呼ぶ（妥協策）
   → キャッシュが温まるので深い再帰が発生しない
"""

import sys


def fib_memo_recursive(n: int, memo: dict = None) -> int:
    """メモ化フィボナッチ（再帰版）。"""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo_recursive(n - 1, memo) + \
              fib_memo_recursive(n - 2, memo)
    return memo[n]


def fib_iterative_warmup(n: int) -> int:
    """反復的にキャッシュを温めてから大きな値を計算する。

    なぜこの方法が有効なのか:
    fib(500) を先に呼ぶと、fib(0)〜fib(500) がキャッシュされる。
    その後 fib(1000) を呼ぶと、fib(501)〜fib(1000) の計算で
    fib(499) と fib(500) はキャッシュから取得される。
    再帰の深さは最大 500 となり、制限内に収まる。
    """
    memo = {}
    step = 400  # 再帰制限よりも十分小さいステップ
    for start in range(0, n + 1, step):
        target = min(start + step, n)
        fib_memo_recursive(target, memo)
    return memo.get(n, fib_memo_recursive(n, memo))


def fib_bottom_up_space_optimized(n: int) -> int:
    """ボトムアップ DP（空間最適化版）。

    なぜ空間 O(1) で済むのか:
    fib(i) の計算には fib(i-1) と fib(i-2) の2つの値だけが必要。
    それより前の値は不要であるため、2変数のみで十分。
    """
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1


def demonstrate_recursion_limit():
    """再帰制限の問題と解決策。"""
    print("再帰制限のエッジケース")
    print("=" * 50)

    # 現在の再帰制限
    print(f"現在の再帰制限: {sys.getrecursionlimit()}")

    # 小さい値は問題なし
    memo = {}
    result = fib_memo_recursive(100, memo)
    print(f"fib(100) = {len(str(result))}桁 [OK]")

    # 大きい値は RecursionError
    try:
        memo2 = {}
        fib_memo_recursive(5000, memo2)
        print("fib(5000) [OK — 再帰制限が十分大きい環境]")
    except RecursionError:
        print("fib(5000) [RecursionError発生]")

    # 反復的ウォームアップで回避
    result_warmup = fib_iterative_warmup(5000)
    print(f"fib(5000) (ウォームアップ版) = {len(str(result_warmup))}桁 [OK]")

    # ボトムアップで回避
    result_bu = fib_bottom_up_space_optimized(5000)
    print(f"fib(5000) (ボトムアップ版) = {len(str(result_bu))}桁 [OK]")

    assert result_warmup == result_bu, "結果が一致しません"


if __name__ == "__main__":
    demonstrate_recursion_limit()
```

### エッジケース2: ブルームフィルタの飽和

```python
"""
エッジケース: ブルームフィルタに想定以上の要素を追加すると偽陽性率が急増する

なぜこれが重要なのか:
ブルームフィルタのサイズは想定要素数から事前に決定される。
想定以上の要素を追加すると、ビット配列がほぼ全て 1 になり
（「飽和」状態）、偽陽性率が 100% に近づく。
これは事実上フィルタが役に立たない状態である。

対策:
1. 要素数を事前に見積もり、余裕を持ったサイズにする
2. Scalable Bloom Filter を使う（要素数に応じて自動拡張）
3. 飽和を検知したら新しいフィルタに切り替える
"""

import math
import hashlib


class MonitoredBloomFilter:
    """飽和監視付きブルームフィルタ。"""

    def __init__(self, expected_items: int, false_positive_rate: float = 0.01):
        self.expected_items = expected_items
        self.target_fpr = false_positive_rate
        m = int(-expected_items * math.log(false_positive_rate) /
                (math.log(2) ** 2))
        self.size = max(m, 64)
        self.num_hashes = max(1, int(round(
            (self.size / expected_items) * math.log(2)
        )))
        self.bit_array = bytearray((self.size + 7) // 8)
        self.count = 0
        self._set_bits = 0  # 1 になっているビット数

    def _hashes(self, item: str) -> list:
        h = hashlib.md5(str(item).encode()).hexdigest()
        h1 = int(h[:16], 16)
        h2 = int(h[16:], 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def _get_bit(self, idx: int) -> bool:
        return bool(self.bit_array[idx // 8] & (1 << (idx % 8)))

    def _set_bit(self, idx: int):
        byte_idx = idx // 8
        bit_off = idx % 8
        if not (self.bit_array[byte_idx] & (1 << bit_off)):
            self._set_bits += 1
            self.bit_array[byte_idx] |= (1 << bit_off)

    def add(self, item: str):
        for pos in self._hashes(item):
            self._set_bit(pos)
        self.count += 1

    def might_contain(self, item: str) -> bool:
        return all(self._get_bit(pos) for pos in self._hashes(item))

    def saturation(self) -> float:
        """飽和率（0.0〜1.0）を返す。"""
        return self._set_bits / self.size

    def estimated_fpr(self) -> float:
        """現在の飽和率から推定される偽陽性率。"""
        sat = self.saturation()
        return sat ** self.num_hashes

    def is_saturated(self, threshold: float = 0.5) -> bool:
        """フィルタが飽和状態かどうかを判定する。

        なぜ閾値 0.5 なのか:
        最適なハッシュ関数数のとき、各ビットが 1 になる確率は
        ちょうど 0.5 である。これを超えると急速に偽陽性率が上昇する。
        """
        return self.saturation() > threshold


def demonstrate_saturation():
    """飽和の進行と偽陽性率の上昇を示す。"""
    print("ブルームフィルタの飽和")
    print("=" * 65)
    print(f"{'追加数':>10s}  {'飽和率':>8s}  {'推定FPR':>10s}  "
          f"{'想定される実FPR':>15s}  {'状態':>8s}")
    print("-" * 65)

    bf = MonitoredBloomFilter(expected_items=1000, false_positive_rate=0.01)

    checkpoints = [100, 500, 1000, 2000, 5000, 10000]

    for target in checkpoints:
        while bf.count < target:
            bf.add(f"item_{bf.count}")

        # 偽陽性率を計測
        fp = 0
        tests = 10000
        for i in range(tests):
            probe = f"probe_{i}"
            if bf.might_contain(probe):
                fp += 1
        actual_fpr = fp / tests

        status = "正常" if not bf.is_saturated() else "飽和!"
        print(f"{bf.count:>10,d}  {bf.saturation():>8.1%}  "
              f"{bf.estimated_fpr():>10.4f}  "
              f"{actual_fpr:>15.4f}  {status:>8s}")


if __name__ == "__main__":
    demonstrate_saturation()
```

### エッジケース3: ハッシュテーブルのリサイズとメモリスパイク

```
ハッシュテーブル（set/dict）のメモリ使用量は一定ではない:

  メモリ ▲
         │
  256KB  │              ┌─────────────  ← リサイズ発生
         │              │                 (2/3 充填でテーブルサイズ2倍)
  128KB  │      ┌───────┘
         │      │
   64KB  │  ┌───┘
         │  │
   32KB  │──┘
         │
         ┼──────────────────────────────► 要素数
         0     1000   2000   3000   4000

  問題: リサイズ時に一時的に新旧2つのテーブルが存在する
  → 最大で定常状態の 3 倍のメモリを消費する瞬間がある
  → メモリ制約の厳しい環境ではこのスパイクに注意が必要

  対策:
  1. 初期サイズを予測して事前確保 (dict.fromkeys() 等)
  2. メモリ制約が厳しい場合はソート済み配列 + 二分探索を検討
  3. ブルームフィルタで代替可能な用途であれば切り替える
```

---

## 10. 実践的な設計判断フレームワーク

### 10.1 意思決定フローチャート

```
時間空間トレードオフの選択フロー:

  [開始]
    │
    ▼
  同じ計算を何回繰り返すか？
    │
    ├─ 1回だけ → トレードオフ不要（素直に計算）
    │
    ├─ 少数回（< 10） → 再計算が安い場合はそのまま
    │
    └─ 多数回（10+） ──┐
                        ▼
                  入力空間の大きさは？
                        │
                  ├─ 小さい（< 10^6）
                  │     │
                  │     ▼
                  │   全パターン事前計算可能？
                  │     ├─ はい → ルックアップテーブル
                  │     └─ いいえ → メモ化 / LRU キャッシュ
                  │
                  ├─ 中程度（10^6〜10^9）
                  │     │
                  │     ▼
                  │   アクセスに局所性があるか？
                  │     ├─ はい → LRU / LFU キャッシュ
                  │     └─ いいえ → ブルームフィルタ（存在判定のみ）
                  │
                  └─ 巨大（> 10^9）
                        │
                        ▼
                  偽陽性を許容できるか？
                        ├─ はい → ブルームフィルタ
                        └─ いいえ → 外部ストレージ（DB、Redis）
```

### 10.2 実務での判断基準

| 判断軸 | 時間優先を選ぶ条件 | 空間優先を選ぶ条件 |
|--------|------------------|------------------|
| レイテンシ要件 | p99 < 10ms など厳格 | バッチ処理で遅延許容 |
| メモリ単価 | クラウドでメモリが安い | 組み込みで RAM 制約あり |
| データの変動性 | 静的データ（参照テーブル） | 動的データ（頻繁に更新） |
| 正確性要件 | 厳密な結果が必要 | 近似・確率的でよい（ブルーム） |
| スケール | 単一マシンで完結 | 分散システムで一貫性が課題 |

---

## 11. 演習問題

### 基礎レベル

**演習 1: メモ化の効果を体感する**

以下の関数 `grid_paths(m, n)` は m x n グリッドの左上から右下までの経路数を計算する。メモ化なしの版を実行し、メモ化版を自分で実装して性能差を確認せよ。

```python
"""
演習1: グリッド経路数の計算にメモ化を適用する

問題:
m x n のグリッドにおいて、左上 (0,0) から右下 (m-1,n-1) まで
右か下にのみ移動する場合の経路数を求めよ。

ヒント:
- grid_paths(m, n) = grid_paths(m-1, n) + grid_paths(m, n-1)
- 基底条件: m == 1 または n == 1 のとき経路は 1 通り
"""

import time


def grid_paths_naive(m: int, n: int) -> int:
    """メモ化なし版。時間計算量は？ 自分で考えてみよ。"""
    if m == 1 or n == 1:
        return 1
    return grid_paths_naive(m - 1, n) + grid_paths_naive(m, n - 1)


# TODO: grid_paths_memo(m, n) をメモ化付きで実装せよ
# def grid_paths_memo(m, n, memo=None):
#     ...


# 検証用
if __name__ == "__main__":
    # 小さい入力で確認
    print(f"grid_paths(3, 3) = {grid_paths_naive(3, 3)}")  # 6
    print(f"grid_paths(4, 4) = {grid_paths_naive(4, 4)}")  # 20

    # m=15, n=15 で素朴版の遅さを体感（数秒かかるはず）
    start = time.perf_counter()
    result = grid_paths_naive(15, 15)
    elapsed = time.perf_counter() - start
    print(f"grid_paths(15, 15) = {result}  ({elapsed:.3f}秒)")
    # メモ化版なら瞬時に完了するはず
```

**演習 2: ルックアップテーブルの構築**

0〜255 の各バイト値について「ビットを反転した値」を返すテーブルを構築し、32ビット整数のビット反転関数を実装せよ。

```python
"""
演習2: ビット反転テーブル

問題:
8ビット値のビット反転テーブルを構築せよ。
例: 0b11010010 → 0b01001011  (上位と下位を入れ替え)

ヒント:
reverse_bits_8(n) は n の 8ビットを反転する。
0b10110000 → 0b00001101

全 256 パターンを事前計算してテーブルに格納する。
"""


def reverse_bits_8(n: int) -> int:
    """8ビット値のビット反転（素朴版）。"""
    result = 0
    for _ in range(8):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# TODO: テーブルを構築
# REVERSE_TABLE = [reverse_bits_8(i) for i in range(256)]

# TODO: 32ビット整数のビット反転をテーブル参照で実装
# def reverse_bits_32(n):
#     ...
```

### 応用レベル

**演習 3: LRU キャッシュの自作**

Python の `OrderedDict` を使わずに、二重連結リストとハッシュマップで LRU キャッシュを実装せよ。`get(key)` と `put(key, value)` が両方 O(1) で動作すること。

```python
"""
演習3: LRU キャッシュの自作実装

要件:
1. get(key): O(1) でキャッシュから取得。ミス時は -1 を返す。
2. put(key, value): O(1) で追加/更新。容量超過時は最古を削除。

ヒント:
- ハッシュマップ: key → ノードへの参照（O(1) アクセス）
- 二重連結リスト: アクセス順を管理（O(1) で移動・削除）
- ダミーの head と tail ノードを使うと境界条件の処理が楽になる
"""


class Node:
    """二重連結リストのノード。"""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCacheManual:
    """LRU キャッシュ — 自作版。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}  # key → Node

        # ダミーノード（番兵）
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    # TODO: _remove(node), _add_to_front(node), get(key), put(key, value)
    # を実装せよ


# テスト
if __name__ == "__main__":
    cache = LRUCacheManual(2)
    cache.put(1, 1)
    cache.put(2, 2)
    assert cache.get(1) == 1      # key=1 にアクセス → 最新に
    cache.put(3, 3)               # 容量超過 → key=2 が追い出される
    assert cache.get(2) == -1     # key=2 は追い出し済み
    cache.put(4, 4)               # 容量超過 → key=1 が追い出される
    assert cache.get(1) == -1
    assert cache.get(3) == 3
    assert cache.get(4) == 4
    print("全テスト通過")
```

### 発展レベル

**演習 4: Counting Bloom Filter の実装**

標準のブルームフィルタに「削除」機能を追加した Counting Bloom Filter を実装せよ。各ビット位置を 4 ビットカウンタにし、追加時にインクリメント、削除時にデクリメントする。

```python
"""
演習4: Counting Bloom Filter

通常のブルームフィルタは削除ができない（ビットを 0 に戻すと
他の要素の情報が失われるため）。Counting Bloom Filter は
各位置にカウンタを持つことでこの問題を解決する。

要件:
1. add(item): 各ハッシュ位置のカウンタをインクリメント
2. remove(item): 各ハッシュ位置のカウンタをデクリメント
3. might_contain(item): 全カウンタが 0 より大きいか判定
4. カウンタオーバーフロー（> 15）時の対処を考慮すること

注意:
- 追加していない要素を remove するとカウンタが不整合になる
- この問題はブルームフィルタの本質的な制約であり、
  Counting Bloom Filter でも完全には解決できない
"""


class CountingBloomFilter:
    """Counting Bloom Filter のスケルトン。"""

    def __init__(self, size: int = 1000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        # 4ビットカウンタ → 1バイトに2カウンタを格納可能
        # ここでは簡略化のためリストで実装
        self.counters = [0] * size

    # TODO: _hashes(item), add(item), remove(item),
    #       might_contain(item) を実装せよ
    # ヒント: remove で counter < 0 にならないようにすること


if __name__ == "__main__":
    cbf = CountingBloomFilter(size=10000, num_hashes=5)
    cbf.add("apple")
    cbf.add("banana")
    assert cbf.might_contain("apple") is True
    assert cbf.might_contain("banana") is True

    cbf.remove("apple")
    assert cbf.might_contain("apple") is False  # 削除成功
    assert cbf.might_contain("banana") is True   # 影響なし
    print("Counting Bloom Filter テスト通過")
```

**演習 5: 空間最適化付き DP — 最長共通部分列**

最長共通部分列（LCS）の DP テーブルを空間最適化せよ。通常は O(m*n) 空間が必要だが、O(min(m,n)) に削減可能である。

```python
"""
演習5: LCS の空間最適化

通常の LCS DP テーブル:
     ""  a  b  c  d  e
  "" [0, 0, 0, 0, 0, 0]
  a  [0, 1, 1, 1, 1, 1]
  c  [0, 1, 1, 2, 2, 2]
  e  [0, 1, 1, 2, 2, 3]

空間 O(m*n) → 文字列が長いとメモリ不足に

ヒント:
- dp[i][j] は dp[i-1][j] と dp[i][j-1] と dp[i-1][j-1] のみに依存
- 2行分だけ保持すれば十分 → 空間 O(n)
- さらに工夫すると 1行 + 1変数 で O(n) に
"""


def lcs_full_table(s1: str, s2: str) -> int:
    """通常版 LCS — 空間 O(m*n)。"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


# TODO: lcs_space_optimized(s1, s2) を空間 O(min(m,n)) で実装せよ
# def lcs_space_optimized(s1, s2):
#     ...

if __name__ == "__main__":
    assert lcs_full_table("abcde", "ace") == 3
    assert lcs_full_table("abc", "def") == 0
    # assert lcs_space_optimized("abcde", "ace") == 3
    print("LCS テスト通過")
```

---

## 12. FAQ

### Q1: メモ化（トップダウン DP）とボトムアップ DP はどちらを使うべきか？

**A:** 両方とも同じ問題を解けるが、特性が異なる。

| 特性 | トップダウン（メモ化） | ボトムアップ |
|------|---------------------|------------|
| 計算する部分問題 | 必要なもののみ | 全て |
| 実装の直感性 | 再帰で自然に書ける | テーブルの添字管理が必要 |
| 再帰オーバーヘッド | あり | なし |
| Python での再帰制限 | 影響を受ける | 影響なし |
| 空間最適化 | 困難 | 可能（不要な行を破棄） |

**推奨:** 問題を最初に理解・実装するときはトップダウン（考えやすいため）。性能が必要な場合はボトムアップに変換する。特に Python では再帰制限の関係でボトムアップが安全である。

### Q2: ブルームフィルタの偽陽性率はどう制御するのか？

**A:** 3つのパラメータ（m, n, k）の関係で制御する。

- **m（ビット配列サイズ）を増やす**: 偽陽性率が下がる。コストはメモリ増加。
- **k（ハッシュ関数数）を最適化する**: k = (m/n) * ln(2) が最適値。小さすぎても大きすぎても偽陽性率が上がる。
- **n（要素数）を事前に見積もる**: 見積もりが甘いとフィルタが飽和する。

実用的な目安:

| 偽陽性率 | 必要なビット数/要素 (m/n) | 最適 k |
|---------|-------------------------|--------|
| 10% | 4.8 | 3 |
| 1% | 9.6 | 7 |
| 0.1% | 14.4 | 10 |
| 0.01% | 19.2 | 13 |

例: 1000 万件のデータに対して偽陽性率 1% を実現するには、m = 9.6 * 10^7 ≈ 1200 万バイト（約 12MB）のメモリが必要である。同じデータをハッシュセットで持つと数百 MB になるため、約 25 倍の空間削減になる。

### Q3: 時間と空間のどちらを優先すべきか？

**A:** 一般的な指針は以下の通り。

1. **まず時間制約を確認する**: ユーザーに直接影響するレイテンシ（API レスポンスタイム、UI 応答速度）が最も重要な場合が多い。メモリは追加購入できるが、ユーザーの忍耐は購入できない。

2. **次にメモリ制約を確認する**: 組み込みシステム、モバイルデバイス、コンテナの Memory Limit、クラウドのメモリ課金など、物理的・経済的なメモリ制約がある場合は空間優先を検討する。

3. **スケーラビリティを考慮する**: データ量が増えたときにどちらの手法がスケールするかを考える。O(n) 空間の手法は n が 10 倍になればメモリも 10 倍になるが、O(1) 空間の手法はデータ量に依存しない。

4. **コスト試算を行う**: クラウド環境では、メモリの増強コストと計算時間のコスト（CPU 時間の課金）を比較して最適な点を選ぶ。

### Q4: lru_cache のデフォルト maxsize=128 は変更すべきか？

**A:** ユースケースによる。

- **DP 問題（部分問題数が有限）**: `maxsize=None` にする。全結果を保持しないと、追い出された値の再計算が発生して計算量が悪化する。
- **API キャッシュ（入力パターンが膨大）**: デフォルトの 128 または `maxsize=256〜1024` 程度。大きくしすぎるとメモリを圧迫する。
- **数値計算（引数が実数）**: 浮動小数点数はハッシュの問題で意図したキャッシュヒットが起きにくい。整数に丸めるか、テーブル参照に切り替えることを検討する。

### Q5: ブルームフィルタの代わりに Cuckoo Filter を使うべきか？

**A:** Cuckoo Filter は以下の場合に有利である。

1. **削除が必要**: ブルームフィルタは削除不可（Counting 版を除く）だが、Cuckoo Filter はネイティブに削除をサポートする。
2. **空間効率**: 偽陽性率が 3% 以下の場合、Cuckoo Filter の方がブルームフィルタより空間効率が良い。
3. **検索速度**: Cuckoo Filter は最大 2 回のメモリアクセスで判定が完了し、キャッシュフレンドリーである。

ただし、Cuckoo Filter には挿入時にリロケーション（既存要素の移動）が必要になる場合があり、最悪ケースでは挿入が失敗する。挿入が頻繁で失敗が許容できない場合はブルームフィルタの方が安全である。

---

## 13. 実世界での適用事例

### 13.1 データベースにおけるトレードオフ

```
データベースの読み書きトレードオフ:

  B-Tree インデックス:
  ┌──────────────────────────────────────────────┐
  │  空間: インデックスはデータの 10-30% の追加空間  │
  │  読み取り: O(log n) — インデックスなしの O(n) から改善│
  │  書き込み: 各 INSERT/UPDATE でインデックスも更新    │
  │                                                  │
  │  トレードオフ:                                     │
  │  インデックスが多い → 読み取り高速 / 書き込み低速    │
  │  インデックスが少ない → 読み取り低速 / 書き込み高速  │
  └──────────────────────────────────────────────┘

  LSM-Tree (LevelDB, RocksDB):
  ┌──────────────────────────────────────────────┐
  │  書き込み: メモリ上の MemTable → 高速            │
  │  読み取り: 複数の SSTable を走査 → やや低速       │
  │  ブルームフィルタ: 各 SSTable に付与               │
  │    → 「このキーは SSTable に存在しない」を高速判定 │
  │    → 不要な SSTable の読み込みをスキップ           │
  │    → 読み取り性能を大幅改善                       │
  └──────────────────────────────────────────────┘
```

### 13.2 Web ブラウザのセーフブラウジング

Google Chrome のセーフブラウジング機能では、悪意ある URL のリストをブルームフィルタとしてブラウザに保持している。ユーザーが URL にアクセスするとき、まずローカルのブルームフィルタで判定し、「たぶん危険」と判定された場合のみ Google のサーバーに問い合わせる。

- ブルームフィルタなし: 全 URL を毎回サーバーに問い合わせ → 遅延 + プライバシー問題
- ブルームフィルタあり: 99%+ のアクセスでサーバー問い合わせ不要 → 高速 + プライバシー保護

### 13.3 Redis のキャッシュパターン

```
Cache-Aside パターン（最も一般的）:

  [アプリケーション]
       │
       ├─ 1. get(key) ──→ [Redis キャッシュ]
       │                      │
       │                      ├─ ヒット → 結果を返す（高速）
       │                      │
       │                      └─ ミス
       │                           │
       ├─ 2. query(key) ──→ [データベース]
       │                      │
       │                      └─ 結果
       │
       └─ 3. set(key, result, TTL) ──→ [Redis キャッシュ]

  このパターンのトレードオフ:
  - 空間: Redis のメモリ分だけ追加コスト
  - 時間: キャッシュヒット時は DB アクセスを完全にスキップ
  - 一貫性: DB 更新後も TTL が切れるまで古いデータが返る
  - 可用性: Redis 障害時は DB に直接アクセスにフォールバック
```

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 14. まとめ

### 核心的な原則

| 原則 | 説明 |
|------|------|
| 空間と時間は交換可能 | メモリを追加で使えば計算時間を削減でき、逆も成り立つ |
| 重複部分問題の検出 | メモ化/DP が有効かどうかの最重要判断基準 |
| テーブルサイズの見積もり | 入力空間が小さい場合のみルックアップテーブルが実用的 |
| 偽陽性の許容判断 | ブルームフィルタは偽陽性を許容する代わりに空間を劇的に削減 |
| キャッシュの無効化 | キャッシュ戦略の選択以上に、無効化戦略の設計が重要 |
| パレートフロンティア | 完璧なトレードオフは存在せず、要件に応じた最適点を選ぶ |

### 各手法の使い分け早見表

| 手法 | 使う場面 | 避ける場面 |
|------|---------|-----------|
| メモ化 | 重複部分問題があるとき | 各部分問題が1回しか呼ばれないとき |
| lru_cache | Python で手軽にキャッシュしたいとき | 引数が mutable なとき |
| ルックアップテーブル | 入力空間が小さく固定のとき | 入力空間が巨大なとき |
| ブルームフィルタ | 大規模データの存在判定で偽陽性が許容できるとき | 偽陽性が致命的なとき |
| ハッシュセット | 汎用的な存在判定・重複検出 | メモリ制約が厳しいとき |
| TTL キャッシュ | 外部データの鮮度が重要なとき | データが不変のとき |

---

## 次に読むべきガイド

- [ハッシュテーブル — 衝突解決とロードファクター](../01-data-structures/03-hash-tables.md) — ハッシュセットやブルームフィルタの基盤となるデータ構造
- [動的計画法 — メモ化とテーブルの詳細](../02-algorithms/04-dynamic-programming.md) — メモ化をさらに深掘りする
- キャッシュアーキテクチャ — CPU キャッシュから CDN まで — ハードウェアレベルでのキャッシュの仕組み

---

## 参考文献

1. Bloom, B.H. (1970). "Space/time trade-offs in hash coding with allowable errors." *Communications of the ACM*, 13(7), 422-426. — ブルームフィルタの原著論文。偽陽性を許容することで空間効率を劇的に改善する確率的データ構造を提案した歴史的論文。

2. Cormen, T.H., Leiserson, C.E., Rivest, R.L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第14章「Dynamic Programming」でメモ化とボトムアップ DP のトレードオフを体系的に解説。第11章「Hash Tables」でハッシュ関数とキャッシュの基礎理論を提供。

3. Mitzenmacher, M. & Upfal, E. (2017). *Probability and Computing: Randomization and Probabilistic Techniques in Algorithms and Data Analysis* (2nd ed.). Cambridge University Press. — ブルームフィルタの偽陽性率の厳密な確率分析と、Counting Bloom Filter など各種バリエーションの理論的基盤。

4. Kirsch, A. & Mitzenmacher, M. (2006). "Less Hashing, Same Performance: Building a Better Bloom Filter." *Proceedings of ESA 2006*, 456-467. — ブルームフィルタで必要なハッシュ関数数を 2 つに削減できることを証明した論文。本ガイドのダブルハッシング実装の理論的根拠。

5. Fan, B., Andersen, D.G., Kaminsky, M., & Mitzenmacher, M. (2014). "Cuckoo Filter: Practically Better Than Bloom." *Proceedings of ACM CoNEXT 2014*. — Cuckoo Filter がブルームフィルタより実用面で優れるケースを示した論文。削除対応と空間効率の改善を提案。

6. Knuth, D.E. (1997). *The Art of Computer Programming, Volume 3: Sorting and Searching* (2nd ed.). Addison-Wesley. — ルックアップテーブルと事前計算の手法を含む、検索アルゴリズムの包括的な参考書。

---

*本ガイドで紹介した全てのコード例は Python 3.8 以降で動作確認が可能である。*
*ベンチマーク結果は実行環境（CPU、メモリ、Python バージョン）によって変動する。*