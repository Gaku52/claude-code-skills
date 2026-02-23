# 高度なデータ構造

> Bloom Filter は「存在しない」ことを100%保証し、Skip List はリンクリストにO(log n)探索を与える。

## この章で学ぶこと

- [ ] Bloom Filter の仕組みと用途を理解する
- [ ] Skip List の構造を説明できる
- [ ] セグメント木とフェニック木の区間クエリを実装できる
- [ ] Union-Find の最適化手法を理解する
- [ ] LRU/LFU キャッシュの設計を説明できる
- [ ] Rope、Merkle Tree 等の特殊データ構造を知る
- [ ] 各データ構造の実務的なユースケースを知る

## 前提知識

- ハッシュテーブル → 参照: [[03-hash-tables.md]]
- 木構造 → 参照: [[04-trees.md]]

---

## 1. Bloom Filter

### 1.1 仕組み

```
Bloom Filter: 集合に要素が「含まれないこと」を高速に判定

  構造: mビットの配列 + k個のハッシュ関数

  挿入: hash1(x), hash2(x), ..., hashk(x) の位置のビットを1にする
  検索: 全てのハッシュ位置が1 → 「おそらく存在」
       1つでも0 → 「確実に存在しない」

  偽陽性率: (1 - e^(-kn/m))^k
  n=1000万, m=1億ビット(12.5MB), k=7 → 偽陽性率 ≈ 0.008 (0.8%)

  最適なハッシュ関数の数: k = (m/n) × ln(2)

  用途:
  - Chrome: 悪意あるURLのチェック
  - Cassandra/HBase: SSTファイルの検索スキップ
  - Medium: 記事推薦の重複排除
  - Bitcoin: SPVノードのトランザクション検証
  - Redis: 大規模キャッシュの事前フィルタリング
```

### 1.2 基本実装

```python
import hashlib
import math

class BloomFilter:
    """Bloom Filter: 確率的データ構造"""

    def __init__(self, expected_items, fp_rate=0.01):
        """
        expected_items: 予想される要素数
        fp_rate: 許容する偽陽性率
        """
        # 最適なビット配列サイズを計算
        self.size = self._optimal_size(expected_items, fp_rate)
        # 最適なハッシュ関数の数を計算
        self.num_hashes = self._optimal_hashes(self.size, expected_items)
        # ビット配列（整数のリストで表現）
        self.bit_array = [0] * self.size
        self.count = 0

    def _optimal_size(self, n, p):
        """最適なビット配列サイズ: m = -(n × ln(p)) / (ln(2))²"""
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    def _optimal_hashes(self, m, n):
        """最適なハッシュ関数数: k = (m/n) × ln(2)"""
        k = (m / n) * math.log(2)
        return max(1, int(k))

    def _hashes(self, item):
        """k個の独立したハッシュ値を生成"""
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices

    def add(self, item):
        """要素を追加"""
        for idx in self._hashes(item):
            self.bit_array[idx] = 1
        self.count += 1

    def contains(self, item):
        """要素の存在チェック"""
        return all(self.bit_array[idx] == 1 for idx in self._hashes(item))

    def estimated_fp_rate(self):
        """現在の偽陽性率の推定"""
        ones = sum(self.bit_array)
        if ones == 0:
            return 0.0
        return (ones / self.size) ** self.num_hashes

    def __contains__(self, item):
        return self.contains(item)

    def __len__(self):
        return self.count


# 使用例
bf = BloomFilter(expected_items=100000, fp_rate=0.01)
print(f"ビット配列サイズ: {bf.size:,} bits ({bf.size // 8:,} bytes)")
print(f"ハッシュ関数数: {bf.num_hashes}")

# 要素の追加
for i in range(100000):
    bf.add(f"user:{i}")

# テスト
print(f"存在チェック（存在する）: {'user:42' in bf}")    # True
print(f"存在チェック（存在しない）: {'user:999999' in bf}")  # おそらくFalse

# 偽陽性率の実測
false_positives = sum(
    1 for i in range(100000, 200000)
    if f"user:{i}" in bf
)
actual_fp_rate = false_positives / 100000
print(f"実測偽陽性率: {actual_fp_rate:.4f}")  # 約 0.01 に近い値
```

### 1.3 ビット配列の最適化実装

```python
import array

class OptimizedBloomFilter:
    """メモリ効率の良い Bloom Filter（ビット単位の操作）"""

    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        # ビット配列をバイト配列で実装
        self.bit_array = bytearray((size + 7) // 8)

    def _set_bit(self, pos):
        """ビットを1にセット"""
        self.bit_array[pos >> 3] |= (1 << (pos & 7))

    def _get_bit(self, pos):
        """ビットを取得"""
        return (self.bit_array[pos >> 3] >> (pos & 7)) & 1

    def _hashes(self, item):
        """Double Hashing で高速にk個のハッシュ値を生成"""
        # h1 と h2 から k個のハッシュ値を導出
        data = str(item).encode()
        h1 = int(hashlib.md5(data).hexdigest(), 16)
        h2 = int(hashlib.sha1(data).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.num_hashes)]

    def add(self, item):
        for idx in self._hashes(item):
            self._set_bit(idx)

    def contains(self, item):
        return all(self._get_bit(idx) for idx in self._hashes(item))

    def memory_usage(self):
        """メモリ使用量をバイト単位で返す"""
        return len(self.bit_array)
```

### 1.4 Counting Bloom Filter

```python
class CountingBloomFilter:
    """削除をサポートする Bloom Filter"""

    def __init__(self, size, num_hashes, counter_bits=4):
        self.size = size
        self.num_hashes = num_hashes
        self.max_count = (1 << counter_bits) - 1  # 4ビット → 最大15
        self.counters = [0] * size

    def _hashes(self, item):
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices

    def add(self, item):
        """要素の追加"""
        for idx in self._hashes(item):
            if self.counters[idx] < self.max_count:
                self.counters[idx] += 1

    def remove(self, item):
        """要素の削除（標準 Bloom Filter では不可能）"""
        indices = self._hashes(item)
        # まず存在確認
        if not all(self.counters[idx] > 0 for idx in indices):
            return False  # 存在しない要素は削除できない

        for idx in indices:
            self.counters[idx] -= 1
        return True

    def contains(self, item):
        return all(self.counters[idx] > 0 for idx in self._hashes(item))

    def memory_usage(self):
        """通常の Bloom Filter の4倍のメモリが必要"""
        return self.size * 4  # 4ビット/カウンター


# Counting Bloom Filter の利用先:
# - CDN でのキャッシュ無効化
# - ストリームデータの重複排除（追加と削除が必要）
# - P2Pネットワークのルーティングテーブル
```

### 1.5 Cuckoo Filter

```python
import hashlib
import random

class CuckooFilter:
    """Cuckoo Filter: Bloom Filter の改良版
    - 削除をサポート
    - 空間効率が Bloom Filter より良好（低い偽陽性率の場合）
    - 検索が高速（2箇所のみ確認）"""

    MAX_KICKS = 500

    def __init__(self, capacity, fingerprint_size=8):
        self.capacity = capacity
        self.fp_size = fingerprint_size
        self.buckets = [[] for _ in range(capacity)]
        self.bucket_size = 4  # 各バケットのエントリ数
        self.count = 0

    def _fingerprint(self, item):
        """フィンガープリント（部分ハッシュ）を生成"""
        h = hashlib.sha256(str(item).encode()).hexdigest()
        fp = int(h[:self.fp_size], 16)
        return fp if fp != 0 else 1  # 0は空を意味するので避ける

    def _hash1(self, item):
        h = hashlib.md5(str(item).encode()).hexdigest()
        return int(h, 16) % self.capacity

    def _hash2(self, idx1, fingerprint):
        """代替インデックス = idx1 XOR hash(fingerprint)"""
        h = hashlib.md5(str(fingerprint).encode()).hexdigest()
        return (idx1 ^ (int(h, 16) % self.capacity)) % self.capacity

    def insert(self, item):
        """要素の挿入"""
        fp = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(idx1, fp)

        # バケット1に空きがあれば挿入
        if len(self.buckets[idx1]) < self.bucket_size:
            self.buckets[idx1].append(fp)
            self.count += 1
            return True

        # バケット2に空きがあれば挿入
        if len(self.buckets[idx2]) < self.bucket_size:
            self.buckets[idx2].append(fp)
            self.count += 1
            return True

        # 追い出し（Cuckoo方式）
        idx = random.choice([idx1, idx2])
        for _ in range(self.MAX_KICKS):
            # ランダムにエントリを選択して追い出し
            victim_idx = random.randrange(len(self.buckets[idx]))
            fp, self.buckets[idx][victim_idx] = self.buckets[idx][victim_idx], fp
            idx = self._hash2(idx, fp)

            if len(self.buckets[idx]) < self.bucket_size:
                self.buckets[idx].append(fp)
                self.count += 1
                return True

        return False  # テーブルが満杯

    def lookup(self, item):
        """要素の検索"""
        fp = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(idx1, fp)

        return fp in self.buckets[idx1] or fp in self.buckets[idx2]

    def delete(self, item):
        """要素の削除"""
        fp = self._fingerprint(item)
        idx1 = self._hash1(item)
        idx2 = self._hash2(idx1, fp)

        if fp in self.buckets[idx1]:
            self.buckets[idx1].remove(fp)
            self.count -= 1
            return True

        if fp in self.buckets[idx2]:
            self.buckets[idx2].remove(fp)
            self.count -= 1
            return True

        return False

    def __contains__(self, item):
        return self.lookup(item)

    def __len__(self):
        return self.count


# Bloom Filter vs Cuckoo Filter の比較:
# ┌──────────────────┬──────────────┬──────────────┐
# │ 特性              │ Bloom Filter │ Cuckoo Filter│
# ├──────────────────┼──────────────┼──────────────┤
# │ 削除サポート      │ ❌           │ ✅           │
# │ 検索速度          │ k回ハッシュ  │ 2回のみ ✅   │
# │ 挿入速度          │ O(k)         │ O(1) 期待    │
# │ 偽陽性率 < 3%     │ やや効率悪い │ 空間効率良好 │
# │ 偽陽性率 > 3%     │ 空間効率良好 │ やや効率悪い │
# │ 同一要素の複数挿入│ 可能         │ 制限あり     │
# └──────────────────┴──────────────┴──────────────┘
```

---

## 2. Skip List

### 2.1 基本構造

```
Skip List: 確率的に平衡する順序付きリスト

  レベル3: head ──────────────────────── 50 ────────── tail
  レベル2: head ──────── 20 ──────────── 50 ────────── tail
  レベル1: head ── 10 ── 20 ── 30 ── 40 ── 50 ── 60 ── tail

  検索: 上のレベルから開始、進めなくなったら下のレベルに
  → 平均 O(log n)

  利点:
  - 赤黒木と同等の性能（O(log n)の検索・挿入・削除）
  - 実装が簡単（赤黒木より遥かにシンプル）
  - 並行処理に適する（ロックフリー実装可能）

  用途: Redis の Sorted Set
```

### 2.2 Skip List の実装

```python
import random

class SkipNode:
    def __init__(self, key=None, value=None, level=0):
        self.key = key
        self.value = value
        # forward[i] はレベル i での次のノード
        self.forward = [None] * (level + 1)

class SkipList:
    """Skip List: 確率的に平衡する順序付きリスト"""

    MAX_LEVEL = 16  # 最大レベル数
    P = 0.5         # レベルアップの確率

    def __init__(self):
        self.header = SkipNode(level=self.MAX_LEVEL)
        self.level = 0  # 現在の最大レベル
        self.size = 0

    def _random_level(self):
        """ランダムなレベルを生成（幾何分布）"""
        level = 0
        while random.random() < self.P and level < self.MAX_LEVEL:
            level += 1
        return level

    def search(self, key):
        """検索: 平均 O(log n)、最悪 O(n)"""
        current = self.header

        # 最上位レベルから下へ
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]

        current = current.forward[0]

        if current and current.key == key:
            return current.value
        return None

    def insert(self, key, value):
        """挿入: 平均 O(log n)"""
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header

        # 各レベルで挿入位置を記録
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        # 既存キーの更新
        if current and current.key == key:
            current.value = value
            return

        # 新しいレベルを決定
        new_level = self._random_level()

        if new_level > self.level:
            for i in range(self.level + 1, new_level + 1):
                update[i] = self.header
            self.level = new_level

        # 新しいノードを作成
        new_node = SkipNode(key, value, new_level)

        # 各レベルにリンクを追加
        for i in range(new_level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

        self.size += 1

    def delete(self, key):
        """削除: 平均 O(log n)"""
        update = [None] * (self.MAX_LEVEL + 1)
        current = self.header

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current and current.key == key:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            # 空のレベルを削除
            while self.level > 0 and self.header.forward[self.level] is None:
                self.level -= 1

            self.size -= 1
            return True

        return False

    def range_query(self, start, end):
        """範囲検索: O(log n + k)、k は結果数"""
        result = []
        current = self.header

        # start 以上の最初のノードを見つける
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < start:
                current = current.forward[i]

        current = current.forward[0]

        # end まで走査
        while current and current.key <= end:
            result.append((current.key, current.value))
            current = current.forward[0]

        return result

    def __len__(self):
        return self.size

    def __contains__(self, key):
        return self.search(key) is not None

    def display(self):
        """Skip List の構造を可視化"""
        for level in range(self.level, -1, -1):
            nodes = []
            current = self.header.forward[level]
            while current:
                nodes.append(str(current.key))
                current = current.forward[level]
            print(f"Level {level}: {' -> '.join(nodes)}")


# 使用例
sl = SkipList()
for val in [3, 6, 7, 9, 12, 19, 17, 26, 21, 25]:
    sl.insert(val, val * 10)

sl.display()
# Level 2: 7 -> 19
# Level 1: 3 -> 7 -> 12 -> 19 -> 25
# Level 0: 3 -> 6 -> 7 -> 9 -> 12 -> 17 -> 19 -> 21 -> 25 -> 26

print(sl.search(12))         # 120
print(sl.range_query(10, 20))  # [(12, 120), (17, 170), (19, 190)]
sl.delete(12)
print(sl.search(12))         # None


# Skip List の計算量:
# ┌──────────┬──────────┬──────────┐
# │ 操作     │ 平均     │ 最悪     │
# ├──────────┼──────────┼──────────┤
# │ 検索     │ O(log n) │ O(n)     │
# │ 挿入     │ O(log n) │ O(n)     │
# │ 削除     │ O(log n) │ O(n)     │
# │ 範囲検索 │ O(log n + k)│ O(n)  │
# └──────────┴──────────┴──────────┘
# 空間: O(n)（期待）
# ノードの平均レベル: 1/(1-p) = 2（p=0.5の場合）
```

### 2.3 Redis の Sorted Set

```
Redis Sorted Set の内部実装:

  Skip List + ハッシュテーブルのハイブリッド
  - Skip List: スコア順のソート、範囲検索
  - ハッシュテーブル: メンバー名からスコアへの O(1) 参照

  使用例（Redis コマンド）:
  ZADD leaderboard 100 "Alice"
  ZADD leaderboard 85 "Bob"
  ZADD leaderboard 92 "Charlie"

  ZRANK leaderboard "Alice"          # 2（0-indexed、スコア順）
  ZRANGE leaderboard 0 -1 WITHSCORES # 全メンバーをスコア順で取得
  ZRANGEBYSCORE leaderboard 85 95    # スコア85-95のメンバー
  ZREVRANK leaderboard "Alice"       # 0（降順でのランク）

  Skip List が赤黒木より選ばれた理由（Redis作者 antirez の説明）:
  1. 実装がシンプル
  2. 範囲検索のパフォーマンスが良い
  3. 並行処理での拡張が容易
  4. 赤黒木と同等の性能（定数倍の差）
```

---

## 3. セグメント木（Segment Tree）

### 3.1 基本概念

```
セグメント木: 区間クエリを O(log n) で処理

  配列: [2, 1, 5, 3, 4, 2]

  セグメント木（区間最小値）:
              1           ← 全体の最小値
            /   \
          1       2       ← 前半/後半の最小値
         / \     / \
        1   5   3   2     ← さらに半分
       / \ / \ / \ / \
      2  1 5  3 4  2      ← 元の配列

  用途:
  - 区間最小値/最大値クエリ（RMQ）
  - 区間和クエリ
  - 区間更新（遅延伝播）
  - 座標圧縮との組み合わせ
  - 競技プログラミングの定番
```

### 3.2 セグメント木の実装

```python
class SegmentTree:
    """セグメント木（区間最小値クエリ + 点更新）"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)  # 十分なサイズを確保
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        """木を構築: O(n)"""
        if start == end:
            self.tree[node] = data[start]
            return

        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        """区間 [left, right] の最小値を取得: O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        # 範囲外
        if right < start or end < left:
            return float('inf')

        # 完全に範囲内
        if left <= start and end <= right:
            return self.tree[node]

        # 部分的に範囲内
        mid = (start + end) // 2
        left_min = self._query(2 * node, start, mid, left, right)
        right_min = self._query(2 * node + 1, mid + 1, end, left, right)
        return min(left_min, right_min)

    def update(self, index, value):
        """index の値を value に更新: O(log n)"""
        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return

        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)

        self.tree[node] = min(self.tree[2 * node], self.tree[2 * node + 1])


# 使用例
data = [2, 1, 5, 3, 4, 2]
st = SegmentTree(data)

print(st.query(0, 5))  # 1（全体の最小値）
print(st.query(2, 4))  # 3（[5,3,4]の最小値）
print(st.query(0, 2))  # 1（[2,1,5]の最小値）

st.update(1, 10)       # data[1] = 10 に更新
print(st.query(0, 2))  # 2（[2,10,5]の最小値）
```

### 3.3 区間和セグメント木

```python
class SumSegmentTree:
    """区間和クエリ + 点更新"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """区間 [left, right] の合計: O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))

    def update(self, index, value):
        """index の値を value に更新: O(log n)"""
        self._update(1, 0, self.n - 1, index, value)

    def _update(self, node, start, end, index, value):
        if start == end:
            self.tree[node] = value
            return
        mid = (start + end) // 2
        if index <= mid:
            self._update(2 * node, start, mid, index, value)
        else:
            self._update(2 * node + 1, mid + 1, end, index, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

# 使用例
data = [1, 3, 5, 7, 9, 11]
st = SumSegmentTree(data)
print(st.query(1, 3))  # 15 (3 + 5 + 7)
st.update(2, 10)       # data[2] = 10
print(st.query(1, 3))  # 20 (3 + 10 + 7)
```

### 3.4 遅延伝播（Lazy Propagation）

```python
class LazySegmentTree:
    """遅延伝播付きセグメント木: 区間更新 + 区間クエリ O(log n)"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 遅延更新の値
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, start, end):
        if start == end:
            self.tree[node] = data[start]
            return
        mid = (start + end) // 2
        self._build(data, 2 * node, start, mid)
        self._build(data, 2 * node + 1, mid + 1, end)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def _push_down(self, node, start, end):
        """遅延更新を子ノードに伝播"""
        if self.lazy[node] != 0:
            mid = (start + end) // 2
            # 左の子
            self.tree[2 * node] += self.lazy[node] * (mid - start + 1)
            self.lazy[2 * node] += self.lazy[node]
            # 右の子
            self.tree[2 * node + 1] += self.lazy[node] * (end - mid)
            self.lazy[2 * node + 1] += self.lazy[node]
            # 自分の遅延をクリア
            self.lazy[node] = 0

    def range_update(self, left, right, value):
        """区間 [left, right] に value を加算: O(log n)"""
        self._range_update(1, 0, self.n - 1, left, right, value)

    def _range_update(self, node, start, end, left, right, value):
        if right < start or end < left:
            return
        if left <= start and end <= right:
            self.tree[node] += value * (end - start + 1)
            self.lazy[node] += value
            return
        self._push_down(node, start, end)
        mid = (start + end) // 2
        self._range_update(2 * node, start, mid, left, right, value)
        self._range_update(2 * node + 1, mid + 1, end, left, right, value)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, left, right):
        """区間 [left, right] の合計: O(log n)"""
        return self._query(1, 0, self.n - 1, left, right)

    def _query(self, node, start, end, left, right):
        if right < start or end < left:
            return 0
        if left <= start and end <= right:
            return self.tree[node]
        self._push_down(node, start, end)
        mid = (start + end) // 2
        return (self._query(2 * node, start, mid, left, right) +
                self._query(2 * node + 1, mid + 1, end, left, right))


# 使用例
data = [1, 3, 5, 7, 9, 11]
lst = LazySegmentTree(data)
print(lst.query(1, 4))       # 24 (3 + 5 + 7 + 9)
lst.range_update(1, 3, 10)   # [1, 13, 15, 17, 9, 11]
print(lst.query(1, 4))       # 54 (13 + 15 + 17 + 9)
```

---

## 4. フェニック木（Binary Indexed Tree / BIT）

### 4.1 基本概念と実装

```python
class FenwickTree:
    """フェニック木（BIT）: 区間和クエリと点更新
    セグメント木よりシンプルだが、機能は区間和に限定"""

    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)  # 1-indexed

    def update(self, i, delta):
        """index i に delta を加算: O(log n)"""
        i += 1  # 1-indexed に変換
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)  # 最下位ビットを加算

    def prefix_sum(self, i):
        """[0, i] の累積和: O(log n)"""
        i += 1  # 1-indexed に変換
        total = 0
        while i > 0:
            total += self.tree[i]
            i -= i & (-i)  # 最下位ビットを減算
        return total

    def range_sum(self, left, right):
        """[left, right] の区間和: O(log n)"""
        if left == 0:
            return self.prefix_sum(right)
        return self.prefix_sum(right) - self.prefix_sum(left - 1)

    @classmethod
    def from_array(cls, arr):
        """配列から O(n) で構築"""
        ft = cls(len(arr))
        for i, val in enumerate(arr):
            ft.update(i, val)
        return ft


# 使用例
data = [1, 3, 5, 7, 9, 11]
ft = FenwickTree.from_array(data)

print(ft.prefix_sum(3))    # 16 (1 + 3 + 5 + 7)
print(ft.range_sum(2, 4))  # 21 (5 + 7 + 9)

ft.update(2, 5)            # data[2] += 5 → [1, 3, 10, 7, 9, 11]
print(ft.range_sum(2, 4))  # 26 (10 + 7 + 9)


# フェニック木 vs セグメント木:
# ┌──────────────────┬──────────────┬──────────────┐
# │ 特性              │ フェニック木 │ セグメント木 │
# ├──────────────────┼──────────────┼──────────────┤
# │ 実装の簡潔さ      │ ✅ 非常に簡潔│ △ やや複雑  │
# │ メモリ            │ O(n) ✅      │ O(4n)        │
# │ 定数倍            │ ✅ 高速     │ △ やや遅い   │
# │ 対応クエリ        │ 区間和のみ   │ 任意 ✅     │
# │ 区間更新          │ △ 工夫要    │ ✅ 遅延伝播  │
# │ 最小値/最大値     │ ❌           │ ✅           │
# └──────────────────┴──────────────┴──────────────┘
```

### 4.2 2次元フェニック木

```python
class FenwickTree2D:
    """2次元フェニック木: 2D 区間和クエリ"""

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, row, col, delta):
        """(row, col) に delta を加算: O(log(R) * log(C))"""
        r = row + 1
        while r <= self.rows:
            c = col + 1
            while c <= self.cols:
                self.tree[r][c] += delta
                c += c & (-c)
            r += r & (-r)

    def prefix_sum(self, row, col):
        """(0,0) から (row, col) までの累積和"""
        total = 0
        r = row + 1
        while r > 0:
            c = col + 1
            while c > 0:
                total += self.tree[r][c]
                c -= c & (-c)
            r -= r & (-r)
        return total

    def range_sum(self, r1, c1, r2, c2):
        """(r1,c1) から (r2,c2) の矩形区間和"""
        total = self.prefix_sum(r2, c2)
        if r1 > 0:
            total -= self.prefix_sum(r1 - 1, c2)
        if c1 > 0:
            total -= self.prefix_sum(r2, c1 - 1)
        if r1 > 0 and c1 > 0:
            total += self.prefix_sum(r1 - 1, c1 - 1)
        return total

# 使用例: 2D マトリクスの区間和
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
ft2d = FenwickTree2D(3, 3)
for r in range(3):
    for c in range(3):
        ft2d.update(r, c, matrix[r][c])

print(ft2d.range_sum(0, 0, 1, 1))  # 12 (1+2+4+5)
print(ft2d.range_sum(1, 1, 2, 2))  # 28 (5+6+8+9)
```

---

## 5. Rope（ロープ）

### 5.1 概念と実装

```python
class RopeNode:
    """Rope: 長い文字列の効率的な操作（テキストエディタ向け）"""

    def __init__(self, text=None, left=None, right=None):
        if text is not None:
            # 葉ノード
            self.text = text
            self.weight = len(text)
            self.left = None
            self.right = None
        else:
            # 内部ノード
            self.text = None
            self.left = left
            self.right = right
            self.weight = self._total_length(left) if left else 0

    def _total_length(self, node):
        if node is None:
            return 0
        if node.text is not None:
            return len(node.text)
        return node.weight + self._total_length(node.right)

class Rope:
    """Rope: テキストエディタ向けの文字列データ構造

    操作の計算量:
    - 連結: O(1)（新しいルートノードを作るだけ）
    - 分割: O(log n)
    - インデックスアクセス: O(log n)
    - 挿入: O(log n)
    - 削除: O(log n)

    通常の文字列との比較:
    - 連結: O(1) vs O(n)（Rope が圧倒的に有利）
    - アクセス: O(log n) vs O(1)（文字列が有利）
    """

    def __init__(self, text=""):
        self.root = RopeNode(text=text) if text else None

    def concat(self, other):
        """2つの Rope を連結: O(1)"""
        if not self.root:
            self.root = other.root
        elif other.root:
            self.root = RopeNode(left=self.root, right=other.root)
        return self

    def index(self, i):
        """i番目の文字を取得: O(log n)"""
        return self._index(self.root, i)

    def _index(self, node, i):
        if node is None:
            raise IndexError("index out of range")

        if node.text is not None:
            # 葉ノード
            return node.text[i]

        if i < node.weight:
            return self._index(node.left, i)
        else:
            return self._index(node.right, i - node.weight)

    def to_string(self):
        """全文字列を取得: O(n)"""
        result = []
        self._collect(self.root, result)
        return "".join(result)

    def _collect(self, node, result):
        if node is None:
            return
        if node.text is not None:
            result.append(node.text)
            return
        self._collect(node.left, result)
        self._collect(node.right, result)

    def __len__(self):
        return self._length(self.root)

    def _length(self, node):
        if node is None:
            return 0
        if node.text is not None:
            return len(node.text)
        return node.weight + self._length(node.right)


# 使用例（テキストエディタの内部表現）
rope1 = Rope("Hello, ")
rope2 = Rope("World!")
rope1.concat(rope2)
print(rope1.to_string())    # "Hello, World!"
print(rope1.index(7))       # 'W'
print(len(rope1))            # 13

# Rope の利用先:
# - Xi Editor (Rust のテキストエディタ)
# - JetBrains IDE のテキストバッファ
# - Visual Studio Code の一部
# - CLion のテキスト管理
```

---

## 6. Merkle Tree

### 6.1 概念と実装

```python
import hashlib

class MerkleNode:
    def __init__(self, data=None, left=None, right=None):
        if data is not None:
            self.hash = hashlib.sha256(data.encode()).hexdigest()
        else:
            combined = left.hash + right.hash
            self.hash = hashlib.sha256(combined.encode()).hexdigest()
        self.left = left
        self.right = right
        self.data = data

class MerkleTree:
    """Merkle Tree: データ整合性の効率的な検証

    用途:
    - Git: コミットとファイルの整合性
    - Bitcoin/Ethereum: トランザクションの検証
    - Amazon DynamoDB: レプリカ間のデータ同期
    - IPFS: コンテンツアドレッシング
    """

    def __init__(self, data_list):
        leaves = [MerkleNode(data=d) for d in data_list]

        # 奇数の場合、最後の要素を複製
        if len(leaves) % 2 == 1:
            leaves.append(MerkleNode(data=data_list[-1]))

        self.root = self._build(leaves)

    def _build(self, nodes):
        """ボトムアップで木を構築: O(n)"""
        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    parent = MerkleNode(left=nodes[i], right=nodes[i + 1])
                else:
                    parent = MerkleNode(left=nodes[i], right=nodes[i])
                next_level.append(parent)
            nodes = next_level
        return nodes[0] if nodes else None

    def get_root_hash(self):
        """ルートハッシュを取得"""
        return self.root.hash if self.root else None

    def get_proof(self, index, data_list):
        """Merkle Proof: 特定のデータの存在証明を生成
        O(log n) のハッシュ値で全体の整合性を証明可能"""
        # 葉ノードを再構築
        leaves = [MerkleNode(data=d) for d in data_list]
        if len(leaves) % 2 == 1:
            leaves.append(MerkleNode(data=data_list[-1]))

        proof = []
        nodes = leaves
        target_index = index

        while len(nodes) > 1:
            next_level = []
            for i in range(0, len(nodes), 2):
                if i + 1 < len(nodes):
                    parent = MerkleNode(left=nodes[i], right=nodes[i + 1])
                    # 証明に必要な兄弟ノードのハッシュを記録
                    if i == target_index or i + 1 == target_index:
                        sibling_idx = i + 1 if i == target_index else i
                        side = "right" if sibling_idx > target_index else "left"
                        proof.append((side, nodes[sibling_idx].hash))
                        target_index = len(next_level)
                else:
                    parent = MerkleNode(left=nodes[i], right=nodes[i])
                next_level.append(parent)
            nodes = next_level

        return proof

    @staticmethod
    def verify_proof(data, proof, root_hash):
        """Merkle Proof を検証: O(log n)"""
        current_hash = hashlib.sha256(data.encode()).hexdigest()

        for side, sibling_hash in proof:
            if side == "right":
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            current_hash = hashlib.sha256(combined.encode()).hexdigest()

        return current_hash == root_hash


# 使用例
data = ["tx1", "tx2", "tx3", "tx4"]
mt = MerkleTree(data)
print(f"Root Hash: {mt.get_root_hash()[:16]}...")

# tx2 の存在証明を生成（O(log n) のデータのみ）
proof = mt.get_proof(1, data)
print(f"Proof size: {len(proof)} hashes")  # log2(4) = 2

# 検証
is_valid = MerkleTree.verify_proof("tx2", proof, mt.get_root_hash())
print(f"Valid: {is_valid}")  # True

# 改ざんされたデータでの検証
is_valid = MerkleTree.verify_proof("tx_fake", proof, mt.get_root_hash())
print(f"Valid: {is_valid}")  # False
```

---

## 7. LFU キャッシュ

### 7.1 実装

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    """Least Frequently Used キャッシュ
    全操作 O(1)"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}            # key -> value
        self.freq = {}             # key -> frequency
        self.freq_to_keys = defaultdict(OrderedDict)  # freq -> OrderedDict of keys
        self.min_freq = 0

    def get(self, key):
        if key not in self.cache:
            return -1

        # 頻度を増加
        self._update_freq(key)
        return self.cache[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return

        if key in self.cache:
            self.cache[key] = value
            self._update_freq(key)
            return

        if len(self.cache) >= self.capacity:
            # 最低頻度で最も古いキーを削除
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.cache[evict_key]
            del self.freq[evict_key]

        self.cache[key] = value
        self.freq[key] = 1
        self.freq_to_keys[1][key] = True
        self.min_freq = 1

    def _update_freq(self, key):
        """キーの頻度を更新"""
        old_freq = self.freq[key]
        new_freq = old_freq + 1
        self.freq[key] = new_freq

        # 古い頻度グループから削除
        del self.freq_to_keys[old_freq][key]
        if not self.freq_to_keys[old_freq]:
            del self.freq_to_keys[old_freq]
            if self.min_freq == old_freq:
                self.min_freq += 1

        # 新しい頻度グループに追加
        self.freq_to_keys[new_freq][key] = True


# 使用例
cache = LFUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
cache.get("a")       # 1 → freq("a") = 2
cache.get("a")       # 1 → freq("a") = 3
cache.get("b")       # 2 → freq("b") = 2
cache.put("d", 4)    # "c" が削除される（freq=1で最も古い）
print(cache.get("c"))  # -1（削除済み）
print(cache.get("a"))  # 1
print(cache.get("d"))  # 4
```

---

## 8. Disjoint Set（素集合）の応用

### 8.1 重み付き Union-Find

```python
class WeightedUnionFind:
    """重み付き Union-Find: 各ノードの根からの相対重みを管理"""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.weight = [0] * n  # 親からの相対重み

    def find(self, x):
        """経路圧縮付き find（重みも更新）"""
        if self.parent[x] != x:
            root = self.find(self.parent[x])
            self.weight[x] += self.weight[self.parent[x]]
            self.parent[x] = root
        return self.parent[x]

    def union(self, x, y, w):
        """x と y を統合、weight(y) - weight(x) = w"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return self.weight[x] - self.weight[y] == w  # 整合性チェック

        # w = weight(y) - weight(x)
        # → weight(root_y) = weight(x) - weight(y) + w
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
            self.weight[root_x] = self.weight[y] - self.weight[x] + w
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.weight[x] - self.weight[y] - w
        else:
            self.parent[root_y] = root_x
            self.weight[root_y] = self.weight[x] - self.weight[y] - w
            self.rank[root_x] += 1

        return True

    def diff(self, x, y):
        """weight(y) - weight(x) を返す"""
        if self.find(x) != self.find(y):
            return None  # 同じ集合に属していない
        return self.weight[x] - self.weight[y]


# 使用例: 相対評価問題
# A は B より 3 高い、B は C より 2 高い → A は C より何高い？
wuf = WeightedUnionFind(3)
wuf.union(0, 1, 3)   # score[1] - score[0] = 3
wuf.union(1, 2, 2)   # score[2] - score[1] = 2
print(wuf.diff(0, 2)) # 5 (score[2] - score[0] = 5)
```

---

## 9. その他の高度なデータ構造

### 9.1 概要一覧

```
┌──────────────────┬──────────────────┬───────────────────┐
│ データ構造        │ 用途             │ 計算量            │
├──────────────────┼──────────────────┼───────────────────┤
│ Segment Tree     │ 区間クエリ       │ O(log n) 更新/検索│
│ Fenwick Tree(BIT)│ 区間和クエリ     │ O(log n)          │
│ Disjoint Set     │ 集合の統合       │ O(α(n)) ≈ O(1)  │
│ LRU Cache        │ キャッシュ管理   │ O(1) 全操作      │
│ LFU Cache        │ 頻度ベースキャッシュ│ O(1) 全操作    │
│ Rope             │ 長い文字列操作   │ O(log n) 連結    │
│ Merkle Tree      │ データ整合性検証 │ O(log n) 検証    │
│ Bloom Filter     │ 存在チェック     │ O(k) 検索/挿入   │
│ Cuckoo Filter    │ 存在チェック+削除│ O(1) 検索        │
│ Skip List        │ 順序付きデータ   │ O(log n) 検索    │
│ Trie             │ 文字列検索       │ O(m) 検索        │
│ Suffix Array     │ 部分文字列検索   │ O(m log n) 検索  │
│ Suffix Tree      │ 部分文字列検索   │ O(m) 検索        │
│ Wavelet Tree     │ 範囲頻度クエリ   │ O(log σ) クエリ  │
│ Persistent DS    │ バージョン管理   │ 操作ごとO(log n) │
│ Van Emde Boas    │ 整数集合         │ O(log log U)     │
│ Splay Tree       │ 自己調整BST      │ 償却O(log n)     │
│ Treap            │ 確率的BST        │ 期待O(log n)     │
└──────────────────┴──────────────────┴───────────────────┘
```

### 9.2 Treap（Tree + Heap）

```python
import random

class TreapNode:
    def __init__(self, key):
        self.key = key
        self.priority = random.random()  # ランダムな優先度
        self.left = None
        self.right = None
        self.size = 1  # 部分木のサイズ

class Treap:
    """Treap: BST性質（キー）+ ヒープ性質（優先度）
    ランダムな優先度により確率的にバランスが保たれる"""

    def __init__(self):
        self.root = None

    def _size(self, node):
        return node.size if node else 0

    def _update(self, node):
        if node:
            node.size = 1 + self._size(node.left) + self._size(node.right)

    def _split(self, node, key):
        """木をキーで分割: 左 < key ≤ 右"""
        if not node:
            return None, None

        if key <= node.key:
            left, node.left = self._split(node.left, key)
            self._update(node)
            return left, node
        else:
            node.right, right = self._split(node.right, key)
            self._update(node)
            return node, right

    def _merge(self, left, right):
        """2つの木をマージ（左の全キー < 右の全キー）"""
        if not left or not right:
            return left or right

        if left.priority > right.priority:
            left.right = self._merge(left.right, right)
            self._update(left)
            return left
        else:
            right.left = self._merge(left, right.left)
            self._update(right)
            return right

    def insert(self, key):
        """挿入: 期待O(log n)"""
        left, right = self._split(self.root, key)
        node = TreapNode(key)
        self.root = self._merge(self._merge(left, node), right)

    def delete(self, key):
        """削除: 期待O(log n)"""
        self.root = self._delete(self.root, key)

    def _delete(self, node, key):
        if not node:
            return None
        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            node = self._merge(node.left, node.right)
        if node:
            self._update(node)
        return node

    def kth(self, k):
        """k番目に小さい要素: O(log n)"""
        return self._kth(self.root, k)

    def _kth(self, node, k):
        if not node:
            return None
        left_size = self._size(node.left)
        if k == left_size + 1:
            return node.key
        elif k <= left_size:
            return self._kth(node.left, k)
        else:
            return self._kth(node.right, k - left_size - 1)


# Treap の利点:
# - 実装が赤黒木より大幅にシンプル
# - Split/Merge ベースで柔軟な操作が可能
# - 期待計算量は O(log n)
# - 区間操作（反転、移動）にも拡張可能
```

### 9.3 永続データ構造（Persistent Data Structure）

```python
class PersistentNode:
    """永続的なノード: 更新時にコピーオンライト"""
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class PersistentBST:
    """永続BST: 全バージョンの木にアクセス可能
    各操作で O(log n) の新ノードのみ作成"""

    def __init__(self):
        self.versions = [None]  # バージョン0: 空の木

    def insert(self, val, version=-1):
        """指定バージョンに挿入して新バージョンを作成"""
        root = self.versions[version]
        new_root = self._insert(root, val)
        self.versions.append(new_root)
        return len(self.versions) - 1  # 新バージョン番号

    def _insert(self, node, val):
        if not node:
            return PersistentNode(val)

        if val < node.val:
            return PersistentNode(node.val,
                                  self._insert(node.left, val),
                                  node.right)  # 右部分木は共有
        elif val > node.val:
            return PersistentNode(node.val,
                                  node.left,  # 左部分木は共有
                                  self._insert(node.right, val))
        return node  # 重複は無視

    def search(self, val, version=-1):
        """指定バージョンで検索"""
        return self._search(self.versions[version], val)

    def _search(self, node, val):
        if not node:
            return False
        if val == node.val:
            return True
        if val < node.val:
            return self._search(node.left, val)
        return self._search(node.right, val)

    def inorder(self, version=-1):
        """指定バージョンの中順走査"""
        result = []
        self._inorder(self.versions[version], result)
        return result

    def _inorder(self, node, result):
        if not node:
            return
        self._inorder(node.left, result)
        result.append(node.val)
        self._inorder(node.right, result)


# 使用例
pbst = PersistentBST()
v1 = pbst.insert(5)   # v1: {5}
v2 = pbst.insert(3)   # v2: {3, 5}
v3 = pbst.insert(7)   # v3: {3, 5, 7}
v4 = pbst.insert(4, version=v2)  # v4: v2ベース → {3, 4, 5}

print(pbst.inorder(v1))  # [5]
print(pbst.inorder(v2))  # [3, 5]
print(pbst.inorder(v3))  # [3, 5, 7]
print(pbst.inorder(v4))  # [3, 4, 5]（v3とは独立）

# 永続データ構造の利用先:
# - Git: ファイルツリーのバージョン管理
# - Clojure/Haskell: イミュータブルなデータ構造
# - React: Virtual DOM の差分検出
# - データベース: MVCC（Multi-Version Concurrency Control）
```

---

## 10. 実践演習

### 演習1: Bloom Filter（基礎）
Bloom Filterを実装し、偽陽性率をパラメータ（m, k）を変えて実測せよ。最適なパラメータを自動計算する機能も実装すること。

### 演習2: セグメント木（応用）
以下をサポートするセグメント木を実装せよ:
- 区間の最小値クエリと点更新
- 区間の合計クエリ
- 遅延伝播による区間更新
- 区間の最大値とその位置の取得

### 演習3: Skip List（応用）
Skip List を実装し、以下の機能を含めよ:
- 挿入、検索、削除
- 範囲検索（start〜end のキーを取得）
- ランクの取得（k番目に小さい要素）
- レベル構造の可視化

### 演習4: LFU キャッシュ（発展）
O(1) の全操作をサポートする LFU キャッシュを実装せよ。TTL（有効期限）のサポートも追加すること。

### 演習5: Merkle Tree（発展）
Merkle Tree を実装し、以下の機能を含めよ:
- ルートハッシュの計算
- Merkle Proof の生成と検証
- データの改ざん検出デモ
- 2つの Merkle Tree の差分検出

### 演習6: Cuckoo Filter（発展）
Cuckoo Filter を実装し、Bloom Filter との性能比較を行え:
- 挿入、検索、削除のベンチマーク
- 偽陽性率の比較
- メモリ使用量の比較

---

## まとめ

| データ構造 | 特性 | 主な用途 |
|-----------|------|---------|
| Bloom Filter | 偽陽性あり、偽陰性なし | 存在チェック、キャッシュ |
| Cuckoo Filter | Bloom Filter + 削除サポート | 削除が必要な存在チェック |
| Skip List | 確率的平衡 O(log n) | Redis Sorted Set |
| Segment Tree | 区間クエリ O(log n) | 競技プログラミング、DB |
| Fenwick Tree | 区間和 O(log n)、簡潔 | 累積和の動的更新 |
| Union-Find | ほぼO(1)の集合統合 | クラスタリング、MST |
| Rope | O(1) 連結 | テキストエディタ |
| Merkle Tree | O(log n) 整合性検証 | Git, ブロックチェーン |
| LFU Cache | 頻度ベース O(1) | CDN、DBキャッシュ |
| Treap | 確率的BST + Split/Merge | 柔軟な順序集合 |
| Persistent DS | バージョン管理 | 関数型プログラミング |

---

## 次に読むべきガイド
→ [[07-choosing-data-structures.md]] -- データ構造の選び方

---

## 参考文献
1. Bloom, B. H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." 1970.
2. Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." 1990.
3. Fan, B. et al. "Cuckoo Filter: Practically Better Than Bloom." CoNEXT 2014.
4. Merkle, R. C. "A Digital Signature Based on a Conventional Encryption Function." CRYPTO 1987.
5. Driscoll, J. R. et al. "Making Data Structures Persistent." STOC 1986.
6. Aragon, C. R., Seidel, R. "Randomized Search Trees." FOCS 1989.
7. De Berg, M. et al. "Computational Geometry." Chapter 10: Segment Trees.
