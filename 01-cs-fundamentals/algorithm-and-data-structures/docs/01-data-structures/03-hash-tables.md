# ハッシュテーブル — ハッシュ関数・衝突解決・ロードファクター

> 平均 O(1) のキー検索を実現するハッシュテーブルの内部構造、衝突解決戦略、性能チューニングを学ぶ。

---

## この章で学ぶこと

1. **ハッシュ関数** の設計原則と良い関数の条件
2. **衝突解決** — チェイン法とオープンアドレス法
3. **ロードファクター** とリハッシュの仕組み
4. **言語別実装** — Python dict, Java HashMap, C++ unordered_map の内部構造
5. **実務応用** — キャッシュ、一意性チェック、集合演算

---

## 1. ハッシュテーブルの基本構造

ハッシュテーブルは「キー」から「値」への写像を O(1) 平均で実現するデータ構造である。内部的にはバケット配列（スロット配列）を持ち、ハッシュ関数によってキーからバケットのインデックスを算出する。

```
キー "apple" → ハッシュ関数 → インデックス 3

  バケット配列:
  [0] → null
  [1] → ("banana", 2)
  [2] → null
  [3] → ("apple", 5)     ← h("apple") = 3
  [4] → null
  [5] → ("cherry", 8)
  [6] → null
  [7] → ("date", 1)

  検索 "apple":
  1. h("apple") = 3
  2. bucket[3] を参照
  3. キーが一致 → 値 5 を返す
  → O(1) 平均
```

### 1.1 基本操作の計算量

| 操作 | 平均 | 最悪 | 備考 |
|------|------|------|------|
| 検索 (get) | O(1) | O(n) | 全キーが同一バケットに衝突した場合 |
| 挿入 (put) | O(1) | O(n) | リハッシュ発生時は O(n) の一括コスト |
| 削除 (delete) | O(1) | O(n) | オープンアドレスでは DELETED マーカー必要 |
| キー列挙 | O(n + m) | O(n + m) | m = バケット数 |

### 1.2 ハッシュテーブルが使われる典型場面

```python
# 1. 出現頻度のカウント
from collections import Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
freq = Counter(words)
print(freq)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})

# 2. 重複チェック（O(n) で完了）
def has_duplicate(arr):
    seen = set()
    for x in arr:
        if x in seen:
            return True
        seen.add(x)
    return False

# 3. 二数の和 (Two Sum) — O(n)
def two_sum(nums, target):
    lookup = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in lookup:
            return [lookup[complement], i]
        lookup[num] = i
    return []

# 4. グループ化
from collections import defaultdict
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())

print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

---

## 2. ハッシュ関数

### 2.1 良いハッシュ関数の条件

```python
# 条件:
# 1. 決定的: 同じ入力 → 同じ出力
# 2. 均一分布: 出力がバケット全体に均等に散らばる
# 3. 高速: 計算が O(キー長) 程度
# 4. 雪崩効果: 入力の小さな変化が出力の大きな変化を生む

# 文字列のハッシュ関数例（多項式ハッシュ）
def polynomial_hash(key, table_size, base=31):
    """多項式ハッシュ — O(len(key))"""
    h = 0
    for char in key:
        h = (h * base + ord(char)) % table_size
    return h

# Python の組み込み hash()
print(hash("hello"))    # 整数値を返す
print(hash(42))         # 整数はそのまま（概ね）
print(hash((1, 2, 3)))  # タプルはハッシュ可能
# hash([1, 2, 3])       # リストはハッシュ不可（mutable）
```

### 2.2 各種ハッシュ関数の実装

```python
# === 除算法 (Division Method) ===
def hash_division(key, table_size):
    """除算法: h(k) = k mod m
    テーブルサイズ m は 2 のべき乗に近い素数を選ぶ
    """
    return key % table_size

# === 乗算法 (Multiplication Method) ===
def hash_multiplication(key, table_size):
    """乗算法: h(k) = floor(m * (k * A mod 1))
    A = (√5 - 1) / 2 ≈ 0.6180339887 が推奨
    テーブルサイズ m に制約がない
    """
    import math
    A = (math.sqrt(5) - 1) / 2  # 黄金比の逆数
    return int(table_size * ((key * A) % 1))

# === FNV-1a ハッシュ ===
def fnv1a_hash(data, table_size):
    """FNV-1a: 文字列ハッシュとして広く使用
    高速で均一分布に優れる
    """
    FNV_OFFSET = 2166136261
    FNV_PRIME = 16777619
    h = FNV_OFFSET
    for byte in data.encode('utf-8'):
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF  # 32bit に制限
    return h % table_size

# === MurmurHash3 簡易版 ===
def murmur3_32(key, seed=0):
    """MurmurHash3 の 32bit 簡易実装
    非暗号学的ハッシュ関数として最も広く使用される
    """
    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    data = key.encode('utf-8') if isinstance(key, str) else key
    length = len(data)
    h = seed

    # ボディ: 4バイトずつ処理
    nblocks = length // 4
    for i in range(nblocks):
        k = int.from_bytes(data[i*4:(i+1)*4], 'little')
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k
        h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
        h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    # テール: 残りのバイト
    tail = data[nblocks * 4:]
    k = 0
    for i in range(len(tail) - 1, -1, -1):
        k = (k << 8) | tail[i]
    if k:
        k = (k * c1) & 0xFFFFFFFF
        k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
        k = (k * c2) & 0xFFFFFFFF
        h ^= k

    # ファイナライゼーション
    h ^= length
    h ^= (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h ^= (h >> 16)

    return h

# テスト
print(polynomial_hash("hello", 16))      # 多項式ハッシュ
print(hash_division(42, 17))              # 除算法
print(hash_multiplication(42, 16))        # 乗算法
print(fnv1a_hash("hello", 16))           # FNV-1a
print(murmur3_32("hello"))               # MurmurHash3
```

### 2.3 暗号学的ハッシュ vs 非暗号学的ハッシュ

```python
import hashlib

# === 暗号学的ハッシュ（セキュリティ用途） ===
# SHA-256: パスワード保存、データ整合性検証
data = "hello world"
sha256_hash = hashlib.sha256(data.encode()).hexdigest()
print(f"SHA-256: {sha256_hash}")

# MD5: チェックサム（セキュリティには非推奨）
md5_hash = hashlib.md5(data.encode()).hexdigest()
print(f"MD5: {md5_hash}")

# === 特性比較 ===
# | 特性         | 暗号学的ハッシュ          | 非暗号学的ハッシュ     |
# |-------------|------------------------|---------------------|
# | 速度        | 遅い（意図的）           | 速い                |
# | 衝突耐性    | 高い（計算量的に困難）     | 低い（十分だが保証なし）|
# | 用途        | セキュリティ、署名        | ハッシュテーブル      |
# | 例          | SHA-256, bcrypt         | MurmurHash, FNV    |
# | 逆像耐性    | あり                    | 不要                |
```

### 2.4 ユニバーサルハッシュ

```python
import random

class UniversalHashFamily:
    """ユニバーサルハッシュファミリー
    任意の2つの異なるキー x, y に対して:
    Pr[h(x) = h(y)] <= 1/m  (m = テーブルサイズ)

    ハッシュ DoS 攻撃への対策に有効
    """
    def __init__(self, table_size, prime=2147483647):
        self.m = table_size
        self.p = prime  # テーブルサイズより大きい素数
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)

    def hash(self, key):
        """h(k) = ((a*k + b) mod p) mod m"""
        return ((self.a * key + self.b) % self.p) % self.m

    def regenerate(self):
        """新しいハッシュ関数をランダムに選択"""
        self.a = random.randint(1, self.p - 1)
        self.b = random.randint(0, self.p - 1)

# 使用例
uhf = UniversalHashFamily(16)
print(uhf.hash(42))
print(uhf.hash(100))
uhf.regenerate()  # 別のハッシュ関数に切り替え
print(uhf.hash(42))  # 異なる結果になる可能性が高い
```

---

## 3. 衝突解決

### 3.1 チェイン法（Separate Chaining）

```
バケット配列 + 連結リスト:

  [0] → null
  [1] → ("banana",2) → ("fig",7) → null
  [2] → null
  [3] → ("apple",5) → ("grape",3) → null
  [4] → null

  h("banana") = h("fig") = 1  → チェインで格納
```

```python
class HashTableChaining:
    """チェイン法によるハッシュテーブル

    各バケットが連結リスト（Pythonではリスト）を持ち、
    衝突したキーを同一バケット内に格納する。
    """
    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        """O(1) 平均"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx][i] = (key, value)
                return
        self.buckets[idx].append((key, value))
        self.count += 1
        if self.count / self.size > 0.75:
            self._rehash()

    def get(self, key):
        """O(1) 平均"""
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        raise KeyError(key)

    def delete(self, key):
        """O(1) 平均"""
        idx = self._hash(key)
        for i, (k, v) in enumerate(self.buckets[idx]):
            if k == key:
                self.buckets[idx].pop(i)
                self.count -= 1
                return v
        raise KeyError(key)

    def contains(self, key):
        """キーの存在確認 — O(1) 平均"""
        idx = self._hash(key)
        return any(k == key for k, _ in self.buckets[idx])

    def keys(self):
        """全キーの列挙 — O(n + m)"""
        result = []
        for bucket in self.buckets:
            for k, v in bucket:
                result.append(k)
        return result

    def items(self):
        """全キー・値ペアの列挙 — O(n + m)"""
        result = []
        for bucket in self.buckets:
            for k, v in bucket:
                result.append((k, v))
        return result

    def _rehash(self):
        """テーブルサイズを倍増し全要素を再配置"""
        old = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        for bucket in old:
            for key, value in bucket:
                self.put(key, value)

    def load_factor(self):
        """現在のロードファクターを返す"""
        return self.count / self.size

    def __repr__(self):
        items = []
        for bucket in self.buckets:
            for k, v in bucket:
                items.append(f"{k!r}: {v!r}")
        return "{" + ", ".join(items) + "}"

# 使用例
ht = HashTableChaining()
ht.put("name", "Alice")
ht.put("age", 30)
ht.put("city", "Tokyo")
print(ht.get("name"))     # "Alice"
print(ht.contains("age")) # True
print(ht.keys())          # ["name", "age", "city"]
ht.delete("age")
print(ht.contains("age")) # False
print(ht.load_factor())   # 0.125
```

### 3.2 チェイン法の改良: 赤黒木チェイン（Java 8+ HashMap）

```python
class HashTableTreeChaining:
    """Java 8+ HashMap の戦略を模倣:
    - バケット内要素が少ない(< 8): 連結リスト
    - バケット内要素が多い(>= 8): 平衡木（赤黒木）に変換

    最悪ケースが O(n) から O(log n) に改善される
    """
    TREEIFY_THRESHOLD = 8
    UNTREEIFY_THRESHOLD = 6

    def __init__(self, size=16):
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0

    def _hash(self, key):
        # Java と同様に上位ビットを下位ビットに混ぜる
        h = hash(key)
        return ((h >> 16) ^ h) % self.size

    def put(self, key, value):
        idx = self._hash(key)
        bucket = self.buckets[idx]

        # 既存キーの更新
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        bucket.append((key, value))
        self.count += 1

        # バケットが閾値を超えたら木化（簡易版: ソート済みリストで代用）
        if len(bucket) >= self.TREEIFY_THRESHOLD:
            self.buckets[idx] = sorted(bucket, key=lambda x: hash(x[0]))

        if self.count / self.size > 0.75:
            self._rehash()

    def get(self, key):
        idx = self._hash(key)
        bucket = self.buckets[idx]

        if len(bucket) >= self.TREEIFY_THRESHOLD:
            # 木化されたバケット: 二分探索（O(log n)）
            return self._tree_search(bucket, key)

        # リストバケット: 線形探索
        for k, v in bucket:
            if k == key:
                return v
        raise KeyError(key)

    def _tree_search(self, sorted_bucket, key):
        """ソート済みバケットでの二分探索"""
        target_hash = hash(key)
        lo, hi = 0, len(sorted_bucket) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            mid_hash = hash(sorted_bucket[mid][0])
            if mid_hash == target_hash and sorted_bucket[mid][0] == key:
                return sorted_bucket[mid][1]
            elif mid_hash < target_hash:
                lo = mid + 1
            else:
                hi = mid - 1
        raise KeyError(key)

    def _rehash(self):
        old = self.buckets
        self.size *= 2
        self.buckets = [[] for _ in range(self.size)]
        self.count = 0
        for bucket in old:
            for key, value in bucket:
                self.put(key, value)
```

### 3.3 オープンアドレス法（線形探索法）

```
h("apple") = 3, h("grape") = 3 → 衝突!

線形探索法: 次の空きスロットを探す
  [0] → null
  [1] → null
  [2] → null
  [3] → ("apple", 5)   ← h("apple") = 3
  [4] → ("grape", 3)   ← h("grape") = 3 → 衝突 → 3+1=4
  [5] → null
```

```python
class HashTableLinearProbing:
    """線形探索法 (Linear Probing) によるオープンアドレスハッシュテーブル

    探索シーケンス: h(k), h(k)+1, h(k)+2, ...
    利点: キャッシュ効率が良い（メモリの連続領域を走査）
    欠点: クラスタリング（一次クラスタリング）が発生しやすい
    """
    DELETED = object()

    def __init__(self, size=16):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def put(self, key, value):
        if self.count / self.size > 0.5:
            self._rehash()
        idx = self._hash(key)
        while self.keys[idx] is not None and self.keys[idx] is not self.DELETED:
            if self.keys[idx] == key:
                self.values[idx] = value
                return
            idx = (idx + 1) % self.size
        self.keys[idx] = key
        self.values[idx] = value
        self.count += 1

    def get(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.size
        raise KeyError(key)

    def delete(self, key):
        idx = self._hash(key)
        while self.keys[idx] is not None:
            if self.keys[idx] == key:
                self.keys[idx] = self.DELETED
                self.values[idx] = None
                self.count -= 1
                return
            idx = (idx + 1) % self.size
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        self.size *= 2
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)
```

### 3.4 二次探索法（Quadratic Probing）

```python
class HashTableQuadraticProbing:
    """二次探索法: 一次クラスタリングを緩和

    探索シーケンス: h(k), h(k)+1², h(k)+2², h(k)+3², ...

    テーブルサイズが素数かつ α < 0.5 なら、
    最初の m/2 個の探索位置が全て異なることが保証される
    """
    DELETED = object()

    def __init__(self, size=17):  # 素数を推奨
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _hash(self, key):
        return hash(key) % self.size

    def _probe(self, key):
        """二次探索のジェネレータ"""
        idx = self._hash(key)
        for i in range(self.size):
            yield (idx + i * i) % self.size

    def put(self, key, value):
        if self.count / self.size > 0.5:
            self._rehash()

        first_deleted = None
        for idx in self._probe(key):
            if self.keys[idx] is None:
                target = first_deleted if first_deleted is not None else idx
                self.keys[target] = key
                self.values[target] = value
                self.count += 1
                return
            elif self.keys[idx] is self.DELETED:
                if first_deleted is None:
                    first_deleted = idx
            elif self.keys[idx] == key:
                self.values[idx] = value
                return

    def get(self, key):
        for idx in self._probe(key):
            if self.keys[idx] is None:
                raise KeyError(key)
            if self.keys[idx] == key:
                return self.values[idx]
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        # 次の素数サイズに拡張
        self.size = self._next_prime(self.size * 2)
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)

    @staticmethod
    def _next_prime(n):
        """n以上の最小の素数を返す"""
        if n <= 2:
            return 2
        candidate = n if n % 2 != 0 else n + 1
        while True:
            if all(candidate % i != 0 for i in range(3, int(candidate**0.5) + 1, 2)):
                return candidate
            candidate += 2
```

### 3.5 ダブルハッシング（Double Hashing）

```python
class HashTableDoubleHashing:
    """ダブルハッシング: 二次クラスタリングも解消

    探索シーケンス: h1(k), h1(k)+h2(k), h1(k)+2*h2(k), ...

    h2(k) がテーブルサイズと互いに素である必要がある
    → テーブルサイズを素数にするか、h2 の値域を制限する
    """
    DELETED = object()

    def __init__(self, size=17):
        self.size = size
        self.keys = [None] * size
        self.values = [None] * size
        self.count = 0

    def _h1(self, key):
        """主ハッシュ関数"""
        return hash(key) % self.size

    def _h2(self, key):
        """副ハッシュ関数（0 にならないこと）
        h2(k) = prime - (k mod prime) で prime < size の素数
        """
        prime = self.size - 2  # size が素数なら size-2 も素数の可能性が高い
        return prime - (hash(key) % prime)

    def _probe(self, key):
        idx = self._h1(key)
        step = self._h2(key)
        for i in range(self.size):
            yield (idx + i * step) % self.size

    def put(self, key, value):
        if self.count / self.size > 0.5:
            self._rehash()
        for idx in self._probe(key):
            if self.keys[idx] is None or self.keys[idx] is self.DELETED:
                self.keys[idx] = key
                self.values[idx] = value
                self.count += 1
                return
            elif self.keys[idx] == key:
                self.values[idx] = value
                return

    def get(self, key):
        for idx in self._probe(key):
            if self.keys[idx] is None:
                raise KeyError(key)
            if self.keys[idx] == key:
                return self.values[idx]
        raise KeyError(key)

    def _rehash(self):
        old_keys, old_values = self.keys, self.values
        self.size = self._next_prime(self.size * 2)
        self.keys = [None] * self.size
        self.values = [None] * self.size
        self.count = 0
        for k, v in zip(old_keys, old_values):
            if k is not None and k is not self.DELETED:
                self.put(k, v)

    @staticmethod
    def _next_prime(n):
        if n <= 2:
            return 2
        candidate = n if n % 2 != 0 else n + 1
        while True:
            if all(candidate % i != 0 for i in range(3, int(candidate**0.5) + 1, 2)):
                return candidate
            candidate += 2
```

### 3.6 Robin Hood ハッシング

```python
class RobinHoodHashTable:
    """Robin Hood ハッシング: Rust の HashMap が採用

    原理: 「貧しい者（探索距離が長い要素）から盗んで
          富める者（探索距離が短い要素）に与える」

    - 挿入時に、既存要素より探索距離が長い場合に入れ替え
    - 最大探索距離が平均化され、最悪ケースが改善される
    - 期待最大探索距離: O(log n)
    """
    EMPTY = None

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity
        self.distances = [0] * capacity  # 各要素の理想位置からの距離

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size >= self.capacity * 7 // 8:  # α < 7/8
            self._rehash()

        idx = self._hash(key)
        dist = 0

        while True:
            if self.keys[idx] is self.EMPTY:
                # 空きスロットに挿入
                self.keys[idx] = key
                self.values[idx] = value
                self.distances[idx] = dist
                self.size += 1
                return

            if self.keys[idx] == key:
                # 既存キーの更新
                self.values[idx] = value
                return

            # Robin Hood: 既存要素より探索距離が長ければ入れ替え
            if dist > self.distances[idx]:
                key, self.keys[idx] = self.keys[idx], key
                value, self.values[idx] = self.values[idx], value
                dist, self.distances[idx] = self.distances[idx], dist

            idx = (idx + 1) % self.capacity
            dist += 1

    def get(self, key):
        idx = self._hash(key)
        dist = 0

        while True:
            if self.keys[idx] is self.EMPTY:
                raise KeyError(key)
            if dist > self.distances[idx]:
                # Robin Hood の性質: これ以上探しても見つからない
                raise KeyError(key)
            if self.keys[idx] == key:
                return self.values[idx]
            idx = (idx + 1) % self.capacity
            dist += 1

    def _rehash(self):
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.distances = [0] * self.capacity
        self.size = 0
        for k, v in zip(old_keys, old_values):
            if k is not self.EMPTY:
                self.put(k, v)
```

### 3.7 カッコウハッシング（Cuckoo Hashing）

```python
class CuckooHashTable:
    """カッコウハッシング: 最悪ケース O(1) の検索を保証

    原理:
    - 2つのハッシュ関数 h1, h2 と2つのテーブル T1, T2
    - 各キーは T1[h1(k)] または T2[h2(k)] のいずれかに格納
    - 検索: 2箇所を見るだけ → O(1) 最悪
    - 挿入: 既存要素を「追い出し」て再配置（カッコウの巣の乗っ取りと同じ）

    サイクルが発生した場合はリハッシュが必要
    """
    def __init__(self, size=16):
        self.size = size
        self.table1 = [None] * size
        self.table2 = [None] * size
        self.count = 0
        self._max_kicks = 500  # 無限ループ防止

    def _h1(self, key):
        return hash(key) % self.size

    def _h2(self, key):
        return hash(key * 2654435761) % self.size  # 別のハッシュ関数

    def get(self, key):
        """O(1) 最悪 — 2箇所をチェックするだけ"""
        idx1 = self._h1(key)
        if self.table1[idx1] is not None and self.table1[idx1][0] == key:
            return self.table1[idx1][1]

        idx2 = self._h2(key)
        if self.table2[idx2] is not None and self.table2[idx2][0] == key:
            return self.table2[idx2][1]

        raise KeyError(key)

    def put(self, key, value):
        # 既存キーの更新チェック
        idx1 = self._h1(key)
        if self.table1[idx1] is not None and self.table1[idx1][0] == key:
            self.table1[idx1] = (key, value)
            return
        idx2 = self._h2(key)
        if self.table2[idx2] is not None and self.table2[idx2][0] == key:
            self.table2[idx2] = (key, value)
            return

        # 新規挿入
        entry = (key, value)
        for _ in range(self._max_kicks):
            # テーブル1に挿入を試みる
            idx1 = self._h1(entry[0])
            if self.table1[idx1] is None:
                self.table1[idx1] = entry
                self.count += 1
                return

            # 既存要素を追い出す
            entry, self.table1[idx1] = self.table1[idx1], entry

            # テーブル2に挿入を試みる
            idx2 = self._h2(entry[0])
            if self.table2[idx2] is None:
                self.table2[idx2] = entry
                self.count += 1
                return

            # 既存要素を追い出す
            entry, self.table2[idx2] = self.table2[idx2], entry

        # サイクル発生 → リハッシュして再試行
        self._rehash()
        self.put(entry[0], entry[1])

    def _rehash(self):
        old_t1 = self.table1
        old_t2 = self.table2
        self.size *= 2
        self.table1 = [None] * self.size
        self.table2 = [None] * self.size
        self.count = 0
        for entry in old_t1 + old_t2:
            if entry is not None:
                self.put(entry[0], entry[1])
```

### 3.8 衝突解決法の詳細比較

```
各探索法のクラスタリング特性:

  線形探索法 (Linear Probing):
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │   │ X │ X │ X │ X │ X │   │   │  ← 一次クラスタ
  └───┴───┴───┴───┴───┴───┴───┴───┘
  連続した占有スロットが成長 → 新しい衝突がクラスタに吸収される

  二次探索法 (Quadratic Probing):
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │   │ X │   │ X │   │   │ X │   │  ← 散在
  └───┴───┴───┴───┴───┴───┴───┴───┘
  一次クラスタは解消。同一ハッシュ値のキーは同じ経路を辿る（二次クラスタ）

  ダブルハッシング:
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ X │   │   │ X │   │ X │   │   │  ← ほぼランダム
  └───┴───┴───┴───┴───┴───┴───┴───┘
  キーごとにステップが異なる → クラスタリングなし
```

---

## 4. ロードファクター

```
ロードファクター α = 要素数 / バケット数

α の影響:
  ┌────────────────────────────────────┐
  │                                    │
  │ 探索時間                          │
  │ ▲                                 │
  │ │                        ╱        │
  │ │                     ╱           │
  │ │                  ╱   チェイン法  │
  │ │               ╱                 │
  │ │          ╱╱╱  オープンアドレス   │
  │ │    ╱╱╱                          │
  │ │╱╱                               │
  │ ┼──────────────────────► α        │
  │ 0   0.25  0.5  0.75  1.0         │
  │                                    │
  │ 推奨 α:                           │
  │   チェイン法: α < 0.75            │
  │   オープンアドレス法: α < 0.5      │
  └────────────────────────────────────┘
```

### 4.1 理論的な探索コスト

```python
def expected_probes_chaining(alpha):
    """チェイン法の期待探索回数
    成功時: 1 + α/2
    失敗時: 1 + α
    """
    return {
        "successful": 1 + alpha / 2,
        "unsuccessful": 1 + alpha
    }

def expected_probes_linear(alpha):
    """線形探索法の期待探索回数
    成功時: (1/2)(1 + 1/(1-α))
    失敗時: (1/2)(1 + 1/(1-α)²)
    """
    if alpha >= 1:
        return {"successful": float('inf'), "unsuccessful": float('inf')}
    return {
        "successful": 0.5 * (1 + 1 / (1 - alpha)),
        "unsuccessful": 0.5 * (1 + 1 / (1 - alpha) ** 2)
    }

def expected_probes_double(alpha):
    """ダブルハッシングの期待探索回数
    成功時: (1/α) * ln(1/(1-α))
    失敗時: 1/(1-α)
    """
    import math
    if alpha >= 1:
        return {"successful": float('inf'), "unsuccessful": float('inf')}
    if alpha == 0:
        return {"successful": 1, "unsuccessful": 1}
    return {
        "successful": (1 / alpha) * math.log(1 / (1 - alpha)),
        "unsuccessful": 1 / (1 - alpha)
    }

# α ごとの比較
for alpha in [0.25, 0.5, 0.75, 0.9]:
    print(f"\n--- α = {alpha} ---")
    chain = expected_probes_chaining(alpha)
    linear = expected_probes_linear(alpha)
    double = expected_probes_double(alpha)
    print(f"チェイン法:   成功 {chain['successful']:.2f}, 失敗 {chain['unsuccessful']:.2f}")
    print(f"線形探索法:   成功 {linear['successful']:.2f}, 失敗 {linear['unsuccessful']:.2f}")
    print(f"ダブルハッシュ: 成功 {double['successful']:.2f}, 失敗 {double['unsuccessful']:.2f}")

# 出力:
# --- α = 0.25 ---
# チェイン法:   成功 1.12, 失敗 1.25
# 線形探索法:   成功 1.17, 失敗 1.39
# ダブルハッシュ: 成功 1.15, 失敗 1.33
#
# --- α = 0.5 ---
# チェイン法:   成功 1.25, 失敗 1.50
# 線形探索法:   成功 1.50, 失敗 2.50
# ダブルハッシュ: 成功 1.39, 失敗 2.00
#
# --- α = 0.75 ---
# チェイン法:   成功 1.38, 失敗 1.75
# 線形探索法:   成功 2.50, 失敗 8.50
# ダブルハッシュ: 成功 1.85, 失敗 4.00
#
# --- α = 0.9 ---
# チェイン法:   成功 1.45, 失敗 1.90
# 線形探索法:   成功 5.50, 失敗 50.50
# ダブルハッシュ: 成功 2.56, 失敗 10.00
```

### 4.2 リハッシュの償却分析

```python
class AmortizedRehashDemo:
    """リハッシュの償却コスト分析

    n 回の挿入の総コスト:
    - 通常の挿入: n × O(1) = O(n)
    - リハッシュ:  1 + 2 + 4 + ... + n = O(n)（等比数列の和）

    総コスト O(n) ÷ n 回 = O(1) 償却

    つまりリハッシュが O(n) かかっても、
    償却分析では各挿入が O(1) になる
    """
    def __init__(self):
        self.size = 2
        self.count = 0
        self.total_cost = 0
        self.rehash_count = 0

    def insert_simulation(self, n):
        """n回の挿入をシミュレート"""
        for i in range(n):
            self.count += 1
            self.total_cost += 1  # 通常の挿入コスト

            if self.count > self.size * 0.75:
                # リハッシュ: 全要素の再挿入コスト
                self.total_cost += self.count
                self.size *= 2
                self.rehash_count += 1

        print(f"挿入回数: {n}")
        print(f"リハッシュ回数: {self.rehash_count}")
        print(f"最終テーブルサイズ: {self.size}")
        print(f"総コスト: {self.total_cost}")
        print(f"償却コスト (平均): {self.total_cost / n:.2f}")

demo = AmortizedRehashDemo()
demo.insert_simulation(1000000)
# 挿入回数: 1000000
# リハッシュ回数: 20
# 最終テーブルサイズ: 2097152
# 総コスト: ~3000000
# 償却コスト (平均): ~3.00  → O(1) 定数
```

---

## 5. 言語別ハッシュテーブル実装の詳細

### 表1: 衝突解決法の比較

| 特性 | チェイン法 | オープンアドレス法 |
|------|-----------|------------------|
| 構造 | 配列 + リスト | 配列のみ |
| メモリ | ポインタ分余分 | 密にパック |
| 最悪検索 | O(n) | O(n) |
| 削除 | 容易 | DELETED マーカー必要 |
| キャッシュ効率 | 低い | 高い |
| 推奨 α | < 0.75 | < 0.5 |
| 実装 | 簡単 | やや複雑 |

### 表2: 言語別ハッシュテーブル実装

| 言語 | 型名 | 衝突解決 | 初期容量 | 最大 α |
|------|------|---------|---------|--------|
| Python | dict | オープンアドレス | 8 | 2/3 |
| Java | HashMap | チェイン(+木) | 16 | 0.75 |
| C++ | unordered_map | チェイン | 実装依存 | 1.0 |
| Go | map | チェイン(バケット) | 実装依存 | 6.5 |
| Rust | HashMap | Robin Hood | 実装依存 | 7/8 |
| C# | Dictionary | チェイン | 3 | 1.0 |
| Ruby | Hash | オープンアドレス | 8 | 実装依存 |

### 5.1 Python dict の内部構造

```python
# Python 3.7+ の dict はコンパクトな2層構造:
#
# 1. ハッシュインデックス配列 (sparse):
#    [_, _, 0, _, 1, _, 2, _]  ← 実際のエントリへのインデックス
#
# 2. エントリ配列 (dense, 挿入順):
#    [("apple", 5), ("banana", 2), ("cherry", 8)]
#
# メリット:
# - 挿入順序が保持される（3.7+ で保証）
# - メモリ効率が良い（以前の実装比で20-25%削減）
# - イテレーションが高速（密な配列を走査するだけ）

# dict の内部サイズ変化を観察
import sys

d = {}
print(f"空の dict: {sys.getsizeof(d)} bytes")  # 64 bytes

for i in range(10):
    d[f"key_{i}"] = i
    print(f"{i+1} 要素: {sys.getsizeof(d)} bytes")

# Python dict のハッシュ探索法:
# - オープンアドレス法（二次探索に近い）
# - 探索シーケンス: j = ((5*j) + 1 + perturb) % size
#   perturb は初期ハッシュ値で、反復ごとに右シフトされる
# - この方法でテーブル全体を走査できる
```

### 5.2 Java HashMap の内部構造

```java
// Java 8+ HashMap の特徴:
//
// 1. 初期容量: 16、ロードファクター: 0.75
// 2. 容量は常に2のべき乗
// 3. バケット内の要素数が8以上 → 赤黒木に変換 (treeify)
// 4. バケット内の要素数が6以下 → リストに戻す (untreeify)
// 5. ハッシュ値の上位ビットを下位に混ぜる (perturbation)
//
// 主要メソッドの実装概要:

// static final int hash(Object key) {
//     int h;
//     return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
// }
//
// // インデックス計算: (n - 1) & hash
// // n が2のべき乗なので、& 演算で mod を高速化
//
// // リサイズ時: 各要素のビットを1つチェックするだけで
// // 新しい位置が決まる（同じか +oldCapacity）
```

### 5.3 各言語での使用例

```python
# === Python ===
# dict: ハッシュテーブル
d = {"name": "Alice", "age": 30}
d["city"] = "Tokyo"          # O(1) 挿入
print(d.get("name", "N/A"))  # O(1) 検索（デフォルト値付き）
del d["age"]                  # O(1) 削除

# defaultdict: デフォルト値付き
from collections import defaultdict
word_count = defaultdict(int)
for w in "hello world hello".split():
    word_count[w] += 1
print(dict(word_count))  # {'hello': 2, 'world': 1}

# Counter: 出現頻度カウント
from collections import Counter
freq = Counter("mississippi")
print(freq.most_common(3))  # [('s', 4), ('i', 4), ('p', 2)]

# OrderedDict: 挿入順序保持（3.7+のdictと同じだが、順序比較が可能）
from collections import OrderedDict
od = OrderedDict()
od["b"] = 2
od["a"] = 1
od.move_to_end("b")  # 末尾に移動（LRUキャッシュに便利）
```

```go
// === Go ===
package main

import "fmt"

func main() {
    // map: ハッシュテーブル
    m := map[string]int{
        "apple":  5,
        "banana": 2,
    }

    // 挿入
    m["cherry"] = 8

    // 検索（2値返し）
    if val, ok := m["apple"]; ok {
        fmt.Println(val) // 5
    }

    // 削除
    delete(m, "banana")

    // イテレーション（順序は非決定的）
    for k, v := range m {
        fmt.Printf("%s: %d\n", k, v)
    }
}
```

```rust
// === Rust ===
use std::collections::HashMap;

fn main() {
    let mut scores = HashMap::new();

    // 挿入
    scores.insert("Alice", 10);
    scores.insert("Bob", 20);

    // 検索
    if let Some(score) = scores.get("Alice") {
        println!("Alice: {}", score);
    }

    // entry API: キーが無ければ挿入
    scores.entry("Charlie").or_insert(30);

    // 値の更新
    let count = scores.entry("Alice").or_insert(0);
    *count += 5;

    // イテレーション
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
}
```

```typescript
// === TypeScript ===
// Object: 文字列キーのみ
const obj: Record<string, number> = { apple: 5, banana: 2 };

// Map: 任意の型をキーに使える
const map = new Map<string, number>();
map.set("apple", 5);
map.set("banana", 2);
console.log(map.get("apple"));  // 5
console.log(map.has("cherry")); // false
console.log(map.size);          // 2

// Set: 値の集合
const set = new Set<number>([1, 2, 3, 2, 1]);
console.log(set.size);  // 3

// WeakMap: ガベージコレクション対応
const weakMap = new WeakMap<object, string>();
let key = {};
weakMap.set(key, "value");
// key = null; でGC可能
```

---

## 6. 実務応用パターン

### 6.1 LRU キャッシュの実装

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used キャッシュ

    OrderedDict を使って O(1) の get/put を実現。
    アクセスされた要素を末尾に移動し、
    容量超過時は先頭（最も古い）要素を削除する。
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        """O(1)"""
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)  # 最近使用としてマーク
        return self.cache[key]

    def put(self, key, value):
        """O(1)"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # 最も古い要素を削除

# 使用例
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # 1（"a" が最近使用に）
cache.put("d", 4)      # 容量超過 → "b" が削除される
print(cache.get("b"))   # -1（削除済み）
```

### 6.2 一致するペアの検出

```python
def find_pairs_with_sum(arr, target):
    """和が target になるペアを全て返す — O(n)

    ハッシュテーブルで補数を管理する
    """
    seen = {}
    pairs = []
    for num in arr:
        complement = target - num
        if complement in seen:
            pairs.append((complement, num))
        seen[num] = True
    return pairs

print(find_pairs_with_sum([1, 5, 7, -1, 5], 6))
# [(1, 5), (-1, 7), (1, 5)]
```

### 6.3 文字列の同型判定（Isomorphic Strings）

```python
def is_isomorphic(s: str, t: str) -> bool:
    """2つの文字列が同型かどうかを判定 — O(n)

    "egg" と "add" → True (e→a, g→d)
    "foo" と "bar" → False
    """
    if len(s) != len(t):
        return False

    s_to_t = {}
    t_to_s = {}

    for cs, ct in zip(s, t):
        if cs in s_to_t:
            if s_to_t[cs] != ct:
                return False
        else:
            s_to_t[cs] = ct

        if ct in t_to_s:
            if t_to_s[ct] != cs:
                return False
        else:
            t_to_s[ct] = cs

    return True

print(is_isomorphic("egg", "add"))   # True
print(is_isomorphic("foo", "bar"))   # False
print(is_isomorphic("paper", "title"))  # True
```

### 6.4 サブ配列の和が k になる個数

```python
def subarray_sum(nums, k):
    """和が k になる連続部分配列の個数 — O(n)

    累積和とハッシュテーブルを組み合わせる
    """
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # 累積和 0 は1回（空の接頭辞）

    for num in nums:
        prefix_sum += num
        # prefix_sum - k が過去に存在すれば、
        # その区間の和が k
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count

print(subarray_sum([1, 1, 1], 2))      # 2
print(subarray_sum([1, 2, 3], 3))      # 2 ([1,2] と [3])
print(subarray_sum([1, -1, 1, 1], 2))  # 3
```

### 6.5 最長連続列（Longest Consecutive Sequence）

```python
def longest_consecutive(nums):
    """最長の連続する整数列の長さを返す — O(n)

    例: [100, 4, 200, 1, 3, 2] → 4 ([1, 2, 3, 4])

    set で O(1) 検索を実現し、各連続列の開始点からのみカウント
    """
    num_set = set(nums)
    max_length = 0

    for num in num_set:
        # num-1 が存在しない → この num は連続列の開始点
        if num - 1 not in num_set:
            current = num
            length = 1
            while current + 1 in num_set:
                current += 1
                length += 1
            max_length = max(max_length, length)

    return max_length

print(longest_consecutive([100, 4, 200, 1, 3, 2]))  # 4
print(longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]))  # 9
```

### 6.6 ブルームフィルタ

```python
import hashlib

class BloomFilter:
    """ブルームフィルタ: 空間効率の良い確率的データ構造

    - 「要素が存在しない」は100%正確（偽陰性なし）
    - 「要素が存在する」は偽陽性の可能性あり
    - 削除不可（Counting Bloom Filter で対応）

    用途: スペルチェック、キャッシュフィルタ、DBクエリ最適化
    """
    def __init__(self, size=1000, num_hashes=3):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [False] * size
        self.count = 0

    def _hashes(self, item):
        """複数のハッシュ値を生成"""
        results = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            results.append(int(h, 16) % self.size)
        return results

    def add(self, item):
        """要素を追加 — O(k), k = ハッシュ関数の数"""
        for idx in self._hashes(item):
            self.bit_array[idx] = True
        self.count += 1

    def might_contain(self, item):
        """要素が存在するかもしれない場合 True
        False なら確実に存在しない
        """
        return all(self.bit_array[idx] for idx in self._hashes(item))

    def false_positive_rate(self):
        """理論的な偽陽性確率
        p ≈ (1 - e^(-kn/m))^k
        k: ハッシュ関数数, n: 要素数, m: ビット配列サイズ
        """
        import math
        k = self.num_hashes
        n = self.count
        m = self.size
        if n == 0:
            return 0.0
        return (1 - math.exp(-k * n / m)) ** k

# 使用例
bf = BloomFilter(size=10000, num_hashes=5)
for word in ["apple", "banana", "cherry", "date"]:
    bf.add(word)

print(bf.might_contain("apple"))   # True
print(bf.might_contain("fig"))     # False（確実に存在しない）
print(bf.might_contain("grape"))   # False（または偽陽性の可能性）
print(f"偽陽性確率: {bf.false_positive_rate():.6f}")
```

### 6.7 一貫性ハッシュ（Consistent Hashing）

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """一貫性ハッシュ: 分散システムのデータ分散に使用

    ノードの追加/削除時に再配置されるキーが最小限になる。
    - 通常のハッシュ: ノード変更 → ほぼ全キーが再配置
    - 一貫性ハッシュ: ノード変更 → K/N 個のキーのみ再配置
      (K: 全キー数, N: ノード数)

    用途: CDN、分散キャッシュ（Memcached, Redis Cluster）、
          分散DB のパーティショニング
    """
    def __init__(self, nodes=None, replicas=100):
        self.replicas = replicas  # 仮想ノード数
        self.ring = []            # ソート済みハッシュ値
        self.hash_to_node = {}    # ハッシュ値 → ノード名

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        """ノードをリングに追加"""
        for i in range(self.replicas):
            virtual_key = f"{node}:replica:{i}"
            h = self._hash(virtual_key)
            self.ring.append(h)
            self.hash_to_node[h] = node
        self.ring.sort()

    def remove_node(self, node):
        """ノードをリングから削除"""
        for i in range(self.replicas):
            virtual_key = f"{node}:replica:{i}"
            h = self._hash(virtual_key)
            self.ring.remove(h)
            del self.hash_to_node[h]

    def get_node(self, key):
        """キーが割り当てられるノードを返す"""
        if not self.ring:
            return None
        h = self._hash(key)
        idx = bisect_right(self.ring, h)
        if idx == len(self.ring):
            idx = 0
        return self.hash_to_node[self.ring[idx]]

# 使用例
ch = ConsistentHash(["server-1", "server-2", "server-3"])
for key in ["user:100", "user:200", "user:300", "user:400", "user:500"]:
    print(f"{key} → {ch.get_node(key)}")

# ノード追加時の影響を確認
print("\n--- server-4 を追加 ---")
ch.add_node("server-4")
for key in ["user:100", "user:200", "user:300", "user:400", "user:500"]:
    print(f"{key} → {ch.get_node(key)}")
# 一部のキーだけが再配置される
```

---

## 7. ハッシュテーブルのセキュリティ

### 7.1 ハッシュ DoS 攻撃と対策

```python
# === ハッシュ DoS 攻撃 ===
# 攻撃者がハッシュ衝突を意図的に起こすキーを大量に送信
# → O(n²) の処理時間でサーバーを DoS する
#
# 対策:
# 1. Python 3.3+: ハッシュランダム化 (PYTHONHASHSEED)
# 2. SipHash: Python 3.4+ のデフォルトハッシュ関数
# 3. ユニバーサルハッシュ: ランダムなハッシュ関数を使用
# 4. リクエストサイズの制限

# Python のハッシュランダム化を確認
import sys
print(f"ハッシュランダム化フラグ: {sys.flags.hash_randomization}")

# PYTHONHASHSEED 環境変数で制御
# PYTHONHASHSEED=0  → ランダム化無効（再現性が必要なテスト用）
# PYTHONHASHSEED=42 → 固定シード
# 未設定             → ランダムシード（デフォルト、推奨）

# SipHash の特徴:
# - 暗号学的に安全ではないが、衝突攻撃に耐性がある
# - 短い入力でも高速（128bit のキーを使用）
# - Python, Rust, Ruby などが採用
```

### 7.2 ハッシュテーブルの時間計算量攻撃への対策

```python
# 定数時間比較（タイミング攻撃対策）
import hmac

def safe_compare(a: str, b: str) -> bool:
    """定数時間での文字列比較

    通常の == は一致しない文字を見つけた時点で即座に返すため、
    処理時間の差でパスワードが推測される可能性がある。
    """
    return hmac.compare_digest(a.encode(), b.encode())

# BAD: タイミング情報が漏れる
# if user_token == stored_token: ...

# GOOD: 定数時間比較
# if safe_compare(user_token, stored_token): ...
```

---

## 8. パフォーマンスベンチマーク

```python
import time
import random
import string

def benchmark_hash_tables():
    """各実装のパフォーマンス比較"""
    n = 100000
    keys = [''.join(random.choices(string.ascii_letters, k=10)) for _ in range(n)]
    values = list(range(n))

    # === Python dict ===
    start = time.perf_counter()
    d = {}
    for k, v in zip(keys, values):
        d[k] = v
    dict_insert = time.perf_counter() - start

    start = time.perf_counter()
    for k in keys:
        _ = d[k]
    dict_lookup = time.perf_counter() - start

    # === チェイン法 ===
    start = time.perf_counter()
    ht = HashTableChaining(size=n * 2)
    for k, v in zip(keys[:10000], values[:10000]):  # 小規模で比較
        ht.put(k, v)
    chain_insert = time.perf_counter() - start

    print(f"Python dict - 挿入 {n} 件: {dict_insert:.4f}s")
    print(f"Python dict - 検索 {n} 件: {dict_lookup:.4f}s")
    print(f"チェイン法  - 挿入 10000件: {chain_insert:.4f}s")
    print(f"\nPython dict は C 実装で高度に最適化されている")

# benchmark_hash_tables()
```

---

## 9. アンチパターン

### アンチパターン1: mutable オブジェクトをキーにする

```python
# BAD: リストはハッシュ不可
d = {}
key = [1, 2, 3]
# d[key] = "value"  # TypeError: unhashable type: 'list'

# GOOD: タプルに変換
d[tuple(key)] = "value"

# BAD: カスタムクラスの __hash__ を変更可能フィールドで定義
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __hash__(self):
        return hash((self.x, self.y))
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p = Point(1, 2)
d = {p: "origin"}
p.x = 10  # ハッシュ値が変わる → d[p] が見つからなくなる!

# GOOD: frozen（不変）にする
from dataclasses import dataclass

@dataclass(frozen=True)
class FrozenPoint:
    x: int
    y: int
    # frozen=True で __hash__ と __eq__ が自動生成
    # フィールドは変更不可

fp = FrozenPoint(1, 2)
d = {fp: "origin"}
# fp.x = 10  # FrozenInstanceError が発生
```

### アンチパターン2: 衝突が多いハッシュ関数

```python
# BAD: 全てのキーが同じバケットに → O(n) 探索
def terrible_hash(key, size):
    return 0  # 全て index 0

# BAD: 下位ビットだけ使う
def bad_hash(key, size):
    return key & 0xF  # 16通りしかない

# BAD: 偶数サイズのテーブルで偶数キーだけが来る
def biased_hash(key, size):
    return key % size  # size=16, key=偶数 → 偶数インデックスのみ使用

# GOOD: Python の hash() + 素数サイズ
def good_hash(key, size):
    return hash(key) % size

# GOOD: 上位ビットも活用（Java 方式）
def better_hash(key, size):
    h = hash(key)
    h ^= (h >> 16)  # 上位ビットを下位に混ぜる
    return h % size
```

### アンチパターン3: 不要なハッシュテーブル使用

```python
# BAD: 小規模データにハッシュテーブルを使用
# → リストの線形探索の方が高速（定数係数が小さい）
small_data = {"a": 1, "b": 2, "c": 3}  # 3要素

# 3要素程度なら、リストの線形探索で十分
# ハッシュ計算のオーバーヘッドの方が大きい
small_list = [("a", 1), ("b", 2), ("c", 3)]

# BAD: キーの範囲が小さいのにハッシュテーブル
# → 直接アドレスも検討
counts = {}
for x in data:  # data が 0-255 の範囲なら
    counts[x] = counts.get(x, 0) + 1

# GOOD: 直接アドレステーブル
counts = [0] * 256
for x in data:
    counts[x] += 1
```

### アンチパターン4: dict のイテレーション中の変更

```python
# BAD: イテレーション中に要素を追加/削除
d = {"a": 1, "b": 2, "c": 3}
# for k in d:
#     if d[k] < 2:
#         del d[k]  # RuntimeError: dictionary changed size during iteration

# GOOD: コピーを作ってからイテレーション
d = {"a": 1, "b": 2, "c": 3}
for k in list(d.keys()):
    if d[k] < 2:
        del d[k]
print(d)  # {'b': 2, 'c': 3}

# GOOD: 辞書内包表記で新しい辞書を作る
d = {"a": 1, "b": 2, "c": 3}
d = {k: v for k, v in d.items() if v >= 2}
print(d)  # {'b': 2, 'c': 3}
```

---

## 10. 面接・競技プログラミングでの頻出パターン

### 10.1 スライディングウィンドウ + ハッシュマップ

```python
def length_of_longest_substring(s: str) -> int:
    """重複のない最長部分文字列の長さ — O(n)

    例: "abcabcbb" → 3 ("abc")
    """
    char_index = {}
    max_length = 0
    start = 0

    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= start:
            start = char_index[char] + 1
        char_index[char] = i
        max_length = max(max_length, i - start + 1)

    return max_length

print(length_of_longest_substring("abcabcbb"))  # 3
print(length_of_longest_substring("bbbbb"))      # 1
print(length_of_longest_substring("pwwkew"))     # 3
```

### 10.2 頻度カウント + Top-K

```python
import heapq
from collections import Counter

def top_k_frequent(nums, k):
    """出現頻度の高い上位 k 個の要素 — O(n log k)

    例: [1,1,1,2,2,3], k=2 → [1, 2]
    """
    freq = Counter(nums)
    return heapq.nlargest(k, freq.keys(), key=freq.get)

print(top_k_frequent([1, 1, 1, 2, 2, 3], 2))  # [1, 2]

# バケットソート版 — O(n)
def top_k_frequent_bucket(nums, k):
    freq = Counter(nums)
    # freq[num] → バケット[freq[num]] に num を追加
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    result = []
    for i in range(len(buckets) - 1, -1, -1):
        result.extend(buckets[i])
        if len(result) >= k:
            return result[:k]
    return result
```

### 10.3 ハッシュマップを使った O(1) データ構造設計

```python
import random

class RandomizedSet:
    """O(1) で insert, remove, getRandom を実現

    dict（値→インデックス） + list（値の配列）を組み合わせる
    """
    def __init__(self):
        self.val_to_idx = {}
        self.vals = []

    def insert(self, val) -> bool:
        """O(1)"""
        if val in self.val_to_idx:
            return False
        self.val_to_idx[val] = len(self.vals)
        self.vals.append(val)
        return True

    def remove(self, val) -> bool:
        """O(1) — 末尾要素と入れ替えて削除"""
        if val not in self.val_to_idx:
            return False
        idx = self.val_to_idx[val]
        last = self.vals[-1]
        # 末尾と入れ替え
        self.vals[idx] = last
        self.val_to_idx[last] = idx
        # 末尾を削除
        self.vals.pop()
        del self.val_to_idx[val]
        return True

    def get_random(self):
        """O(1)"""
        return random.choice(self.vals)

rs = RandomizedSet()
rs.insert(1)
rs.insert(2)
rs.insert(3)
rs.remove(2)
print(rs.get_random())  # 1 or 3
```

---

## 11. FAQ

### Q1: Python の dict はなぜ挿入順序を保持するか？

**A:** Python 3.7 以降、dict はコンパクトな配列構造を採用し、挿入順序を保持する。内部的にはハッシュインデックス配列と、挿入順の密な配列の2層構造になっている。この設計はメモリ効率が20-25%改善され、イテレーションも密な配列を走査するだけなので高速になった。CPython 3.6 で実装され、3.7 で言語仕様として保証された。

### Q2: ハッシュテーブルの最悪ケース O(n) はどう避けるか？

**A:** 複数の対策がある:
1. **Java 8+ HashMap**: 衝突が多いバケットを赤黒木に変換し O(log n) に改善
2. **ユニバーサルハッシュ**: ランダムなハッシュ関数で攻撃的な入力に対応
3. **カッコウハッシュ**: 最悪ケースでも O(1) の検索を保証
4. **Robin Hood ハッシング**: 探索距離を平均化
5. **適切なロードファクター管理**: 閾値を超えたらリハッシュ

### Q3: set と dict の内部構造は同じか？

**A:** Python では set と dict はほぼ同じハッシュテーブル構造。set は値を持たない分メモリ効率が良い。操作（in, add, remove）は同じ O(1) 平均。ただし set は集合演算（和、積、差）を高速にサポートする追加機能がある。

### Q4: ハッシュテーブルの初期サイズはどう決めるべきか？

**A:** 予想される要素数をロードファクターの閾値で割ったサイズが目安。例えば1000要素、α=0.75 なら約1334以上のバケットが必要。リハッシュは O(n) のコストがかかるため、事前にサイズがわかっている場合は大きめに確保する。Python の dict は `dict.fromkeys(range(n))` で事前確保できないが、Java の HashMap は `new HashMap<>(initialCapacity)` で指定可能。

### Q5: ハッシュテーブルと平衡二分探索木の使い分けは？

**A:**
- **ハッシュテーブル**: 平均 O(1) の検索。順序不要な場面で最速。
- **平衡BST（TreeMap等）**: O(log n) だが、キーの順序を保持。範囲検索、最小値/最大値、順序走査が必要な場合に使用。
- 実務指針: 単純なキー検索にはハッシュテーブル、順序関連の操作が必要なら BST。

### Q6: 並行処理でのハッシュテーブルは？

**A:** 通常のハッシュテーブルはスレッドセーフではない。対策:
- Java: `ConcurrentHashMap`（セグメント単位のロック）
- Python: `threading.Lock` でラップ、または `multiprocessing.Manager().dict()`
- Go: `sync.Map`（読み取り多い場面に最適化）
- 一般: ストライプロック（バケット群ごとにロック）で並行性を向上

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| ハッシュ関数 | 均一分布・決定的・高速が条件。SipHash が標準 |
| チェイン法 | 実装が簡単。削除も容易。Java HashMap が採用 |
| オープンアドレス法 | キャッシュ効率が良い。Python dict が採用 |
| Robin Hood | 探索距離を平均化。Rust HashMap が採用 |
| カッコウハッシュ | 最悪 O(1) 検索。理論的に重要 |
| ロードファクター | 性能維持の鍵。閾値超過でリハッシュ |
| リハッシュ | テーブルサイズ倍増 + 全要素再挿入。償却 O(1) |
| キーの条件 | immutable かつ __hash__ と __eq__ が整合的 |
| セキュリティ | ハッシュランダム化で DoS 攻撃を防御 |
| 実務応用 | LRU キャッシュ、一貫性ハッシュ、ブルームフィルタ |

---

## 次に読むべきガイド

- [木構造 — BST と平衡木](./04-trees.md)
- [時間空間トレードオフ — ブルームフィルタ](../00-complexity/02-space-time-tradeoff.md)

---

## 参考文献

1. Cormen, T.H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — 第11章「Hash Tables」
2. Knuth, D.E. (1998). *The Art of Computer Programming, Volume 3*. Addison-Wesley. — ハッシュ法の理論
3. Python Developer's Guide. "Dictionaries." — https://docs.python.org/3/c-api/dict.html
4. Pagh, R. & Rodler, F.F. (2004). "Cuckoo hashing." *Journal of Algorithms*, 51(2), 122-144.
5. Celis, P. (1986). *Robin Hood Hashing*. Technical Report CS-86-14, University of Waterloo.
6. Bloom, B.H. (1970). "Space/time trade-offs in hash coding with allowable errors." *Communications of the ACM*, 13(7), 422-426.
7. Karger, D. et al. (1997). "Consistent hashing and random trees." *Proceedings of the 29th annual ACM symposium on Theory of computing*.
