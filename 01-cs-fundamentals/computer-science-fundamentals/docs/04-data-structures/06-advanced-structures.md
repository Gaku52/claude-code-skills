# 高度なデータ構造

> Bloom Filter は「存在しない」ことを100%保証し、Skip List はリンクリストにO(log n)探索を与える。

## この章で学ぶこと

- [ ] Bloom Filter の仕組みと用途を理解する
- [ ] Skip List の構造を説明できる
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

  用途:
  - Chrome: 悪意あるURLのチェック
  - Cassandra/HBase: SSTファイルの検索スキップ
  - Medium: 記事推薦の重複排除
  - Bitcoin: SPVノードのトランザクション検証
```

### 1.2 実装

```python
import mmh3  # MurmurHash3

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size

    def add(self, item):
        for i in range(self.num_hashes):
            idx = mmh3.hash(item, i) % self.size
            self.bit_array[idx] = 1

    def contains(self, item):
        for i in range(self.num_hashes):
            idx = mmh3.hash(item, i) % self.size
            if self.bit_array[idx] == 0:
                return False  # 確実に存在しない
        return True  # おそらく存在する（偽陽性の可能性あり）
```

---

## 2. Skip List

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

---

## 3. その他の高度なデータ構造

```
┌──────────────────┬──────────────────┬───────────────────┐
│ データ構造        │ 用途             │ 計算量            │
├──────────────────┼──────────────────┼───────────────────┤
│ Segment Tree     │ 区間クエリ       │ O(log n) 更新/検索│
│ Fenwick Tree(BIT)│ 区間和クエリ     │ O(log n)          │
│ Disjoint Set     │ 集合の統合       │ O(α(n)) ≈ O(1)  │
│ LRU Cache        │ キャッシュ管理   │ O(1) 全操作      │
│ Rope             │ 長い文字列操作   │ O(log n) 連結    │
│ Merkle Tree      │ データ整合性検証 │ O(log n) 検証    │
└──────────────────┴──────────────────┴───────────────────┘
```

---

## 4. 実践演習

### 演習1: Bloom Filter（基礎）
Bloom Filterを実装し、偽陽性率をパラメータ（m, k）を変えて実測せよ。

### 演習2: セグメント木（応用）
区間の最小値クエリと点更新をサポートするセグメント木を実装せよ。

### 演習3: LRUキャッシュ（発展）
スレッドセーフなLRUキャッシュを実装し、ベンチマークを取れ。

---

## まとめ

| データ構造 | 特性 | 主な用途 |
|-----------|------|---------|
| Bloom Filter | 偽陽性あり、偽陰性なし | 存在チェック、キャッシュ |
| Skip List | 確率的平衡 O(log n) | Redis Sorted Set |
| Segment Tree | 区間クエリ O(log n) | 競技プログラミング、DB |
| Union-Find | ほぼO(1)の集合統合 | クラスタリング、MST |

---

## 次に読むべきガイド
→ [[07-choosing-data-structures.md]] — データ構造の選び方

---

## 参考文献
1. Bloom, B. H. "Space/Time Trade-offs in Hash Coding with Allowable Errors." 1970.
2. Pugh, W. "Skip Lists: A Probabilistic Alternative to Balanced Trees." 1990.
