# 探索アルゴリズム

> 「データを見つける」ことはコンピューティングの最も基本的な操作であり、探索の効率がシステム全体の性能を左右する。

## この章で学ぶこと

- [ ] 線形探索と二分探索の違いと使い分けを理解する
- [ ] 二分探索の応用パターンを実装できる
- [ ] ハッシュベースの探索の仕組みを理解する

## 前提知識

- 計算量解析 → 参照: [[01-complexity-analysis.md]]

---

## 1. 線形探索

### 1.1 基本

```python
def linear_search(arr, target):
    """先頭から順に探す — 最も単純な探索"""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

# 計算量: O(n)
# 前提条件: なし（ソート不要）
# 用途: 小さいデータ、ソートされていないデータ

# 実務で線形探索が適切な場面:
# - 要素数が少ない (n < 50程度)
# - 一度しか検索しない
# - データが頻繁に変更される（ソート維持コスト > 探索コスト）
```

---

## 2. 二分探索

### 2.1 基本実装

```python
def binary_search(arr, target):
    """ソート済み配列を半分ずつ絞り込む"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # オーバーフロー防止
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# 計算量: O(log n)
# 前提条件: 配列がソート済み
# n = 10億 でもたった30回の比較で探索完了！

# 二分探索の動作例: arr = [1, 3, 5, 7, 9, 11, 13], target = 9
# Step 1: left=0, right=6, mid=3 → arr[3]=7 < 9 → left=4
# Step 2: left=4, right=6, mid=5 → arr[5]=11 > 9 → right=4
# Step 3: left=4, right=4, mid=4 → arr[4]=9 == 9 → found!
```

### 2.2 よくあるバグと対策

```python
# 二分探索のバグ Top 3:

# バグ1: 中間値のオーバーフロー
mid = (left + right) // 2       # ❌ left + right がオーバーフロー
mid = left + (right - left) // 2  # ✅ 安全

# バグ2: 無限ループ
# left=3, right=4, mid=3 の時:
# arr[mid] < target → left = mid (❌ 進まない！)
# 正しくは: left = mid + 1

# バグ3: off-by-one エラー
# while left < right vs while left <= right
# → ループ条件と更新式の組み合わせで挙動が変わる

# 安全なテンプレート:
# パターンA: 完全一致（上記の基本実装）
# while left <= right, left = mid + 1, right = mid - 1

# パターンB: 境界探索（lower_bound / upper_bound）
# → 次のセクション参照
```

### 2.3 二分探索の応用パターン

```python
import bisect

# 1. lower_bound: target 以上の最小のインデックス
def lower_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left
# Python: bisect.bisect_left(arr, target)

# 2. upper_bound: target より大きい最小のインデックス
def upper_bound(arr, target):
    left, right = 0, len(arr)
    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:
            left = mid + 1
        else:
            right = mid
    return left
# Python: bisect.bisect_right(arr, target)

# 3. 答えで二分探索（Binary Search on Answer）
# 「条件を満たす最小の値は？」を二分探索で求める

# 例: n本のロープを全て切って同じ長さにする。最大何cmにできるか？
def max_rope_length(ropes, k):
    """ropes を切って k 本以上の同じ長さのロープを作る最大長"""
    def can_cut(length):
        return sum(r // length for r in ropes) >= k

    left, right = 1, max(ropes)
    while left <= right:
        mid = left + (right - left) // 2
        if can_cut(mid):
            left = mid + 1
        else:
            right = mid - 1
    return right

# 4. 回転ソート済み配列での二分探索
def search_rotated(arr, target):
    """[4,5,6,7,0,1,2] のような配列で探索"""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:  # 左半分がソート済み
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半分がソート済み
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

---

## 3. ハッシュベース探索

### 3.1 ハッシュテーブル

```python
# ハッシュテーブル: O(1) の期待時間で探索

# Python の dict / set は内部的にハッシュテーブル
# JavaScript の Map / Set も同様

# 基本操作:
d = {}
d["key"] = "value"    # 挿入: O(1) 期待
val = d["key"]         # 探索: O(1) 期待
del d["key"]           # 削除: O(1) 期待
"key" in d             # 存在確認: O(1) 期待

# ハッシュ関数の要件:
# 1. 決定的: 同じ入力 → 同じ出力
# 2. 均一分布: 出力が偏らない
# 3. 高速: 計算コストが低い

# 衝突（collision）の解決:
# 1. チェイニング: 同じスロットにリンクリストで格納
# 2. オープンアドレス法: 別のスロットを探す（線形探索、二次探索）

# Python dict の実装:
# - オープンアドレス法（ランダム探索）
# - ロードファクター: 2/3 でリハッシュ
# - ハッシュ関数: SipHash（セキュリティ対策）
```

### 3.2 二分探索 vs ハッシュテーブル

```
探索方法の比較:

  ┌──────────────────┬──────────────┬──────────────────┐
  │ 特性             │ 二分探索      │ ハッシュテーブル  │
  ├──────────────────┼──────────────┼──────────────────┤
  │ 時間計算量(探索)  │ O(log n)     │ O(1) 期待        │
  │ 最悪計算量       │ O(log n)     │ O(n)             │
  │ 前提条件         │ ソート済み   │ ハッシュ関数      │
  │ 空間効率         │ O(1) 追加    │ O(n)             │
  │ 範囲検索         │ 効率的       │ 非効率           │
  │ 順序付き列挙     │ 容易         │ 不可能           │
  │ キャッシュ効率   │ 良い         │ 悪い             │
  │ 実装の複雑さ     │ 中           │ 低               │
  └──────────────────┴──────────────┴──────────────────┘

  使い分け:
  - 完全一致のみ → ハッシュテーブル
  - 範囲検索が必要 → 二分探索（ソート済み配列 or BST）
  - メモリ制約 → 二分探索
  - 最悪ケース保証 → 二分探索
```

---

## 4. 実務での探索

### 4.1 データベースのインデックス

```
データベースの探索:

  インデックスなし:
    SELECT * FROM users WHERE email = 'test@example.com';
    → フルテーブルスキャン: O(n) — 100万行 → 数秒

  B-Treeインデックス:
    CREATE INDEX idx_email ON users(email);
    → B-Tree探索: O(log n) — 100万行 → 数ms

  ハッシュインデックス:
    → 完全一致: O(1)
    → 範囲検索不可: WHERE age > 20 には使えない

  B-Treeの特徴:
  - ディスクI/Oに最適化（各ノードが1ページ=4KB〜16KB）
  - 扇出数(fanout)が大きい（通常100-500）
  - 高さが非常に低い: 100万件でも高さ3-4
  - 範囲検索が効率的（葉ノードがリンクされている）
```

### 4.2 全文検索

```
全文検索の仕組み:

  転置インデックス（Inverted Index）:
    文書1: "The quick brown fox"
    文書2: "The lazy brown dog"
    文書3: "Quick fox jumps"

    ┌──────────┬──────────────┐
    │ 単語      │ 文書ID       │
    ├──────────┼──────────────┤
    │ brown    │ [1, 2]       │
    │ dog      │ [2]          │
    │ fox      │ [1, 3]       │
    │ jumps    │ [3]          │
    │ lazy     │ [2]          │
    │ quick    │ [1, 3]       │
    │ the      │ [1, 2]       │
    └──────────┴──────────────┘

    検索 "quick fox":
    quick → [1, 3]
    fox   → [1, 3]
    AND   → [1, 3]  ← 共通の文書

  使用技術:
  - Elasticsearch: Lucene ベースの分散全文検索エンジン
  - PostgreSQL: tsvector + GINインデックス
  - SQLite FTS5: 軽量全文検索
```

---

## 5. 実践演習

### 演習1: 二分探索の実装（基礎）
以下のバリエーションを全て実装せよ:
1. 完全一致探索
2. lower_bound（target以上の最小インデックス）
3. upper_bound（targetより大きい最小インデックス）

### 演習2: 答えで二分探索（応用）
N人の生徒にM個のチョコレートを配る。各チョコの甘さが与えられる。全員に連続する区間のチョコを配る時、甘さの合計の最小値を最大化せよ。

### 演習3: ハッシュテーブルの実装（発展）
チェイニング方式のハッシュテーブルをゼロから実装し、挿入・検索・削除・リサイズをサポートせよ。

---

## FAQ

### Q1: 二分探索はソート済み配列以外にも使えますか？
**A**: 「答えで二分探索」は非常に強力。「条件を満たすか？」が単調（あるしきい値を境に Yes/No が切り替わる）なら、そのしきい値を二分探索で求められる。最適化問題の判定問題への帰着。

### Q2: ハッシュテーブルの最悪 O(n) は問題にならないのですか？
**A**: 実用上はほぼ問題にならない。適切なハッシュ関数とロードファクター管理で衝突を最小化できる。ただし、意図的な衝突攻撃（Hash DoS）には注意が必要。Python は SipHash を使用して対策している。

### Q3: データベースのインデックスはいくつ作るべきですか？
**A**: クエリパターンに応じて必要最小限。インデックスは読み取りを高速化するが、書き込み（INSERT/UPDATE/DELETE）を遅くする。各インデックスは追加のストレージも消費する。

---

## まとめ

| 探索方法 | 計算量 | 前提 | 用途 |
|---------|--------|------|------|
| 線形探索 | O(n) | なし | 小規模/未ソート |
| 二分探索 | O(log n) | ソート済み | 大規模/範囲検索 |
| ハッシュ | O(1)期待 | ハッシュ関数 | 完全一致 |
| B-Tree | O(log n) | 構築済み | DB/ファイルシステム |
| 転置インデックス | O(1)〜O(k) | 構築済み | 全文検索 |

---

## 次に読むべきガイド
→ [[04-graph-algorithms.md]] — グラフアルゴリズム

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapter 12: Binary Search Trees.
2. Knuth, D. E. "The Art of Computer Programming." Vol. 3: Sorting and Searching.
3. Comer, D. "The Ubiquitous B-Tree." ACM Computing Surveys, 1979.
