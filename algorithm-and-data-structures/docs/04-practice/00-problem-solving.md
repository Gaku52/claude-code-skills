# 問題解決法

> アルゴリズム問題を体系的に解くための思考法を、パターン認識・制約分析・段階的改善の3段階で習得する

## この章で学ぶこと

1. **パターン認識**で問題を既知のアルゴリズムカテゴリに分類できる
2. **制約分析**から許容される計算量を逆算し、適切なアルゴリズムを選択できる
3. **段階的改善**で愚直解から最適解へと効率的に到達する手順を身につける

---

## 1. 問題解決の全体フレームワーク

```
┌──────────────────────────────────────────────┐
│        問題解決の5ステップ                     │
├──────────────────────────────────────────────┤
│                                               │
│  Step 1: 問題を理解する                       │
│    → 入力/出力を明確化、エッジケースを特定     │
│                                               │
│  Step 2: 具体例で手を動かす                    │
│    → 小さな入力で手計算、パターンを発見        │
│                                               │
│  Step 3: 制約を分析する                        │
│    → n の範囲から O(?) を逆算                  │
│                                               │
│  Step 4: アルゴリズムを選択・設計する           │
│    → パターン認識 → 既知手法の適用/組合せ       │
│                                               │
│  Step 5: 実装・検証・改善する                   │
│    → 愚直解 → 正解確認 → 最適化               │
│                                               │
└──────────────────────────────────────────────┘
```

---

## 2. 制約分析: n から計算量を逆算する

```
1秒で処理可能な演算数 ≈ 10^8 〜 10^9

制約 n の範囲 → 許容される計算量:

  n ≤ 10      → O(n!)        全順列探索、バックトラッキング
  n ≤ 20      → O(2^n)       ビットDP、部分集合列挙
  n ≤ 50      → O(n^4)       4重ループ（稀）
  n ≤ 500     → O(n^3)       Floyd-Warshall、区間DP
  n ≤ 5,000   → O(n^2)       DP（2次元）、全対比較
  n ≤ 100,000 → O(n log n)   ソート、セグメント木、二分探索
  n ≤ 10^6    → O(n)         線形走査、尺取り法
  n ≤ 10^9    → O(log n)     二分探索、繰り返し二乗法
  n ≤ 10^18   → O(1) or O(log n)  数学的公式、行列累乗
```

### 制約分析の実践例

```python
# 問題: 配列から和が target となるペアを見つける
# 制約: n ≤ 10^5

# 制約分析:
# n = 10^5 → O(n²) = 10^10 → TLE（タイムオーバー）
# → O(n log n) か O(n) が必要

# O(n²): 全ペア列挙（NG）
def two_sum_brute(arr, target):
    for i in range(len(arr)):
        for j in range(i+1, len(arr)):
            if arr[i] + arr[j] == target:
                return (i, j)

# O(n log n): ソート + 二分探索 or 尺取り（OK）
def two_sum_sort(arr, target):
    sorted_arr = sorted(enumerate(arr), key=lambda x: x[1])
    l, r = 0, len(arr) - 1
    while l < r:
        s = sorted_arr[l][1] + sorted_arr[r][1]
        if s == target:
            return (sorted_arr[l][0], sorted_arr[r][0])
        elif s < target:
            l += 1
        else:
            r -= 1

# O(n): ハッシュマップ（最適）
def two_sum_hash(arr, target):
    seen = {}
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
```

---

## 3. パターン認識マップ

```
問題のキーワード → アルゴリズム候補:

┌──────────────────────────────────────────────────────┐
│ キーワード           → 候補アルゴリズム              │
├──────────────────────────────────────────────────────┤
│ "最短"               → BFS, Dijkstra, Bellman-Ford   │
│ "最大/最小"          → DP, 貪欲法, 二分探索          │
│ "数え上げ"           → DP, 組合せ数学                │
│ "全探索"             → バックトラッキング, ビットDP  │
│ "部分列"             → DP (LIS, LCS)                 │
│ "連結"               → Union-Find, BFS/DFS           │
│ "区間"               → セグメント木, 尺取り法        │
│ "文字列パターン"     → KMP, Z-algorithm, Trie        │
│ "グラフ+重み"        → Dijkstra, MST                 │
│ "辞書順最小"         → 貪欲法, スタック              │
│ "二部グラフ"         → 二部マッチング, 2色彩色       │
│ "割り当て"           → ネットワークフロー            │
│ "MOD 10^9+7"         → DP + 繰り返し二乗法           │
└──────────────────────────────────────────────────────┘
```

---

## 4. 典型パターン: 尺取り法（Two Pointers / Sliding Window）

```python
def max_subarray_sum_k(arr: list, k: int) -> int:
    """長さ k の部分配列の最大和（固定長ウィンドウ）"""
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # スライド
        max_sum = max(max_sum, window_sum)

    return max_sum

def min_subarray_len(arr: list, target: int) -> int:
    """和が target 以上となる最短の連続部分配列（可変長ウィンドウ）"""
    n = len(arr)
    min_len = float('inf')
    left = 0
    current_sum = 0

    for right in range(n):
        current_sum += arr[right]

        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1

    return min_len if min_len != float('inf') else 0

print(max_subarray_sum_k([1, 4, 2, 10, 23, 3, 1, 0, 20], 4))  # 39
print(min_subarray_len([2, 3, 1, 2, 4, 3], 7))                  # 2
```

---

## 5. 典型パターン: 二分探索で答えを決め打ち

```python
def min_max_partition(arr: list, k: int) -> int:
    """配列を k 分割したとき、各区間の和の最大値を最小化する"""

    def can_partition(max_sum: int) -> bool:
        """最大和が max_sum 以下で k 分割可能か？"""
        count = 1
        current_sum = 0
        for num in arr:
            if current_sum + num > max_sum:
                count += 1
                current_sum = num
                if count > k:
                    return False
            else:
                current_sum += num
        return True

    # 答えの二分探索
    lo = max(arr)          # 最小: 最大要素（1要素の区間）
    hi = sum(arr)          # 最大: 全体が1区間
    result = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        if can_partition(mid):
            result = mid
            hi = mid - 1
        else:
            lo = mid + 1

    return result

print(min_max_partition([7, 2, 5, 10, 8], 2))  # 18 ([7,2,5] [10,8])
```

---

## 6. 典型パターン: 累積和

```python
def prefix_sum_queries(arr: list, queries: list) -> list:
    """複数の区間和クエリを O(1) で回答"""
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    results = []
    for l, r in queries:
        results.append(prefix[r + 1] - prefix[l])

    return results

# 2次元累積和
def prefix_sum_2d(matrix: list) -> list:
    """2次元累積和の構築"""
    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(rows):
        for j in range(cols):
            prefix[i+1][j+1] = (matrix[i][j]
                                + prefix[i][j+1]
                                + prefix[i+1][j]
                                - prefix[i][j])
    return prefix

def query_2d(prefix, r1, c1, r2, c2):
    """(r1,c1)-(r2,c2) の矩形和"""
    return (prefix[r2+1][c2+1]
            - prefix[r1][c2+1]
            - prefix[r2+1][c1]
            + prefix[r1][c1])

arr = [3, 1, 4, 1, 5, 9, 2, 6]
print(prefix_sum_queries(arr, [(0, 3), (2, 5), (0, 7)]))
# [9, 19, 31]
```

---

## 7. 段階的改善の実践

```
問題: n 個の点から最近接点対を求める

段階1: 愚直解 O(n²)
  → 全ペアの距離を計算
  → 正解の確認に使う

段階2: ソート活用 O(n log n + α)
  → x座標でソートし、近い点のみ比較
  → まだ最悪 O(n²) の可能性

段階3: 分割統治 O(n log n)
  → 左右に分割 + ストリップ内の限定比較
  → 最適解

段階4: ランダム化 O(n) 期待値
  → グリッド法
  → 実用上最速だが分析が複雑
```

```python
# 段階的改善の実装例

# 段階1: O(n²) 愚直解
def closest_pair_brute(points):
    min_dist = float('inf')
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            d = dist(points[i], points[j])
            min_dist = min(min_dist, d)
    return min_dist

# 段階2: ソート + 枝刈り
def closest_pair_sorted(points):
    points.sort()  # x座標でソート
    min_dist = float('inf')
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            if points[j][0] - points[i][0] >= min_dist:
                break  # 枝刈り
            d = dist(points[i], points[j])
            min_dist = min(min_dist, d)
    return min_dist

# 段階3: 分割統治 O(n log n) → 02-algorithms/06-divide-conquer.md 参照
```

---

## 8. エッジケースチェックリスト

```
入力の境界:
  □ n = 0 (空入力)
  □ n = 1 (最小入力)
  □ n = 最大値 (制約上限)

値の特殊性:
  □ 全要素が同じ値
  □ 既にソート済み / 逆順
  □ 負の値 / ゼロ
  □ 整数オーバーフロー

グラフ:
  □ 自己ループ
  □ 多重辺
  □ 非連結グラフ
  □ 木（辺 = 頂点-1）

文字列:
  □ 空文字列
  □ 1文字
  □ 全文字が同じ
```

---

## 9. 問題カテゴリ判定フローチャート

```
問題を読む
    │
    ├─ 最適化問題？
    │   ├─ 貪欲選択性質あり？ → 貪欲法
    │   ├─ 重複部分問題あり？ → DP
    │   └─ 制約が小さい？ → 全探索/バックトラック
    │
    ├─ グラフ問題？
    │   ├─ 最短経路？ → BFS/Dijkstra/Bellman-Ford
    │   ├─ 連結性？ → Union-Find/BFS/DFS
    │   ├─ 順序関係？ → トポロジカルソート
    │   └─ マッチング？ → ネットワークフロー
    │
    ├─ 区間/配列問題？
    │   ├─ 区間クエリ？ → セグメント木/BIT
    │   ├─ 部分配列の和？ → 累積和/尺取り法
    │   └─ 二分探索可能？ → 答えの二分探索
    │
    └─ 文字列問題？
        ├─ パターン検索？ → KMP/Z-algorithm
        ├─ 接頭辞/辞書？ → Trie
        └─ 部分列？ → DP (LCS/LIS)
```

---

## 10. アルゴリズム選択比較表

| 問題の性質 | 第一候補 | 第二候補 | 避けるべき |
|:---|:---|:---|:---|
| 最短経路（重みなし） | BFS | - | Dijkstra（無駄） |
| 最短経路（非負重み） | Dijkstra | A* | Bellman-Ford（遅い） |
| 連結成分（動的） | Union-Find | - | BFS毎回（遅い） |
| 区間和（更新あり） | BIT | セグメント木 | 毎回再計算（遅い） |
| 区間最小値 | セグメント木 | Sparse Table | BIT（非対応） |
| ソート済み配列の検索 | 二分探索 | - | 線形探索（遅い） |

## 計算量の目安

| 操作 | Python実測（目安） | 注意点 |
|:---|:---|:---|
| 10^6 回のループ | ~0.1秒 | Pythonは遅い |
| 10^7 回のループ | ~1秒 | 制限ギリギリ |
| 10^8 回のループ | ~10秒 | TLE確実 |
| sort(10^6要素) | ~0.3秒 | TimSort は高速 |
| dict操作(10^6回) | ~0.2秒 | ハッシュは定数大 |

---

## 11. アンチパターン

### アンチパターン1: 制約を無視した実装

```python
# BAD: n=10^5 なのに O(n²) を実装
# → 10^10 回の演算 → TLE

# GOOD: 制約を先に確認し、許容される計算量を見積もる
# n=10^5 → O(n log n) 以下が必要
```

### アンチパターン2: いきなり最適解を目指す

```python
# BAD: 最初から最適なアルゴリズムを実装しようとする
# → バグが入りやすい、デバッグが困難

# GOOD: まず愚直解を実装し、正しい出力を確認してから最適化
# 1. O(n²) の愚直解で正解を確認
# 2. O(n log n) の解を実装
# 3. 愚直解と比較してバグを検出
```

---

## 12. FAQ

### Q1: 問題を見て何も思いつかない場合は？

**A:** (1) 小さな具体例で手計算する。(2) 制約から計算量を逆算する。(3) 類似問題を思い出す。(4) 愚直解を書いてみる。(5) 愚直解のボトルネックを特定し、データ構造やアルゴリズムで改善する。「何も思いつかない」は多くの場合「具体例が足りない」。

### Q2: バグが見つからない場合は？

**A:** (1) 愚直解と比較するストレステストを実装する。(2) 小さなランダム入力を大量生成する。(3) エッジケース（空入力、最小最大、同値）を手動テストする。(4) 変数の中間状態をプリントする。(5) 問題文を再度読み直す（誤読の可能性）。

### Q3: 計算量の定数倍が気になる場合は？

**A:** Python は C++ の 30-100 倍遅いのが一般的。対策: (1) PyPy を使う（3-5倍高速）。(2) リスト内包表記を使う（for文より高速）。(3) `sys.stdin.readline` で入力を高速化。(4) 再帰を反復に変換。(5) 最終手段として C++ で書き直す。

---

## 13. まとめ

| 項目 | 要点 |
|:---|:---|
| 制約分析 | n の範囲から O(?) を逆算して適切なアルゴリズムを選ぶ |
| パターン認識 | キーワードと問題構造から既知の手法にマッピング |
| 段階的改善 | 愚直解→正解確認→最適化の3段階で進める |
| エッジケース | 空入力・最小最大・特殊値を必ず確認 |
| 典型テクニック | 尺取り法、答えの二分探索、累積和 |
| デバッグ | ストレステストと愚直解の比較が最強 |

---

## 次に読むべきガイド

- [競技プログラミング](./01-competitive-programming.md) -- 問題解決力を実戦で鍛える
- [動的計画法](../02-algorithms/04-dynamic-programming.md) -- 最も出題頻度の高いパラダイム
- [グラフ走査](../02-algorithms/02-graph-traversal.md) -- グラフ問題の基礎

---

## 参考文献

1. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Part I: Problem Solving
2. Polya, G. (1945). *How to Solve It*. Princeton University Press.
3. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapter 1: Introduction
4. Cormen, T. H. et al. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第1部
