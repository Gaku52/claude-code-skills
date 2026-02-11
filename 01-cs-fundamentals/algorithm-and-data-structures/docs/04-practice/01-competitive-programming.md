# 競技プログラミング

> AtCoder・LeetCode を中心に、競技プログラミングの典型テクニック・戦略・実践パターンを体系的に習得する

## この章で学ぶこと

1. **AtCoder と LeetCode** のレベル体系と効率的な学習ロードマップを理解する
2. **頻出の典型テクニック**（座標圧縮、bit全探索、しゃくとり法等）を実装できる
3. **コンテスト中の戦略**（時間配分、テンプレート活用、デバッグ手法）を身につける

---

## 1. 競技プログラミングの全体像

```
┌────────────────────────────────────────────────────┐
│            競技プログラミングの世界                  │
├──────────────────┬─────────────────────────────────┤
│  AtCoder          │  LeetCode                      │
│  (コンテスト型)    │  (面接対策型)                   │
├──────────────────┼─────────────────────────────────┤
│ ABC: 初級〜中級   │ Easy: 基本データ構造            │
│ ARC: 中級〜上級   │ Medium: アルゴリズム応用         │
│ AGC: 上級〜超上級 │ Hard: 高度な組合せ/DP/フロー    │
├──────────────────┼─────────────────────────────────┤
│ レーティング:     │ レーティング:                    │
│ 灰 <400          │ (コンテスト参加で付与)           │
│ 茶 400-799       │                                  │
│ 緑 800-1199      │ 主な目的:                        │
│ 水 1200-1599     │ ・コーディング面接対策            │
│ 青 1600-1999     │ ・アルゴリズム力の証明            │
│ 黄 2000-2399     │                                  │
│ 橙 2400-2799     │                                  │
│ 赤 2800+         │                                  │
└──────────────────┴─────────────────────────────────┘
```

---

## 2. Python テンプレート

```python
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations, permutations, accumulate
from heapq import heappush, heappop
from bisect import bisect_left, bisect_right
from functools import lru_cache

# 高速入力
input = sys.stdin.readline

def main():
    # --- 入力 ---
    N = int(input())
    A = list(map(int, input().split()))
    # S = input().strip()

    # --- 解法 ---
    ans = 0

    # --- 出力 ---
    print(ans)

if __name__ == "__main__":
    main()
```

### 入出力の高速化

```python
import sys
input = sys.stdin.readline

# 複数行の入力を一括読み込み
def read_all():
    return sys.stdin.read().split()

# 大量出力
def fast_output(results):
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

# 再帰制限の拡張（DFS用）
sys.setrecursionlimit(10**6)
```

---

## 3. 典型テクニック: bit全探索

n が小さい（n <= 20）とき、全部分集合を 2^n で列挙する。

```python
def bit_bruteforce(n: int, items: list) -> list:
    """bit全探索 - O(2^n * n)"""
    results = []
    for mask in range(1 << n):  # 0 から 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # i番目のビットが立っている
                subset.append(items[i])
        results.append(subset)
    return results

# 例: 部分和問題（n <= 20）
def subset_sum(arr: list, target: int) -> bool:
    """target と一致する部分集合が存在するか"""
    n = len(arr)
    for mask in range(1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += arr[i]
        if total == target:
            return True
    return False

print(subset_sum([3, 7, 1, 8, 4], 12))  # True (3+1+8)

# 半分全列挙（n <= 40 に拡張）
def meet_in_the_middle(arr: list, target: int) -> bool:
    """半分全列挙 - O(2^(n/2) * n)"""
    n = len(arr)
    half = n // 2

    # 前半の部分和を列挙
    left_sums = set()
    for mask in range(1 << half):
        s = sum(arr[i] for i in range(half) if mask & (1 << i))
        left_sums.add(s)

    # 後半の部分和で target - s が前半にあるか確認
    for mask in range(1 << (n - half)):
        s = sum(arr[half + i] for i in range(n - half) if mask & (1 << i))
        if target - s in left_sums:
            return True

    return False
```

---

## 4. 典型テクニック: 座標圧縮

大きな座標値を、相対順序を保ったまま 0, 1, 2, ... に圧縮する。

```
元の座標:    [100, 5000, 300, 100, 10000]
圧縮後:      [0, 2, 1, 0, 3]

圧縮テーブル:
  100   → 0
  300   → 1
  5000  → 2
  10000 → 3
```

```python
def coordinate_compress(arr: list) -> tuple:
    """座標圧縮 - O(n log n)"""
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}
    compressed = [compress[v] for v in arr]
    return compressed, sorted_unique

data = [100, 5000, 300, 100, 10000]
compressed, mapping = coordinate_compress(data)
print(compressed)  # [0, 2, 1, 0, 3]
print(mapping)     # [100, 300, 5000, 10000]

# 応用: 転倒数の計算（BITと組み合わせ）
def count_inversions_compressed(arr):
    compressed, _ = coordinate_compress(arr)
    n = len(compressed)
    max_val = max(compressed) + 1

    # BIT で右から走査
    bit = [0] * (max_val + 1)

    def bit_update(i, delta=1):
        i += 1
        while i <= max_val:
            bit[i] += delta
            i += i & (-i)

    def bit_query(i):
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    inversions = 0
    for i in range(n - 1, -1, -1):
        inversions += bit_query(compressed[i] - 1)
        bit_update(compressed[i])

    return inversions
```

---

## 5. 典型テクニック: MOD演算

大きな数の演算で桁溢れを防ぐため、MOD 10^9+7 で計算する。

```python
MOD = 10**9 + 7

# 基本演算
def mod_add(a, b):
    return (a + b) % MOD

def mod_mul(a, b):
    return (a * b) % MOD

# 繰り返し二乗法
def mod_pow(base, exp, mod=MOD):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        exp >>= 1
        base = base * base % mod
    return result

# モジュラ逆元（フェルマーの小定理: a^(-1) ≡ a^(p-2) mod p）
def mod_inv(a, mod=MOD):
    return mod_pow(a, mod - 2, mod)

# 組合せ数 nCr の前計算
class Combinatorics:
    def __init__(self, max_n: int, mod: int = MOD):
        self.mod = mod
        self.fact = [1] * (max_n + 1)
        self.inv_fact = [1] * (max_n + 1)

        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i - 1] * i % mod

        self.inv_fact[max_n] = mod_pow(self.fact[max_n], mod - 2, mod)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % mod

    def comb(self, n: int, r: int) -> int:
        """nCr mod p"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def perm(self, n: int, r: int) -> int:
        """nPr mod p"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[n - r] % self.mod

# 使用例
comb = Combinatorics(200000)
print(comb.comb(100000, 50000) % MOD)
```

---

## 6. 典型テクニック: ダイクストラ（競プロ版）

```python
import heapq

def dijkstra(graph: list, start: int) -> list:
    """ダイクストラ法（競プロ用高速版）
    graph[u] = [(v, cost), ...]
    """
    INF = float('inf')
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]  # (距離, 頂点)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, cost in graph[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist

# 入力例（AtCoder形式）
# N M
# a1 b1 c1
# ...
def solve_shortest_path():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        a, b, c = map(int, input().split())
        a -= 1; b -= 1  # 0-indexed
        graph[a].append((b, c))
        graph[b].append((a, c))

    dist = dijkstra(graph, 0)
    print(dist[N - 1])
```

---

## 7. 典型テクニック: 累積和 + 二分探索

```python
from itertools import accumulate
from bisect import bisect_left, bisect_right

def solve_range_queries():
    """累積和で区間クエリを高速化"""
    A = [3, 1, 4, 1, 5, 9, 2, 6]
    prefix = list(accumulate(A, initial=0))
    # prefix = [0, 3, 4, 8, 9, 14, 23, 25, 31]

    # 区間 [l, r) の和 = prefix[r] - prefix[l]
    print(prefix[5] - prefix[1])  # 11 (1+4+1+5)

def solve_with_binary_search():
    """ソート済み配列で二分探索を活用"""
    A = sorted([3, 1, 4, 1, 5, 9, 2, 6])
    # A = [1, 1, 2, 3, 4, 5, 6, 9]

    # x 以上の要素数
    def count_ge(x):
        return len(A) - bisect_left(A, x)

    # x 以下の要素数
    def count_le(x):
        return bisect_right(A, x)

    print(count_ge(3))  # 5 (3,4,5,6,9)
    print(count_le(4))  # 5 (1,1,2,3,4)
```

---

## 8. AtCoder ABC 頻出パターン

```
A問題 (100点) - 基本演算、条件分岐
B問題 (200点) - ループ、文字列操作
C問題 (300点) - 全探索、ソート、累積和
D問題 (400点) - DP、二分探索、グラフBFS
E問題 (500点) - セグメント木、Union-Find、高度なDP
F問題 (500-600点) - 数論、フロー、高度なデータ構造

レベル別の学習目標:
  灰→茶: A,B を確実に、C を半分
  茶→緑: A,B,C を確実に、D を半分
  緑→水: A〜D を確実に、E を半分
  水→青: A〜E を確実に、F に挑戦
```

---

## 9. LeetCode 頻出パターン

```
パターン別出題頻度:

  配列/ハッシュマップ      ████████████████  35%
  二分探索/尺取り         ██████████        15%
  木/グラフ              ████████          15%
  動的計画法             ██████████        15%
  スタック/キュー         ████              8%
  文字列                 ████              7%
  その他                 ██                5%

面接頻出50問（Blind 75 / NeetCode 150 ベース）:
  Easy:  Two Sum, Valid Parentheses, Merge Two Lists
  Medium: 3Sum, LRU Cache, Course Schedule, Word Break
  Hard:  Merge K Lists, Trapping Rain Water, Word Ladder II
```

---

## 10. コンテスト中の戦略

```
時間配分（AtCoder ABC 100分の場合）:

  0-5分:   A問題（確実に取る）
  5-15分:  B問題（確実に取る）
  15-35分: C問題（しっかり考える）
  35-65分: D問題（本番の勝負所）
  65-90分: E問題（取れたらボーナス）
  90-100分: 見直し・提出

デバッグ戦略:
  1. まず紙に小さな例を手計算
  2. 愚直解を書いてストレステスト
  3. print デバッグで中間状態確認
  4. オフバイワンエラーを集中的にチェック

メンタル管理:
  ・1問詰まったら次に進む
  ・WA（不正解）でパニックにならない
  ・最後5分は新しい提出をしない（ペナルティ回避）
```

---

## 11. プラットフォーム比較表

| 特性 | AtCoder | LeetCode | Codeforces |
|:---|:---|:---|:---|
| 主な用途 | 競技 | 面接対策 | 競技 |
| 言語 | 日本語 | 英語 | 英語 |
| 難易度表示 | 色レーティング | Easy/Medium/Hard | *A-*F |
| コンテスト頻度 | 週1回(土曜) | 週2回 | 週2-3回 |
| 入力形式 | 標準入力 | 関数引数 | 標準入力 |
| Python対応 | PyPy可 | Python3 | PyPy可 |
| 解説 | 公式editorial | Discussion | 有志editorial |

## 典型テクニック速見表

| テクニック | 計算量 | 適用場面 | 出現頻度 |
|:---|:---|:---|:---|
| 累積和 | O(n) 前処理 | 区間和クエリ | 非常に高い |
| 二分探索 | O(log n) | 単調性のある探索 | 非常に高い |
| 尺取り法 | O(n) | 条件を満たす区間 | 高い |
| bit全探索 | O(2^n) | n<=20の全列挙 | 中程度 |
| 座標圧縮 | O(n log n) | 大きな値の離散化 | 中程度 |
| MOD演算 | O(n) | 巨大な数の計算 | 非常に高い |
| ダブリング | O(n log n) | LCA, 繰り返し遷移 | 中程度 |
| 行列累乗 | O(k³ log n) | 線形漸化式の高速化 | 低い |

---

## 12. アンチパターン

### アンチパターン1: Pythonの速度を過信

```python
# BAD: Python で 10^8 回のループ
# → 約10秒かかり TLE 確実

# GOOD: 対策
# 1. PyPy で提出（3-5倍高速）
# 2. リスト内包表記を使う
# 3. NumPy を活用（AtCoder では使用可能）
# 4. 最悪ケースは C++ に切り替え

# リスト内包表記 vs for文
# BAD
result = []
for i in range(n):
    result.append(i * i)

# GOOD (2-3倍高速)
result = [i * i for i in range(n)]
```

### アンチパターン2: コーナーケースの未考慮

```python
# BAD: N=1 のケースを忘れる
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    # N=1 のとき A[1] でインデックスエラー!

# GOOD: エッジケースを先に処理
def solve():
    N = int(input())
    A = list(map(int, input().split()))
    if N == 1:
        print(A[0])
        return
    # 以降 N >= 2 を前提に処理
```

---

## 13. FAQ

### Q1: 競技プログラミングを始めるのに最適な言語は？

**A:** C++ が最も推奨される（実行速度が速く、STL が強力）。Python は書きやすいが速度がネック（PyPy で緩和可能）。初学者は Python で始めて、速度が必要になったら C++ に移行するのが現実的。LeetCode は Python が主流。AtCoder は C++ と Python が二大勢力。

### Q2: 効率的な練習方法は？

**A:** (1) AtCoder Problems（kenkoooo）で茶〜緑 difficulty を毎日1-2問。(2) 解けなかった問題の editorial を読み、理解してから再実装。(3) 典型90問（E869120作）を進める。(4) コンテストには毎週参加して実戦経験を積む。量より質、理解の深さが重要。

### Q3: 実務に競技プログラミングは役立つか？

**A:** 直接的には限定的だが、間接的に大きく役立つ。(1) アルゴリズムの引き出しが増える。(2) 計算量を意識したコーディング習慣がつく。(3) エッジケースへの感度が上がる。(4) コードの正確性を検証する能力が向上する。ただし、実務で求められる設計力・保守性とは別のスキル。

### Q4: AtCoderで緑になるまでの期間は？

**A:** 個人差が大きいが、プログラミング経験者が毎日1-2時間練習して3-6か月が目安。数学的素養があると上達が速い。重要なのは (1) ABCの過去問を最低100問解く、(2) 典型パターンを身につける、(3) 毎週コンテストに参加する、の3つを継続すること。

---

## 14. まとめ

| 項目 | 要点 |
|:---|:---|
| AtCoder | 日本語、週1コンテスト、色レーティングでレベルが明確 |
| LeetCode | 英語、面接対策の定番、パターン別に練習可能 |
| テンプレート | 高速入力・再帰制限拡張・MOD演算を定型化 |
| 典型テクニック | bit全探索・座標圧縮・累積和・MOD演算が頻出 |
| コンテスト戦略 | 時間配分・デバッグ手法・メンタル管理が重要 |
| 練習方法 | 毎日1-2問、editorial精読、コンテスト参加を継続 |

---

## 次に読むべきガイド

- [問題解決法](./00-problem-solving.md) -- アルゴリズム問題への体系的アプローチ
- [動的計画法](../02-algorithms/04-dynamic-programming.md) -- 最頻出のアルゴリズムパラダイム
- [セグメント木](../03-advanced/01-segment-tree.md) -- 中級以上で必須のデータ構造

---

## 参考文献

1. 秋葉拓哉ほか (2012). 『プログラミングコンテストチャレンジブック 第2版』. マイナビ出版.
2. E869120. "競プロ典型 90 問." https://github.com/E869120/kyopro-tenkei-90
3. kenkoooo. "AtCoder Problems." https://kenkoooo.com/atcoder/
4. NeetCode. "NeetCode 150." https://neetcode.io/
5. Halim, S. & Halim, F. (2013). *Competitive Programming 3*.
