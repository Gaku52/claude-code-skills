# 競技プログラミング

> AtCoder・LeetCode・Codeforces を中心に、競技プログラミングの典型テクニック・戦略・実践パターンを体系的に習得する

## この章で学ぶこと

1. **AtCoder と LeetCode** のレベル体系と効率的な学習ロードマップを理解する
2. **頻出の典型テクニック**（座標圧縮、bit 全探索、しゃくとり法、MOD 演算等）を実装できる
3. **コンテスト中の戦略**（時間配分、テンプレート活用、デバッグ手法）を身につける
4. **面接対策としての活用法**（LeetCode パターン分類、頻出問題の解法フレームワーク）を把握する
5. **段階的な成長プラン**を設計し、継続的にレーティングを上げる方法を知る


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [問題解決法](./00-problem-solving.md) の内容を理解していること

---

## 目次

1. [競技プログラミングの全体像](#1-競技プログラミングの全体像)
2. [Python テンプレートと高速化技法](#2-python-テンプレートと高速化技法)
3. [典型テクニック: bit 全探索と半分全列挙](#3-典型テクニック-bit-全探索と半分全列挙)
4. [典型テクニック: 座標圧縮と転倒数](#4-典型テクニック-座標圧縮と転倒数)
5. [典型テクニック: しゃくとり法（Two Pointers）](#5-典型テクニック-しゃくとり法two-pointers)
6. [典型テクニック: MOD 演算と組合せ計算](#6-典型テクニック-mod-演算と組合せ計算)
7. [グラフアルゴリズムの競プロ実装](#7-グラフアルゴリズムの競プロ実装)
8. [累積和・imos 法・二分探索](#8-累積和imos-法二分探索)
9. [AtCoder / LeetCode / Codeforces パターン分析](#9-atcoder--leetcode--codeforces-パターン分析)
10. [コンテスト戦略とメンタルモデル](#10-コンテスト戦略とメンタルモデル)
11. [プラットフォーム比較と学習ロードマップ](#11-プラットフォーム比較と学習ロードマップ)
12. [演習問題（3 段階）](#12-演習問題3-段階)
13. [アンチパターン](#13-アンチパターン)
14. [FAQ](#14-faq)
15. [まとめ](#15-まとめ)
16. [参考文献](#16-参考文献)

---

## 1. 競技プログラミングの全体像

### 1.1 競技プログラミングとは

競技プログラミング（Competitive Programming、略して「競プロ」）は、制限時間内にアルゴリズムの問題を解くことを競う知的スポーツである。与えられた問題に対して正しい出力を返すプログラムを書き、実行時間・メモリ使用量の制約を満たしながら、できるだけ多くの問題を素早く解くことが求められる。

競技プログラミングの本質は「問題を読んで数学的・アルゴリズム的にモデル化し、効率的な解法を実装する」という一連のプロセスにある。これは単なるコーディング能力ではなく、問題分析力・アルゴリズム設計力・実装力・デバッグ力を総合的に要求する。

### 1.2 主要プラットフォームの位置づけ

```
┌──────────────────────────────────────────────────────────────────────┐
│                  競技プログラミングの世界地図                          │
├─────────────────────┬───────────────────┬────────────────────────────┤
│  AtCoder (日本)      │  LeetCode (米国)   │  Codeforces (ロシア)       │
│  コンテスト重視       │  面接対策重視       │  コンテスト重視             │
├─────────────────────┼───────────────────┼────────────────────────────┤
│ ABC: 初級〜中級      │ Easy:   基本       │ Div.4: 初級                │
│ ARC: 中級〜上級      │ Medium: 応用       │ Div.3: 初級〜中級          │
│ AGC: 上級〜最上級    │ Hard:   高度       │ Div.2: 中級〜上級          │
│ AHC: ヒューリス      │                   │ Div.1: 上級〜最上級        │
│      ティック        │                   │ Global: 最上級             │
├─────────────────────┼───────────────────┼────────────────────────────┤
│ レーティング:        │ レーティング:      │ レーティング:               │
│ 灰 <400   (入門)    │ コンテスト参加で   │ Newbie   <1200              │
│ 茶 400-799 (初級)   │ 付与される         │ Pupil    1200-1399         │
│ 緑 800-1199(中級)   │                   │ Specialist 1400-1599       │
│ 水 1200-1599(上級)  │ 主な目的:          │ Expert   1600-1899         │
│ 青 1600-1999(準上)  │ ・FAANG面接対策    │ CM       1900-2099         │
│ 黄 2000-2399(超上)  │ ・アルゴリズム力   │ Master   2100-2299         │
│ 橙 2400-2799(極上)  │   の証明           │ IM       2300-2399         │
│ 赤 2800+   (最上)   │ ・スキル可視化     │ GM       2400-2599         │
│                     │                   │ IGM      2600-2999         │
│                     │                   │ LGM      3000+             │
└─────────────────────┴───────────────────┴────────────────────────────┘
```

### 1.3 AtCoder のコンテスト体系

AtCoder は日本最大の競技プログラミングプラットフォームであり、高品質な問題と丁寧な解説が特徴である。

**ABC（AtCoder Beginner Contest）** は毎週土曜日 21:00（JST）に開催される。A〜G の 7 問構成で、100 分間の制限時間がある。A・B は基本的なプログラミング力、C・D は典型的なアルゴリズム知識、E・F・G は高度なデータ構造やアルゴリズムの知識が問われる。

**ARC（AtCoder Regular Contest）** は月 1-2 回開催される中上級向けコンテストで、数学的思考力を強く要求する問題が多い。

**AGC（AtCoder Grand Contest）** は不定期開催の最上級コンテストで、世界トップクラスの難問が出題される。

**AHC（AtCoder Heuristic Contest）** はマラソン型（最適化型）コンテストで、厳密解ではなく近似解の品質を競う。

### 1.4 レーティングと実力の対応関係

```
AtCoder レーティングと到達難易度の目安:

  レート    色     到達目安           必要な知識レベル
  ──────────────────────────────────────────────────────────────────
  < 400    灰     すぐ到達可能       プログラミング基礎
  400-799  茶     1-3か月           基本アルゴリズム（ソート、探索）
  800-1199 緑     3-6か月           典型テクニック（DP, BFS, 累積和）
  1200-1599 水    6か月-1年         セグ木、Union-Find、高度な DP
  1600-1999 青    1-2年             数論、フロー、高度なグラフ理論
  2000-2399 黄    2-3年             数学的洞察力、高難度の構築
  2400-2799 橙    3年以上           世界レベルの問題解決能力
  2800+    赤     極めて困難         国際情報オリンピックメダリスト級

  ※ 到達目安は毎日 1-2 時間練習した場合の概算
  ※ 数学的素養、プログラミング経験により大きく異なる
```

### 1.5 競技プログラミングと実務の関係

競技プログラミングで鍛えられるスキルは、ソフトウェアエンジニアリングの基礎力に直結する部分と、そうでない部分がある。

**直結するスキル:**
- 計算量を意識した設計（O(N^2) で TLE するなら O(N log N) を考える習慣）
- エッジケースへの感度（空配列、要素 1 個、最大値付近の入力）
- データ構造の適切な選択（ハッシュマップ vs ソート済み配列 vs ヒープ）
- アルゴリズムの引き出し（グラフ探索、DP、二分探索が適用できる場面の認識）

**実務で別途必要なスキル:**
- 保守性の高いコード設計（変数名、関数分割、テスト容易性）
- チーム開発（コードレビュー、ドキュメント、コミュニケーション）
- システム設計（分散システム、データベース設計、API 設計）

---

## 2. Python テンプレートと高速化技法

### 2.1 基本テンプレート

競技プログラミングでは、入出力の定型パターンをテンプレートとして準備しておくことで、本質的な問題解決に集中できる。以下は Python での標準テンプレートである。

```python
import sys
from collections import defaultdict, deque, Counter
from itertools import combinations, permutations, accumulate, product
from heapq import heappush, heappop, heapify
from bisect import bisect_left, bisect_right
from functools import lru_cache
from math import gcd, lcm, isqrt, inf

# 高速入力（sys.stdin.readline は input() の約 3 倍速い）
input = sys.stdin.readline

def main():
    # ============================
    # 入力パターン集
    # ============================

    # 整数 1 つ
    N = int(input())

    # 整数 2 つ
    # N, M = map(int, input().split())

    # 整数のリスト
    A = list(map(int, input().split()))

    # 文字列
    # S = input().strip()  # strip() で末尾の改行を除去

    # N 行の入力（グラフの辺など）
    # edges = [tuple(map(int, input().split())) for _ in range(M)]

    # グリッド入力
    # grid = [input().strip() for _ in range(H)]

    # ============================
    # 解法をここに記述
    # ============================
    ans = 0

    # ============================
    # 出力
    # ============================
    print(ans)

if __name__ == "__main__":
    main()
```

### 2.2 入出力の高速化

Python は入出力がボトルネックになることが多い。以下のテクニックで大幅に高速化できる。

```python
import sys

# 方法 1: sys.stdin.readline を input に差し替え
input = sys.stdin.readline

# 方法 2: 全入力を一括読み込み（最速）
def solve():
    data = sys.stdin.read().split()
    idx = 0
    def rd():
        nonlocal idx
        idx += 1
        return data[idx - 1]

    N = int(rd())
    A = [int(rd()) for _ in range(N)]

    # 解法...
    ans = sum(A)
    print(ans)

solve()

# 方法 3: 大量出力の高速化
def fast_output(results):
    """リストの各要素を改行区切りで出力"""
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

# 方法 4: 再帰制限の拡張（DFS で深い再帰が必要な場合）
sys.setrecursionlimit(10**6)

# 方法 5: スレッドで再帰スタックサイズを確保
import threading
def main():
    sys.setrecursionlimit(10**6)
    # 再帰が深い DFS などの処理
    pass

threading.Thread(target=main, daemon=True).start()
```

### 2.3 C++ テンプレート（参考）

速度が求められる問題では C++ への切り替えが必要になることがある。以下は最小限の C++ テンプレートである。

```cpp
#include <bits/stdc++.h>
using namespace std;

// 型エイリアス
using ll = long long;
using pii = pair<int, int>;
using vi = vector<int>;
using vll = vector<ll>;

// 定数
const int INF = 1e9;
const ll LINF = 1e18;
const int MOD = 1e9 + 7;

// マクロ
#define rep(i, n) for (int i = 0; i < (n); i++)
#define all(v) (v).begin(), (v).end()

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    cin >> N;
    vi A(N);
    rep(i, N) cin >> A[i];

    ll ans = 0;
    // 解法をここに記述

    cout << ans << endl;
    return 0;
}
```

### 2.4 Python 高速化テクニック一覧

```
┌─────────────────────────────────────────────────────────────────┐
│              Python 高速化テクニック体系図                       │
├─────────────────┬───────────────────────────────────────────────┤
│ 入出力          │ sys.stdin.readline, sys.stdin.read()          │
│                 │ sys.stdout.write(), print のフラッシュ抑制     │
├─────────────────┼───────────────────────────────────────────────┤
│ ループ          │ リスト内包表記（for文の2-3倍速）               │
│                 │ map/filter の活用                              │
│                 │ ループ変数のローカル化                          │
├─────────────────┼───────────────────────────────────────────────┤
│ データ構造      │ set/dict のルックアップ O(1)                    │
│                 │ deque の両端操作 O(1)                          │
│                 │ heapq の優先度付きキュー                       │
│                 │ SortedContainers（pip install 不可の場合は BIT）│
├─────────────────┼───────────────────────────────────────────────┤
│ 算術            │ ビット演算で分岐を削減                          │
│                 │ math.isqrt, math.gcd の C 実装を活用            │
│                 │ numpy（AtCoder で使用可能）                     │
├─────────────────┼───────────────────────────────────────────────┤
│ 提出言語        │ PyPy（CPython の 3-10 倍速い）                  │
│                 │ Cython（コンパイル型で高速）                     │
├─────────────────┼───────────────────────────────────────────────┤
│ メモ化          │ @lru_cache（再帰 DP で必須）                    │
│                 │ 手動メモ化（dict ベース）                       │
└─────────────────┴───────────────────────────────────────────────┘
```

---

## 3. 典型テクニック: bit 全探索と半分全列挙

### 3.1 bit 全探索の基本

n 個の要素の全部分集合を列挙するテクニック。n が小さい（n <= 20 程度）とき、2^n 通りの組合せを整数のビット表現で管理する。ビットが 1 ならその要素を選択、0 なら非選択と対応させる。

```
要素:   [A, B, C, D]   (n = 4)
ビット: 0000 → {}
        0001 → {A}
        0010 → {B}
        0011 → {A, B}
        0100 → {C}
        ...
        1111 → {A, B, C, D}

全部で 2^4 = 16 通り
```

### 3.2 bit 全探索の実装

```python
def bit_bruteforce(n: int, items: list) -> list:
    """bit全探索で全部分集合を列挙する

    計算量: O(2^n * n)
    制約:   n <= 20 程度（2^20 = 約100万）
    """
    results = []
    for mask in range(1 << n):       # 0 から 2^n - 1 まで
        subset = []
        for i in range(n):
            if mask & (1 << i):      # i 番目のビットが立っているか
                subset.append(items[i])
        results.append(subset)
    return results

# 使用例
items = ['A', 'B', 'C']
subsets = bit_bruteforce(3, items)
# [[], ['A'], ['B'], ['A','B'], ['C'], ['A','C'], ['B','C'], ['A','B','C']]
```

### 3.3 部分和問題への適用

部分和問題は bit 全探索の代表的な適用先である。「配列の要素からいくつか選んで合計を target にできるか」を判定する。

```python
def subset_sum(arr: list, target: int) -> bool:
    """部分和問題を bit 全探索で解く

    計算量: O(2^n * n)
    制約:   n <= 20
    """
    n = len(arr)
    for mask in range(1 << n):
        total = 0
        for i in range(n):
            if mask & (1 << i):
                total += arr[i]
        if total == target:
            return True
    return False

# テスト
print(subset_sum([3, 7, 1, 8, 4], 12))   # True  (3 + 1 + 8 = 12)
print(subset_sum([3, 7, 1, 8, 4], 2))    # False
print(subset_sum([3, 7, 1, 8, 4], 23))   # True  (3 + 7 + 1 + 8 + 4 = 23)

# より効率的な実装（popcount で判定不要な場合のスキップ）
def subset_sum_optimized(arr: list, target: int) -> list:
    """target と一致する部分集合を全て返す"""
    n = len(arr)
    results = []
    for mask in range(1 << n):
        total = sum(arr[i] for i in range(n) if mask & (1 << i))
        if total == target:
            subset = [arr[i] for i in range(n) if mask & (1 << i)]
            results.append(subset)
    return results

print(subset_sum_optimized([3, 7, 1, 8, 4], 12))
# [[3, 1, 8], [8, 4]]
```

### 3.4 半分全列挙（Meet in the Middle）

n が 40 程度まで拡張する場合、配列を前半・後半に分割し、それぞれ 2^(n/2) で列挙してマッチングする。

```
全探索:     O(2^40) = 約 1 兆 → TLE
半分全列挙: O(2^20 * 20) = 約 2000 万 → 間に合う

┌────────────────────────────────────────────┐
│  配列 A を前半 L と後半 R に分割           │
│                                            │
│  L の全部分和を列挙: {s_1, s_2, ..., s_k}  │
│                ↓                           │
│  R の各部分和 t に対して                    │
│  target - t が L の集合に含まれるか確認     │
│                                            │
│  計算量: O(2^(n/2) * n)                    │
└────────────────────────────────────────────┘
```

```python
def meet_in_the_middle(arr: list, target: int) -> bool:
    """半分全列挙で部分和問題を解く

    計算量: O(2^(n/2) * n)
    制約:   n <= 40
    """
    n = len(arr)
    half = n // 2

    # 前半の全部分和を集合に格納
    left_sums = set()
    for mask in range(1 << half):
        s = sum(arr[i] for i in range(half) if mask & (1 << i))
        left_sums.add(s)

    # 後半の各部分和について、target との差が前半に存在するか確認
    for mask in range(1 << (n - half)):
        s = sum(arr[half + i] for i in range(n - half) if mask & (1 << i))
        if target - s in left_sums:
            return True

    return False

# テスト: n = 30 の部分和問題も高速に解ける
import random
arr = [random.randint(1, 10**9) for _ in range(30)]
target = sum(arr[:5])  # 最初の 5 要素の和を target にする
print(meet_in_the_middle(arr, target))  # True
```

### 3.5 bit 演算の小技集

```python
# ビット操作の基本テクニック（競プロで頻出）

# 1. i 番目のビットが立っているか
def is_set(mask, i):
    return bool(mask & (1 << i))

# 2. i 番目のビットを立てる
def set_bit(mask, i):
    return mask | (1 << i)

# 3. i 番目のビットを消す
def clear_bit(mask, i):
    return mask & ~(1 << i)

# 4. 最下位の立っているビットを取得
def lowest_bit(mask):
    return mask & (-mask)

# 5. 部分集合の列挙（mask の部分集合を降順で列挙）
def enumerate_subsets(mask):
    """mask のビットが立っている位置の部分集合を列挙"""
    sub = mask
    while sub > 0:
        yield sub
        sub = (sub - 1) & mask
    yield 0  # 空集合

# 6. popcount（立っているビットの数）
def popcount(mask):
    return bin(mask).count('1')

# 使用例: 3 つ選ぶ組合せを列挙
n = 5
for mask in range(1 << n):
    if popcount(mask) == 3:
        selected = [i for i in range(n) if mask & (1 << i)]
        print(selected)
# [0,1,2], [0,1,3], [0,1,4], [0,2,3], [0,2,4], ...
```

---

## 4. 典型テクニック: 座標圧縮と転倒数

### 4.1 座標圧縮の概要

座標圧縮（Coordinate Compression）は、大きな座標値を相対順序を保ったまま 0, 1, 2, ... に変換するテクニックである。値の大きさそのものではなく相対的な順序関係だけが重要な問題で使われる。

```
元の座標:    [100, 5000, 300, 100, 10000]

ステップ 1: 重複を除去してソート
  sorted_unique = [100, 300, 5000, 10000]

ステップ 2: 各値に 0-indexed の番号を割り当て
  100   → 0
  300   → 1
  5000  → 2
  10000 → 3

ステップ 3: 元の配列を変換
  圧縮後: [0, 2, 1, 0, 3]

メモリ効率:
  元の座標範囲: 0 〜 10000 → 配列サイズ 10001 必要
  圧縮後の範囲: 0 〜 3     → 配列サイズ 4 で十分
```

### 4.2 座標圧縮の実装

```python
def coordinate_compress(arr: list) -> tuple:
    """座標圧縮を行う

    戻り値: (圧縮後の配列, 復元用の配列)
    計算量: O(n log n)
    """
    sorted_unique = sorted(set(arr))
    compress = {v: i for i, v in enumerate(sorted_unique)}
    compressed = [compress[v] for v in arr]
    return compressed, sorted_unique

# 基本的な使用例
data = [100, 5000, 300, 100, 10000]
compressed, mapping = coordinate_compress(data)
print(compressed)  # [0, 2, 1, 0, 3]
print(mapping)     # [100, 300, 5000, 10000]

# 復元
original = [mapping[c] for c in compressed]
print(original)    # [100, 5000, 300, 100, 10000]
```

### 4.3 座標圧縮の応用: 転倒数の計算

転倒数（Inversion Count）は、配列中の i < j かつ A[i] > A[j] を満たすペア (i, j) の個数である。座標圧縮と BIT（Binary Indexed Tree）を組み合わせて O(n log n) で計算できる。

```python
def count_inversions(arr: list) -> int:
    """転倒数を座標圧縮 + BIT で計算する

    計算量: O(n log n)
    """
    # 座標圧縮
    compressed, _ = coordinate_compress(arr)
    n = len(compressed)
    max_val = max(compressed) + 1

    # BIT（Binary Indexed Tree）
    bit = [0] * (max_val + 2)

    def bit_update(i, delta=1):
        i += 1  # 1-indexed
        while i <= max_val + 1:
            bit[i] += delta
            i += i & (-i)

    def bit_query(i):
        """[0, i] の累積和"""
        i += 1
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    # 右から走査し、自分より小さい値がいくつ既に登場しているかを数える
    inversions = 0
    for i in range(n - 1, -1, -1):
        if compressed[i] > 0:
            inversions += bit_query(compressed[i] - 1)
        bit_update(compressed[i])

    return inversions

# テスト
print(count_inversions([3, 1, 2]))      # 2  (3>1, 3>2)
print(count_inversions([1, 2, 3]))      # 0  (ソート済み)
print(count_inversions([3, 2, 1]))      # 3  (3>2, 3>1, 2>1)
print(count_inversions([5, 2, 6, 1]))   # 4  (5>2, 5>1, 2>1, 6>1)
```

### 4.4 座標圧縮が必要な典型問題パターン

| パターン | 説明 | 座標圧縮の役割 |
|:---|:---|:---|
| 区間スケジューリング | 区間の端点が巨大な値 | 端点を圧縮して imos 法を適用 |
| 転倒数 | 値の範囲が広い | BIT のサイズを値の種類数に縮小 |
| 矩形の面積和 | 座標が 10^9 などの巨大な値 | 座標平面を離散化 |
| 区間の被覆判定 | 区間の端点が多数 | イベントソートの前処理 |
| 順位統計 | 値が離散的でない | 順位ベースのクエリに変換 |

---

## 5. 典型テクニック: しゃくとり法（Two Pointers）

### 5.1 しゃくとり法の概要

しゃくとり法（尺取り法、Two Pointers）は、配列上で 2 つのポインタ（left, right）を管理し、条件を満たす区間を効率的に探索するテクニックである。愚直に全区間を調べると O(N^2) だが、しゃくとり法を使えば O(N) で解ける。

「条件を満たす区間 [l, r) において、r を右に伸ばすと条件が成立しやすくなり、l を右に縮めると条件が成立しにくくなる」という単調性が成り立つ場合に適用できる。

```
しゃくとり法の動作イメージ:

配列:  [2, 5, 1, 3, 7, 2, 4]
条件:  区間の和が 10 以下

step 1:  [2]               sum=2   → 条件OK, right を伸ばす
step 2:  [2, 5]            sum=7   → 条件OK, right を伸ばす
step 3:  [2, 5, 1]         sum=8   → 条件OK, right を伸ばす
step 4:  [2, 5, 1, 3]      sum=11  → 条件NG, left を縮める
step 5:     [5, 1, 3]      sum=9   → 条件OK, right を伸ばす
step 6:     [5, 1, 3, 7]   sum=16  → 条件NG, left を縮める
step 7:        [1, 3, 7]   sum=11  → 条件NG, left を縮める
step 8:           [3, 7]   sum=10  → 条件OK, right を伸ばす
...

left と right は各々最大 N 回しか進まないため全体 O(N)
```

### 5.2 しゃくとり法の基本パターン

```python
def two_pointers_max_length(arr: list, threshold: int) -> int:
    """和が threshold 以下の最長連続部分列の長さを求める

    計算量: O(N)
    """
    n = len(arr)
    left = 0
    current_sum = 0
    max_len = 0

    for right in range(n):
        current_sum += arr[right]

        # 条件を満たさなくなるまで left を進める
        while current_sum > threshold:
            current_sum -= arr[left]
            left += 1

        # この時点で [left, right] は条件を満たす最長区間
        max_len = max(max_len, right - left + 1)

    return max_len

# テスト
print(two_pointers_max_length([2, 5, 1, 3, 7, 2, 4], 10))  # 3 ([2,5,1] or [5,1,3])

def two_pointers_count(arr: list, threshold: int) -> int:
    """和が threshold 以下の連続部分列の個数を求める

    計算量: O(N)
    ポイント: [l, r] が条件を満たすとき、[l, l], [l, l+1], ..., [l, r] の
              r - l + 1 個の区間も全て条件を満たす
    """
    n = len(arr)
    left = 0
    current_sum = 0
    count = 0

    for right in range(n):
        current_sum += arr[right]
        while current_sum > threshold:
            current_sum -= arr[left]
            left += 1
        count += right - left + 1  # right を右端とする条件を満たす区間の数

    return count

print(two_pointers_count([2, 5, 1, 3, 7, 2, 4], 10))  # 16
```

### 5.3 しゃくとり法の応用: 種類数の管理

「区間内の異なる値の種類数が K 以下」のような条件でも使える。

```python
from collections import defaultdict

def at_most_k_distinct(arr: list, k: int) -> int:
    """異なる値が最大 k 種類の最長連続部分列の長さ

    計算量: O(N)
    """
    n = len(arr)
    left = 0
    freq = defaultdict(int)
    distinct = 0
    max_len = 0

    for right in range(n):
        if freq[arr[right]] == 0:
            distinct += 1
        freq[arr[right]] += 1

        while distinct > k:
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                distinct -= 1
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len

# テスト: 異なる値が最大 2 種類
print(at_most_k_distinct([1, 2, 1, 2, 3, 3, 4], 2))  # 4 ([1,2,1,2])

def exactly_k_distinct(arr: list, k: int) -> int:
    """異なる値がちょうど k 種類の連続部分列の個数

    テクニック: exactly(k) = at_most(k) - at_most(k-1)
    """
    def at_most(k):
        if k < 0:
            return 0
        left = 0
        freq = defaultdict(int)
        distinct = 0
        count = 0
        for right in range(len(arr)):
            if freq[arr[right]] == 0:
                distinct += 1
            freq[arr[right]] += 1
            while distinct > k:
                freq[arr[left]] -= 1
                if freq[arr[left]] == 0:
                    distinct -= 1
                left += 1
            count += right - left + 1
        return count

    return at_most(k) - at_most(k - 1)

print(exactly_k_distinct([1, 2, 1, 2, 3], 2))  # 7
```

### 5.4 しゃくとり法の適用判定フローチャート

```
問題を見たとき:
  │
  ├─ 「連続部分列」「区間」のキーワードがある？
  │   ├─ No → しゃくとり法は使えない可能性が高い
  │   └─ Yes
  │       │
  │       ├─ 区間を広げると条件が「緩和」される？
  │       │   （例: 和が大きくなる、種類数が増える）
  │       │   ├─ Yes → しゃくとり法が適用可能
  │       │   └─ No  → 別の手法を検討
  │       │
  │       └─ 条件に「単調性」があるか？
  │           ├─ Yes → しゃくとり法
  │           └─ No  → セグメント木や他の手法
  │
  └─ 「2つの配列」を同時に走査する必要がある？
      └─ Yes → Two Pointers（マージ操作、ソート済み配列の活用）
```

---

## 6. 典型テクニック: MOD 演算と組合せ計算

### 6.1 MOD 演算の必要性

競技プログラミングでは「答えを 10^9 + 7 で割った余りを求めよ」という形式の問題が非常に多い。これは以下の理由による。

1. 答えが非常に大きくなる問題（フィボナッチ数列の第 10^18 項など）で桁溢れを防ぐ
2. 多倍長整数演算のオーバーヘッドを避け、計算量を削減する
3. 解の一意性を保証する（10^9 + 7 は素数なので、逆元が存在する）

```
よく使われる MOD の値:
  998244353 = 119 * 2^23 + 1  （NTT に適した素数）
  10^9 + 7  = 1000000007      （最も一般的）
  10^9 + 9  = 1000000009      （ハッシュ用）
```

### 6.2 MOD 演算の基本実装

```python
MOD = 10**9 + 7

# 基本演算（加算・乗算は素直に MOD を取るだけ）
def mod_add(a, b, mod=MOD):
    return (a + b) % mod

def mod_sub(a, b, mod=MOD):
    return (a - b) % mod  # Python は負の余りを正しく処理する

def mod_mul(a, b, mod=MOD):
    return (a * b) % mod

# 繰り返し二乗法（累乗の高速計算）
def mod_pow(base, exp, mod=MOD):
    """base^exp mod mod を O(log exp) で計算する"""
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:              # exp が奇数
            result = result * base % mod
        exp >>= 1               # exp を半分にする
        base = base * base % mod
    return result

# Python 組み込みの pow(base, exp, mod) も同じ計算をする
# pow(2, 10, MOD) == mod_pow(2, 10, MOD) == 1024

# モジュラ逆元（フェルマーの小定理: a^(-1) ≡ a^(p-2) mod p, p は素数）
def mod_inv(a, mod=MOD):
    """a の mod 上の逆元を求める（mod が素数の場合）"""
    return mod_pow(a, mod - 2, mod)

# 除算: a / b mod p = a * b^(-1) mod p
def mod_div(a, b, mod=MOD):
    return a * mod_inv(b, mod) % mod
```

### 6.3 組合せ計算（nCr mod p）

組合せ数 nCr を MOD 付きで高速に計算するには、階乗とその逆元を前計算する。

```python
class Combinatorics:
    """組合せ計算を O(1) で行うためのクラス

    前計算: O(max_n)
    クエリ: O(1)
    """
    def __init__(self, max_n: int, mod: int = MOD):
        self.mod = mod
        self.fact = [1] * (max_n + 1)       # fact[i] = i! mod p
        self.inv_fact = [1] * (max_n + 1)   # inv_fact[i] = (i!)^(-1) mod p

        # 階乗の前計算
        for i in range(1, max_n + 1):
            self.fact[i] = self.fact[i - 1] * i % mod

        # 逆元の前計算（最大値から逆算）
        self.inv_fact[max_n] = pow(self.fact[max_n], mod - 2, mod)
        for i in range(max_n - 1, -1, -1):
            self.inv_fact[i] = self.inv_fact[i + 1] * (i + 1) % mod

    def comb(self, n: int, r: int) -> int:
        """nCr mod p を O(1) で返す"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[r] % self.mod * self.inv_fact[n - r] % self.mod

    def perm(self, n: int, r: int) -> int:
        """nPr mod p を O(1) で返す"""
        if r < 0 or r > n:
            return 0
        return self.fact[n] * self.inv_fact[n - r] % self.mod

    def homo(self, n: int, r: int) -> int:
        """重複組合せ nHr = (n+r-1)Cr mod p"""
        return self.comb(n + r - 1, r)

    def catalan(self, n: int) -> int:
        """カタラン数 C_n = (2n)Cn / (n+1)"""
        return self.comb(2 * n, n) * pow(n + 1, self.mod - 2, self.mod) % self.mod

# 使用例
comb = Combinatorics(200000)
print(comb.comb(10, 3))         # 120
print(comb.comb(100000, 50000)) # 大きな値 mod 10^9+7
print(comb.perm(5, 3))          # 60
print(comb.catalan(5))          # 42
```

### 6.4 MOD 演算で気をつけるべきポイント

```
┌──────────────────────────────────────────────────────────────┐
│        MOD 演算の落とし穴と対策                               │
├────────────────────┬─────────────────────────────────────────┤
│ 落とし穴            │ 対策                                    │
├────────────────────┼─────────────────────────────────────────┤
│ 引き算で負になる    │ (a - b % MOD + MOD) % MOD               │
│                    │ ※ Python は自動で正の余りを返すので不要  │
├────────────────────┼─────────────────────────────────────────┤
│ 除算に MOD を       │ 逆元を掛ける: a * mod_inv(b) % MOD      │
│ 直接適用する        │ 割り算に % を使ってはいけない             │
├────────────────────┼─────────────────────────────────────────┤
│ 中間値のオーバー    │ 各ステップで % MOD を取る                │
│ フロー (C++)       │ Python は多倍長なので問題にならない       │
├────────────────────┼─────────────────────────────────────────┤
│ MOD が素数でない    │ フェルマーの小定理が使えない              │
│ 場合の逆元          │ 拡張ユークリッドの互除法を使う            │
├────────────────────┼─────────────────────────────────────────┤
│ 998244353 と        │ 問題文をよく読む                         │
│ 10^9+7 を間違える   │ MOD は定数として最初に定義する            │
└────────────────────┴─────────────────────────────────────────┘
```

---

## 7. グラフアルゴリズムの競プロ実装

### 7.1 グラフの入力パターン

競技プログラミングでは、グラフの入力形式にいくつかの典型パターンがある。

```python
# パターン 1: 隣接リスト（重みなし）
# 入力:
# N M
# a1 b1
# a2 b2
# ...

def read_graph_unweighted():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        a, b = map(int, input().split())
        a -= 1; b -= 1   # 0-indexed に変換
        graph[a].append(b)
        graph[b].append(a)  # 無向グラフの場合
    return N, graph

# パターン 2: 隣接リスト（重みあり）
# 入力:
# N M
# a1 b1 c1
# ...

def read_graph_weighted():
    N, M = map(int, input().split())
    graph = [[] for _ in range(N)]
    for _ in range(M):
        a, b, c = map(int, input().split())
        a -= 1; b -= 1
        graph[a].append((b, c))
        graph[b].append((a, c))
    return N, graph

# パターン 3: 木（親の指定）
# 入力: N 頂点の木で、頂点 i (2 <= i <= N) の親が p_i
# p_2 p_3 ... p_N

def read_tree_parent():
    N = int(input())
    parent = [-1] + [-1] + list(map(int, input().split()))
    # parent[i] = 頂点 i の親 (1-indexed)
    children = [[] for _ in range(N + 1)]
    for i in range(2, N + 1):
        children[parent[i]].append(i)
    return N, parent, children
```

### 7.2 ダイクストラ法（競プロ版）

```python
import heapq

def dijkstra(graph: list, start: int) -> list:
    """ダイクストラ法（優先度付きキュー版）

    graph[u] = [(v, cost), ...]
    計算量: O((V + E) log V)
    """
    INF = float('inf')
    n = len(graph)
    dist = [INF] * n
    dist[start] = 0
    pq = [(0, start)]  # (距離, 頂点)

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:     # 既により短い経路が見つかっている
            continue
        for v, cost in graph[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist

# 経路復元が必要な場合
def dijkstra_with_path(graph: list, start: int, goal: int) -> tuple:
    """ダイクストラ法 + 経路復元"""
    INF = float('inf')
    n = len(graph)
    dist = [INF] * n
    prev_node = [-1] * n
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == goal:
            break
        for v, cost in graph[u]:
            nd = d + cost
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                heapq.heappush(pq, (nd, v))

    # 経路復元
    if dist[goal] == INF:
        return INF, []
    path = []
    v = goal
    while v != -1:
        path.append(v)
        v = prev_node[v]
    path.reverse()
    return dist[goal], path
```

### 7.3 BFS / DFS の競プロテンプレート

```python
from collections import deque

def bfs(graph: list, start: int) -> list:
    """BFS（幅優先探索）- 重みなしグラフの最短距離

    計算量: O(V + E)
    """
    n = len(graph)
    dist = [-1] * n
    dist[start] = 0
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if dist[v] == -1:   # 未訪問
                dist[v] = dist[u] + 1
                queue.append(v)

    return dist

def dfs_iterative(graph: list, start: int) -> list:
    """DFS（深さ優先探索）- スタック版（再帰上限回避）

    計算量: O(V + E)
    """
    n = len(graph)
    visited = [False] * n
    order = []
    stack = [start]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)
        for v in graph[u]:
            if not visited[v]:
                stack.append(v)

    return order

# グリッド上の BFS（迷路の最短経路など）
def grid_bfs(grid: list, start: tuple, goal: tuple) -> int:
    """グリッド上の BFS

    grid[i][j] = '.' (通行可能) or '#' (壁)
    """
    H, W = len(grid), len(grid[0])
    dist = [[-1] * W for _ in range(H)]
    sy, sx = start
    gy, gx = goal
    dist[sy][sx] = 0
    queue = deque([(sy, sx)])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右左下上

    while queue:
        y, x = queue.popleft()
        if (y, x) == (gy, gx):
            return dist[y][x]
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] != '#' and dist[ny][nx] == -1:
                dist[ny][nx] = dist[y][x] + 1
                queue.append((ny, nx))

    return -1  # 到達不可能
```

### 7.4 Union-Find（素集合データ構造）

```python
class UnionFind:
    """Union-Find（素集合データ構造）

    経路圧縮 + union by rank で
    find, union ともにならし O(alpha(N)) ≈ O(1)
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_groups = n

    def find(self, x: int) -> int:
        """x の根を返す（経路圧縮付き）"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        """x と y を同じグループに統合する。既に同じなら False"""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.num_groups -= 1
        return True

    def connected(self, x: int, y: int) -> bool:
        """x と y が同じグループか"""
        return self.find(x) == self.find(y)

    def group_size(self, x: int) -> int:
        """x が属するグループのサイズ"""
        return self.size[self.find(x)]

# 使用例: 連結成分の数を求める
def count_connected_components():
    N, M = map(int, input().split())
    uf = UnionFind(N)
    for _ in range(M):
        a, b = map(int, input().split())
        uf.union(a - 1, b - 1)
    print(uf.num_groups)
```

### 7.5 グラフアルゴリズム選択チャート

```
問題の性質から適切なアルゴリズムを選ぶ:

  最短経路を求めたい
  │
  ├─ 重みなし？ → BFS  O(V+E)
  │
  ├─ 重みが全て非負？
  │   ├─ 単一始点？ → ダイクストラ  O((V+E) log V)
  │   └─ 全頂点間？ → ワーシャルフロイド  O(V^3)
  │
  └─ 負の重みあり？
      ├─ 負閉路なし → ベルマンフォード  O(VE)
      └─ 負閉路検出も → ベルマンフォード（N 回目の更新チェック）

  連結性を判定したい
  │
  ├─ 静的（辺の追加のみ）→ Union-Find  O(α(N))
  └─ 動的（辺の削除あり）→ オフライン処理 or Link-Cut Tree

  木の問題
  │
  ├─ LCA（最近共通祖先）→ ダブリング  O(N log N)前処理, O(log N)クエリ
  ├─ パスの重み → Euler Tour + セグメント木
  └─ 部分木の情報 → DFS 順序 + BIT/セグメント木
```

---

## 8. 累積和・imos 法・二分探索

### 8.1 累積和（Prefix Sum）

累積和は、配列の区間和クエリを O(1) で答えるための前処理テクニックである。前計算 O(N)、クエリ O(1) という時間計算量で、区間和を扱う問題の定番手法。

```python
from itertools import accumulate

def prefix_sum_demo():
    """1 次元累積和の基本"""
    A = [3, 1, 4, 1, 5, 9, 2, 6]

    # 累積和の構築（先頭に 0 を置く）
    prefix = [0] + list(accumulate(A))
    # prefix = [0, 3, 4, 8, 9, 14, 23, 25, 31]

    # 区間 [l, r) の和 = prefix[r] - prefix[l]
    # 例: A[1] + A[2] + A[3] + A[4] = 1+4+1+5 = 11
    print(prefix[5] - prefix[1])  # 11

    # 全区間の和
    print(prefix[8] - prefix[0])  # 31

# 2 次元累積和
def prefix_sum_2d(grid: list) -> list:
    """2 次元累積和の構築

    H x W のグリッドに対して O(HW) で前計算
    矩形区間の和を O(1) で取得
    """
    H = len(grid)
    W = len(grid[0])
    # (H+1) x (W+1) の累積和テーブル
    ps = [[0] * (W + 1) for _ in range(H + 1)]

    for i in range(H):
        for j in range(W):
            ps[i + 1][j + 1] = (
                grid[i][j]
                + ps[i][j + 1]
                + ps[i + 1][j]
                - ps[i][j]
            )
    return ps

def query_2d(ps, r1, c1, r2, c2):
    """矩形 [r1, r2) x [c1, c2) の和を O(1) で取得"""
    return ps[r2][c2] - ps[r1][c2] - ps[r2][c1] + ps[r1][c1]

# テスト
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
ps = prefix_sum_2d(grid)
print(query_2d(ps, 0, 0, 2, 2))  # 1+2+4+5 = 12
print(query_2d(ps, 1, 1, 3, 3))  # 5+6+8+9 = 28
```

### 8.2 imos 法（差分配列）

imos 法は、多数の区間加算操作を効率的に処理するテクニック。各区間の開始点と終了点にだけ +1, -1 の操作を記録し、最後に累積和を取ることで全区間の加算結果を得る。

```
imos 法の動作例:

区間 [1, 4) に +1, [2, 6) に +1, [3, 5) に +1 を加算

ステップ 1: 差分配列に記録
  index:  0  1  2  3  4  5  6  7
  diff:  [0, +1, +1, +1, -1, -1, -1, 0]
                        ↑ [1,4)の-1  ↑ [2,6)の-1
                   ↑ [3,5)の+1

ステップ 2: 累積和を取る
  result: [0, 1, 2, 3, 2, 1, 0, 0]

  区間の重なりが自動的に計算される
```

```python
def imos_1d(n: int, intervals: list) -> list:
    """1 次元 imos 法

    intervals: [(l, r), ...] 各区間 [l, r) に +1 を加算
    計算量: O(N + Q)  (N: 配列サイズ, Q: 区間の数)
    """
    diff = [0] * (n + 1)
    for l, r in intervals:
        diff[l] += 1
        diff[r] -= 1

    # 累積和で復元
    result = [0] * n
    result[0] = diff[0]
    for i in range(1, n):
        result[i] = result[i - 1] + diff[i]

    return result

# テスト
intervals = [(1, 4), (2, 6), (3, 5)]
print(imos_1d(7, intervals))  # [0, 1, 2, 3, 2, 1, 0]

def imos_2d(H: int, W: int, rectangles: list) -> list:
    """2 次元 imos 法

    rectangles: [(r1, c1, r2, c2), ...] 矩形 [r1,r2) x [c1,c2) に +1
    計算量: O(HW + Q)
    """
    diff = [[0] * (W + 1) for _ in range(H + 1)]
    for r1, c1, r2, c2 in rectangles:
        diff[r1][c1] += 1
        diff[r1][c2] -= 1
        diff[r2][c1] -= 1
        diff[r2][c2] += 1

    # 横方向の累積和
    for i in range(H):
        for j in range(1, W):
            diff[i][j] += diff[i][j - 1]
    # 縦方向の累積和
    for j in range(W):
        for i in range(1, H):
            diff[i][j] += diff[i - 1][j]

    return [row[:W] for row in diff[:H]]
```

### 8.3 二分探索（Binary Search）

二分探索は「答えに単調性がある」場合に、探索範囲を半分ずつ狭めて O(log N) で答えを見つけるテクニック。

```python
from bisect import bisect_left, bisect_right

# パターン 1: ソート済み配列での検索
def binary_search_in_sorted(arr: list):
    """bisect モジュールの活用"""
    # arr = [1, 1, 2, 3, 4, 5, 6, 9]

    # x 以上の最小インデックス
    idx = bisect_left(arr, 3)   # 3

    # x より大きい最小インデックス
    idx = bisect_right(arr, 3)  # 4

    # x 以上の要素数
    count_ge = len(arr) - bisect_left(arr, 3)  # 5

    # x 以下の要素数
    count_le = bisect_right(arr, 3)             # 4

    # x が存在するか
    idx = bisect_left(arr, 3)
    exists = idx < len(arr) and arr[idx] == 3   # True

# パターン 2: 答えで二分探索（二分探索の一般化）
def binary_search_on_answer():
    """「条件を満たす最小（最大）の値を求めよ」型の問題"""

    def is_ok(mid: int) -> bool:
        """mid が条件を満たすか判定する関数"""
        # 問題に応じて実装
        return True

    # 条件を満たす最小値を求める
    lo, hi = 0, 10**18  # 探索範囲
    while lo < hi:
        mid = (lo + hi) // 2
        if is_ok(mid):
            hi = mid       # 条件を満たす → 左半分を探索
        else:
            lo = mid + 1   # 条件を満たさない → 右半分を探索
    # lo == hi が答え

    # 実数の二分探索（浮動小数点の場合）
    lo, hi = 0.0, 1e18
    for _ in range(100):   # 十分な回数ループ
        mid = (lo + hi) / 2
        if is_ok(int(mid)):
            hi = mid
        else:
            lo = mid
    # lo ≈ hi が答え

# パターン 3: 最小値の最大化（典型問題）
def minimize_maximum_distance(positions: list, k: int) -> int:
    """N 個の位置に K 個の中継点を追加し、
    隣接する点の最大距離を最小化する

    「答え d 以下にできるか？」を二分探索
    """
    positions.sort()

    def can_achieve(d):
        """最大距離を d 以下にするために必要な中継点数"""
        count = 0
        for i in range(len(positions) - 1):
            gap = positions[i + 1] - positions[i]
            count += (gap - 1) // d  # この区間に必要な中継点数
        return count <= k

    lo, hi = 1, positions[-1] - positions[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if can_achieve(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo
```

### 8.4 累積和・imos 法・二分探索の使い分け

| 手法 | 前計算 | クエリ | 適用場面 |
|:---|:---|:---|:---|
| 累積和 | O(N) | O(1) | 静的な区間和クエリ |
| 2D 累積和 | O(HW) | O(1) | 矩形の和クエリ |
| imos 法 | O(N+Q) | - | 多数の区間加算を一括処理 |
| 2D imos 法 | O(HW+Q) | - | 多数の矩形加算を一括処理 |
| 二分探索 | O(N log N) | O(log N) | ソート済み配列の検索 |
| 答えで二分探索 | - | O(log X * f(N)) | 単調性のある最適化問題 |

---

## 9. AtCoder / LeetCode / Codeforces パターン分析

### 9.1 AtCoder ABC 頻出パターン

ABC（AtCoder Beginner Contest）は A〜G の 7 問構成で、各問題の難易度と典型パターンは以下の通り。

```
問題  配点    想定ランク    典型パターン
─────────────────────────────────────────────────────────────
A     100点   灰           四則演算、条件分岐、文字列出力
B     200点   灰〜茶       ループ、文字列操作、シミュレーション
C     300点   茶〜緑       全探索、ソート、累積和、貪欲法
D     400点   緑〜水       DP、二分探索、BFS/DFS、Union-Find
E     500点   水〜青       セグメント木、高度な DP、数論
F     500点   青〜黄       フロー、行列累乗、高度な組合せ論
G     600点   黄〜橙       構築問題、高度なデータ構造の組合せ
```

**レベル別の学習目標と推奨パターン:**

| 目標 | 安定して解くべき問題 | 習得すべきパターン |
|:---|:---|:---|
| 灰 → 茶 | A, B を確実、C を半分 | ループ、条件分岐、ソート |
| 茶 → 緑 | A, B, C を確実、D を半分 | DP 基礎、BFS、累積和 |
| 緑 → 水 | A〜D を確実、E を半分 | セグ木、Union-Find、高度 DP |
| 水 → 青 | A〜E を確実、F に挑戦 | 数論、フロー、包除原理 |

### 9.2 LeetCode 頻出パターン

LeetCode は面接対策として利用されることが多く、問題はパターン別に分類できる。Blind 75 や NeetCode 150 と呼ばれる頻出問題リストが広く知られている。

```
パターン別出題頻度と重要度:

  カテゴリ               出題頻度       面接での重要度
  ─────────────────────────────────────────────────────
  配列 / ハッシュマップ   ████████████   極めて高い
  二分探索               ██████████     高い
  スライディングウィンドウ █████████     高い
  Two Pointers           ████████      高い
  木 / グラフ             ████████      高い
  動的計画法             ██████████     非常に高い
  バックトラッキング      ██████        中程度
  スタック / キュー       ██████        中程度
  ヒープ                 █████         中程度
  グリーディ             █████         中程度
  文字列                 ████          中程度
  トライ / セグメント木   ███           低い（上級向け）
```

**面接で頻出する問題例（難易度別）:**

- **Easy:** Two Sum, Valid Parentheses, Merge Two Sorted Lists, Best Time to Buy and Sell Stock, Valid Palindrome, Linked List Cycle, Invert Binary Tree
- **Medium:** 3Sum, Container With Most Water, LRU Cache, Course Schedule, Word Break, Coin Change, Number of Islands, Group Anagrams, Longest Substring Without Repeating Characters
- **Hard:** Merge K Sorted Lists, Trapping Rain Water, Word Ladder II, Sliding Window Maximum, Minimum Window Substring

### 9.3 Codeforces の特徴とパターン

Codeforces は問題数が圧倒的に多く、数学的な考察を要求する問題が AtCoder 以上に出題される。Div.2 の A〜E が一般的な出題形式。

```
Div.2 の各問題の特徴:

  A (800-1000):  基本的な数学、実装
  B (1000-1300): 貪欲法、構築、場合分け
  C (1300-1600): 二分探索、DP、グラフ基礎
  D (1600-2000): 高度な DP、数論、セグメント木
  E (2000-2400): 高難度の組合せ問題、構築

特徴的な出題傾向:
  - 構築問題（Constructive Algorithm）が非常に多い
  - 数論（GCD, LCM, 素因数分解）の出題頻度が高い
  - インタラクティブ問題が定期的に出題される
  - ゲーム理論（Nim, Sprague-Grundy）がときどき出題される
```

### 9.4 プラットフォーム横断比較表

| 特性 | AtCoder | LeetCode | Codeforces |
|:---|:---|:---|:---|
| 主な用途 | 競技・スキル向上 | 面接対策 | 競技・スキル向上 |
| 言語 | 日本語 | 英語 | 英語 |
| 難易度表示 | 色レーティング | Easy/Medium/Hard | *A-*F + Rating |
| コンテスト頻度 | 週 1 回 (土曜) | 週 2 回 | 週 2-3 回 |
| 入力形式 | 標準入力 | 関数引数 | 標準入力 |
| Python 対応 | PyPy 可 | Python3 | PyPy 可 |
| 解説 | 公式 editorial | Discussion | 有志 editorial |
| 問題数 | 約 5000+ | 約 3000+ | 約 10000+ |
| 特徴 | 高品質・丁寧な解説 | 企業別問題集 | 問題量が圧倒的 |
| コミュニティ | 日本語主体 | 英語主体 | 英語・ロシア語主体 |

---

## 10. コンテスト戦略とメンタルモデル

### 10.1 時間配分の基本戦略

```
AtCoder ABC 100 分間の最適時間配分:

  0-5 分:    A 問題を確実に解く（ウォームアップ）
  5-15 分:   B 問題を確実に解く（慎重に実装）
  15-35 分:  C 問題に取り組む（ここが分かれ目）
  35-65 分:  D 問題に集中する（最も配点効率が高い）
  65-90 分:  E 問題に挑戦する（解ければ大幅加点）
  90-100 分: 見直し・未 AC の問題を再検討

  重要ポイント:
  ・1 問に 30 分以上かけない（次の問題に進む判断力が重要）
  ・最後 5 分は新しい提出をしない（ペナルティ回避）
  ・WA が出たら、まずサンプルケースを再確認

LeetCode Weekly Contest 90 分間の最適時間配分:

  0-5 分:    Q1 (Easy) を素早く解く
  5-20 分:   Q2 (Medium) を解く
  20-55 分:  Q3 (Medium-Hard) に取り組む
  55-90 分:  Q4 (Hard) に挑戦
```

### 10.2 デバッグ戦略

コンテスト中に WA（Wrong Answer）や RE（Runtime Error）が出た場合の体系的なデバッグ手法を整理する。

```python
# デバッグ戦略 1: 愚直解とのストレステスト
import random

def brute_force(arr):
    """愚直な O(N^2) 解"""
    # ...正しいが遅い解法
    pass

def optimized(arr):
    """最適化した O(N log N) 解"""
    # ...高速だが正しいか不明な解法
    pass

def stress_test(n_tests=10000, max_n=10, max_val=100):
    """ランダムテストで愚直解と最適解を比較"""
    for _ in range(n_tests):
        n = random.randint(1, max_n)
        arr = [random.randint(1, max_val) for _ in range(n)]
        expected = brute_force(arr)
        actual = optimized(arr)
        if expected != actual:
            print(f"MISMATCH: arr={arr}")
            print(f"  expected={expected}, actual={actual}")
            return
    print("All tests passed!")

# デバッグ戦略 2: エッジケースチェックリスト
EDGE_CASES = """
  N = 0 (空入力)
  N = 1 (要素 1 つ)
  N = 2 (最小のペア)
  全要素が同じ値
  全要素が最大値 (10^9 等)
  ソート済み (昇順)
  ソート済み (降順)
  負の値が含まれる場合
  オーバーフローする可能性のある計算
  0-indexed と 1-indexed の変換ミス
"""
```

### 10.3 問題を読み解くフレームワーク

問題を見たとき、以下の手順で解法を絞り込む。

```
ステップ 1: 制約を確認する
  │
  ├─ N <= 8        → bit 全探索 or 順列全探索
  ├─ N <= 20       → bit 全探索
  ├─ N <= 40       → 半分全列挙
  ├─ N <= 300      → O(N^3) が通る（DP, ワーシャルフロイド）
  ├─ N <= 3000     → O(N^2) が通る
  ├─ N <= 10^5     → O(N log N) or O(N sqrt(N))
  ├─ N <= 10^6     → O(N) or O(N log N)
  └─ N <= 10^18    → O(log N) or O(sqrt(N))（数学的手法）

ステップ 2: 問題の型を分類する
  │
  ├─ 最短経路      → BFS / ダイクストラ / ベルマンフォード
  ├─ 最大・最小の最適化 → DP / 貪欲法 / 二分探索
  ├─ 数え上げ      → DP / 包除原理 / MOD 演算
  ├─ 連結性        → Union-Find / BFS / DFS
  ├─ 区間クエリ    → セグメント木 / BIT / 累積和
  └─ 文字列マッチ  → KMP / Z-algorithm / ローリングハッシュ

ステップ 3: 計算量を見積もって実装に移る
```

### 10.4 メンタル管理

```
コンテスト中のメンタル管理のルール:

  1. 1 問に詰まったら次に進む
     → 後の問題の方が簡単な場合がある（特に E が D より解きやすいケース）

  2. WA（不正解）でパニックにならない
     → まずサンプルケースを手で確認、次にエッジケースを検討

  3. 順位は気にしない
     → コンテスト中に順位表を見ると焦る原因になる

  4. 提出前に 1 分間の見直し
     → 変数名の typo、off-by-one エラー、出力形式の確認

  5. 終了後は必ず振り返りをする
     → 解けなかった問題の editorial を読み、1 週間以内に自力で再実装
```

---

## 11. プラットフォーム比較と学習ロードマップ

### 11.1 典型テクニック速見表

| テクニック | 計算量 | 適用場面 | 出現頻度 |
|:---|:---|:---|:---|
| 累積和 | O(N) 前処理 | 区間和クエリ | 非常に高い |
| 二分探索 | O(log N) | 単調性のある探索 | 非常に高い |
| しゃくとり法 | O(N) | 条件を満たす区間 | 高い |
| bit 全探索 | O(2^N) | N<=20 の全列挙 | 中程度 |
| 座標圧縮 | O(N log N) | 大きな値の離散化 | 中程度 |
| MOD 演算 | O(N) | 巨大な数の計算 | 非常に高い |
| Union-Find | O(alpha(N)) | 連結成分管理 | 高い |
| ダイクストラ | O((V+E) log V) | 重み付き最短経路 | 高い |
| ダブリング | O(N log N) | LCA, 繰り返し遷移 | 中程度 |
| セグメント木 | O(N) 構築, O(log N) クエリ | 区間クエリ全般 | 高い（水以上） |
| 行列累乗 | O(K^3 log N) | 線形漸化式の高速化 | 低い |
| 遅延評価セグ木 | O(log N) | 区間更新 + 区間クエリ | 中程度（青以上） |

### 11.2 段階別学習ロードマップ

**Phase 1: 入門（灰 → 茶, 目安 1-3 か月）**

学習内容:
- プログラミング言語の基本文法（Python or C++）
- 入出力の処理、ループ、条件分岐
- ソートアルゴリズムの使い方
- 線形探索と基本的な全探索

推奨問題:
- AtCoder ABC の A, B 問題を 50 問以上
- AtCoder Beginners Selection（公式推奨 10 問）

**Phase 2: 基礎（茶 → 緑, 目安 3-6 か月）**

学習内容:
- 累積和、二分探索
- BFS / DFS の基本
- DP の基本（ナップサック、LCS、LIS）
- 貪欲法の基本パターン
- bit 全探索

推奨問題:
- AtCoder ABC の C, D 問題を 100 問以上
- 競プロ典型 90 問（E869120 作）の星 2-3
- Educational DP Contest（EDPC）の A〜K

**Phase 3: 中級（緑 → 水, 目安 6 か月-1 年）**

学習内容:
- セグメント木と BIT
- Union-Find
- ダイクストラ法、ワーシャルフロイド法
- 高度な DP（桁 DP、木 DP、ビット DP）
- 数論の基本（素因数分解、MOD 逆元、中国剰余定理）

推奨問題:
- AtCoder ABC の E, F 問題
- 競プロ典型 90 問の星 4-5
- AtCoder Library Practice Contest

**Phase 4: 上級（水 → 青以上, 目安 1 年以上）**

学習内容:
- 遅延評価セグメント木
- 最大フロー、最小カット
- 行列累乗、包除原理
- 文字列アルゴリズム（SA, LCP, Aho-Corasick）
- 平面走査法、凸包

推奨問題:
- ARC, AGC の過去問
- Codeforces Div.1 の問題
- IOI / ICPC の過去問

### 11.3 学習リソース一覧

```
┌──────────────────────────────────────────────────────────────────┐
│                  学習リソースマップ                               │
├───────────────┬──────────────────────────────────────────────────┤
│ 書籍          │ ・プログラミングコンテストチャレンジブック(蟻本)  │
│               │ ・問題解決力を鍛える! アルゴリズムとデータ構造     │
│               │ ・Competitive Programming 3 (Halim)               │
│               │ ・アルゴリズムイントロダクション (CLRS)            │
├───────────────┼──────────────────────────────────────────────────┤
│ Webサイト     │ ・AtCoder Problems (kenkoooo)                     │
│               │ ・NeetCode 150                                    │
│               │ ・競プロ典型90問                                   │
│               │ ・EDPC (Educational DP Contest)                   │
│               │ ・algo-method                                     │
├───────────────┼──────────────────────────────────────────────────┤
│ ライブラリ    │ ・AtCoder Library (ACL)                           │
│               │ ・Python: sortedcontainers, networkx              │
│               │ ・C++: bits/stdc++.h, pb_ds                      │
├───────────────┼──────────────────────────────────────────────────┤
│ コミュニティ  │ ・AtCoder 公式 Discord                            │
│               │ ・Twitter/X の #競プロ タグ                       │
│               │ ・Codeforces Blog                                 │
│               │ ・LeetCode Discussion                            │
└───────────────┴──────────────────────────────────────────────────┘
```

---

## 12. 演習問題（3 段階）

### 12.1 初級演習（灰〜茶レベル）

**演習 1-1: 部分和判定**

> 配列 A = [2, 7, 3, 5, 11] と目標値 target = 15 が与えられる。
> A の要素からいくつか選んで合計が target になるか判定せよ。

ヒント: N <= 5 なので bit 全探索（2^5 = 32 通り）で十分。

```python
# 解答例
def solve_1_1():
    A = [2, 7, 3, 5, 11]
    target = 15
    n = len(A)
    for mask in range(1 << n):
        total = sum(A[i] for i in range(n) if mask & (1 << i))
        if total == target:
            subset = [A[i] for i in range(n) if mask & (1 << i)]
            print(f"Yes: {subset}")
            return
    print("No")

solve_1_1()  # Yes: [7, 3, 5] (合計 15)
```

**演習 1-2: 最長連続部分列**

> 配列 A = [1, 3, 2, 5, 4, 7, 6, 8] から、連続する部分列で和が 12 以下となる最長の長さを求めよ。

ヒント: しゃくとり法を使う。

**演習 1-3: グリッド最短路**

> 以下の 5x5 グリッドで、左上 (0,0) から右下 (4,4) への最短経路の長さを求めよ。'.' は通行可能、'#' は壁。

```
.....
.#.#.
.#...
...#.
.#...
```

ヒント: BFS を使う。

### 12.2 中級演習（緑〜水レベル）

**演習 2-1: 転倒数の計算**

> 配列 A = [5, 3, 1, 4, 2] の転倒数を求めよ。
> 座標圧縮 + BIT を使って O(N log N) で解け。

期待される出力: 7（ペア: (5,3), (5,1), (5,4), (5,2), (3,1), (3,2), (4,2)）

**演習 2-2: 最小値の最大化**

> 数直線上の位置 [1, 5, 12, 23, 37, 50] に 2 つの中継点を追加する。隣接する点間の最大距離を最小化せよ。

ヒント: 「最大距離を d 以下にできるか？」を二分探索する。

**演習 2-3: 区間の種類数**

> 配列 A = [1, 2, 1, 3, 2, 3, 1, 4] から、異なる値がちょうど 3 種類の連続部分列の個数を求めよ。

ヒント: exactly(k) = at_most(k) - at_most(k-1) の式を使う。

### 12.3 上級演習（水〜青レベル）

**演習 3-1: 組合せ数え上げ**

> N 人を K 個のグループに分ける方法の数を 10^9 + 7 で割った余りを求めよ。
> 各グループは 1 人以上。N = 10, K = 3 の場合の答えを計算せよ。

ヒント: 第 2 種スターリング数 S(N, K) を DP で求める。

**演習 3-2: 木の直径**

> N 頂点の木が与えられたとき、最も遠い 2 頂点間の距離（木の直径）を求めよ。

ヒント: 任意の頂点から BFS で最遠頂点を求め、そこからもう一度 BFS する（2 回 BFS）。

**演習 3-3: 区間スケジューリング重み付き版**

> N 個のジョブが与えられ、各ジョブには開始時刻 s_i、終了時刻 e_i、報酬 w_i がある。時間が重ならないようにジョブを選び、報酬の合計を最大化せよ。

ヒント: 終了時刻でソートし、二分探索 + DP を使う。

---

## 13. アンチパターン

### アンチパターン 1: Python の速度を過信する

Python（CPython）は C++ と比較して 50-100 倍遅い。10^7 回程度のループが実質的な上限であり、10^8 回のループは確実に TLE（Time Limit Exceeded）となる。

```python
# BAD: Python で 10^8 回のループ → 約 10 秒、TLE 確実
result = 0
for i in range(10**8):
    result += i

# GOOD: 対策 1 - PyPy で提出（3-10 倍高速）
# 提出言語を「PyPy3」に変更するだけ

# GOOD: 対策 2 - リスト内包表記（for 文の 2-3 倍速い）
# BAD
result = []
for i in range(n):
    result.append(i * i)

# GOOD
result = [i * i for i in range(n)]

# GOOD: 対策 3 - 組み込み関数を活用
# BAD
total = 0
for x in arr:
    total += x

# GOOD
total = sum(arr)

# GOOD: 対策 4 - ローカル変数化（グローバルアクセスは遅い）
def solve():
    # ローカル変数は LOAD_FAST 命令で高速アクセス
    n = len(arr)
    for i in range(n):
        pass
```

### アンチパターン 2: コーナーケースの未考慮

コンテストでの WA の多くは、特殊な入力（コーナーケース）に対するバグが原因である。

```python
# BAD: N=1 のケースを忘れる
def solve_bad():
    N = int(input())
    A = list(map(int, input().split()))
    # N=1 のとき A[1] でインデックスエラー!
    print(A[0] + A[1])

# GOOD: エッジケースを先に処理
def solve_good():
    N = int(input())
    A = list(map(int, input().split()))
    if N == 1:
        print(A[0])
        return
    # 以降 N >= 2 を前提に処理
    print(A[0] + A[1])
```

### アンチパターン 3: 浮動小数点の比較

```python
# BAD: 浮動小数点の等値比較
if 0.1 + 0.2 == 0.3:  # False になる!
    print("equal")

# GOOD: 十分小さい epsilon で比較
EPS = 1e-9
if abs((0.1 + 0.2) - 0.3) < EPS:
    print("equal")

# BETTER: 可能なら整数に変換して計算
# 座標が小数 → 10^6 倍して整数化
# 確率 → 分数のまま計算（MOD 逆元で除算）
```

### アンチパターン 4: 再帰の深さ制限を忘れる

```python
# BAD: Python のデフォルト再帰上限は 1000
def dfs(v, graph, visited):
    visited[v] = True
    for u in graph[v]:
        if not visited[u]:
            dfs(u, graph, visited)
# N = 10^5 の木で RecursionError

# GOOD: 再帰制限を拡張
import sys
sys.setrecursionlimit(10**6)

# BETTER: スタックベースの反復 DFS に書き換え
def dfs_iterative(start, graph):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in graph[v]:
            if u not in visited:
                stack.append(u)
```

---

## 14. FAQ

### Q1: 競技プログラミングを始めるのに最適な言語は？

**A:** C++ が最も推奨される言語である。実行速度が速く、STL（Standard Template Library）が競技プログラミング向けに非常に強力（set, map, priority_queue, lower_bound 等）。ただし、Python は書きやすさで優れており、特に LeetCode では Python が主流。初学者は Python で始めて、速度が必要になったら C++ に移行するのが現実的なパスである。AtCoder では PyPy を使えば多くの問題で Python でも AC できる。

### Q2: 効率的な練習方法は？

**A:** 以下の 4 ステップを推奨する。
1. **毎日 1-2 問**: AtCoder Problems（kenkoooo）で自分のレーティング付近の difficulty の問題を解く
2. **editorial 精読**: 解けなかった問題は 30 分考えてから editorial を読み、理解した上で自力で再実装する
3. **典型問題の周回**: 競プロ典型 90 問や EDPC を繰り返し解いてパターンを身体に染み込ませる
4. **毎週コンテスト参加**: ABC に毎週参加して本番の緊張感と時間制限の中で問題を解く経験を積む

量より質、理解の深さが重要。1 問を完全に理解することは、10 問を表面的に解くよりも価値がある。

### Q3: 実務に競技プログラミングは役立つか？

**A:** 直接的には限定的だが、間接的に大きく役立つ。

**役立つ側面:**
- アルゴリズムの引き出しが増え、効率的なコードを書けるようになる
- 計算量を意識したコーディング習慣が身につく
- エッジケースへの感度が高まり、バグの少ないコードを書ける
- コードの正確性を検証する能力（テスト設計力）が向上する
- 技術面接で大きなアドバンテージになる

**別途必要なスキル:**
- 保守性の高いコード設計（変数命名、関数分割、テスト容易性）
- チーム開発のスキル（コードレビュー、ドキュメント作成）
- システム全体の設計能力（分散システム、DB 設計）

### Q4: AtCoder で緑になるまでの期間は？

**A:** 個人差が大きいが、プログラミング経験者が毎日 1-2 時間練習して 3-6 か月が目安。数学的素養があると上達が速い。重要なのは以下の 3 つを継続すること。
1. ABC の過去問を最低 100 問解く
2. 典型パターン（DP, BFS, 累積和, 二分探索）を確実に身につける
3. 毎週コンテストに参加して実戦経験を積む

### Q5: LeetCode と AtCoder、どちらを優先すべきか？

**A:** 目的による。FAANG 等の外資系企業への転職・就職を目指すなら LeetCode を優先すべき。面接では LeetCode の Medium レベルの問題が出題されることが多い。純粋にアルゴリズム力を高めたい場合や日本企業の技術面接対策なら AtCoder の方が適している。両方を並行して進めるのが理想的だが、時間が限られる場合は目的に応じて選択する。

### Q6: コンテスト中に解けない問題があったらどうするか？

**A:** 以下の判断基準で行動する。
1. **15 分考えても方針が立たない** → 次の問題に進む。後の問題の方が簡単な場合がある
2. **方針は立つが実装が間に合わない** → 残り時間と相談。30 分以上あれば挑戦、なければ他の問題の見直し
3. **WA が出続ける** → サンプルケースの手計算 → エッジケースの確認 → 3 回 WA したら次に進む

コンテスト後は必ず解けなかった問題の editorial を読み、1 週間以内に自力で AC する習慣をつける。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 15. まとめ

| 項目 | 要点 |
|:---|:---|
| AtCoder | 日本語、週 1 コンテスト、色レーティングでレベルが明確 |
| LeetCode | 英語、面接対策の定番、パターン別に練習可能 |
| Codeforces | 英語、問題数が圧倒的、数学的問題が豊富 |
| テンプレート | 高速入力・再帰制限拡張・MOD 演算を定型化して準備 |
| 典型テクニック | bit 全探索・座標圧縮・しゃくとり法・累積和・MOD 演算が頻出 |
| グラフ | BFS / DFS / ダイクストラ / Union-Find が中級以上で必須 |
| コンテスト戦略 | 時間配分・デバッグ手法・メンタル管理が成績を左右する |
| 練習方法 | 毎日 1-2 問、editorial 精読、コンテスト参加を継続 |
| 成長の鍵 | 量より質、パターン認識力、継続的な振り返り |

---

## 次に読むべきガイド

- [問題解決法](./00-problem-solving.md) -- アルゴリズム問題への体系的アプローチ
- [動的計画法](../02-algorithms/04-dynamic-programming.md) -- 最頻出のアルゴリズムパラダイム
- [セグメント木](../03-advanced/01-segment-tree.md) -- 中級以上で必須のデータ構造

---

## 16. 参考文献

1. 秋葉拓哉, 岩田陽一, 北川宜稔 (2012). 『プログラミングコンテストチャレンジブック 第 2 版』. マイナビ出版. -- 通称「蟻本」。競技プログラミングの入門書として最も定評がある。
2. 大槻兼資 (2020). 『問題解決力を鍛える! アルゴリズムとデータ構造』. 講談社. -- 日本語で書かれた現代的なアルゴリズム教科書。豊富な図解と丁寧な解説が特徴。
3. E869120. "競プロ典型 90 問." https://github.com/E869120/kyopro-tenkei-90 -- 典型テクニック 90 個を体系的に学べる問題集。初級から上級まで幅広くカバー。
4. kenkoooo. "AtCoder Problems." https://kenkoooo.com/atcoder/ -- AtCoder の過去問を difficulty 付きで一覧表示。進捗管理にも使える必須ツール。
5. NeetCode. "NeetCode 150." https://neetcode.io/ -- LeetCode の頻出 150 問をパターン別に整理。面接対策の定番リソース。
6. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- 世界的に有名な競技プログラミングの教科書。網羅的なアルゴリズム解説。
7. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms, 4th Edition*. MIT Press. -- 通称 CLRS。アルゴリズムの理論的基盤を学ぶ上で最も権威ある教科書。
8. AtCoder Library (ACL). https://github.com/atcoder/ac-library -- AtCoder 公式のアルゴリズムライブラリ。セグメント木、遅延評価、フロー等の高品質な実装を提供。

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
