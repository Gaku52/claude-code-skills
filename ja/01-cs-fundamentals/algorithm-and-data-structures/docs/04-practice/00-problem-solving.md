# 問題解決法

> アルゴリズム問題を体系的に解くための思考法を、パターン認識・制約分析・段階的改善の3段階で習得する

## この章で学ぶこと

1. **パターン認識** で問題を既知のアルゴリズムカテゴリに分類できる
2. **制約分析** から許容される計算量を逆算し、適切なアルゴリズムを選択できる
3. **段階的改善** で愚直解から最適解へと効率的に到達する手順を身につける
4. **ストレステスト** と愚直解比較による堅牢なデバッグ手法を実践できる
5. **典型パターン** を12種以上マスターし、未知の問題にも対応できる引き出しを持つ


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. 問題解決の全体フレームワーク

問題解決能力は単なるアルゴリズム知識ではない。問題を読み、構造を把握し、既知の手法にマッピングし、段階的に解を構築する一連の思考プロセスである。George Polya が『いかにして問題をとくか（How to Solve It）』で提唱した4段階フレームワークをアルゴリズム問題向けに拡張すると、以下の5ステップになる。

```
┌──────────────────────────────────────────────────────────────────────┐
│                    問題解決の5ステップ                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Step 1: 問題を理解する (Understand)                                 │
│    ├─ 入力/出力の形式を明確化する                                    │
│    ├─ 制約条件（n の範囲、値の範囲、時間制限）を確認する             │
│    ├─ エッジケースを特定する（空入力、最小値、最大値）               │
│    └─ 問題文を自分の言葉で言い換える                                 │
│                                                                      │
│  Step 2: 具体例で手を動かす (Explore)                                │
│    ├─ 小さな入力（n=3〜5）で手計算する                               │
│    ├─ 複数の具体例でパターンを発見する                               │
│    ├─ 図やテーブルを描いて視覚化する                                 │
│    └─ 反例がないか確認する                                           │
│                                                                      │
│  Step 3: 制約を分析する (Analyze)                                    │
│    ├─ n の範囲から許容 O(?) を逆算する                               │
│    ├─ 空間計算量の制約も確認する（ML = メモリ制限）                   │
│    ├─ 使用言語の定数倍を考慮する                                     │
│    └─ 特殊な制約（座標圧縮が必要、MOD演算等）を見抜く               │
│                                                                      │
│  Step 4: アルゴリズムを選択・設計する (Design)                       │
│    ├─ パターン認識で既知手法にマッピングする                         │
│    ├─ 複数の候補を計算量で比較する                                   │
│    ├─ 必要なデータ構造を選定する                                     │
│    └─ 擬似コードで設計を検証する                                     │
│                                                                      │
│  Step 5: 実装・検証・改善する (Implement & Verify)                   │
│    ├─ 愚直解をまず実装し、正しい出力を確認する                       │
│    ├─ 最適解を実装する                                               │
│    ├─ ストレステストで愚直解と比較する                               │
│    └─ エッジケースを網羅的にテストする                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 Step 1 の実践: 問題を理解する

問題を正確に理解できていないまま実装に進むのは、最も多いミスの1つである。問題文を読む際は以下の点を必ず確認する。

```python
# 問題理解のチェックリスト（実装前に必ず確認）

class ProblemUnderstanding:
    """問題を理解するためのフレームワーク"""

    def __init__(self, problem_text: str):
        self.problem_text = problem_text
        self.input_format = None    # 入力形式
        self.output_format = None   # 出力形式
        self.constraints = {}       # 制約条件
        self.edge_cases = []        # エッジケース
        self.examples = []          # 具体例

    def parse_constraints(self):
        """制約条件を構造化して抽出する"""
        # 典型的な制約パターン
        constraint_patterns = {
            'n_range': '1 <= n <= ?',
            'value_range': '値の範囲',
            'time_limit': '実行時間制限（通常2秒）',
            'memory_limit': 'メモリ制限（通常256MB）',
            'special': 'MOD, 座標範囲, 文字種等',
        }
        return constraint_patterns

    def identify_edge_cases(self):
        """エッジケースを体系的に洗い出す"""
        common_edges = [
            "n = 0 (空入力)",
            "n = 1 (最小入力)",
            "n = 最大制約 (TLE/MLEの境界)",
            "全要素が同一値",
            "既にソート済み / 完全逆順",
            "負の値・ゼロを含む",
            "答えが存在しない場合",
            "答えが複数存在する場合",
        ]
        return common_edges

    def rephrase(self) -> str:
        """問題を自分の言葉で言い換える"""
        # 「〜を求めよ」→「〜を最小化/最大化する」→「〜の条件を満たすものを数える」
        # この言い換えが正しいか、サンプルで検証する
        pass
```

### 1.2 Step 2 の実践: 具体例で手を動かす

アルゴリズムを考える前に、まず3つ以上の具体例を手計算する。これにより問題の本質的な構造が見えてくる。

```
具体例の作り方ガイド:

   例1: サンプル入力をそのまま追う
         → 問題の意味を正確に理解する

   例2: 自分で小さな入力を作る (n=3〜5)
         → 手計算でパターンを発見する

   例3: エッジケースを試す
         → 空入力、最小入力、特殊値

   例4: 大きめの入力を試す (n=10〜20)
         → アルゴリズムの一般化を検証する

   ┌─────────────────────────────────────────────────┐
   │  手計算のコツ                                     │
   │                                                   │
   │  ・表を作って状態遷移を追う（DPに繋がる）        │
   │  ・図を描いて空間関係を把握する（幾何・グラフ）  │
   │  ・途中経過をメモする（貪欲法の正当性確認）      │
   │  ・「なぜこの答えが最適か」を言語化する          │
   └─────────────────────────────────────────────────┘
```

### 1.3 Step 3〜5 の連携

Steps 3〜5 は直線的ではなく、フィードバックループを形成する。

```
    ┌──────────────┐
    │ Step 3: 分析 │◄─────────────────────┐
    └──────┬───────┘                       │
           │ 計算量の見積もり               │ 計算量が合わない
           ▼                               │ → 別のアルゴリズムへ
    ┌──────────────┐                       │
    │ Step 4: 設計 │───────────────────────┘
    └──────┬───────┘
           │ 擬似コード
           ▼
    ┌──────────────┐
    │ Step 5: 実装 │───┐
    └──────┬───────┘   │ WA（不正解）
           │            │ → 愚直解と比較
           ▼            │
    ┌──────────────┐   │
    │   検証完了    │◄──┘
    └──────────────┘
```

---

## 2. 制約分析: n から計算量を逆算する

### 2.1 基本原則

制約分析はアルゴリズム選択の出発点である。多くの問題では、制約条件を見た瞬間に使うべきアルゴリズムの計算量クラスが決まる。これは「制約が答えを教えてくれる」と表現される重要な原則だ。

```
1秒で処理可能な演算数 ≈ 10^8 〜 10^9 （C++基準）
Python の場合: 約 10^6 〜 10^7 （C++の30〜100倍遅い）

制約 n の範囲 → 許容される計算量:

  n ≤ 8       → O(n! * n)    全順列+各順列に対する処理
  n ≤ 10      → O(n!)        全順列探索、バックトラッキング
  n ≤ 20      → O(2^n * n)   ビットDP、部分集合列挙
  n ≤ 25      → O(2^(n/2))   半分全列挙（Meet in the Middle）
  n ≤ 50      → O(n^4)       4重ループ（稀）
  n ≤ 300     → O(n^3)       Floyd-Warshall、区間DP
  n ≤ 3,000   → O(n^2 log n) 一部の工夫付き2重ループ
  n ≤ 5,000   → O(n^2)       DP（2次元）、全対比較
  n ≤ 100,000 → O(n√n)       平方分割、Mo's algorithm
  n ≤ 200,000 → O(n log n)   ソート、セグメント木、二分探索
  n ≤ 10^6    → O(n)         線形走査、尺取り法、BFS
  n ≤ 10^7    → O(n)         線形（定数倍に注意）
  n ≤ 10^9    → O(√n) or O(log n)  素因数分解、二分探索
  n ≤ 10^18   → O(log n) or O(1)   行列累乗、数学的公式
```

### 2.2 制約分析の実践例

```python
# ============================================================
# 例題: 配列から和が target となるペアを見つける
# 制約: 1 <= n <= 10^5, -10^9 <= arr[i] <= 10^9
# ============================================================

# 制約分析:
# n = 10^5 → O(n^2) = 10^10 → TLE（タイムオーバー）
# → O(n log n) か O(n) が必要
# → 候補: ソート+二分探索、ソート+尺取り、ハッシュマップ

# === 解法1: O(n^2) 全ペア列挙（TLEだが正解確認用） ===
def two_sum_brute(arr: list[int], target: int) -> tuple[int, int] | None:
    """愚直解: 全ペアを列挙する O(n^2)"""
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == target:
                return (i, j)
    return None

# === 解法2: O(n log n) ソート + 二分探索 ===
def two_sum_sort(arr: list[int], target: int) -> tuple[int, int] | None:
    """ソート+尺取り法 O(n log n)"""
    indexed = sorted(enumerate(arr), key=lambda x: x[1])
    left, right = 0, len(arr) - 1

    while left < right:
        current_sum = indexed[left][1] + indexed[right][1]
        if current_sum == target:
            return (indexed[left][0], indexed[right][0])
        elif current_sum < target:
            left += 1
        else:
            right -= 1
    return None

# === 解法3: O(n) ハッシュマップ（最適解） ===
def two_sum_hash(arr: list[int], target: int) -> tuple[int, int] | None:
    """ハッシュマップ O(n)"""
    seen: dict[int, int] = {}
    for i, num in enumerate(arr):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return None

# === ストレステストで3解法を比較 ===
import random

def stress_test_two_sum(n_tests: int = 1000, max_n: int = 100):
    """ランダムテストで3つの解法を比較検証"""
    for test_id in range(n_tests):
        n = random.randint(2, max_n)
        arr = [random.randint(-100, 100) for _ in range(n)]
        target = random.randint(-200, 200)

        result_brute = two_sum_brute(arr, target)
        result_sort = two_sum_sort(arr, target)
        result_hash = two_sum_hash(arr, target)

        # 存在性の一致を確認
        has_brute = result_brute is not None
        has_sort = result_sort is not None
        has_hash = result_hash is not None

        assert has_brute == has_sort == has_hash, (
            f"Test {test_id}: 不一致! arr={arr}, target={target}"
        )

        # 解が存在する場合、正しいペアか確認
        if has_brute:
            i, j = result_hash
            assert arr[i] + arr[j] == target, (
                f"Test {test_id}: ハッシュ解が不正"
            )
    print(f"全{n_tests}テスト通過")

# stress_test_two_sum()  # 実行して検証
```

### 2.3 空間計算量の制約

時間計算量だけでなく、空間計算量の考慮も重要である。

```
メモリ制限の目安（256MB の場合）:

  int型配列:     約 6.4 * 10^7 要素
  long long配列: 約 3.2 * 10^7 要素
  2次元配列:     n * m <= 約 6.4 * 10^7

  Python の場合（メモリ効率が悪い）:
    list[int]:   1要素あたり約28バイト → 約 9 * 10^6 要素
    numpy array: 1要素あたり8バイト → 約 3.2 * 10^7 要素

  頻出パターン:
    n = 10^5 の2次元DP → O(n^2) = 10^10 → MLE
    → 「行のみ保持」で O(n) に削減する（DP のスクロール最適化）
```

### 2.4 言語別の定数倍と対策

```
┌─────────────┬──────────────┬────────────────────────────────────┐
│ 言語        │ 速度倍率     │ 対策                               │
│             │（C++比）     │                                    │
├─────────────┼──────────────┼────────────────────────────────────┤
│ C++         │ 1x           │ 基準                               │
│ Java        │ 2-3x         │ FastReader使用、Scanner避ける       │
│ Python      │ 30-100x      │ PyPy使用、内包表記、sys.stdin      │
│ Go          │ 1-2x         │ bufio.Scanner使用                  │
│ Rust        │ 0.8-1.2x     │ C++同等                            │
│ JavaScript  │ 3-10x        │ TypedArray活用                     │
└─────────────┴──────────────┴────────────────────────────────────┘

Python 高速化テクニック集:

  1. 入出力の高速化
     import sys
     input = sys.stdin.readline  # 10倍以上高速
     print = sys.stdout.write    # 大量出力時に有効

  2. リスト内包表記 > for ループ
     # 遅い: for i in range(n): result.append(i*i)
     # 速い: result = [i*i for i in range(n)]

  3. 再帰の上限引き上げ
     sys.setrecursionlimit(300000)  # デフォルト1000

  4. collections の活用
     from collections import deque, defaultdict, Counter

  5. 局所変数は大域変数より高速
     def solve():
         # ここに全処理を書く（関数内のローカル変数参照は高速）
         pass
     solve()
```

---

## 3. パターン認識マップ

### 3.1 キーワード・アルゴリズム対応表

問題文に含まれるキーワードから、使うべきアルゴリズムを推定する。これは経験に基づくヒューリスティクスであり、絶対的なルールではないが、正解率は高い。

```
問題のキーワード → アルゴリズム候補:

┌──────────────────────────────────────────────────────────────────────┐
│ キーワード              │ 候補アルゴリズム                           │
├──────────────────────────────────────────────────────────────────────┤
│ "最短距離" "最小手数"   │ BFS（重みなし）, Dijkstra（重みあり）     │
│ "最短経路" "コスト最小" │ Dijkstra, Bellman-Ford, Floyd-Warshall    │
│ "最大/最小を求めよ"     │ DP, 貪欲法, 二分探索（答え決め打ち）     │
│ "〜の数を求めよ"        │ DP, 組合せ数学, 包除原理                  │
│ "全ての組み合わせ"      │ バックトラッキング, ビットDP              │
│ "部分列" "部分文字列"   │ DP (LIS, LCS), 尺取り法                  │
│ "連結か" "到達可能か"   │ Union-Find, BFS/DFS                       │
│ "区間" "範囲"           │ セグメント木, BIT, 尺取り法              │
│ "文字列の一致"          │ KMP, Z-algorithm, Rabin-Karp             │
│ "辞書順最小"            │ 貪欲法, スタック                          │
│ "二部グラフ"            │ 二部マッチング, 2色彩色                  │
│ "割り当て" "フロー"     │ ネットワークフロー, 二部マッチング        │
│ "MOD 10^9+7"            │ DP + 繰り返し二乗法 + modular inverse    │
│ "木" "根付き木"         │ DFS, オイラーツアー, LCA                  │
│ "閉路" "サイクル"       │ DFS（帰りがけ順）, Union-Find            │
│ "座標" "点" "距離"      │ 幾何, ソート, 掃引線                      │
│ "ゲーム" "先手後手"     │ Grundy数, DP, ゲーム理論                 │
│ "期待値" "確率"         │ 確率DP, 行列累乗                          │
│ "回文"                  │ Manacher, DP, ハッシュ                    │
│ "転倒数" "交差数"       │ BIT, マージソート                         │
│ "XOR"                   │ Trie（ビット列）, 線形基底               │
│ "GCD" "LCM" "素数"     │ ユークリッド互除法, エラトステネスの篩    │
│ "括弧" "ネスト"         │ スタック                                  │
│ "k番目" "中央値"        │ 二分探索, 順序統計量                      │
│ "部分和" "ナップサック"  │ DP, 半分全列挙                           │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 問題構造からの分類

キーワードだけでなく、問題の構造的特徴からもアルゴリズムを推定できる。

```python
def classify_problem(problem_features: dict) -> list[str]:
    """
    問題の特徴からアルゴリズム候補を推定する

    problem_features の例:
    {
        'optimization': True,          # 最適化問題か
        'counting': False,             # 数え上げ問題か
        'decision': False,             # 判定問題か
        'graph': True,                 # グラフが関与するか
        'has_weights': True,           # 重み/コストがあるか
        'constraint_n': 100000,        # n の制約
        'overlapping_subproblems': True,  # 重複部分問題
        'greedy_choice': False,        # 貪欲選択性質
        'monotonic': True,             # 単調性があるか
    }
    """
    candidates = []
    n = problem_features.get('constraint_n', 0)

    # 最適化問題の分岐
    if problem_features.get('optimization'):
        if problem_features.get('greedy_choice'):
            candidates.append('貪欲法')
        if problem_features.get('overlapping_subproblems'):
            candidates.append('動的計画法')
        if problem_features.get('monotonic'):
            candidates.append('二分探索（答えの決め打ち）')
        if n <= 20:
            candidates.append('ビットDP / 全探索')

    # グラフ問題の分岐
    if problem_features.get('graph'):
        if problem_features.get('has_weights'):
            candidates.append('Dijkstra / Bellman-Ford')
        else:
            candidates.append('BFS / DFS')

    # 数え上げ問題
    if problem_features.get('counting'):
        candidates.append('DP')
        candidates.append('組合せ数学')

    # 判定問題
    if problem_features.get('decision'):
        if problem_features.get('monotonic'):
            candidates.append('二分探索')

    return candidates
```

### 3.3 データ構造の選択指針

```
問題の要求 → 適切なデータ構造:

┌──────────────────────────────────────────────────────────────────────┐
│ 要求される操作               │ データ構造                           │
├──────────────────────────────────────────────────────────────────────┤
│ 先頭/末尾への追加・削除      │ deque (両端キュー)                   │
│ 最小値/最大値の取り出し      │ heapq (優先度付きキュー)             │
│ 要素の挿入・検索・削除 O(1)  │ dict / set (ハッシュ)                │
│ 要素の挿入・検索・削除 O(lgn)│ SortedList (平衡二分探索木)          │
│ 区間の和・最大値の更新       │ セグメント木 / BIT                   │
│ 集合の併合・同一判定         │ Union-Find (素集合)                  │
│ 文字列の接頭辞検索           │ Trie (トライ木)                      │
│ LIFO (最後に入れたものを先に)│ stack (list)                         │
│ FIFO (先に入れたものを先に)  │ deque                                │
│ ソート済みで k 番目          │ BIT / 平衡二分探索木                 │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. 典型パターン1: 尺取り法（Two Pointers / Sliding Window）

### 4.1 尺取り法の基本概念

尺取り法は、配列上の連続部分列に関する問題を効率的に解くテクニックである。2つのポインタ（left と right）を使い、条件を満たす区間を管理しながらスライドさせる。

```
固定長ウィンドウ:
  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ 1 │ 4 │ 2 │10 │23 │ 3 │ 1 │ 0 │  配列
  └───┴───┴───┴───┴───┴───┴───┴───┘
  ├─── k=4 ────┤                      ウィンドウ和 = 17
      ├─── k=4 ────┤                  ウィンドウ和 = 39 ← 最大
          ├─── k=4 ────┤              ウィンドウ和 = 38
              ├─── k=4 ────┤          ウィンドウ和 = 37
                  ├─── k=4 ────┤      ウィンドウ和 = 27

  新しい要素を追加し、古い要素を除去 → O(1) で更新

可変長ウィンドウ:
  ┌───┬───┬───┬───┬───┬───┐
  │ 2 │ 3 │ 1 │ 2 │ 4 │ 3 │  配列, target=7
  └───┴───┴───┴───┴───┴───┘
   L       R                    sum=6 < 7 → R を進める
   L           R                sum=8 >= 7 → 長さ4記録, L を進める
       L       R                sum=6 < 7 → R を進める
       L           R            sum=10 >= 7 → 長さ3記録, L を進める
           L       R            sum=7 >= 7 → 長さ2記録 ← 最短
```

### 4.2 実装パターン

```python
# ============================================================
# パターン1: 固定長ウィンドウ
# ============================================================
def max_subarray_sum_k(arr: list[int], k: int) -> int:
    """長さ k の部分配列の最大和"""
    if len(arr) < k:
        return 0

    # 最初のウィンドウ
    window_sum = sum(arr[:k])
    max_sum = window_sum

    # ウィンドウをスライド
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # O(1) で更新
        max_sum = max(max_sum, window_sum)

    return max_sum


# ============================================================
# パターン2: 可変長ウィンドウ（条件を満たす最短区間）
# ============================================================
def min_subarray_len(arr: list[int], target: int) -> int:
    """和が target 以上となる最短の連続部分配列の長さ"""
    n = len(arr)
    min_len = float('inf')
    left = 0
    current_sum = 0

    for right in range(n):
        current_sum += arr[right]

        # 条件を満たす限り left を進める（区間を縮める）
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1

    return min_len if min_len != float('inf') else 0


# ============================================================
# パターン3: 可変長ウィンドウ（条件を満たす最長区間）
# ============================================================
def longest_substring_k_distinct(s: str, k: int) -> int:
    """異なる文字が高々 k 種の最長部分文字列"""
    from collections import defaultdict

    char_count: dict[str, int] = defaultdict(int)
    left = 0
    max_len = 0

    for right in range(len(s)):
        char_count[s[right]] += 1

        # 条件を違反したら left を進める
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len


# ============================================================
# パターン4: 2つの配列に対する尺取り法（マージ操作）
# ============================================================
def merge_sorted_arrays(arr1: list[int], arr2: list[int]) -> list[int]:
    """ソート済み2配列のマージ O(n + m)"""
    result = []
    i, j = 0, 0

    while i < len(arr1) and j < len(arr2):
        if arr1[i] <= arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1

    # 残りを追加
    result.extend(arr1[i:])
    result.extend(arr2[j:])
    return result


# ============================================================
# テスト
# ============================================================
print(max_subarray_sum_k([1, 4, 2, 10, 23, 3, 1, 0, 20], 4))  # 39
print(min_subarray_len([2, 3, 1, 2, 4, 3], 7))                  # 2
print(longest_substring_k_distinct("eceba", 2))                  # 3 ("ece")
print(merge_sorted_arrays([1, 3, 5], [2, 4, 6]))                # [1,2,3,4,5,6]
```

### 4.3 尺取り法の適用条件

尺取り法が適用できるための条件を理解しておくことが重要である。

```
尺取り法が使える条件:

  1. 区間の拡大（right を進める）で「コスト」が単調に増加する
  2. 区間の縮小（left を進める）で「コスト」が単調に減少する
  3. つまり、区間 [l, r] に対する判定関数が単調性を持つ

  例: 区間和 >= target
    → 区間を広げると和が増える（単調増加）
    → 区間を狭めると和が減る（単調減少）
    → 尺取り法が使える

  反例: 区間内の最大値 - 最小値 <= k
    → 区間を広げても差が増えるとは限らない
    → 単純な尺取り法は使えない
    → ただし、sortedcontainers 等で拡張すれば可能

適用可能な典型問題:
  ・和が target 以上/以下の最短/最長区間
  ・異なる要素が k 個以下の最長区間
  ・条件を満たす区間の数え上げ
  ・2つのソート済み配列のマージ
```

### 4.4 尺取り法の計算量分析

```
なぜ O(n) なのか？

  left と right はそれぞれ最大 n 回しか進まない。
  各要素は left に1回、right に1回しかアクセスされない。
  → 合計の操作回数は最大 2n → O(n)

  ┌──────────────────────────────────────────┐
  │  right: 0 → 1 → 2 → ... → n-1          │
  │  left:  0 → 0 → 1 → ... → n-1          │
  │                                          │
  │  right の移動回数: n                     │
  │  left の移動回数: 最大 n                 │
  │  合計: 最大 2n = O(n)                    │
  └──────────────────────────────────────────┘
```

---

## 5. 典型パターン2: 二分探索で答えを決め打ち

### 5.1 「答えで二分探索」の考え方

通常の二分探索は「ソート済み配列から値を探す」が、ここでの二分探索は「答えの値を仮定して実現可能かを判定する」というパラダイムである。最適化問題を判定問題に変換する強力なテクニックだ。

```
通常の二分探索:
  「配列の中に値 x は存在するか？」 → O(log n)

答えの二分探索:
  「答えが x 以下で実現可能か？」 → O(log(答えの範囲) * 判定の計算量)

  答えの二分探索が使える条件:
    1. 答えに単調性がある（答えを大きくすると実現しやすくなる等）
    2. 「答えが x のとき実現可能か」を効率的に判定できる

  ┌──────────────────────────────────────────────────────┐
  │  答えの空間: [lo, hi]                                │
  │                                                      │
  │  lo ──────── mid ──────── hi                         │
  │  ← 不可能 →│← 可能 →→→→→→→→→│                      │
  │             ▲                                        │
  │         最小の答え                                    │
  │                                                      │
  │  can_achieve(mid) == True  → hi = mid (左に詰める)   │
  │  can_achieve(mid) == False → lo = mid + 1 (右に詰める)│
  └──────────────────────────────────────────────────────┘
```

### 5.2 実装パターン

```python
# ============================================================
# 例題1: 配列を k 分割したとき、各区間の和の最大値を最小化する
# (Painter's Partition Problem / Split Array Largest Sum)
# ============================================================

def min_max_partition(arr: list[int], k: int) -> int:
    """配列を k 分割したとき、各区間の和の最大値を最小化する"""

    def can_partition(max_sum: int) -> bool:
        """各区間の和が max_sum 以下で k 分割可能か？"""
        count = 1       # 現在の分割数
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
    lo = max(arr)      # 最小: 最大要素1つだけの区間
    hi = sum(arr)      # 最大: 全体が1区間
    result = hi

    while lo <= hi:
        mid = (lo + hi) // 2
        if can_partition(mid):
            result = mid
            hi = mid - 1   # より小さい答えを探す
        else:
            lo = mid + 1   # 答えを大きくする

    return result


# ============================================================
# 例題2: n 本の木から長さ L の丸太を k 本切り出す。L の最大値は？
# ============================================================

def max_log_length(trees: list[int], k: int) -> int:
    """n 本の木から長さ L の丸太を k 本以上切り出せる最大の L"""

    def count_logs(length: int) -> int:
        """長さ length で切り出せる丸太の本数"""
        return sum(t // length for t in trees)

    lo, hi = 1, max(trees)
    result = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        if count_logs(mid) >= k:
            result = mid
            lo = mid + 1    # より大きな L を探す
        else:
            hi = mid - 1

    return result


# ============================================================
# 例題3: 実数値の二分探索（精度指定）
# ============================================================

def sqrt_binary_search(x: float, eps: float = 1e-9) -> float:
    """x の平方根を二分探索で求める"""
    lo, hi = 0.0, max(1.0, x)

    while hi - lo > eps:
        mid = (lo + hi) / 2
        if mid * mid <= x:
            lo = mid
        else:
            hi = mid

    return lo


# テスト
print(min_max_partition([7, 2, 5, 10, 8], 2))   # 18 ([7,2,5] と [10,8])
print(max_log_length([10, 24, 15], 7))           # 6
print(f"{sqrt_binary_search(2):.10f}")           # 1.4142135624
```

### 5.3 二分探索の典型バグと対策

```
┌──────────────────────────────────────────────────────────────────────┐
│  二分探索で陥りやすいバグ TOP 5                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 無限ループ                                                       │
│     原因: lo と hi の更新が不適切で収束しない                        │
│     対策: lo = mid + 1 / hi = mid - 1 で必ず範囲を狭める            │
│                                                                      │
│  2. Off-by-one エラー                                                │
│     原因: lo <= hi vs lo < hi の混同                                │
│     対策: 条件とreturnを一貫させる。テンプレートを使う               │
│                                                                      │
│  3. 整数オーバーフロー                                               │
│     原因: mid = (lo + hi) / 2 で lo + hi がオーバーフロー           │
│     対策: mid = lo + (hi - lo) // 2                                  │
│                                                                      │
│  4. 単調性の方向を間違える                                           │
│     原因: 最小化なのに lo = mid + 1 にする（逆）                    │
│     対策: 判定関数の True/False と探索方向を表に書いて確認           │
│                                                                      │
│  5. 実数二分探索の精度不足                                           │
│     原因: eps が大きすぎる / 反復回数が少ない                       │
│     対策: 反復回数固定（100回等）にする方が安全                      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

安全な二分探索テンプレート（整数版）:

  # 最小値を求める場合（条件を満たす最小の答え）
  lo, hi = 下限, 上限
  while lo < hi:
      mid = (lo + hi) // 2
      if is_feasible(mid):
          hi = mid        # mid は候補、左をさらに探す
      else:
          lo = mid + 1    # mid は不可、右へ
  answer = lo             # lo == hi が答え

  # 最大値を求める場合（条件を満たす最大の答え）
  lo, hi = 下限, 上限
  while lo < hi:
      mid = (lo + hi + 1) // 2  # 切り上げ（重要！）
      if is_feasible(mid):
          lo = mid        # mid は候補、右をさらに探す
      else:
          hi = mid - 1    # mid は不可、左へ
  answer = lo             # lo == hi が答え
```

---

## 6. 典型パターン3: 累積和

### 6.1 累積和の基本

累積和は、区間の和を O(1) で求めるための前処理テクニックである。前処理に O(n)、各クエリに O(1) で回答できる。

```
元の配列:     [3, 1, 4, 1, 5, 9, 2, 6]
累積和配列:   [0, 3, 4, 8, 9, 14, 23, 25, 31]
               ↑  prefix[0] = 0（番兵）

区間 [2, 5] の和 = prefix[6] - prefix[2]
                  = 23 - 4 = 19

  検証: arr[2]+arr[3]+arr[4]+arr[5] = 4+1+5+9 = 19  ✓

  ┌───┬───┬───┬───┬───┬───┬───┬───┐
  │ 3 │ 1 │ 4 │ 1 │ 5 │ 9 │ 2 │ 6 │  元の配列
  └───┴───┴───┴───┴───┴───┴───┴───┘
  idx: 0   1   2   3   4   5   6   7

  prefix[i+1] = prefix[i] + arr[i]
  sum(arr[l..r]) = prefix[r+1] - prefix[l]
```

### 6.2 1次元・2次元の実装

```python
# ============================================================
# 1次元累積和
# ============================================================
def prefix_sum_queries(arr: list[int], queries: list[tuple[int, int]]) -> list[int]:
    """複数の区間和クエリを O(1) で回答"""
    n = len(arr)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + arr[i]

    results = []
    for l, r in queries:
        results.append(prefix[r + 1] - prefix[l])

    return results


# ============================================================
# 2次元累積和
# ============================================================
def build_prefix_sum_2d(matrix: list[list[int]]) -> list[list[int]]:
    """2次元累積和の構築 O(rows * cols)"""
    if not matrix or not matrix[0]:

    rows, cols = len(matrix), len(matrix[0])
    prefix = [[0] * (cols + 1) for _ in range(rows + 1)]

    for i in range(rows):
        for j in range(cols):
            prefix[i + 1][j + 1] = (
                matrix[i][j]
                + prefix[i][j + 1]
                + prefix[i + 1][j]
                - prefix[i][j]         # 包除原理
            )

    return prefix


def query_2d(prefix: list[list[int]], r1: int, c1: int, r2: int, c2: int) -> int:
    """(r1,c1)-(r2,c2) の矩形和 O(1)"""
    return (
        prefix[r2 + 1][c2 + 1]
        - prefix[r1][c2 + 1]
        - prefix[r2 + 1][c1]
        + prefix[r1][c1]              # 包除原理
    )


# ============================================================
# 累積和の応用: いもす法（差分配列）
# ============================================================
def range_add_queries(n: int, queries: list[tuple[int, int, int]]) -> list[int]:
    """
    いもす法: 区間加算クエリを効率的に処理する
    queries: [(l, r, val), ...] → arr[l..r] に val を加算
    全クエリ処理後の配列を返す

    計算量: O(n + Q)  （Q = クエリ数）
    """
    diff = [0] * (n + 1)   # 差分配列

    for l, r, val in queries:
        diff[l] += val
        if r + 1 < n:
            diff[r + 1] -= val

    # 差分配列の累積和 = 最終的な配列
    result = [0] * n
    result[0] = diff[0]
    for i in range(1, n):
        result[i] = result[i - 1] + diff[i]

    return result


# テスト
arr = [3, 1, 4, 1, 5, 9, 2, 6]
print(prefix_sum_queries(arr, [(0, 3), (2, 5), (0, 7)]))
# [9, 19, 31]

matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
prefix = build_prefix_sum_2d(matrix)
print(query_2d(prefix, 0, 0, 1, 1))  # 1+2+4+5 = 12
print(query_2d(prefix, 1, 1, 2, 2))  # 5+6+8+9 = 28

print(range_add_queries(5, [(1, 3, 2), (0, 2, 1), (2, 4, 3)]))
# [1, 3, 6, 5, 3]
```

### 6.3 累積和の変種

```
┌────────────────────────┬───────────────────────────────────────────┐
│ 変種                   │ 用途                                     │
├────────────────────────┼───────────────────────────────────────────┤
│ 通常の累積和           │ 区間和クエリ O(1)                        │
│ 2次元累積和            │ 矩形和クエリ O(1)                        │
│ いもす法（差分配列）   │ 区間加算 O(1) + 最終状態復元 O(n)       │
│ 累積XOR               │ 区間XOR O(1)                             │
│ 累積GCD               │ 区間GCD（右から左への累積も必要）        │
│ 累積最大/最小          │ 接頭辞最大値クエリ O(1)                  │
│                        │ ※任意区間には Sparse Table が必要       │
└────────────────────────┴───────────────────────────────────────────┘
```

---

## 7. 典型パターン4: 動的計画法の問題への適用

### 7.1 DP適用の判断基準

動的計画法（DP）はアルゴリズム問題で最も出題頻度の高いパラダイムである。DP が適用できるかの判断基準は以下の2つだ。

```
DP 適用の2条件:

  1. 最適部分構造 (Optimal Substructure)
     → 全体の最適解が部分問題の最適解から構成できる
     → 例: 最短経路の部分経路も最短経路

  2. 重複部分問題 (Overlapping Subproblems)
     → 同じ部分問題が何度も現れる
     → 例: フィボナッチ数列で fib(5) の計算に fib(3) が複数回必要

  DP の設計手順:
    Step 1: 状態を定義する → dp[i] は何を表すか？
    Step 2: 遷移式を立てる → dp[i] = f(dp[j], ...) の形
    Step 3: 初期条件を設定する → dp[0] = ?
    Step 4: 計算順序を決める → 小さい状態から大きい状態へ
    Step 5: 答えを特定する → dp[n] ? max(dp) ? dp[n][m] ?
```

### 7.2 DP の典型パターン集

```python
# ============================================================
# パターン1: 1次元DP（最長増加部分列 LIS）
# ============================================================
import bisect

def lis_length(arr: list[int]) -> int:
    """最長増加部分列の長さ O(n log n)"""
    # tails[i] = 長さ (i+1) の増加部分列の末尾要素の最小値
    tails: list[int] = []

    for num in arr:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)   # 新しい長さの部分列
        else:
            tails[pos] = num    # より小さい末尾に更新

    return len(tails)


# ============================================================
# パターン2: 2次元DP（ナップサック問題）
# ============================================================
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    """0-1 ナップサック問題 O(n * capacity)"""
    n = len(weights)
    # dp[j] = 容量 j で得られる最大価値（1次元に圧縮）
    dp = [0] * (capacity + 1)

    for i in range(n):
        # 逆順にループ（各品物を1回だけ使うため）
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[capacity]


# ============================================================
# パターン3: 区間DP（行列連鎖乗算）
# ============================================================
def matrix_chain_order(dims: list[int]) -> int:
    """
    行列連鎖乗算の最小コスト O(n^3)
    dims[i-1] x dims[i] が i 番目の行列のサイズ
    """
    n = len(dims) - 1  # 行列の数
    # dp[i][j] = 行列 i..j を掛け合わせる最小コスト
    dp = [[0] * n for _ in range(n)]

    # 区間の長さを 2 から n まで増やす
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]


# ============================================================
# パターン4: ビットDP（巡回セールスマン問題 TSP）
# ============================================================
def tsp(dist: list[list[int]]) -> int:
    """
    巡回セールスマン問題 O(2^n * n^2)
    dist[i][j] = 都市 i から j への距離
    """
    n = len(dist)
    INF = float('inf')
    # dp[S][i] = 集合 S の都市を訪問済みで、現在都市 i にいる最小コスト
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0  # 都市0からスタート

    for S in range(1 << n):
        for u in range(n):
            if dp[S][u] == INF:
                continue
            if not (S >> u & 1):
                continue
            for v in range(n):
                if S >> v & 1:
                    continue  # 既に訪問済み
                new_S = S | (1 << v)
                dp[new_S][v] = min(dp[new_S][v], dp[S][u] + dist[u][v])

    # 全都市を訪問して都市0に戻る
    full = (1 << n) - 1
    return min(dp[full][i] + dist[i][0] for i in range(n))


# テスト
print(lis_length([10, 9, 2, 5, 3, 7, 101, 18]))  # 4 ([2,3,7,101] or [2,5,7,101])
print(knapsack([2, 3, 4, 5], [3, 4, 5, 6], 8))    # 10
print(matrix_chain_order([10, 30, 5, 60]))         # 4500

dist_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0],
]
print(tsp(dist_matrix))  # 80 (0→1→3→2→0)
```

### 7.3 DP の状態設計ガイド

```
┌───────────────────────────────────────────────────────────────────┐
│  DP の状態設計チートシート                                        │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  問題のタイプ     │ 典型的な状態定義                              │
│  ─────────────────┼──────────────────────────────────────────────│
│  配列/文字列      │ dp[i] = 先頭 i 要素での最適値                │
│  ナップサック系   │ dp[i][w] = i番目まで見て容量wでの最適値      │
│  区間             │ dp[l][r] = 区間[l,r]での最適値               │
│  木               │ dp[v] = 頂点vを根とする部分木の最適値        │
│  DAG上            │ dp[v] = 頂点vへの最適パス                    │
│  ビットマスク     │ dp[S] = 集合Sを処理済みの最適値              │
│  桁               │ dp[pos][tight][...] = 桁DPの状態             │
│  確率/期待値      │ dp[state] = stateでの期待値                  │
│  ゲーム           │ dp[state] = stateでの勝敗/Grundy数           │
│                                                                   │
│  状態削減テクニック:                                              │
│    ・スクロール配列: dp[i] が dp[i-1] のみに依存 → 1次元に圧縮  │
│    ・座標圧縮: 値の範囲が大きいとき、出現値のみにマッピング      │
│    ・状態の対称性: 回転・反転で等価な状態をまとめる              │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

## 8. 段階的改善の実践

### 8.1 段階的改善の方法論

いきなり最適解を目指すのではなく、愚直解から出発して段階的に改善する。これが問題解決の最も確実なアプローチである。

```
段階的改善のフロー:

  ┌──────────────────┐
  │  段階1: 愚直解    │  計算量は度外視
  │  (Brute Force)   │  正しい答えを出すことだけに集中
  └────────┬─────────┘
           │ ✓ 正解を確認
           ▼
  ┌──────────────────┐
  │  段階2: 観察      │  愚直解のどこがボトルネックか？
  │  (Observation)   │  無駄な計算はどこか？
  └────────┬─────────┘
           │ ボトルネック特定
           ▼
  ┌──────────────────┐
  │  段階3: 最適化    │  データ構造やアルゴリズムを適用
  │  (Optimization)  │  計算量を改善
  └────────┬─────────┘
           │ ✓ ストレステストで検証
           ▼
  ┌──────────────────┐
  │  段階4: 定数倍    │  必要に応じて定数倍を改善
  │  (Fine-tuning)   │  言語固有の最適化
  └──────────────────┘
```

### 8.2 実践例: 最近接点対問題

```
問題: n 個の2次元の点から、最も距離が近い点のペアを求める

段階1: 愚直解 O(n^2)
  → 全ペアの距離を計算
  → 正解の確認に使う（ストレステストの基準）

段階2: ソート活用 O(n log n + α)
  → x座標でソートし、近い点のみ比較
  → 平均的には高速だが、最悪 O(n^2) の可能性

段階3: 分割統治 O(n log n)
  → 左右に分割 + ストリップ内の限定比較
  → 最適解（決定的アルゴリズム）

段階4: ランダム化 O(n) 期待値
  → グリッド法
  → 期待計算量は最速だが分析が複雑
```

```python
import math
import random

def dist(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """2点間のユークリッド距離"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# === 段階1: O(n^2) 愚直解 ===
def closest_pair_brute(points: list[tuple[float, float]]) -> float:
    """全ペアの距離を計算"""
    n = len(points)
    min_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(points[i], points[j])
            min_dist = min(min_dist, d)
    return min_dist


# === 段階2: ソート + 枝刈り ===
def closest_pair_sorted(points: list[tuple[float, float]]) -> float:
    """x座標でソートし、距離が min_dist 以上なら枝刈り"""
    points_sorted = sorted(points)  # x座標でソート
    n = len(points_sorted)
    min_dist = float('inf')

    for i in range(n):
        for j in range(i + 1, n):
            # x座標の差だけで min_dist 以上なら、以降は全てスキップ
            if points_sorted[j][0] - points_sorted[i][0] >= min_dist:
                break
            d = dist(points_sorted[i], points_sorted[j])
            min_dist = min(min_dist, d)

    return min_dist


# === 段階3: 分割統治 O(n log n) ===
def closest_pair_dnc(points: list[tuple[float, float]]) -> float:
    """分割統治法による最近接点対"""
    def solve(pts_x: list, pts_y: list) -> float:
        n = len(pts_x)
        if n <= 3:
            return closest_pair_brute(pts_x)

        mid = n // 2
        mid_x = pts_x[mid][0]

        # 左右に分割
        left_x = pts_x[:mid]
        right_x = pts_x[mid:]
        left_set = set(map(id, left_x))

        left_y = [p for p in pts_y if id(p) in left_set]  # 簡易実装
        right_y = [p for p in pts_y if id(p) not in left_set]

        d_left = solve(left_x, left_y)
        d_right = solve(right_x, right_y)
        d = min(d_left, d_right)

        # ストリップ内の点を確認
        strip = [p for p in pts_y if abs(p[0] - mid_x) < d]

        for i in range(len(strip)):
            j = i + 1
            while j < len(strip) and strip[j][1] - strip[i][1] < d:
                d = min(d, dist(strip[i], strip[j]))
                j += 1

        return d

    pts_x = sorted(points, key=lambda p: p[0])
    pts_y = sorted(points, key=lambda p: p[1])
    return solve(pts_x, pts_y)


# === ストレステスト ===
def stress_test_closest_pair(n_tests: int = 200):
    """愚直解と最適解を比較するストレステスト"""
    for test_id in range(n_tests):
        n = random.randint(2, 50)
        points = [(random.uniform(-100, 100), random.uniform(-100, 100))
                   for _ in range(n)]

        d_brute = closest_pair_brute(points)
        d_sorted = closest_pair_sorted(points)
        d_dnc = closest_pair_dnc(points)

        assert abs(d_brute - d_sorted) < 1e-9, f"Test {test_id}: sorted 不一致"
        assert abs(d_brute - d_dnc) < 1e-9, f"Test {test_id}: dnc 不一致"

    print(f"全{n_tests}テスト通過")

# stress_test_closest_pair()
```

### 8.3 ストレステストの汎用テンプレート

```python
import random
import time

def stress_test(brute_fn, optimized_fn, generator_fn,
                comparator=None, n_tests: int = 1000, verbose: bool = False):
    """
    汎用ストレステストフレームワーク

    brute_fn:      愚直解（正しい答えを返す関数）
    optimized_fn:  最適化解（テスト対象の関数）
    generator_fn:  テスト入力を生成する関数
    comparator:    結果の比較関数（None ならば == で比較）
    """
    compare = comparator or (lambda a, b: a == b)

    for test_id in range(n_tests):
        test_input = generator_fn()
        # 同じ入力に対する結果を比較
        result_brute = brute_fn(*test_input)
        result_opt = optimized_fn(*test_input)

        if not compare(result_brute, result_opt):
            print(f"不一致! テスト #{test_id}")
            print(f"  入力:     {test_input}")
            print(f"  愚直解:   {result_brute}")
            print(f"  最適化解: {result_opt}")
            return False

        if verbose and test_id % 100 == 0:
            print(f"  テスト #{test_id} 通過")

    print(f"全{n_tests}テスト通過")
    return True


# 使用例
def gen_two_sum_input():
    n = random.randint(2, 100)
    arr = [random.randint(-100, 100) for _ in range(n)]
    target = random.randint(-200, 200)
    return (arr, target)

# stress_test(two_sum_brute, two_sum_hash, gen_two_sum_input, n_tests=5000)
```

---

## 9. エッジケースチェックリストと問題カテゴリ判定

### 9.1 包括的エッジケースチェックリスト

エッジケースの見落としは WA（Wrong Answer）の最大の原因である。問題のタイプごとに体系的にチェックする。

```
┌──────────────────────────────────────────────────────────────────────┐
│  汎用エッジケースチェックリスト                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  入力の境界:                                                         │
│    □ n = 0 (空入力)  → 特殊な返り値が必要か？                       │
│    □ n = 1 (最小入力) → ループが1回も実行されないか？               │
│    □ n = 最大値 (制約上限) → TLE/MLEにならないか？                  │
│                                                                      │
│  値の特殊性:                                                         │
│    □ 全要素が同じ値 → ソートが意味をなさない                        │
│    □ 既にソート済み / 逆順 → 最良/最悪ケース                        │
│    □ 負の値 / ゼロ → 積の符号、除算のゼロ割り                      │
│    □ 整数オーバーフロー → Python以外では注意                        │
│    □ 値が非常に大きい / 非常に小さい → 精度問題                     │
│                                                                      │
│  グラフ固有:                                                         │
│    □ 自己ループ → dist[v][v] = 0 の確認                             │
│    □ 多重辺 → 最小コストの辺を選ぶか？                              │
│    □ 非連結グラフ → 到達不能の場合の返り値                          │
│    □ 木（辺 = 頂点-1） → サイクルなしの前提                        │
│    □ 頂点数1のグラフ → 辺が0本                                      │
│    □ 完全グラフ → 辺数 O(n^2) でメモリ注意                         │
│                                                                      │
│  文字列固有:                                                         │
│    □ 空文字列 → len=0 での処理                                      │
│    □ 1文字 → 回文判定等で特殊扱い                                   │
│    □ 全文字が同じ → "aaaa" 等                                       │
│    □ 大文字/小文字の混在 → 正規化が必要か                           │
│    □ 特殊文字 → スペース、記号の扱い                                │
│                                                                      │
│  数値固有:                                                           │
│    □ 0除算 → 分母が0になるケース                                    │
│    □ 負の数のMOD → Python: -1 % 3 = 2, C++: -1 % 3 = -1           │
│    □ 浮動小数点の比較 → abs(a-b) < eps を使う                      │
│    □ 大きな数のMOD → 途中でMODを取らないとオーバーフロー           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 問題カテゴリ判定フローチャート

```
問題を読む
    │
    ├─ 最適化問題？ (最大化/最小化を求める)
    │   ├─ 貪欲選択性質あり？
    │   │   ├─ Yes → 貪欲法
    │   │   └─ 不明 → 反例を探す。なければ貪欲法を試す
    │   ├─ 重複部分問題あり？
    │   │   ├─ 状態数が少ない → DP
    │   │   └─ 状態数が多い → メモ化再帰 or 状態圧縮
    │   ├─ 答えに単調性あり？ → 二分探索（答えの決め打ち）
    │   └─ 制約が小さい（n ≤ 20）？ → ビットDP / 全探索
    │
    ├─ グラフ問題？
    │   ├─ 最短経路？
    │   │   ├─ 重みなし → BFS
    │   │   ├─ 非負重み → Dijkstra
    │   │   ├─ 負辺あり → Bellman-Ford
    │   │   └─ 全対全 → Floyd-Warshall (n ≤ 300)
    │   ├─ 連結性？
    │   │   ├─ 静的 → BFS/DFS
    │   │   └─ 動的（辺の追加） → Union-Find
    │   ├─ 順序関係？ → トポロジカルソート
    │   ├─ 最小全域木？ → Kruskal / Prim
    │   ├─ マッチング？ → 二部マッチング / ネットワークフロー
    │   └─ 強連結成分？ → Tarjan / Kosaraju
    │
    ├─ 区間/配列問題？
    │   ├─ 区間クエリ（更新なし）？ → 累積和 / Sparse Table
    │   ├─ 区間クエリ（更新あり）？ → セグメント木 / BIT
    │   ├─ 部分配列の和/積？ → 累積和 / 尺取り法
    │   ├─ 二分探索可能？ → 答えの二分探索
    │   └─ オフラインクエリ？ → Mo's algorithm
    │
    ├─ 文字列問題？
    │   ├─ パターン検索？ → KMP / Z-algorithm / Rabin-Karp
    │   ├─ 接頭辞の検索？ → Trie
    │   ├─ 部分列の比較？ → DP (LCS / Edit Distance)
    │   ├─ 回文？ → Manacher / DP
    │   └─ 接尾辞に関する問題？ → Suffix Array
    │
    ├─ 数学問題？
    │   ├─ 素数判定/因数分解？ → エラトステネスの篩 / 試し割り
    │   ├─ GCD/LCM？ → ユークリッド互除法
    │   ├─ 組合せ数？ → Pascal三角形 / 逆元
    │   ├─ MOD演算？ → 繰り返し二乗法 / フェルマーの小定理
    │   └─ 行列？ → 行列累乗
    │
    └─ ゲーム理論？
        ├─ 二人零和ゲーム？ → Minimax / Alpha-Beta
        ├─ Nim系？ → Grundy数 / XOR
        └─ 一般のゲーム？ → DP + ゲーム理論
```

### 9.3 アルゴリズム選択比較表

| 問題の性質 | 第一候補 | 第二候補 | 避けるべき手法 |
|:---|:---|:---|:---|
| 最短経路（重みなし） | BFS O(V+E) | --- | Dijkstra（無駄に複雑） |
| 最短経路（非負重み） | Dijkstra O(E log V) | A* | Bellman-Ford O(VE)（遅い） |
| 最短経路（負辺あり） | Bellman-Ford O(VE) | SPFA | Dijkstra（誤答） |
| 全対最短経路 | Floyd-Warshall O(V^3) | Dijkstra V回 | BFS V回（重みあり不可） |
| 連結成分（静的） | Union-Find O(alpha(n)) | BFS/DFS O(V+E) | --- |
| 連結成分（動的追加） | Union-Find | --- | BFS毎回（遅い） |
| 区間和（更新なし） | 累積和 O(1)/クエリ | --- | セグメント木（過剰） |
| 区間和（更新あり） | BIT O(log n) | セグメント木 | 毎回再計算 O(n) |
| 区間最小値（更新なし） | Sparse Table O(1)/クエリ | セグメント木 | --- |
| 区間最小値（更新あり） | セグメント木 O(log n) | --- | BIT（非対応） |
| ソート済み配列の検索 | 二分探索 O(log n) | --- | 線形探索 O(n) |
| 要素の挿入・削除・検索 | ハッシュ O(1) 平均 | 平衡BST O(log n) | 配列の線形探索 |
| 最大/最小の動的管理 | ヒープ O(log n) | --- | ソート毎回 O(n log n) |

### 9.4 計算量の比較表

| 操作 | Python での目安 (n=10^6) | C++ での目安 (n=10^6) | 注意点 |
|:---|:---|:---|:---|
| 単純ループ | 約0.1秒 | 約0.003秒 | Python は30-100倍遅い |
| ソート | 約0.3秒 | 約0.06秒 | TimSort は高速 |
| dict/set 操作 | 約0.2秒 | 約0.05秒 | ハッシュの定数倍 |
| BFS/DFS | 約0.5秒 | 約0.01秒 | Python は再帰上限に注意 |
| セグメント木構築 | 約1秒 | 約0.02秒 | Python では厳しい場合あり |
| 二分探索 (log n 回) | 約0.00002秒 | 約0.000001秒 | 判定関数の計算量に依存 |

---

## 10. アンチパターン集

### アンチパターン1: 制約を無視した実装

```python
# ================================================================
# BAD: n=10^5 なのに O(n^2) を実装
# → 10^10 回の演算 → TLE（制限時間超過）
# ================================================================

def find_duplicate_bad(arr: list[int]) -> int:
    """O(n^2): 全ペアを比較して重複を探す"""
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return arr[i]
    return -1

# ================================================================
# GOOD: 制約を先に確認し、許容される計算量を見積もる
# n=10^5 → O(n log n) 以下が必要
# ================================================================

def find_duplicate_good(arr: list[int]) -> int:
    """O(n): ハッシュセットで重複を検出"""
    seen: set[int] = set()
    for num in arr:
        if num in seen:
            return num
        seen.add(num)
    return -1
```

### アンチパターン2: いきなり最適解を目指す

```python
# ================================================================
# BAD: 最初から複雑な最適アルゴリズムを実装しようとする
# → バグが入りやすい、デバッグが困難、時間を浪費
# ================================================================

# いきなりセグメント木 + 座標圧縮 + オフラインクエリを書き始める...

# ================================================================
# GOOD: まず愚直解を実装し、正しい出力を確認してから最適化
# ================================================================

# Step 1: O(n^2) の愚直解で正解を確認
# Step 2: O(n log n) の最適解を実装
# Step 3: ストレステストで愚直解と最適解を比較してバグを検出
# Step 4: 大きな入力での実行時間を確認
```

### アンチパターン3: グローバル変数の乱用

```python
# ================================================================
# BAD: グローバル変数を多用し、状態管理が混乱
# ================================================================

visited = set()       # グローバル → テストケース間で初期化忘れ
result = []           # グローバル → 複数回呼び出しで汚染

def dfs_bad(graph, v):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs_bad(graph, u)

# ================================================================
# GOOD: 状態を引数や返り値で管理、または関数内にカプセル化
# ================================================================

def dfs_good(graph: dict, start: int) -> set[int]:
    """状態を関数内にカプセル化"""
    visited: set[int] = set()
    stack = [start]

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        for u in graph[v]:
            if u not in visited:
                stack.append(u)

    return visited
```

### アンチパターン4: 浮動小数点の直接比較

```python
# ================================================================
# BAD: 浮動小数点数を == で比較
# ================================================================

def is_right_triangle_bad(a: float, b: float, c: float) -> bool:
    return a*a + b*b == c*c  # 浮動小数点誤差で False になる可能性

# ================================================================
# GOOD: 誤差を考慮した比較
# ================================================================

EPS = 1e-9

def is_right_triangle_good(a: float, b: float, c: float) -> bool:
    return abs(a*a + b*b - c*c) < EPS
```

### アンチパターン5: 再帰の深さ制限を忘れる

```python
# ================================================================
# BAD: Python のデフォルト再帰上限（1000）を超える
# ================================================================

def dfs_recursive_bad(graph, v, visited):
    visited.add(v)
    for u in graph[v]:
        if u not in visited:
            dfs_recursive_bad(graph, u, visited)
    # n=10000 の線形グラフで RecursionError

# ================================================================
# GOOD: 再帰上限を引き上げるか、スタックで実装
# ================================================================

import sys
sys.setrecursionlimit(300000)  # 再帰上限を引き上げ

# または、スタックベースの反復実装（推奨）
def dfs_iterative(graph: dict, start: int) -> list[int]:
    visited = set()
    stack = [start]
    order = []

    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        order.append(v)
        for u in reversed(graph[v]):
            if u not in visited:
                stack.append(u)

    return order
```

---

## 11. 演習問題（3段階）

### Level 1: 基礎（制約分析とパターン認識）

**演習 1-1: 制約から計算量を逆算する**

以下の各問題について、許容される計算量と使うべきアルゴリズムを推定せよ。

```
(a) n <= 15 の配列が与えられる。全ての部分集合の中から和が最大のものを求めよ。
    → 許容計算量: O(?)  → 候補アルゴリズム: ?

(b) n <= 200,000 の配列が与えられる。区間 [l, r] の和を Q 回求めよ。(Q <= 200,000)
    → 許容計算量: O(?)  → 候補アルゴリズム: ?

(c) n <= 300 の頂点を持つ重み付きグラフで、全対間の最短距離を求めよ。
    → 許容計算量: O(?)  → 候補アルゴリズム: ?

(d) 長さ n <= 10^6 の文字列 S と、長さ m <= 10^6 のパターン P が与えられる。
    S 中に P が出現する全ての位置を求めよ。
    → 許容計算量: O(?)  → 候補アルゴリズム: ?
```

<details>
<summary>解答</summary>

```
(a) n <= 15 → O(2^n) = O(32768) → ビットマスクで全部分集合を列挙。
    各部分集合の和を計算 → O(2^n * n) でも十分。

(b) n, Q <= 200,000 → 前処理 O(n)、各クエリ O(1) が理想。
    → 累積和。前処理 O(n)、クエリ O(1)、合計 O(n + Q)。

(c) n <= 300 → O(n^3) = O(2.7 * 10^7) → Floyd-Warshall。
    Dijkstra を n 回でも O(n^2 log n) で可能だが、Floyd-Warshall の方がシンプル。

(d) n, m <= 10^6 → O(n + m) が必要。
    → KMP法 または Z-algorithm。愚直な O(nm) は TLE。
```
</details>

**演習 1-2: パターンマッチング**

以下の各問題のキーワードから、使うべきアルゴリズムを推定せよ。

```
(a) 「グラフ上で頂点 s から頂点 t への最短経路を求めよ。辺の重みは全て 1。」
(b) 「配列から連続する部分列で、和が K 以上となる最短のものを求めよ。」
(c) 「N 個の都市を全て訪問して戻ってくる最短経路を求めよ。N <= 20。」
(d) 「文字列 S の中で最長の回文部分文字列を求めよ。」
```

<details>
<summary>解答</summary>

```
(a) 重みなし最短経路 → BFS O(V + E)
(b) 連続部分列 + 和の条件 + 最短 → 尺取り法 O(n)
(c) 全都市訪問 + N <= 20 → ビットDP (TSP) O(2^n * n^2)
(d) 最長回文部分文字列 → Manacher O(n) または DP O(n^2)
```
</details>

### Level 2: 応用（段階的改善の実践）

**演習 2-1: 最長部分配列問題**

長さ n の整数配列 arr と正整数 k が与えられる。異なる要素が高々 k 種類の連続部分配列の最長の長さを求めよ。

制約: 1 <= n <= 10^5, 1 <= k <= n

```
入力例: arr = [1, 2, 1, 2, 3], k = 2
出力例: 4  (部分配列 [1, 2, 1, 2])
```

段階的に解を構築せよ:
1. O(n^3) の愚直解を実装
2. O(n^2) に改善
3. O(n) に最適化（尺取り法）
4. ストレステストで検証

<details>
<summary>解答</summary>

```python
from collections import defaultdict

# 段階1: O(n^3) 全部分配列を列挙し、各部分配列の異なり数を数える
def longest_k_distinct_brute(arr: list[int], k: int) -> int:
    n = len(arr)
    max_len = 0
    for i in range(n):
        for j in range(i, n):
            distinct = len(set(arr[i:j+1]))  # O(n) の集合構築
            if distinct <= k:
                max_len = max(max_len, j - i + 1)
    return max_len

# 段階2: O(n^2) setを逐次更新
def longest_k_distinct_n2(arr: list[int], k: int) -> int:
    n = len(arr)
    max_len = 0
    for i in range(n):
        count = defaultdict(int)
        distinct = 0
        for j in range(i, n):
            if count[arr[j]] == 0:
                distinct += 1
            count[arr[j]] += 1
            if distinct <= k:
                max_len = max(max_len, j - i + 1)
            else:
                break  # これ以上伸ばしても条件を満たさない
    return max_len

# 段階3: O(n) 尺取り法
def longest_k_distinct_optimal(arr: list[int], k: int) -> int:
    n = len(arr)
    count = defaultdict(int)
    left = 0
    max_len = 0

    for right in range(n):
        count[arr[right]] += 1

        while len(count) > k:
            count[arr[left]] -= 1
            if count[arr[left]] == 0:
                del count[arr[left]]
            left += 1

        max_len = max(max_len, right - left + 1)

    return max_len
```
</details>

**演習 2-2: 二分探索の応用**

n 個の正整数からなる配列 arr と正整数 m が与えられる。arr を m 個の連続部分配列に分割するとき、各部分配列の和の最大値を最小化せよ。

制約: 1 <= m <= n <= 10^5, 1 <= arr[i] <= 10^4

```
入力例: arr = [7, 2, 5, 10, 8], m = 2
出力例: 18  (分割: [7, 2, 5] と [10, 8]、和はそれぞれ 14 と 18)
```

<details>
<summary>解答</summary>

本章の「5.2 実装パターン」の `min_max_partition` 関数を参照。答えの二分探索で解く。判定関数 `can_partition(max_sum)` は、各区間の和が max_sum 以下で m 分割可能かを貪欲に判定する。

計算量: O(n log(sum(arr)))
</details>

### Level 3: 発展（複合問題）

**演習 3-1: 総合問題**

n 人の人がいて、それぞれ能力値 a[i] を持つ。k 個のチームに分け、各チーム内の能力値の差（最大値 - 最小値）の合計を最小化せよ。各チームは1人以上で構成される。

制約: 1 <= k <= n <= 5000

```
入力例: n=5, k=2, a=[3, 1, 7, 5, 2]
        ソート後: [1, 2, 3, 5, 7]
        分割例: [1, 2, 3] と [5, 7] → 差は (3-1)+(7-5) = 2+2 = 4
出力例: 4
```

ヒント: ソートした後、DP で解く。dp[i][j] = 先頭 i 人を j チームに分けたときの最小コスト。

<details>
<summary>解答の方針</summary>

```python
def min_team_diff(a: list[int], k: int) -> int:
    a.sort()
    n = len(a)
    INF = float('inf')

    # dp[i][j] = a[0..i-1] を j チームに分けた最小コスト
    dp = [[INF] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 0

    for j in range(1, k + 1):
        for i in range(j, n + 1):
            # 最後のチームが a[m..i-1] (1人以上)
            for m in range(j - 1, i):
                cost = a[i - 1] - a[m]  # ソート済みなので max-min = 末尾-先頭
                dp[i][j] = min(dp[i][j], dp[m][j - 1] + cost)

    return dp[n][k]

# 計算量: O(n^2 * k)
# n=5000, k=5000 では最悪 O(n^3) だが、k が小さいケースでは十分高速。
```
</details>

---

## 12. FAQ（よくある質問）

### Q1: 問題を見て何も思いつかない場合はどうすればよいか？

**A:** 以下の5段階を順に試す。

1. **小さな具体例で手計算する。** n=3 から 5 の例を3つ以上作り、紙の上で解く。パターンが見えてくることが多い。
2. **制約から計算量を逆算する。** n が 10^5 以下なら O(n log n) が必要、と分かるだけで候補が大幅に絞られる。
3. **パターン認識マップを参照する。** キーワード（最短、最大、数え上げ等）からアルゴリズム候補を列挙する。
4. **愚直解を書いてみる。** 正しい答えを出すコードがあると、最適化の足掛かりになる。愚直解のボトルネック（どこが O(n^2) か）を特定し、データ構造やアルゴリズムで改善する。
5. **類似問題を思い出す。** 過去に解いた似た問題の解法を転用できないか考える。

「何も思いつかない」は多くの場合「具体例が足りない」か「制約分析をしていない」のどちらかである。

### Q2: 愚直解は正しいがTLEになる。どう最適化すればよいか？

**A:** 以下のチェックリストを上から順に確認する。

1. **不要な計算の除去**: 同じ値を何度も計算していないか？ → メモ化/累積和
2. **データ構造の変更**: 線形探索をハッシュや二分探索に置き換えられないか？
3. **アルゴリズムの変更**: O(n^2) を O(n log n) のアルゴリズムに変えられないか？
4. **二分探索の適用**: 答えに単調性があれば、答えの二分探索が使えないか？
5. **前処理の活用**: クエリに O(1) で答えるための前処理（累積和、Sparse Table等）
6. **定数倍の改善**: Python特有の高速化（sys.stdin, 内包表記, PyPy）

### Q3: DP の遷移式が立てられない場合はどうすればよいか？

**A:** 以下の手順で考える。

1. **状態の定義を決める。** 「dp[i] は何を表すか？」を日本語で明確に定義する。曖昧な定義では遷移式も曖昧になる。
2. **手計算で遷移を追う。** dp[0], dp[1], dp[2], ... を具体的に手計算し、「dp[3] を求めるために何が必要だったか」を書き出す。
3. **末尾に注目する。** 「最後の要素をどう扱うか」で場合分けすると遷移式が見えやすい。例: 最後の要素を選ぶ/選ばない。
4. **状態に次元を追加する。** dp[i] だけでは情報が足りない場合、dp[i][j] のように次元を増やす。例: ナップサック問題では重さの次元が必要。
5. **既知の DP パターンとの類似性を探す。** LIS, LCS, ナップサック, 区間DP, ビットDP 等、典型パターンとの対応関係を考える。

### Q4: コンテスト中の時間配分はどうすればよいか？

**A:** 一般的なコンテスト（2時間6問形式）での推奨時間配分は以下の通り。

```
A問題: 5分    （実装のみ）
B問題: 10分   （簡単なアルゴリズム）
C問題: 20分   （標準的なアルゴリズム）
D問題: 30分   （応用的なアルゴリズム）
E問題: 30分   （高度なアルゴリズム）
F問題: 25分   （非常に高度 / 諦めることも戦略）

重要: 20分考えて方針が立たない問題はスキップし、次の問題に移る。
      残り時間で戻ってきて再挑戦する。
```

### Q5: Python と C++ のどちらを使うべきか？

**A:** 状況による。

```
Python が向いているケース:
  ・多倍長整数が必要（Python は標準でサポート）
  ・実装が複雑で、バグを減らしたい
  ・n が 10^5 程度で、定数倍がボトルネックにならない
  ・ライブラリ（itertools, collections 等）が活用できる

C++ が向いているケース:
  ・n が 10^6 以上で定数倍が重要
  ・実行時間制限が厳しい（1〜2秒）
  ・STL のデータ構造（set, map, priority_queue）が便利

折衷案: PyPy を使う
  ・Python の書きやすさで、3〜5倍の高速化
  ・ただし一部のライブラリが使えない場合がある
```

### Q6: ストレステストはどの程度の規模で実行すべきか？

**A:** テスト入力のサイズと回数の目安は以下の通り。

```
目的別のテスト設定:

  正解性の確認（愚直解との比較）:
    ・入力サイズ: n = 1〜50（愚直解が高速に動く範囲）
    ・テスト回数: 1000〜10000回
    ・時間: 数秒〜数十秒で完了するように調整

  エッジケースの確認:
    ・n = 0, 1, 2 を明示的にテスト
    ・全要素同一値、ソート済み、逆順
    ・値の最大値、最小値、ゼロ

  性能の確認:
    ・n = 制約の最大値（10^5, 10^6 等）
    ・最悪ケースを手動で構築（ソート済み、逆順、ランダム等）
    ・1〜3ケースで実行時間を測定
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

## 13. まとめ

| 項目 | 要点 |
|:---|:---|
| 問題解決5ステップ | 理解 → 具体例 → 制約分析 → 設計 → 実装検証 |
| 制約分析 | n の範囲から O(?) を逆算して適切なアルゴリズムを選ぶ |
| パターン認識 | キーワードと問題構造から既知の手法にマッピング |
| 段階的改善 | 愚直解 → 正解確認 → 最適化の3段階で進める |
| エッジケース | 空入力・最小最大・特殊値・グラフ特有ケースを必ず確認 |
| 典型テクニック | 尺取り法、答えの二分探索、累積和、DP、ビットDP |
| ストレステスト | 愚直解との比較が最強のデバッグ手法 |
| アンチパターン | 制約無視、いきなり最適解、グローバル変数乱用、浮動小数点直接比較 |
| 時間配分 | 20分考えて方針が立たなければスキップする勇気 |

---

## 次に読むべきガイド

- [競技プログラミング](./01-competitive-programming.md) -- 問題解決力を実戦で鍛える
- [動的計画法](../02-algorithms/04-dynamic-programming.md) -- 最も出題頻度の高いパラダイム
- [グラフ走査](../02-algorithms/02-graph-traversal.md) -- グラフ問題の基礎
- ソート -- 前処理として頻出
- データ構造基礎 -- 適切なデータ構造の選択

---

## 参考文献

1. Skiena, S. S. (2020). *The Algorithm Design Manual* (3rd ed.). Springer. -- Part I: Practical Algorithm Design は問題解決の全体フレームワークを詳述。第1章から第8章がパターン認識と設計手法の基盤となる。

2. Polya, G. (1945). *How to Solve It*. Princeton University Press. -- 数学的問題解決の古典。「理解する → 計画を立てる → 実行する → 振り返る」の4段階フレームワークは、アルゴリズム問題解決にも直接適用できる。

3. Halim, S. & Halim, F. (2013). *Competitive Programming 3*. -- Chapter 1 から 3 が問題解決の実践的テクニックを網羅。パターン認識、制約分析、典型アルゴリズムの適用方法が豊富な例題とともに解説されている。

4. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. -- 第1部「基礎」が計算量分析の理論的基盤を提供。各アルゴリズムの正当性証明と計算量解析の手法は全てここに基づく。

5. Laaksonen, A. (2017). *Competitive Programmer's Handbook*. -- 無料で公開されているオンライン教材。二分探索、DP、グラフアルゴリズム等の典型パターンが簡潔にまとめられており、パターン認識の訓練に最適。URL: https://cses.fi/book/book.pdf

6. Sedgewick, R. & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley. -- 実装寄りの教科書。Java によるコード例が豊富で、データ構造の選択指針と計算量解析の実践的な応用が学べる。
