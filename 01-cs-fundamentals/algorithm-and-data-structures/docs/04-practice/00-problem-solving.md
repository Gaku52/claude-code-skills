# 問題解決法

> アルゴリズム問題を体系的に解くための思考法を、パターン認識・制約分析・段階的改善の3段階で習得する

## この章で学ぶこと

1. **パターン認識** で問題を既知のアルゴリズムカテゴリに分類できる
2. **制約分析** から許容される計算量を逆算し、適切なアルゴリズムを選択できる
3. **段階的改善** で愚直解から最適解へと効率的に到達する手順を身につける
4. **ストレステスト** と愚直解比較による堅牢なデバッグ手法を実践できる
5. **典型パターン** を12種以上マスターし、未知の問題にも対応できる引き出しを持つ

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
        return [[0]]

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
