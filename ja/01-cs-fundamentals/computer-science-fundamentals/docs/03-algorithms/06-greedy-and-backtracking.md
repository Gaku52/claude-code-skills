# 貪欲法とバックトラック

> 貪欲法は「今この瞬間の最善手」を選び続ける楽観的戦略。バックトラックは「全ての可能性を試し、失敗したら引き返す」慎重な戦略。

## この章で学ぶこと

- [ ] 貪欲法が最適解を与える条件を理解する
- [ ] 典型的な貪欲法アルゴリズムを実装できる
- [ ] バックトラックの仕組みと枝刈りを理解する
- [ ] 貪欲法の正当性を証明する手法を学ぶ
- [ ] 枝刈りの高度なテクニックを習得する
- [ ] 実務での近似アルゴリズムとヒューリスティックを理解する

## 前提知識


---

## 1. 貪欲法（Greedy Algorithm）

### 1.1 基本概念

```
貪欲法: 各ステップで局所的に最適な選択をする

  特徴:
  - 一度選んだら変更しない（後戻りなし）
  - 高速（通常 O(n log n) 以下）
  - 最適解の保証には証明が必要

  貪欲法が最適になる2つの条件:
  1. 貪欲選択性質: 局所最適な選択が全体最適に含まれる
  2. 最適部分構造: 残りの部分問題も最適に解ける
```

### 1.2 典型的な貪欲法

```python
# 1. 活動選択問題（区間スケジューリング）
def activity_selection(activities):
    """重ならない最大数の活動を選ぶ"""
    # 終了時刻でソート
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]

    for start, end in activities[1:]:
        if start >= last_end:
            selected.append((start, end))
            last_end = end

    return selected

# 例: [(1,4), (3,5), (0,6), (5,7), (3,9), (5,9), (6,10), (8,11)]
# → [(1,4), (5,7), (8,11)] — 3つの活動が最大

# 2. ハフマン符号（最適前置符号）
import heapq

def huffman(freq):
    """出現頻度から最適なハフマン木を構築"""
    heapq.heapify(heap)

    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])

    return sorted(heap[0][1:], key=lambda x: (len(x[1]), x[0]))

# 3. お釣り問題（特定の硬貨セットのみ最適）
def coin_greedy(coins, amount):
    """大きい硬貨から貪欲に使う"""
    coins.sort(reverse=True)
    result = []
    for coin in coins:
        while amount >= coin:
            result.append(coin)
            amount -= coin
    return result if amount == 0 else None

# 注意: coins=[1,5,10,25] では最適
# coins=[1,3,4] で amount=6 → 貪欲: [4,1,1]=3枚 ≠ 最適: [3,3]=2枚
```

### 1.3 貪欲法 vs DP の判断

```
いつ貪欲法が使えるか:

  ✅ 貪欲法が最適:
  - 区間スケジューリング（終了時刻でソート）
  - ハフマン符号
  - クラスカル法（最小全域木）
  - ダイクストラ法（最短経路）
  - 米国の硬貨での釣り銭
  - 分数ナップサック
  - タスクスケジューリング（デッドライン付き）

  ❌ 貪欲法が最適でない（DPが必要）:
  - 0-1 ナップサック
  - コイン問題（一般の硬貨セット）
  - 最長共通部分列
  - 編集距離
  - 巡回セールスマン問題

  判断のヒント:
  - 「各ステップで後悔しない選択」ができるか？
  - 反例が見つからないか？
  - マトロイド構造を持つか？（数学的に厳密な判定）
```

### 1.4 貪欲法の正当性証明

```python
# 貪欲法の正しさを証明する3つの方法

# 方法1: 交換論法（Exchange Argument）
# 「最適解が貪欲解と異なると仮定し、貪欲解に変換しても
#  悪くならないことを示す」

# 例: 区間スケジューリングの証明
# 最適解 O と貪欲解 G を考える。
# Oの最初の活動 o1 と Gの最初の活動 g1 について:
# g1は終了時刻が最小 → g1.end <= o1.end
# o1 を g1 に置き換えても、残りの活動との干渉は増えない
# → |G| >= |O| → 貪欲解は最適

# 方法2: 帰納法
# 「k個選んだ時点で最適解に含まれる活動が
#  k個含まれていることを帰納的に示す」

# 方法3: マトロイド理論
# 問題がマトロイド構造を持つなら、貪欲法が最適
# マトロイドの3条件:
# 1. 空集合は独立集合
# 2. 独立集合の部分集合は独立集合
# 3. 交換公理: |A| < |B| なら B\A に要素xが存在し A∪{x} も独立集合
```

### 1.5 分数ナップサック問題

```python
def fractional_knapsack(weights, values, capacity):
    """品物を分割して入れられるナップサック問題
    → 価値密度（value/weight）の高い順に詰める貪欲法が最適"""

    n = len(weights)
    # 価値密度でソート
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)

    total_value = 0
    remaining = capacity

    fractions = [0.0] * n

    for i in items:
        if remaining <= 0:
            break
        if weights[i] <= remaining:
            # 丸ごと入れる
            fractions[i] = 1.0
            total_value += values[i]
            remaining -= weights[i]
        else:
            # 一部だけ入れる
            fraction = remaining / weights[i]
            fractions[i] = fraction
            total_value += values[i] * fraction
            remaining = 0

    return total_value, fractions

# 例: weights=[10, 20, 30], values=[60, 100, 120], capacity=50
# 密度: [6, 5, 4]
# 品物0(10kg, 60円)全部 + 品物1(20kg, 100円)全部 + 品物2(20/30, 80円)
# = 60 + 100 + 80 = 240円

# 0-1ナップサックとの違い:
# 分数ナップサック: 品物を分割可能 → 貪欲法で最適 O(n log n)
# 0-1ナップサック: 品物は分割不可 → DPが必要 O(nW)
```

### 1.6 タスクスケジューリング

```python
def task_scheduling_with_deadline(tasks):
    """各タスクに利益とデッドラインがある場合、最大利益を得るスケジュール
    tasks: [(profit, deadline), ...]"""

    # 利益の降順にソート
    tasks.sort(key=lambda x: x[0], reverse=True)

    max_deadline = max(t[1] for t in tasks)
    slots = [False] * (max_deadline + 1)  # 時間スロット
    total_profit = 0
    scheduled = []

    for profit, deadline in tasks:
        # デッドラインから逆順に空きスロットを探す
        for slot in range(deadline, 0, -1):
            if not slots[slot]:
                slots[slot] = True
                total_profit += profit
                scheduled.append((profit, deadline, slot))
                break

    return total_profit, scheduled

# 例:
tasks = [(100, 2), (19, 1), (27, 2), (25, 1), (15, 3)]
profit, schedule = task_scheduling_with_deadline(tasks)
# 利益順: 100, 27, 25, 19, 15
# スロット1: 27(d=2をスロット2に、25をd=1のスロット1に... )
# 最適: スロット1=27, スロット2=100, スロット3=15 → 142

# 最小遅延スケジューリング
def min_lateness_scheduling(jobs):
    """各ジョブに処理時間と締切がある場合、最大遅延を最小化
    jobs: [(processing_time, deadline), ...]"""

    # 締切が早い順にソート（EDF: Earliest Deadline First）
    indexed_jobs = sorted(enumerate(jobs), key=lambda x: x[1][1])

    current_time = 0
    max_lateness = 0
    schedule = []

    for idx, (proc_time, deadline) in indexed_jobs:
        current_time += proc_time
        lateness = max(0, current_time - deadline)
        max_lateness = max(max_lateness, lateness)
        schedule.append({
            'job': idx,
            'start': current_time - proc_time,
            'finish': current_time,
            'lateness': lateness
        })

    return max_lateness, schedule

# 例: ジョブ: [(1, 2), (2, 4), (3, 6), (4, 8)]
# 処理順: そのまま（締切順にソート済み）
# 完了時刻: 1, 3, 6, 10
# 遅延: 0, 0, 0, 2 → 最大遅延 = 2
```

### 1.7 区間関連の貪欲法

```python
def min_intervals_to_cover(intervals, start, end):
    """区間[start, end]を最小数の部分区間でカバーする"""
    intervals.sort()
    count = 0
    i = 0
    current = start

    while current < end and i < len(intervals):
        best_end = current

        # currentを含む区間の中で、最も遠くまで伸びるものを選ぶ
        while i < len(intervals) and intervals[i][0] <= current:
            best_end = max(best_end, intervals[i][1])
            i += 1

        if best_end == current:
            return -1  # カバーできない

        count += 1
        current = best_end

    return count if current >= end else -1

# 例: intervals=[(0,3),(2,5),(3,7),(6,10)], start=0, end=10
# 選択: (0,3) → (2,5) → (3,7) → (6,10) = 4個...
# 最適: (0,3) → (3,7) → (6,10) = 3個

# 重なり合う区間のマージ
def merge_intervals(intervals):
    """重なり合う区間をマージする"""
    if not intervals:
        return []

    intervals.sort()
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged

# 例: [(1,3), (2,6), (8,10), (15,18)]
# → [(1,6), (8,10), (15,18)]

# 最小数の矢で風船を割る
def min_arrows_to_burst_balloons(balloons):
    """重なりのある風船を最小の矢で割る（区間スケジューリングの変形）"""
    if not balloons:
        return 0

    balloons.sort(key=lambda x: x[1])
    arrows = 1
    end = balloons[0][1]

    for s, e in balloons[1:]:
        if s > end:  # 新しい矢が必要
            arrows += 1
            end = e

    return arrows

# 例: balloons = [(10,16), (2,8), (1,6), (7,12)]
# ソート後: [(1,6), (2,8), (7,12), (10,16)]
# 矢1: x=6 → (1,6)と(2,8)を割る
# 矢2: x=12 → (7,12)と(10,16)を割る
# → 2本
```

### 1.8 Kruskal法とPrim法（最小全域木）

```python
class UnionFind:
    """Union-Find（素集合データ構造）"""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 経路圧縮
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def kruskal(n, edges):
    """Kruskal法: 辺を重みの昇順に追加（貪欲法）"""
    edges.sort(key=lambda x: x[2])  # (u, v, weight)
    uf = UnionFind(n)
    mst = []
    total_weight = 0

    for u, v, w in edges:
        if uf.union(u, v):  # サイクルにならなければ追加
            mst.append((u, v, w))
            total_weight += w
            if len(mst) == n - 1:
                break

    return total_weight, mst

# 計算量: O(E log E)  (ソートが支配的)
# 正当性: カットの性質 → 最小重みのクロスエッジは安全

def prim(n, adj):
    """Prim法: 頂点を1つずつ追加（優先度キューで貪欲に選択）"""
    import heapq
    visited = [False] * n
    mst = []
    total_weight = 0
    heap = [(0, 0, -1)]  # (weight, vertex, parent)

    while heap and len(mst) < n:
        w, u, parent = heapq.heappop(heap)
        if visited[u]:
            continue
        visited[u] = True
        total_weight += w
        if parent != -1:
            mst.append((parent, u, w))

        for v, weight in adj[u]:
            if not visited[v]:
                heapq.heappush(heap, (weight, v, u))

    return total_weight, mst

# 計算量: O(E log V)  (ヒープ操作)
# Kruskal: 辺が少ないグラフ（疎グラフ）に有利
# Prim: 頂点が少ないグラフ（密グラフ）に有利

# 使用例
edges = [(0,1,4), (0,7,8), (1,2,8), (1,7,11), (2,3,7),
         (2,5,4), (2,8,2), (3,4,9), (3,5,14), (4,5,10),
         (5,6,2), (6,7,1), (6,8,6), (7,8,7)]
weight, mst = kruskal(9, edges)
print(f"最小全域木の重み: {weight}")  # 37
```

### 1.9 ダイクストラ法

```python
import heapq

def dijkstra(adj, start):
    """ダイクストラ法: 単一始点最短経路（非負重みのみ）"""
    n = len(adj)
    dist = [float('inf')] * n
    dist[start] = 0
    prev = [-1] * n
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # 既により短い経路が見つかっている

        for v, w in adj[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev

def reconstruct_path(prev, start, end):
    """最短経路を復元"""
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path if path[0] == start else []

# 使用例
adj = [
    [(1, 4), (7, 8)],     # 0
    [(0, 4), (2, 8)],     # 1
    [(1, 8), (3, 7)],     # 2
    [(2, 7), (4, 9)],     # 3
    [(3, 9), (5, 10)],    # 4
    [(4, 10), (6, 2)],    # 5
    [(5, 2), (7, 1)],     # 6
    [(0, 8), (6, 1)],     # 7
]
dist, prev = dijkstra(adj, 0)
path = reconstruct_path(prev, 0, 4)
print(f"0→4の最短距離: {dist[4]}")   # 19
print(f"経路: {path}")                # [0, 1, 2, 3, 4]

# なぜ貪欲法が正しいのか？
# ダイクストラ法では、ヒープから取り出した頂点の距離は確定している
# 理由: 非負重みなので、未処理の頂点を経由しても距離は短くならない
# 注意: 負の重みがあると貪欲選択性質が崩れる → Bellman-Fordを使う
```

---

## 2. バックトラック

### 2.1 基本概念

```
バックトラック: 解の候補を構築し、制約違反で引き返す

  全探索との違い:
  - 全探索: 全ての組み合わせを生成してからチェック
  - バックトラック: 構築途中で制約違反を検出→枝刈り

  探索木のイメージ:
          root
        /  |  \
       a   b   c     ← 1文字目の選択
      /|\ /|\ /|\
     a b c a b c ...  ← 2文字目の選択
     ↑     ↑
     OK    制約違反→戻る（バックトラック）
```

### 2.2 バックトラックのテンプレート

```python
def backtrack_template(candidates, constraints):
    """バックトラックの一般的なテンプレート"""
    results = []

    def backtrack(state, choices):
        # 基底条件: 解が完成した
        if is_solution(state):
            results.append(state.copy())
            return

        for choice in choices:
            # 枝刈り: 制約に違反するなら探索しない
            if not is_valid(state, choice, constraints):
                continue

            # 選択を行う
            state.append(choice)  # make choice

            # 再帰的に探索
            backtrack(state, next_choices(choices, choice))

            # 選択を取り消す（バックトラック）
            state.pop()  # undo choice

    backtrack([], candidates)
    return results

# バックトラックの3要素:
# 1. 選択（Choice）: 何を選ぶか
# 2. 制約（Constraint）: 何が有効な選択か
# 3. 目標（Goal）: いつ解が完成するか
```

### 2.3 典型的なバックトラック

```python
# 1. N-Queens問題
def solve_n_queens(n):
    """N×Nのボードにクイーンを互いに攻撃しないように配置"""
    solutions = []

    def is_safe(board, row, col):
        for i in range(row):
            if board[i] == col:  # 同じ列
                return False
            if abs(board[i] - col) == abs(i - row):  # 対角線
                return False
        return True

    def backtrack(board, row):
        if row == n:
            solutions.append(board[:])
            return
        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(board, row + 1)
                # board[row] は次のイテレーションで上書きされるので
                # 明示的な「元に戻す」操作は不要

    backtrack([0] * n, 0)
    return solutions

# N-Queensの解の数:
# N=4: 2, N=5: 10, N=6: 4, N=7: 40, N=8: 92, N=12: 14200

# 解のビジュアライズ
def print_queens(board):
    n = len(board)
    for row in range(n):
        line = ""
        for col in range(n):
            if board[row] == col:
                line += "Q "
            else:
                line += ". "
        print(line)
    print()

# 2. 順列の生成
def permutations(nums):
    result = []
    def backtrack(path, remaining):
        if not remaining:
            result.append(path[:])
            return
        for i in range(len(remaining)):
            path.append(remaining[i])
            backtrack(path, remaining[:i] + remaining[i+1:])
            path.pop()  # バックトラック（元に戻す）
    backtrack([], nums)
    return result

# 重複要素ありの順列
def permutations_with_duplicates(nums):
    """重複する要素がある場合、重複する順列を除外"""
    nums.sort()
    result = []
    used = [False] * len(nums)

    def backtrack(path):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            # 重複の排除: 同じ値の要素は、前の要素を使った後にのみ使う
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            used[i] = True
            path.append(nums[i])
            backtrack(path)
            path.pop()
            used[i] = False

    backtrack([])
    return result

# 例: [1, 1, 2] → [[1,1,2], [1,2,1], [2,1,1]]

# 3. 数独ソルバー
def solve_sudoku(board):
    def is_valid(board, row, col, num):
        for i in range(9):
            if board[row][i] == num: return False
            if board[i][col] == num: return False
        box_r, box_c = 3 * (row // 3), 3 * (col // 3)
        for i in range(box_r, box_r + 3):
            for j in range(box_c, box_c + 3):
                if board[i][j] == num: return False
        return True

    def backtrack():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    for num in range(1, 10):
                        if is_valid(board, i, j, num):
                            board[i][j] = num
                            if backtrack():
                                return True
                            board[i][j] = 0  # バックトラック
                    return False  # どの数字も入らない
        return True  # 全マス埋まった
    backtrack()
```

### 2.4 組み合わせの列挙

```python
# 組み合わせ（Combination）
def combinations(nums, k):
    """numsからk個を選ぶ全ての組み合わせ"""
    result = []

    def backtrack(start, path):
        if len(path) == k:
            result.append(path[:])
            return

        # 残りの要素数が足りない場合の枝刈り
        remaining = len(nums) - start
        needed = k - len(path)
        if remaining < needed:
            return

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# 例: combinations([1,2,3,4], 2)
# → [[1,2], [1,3], [1,4], [2,3], [2,4], [3,4]]

# 和が target になる組み合わせ（各要素1回のみ）
def combination_sum_unique(candidates, target):
    """重複なしで和がtargetになる組み合わせ"""
    candidates.sort()
    result = []

    def backtrack(start, remaining, path):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            # 同じ値の重複を排除
            if i > start and candidates[i] == candidates[i-1]:
                continue
            if candidates[i] > remaining:
                break  # ソート済みなのでこれ以降は全て超過

            path.append(candidates[i])
            backtrack(i + 1, remaining - candidates[i], path)
            path.pop()

    backtrack(0, target, [])
    return result

# 和が target になる組み合わせ（各要素何度でも使用可）
def combination_sum_repeat(candidates, target):
    """要素を繰り返し使って和がtargetになる組み合わせ"""
    candidates.sort()
    result = []

    def backtrack(start, remaining, path):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break

            path.append(candidates[i])
            backtrack(i, remaining - candidates[i], path)  # iから再開（繰り返し可）
            path.pop()

    backtrack(0, target, [])
    return result

# 例: candidates=[2,3,6,7], target=7
# → [[2,2,3], [7]]
```

### 2.5 部分集合の列挙

```python
def subsets(nums):
    """全ての部分集合を列挙（バックトラック版）"""
    result = []

    def backtrack(start, path):
        result.append(path[:])  # 全ての途中状態も解

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# 重複要素ありの部分集合
def subsets_with_duplicates(nums):
    """重複要素がある場合の部分集合列挙"""
    nums.sort()
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            # 同じレベルで同じ値をスキップ
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# 例: [1, 2, 2]
# → [[], [1], [1,2], [1,2,2], [2], [2,2]]
```

### 2.6 枝刈りの技法

```
枝刈り（Pruning）: 不要な探索を事前に切り捨てる

  1. 実行可能性枝刈り: 制約違反が確定したら探索打ち切り
     → N-Queens: 同じ列/対角線にクイーンがあったら即終了

  2. 最適性枝刈り: 現時点で最適解に届かないなら打ち切り
     → 分岐限定法: 残りの最良見積もりが暫定最良解以下なら切る

  3. 対称性枝刈り: 対称な解を1つだけ探索
     → N-Queens: 最初のクイーンを上半分に限定

  4. 順序枝刈り: 探索の順序を工夫して早期終了を促す
     → ソートして大きい値から試す → 制約に早く引っかかる

  5. 前計算枝刈り: 事前に不可能な状態を計算しておく
     → 数独: 各セルの候補をビットマスクで管理

  枝刈りの効果:
  - N-Queens (N=8): 全探索 16,777,216通り → 枝刈り 15,720通り
  - 数独: 全探索 6.67×10²¹ → 枝刈りで瞬時
```

### 2.7 高度なバックトラック: 制約伝播付き数独ソルバー

```python
def solve_sudoku_advanced(board):
    """制約伝播 + バックトラックの高速数独ソルバー"""

    # 各セルの候補をビットマスクで管理
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9

    empty_cells = []

    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                bit = 1 << board[i][j]
                rows[i] |= bit
                cols[j] |= bit
                boxes[(i // 3) * 3 + j // 3] |= bit
            else:
                empty_cells.append((i, j))

    def get_candidates(i, j):
        """セル(i,j)に入れられる数字のリスト"""
        used = rows[i] | cols[j] | boxes[(i // 3) * 3 + j // 3]
        return [num for num in range(1, 10) if not (used & (1 << num))]

    def backtrack(idx):
        if idx == len(empty_cells):
            return True

        i, j = empty_cells[idx]
        box_idx = (i // 3) * 3 + j // 3

        for num in get_candidates(i, j):
            bit = 1 << num
            board[i][j] = num
            rows[i] |= bit
            cols[j] |= bit
            boxes[box_idx] |= bit

            if backtrack(idx + 1):
                return True

            board[i][j] = 0
            rows[i] ^= bit
            cols[j] ^= bit
            boxes[box_idx] ^= bit

        return False

    # MRV (Minimum Remaining Values) ヒューリスティック
    # 候補数が少ないセルから先に埋める
    empty_cells.sort(key=lambda cell: len(get_candidates(cell[0], cell[1])))

    backtrack(0)
    return board

# MRVヒューリスティックにより、探索木の分岐数を最小化
# 候補が1つしかないセル（裸のシングル）は即座に確定
```

---

## 3. 全探索のテクニック

### 3.1 ビット全探索

```python
# ビット全探索: 2^n 通りの部分集合を列挙

def subsets_bitmask(nums):
    """全ての部分集合を列挙"""
    n = len(nums)
    result = []
    for mask in range(1 << n):  # 0 から 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # i番目のビットが立っているか
                subset.append(nums[i])
        result.append(subset)
    return result

# nums = [1, 2, 3]
# mask=000 → []
# mask=001 → [1]
# mask=010 → [2]
# mask=011 → [1, 2]
# mask=100 → [3]
# mask=101 → [1, 3]
# mask=110 → [2, 3]
# mask=111 → [1, 2, 3]

# 適用条件: n ≤ 20 程度（2^20 ≈ 100万）
```

### 3.2 半分全列挙（Meet in the Middle）

```python
def subset_sum_meet_in_middle(nums, target):
    """Meet in the Middle: 2^n → 2^(n/2) × 2 に分割"""
    n = len(nums)
    half = n // 2

    # 前半の部分集合の和を列挙
    sums_first = {}
    for mask in range(1 << half):
        s = sum(nums[i] for i in range(half) if mask & (1 << i))
        sums_first[s] = sums_first.get(s, 0) + 1

    # 後半の部分集合で target - s を探す
    count = 0
    remaining = n - half
    for mask in range(1 << remaining):
        s = sum(nums[half + i] for i in range(remaining) if mask & (1 << i))
        complement = target - s
        if complement in sums_first:
            count += sums_first[complement]

    return count

# 計算量: O(2^(n/2) × n)
# n=40 の場合: 2^40 ≈ 1兆 → 2^20 ≈ 100万（現実的）

# 使用例
nums = list(range(1, 41))  # 1から40
target = 410  # 1+2+...+40 = 820 の半分
# 2^40通りの全探索は不可能だが、Meet in the Middleなら可能
```

### 3.3 探索の状態空間と計算量

```
各探索手法の計算量:

  ┌──────────────────────┬─────────────┬──────────────┐
  │ 手法                  │ 計算量       │ 適用範囲      │
  ├──────────────────────┼─────────────┼──────────────┤
  │ 全順列               │ O(n!)        │ n ≤ 10       │
  │ ビット全探索         │ O(2^n × n)   │ n ≤ 20       │
  │ Meet in the Middle   │ O(2^(n/2) × n)│ n ≤ 40      │
  │ バックトラック       │ O(指数) ※    │ 枝刈り次第    │
  │ DFS/BFS             │ O(V + E)     │ グラフの大きさ │
  └──────────────────────┴─────────────┴──────────────┘

  ※ バックトラックの計算量は枝刈りの効率に大きく依存する
  枝刈りなし: O(n!) や O(k^n) 程度
  良い枝刈り: 問題によっては O(n × k) に近づく
```

---

## 4. 分岐限定法（Branch and Bound）

### 4.1 基本概念

```python
def branch_and_bound_knapsack(weights, values, capacity):
    """分岐限定法による0-1ナップサック問題"""
    n = len(weights)

    # 価値密度でソート（上界計算のため）
    items = sorted(range(n), key=lambda i: values[i] / weights[i], reverse=True)
    sorted_weights = [weights[i] for i in items]
    sorted_values = [values[i] for i in items]

    def upper_bound(idx, remaining_cap, current_value):
        """分数ナップサックで上界を計算"""
        bound = current_value
        cap = remaining_cap

        for i in range(idx, n):
            if sorted_weights[i] <= cap:
                bound += sorted_values[i]
                cap -= sorted_weights[i]
            else:
                bound += sorted_values[i] * (cap / sorted_weights[i])
                break

        return bound

    best = [0]

    def backtrack(idx, remaining_cap, current_value):
        if current_value > best[0]:
            best[0] = current_value

        if idx == n:
            return

        # 枝刈り: 上界が現在の最良解以下なら探索しない
        if upper_bound(idx, remaining_cap, current_value) <= best[0]:
            return

        # 品物を入れる
        if sorted_weights[idx] <= remaining_cap:
            backtrack(idx + 1, remaining_cap - sorted_weights[idx],
                     current_value + sorted_values[idx])

        # 品物を入れない
        backtrack(idx + 1, remaining_cap, current_value)

    backtrack(0, capacity, 0)
    return best[0]

# 分岐限定法のポイント:
# 1. 上界（楽観的見積もり）を計算する
# 2. 上界が暫定最良解以下なら、そのブランチを切る
# 3. 良い初期解があると多くの枝が切れる
# → 良い初期解のために、先に貪欲法で近似解を求めるのが有効
```

### 4.2 TSPの分岐限定法

```python
def tsp_branch_and_bound(dist):
    """TSP を分岐限定法で解く"""
    n = len(dist)
    INF = float('inf')
    best_cost = [INF]
    best_path = [None]

    def lower_bound(visited, current, cost):
        """未訪問都市の最小出辺の合計を下界として計算"""
        bound = cost
        for i in range(n):
            if i not in visited:
                min_edge = min(
                    dist[i][j] for j in range(n)
                    if j != i and (j not in visited or j == 0)
                )
                bound += min_edge
        return bound

    def backtrack(current, visited, path, cost):
        if len(visited) == n:
            total = cost + dist[current][0]
            if total < best_cost[0]:
                best_cost[0] = total
                best_path[0] = path[:]
            return

        # 下界による枝刈り
        if lower_bound(visited, current, cost) >= best_cost[0]:
            return

        for next_city in range(n):
            if next_city not in visited:
                new_cost = cost + dist[current][next_city]
                if new_cost < best_cost[0]:
                    visited.add(next_city)
                    path.append(next_city)
                    backtrack(next_city, visited, path, new_cost)
                    path.pop()
                    visited.remove(next_city)

    backtrack(0, {0}, [0], 0)
    best_path[0].append(0)
    return best_cost[0], best_path[0]
```

---

## 5. 実務での近似アルゴリズム

### 5.1 NP困難問題への対処法

```
NP困難問題に対する実務的なアプローチ:

  1. 厳密解法（小さい入力）
     - ブルートフォース: n ≤ 10
     - ビット全探索: n ≤ 20
     - ビットDP: n ≤ 20
     - 分岐限定法: n ≤ 30程度（問題による）

  2. 近似アルゴリズム（保証付き）
     - 頂点被覆: 2-近似（最適解の2倍以内）
     - TSP（三角不等式）: 1.5-近似（Christofides）
     - 集合被覆: O(log n)-近似

  3. ヒューリスティック（保証なし）
     - 焼きなまし法（SA）
     - 遺伝的アルゴリズム（GA）
     - タブーサーチ
     - 局所探索
     - ランダム化アルゴリズム

  4. 問題の制限・緩和
     - 特殊ケースに帰着
     - 入力サイズの制限
     - 解の品質の妥協
```

### 5.2 焼きなまし法の実装例

```python
import random
import math

def simulated_annealing_tsp(dist, initial_temp=10000, cooling_rate=0.9995,
                            min_temp=1e-8, max_iterations=1000000):
    """焼きなまし法によるTSPの近似解"""
    n = len(dist)

    # 初期解: ランダム順列
    current = list(range(n))
    random.shuffle(current)

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    current_cost = tour_cost(current)
    best = current[:]
    best_cost = current_cost
    temp = initial_temp

    for iteration in range(max_iterations):
        if temp < min_temp:
            break

        # 近傍: 2-opt (ランダムな2辺を入れ替え)
        i, j = sorted(random.sample(range(n), 2))
        new_tour = current[:i] + current[i:j+1][::-1] + current[j+1:]
        new_cost = tour_cost(new_tour)

        delta = new_cost - current_cost

        # 改善なら必ず受理、改悪でも確率的に受理
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = new_tour
            current_cost = new_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        temp *= cooling_rate

    return best_cost, best

# 焼きなまし法のパラメータチューニング:
# - 初期温度: 大きいほど初期の探索範囲が広い
# - 冷却速度: 1に近いほどゆっくり冷える（高品質だが遅い）
# - 近傍の定義: 問題に適した操作を選ぶ
```

### 5.3 局所探索と2-opt

```python
def two_opt_tsp(dist):
    """2-opt局所探索によるTSPの改善"""
    n = len(dist)
    tour = list(range(n))

    def tour_cost(tour):
        return sum(dist[tour[i]][tour[(i+1) % n]] for i in range(n))

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue  # 同じ辺

                # 2辺を入れ替えた場合のコスト変化
                delta = (
                    dist[tour[i]][tour[j]] +
                    dist[tour[i+1]][tour[(j+1) % n]] -
                    dist[tour[i]][tour[i+1]] -
                    dist[tour[j]][tour[(j+1) % n]]
                )

                if delta < -1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    improved = True

    return tour_cost(tour), tour

# 2-optの計算量: O(n^2) per iteration
# 通常は少数回のイテレーションで収束
# 最適解の保証はないが、実務的に良い解が得られる
```

---

## 6. 実践演習

### 演習1: 貪欲法（基礎）
分数ナップサック問題（品物を切り分けてよい場合）を貪欲法で解け。0-1ナップサックとの違いを述べよ。

### 演習2: バックトラック（応用）
与えられた数字の配列から、和が target になる全ての組み合わせを求めよ（同じ要素は1回のみ使用可能）。

### 演習3: 区間スケジューリング（応用）
重み付き区間スケジューリング問題を解け。各活動に利益があり、重ならない活動の利益の合計を最大化せよ（ヒント: DP + 二分探索）。

### 演習4: グラフ問題（応用）
Kruskal法で最小全域木を求めるプログラムを実装せよ。Union-Findデータ構造も実装すること。

### 演習5: バックトラック応用（発展）
数独ソルバーを制約伝播付きで実装し、通常のバックトラックとの性能差を比較せよ。

### 演習6: 最適化（発展）
巡回セールスマン問題を、バックトラック+枝刈り（分岐限定法）で解くプログラムを実装し、都市数を増やした時の実行時間の変化を計測せよ。

### 演習7: 近似アルゴリズム（発展）
焼きなまし法でTSPを解き、厳密解（ビットDP）との品質差を比較せよ。パラメータの影響を分析せよ。


---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
---

## FAQ

### Q1: 貪欲法の正しさをどう証明しますか？
**A**: 3つの方法: (1)交換論法: 最適解を貪欲解に変換しても悪くならないことを示す (2)帰納法: 各ステップで最適性が維持されることを示す (3)マトロイド理論: 問題がマトロイド構造を持つことを示す。実務的には、まず反例を探すのが最も効率的。小さいケースで全探索と結果を比較してみるとよい。

### Q2: バックトラックと動的計画法の使い分けは？
**A**: 部分問題が重複するならDP。重複がなく全パターン列挙が必要ならバックトラック。「全ての解を列挙する」問題はバックトラック。「最適値だけ求める」問題はDP向き。ただし、バックトラック + メモ化 = トップダウンDP という関係もある。

### Q3: NP困難な問題に実務でどう対処しますか？
**A**: (1)近似アルゴリズム（最適解の定数倍以内を保証）(2)ヒューリスティック（焼きなまし法、遺伝的アルゴリズム）(3)問題サイズの制限（n<=20ならビット全探索）(4)特殊ケースへの帰着。まず問題の構造を分析し、利用できる特殊性がないか確認することが重要。

### Q4: 枝刈りの効果をどう評価しますか？
**A**: (1)探索ノード数をカウントして枝刈りなしと比較 (2)実行時間を計測 (3)理論的な計算量の改善を分析。良い枝刈りは探索空間を指数的に削減する。しかし枝刈りのコスト自体が高すぎると逆効果になるので、簡単に計算できる境界値を使うことが重要。

### Q5: 貪欲法が使えるのにDPを使うのは問題ですか？
**A**: 正しい答えは得られるが、計算量が無駄に大きくなる。例えば区間スケジューリングは貪欲法で O(n log n) だが、DPで解くと O(n^2) になる。ただし、貪欲法の正しさに自信がない場合は、安全策としてDPを使うのも合理的な判断。

### Q6: Meet in the Middle はどういう問題に使えますか？
**A**: 入力を2つに分割し、各半分を独立に全探索した後に結果をマージできる問題に使える。典型的なのは部分和問題（n<=40）。分割した各半分が2^(n/2)通りで済むため、全体として2^n から 2^(n/2)×2 に削減される。ソートして二分探索、またはハッシュマップでマージする。

---

## まとめ

| 手法 | 計算量 | 最適性 | 用途 |
|------|--------|--------|------|
| 貪欲法 | O(n log n)~ | 条件付き最適 | 区間スケジューリング、MST、最短経路 |
| バックトラック | O(指数)~ | 完全探索（枝刈りで高速化）| N-Queens、数独、組合せ列挙 |
| ビット全探索 | O(2^n x n) | 完全 | n<=20の部分集合問題 |
| Meet in the Middle | O(2^(n/2) x n) | 完全 | n<=40の部分集合問題 |
| 分岐限定法 | O(指数) ※ | 完全（枝刈り効率に依存） | ナップサック、TSP |
| 焼きなまし法 | ユーザ指定 | 近似（保証なし） | NP困難な最適化問題 |
| 近似アルゴリズム | 多項式 | 近似（保証あり） | 頂点被覆、集合被覆 |

---

## 次に読むべきガイド

---

## 参考文献
1. Cormen, T. H. et al. "Introduction to Algorithms." Chapters 16-17.
2. Skiena, S. S. "The Algorithm Design Manual." Chapters 8-9.
3. Papadimitriou, C., Steiglitz, K. "Combinatorial Optimization." Dover, 1998.
4. Aarts, E., Korst, J. "Simulated Annealing and Boltzmann Machines." 1989.
5. Cook, W. "In Pursuit of the Traveling Salesman." Princeton University Press, 2012.
6. Lawler, E. L. et al. "The Traveling Salesman Problem." Wiley, 1985.
