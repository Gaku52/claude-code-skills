# アルゴリズムとデータ構造

> アルゴリズムとデータ構造はプログラミングの基盤。計算量分析、ソートアルゴリズム、木構造、グラフアルゴリズム、動的計画法、競技プログラミングのテクニックまで体系的に解説する。

## このSkillの対象者

- アルゴリズムを体系的に学びたいエンジニア
- コーディング面接の準備をしている方
- 競技プログラミングに取り組みたい方

## 前提知識

- 基本的なプログラミング（ループ、条件分岐、関数）
- 数学の基礎（対数、指数、集合）

## 学習ガイド

### 00-basics — 基礎

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/00-basics/00-complexity-analysis.md]] | Big-O、Big-Ω、Big-Θ、時間計算量、空間計算量 |
| 01 | [[docs/00-basics/01-recursion.md]] | 再帰、メモ化、末尾再帰、分割統治法 |
| 02 | [[docs/00-basics/02-search-algorithms.md]] | 線形探索、二分探索、三分探索、補間探索 |
| 03 | [[docs/00-basics/03-two-pointers-and-sliding-window.md]] | 二分探索応用、尺取り法、スライディングウィンドウ |

### 01-sorting — ソートアルゴリズム

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/01-sorting/00-comparison-sorts.md]] | バブル、挿入、選択、マージ、クイック、ヒープソート |
| 01 | [[docs/01-sorting/01-non-comparison-sorts.md]] | 計数ソート、基数ソート、バケットソート |
| 02 | [[docs/01-sorting/02-sorting-applications.md]] | 安定性、部分ソート、外部ソート、実装の選択基準 |

### 02-data-structures — データ構造

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/02-data-structures/00-array-and-linked-list.md]] | 配列、連結リスト、スタック、キュー、Deque |
| 01 | [[docs/02-data-structures/01-hash-table.md]] | ハッシュテーブル、衝突解決、ハッシュ関数、Bloom Filter |
| 02 | [[docs/02-data-structures/02-tree-structures.md]] | 二分木、BST、AVL、赤黒木、B-Tree |
| 03 | [[docs/02-data-structures/03-heap-and-priority-queue.md]] | ヒープ、優先度キュー、Fibonacci Heap |
| 04 | [[docs/02-data-structures/04-trie-and-advanced.md]] | Trie、Segment Tree、Fenwick Tree、Union-Find |

### 03-graph — グラフアルゴリズム

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/03-graph/00-graph-basics.md]] | グラフ表現、BFS、DFS、トポロジカルソート |
| 01 | [[docs/03-graph/01-shortest-path.md]] | Dijkstra、Bellman-Ford、Floyd-Warshall、A* |
| 02 | [[docs/03-graph/02-mst-and-flow.md]] | 最小全域木（Kruskal/Prim）、最大フロー（Ford-Fulkerson） |

### 04-advanced — 高度なアルゴリズム

| # | ファイル | 内容 |
|---|---------|------|
| 00 | [[docs/04-advanced/00-dynamic-programming.md]] | DP の基礎、部分問題、ナップサック、LCS、LIS |
| 01 | [[docs/04-advanced/01-greedy.md]] | 貪欲法、活動選択、ハフマン符号、証明テクニック |
| 02 | [[docs/04-advanced/02-string-algorithms.md]] | 文字列検索（KMP/Rabin-Karp/Aho-Corasick）、接尾辞配列 |
| 03 | [[docs/04-advanced/03-computational-geometry.md]] | 凸包、線分交差、最近点対、幾何アルゴリズム |
| 04 | [[docs/04-advanced/04-interview-patterns.md]] | 面接頻出パターン、LeetCode 攻略、問題分類 |

## クイックリファレンス

```
計算量チートシート:
  O(1)        — ハッシュテーブル lookup
  O(log n)    — 二分探索
  O(n)        — 線形探索
  O(n log n)  — マージソート、クイックソート（平均）
  O(n²)       — バブルソート、挿入ソート
  O(2^n)      — 部分集合列挙
  O(n!)       — 順列列挙
```

## 参考文献

1. Cormen, T. et al. "Introduction to Algorithms." MIT Press, 2022.
2. Sedgewick, R. "Algorithms." Addison-Wesley, 2011.
3. Skiena, S. "The Algorithm Design Manual." Springer, 2020.
