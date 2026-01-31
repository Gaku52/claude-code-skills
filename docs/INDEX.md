# 検索キーワード一覧・索引 (Keyword Index)

このファイルは、claude-code-skillsリポジトリ内の全情報を検索可能にするための包括的なインデックスです。

**最終更新**: 2026-01-04
**総証明数**: 30個 (アルゴリズム22 + 分散システム5 + DB設計2 + React 1)
**総論文引用数**: 255+本
**現在のスコア**: 90/100点 (MIT+ Level)

---

## 💡 このリポジトリの使い方

### 役割分担
- **このリポジトリ**: 原則、パターン、ベストプラクティス、数学的証明
- **公式ドキュメント**: 最新API、詳細仕様、マイグレーションガイド

### 学習フロー
1. **証明・理論** → このリポジトリで完結（不変の知識）
2. **スキルガイド** → このリポジトリで原則を学ぶ → 公式ドキュメントで最新詳細を確認
3. **npmパッケージ** → このリポジトリで完結（実装とAPI）

**各項目に公式ドキュメントへのリンクがあります。最短経路で目的地へ到達できます。**

---

## 📖 使い方

### 基本的な検索方法

1. **キーワードで検索**: このファイル内で `Ctrl+F` (Mac: `Cmd+F`) で検索
2. **カテゴリから探す**: 下記のカテゴリ別索引を参照
3. **直接移動**: ファイルパスをクリックして該当ドキュメントへ

### 具体的な検索例

**例1: アルゴリズムを名前で探す**
```
検索キーワード: "Dijkstra"
→ 結果: Dijkstra's Algorithm の証明が見つかる
→ ファイル: backend-development/guides/algorithms/dijkstra-algorithm-proof.md
```

**例2: 計算量で探す**
```
検索キーワード: "O(log n)"
→ 結果: Binary Search, Fenwick Tree, Segment Tree などが見つかる
→ すべて O(log n) の計算量を持つアルゴリズム
```

**例3: 性能指標で探す**
```
検索キーワード: "speedup"
→ 結果: FFT (852×), Binary Search (4027×), Fenwick Tree (1736×) など
→ 実験で確認された高速化倍率
```

**例4: 統計手法で探す**
```
検索キーワード: "Cohen's d"
→ 結果: t検定、効果量、統計パッケージの説明が見つかる
→ ファイル: packages/stats/src/ttest.ts など
```

**例5: 分散システムで探す**
```
検索キーワード: "Paxos"
→ 結果: Paxos証明、TLA+仕様、比較データが見つかる
→ ファイル: 02-paxos-consensus-proof.md, 03-paxos-consensus.tla
```

**例6: ファイル形式で探す**
```
検索キーワード: ".tla"
→ 結果: すべてのTLA+仕様ファイルが見つかる
→ 3つの形式検証ファイル
```

**例7: パッケージ機能で探す**
```
検索キーワード: "paired t-test"
→ 結果: 統計パッケージのt検定機能が見つかる
→ ファイル: packages/stats/src/ttest.ts, examples/stats-example.ts
```

**例8: デモで探す**
```
検索キーワード: "demo"
→ 結果: 3つのインタラクティブデモが見つかる
→ Landing Page, Statistics Playground, CRDT Demo
```

---

## 🗂️ カテゴリ別索引

### 1. アルゴリズム証明 (Algorithm Proofs) - 22個

**このリポジトリで完結**: 数学的証明、計算量解析、実験データ

| 名前 | 計算量 | 証明ファイル | 公式実装例 |
|------|-------|------------|-----------|
| A* Pathfinding | O((V+E)log V) | [proof](backend-development/guides/algorithms/astar-pathfinding-proof.md) | [Wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm#Pseudocode) |
| AVL Tree | O(log n) | [proof](backend-development/guides/algorithms/avl-tree-proof.md) | [C++ std::map](https://en.cppreference.com/w/cpp/container/map) (Red-Black実装が多い) |
| **Binary Search** | **O(log n)** | [proof](backend-development/guides/algorithms/binary-search-proof.md) | [C++ std::binary_search](https://en.cppreference.com/w/cpp/algorithm/binary_search), [Python bisect](https://docs.python.org/3/library/bisect.html) |
| Bloom Filter | O(k) | [proof](backend-development/guides/algorithms/bloom-filter-proof.md) | [Redis Bloom](https://redis.io/docs/stack/bloom/) |
| Convex Hull | O(n log n) | [proof](backend-development/guides/algorithms/convex-hull-proof.md) | [SciPy ConvexHull](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html) |
| **Dijkstra** | **O((V+E)log V)** | [proof](backend-development/guides/algorithms/dijkstra-algorithm-proof.md) | [NetworkX dijkstra](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html) |
| Dynamic Programming | O(nm) 典型 | [proof](backend-development/guides/algorithms/dynamic-programming-proof.md) | [Python functools.lru_cache](https://docs.python.org/3/library/functools.html#functools.lru_cache) |
| Fenwick Tree | O(log n) | [proof](backend-development/guides/algorithms/fenwick-tree-proof.md) | [C++ Fenwick Tree](https://cp-algorithms.com/data_structures/fenwick.html) |
| **FFT** | **O(n log n)** | [proof](backend-development/guides/algorithms/fft-proof.md) | [NumPy FFT](https://numpy.org/doc/stable/reference/routines.fft.html), [FFTW](https://www.fftw.org) |
| Graph Traversal | O(V+E) | [proof](backend-development/guides/algorithms/graph-traversal-proof.md) | [Python BFS/DFS](https://docs.python.org/3/library/collections.html#collections.deque) |
| Hash Table | O(1) 平均 | [proof](backend-development/guides/algorithms/hash-table-proof.md) | [C++ std::unordered_map](https://en.cppreference.com/w/cpp/container/unordered_map), [Python dict](https://docs.python.org/3/library/stdtypes.html#dict) |
| Minimum Spanning Tree | O(E log V) | [proof](backend-development/guides/algorithms/minimum-spanning-tree-proof.md) | [NetworkX MST](https://networkx.org/documentation/stable/reference/algorithms/tree.html) |
| Network Flow | O(VE²) | [proof](backend-development/guides/algorithms/network-flow-proof.md) | [NetworkX Flow](https://networkx.org/documentation/stable/reference/algorithms/flow.html) |
| Red-Black Tree | O(log n) | [proof](backend-development/guides/algorithms/red-black-tree-proof.md) | [C++ std::map](https://en.cppreference.com/w/cpp/container/map), [Java TreeMap](https://docs.oracle.com/javase/8/docs/api/java/util/TreeMap.html) |
| Segment Tree | O(log n) | [proof](backend-development/guides/algorithms/segment-tree-proof.md) | [C++ Segment Tree](https://cp-algorithms.com/data_structures/segment_tree.html) |
| Skip List | O(log n) 期待値 | [proof](backend-development/guides/algorithms/skip-list-proof.md) | [Redis Sorted Set](https://redis.io/docs/data-types/sorted-sets/) |
| **Sorting** | **O(n log n)** | [proof](backend-development/guides/algorithms/sorting-algorithms-proof.md) | [C++ std::sort](https://en.cppreference.com/w/cpp/algorithm/sort), [Python sorted](https://docs.python.org/3/library/functions.html#sorted) |
| Strassen | O(n^2.807) | [proof](backend-development/guides/algorithms/strassen-matrix-multiplication-proof.md) | [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) |
| **String Matching** | **O(n+m)** | [proof](backend-development/guides/algorithms/string-matching-proof.md) | [C++ std::search](https://en.cppreference.com/w/cpp/algorithm/search), [Python str.find](https://docs.python.org/3/library/stdtypes.html#str.find) |
| Topological Sort | O(V+E) | [proof](backend-development/guides/algorithms/topological-sort-proof.md) | [NetworkX topological_sort](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.topological_sort.html) |
| Trie | O(m) | [proof](backend-development/guides/algorithms/trie-proof.md) | [Python trie](https://pypi.org/project/pygtrie/) |
| Union-Find | O(α(n)) | [proof](backend-development/guides/algorithms/union-find-proof.md) | [NetworkX union-find](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.mst.minimum_spanning_tree.html) |

---

### 2. 分散システム証明 (Distributed Systems Proofs) - 5個

| 名前 | キーワード | ファイル | 主要結果 |
|------|-----------|---------|---------|
| CAP Theorem | CAP定理, 不可能性, C∧A∧P | [01-cap-theorem-proof.md](_IMPROVEMENTS/phase3/distributed-systems/01-cap-theorem-proof.md) | 数学的証明完了 |
| Paxos Consensus | パクソス, 合意形成, 100% safety | [02-paxos-consensus-proof.md](_IMPROVEMENTS/phase3/distributed-systems/02-paxos-consensus-proof.md) | 98% agreement |
| Raft Consensus | ラフト, 合意形成, 43% faster | [03-raft-consensus-proof.md](_IMPROVEMENTS/phase3/distributed-systems/03-raft-consensus-proof.md) | Paxosより43%高速 |
| Distributed Transactions | 2PC, 3PC, 分散トランザクション | [04-distributed-transactions-proof.md](_IMPROVEMENTS/phase3/distributed-systems/04-distributed-transactions-proof.md) | 原子性保証 |
| Eventual Consistency & CRDT | 結果整合性, 無矛盾複製データ型 | [05-eventual-consistency-crdt-proof.md](_IMPROVEMENTS/phase3/distributed-systems/05-eventual-consistency-crdt-proof.md) | 480-650ms収束 |

---

### 3. 形式検証 (Formal Verification - TLA+) - 5個

#### TLA+ 仕様ファイル (.tla) - 3個

| 名前 | キーワード | ファイル | 検証状態数 |
|------|-----------|---------|----------|
| Two-Phase Commit | 2PC, 原子性, ブロッキング | [02-two-phase-commit.tla](_IMPROVEMENTS/phase3/tla-plus/02-two-phase-commit.tla) | 12,500 states |
| Paxos Consensus | パクソス, 安全性, 活性 | [03-paxos-consensus.tla](_IMPROVEMENTS/phase3/tla-plus/03-paxos-consensus.tla) | 50,000 states |
| Raft Consensus | ラフト, リーダー選出 | [04-raft-consensus.tla](_IMPROVEMENTS/phase3/tla-plus/04-raft-consensus.tla) | 90,000 states |

**合計検証状態数**: 152,500+ states

#### TLA+ ドキュメント (.md) - 2個

| 名前 | キーワード | ファイル |
|------|-----------|---------|
| TLA+ Introduction | TLA+, 形式仕様, 入門 | [01-tla-plus-introduction.md](_IMPROVEMENTS/phase3/tla-plus/01-tla-plus-introduction.md) |
| TLA+ Summary | まとめ, ベストプラクティス | [05-tla-plus-summary.md](_IMPROVEMENTS/phase3/tla-plus/05-tla-plus-summary.md) |

---

### 4. 統計・実験手法 (Statistics & Methodology) - 3個

| 名前 | キーワード | ファイル | 内容 |
|------|-----------|---------|------|
| Statistical Methodology | 統計手法, t検定, Cohen's d | [01-statistical-methodology.md](_IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md) | 完全な統計手法ガイド |
| Experiment Template (TypeScript) | 実験テンプレート, TypeScript, 再現性 | [02-experiment-template.ts](_IMPROVEMENTS/phase3/experiment-templates/02-experiment-template.ts) | 実行可能テンプレート |
| Reporting Template | レポートテンプレート, 論文形式 | [03-reporting-template.md](_IMPROVEMENTS/phase3/experiment-templates/03-reporting-template.md) | 学術論文形式 |

---

### 5. npmパッケージ (npm Packages) - 2個

#### @claude-code-skills/stats
**場所**: `packages/stats/`
**機能**: 統計分析ライブラリ (800+ lines)

| ファイル | 内容 | キーワード |
|---------|------|-----------|
| [types.ts](packages/stats/src/types.ts) | 型定義 | TTestResult, RegressionResult |
| [distributions.ts](packages/stats/src/distributions.ts) | 確率分布 | Normal distribution, t-distribution |
| [ttest.ts](packages/stats/src/ttest.ts) | t検定 | paired t-test, independent t-test, Cohen's d |
| [regression.ts](packages/stats/src/regression.ts) | 回帰分析 | linear regression, log-log, R² |
| [utils.ts](packages/stats/src/utils.ts) | ユーティリティ | mean, SD, CI, outliers |
| [experiment.ts](packages/stats/src/experiment.ts) | 実験フレームワーク | before-after, sample size |
| [index.ts](packages/stats/src/index.ts) | エントリーポイント | エクスポート |
| [README.md](packages/stats/README.md) | ドキュメント | API reference, examples |

#### @claude-code-skills/crdt
**場所**: `packages/crdt/`
**機能**: CRDT実装ライブラリ (700+ lines)

| ファイル | 内容 | キーワード |
|---------|------|-----------|
| [types.ts](packages/crdt/src/types.ts) | 型定義 | CRDT interface, semilattice |
| [g-counter.ts](packages/crdt/src/g-counter.ts) | G-Counter | grow-only, join-semilattice |
| [pn-counter.ts](packages/crdt/src/pn-counter.ts) | PN-Counter | increment/decrement |
| [lww-set.ts](packages/crdt/src/lww-set.ts) | LWW-Element-Set | last-write-wins, timestamp |
| [or-set.ts](packages/crdt/src/or-set.ts) | OR-Set | observed-remove, unique tags |
| [index.ts](packages/crdt/src/index.ts) | エントリーポイント | エクスポート |
| [README.md](packages/crdt/README.md) | ドキュメント | API reference, convergence proof |

---

### 6. インタラクティブデモ (Interactive Demos) - 3個

| 名前 | URL | ファイル | 内容 |
|------|-----|---------|------|
| Landing Page | / | [demos/index.html](demos/index.html) | プロジェクト概要 |
| Statistics Playground | /stats-playground/ | [demos/stats-playground/index.html](demos/stats-playground/index.html) | t検定計算機 |
| CRDT Demo | /crdt-demo/ | [demos/crdt-demo/index.html](demos/crdt-demo/index.html) | CRDT可視化 |

**デモサイト**: https://gaku52.github.io/claude-code-skills/

---

### 7. 使用例 (Example Code) - 2個

| ファイル | 内容 | キーワード |
|---------|------|-----------|
| [stats-example.ts](examples/stats-example.ts) | 統計ライブラリ使用例 | paired t-test, regression, experiment |
| [crdt-example.ts](examples/crdt-example.ts) | CRDTライブラリ使用例 | G-Counter, OR-Set, convergence |

---

### 8. データベース設計証明 (Database Design Proofs) - 1個

| 名前 | キーワード | ファイル |
|------|-----------|---------|
| B-Tree Operations | B木, データベースインデックス | [btree-operations-proof.md](database-design/guides/algorithms/btree-operations-proof.md) |

---

### 9. React アルゴリズム証明 (React Algorithm Proofs) - 2個

| 名前 | キーワード | ファイル |
|------|-----------|---------|
| Fiber Reconciliation | Fiber, 再調整, React内部 | [fiber-reconciliation-proof.md](react-development/guides/algorithms/fiber-reconciliation-proof.md) |
| Virtual DOM Diffing | Virtual DOM, Diff, 最適化 | [virtual-dom-diffing-proof.md](react-development/guides/algorithms/virtual-dom-diffing-proof.md) |

---

### 10. プロジェクト管理ドキュメント

| 名前 | ファイル | 内容 |
|------|---------|------|
| **メインREADME** | [README.md](README.md) | プロジェクト概要、スコア、主要成果 |
| **検索索引** | [INDEX.md](INDEX.md) | このファイル (全Markdownの検索) |
| **ナビゲーション** | [NAVIGATION.md](NAVIGATION.md) | 作者向けクイックナビゲーション |
| **メンテナンス** | [MAINTENANCE.md](MAINTENANCE.md) | 日々の更新・メンテナンス方法 |

#### Phase レポート (4個)

| フェーズ | スコア変化 | ファイル | 主な成果 |
|---------|-----------|---------|---------|
| Phase 1 | 38→55点 | [PHASE1-COMPLETION-REPORT.md](_IMPROVEMENTS/PHASE1-COMPLETION-REPORT.md) | 統計厳格化 (n≥30, p<0.001) |
| Phase 2 | 55→68点 | [PHASE2-COMPLETION-REPORT.md](_IMPROVEMENTS/PHASE2-COMPLETION-REPORT.md) | 22個のアルゴリズム証明 |
| Phase 3 | 68→81点 | [PHASE3-COMPLETION-REPORT.md](_IMPROVEMENTS/PHASE3-COMPLETION-REPORT.md) | 分散システム + TLA+ |
| Phase 4 | 81→90点 | [PHASE4-COMPLETION-REPORT.md](_IMPROVEMENTS/PHASE4-COMPLETION-REPORT.md) | npm packages + デモ |

#### その他の重要ドキュメント

| ファイル | 内容 |
|---------|------|
| [90-POINT-ROADMAP.md](_IMPROVEMENTS/90-POINT-ROADMAP.md) | 90点到達のロードマップ |
| [MIT-EVALUATION-REPORT.md](_IMPROVEMENTS/MIT-EVALUATION-REPORT.md) | MIT基準の評価レポート |
| [QUICK-START.md](_IMPROVEMENTS/QUICK-START.md) | クイックスタートガイド |

---

### 11. CI/CD & インフラストラクチャ

| ファイル | 内容 | 機能 |
|---------|------|------|
| [.github/workflows/ci.yml](.github/workflows/ci.yml) | CI pipeline | ビルド、テスト、lint |
| [.github/workflows/pages.yml](.github/workflows/pages.yml) | GitHub Pages deploy | デモの自動デプロイ |
| [pnpm-workspace.yaml](pnpm-workspace.yaml) | Monorepo config | Workspace設定 |
| [package.json](package.json) | Root package | 統一スクリプト |
| [tsconfig.base.json](tsconfig.base.json) | TypeScript config | 共有設定 |

---

### 12. Skills ディレクトリ (24個のスキルガイド)

**使い方**: このリポジトリで原則・パターンを学ぶ → 公式ドキュメントで最新詳細を確認

| スキル | SKILL.md | 公式ドキュメント | 学ぶべきこと |
|--------|---------|----------------|------------|
| **Backend Development** | [SKILL.md](backend-development/SKILL.md) | [Node.js](https://nodejs.org/docs), [Express](https://expressjs.com) | 原則: API設計、エラーハンドリング<br>公式: 最新API仕様 |
| **CI/CD Automation** | [SKILL.md](ci-cd-automation/SKILL.md) | [GitHub Actions](https://docs.github.com/actions), [Fastlane](https://docs.fastlane.tools) | 原則: パイプライン設計<br>公式: 最新ワークフロー構文 |
| **CLI Development** | [SKILL.md](cli-development/SKILL.md) | [Commander.js](https://github.com/tj/commander.js), [Click](https://click.palletsprojects.com) | 原則: CLI設計、引数パース<br>公式: ライブラリAPI |
| **Code Review** | [SKILL.md](code-review/SKILL.md) | [GitHub Review](https://docs.github.com/pull-requests) | 原則: レビュープロセス<br>公式: GitHub機能 |
| **Database Design** | [SKILL.md](database-design/SKILL.md) | [PostgreSQL](https://www.postgresql.org/docs/), [Prisma](https://www.prisma.io/docs) | 原則: 正規化、インデックス設計<br>公式: SQL仕様、ORM |
| **Dependency Management** | [SKILL.md](dependency-management/SKILL.md) | [npm](https://docs.npmjs.com), [pnpm](https://pnpm.io), [CocoaPods](https://guides.cocoapods.org) | 原則: バージョン管理<br>公式: パッケージマネージャー |
| **Documentation** | [SKILL.md](documentation/SKILL.md) | [Markdown Guide](https://www.markdownguide.org), [TypeDoc](https://typedoc.org) | 原則: ドキュメント構造<br>公式: ツール仕様 |
| **Frontend Performance** | [SKILL.md](frontend-performance/SKILL.md) | [Web.dev Performance](https://web.dev/performance/), [Core Web Vitals](https://web.dev/vitals/) | 原則: 最適化パターン<br>公式: 測定ツール、ベストプラクティス |
| **Git Workflow** | [SKILL.md](git-workflow/SKILL.md) | [Git Docs](https://git-scm.com/doc), [GitHub Flow](https://docs.github.com/get-started/quickstart/github-flow) | 原則: ブランチ戦略<br>公式: Gitコマンド仕様 |
| **Incident Logger** | [SKILL.md](incident-logger/SKILL.md) | [Postmortem Templates](https://sre.google/workbook/postmortem-culture/) | 原則: インシデント管理<br>公式: SREベストプラクティス |
| **iOS Development** | [SKILL.md](ios-development/SKILL.md) | [Apple Developer](https://developer.apple.com/documentation/), [SwiftUI](https://developer.apple.com/xcode/swiftui/) | 原則: MVVM、Architecture<br>公式: SwiftUI、UIKit API |
| **iOS Project Setup** | [SKILL.md](ios-project-setup/SKILL.md) | [Xcode](https://developer.apple.com/documentation/xcode), [SPM](https://www.swift.org/package-manager/) | 原則: プロジェクト構成<br>公式: Xcode設定、ビルドシステム |
| **iOS Security** | [SKILL.md](ios-security/SKILL.md) | [iOS Security Guide](https://support.apple.com/guide/security/welcome/web), [Keychain](https://developer.apple.com/documentation/security/keychain_services) | 原則: セキュリティパターン<br>公式: iOS Security API |
| **Lessons Learned** | [SKILL.md](lessons-learned/SKILL.md) | [Retrospective Techniques](https://retromat.org) | 原則: 振り返りプロセス<br>公式: ファシリテーション技法 |
| **Networking & Data** | [SKILL.md](networking-data/SKILL.md) | [URLSession](https://developer.apple.com/documentation/foundation/urlsession), [Core Data](https://developer.apple.com/documentation/coredata) | 原則: 通信・永続化パターン<br>公式: Apple Framework |
| **Next.js** | [SKILL.md](nextjs-development/SKILL.md) | [Next.js Docs](https://nextjs.org/docs) | 原則: App Router、SSR<br>公式: Caching、最新機能 |
| **Node.js** | [SKILL.md](nodejs-development/SKILL.md) | [Node.js Docs](https://nodejs.org/docs/latest/api/) | 原則: 非同期パターン<br>公式: Node.js API |
| **Python** | [SKILL.md](python-development/SKILL.md) | [Python Docs](https://docs.python.org/3/), [FastAPI](https://fastapi.tiangolo.com) | 原則: 型ヒント、FastAPI<br>公式: 標準ライブラリ |
| **Quality Assurance** | [SKILL.md](quality-assurance/SKILL.md) | [Testing Best Practices](https://testing-library.com/docs/) | 原則: QAプロセス<br>公式: テストツール |
| **React** | [SKILL.md](react-development/SKILL.md) | [React.dev](https://react.dev) | 原則: Hooks、パフォーマンス<br>公式: 最新API、Server Components |
| **Script Development** | [SKILL.md](script-development/SKILL.md) | [Bash Guide](https://www.gnu.org/software/bash/manual/), [Python Scripts](https://docs.python.org/3/tutorial/) | 原則: 自動化パターン<br>公式: シェル・言語仕様 |
| **SwiftUI Patterns** | [SKILL.md](swiftui-patterns/SKILL.md) | [SwiftUI Tutorials](https://developer.apple.com/tutorials/swiftui) | 原則: 状態管理、レイアウト<br>公式: SwiftUI API |
| **Testing Strategy** | [SKILL.md](testing-strategy/SKILL.md) | [Jest](https://jestjs.io), [Testing Library](https://testing-library.com) | 原則: テスト戦略<br>公式: フレームワークAPI |
| **Web Accessibility** | [SKILL.md](web-accessibility/SKILL.md) | [WCAG](https://www.w3.org/WAI/WCAG21/quickref/), [MDN Accessibility](https://developer.mozilla.org/docs/Web/Accessibility) | 原則: アクセシビリティ原則<br>公式: WCAG仕様 |
| **Web Development** | [SKILL.md](web-development/SKILL.md) | [MDN Web Docs](https://developer.mozilla.org) | 原則: フレームワーク選定<br>公式: Web標準 |

---

## 🔍 キーワード検索マップ

### パフォーマンス関連
- **高速化**: FFT (852×), Binary Search (4027×), Fenwick Tree (1736×), Segment Tree (1205×), String Matching (183×)
- **計算量**: O(log n), O(n log n), O(n²), O(n^2.807), Master定理
- **R²値**: 0.9997 (FFT), 0.9997 (Binary Search), 0.9998 (Fenwick Tree)

### 統計関連
- **有意性検定**: paired t-test, independent t-test, p-value < 0.001
- **効果量**: Cohen's d, 実用的有意性
- **信頼区間**: 95% CI, 標準誤差
- **回帰分析**: linear regression, log-log regression, R²

### 分散システム関連
- **合意形成**: Paxos, Raft, 2PC, 3PC
- **一貫性**: CAP定理, 強結果整合性, eventual consistency
- **CRDT**: G-Counter, PN-Counter, LWW-Set, OR-Set
- **形式検証**: TLA+, model checking, safety, liveness

### データ構造関連
- **木構造**: AVL Tree, Red-Black Tree, Trie, B-Tree
- **区間クエリ**: Fenwick Tree, Segment Tree, RMQ
- **素集合**: Union-Find, 経路圧縮
- **確率的**: Bloom Filter, Skip List

### アルゴリズム設計技法
- **分割統治**: Sorting, FFT, Strassen
- **動的計画法**: Dynamic Programming (LCS, Knapsack)
- **貪欲法**: Dijkstra, MST (Kruskal, Prim)
- **文字列**: String Matching (KMP, Rabin-Karp)
- **グラフ**: Graph Traversal, Topological Sort, Network Flow
- **計算幾何**: Convex Hull

---

## 📊 統計情報

### 証明の分布
| カテゴリ | 証明数 | 割合 |
|---------|-------|------|
| アルゴリズム | 22 | 71.0% |
| 分散システム | 5 | 16.1% |
| 形式検証 (TLA+) | 3 | 9.7% |
| データベース | 1 | 3.2% |
| **合計** | **31** | **100%** |

### ファイル数統計
| カテゴリ | ファイル数 |
|---------|----------|
| アルゴリズム証明 (.md) | 22 |
| 分散システム証明 (.md) | 5 |
| TLA+ 仕様 (.tla) | 3 |
| TLA+ ドキュメント (.md) | 2 |
| 統計テンプレート (.md, .ts) | 3 |
| npm packages (2パッケージ) | 13 TypeScriptファイル |
| インタラクティブデモ (.html) | 3 |
| 使用例 (.ts) | 2 |
| Phaseレポート (.md) | 4 |
| プロジェクト管理 (.md) | 4 |
| **合計 (主要ファイル)** | **61+** |

### 引用論文の分布
| 分野 | 論文数 | 割合 |
|------|-------|------|
| アルゴリズム | 150 | 58.8% |
| 統計 | 57 | 22.4% |
| 分散システム | 40 | 15.7% |
| 形式検証 | 8 | 3.1% |
| **合計** | **255+** | **100%** |

### コード量
| コンポーネント | 行数 |
|--------------|------|
| Stats package | 800+ |
| CRDT package | 700+ |
| Examples | 200+ |
| Demos (HTML/CSS/JS) | 1,500+ |
| **合計** | **3,200+** |

---

## 🎯 学習パス

### 初級者向け (3-5時間)
1. [Binary Search](backend-development/guides/algorithms/binary-search-proof.md) - 基本的な分割統治
2. [Sorting Algorithms](backend-development/guides/algorithms/sorting-algorithms-proof.md) - 基本的なソート
3. [統計手法入門](_IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md) - 実験計画法

### 中級者向け (8-10時間)
1. [Fenwick Tree](backend-development/guides/algorithms/fenwick-tree-proof.md) - 高度なデータ構造
2. [Dijkstra's Algorithm](backend-development/guides/algorithms/dijkstra-algorithm-proof.md) - グラフアルゴリズム
3. [String Matching](backend-development/guides/algorithms/string-matching-proof.md) - KMP, Rabin-Karp
4. [Dynamic Programming](backend-development/guides/algorithms/dynamic-programming-proof.md) - DP技法

### 上級者向け (12-15時間)
1. [FFT](backend-development/guides/algorithms/fft-proof.md) - 高度な数値計算
2. [Paxos](_IMPROVEMENTS/phase3/distributed-systems/02-paxos-consensus-proof.md) - 分散合意形成
3. [Raft](_IMPROVEMENTS/phase3/distributed-systems/03-raft-consensus-proof.md) - モダンな合意形成
4. [CRDT](_IMPROVEMENTS/phase3/distributed-systems/05-eventual-consistency-crdt-proof.md) - 最新の分散システム理論

### 研究者向け (20+ 時間)
1. [TLA+ Introduction](_IMPROVEMENTS/phase3/tla-plus/01-tla-plus-introduction.md) - 形式検証入門
2. [Statistical Methodology](_IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md) - 研究手法
3. [Experiment Template](_IMPROVEMENTS/phase3/experiment-templates/02-experiment-template.ts) - 再現可能な実験
4. 全22個のアルゴリズム証明を体系的に学習

---

## 🔗 クイックリンク

### よく参照されるファイル
- [プロジェクト概要](README.md) - **最初に読む**
- [検索索引](INDEX.md) - **このファイル**
- [ナビゲーション](NAVIGATION.md) - **作者向けクイックガイド**
- [メンテナンス](MAINTENANCE.md) - **日々の更新方法**
- [最新完了レポート](_IMPROVEMENTS/PHASE4-COMPLETION-REPORT.md) - **Phase 4成果**

### パッケージドキュメント
- [Stats Package README](packages/stats/README.md) - **API reference**
- [CRDT Package README](packages/crdt/README.md) - **API reference**
- [Stats Example Code](examples/stats-example.ts) - **使用例**
- [CRDT Example Code](examples/crdt-example.ts) - **使用例**

### デモ
- [デモサイトトップ](https://gaku52.github.io/claude-code-skills/) - **オンラインデモ**
- [統計プレイグラウンド](https://gaku52.github.io/claude-code-skills/stats-playground/) - **t検定計算機**
- [CRDTデモ](https://gaku52.github.io/claude-code-skills/crdt-demo/) - **収束可視化**

---

## 📌 検索のヒント

### このファイルで効率的に検索する方法

**1. 特定のアルゴリズムを探す**
```
例: "Dijkstra", "FFT", "String Matching", "Union-Find"
```

**2. 特定の技法を探す**
```
例: "分割統治", "動的計画法", "貪欲法", "グラフ"
```

**3. パフォーマンス指標で探す**
```
例: "speedup", "R²", "O(log n)", "O(n log n)"
```

**4. 統計手法を探す**
```
例: "t-test", "Cohen's d", "regression", "confidence interval"
```

**5. 分散システムを探す**
```
例: "Paxos", "Raft", "CRDT", "CAP", "2PC", "3PC"
```

**6. 形式検証を探す**
```
例: "TLA+", ".tla", "model checking", "safety", "liveness"
```

**7. ファイル形式で探す**
```
例: ".md", ".ts", ".tla", ".html"
```

**8. パッケージ・デモを探す**
```
例: "stats", "crdt", "demo", "example", "package"
```

---

## 🏆 主要な成果

### Phase 4 で追加されたもの (81→90点)

**npmパッケージ (2個)**:
- @claude-code-skills/stats (800+ lines)
- @claude-code-skills/crdt (700+ lines)

**インタラクティブデモ (3個)**:
- Landing Page
- Statistics Playground
- CRDT Demo

**ドキュメント (3個)**:
- [MAINTENANCE.md](MAINTENANCE.md) - メンテナンス方法
- [INDEX.md](INDEX.md) - このファイル
- [NAVIGATION.md](NAVIGATION.md) - ナビゲーションガイド

---

## 📝 全Markdownファイル一覧

### プロジェクトルート (8個)
- README.md
- INDEX.md (このファイル)
- NAVIGATION.md
- MAINTENANCE.md
- CHANGELOG.md
- CODE_OF_CONDUCT.md
- CONTRIBUTING.md
- SECURITY.md

### _IMPROVEMENTS ディレクトリ (23個)
- Phase レポート: 4個
- Phase別ドキュメント: 19個

### backend-development ディレクトリ (26個)
- アルゴリズム証明: 22個
- ガイド: 3個
- SKILL.md: 1個

### その他 24 Skill ディレクトリ
各ディレクトリに SKILL.md + 複数のガイド

**総Markdownファイル数**: 250+個

---

**このインデックスは定期的に更新されます。**

**最終更新**: 2026-01-04
**管理者**: Gaku
**リポジトリ**: [claude-code-skills](https://github.com/Gaku52/claude-code-skills)
**スコア**: 90/100 (MIT+ Level)
