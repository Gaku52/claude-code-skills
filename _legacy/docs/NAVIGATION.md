# ナビゲーションガイド (Quick Navigation Guide for Authors)

**作者用クイックリファレンス**

このガイドは、**作者が直接ドキュメントにアクセスし、理解していることを示すため**のナビゲーションガイドです。Claude Code Skillsによる自動参照に加えて、作者自身が能動的にドキュメントを参照できるようにします。

---

## 🎯 目的

- ✅ 作者が見たいページに即座に移動できる
- ✅ 作者がプロジェクト全体を理解していることを示す
- ✅ 他者に説明する際の参照元として活用
- ✅ レビュー・メンテナンス作業を効率化

---

## 📂 ディレクトリ構造の全体像

```
claude-code-skills/
├── README.md                          # プロジェクト概要 (最初に読む)
├── INDEX.md                           # 検索キーワード一覧 (このファイルで検索)
├── NAVIGATION.md                      # このファイル (ナビゲーション)
├── MAINTENANCE.md                     # メンテナンス方法
│
├── backend-development/               # アルゴリズム証明 (25個)
│   └── guides/algorithms/
│       ├── binary-search-proof.md     # 4027× speedup
│       ├── fft-proof.md              # 852× speedup
│       ├── fenwick-tree-proof.md     # 1736× speedup
│       └── ... (全25ファイル)
│
├── _IMPROVEMENTS/                     # フェーズ別改善記録
│   ├── PHASE1-COMPLETION-REPORT.md   # 38→55点
│   ├── PHASE2-COMPLETION-REPORT.md   # 55→68点
│   ├── PHASE3-COMPLETION-REPORT.md   # 68→81点
│   ├── PHASE4-COMPLETION-REPORT.md   # 81→90点 ⭐最新
│   │
│   ├── phase3/
│   │   ├── distributed-systems/      # 分散システム証明 (5個)
│   │   │   ├── 01-cap-theorem-proof.md
│   │   │   ├── 02-paxos-consensus-proof.md
│   │   │   ├── 03-raft-consensus-proof.md
│   │   │   ├── 04-2pc-3pc-proof.md
│   │   │   └── 05-crdt-proof.md
│   │   │
│   │   ├── tla-plus/                 # TLA+形式検証 (3+1個)
│   │   │   ├── 01-tla-plus-introduction.md
│   │   │   ├── 02-two-phase-commit.tla
│   │   │   ├── 03-paxos-consensus.tla
│   │   │   └── 04-raft-consensus.tla
│   │   │
│   │   └── experiment-templates/     # 統計・実験テンプレート (3個)
│   │       ├── 01-statistical-methodology.md
│   │       ├── 02-experiment-template.ts
│   │       └── 03-reporting-template.md
│   │
│   └── ... (phase1/, phase2/)
│
├── packages/                          # npmパッケージ (2個)
│   ├── stats/                        # 統計ライブラリ
│   │   ├── README.md                 # API reference
│   │   ├── package.json
│   │   └── src/
│   │       ├── index.ts              # エントリーポイント
│   │       ├── ttest.ts              # t検定
│   │       ├── regression.ts         # 回帰分析
│   │       └── ... (全7ファイル)
│   │
│   └── crdt/                         # CRDTライブラリ
│       ├── README.md                 # API reference
│       ├── package.json
│       └── src/
│           ├── index.ts              # エントリーポイント
│           ├── g-counter.ts          # G-Counter
│           ├── or-set.ts             # OR-Set
│           └── ... (全6ファイル)
│
├── demos/                             # インタラクティブデモ (3個)
│   ├── index.html                    # ランディングページ
│   ├── stats-playground/
│   │   └── index.html               # 統計計算機
│   └── crdt-demo/
│       └── index.html               # CRDTデモ
│
├── examples/                          # 使用例 (2個)
│   ├── stats-example.ts              # 統計ライブラリ使用例
│   └── crdt-example.ts               # CRDTライブラリ使用例
│
└── .github/workflows/                 # CI/CD
    ├── ci.yml                        # ビルド・テスト
    └── pages.yml                     # GitHub Pages自動デプロイ
```

---

## 🚀 よくあるシナリオ別ナビゲーション

### シナリオ1: プロジェクト全体を理解したい

**順番に読む**:
1. [README.md](README.md) - プロジェクト概要、スコア、主要成果
2. [PHASE4-COMPLETION-REPORT.md](_IMPROVEMENTS/PHASE4-COMPLETION-REPORT.md) - 最新の達成内容 (81→90点)
3. [INDEX.md](INDEX.md) - 全コンテンツのカタログ

**推定時間**: 15分

---

### シナリオ2: 特定のアルゴリズムを確認したい

#### 例: Binary Searchの証明を見たい

**直接移動**:
```bash
# ファイルを開く
cat backend-development/guides/algorithms/binary-search-proof.md

# または、エディタで開く
open backend-development/guides/algorithms/binary-search-proof.md
```

**確認すべき項目**:
- [ ] 数学的証明 (帰納法)
- [ ] 計算量解析 (O(log n))
- [ ] 実験結果 (4027× speedup)
- [ ] R²値 (0.9997)
- [ ] 引用論文 (4-6本)

**関連ファイル**:
- [アルゴリズム一覧](INDEX.md#1-アルゴリズム-algorithms---25個の証明)

---

### シナリオ3: 分散システムの証明を確認したい

#### 例: Paxosの証明を見たい

**直接移動**:
```bash
cat _IMPROVEMENTS/phase3/distributed-systems/02-paxos-consensus-proof.md
```

**確認すべき項目**:
- [ ] Safety保証 (100%)
- [ ] Agreement成功率 (98%)
- [ ] TLA+仕様との対応 (03-paxos-consensus.tla)
- [ ] 実験データ (n≥30, p<0.001)

**関連ファイル**:
- [TLA+ Paxos仕様](_IMPROVEMENTS/phase3/tla-plus/03-paxos-consensus.tla)
- [分散システム一覧](INDEX.md#2-分散システム-distributed-systems---5個の証明)

---

### シナリオ4: npmパッケージの使い方を確認したい

#### Statsパッケージ

**ドキュメント**:
```bash
cat packages/stats/README.md
```

**実装を見る**:
```bash
# t検定の実装
cat packages/stats/src/ttest.ts

# 使用例
cat examples/stats-example.ts
```

**デモを試す**:
- ブラウザで開く: [demos/stats-playground/index.html](demos/stats-playground/index.html)
- オンライン: https://gaku52.github.io/claude-code-skills/stats-playground/

#### CRDTパッケージ

**ドキュメント**:
```bash
cat packages/crdt/README.md
```

**実装を見る**:
```bash
# G-Counterの実装 (最も基本的なCRDT)
cat packages/crdt/src/g-counter.ts

# OR-Setの実装 (最も複雑なCRDT)
cat packages/crdt/src/or-set.ts

# 使用例
cat examples/crdt-example.ts
```

**デモを試す**:
- ブラウザで開く: [demos/crdt-demo/index.html](demos/crdt-demo/index.html)
- オンライン: https://gaku52.github.io/claude-code-skills/crdt-demo/

---

### シナリオ5: 統計手法・実験計画を確認したい

**メインドキュメント**:
```bash
cat _IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md
```

**確認すべき項目**:
- [ ] サンプルサイズ計算 (n≥30)
- [ ] t検定の実施方法
- [ ] Cohen's dの解釈
- [ ] 信頼区間の計算
- [ ] R²値の意味

**実行可能なテンプレート**:
```bash
# TypeScript実装を確認
cat _IMPROVEMENTS/phase3/experiment-templates/02-experiment-template.ts

# 実際に実行
npx tsx _IMPROVEMENTS/phase3/experiment-templates/02-experiment-template.ts
```

**レポート形式**:
```bash
cat _IMPROVEMENTS/phase3/experiment-templates/03-reporting-template.md
```

---

### シナリオ6: 形式検証 (TLA+) を確認したい

**入門ドキュメント**:
```bash
cat _IMPROVEMENTS/phase3/tla-plus/01-tla-plus-introduction.md
```

**TLA+仕様を見る**:
```bash
# Two-Phase Commit (最もシンプル)
cat _IMPROVEMENTS/phase3/tla-plus/02-two-phase-commit.tla

# Paxos (中級)
cat _IMPROVEMENTS/phase3/tla-plus/03-paxos-consensus.tla

# Raft (最も複雑)
cat _IMPROVEMENTS/phase3/tla-plus/04-raft-consensus.tla
```

**検証結果の確認**:
- Two-Phase Commit: 12,500 states
- Paxos: 50,000 states
- Raft: 90,000 states
- **合計**: 152,500+ states verified ✅

---

### シナリオ7: プロジェクトの歴史を確認したい

**フェーズ別レポート** (時系列順):

1. **Phase 1** (38→55点): 統計厳格化
   ```bash
   cat _IMPROVEMENTS/PHASE1-COMPLETION-REPORT.md
   ```

2. **Phase 2** (55→68点): 25個のアルゴリズム証明追加
   ```bash
   cat _IMPROVEMENTS/PHASE2-COMPLETION-REPORT.md
   ```

3. **Phase 3** (68→81点): 分散システム + TLA+
   ```bash
   cat _IMPROVEMENTS/PHASE3-COMPLETION-REPORT.md
   ```

4. **Phase 4** (81→90点): 実用化 (npm + デモ)
   ```bash
   cat _IMPROVEMENTS/PHASE4-COMPLETION-REPORT.md
   ```

**各フェーズで何を達成したかが明確に記録されています。**

---

### シナリオ8: メンテナンス方法を確認したい

**メンテナンスガイド**:
```bash
cat MAINTENANCE.md
```

**確認すべき項目**:
- [ ] 週次・月次・四半期・年次タスク
- [ ] 新しい論文の追加方法
- [ ] npmパッケージの更新手順
- [ ] 統計手法の追加プロセス
- [ ] CRDTの追加プロセス
- [ ] スコア更新のガイドライン
- [ ] 長期目標 (2026-2028)

---

## 📋 作者が確認すべきチェックリスト

### プロジェクト理解のチェック

**基本理解** (必須):
- [ ] プロジェクトの目的を説明できる
- [ ] 現在のスコア (90/100) の内訳を説明できる
- [ ] 34個の証明がどのカテゴリに分類されるか知っている
- [ ] 255+論文がどのように引用されているか理解している

**技術理解** (推奨):
- [ ] t検定とCohen's dの違いを説明できる
- [ ] PaxosとRaftの違いを説明できる
- [ ] CRDTの3つの性質 (結合的・可換的・冪等的) を説明できる
- [ ] TLA+が何のために使われているか理解している

**実装理解** (上級):
- [ ] Statsパッケージの主要APIを説明できる
- [ ] CRDTパッケージの4つの型を説明できる
- [ ] デモページの動作原理を説明できる
- [ ] CI/CDパイプラインの流れを説明できる

---

## 🔗 頻繁にアクセスするファイルのショートカット

### トップ10 (使用頻度順)

1. **[README.md](README.md)** - プロジェクト概要
2. **[INDEX.md](INDEX.md)** - 検索・索引
3. **[PHASE4-COMPLETION-REPORT.md](_IMPROVEMENTS/PHASE4-COMPLETION-REPORT.md)** - 最新成果
4. **[MAINTENANCE.md](MAINTENANCE.md)** - メンテナンス方法
5. **[packages/stats/README.md](packages/stats/README.md)** - Stats API
6. **[packages/crdt/README.md](packages/crdt/README.md)** - CRDT API
7. **[01-statistical-methodology.md](_IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md)** - 統計手法
8. **[02-paxos-consensus-proof.md](_IMPROVEMENTS/phase3/distributed-systems/02-paxos-consensus-proof.md)** - Paxos証明
9. **[binary-search-proof.md](backend-development/guides/algorithms/binary-search-proof.md)** - Binary Search
10. **[fft-proof.md](backend-development/guides/algorithms/fft-proof.md)** - FFT証明

---

## 🎓 学習パス別ナビゲーション

### パス1: 統計・実験計画を学びたい

1. [統計手法ガイド](_IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md) - 理論
2. [実験テンプレート](_IMPROVEMENTS/phase3/experiment-templates/02-experiment-template.ts) - 実装
3. [Statsパッケージ実装](packages/stats/src/ttest.ts) - 実際のコード
4. [Statsデモ](demos/stats-playground/index.html) - インタラクティブ体験

**推定時間**: 2時間

---

### パス2: アルゴリズムを学びたい

#### 初級
1. [Binary Search](backend-development/guides/algorithms/binary-search-proof.md)
2. [Merge Sort](backend-development/guides/algorithms/merge-sort-proof.md)

#### 中級
3. [Fenwick Tree](backend-development/guides/algorithms/fenwick-tree-proof.md)
4. [Dijkstra](backend-development/guides/algorithms/dijkstra-proof.md)

#### 上級
5. [FFT](backend-development/guides/algorithms/fft-proof.md)
6. [Strassen行列乗算](backend-development/guides/algorithms/strassen-proof.md)

**推定時間**: 5時間 (全部読む場合)

---

### パス3: 分散システムを学びたい

1. [CAP定理](_IMPROVEMENTS/phase3/distributed-systems/01-cap-theorem-proof.md) - 基礎理論
2. [Paxos](_IMPROVEMENTS/phase3/distributed-systems/02-paxos-consensus-proof.md) - 古典的合意形成
3. [Raft](_IMPROVEMENTS/phase3/distributed-systems/03-raft-consensus-proof.md) - モダンな合意形成
4. [CRDT](_IMPROVEMENTS/phase3/distributed-systems/05-crdt-proof.md) - 無矛盾複製
5. [CRDTパッケージ実装](packages/crdt/src/g-counter.ts) - 実装
6. [CRDTデモ](demos/crdt-demo/index.html) - 可視化

**推定時間**: 4時間

---

### パス4: 形式検証を学びたい

1. [TLA+入門](_IMPROVEMENTS/phase3/tla-plus/01-tla-plus-introduction.md)
2. [Two-Phase Commit仕様](_IMPROVEMENTS/phase3/tla-plus/02-two-phase-commit.tla) - 最もシンプル
3. [Paxos仕様](_IMPROVEMENTS/phase3/tla-plus/03-paxos-consensus.tla) - 中級
4. [Raft仕様](_IMPROVEMENTS/phase3/tla-plus/04-raft-consensus.tla) - 高度

**推定時間**: 3時間 (TLA+ Toolbox必要)

---

## 💡 効率的なナビゲーションのヒント

### ヒント1: プロジェクトルートから相対パスで指定

すべてのパスは `/Users/gaku/claude-code-skills/` からの相対パスです。

```bash
cd /Users/gaku/claude-code-skills
cat README.md
cat packages/stats/README.md
```

### ヒント2: INDEX.mdでキーワード検索

特定のトピックを探す場合:
```bash
# INDEX.mdを開く
open INDEX.md

# Cmd+F (Mac) / Ctrl+F (Win) で検索
# 例: "FFT", "Paxos", "t-test"
```

### ヒント3: GitHub上で見る

オンラインで参照する場合:
```
https://github.com/Gaku52/claude-code-skills/blob/main/[ファイルパス]
```

例:
- https://github.com/Gaku52/claude-code-skills/blob/main/README.md
- https://github.com/Gaku52/claude-code-skills/blob/main/packages/stats/README.md

### ヒント4: VSCodeのワークスペース検索

プロジェクト全体から検索:
```
Cmd+Shift+F (Mac) / Ctrl+Shift+F (Win)
```

キーワード例: "Cohen's d", "join-semilattice", "O(log n)"

---

## 📞 作者向けクイックレファレンス

### 誰かに説明する時に見るべきファイル

**「このプロジェクトは何ですか?」**
→ [README.md](README.md)

**「どんな証明がありますか?」**
→ [INDEX.md](INDEX.md)

**「統計はどうやっていますか?」**
→ [統計手法ガイド](_IMPROVEMENTS/phase3/experiment-templates/01-statistical-methodology.md)

**「実装はありますか?」**
→ [Statsパッケージ](packages/stats/README.md), [CRDTパッケージ](packages/crdt/README.md)

**「デモはありますか?」**
→ https://gaku52.github.io/claude-code-skills/

**「最新の成果は?」**
→ [Phase 4レポート](_IMPROVEMENTS/PHASE4-COMPLETION-REPORT.md)

**「メンテナンス方法は?」**
→ [MAINTENANCE.md](MAINTENANCE.md)

---

## 🎯 まとめ

このナビゲーションガイドを使えば:

✅ **作者が能動的にドキュメントにアクセスできる**
✅ **プロジェクト理解を示すことができる**
✅ **効率的にレビュー・メンテナンスできる**
✅ **他者に説明する際の参照元として使える**

**重要**: このガイドはClaude Code Skillsの自動参照を**補完**するものであり、作者自身の理解と能動的なアクセスを可能にします。

---

**最終更新**: 2026-01-04
**作成者**: Gaku
**目的**: 作者が直接ドキュメントにアクセスし、理解を示すため
