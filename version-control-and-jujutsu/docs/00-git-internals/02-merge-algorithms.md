# マージアルゴリズム

> Gitのマージ戦略（3-way merge, recursive, ort）とrebaseの内部動作を解説し、コンフリクト解決の原理とマージ戦略の選択基準を理解する。

## この章で学ぶこと

1. **3-way mergeの原理** — 共通祖先を用いたマージの基本アルゴリズム
2. **Gitのマージ戦略**（recursive, ort, octopus, ours）の違いと使い分け
3. **rebaseの内部動作** — cherry-pickの連鎖としてのrebaseとマージとの比較

---

## 1. 2-way merge vs 3-way merge

### 1.1 2-way mergeの限界

```
ファイルの状態:
  ブランチA:  Line 1 / Line 2-modified / Line 3
  ブランチB:  Line 1 / Line 2          / Line 3-modified

2-way merge（AとBだけ比較）:
  → Line 2 が違う。どちらが正しい？ 判断不可能
  → Line 3 が違う。どちらが正しい？ 判断不可能
  → 全ての差異がコンフリクトになる
```

### 1.2 3-way mergeの原理

```
共通祖先(Base): Line 1 / Line 2          / Line 3
ブランチA:      Line 1 / Line 2-modified / Line 3
ブランチB:      Line 1 / Line 2          / Line 3-modified

3-way merge（Base + A + B を比較）:
  Line 1: A=Base, B=Base → 変更なし → "Line 1"
  Line 2: A≠Base, B=Base → Aが変更  → "Line 2-modified"
  Line 3: A=Base, B≠Base → Bが変更  → "Line 3-modified"

結果: Line 1 / Line 2-modified / Line 3-modified
  → コンフリクトなし！
```

```
┌─────────────────────────────────────────────────────┐
│              3-way merge のアルゴリズム               │
│                                                     │
│           Base (共通祖先)                            │
│           /          \                              │
│          /            \                             │
│    Branch A        Branch B                         │
│          \            /                             │
│           \          /                              │
│        3-way merge判定                              │
│                                                     │
│  各行について:                                       │
│  ┌──────────┬──────────┬───────────────────────┐    │
│  │ A=Base?  │ B=Base?  │ 結果                  │    │
│  ├──────────┼──────────┼───────────────────────┤    │
│  │ Yes      │ Yes      │ Base（変更なし）      │    │
│  │ No       │ Yes      │ A を採用              │    │
│  │ Yes      │ No       │ B を採用              │    │
│  │ No       │ No       │ A=B なら採用、        │    │
│  │          │          │ A≠B ならコンフリクト  │    │
│  └──────────┴──────────┴───────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

---

## 2. 共通祖先（merge base）の特定

```bash
# 2つのブランチの共通祖先を特定
$ git merge-base main feature/auth
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# 複数の共通祖先がある場合（criss-cross merge）
$ git merge-base --all main feature/auth
a1b2c3d4...
f5e6d7c8...
```

```
単純なケース:
       o---o---o  feature
      /
 o---o---o---o  main
      ^
      merge base (1つ)

criss-cross merge（複数の共通祖先）:
       o---A---o  feature
      / \ / \
 o---o   X   o---?
      \ / \ /
       o---B---o  main
           ^
      merge base が A と B の2つ
```

---

## 3. Gitのマージ戦略

### 3.1 戦略の一覧と比較

| 戦略       | 対象       | 共通祖先の扱い              | 用途                       |
|------------|------------|-----------------------------|-----------------------------|
| `ort`      | 2ブランチ  | 仮想共通祖先を再帰的に構築  | デフォルト（Git 2.34+）     |
| `recursive`| 2ブランチ  | 仮想共通祖先を再帰的に構築  | 旧デフォルト（Git 2.33以前）|
| `resolve`  | 2ブランチ  | 1つだけ使用                 | 単純な場合                  |
| `octopus`  | 3+ブランチ | 各ペアの共通祖先             | 複数ブランチの一括マージ    |
| `ours`     | N ブランチ | 使用しない                   | 自分側の内容を強制採用      |
| `subtree`  | 2ブランチ  | サブツリー対応               | サブプロジェクト統合        |

### 3.2 ort戦略（Ostensibly Recursive's Twin）

Git 2.34以降のデフォルト。`recursive`の完全書き換え版。

```bash
# ort戦略の明示的な使用
$ git merge -s ort feature/auth

# ort戦略のオプション
$ git merge -X ours feature/auth      # コンフリクト時に自分側を優先
$ git merge -X theirs feature/auth    # コンフリクト時に相手側を優先
$ git merge -X patience feature/auth  # patience diffアルゴリズム使用
```

**ortがrecursiveより優れている点**:

| 項目             | recursive           | ort                  |
|------------------|---------------------|----------------------|
| パフォーマンス   | O(n^2)のケースあり  | 常にO(n log n)       |
| 一時ファイル     | 作業ディレクトリ使用| メモリ内で完結       |
| リネーム検出     | 遅い場合がある      | 高速化               |
| クリーンな実装   | 歴史的経緯で複雑    | ゼロから再設計       |

### 3.3 recursive戦略の再帰的マージ

criss-cross mergeで複数の共通祖先がある場合:

```
手順:
1. 複数の共通祖先 (A, B) を発見
2. A と B を（再帰的に）マージして仮想共通祖先 V を作成
3. V を base として通常の3-way mergeを実行

       o---A---o---o  feature
      / \ / \       \
 o---o   X   o      merge
      \ / \ /       /
       o---B---o---o  main

 1. merge-base(feature, main) = {A, B}
 2. V = merge(A, B)    ← 再帰的マージ
 3. result = 3-way-merge(V, feature, main)
```

### 3.4 fast-forwardマージ

```bash
# fast-forwardが可能な場合
#   main:    A---B
#   feature: A---B---C---D

$ git checkout main
$ git merge feature
# → mainのポインタをDに移動するだけ（新commitは作らない）

# fast-forwardを強制的にno-ffにする
$ git merge --no-ff feature
# → マージコミットを必ず作成する

# fast-forward限定（不可能ならエラー）
$ git merge --ff-only feature
```

```
┌──────────────────────────────────────────┐
│  fast-forward merge                      │
│                                          │
│  Before:                                 │
│  main ──→ A ── B                         │
│                  \                        │
│                   C ── D ←── feature      │
│                                          │
│  After (--ff):                           │
│  main ──────────────→ D ←── feature      │
│  (コミット履歴: A-B-C-D)                 │
│                                          │
│  After (--no-ff):                        │
│  main ──→ A ── B ──────── M             │
│                  \        /              │
│                   C ── D ←── feature     │
│  (Mはマージコミット)                     │
└──────────────────────────────────────────┘
```

---

## 4. コンフリクト解決の内部動作

### 4.1 ステージングエリアのステージ番号

コンフリクト中、インデックスには同一ファイルの3つのバージョンが格納される。

```bash
# コンフリクト状態のインデックスを確認
$ git ls-files -u
100644 abc123... 1	src/auth.js    # Stage 1: 共通祖先 (base)
100644 def456... 2	src/auth.js    # Stage 2: 自分側 (ours / HEAD)
100644 789abc... 3	src/auth.js    # Stage 3: 相手側 (theirs / MERGE_HEAD)

# 各ステージの内容を確認
$ git show :1:src/auth.js    # base
$ git show :2:src/auth.js    # ours
$ git show :3:src/auth.js    # theirs
```

### 4.2 コンフリクトマーカー

```javascript
// コンフリクトが発生したファイルの内容
function authenticate(user) {
<<<<<<< HEAD
  return bcrypt.compare(user.password, hash);
||||||| abc123 (merge.conflictStyle = zdiff3 の場合)
  return checkPassword(user.password);
=======
  return argon2.verify(hash, user.password);
>>>>>>> feature/auth
}
```

```bash
# コンフリクトスタイルの設定
$ git config merge.conflictStyle zdiff3
# zdiff3: base（共通祖先）の内容も表示される → 判断が容易に
```

### 4.3 rerere（Reuse Recorded Resolution）

```bash
# rerereを有効化
$ git config rerere.enabled true

# 動作の仕組み:
# 1. コンフリクトが発生 → コンフリクトのパターンを記録
# 2. 手動で解決 → 解決方法を .git/rr-cache/ に保存
# 3. 同じコンフリクトが再発 → 自動的に以前の解決を適用

# 記録された解決の確認
$ git rerere status
$ git rerere diff
```

---

## 5. rebaseの内部動作

### 5.1 rebaseの仕組み

```bash
$ git checkout feature
$ git rebase main
```

```
Before:
     C---D---E  feature (HEAD)
    /
A---B---F---G  main

rebaseの内部手順:
1. feature と main の共通祖先 B を特定
2. B..feature の各commit (C, D, E) のpatchを取得
3. main (G) の上にpatchを順次適用（cherry-pick）
4. feature refを最後のcommitに更新

After:
                C'---D'---E'  feature (HEAD)
               /
A---B---F---G  main

※ C', D', E' は新しいcommitオブジェクト（SHA-1が異なる）
※ 元の C, D, E はreflogから到達可能（GCまで保持）
```

### 5.2 rebase vs merge

```bash
# merge: 履歴を保存、マージコミット作成
$ git checkout main
$ git merge feature

# rebase: 履歴を直線化、マージコミットなし
$ git checkout feature
$ git rebase main
$ git checkout main
$ git merge feature  # fast-forward
```

| 項目             | merge                        | rebase                        |
|------------------|------------------------------|-------------------------------|
| 履歴の形状       | 分岐・合流が残る             | 直線的になる                  |
| コミットの同一性 | 元コミットのSHA-1を保持      | 新しいSHA-1に変わる           |
| コンフリクト解決 | 一度だけ                     | 各コミットごとに発生しうる    |
| 公開ブランチ     | 安全                         | 危険（push --force 必要）     |
| bisect適性       | マージコミットが邪魔な場合   | 直線履歴で二分探索が容易      |

---

## 6. リネーム検出

```bash
# Gitはリネームを明示的に記録しない
# マージ時にヒューリスティックでリネームを検出する

# リネーム検出の閾値（類似度）を設定
$ git merge -X rename-threshold=50 feature
# → 50%以上の類似度でリネームと判定

# diffでのリネーム検出
$ git diff --find-renames=50 HEAD~1
# rename src/old.js => src/new.js (85%)
```

```
┌──────────────────────────────────────────────────┐
│  リネーム検出の仕組み                             │
│                                                  │
│  Base:    src/auth.js (内容A)                    │
│  Ours:    src/auth.js (内容A')  ← 内容を修正    │
│  Theirs:  lib/auth.js (内容A)   ← ファイル移動  │
│                                                  │
│  検出フロー:                                      │
│  1. Base→Theirs で src/auth.js 削除、            │
│     lib/auth.js 追加を検出                        │
│  2. 内容の類似度を計算（A と A で100%一致）      │
│  3. "src/auth.js → lib/auth.js" のリネームと判定 │
│  4. Ours の修正を lib/auth.js に適用             │
│                                                  │
│  結果: lib/auth.js (内容A') ← 移動+修正の統合   │
└──────────────────────────────────────────────────┘
```

---

## 7. アンチパターン

### アンチパターン1: 公開済みブランチのrebase

```bash
# NG: mainやdevelopなど共有ブランチをrebase
$ git checkout main
$ git rebase feature
$ git push --force origin main
# → 他のメンバーのローカルmainと履歴が不整合
# → 他のメンバーが強制的にreset --hardする必要がある

# OK: 自分専用のfeatureブランチのみrebaseする
$ git checkout feature/my-work
$ git rebase main
$ git push --force-with-lease origin feature/my-work
# --force-with-lease: 他者のpushがあれば拒否される
```

**理由**: rebaseはコミットのSHA-1を変更する。共有ブランチのSHA-1が変わると、他の開発者のローカル履歴と矛盾が生じ、データ損失のリスクがある。

### アンチパターン2: マージ戦略`ours`の誤用

```bash
# NG: "ours"戦略で相手の変更を完全に無視
$ git merge -s ours feature/important-fix
# → feature/important-fix の変更が一切反映されない
# → 履歴上はマージ済みに見えるため、再マージも不可

# OK: 意図的に履歴を閉じる場合のみ使用
$ git merge -s ours legacy/deprecated-feature
# → 明確に "このブランチの内容は不要" という意思表示として使用
```

**理由**: `-s ours`はマージ「戦略」であり、コンフリクト時の`-X ours`（マージ「オプション」）とは全く異なる。前者は相手側の変更を完全に捨てる。

---

## 8. FAQ

### Q1. コンフリクトが発生した場合、マージを中断できるか？

**A1.** はい、`git merge --abort`でマージ前の状態に完全に復帰できます。rebase中のコンフリクトは`git rebase --abort`で中断できます。いずれの場合も、作業ディレクトリとインデックスがマージ/リベース前の状態に戻ります。

### Q2. octopusマージはどのような場面で使うのか？

**A2.** 3つ以上のブランチを同時にマージする場合に使います。典型的にはリリース準備時に複数のfeatureブランチを統合するケースです。ただし、コンフリクトが発生するとoctopusマージは自動的に中断されます。コンフリクトが予想される場合は個別にマージする方が安全です。

```bash
$ git merge feature/a feature/b feature/c
# → 自動的にoctopus戦略が選択される
```

### Q3. rebase中にコンフリクトが発生した場合の対処法は？

**A3.** rebaseは各コミットを順次適用するため、コミットごとにコンフリクトが発生しえます。

```bash
# 1. コンフリクトを手動で解決
$ vim src/conflicted-file.js

# 2. 解決したファイルをステージ
$ git add src/conflicted-file.js

# 3. rebaseを続行
$ git rebase --continue

# または、このコミットをスキップ
$ git rebase --skip

# または、rebase全体を中断
$ git rebase --abort
```

---

## まとめ

| 概念                | 要点                                                          |
|---------------------|---------------------------------------------------------------|
| 3-way merge         | 共通祖先を基準に各行の変更元を判定、2-way mergeより賢い       |
| merge base          | 2ブランチの最近共通祖先、criss-crossでは複数存在しうる        |
| ort戦略             | Git 2.34+のデフォルト、recursive の高速・安定な書き換え版      |
| fast-forward        | ブランチポインタの移動のみ、`--no-ff`でマージコミット強制     |
| コンフリクト        | インデックスのStage 1/2/3で3バージョンを管理                  |
| rebase              | cherry-pickの連鎖、履歴の直線化、SHA-1は変化する              |
| rerere              | コンフリクト解決を記録・再利用する仕組み                      |

---

## 次に読むべきガイド

- [インタラクティブRebase](../01-advanced-git/00-interactive-rebase.md) — squash、fixup、rewordの実践
- [Packfile/GC](./03-packfile-gc.md) — マージ後のオブジェクト最適化
- [bisect/blame](../01-advanced-git/02-bisect-blame.md) — マージ履歴上でのバグ特定

---

## 参考文献

1. **Pro Git Book** — "Basic Branching and Merging" https://git-scm.com/book/en/v2/Git-Branching-Basic-Branching-and-Merging
2. **Elijah Newren** — "Git's new default merge strategy: ort" https://github.blog/2021-08-16-highlights-from-git-2-33/#merge-ort
3. **Git公式ドキュメント** — `git-merge`, `git-rebase`, `git-rerere` https://git-scm.com/docs
4. **A Formal Investigation of Diff3** — Sanjeev Khanna, Keshav Kunal, Benjamin C. Pierce https://www.cis.upenn.edu/~bcpierce/papers/diff3-short.pdf
