# マージアルゴリズム

> Gitのマージ戦略（3-way merge, recursive, ort）とrebaseの内部動作を解説し、コンフリクト解決の原理とマージ戦略の選択基準を理解する。

## この章で学ぶこと

1. **3-way mergeの原理** — 共通祖先を用いたマージの基本アルゴリズム
2. **Gitのマージ戦略**（recursive, ort, octopus, ours）の違いと使い分け
3. **rebaseの内部動作** — cherry-pickの連鎖としてのrebaseとマージとの比較
4. **コンフリクト解決の詳細** — ステージ番号、マーカー、rerere、手動解決パターン
5. **リネーム検出とdiffアルゴリズム** — マージ品質に影響する検出ロジック

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

2-way mergeは「2つのバージョンを直接比較する」だけのアルゴリズムである。差異が見つかった場合、どちらのバージョンが「意図的な変更」でどちらが「元のまま」なのかを判定できないため、全ての差異をコンフリクトとして報告する。これはユーザーにとって非常に煩雑であり、実用的ではない。

実際のパッチツール（`diff` + `patch`）やエディタのマージ機能では、この2-way比較しかできない場合がある。その場合、開発者が全ての差異を手動で判断する必要がある。

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

### 1.3 3-way mergeの詳細な判定ロジック

3-way mergeでは、ファイルを行（hunk）単位に分割し、各行について共通祖先との差分を計算する。判定ロジックの詳細は以下の通り。

```
┌───────────────────────────────────────────────────────┐
│  3-way merge の詳細判定フロー                          │
│                                                       │
│  入力: Base, Ours (A), Theirs (B) の3つのファイル      │
│                                                       │
│  Step 1: diff(Base, A) → patchA を生成               │
│  Step 2: diff(Base, B) → patchB を生成               │
│                                                       │
│  Step 3: patchA と patchB の各hunkを分類              │
│                                                       │
│  Case 1: hunkがpatchAにのみ存在                       │
│    → Aの変更を採用                                    │
│                                                       │
│  Case 2: hunkがpatchBにのみ存在                       │
│    → Bの変更を採用                                    │
│                                                       │
│  Case 3: 同じhunkがpatchA, patchBの両方に存在         │
│    3a: 変更内容が同一 → 片方を採用（重複変更）        │
│    3b: 変更内容が異なるが行範囲が重ならない → 両方採用 │
│    3c: 変更内容が異なり行範囲が重なる → コンフリクト  │
│                                                       │
│  Case 4: どちらのpatchにも存在しない行                 │
│    → Baseの内容をそのまま保持                         │
└───────────────────────────────────────────────────────┘
```

### 1.4 テキスト行以外の3-way merge

3-way mergeは行単位の比較が基本だが、以下のケースでは特別な処理が必要になる。

```bash
# バイナリファイルの場合
$ git merge feature
# warning: Cannot merge binary files: assets/logo.png
# → バイナリファイルは行分割できないため、常にコンフリクト扱い
# → ours/theirsのどちらかを手動で選択する

# バイナリのコンフリクト解決
$ git checkout --ours assets/logo.png     # 自分側を採用
$ git checkout --theirs assets/logo.png   # 相手側を採用
$ git add assets/logo.png

# カスタムマージドライバーの設定
$ cat .gitattributes
*.psd merge=binary        # バイナリとして扱う（常にコンフリクト）
*.lock merge=ours         # 自分側を常に採用（lockファイル）
*.pbxproj merge=union     # 両方の変更を結合（Xcodeプロジェクト）
```

```bash
# マージドライバーの定義
$ git config merge.union.driver "union-merge %O %A %B"
$ git config merge.custom.driver "custom-merge-tool %O %A %B %P"
# %O = 共通祖先（base）
# %A = 自分側（ours）
# %B = 相手側（theirs）
# %P = ファイルパス
```

---

## 2. 共通祖先（merge base）の特定

### 2.1 基本的なmerge base

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

### 2.2 merge baseの計算アルゴリズム

merge baseの計算はDAG（有向非巡回グラフ）上のLCA（Lowest Common Ancestor、最近共通祖先）問題に帰着される。

```
┌────────────────────────────────────────────────────────┐
│  LCA（Lowest Common Ancestor）の計算                    │
│                                                        │
│  Git DAG:                                              │
│      A---B---C---D  main                               │
│       \         /                                      │
│        E---F---G    feature                            │
│                                                        │
│  LCA(D, G) を計算:                                     │
│  1. D の祖先集合: {D, C, B, A, G, F, E}               │
│  2. G の祖先集合: {G, F, E, A}                         │
│  3. 共通祖先: {A, G, F, E} ∩ {D, C, B, A, G, F, E}   │
│  4. 最も「低い」（最新の）共通祖先 = A                  │
│                                                        │
│  実装: BFSで両方から同時に祖先を辿り、最初の合流点     │
└────────────────────────────────────────────────────────┘
```

```bash
# merge baseの詳細な確認
$ git merge-base --is-ancestor A B
# exit 0: AはBの祖先
# exit 1: AはBの祖先ではない

# fork-pointの検出（reflogベース）
$ git merge-base --fork-point main feature
# → featureがmainから分岐した正確なポイントを検出
# → reflogを使うため、mainがrebaseされた場合にも対応

# octopus mergeのmerge base
$ git merge-base --octopus branch-a branch-b branch-c
# → 3ブランチ共通の祖先を計算
```

### 2.3 criss-cross mergeの詳細

criss-cross mergeは複雑な履歴パターンで、複数の共通祖先が存在する状況を指す。

```
criss-cross merge の発生パターン:

Step 1: 初期状態
  A---B  main
   \
    C  feature

Step 2: featureをmainにマージ（M1）
  A---B---M1  main
   \     /
    C---+     feature

Step 3: mainをfeatureにマージ（M2）
  A---B---M1  main
   \     / \
    C---+---M2  feature

Step 4: 両ブランチが独立に進行
  A---B---M1---D  main
   \     / \
    C---+---M2---E  feature

Step 5: mainとfeatureをマージしたい
  merge-base(D, E) = {M1, M2} ← 2つの共通祖先！

対処法（recursive/ort戦略）:
  1. M1とM2を仮想的にマージしてV（仮想共通祖先）を作成
  2. Vを使って3-way mergeを実行
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
| 並列処理         | 非対応              | 部分的に並列化可能   |
| メモリ使用量     | ディスクI/O多い     | メモリ上で効率的     |

```bash
# ortの性能改善を示す例（大規模リポジトリ）
# Linux kernelリポジトリでのベンチマーク（参考値）
# recursive: 25.3秒
# ort:        4.1秒  (約6倍高速)

# ortが作業ディレクトリを使用しないことの確認
$ git merge -s ort feature/auth
# → マージ中に作業ディレクトリのファイルが一時的に変更されない
# → 他のプロセスが同時にファイルを参照しても安全
```

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

```bash
# recursive戦略の再帰深さ制限
# デフォルトでは再帰の深さに制限なし（実用上は問題にならない）
# 理論上は無限ループしない（DAGなので祖先は必ず有限）

# recursiveで仮想共通祖先の作成時にコンフリクトが発生する場合
# → コンフリクトマーカーを含んだ状態で仮想共通祖先を作成
# → 最終マージのコンフリクト解決結果に影響する可能性がある
```

### 3.4 resolve戦略

```bash
# resolve戦略: 複数の共通祖先がある場合、1つだけを選択
$ git merge -s resolve feature/auth

# 使用場面:
# - recursive/ortでコンフリクトが発生し、resolveだと成功する場合がある
# - criss-cross mergeの状況で、特定の共通祖先を使いたい場合
# - 非常にまれだが、デバッグ目的で使用することもある
```

### 3.5 octopus戦略

```bash
# 3つ以上のブランチを同時にマージ
$ git merge feature/a feature/b feature/c
# → 自動的にoctopus戦略が選択される

# octopus戦略の制約
# - コンフリクトが発生すると自動的に中断される
# - 手動コンフリクト解決は不可能
# - コンフリクトが予想される場合は個別にマージすべき
```

```
┌────────────────────────────────────────────────────┐
│  octopus merge の動作                               │
│                                                    │
│  入力: main, feature/a, feature/b, feature/c       │
│                                                    │
│  Step 1: main + feature/a をマージ → 中間結果1     │
│  Step 2: 中間結果1 + feature/b をマージ → 中間結果2│
│  Step 3: 中間結果2 + feature/c をマージ → 最終結果 │
│                                                    │
│  いずれかのステップでコンフリクト → 全体を中断     │
│                                                    │
│  結果のcommit:                                     │
│      feature/a  feature/b  feature/c               │
│           \         |         /                    │
│            \        |        /                     │
│             o───────o───────o                      │
│                     |                              │
│                   merge                            │
│                     |                              │
│                   main                             │
│  → 3つの親を持つマージcommit                       │
└────────────────────────────────────────────────────┘
```

### 3.6 ours戦略（マージ戦略としてのours）

```bash
# -s ours: 戦略としてのours（相手の変更を完全に無視）
$ git merge -s ours feature/deprecated
# → 自分側（HEAD）の内容をそのまま保持
# → feature/deprecatedの変更は一切反映されない
# → しかし、履歴上はマージ済みとして記録される

# 主な用途:
# 1. 不要なブランチの履歴を閉じる
# 2. リリースブランチの不要な変更をスキップ
# 3. 意図的に特定のブランチの変更を拒否する

# 重要: -s ours と -X ours は全く異なる
$ git merge -s ours feature    # 戦略: 相手の変更を全て無視
$ git merge -X ours feature    # オプション: コンフリクト時のみ自分側を優先
```

### 3.7 subtree戦略

```bash
# subtree戦略: サブプロジェクトの統合
$ git merge -s subtree library-repo/main

# 使用例: 別リポジトリのコードをサブディレクトリに統合
$ git remote add library-repo https://github.com/example/lib.git
$ git fetch library-repo
$ git merge -s subtree --allow-unrelated-histories library-repo/main

# subtree戦略の動作:
# 1. 相手側のファイルツリーが自分側のサブディレクトリに対応するか自動検出
# 2. パスの対応関係を調整してから3-way mergeを実行
# 3. git-subtreeコマンドの内部でも使用される
```

### 3.8 fast-forwardマージ

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

```bash
# チームでの推奨設定: --no-ffを強制
$ git config merge.ff false
# → 全てのマージでマージコミットが作成される
# → 「いつマージされたか」が履歴から明確にわかる

# 個人ブランチでは--ff-onlyを使い、rebase + ff-onlyの運用
$ git config pull.ff only
# → pullがfast-forwardできない場合はエラー（rebaseを促す）
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

# ステージ番号の意味
# Stage 0: 通常のファイル（コンフリクトなし）
# Stage 1: Base（共通祖先）
# Stage 2: Ours（HEAD側）
# Stage 3: Theirs（MERGE_HEAD側）
```

```
┌───────────────────────────────────────────────────────┐
│  コンフリクト時のインデックス状態                       │
│                                                       │
│  通常時:                                              │
│  ┌────────┬──────────┬──────────────┐                │
│  │ Stage  │ SHA-1    │ ファイル名    │                │
│  ├────────┼──────────┼──────────────┤                │
│  │ 0      │ abc123.. │ src/auth.js  │                │
│  │ 0      │ def456.. │ src/utils.js │                │
│  └────────┴──────────┴──────────────┘                │
│                                                       │
│  コンフリクト時:                                       │
│  ┌────────┬──────────┬──────────────┐                │
│  │ Stage  │ SHA-1    │ ファイル名    │                │
│  ├────────┼──────────┼──────────────┤                │
│  │ 1      │ 111aaa.. │ src/auth.js  │ ← base        │
│  │ 2      │ 222bbb.. │ src/auth.js  │ ← ours        │
│  │ 3      │ 333ccc.. │ src/auth.js  │ ← theirs      │
│  │ 0      │ def456.. │ src/utils.js │ ← 正常        │
│  └────────┴──────────┴──────────────┘                │
│                                                       │
│  git add src/auth.js で:                              │
│  Stage 1, 2, 3 が削除され、Stage 0 に解決版が格納    │
└───────────────────────────────────────────────────────┘
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

# 利用可能なコンフリクトスタイル
# merge: デフォルト。ours/theirsの2つだけ表示
# diff3: base + ours + theirsの3つを表示
# zdiff3: diff3の改良版。baseとours/theirsの共通部分を省略（Git 2.35+）
```

```
┌────────────────────────────────────────────────────────┐
│  コンフリクトスタイルの比較                              │
│                                                        │
│  merge (デフォルト):                                   │
│  <<<<<<< HEAD                                          │
│    return bcrypt.compare(user.password, hash);         │
│  =======                                               │
│    return argon2.verify(hash, user.password);          │
│  >>>>>>> feature/auth                                  │
│  → baseが見えないので「何を変えたか」の判断が困難      │
│                                                        │
│  diff3:                                                │
│  <<<<<<< HEAD                                          │
│    return bcrypt.compare(user.password, hash);         │
│  ||||||| merged common ancestors                       │
│    return checkPassword(user.password);                │
│  =======                                               │
│    return argon2.verify(hash, user.password);          │
│  >>>>>>> feature/auth                                  │
│  → baseが見えるので変更の意図が明確                    │
│                                                        │
│  zdiff3 (推奨):                                        │
│  <<<<<<< HEAD                                          │
│    return bcrypt.compare(user.password, hash);         │
│  ||||||| abc123                                        │
│    return checkPassword(user.password);                │
│  =======                                               │
│    return argon2.verify(hash, user.password);          │
│  >>>>>>> feature/auth                                  │
│  → diff3と同様だが、共通行を省略して見やすい           │
└────────────────────────────────────────────────────────┘
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

# rerereのキャッシュを手動管理
$ git rerere forget <pathspec>       # 特定ファイルの記録を削除
$ git rerere gc                       # 古い記録を削除
```

```bash
# rerereの具体的な使用シナリオ

# シナリオ: featureブランチをmainにrebaseするたびに同じコンフリクトが発生

# 1回目: 手動で解決
$ git rebase main
# CONFLICT! src/config.js
$ vim src/config.js       # 手動で解決
$ git add src/config.js
$ git rebase --continue
# → rerereが解決パターンを記録

# 2回目以降: 自動で解決
$ git rebase main
# CONFLICT! src/config.js
# Resolved 'src/config.js' using previous resolution.
$ git add src/config.js    # rerereの解決結果を確認して追加
$ git rebase --continue

# rerereキャッシュの中身
$ ls .git/rr-cache/
abc123def456.../
├── preimage    # コンフリクトの状態（マーカー付き）
└── postimage   # 解決後の状態
```

### 4.4 コンフリクト解決のベストプラクティス

```bash
# コンフリクトの確認
$ git diff --name-only --diff-filter=U
# → コンフリクトしているファイルの一覧

# 各ファイルのコンフリクト箇所をカウント
$ grep -c "<<<<<<< HEAD" src/auth.js
3

# ours/theirsで一括解決
$ git checkout --ours src/auth.js      # 自分側で一括解決
$ git checkout --theirs src/auth.js    # 相手側で一括解決
$ git add src/auth.js

# merge-toolを使った対話的解決
$ git mergetool
# → 設定されたマージツール（vimdiff, meld, kdiff3等）が起動

# mergeツールの設定
$ git config merge.tool vimdiff
$ git config mergetool.vimdiff.layout "LOCAL,BASE,REMOTE / MERGED"
$ git config mergetool.keepBackup false
# keepBackup=false: .origファイルを作成しない
```

```bash
# マージの中断と再開
$ git merge --abort         # マージを完全に中断（マージ前に戻る）
$ git merge --quit          # マージを中断（作業ディレクトリの変更は保持）
$ git merge --continue      # コンフリクト解決後にマージを続行

# rebase中のコンフリクト
$ git rebase --abort        # rebaseを完全に中断
$ git rebase --skip         # 現在のコミットをスキップして続行
$ git rebase --continue     # コンフリクト解決後にrebaseを続行
```

### 4.5 複雑なコンフリクト解決パターン

```bash
# パターン1: 両方の変更を取り込む（手動マージ）
# Before:
# <<<<<<< HEAD
#   validateEmail(email);
#   validatePassword(password);
# =======
#   validateEmail(email);
#   sanitizeInput(email);
# >>>>>>> feature/security

# After:（両方の変更を採用）
#   validateEmail(email);
#   validatePassword(password);
#   sanitizeInput(email);

# パターン2: 構造的なコンフリクト（関数の並び順が変わった場合）
# → diffアルゴリズムの変更で改善できることがある
$ git merge -X diff-algorithm=histogram feature
$ git merge -X diff-algorithm=patience feature

# パターン3: 大量のコンフリクトを効率的に解決
# コンフリクトファイルの一覧とカウント
$ git diff --name-only --diff-filter=U | wc -l
42

# パターンに基づく一括解決
$ git diff --name-only --diff-filter=U | xargs -I{} git checkout --theirs {}
$ git diff --name-only --diff-filter=U | xargs git add
# → 全てtheirsで解決（内容を後で確認する前提）
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

### 5.2 rebaseの内部実装（cherry-pickの連鎖）

```bash
# rebase の内部動作は以下のcherry-pickと同等
$ git checkout main                # mainに移動
$ git cherry-pick C               # C を適用 → C'
$ git cherry-pick D               # D を適用 → D'
$ git cherry-pick E               # E を適用 → E'
$ git branch -f feature HEAD      # featureを更新
$ git checkout feature             # featureに移動

# ただし実際のrebaseは:
# 1. detached HEAD状態で実行
# 2. ORIG_HEAD にrebase前のHEADを保存
# 3. .git/rebase-merge/ または .git/rebase-apply/ に状態を保存
# 4. 全cherry-pick完了後にref更新
```

```
┌───────────────────────────────────────────────────────┐
│  rebase中の.gitディレクトリの状態                       │
│                                                       │
│  .git/rebase-merge/                                   │
│  ├── head-name     ← "refs/heads/feature"             │
│  ├── onto          ← rebase先のcommit SHA-1           │
│  ├── orig-head     ← rebase前のHEAD SHA-1             │
│  ├── msgnum        ← 現在処理中のcommit番号           │
│  ├── end           ← 処理するcommitの総数             │
│  ├── interactive   ← インタラクティブモードのフラグ    │
│  └── done          ← 処理済みのコマンド一覧           │
│                                                       │
│  これらのファイルがrebase中断・再開の状態管理に使用    │
│  rebase --abort でこのディレクトリが削除される         │
└───────────────────────────────────────────────────────┘
```

### 5.3 rebaseとcherry-pickの違い

```bash
# cherry-pick: 特定のコミットを現在のブランチに適用
$ git cherry-pick abc123
# → abc123の変更（差分）を現在のHEADに適用
# → 新しいcommitが作成される（元のcommitのメッセージを引き継ぐ）
# → cherry-pick元と先で同じ変更が別のcommitとして存在する

# rebase: ブランチ全体を移動
$ git rebase main
# → 内部的にはcherry-pickの連鎖だが、ブランチrefも更新される
# → 元のcommitへの参照はreflogのみ
```

```bash
# cherry-pickの詳細オプション
$ git cherry-pick abc123 --no-commit
# → 変更を適用するがcommitは作成しない（作業ディレクトリにのみ反映）

$ git cherry-pick abc123..def456
# → 範囲指定で複数のcommitをcherry-pick

$ git cherry-pick -x abc123
# → コミットメッセージに "(cherry picked from commit abc123)" を追記
# → 追跡に便利

$ git cherry-pick -m 1 MERGE_COMMIT
# → マージコミットをcherry-pick（-m 1 で第一親をbaseに使用）
```

### 5.4 rebase vs merge

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
| undo可能性       | git revertで取り消せる       | reflogからの復元が必要        |
| 追跡性           | マージコミットがマイルストーン| 履歴が平坦で把握しにくい場合も|

### 5.5 rebase --onto

```bash
# --onto: より柔軟なrebase先の指定
$ git rebase --onto NEW_BASE OLD_BASE BRANCH

# 使用例1: featureブランチの一部だけを別のブランチに移動
# Before:
#     C---D---E  feature (HEAD)
#    /
# A---B---F  main
#        \
#         G---H  release

# featureのD, Eだけをreleaseに移動したい
$ git rebase --onto release C feature
# → D'---E' がreleaseの上に作成される

# 使用例2: マージ済みのcommitをスキップしてrebase
# Before:
#     C---D---E---F  feature
#    /       \
# A---B-------M  main (Dがマージ済み)

# D を除いて feature を main にrebase
$ git rebase --onto main D feature
# → C'---E'---F' がmainの上に作成される
```

```
┌────────────────────────────────────────────────────────┐
│  rebase --onto の動作図                                 │
│                                                        │
│  git rebase --onto NEW_BASE OLD_BASE BRANCH            │
│                                                        │
│  1. OLD_BASE..BRANCH の範囲のcommitを抽出              │
│  2. NEW_BASE の上にcherry-pick                         │
│  3. BRANCH refを更新                                   │
│                                                        │
│  例: git rebase --onto release C feature               │
│                                                        │
│  Before:                                               │
│      C---D---E  feature                                │
│     /                                                  │
│  A---B---F  main                                       │
│         \                                              │
│          G---H  release                                │
│                                                        │
│  抽出範囲: C..feature = {D, E}                         │
│  適用先: release (H)                                   │
│                                                        │
│  After:                                                │
│      C  (orphaned)                                     │
│     /                                                  │
│  A---B---F  main                                       │
│         \                                              │
│          G---H  release                                │
│                \                                       │
│                 D'---E'  feature                        │
└────────────────────────────────────────────────────────┘
```

---

## 6. diffアルゴリズム

マージの品質はdiffアルゴリズムに大きく依存する。Gitは複数のdiffアルゴリズムをサポートしている。

### 6.1 利用可能なアルゴリズム

```bash
# diffアルゴリズムの指定
$ git diff --diff-algorithm=myers      # デフォルト（Myers）
$ git diff --diff-algorithm=minimal    # 最小差分（遅いが正確）
$ git diff --diff-algorithm=patience   # patience diff
$ git diff --diff-algorithm=histogram  # histogram diff

# 永続設定
$ git config diff.algorithm histogram
```

| アルゴリズム | 特徴                                              | 適用場面                   |
|-------------|---------------------------------------------------|----------------------------|
| `myers`     | デフォルト、高速、LCS（最長共通部分列）ベース      | 一般的な用途               |
| `minimal`   | 最小差分を保証、低速                               | 差分の正確さが重要な場合   |
| `patience`  | ユニークな行をアンカーにして差分計算               | 構造的な変更が多い場合     |
| `histogram` | patience改良版、繰り返し行に強い                   | 推奨（多くの場面で高品質） |

### 6.2 patience diffの仕組み

```
┌────────────────────────────────────────────────────────┐
│  patience diff のアルゴリズム                            │
│                                                        │
│  1. 両ファイルでユニーク（1回だけ出現）な行を抽出      │
│  2. ユニーク行同士のLCS（最長共通部分列）を計算         │
│  3. LCSをアンカーとして差分の「骨格」を決定            │
│  4. アンカー間の領域をMyers diffで処理                 │
│                                                        │
│  メリット:                                             │
│  - 関数の移動や並べ替えに強い                          │
│  - 空行やブレースだけの行に引きずられない              │
│  - 意味的に正しい差分が得られやすい                    │
│                                                        │
│  例:                                                   │
│  File A:              File B:                          │
│  function foo() {     function bar() {                 │
│    return 1;            return 42;                     │
│  }                    }                                │
│  function bar() {     function foo() {                 │
│    return 42;           return 1;                      │
│  }                    }                                │
│                                                        │
│  Myers: "}" の位置で差分がずれる可能性                  │
│  Patience: 関数名がユニークなのでアンカーになる         │
│  → 「foo/barの順序が入れ替わった」と正しく検出         │
└────────────────────────────────────────────────────────┘
```

---

## 7. リネーム検出

```bash
# Gitはリネームを明示的に記録しない
# マージ時にヒューリスティックでリネームを検出する

# リネーム検出の閾値（類似度）を設定
$ git merge -X rename-threshold=50 feature
# → 50%以上の類似度でリネームと判定

# diffでのリネーム検出
$ git diff --find-renames=50 HEAD~1
# rename src/old.js => src/new.js (85%)

# コピー検出も可能
$ git diff --find-copies --find-copies-harder HEAD~1
# copy src/template.js => src/new-page.js (72%)
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

### 7.1 リネーム検出のパフォーマンス

```bash
# リネーム検出の制限設定
$ git config merge.renameLimit 10000
# デフォルト: 7000
# → 追加+削除されたファイルの組み合わせ数の上限
# → 超える場合はリネーム検出がスキップされる

$ git config diff.renameLimit 10000
# diff時のリネーム検出制限

# ort戦略でのリネーム検出の高速化
# ort は以下の最適化を行う:
# 1. ディレクトリリネームの検出（ファイル単位ではなくディレクトリ単位）
# 2. 前回のマージ結果のキャッシュ
# 3. 不要な類似度計算のスキップ
```

### 7.2 ディレクトリリネームの検出

```bash
# ort戦略ではディレクトリリネームも検出する

# 例:
# Base: src/components/auth/login.js, src/components/auth/register.js
# Ours: src/modules/auth/login.js, src/modules/auth/register.js (ディレクトリ移動)
# Theirs: src/components/auth/login.js, src/components/auth/register.js,
#          src/components/auth/forgot-password.js (新ファイル追加)

# ort戦略の結果:
# → src/components/auth/ → src/modules/auth/ のディレクトリリネームを検出
# → Theirsの新ファイルも src/modules/auth/forgot-password.js に配置
# → 手動でファイルを移動する必要がない
```

---

## 8. マージの高度なオプション

### 8.1 マージオプション（-X）の一覧

```bash
# コンフリクト解決のオプション
$ git merge -X ours feature         # コンフリクト時に自分側を優先
$ git merge -X theirs feature       # コンフリクト時に相手側を優先

# 空白の扱い
$ git merge -X ignore-space-change feature    # 空白の量の変更を無視
$ git merge -X ignore-all-space feature       # 全ての空白を無視
$ git merge -X ignore-space-at-eol feature    # 行末空白を無視

# リネーム
$ git merge -X rename-threshold=40 feature    # リネーム検出閾値を変更

# diffアルゴリズム
$ git merge -X diff-algorithm=histogram feature

# サブツリー
$ git merge -X subtree=path/to/dir feature    # サブツリーパスの指定

# find-renames（リネーム検出の詳細制御）
$ git merge -X find-renames=30 feature        # 30%一致でリネーム判定
```

### 8.2 --no-commitと--squash

```bash
# --no-commit: マージ結果をコミットせずにステージに留める
$ git merge --no-commit feature
# → コンフリクトがなくてもコミットしない
# → 内容を確認・修正してから手動でcommitできる
$ git diff --staged   # マージ結果を確認
$ git commit -m "merge: feature branch with modifications"

# --squash: マージ結果を1つのcommitにまとめる
$ git merge --squash feature
# → featureの全変更を作業ディレクトリとインデックスに適用
# → マージcommitではなく通常のcommitとして記録
# → 親はHEADのみ（featureブランチへの参照なし）
$ git commit -m "feat: squash merge of feature branch"

# 注意: --squashはfeatureブランチを「マージ済み」として記録しない
# → 同じfeatureブランチを再度マージしようとするとコンフリクトする
# → featureブランチは使い終わったら削除すべき
```

---

## 9. アンチパターン

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

### アンチパターン3: コンフリクトを理解せずに解決する

```bash
# NG: コンフリクトの内容を確認せずに一括解決
$ git checkout --theirs .
$ git add .
$ git commit
# → 自分の変更が全て失われる可能性がある
# → テストもせずにマージ完了としてしまう

# OK: 各コンフリクトを個別に確認して解決
$ git diff --name-only --diff-filter=U    # コンフリクトファイル一覧
$ git show :1:src/auth.js > /tmp/base.js  # base版を確認
$ git show :2:src/auth.js > /tmp/ours.js  # ours版を確認
$ git show :3:src/auth.js > /tmp/theirs.js # theirs版を確認
# → 3つのバージョンを比較して適切に解決
$ git mergetool src/auth.js               # マージツールで解決
```

### アンチパターン4: マージコミットメッセージのデフォルト使用

```bash
# NG: デフォルトのマージメッセージをそのまま使用
$ git merge feature
# "Merge branch 'feature' into main"  ← 情報が少なすぎる

# OK: 意味のあるマージメッセージを記述
$ git merge --no-ff feature -m "merge: feature/auth - OAuth2認証の追加

- Google/GitHubプロバイダー対応
- セッション管理の統合
- 既存のパスワード認証との互換性を維持

Closes #123"
```

### アンチパターン5: rebase中にpushしてしまう

```bash
# NG: rebaseが完了する前にpush
$ git rebase main
# コンフリクト発生...
$ git push origin feature   # ← まだrebase途中
# → rebase途中の不完全な状態がリモートに残る

# OK: rebaseを完了してからpush
$ git rebase main
# コンフリクト解決...
$ git rebase --continue
# 全コミットの適用完了
$ git push --force-with-lease origin feature
```

---

## 10. FAQ

### Q1. コンフリクトが発生した場合、マージを中断できるか？

**A1.** はい、`git merge --abort`でマージ前の状態に完全に復帰できます。rebase中のコンフリクトは`git rebase --abort`で中断できます。いずれの場合も、作業ディレクトリとインデックスがマージ/リベース前の状態に戻ります。

```bash
# マージの中断
$ git merge --abort

# rebaseの中断
$ git rebase --abort

# cherry-pickの中断
$ git cherry-pick --abort
```

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

### Q4. merge.conflictStyleは何を設定すべきか？

**A4.** **zdiff3を推奨します**（Git 2.35以降）。共通祖先の内容が表示されるため、「何が変更されたか」を理解しやすく、正確なコンフリクト解決が可能になります。

```bash
$ git config --global merge.conflictStyle zdiff3
```

diff3/zdiff3では、コンフリクトマーカー内に共通祖先（base）の内容が`|||||||`区切りで表示されます。これにより、「ours側は何を変えたか」「theirs側は何を変えたか」が一目瞭然になります。

### Q5. rebaseとmergeのどちらを使うべきか？

**A5.** チームの方針に依存しますが、一般的なガイドラインは以下の通りです。

| 状況                           | 推奨                    | 理由                                        |
|--------------------------------|------------------------|---------------------------------------------|
| 個人のfeatureブランチ          | rebase                 | 直線的な履歴で読みやすい                     |
| 共有ブランチ（main, develop）  | merge                  | 他の開発者に影響を与えない                   |
| 長期ブランチのmainへの追従     | merge（またはrebase）  | コンフリクト頻度が高いならmergeが安全        |
| PRマージ                       | --no-ff merge          | マージポイントが明確                         |
| 小さな修正の統合               | rebase + ff merge      | 履歴が綺麗                                   |

### Q6. マージコミットを後から取り消すには？

**A6.** `git revert -m 1 MERGE_COMMIT`で取り消せます。`-m 1`は「第一親（mainline）を基準にする」という意味です。

```bash
# マージcommitの取り消し
$ git revert -m 1 abc123
# → マージで導入された変更を打ち消すcommitが作成される

# 注意: revertしたマージを再度マージしたい場合
# → revertのrevertが必要
$ git revert def456  # def456 = 上記のrevert commit
$ git merge feature  # 再マージが可能になる
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
| diffアルゴリズム    | histogram推奨、patience は構造変更に強い                      |
| リネーム検出        | 類似度ベースのヒューリスティック、ort戦略で高速化              |
| zdiff3              | 推奨コンフリクトスタイル、base表示で判断が容易               |
| --onto              | rebaseの柔軟な移動先指定、部分的なcommit移動が可能           |

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
5. **Elijah Newren** — "Merge strategies in Git" https://git-scm.com/docs/merge-strategies
6. **Git公式ドキュメント** — `gitattributes` merge drivers https://git-scm.com/docs/gitattributes
