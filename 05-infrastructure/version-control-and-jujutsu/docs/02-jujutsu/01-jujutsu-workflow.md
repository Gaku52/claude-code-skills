# Jujutsuワークフロー

> Jujutsuの変更セット（changeset）管理と自動リベース機能を活用した実践的な開発ワークフローを習得し、Gitでは困難だった柔軟なコミット操作を実現する。

## この章で学ぶこと

1. **変更セットの操作** — jj squash, jj split, jj move による柔軟なcommit編集
2. **自動リベースの仕組み** — 親commitの変更時に子commitが自動的にリベースされる動作
3. **実践的なブランチレス開発** — ブックマーク（旧branch）を使った効率的な開発フロー
4. **並行作業の管理** — 複数の作業を同時に進行するためのテクニック
5. **コミットの整理と最適化** — レビュー前のcommit履歴を整理するワークフロー

---

## 1. 変更セットの基本操作

### 1.1 jj new — 新しいcommitの作成

```bash
# 現在のworking copyの上に新しいcommitを作成
$ jj new
# → 現在の変更が確定し、新しい空のworking copy commitが作成される

# 特定のcommitの上に新しいcommitを作成
$ jj new qpvuntsm
# → qpvuntsm の子として新しいworking copy commitを作成
# → 元のworking copyの位置にあったcommitは自動的に上に移動

# マージcommitの作成（複数の親を指定）
$ jj new commit-a commit-b
# → commit-a と commit-b の両方を親に持つmerge commitを作成

# メッセージ付きで新しいcommitを作成
$ jj new -m "feat: 新機能の基盤"
# → 新しいcommitを作成し、同時にメッセージを設定

# 特定のrevisionの後に挿入
$ jj new --after aaa --before bbb
# → aaa と bbb の間に新しいcommitを挿入
```

```
┌─────────────────────────────────────────────────────┐
│  jj new の動作                                       │
│                                                     │
│  Before:                                            │
│  @  rlvkpntz  feat: 認証機能  ← working copy       │
│  ○  qpvuntsm  feat: 初期設定                        │
│  ◆  zzzzzzzz  root()                                │
│                                                     │
│  $ jj new                                           │
│                                                     │
│  After:                                             │
│  @  xtkvpqwm  (empty) (no description)  ← 新working│
│  ○  rlvkpntz  feat: 認証機能  ← 確定済み            │
│  ○  qpvuntsm  feat: 初期設定                        │
│  ◆  zzzzzzzz  root()                                │
│                                                     │
│  $ jj new qpvuntsm                                  │
│                                                     │
│  After:                                             │
│  ○  rlvkpntz  feat: 認証機能  ← 自動リベース        │
│  @  newcommit (empty)         ← 挿入されたcommit    │
│  ○  qpvuntsm  feat: 初期設定                        │
│  ◆  zzzzzzzz  root()                                │
└─────────────────────────────────────────────────────┘
```

### 1.2 jj edit — 過去のcommitを編集位置にする

```bash
# 過去のcommitをworking copyにする
$ jj edit qpvuntsm
# → qpvuntsm がworking copyになり、直接編集可能に
# → その上にあるcommitは自動的にリベースされる

$ vim src/config.js  # qpvuntsmのcommitを直接編集
# → 子commitが自動リベースされる

# change IDで指定（推奨）
$ jj edit rlvkpntz
# → commit IDではなくchange IDを使うことで、rebase後も追跡可能

# ブックマーク名で指定
$ jj edit feature-auth
# → ブックマークが指すcommitをworking copyにする
```

```
┌────────────────────────────────────────────────────┐
│  jj edit の動作                                     │
│                                                    │
│  Before:                                           │
│  @  rlvkpntz  feat: 認証機能                       │
│  ○  qpvuntsm  feat: 初期設定                       │
│                                                    │
│  $ jj edit qpvuntsm                                │
│                                                    │
│  After:                                            │
│  ○  rlvkpntz  feat: 認証機能  ← 自動リベース対象  │
│  @  qpvuntsm  feat: 初期設定  ← 直接編集可能       │
│                                                    │
│  ファイルを編集すると:                              │
│  ○  rlvkpntz' feat: 認証機能  ← 自動リベース!     │
│  @  qpvuntsm' feat: 初期設定  ← 変更された         │
│                                                    │
│  重要: editはjj newとは異なり、新しいcommitを       │
│  作成しない。既存のcommitを直接編集する。           │
└────────────────────────────────────────────────────┘
```

### 1.3 jj squash — 変更の統合

```bash
# working copyの変更を親commitに統合
$ jj squash
# → working copyの全変更が親commitに移動
# → working copyは空になる

# メッセージを同時に設定
$ jj squash -m "feat: 認証機能の完成版"
# → 統合後のcommitにメッセージを設定

# 特定のファイルだけ親commitに統合
$ jj squash --keep src/auth.js
# → src/auth.js の変更だけ親に移動、他はworking copyに残る

# 特定のcommitに対してsquash
$ jj squash --from rlvkpntz --into qpvuntsm
# → rlvkpntzの変更をqpvuntsmに統合

# パスを指定して部分的にsquash
$ jj squash src/auth.js src/middleware.js
# → 指定ファイルの変更のみ親に移動

# インタラクティブにsquashする内容を選択
$ jj squash -i
# → diff-editorが開き、squashする変更を選択できる
```

```
┌────────────────────────────────────────────────────┐
│  jj squash の動作パターン                           │
│                                                    │
│  パターン1: 全変更を親に統合                        │
│  Before:              After:                       │
│  @  B (changes)       @  B (empty)                 │
│  ○  A                 ○  A' (A + B's changes)      │
│                                                    │
│  パターン2: 特定ファイルのみ統合                    │
│  Before:              After:                       │
│  @  B (a.js, b.js)    @  B' (b.js only)            │
│  ○  A                 ○  A' (A + a.js changes)     │
│                                                    │
│  パターン3: 任意のcommit間での統合                  │
│  Before:              After:                       │
│  ○  C                 ○  C' (自動リベース)          │
│  ○  B (target)        ○  B' (empty, abandonされる)  │
│  ○  A (dest)          ○  A' (A + B's changes)      │
│                                                    │
│  $ jj squash --from B --into A                     │
└────────────────────────────────────────────────────┘
```

### 1.4 jj split — commitの分割

```bash
# working copyの変更をインタラクティブに分割
$ jj split
# → エディタが開き、最初のcommitに含める変更を選択
# → 残りは新しいcommitになる

# ファイル単位で分割
$ jj split src/auth.js src/middleware.js
# → 指定ファイルの変更が最初のcommitに
# → 残りのファイルの変更が次のcommitに

# 過去のcommitを分割
$ jj split -r rlvkpntz
# → rlvkpntz をインタラクティブに分割
# → 子commitは自動リベース

# パスパターンで分割
$ jj split "src/**/*.test.js"
# → テストファイルの変更を最初のcommitに
# → それ以外を次のcommitに
```

```
┌────────────────────────────────────────────────────┐
│  jj split の動作                                    │
│                                                    │
│  Before:                                           │
│  @  rlvkpntz  feat: 認証+UI                       │
│  │  (auth.js, middleware.js, Login.jsx を変更)      │
│  ○  qpvuntsm  ...                                  │
│                                                    │
│  $ jj split src/auth.js src/middleware.js           │
│                                                    │
│  After:                                            │
│  @  nwmqklop  (working copy, Login.jsx の変更)     │
│  ○  rlvkpntz  feat: 認証+UI                       │
│  │  (auth.js, middleware.js の変更のみ)             │
│  ○  qpvuntsm  ...                                  │
│                                                    │
│  splitの後は各commitのメッセージを修正する:         │
│  $ jj describe -r rlvkpntz -m "feat: 認証ロジック" │
│  $ jj describe -m "feat: ログインUI"               │
└────────────────────────────────────────────────────┘
```

### 1.5 jj move — 変更の移動（非推奨、squash推奨）

```bash
# 注意: jj move は古いコマンドで、jj squash --from --into に置き換えられた
# 互換性のために残っているが、squashの使用を推奨

# 旧: jj move --from A --to B
# 新: jj squash --from A --into B

# 使用例（squashで代替）
$ jj squash --from rlvkpntz --into qpvuntsm
# → rlvkpntz の変更を qpvuntsm に移動
```

### 1.6 jj diffedit — commitの内容をdiffエディタで直接編集

```bash
# working copyの変更をdiffエディタで編集
$ jj diffedit
# → diff-editorが開き、変更を追加・削除できる

# 過去のcommitをdiffエディタで編集
$ jj diffedit -r rlvkpntz
# → rlvkpntz の変更内容をdiffエディタで直接編集
# → 子commitは自動リベース

# 特定のファイルのみ編集
$ jj diffedit -r rlvkpntz src/auth.js
```

---

## 2. 自動リベース

### 2.1 自動リベースの仕組み

Jujutsuの最も強力な機能の一つ。**親commitが変更されると、子commit以降が自動的にリベースされる**。

```bash
# 3つのcommitが積まれた状態
$ jj log
@  ccc  feat: UI実装
○  bbb  feat: APIエンドポイント
○  aaa  feat: 初期設定
◆  root()

# 中間のcommitを直接編集
$ jj edit bbb
$ vim src/api.js
$ jj new  # 編集を確定して新しいworking copyへ

# → ccc が自動的にリベースされる！
$ jj log
@  ddd  (empty)
○  ccc' feat: UI実装           ← 自動リベース済み（SHA変更）
○  bbb' feat: APIエンドポイント ← 編集された
○  aaa  feat: 初期設定
◆  root()
```

```
┌─────────────────────────────────────────────────────┐
│  自動リベースの図解                                  │
│                                                     │
│  Git で同じことをする場合:                           │
│  1. git rebase -i で対象commitをeditに設定          │
│  2. 修正を加える                                    │
│  3. git commit --amend                              │
│  4. git rebase --continue                           │
│  5. コンフリクトがあれば各commitで解決              │
│  → 5ステップ + コンフリクト解決                     │
│                                                     │
│  Jujutsu で同じことをする場合:                      │
│  1. jj edit bbb                                     │
│  2. 修正を加える                                    │
│  → 2ステップ、自動リベース                          │
│  → コンフリクトは commit に記録（後で解決可）       │
└─────────────────────────────────────────────────────┘
```

### 2.2 自動リベースの連鎖

```bash
# 複数の子commitがある場合も全て自動リベース
$ jj log --no-graph
aaa  feat: 基盤
├── bbb  feat: 認証
│   └── ccc  feat: 認証テスト
└── ddd  feat: UI
    └── eee  feat: UIテスト

# aaa を編集すると、bbb, ccc, ddd, eee の全てが自動リベース
$ jj edit aaa
$ vim src/base.js
# → 5つの子commit全てが新しいaaa'の上にリベースされる
```

### 2.3 自動リベースとコンフリクトの関係

```bash
# 自動リベース時にコンフリクトが発生した場合
$ jj edit aaa
$ vim src/shared.js   # bbb でも変更しているファイルを編集

$ jj log
○  ccc' feat: 認証テスト
○  bbb' feat: 認証          conflict    ← コンフリクト発生
○  aaa' feat: 基盤          ← 編集された

# コンフリクトは記録されるが、リベースは完了している
# bbb' に移動してコンフリクトを解決
$ jj edit bbb'
$ vim src/shared.js   # コンフリクトマーカーを解決

# コンフリクト解決後、ccc' も自動的にリベースされる
$ jj log
○  ccc'' feat: 認証テスト    ← 再リベース
○  bbb'' feat: 認証           ← コンフリクト解決済み
○  aaa'  feat: 基盤
```

### 2.4 自動リベースが発生しないケース

```
┌────────────────────────────────────────────────────┐
│  自動リベースが発生しないケース                      │
│                                                    │
│  1. immutableなcommitの子は自動リベースされない     │
│     → trunk() や tags() 以前のcommit               │
│                                                    │
│  2. abandonしたcommitの子は親の親に接続される       │
│     → リベースではなく「再接続」                    │
│                                                    │
│  3. 別のworkspaceのworking copyは影響を受けない     │
│     → ワークスペース間は独立                        │
│                                                    │
│  4. jj rebase -r（単一commit）の場合                │
│     → 指定commitのみ移動、子は元の位置に残る        │
│     → -s（subtree）とは異なる動作                   │
└────────────────────────────────────────────────────┘
```

---

## 3. ブックマーク（旧branch）

### 3.1 ブックマークの基本

```bash
# ブックマークの作成（Git branchに相当）
$ jj bookmark create feature-auth -r @
# → 現在のworking copy commitに "feature-auth" ブックマークを設定

# ブックマークの一覧
$ jj bookmark list
feature-auth: rlvkpntz abc12345
main: qpvuntsm def67890

# 全てのブックマーク（リモート含む）を表示
$ jj bookmark list --all
feature-auth: rlvkpntz abc12345
main: qpvuntsm def67890
main@origin: qpvuntsm def67890

# ブックマークの移動
$ jj bookmark set feature-auth -r @

# ブックマークの削除
$ jj bookmark delete feature-auth

# リモートブックマークの追跡
$ jj bookmark track main@origin

# ブックマークの名前変更
$ jj bookmark rename old-name new-name

# ブックマークの追跡解除
$ jj bookmark untrack feature@origin
```

### 3.2 ブランチレス開発のワークフロー

```bash
# Jujutsuではブランチ名をつけなくても開発できる
$ jj new main          # mainの上に新しいcommitを作成
$ vim src/feature.js
$ jj describe -m "feat: 新機能のプロトタイプ"

# 別の作業をしたくなったら
$ jj new main          # mainの上にもう1つcommitを作成
$ vim src/hotfix.js
$ jj describe -m "fix: 緊急バグ修正"

# ログで確認
$ jj log
○  xxx  fix: 緊急バグ修正
│ ○  yyy  feat: 新機能のプロトタイプ
├─┘
◆  main  ...
```

```
┌────────────────────────────────────────────────────┐
│  ブランチレス開発の利点                              │
│                                                    │
│  Git:                                              │
│  $ git checkout -b feature/auth   # ブランチ作成   │
│  $ ... 作業 ...                                    │
│  $ git checkout -b hotfix/bug     # 別ブランチ作成 │
│  $ ... 作業 ...                                    │
│  $ git checkout feature/auth      # 戻る           │
│  → ブランチの切り替えが煩雑                        │
│  → 未コミットの変更があるとstashが必要              │
│                                                    │
│  Jujutsu:                                          │
│  $ jj new main                    # 新commit       │
│  $ ... 作業 ...                                    │
│  $ jj new main                    # 別のcommit     │
│  $ ... 作業 ...                                    │
│  $ jj edit <change-id>            # 任意に移動     │
│  → 全てがcommitなのでstash不要                     │
│  → ブランチ名の管理が不要                          │
│  → push時にだけブックマークを設定                  │
└────────────────────────────────────────────────────┘
```

### 3.3 ブックマークとpushの関係

```bash
# ブックマークなしではpushできない
$ jj git push
# Nothing to push

# pushするにはブックマークが必要
# 方法1: 明示的にブックマークを作成してpush
$ jj bookmark create feature-auth -r @
$ jj git push --bookmark feature-auth --allow-new

# 方法2: --change オプションで自動ブックマーク
$ jj git push --change @
# → change IDからブックマーク名が自動生成される
# → 例: "push-rlvkpntzqwop" のようなブランチ名

# 方法3: 複数のブックマークを一度にpush
$ jj git push --bookmark feature-auth --bookmark feature-ui

# ブックマークのpush状態を確認
$ jj bookmark list --all
feature-auth: rlvkpntz abc12345
  @origin: rlvkpntz abc12345    ← リモートと同期済み
feature-ui: qpvuntsm def67890
  @origin (behind): rlvkpntz old12345  ← ローカルが先行
```

### 3.4 ブックマークの自動更新

```bash
# ブックマークが設定されたcommitを編集すると
# ブックマークは自動的に新しいcommit IDを追跡する

$ jj log
@  rlvkpntz  feature-auth  feat: 認証機能
○  main ...

$ jj edit rlvkpntz    # feature-auth のcommitを編集
$ vim src/auth.js     # ファイルを修正

# ブックマークは自動的に新しいcommitを追跡
$ jj bookmark list
feature-auth: rlvkpntz abc12345   ← change IDは同じ
# → commit IDは変わるが、change IDは変わらない
# → ブックマークはchange IDを介して追跡される
```

---

## 4. コミットの並べ替えと挿入

### 4.1 jj rebase — コミットの移動

```bash
# 単一commitの親を変更（子commitは元の位置に残る）
$ jj rebase -r rlvkpntz -d main
# → rlvkpntz のみが main の子に移動
# → rlvkpntz の元の子commitは rlvkpntz の親に接続される

# commitとその子孫全体を移動
$ jj rebase -s rlvkpntz -d main
# → rlvkpntz以降の全commitをmainの上に移動

# 範囲指定でのリベース（ブランチの先端まで）
$ jj rebase -b feature-auth -d main
# → feature-authブックマークまでのcommitをmainの上に移動

# 複数の親を指定（マージcommitの作成を伴うリベース）
$ jj rebase -r rlvkpntz -d main -d feature-other
# → rlvkpntz がmainとfeature-otherの両方を親に持つようになる
```

```
┌────────────────────────────────────────────────────┐
│  jj rebase の3つのモード                            │
│                                                    │
│  -r (revision): 単一commitのみ移動                 │
│  Before:          After:                           │
│  ○  C             ○  C' (Aの子に接続)              │
│  ○  B             │ ○  B' (mainの子に移動)         │
│  ○  A             ○  A                             │
│  ◆  main          ◆  main                          │
│  $ jj rebase -r B -d main                          │
│                                                    │
│  -s (source): commitとその子孫を移動               │
│  Before:          After:                           │
│  ○  C             ○  C' (B'の子)                   │
│  ○  B             ○  B' (mainの子に移動)           │
│  ○  A             ○  A                             │
│  ◆  main          ◆  main                          │
│  $ jj rebase -s B -d main                          │
│                                                    │
│  -b (branch): ブランチのルートから先端まで移動      │
│  Before:          After:                           │
│  ○  C             ○  C'                            │
│  ○  B             ○  B'                            │
│  ○  A             ○  A' (mainの子に移動)           │
│  ◆  main          ◆  main                          │
│  $ jj rebase -b C -d main                          │
└────────────────────────────────────────────────────┘
```

### 4.2 コミット間への挿入

```bash
# 既存の2つのcommitの間に新しいcommitを挿入
$ jj new --after aaa --before bbb
# → aaa と bbb の間に新しいcommitが挿入される
# → bbb以降は自動リベース

# 結果:
# ○  bbb'  feat: API      ← 自動リベースされた
# ○  new   (working copy)  ← 挿入された新commit
# ○  aaa   feat: 初期設定
```

```
┌────────────────────────────────────────────────────┐
│  commit挿入の図解                                   │
│                                                    │
│  Before:             After:                        │
│  ○  bbb             ○  bbb' (自動リベース)         │
│  ○  aaa             @  new  (挿入された)           │
│                      ○  aaa                        │
│                                                    │
│  Git で同じことをする場合:                          │
│  1. git rebase -i でaaa以降をedit                  │
│  2. aaaの後で停止                                  │
│  3. 新しいcommitを作成                             │
│  4. git rebase --continue                          │
│  → 非常に手間がかかる                              │
└────────────────────────────────────────────────────┘
```

### 4.3 コミットの順序入れ替え

```bash
# 2つのcommitの順序を入れ替える
# Before:
# ○  B  feat: UI
# ○  A  feat: 認証

# Aを B の上に移動
$ jj rebase -r A -d B
# → A が B の上に移動

# After:
# ○  A' feat: 認証
# ○  B  feat: UI

# より複雑な入れ替え（3つのcommitの順序変更）
# Before: C → B → A → main
# Goal:   A → C → B → main

$ jj rebase -r A -d main     # まずAをmainの直上に
$ jj rebase -s B -d A        # B以降をAの上に
# → A → B → C → main になる
# さらに
$ jj rebase -r C -d A        # CをAの直上に（Bの前に挿入）
# → A → C' → B' → main になる
```

---

## 5. 並行作業の管理

### 5.1 複数の作業を同時進行

```bash
# 作業1: 認証機能
$ jj new main
$ jj describe -m "feat: 認証機能"
$ vim src/auth.js

# 作業2: 認証機能の上にUI
$ jj new
$ jj describe -m "feat: ログインUI"
$ vim src/Login.jsx

# 作業3: mainから別の作業を開始（認証とは独立）
$ jj new main
$ jj describe -m "fix: パフォーマンス改善"
$ vim src/perf.js

# 全ての作業を一覧
$ jj log -r 'heads(all())'

# 作業の切り替え
$ jj edit rlvkpntz    # 認証機能の作業に戻る
$ jj edit qpvuntsm    # パフォーマンス改善の作業に切り替え
```

### 5.2 作業の合流（マージ）

```bash
# 2つの作業をマージ
$ jj new feature-auth perf-fix
$ jj describe -m "merge: 認証とパフォーマンス改善を統合"

# あるいは特定のcommitをrebaseで合流
$ jj rebase -r perf-fix -d feature-auth
```

### 5.3 ワークスペースを使った並行作業

```bash
# ワークスペースは同一リポジトリを複数のディレクトリで扱う機能
# 各ワークスペースは独立したworking copyを持つ

# 新しいワークスペースの作成
$ jj workspace add ../my-project-hotfix
# → ../my-project-hotfix/ に新しいワークスペースが作成される
# → 同じリポジトリを共有しつつ、独立したworking copyを持つ

# ワークスペースの一覧
$ jj workspace list
default: rlvkpntz abc12345
hotfix: qpvuntsm def67890

# hotfixワークスペースで作業
$ cd ../my-project-hotfix
$ jj new main
$ vim src/fix.js
$ jj describe -m "fix: 緊急修正"

# 元のワークスペースに戻る
$ cd ../my-project
$ jj log  # hotfixの変更も見える

# ワークスペースの削除
$ jj workspace forget hotfix
```

```
┌────────────────────────────────────────────────────┐
│  ワークスペースの利点                               │
│                                                    │
│  Git worktreeと類似しているが、以下が異なる:        │
│                                                    │
│  1. ワークスペース間でcommitが自動的に共有される    │
│  2. あるワークスペースでのrebaseが他にも反映される  │
│  3. Operation Logがリポジトリ全体で共有される       │
│                                                    │
│  典型的なユースケース:                              │
│  - メインの開発作業 + hotfix作業                   │
│  - ビルド確認用の別ディレクトリ                    │
│  - CI/CD用の隔離された環境                         │
└────────────────────────────────────────────────────┘
```

### 5.4 独立した変更の管理パターン

```bash
# パターン1: フィーチャーフラグを使った段階的開発
$ jj new main
$ jj describe -m "feat: フィーチャーフラグ基盤"
$ vim src/feature-flags.js
$ jj new
$ jj describe -m "feat: ダークモード（フラグ付き）"
$ vim src/dark-mode.js

# パターン2: レビュー待ちの間に次の作業を開始
$ jj log
○  review-1  feat: 認証機能  ← レビュー中
◆  main

$ jj new review-1    # レビュー中のcommitの上に積む
$ jj describe -m "feat: 認証に基づくAPI"
$ vim src/api.js
# → review-1 がマージされてmainが更新されたら
# → jj git fetch && jj rebase -d main@origin で更新

# パターン3: 複数の独立した修正
$ jj new main -m "fix: ヘッダーのレイアウト修正"
$ vim src/header.css

$ jj new main -m "fix: フッターのリンク修正"
$ vim src/footer.html

$ jj new main -m "docs: READMEの更新"
$ vim README.md

# 各修正を個別にpush
$ jj git push --change rlvkpntz   # ヘッダー修正
$ jj git push --change qpvuntsm   # フッター修正
$ jj git push --change xtkvpqwm   # README更新
```

---

## 6. abandon と restore

### 6.1 jj abandon — commitの破棄

```bash
# commitの破棄（内容は削除、子commitは親に接続）
$ jj abandon rlvkpntz
# → rlvkpntz が削除され、子commitの親がrlvkpntzの親に変更される

# 複数のcommitを一度にabandon
$ jj abandon rlvkpntz qpvuntsm
# → 2つのcommitを同時に破棄

# revsetで条件指定してabandon
$ jj abandon 'empty() & mine()'
# → 自分の空のcommitを全て破棄

# working copyをabandon（変更の破棄）
$ jj abandon @
# → 現在のworking copyの変更を全て破棄
# → 新しい空のworking copyが親の上に作成される
```

### 6.2 jj restore — ファイルの復元

```bash
# 操作の取り消し
$ jj undo
# → 直前のjjコマンドを完全に取り消す

# 特定ファイルの復元
$ jj restore --from main src/config.js
# → mainのsrc/config.jsの内容をworking copyに復元

# 特定のrevisionから複数ファイルを復元
$ jj restore --from @- src/auth.js src/api.js
# → 親commitから2つのファイルを復元

# 全ファイルを特定のrevisionから復元
$ jj restore --from main
# → working copyの全ファイルをmainの状態に復元

# パスパターンで復元
$ jj restore --from @- "src/**/*.test.js"
# → テストファイルのみを親commitの状態に復元
```

### 6.3 jj backout — 変更の打ち消し

```bash
# 特定のcommitの変更を打ち消す新しいcommitを作成
$ jj backout -r rlvkpntz
# → rlvkpntz の変更を逆にした新しいcommitが作成される
# → Git の git revert に相当

# 打ち消しcommitの配置先を指定
$ jj backout -r rlvkpntz -d @
# → working copyの子として打ち消しcommitを作成
```

| 操作           | 説明                                                  |
|----------------|-------------------------------------------------------|
| `jj abandon`   | commitを削除、子commitは親に再接続                    |
| `jj undo`      | 直前のjjコマンドを完全に取り消し                      |
| `jj restore`   | 特定revision/ファイルの内容をworking copyに復元       |
| `jj op restore`| 特定の操作時点にリポジトリ全体を復元                  |
| `jj backout`   | commitの変更を打ち消す新しいcommitを作成（git revert） |

---

## 7. 実践的なワークフローパターン

### 7.1 スタックドPR（積み上げPR）ワークフロー

```bash
# PRを積み上げて段階的にレビュー・マージする

# Step 1: 基盤となる型定義
$ jj new main
$ vim src/types.ts
$ jj describe -m "feat: 型定義の追加"
$ jj bookmark create pr/types -r @

# Step 2: 型定義を使った認証ロジック
$ jj new
$ vim src/auth.ts
$ jj describe -m "feat: 認証ロジック"
$ jj bookmark create pr/auth -r @

# Step 3: 認証を使ったAPIエンドポイント
$ jj new
$ vim src/api.ts
$ jj describe -m "feat: APIエンドポイント"
$ jj bookmark create pr/api -r @

# 各ブックマークをpush
$ jj git push --bookmark pr/types --allow-new
$ jj git push --bookmark pr/auth --allow-new
$ jj git push --bookmark pr/api --allow-new

# ベースの変更を修正（型定義を更新）→ 全てが自動リベース
$ jj edit pr/types
$ vim src/types.ts
$ jj new    # 修正を確定
# → pr/auth と pr/api が自動リベース！
# → 各PRを再pushするだけ
$ jj git push --bookmark pr/types
$ jj git push --bookmark pr/auth
$ jj git push --bookmark pr/api
```

### 7.2 レビュー対応ワークフロー

```bash
# レビューコメントに対応する

# レビュー対象のcommitを直接編集
$ jj edit pr/auth
$ vim src/auth.ts    # レビューコメントに対応した修正
$ jj new             # 修正を確定

# あるいは新しいcommitとして修正を追加し、後でsquash
$ jj new pr/auth
$ vim src/auth.ts
$ jj describe -m "fix: レビュー対応 - エラーハンドリング追加"
$ jj squash          # 修正を pr/auth に統合

# 再push
$ jj git push --bookmark pr/auth
```

### 7.3 mainへの追従ワークフロー

```bash
# mainが更新された場合のリベース

# リモートの最新を取得
$ jj git fetch

# 現在の作業をmainの最新にリベース
$ jj rebase -d main@origin

# スタック全体をリベース
$ jj rebase -s <stack-root> -d main@origin

# コンフリクトが発生した場合
$ jj log -r 'conflict()'
# → コンフリクトのあるcommitを確認
$ jj edit <conflict-commit>
$ vim <conflicting-file>
$ jj new    # 解決を確定
```

### 7.4 緊急hotfixワークフロー

```bash
# 開発中に緊急のhotfixが必要になった場合

# 現在の作業の状態を記録（不要、全てcommit済み）
$ jj log  # 現在の作業を確認

# mainからhotfixを作成
$ jj new main@origin
$ vim src/critical-fix.js
$ jj describe -m "fix: セキュリティ脆弱性の修正"

# 即座にpush
$ jj bookmark create hotfix -r @
$ jj git push --bookmark hotfix --allow-new

# 元の作業に戻る
$ jj edit <元のchange-id>
# → stash不要、コンテキスト切り替えが瞬時

# hotfixがマージされた後、自分の作業をリベース
$ jj git fetch
$ jj rebase -d main@origin
```

### 7.5 リファクタリングとフィーチャーの分離

```bash
# 開発中にリファクタリングの必要性に気づいた場合

# 現在の作業にリファクタリングとフィーチャーが混在
$ jj log
@  feature-commit  feat: 新機能（リファクタリング含む）
○  main

# splitでリファクタリングとフィーチャーを分離
$ jj split src/refactored-file.js
# → 1つ目のcommit: リファクタリング部分
# → 2つ目のcommit: フィーチャー部分

# メッセージを修正
$ jj describe -r @- -m "refactor: コード構造の改善"
$ jj describe -m "feat: 新機能の実装"

# リファクタリングを先にmainにマージすることも可能
$ jj bookmark create pr/refactor -r @-
$ jj git push --bookmark pr/refactor --allow-new
```

### 7.6 大規模な変更を小さなcommitに分割

```bash
# 大きな変更を段階的にcommitに分割するワークフロー

# まず全ての変更を1つのcommitに入れる
$ jj describe -m "feat: 大規模な機能追加（WIP）"

# ファイル単位で分割
$ jj split src/types.ts
# → types.ts の変更が1つ目のcommit
$ jj describe -r @- -m "feat: 型定義の追加"

$ jj split src/auth.ts src/auth.test.ts
# → auth関連の変更が1つ目のcommit
$ jj describe -r @- -m "feat: 認証ロジックの実装"

# 残りの変更に最終的なメッセージを設定
$ jj describe -m "feat: UIコンポーネントの実装"

# 結果:
$ jj log
@  xxx  feat: UIコンポーネントの実装
○  yyy  feat: 認証ロジックの実装
○  zzz  feat: 型定義の追加
◆  main
```

---

## 8. アンチパターン

### アンチパターン1: 全ての変更を1つのcommitに入れ続ける

```bash
# NG: jj newを使わずに全変更を1つのcommitに蓄積
$ vim src/auth.js
$ vim src/ui.js
$ vim src/api.js
$ jj describe -m "feat: 全部入り"
# → 巨大な1つのcommitになり、レビューしづらい

# OK: 論理的な単位でcommitを分ける
$ vim src/auth.js
$ jj describe -m "feat: 認証ロジック"
$ jj new
$ vim src/api.js
$ jj describe -m "feat: APIエンドポイント"
$ jj new
$ vim src/ui.js
$ jj describe -m "feat: UI実装"
```

**理由**: Jujutsuのworking copy = commitモデルでは、`jj new`を意識的に使って変更を分割する必要がある。Gitの`git add -p`に相当する部分選択は`jj split`で後からでも可能。

### アンチパターン2: change IDとcommit IDを混同する

```bash
# NG: commit ID（SHA-1）でrevisionを参照し続ける
$ jj rebase -r abc12345 -d main
# → rebase後にSHA-1が変わり、以前のIDが無効になる可能性

# OK: change IDで参照する
$ jj rebase -r rlvkpntz -d main
# → change IDはrebase後も変わらない
```

**理由**: commit IDはGitのSHA-1ハッシュでありcommitの内容に依存するため、rebaseで変化する。change IDはJujutsu独自の識別子で、内容が変わっても追跡可能。

### アンチパターン3: ブックマークを頻繁に手動で移動する

```bash
# NG: commitを編集するたびにブックマークを手動で移動
$ jj edit feature-auth
$ vim src/auth.js
$ jj new
$ jj bookmark set feature-auth -r ???  # どこに設定すべきか混乱

# OK: ブックマークは push 直前に設定する
$ jj edit rlvkpntz     # change IDで参照
$ vim src/auth.js
$ jj new
$ jj bookmark set feature-auth -r rlvkpntz  # push直前に設定
$ jj git push --bookmark feature-auth
```

**理由**: ブックマークはGitのブランチに相当するもので、主にpush/fetch時のリモートとの対応付けに使用する。日常の開発中はchange IDで参照し、pushが必要な時にだけブックマークを操作する。

### アンチパターン4: jj edit と jj new の使い分けを誤る

```bash
# NG: 新しい作業を始めるのに jj edit を使う
$ jj edit main    # ← immutableで編集できない上、意図と異なる

# OK: 新しい作業は jj new で始める
$ jj new main     # mainの上に新しいcommitを作成

# NG: 過去のcommitを修正するのに jj new を使う
$ jj new rlvkpntz   # ← 新しいcommitが作成されてしまう

# OK: 過去のcommitの修正は jj edit を使う
$ jj edit rlvkpntz   # そのcommitを直接編集
```

**理由**: `jj new`は新しいcommitを作成し、`jj edit`は既存のcommitをworking copyにして直接編集する。目的に応じて正しく使い分ける必要がある。

---

## 9. FAQ

### Q1. `jj new`と`jj commit`の違いは何か？

**A1.** `jj commit`は`jj new`とほぼ同じですが、**コミットメッセージの入力を同時に行う**ショートカットです。

```bash
# 以下は同等の操作
$ jj describe -m "feat: 新機能" && jj new
$ jj commit -m "feat: 新機能"
```

`jj commit`はGitからの移行者向けの利便性コマンドで、内部的には「describeしてからnew」と同じ動作をします。

### Q2. 自動リベースでコンフリクトが発生した場合はどうなるか？

**A2.** コンフリクトはcommitに記録されます。**リベースは中断されません**。コンフリクトのあるcommitは`jj log`で`conflict`マークが表示されます。`jj edit`でそのcommitに移動し、ファイルを編集してコンフリクトを解決できます。急ぎでなければ後回しにすることも可能です。

### Q3. Jujutsuでstashに相当する操作は何か？

**A3.** Jujutsuではstashは**不要**です。全ての変更はcommitとして保存されるため、別の作業に移りたい場合は以下のようにします。

```bash
# Gitでのstash相当の操作（Jujutsu）
$ jj new main       # mainの上に新しいcommitを作成して作業開始
# → 前のworking copyの変更はそのまま確定済みcommitとして残る
# → 戻りたくなったら jj edit <change-id> で即座に戻れる
```

### Q4. jj squash と jj edit はどう使い分けるか？

**A4.** 以下のように使い分けます。

```bash
# jj edit: 過去のcommitを直接修正したい時
# → commitの内容を直接変更する
# → ファイルを編集してそのcommit自体を変更
$ jj edit rlvkpntz
$ vim src/auth.js    # commitの内容を修正
$ jj new

# jj squash: working copyの変更を親commitに統合したい時
# → 現在の作業を直前のcommitにまとめる
$ vim src/auth.js    # working copyで作業
$ jj squash          # 変更を親commitに統合

# jj squash --from --into: 任意の2つのcommit間で統合
$ jj squash --from bbb --into aaa
```

### Q5. ワークスペースとは何か？Gitのworktreeと同じか？

**A5.** ワークスペースはGitのworktreeに類似していますが、Jujutsuのモデルに基づいて設計されています。

```bash
# ワークスペースの主な違い
# - Git worktree: 各worktreeが独立したブランチを持つ
# - jj workspace: リポジトリ全体の状態を共有、各workspaceは独立したworking copyを持つ

# ワークスペースの作成
$ jj workspace add ../my-project-test
# → 同じリポジトリを参照する新しいディレクトリが作成される

# ワークスペース間の操作
# workspace-1 で行った変更は、workspace-2 の jj log でも見える
```

### Q6. jj rebase -r と -s と -b の違いは？

**A6.**

| オプション | 移動対象            | 子commitの扱い                |
|------------|---------------------|-------------------------------|
| `-r`       | 指定commitのみ      | 子は指定commitの親に接続      |
| `-s`       | 指定commit+全子孫   | 子孫も一緒に移動              |
| `-b`       | ルートから指定commit | 範囲全体が移動                |

```bash
# -r: 単一commitのみ移動
$ jj rebase -r B -d main
# B のみが main の子に。B の元の子は B の親に接続される

# -s: サブツリー全体を移動
$ jj rebase -s B -d main
# B とその全ての子孫が main の下に移動

# -b: ブランチのルートから先端まで
$ jj rebase -b tip -d main
# tip までの全commitが main の下に移動
```

---

## まとめ

| 概念             | 要点                                                          |
|------------------|---------------------------------------------------------------|
| jj new           | 新しいcommitを開始、前の変更を確定                            |
| jj edit          | 過去のcommitを直接編集、子は自動リベース                      |
| jj squash        | working copyの変更を親commitに統合                            |
| jj split         | 1つのcommitを複数に分割                                       |
| jj rebase        | commitの親を変更、子は自動リベース                            |
| jj diffedit      | diffエディタでcommitの内容を直接編集                          |
| 自動リベース     | 親commit変更時に子commit以降が自動的にリベースされる           |
| ブックマーク     | Gitブランチに相当、push時に必要                                |
| jj abandon       | commitを破棄、子commitは親に再接続                            |
| jj backout       | commitの変更を打ち消す（git revert相当）                      |
| ワークスペース   | 同一リポジトリの複数のworking copy                            |

---

## 次に読むべきガイド

- [Jujutsu応用](./02-jujutsu-advanced.md) — revset、テンプレート、Git連携の高度な使い方
- [Git→Jujutsu移行](./03-git-to-jujutsu.md) — 操作対応表と移行ガイド
- [Jujutsu入門](./00-jujutsu-introduction.md) — 基本概念の復習

---

## 参考文献

1. **Jujutsu公式ドキュメント** — "Tutorial" https://martinvonz.github.io/jj/latest/tutorial/
2. **Jujutsu GitHubリポジトリ** — "Working Copy" https://github.com/martinvonz/jj/blob/main/docs/working-copy.md
3. **Chris Krycho** — "jj init: Jujutsu tips and tricks" https://v5.chriskrycho.com/essays/jj-init/
4. **Austin Seipp** — "Stacked PRs with Jujutsu" https://austinseipp.com/posts/2024-07-10-jj-hierarchies
5. **Jujutsu公式ドキュメント** — "Workspaces" https://martinvonz.github.io/jj/latest/working-copy/#workspaces
