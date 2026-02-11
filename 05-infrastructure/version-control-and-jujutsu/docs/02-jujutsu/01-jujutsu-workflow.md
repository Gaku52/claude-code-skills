# Jujutsuワークフロー

> Jujutsuの変更セット（changeset）管理と自動リベース機能を活用した実践的な開発ワークフローを習得し、Gitでは困難だった柔軟なコミット操作を実現する。

## この章で学ぶこと

1. **変更セットの操作** — jj squash, jj split, jj move による柔軟なcommit編集
2. **自動リベースの仕組み** — 親commitの変更時に子commitが自動的にリベースされる動作
3. **実践的なブランチレス開発** — ブックマーク（旧branch）を使った効率的な開発フロー

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
└────────────────────────────────────────────────────┘
```

### 1.3 jj squash — 変更の統合

```bash
# working copyの変更を親commitに統合
$ jj squash
# → working copyの全変更が親commitに移動
# → working copyは空になる

# 特定のファイルだけ親commitに統合
$ jj squash --keep src/auth.js
# → src/auth.js の変更だけ親に移動、他はworking copyに残る

# 特定のcommitに対してsquash
$ jj squash --from rlvkpntz --into qpvuntsm
# → rlvkpntzの変更をqpvuntsmに統合
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
└────────────────────────────────────────────────────┘
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

# ブックマークの移動
$ jj bookmark set feature-auth -r @

# ブックマークの削除
$ jj bookmark delete feature-auth

# リモートブックマークの追跡
$ jj bookmark track main@origin
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

---

## 4. コミットの並べ替えと挿入

### 4.1 jj rebase — コミットの移動

```bash
# 単一commitの親を変更
$ jj rebase -r rlvkpntz -d main
# → rlvkpntz の親を main に変更（子commitも自動リベース）

# commitとその子孫全体を移動
$ jj rebase -s rlvkpntz -d main
# → rlvkpntz以降の全commitをmainの上に移動

# 範囲指定でのリベース
$ jj rebase -b feature-auth -d main
# → feature-authブックマークまでのcommitをmainの上に移動
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
```

### 5.2 作業の合流（マージ）

```bash
# 2つの作業をマージ
$ jj new feature-auth perf-fix
$ jj describe -m "merge: 認証とパフォーマンス改善を統合"

# あるいは特定のcommitをrebaseで合流
$ jj rebase -r perf-fix -d feature-auth
```

---

## 6. abandon と restore

```bash
# commitの破棄（内容は削除、子commitは親に接続）
$ jj abandon rlvkpntz
# → rlvkpntz が削除され、子commitの親がrlvkpntzの親に変更される

# 操作の取り消し
$ jj undo
# → 直前のjjコマンドを完全に取り消す

# 特定ファイルの復元
$ jj restore --from main src/config.js
# → mainのsrc/config.jsの内容をworking copyに復元
```

| 操作           | 説明                                                  |
|----------------|-------------------------------------------------------|
| `jj abandon`   | commitを削除、子commitは親に再接続                    |
| `jj undo`      | 直前のjjコマンドを完全に取り消し                      |
| `jj restore`   | 特定revision/ファイルの内容をworking copyに復元       |
| `jj op restore`| 特定の操作時点にリポジトリ全体を復元                  |

---

## 7. アンチパターン

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

---

## 8. FAQ

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

---

## まとめ

| 概念             | 要点                                                          |
|------------------|---------------------------------------------------------------|
| jj new           | 新しいcommitを開始、前の変更を確定                            |
| jj edit          | 過去のcommitを直接編集、子は自動リベース                      |
| jj squash        | working copyの変更を親commitに統合                            |
| jj split         | 1つのcommitを複数に分割                                       |
| jj rebase        | commitの親を変更、子は自動リベース                            |
| 自動リベース     | 親commit変更時に子commit以降が自動的にリベースされる           |
| ブックマーク     | Gitブランチに相当、push時に必要                                |
| jj abandon       | commitを破棄、子commitは親に再接続                            |

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
