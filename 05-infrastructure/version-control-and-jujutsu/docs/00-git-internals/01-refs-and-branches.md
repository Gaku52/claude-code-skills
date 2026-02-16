# Ref・ブランチ

> GitのRef（参照）機構を深堀りし、HEAD、ブランチ、タグ、reflogの内部表現とdetached HEAD状態の正しい理解・復旧方法を解説する。

## この章で学ぶこと

1. **Refの種類と内部表現** — ブランチ、タグ、リモート追跡ブランチがファイルシステム上でどう管理されるか
2. **HEADの仕組みとdetached HEAD** — シンボリック参照の動作原理と安全な運用方法
3. **reflogによる履歴復元** — 失われたコミットの追跡と救出テクニック
4. **packed-refsと参照解決** — 大量のrefの最適化と解決優先順位
5. **タグの内部表現** — lightweight tagとannotated tagの違い
6. **ブランチ運用パターン** — 実務で遭遇する様々なブランチ操作と内部動作の理解

---

## 1. Refとは何か

Refは**SHA-1ハッシュへのポインタ**であり、`.git/refs/`配下のテキストファイルとして保存される。Gitのオブジェクトモデル（blob、tree、commit、tag）では、すべてのオブジェクトがSHA-1（またはSHA-256）ハッシュで一意に識別されるが、40文字のハッシュを直接覚えるのは人間には困難である。Refはこのハッシュに**人間が読みやすい名前**を与える仕組みである。

### 1.1 Refのファイルシステム上の配置

```
.git/
├── HEAD                          ← シンボリック参照
├── ORIG_HEAD                     ← merge/rebase/reset前のHEAD位置
├── MERGE_HEAD                    ← マージ中の相手側HEAD
├── FETCH_HEAD                    ← fetch結果の一時参照
├── CHERRY_PICK_HEAD              ← cherry-pick中の参照
├── REVERT_HEAD                   ← revert中の参照
├── refs/
│   ├── heads/                    ← ローカルブランチ
│   │   ├── main                  ← "main"ブランチ
│   │   ├── develop               ← "develop"ブランチ
│   │   └── feature/auth          ← "feature/auth"ブランチ
│   ├── tags/                     ← タグ
│   │   ├── v1.0.0
│   │   └── v2.0.0
│   ├── remotes/                  ← リモート追跡ブランチ
│   │   ├── origin/
│   │   │   ├── main
│   │   │   ├── develop
│   │   │   └── feature/auth
│   │   └── upstream/
│   │       └── main
│   ├── stash                     ← stashの最新エントリ
│   └── notes/                    ← git notesの参照
│       └── commits
├── packed-refs                   ← pack済みref（最適化）
└── logs/                         ← reflog
    ├── HEAD
    └── refs/
        ├── heads/
        │   ├── main
        │   └── feature/auth
        └── remotes/
            └── origin/
                └── main
```

### 1.2 Refの実体を確認する

```bash
# ブランチの実体を確認
$ cat .git/refs/heads/main
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# HEADの実体を確認（シンボリック参照）
$ cat .git/HEAD
ref: refs/heads/main

# git rev-parseでRefをSHA-1に変換
$ git rev-parse main
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

$ git rev-parse HEAD
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# refが指すオブジェクトの型を確認
$ git cat-file -t refs/heads/main
commit

# 全てのrefを一覧表示
$ git for-each-ref --format='%(refname) %(objecttype) %(objectname:short)' refs/
refs/heads/develop commit a1b2c3d
refs/heads/feature/auth commit f5e6d7c
refs/heads/main commit a1b2c3d
refs/remotes/origin/main commit a1b2c3d
refs/tags/v1.0.0 tag 1234567
refs/tags/v2.0.0 commit 89abcde
```

### 1.3 Refの名前解決ルール

Gitは省略されたref名を以下の順序で解決する。この順序を理解しておくことで、同名のブランチとタグが存在する場合の挙動を予測できる。

```
git rev-parse <name> の解決順序:

1. <name> をそのまま試す（例: HEAD、ORIG_HEAD）
2. refs/<name>
3. refs/tags/<name>
4. refs/heads/<name>
5. refs/remotes/<name>
6. refs/remotes/<name>/HEAD
```

```bash
# 同名のブランチとタグがある場合の問題
$ git branch v1.0.0       # ブランチ "v1.0.0" を作成
$ git checkout v1.0.0     # タグ? ブランチ? → 警告が出る

warning: refname 'v1.0.0' is ambiguous.

# 明示的に指定する方法
$ git checkout refs/heads/v1.0.0    # ブランチを指定
$ git checkout refs/tags/v1.0.0     # タグを指定（detached HEAD）

# rev-parseでの明示的な解決
$ git rev-parse refs/heads/v1.0.0   # ブランチのSHA-1
$ git rev-parse refs/tags/v1.0.0    # タグのSHA-1
```

### 1.4 特殊なRef

Gitには特定の操作中に自動的に設定される特殊なRefがある。

| Ref名               | 設定タイミング              | 用途                                     |
|---------------------|-----------------------------|-----------------------------------------|
| `HEAD`              | 常時                        | 現在のチェックアウト位置                 |
| `ORIG_HEAD`         | merge/rebase/reset後        | 操作前のHEAD位置（取り消し用）          |
| `MERGE_HEAD`        | merge中                     | マージ中の相手ブランチのHEAD            |
| `FETCH_HEAD`        | fetch後                     | 最後にfetchした結果                      |
| `CHERRY_PICK_HEAD`  | cherry-pick中               | cherry-pick対象のコミット               |
| `REVERT_HEAD`       | revert中                    | revert対象のコミット                     |
| `BISECT_HEAD`       | bisect中                    | 現在のbisectチェックポイント            |

```bash
# ORIG_HEADを使った操作の取り消し
$ git merge feature/auth
# マージを取り消したい場合:
$ git reset --hard ORIG_HEAD

# FETCH_HEADの確認
$ git fetch origin
$ cat .git/FETCH_HEAD
a1b2c3d4e5f6... branch 'main' of https://github.com/user/repo

# MERGE_HEADはマージ中のみ存在
$ git merge feature/auth
# コンフリクト発生中:
$ cat .git/MERGE_HEAD
f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4
# マージ完了後はファイルが削除される
```

---

## 2. HEADの仕組み

HEADはGitで最も重要なRefであり、**現在のチェックアウト位置**を示す。通常はブランチへのシンボリック参照だが、特定のコミットを直接指すこともある（detached HEAD）。

### 2.1 通常のHEAD（attached）

```
┌───────────────────────────────────────┐
│  .git/HEAD                            │
│  "ref: refs/heads/feature/auth"       │
│         │                             │
│         ▼                             │
│  .git/refs/heads/feature/auth         │
│  "c3d4e5f6..."                        │
│         │                             │
│         ▼                             │
│  commit c3d4e5f6...                   │
│    ├── tree ...                       │
│    ├── parent ...                     │
│    └── message: "Add login form"      │
└───────────────────────────────────────┘

新しいコミット時:
  1. 新commitオブジェクト作成（parent = c3d4e5f6...）
  2. refs/heads/feature/auth を新commitのSHA-1に更新
  3. HEADは refs/heads/feature/auth を指したまま
```

HEADがブランチを間接参照している状態では、`git commit`を実行すると**ブランチのポインタが自動的に前進**する。これがGitの通常の動作であり、ブランチが「成長する」仕組みの本質である。

```bash
# HEADの状態を確認するコマンド群
$ git symbolic-ref HEAD
refs/heads/feature/auth    # ブランチ名が返る = attached

$ git rev-parse HEAD
c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0a1b2    # SHA-1が返る

$ git rev-parse --abbrev-ref HEAD
feature/auth    # 短縮形のブランチ名

# コミット前後のブランチ位置の変化
$ git rev-parse feature/auth
c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0a1b2

$ echo "new content" >> file.txt && git add file.txt && git commit -m "update"

$ git rev-parse feature/auth
d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0a1b2c3    # 新しいSHA-1に更新
```

### 2.2 detached HEAD

detached HEADは、HEADがブランチではなく**特定のコミットを直接指している状態**である。

```bash
# detached HEADになる主な操作
$ git checkout a1b2c3d                # 特定コミットのチェックアウト
$ git checkout v1.0.0                 # タグのチェックアウト
$ git checkout origin/main            # リモート追跡ブランチのチェックアウト
$ git rebase --onto main feature HEAD~3  # rebase操作の途中

# HEADの状態確認
$ cat .git/HEAD
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# ← "ref:" プレフィックスがない = detached

$ git symbolic-ref HEAD
fatal: ref HEAD is not a symbolic ref
# ← symbolic-refはdetached HEADではエラーになる

$ git status
HEAD detached at a1b2c3d
# Gitはdetached HEAD状態を明確に表示する
```

```
┌────────────────────────────────────────┐
│  通常のHEAD（attached）               │
│                                        │
│  HEAD ──→ refs/heads/main ──→ commit   │
│                                        │
├────────────────────────────────────────┤
│  detached HEAD                         │
│                                        │
│  HEAD ──→ commit（直接参照）           │
│  refs/heads/main ──→ 別のcommit       │
│                                        │
│  ※ この状態で新コミットを作ると       │
│    どのブランチにも属さないコミット    │
│    が生成される（GC対象になりうる）    │
└────────────────────────────────────────┘
```

### 2.3 detached HEADの正しい活用シーン

detached HEADは必ずしも危険な状態ではない。以下のようなユースケースでは意図的に使用される。

```bash
# ユースケース1: 過去のコミットを一時的に確認する
$ git checkout v1.0.0
# テストを実行して過去のバージョンの動作を確認
$ make test
# 確認が終わったら元のブランチに戻る
$ git checkout main

# ユースケース2: CI/CDパイプラインでのタグチェックアウト
# Jenkins/GitHub Actionsなどでタグベースのビルドを行う
$ git checkout v2.1.0
$ docker build -t myapp:2.1.0 .

# ユースケース3: bisect中の自動チェックアウト
$ git bisect start
$ git bisect bad HEAD
$ git bisect good v1.0.0
# → Gitが自動的にdetached HEADで中間コミットをチェックアウト

# ユースケース4: worktreeでの一時的な作業
$ git worktree add /tmp/hotfix v1.0.0
# → worktreeはdetached HEADで作成可能
```

### 2.4 detached HEADからの復帰

```bash
# 方法1: 新しいブランチを作成して退避
$ git checkout -b rescue-branch
# → 現在のHEAD位置に新ブランチを作成し、attachedに戻る

# 方法2: 既存ブランチに戻る
$ git checkout main
# → detached HEAD中に作ったコミットがある場合、
#    reflogにのみ記録される（ブランチには属さない）

# 方法3: detached HEAD中に作ったコミットを救出
$ git reflog
# a1b2c3d HEAD@{0}: checkout: moving from main to a1b2c3d
# f5e6d7c HEAD@{1}: commit: important work in detached state
$ git branch rescue-branch f5e6d7c

# 方法4: Git 2.23以降のswitchコマンド
$ git switch main              # ブランチに戻る
$ git switch -c new-branch     # 新ブランチを作って戻る
$ git switch --detach v1.0.0   # 意図的にdetachする（明示的）

# 方法5: detached HEAD中の複数コミットをまとめて救出
$ git reflog
# abc1234 HEAD@{0}: commit: third fix
# def5678 HEAD@{1}: commit: second fix
# 789abcd HEAD@{2}: commit: first fix
# a1b2c3d HEAD@{3}: checkout: moving from main to a1b2c3d
$ git branch rescue-branch abc1234
# → abc1234から辿れる全コミット（first/second/third fix）が保護される
```

### 2.5 HEADの内部操作

`git update-ref`コマンドを使うことで、低レベルでRefを操作できる。通常のGitコマンドの裏側で実行されている処理を理解するのに役立つ。

```bash
# HEADが指すブランチを変更（git checkoutの内部動作に近い）
$ git symbolic-ref HEAD refs/heads/feature/auth

# ブランチを新しいコミットに更新（git commitの内部動作の一部）
$ git update-ref refs/heads/main a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# update-refは安全なref更新を行う
# - reflogエントリを自動作成
# - ロックファイル(.lock)を使用して並行アクセスを保護
$ ls .git/refs/heads/main.lock
# → update-ref実行中のみ存在する一時ファイル

# refの削除
$ git update-ref -d refs/heads/old-branch
# → refs/heads/old-branchファイルを削除し、reflogにも記録
```

---

## 3. ブランチの操作と内部動作

### 3.1 ブランチの作成・削除の内部動作

```bash
# ブランチ作成 = ファイル作成
$ git branch feature/new-ui
# → .git/refs/heads/feature/new-ui にHEADのSHA-1を書き込み
# → .git/logs/refs/heads/feature/new-ui にreflogエントリを作成

# 特定コミットからブランチ作成
$ git branch feature/from-tag v1.0.0
# → v1.0.0が指すSHA-1を書き込み

# ブランチ削除 = ファイル削除
$ git branch -d feature/new-ui
# → .git/refs/heads/feature/new-ui を削除
#    (commitオブジェクト自体は削除されない)
#    マージ済みでない場合はエラーになる

# 強制削除（マージ状態を確認しない）
$ git branch -D feature/new-ui
# → -d --force と同等、マージ済みでなくても削除

# ブランチ名の変更 = ファイルのリネーム + reflog更新
$ git branch -m old-name new-name
# → refs/heads/old-name → refs/heads/new-name
# → logs/refs/heads/old-name → logs/refs/heads/new-name
# → configのブランチ設定も更新
```

### 3.2 ブランチの内部操作を手動で再現する

Gitのブランチは本質的には「commitオブジェクトのSHA-1が書かれたテキストファイル」に過ぎない。この事実を確認するために、手動でブランチを操作してみる。

```bash
# 現在のHEADのSHA-1を確認
$ git rev-parse HEAD
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# 手動でブランチを作成（git branchの代替）
$ echo "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0" > .git/refs/heads/manual-branch
# → git branch -a で manual-branch が表示される

# ただし上記の方法はreflogが作成されず非推奨
# 正しい低レベル操作:
$ git update-ref refs/heads/manual-branch HEAD
# → reflogエントリも作成される

# 手動でブランチを移動（git reset --hardの内部動作に近い）
$ git update-ref refs/heads/main f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4
# → mainブランチが別のコミットを指すようになる

# ブランチのトラッキング設定
$ git branch --set-upstream-to=origin/main main
# → .git/config に以下が書き込まれる:
# [branch "main"]
#     remote = origin
#     merge = refs/heads/main
```

### 3.3 ブランチの階層構造（名前空間）

Gitのブランチ名にはスラッシュ（`/`）を含めることができ、ファイルシステム上ではディレクトリ階層として表現される。

```bash
# スラッシュを含むブランチ名
$ git branch feature/auth/login
$ git branch feature/auth/signup
$ git branch feature/ui/dashboard

# ファイルシステム上の構造
$ find .git/refs/heads -type f
.git/refs/heads/main
.git/refs/heads/feature/auth/login
.git/refs/heads/feature/auth/signup
.git/refs/heads/feature/ui/dashboard

# 注意: "feature/auth" というブランチと "feature/auth/login" は共存できない
# → "feature/auth" はファイルだが "feature/auth/" はディレクトリになるため
$ git branch feature/auth
fatal: cannot lock ref 'refs/heads/feature/auth':
  'refs/heads/feature/auth/login' exists; cannot create 'refs/heads/feature/auth'

# ブランチ一覧のフィルタリング
$ git branch --list 'feature/*'
  feature/auth/login
  feature/auth/signup
  feature/ui/dashboard

$ git branch --list 'feature/auth/*'
  feature/auth/login
  feature/auth/signup
```

### 3.4 リモート追跡ブランチ

```bash
# リモート追跡ブランチの一覧
$ git branch -r
  origin/main
  origin/develop
  origin/feature/auth
  upstream/main

# 全ブランチ（ローカル + リモート追跡）
$ git branch -a
  develop
  feature/auth
* main
  remotes/origin/develop
  remotes/origin/feature/auth
  remotes/origin/main
  remotes/upstream/main

# fetch時の動作
$ git fetch origin
# → refs/remotes/origin/* を更新
# → ローカルブランチは変更しない

# リモート追跡ブランチの更新ルール（refspec）
$ cat .git/config
[remote "origin"]
    url = https://github.com/user/repo.git
    fetch = +refs/heads/*:refs/remotes/origin/*

[remote "upstream"]
    url = https://github.com/upstream/repo.git
    fetch = +refs/heads/*:refs/remotes/upstream/*
```

```
┌─────────────────────────────────────────────────────┐
│                    refspec の構造                     │
│                                                     │
│  +refs/heads/*:refs/remotes/origin/*                │
│  │    │              │                              │
│  │    │              └── ローカル側のref             │
│  │    └── リモート側のref                            │
│  └── "+" = 非fast-forwardでも強制更新               │
│                                                     │
│  例: origin/main が更新された場合                    │
│  refs/heads/main (remote)                           │
│    → refs/remotes/origin/main (local)               │
└─────────────────────────────────────────────────────┘
```

### 3.5 高度なrefspec操作

```bash
# 特定ブランチのみfetch
$ git fetch origin main
# → refs/remotes/origin/main のみ更新

# カスタムrefspecでfetch
$ git fetch origin +refs/heads/release/*:refs/remotes/origin/release/*
# → release/ で始まるブランチのみ取得

# pushのrefspec
$ git push origin main:main
# → ローカルのmain をリモートの main にpush

$ git push origin feature/auth:refs/heads/feature/auth
# → 明示的なrefspec指定

# リモートブランチの削除
$ git push origin --delete feature/old
# → リモートの feature/old ブランチを削除
# → ローカルの refs/remotes/origin/feature/old も削除

# refspecでリモートブランチを削除する別の方法
$ git push origin :feature/old
# → "空" をfeature/oldにpush = 削除

# 不要になったリモート追跡ブランチの整理
$ git remote prune origin
# → リモートに存在しなくなったrefs/remotes/origin/*を削除

$ git fetch --prune origin
# → fetchと同時にpruneも実行（推奨設定）

# 自動pruneの設定
$ git config fetch.prune true
# → 毎回のfetchで自動的にpruneが実行される
```

### 3.6 上流ブランチ（upstream）の設定と活用

```bash
# 上流ブランチの設定
$ git branch --set-upstream-to=origin/main main
# または
$ git push -u origin feature/auth
# → push時に自動的に上流ブランチを設定

# 上流ブランチの確認
$ git branch -vv
* feature/auth abc1234 [origin/feature/auth: ahead 2] latest commit
  main         def5678 [origin/main] synced commit
  develop      789abcd [origin/develop: behind 3] older commit

# 上流ブランチとの差分確認
$ git log @{upstream}..HEAD    # ローカルにあってリモートにないコミット
$ git log HEAD..@{upstream}    # リモートにあってローカルにないコミット
$ git log @{upstream}...HEAD   # 双方の差分（対称差分）

# @{push}との違い（Git 2.5+）
$ git log @{push}..HEAD
# → pushする先のブランチとの差分（triangular workflowで有用）
# 例: fetchはupstreamから、pushはoriginへ、という運用
```

---

## 4. タグの内部表現

### 4.1 lightweight tag vs annotated tag

Gitのタグには2種類があり、内部表現が異なる。

```bash
# lightweight tag の作成
$ git tag v1.0.0-rc1
# → .git/refs/tags/v1.0.0-rc1 にcommitのSHA-1を直接保存
# → タグオブジェクトは作成されない

$ cat .git/refs/tags/v1.0.0-rc1
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

$ git cat-file -t v1.0.0-rc1
commit    # ← commitオブジェクトを直接指している

# annotated tag の作成
$ git tag -a v1.0.0 -m "Release version 1.0.0"
# → tagオブジェクトが作成される
# → .git/refs/tags/v1.0.0 にtagオブジェクトのSHA-1を保存

$ git cat-file -t v1.0.0
tag       # ← tagオブジェクトを指している

$ git cat-file -p v1.0.0
object a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
type commit
tag v1.0.0
tagger Gaku <gaku@example.com> 1707600000 +0900

Release version 1.0.0
```

```
┌────────────────────────────────────────────────────┐
│  lightweight tag                                    │
│                                                    │
│  refs/tags/v1.0.0-rc1 ──→ commit object            │
│  （SHA-1を直接保存）                                │
│                                                    │
├────────────────────────────────────────────────────┤
│  annotated tag                                      │
│                                                    │
│  refs/tags/v1.0.0 ──→ tag object ──→ commit object │
│  （tagオブジェクトを経由）                           │
│                                                    │
│  tag objectの内容:                                  │
│  - object: 対象commitのSHA-1                        │
│  - type: commit                                     │
│  - tag: タグ名                                      │
│  - tagger: 作成者情報                               │
│  - message: タグメッセージ                           │
│  - GPG signature（署名付きの場合）                   │
└────────────────────────────────────────────────────┘
```

### 4.2 署名付きタグ

```bash
# GPG署名付きタグの作成
$ git tag -s v1.0.0 -m "Signed release v1.0.0"
# → tagオブジェクトにGPG署名が含まれる

# 署名の検証
$ git tag -v v1.0.0
object a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
type commit
tag v1.0.0
tagger Gaku <gaku@example.com> 1707600000 +0900

Signed release v1.0.0
gpg: Signature made Mon 12 Feb 2024 10:00:00 AM JST
gpg: Good signature from "Gaku <gaku@example.com>"

# SSH署名（Git 2.34+）
$ git config gpg.format ssh
$ git config user.signingKey ~/.ssh/id_ed25519.pub
$ git tag -s v2.0.0 -m "SSH signed release v2.0.0"
```

### 4.3 タグのpush

タグはデフォルトでは`git push`でリモートに送信されない。明示的な操作が必要。

```bash
# 個別タグのpush
$ git push origin v1.0.0

# 全タグのpush
$ git push origin --tags
# → lightweight tag と annotated tag の両方がpushされる

# annotated tagのみpush（Git 2.4+）
$ git push origin --follow-tags
# → annotated tagのみ選択的にpush

# リモートのタグを削除
$ git push origin --delete v1.0.0
# または
$ git push origin :refs/tags/v1.0.0

# リモートからタグを再取得
$ git fetch origin --tags
# → ローカルに存在しないタグをリモートから取得
```

### 4.4 タグの"peeling"

packed-refsやfor-each-refでは、annotated tagが最終的に指すcommitのSHA-1も記録される。これを「peeling」と呼ぶ。

```bash
# peelされたタグの確認
$ git for-each-ref --format='%(refname) %(objectname:short) → %(objectname:short=,deref)' refs/tags/

# peel先のcommit SHA-1を直接取得
$ git rev-parse v1.0.0^{}
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# ^{} は tagオブジェクトを「剥がして」内部のcommitを返す

# packed-refsでのpeeled表現
$ cat .git/packed-refs
# pack-refs with: peeled fully-peeled sorted
f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4 refs/tags/v1.0.0
^a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# ^ から始まる行がpeeled SHA-1（tagが指すcommit）
```

---

## 5. reflog — 操作履歴の記録

### 5.1 reflogの基本

reflogは**refの変更履歴**を記録するローカル専用の仕組みである。`git clone`や`git push`では転送されない。リポジトリローカルの「操作日誌」であり、誤操作からの復旧の最後の手段となる。

```bash
# HEADのreflogを表示
$ git reflog
a1b2c3d HEAD@{0}: commit: feat: add authentication
f5e6d7c HEAD@{1}: checkout: moving from feature to main
b8c9d0e HEAD@{2}: commit: fix: typo in README
1234567 HEAD@{3}: merge feature/auth: Fast-forward
89abcde HEAD@{4}: commit: refactor: extract utils
fedcba0 HEAD@{5}: rebase (finish): returning to refs/heads/main
fedcba0 HEAD@{6}: rebase (pick): update config
1111111 HEAD@{7}: rebase (start): checkout origin/main

# 特定ブランチのreflog
$ git reflog show main
a1b2c3d main@{0}: commit: feat: add authentication
f5e6d7c main@{1}: merge feature/ui: Merge made by 'ort'
b8c9d0e main@{2}: commit: initial setup

# 日時指定でのアクセス
$ git show main@{2.days.ago}
$ git show HEAD@{2024-02-01}
$ git show HEAD@{yesterday}
$ git show main@{1.week.ago}

# reflogの詳細表示
$ git reflog --format='%C(auto)%h %gd %gs %ci'
a1b2c3d HEAD@{0} commit: feat: add authentication 2024-02-12 10:00:00 +0900
f5e6d7c HEAD@{1} checkout: moving from feature to main 2024-02-12 09:45:00 +0900

# reflogのdiff表示
$ git diff HEAD@{0} HEAD@{3}
# → 3操作前との差分を表示
```

### 5.2 reflogの保存場所と形式

```bash
# reflogファイルの確認
$ cat .git/logs/HEAD
# 各行: 旧SHA-1 新SHA-1 操作者 タイムスタンプ 操作内容

$ cat .git/logs/refs/heads/main
0000000... a1b2c3d... Gaku <gaku@example.com> 1707600000 +0900	commit (initial): first commit
a1b2c3d... f5e6d7c... Gaku <gaku@example.com> 1707603600 +0900	commit: second commit
f5e6d7c... b8c9d0e... Gaku <gaku@example.com> 1707607200 +0900	merge feature/auth: Merge made by 'ort'
```

```
┌───────────────────────────────────────────────────────────┐
│  reflogエントリの形式                                      │
│                                                           │
│  <旧SHA-1> <新SHA-1> <名前> <<メール>> <UNIXtime> <TZ>\t<メッセージ>  │
│                                                           │
│  例:                                                       │
│  a1b2c3d... f5e6d7c... Gaku <g@ex.com> 1707600000 +0900  │
│  \tcommit: add feature                                    │
│                                                           │
│  旧SHA-1が 0000000... の場合 = ブランチの新規作成          │
│  新SHA-1が 0000000... の場合 = ブランチの削除              │
└───────────────────────────────────────────────────────────┘
```

### 5.3 reflogの有効期限

| 種別                     | デフォルト期限 | 設定キー                     |
|--------------------------|----------------|------------------------------|
| 到達可能なエントリ       | 90日           | `gc.reflogExpire`            |
| 到達不可能なエントリ     | 30日           | `gc.reflogExpireUnreachable` |

```bash
# reflogの期限を変更
$ git config gc.reflogExpire "180 days"
$ git config gc.reflogExpireUnreachable "60 days"

# 特定のrefに対する個別設定
$ git config gc.main.reflogExpire "365 days"
# → mainブランチのreflogは1年間保持

# 手動でreflogを期限切れにする
$ git reflog expire --expire=now --all
# → 全refの全reflogエントリを即座に期限切れに（危険！）

# 特定のrefのreflogのみ期限切れにする
$ git reflog expire --expire=30.days.ago refs/heads/feature/old

# dry-runで確認
$ git reflog expire --expire=30.days.ago --dry-run --all
# → 実際には削除せず、削除対象を表示
```

### 5.4 reflogを使った復旧テクニック

```bash
# テクニック1: git reset --hard の取り消し
$ git reset --hard HEAD~3    # 直近3コミットを破棄
# "やっぱり元に戻したい"
$ git reflog
# a1b2c3d HEAD@{0}: reset: moving to HEAD~3
# f5e6d7c HEAD@{1}: commit: important commit 3
# b8c9d0e HEAD@{2}: commit: important commit 2
# 1234567 HEAD@{3}: commit: important commit 1
$ git reset --hard HEAD@{1}
# → reset前の状態に復帰

# テクニック2: 削除したブランチの復元
$ git branch -D feature/important
# "削除すべきではなかった"
$ git reflog
# → feature/importantの最後のコミットを探す
$ git branch feature/important HEAD@{2}

# テクニック3: 失敗したrebaseの取り消し
$ git rebase main
# コンフリクトだらけで収拾がつかない
$ git rebase --abort    # rebase中なら --abort が使える

# rebase完了後に元に戻したい場合
$ git reflog
# → rebase開始前のHEAD位置を探す
$ git reset --hard HEAD@{5}    # rebase前の状態に復帰

# テクニック4: amend前のコミットを取得
$ git commit --amend -m "corrected message"
# "amend前のコミットも保存しておきたい"
$ git reflog
# a1b2c3d HEAD@{0}: commit (amend): corrected message
# f5e6d7c HEAD@{1}: commit: original message
$ git branch backup-original f5e6d7c

# テクニック5: stash dropの復元
$ git stash drop stash@{0}
# "ドロップしたstashを取り戻したい"
$ git fsck --no-reflogs | grep commit
# dangling commit f5e6d7c...
$ git stash apply f5e6d7c
```

### 5.5 reflogとgit fsckの連携

reflogの期限が切れた後でも、GCが実行されるまではオブジェクト自体は残っている可能性がある。`git fsck`で到達不可能なオブジェクトを探索できる。

```bash
# 到達不可能なオブジェクトを探す
$ git fsck --unreachable
unreachable commit a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
unreachable blob f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4
unreachable tree 1234567890abcdef1234567890abcdef12345678

# 到達不可能なコミットを lost-found に保存
$ git fsck --lost-found
# → .git/lost-found/commit/ にコミットのSHA-1ファイルが作成される
# → .git/lost-found/other/ にその他のオブジェクトが保存される

# dangling object（どこからも参照されていないオブジェクト）の確認
$ git fsck --no-reflogs
# → reflogからの到達可能性を無視して判定
# → reflogでのみ保護されているオブジェクトも表示される

# 特定のdanglingコミットの内容を確認
$ git log --oneline --graph a1b2c3d4
$ git show a1b2c3d4
```

---

## 6. packed-refs

大量のrefがある場合、個別ファイルではなく`packed-refs`にまとめて性能を向上させる。

### 6.1 packed-refsの構造

```bash
# packed-refsの中身
$ cat .git/packed-refs
# pack-refs with: peeled fully-peeled sorted
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 refs/heads/main
b8c9d0e1f2a3b4c5d6e7f8a9b0a1b2c3d4e5f6a7 refs/heads/develop
f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4 refs/tags/v1.0.0
^a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
89abcdef0123456789abcdef0123456789abcdef refs/tags/v2.0.0
# ^ = peeled tag（tagオブジェクトが指すcommitのSHA-1）
# 行頭の # はコメント

# 手動でpackする
$ git pack-refs --all
# → 全てのloose refをpacked-refsに統合
# → アクティブなブランチ（現在のHEAD）のloose refは残る場合がある

# packのみ（looseを削除しない）
$ git pack-refs --no-prune
```

### 6.2 ref解決の優先順位と動作

```
┌─────────────────────────────────────────────────┐
│  ref解決の優先順位                               │
│                                                 │
│  1. .git/refs/heads/<name>  （loose ref）       │
│  2. .git/packed-refs 内の該当行                  │
│                                                 │
│  → looseが存在すればそちらが優先                 │
│  → ブランチ更新時は loose に書き込み             │
│  → git gc / pack-refs で packed に統合           │
│                                                 │
│  更新の流れ:                                     │
│  1. git commit → loose ref が更新される          │
│  2. looseとpackedの両方に同じrefが存在する場合    │
│     → looseが最新（packed側は古いまま）          │
│  3. git gc → loose を packed に統合              │
│     → loose ファイルを削除                       │
│  4. 次のcommit → 再びloose refが作成される       │
└─────────────────────────────────────────────────┘
```

### 6.3 packed-refsの性能への影響

```bash
# 大量のブランチ/タグがある場合の比較
# (例: 10,000タグのリポジトリ)

# looseオブジェクトの場合:
# → 10,000個のファイルがrefs/tags/に作成される
# → ファイルシステムのinode消費が激しい
# → ls-remote, branch -a 等が遅くなる

# packed-refsの場合:
# → 1ファイルに全タグが格納される
# → ファイルシステムの負荷が大幅に減少
# → 参照解決も高速化（sequential read vs random seek）

# パフォーマンス測定
$ time git for-each-ref refs/tags/ | wc -l
# packed-refs: ~0.01s
# loose refs:  ~0.5s  (10,000ファイルの場合)

# reftableフォーマット（Git 2.45+、実験的）
# → packed-refsのさらなる進化版
# → バイナリ形式で高速なルックアップが可能
# → JGit発のフォーマットをC実装に移植
$ git config core.repositoryFormatVersion 1
$ git config extensions.refStorage reftable
# → 注意: reftableは実験的機能であり、互換性リスクがある
```

---

## 7. シンボリック参照

シンボリック参照は**他のrefを間接参照するref**である。最も一般的な例はHEADであり、通常はブランチrefを指す。

### 7.1 基本操作

```bash
# HEADが最も一般的なシンボリック参照
$ git symbolic-ref HEAD
refs/heads/main

# カスタムシンボリック参照の作成
$ git symbolic-ref refs/custom/current refs/heads/feature/auth
# → refs/custom/current は feature/auth を間接参照する

# detached HEAD時はエラーになる
$ git symbolic-ref HEAD
fatal: ref HEAD is not a symbolic ref

# シンボリック参照の安全な確認（エラーを回避）
$ git symbolic-ref --quiet HEAD && echo "attached" || echo "detached"
```

### 7.2 シンボリック参照の実用例

```bash
# リモートのデフォルトブランチ（HEAD）
$ git remote show origin
# → "HEAD branch: main" と表示される
# → これは refs/remotes/origin/HEAD がシンボリック参照

$ cat .git/refs/remotes/origin/HEAD
ref: refs/remotes/origin/main

# origin/HEADを更新
$ git remote set-head origin develop
# → refs/remotes/origin/HEAD が refs/remotes/origin/develop を指すように変更

# origin/HEADを自動検出で設定
$ git remote set-head origin --auto
# → リモートに問い合わせてデフォルトブランチを自動設定

# ワークツリーでのHEAD
# メインワークツリー: .git/HEAD
# 追加ワークツリー: .git/worktrees/<name>/HEAD
$ git worktree add ../feature-worktree feature/auth
$ cat .git/worktrees/feature-worktree/HEAD
ref: refs/heads/feature/auth
```

---

## 8. ブランチ保護と運用パターン

### 8.1 ローカルでのブランチ保護

```bash
# receive.denyNonFastForwardsによる保護（共有リポジトリ）
$ git config receive.denyNonFastForwards true
# → 非fast-forwardのpushを全て拒否

# receive.denyDeletesによる削除防止
$ git config receive.denyDeletes true
# → ブランチ/タグの削除pushを拒否

# pre-receiveフックによるブランチ保護（サーバーサイド）
# .git/hooks/pre-receive:
#!/bin/bash
while read oldrev newrev refname; do
  if [ "$refname" = "refs/heads/main" ]; then
    # mainブランチへの直接pushを拒否
    echo "ERROR: Direct push to main is not allowed."
    echo "Please create a pull request instead."
    exit 1
  fi
done

# update フックによる個別ref制御
# .git/hooks/update:
#!/bin/bash
refname="$1"
oldrev="$2"
newrev="$3"
if [ "$refname" = "refs/heads/main" ] && \
   [ "$(git merge-base $oldrev $newrev)" != "$oldrev" ]; then
    echo "ERROR: Non-fast-forward push to main is not allowed."
    exit 1
fi
```

### 8.2 ブランチの整理と棚卸し

```bash
# マージ済みブランチの一覧
$ git branch --merged main
  feature/auth        # mainにマージ済み
  feature/old-ui      # mainにマージ済み
* main

# 未マージブランチの一覧
$ git branch --no-merged main
  feature/wip         # 進行中の作業

# マージ済みブランチを一括削除
$ git branch --merged main | grep -v '^\*' | grep -v 'main' | xargs git branch -d

# リモートで削除済みだがローカルに残っている追跡ブランチの確認
$ git remote prune origin --dry-run
Pruning origin
 * [would prune] origin/feature/deleted-remote

# 最終コミット日時でソートしたブランチ一覧
$ git for-each-ref --sort=-committerdate --format='%(committerdate:short) %(refname:short)' refs/heads/
2024-02-12 main
2024-02-10 feature/auth
2024-01-15 feature/old
2023-11-20 feature/ancient

# 3ヶ月以上更新のないブランチを検出するスクリプト
$ git for-each-ref --sort=committerdate --format='%(committerdate:unix) %(refname:short)' refs/heads/ | \
  while read timestamp branch; do
    if [ "$timestamp" -lt "$(date -d '3 months ago' +%s)" ]; then
      echo "Stale: $branch ($(date -d @$timestamp +%Y-%m-%d))"
    fi
  done
```

### 8.3 ブランチの命名規則

実務でよく使われるブランチ命名パターンとその内部動作への影響を整理する。

```
┌────────────────────────────────────────────────────────┐
│  ブランチ命名規則の例                                   │
│                                                        │
│  パターン          例                     用途          │
│  ──────────────────────────────────────────────────── │
│  feature/<name>    feature/user-auth      新機能        │
│  bugfix/<name>     bugfix/login-crash     バグ修正      │
│  hotfix/<name>     hotfix/security-patch  緊急修正      │
│  release/<ver>     release/2.1.0          リリース準備  │
│  chore/<name>      chore/update-deps      メンテナンス  │
│  refactor/<name>   refactor/auth-module   リファクタ    │
│                                                        │
│  注意: "feature" と "feature/x" は共存不可             │
│  → ディレクトリとファイルの衝突                         │
│  → 命名規則を決めたらチームで統一する                   │
└────────────────────────────────────────────────────────┘
```

```bash
# ブランチ名に使えない文字
# - スペース、~、^、:、?、*、[、\
# - ".." を含む名前
# - "." で始まる名前
# - "/" で終わる名前
# - ".lock" で終わる名前
# - ASCII制御文字

# ブランチ名のバリデーション
$ git check-ref-format --branch "feature/valid-name"
feature/valid-name    # 有効

$ git check-ref-format --branch "feature/invalid..name"
fatal: 'feature/invalid..name' is not a valid branch name
```

---

## 9. Refの並行アクセス制御

### 9.1 ロックファイルによる排他制御

Gitはref更新時にロックファイル（`.lock`サフィックス）を使用して並行アクセスを制御する。

```bash
# ロックの仕組み
# 1. refs/heads/main を更新する場合:
#    → .git/refs/heads/main.lock を作成（排他ロック取得）
#    → main.lock に新しいSHA-1を書き込み
#    → main.lock → main にアトミックにリネーム
#    → ロック解放

# ロック競合が発生した場合のエラー
$ git checkout feature/auth
error: Unable to create '/path/to/repo/.git/refs/heads/main.lock':
  File exists.
Another git process seems to be running in this repository.
If no other git process is running, remove the file manually.

# 強制ロック解除（他のgitプロセスが本当に動いていないことを確認してから）
$ rm .git/refs/heads/main.lock

# index.lockも同様の仕組み
$ rm .git/index.lock    # インデックスのロック解除
```

### 9.2 CAS（Compare-And-Swap）によるref更新

```bash
# update-refでのCAS操作
$ git update-ref refs/heads/main <new-sha1> <expected-old-sha1>
# → expected-old-sha1 と現在のSHA-1が一致する場合のみ更新
# → 一致しない場合はエラー（他のプロセスが先に更新した）

# pushでのCAS（--force-with-lease）
$ git push --force-with-lease origin main
# → リモートのmainが最後にfetchした時から変わっていない場合のみforce push
# → 他の開発者のpushを上書きするリスクを低減

# 期待値を明示するforce-with-lease
$ git push --force-with-lease=main:a1b2c3d origin main
# → リモートのmainが a1b2c3d の場合のみforce push
```

---

## 10. アンチパターン

### アンチパターン1: detached HEADでの長時間作業

```bash
# NG: detached HEAD状態で何日も作業を続ける
$ git checkout v1.0.0
# (detached HEAD)
$ ... 数日間の作業 ...
$ git commit -m "important changes"
# → ブランチに属さないコミットが作られる
# → git gcで消失する可能性がある

# OK: 必ずブランチを作成してから作業する
$ git checkout v1.0.0
$ git checkout -b hotfix/v1.0.0-patch
$ ... 作業 ...
$ git commit -m "important changes"
```

**理由**: detached HEADで作成されたコミットはどのrefからも到達不可能になった時点でGCの対象になる。reflogの期限（デフォルト30日）を過ぎると完全に消失する。

### アンチパターン2: reflogに依存した「バックアップ」戦略

```bash
# NG: "reflogがあるからreset --hardしても大丈夫"
$ git reset --hard HEAD~5
# → reflogには残るが、30-90日で期限切れ
# → git gcでオブジェクト自体が削除される可能性

# OK: 明示的にブランチやタグで保護する
$ git tag backup/before-cleanup HEAD
$ git reset --hard HEAD~5
# → tagがある限りGCされない
```

**理由**: reflogはローカル専用で、`git clone`や`git push`では転送されない。サーバー側にはreflogが存在しない場合もある。

### アンチパターン3: ブランチ名とタグ名の衝突

```bash
# NG: タグと同名のブランチを作成
$ git tag release-v1.0
$ git branch release-v1.0
# → 参照が曖昧になり、コマンドによって解決結果が異なる
# → checkout時は警告が出るが、他のコマンドでは暗黙的に片方が選ばれる

# OK: 命名規則でブランチとタグの名前空間を明確に分離
$ git tag v1.0.0                    # タグ: vX.Y.Z
$ git branch release/1.0.0         # ブランチ: release/X.Y.Z
```

**理由**: Gitのref解決順序（タグ → ブランチの順）により、同名のrefが存在すると予期しないrefが選択される可能性がある。

### アンチパターン4: packed-refsの手動編集

```bash
# NG: packed-refsファイルを直接テキストエディタで編集
$ vim .git/packed-refs
# → ソート順が崩れたり、チェックサムが不整合になる可能性

# OK: Gitコマンドを通じて操作
$ git update-ref refs/heads/main <new-sha1>
$ git pack-refs --all
```

**理由**: packed-refsは内部フォーマットの整合性（ソート順、peeled行の位置）が重要であり、手動編集は破損リスクが高い。

### アンチパターン5: 全ブランチの一括force push

```bash
# NG: 全ブランチを一括force push
$ git push --force --all origin
# → リモートの全ブランチを上書き
# → 他の開発者の作業が消失する可能性

# OK: 必要なブランチのみを個別にforce push
$ git push --force-with-lease origin feature/my-branch
# → 自分専用のブランチのみ、安全にforce push
```

**理由**: `--force --all`はリモートの全ブランチを無条件に上書きする。チーム開発では他のメンバーのpushを巻き戻す危険がある。`--force-with-lease`を使えば、他のpushがあった場合に拒否される。

---

## 11. 実務シナリオ集

### シナリオ1: ブランチの分岐点を調べる

```bash
# feature/authがmainから分岐した地点を特定
$ git merge-base main feature/auth
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# 分岐後のコミット数を確認
$ git rev-list --count main..feature/auth
5    # feature/authにあってmainにないコミット数

$ git rev-list --count feature/auth..main
3    # mainにあってfeature/authにないコミット数

# グラフで分岐を視覚化
$ git log --oneline --graph main feature/auth
* abc1234 (feature/auth) latest feature commit
* def5678 add feature logic
* 789abcd start feature
| * fedcba0 (main) latest main commit
| * 1111111 fix bug
| * 2222222 update docs
|/
* a1b2c3d common ancestor (merge-base)
```

### シナリオ2: 複数のリモートからの同期

```bash
# フォークしたリポジトリで、上流の変更を取り込む
$ git remote add upstream https://github.com/original/repo.git
$ git fetch upstream
$ git checkout main
$ git merge upstream/main

# 複数リモートのref状況を一覧
$ git for-each-ref --format='%(refname:short) %(upstream:short) %(upstream:track)' refs/heads/
main origin/main [ahead 0, behind 0]
feature/auth origin/feature/auth [ahead 2]
```

### シナリオ3: refs/notesの活用

```bash
# コミットにメモ（note）を追加
$ git notes add -m "このコミットはパフォーマンスに影響あり" abc1234
# → refs/notes/commits にnoteオブジェクトが保存される

# noteの表示
$ git log --show-notes abc1234
commit abc1234...
    fix: update query
Notes:
    このコミットはパフォーマンスに影響あり

# noteのpush（明示的に指定が必要）
$ git push origin refs/notes/commits
```

### シナリオ4: replace refによるコミットの差し替え

```bash
# コミットメッセージを修正したいが、pushbackはしたくない場合
$ git replace abc1234 def5678
# → refs/replace/abc1234 が作成される
# → abc1234へのアクセスがdef5678に透過的に差し替えられる

# replaceの確認
$ git replace -l
abc1234

# replaceの削除
$ git replace -d abc1234

# replaceされたコミットを表示
$ git log --no-replace-objects abc1234
# → 元のコミットが表示される（replace無視）
```

---

## 12. FAQ

### Q1. `git branch -d`で削除したブランチを復元できるか？

**A1.** はい、reflogを使えば復元できます。

```bash
$ git reflog
# 削除前のブランチが指していたcommit SHA-1を見つける
$ git branch recovered-branch <SHA-1>
```

ただし、reflogの有効期限内に限ります。期限切れ後は`git fsck --lost-found`で到達不可能オブジェクトから探す必要があります。

### Q2. HEADとORIG_HEADの違いは何か？

**A2.** `HEAD`は現在のチェックアウト位置を指すシンボリック参照です。`ORIG_HEAD`は`merge`、`rebase`、`reset`など**HEADを大きく移動させる操作の直前の位置**を記録する特殊参照です。操作を取り消したい場合に`git reset --hard ORIG_HEAD`のように使用します。

### Q3. `refs/stash`はどのような仕組みか？

**A3.** `git stash`はワーキングディレクトリの変更をcommitオブジェクトとして保存し、`refs/stash`がその最新のstashエントリを指します。過去のstashエントリはreflog（`stash@{0}`, `stash@{1}`, ...）として保持されます。内部的には通常のcommit/tree/blobオブジェクトで構成されています。

```bash
# stashの内部構造を確認
$ git cat-file -p refs/stash
tree d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0a1b2c3
parent a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
parent f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4
author Gaku <gaku@example.com> 1707600000 +0900
committer Gaku <gaku@example.com> 1707600000 +0900

WIP on main: a1b2c3d commit message

# stashコミットは2つ（またはuntracked含め3つ）のparentを持つ
# parent 1: stash時のHEADコミット
# parent 2: インデックスの状態を記録したコミット
# parent 3: --include-untracked時のuntracked filesコミット
```

### Q4. SHA-1からSHA-256への移行はrefにどのような影響があるか？

**A4.** Git 2.29以降、SHA-256をオブジェクトハッシュとして使用する実験的サポートが追加されています。SHA-256リポジトリでは、refファイルに格納されるハッシュが64文字（40文字ではなく）になります。ただし、SHA-1とSHA-256のリポジトリ間の相互運用性は限定的であり、実務での移行はまだ先の話です。

```bash
# SHA-256リポジトリの作成（実験的）
$ git init --object-format=sha256 my-repo
$ cd my-repo
$ echo "test" | git hash-object --stdin
# → 64文字のSHA-256ハッシュが返る
```

### Q5. refの操作をフックで監視する方法は？

**A5.** `reference-transaction`フック（Git 2.28+）を使うと、全てのref更新をフックで監視できます。

```bash
# .git/hooks/reference-transaction の例
#!/bin/bash
# $1 = "prepared" | "committed" | "aborted"
while read oldvalue newvalue refname; do
  if [ "$1" = "committed" ]; then
    echo "Ref updated: $refname $oldvalue -> $newvalue" >> /tmp/git-ref-log.txt
  fi
done
```

### Q6. git worktreeとHEADの関係は？

**A6.** 各worktreeは独自のHEADを持ちます。メインのworktreeは`.git/HEAD`を使い、追加worktreeは`.git/worktrees/<name>/HEAD`を使います。重要な制約として、**複数のworktreeで同じブランチをチェックアウトすることはできません**。

```bash
# worktree追加
$ git worktree add ../feature-wt feature/auth
# → .git/worktrees/feature-wt/HEAD = "ref: refs/heads/feature/auth"

# 同じブランチをチェックアウトしようとするとエラー
$ git worktree add ../another-wt feature/auth
fatal: 'feature/auth' is already checked out at '/path/to/feature-wt'

# worktree一覧
$ git worktree list
/path/to/main      a1b2c3d [main]
/path/to/feature-wt f5e6d7c [feature/auth]
```

---

## まとめ

| 概念                   | 要点                                                          |
|------------------------|---------------------------------------------------------------|
| Ref                    | SHA-1ハッシュへのポインタ、`.git/refs/`配下にテキストファイル |
| ブランチ               | `refs/heads/<name>` に保存、作成・削除はファイル操作         |
| HEAD                   | シンボリック参照、通常はブランチを間接参照                    |
| detached HEAD          | HEADがcommitを直接参照、ブランチなしで危険                   |
| reflog                 | refの変更履歴、ローカル専用、30-90日で期限切れ               |
| packed-refs            | 大量refの最適化、looseが優先                                 |
| リモート追跡ブランチ   | `refs/remotes/<remote>/<branch>`、fetch時に更新              |
| lightweight tag        | commitを直接指すref、メタデータなし                           |
| annotated tag          | tagオブジェクトを経由、作成者・メッセージ・署名を格納        |
| シンボリック参照       | 他のrefへの間接参照、HEADが代表例                            |
| ロックファイル         | `.lock`サフィックスで並行アクセスを排他制御                   |
| ORIG_HEAD              | 破壊的操作前のHEAD位置を記録、取り消しに使用                |

---

## 次に読むべきガイド

- [Gitオブジェクトモデル](./00-git-object-model.md) — blob/tree/commit/tagの基礎
- [マージアルゴリズム](./02-merge-algorithms.md) — 3-way mergeとortの内部動作
- [Packfile/GC](./03-packfile-gc.md) — オブジェクトの圧縮とガベージコレクション
- [インタラクティブRebase](../01-advanced-git/00-interactive-rebase.md) — HEADの書き換え操作

---

## 参考文献

1. **Pro Git Book** — Scott Chacon, Ben Straub "Git Internals - Git References" https://git-scm.com/book/en/v2/Git-Internals-Git-References
2. **Git公式ドキュメント** — `git-symbolic-ref`, `git-reflog`, `git-update-ref`, `git-for-each-ref` https://git-scm.com/docs
3. **GitHub Blog** — "Commits are snapshots, not diffs" https://github.blog/2020-12-17-commits-are-snapshots-not-diffs/
4. **Git公式ドキュメント** — `git-pack-refs`, `git-check-ref-format` https://git-scm.com/docs
5. **Derrick Stolee** — "Scaling monorepo maintenance" https://github.blog/2021-04-29-scaling-monorepo-maintenance/
6. **Git Reference Transaction Hook** — https://git-scm.com/docs/githooks#_reference_transaction
7. **reftable specification** — https://www.git-scm.com/docs/reftable
