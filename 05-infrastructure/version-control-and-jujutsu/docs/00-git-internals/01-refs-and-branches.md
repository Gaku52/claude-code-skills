# Ref・ブランチ

> GitのRef（参照）機構を深堀りし、HEAD、ブランチ、タグ、reflogの内部表現とdetached HEAD状態の正しい理解・復旧方法を解説する。

## この章で学ぶこと

1. **Refの種類と内部表現** — ブランチ、タグ、リモート追跡ブランチがファイルシステム上でどう管理されるか
2. **HEADの仕組みとdetached HEAD** — シンボリック参照の動作原理と安全な運用方法
3. **reflogによる履歴復元** — 失われたコミットの追跡と救出テクニック

---

## 1. Refとは何か

Refは**SHA-1ハッシュへのポインタ**であり、`.git/refs/`配下のテキストファイルとして保存される。

```
.git/
├── HEAD                          ← シンボリック参照
├── refs/
│   ├── heads/                    ← ローカルブランチ
│   │   ├── main                  ← "main"ブランチ
│   │   └── feature/auth          ← "feature/auth"ブランチ
│   ├── tags/                     ← タグ
│   │   ├── v1.0.0
│   │   └── v2.0.0
│   └── remotes/                  ← リモート追跡ブランチ
│       └── origin/
│           ├── main
│           └── feature/auth
├── packed-refs                   ← pack済みref（最適化）
└── logs/                         ← reflog
    ├── HEAD
    └── refs/
        └── heads/
            └── main
```

```bash
# ブランチの実体を確認
$ cat .git/refs/heads/main
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0

# HEADの実体を確認（シンボリック参照）
$ cat .git/HEAD
ref: refs/heads/main
```

---

## 2. HEADの仕組み

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

### 2.2 detached HEAD

```bash
# 特定のコミットに直接チェックアウト
$ git checkout a1b2c3d
# → HEADがブランチではなくcommitを直接指す

$ cat .git/HEAD
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# ← "ref:" プレフィックスがない = detached
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

### 2.3 detached HEADからの復帰

```bash
# 方法1: 新しいブランチを作成して退避
$ git checkout -b rescue-branch

# 方法2: 既存ブランチに戻る
$ git checkout main

# 方法3: detached HEAD中に作ったコミットを救出
$ git reflog
# a1b2c3d HEAD@{0}: checkout: moving from main to a1b2c3d
# f5e6d7c HEAD@{1}: commit: important work in detached state
$ git branch rescue-branch f5e6d7c
```

---

## 3. ブランチの操作と内部動作

### 3.1 ブランチの作成・削除の内部動作

```bash
# ブランチ作成 = ファイル作成
$ git branch feature/new-ui
# → .git/refs/heads/feature/new-ui にHEADのSHA-1を書き込み

# ブランチ削除 = ファイル削除
$ git branch -d feature/new-ui
# → .git/refs/heads/feature/new-ui を削除
#    (commitオブジェクト自体は削除されない)

# ブランチ名の変更 = ファイルのリネーム
$ git branch -m old-name new-name
# → refs/heads/old-name → refs/heads/new-name
```

### 3.2 リモート追跡ブランチ

```bash
# リモート追跡ブランチの一覧
$ git branch -r
  origin/main
  origin/feature/auth
  upstream/main

# fetch時の動作
$ git fetch origin
# → refs/remotes/origin/* を更新
# → ローカルブランチは変更しない

# リモート追跡ブランチの更新ルール（refspec）
$ cat .git/config
[remote "origin"]
    url = https://github.com/user/repo.git
    fetch = +refs/heads/*:refs/remotes/origin/*
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

---

## 4. reflog — 操作履歴の記録

### 4.1 reflogの基本

```bash
# HEADのreflogを表示
$ git reflog
a1b2c3d HEAD@{0}: commit: feat: add authentication
f5e6d7c HEAD@{1}: checkout: moving from feature to main
b8c9d0e HEAD@{2}: commit: fix: typo in README
1234567 HEAD@{3}: merge feature/auth: Fast-forward

# 特定ブランチのreflog
$ git reflog show main
a1b2c3d main@{0}: commit: feat: add authentication
f5e6d7c main@{1}: merge feature/ui: Merge made by 'ort'

# 日時指定でのアクセス
$ git show main@{2.days.ago}
$ git show HEAD@{2024-02-01}
```

### 4.2 reflogの保存場所

```bash
# reflogファイルの確認
$ cat .git/logs/HEAD
# 各行: 旧SHA-1 新SHA-1 操作者 タイムスタンプ 操作内容

$ cat .git/logs/refs/heads/main
0000000... a1b2c3d... Gaku <gaku@example.com> 1707600000 +0900	commit (initial): first commit
a1b2c3d... f5e6d7c... Gaku <gaku@example.com> 1707603600 +0900	commit: second commit
```

### 4.3 reflogの有効期限

| 種別                     | デフォルト期限 | 設定キー                     |
|--------------------------|----------------|------------------------------|
| 到達可能なエントリ       | 90日           | `gc.reflogExpire`            |
| 到達不可能なエントリ     | 30日           | `gc.reflogExpireUnreachable` |

```bash
# reflogの期限を変更
$ git config gc.reflogExpire "180 days"
$ git config gc.reflogExpireUnreachable "60 days"

# 手動でreflogを期限切れにする
$ git reflog expire --expire=now --all
```

---

## 5. packed-refs

大量のrefがある場合、個別ファイルではなく`packed-refs`にまとめて性能を向上させる。

```bash
# packed-refsの中身
$ cat .git/packed-refs
# pack-refs with: peeled fully-peeled sorted
a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0 refs/heads/main
f5e6d7c8b9a0e1f2d3c4b5a6d7e8f9a0b1c2d3e4 refs/tags/v1.0.0
^a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0
# ^ = peeled tag（tagが指すcommitのSHA-1）

# 手動でpackする
$ git pack-refs --all
```

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
└─────────────────────────────────────────────────┘
```

---

## 6. シンボリック参照

```bash
# HEADが最も一般的なシンボリック参照
$ git symbolic-ref HEAD
refs/heads/main

# カスタムシンボリック参照の作成
$ git symbolic-ref refs/custom/current refs/heads/feature/auth

# detached HEAD時はエラーになる
$ git symbolic-ref HEAD
fatal: ref HEAD is not a symbolic ref
```

---

## 7. アンチパターン

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

---

## 8. FAQ

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

---

## 次に読むべきガイド

- [Gitオブジェクトモデル](./00-git-object-model.md) — blob/tree/commit/tagの基礎
- [マージアルゴリズム](./02-merge-algorithms.md) — 3-way mergeとortの内部動作
- [インタラクティブRebase](../01-advanced-git/00-interactive-rebase.md) — HEADの書き換え操作

---

## 参考文献

1. **Pro Git Book** — Scott Chacon, Ben Straub "Git Internals - Git References" https://git-scm.com/book/en/v2/Git-Internals-Git-References
2. **Git公式ドキュメント** — `git-symbolic-ref`, `git-reflog`, `git-update-ref` https://git-scm.com/docs
3. **GitHub Blog** — "Commits are snapshots, not diffs" https://github.blog/2020-12-17-commits-are-snapshots-not-diffs/
