# Worktree/Submodule

> `git worktree`による複数作業ディレクトリの管理と、`git submodule`による外部リポジトリの統合手法を解説し、大規模プロジェクトでの効率的な運用方法を習得する。

## この章で学ぶこと

1. **git worktreeの仕組みと活用法** -- 1つのリポジトリで複数のブランチを同時にチェックアウトする手法
2. **git submoduleの内部構造と運用** -- 外部リポジトリの依存管理とバージョン固定の仕組み
3. **代替手段との比較** -- subtree merge、モノレポ、パッケージマネージャーとの使い分け
4. **大規模プロジェクトでのベストプラクティス** -- CI/CD連携、チーム運用、トラブルシューティング


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [インタラクティブRebase](./00-interactive-rebase.md) の内容を理解していること

---

## 1. git worktree

### 1.1 worktreeとは

1つの`.git`ディレクトリを共有しながら、**複数のブランチを別々のディレクトリに同時チェックアウト**できる機能。

```bash
# worktreeの追加
$ git worktree add ../hotfix-v1 hotfix/v1.0.1
# → ../hotfix-v1 ディレクトリに hotfix/v1.0.1 をチェックアウト

# 新しいブランチを作成しつつworktreeを追加
$ git worktree add -b feature/new-ui ../new-ui main
# → main から feature/new-ui を作成し、../new-ui にチェックアウト

# worktreeの一覧
$ git worktree list
/home/user/project          abc1234 [main]
/home/user/hotfix-v1        def5678 [hotfix/v1.0.1]
/home/user/new-ui           789abcd [feature/new-ui]

# 詳細表示（porcelain形式）
$ git worktree list --porcelain
worktree /home/user/project
HEAD abc1234567890abcdef1234567890abcdef123456
branch refs/heads/main

worktree /home/user/hotfix-v1
HEAD def567890abcdef1234567890abcdef1234567890
branch refs/heads/hotfix/v1.0.1
```

```
┌──────────────────────────────────────────────────────┐
│  worktree のディレクトリ構造                          │
│                                                      │
│  /home/user/                                         │
│  ├── project/              ← メインworktree          │
│  │   ├── .git/             ← 共有オブジェクトDB      │
│  │   │   ├── objects/                                │
│  │   │   ├── refs/                                   │
│  │   │   ├── worktrees/                              │
│  │   │   │   ├── hotfix-v1/   ← worktree固有情報    │
│  │   │   │   │   ├── HEAD                            │
│  │   │   │   │   ├── index                           │
│  │   │   │   │   └── gitdir                          │
│  │   │   │   └── new-ui/      ← worktree固有情報    │
│  │   │   │       ├── HEAD                            │
│  │   │   │       ├── index                           │
│  │   │   │       └── gitdir                          │
│  │   │   └── ...                                     │
│  │   └── src/              ← mainの作業ファイル      │
│  │                                                   │
│  ├── hotfix-v1/            ← linked worktree         │
│  │   ├── .git              ← テキストファイル(パス)  │
│  │   └── src/              ← hotfixの作業ファイル    │
│  │                                                   │
│  └── new-ui/               ← linked worktree         │
│      ├── .git              ← テキストファイル(パス)  │
│      └── src/              ← new-uiの作業ファイル    │
└──────────────────────────────────────────────────────┘
```

### 1.2 worktreeの管理

```bash
# worktreeの削除
$ git worktree remove ../hotfix-v1
# → ディレクトリを削除し、.git/worktrees/ からも削除

# 未コミットの変更がある場合は強制削除
$ git worktree remove --force ../hotfix-v1

# 手動でディレクトリを削除した場合のクリーンアップ
$ rm -rf ../hotfix-v1
$ git worktree prune
# → 存在しないworktreeの参照を削除

# worktreeをロック（自動pruneの防止）
$ git worktree lock ../new-ui --reason "長期作業中"
$ git worktree unlock ../new-ui

# worktreeの移動
$ git worktree move ../new-ui ../new-ui-v2
# → ディレクトリ名を変更（Git 2.17+）
```

### 1.3 worktreeの内部構造

```bash
# linked worktreeの .git ファイルの中身
$ cat ../hotfix-v1/.git
gitdir: /home/user/project/.git/worktrees/hotfix-v1

# メインリポジトリ側のworktree情報
$ cat /home/user/project/.git/worktrees/hotfix-v1/gitdir
/home/user/hotfix-v1/.git

# worktree固有のHEAD
$ cat /home/user/project/.git/worktrees/hotfix-v1/HEAD
ref: refs/heads/hotfix/v1.0.1

# worktree固有のindex（ステージング情報）
$ ls -la /home/user/project/.git/worktrees/hotfix-v1/index
```

```
┌──────────────────────────────────────────────────────┐
│  worktree間で共有されるもの / 共有されないもの         │
│                                                      │
│  共有される:                                         │
│  ├── .git/objects/     ← 全オブジェクト              │
│  ├── .git/refs/        ← 全ブランチ・タグ            │
│  ├── .git/config       ← リポジトリ設定              │
│  ├── .git/hooks/       ← フックスクリプト            │
│  └── .git/info/        ← exclude等                   │
│                                                      │
│  共有されない（worktree固有）:                       │
│  ├── HEAD              ← 現在のブランチ              │
│  ├── index             ← ステージング状態            │
│  ├── MERGE_HEAD        ← マージ中の状態              │
│  ├── REBASE_HEAD       ← rebase中の状態              │
│  └── 作業ディレクトリ  ← 実際のファイル              │
└──────────────────────────────────────────────────────┘
```

### 1.4 worktreeの活用パターン

```bash
# パターン1: PRレビュー中に別の作業をする
$ git worktree add ../review-pr-42 origin/feature/pr-42
$ cd ../review-pr-42
$ npm install && npm test
$ cd ../project
# → メインの作業ディレクトリを汚さずにレビュー

# パターン2: ビルドの同時実行
$ git worktree add ../build-release release/v2.0
$ cd ../build-release && npm run build &
$ cd ../project && npm run dev
# → リリースビルドと開発サーバーを同時実行

# パターン3: 複数バージョンの動作比較
$ git worktree add ../v1 v1.0.0
$ git worktree add ../v2 v2.0.0
# → 2つのバージョンを並べて動作確認

# パターン4: 緊急のhotfix対応
$ git worktree add -b hotfix/critical ../hotfix main
$ cd ../hotfix
# ... 修正作業 ...
$ git commit -m "fix: critical security issue"
$ git push origin hotfix/critical
$ cd ../project
$ git worktree remove ../hotfix
# → メインの開発作業を中断せずにhotfixを完了

# パターン5: ドキュメントの同時編集
$ git worktree add ../docs-edit docs/main
$ cd ../docs-edit
# → ドキュメント専用のworktreeで作業
# → メインの開発worktreeのnpm install等の影響を受けない

# パターン6: CI用のクリーンビルド
$ git worktree add --detach ../ci-build HEAD
$ cd ../ci-build
$ npm ci && npm run build && npm test
$ cd ../project
$ git worktree remove ../ci-build
# → クリーンな状態でビルド・テストを実行
```

### 1.5 worktreeとブランチ操作

```bash
# worktree内でのブランチ操作
$ cd ../hotfix-v1
$ git branch                    # 全ブランチを表示（全worktreeで共通）
$ git fetch origin              # フェッチ（全worktreeに反映）
$ git stash                     # このworktreeの変更をstash

# worktreeでチェックアウトできないケース
$ git worktree add ../test main
# fatal: 'main' is already checked out at '/home/user/project'
# → 同じブランチを複数worktreeでチェックアウトすることは不可

# 回避策: detached HEADで同じコミットを参照
$ git worktree add --detach ../test HEAD
# → ブランチではなくコミットを直接チェックアウト
```

### 1.6 worktreeの制約と注意点

| 制約                           | 説明                                          |
|-------------------------------|-----------------------------------------------|
| 同一ブランチの重複チェックアウト | 同じブランチを複数worktreeでチェックアウト不可 |
| ベアリポジトリ                 | worktreeの追加は可能だがメインworktreeがない    |
| サブモジュール                 | worktreeごとにサブモジュールの初期化が必要      |
| GC                            | メインworktreeの.git/objectsを共有              |
| node_modules                  | worktreeごとにnpm installが必要                |
| IDE設定                        | worktreeごとに.ideaや.vscode設定が必要         |

```bash
# worktreeでサブモジュールを初期化する例
$ git worktree add ../review origin/feature/review
$ cd ../review
$ git submodule update --init --recursive
# → worktreeごとにサブモジュールの初期化が必要

# worktreeで依存関係をインストールする例
$ git worktree add ../test-branch test-branch
$ cd ../test-branch
$ npm install
# → node_modulesはworktreeごとに独立
```

### 1.7 worktreeを使ったスクリプト自動化

```bash
#!/bin/bash
# review-pr.sh - PRレビュー用worktreeを自動作成
set -euo pipefail

PR_NUMBER=$1
BRANCH="origin/pr/${PR_NUMBER}"
WORKTREE_DIR="../review-pr-${PR_NUMBER}"

# リモートの最新を取得
git fetch origin

# worktreeを作成
git worktree add "$WORKTREE_DIR" "$BRANCH"

# 依存関係のインストールとテスト
cd "$WORKTREE_DIR"
if [ -f package.json ]; then
    npm install
    npm test
fi

echo "Review worktree created at: $WORKTREE_DIR"
echo "To clean up: git worktree remove $WORKTREE_DIR"
```

```bash
#!/bin/bash
# cleanup-worktrees.sh - 不要なworktreeを一括削除
set -euo pipefail

echo "Current worktrees:"
git worktree list

# マージ済みブランチのworktreeを検出
git worktree list --porcelain | while read -r line; do
    if [[ "$line" == "branch refs/heads/"* ]]; then
        branch="${line#branch refs/heads/}"
        if git branch --merged main | grep -q "$branch" && [ "$branch" != "main" ]; then
            echo "Removing merged worktree for branch: $branch"
            worktree_path=$(git worktree list | grep "$branch" | awk '{print $1}')
            git worktree remove "$worktree_path" 2>/dev/null || true
        fi
    fi
done

# 存在しないworktreeのクリーンアップ
git worktree prune
echo "Cleanup complete."
```

---

## 2. git submodule

### 2.1 submoduleの基本

```bash
# サブモジュールの追加
$ git submodule add https://github.com/lib/utils.git vendor/utils
# → .gitmodules ファイルが作成される
# → vendor/utils/ にリポジトリがクローンされる
# → 特定のcommit SHA-1がインデックスに記録される

# .gitmodulesの内容
$ cat .gitmodules
[submodule "vendor/utils"]
    path = vendor/utils
    url = https://github.com/lib/utils.git
```

```
┌─────────────────────────────────────────────────────┐
│  submodule の仕組み                                  │
│                                                     │
│  親リポジトリのtreeオブジェクト:                     │
│  100644 blob abc123  .gitmodules                    │
│  100644 blob def456  README.md                      │
│  160000 commit 789abc vendor/utils  ← commitを参照! │
│         ^^^^^^                                      │
│         mode 160000 = submodule                     │
│                                                     │
│  → 親リポジトリは vendor/utils の特定commitを記録   │
│  → vendor/utils/ 内部は独立したリポジトリ           │
│  → .gitmodules にURLとパスのマッピングを保持        │
└─────────────────────────────────────────────────────┘
```

### 2.2 submoduleの初期化とクローン

```bash
# クローン時にサブモジュールも取得
$ git clone --recurse-submodules https://github.com/user/project.git

# クローン後にサブモジュールを初期化
$ git submodule init
$ git submodule update
# または一括で
$ git submodule update --init --recursive

# 全サブモジュールの状態確認
$ git submodule status
 789abcdef1234567890abcdef1234567890abcdef vendor/utils (v2.3.0)
+fedcba9876543210fedcba9876543210fedcba98 vendor/auth (heads/main)
-0123456789abcdef0123456789abcdef01234567 vendor/ui
```

**ステータスマーカーの意味**:

| マーカー | 意味                                              |
|----------|---------------------------------------------------|
| (空白)   | 記録されたcommitにチェックアウト済み               |
| `+`      | 記録と異なるcommitにチェックアウトされている       |
| `-`      | 未初期化                                           |
| `U`      | マージコンフリクト中                               |

### 2.3 submoduleの更新

```bash
# 親リポジトリが記録しているcommitに合わせる
$ git submodule update
# → detached HEAD状態になる

# リモートの最新を取得してサブモジュールを更新
$ git submodule update --remote
# → .gitmodulesのbranch設定（デフォルトmain）の最新commitに更新
# → 親リポジトリのインデックスも更新される

# 特定のサブモジュールだけ更新
$ git submodule update --remote vendor/utils
$ git add vendor/utils
$ git commit -m "chore: update vendor/utils to latest"

# マージ戦略を指定して更新
$ git submodule update --remote --merge
# → 現在のブランチにリモートの変更をmerge

$ git submodule update --remote --rebase
# → 現在の作業をリモートの最新に対してrebase
```

```
┌────────────────────────────────────────────────────┐
│  submodule update のフロー                          │
│                                                    │
│  git submodule update (--remote なし):             │
│  1. 親リポジトリの記録commitを読む                 │
│  2. サブモジュールをそのcommitにcheckout           │
│  → 常に "固定されたバージョン" になる              │
│                                                    │
│  git submodule update --remote:                    │
│  1. サブモジュールのリモートからfetch              │
│  2. 設定されたブランチの最新commitを取得           │
│  3. サブモジュールをそのcommitにcheckout           │
│  4. 親リポジトリのインデックスを更新               │
│  → "最新バージョン" に追従する                     │
│                                                    │
│  git submodule update --remote --merge:            │
│  1. リモートからfetch                              │
│  2. 現在のブランチにmerge                          │
│  → サブモジュール内でブランチ作業中に有効          │
└────────────────────────────────────────────────────┘
```

### 2.4 submodule内での開発

```bash
# サブモジュール内で作業する場合
$ cd vendor/utils
$ git checkout main               # detached HEADからブランチに切替
$ vim src/index.js                # 修正
$ git add . && git commit -m "fix: bug in utils"
$ git push origin main            # サブモジュールのリモートにpush

# 親リポジトリに戻って記録を更新
$ cd ../..
$ git add vendor/utils
$ git commit -m "chore: update vendor/utils submodule"
```

### 2.5 submoduleの特定バージョンへの固定

```bash
# 特定のタグにサブモジュールを固定
$ cd vendor/utils
$ git fetch --tags
$ git checkout v2.3.0
$ cd ../..
$ git add vendor/utils
$ git commit -m "chore: pin vendor/utils to v2.3.0"

# 特定のコミットに固定
$ cd vendor/utils
$ git checkout abc123def456
$ cd ../..
$ git add vendor/utils
$ git commit -m "chore: pin vendor/utils to known-good commit"

# ブランチのHEADに追従する設定
$ git config -f .gitmodules submodule.vendor/utils.branch develop
$ git submodule update --remote vendor/utils
```

### 2.6 submoduleの削除

```bash
# サブモジュールの完全な削除（3段階必要）
$ git submodule deinit -f vendor/utils   # 1. 設定の無効化
$ git rm -f vendor/utils                  # 2. ファイルとインデックスから削除
$ rm -rf .git/modules/vendor/utils        # 3. キャッシュの削除
$ git commit -m "chore: remove vendor/utils submodule"
```

```
┌──────────────────────────────────────────────────────┐
│  submodule削除時に影響を受けるファイル/ディレクトリ   │
│                                                      │
│  1. .gitmodules             ← サブモジュールの設定   │
│  2. .git/config             ← ローカル設定           │
│  3. .git/modules/<path>/   ← キャッシュされたリポジトリ│
│  4. <path>/                ← 実際のファイル           │
│  5. インデックス            ← mode 160000 のエントリ │
│                                                      │
│  git submodule deinit: 2を削除                       │
│  git rm:              1, 4, 5を削除                  │
│  rm -rf:              3を削除（手動）                 │
└──────────────────────────────────────────────────────┘
```

### 2.7 submoduleのURLとパスの変更

```bash
# URLの変更
$ git config -f .gitmodules submodule.vendor/utils.url git@github.com:org/utils.git
$ git submodule sync
$ git submodule update --init

# パスの変更（サブモジュールの移動）
$ git mv vendor/utils lib/utils
# → .gitmodulesのパスも自動更新（Git 2.17+）
$ git commit -m "chore: move vendor/utils to lib/utils"

# URLの一括書き換え（HTTPS → SSH）
$ git config --global url."git@github.com:".insteadOf "https://github.com/"
# → 全てのHTTPS URLがSSHに変換される
```

---

## 3. subtree mergeとの比較

```bash
# subtree addでの外部リポジトリ統合
$ git subtree add --prefix=vendor/utils \
    https://github.com/lib/utils.git main --squash

# subtreeの更新
$ git subtree pull --prefix=vendor/utils \
    https://github.com/lib/utils.git main --squash

# subtreeからの変更を上流にpush
$ git subtree push --prefix=vendor/utils \
    https://github.com/lib/utils.git develop
```

| 項目               | submodule                    | subtree                      |
|--------------------|------------------------------|------------------------------|
| リポジトリ構造     | 親とは別の独立リポジトリ     | 親リポジトリに統合           |
| クローン           | `--recurse-submodules`必要   | 通常のcloneで完結            |
| バージョン管理     | commit SHA-1で厳密に固定     | マージコミットで管理         |
| 更新の容易さ       | `submodule update`           | `subtree pull`               |
| .gitmodulesの管理  | 必要                         | 不要                         |
| 履歴の独立性       | 完全に分離                   | 親の履歴に混在               |
| CIでの扱い         | 追加ステップが必要           | 特別な処理不要               |
| 推奨用途           | 大きな外部ライブラリ         | 小さな共有コード             |
| 上流への貢献       | サブモジュール内で直接push   | `subtree push`で抽出         |
| ディスク使用量     | 独立クローン分               | 親リポジトリに含まれる       |

### 3.1 subtreeの詳細な使い方

```bash
# subtree add（初回追加）
$ git subtree add --prefix=lib/shared \
    git@github.com:org/shared-lib.git main --squash
# → lib/shared/ に外部リポジトリの内容を配置
# → --squash で外部リポジトリの履歴を1つのコミットにまとめる

# subtree pull（更新）
$ git subtree pull --prefix=lib/shared \
    git@github.com:org/shared-lib.git main --squash
# → 最新の変更を取り込む

# subtree push（上流への貢献）
$ git subtree push --prefix=lib/shared \
    git@github.com:org/shared-lib.git feature/my-fix
# → lib/shared/ への変更を外部リポジトリのブランチにpush

# subtree split（履歴の抽出）
$ git subtree split --prefix=lib/shared --branch=shared-only
# → lib/shared/ に関する履歴だけを抽出してブランチを作成
```

### 3.2 モノレポとの比較

```
┌──────────────────────────────────────────────────────┐
│  依存管理の4つのアプローチ                            │
│                                                      │
│  1. submodule                                        │
│     独立リポジトリを参照。バージョン固定が容易        │
│     適用: 外部ライブラリ、大きな依存                  │
│                                                      │
│  2. subtree                                          │
│     コードを直接統合。クローンが容易                  │
│     適用: 小さな共有ライブラリ                        │
│                                                      │
│  3. モノレポ                                         │
│     全てのコードを1つのリポジトリに配置               │
│     適用: 組織内の密結合プロジェクト                  │
│     ツール: Nx, Turborepo, Bazel                     │
│                                                      │
│  4. パッケージマネージャー                           │
│     npm, pip, gem 等でバージョン管理                  │
│     適用: 公開ライブラリ、明確なAPI境界               │
│                                                      │
│  判断基準:                                           │
│  - 変更頻度が高い → モノレポ or submodule             │
│  - 安定したAPI → パッケージマネージャー              │
│  - クローン簡易性が重要 → subtree                    │
│  - 厳密なバージョン管理 → submodule                  │
└──────────────────────────────────────────────────────┘
```

---

## 4. foreach -- 一括操作

```bash
# 全サブモジュールで同じコマンドを実行
$ git submodule foreach 'git fetch origin && git checkout main && git pull'

# ネストされたサブモジュールも含む
$ git submodule foreach --recursive 'git clean -fdx'

# 条件付き実行
$ git submodule foreach '
  if [ -f package.json ]; then
    npm install
  fi
'

# サブモジュールの名前やパスを使用
$ git submodule foreach 'echo "Processing: $name at $sm_path (toplevel: $toplevel)"'
# $name:     サブモジュール名（.gitmodulesのセクション名）
# $sm_path:  サブモジュールのパス
# $toplevel: 親リポジトリのトップレベルディレクトリ
# $sha1:     サブモジュールの現在のcommit SHA-1
# $displaypath: 表示用パス

# 全サブモジュールのステータスサマリー
$ git submodule foreach 'echo "$sm_path: $(git describe --always --dirty)"'

# 全サブモジュールで未コミットの変更があるか確認
$ git submodule foreach 'git status --porcelain | grep -q . && echo "$sm_path has changes" || echo "$sm_path is clean"'
```

### 4.1 foreachの実践的なスクリプト

```bash
#!/bin/bash
# update-all-submodules.sh - 全サブモジュールを安全に更新
set -euo pipefail

echo "=== Fetching all submodules ==="
git submodule foreach 'git fetch origin 2>/dev/null'

echo ""
echo "=== Status before update ==="
git submodule status

echo ""
echo "=== Updating to remote HEAD ==="
git submodule update --remote

echo ""
echo "=== Status after update ==="
git submodule status

# 変更があればコミット
if ! git diff --cached --quiet; then
    echo ""
    echo "=== Committing submodule updates ==="
    git add -A
    git commit -m "chore: update all submodules to latest"
else
    echo ""
    echo "All submodules are up to date."
fi
```

---

## 5. 実用的な.gitmodules設定

```bash
# ブランチの指定（update --remote 時に使用）
$ git config -f .gitmodules submodule.vendor/utils.branch develop

# shallow clone（高速化）
$ git config -f .gitmodules submodule.vendor/utils.shallow true

# URLの書き換え（プライベートリポジトリ対応）
$ git config url."git@github.com:".insteadOf "https://github.com/"

# 特定のサブモジュールのfetch設定
$ git config -f .gitmodules submodule.vendor/utils.fetchRecurseSubmodules false

# update戦略の設定
$ git config -f .gitmodules submodule.vendor/utils.update merge
# → update時にmergeを使用（デフォルトはcheckout）

# .gitmodulesの最終形
$ cat .gitmodules
[submodule "vendor/utils"]
    path = vendor/utils
    url = https://github.com/lib/utils.git
    branch = develop
    shallow = true
[submodule "vendor/auth"]
    path = vendor/auth
    url = git@github.com:org/auth-lib.git
    branch = main
    update = merge
[submodule "vendor/ui"]
    path = vendor/ui
    url = https://github.com/org/ui-components.git
    branch = stable
    shallow = true
    fetchRecurseSubmodules = false
```

### 5.1 .gitmodules設定項目の一覧

| 設定項目                    | 説明                                         | デフォルト  |
|----------------------------|----------------------------------------------|-------------|
| `path`                     | サブモジュールの配置パス                      | (必須)      |
| `url`                      | リポジトリのURL                               | (必須)      |
| `branch`                   | `--remote`更新時に追従するブランチ           | (リモートHEAD) |
| `update`                   | 更新戦略 (checkout/merge/rebase/none)        | checkout    |
| `shallow`                  | shallow cloneを使用                          | false       |
| `fetchRecurseSubmodules`   | fetch時にサブモジュールも再帰的にfetch       | (設定依存)  |
| `ignore`                   | status/diffでの無視レベル (dirty/untracked/all/none) | none  |

---

## 6. CI/CD環境でのsubmodule運用

### 6.1 GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout with submodules
        uses: actions/checkout@v4
        with:
          submodules: recursive    # サブモジュールを再帰的にclone
          token: ${{ secrets.PAT_TOKEN }}  # プライベートサブモジュール用

      # shallow submodule（高速化）
      # - name: Checkout with shallow submodules
      #   uses: actions/checkout@v4
      #   with:
      #     submodules: recursive
      #     fetch-depth: 1         # shallow clone

      - name: Build
        run: npm run build

      - name: Test
        run: npm test
```

### 6.2 GitLab CI

```yaml
# .gitlab-ci.yml
variables:
  GIT_SUBMODULE_STRATEGY: recursive   # サブモジュールを再帰的に取得
  GIT_SUBMODULE_DEPTH: 1              # shallow clone

build:
  script:
    - npm install
    - npm run build

# プライベートサブモジュールの場合
# Settings > CI/CD > Variables に CI_JOB_TOKEN を設定
# .gitmodulesのURLを相対パスに変更:
# [submodule "lib/shared"]
#     path = lib/shared
#     url = ../../group/shared-lib.git
```

### 6.3 Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                checkout([
                    $class: 'GitSCM',
                    extensions: [
                        [$class: 'SubmoduleOption',
                         disableSubmodules: false,
                         parentCredentials: true,
                         recursiveSubmodules: true,
                         reference: '',
                         trackingSubmodules: false]
                    ],
                    userRemoteConfigs: [[
                        url: 'https://github.com/org/project.git',
                        credentialsId: 'github-credentials'
                    ]]
                ])
            }
        }
        stage('Build') {
            steps {
                sh 'npm install && npm run build'
            }
        }
    }
}
```

---

## 7. トラブルシューティング

### 7.1 よくあるsubmoduleのエラーと対処

```bash
# エラー1: "fatal: reference is not a tree: <sha1>"
# 原因: 親リポジトリが参照するcommitがサブモジュールのリモートに存在しない
$ cd vendor/utils
$ git fetch origin
$ git log --oneline --all | head -5
# → 参照先のcommitが存在するか確認
# 対処: サブモジュールの開発者がpushし忘れている可能性がある

# エラー2: "fatal: No url found for submodule path 'vendor/utils'"
# 原因: .gitmodulesに設定があるがgit submodule initされていない
$ git submodule init
$ git submodule update

# エラー3: サブモジュールがdetached HEADになる
# 原因: git submodule updateはデフォルトでcheckout（detached HEAD）
$ cd vendor/utils
$ git checkout main   # ブランチに切り替え
# または、update戦略をmergeに変更
$ git config -f .gitmodules submodule.vendor/utils.update merge

# エラー4: "Submodule path 'vendor/utils' already exists in the index"
# 原因: 不完全な削除後に再追加しようとしている
$ git rm -f vendor/utils
$ rm -rf .git/modules/vendor/utils
$ git submodule add <url> vendor/utils

# エラー5: ネストされたサブモジュールが初期化されない
$ git submodule update --init --recursive
# --recursive を忘れるとネストされたサブモジュールは初期化されない
```

### 7.2 worktreeのトラブルシューティング

```bash
# エラー1: "fatal: '<branch>' is already checked out"
# 対処: 別のworktreeで使用中のブランチは使えない
$ git worktree list  # どのworktreeがそのブランチを使っているか確認
# → そのworktreeを削除するか、別のブランチ名を使う

# エラー2: worktreeが壊れた（参照先が見つからない）
$ git worktree repair
# → 壊れたworktreeのリンクを修復（Git 2.30+）

# エラー3: worktreeの.gitファイルが破損
$ cat ../hotfix-v1/.git
# → "gitdir: ..." の内容を確認
# → パスが正しくない場合は手動で修正

# エラー4: worktreeを移動した後にリンクが壊れた
$ git worktree repair ../new-location
# → 移動先のパスでリンクを修復
```

---

## 8. アンチパターン

### アンチパターン1: submoduleの更新忘れ

```bash
# NG: サブモジュールの変更をpushせずに親リポジトリをpush
$ cd vendor/utils
$ git commit -m "fix: critical bug"
# vendor/utilsのリモートにpushし忘れ
$ cd ../..
$ git add vendor/utils
$ git commit -m "update submodule"
$ git push origin main
# → 他のメンバーが submodule update すると、存在しないcommitを参照してエラー

# OK: 常にサブモジュール側を先にpush
$ cd vendor/utils && git push origin main
$ cd ../.. && git add vendor/utils && git commit && git push
# または push時に自動チェック
$ git push --recurse-submodules=check origin main
$ git push --recurse-submodules=on-demand origin main  # 自動push
```

**理由**: 親リポジトリはサブモジュールのcommit SHA-1を記録するだけ。そのcommitがリモートに存在しなければ、他の開発者はcheckoutできない。

### アンチパターン2: worktreeのパスを絶対パスでスクリプトに埋め込む

```bash
# NG: 絶対パスをハードコード
BUILD_DIR="/home/user/build-release"
git worktree add "$BUILD_DIR" release/v2.0

# OK: 相対パスや変数を使用
PROJECT_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="${PROJECT_ROOT}/../build-release"
git worktree add "$BUILD_DIR" release/v2.0
```

**理由**: 開発者ごとにディレクトリ構造が異なる。相対パスやgitコマンドで動的に解決すべき。

### アンチパターン3: サブモジュールをブランチ追従モードで無管理に運用

```bash
# NG: --remote で常に最新を追従、テストなしで統合
$ git submodule update --remote
$ git add -A && git commit -m "update submodules" && git push
# → 破壊的変更が自動的に取り込まれる可能性

# OK: バージョンを明示的に管理
$ cd vendor/utils
$ git fetch origin
$ git log --oneline origin/main..HEAD  # 差分を確認
$ git checkout v2.4.0                   # 特定バージョンに固定
$ cd ../..
$ git add vendor/utils
$ git commit -m "chore: update vendor/utils to v2.4.0"
```

**理由**: サブモジュールは依存関係。無制御な自動更新は本番環境のバグにつながる。Dependabotなどのツールを使って管理するのが望ましい。

### アンチパターン4: worktreeを大量に放置する

```bash
# NG: worktreeを作成するだけで放置
$ git worktree add ../review-1 feature/a
$ git worktree add ../review-2 feature/b
$ git worktree add ../review-3 feature/c
# ... 数週間放置 ...
# → ディスク容量を圧迫、ブランチの削除もできなくなる

# OK: 定期的にクリーンアップ
$ git worktree list
$ git worktree remove ../review-1
$ git worktree prune
```

**理由**: worktreeが存在する限り、そのブランチは削除できず、作業ファイル分のディスクを占有し続ける。

---

## 9. 高度なサブモジュール運用

### 9.1 サブモジュールの差分表示

```bash
# サブモジュールの変更をサマリーで表示
$ git diff --submodule=short
# → サブモジュールのcommit変更を表示

$ git diff --submodule=log
# → サブモジュールの変更されたcommitのlog一覧を表示

$ git diff --submodule=diff
# → サブモジュール内の実際のdiffを表示

# デフォルトの差分表示形式を設定
$ git config --global diff.submodule log
```

### 9.2 サブモジュールのブランチ管理

```bash
# 全サブモジュールで特定のブランチに切り替え
$ git submodule foreach 'git checkout develop || true'

# 全サブモジュールの状態をdetached HEADからブランチに変更
$ git submodule foreach '
  branch=$(git config -f $toplevel/.gitmodules submodule.$name.branch || echo main)
  git checkout $branch 2>/dev/null || git checkout -b $branch
'

# サブモジュール内のブランチを一括表示
$ git submodule foreach 'echo "$sm_path: $(git branch --show-current || echo DETACHED)"'
```

### 9.3 サブモジュールのセキュリティ

```bash
# fsck でサブモジュールのURLが安全か検証
$ git config --global protocol.file.allow always
# → fileプロトコルを明示的に許可（Git 2.38.1+のセキュリティ修正以降）

# サブモジュールのURLに対する制限
$ git config --global submodule.fetchJobs 4
# → 並列fetchのジョブ数を制限

# URLの検証
$ git submodule foreach 'echo "$name: $(git remote get-url origin)"'
# → 全サブモジュールのURLを一括確認
```


---

## 実践演習

### 演習1: 基本的な実装

以下の要件を満たすコードを実装してください。

**要件:**
- 入力データの検証を行うこと
- エラーハンドリングを適切に実装すること
- テストコードも作成すること

```python
# 演習1: 基本実装のテンプレート
class Exercise1:
    """基本的な実装パターンの演習"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """入力値の検証"""
        if value is None:
            raise ValueError("入力値がNoneです")
        return True

    def process(self, value):
        """データ処理のメインロジック"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """処理結果の取得"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# テスト
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "例外が発生するべき"
    except ValueError:
        pass

    print("全テスト合格!")

test_exercise1()
```

### 演習2: 応用パターン

基本実装を拡張して、以下の機能を追加してください。

```python
# 演習2: 応用パターン
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """応用パターンの演習"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """アイテムの追加（サイズ制限付き）"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """キーによる検索"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """キーによる削除"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """統計情報"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# テスト
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # サイズ制限
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("応用テスト全合格!")

test_advanced()
```

### 演習3: パフォーマンス最適化

以下のコードのパフォーマンスを改善してください。

```python
# 演習3: パフォーマンス最適化
import time
from functools import lru_cache

# 最適化前（O(n^2)）
def slow_search(data: list, target: int) -> int:
    """非効率な検索"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# 最適化後（O(n)）
def fast_search(data: list, target: int) -> tuple:
    """ハッシュマップを使った効率的な検索"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# ベンチマーク
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"非効率版: {slow_time:.4f}秒")
    print(f"効率版:   {fast_time:.6f}秒")
    print(f"高速化率: {slow_time/fast_time:.0f}倍")

benchmark()
```

**ポイント:**
- アルゴリズムの計算量を意識する
- 適切なデータ構造を選択する
- ベンチマークで効果を測定する

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

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |
---

## 10. FAQ

### Q1. worktreeとgit cloneの違いは何か？

**A1.** worktreeは**オブジェクトデータベースを共有**します。cloneは全てを複製するため、ディスク使用量が倍増します。worktreeは同一リポジトリの別ブランチを並行作業する場合に最適で、cloneは完全に独立した作業環境が必要な場合に使います。

| 項目               | worktree            | clone               |
|--------------------|---------------------|----------------------|
| .git/objects       | 共有（リンク）      | 独立したコピー       |
| ディスク使用量     | 作業ファイルのみ追加| 全データの複製       |
| ブランチの制約     | 同一ブランチ不可    | 制約なし             |
| fetchの反映        | 即座に全worktreeに  | 各cloneで個別に必要  |
| hooks              | 共有                | 独立                 |
| config             | 共有                | 独立                 |

### Q2. サブモジュールのURLを変更するにはどうすればよいか？

**A2.** 以下の手順で変更します。

```bash
# 1. .gitmodulesを編集
$ git config -f .gitmodules submodule.vendor/utils.url git@github.com:org/utils.git

# 2. ローカル設定を同期
$ git submodule sync

# 3. サブモジュールを再初期化
$ git submodule update --init

# 4. 変更をコミット
$ git add .gitmodules
$ git commit -m "chore: update submodule URL for vendor/utils"
```

### Q3. サブモジュールを含むリポジトリでCIを設定する際のポイントは？

**A3.** 以下の3点が重要です。

1. **クローン時に`--recurse-submodules`を指定**するか、`git submodule update --init --recursive`を実行する
2. **shallow cloneとの組み合わせ**: `git clone --depth=1 --recurse-submodules --shallow-submodules`で最小限のデータ取得
3. **SSH鍵またはトークンの設定**: プライベートサブモジュールへのアクセスに認証が必要。GitHub Actionsでは`persist-credentials: true`と適切なトークンスコープを設定する

### Q4. worktreeを使用中にgit gcを実行するとどうなるか？

**A4.** GCはメインworktreeの`.git/objects/`に対して実行されます。linked worktreeのオブジェクトも同じデータベースに格納されているため、**全worktreeで参照されているオブジェクトは保護されます**。ただし、worktreeを手動で削除した（`git worktree remove`を使わずに`rm -rf`で消した）場合、そのworktreeが参照していたオブジェクトがGCで回収される可能性があります。

```bash
# 安全なクリーンアップ手順
$ git worktree prune          # 壊れたworktree参照を削除
$ git gc --prune=now          # 不要オブジェクトを削除
```

### Q5. サブモジュールの代わりにGitのsparse-checkoutを使う方法は？

**A5.** sparse-checkoutはモノレポの一部だけをチェックアウトする機能で、サブモジュールとは異なるアプローチです。

```bash
# sparse-checkout（Git 2.25+）
$ git clone --filter=blob:none --sparse https://github.com/org/monorepo.git
$ cd monorepo
$ git sparse-checkout set lib/utils lib/auth
# → lib/utils/ と lib/auth/ だけがチェックアウトされる
# → 他のディレクトリのファイルはダウンロードされない

# サブモジュールとの違い:
# - sparse-checkout: 1つのリポジトリの一部を取得
# - submodule: 別のリポジトリを参照
```

### Q6. ネストされたサブモジュール（サブモジュールの中にサブモジュール）は推奨されるか？

**A6.** 技術的には可能ですが、**一般的には推奨されません**。ネストが深くなるほど以下の問題が増大します。

- `--recursive`を忘れると部分的にしか初期化されない
- 更新の順序が複雑になる
- CI/CDの設定が煩雑になる
- トラブルシューティングが困難

代替案として、全てのサブモジュールを親リポジトリの直下にフラットに配置することを検討してください。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| 概念                   | 要点                                                          |
|------------------------|---------------------------------------------------------------|
| worktree               | .gitを共有して複数ブランチを同時チェックアウト                |
| linked worktree        | `.git`テキストファイルでメインリポジトリを参照                |
| worktree repair        | 壊れたworktreeのリンクを修復（Git 2.30+）                    |
| submodule              | 外部リポジトリのcommit SHA-1を親リポジトリのtreeに記録       |
| .gitmodules            | サブモジュールのURL・パス・ブランチのマッピング              |
| submodule update       | 親が記録したcommitにサブモジュールをcheckout                 |
| submodule sync         | .gitmodulesの変更をローカル設定に反映                        |
| subtree                | 外部コードを親リポジトリの履歴に統合する代替手法             |
| --recurse-submodules   | clone/push/pull時にサブモジュールも自動処理                  |
| sparse-checkout        | モノレポの一部だけをチェックアウトする機能                    |

---

## 次に読むべきガイド

- [Packfile/GC](../00-git-internals/03-packfile-gc.md) -- worktreeとGCの関係
- [Git Hooks](./03-hooks-automation.md) -- サブモジュール更新の自動化
- [Jujutsu入門](../02-jujutsu/00-jujutsu-introduction.md) -- サブモジュールの代替アプローチ

---

## 参考文献

1. **Pro Git Book** -- "Git Tools - Submodules" https://git-scm.com/book/en/v2/Git-Tools-Submodules
2. **Git公式ドキュメント** -- `git-worktree`, `git-submodule` https://git-scm.com/docs
3. **GitHub Blog** -- "Working with submodules" https://github.blog/2016-02-01-working-with-submodules/
4. **Atlassian Git Tutorial** -- "Git subtree" https://www.atlassian.com/git/tutorials/git-subtree
5. **GitHub Docs** -- "About Git sparse-checkout" https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-sparse-checkout
