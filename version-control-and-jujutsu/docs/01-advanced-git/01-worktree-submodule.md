# Worktree/Submodule

> `git worktree`による複数作業ディレクトリの管理と、`git submodule`による外部リポジトリの統合手法を解説し、大規模プロジェクトでの効率的な運用方法を習得する。

## この章で学ぶこと

1. **git worktreeの仕組みと活用法** — 1つのリポジトリで複数のブランチを同時にチェックアウトする手法
2. **git submoduleの内部構造と運用** — 外部リポジトリの依存管理とバージョン固定の仕組み
3. **代替手段との比較** — subtree merge、モノレポ、パッケージマネージャーとの使い分け

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

# 手動でディレクトリを削除した場合のクリーンアップ
$ rm -rf ../hotfix-v1
$ git worktree prune
# → 存在しないworktreeの参照を削除

# worktreeをロック（自動pruneの防止）
$ git worktree lock ../new-ui --reason "長期作業中"
$ git worktree unlock ../new-ui
```

### 1.3 worktreeの活用パターン

```bash
# パターン1: PRレビュー中に別の作業をする
$ git worktree add ../review-pr-42 origin/feature/pr-42
$ cd ../review-pr-42
$ npm test
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
```

### 1.4 worktreeの制約

| 制約                           | 説明                                          |
|-------------------------------|-----------------------------------------------|
| 同一ブランチの重複チェックアウト | 同じブランチを複数worktreeでチェックアウト不可 |
| ベアリポジトリ                 | worktreeの追加は可能だがメインworktreeがない    |
| サブモジュール                 | worktreeごとにサブモジュールの初期化が必要      |
| GC                            | メインworktreeの.git/objectsを共有              |

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

### 2.5 submoduleの削除

```bash
# サブモジュールの完全な削除（3段階必要）
$ git submodule deinit -f vendor/utils   # 1. 設定の無効化
$ git rm -f vendor/utils                  # 2. ファイルとインデックスから削除
$ rm -rf .git/modules/vendor/utils        # 3. キャッシュの削除
$ git commit -m "chore: remove vendor/utils submodule"
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

---

## 4. foreach — 一括操作

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

# .gitmodulesの最終形
$ cat .gitmodules
[submodule "vendor/utils"]
    path = vendor/utils
    url = https://github.com/lib/utils.git
    branch = develop
    shallow = true
```

---

## 6. アンチパターン

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

---

## 7. FAQ

### Q1. worktreeとgit cloneの違いは何か？

**A1.** worktreeは**オブジェクトデータベースを共有**します。cloneは全てを複製するため、ディスク使用量が倍増します。worktreeは同一リポジトリの別ブランチを並行作業する場合に最適で、cloneは完全に独立した作業環境が必要な場合に使います。

| 項目               | worktree            | clone               |
|--------------------|---------------------|----------------------|
| .git/objects       | 共有（リンク）      | 独立したコピー       |
| ディスク使用量     | 作業ファイルのみ追加| 全データの複製       |
| ブランチの制約     | 同一ブランチ不可    | 制約なし             |
| fetchの反映        | 即座に全worktreeに  | 各cloneで個別に必要  |

### Q2. サブモジュールのURLを変更するにはどうすればよいか？

**A2.** 以下の手順で変更します。

```bash
# 1. .gitmodulesを編集
$ git config -f .gitmodules submodule.vendor/utils.url git@github.com:org/utils.git

# 2. ローカル設定を同期
$ git submodule sync

# 3. サブモジュールを再初期化
$ git submodule update --init
```

### Q3. サブモジュールを含むリポジトリでCIを設定する際のポイントは？

**A3.** 以下の3点が重要です。

1. **クローン時に`--recurse-submodules`を指定**するか、`git submodule update --init --recursive`を実行する
2. **shallow cloneとの組み合わせ**: `git clone --depth=1 --recurse-submodules --shallow-submodules`で最小限のデータ取得
3. **SSH鍵またはトークンの設定**: プライベートサブモジュールへのアクセスに認証が必要。GitHub Actionsでは`persist-credentials: true`と適切なトークンスコープを設定する

---

## まとめ

| 概念                   | 要点                                                          |
|------------------------|---------------------------------------------------------------|
| worktree               | .gitを共有して複数ブランチを同時チェックアウト                |
| linked worktree        | `.git`テキストファイルでメインリポジトリを参照                |
| submodule              | 外部リポジトリのcommit SHA-1を親リポジトリのtreeに記録       |
| .gitmodules            | サブモジュールのURL・パス・ブランチのマッピング              |
| submodule update       | 親が記録したcommitにサブモジュールをcheckout                 |
| subtree                | 外部コードを親リポジトリの履歴に統合する代替手法             |
| --recurse-submodules   | clone/push/pull時にサブモジュールも自動処理                  |

---

## 次に読むべきガイド

- [Packfile/GC](../00-git-internals/03-packfile-gc.md) — worktreeとGCの関係
- [Git Hooks](./03-hooks-automation.md) — サブモジュール更新の自動化
- [Jujutsu入門](../02-jujutsu/00-jujutsu-introduction.md) — サブモジュールの代替アプローチ

---

## 参考文献

1. **Pro Git Book** — "Git Tools - Submodules" https://git-scm.com/book/en/v2/Git-Tools-Submodules
2. **Git公式ドキュメント** — `git-worktree`, `git-submodule` https://git-scm.com/docs
3. **GitHub Blog** — "Working with submodules" https://github.blog/2016-02-01-working-with-submodules/
4. **Atlassian Git Tutorial** — "Git subtree" https://www.atlassian.com/git/tutorials/git-subtree
