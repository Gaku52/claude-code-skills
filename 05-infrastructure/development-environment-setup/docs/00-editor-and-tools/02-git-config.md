# Git 設定

> Git の初期設定からGPG署名・SSH鍵・diff/mergeツール連携まで、プロフェッショナルなGit環境を構築するための完全ガイド。

## この章で学ぶこと

1. `.gitconfig` の体系的な設定と実用的なエイリアス構築
2. GPG 署名と SSH 鍵の正しい設定方法
3. diff/merge ツールの統合と credential helper の管理
4. 複数アカウントの使い分けと Git hooks の活用
5. トラブルシューティングとパフォーマンス最適化

---

## 1. Git の初期設定

### 1.1 基本設定

```bash
# ユーザー情報
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# デフォルトブランチ名
git config --global init.defaultBranch main

# エディタ設定
git config --global core.editor "code --wait"

# 改行コード処理
# macOS/Linux
git config --global core.autocrlf input
# Windows
git config --global core.autocrlf true

# 日本語ファイル名の表示
git config --global core.quotepath false

# カラー出力
git config --global color.ui auto

# ページャー設定 (delta 推奨)
git config --global core.pager "delta"

# 大文字小文字を区別
git config --global core.ignorecase false

# シンボリックリンクを追跡
git config --global core.symlinks true
```

### 1.2 設定ファイルの階層

```
Git 設定の優先順位 (高い順):

┌─────────────────────────────────────────────┐
│  1. Local    (.git/config)                   │
│     → リポジトリ固有の設定                    │
│     → git config --local                     │
│     → 例: プロジェクト固有の user.email       │
├─────────────────────────────────────────────┤
│  2. Global   (~/.gitconfig)                  │
│     → ユーザー共通の設定                      │
│     → git config --global                    │
│     → 例: 個人のエイリアス、エディタ設定      │
├─────────────────────────────────────────────┤
│  3. System   (/etc/gitconfig)                │
│     → システム全体の設定                      │
│     → git config --system                    │
│     → 例: 組織のプロキシ設定                  │
├─────────────────────────────────────────────┤
│  4. Portable (/path/to/.gitconfig)           │
│     → GIT_CONFIG_GLOBAL 環境変数で指定       │
│     → dotfiles リポジトリからの共有設定      │
└─────────────────────────────────────────────┘

※ 同じキーが複数レベルで設定された場合、
   上位（Local）の値が優先される
```

### 1.3 設定の確認と管理

```bash
# 全設定の一覧表示 (適用元も表示)
git config --list --show-origin

# 特定のキーの値を確認
git config user.email

# 設定の適用元を確認
git config --show-origin user.email

# 設定のスコープを確認
git config --show-scope --list

# 設定を削除
git config --global --unset user.signingkey

# 設定ファイルを直接編集
git config --global --edit

# 条件付き設定の確認
git config --list --show-origin | grep includeIf
```

---

## 2. .gitconfig 完全版

### 2.1 推奨設定

```ini
# ~/.gitconfig

[user]
    name = Your Name
    email = your.email@example.com
    signingkey = YOUR_GPG_KEY_ID

[core]
    editor = code --wait
    autocrlf = input
    quotepath = false
    pager = delta
    ignorecase = false
    whitespace = trailing-space,space-before-tab
    precomposeunicode = true
    fsmonitor = true
    untrackedcache = true
    symlinks = true
    # 大規模リポジトリでのパフォーマンス改善
    # fsmonitor = true  # Git 2.37+ (Watchman or built-in)

[init]
    defaultBranch = main

[color]
    ui = auto

[color "branch"]
    current = yellow reverse
    local = yellow
    remote = green

[color "status"]
    added = green
    changed = yellow
    untracked = red

[push]
    default = current
    autoSetupRemote = true
    followTags = true
    # Git 2.40+: push 時に新しいブランチも自動設定
    # useForceIfIncludes = true  # force-with-lease の強化版

[pull]
    rebase = true

[fetch]
    prune = true
    prunetags = true
    # 並列フェッチ (Git 2.36+)
    parallel = 0  # CPUコア数に自動設定
    writeCommitGraph = true

[merge]
    conflictstyle = zdiff3
    tool = vscode
    # マージコミットに対象ブランチのログを含める
    log = 20

[mergetool]
    keepBackup = false
    prompt = false

[mergetool "vscode"]
    cmd = code --wait --merge $REMOTE $LOCAL $BASE $MERGED

[diff]
    tool = vscode
    colorMoved = default
    algorithm = histogram
    # バイナリファイルの差分表示
    # wordRegex = .  # 単語単位の差分
    renames = copies
    mnemonicPrefix = true
    submodule = log

[difftool "vscode"]
    cmd = code --wait --diff $LOCAL $REMOTE

[difftool]
    prompt = false

[rebase]
    autosquash = true
    autostash = true
    # rebase 時のコンフリクト検出を改善
    updateRefs = true
    # instructionFormat = "%s [%an, %ar]"

[rerere]
    enabled = true
    # rerere の自動ステージング
    autoupdate = true

[commit]
    gpgsign = true
    verbose = true
    # コミットメッセージテンプレート
    # template = ~/.gitmessage

[tag]
    gpgsign = true
    sort = version:refname

[interactive]
    diffFilter = delta --color-only

[delta]
    navigate = true
    light = false
    side-by-side = true
    line-numbers = true
    syntax-theme = Dracula
    file-style = bold yellow ul
    file-decoration-style = none
    hunk-header-decoration-style = cyan box ul
    minus-style = syntax "#3f0001"
    plus-style = syntax "#003800"
    line-numbers-minus-style = "#ff0000"
    line-numbers-plus-style = "#00ff00"
    map-styles = "bold purple => syntax #330033, bold cyan => syntax #003333"

[branch]
    sort = -committerdate
    # 新しいブランチ作成時に自動でリモート追跡
    autoSetupMerge = always

[column]
    ui = auto

[help]
    autocorrect = prompt
    # autocorrect = 10  # 1秒後に自動修正実行

[log]
    date = iso
    # abbrevCommit = true
    # follow = true  # ファイル名変更を自動追跡

[status]
    showUntrackedFiles = all
    submoduleSummary = true
    # short = true  # デフォルトで短縮形式

[stash]
    showPatch = true

[transfer]
    # Git オブジェクトの整合性チェック
    fsckObjects = true

[receive]
    fsckObjects = true

[url "git@github.com:"]
    insteadOf = https://github.com/

# GitHub CLI との統合
[credential "https://github.com"]
    helper = !/usr/bin/gh auth git-credential

# Git LFS
[filter "lfs"]
    clean = git-lfs clean -- %f
    smudge = git-lfs smudge -- %f
    process = git-lfs filter-process
    required = true
```

### 2.2 コミットメッセージテンプレート

```bash
# テンプレートファイルを作成
cat << 'EOF' > ~/.gitmessage

# <type>(<scope>): <subject>
#
# <body>
#
# <footer>
#
# ─── Type ───────────────────────────
# feat:     新機能
# fix:      バグ修正
# docs:     ドキュメント変更
# style:    コードスタイル変更 (動作に影響しない)
# refactor: リファクタリング
# perf:     パフォーマンス改善
# test:     テスト追加・修正
# build:    ビルドシステム変更
# ci:       CI 設定変更
# chore:    その他の変更
# revert:   コミットの取り消し
#
# ─── Rules ──────────────────────────
# Subject: 50文字以内、命令形、ピリオド不要
# Body: 72文字で折り返し、何を・なぜ変えたか
# Footer: Breaking Changes, Issue参照
#
# ─── Example ────────────────────────
# feat(auth): add JWT token refresh mechanism
#
# Implement automatic token refresh when the access
# token expires. The refresh token is stored securely
# in an HTTP-only cookie.
#
# Closes #123
# BREAKING CHANGE: API now requires Authorization header
EOF

# テンプレートを有効化
git config --global commit.template ~/.gitmessage
```

---

## 3. エイリアス

### 3.1 実用的エイリアス集

```ini
# ~/.gitconfig の [alias] セクション

[alias]
    # ─── 基本操作の短縮 ───
    s = status
    a = add
    aa = add --all
    ap = add --patch
    c = commit
    cm = commit -m
    ca = commit --amend
    can = commit --amend --no-edit
    co = checkout
    sw = switch
    sc = switch -c
    br = branch
    d = diff
    ds = diff --staged
    dw = diff --word-diff
    p = push
    pf = push --force-with-lease
    pl = pull
    f = fetch --all --prune
    m = merge
    rb = rebase
    rbi = rebase -i
    cp = cherry-pick
    st = stash
    stp = stash pop
    stl = stash list

    # ─── ログ表示 ───
    lg = log --oneline --graph --decorate --all -20
    ll = log --pretty=format:'%C(yellow)%h%C(reset) %C(green)(%cr)%C(reset) %s %C(bold blue)<%an>%C(reset)%C(red)%d%C(reset)' --abbrev-commit -20
    hist = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --all
    lp = log --patch -5
    ls = log --stat --oneline -10

    # ─── ファイル変更の追跡 ───
    filelog = log --follow -p --
    blame-line = "!f() { git log -L $1,$1:$2; }; f"
    contributors = shortlog --summary --numbered --email

    # ─── 便利コマンド ───
    # 直前のコミットを修正 (メッセージそのまま)
    amend = commit --amend --no-edit

    # ステージング取り消し
    unstage = restore --staged

    # 最後のコミットを取り消し (変更は保持)
    undo = reset --soft HEAD~1

    # 変更を一時退避
    stash-all = stash push --include-untracked -m

    # ブランチを最新に更新
    sync = !git fetch --all --prune && git pull --rebase

    # マージ済みブランチを削除
    cleanup = !git branch --merged | grep -v '\\*\\|main\\|master\\|develop' | xargs -n 1 git branch -d

    # リモートのマージ済みブランチも削除
    cleanup-remote = !git fetch --prune && git branch -r --merged origin/main | grep -v 'main\\|develop' | sed 's/origin\\///' | xargs -n 1 git push origin --delete

    # ファイルの変更履歴
    history = log --follow -p --

    # 今日のコミット
    today = log --since='midnight' --oneline --author='Your Name'

    # 今週のコミット
    week = log --since='1 week ago' --oneline --author='Your Name'

    # WIP コミット
    wip = !git add -A && git commit -m 'WIP: work in progress [skip ci]'

    # WIP 取り消し
    unwip = !git log -1 --format='%s' | grep -q 'WIP' && git reset HEAD~1

    # 初期コミット (空)
    init-commit = !git init && git commit --allow-empty -m 'chore: initial commit'

    # ブランチ名をコピー
    branch-name = rev-parse --abbrev-ref HEAD

    # 最近のブランチ一覧
    recent = branch --sort=-committerdate --format='%(committerdate:relative)\t%(refname:short)\t%(subject)' -20

    # 差分の統計
    stat = diff --stat

    # コミット間の差分ファイル一覧
    changed = diff --name-only

    # コンフリクトファイル一覧
    conflicts = diff --name-only --diff-filter=U

    # Git root ディレクトリ表示
    root = rev-parse --show-toplevel

    # fixup コミット (autosquash 用)
    fixup = "!f() { git commit --fixup=$1; }; f"

    # タグ一覧 (バージョン順)
    tags = tag -l --sort=-version:refname

    # リモート URL 表示
    remote-url = remote get-url origin

    # ブランチ比較 (何コミット差があるか)
    ahead = "!f() { git rev-list --count HEAD..${1:-origin/main}; }; f"
    behind = "!f() { git rev-list --count ${1:-origin/main}..HEAD; }; f"

    # インタラクティブな add (パッチモード)
    patch = add --patch

    # すべての変更を取り消し (注意して使用)
    nuke = !git reset --hard HEAD && git clean -fd

    # ローカルの main を最新に
    update-main = !git checkout main && git pull && git checkout -

    # PR 用: 現在のブランチの全コミットを表示
    pr-log = "!git log --oneline $(git merge-base HEAD main)..HEAD"

    # PR 用: 現在のブランチの全変更ファイルを表示
    pr-files = "!git diff --name-only $(git merge-base HEAD main)..HEAD"

    # PR 用: 現在のブランチの差分統計
    pr-stat = "!git diff --stat $(git merge-base HEAD main)..HEAD"
```

### 3.2 エイリアスの動作

```
git lg の出力例:

  * a1b2c3d (HEAD -> feature/auth, origin/feature/auth) Add JWT middleware
  * d4e5f6g Add login endpoint
  * g7h8i9j Add user model
  | * j0k1l2m (origin/feature/ui) Update dashboard layout
  | * m3n4o5p Add sidebar component
  |/
  * p6q7r8s (origin/main, main) Release v1.2.0
  * s9t0u1v Fix database connection pool
  * v2w3x4y Add health check endpoint

git recent の出力例:

  2 hours ago     feature/auth      Add JWT middleware
  5 hours ago     feature/ui        Update dashboard layout
  1 day ago       fix/db-pool       Fix connection pool leak
  3 days ago      main              Release v1.2.0
  1 week ago      feature/search    Add full-text search

git pr-stat の出力例:

  src/middleware/auth.ts   | 45 ++++++++++++
  src/routes/login.ts      | 32 +++++++++
  src/models/user.ts       | 28 +++++++++
  tests/auth.test.ts       | 67 ++++++++++++++++
  4 files changed, 172 insertions(+)
```

---

## 4. SSH 鍵の設定

### 4.1 Ed25519 鍵の生成

```bash
# SSH 鍵を生成 (Ed25519 推奨)
ssh-keygen -t ed25519 -C "your.email@example.com"

# 鍵の保存場所 (デフォルト: ~/.ssh/id_ed25519)
# パスフレーズを設定 (空にしない)

# SSH エージェントに鍵を登録
eval "$(ssh-agent -s)"

# macOS: キーチェーンに保存
ssh-add --apple-use-keychain ~/.ssh/id_ed25519

# Linux
ssh-add ~/.ssh/id_ed25519

# 公開鍵をコピーして GitHub に登録
cat ~/.ssh/id_ed25519.pub | pbcopy  # macOS
cat ~/.ssh/id_ed25519.pub | xclip -selection clipboard  # Linux
# GitHub → Settings → SSH and GPG keys → New SSH key

# ─── 鍵の種類比較 ───
# Ed25519:  推奨。高速、安全、鍵サイズが小さい
# RSA 4096: レガシー互換。古いシステムとの接続に必要な場合
# ECDSA:    非推奨。NSA関与の疑念あり
```

### 4.2 SSH config

```bash
# ~/.ssh/config

# ─── GitHub (個人) ───
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    AddKeysToAgent yes
    UseKeychain yes  # macOS のみ
    IdentitiesOnly yes

# ─── GitHub (会社用アカウント) ───
Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
    AddKeysToAgent yes
    IdentitiesOnly yes

# ─── GitLab ───
Host gitlab.com
    HostName gitlab.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    PreferredAuthentications publickey

# ─── Bitbucket ───
Host bitbucket.org
    HostName bitbucket.org
    User git
    IdentityFile ~/.ssh/id_ed25519

# ─── 自社 Git サーバー ───
Host git.company.com
    HostName git.company.com
    User git
    Port 2222
    IdentityFile ~/.ssh/id_ed25519_work
    ProxyCommand ssh -q -W %h:%p bastion.company.com

# ─── 踏み台サーバー経由 ───
Host bastion.company.com
    HostName bastion.company.com
    User your-username
    IdentityFile ~/.ssh/id_ed25519_work

# ─── 全ホスト共通設定 ───
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
    AddKeysToAgent yes
    Compression yes

# 使い分け:
# git clone git@github.com:personal/repo.git       # 個人
# git clone git@github-work:company/repo.git       # 会社
# git clone git@gitlab.com:team/repo.git            # GitLab
```

### 4.3 SSH 鍵のセキュリティベストプラクティス

```bash
# ─── 鍵のパーミッション確認 ───
ls -la ~/.ssh/
# 正しいパーミッション:
# drwx------  ~/.ssh/          (700)
# -rw-------  id_ed25519       (600)
# -rw-r--r--  id_ed25519.pub   (644)
# -rw-------  config           (600)

# パーミッション修正
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/config

# ─── 鍵のフィンガープリント確認 ───
ssh-keygen -lf ~/.ssh/id_ed25519.pub
# 256 SHA256:xxxxx your.email@example.com (ED25519)

# ─── 複数鍵の管理 ───
# ssh-add で登録済み鍵を確認
ssh-add -l

# 特定の鍵を削除
ssh-add -d ~/.ssh/id_ed25519_old

# 全鍵を削除
ssh-add -D

# ─── 鍵のローテーション ───
# 1. 新しい鍵を生成
ssh-keygen -t ed25519 -C "your.email@example.com" -f ~/.ssh/id_ed25519_new
# 2. GitHub に新しい公開鍵を追加
# 3. SSH config を更新
# 4. テスト
ssh -T git@github.com
# 5. 古い公開鍵を GitHub から削除
# 6. 古い鍵ファイルを削除
```

### 4.4 接続テスト

```bash
# GitHub
ssh -T git@github.com
# Hi username! You've successfully authenticated, but GitHub
# does not provide shell access.

# GitLab
ssh -T git@gitlab.com
# Welcome to GitLab, @username!

# 詳細デバッグ
ssh -vT git@github.com
# デバッグ出力で認証プロセスを確認

# 会社用アカウントのテスト
ssh -T git@github-work
```

---

## 5. GPG 署名

### 5.1 GPG 鍵の生成と設定

```bash
# GPG 鍵を生成
gpg --full-generate-key
# 種類: RSA and RSA (default)
# 鍵長: 4096
# 有効期限: 2y (推奨)
# 名前とメールアドレス: Git と同じものを入力

# 鍵 ID を確認
gpg --list-secret-keys --keyid-format=long
# sec   rsa4096/3AA5C34371567BD2 2024-01-01 [SC] [expires: 2026-01-01]
#       ABCDEF1234567890ABCDEF1234567890ABCDEF12
# uid                 [ultimate] Your Name <your.email@example.com>
# ssb   rsa4096/42B317FD4BA89E7A 2024-01-01 [E] [expires: 2026-01-01]

# 鍵 ID: 3AA5C34371567BD2

# Git に設定
git config --global user.signingkey 3AA5C34371567BD2
git config --global commit.gpgsign true
git config --global tag.gpgsign true

# 公開鍵をエクスポートして GitHub に登録
gpg --armor --export 3AA5C34371567BD2 | pbcopy
# GitHub → Settings → SSH and GPG keys → New GPG key

# macOS: pinentry-mac でパスフレーズ入力を GUI 化
brew install pinentry-mac
echo "pinentry-program $(which pinentry-mac)" >> ~/.gnupg/gpg-agent.conf
gpgconf --kill gpg-agent

# ─── GPG Agent の設定 ───
# ~/.gnupg/gpg-agent.conf
cat << 'EOF' > ~/.gnupg/gpg-agent.conf
# キャッシュ時間 (秒)
default-cache-ttl 28800      # 8時間
max-cache-ttl 86400           # 24時間
# macOS
pinentry-program /opt/homebrew/bin/pinentry-mac
# Linux (GUI)
# pinentry-program /usr/bin/pinentry-gnome3
# Linux (CUI)
# pinentry-program /usr/bin/pinentry-tty
EOF

# GPG Agent を再起動
gpgconf --kill gpg-agent
gpg-agent --daemon
```

### 5.2 SSH 鍵による署名 (GPG の代替)

```bash
# Git 2.34+ で SSH 鍵を使った署名が可能
# GPG のセットアップが面倒な場合の代替手段

# SSH 鍵で署名する設定
git config --global gpg.format ssh
git config --global user.signingkey ~/.ssh/id_ed25519.pub
git config --global commit.gpgsign true

# 署名の検証に使う公開鍵リスト
git config --global gpg.ssh.allowedSignersFile ~/.ssh/allowed_signers

# ~/.ssh/allowed_signers
# メールアドレスと公開鍵のマッピング
cat << 'EOF' > ~/.ssh/allowed_signers
your.email@example.com ssh-ed25519 AAAA... your.email@example.com
colleague@example.com ssh-ed25519 BBBB... colleague@example.com
EOF

# 署名の検証
git log --show-signature -1
git verify-commit HEAD
git verify-tag v1.0.0

# GPG vs SSH 署名の比較:
# GPG: 業界標準、GitHub "Verified" バッジ対応、設定が複雑
# SSH: 設定が簡単、Git 2.34+ 必須、GitHub 対応済み
```

### 5.3 署名の検証フロー

```
GPG 署名付きコミットのフロー:

  開発者の環境                    GitHub
  ┌──────────────────┐          ┌──────────────────┐
  │ git commit       │          │                  │
  │      │           │          │  公開鍵を        │
  │      ▼           │          │  登録済み        │
  │ GPG/SSH 秘密鍵で │   push   │      │           │
  │ 署名を付与       │ ───────→ │      ▼           │
  │      │           │          │  公開鍵で        │
  │      ▼           │          │  署名を検証      │
  │ 署名付き         │          │      │           │
  │ コミット         │          │      ▼           │
  │ 完成             │          │  ✅ Verified     │
  └──────────────────┘          └──────────────────┘

  Vigilant Mode 有効時:
  ┌──────────────────┐
  │ 署名なしコミット  │ → ⚠️ Unverified 表示
  │ 署名あり (有効)   │ → ✅ Verified 表示
  │ 署名あり (無効)   │ → ❌ Invalid 表示
  └──────────────────┘
```

---

## 6. Credential Helper

### 6.1 プラットフォーム別設定

```bash
# macOS (Keychain)
git config --global credential.helper osxkeychain

# Windows (Credential Manager)
git config --global credential.helper manager

# Linux (libsecret)
sudo apt install libsecret-1-0 libsecret-1-dev
sudo make --directory=/usr/share/doc/git/contrib/credential/libsecret
git config --global credential.helper /usr/share/doc/git/contrib/credential/libsecret/git-credential-libsecret

# Linux (GNOME Keyring)
git config --global credential.helper /usr/lib/git-core/git-credential-libsecret

# GitHub CLI (推奨: 全プラットフォーム共通)
gh auth login
gh auth setup-git
git config --global credential.helper '!gh auth git-credential'

# キャッシュ (一時的、サーバー向け)
git config --global credential.helper 'cache --timeout=3600'
```

### 6.2 認証方式比較

| 方式 | セキュリティ | 利便性 | 推奨度 | 備考 |
|------|------------|--------|--------|------|
| SSH 鍵 | 高 | 高 | 最推奨 | パスフレーズ + Agent |
| gh auth (CLI) | 高 | 高 | 推奨 | OAuth ベース |
| SSH 署名 | 高 | 高 | 推奨 | Git 2.34+ |
| Personal Access Token | 中 | 中 | 可 | スコープ制限必須 |
| Fine-grained PAT | 高 | 中 | 推奨 | リポジトリ単位制限 |
| パスワード認証 | 低 | - | 廃止済み | 2021年8月廃止 |

### 6.3 Personal Access Token の管理

```bash
# Fine-grained Personal Access Token の作成手順:
# 1. GitHub → Settings → Developer settings
# 2. Personal access tokens → Fine-grained tokens
# 3. Generate new token
# 4. 設定:
#    - Token name: 用途を明記 (e.g., "CI/CD pipeline")
#    - Expiration: 最大90日推奨
#    - Repository access: Only select repositories
#    - Permissions: 最小権限の原則
#      - Contents: Read-only (クローンのみ)
#      - Contents: Read and write (push も必要な場合)
#      - Pull requests: Read and write (PR 操作)

# トークンの保存 (macOS)
# Keychain Access に自動保存される

# トークンの更新
# 期限切れ時に再認証:
gh auth refresh
# または: Credential Manager から古いエントリを削除
```

---

## 7. diff / merge ツール

### 7.1 delta の設定

```bash
# delta インストール
brew install git-delta  # macOS
sudo apt install git-delta  # Ubuntu

# ~/.gitconfig で設定 (前述の [delta] セクション参照)

# 使用例
git diff              # 自動で delta が適用される
git log -p            # パッチ表示も delta で美麗に
git show HEAD         # コミット詳細も delta 対応
git stash show -p     # stash の差分も delta で表示
git blame file.ts     # blame も delta 対応

# delta の表示モード切替
# サイドバイサイド (デフォルト設定)
git diff
# 統合表示 (一時的に変更)
git -c delta.side-by-side=false diff
# 行番号なし
git -c delta.line-numbers=false diff
```

### 7.2 delta のテーマカスタマイズ

```ini
# ~/.gitconfig の [delta] セクション (詳細版)

[delta]
    navigate = true
    light = false
    side-by-side = true
    line-numbers = true
    syntax-theme = Catppuccin Mocha

    # ファイルヘッダー
    file-style = bold yellow ul
    file-decoration-style = none
    file-added-label = [+]
    file-modified-label = [~]
    file-removed-label = [-]
    file-renamed-label = [→]

    # ハンクヘッダー
    hunk-header-decoration-style = cyan box ul
    hunk-header-style = file line-number syntax

    # 差分表示
    minus-style = syntax "#3B1219"
    minus-emph-style = syntax "#6F1223"
    plus-style = syntax "#1A2B1A"
    plus-emph-style = syntax "#2B4B2B"

    # 行番号
    line-numbers-minus-style = "#F38BA8"
    line-numbers-plus-style = "#A6E3A1"
    line-numbers-zero-style = "#585B70"
    line-numbers-left-format = "{nm:>4} "
    line-numbers-right-format = "{np:>4} │ "

    # blame
    blame-palette = "#1e1e2e #181825 #11111b #313244 #45475a"
    blame-format = "{author:<18} {commit:<8} {timestamp:<16}"

    # merge conflict
    merge-conflict-begin-symbol = ▼
    merge-conflict-end-symbol = ▲
    merge-conflict-ours-diff-header-style = yellow bold
    merge-conflict-theirs-diff-header-style = cyan bold

    # インライン差分 (単語単位)
    inline-hint-style = syntax
```

### 7.3 VS Code をdiff/mergeツールに設定

```bash
# diff ツール
git config --global diff.tool vscode
git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'
git config --global difftool.prompt false

# merge ツール
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait --merge $REMOTE $LOCAL $BASE $MERGED'
git config --global mergetool.keepBackup false
git config --global mergetool.prompt false

# 使用方法
git difftool                 # VS Code で差分表示
git difftool --dir-diff      # ディレクトリ単位の差分表示
git mergetool                # VS Code でコンフリクト解決

# IntelliJ IDEA を使う場合
git config --global diff.tool intellij
git config --global difftool.intellij.cmd 'idea diff $LOCAL $REMOTE'
git config --global merge.tool intellij
git config --global mergetool.intellij.cmd 'idea merge $LOCAL $REMOTE $BASE $MERGED'
```

### 7.4 コンフリクト解決のフロー

```
zdiff3 形式のコンフリクト表示:

<<<<<<< HEAD (現在のブランチ)
  const timeout = 5000;
||||||| parent of abc1234 (共通祖先)
  const timeout = 3000;
=======
  const timeout = 10000;
>>>>>>> feature/update-config (マージ元)

zdiff3 の利点:
  - 共通祖先 (|||||||) が表示される
  - 「何から何に変えたか」が一目瞭然
  - 標準の diff3 より見やすい

コンフリクト解決のステップ:
  1. git mergetool で VS Code を開く
  2. 3ウェイマージビューで変更を比較
  3. 「Accept Current」「Accept Incoming」「Accept Both」から選択
  4. 必要に応じて手動編集
  5. 保存してエディタを閉じる
  6. git add <resolved-file>
  7. git merge --continue (または git rebase --continue)

rerere が有効な場合:
  - 一度解決したパターンが記録される
  - 同じコンフリクトは自動で解決される
  - git rerere status で記録を確認
```

---

## 8. グローバル .gitignore

### 8.1 設定

```bash
# グローバル gitignore を設定
git config --global core.excludesfile ~/.gitignore_global

# ~/.gitignore_global
cat << 'EOF' > ~/.gitignore_global
# ─── OS ───
.DS_Store
.DS_Store?
._*
Thumbs.db
Desktop.ini
ehthumbs.db
$RECYCLE.BIN/

# ─── エディタ / IDE ───
*.swp
*.swo
*~
.idea/
*.iml
.vscode/.history
.vscode/settings.json
*.code-workspace
.fleet/
*.sublime-project
*.sublime-workspace
.vs/

# ─── 環境変数 ───
.env
.env.local
.env.*.local
.env.development.local
.env.test.local
.env.production.local

# ─── ビルド / キャッシュ ───
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
pnpm-debug.log*

# ─── ランタイム ───
*.pid
*.seed
*.pid.lock

# ─── その他 ───
.direnv/
.envrc
.tool-versions
.mise.local.toml
.claude/
.cursorrules
*.bak
*.backup
*.orig
EOF
```

### 8.2 プロジェクト固有の .gitignore

```bash
# gitignore.io を使って生成
curl -sL "https://www.toptal.com/developers/gitignore/api/node,react,typescript,vscode,macos" > .gitignore

# gh CLI での生成
gh api /gitignore/templates/Node -q .source >> .gitignore

# よく使う言語・フレームワーク別テンプレート
# https://github.com/github/gitignore
```

---

## 9. 複数アカウントの使い分け

### 9.1 includeIf によるディレクトリベース切替

```ini
# ~/.gitconfig
[user]
    name = Personal Name
    email = personal@example.com
    signingkey = PERSONAL_GPG_KEY_ID

# 会社のディレクトリ
[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work

# OSS プロジェクト用
[includeIf "gitdir:~/oss/"]
    path = ~/.gitconfig-oss

# 特定のリポジトリ
[includeIf "gitdir:~/work/secret-project/"]
    path = ~/.gitconfig-secret

# リモート URL ベースの切替 (Git 2.36+)
[includeIf "hasconfig:remote.*.url:git@github-work:**"]
    path = ~/.gitconfig-work
```

```ini
# ~/.gitconfig-work
[user]
    name = Work Name
    email = work@company.com
    signingkey = WORK_GPG_KEY_ID

[core]
    sshCommand = ssh -i ~/.ssh/id_ed25519_work

# 会社のプロキシ設定
[http]
    proxy = http://proxy.company.com:8080

[url "git@github-work:"]
    insteadOf = git@github.com:company/
```

### 9.2 設定確認スクリプト

```bash
#!/bin/bash
# ~/bin/git-check-config.sh
# 現在のディレクトリの Git 設定を確認

echo "=== Git Configuration Check ==="
echo ""
echo "User:"
echo "  Name:  $(git config user.name)"
echo "  Email: $(git config user.email)"
echo ""
echo "Signing:"
echo "  Key:    $(git config user.signingkey)"
echo "  GPG:    $(git config commit.gpgsign)"
echo ""
echo "Remote:"
echo "  URL:    $(git remote get-url origin 2>/dev/null || echo 'N/A')"
echo ""
echo "SSH:"
echo "  Command: $(git config core.sshCommand || echo 'default')"
echo ""
echo "Config source:"
git config --show-origin user.email
```

---

## 10. Git Hooks

### 10.1 クライアントサイドフック

```bash
# ─── pre-commit フック ───
# .git/hooks/pre-commit (プロジェクト固有)
cat << 'HOOK' > .git/hooks/pre-commit
#!/bin/bash
# Lint & Format チェック
echo "Running pre-commit checks..."

# ステージされたファイルのみ対象
STAGED_FILES=$(git diff --cached --name-only --diff-filter=d)

# TypeScript/JavaScript のリント
TS_FILES=$(echo "$STAGED_FILES" | grep -E '\.(ts|tsx|js|jsx)$')
if [ -n "$TS_FILES" ]; then
    echo "Linting TypeScript/JavaScript files..."
    npx eslint $TS_FILES --quiet
    if [ $? -ne 0 ]; then
        echo "❌ ESLint check failed"
        exit 1
    fi
fi

# 機密情報のチェック
if git diff --cached --diff-filter=d | grep -iE '(password|secret|api_key|token)[\s]*=[\s]*["\x27][^"\x27]+'; then
    echo "❌ Possible secret detected in staged changes"
    echo "Please review and remove sensitive data"
    exit 1
fi

echo "✅ Pre-commit checks passed"
HOOK
chmod +x .git/hooks/pre-commit
```

### 10.2 Husky + lint-staged (推奨)

```bash
# Husky のセットアップ
npx husky init

# lint-staged のインストール
npm install -D lint-staged
```

```json
// package.json
{
  "lint-staged": {
    "*.{ts,tsx,js,jsx}": [
      "eslint --fix",
      "prettier --write"
    ],
    "*.{json,md,yml,yaml}": [
      "prettier --write"
    ],
    "*.css": [
      "stylelint --fix",
      "prettier --write"
    ]
  }
}
```

```bash
# .husky/pre-commit
npx lint-staged

# .husky/commit-msg
npx --no-install commitlint --edit $1

# .husky/pre-push
npm test
```

### 10.3 commitlint の設定

```bash
# インストール
npm install -D @commitlint/cli @commitlint/config-conventional

# commitlint.config.js
cat << 'EOF' > commitlint.config.js
module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [2, 'always', [
      'feat', 'fix', 'docs', 'style', 'refactor',
      'perf', 'test', 'build', 'ci', 'chore', 'revert'
    ]],
    'subject-max-length': [2, 'always', 72],
    'body-max-line-length': [2, 'always', 100],
    'header-max-length': [2, 'always', 100],
  },
};
EOF
```

---

## 11. Git LFS (Large File Storage)

### 11.1 セットアップ

```bash
# インストール
brew install git-lfs
git lfs install

# トラッキング対象の設定
git lfs track "*.psd"
git lfs track "*.zip"
git lfs track "*.mp4"
git lfs track "*.woff2"
git lfs track "*.model"      # ML モデル
git lfs track "assets/**"    # ディレクトリ単位

# .gitattributes が自動生成される
cat .gitattributes
# *.psd filter=lfs diff=lfs merge=lfs -text
# *.zip filter=lfs diff=lfs merge=lfs -text

# .gitattributes を必ずコミット
git add .gitattributes
git commit -m "chore: configure Git LFS tracking"

# LFS の状態確認
git lfs status
git lfs ls-files          # LFS で管理されているファイル一覧
git lfs env               # LFS の設定情報
```

---

## 12. アンチパターン

### 12.1 GPG / SSH 鍵にパスフレーズを設定しない

```
❌ アンチパターン: ssh-keygen 実行時にパスフレーズを空にする

問題:
  - 秘密鍵が漏洩した場合、即座に悪用される
  - セキュリティ監査で指摘される
  - チームのセキュリティポリシー違反

✅ 正しいアプローチ:
  - 必ずパスフレーズを設定する
  - SSH Agent + Keychain でパスフレーズ入力を自動化
  - ssh-add --apple-use-keychain でセッション間で持続
  - GPG Agent のキャッシュ時間を適切に設定
```

### 12.2 `git pull` をデフォルトのまま使う

```
❌ アンチパターン: pull.rebase を設定せずに git pull を実行

問題:
  - 不要なマージコミットが大量発生
  - git log が読みにくくなる

  * abc (HEAD) Merge branch 'main' into feature
  |\
  | * def (main) Update README
  * | ghi Add feature
  |/
  * jkl Previous commit

✅ 正しいアプローチ:
  - git config --global pull.rebase true
  - きれいなリニア履歴を維持

  * abc (HEAD -> feature) Add feature
  * def (main) Update README
  * jkl Previous commit
```

### 12.3 force push を main/master に実行する

```
❌ アンチパターン: git push --force origin main

問題:
  - 他のメンバーのコミットが消失する
  - CI/CD パイプラインが壊れる
  - 復旧が困難

✅ 正しいアプローチ:
  - feature ブランチでも --force-with-lease を使う
  - main/master には force push を禁止する (GitHub Branch Protection)
  - GitHub → Settings → Branches → Branch protection rules
    ✅ Do not allow force pushes
    ✅ Require pull request reviews before merging
    ✅ Require status checks to pass before merging
```

### 12.4 大きなバイナリファイルを直接コミット

```
❌ アンチパターン: 画像・動画・モデルファイルを Git に直接コミット

問題:
  - リポジトリサイズが膨張
  - clone / fetch が遅くなる
  - 一度コミットすると履歴から完全削除が困難

✅ 正しいアプローチ:
  - Git LFS を使って大きなファイルを管理
  - .gitattributes でトラッキングルールを定義
  - バイナリファイルの上限サイズを決める (例: 1MB)
  - CI で pre-commit フックによるサイズチェック
```

---

## 13. FAQ

### Q1: 会社と個人で異なる Git アカウントを使い分けるには？

**A:** `includeIf` ディレクティブを使う。`~/work/` 配下のリポジトリでは会社の設定を自動適用し、それ以外では個人の設定を使う。SSH の config も分けることで、clone 時に `git@github-work:company/repo.git` のように使い分けられる。詳細はセクション9を参照。

### Q2: `rerere` とは何？有効にすべき？

**A:** REuse REcorded REsolution の略。一度解決したコンフリクトのパターンを記録し、同じコンフリクトが再発した際に自動で同じ解決を適用する。feature ブランチを頻繁に rebase する環境では必須レベルの設定。`git config --global rerere.enabled true` で有効化する。`rerere.autoupdate = true` にすると、自動解決後にステージングも自動で行われる。

### Q3: コミット署名は必要？

**A:** 個人開発なら任意だが、チーム開発・OSS では強く推奨。GitHub の "Verified" バッジはコミットの真正性を証明する。GitHub Actions の bot コミットがなりすましでないことの確認にも使われる。Organization の設定で署名必須にすることも可能。GPG が面倒なら SSH 鍵署名 (Git 2.34+) が手軽で推奨。

### Q4: `git push --force` と `--force-with-lease` の違いは？

**A:** `--force` はリモートの状態を無条件に上書きする。他のメンバーのコミットを消す危険がある。`--force-with-lease` はリモートが自分の知っている状態と同じ場合のみ強制 push する。他の人が push していた場合は拒否される。feature ブランチの rebase 後には `--force-with-lease` を使うべき。main/master への force push は設定で禁止する。

### Q5: histogram diff とは何か？

**A:** `diff.algorithm = histogram` は patience diff の改良版で、コードの移動や構造的な変更をより賢く検出する。デフォルトの Myers diff よりも「意味のある」差分を生成する傾向がある。特に関数の追加・削除・移動が多いリファクタリング時に効果的。パフォーマンスへの影響はほぼない。

### Q6: fsmonitor とは何か？有効にすべき？

**A:** `core.fsmonitor = true` はファイルシステムの変更を監視するデーモンを使い、`git status` や `git diff` の速度を劇的に改善する。10万ファイル超の大規模リポジトリでは数十倍の高速化が見られる。Git 2.37+ の組み込み FSMonitor か、Facebook の Watchman が使える。小規模リポジトリでは効果が薄いが、有効にしてデメリットはほぼない。

---

## 14. まとめ

| 設定項目 | 推奨値 | 理由 |
|---------|--------|------|
| `init.defaultBranch` | `main` | 業界標準 |
| `pull.rebase` | `true` | きれいな履歴 |
| `fetch.prune` | `true` | 削除済みブランチの自動掃除 |
| `merge.conflictstyle` | `zdiff3` | 共通祖先表示 |
| `commit.gpgsign` | `true` | コミット検証 |
| `push.autoSetupRemote` | `true` | 初回 push の手間削減 |
| `rebase.autosquash` | `true` | fixup コミット自動整理 |
| `rebase.updateRefs` | `true` | スタックされた PR の自動更新 |
| `rerere.enabled` | `true` | コンフリクト自動再解決 |
| `diff.algorithm` | `histogram` | より賢い差分検出 |
| `core.pager` | `delta` | 美しい diff 表示 |
| `core.fsmonitor` | `true` | 大規模リポジトリの高速化 |
| `branch.sort` | `-committerdate` | 最新ブランチ優先表示 |
| `tag.sort` | `version:refname` | セマンティックバージョン順 |
| `help.autocorrect` | `prompt` | タイポ時に候補提示 |

---

## 次に読むべきガイド

- [03-ai-tools.md](./03-ai-tools.md) -- AI 開発ツールの導入
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) -- チーム共通の Git 規約
- [../01-runtime-and-package/03-linter-formatter.md](../01-runtime-and-package/03-linter-formatter.md) -- pre-commit フック連携

---

## 参考文献

1. **Pro Git (2nd Edition)** -- https://git-scm.com/book/ja/v2 -- Git の最も包括的な無料リファレンス。日本語版あり。
2. **GitHub SSH ドキュメント** -- https://docs.github.com/ja/authentication/connecting-to-github-with-ssh -- 公式の SSH 設定ガイド。
3. **git-delta** -- https://github.com/dandavison/delta -- delta の公式リポジトリ。設定例が豊富。
4. **gitconfig のベストプラクティス** -- https://jvns.ca/blog/2024/02/16/popular-git-config-options/ -- Julia Evans による実用的な解説。
5. **Conventional Commits** -- https://www.conventionalcommits.org/ja/ -- コミットメッセージ規約の標準仕様。日本語版。
6. **Git LFS Documentation** -- https://git-lfs.com/ -- Git LFS の公式ドキュメント。
7. **Husky** -- https://typicode.github.io/husky/ -- Git hooks の管理ツール。
8. **GitHub Branch Protection** -- https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-a-branch-protection-rule -- ブランチ保護ルールの設定。
9. **SSH 鍵署名 (Git Blog)** -- https://github.blog/changelog/2022-08-23-ssh-commit-verification-now-supported/ -- SSH 署名の公式アナウンス。
10. **Git Performance** -- https://git-scm.com/docs/git-maintenance -- Git のメンテナンスと最適化コマンド。
