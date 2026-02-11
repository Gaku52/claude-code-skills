# Git 設定

> Git の初期設定からGPG署名・SSH鍵・diff/mergeツール連携まで、プロフェッショナルなGit環境を構築するための完全ガイド。

## この章で学ぶこと

1. `.gitconfig` の体系的な設定と実用的なエイリアス構築
2. GPG 署名と SSH 鍵の正しい設定方法
3. diff/merge ツールの統合と credential helper の管理

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
```

### 1.2 設定ファイルの階層

```
Git 設定の優先順位 (高い順):

┌─────────────────────────────────────────────┐
│  1. Local    (.git/config)                   │
│     → リポジトリ固有の設定                    │
│     → git config --local                     │
├─────────────────────────────────────────────┤
│  2. Global   (~/.gitconfig)                  │
│     → ユーザー共通の設定                      │
│     → git config --global                    │
├─────────────────────────────────────────────┤
│  3. System   (/etc/gitconfig)                │
│     → システム全体の設定                      │
│     → git config --system                    │
└─────────────────────────────────────────────┘

※ 同じキーが複数レベルで設定された場合、
   上位（Local）の値が優先される
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

[init]
    defaultBranch = main

[color]
    ui = auto

[push]
    default = current
    autoSetupRemote = true
    followTags = true

[pull]
    rebase = true

[fetch]
    prune = true
    prunetags = true

[merge]
    conflictstyle = zdiff3
    tool = vscode

[mergetool "vscode"]
    cmd = code --wait --merge $REMOTE $LOCAL $BASE $MERGED

[diff]
    tool = vscode
    colorMoved = default
    algorithm = histogram

[difftool "vscode"]
    cmd = code --wait --diff $LOCAL $REMOTE

[rebase]
    autosquash = true
    autostash = true

[rerere]
    enabled = true

[commit]
    gpgsign = true
    verbose = true

[tag]
    gpgsign = true

[interactive]
    diffFilter = delta --color-only

[delta]
    navigate = true
    light = false
    side-by-side = true
    line-numbers = true
    syntax-theme = Dracula

[branch]
    sort = -committerdate

[column]
    ui = auto

[help]
    autocorrect = prompt
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
    c = commit
    cm = commit -m
    co = checkout
    sw = switch
    br = branch
    d = diff
    ds = diff --staged
    p = push
    pl = pull

    # ─── ログ表示 ───
    lg = log --oneline --graph --decorate --all -20
    ll = log --pretty=format:'%C(yellow)%h%C(reset) %C(green)(%cr)%C(reset) %s %C(bold blue)<%an>%C(reset)%C(red)%d%C(reset)' --abbrev-commit -20
    hist = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit --all

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

    # ファイルの変更履歴
    filelog = log --follow -p --

    # 今日のコミット
    today = log --since='midnight' --oneline --author='Your Name'

    # WIP コミット
    wip = !git add -A && git commit -m 'WIP: work in progress [skip ci]'

    # WIP 取り消し
    unwip = !git log -1 --format='%s' | grep -q 'WIP' && git reset HEAD~1
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
# GitHub → Settings → SSH and GPG keys → New SSH key
```

### 4.2 SSH config

```bash
# ~/.ssh/config
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519
    AddKeysToAgent yes
    UseKeychain yes  # macOS のみ

# 会社用アカウント (複数アカウント対応)
Host github-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
    AddKeysToAgent yes

# 使い分け:
# git clone git@github.com:personal/repo.git       # 個人
# git clone git@github-work:company/repo.git       # 会社
```

### 4.3 接続テスト

```bash
ssh -T git@github.com
# Hi username! You've successfully authenticated, but GitHub
# does not provide shell access.
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
```

### 5.2 署名の検証フロー

```
GPG 署名付きコミットのフロー:

  開発者の環境                    GitHub
  ┌──────────────┐              ┌──────────────┐
  │ git commit   │              │              │
  │      │       │              │  公開鍵を    │
  │      ▼       │              │  登録済み    │
  │ GPG 秘密鍵で │    push     │      │       │
  │ 署名を付与   │ ──────────→ │      ▼       │
  │      │       │              │  公開鍵で    │
  │      ▼       │              │  署名を検証  │
  │ 署名付き     │              │      │       │
  │ コミット     │              │      ▼       │
  │ 完成         │              │  ✅ Verified │
  └──────────────┘              └──────────────┘
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

# GitHub CLI (推奨: 全プラットフォーム共通)
gh auth login
gh auth setup-git
git config --global credential.helper '!gh auth git-credential'
```

### 6.2 認証方式比較

| 方式 | セキュリティ | 利便性 | 推奨度 |
|------|------------|--------|--------|
| SSH 鍵 | 高 | 高 | 最推奨 |
| gh auth (CLI) | 高 | 高 | 推奨 |
| Personal Access Token | 中 | 中 | 可 |
| パスワード認証 | 低 | - | 廃止済み |

---

## 7. diff / merge ツール

### 7.1 delta の設定

```bash
# delta インストール
brew install git-delta  # macOS
sudo apt install git-delta  # Ubuntu (snap)

# ~/.gitconfig で設定 (前述の [delta] セクション参照)

# 使用例
git diff              # 自動で delta が適用される
git log -p            # パッチ表示も delta で美麗に
git show HEAD         # コミット詳細も delta 対応
```

### 7.2 VS Code をdiff/mergeツールに設定

```bash
# diff ツール
git config --global diff.tool vscode
git config --global difftool.vscode.cmd 'code --wait --diff $LOCAL $REMOTE'

# merge ツール
git config --global merge.tool vscode
git config --global mergetool.vscode.cmd 'code --wait --merge $REMOTE $LOCAL $BASE $MERGED'
git config --global mergetool.keepBackup false

# 使用方法
git difftool                 # VS Code で差分表示
git mergetool                # VS Code でコンフリクト解決
```

### 7.3 コンフリクト解決のフロー

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
Thumbs.db
Desktop.ini

# ─── エディタ ───
*.swp
*.swo
*~
.idea/
*.iml
.vscode/.history
*.code-workspace

# ─── 環境 ───
.env.local
.env.*.local

# ─── ビルド ───
*.log
npm-debug.log*

# ─── その他 ───
.direnv/
.envrc
EOF
```

---

## 9. アンチパターン

### 9.1 GPG / SSH 鍵にパスフレーズを設定しない

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
```

### 9.2 `git pull` をデフォルトのまま使う

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

---

## 10. FAQ

### Q1: 会社と個人で異なる Git アカウントを使い分けるには？

**A:** `includeIf` ディレクティブを使う。

```ini
# ~/.gitconfig
[user]
    name = Personal Name
    email = personal@example.com

[includeIf "gitdir:~/work/"]
    path = ~/.gitconfig-work

# ~/.gitconfig-work
[user]
    name = Work Name
    email = work@company.com
    signingkey = WORK_GPG_KEY_ID
```

### Q2: `rerere` とは何？有効にすべき？

**A:** REuse REcorded REsolution の略。一度解決したコンフリクトのパターンを記録し、同じコンフリクトが再発した際に自動で同じ解決を適用する。feature ブランチを頻繁に rebase する環境では必須レベルの設定。`git config --global rerere.enabled true` で有効化する。

### Q3: コミット署名は必要？

**A:** 個人開発なら任意だが、チーム開発・OSS では強く推奨。GitHub の "Verified" バッジはコミットの真正性を証明する。GitHub Actions の bot コミットがなりすましでないことの確認にも使われる。Organization の設定で署名必須にすることも可能。

---

## 11. まとめ

| 設定項目 | 推奨値 | 理由 |
|---------|--------|------|
| `init.defaultBranch` | `main` | 業界標準 |
| `pull.rebase` | `true` | きれいな履歴 |
| `fetch.prune` | `true` | 削除済みブランチの自動掃除 |
| `merge.conflictstyle` | `zdiff3` | 共通祖先表示 |
| `commit.gpgsign` | `true` | コミット検証 |
| `push.autoSetupRemote` | `true` | 初回 push の手間削減 |
| `rebase.autosquash` | `true` | fixup コミット自動整理 |
| `rerere.enabled` | `true` | コンフリクト自動再解決 |
| `diff.algorithm` | `histogram` | より賢い差分検出 |
| `core.pager` | `delta` | 美しい diff 表示 |

---

## 次に読むべきガイド

- [03-ai-tools.md](./03-ai-tools.md) — AI 開発ツールの導入
- [../03-team-setup/00-project-standards.md](../03-team-setup/00-project-standards.md) — チーム共通の Git 規約
- [../01-runtime-and-package/03-linter-formatter.md](../01-runtime-and-package/03-linter-formatter.md) — pre-commit フック連携

---

## 参考文献

1. **Pro Git (2nd Edition)** — https://git-scm.com/book/ja/v2 — Git の最も包括的な無料リファレンス。日本語版あり。
2. **GitHub SSH ドキュメント** — https://docs.github.com/ja/authentication/connecting-to-github-with-ssh — 公式の SSH 設定ガイド。
3. **git-delta** — https://github.com/dandavison/delta — delta の公式リポジトリ。設定例が豊富。
4. **gitconfig のベストプラクティス** — https://jvns.ca/blog/2024/02/16/popular-git-config-options/ — Julia Evans による実用的な解説。
