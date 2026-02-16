# シェル設定

> シェルの設定をカスタマイズすることで、生産性は劇的に向上する。設定ファイルを育てることは、自分だけの最強の開発環境を構築することに等しい。

## この章で学ぶこと

- [ ] .bashrc / .zshrc の役割を理解する
- [ ] 設定ファイルの読み込み順序を把握する
- [ ] エイリアスと環境変数を設定できる
- [ ] シェル関数で複雑な処理を自動化できる
- [ ] プロンプトのカスタマイズ方法を知る
- [ ] 補完機能を最大限に活用できる
- [ ] 履歴管理を最適化できる
- [ ] モダンなシェルツールを導入・設定できる
- [ ] 複数マシン間で設定を同期する方法を知る
- [ ] トラブルシューティングの手法を習得する

---

## 1. 設定ファイルの読み込み順序

シェル設定を正しく管理するためには、設定ファイルがいつ・どの順序で読み込まれるかを理解することが不可欠である。

### 1.1 bash の設定ファイル

```
bash の読み込み順序:

■ ログインシェル（ターミナル起動時、SSH接続時）:
  /etc/profile
  → ~/.bash_profile  （存在する場合）
  → ~/.bash_login    （.bash_profile が無い場合）
  → ~/.profile       （上の2つが無い場合）

■ 非ログインシェル（新しいターミナルタブ、bash コマンド実行時）:
  ~/.bashrc

■ ログアウト時:
  ~/.bash_logout

重要なポイント:
  - ログインシェルでは .bashrc は自動的に読まれない
  - よくある対策: .bash_profile の中で .bashrc を source する
```

```bash
# ~/.bash_profile の推奨構成
# ログインシェルでも .bashrc の設定を使えるようにする

# .bashrc があれば読み込む
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi

# ログインシェル固有の設定（環境変数など）
export PATH="$HOME/.local/bin:$PATH"
export EDITOR="vim"
export VISUAL="vim"
export LANG="ja_JP.UTF-8"
export LC_ALL="ja_JP.UTF-8"
```

### 1.2 zsh の設定ファイル

```
zsh の読み込み順序（番号順に実行される）:

■ 常に読まれる:
  1. /etc/zshenv
  2. ~/.zshenv

■ ログインシェルの場合（追加で読まれる）:
  3. /etc/zprofile
  4. ~/.zprofile

■ インタラクティブシェルの場合（追加で読まれる）:
  5. /etc/zshrc
  6. ~/.zshrc

■ ログインシェルの場合（さらに追加で読まれる）:
  7. /etc/zlogin
  8. ~/.zlogin

■ ログアウト時:
  ~/.zlogout
  /etc/zlogout

実務的な使い分け:
  ~/.zshenv    → 環境変数（PATH, EDITOR）
                 ※非インタラクティブでも読まれるため注意
  ~/.zprofile  → ログインシェル固有の処理
  ~/.zshrc     → エイリアス、関数、補完設定、プロンプト設定
                 ※最も多くの設定を書く場所
  ~/.zlogin    → ログイン時の表示メッセージなど
  ~/.zlogout   → ログアウト時のクリーンアップ
```

### 1.3 設定ファイルの使い分けガイド

```bash
# ============================================
# ~/.zshenv — 環境変数（常に読まれる）
# ============================================
# 注意: このファイルは全てのzshプロセスで読まれるため、
#       出力を伴うコマンドは書かないこと

export EDITOR="vim"
export VISUAL="vim"
export PAGER="less"
export LANG="ja_JP.UTF-8"

# XDG Base Directory
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_STATE_HOME="$HOME/.local/state"

# PATH設定
typeset -U path  # 重複を自動排除（zsh固有の機能）
path=(
    "$HOME/.local/bin"
    "$HOME/.cargo/bin"
    "$HOME/go/bin"
    "/usr/local/bin"
    $path
)
export PATH
```

```bash
# ============================================
# ~/.zshrc — インタラクティブシェル設定
# ============================================
# エイリアス、関数、補完、プロンプト等をここに書く

# === エイリアス ===
alias ll='ls -lah'
alias la='ls -A'
# ... (後述の詳細セクション参照)

# === 補完設定 ===
autoload -Uz compinit && compinit
# ... (後述)

# === 履歴設定 ===
HISTSIZE=100000
SAVEHIST=100000
# ... (後述)
```

### 1.4 読み込み順序のデバッグ

```bash
# どのファイルが読まれているか確認する方法

# 方法1: 各設定ファイルの先頭に echo を追加
# ~/.zshenv に追加:
echo "Loading .zshenv"

# ~/.zshrc に追加:
echo "Loading .zshrc"

# 方法2: zsh の起動時トレース
zsh -x 2>&1 | head -50

# 方法3: zsh のファイル読み込みトレース
# SOURCE_TRACE を有効にする
zsh -o SOURCE_TRACE

# 方法4: bash のデバッグモード
bash -x --login 2>&1 | head -50

# 設定変更を反映する方法
source ~/.zshrc            # 現在のシェルに反映（再読み込み）
exec zsh                   # シェルを再起動（よりクリーン）
exec bash                  # bashの場合

# 設定ファイルの読み込みにかかる時間を計測
time zsh -i -c exit        # zshの起動時間
time bash -i -c exit       # bashの起動時間

# 設定ファイルのプロファイリング（zsh）
# ~/.zshrc の先頭に追加:
zmodload zsh/zprof
# ~/.zshrc の末尾に追加:
zprof

# これにより、どの処理に時間がかかっているか可視化できる
```

---

## 2. 環境変数

### 2.1 基本的な環境変数

```bash
# ============================================
# 基本的な環境変数の設定
# ============================================

# デフォルトエディタ
export EDITOR="vim"                    # CUIエディタ
export VISUAL="code"                   # GUIエディタ（git commit等で使用）

# ロケール設定
export LANG="ja_JP.UTF-8"            # 日本語UTF-8
export LC_ALL="ja_JP.UTF-8"          # 全カテゴリのロケール
export LC_COLLATE="C"                 # ソート順をASCII準拠に（ls等に影響）

# ページャ設定
export PAGER="less"
export LESS="-iMRSX"
# -i: 小文字検索で大文字小文字無視
# -M: 詳細なプロンプト表示
# -R: ANSIカラーコードを解釈
# -S: 長い行を折り返さない
# -X: 終了時に画面をクリアしない

# manページのカラー表示
export LESS_TERMCAP_mb=$'\e[1;31m'     # 点滅開始
export LESS_TERMCAP_md=$'\e[1;36m'     # 太字開始（シアン）
export LESS_TERMCAP_me=$'\e[0m'        # 点滅/太字終了
export LESS_TERMCAP_so=$'\e[01;33m'    # ステータスライン開始（黄色）
export LESS_TERMCAP_se=$'\e[0m'        # ステータスライン終了
export LESS_TERMCAP_us=$'\e[1;32m'     # 下線開始（緑）
export LESS_TERMCAP_ue=$'\e[0m'        # 下線終了

# ヒストリ関連
export HISTTIMEFORMAT="%F %T "         # 履歴にタイムスタンプを追加（bash）

# GPG設定（git署名等で使用）
export GPG_TTY=$(tty)
```

### 2.2 PATH の管理

```bash
# ============================================
# PATH の管理
# ============================================

# 基本的なPATH追加
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/bin:$PATH"

# 言語・フレームワーク固有のPATH

# Homebrew (macOS)
eval "$(/opt/homebrew/bin/brew shellenv)"  # Apple Silicon Mac
# eval "$(/usr/local/bin/brew shellenv)"   # Intel Mac

# Node.js (nvm)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && source "$NVM_DIR/bash_completion"

# Node.js (fnm) — nvmより高速な代替
eval "$(fnm env --use-on-cd)"

# Python (pyenv)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Ruby (rbenv)
eval "$(rbenv init -)"

# Go
export GOPATH="$HOME/go"
export PATH="$GOPATH/bin:$PATH"

# Rust
source "$HOME/.cargo/env"

# Java (jenv)
export PATH="$HOME/.jenv/bin:$PATH"
eval "$(jenv init -)"

# Deno
export DENO_INSTALL="$HOME/.deno"
export PATH="$DENO_INSTALL/bin:$PATH"

# PATHの重複排除（zsh固有）
typeset -U path

# PATHの内容確認
echo $PATH | tr ':' '\n'              # PATHを1行ずつ表示
echo $PATH | tr ':' '\n' | nl        # 番号付きで表示

# 特定のコマンドの場所を確認
which python3                          # コマンドのパス
type python3                           # より詳細な情報
where python3                          # 全候補（zsh）
```

### 2.3 プロジェクト固有の環境変数

```bash
# ============================================
# プロジェクト固有の環境変数管理
# ============================================

# direnv を使ったディレクトリ単位の環境変数管理
# brew install direnv
eval "$(direnv hook zsh)"    # ~/.zshrc に追加

# プロジェクトルートに .envrc を作成
# .envrc の例:
export DATABASE_URL="postgresql://localhost/myapp_dev"
export REDIS_URL="redis://localhost:6379"
export API_KEY="dev-api-key-12345"
export NODE_ENV="development"
export AWS_PROFILE="myproject-dev"

# layout機能で言語バージョンを自動切り替え
layout python3              # Python venv を自動作成・有効化
layout ruby                 # Ruby バージョンを自動切り替え
layout node                 # Node.js バージョンを自動切り替え

# direnv の使い方
direnv allow                # .envrc を信頼する（初回・変更時に必要）
direnv deny                 # .envrc を拒否する
direnv edit                 # .envrc を編集（保存時に自動 allow）

# ディレクトリに入ると自動的に環境変数がセットされ、
# 出ると自動的に解除される

# .envrc のセキュリティ
# - .envrc は direnv allow するまで実行されない
# - git管理する場合、機密情報は .env.local に分離
# - .envrc 内で .env.local を読み込む:
dotenv_if_exists .env.local
```

---

## 3. エイリアス

### 3.1 基本的なエイリアス

```bash
# ============================================
# 基本コマンドの改善
# ============================================

# ls 系
alias ls='ls --color=auto'         # カラー表示（Linux）
alias ls='ls -G'                   # カラー表示（macOS）
alias ll='ls -lah'                 # 詳細表示
alias la='ls -A'                   # 隠しファイル含む
alias lt='ls -lahtr'               # 更新日時の新しい順
alias lS='ls -lahS'                # サイズ順

# モダンな代替ツールがある場合
if command -v eza &>/dev/null; then
    alias ls='eza --color=auto --icons'
    alias ll='eza -lah --icons --git'
    alias la='eza -a --icons'
    alias lt='eza -la --sort=modified --icons'
    alias tree='eza --tree --icons'
fi

if command -v bat &>/dev/null; then
    alias cat='bat --paging=never'
    alias catp='bat --plain'        # プレーンモード
fi

if command -v fd &>/dev/null; then
    alias find='fd'
fi

# ディレクトリ移動
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..'
alias -- -='cd -'                   # 前のディレクトリに戻る

# mkdir は常にネスト作成
alias mkdir='mkdir -pv'

# grep にカラー表示
alias grep='grep --color=auto'
alias egrep='egrep --color=auto'
alias fgrep='fgrep --color=auto'

# 危険なコマンドの安全化
alias rm='rm -i'                   # 確認付き削除
alias cp='cp -i'                   # 確認付きコピー
alias mv='mv -i'                   # 確認付き移動
alias ln='ln -i'                   # 確認付きリンク

# 安全化を無効にしたい場合
# \rm file.txt                     # バックスラッシュでエイリアス無視
# command rm file.txt              # command で直接実行
```

### 3.2 Git エイリアス

```bash
# ============================================
# Git エイリアス
# ============================================

# 基本操作
alias g='git'
alias gs='git status'
alias ga='git add'
alias gaa='git add --all'
alias gc='git commit'
alias gcm='git commit -m'
alias gca='git commit --amend'
alias gcan='git commit --amend --no-edit'
alias gp='git push'
alias gpf='git push --force-with-lease'   # 安全なforce push
alias gpl='git pull'
alias gplr='git pull --rebase'

# ブランチ操作
alias gb='git branch'
alias gba='git branch -a'
alias gbd='git branch -d'
alias gbD='git branch -D'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gsw='git switch'
alias gswc='git switch -c'

# 差分・ログ
alias gd='git diff'
alias gds='git diff --staged'
alias gdn='git diff --name-only'
alias gl='git log --oneline --graph --decorate -20'
alias gla='git log --oneline --graph --decorate --all'
alias glp='git log --pretty=format:"%C(yellow)%h%C(reset) %C(green)(%cr)%C(reset) %s %C(blue)<%an>%C(reset)" --abbrev-commit'

# スタッシュ
alias gst='git stash'
alias gstl='git stash list'
alias gstp='git stash pop'
alias gsta='git stash apply'
alias gstd='git stash drop'

# リモート
alias gf='git fetch --all --prune'
alias grb='git rebase'
alias grbc='git rebase --continue'
alias grba='git rebase --abort'

# リセット
alias grs='git reset --soft HEAD~1'      # 直前のコミットを取り消し（変更は保持）
alias grh='git reset --hard HEAD~1'       # 直前のコミットを完全に取り消し

# cherry-pick
alias gcp='git cherry-pick'
alias gcpc='git cherry-pick --continue'
alias gcpa='git cherry-pick --abort'

# クリーンアップ
alias gclean='git clean -fd'
alias gprune='git remote prune origin'
```

### 3.3 Docker エイリアス

```bash
# ============================================
# Docker エイリアス
# ============================================

alias d='docker'
alias dc='docker compose'
alias dcu='docker compose up -d'
alias dcd='docker compose down'
alias dcr='docker compose restart'
alias dcl='docker compose logs -f'
alias dce='docker compose exec'
alias dcb='docker compose build --no-cache'

alias dps='docker ps'
alias dpsa='docker ps -a'
alias di='docker images'
alias drm='docker rm'
alias drmi='docker rmi'
alias dex='docker exec -it'
alias dlogs='docker logs -f'

# Docker クリーンアップ
alias dprune='docker system prune -af'
alias dvprune='docker volume prune -f'
alias diprune='docker image prune -af'
```

### 3.4 Kubernetes エイリアス

```bash
# ============================================
# Kubernetes エイリアス
# ============================================

alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get services'
alias kgd='kubectl get deployments'
alias kgn='kubectl get nodes'
alias kga='kubectl get all'
alias kaf='kubectl apply -f'
alias kdf='kubectl delete -f'
alias kdp='kubectl describe pod'
alias kds='kubectl describe service'
alias kdd='kubectl describe deployment'
alias kl='kubectl logs -f'
alias kex='kubectl exec -it'
alias kctx='kubectl config use-context'
alias kns='kubectl config set-context --current --namespace'

# kubectx / kubens がインストール済みの場合
# alias kctx='kubectx'
# alias kns='kubens'
```

### 3.5 その他の実用的なエイリアス

```bash
# ============================================
# ネットワーク
# ============================================
alias myip='curl -s ifconfig.me'
alias localip='ipconfig getifaddr en0'    # macOS
alias ports='netstat -tulanp'              # 使用中のポート
alias ports='lsof -i -P -n | grep LISTEN' # macOS

# ============================================
# システム情報
# ============================================
alias df='df -h'                     # ディスク使用量（人間可読）
alias du='du -h'                     # ディレクトリサイズ
alias free='free -h'                 # メモリ使用量（Linux）
alias top='htop'                     # htopがあれば使う
alias psg='ps aux | grep -v grep | grep'  # プロセス検索

# ============================================
# ファイル操作
# ============================================
alias tarx='tar -xvf'               # 展開
alias tarc='tar -czvf'              # 圧縮
alias dush='du -sh * | sort -rh'    # サイズ順でディレクトリ表示
alias count='find . -type f | wc -l'  # ファイル数カウント

# ============================================
# 開発
# ============================================
alias py='python3'
alias pip='pip3'
alias serve='python3 -m http.server 8000'  # 簡易HTTPサーバー
alias json='python3 -m json.tool'          # JSONフォーマット

# ============================================
# クリップボード（macOS）
# ============================================
alias pbp='pbpaste'
alias pbc='pbcopy'
alias copy='pbcopy'
alias paste='pbpaste'

# ============================================
# タイムスタンプ
# ============================================
alias now='date +"%Y-%m-%d %H:%M:%S"'
alias timestamp='date +%s'
alias week='date +%V'
```

### 3.6 グローバルエイリアス（zsh限定）

```bash
# ============================================
# グローバルエイリアス（コマンドの途中でも展開される）
# ============================================

# パイプの省略形
alias -g G='| grep'
alias -g L='| less'
alias -g H='| head'
alias -g T='| tail'
alias -g S='| sort'
alias -g U='| uniq'
alias -g W='| wc -l'
alias -g C='| pbcopy'          # macOS クリップボード
alias -g J='| jq .'           # JSON整形
alias -g N='> /dev/null 2>&1' # 出力破棄

# 使用例:
# ps aux G nginx           → ps aux | grep nginx
# cat file.txt L           → cat file.txt | less
# ls -la S                 → ls -la | sort
# curl api.example.com J   → curl api.example.com | jq .

# サフィックスエイリアス（拡張子でコマンドを自動選択）
alias -s md='code'             # .md ファイルを code で開く
alias -s json='code'           # .json ファイルを code で開く
alias -s py='python3'          # .py ファイルを python3 で実行
alias -s sh='bash'             # .sh ファイルを bash で実行
alias -s txt='less'            # .txt ファイルを less で表示
alias -s log='less'            # .log ファイルを less で表示

# 使用例:
# README.md                → code README.md
# script.py                → python3 script.py
```

---

## 4. シェル関数

### 4.1 基本的なシェル関数

```bash
# ============================================
# ディレクトリ作成して移動
# ============================================
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# ============================================
# ファイルのバックアップを作成
# ============================================
bak() {
    local file="$1"
    if [ -z "$file" ]; then
        echo "Usage: bak <file>"
        return 1
    fi
    cp -a "$file" "${file}.bak.$(date +%Y%m%d_%H%M%S)"
    echo "Backed up: ${file}.bak.$(date +%Y%m%d_%H%M%S)"
}

# ============================================
# アーカイブの展開（形式を自動判別）
# ============================================
extract() {
    if [ -f "$1" ]; then
        case "$1" in
            *.tar.bz2)   tar xjf "$1"    ;;
            *.tar.gz)    tar xzf "$1"    ;;
            *.tar.xz)    tar xJf "$1"    ;;
            *.tar.zst)   tar --zstd -xf "$1" ;;
            *.bz2)       bunzip2 "$1"    ;;
            *.rar)       unrar x "$1"    ;;
            *.gz)        gunzip "$1"     ;;
            *.tar)       tar xf "$1"     ;;
            *.tbz2)      tar xjf "$1"    ;;
            *.tgz)       tar xzf "$1"    ;;
            *.zip)       unzip "$1"      ;;
            *.Z)         uncompress "$1" ;;
            *.7z)        7z x "$1"       ;;
            *.xz)        unxz "$1"       ;;
            *.zst)       unzstd "$1"     ;;
            *)           echo "'$1' cannot be extracted via extract()" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# ============================================
# 指定したポートで何が動いているか確認
# ============================================
port() {
    lsof -i :"$1"
}

# ============================================
# 特定のポートのプロセスをkill
# ============================================
killport() {
    local port="$1"
    if [ -z "$port" ]; then
        echo "Usage: killport <port>"
        return 1
    fi
    local pid=$(lsof -ti :"$port")
    if [ -n "$pid" ]; then
        echo "Killing process $pid on port $port"
        kill -9 "$pid"
    else
        echo "No process found on port $port"
    fi
}

# ============================================
# ディレクトリのサイズを見やすく表示
# ============================================
dirsize() {
    du -sh "${1:-.}"/* 2>/dev/null | sort -rh | head -20
}

# ============================================
# Git関連の便利関数
# ============================================

# 新しいブランチを作ってpush
gnew() {
    git checkout -b "$1" && git push -u origin "$1"
}

# コミットメッセージ付きで一括add&commit
gac() {
    git add --all && git commit -m "$*"
}

# 直近のコミットログをfzfで検索してcheckout
gshow() {
    local commit
    commit=$(git log --oneline --graph --decorate --all | fzf --preview 'git show {2}' | awk '{print $2}')
    [ -n "$commit" ] && git show "$commit"
}

# ブランチをfzfで選択して切り替え
fbr() {
    local branch
    branch=$(git branch -a | sed 's/^\*//' | sed 's/^ *//' | fzf --preview 'git log --oneline --graph -20 {}')
    [ -n "$branch" ] && git checkout "$branch"
}

# ============================================
# fzf を使った検索関数
# ============================================

# ファイルをfzfで検索してエディタで開く
fe() {
    local file
    file=$(fzf --preview 'bat --color=always --line-range :100 {}' --preview-window=right:60%)
    [ -n "$file" ] && ${EDITOR:-vim} "$file"
}

# ディレクトリをfzfで検索してcd
fcd() {
    local dir
    dir=$(find . -type d -not -path '*/\.*' 2>/dev/null | fzf --preview 'ls -la {}')
    [ -n "$dir" ] && cd "$dir"
}

# コマンド履歴をfzfで検索
fh() {
    local cmd
    cmd=$(history | sort -rn | awk '{$1=""; print $0}' | sed 's/^ //' | sort -u | fzf)
    [ -n "$cmd" ] && eval "$cmd"
}

# プロセスをfzfで検索してkill
fkill() {
    local pid
    pid=$(ps aux | sed 1d | fzf -m --header='Select process to kill' | awk '{print $2}')
    [ -n "$pid" ] && echo "$pid" | xargs kill -9
}

# ============================================
# ネットワーク関連
# ============================================

# HTTP ステータスコードを確認
httpstatus() {
    curl -o /dev/null -s -w "%{http_code}\n" "$1"
}

# SSL証明書の有効期限確認
sslexpiry() {
    echo | openssl s_client -connect "$1":443 -servername "$1" 2>/dev/null | openssl x509 -noout -enddate
}

# DNS情報を見やすく表示
dns() {
    echo "--- A Record ---"
    dig +short "$1" A
    echo "--- AAAA Record ---"
    dig +short "$1" AAAA
    echo "--- MX Record ---"
    dig +short "$1" MX
    echo "--- NS Record ---"
    dig +short "$1" NS
    echo "--- TXT Record ---"
    dig +short "$1" TXT
}

# ============================================
# 計算関数
# ============================================
calc() {
    echo "scale=4; $*" | bc -l
}

# ファイルサイズを人間可読で表示
fsize() {
    if [ -f "$1" ]; then
        ls -lh "$1" | awk '{print $5, $9}'
    else
        echo "File not found: $1"
    fi
}

# ============================================
# 開発関連
# ============================================

# Node.jsプロジェクトの初期化
node-init() {
    mkdir -p "$1" && cd "$1"
    npm init -y
    echo "node_modules/" > .gitignore
    echo "dist/" >> .gitignore
    echo ".env" >> .gitignore
    git init
    echo "Project '$1' initialized"
}

# Docker コンテナに入る
denter() {
    docker exec -it "$1" /bin/sh -c "if command -v bash > /dev/null; then bash; else sh; fi"
}

# ============================================
# テキスト処理
# ============================================

# 文字列のBase64エンコード/デコード
b64e() { echo -n "$1" | base64; }
b64d() { echo -n "$1" | base64 --decode; echo; }

# URLエンコード/デコード
urlencode() { python3 -c "import urllib.parse; print(urllib.parse.quote('$1'))"; }
urldecode() { python3 -c "import urllib.parse; print(urllib.parse.unquote('$1'))"; }

# UUIDを生成
uuid() { python3 -c "import uuid; print(uuid.uuid4())"; }

# ランダムパスワード生成
genpass() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d '/+=' | cut -c1-"$length"
}
```

---

## 5. プロンプトのカスタマイズ

### 5.1 bash のプロンプト

```bash
# ============================================
# bash プロンプトの基本
# ============================================

# PS1 の特殊文字
# \u: ユーザー名
# \h: ホスト名（短縮）
# \H: ホスト名（完全）
# \w: カレントディレクトリ（フルパス）
# \W: カレントディレクトリ（ベース名のみ）
# \d: 日付
# \t: 時刻（24時間制 HH:MM:SS）
# \T: 時刻（12時間制 HH:MM:SS）
# \@: 時刻（12時間制 AM/PM）
# \n: 改行
# \$: root なら #, 一般ユーザーなら $
# \!: 履歴番号
# \#: コマンド番号
# \[...\]: 非表示文字の囲み（プロンプト長の計算から除外）

# シンプルなプロンプト
export PS1='\u@\h:\w\$ '

# カラー付きプロンプト
export PS1='\[\e[1;32m\]\u@\h\[\e[0m\]:\[\e[1;34m\]\w\[\e[0m\]\$ '
# 緑のユーザー名@ホスト名、青のディレクトリ

# Git ブランチ表示付きプロンプト
parse_git_branch() {
    git branch 2>/dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}

parse_git_status() {
    local status=$(git status --porcelain 2>/dev/null)
    if [ -n "$status" ]; then
        echo "*"  # 未コミットの変更あり
    fi
}

export PS1='\[\e[1;32m\]\u\[\e[0m\]:\[\e[1;34m\]\w\[\e[0;33m\]$(parse_git_branch)$(parse_git_status)\[\e[0m\]\$ '

# 複数行のプロンプト（情報量が多い場合）
export PS1='\n\[\e[1;32m\]\u@\h\[\e[0m\] \[\e[1;34m\]\w\[\e[0;33m\]$(parse_git_branch)\[\e[0m\]\n\$ '

# PS2: 継続行のプロンプト
export PS2='> '

# PS4: デバッグ用プロンプト（set -x 時）
export PS4='+ ${BASH_SOURCE}:${LINENO}: ${FUNCNAME[0]:+${FUNCNAME[0]}(): }'
```

### 5.2 zsh のプロンプト

```bash
# ============================================
# zsh プロンプトの基本
# ============================================

# zsh のプロンプト変数
# %n: ユーザー名
# %m: ホスト名（短縮）
# %M: ホスト名（完全）
# %~: カレントディレクトリ（~省略）
# %/: カレントディレクトリ（フルパス）
# %d: 日付
# %T: 時刻（HH:MM）
# %*: 時刻（HH:MM:SS）
# %#: root なら #, 一般ユーザーなら %
# %?: 直前のコマンドの終了コード
# %F{color}...%f: フォアグラウンドカラー
# %B...%b: 太字
# %U...%u: 下線

# シンプルなプロンプト
PROMPT='%n@%m:%~%# '

# カラー付きプロンプト
PROMPT='%F{green}%n%f@%F{blue}%m%f:%F{yellow}%~%f %# '

# Git情報を表示（vcs_info）
autoload -Uz vcs_info
precmd_vcs_info() { vcs_info }
precmd_functions+=( precmd_vcs_info )
setopt prompt_subst

zstyle ':vcs_info:*' enable git
zstyle ':vcs_info:*' formats ' %F{magenta}(%b)%f'
zstyle ':vcs_info:*' actionformats ' %F{magenta}(%b|%a)%f'
zstyle ':vcs_info:git:*' check-for-changes true
zstyle ':vcs_info:git:*' stagedstr '%F{green}+%f'
zstyle ':vcs_info:git:*' unstagedstr '%F{red}*%f'
zstyle ':vcs_info:git:*' formats ' %F{magenta}(%b%c%u)%f'

PROMPT='%F{green}%n%f:%F{blue}%~%f${vcs_info_msg_0_} %# '

# 右側プロンプト（RPROMPT）
RPROMPT='%F{gray}%*%f'                   # 時刻を右側に表示
RPROMPT='%(?..%F{red}[%?]%f) %F{gray}%*%f'  # エラー時に終了コードも表示

# コマンド実行時にRPROMPTを消す
setopt TRANSIENT_RPROMPT
```

### 5.3 Starship（モダンなプロンプト）

```bash
# ============================================
# Starship のインストールと設定
# ============================================

# インストール
brew install starship            # macOS
# curl -sS https://starship.rs/install.sh | sh   # Linux

# シェルに設定を追加
# ~/.zshrc に追加:
eval "$(starship init zsh)"

# ~/.bashrc に追加:
eval "$(starship init bash)"

# Starship の設定ファイル: ~/.config/starship.toml
```

```toml
# ~/.config/starship.toml

# プロンプトの全体フォーマット
format = """
$username\
$hostname\
$directory\
$git_branch\
$git_status\
$python\
$nodejs\
$rust\
$golang\
$docker_context\
$kubernetes\
$aws\
$line_break\
$character"""

# ディレクトリ設定
[directory]
truncation_length = 5
truncate_to_repo = true
style = "bold blue"
format = "[$path]($style)[$read_only]($read_only_style) "

# Git ブランチ
[git_branch]
symbol = " "
style = "bold purple"
format = "on [$symbol$branch]($style) "

# Git ステータス
[git_status]
conflicted = "="
ahead = "⇡${count}"
behind = "⇣${count}"
diverged = "⇕⇡${ahead_count}⇣${behind_count}"
untracked = "?${count}"
stashed = "$${count}"
modified = "!${count}"
staged = "+${count}"
renamed = "»${count}"
deleted = "✘${count}"
format = '([$all_status$ahead_behind]($style) )'

# プロンプトのキャラクター
[character]
success_symbol = "[❯](bold green)"
error_symbol = "[❯](bold red)"

# Python
[python]
symbol = " "
format = 'via [${symbol}${pyenv_prefix}(${version} )(\($virtualenv\) )]($style)'

# Node.js
[nodejs]
symbol = " "
format = "via [$symbol($version )]($style)"

# Docker
[docker_context]
symbol = " "
format = "via [$symbol$context]($style) "

# Kubernetes
[kubernetes]
disabled = false
symbol = "☸ "
format = '[$symbol$context( \($namespace\))]($style) '

# AWS
[aws]
symbol = " "
format = '[$symbol($profile )(\($region\) )]($style)'

# 実行時間
[cmd_duration]
min_time = 2_000          # 2秒以上のコマンドで表示
format = "took [$duration]($style) "
```

### 5.4 Oh My Zsh

```bash
# ============================================
# Oh My Zsh のインストールと設定
# ============================================

# インストール
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# ~/.zshrc での設定
export ZSH="$HOME/.oh-my-zsh"

# テーマ設定
ZSH_THEME="robbyrussell"          # デフォルト
# ZSH_THEME="agnoster"            # 人気テーマ
# ZSH_THEME="powerlevel10k/powerlevel10k"  # 高機能テーマ

# プラグイン設定
plugins=(
    git                           # Git エイリアスと補完
    zsh-autosuggestions           # コマンド入力時にサジェスト
    zsh-syntax-highlighting       # コマンドのシンタックスハイライト
    docker                        # Docker 補完
    docker-compose                # docker compose 補完
    kubectl                       # kubectl 補完
    aws                           # AWS CLI 補完
    node                          # Node.js 関連
    npm                           # npm 補完
    python                        # Python 関連
    pip                           # pip 補完
    brew                          # Homebrew 補完
    macos                         # macOS ユーティリティ
    fzf                           # fzf 連携
    z                             # ディレクトリ高速移動
    history-substring-search      # 履歴のサブストリング検索
    colored-man-pages             # manページのカラー表示
    extract                       # 各種アーカイブの展開
    web-search                    # ターミナルからWeb検索
    copypath                      # 現在のパスをコピー
    copybuffer                    # 現在のコマンドラインをコピー
    direnv                        # direnv 連携
)

source $ZSH/oh-my-zsh.sh

# 追加プラグインのインストール（Oh My Zsh のカスタムプラグイン）
# git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM}/plugins/zsh-autosuggestions
# git clone https://github.com/zsh-users/zsh-syntax-highlighting ${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting
```

---

## 6. 履歴管理

### 6.1 履歴の設定

```bash
# ============================================
# bash の履歴設定
# ============================================
export HISTSIZE=100000               # メモリ上の履歴数
export HISTFILESIZE=200000           # ファイルに保存する履歴数
export HISTFILE=~/.bash_history      # 履歴ファイルのパス
export HISTCONTROL=ignoreboth        # 重複と空白開始を無視
# ignoredups:   連続する重複を無視
# ignorespace:  スペース開始のコマンドを無視
# ignoreboth:   上記両方
# erasedups:    全履歴から重複を削除

export HISTTIMEFORMAT="%F %T "       # タイムスタンプ付き
export HISTIGNORE="ls:cd:pwd:exit:clear:history"  # 記録しないコマンド

shopt -s histappend                  # 追記モード（上書きしない）
shopt -s cmdhist                     # 複数行コマンドを1行で保存

# プロンプト表示ごとに履歴を保存（複数ターミナル対応）
PROMPT_COMMAND="history -a; history -c; history -r; $PROMPT_COMMAND"
```

```bash
# ============================================
# zsh の履歴設定
# ============================================
HISTSIZE=100000                      # メモリ上の履歴数
SAVEHIST=200000                      # ファイルに保存する履歴数
HISTFILE=~/.zsh_history              # 履歴ファイルのパス

# 履歴のオプション
setopt SHARE_HISTORY                 # 複数ターミナルで履歴を共有
setopt HIST_IGNORE_DUPS              # 連続する重複を無視
setopt HIST_IGNORE_ALL_DUPS          # 全履歴から重複を削除
setopt HIST_IGNORE_SPACE             # スペース開始のコマンドを無視
setopt HIST_FIND_NO_DUPS             # 検索時に重複を表示しない
setopt HIST_REDUCE_BLANKS            # 余分な空白を削除
setopt HIST_VERIFY                   # 履歴展開を即実行せず確認
setopt HIST_SAVE_NO_DUPS             # 保存時に重複を除去
setopt HIST_EXPIRE_DUPS_FIRST        # 容量超過時に重複を先に削除
setopt INC_APPEND_HISTORY            # コマンド実行ごとに即時保存
setopt EXTENDED_HISTORY              # タイムスタンプを記録

# 履歴から除外するコマンドのパターン
HISTORY_IGNORE="(ls|cd|pwd|exit|clear|history|bg|fg)"
```

### 6.2 履歴の活用テクニック

```bash
# ============================================
# 履歴の検索と再利用
# ============================================

# Ctrl+R: インタラクティブ逆方向検索（最も頻繁に使う）
# 入力開始 → 候補表示 → Ctrl+R で次の候補 → Enter で実行

# 履歴展開
!!                    # 直前のコマンドを再実行
!$                    # 直前のコマンドの最後の引数
!^                    # 直前のコマンドの最初の引数
!*                    # 直前のコマンドの全引数
!n                    # 履歴番号nのコマンド
!-n                   # n個前のコマンド
!string               # stringで始まる最新のコマンド
!?string?             # stringを含む最新のコマンド

# 修飾子
!!:s/old/new          # 直前のコマンドのold→new置換
^old^new              # 上記の短縮形

# Alt+. （Option+.）: 直前のコマンドの最後の引数を挿入
# 繰り返し押すと、さらに前のコマンドの最後の引数

# 履歴の管理コマンド
history               # 履歴一覧
history 20            # 直近20件
history -c            # 履歴クリア（メモリ上）
history -w            # 履歴をファイルに書き込み
fc -l 1               # 全履歴表示（zsh）
fc -l -20             # 直近20件（zsh）

# fzf を使った高度な履歴検索
# Ctrl+R が fzf のインターフェースに置き換わる（fzf インストール時）
```

### 6.3 機密情報の履歴への記録を防ぐ

```bash
# ============================================
# セキュリティ: 機密情報の履歴記録防止
# ============================================

# 方法1: コマンドの先頭にスペースを付ける（HIST_IGNORE_SPACE が有効な場合）
 export API_KEY="secret-key-12345"   # 先頭にスペース → 履歴に残らない

# 方法2: 環境変数ファイルから読み込む
source ~/.env.secret

# 方法3: キーチェーン/シークレットマネージャを使う
# macOS の場合:
security find-generic-password -s "myapp" -w  # キーチェーンから取得

# 方法4: 特定のコマンドを履歴から除外
HISTIGNORE="*secret*:*password*:*token*:*API_KEY*"

# 履歴ファイルのパーミッション
chmod 600 ~/.zsh_history
chmod 600 ~/.bash_history
```

---

## 7. 補完機能の設定

### 7.1 zsh の補完設定

```bash
# ============================================
# zsh 補完の詳細設定
# ============================================

# 補完システムの初期化
autoload -Uz compinit
compinit

# 補完のキャッシュ（起動時間短縮）
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$XDG_CACHE_HOME/zsh/zcompcache"

# 大文字小文字を無視した補完
zstyle ':completion:*' matcher-list 'm:{a-zA-Z}={A-Za-z}' 'r:|=*' 'l:|=* r:|=*'

# 補完メニューの有効化
zstyle ':completion:*' menu select

# 補完候補のグループ化
zstyle ':completion:*' group-name ''
zstyle ':completion:*:descriptions' format '%F{yellow}-- %d --%f'
zstyle ':completion:*:corrections' format '%F{green}-- %d (errors: %e) --%f'
zstyle ':completion:*:messages' format '%F{purple}-- %d --%f'
zstyle ':completion:*:warnings' format '%F{red}-- no matches found --%f'

# 補完候補のカラー表示
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}"

# ディレクトリの補完時にスラッシュを自動追加
zstyle ':completion:*' squeeze-slashes true

# killコマンドの補完でプロセス名を表示
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#)*=0=01;31'
zstyle ':completion:*:*:kill:*' menu yes select
zstyle ':completion:*:kill:*' force-list always

# SSH/SCP のホスト名補完
zstyle ':completion:*:ssh:*' hosts $(awk '/^Host / && !/\*/{print $2}' ~/.ssh/config 2>/dev/null)
zstyle ':completion:*:scp:*' hosts $(awk '/^Host / && !/\*/{print $2}' ~/.ssh/config 2>/dev/null)

# man ページのセクション補完
zstyle ':completion:*:manuals' separate-sections true
zstyle ':completion:*:manuals.(^1*)' insert-sections true

# 補完時のキーバインド
bindkey '^[[Z' reverse-menu-complete   # Shift+Tab で逆方向の補完
```

### 7.2 bash の補完設定

```bash
# ============================================
# bash 補完の設定
# ============================================

# bash-completion パッケージの読み込み
if [ -f /etc/bash_completion ]; then
    source /etc/bash_completion
elif [ -f /usr/share/bash-completion/bash_completion ]; then
    source /usr/share/bash-completion/bash_completion
fi

# macOS (Homebrew) の場合
if [ -f "$(brew --prefix)/etc/bash_completion" ]; then
    source "$(brew --prefix)/etc/bash_completion"
fi

# 大文字小文字を無視した補完
bind "set completion-ignore-case on"

# 部分一致で補完候補を表示
bind "set show-all-if-ambiguous on"

# Tab1回で候補表示
bind "set show-all-if-unmodified on"

# カラー表示
bind "set colored-stats on"

# 補完時にファイルタイプを表示
bind "set visible-stats on"

# 補完候補をページングせず一度に表示
bind "set page-completions off"

# シンボリックリンクの補完時にスラッシュを追加
bind "set mark-symlinked-directories on"
```

### 7.3 各種ツールの補完設定

```bash
# ============================================
# ツール固有の補完設定
# ============================================

# Docker
if command -v docker &>/dev/null; then
    # Docker の補完（zsh）
    # docker completion は通常自動で有効
    # 手動設定が必要な場合:
    # mkdir -p ~/.zsh/completions
    # docker completion zsh > ~/.zsh/completions/_docker
    fpath=(~/.zsh/completions $fpath)
fi

# kubectl
if command -v kubectl &>/dev/null; then
    source <(kubectl completion zsh)      # zsh
    # source <(kubectl completion bash)   # bash
fi

# Helm
if command -v helm &>/dev/null; then
    source <(helm completion zsh)
fi

# AWS CLI
if command -v aws_completer &>/dev/null; then
    complete -C aws_completer aws         # bash
    # autoload bashcompinit; bashcompinit # zsh で bash 補完を使う場合
    # complete -C aws_completer aws
fi

# Terraform
if command -v terraform &>/dev/null; then
    complete -C terraform terraform       # bash
    autoload -U +X bashcompinit && bashcompinit
    complete -o nospace -C terraform terraform  # zsh
fi

# GitHub CLI
if command -v gh &>/dev/null; then
    eval "$(gh completion -s zsh)"
fi

# npm の補完
if command -v npm &>/dev/null; then
    eval "$(npm completion)"
fi

# pip の補完
if command -v pip3 &>/dev/null; then
    eval "$(pip3 completion --zsh)"       # zsh
    # eval "$(pip3 completion --bash)"    # bash
fi

# rustup と cargo の補完
if command -v rustup &>/dev/null; then
    eval "$(rustup completions zsh)"
    eval "$(rustup completions zsh cargo)"
fi
```

---

## 8. モダンなシェルツールの導入

### 8.1 必須ツール一覧

```bash
# ============================================
# モダンなCLIツールのインストール（macOS）
# ============================================

# パッケージマネージャ
brew install \
    fzf              # ファジー検索（最重要）
    zoxide           # スマートcd（z コマンド）
    bat              # cat の改良版（シンタックスハイライト）
    eza              # ls の改良版（アイコン、Git連携）
    ripgrep          # grep の高速版（rg）
    fd               # find の高速版
    delta            # diff の改良版（Git diff用）
    jq               # JSON処理
    yq               # YAML処理
    tldr             # manの簡易版（使用例中心）
    htop             # top の改良版
    ncdu             # ディスク使用量の可視化
    starship         # モダンなプロンプト
    direnv           # ディレクトリ単位の環境変数

# Linuxの場合（Ubuntu/Debian）
sudo apt install -y \
    fzf zoxide bat exa ripgrep fd-find \
    jq delta htop ncdu direnv
```

### 8.2 fzf の設定

```bash
# ============================================
# fzf の詳細設定
# ============================================

# fzf のインストール後の設定
# $(brew --prefix)/opt/fzf/install   # インストールスクリプト実行

# デフォルトオプション
export FZF_DEFAULT_OPTS='
    --height 60%
    --layout=reverse
    --border rounded
    --info inline
    --multi
    --preview-window=right:60%:wrap
    --bind "ctrl-a:select-all"
    --bind "ctrl-d:deselect-all"
    --bind "ctrl-t:toggle-all"
    --bind "ctrl-/:toggle-preview"
    --color=fg:#c0caf5,bg:#1a1b26,hl:#ff9e64
    --color=fg+:#c0caf5,bg+:#292e42,hl+:#ff9e64
    --color=info:#7aa2f7,prompt:#7dcfff,pointer:#ff007c
    --color=marker:#9ece6a,spinner:#9ece6a,header:#9ece6a
'

# デフォルトの検索コマンド
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'

# Ctrl+T: ファイル検索
export FZF_CTRL_T_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_CTRL_T_OPTS='
    --preview "bat --color=always --line-range :100 {}"
    --bind "enter:become(${EDITOR:-vim} {+})"
'

# Alt+C: ディレクトリ検索してcd
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'
export FZF_ALT_C_OPTS='
    --preview "eza --tree --level=2 --icons {}"
'

# Ctrl+R: コマンド履歴検索
export FZF_CTRL_R_OPTS='
    --preview "echo {}"
    --preview-window=down:3:hidden:wrap
    --bind "ctrl-/:toggle-preview"
'

# fzf の読み込み
[ -f ~/.fzf.zsh ] && source ~/.fzf.zsh
```

### 8.3 zoxide の設定

```bash
# ============================================
# zoxide の設定
# ============================================

# 初期化（~/.zshrc に追加）
eval "$(zoxide init zsh)"

# bash の場合
# eval "$(zoxide init bash)"

# 使い方
z projects             # 過去の訪問履歴から最適な "projects" ディレクトリに移動
z proj                 # 部分一致でも移動可能
zi                     # インタラクティブ選択（fzf連携）
z -                    # 前のディレクトリに戻る

# zoxide のデータベース操作
zoxide query           # データベースの内容を表示
zoxide query -l        # スコア付きで表示
zoxide add /path       # パスを手動追加
zoxide remove /path    # パスを手動削除
```

### 8.4 Git diff の改善（delta）

```bash
# ============================================
# delta の設定（~/.gitconfig に追加）
# ============================================
```

```ini
# ~/.gitconfig
[core]
    pager = delta

[interactive]
    diffFilter = delta --color-only

[delta]
    navigate = true
    side-by-side = true
    line-numbers = true
    syntax-theme = Dracula
    plus-style = "syntax #003800"
    minus-style = "syntax #3f0001"
    plus-emph-style = "syntax #006000"
    minus-emph-style = "syntax #600000"

[merge]
    conflictstyle = diff3

[diff]
    colorMoved = default
```

---

## 9. 設定の同期とバージョン管理

### 9.1 dotfiles リポジトリの構成

```bash
# ============================================
# dotfiles の管理
# ============================================

# dotfiles リポジトリの構成例
# ~/.dotfiles/
# ├── zsh/
# │   ├── .zshrc
# │   ├── .zshenv
# │   └── .zprofile
# ├── bash/
# │   ├── .bashrc
# │   └── .bash_profile
# ├── git/
# │   ├── .gitconfig
# │   └── .gitignore_global
# ├── vim/
# │   └── .vimrc
# ├── config/
# │   ├── starship.toml
# │   └── ...
# ├── install.sh
# ├── Makefile
# └── README.md

# シンボリックリンクでセットアップ
ln -sf ~/.dotfiles/zsh/.zshrc ~/.zshrc
ln -sf ~/.dotfiles/zsh/.zshenv ~/.zshenv
ln -sf ~/.dotfiles/git/.gitconfig ~/.gitconfig
ln -sf ~/.dotfiles/config/starship.toml ~/.config/starship.toml
```

### 9.2 自動セットアップスクリプト

```bash
#!/bin/bash
# install.sh — dotfiles の自動セットアップ

set -euo pipefail

DOTFILES_DIR="$HOME/.dotfiles"

echo "=== Setting up dotfiles ==="

# シンボリックリンクの作成
create_symlink() {
    local src="$1"
    local dst="$2"

    if [ -e "$dst" ] && [ ! -L "$dst" ]; then
        echo "Backing up existing $dst to ${dst}.bak"
        mv "$dst" "${dst}.bak"
    fi

    ln -sf "$src" "$dst"
    echo "Linked: $src -> $dst"
}

# zsh
create_symlink "$DOTFILES_DIR/zsh/.zshrc" "$HOME/.zshrc"
create_symlink "$DOTFILES_DIR/zsh/.zshenv" "$HOME/.zshenv"

# git
create_symlink "$DOTFILES_DIR/git/.gitconfig" "$HOME/.gitconfig"
create_symlink "$DOTFILES_DIR/git/.gitignore_global" "$HOME/.gitignore_global"

# Starship
mkdir -p "$HOME/.config"
create_symlink "$DOTFILES_DIR/config/starship.toml" "$HOME/.config/starship.toml"

# macOS の場合: Homebrew パッケージのインストール
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "=== Installing Homebrew packages ==="
    if ! command -v brew &>/dev/null; then
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew bundle --file="$DOTFILES_DIR/Brewfile"
fi

echo "=== Setup complete ==="
```

### 9.3 Brewfile での依存管理

```ruby
# Brewfile — Homebrew のパッケージ管理
# 使い方: brew bundle --file=Brewfile

# タップ
tap "homebrew/cask-fonts"

# CLI ツール
brew "zsh"
brew "fzf"
brew "zoxide"
brew "bat"
brew "eza"
brew "ripgrep"
brew "fd"
brew "git-delta"
brew "jq"
brew "yq"
brew "tldr"
brew "htop"
brew "ncdu"
brew "starship"
brew "direnv"
brew "gh"
brew "gnupg"
brew "wget"
brew "tree"
brew "watch"
brew "tmux"

# 開発ツール
brew "node"
brew "python@3"
brew "go"
brew "rustup"

# GUI アプリ（Cask）
cask "visual-studio-code"
cask "iterm2"
cask "docker"
cask "rectangle"

# フォント（Nerd Fonts）
cask "font-hack-nerd-font"
cask "font-fira-code-nerd-font"
cask "font-jetbrains-mono-nerd-font"
```

---

## 10. トラブルシューティング

### 10.1 よくある問題と解決策

```bash
# ============================================
# シェル設定のトラブルシューティング
# ============================================

# 問題1: 設定が反映されない
# 原因: 設定ファイルを読み込んでいない、または誤ったファイルに書いている
source ~/.zshrc                    # 手動で再読み込み
exec zsh                           # シェルを完全に再起動
echo $SHELL                        # 現在のデフォルトシェルを確認
cat /etc/shells                    # 利用可能なシェル一覧
chsh -s $(which zsh)               # デフォルトシェルを変更

# 問題2: コマンドが見つからない（command not found）
which command_name                 # コマンドの場所を確認
echo $PATH                        # PATHの内容確認
type command_name                  # コマンドの種類確認
hash -r                            # コマンドのハッシュテーブルをリセット（bash）
rehash                             # コマンドのハッシュテーブルをリセット（zsh）

# 問題3: エイリアスが効かない
alias                              # 定義済みエイリアスの一覧
type command_name                  # そのコマンドがエイリアスか確認
unalias command_name               # エイリアスを解除

# 問題4: シェルの起動が遅い
time zsh -i -c exit                # 起動時間を計測
# zshrc の先頭に zmodload zsh/zprof、末尾に zprof を追加
# → どの処理が遅いか特定

# 一般的な遅延原因:
# - nvm の読み込み（遅い場合は fnm に乗り換え）
# - compinit の重複実行
# - 大量のプラグイン読み込み
# - ネットワークアクセスを伴う処理

# 問題5: 文字化け
locale                             # ロケール設定確認
echo $LANG                        # LANG確認
# 対策: export LANG="ja_JP.UTF-8" を .zshrc に追加

# 問題6: 補完が効かない
rm -f ~/.zcompdump*                # 補完キャッシュを削除
autoload -Uz compinit && compinit  # 補完システムを再初期化
```

### 10.2 設定ファイルのベストプラクティス

```bash
# ============================================
# 設定ファイル管理のベストプラクティス
# ============================================

# 1. 設定をモジュール化する
# ~/.zshrc から個別ファイルを読み込む構成

# ~/.zshrc
for config_file in ~/.zsh/conf.d/*.zsh(N); do
    source "$config_file"
done

# ~/.zsh/conf.d/
# ├── 01-env.zsh          # 環境変数
# ├── 02-history.zsh       # 履歴設定
# ├── 03-completion.zsh    # 補完設定
# ├── 04-aliases.zsh       # エイリアス
# ├── 05-functions.zsh     # 関数
# ├── 06-keybindings.zsh   # キーバインド
# ├── 07-prompt.zsh        # プロンプト
# ├── 08-tools.zsh         # ツール設定
# └── 99-local.zsh         # マシン固有設定

# 2. マシン固有の設定を分離する
if [ -f ~/.zshrc.local ]; then
    source ~/.zshrc.local
fi

# 3. OS判定を入れる
case "$OSTYPE" in
    darwin*)
        # macOS固有の設定
        alias ls='ls -G'
        ;;
    linux*)
        # Linux固有の設定
        alias ls='ls --color=auto'
        ;;
esac

# 4. コマンドの存在チェックを入れる
if command -v eza &>/dev/null; then
    alias ls='eza --icons'
fi

# 5. 設定変更前にバックアップを取る
cp ~/.zshrc ~/.zshrc.bak.$(date +%Y%m%d)
```

---

## 11. キーバインドの設定

### 11.1 zsh のキーバインド

```bash
# ============================================
# zsh キーバインド設定
# ============================================

# Emacs モード（デフォルト）
bindkey -e

# Vi モードを使いたい場合
# bindkey -v
# export KEYTIMEOUT=1

# 基本的なキーバインド
bindkey '^A' beginning-of-line       # Ctrl+A: 行頭
bindkey '^E' end-of-line             # Ctrl+E: 行末
bindkey '^K' kill-line               # Ctrl+K: カーソルから行末まで削除
bindkey '^U' backward-kill-line      # Ctrl+U: カーソルから行頭まで削除
bindkey '^W' backward-kill-word      # Ctrl+W: 単語を後方削除
bindkey '^Y' yank                    # Ctrl+Y: ペースト（killリングから）
bindkey '^L' clear-screen            # Ctrl+L: 画面クリア
bindkey '^R' history-incremental-search-backward  # Ctrl+R: 履歴逆検索

# 単語移動
bindkey '^[b' backward-word          # Alt+B: 前の単語へ
bindkey '^[f' forward-word           # Alt+F: 次の単語へ
bindkey '^[d' kill-word              # Alt+D: 単語を前方削除

# macOS の Option キー対応
bindkey '\e[1;3D' backward-word      # Option+Left
bindkey '\e[1;3C' forward-word       # Option+Right

# ホーム/エンドキー
bindkey '^[[H' beginning-of-line     # Home
bindkey '^[[F' end-of-line           # End

# Delete キー
bindkey '^[[3~' delete-char          # Delete

# 履歴検索の改善
bindkey '^P' up-line-or-search       # Ctrl+P: 上方向の履歴検索
bindkey '^N' down-line-or-search     # Ctrl+N: 下方向の履歴検索
bindkey '^[[A' up-line-or-search     # 上矢印
bindkey '^[[B' down-line-or-search   # 下矢印

# 部分一致履歴検索（入力中の文字列で検索）
autoload -Uz up-line-or-beginning-search down-line-or-beginning-search
zle -N up-line-or-beginning-search
zle -N down-line-or-beginning-search
bindkey '^[[A' up-line-or-beginning-search    # 上矢印
bindkey '^[[B' down-line-or-beginning-search  # 下矢印

# カスタムウィジェット: sudo を先頭に付ける
sudo-command-line() {
    [[ -z $BUFFER ]] && zle up-history
    if [[ $BUFFER == sudo\ * ]]; then
        LBUFFER="${LBUFFER#sudo }"
    else
        LBUFFER="sudo $LBUFFER"
    fi
}
zle -N sudo-command-line
bindkey '^[s' sudo-command-line      # Alt+S: sudo をトグル

# 現在のコマンドラインをエディタで編集
autoload -Uz edit-command-line
zle -N edit-command-line
bindkey '^X^E' edit-command-line     # Ctrl+X Ctrl+E: エディタで編集
```

---

## 12. 実践演習

### 演習1: [基礎] ── 最小限の .zshrc を作成する

```bash
# 要件:
# 1. 基本的な環境変数（EDITOR, LANG, PATH）を設定
# 2. 便利なエイリアスを5つ以上定義
# 3. 履歴設定を適切に設定
# 4. 補完を有効化

# 解答例:
cat > ~/.zshrc.exercise1 << 'EOF'
# ========== 環境変数 ==========
export EDITOR="vim"
export LANG="ja_JP.UTF-8"
export PATH="$HOME/.local/bin:$PATH"

# ========== エイリアス ==========
alias ll='ls -lah'
alias la='ls -A'
alias ..='cd ..'
alias ...='cd ../..'
alias gs='git status'
alias gc='git commit'
alias gp='git push'

# ========== 履歴設定 ==========
HISTSIZE=50000
SAVEHIST=50000
HISTFILE=~/.zsh_history
setopt SHARE_HISTORY
setopt HIST_IGNORE_DUPS
setopt HIST_IGNORE_SPACE

# ========== 補完 ==========
autoload -Uz compinit && compinit
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'
zstyle ':completion:*' menu select
EOF
```

### 演習2: [中級] ── fzf を活用した関数を作成する

```bash
# 要件:
# 1. fzf でファイルを検索してエディタで開く関数
# 2. fzf で Git ブランチを切り替える関数
# 3. fzf でプロセスを選択して kill する関数

# 解答例:
cat > ~/.zshrc.exercise2 << 'FUNC_EOF'
# fzf でファイルを検索してエディタで開く
fopen() {
    local file
    file=$(fd --type f --hidden --exclude .git | fzf \
        --preview 'bat --color=always --line-range :100 {}' \
        --preview-window=right:60%)
    [ -n "$file" ] && ${EDITOR:-vim} "$file"
}

# fzf で Git ブランチを切り替える
fbranch() {
    local branch
    branch=$(git branch -a | sed 's/^\* //' | sed 's/^ *//' | \
        fzf --preview 'git log --oneline --graph -20 {}')
    if [ -n "$branch" ]; then
        branch=$(echo "$branch" | sed 's|remotes/origin/||')
        git checkout "$branch"
    fi
}

# fzf でプロセスを選択して kill
fkill() {
    local pids
    pids=$(ps aux | sed 1d | fzf -m --header='Select process to kill' | awk '{print $2}')
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill -9
        echo "Killed: $pids"
    fi
}
FUNC_EOF
```

### 演習3: [上級] ── 完全なシェル環境セットアップスクリプト

```bash
# 要件:
# 1. OS判定（macOS/Linux）を行い適切なパッケージマネージャでツールをインストール
# 2. dotfiles をシンボリックリンクで配置
# 3. zsh プラグインをインストール
# 4. セットアップ完了後にテストを実行

# 解答例のフレームワーク:
cat > ~/setup.sh << 'SETUP_EOF'
#!/bin/bash
set -euo pipefail

# OS判定
detect_os() {
    case "$OSTYPE" in
        darwin*) echo "macos" ;;
        linux*)  echo "linux" ;;
        *)       echo "unknown"; exit 1 ;;
    esac
}

OS=$(detect_os)
echo "Detected OS: $OS"

# パッケージインストール
install_packages() {
    local packages=(fzf zoxide bat ripgrep fd jq starship direnv)

    if [ "$OS" = "macos" ]; then
        if ! command -v brew &>/dev/null; then
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        brew install "${packages[@]}" eza git-delta
    elif [ "$OS" = "linux" ]; then
        sudo apt update
        sudo apt install -y "${packages[@]}"
    fi
}

# zsh プラグイン
install_zsh_plugins() {
    local ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"

    local plugins=(
        "https://github.com/zsh-users/zsh-autosuggestions"
        "https://github.com/zsh-users/zsh-syntax-highlighting"
        "https://github.com/zsh-users/zsh-completions"
    )

    for plugin_url in "${plugins[@]}"; do
        local plugin_name=$(basename "$plugin_url")
        local target_dir="$ZSH_CUSTOM/plugins/$plugin_name"
        if [ ! -d "$target_dir" ]; then
            git clone "$plugin_url" "$target_dir"
            echo "Installed: $plugin_name"
        else
            echo "Already installed: $plugin_name"
        fi
    done
}

# テスト
run_tests() {
    echo "=== Running tests ==="
    local failed=0

    for cmd in fzf zoxide bat rg fd jq starship direnv; do
        if command -v "$cmd" &>/dev/null; then
            echo "  [OK] $cmd"
        else
            echo "  [NG] $cmd not found"
            ((failed++))
        fi
    done

    if [ "$failed" -eq 0 ]; then
        echo "All tests passed!"
    else
        echo "$failed test(s) failed"
    fi
}

install_packages
install_zsh_plugins
run_tests
SETUP_EOF

chmod +x ~/setup.sh
```

---

## まとめ

| 設定 | ファイル | 用途 |
|------|---------|------|
| 環境変数 | .zshenv / .bash_profile | PATH, EDITOR, LANG等 |
| エイリアス | .zshrc / .bashrc | コマンドの短縮 |
| 関数 | .zshrc / .bashrc | 複雑なコマンドの自動化 |
| プロンプト | .zshrc / .bashrc | 表示のカスタマイズ |
| 履歴 | .zshrc / .bashrc | 履歴の保存・共有設定 |
| 補完 | .zshrc / .bashrc | Tab補完の強化 |
| キーバインド | .zshrc / .bashrc | ショートカットキーの設定 |
| ツール設定 | 各ツールの設定ファイル | fzf, starship, delta等 |
| dotfiles管理 | ~/.dotfiles/ | バージョン管理と同期 |

### ベストプラクティスのまとめ

1. **設定をモジュール化する** -- 1つの巨大な .zshrc ではなく、機能ごとにファイルを分割する
2. **dotfiles をGit管理する** -- 設定の変更履歴を追跡し、複数マシンで同期する
3. **OS判定とコマンド存在チェックを入れる** -- 移植性の高い設定を書く
4. **危険なコマンドにはセーフガードを設ける** -- rm -i, cp -i 等のエイリアス
5. **起動時間を定期的に計測する** -- プラグインの追加で遅くなりすぎないよう注意
6. **機密情報は設定ファイルに直接書かない** -- direnv, キーチェーン, シークレットマネージャを活用
7. **モダンなツールを積極的に導入する** -- fzf, zoxide, bat, eza, ripgrep, fd で生産性向上

---

## 次に読むべきガイド
→ [[02-man-and-help.md]] — マニュアルとヘルプ

---

## 参考文献
1. Robbins, A. "bash Pocket Reference." 2nd Ed, O'Reilly, 2016.
2. Kiddle, O., Peek, J., Stephenson, P. "From Bash to Z Shell: Conquering the Command Line." Apress, 2004.
3. Janssens, J. "Data Science at the Command Line." 2nd Ed, O'Reilly, 2021.
4. Neil, D. "Practical Vim." 2nd Ed, Pragmatic Bookshelf, 2015.
5. Starship 公式ドキュメント: https://starship.rs/
6. Oh My Zsh 公式リポジトリ: https://github.com/ohmyzsh/ohmyzsh
7. fzf 公式リポジトリ: https://github.com/junegunn/fzf
8. zoxide 公式リポジトリ: https://github.com/ajeetdsouza/zoxide
