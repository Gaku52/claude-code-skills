# CLI 生産性向上

> ツールと設定を最適化し、CLI での作業速度を最大化する。

## この章で学ぶこと

- [ ] fzf, zoxide 等のモダンツールで操作を高速化できる
- [ ] エイリアス・関数でコマンドを短縮できる
- [ ] CLI ワークフローを最適化できる
- [ ] シェル補完を設定して入力を最小化できる
- [ ] CLI でのテキスト処理を高速に行える
- [ ] ターミナルエミュレータの選択と設定ができる

---

## 1. モダン CLI ツール

### 1.1 fzf（ファジーファインダー）

```bash
# ── インストール ──
# macOS
brew install fzf
$(brew --prefix)/opt/fzf/install      # キーバインドと補完を設定

# Ubuntu/Debian
sudo apt install fzf
# または最新版を Git から
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

# ── 基本的な使い方 ──
fzf                              # カレントディレクトリのファイルを検索
vim $(fzf)                       # 選択したファイルをvimで開く
cat $(fzf)                       # 選択したファイルの内容を表示

# ── キーバインド（シェル統合後） ──
# Ctrl+R    — コマンド履歴をファジー検索
# Ctrl+T    — ファイルパスをファジー検索して挿入
# Alt+C     — ディレクトリをファジー検索して cd

# ── パイプと組み合わせ ──
ps aux | fzf                     # プロセスを検索
git log --oneline | fzf          # コミットを検索
docker ps | fzf                  # コンテナを検索
kubectl get pods | fzf           # Pod を検索
env | fzf                        # 環境変数を検索
history | fzf                    # 履歴を検索

# ── プレビュー機能 ──
# ファイル内容をプレビュー（bat使用）
fzf --preview 'bat --color=always --line-range :100 {}'

# ファイル内容をプレビュー（head使用）
fzf --preview 'head -50 {}'

# ディレクトリの中身をプレビュー
fd -t d | fzf --preview 'eza -la --git {}'

# Git ログのプレビュー
git log --oneline | fzf --preview 'git show --color=always {1}'

# ── 複数選択 ──
# Tab で複数選択、Enter で確定
fzf --multi                      # 複数選択モード（-m でも可）
vim $(fzf -m)                    # 複数ファイルを選択して vim で開く
rm $(fzf -m)                     # 複数ファイルを選択して削除

# ── レイアウトとオプション ──
fzf --height 40%                 # 画面の40%で表示
fzf --layout=reverse             # 上から下へ表示
fzf --border                     # ボーダー表示
fzf --header "Select a file"     # ヘッダーテキスト
fzf --prompt ">> "               # プロンプトカスタマイズ

# ── fzf のデフォルト設定 ──
# ~/.zshrc に追加:
export FZF_DEFAULT_OPTS="
  --height 60%
  --layout=reverse
  --border rounded
  --preview-window right:50%
  --bind 'ctrl-/:toggle-preview'
  --bind 'ctrl-a:select-all'
  --bind 'ctrl-d:deselect-all'
  --color=bg+:#313244,bg:#1e1e2e,spinner:#f5e0dc,hl:#f38ba8
  --color=fg:#cdd6f4,header:#f38ba8,info:#cba6f7,pointer:#f5e0dc
  --color=marker:#f5e0dc,fg+:#cdd6f4,prompt:#cba6f7,hl+:#f38ba8
"

# Ctrl+T で fd を使用（高速）
export FZF_CTRL_T_COMMAND="fd --type f --hidden --follow --exclude .git"
export FZF_CTRL_T_OPTS="--preview 'bat --color=always --line-range :100 {}'"

# Alt+C で fd を使用
export FZF_ALT_C_COMMAND="fd --type d --hidden --follow --exclude .git"
export FZF_ALT_C_OPTS="--preview 'eza -la --git {}'"

# Ctrl+R のオプション
export FZF_CTRL_R_OPTS="
  --preview 'echo {}'
  --preview-window up:3:hidden:wrap
  --bind 'ctrl-/:toggle-preview'
"
```

### 1.2 fzf の実践的な活用パターン

```bash
# ── Git 連携関数 ──

# ブランチ選択して checkout
fco() {
    local branch
    branch=$(git branch -a | sed 's/^..//' | sed 's#remotes/origin/##' | sort -u |
        fzf --height 40% --preview 'git log --oneline -20 {}')
    [ -n "$branch" ] && git checkout "$branch"
}

# コミットハッシュを選択（git show / cherry-pick 等に）
fcommit() {
    local commit
    commit=$(git log --oneline --all --graph --decorate |
        fzf --ansi --no-sort --preview 'echo {} | grep -o "[a-f0-9]\{7,\}" | head -1 | xargs git show --color=always' |
        grep -o "[a-f0-9]\{7,\}" | head -1)
    [ -n "$commit" ] && echo "$commit"
}

# ステージング対象をインタラクティブに選択
fga() {
    local files
    files=$(git diff --name-only |
        fzf --multi --preview 'git diff --color=always {}')
    [ -n "$files" ] && echo "$files" | xargs git add
}

# stash をインタラクティブに選択して適用
fstash() {
    local stash
    stash=$(git stash list |
        fzf --preview 'echo {} | cut -d: -f1 | xargs git stash show -p --color=always' |
        cut -d: -f1)
    [ -n "$stash" ] && git stash apply "$stash"
}

# ── Docker 連携 ──

# コンテナを選択してシェルに入る
dexec() {
    local container
    container=$(docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' |
        fzf --height 40% | awk '{print $1}')
    [ -n "$container" ] && docker exec -it "$container" "${1:-bash}"
}

# コンテナを選択してログを表示
dlogs() {
    local container
    container=$(docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}' |
        fzf --height 40% | awk '{print $1}')
    [ -n "$container" ] && docker logs -f "$container"
}

# イメージを選択して削除
drmi() {
    local images
    images=$(docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}' |
        fzf --multi --height 40% | awk '{print $1}')
    [ -n "$images" ] && echo "$images" | xargs docker rmi
}

# ── プロセス管理 ──

# プロセスを選択して kill
fkill() {
    local pid
    pid=$(ps aux | fzf --header-lines=1 --height 40% | awk '{print $2}')
    [ -n "$pid" ] && echo "Killing PID $pid" && kill "${1:--9}" "$pid"
}

# ── SSH 接続 ──

# SSH先をインタラクティブに選択
fssh() {
    local host
    host=$(awk '/^Host / && !/\*/ {print $2}' ~/.ssh/config |
        fzf --height 30% --header "SSH to:")
    [ -n "$host" ] && ssh "$host"
}

# ── ファイルブラウザ ──

# インタラクティブなファイルブラウジング（ディレクトリ移動 + プレビュー）
fbrowse() {
    while true; do
        local selection
        selection=$(ls -1ap | fzf --header "$(pwd)" \
            --preview '[[ -d {} ]] && eza -la {} || bat --color=always {}' \
            --expect=ctrl-o,ctrl-h)
        local key=$(echo "$selection" | head -1)
        local file=$(echo "$selection" | tail -1)
        [ -z "$file" ] && break
        if [ "$key" = "ctrl-h" ]; then
            cd ..
        elif [ -d "$file" ]; then
            cd "$file"
        else
            ${EDITOR:-vim} "$file"
            break
        fi
    done
}
```

### 1.3 zoxide（スマートディレクトリ移動）

```bash
# ── インストール ──
brew install zoxide              # macOS
curl -sS https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | bash

# シェル初期化に追加
eval "$(zoxide init zsh)"        # .zshrc
eval "$(zoxide init bash)"       # .bashrc
eval "$(zoxide init fish)"       # config.fish

# ── 基本操作 ──
z project                        # "project" を含むよく行くディレクトリへ
z doc                            # "doc" を含むディレクトリへ
z foo bar                        # "foo" と "bar" 両方を含むパスへ
zi                               # fzf連携でインタラクティブ選択

# ── 仕組み ──
# zoxide は cd コマンドをフックして訪問したディレクトリを記録
# 訪問頻度と最後のアクセス時刻に基づいてスコアを計算
# z で移動する際にスコアが最も高いディレクトリを選択

# ── データ管理 ──
zoxide query                     # 記録されたディレクトリ一覧
zoxide query --list              # スコア付き一覧
zoxide query -s project          # "project" を含むエントリのスコア
zoxide add /path/to/dir          # 手動でパスを追加
zoxide remove /path/to/dir       # パスを削除

# ── cd の完全な置き換え ──
# .zshrc に追加して cd を zoxide に完全置換:
alias cd='z'

# ── __zoxide_zi のカスタマイズ ──
export _ZO_FZF_OPTS="
  --height 40%
  --layout=reverse
  --preview 'eza -la --git {2..}'
  --preview-window right:40%
"
```

### 1.4 bat（cat の代替）

```bash
# ── インストール ──
brew install bat                 # macOS
sudo apt install bat             # Ubuntu (batcat という名前になることがある)

# ── 基本操作 ──
bat file.py                      # シンタックスハイライト + 行番号
bat -l json data.txt             # 言語を明示的に指定
bat --diff file1 file2           # 差分表示
bat -A file.txt                  # 制御文字を表示
bat --line-range 10:20 file.py   # 10-20行目のみ表示
bat -p file.py                   # プレーンモード（行番号なし）

# ── テーマ ──
bat --list-themes                # 利用可能テーマ一覧
export BAT_THEME="Catppuccin Mocha"   # テーマ設定

# テーマのプレビュー
bat --list-themes | fzf --preview="bat --theme={} --color=always /path/to/sample.py"

# ── 設定ファイル ──
# ~/.config/bat/config
# --theme="Catppuccin Mocha"
# --style="numbers,changes,header,grid"
# --italic-text=always
# --map-syntax "*.conf:INI"
# --map-syntax ".ignore:Git Ignore"

# ── 他ツールとの連携 ──
# man ページのカラー表示
export MANPAGER="sh -c 'col -bx | bat -l man -p'"
export MANROFFOPT="-c"

# help の出力をカラー表示
alias bathelp='bat --plain --language=help'
help() {
    "$@" --help 2>&1 | bathelp
}

# git diff を bat で表示（delta 推奨だが bat でも可能）
git diff | bat -l diff
```

### 1.5 eza（ls の代替）

```bash
# ── インストール ──
brew install eza                 # macOS
cargo install eza                # Rust から

# ── 基本操作 ──
eza                              # カラー表示
eza -la                          # 詳細表示（隠しファイル含む）
eza -la --git                    # Git状態表示
eza --tree --level=2             # ツリー表示（2階層）
eza --tree --level=3 --git-ignore # ツリー（.gitignore尊重）
eza --icons                      # アイコン表示
eza -la --group                  # グループ表示
eza -la --header                 # ヘッダー行付き
eza -la --time-style=long-iso    # ISO形式の日時
eza -la --sort=modified          # 更新日時順
eza -la --sort=size              # サイズ順
eza -la --sort=extension         # 拡張子順
eza -la --reverse                # 逆順
eza --only-dirs                  # ディレクトリのみ
eza --only-files                 # ファイルのみ

# ── フィルタリング ──
eza -la --ignore-glob="*.pyc|__pycache__|node_modules"
eza -la --git-ignore             # .gitignore に基づくフィルタ

# ── エイリアス推奨設定 ──
alias ls='eza --icons'
alias ll='eza -la --icons --git --header'
alias lt='eza --tree --level=2 --icons'
alias lta='eza --tree --level=3 --icons --git-ignore'
alias lm='eza -la --sort=modified --icons'
alias lS='eza -la --sort=size --icons --reverse'
```

### 1.6 fd（find の代替）

```bash
# ── インストール ──
brew install fd                  # macOS
sudo apt install fd-find         # Ubuntu (fdfind という名前)

# ── 基本操作 ──
fd pattern                       # ファイル名でパターン検索
fd -e py                         # .py ファイルのみ
fd -e py -e js                   # .py と .js ファイル
fd -t d                          # ディレクトリのみ
fd -t f                          # ファイルのみ
fd -t l                          # シンボリックリンクのみ
fd -t x                          # 実行可能ファイルのみ
fd -H pattern                    # 隠しファイル含む
fd -I pattern                    # .gitignore 無視
fd -g '*.py'                     # glob パターン（正規表現ではなく）
fd -F 'exact_name'               # 完全一致
fd --max-depth 2 pattern         # 深さ制限

# ── コマンド実行 ──
fd pattern --exec wc -l          # 見つけたファイルの行数
fd -e py --exec python -c "import py_compile; py_compile.compile('{}')"
fd -e py --exec-batch wc -l      # まとめて実行（高速）
fd -e log --changed-within 1d    # 過去1日以内に変更されたログ
fd -e tmp --changed-before 7d --exec rm  # 7日以上前の tmp を削除

# ── 除外パターン ──
fd -E node_modules -E .git pattern
fd --ignore-file .fdignore pattern  # .fdignore ファイルを使用

# ── 実践例 ──
# 大きなファイルを探す
fd -t f --exec-batch ls -lhS | sort -rh -k5 | head -20

# TODO コメントがあるファイルを探す
fd -e py --exec grep -l "TODO" {}

# 空ディレクトリを探す
fd -t d --exec sh -c '[ -z "$(ls -A {})" ] && echo {}'
```

### 1.7 ripgrep（grep の代替）

```bash
# ── インストール ──
brew install ripgrep             # macOS
sudo apt install ripgrep         # Ubuntu

# ── 基本操作 ──
rg pattern                       # 再帰検索（.gitignore尊重）
rg -i pattern                    # 大文字小文字を無視
rg -w pattern                    # 単語境界マッチ
rg -F 'literal string'           # 正規表現ではなくリテラル検索
rg -v pattern                    # パターンに一致しない行
rg -c pattern                    # マッチ数のカウント
rg -l pattern                    # ファイル名のみ表示
rg -n pattern                    # 行番号表示（デフォルト）

# ── ファイルタイプ指定 ──
rg -t py pattern                 # Pythonファイルのみ
rg -t js -t ts pattern           # JavaScript + TypeScript
rg -T html pattern               # HTMLファイルを除外
rg --type-list                   # 利用可能なタイプ一覧

# ── コンテキスト表示 ──
rg -A 3 pattern                  # マッチ後3行
rg -B 3 pattern                  # マッチ前3行
rg -C 3 pattern                  # マッチ前後3行

# ── 高度な検索 ──
rg 'fn\s+\w+\(' -t rust         # Rust の関数定義
rg 'class\s+\w+' -t py          # Python のクラス定義
rg 'TODO|FIXME|HACK' -t py -t js # 複数パターン
rg -U 'def\s+\w+.*\n\s+"""'     # マルチラインマッチ
rg --json pattern | jq           # JSON出力

# ── 置換（プレビュー） ──
rg 'old_name' --replace 'new_name'  # 置換結果をプレビュー（ファイルは変更しない）
# 実際の置換は sed や sd と組み合わせる
rg -l 'old_name' | xargs sed -i 's/old_name/new_name/g'

# ── 設定ファイル ──
# ~/.config/ripgrep/config (RIPGREP_CONFIG_PATH で指定)
export RIPGREP_CONFIG_PATH="$HOME/.config/ripgrep/config"
# --smart-case
# --hidden
# --glob=!.git
# --glob=!node_modules
# --colors=line:fg:yellow
# --colors=match:fg:red
# --colors=match:style:bold
```

### 1.8 その他のモダンツール

```bash
# ── delta — git diff のシンタックスハイライト ──
brew install git-delta
# ~/.gitconfig に追加:
# [core]
#     pager = delta
# [interactive]
#     diffFilter = delta --color-only
# [delta]
#     navigate = true
#     side-by-side = true
#     line-numbers = true
#     syntax-theme = Catppuccin Mocha

# ── sd — sed の代替（より直感的な置換） ──
brew install sd
sd 'old_pattern' 'new_pattern' file.txt     # ファイル内置換
sd -F 'literal' 'replacement' file.txt      # リテラル置換
fd -e py | xargs sd 'old_func' 'new_func'   # 複数ファイルで置換

# ── dust — du の代替（ディスク使用量の可視化） ──
brew install dust
dust                             # カレントディレクトリのディスク使用量
dust -d 2                        # 深さ2まで
dust -r                          # 逆順（小さい順）

# ── procs — ps の代替 ──
brew install procs
procs                            # カラー表示のプロセス一覧
procs --tree                     # ツリー表示
procs --watch                    # リアルタイム更新
procs nginx                      # nginx 関連プロセスのみ

# ── bottom (btm) — top の代替 ──
brew install bottom
btm                              # インタラクティブなシステムモニタ

# ── hyperfine — ベンチマークツール ──
brew install hyperfine
hyperfine 'fd -e py'             # コマンドのベンチマーク
hyperfine 'fd -e py' 'find . -name "*.py"'  # 2つのコマンドを比較
hyperfine --warmup 3 'npm run build'  # ウォームアップ3回

# ── tokei — コード行数カウント ──
brew install tokei
tokei                            # リポジトリのコード統計
tokei -t Python,Rust             # 言語を指定

# ── jq — JSON プロセッサ ──
brew install jq
echo '{"name":"Alice","age":30}' | jq '.'        # 整形
echo '{"name":"Alice","age":30}' | jq '.name'     # フィールド抽出
curl -s api.example.com/data | jq '.items[] | {id, name}'  # 配列の各要素から抽出

# ── yq — YAML プロセッサ（jq のYAML版） ──
brew install yq
yq '.services' docker-compose.yml     # YAML からフィールド抽出
yq -i '.version = "3"' config.yml     # YAML をインプレース編集

# ── tldr — man の簡易版 ──
brew install tldr
tldr tar                         # tar の使用例
tldr curl                        # curl の使用例

# ── glow — ターミナルでMarkdownレンダリング ──
brew install glow
glow README.md                   # Markdownを整形表示
glow -p README.md                # ページャーで表示

# ── difftastic — 構造的な diff ──
brew install difftastic
difft file1.py file2.py          # AST ベースの差分
# git と統合:
# [diff]
#     external = difft
```

---

## 2. シェル設定の最適化

### 2.1 エイリアス

```bash
# ~/.zshrc（または ~/.bashrc）

# ── ナビゲーション ──
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias .....='cd ../../../..'
alias -- -='cd -'                # 前のディレクトリに戻る

# ── ls → eza ──
alias ls='eza --icons'
alias ll='eza -la --icons --git --header'
alias la='eza -la --icons'
alias lt='eza --tree --level=2 --icons'
alias lta='eza --tree --level=3 --icons --git-ignore'
alias lm='eza -la --sort=modified --icons'

# ── cat → bat ──
alias cat='bat --paging=never'
alias catp='bat --plain'         # プレーンモード

# ── grep → ripgrep ──
alias grep='rg'

# ── 安全な操作 ──
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'
alias mkdir='mkdir -p'

# ── Git ショートカット ──
alias g='git'
alias gs='git status -sb'
alias ga='git add'
alias gaa='git add -A'
alias gc='git commit'
alias gcm='git commit -m'
alias gca='git commit --amend'
alias gp='git push'
alias gpl='git pull --rebase'
alias gl='git log --oneline -20'
alias glg='git log --graph --oneline --decorate --all'
alias gd='git diff'
alias gds='git diff --staged'
alias gco='git checkout'
alias gcb='git checkout -b'
alias gb='git branch'
alias gba='git branch -a'
alias gst='git stash'
alias gstp='git stash pop'
alias gcp='git cherry-pick'
alias grb='git rebase'
alias grbi='git rebase -i'

# ── Docker ──
alias d='docker'
alias dc='docker compose'
alias dcu='docker compose up -d'
alias dcd='docker compose down'
alias dcl='docker compose logs -f'
alias dps='docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
alias dpsa='docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'
alias dimg='docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"'
alias dprune='docker system prune -af'

# ── Kubernetes ──
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get svc'
alias kgd='kubectl get deployments'
alias kga='kubectl get all'
alias kaf='kubectl apply -f'
alias kdf='kubectl delete -f'
alias klog='kubectl logs -f'
alias kexec='kubectl exec -it'
alias kctx='kubectl config use-context'
alias kns='kubectl config set-context --current --namespace'

# ── ネットワーク ──
alias myip='curl -s ifconfig.me'
alias localip="ipconfig getifaddr en0"
alias ports='netstat -tulanp'
alias listen='lsof -i -P | grep LISTEN'

# ── システム ──
alias df='df -h'
alias du='du -h'
alias free='free -h 2>/dev/null || vm_stat'
alias top='btm 2>/dev/null || htop 2>/dev/null || top'

# ── その他 ──
alias path='echo $PATH | tr ":" "\n" | nl'
alias now='date +"%Y-%m-%d %H:%M:%S"'
alias week='date +%V'
alias cls='clear'
alias h='history'
alias j='jobs -l'
```

### 2.2 便利な関数

```bash
# ── ディレクトリ作成 + 移動 ──
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# ── ファイル/ディレクトリのサイズ表示 ──
sizeof() {
    du -sh "$@" 2>/dev/null | sort -rh
}

# ── ポートを使っているプロセスを表示 ──
port() {
    lsof -i :"$1"
}

# ── 指定秒後にアラーム ──
timer() {
    local seconds="${1:-60}"
    echo "Timer: ${seconds}s"
    sleep "$seconds" && printf '\a' && echo "Time's up!"
}

# ── JSON整形 ──
json() {
    if [ -t 0 ]; then
        cat "$@" | jq '.'
    else
        jq '.'
    fi
}

# ── 天気 ──
weather() {
    curl -s "wttr.in/${1:-Tokyo}?format=3"
}

# ── extract: アーカイブを自動判別して展開 ──
extract() {
    if [ -f "$1" ]; then
        case "$1" in
            *.tar.bz2) tar xjf "$1"    ;;
            *.tar.gz)  tar xzf "$1"    ;;
            *.tar.xz)  tar xJf "$1"    ;;
            *.tar.zst) tar --zstd -xf "$1" ;;
            *.bz2)     bunzip2 "$1"    ;;
            *.gz)      gunzip "$1"     ;;
            *.tar)     tar xf "$1"     ;;
            *.tbz2)    tar xjf "$1"    ;;
            *.tgz)     tar xzf "$1"    ;;
            *.zip)     unzip "$1"      ;;
            *.Z)       uncompress "$1" ;;
            *.7z)      7z x "$1"       ;;
            *.rar)     unrar x "$1"    ;;
            *)         echo "Cannot extract '$1'" ;;
        esac
    else
        echo "'$1' is not a valid file"
    fi
}

# ── backup: ファイルのバックアップを作成 ──
backup() {
    cp -a "$1" "$1.bak.$(date +%Y%m%d_%H%M%S)"
}

# ── retry: コマンドを指定回数リトライ ──
retry() {
    local max_attempts="${1:-3}"
    local delay="${2:-5}"
    shift 2
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        echo "Attempt $attempt/$max_attempts: $*"
        if "$@"; then
            echo "Success on attempt $attempt"
            return 0
        fi
        echo "Failed. Waiting ${delay}s..."
        sleep "$delay"
        attempt=$((attempt + 1))
    done
    echo "All $max_attempts attempts failed"
    return 1
}

# ── note: クイックメモ ──
note() {
    local note_dir="$HOME/notes"
    mkdir -p "$note_dir"
    if [ $# -eq 0 ]; then
        ${EDITOR:-vim} "$note_dir/$(date +%Y-%m-%d).md"
    else
        echo "$(date +%H:%M) $*" >> "$note_dir/$(date +%Y-%m-%d).md"
        echo "Note added."
    fi
}

# ── serve: カレントディレクトリをHTTPサーバーで公開 ──
serve() {
    local port="${1:-8000}"
    echo "Serving on http://localhost:$port"
    python3 -m http.server "$port"
}

# ── cheat: コマンドのチートシート表示 ──
cheat() {
    curl -s "cheat.sh/$1"
}

# ── calc: コマンドラインの電卓 ──
calc() {
    python3 -c "from math import *; print($*)"
}

# ── colors: 256色の表示テスト ──
colors() {
    for i in {0..255}; do
        printf "\x1b[38;5;${i}m%3d " "$i"
        if (( (i + 1) % 16 == 0 )); then
            printf "\n"
        fi
    done
    printf "\x1b[0m\n"
}

# ── up: N階層上に移動 ──
up() {
    local count="${1:-1}"
    local path=""
    for i in $(seq 1 "$count"); do
        path="../$path"
    done
    cd "$path" || return
}

# ── tre: eza --tree の省略形（Git対応・深さ指定） ──
tre() {
    eza --tree --level="${1:-2}" --icons --git-ignore --git
}

# ── urlencode / urldecode ──
urlencode() {
    python3 -c "import urllib.parse; print(urllib.parse.quote('$*'))"
}
urldecode() {
    python3 -c "import urllib.parse; print(urllib.parse.unquote('$*'))"
}

# ── base64 エンコード/デコード ──
b64e() { echo -n "$*" | base64; }
b64d() { echo "$*" | base64 --decode; }

# ── whatismyip: 詳細なIP情報 ──
whatismyip() {
    curl -s "https://ipinfo.io" | jq '.'
}
```

### 2.3 Zsh 固有の設定

```bash
# ── Zsh のオプション設定 ──
setopt AUTO_CD               # ディレクトリ名だけで cd
setopt AUTO_PUSHD            # cd 時に自動で pushd
setopt PUSHD_IGNORE_DUPS     # pushd で重複を無視
setopt PUSHD_MINUS           # + と - の意味を入れ替え
setopt CORRECT               # コマンドのスペルチェック
setopt CORRECT_ALL           # 引数のスペルチェックも
setopt NO_BEEP               # ビープ音を無効化
setopt INTERACTIVE_COMMENTS  # コメントを許可
setopt EXTENDED_GLOB         # 拡張グロブ (#, ~, ^ 等)
setopt NULL_GLOB             # グロブがマッチしなくてもエラーにしない

# ── 履歴設定 ──
HISTFILE=~/.zsh_history
HISTSIZE=100000
SAVEHIST=100000
setopt HIST_IGNORE_ALL_DUPS  # 重複を無視
setopt HIST_IGNORE_SPACE     # スペースで始まるコマンドを記録しない
setopt HIST_REDUCE_BLANKS    # 余分な空白を削除
setopt SHARE_HISTORY         # 複数のセッション間で履歴を共有
setopt APPEND_HISTORY        # 履歴を追記
setopt INC_APPEND_HISTORY    # コマンド実行直後に追記
setopt HIST_VERIFY           # !! で直前のコマンドをすぐ実行せず展開

# ── 補完設定 ──
autoload -Uz compinit
compinit

# 補完の表示を改善
zstyle ':completion:*' menu select                   # メニュー選択
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'  # 小文字で大文字もマッチ
zstyle ':completion:*' list-colors "${(s.:.)LS_COLORS}" # カラー表示
zstyle ':completion:*:descriptions' format '%F{yellow}-- %d --%f'
zstyle ':completion:*:warnings' format '%F{red}-- no matches found --%f'
zstyle ':completion:*' group-name ''                 # グループ化
zstyle ':completion:*' squeeze-slashes true           # // を / に

# 補完のキャッシュ
zstyle ':completion:*' use-cache on
zstyle ':completion:*' cache-path "$HOME/.zcompcache"

# ── キーバインド（vi モード推奨） ──
bindkey -v                       # vi モードに設定
export KEYTIMEOUT=1              # モード切替を高速化

# vi モードでも便利なキーバインドを維持
bindkey '^R' history-incremental-search-backward
bindkey '^A' beginning-of-line
bindkey '^E' end-of-line
bindkey '^W' backward-kill-word
bindkey '^K' kill-line
bindkey '^U' kill-whole-line

# ── Zsh プラグインマネージャー ──
# zinit（推奨）
# bash -c "$(curl --fail --show-error --silent --location \
#   https://raw.githubusercontent.com/zdharma-continuum/zinit/HEAD/scripts/install.sh)"

# zinit でプラグインをロード
# zinit light zsh-users/zsh-autosuggestions
# zinit light zsh-users/zsh-syntax-highlighting
# zinit light zsh-users/zsh-completions
```

---

## 3. Starship プロンプト

### 3.1 基本設定

```bash
# ── インストール ──
brew install starship            # macOS
curl -sS https://starship.rs/install.sh | sh   # Linux

# .zshrc に追加
eval "$(starship init zsh)"

# .bashrc に追加
eval "$(starship init bash)"
```

### 3.2 設定ファイル（~/.config/starship.toml）

```toml
# ~/.config/starship.toml

# ── プロンプト全体 ──
# プロンプトの表示フォーマット
format = """
$username\
$hostname\
$directory\
$git_branch\
$git_status\
$nodejs\
$python\
$rust\
$golang\
$docker_context\
$kubernetes\
$aws\
$cmd_duration\
$line_break\
$character"""

# 右プロンプト
right_format = "$time"

# ── キャラクター（プロンプト記号） ──
[character]
success_symbol = "[>](bold green)"
error_symbol = "[>](bold red)"
vimcmd_symbol = "[<](bold green)"

# ── ディレクトリ ──
[directory]
truncation_length = 3
truncate_to_repo = true
style = "bold cyan"
format = "[$path]($style)[$read_only]($read_only_style) "
read_only = " (RO)"

# ── Git ブランチ ──
[git_branch]
format = "[$symbol$branch(:$remote_branch)]($style) "
symbol = " "
style = "bold purple"

# ── Git ステータス ──
[git_status]
format = '([\[$all_status$ahead_behind\]]($style) )'
conflicted = "="
ahead = "^${count}"
behind = "v${count}"
diverged = "^${ahead_count}v${behind_count}"
untracked = "?${count}"
stashed = "$${count}"
modified = "!${count}"
staged = "+${count}"
renamed = "~${count}"
deleted = "-${count}"
style = "bold red"

# ── 言語・ランタイム ──
[nodejs]
format = "[$symbol($version)]($style) "
symbol = " "
detect_files = ["package.json", ".node-version"]

[python]
format = "[$symbol$pyenv_prefix($version)( \\($virtualenv\\))]($style) "
symbol = " "

[rust]
format = "[$symbol($version)]($style) "
symbol = " "

[golang]
format = "[$symbol($version)]($style) "
symbol = " "

# ── Docker ──
[docker_context]
format = "[$symbol$context]($style) "
symbol = " "
only_with_files = true

# ── Kubernetes ──
[kubernetes]
disabled = false
format = "[$symbol$context(/$namespace)]($style) "
symbol = "K8s "
style = "bold blue"

# ── AWS ──
[aws]
format = "[$symbol($profile)(\\($region\\))]($style) "
symbol = " "
style = "bold yellow"

# ── コマンド実行時間 ──
[cmd_duration]
min_time = 3000
format = "took [$duration]($style) "
style = "bold yellow"

# ── 時刻（右プロンプト） ──
[time]
disabled = false
format = "[$time]($style)"
time_format = "%H:%M"
style = "dimmed white"

# ── ホスト名（SSH接続時のみ表示） ──
[hostname]
ssh_only = true
format = "[@$hostname]($style) "
style = "bold green"

# ── ユーザー名（root時のみ表示） ──
[username]
show_always = false
format = "[$user]($style) "
style_root = "bold red"
```

### 3.3 プリセットとカスタマイズ

```bash
# ── プリセットの適用 ──
# Nerd Font Symbols
starship preset nerd-font-symbols -o ~/.config/starship.toml

# Bracketed Segments（角括弧スタイル）
starship preset bracketed-segments -o ~/.config/starship.toml

# Plain Text Symbols（Nerd Font なしでも使える）
starship preset plain-text-symbols -o ~/.config/starship.toml

# Tokyo Night
starship preset tokyo-night -o ~/.config/starship.toml

# ── 環境ごとの設定切替 ──
# STARSHIP_CONFIG 環境変数で設定ファイルを切り替え
export STARSHIP_CONFIG=~/.config/starship/work.toml    # 仕事用
export STARSHIP_CONFIG=~/.config/starship/personal.toml # 個人用
```

---

## 4. キーボードショートカット

### 4.1 Readline / Zsh のキーバインド

```bash
# ── カーソル移動 ──
# Ctrl+A    → 行頭
# Ctrl+E    → 行末
# Ctrl+F    → 1文字前進（→と同じ）
# Ctrl+B    → 1文字後退（←と同じ）
# Alt+F     → 1単語前進
# Alt+B     → 1単語後退

# ── 編集 ──
# Ctrl+U    → カーソルから行頭まで削除
# Ctrl+K    → カーソルから行末まで削除
# Ctrl+W    → 直前の単語を削除
# Alt+D     → 次の単語を削除
# Ctrl+Y    → 削除した内容をペースト（yank）
# Ctrl+T    → カーソル前後の文字を入れ替え
# Alt+T     → カーソル前後の単語を入れ替え
# Alt+U     → 単語を大文字に変換
# Alt+L     → 単語を小文字に変換
# Alt+C     → 単語の先頭を大文字に
# Ctrl+_    → Undo（直前の編集を取り消し）

# ── 履歴 ──
# Ctrl+R    → 履歴の逆方向検索（fzf連携推奨）
# Ctrl+S    → 履歴の順方向検索
# Ctrl+P    → 前のコマンド（↑と同じ）
# Ctrl+N    → 次のコマンド（↓と同じ）
# !!        → 直前のコマンドを再実行
# !$        → 直前のコマンドの最後の引数
# !^        → 直前のコマンドの最初の引数
# !:n       → 直前のコマンドのn番目の引数
# !:n-m     → 直前のコマンドのn〜m番目の引数
# !*        → 直前のコマンドの全引数
# !cmd      → "cmd" で始まる直近のコマンドを実行
# !?str     → "str" を含む直近のコマンドを実行
# ^old^new  → 直前コマンドの "old" を "new" に置換して実行

# ── 制御 ──
# Ctrl+C    → 現在のコマンドを中断
# Ctrl+Z    → 現在のコマンドを一時停止（bg/fgで再開）
# Ctrl+D    → EOFを送信（シェル終了 / 入力終了）
# Ctrl+L    → 画面クリア
# Ctrl+S    → 画面出力の一時停止
# Ctrl+Q    → 画面出力の再開
# Ctrl+\\   → SIGQUIT送信（コアダンプ付き終了）

# ── Zsh 固有 ──
# Tab Tab   → 補完候補一覧
# Ctrl+X Ctrl+E → エディタでコマンド編集（$EDITOR）
# Alt+H     → man ページを表示
# Alt+?     → which コマンドを実行
# Esc .     → 直前のコマンドの最後の引数を挿入
```

### 4.2 カスタムキーバインド

```bash
# ~/.zshrc に追加

# ── fzf 連携のカスタムバインド ──

# Ctrl+G でファジー Git ブランチ切替
bindkey -s '^g' 'fco\n'

# Ctrl+O で fzf でファイルを開く
fzf-open-file() {
    local file
    file=$(fzf --preview 'bat --color=always {}')
    if [ -n "$file" ]; then
        BUFFER="${EDITOR:-vim} $file"
        zle accept-line
    fi
    zle reset-prompt
}
zle -N fzf-open-file
bindkey '^o' fzf-open-file

# Alt+C でディレクトリに移動（zoxide + fzf）
fzf-cd() {
    local dir
    dir=$(zoxide query -l | fzf --height 40% --preview 'eza -la {}')
    if [ -n "$dir" ]; then
        BUFFER="cd $dir"
        zle accept-line
    fi
    zle reset-prompt
}
zle -N fzf-cd
bindkey '\ec' fzf-cd

# ── vi モードのカスタマイズ ──
# vi モードの表示をカスタマイズ（カーソル形状を変更）
function zle-keymap-select {
    case $KEYMAP in
        vicmd)      echo -ne '\e[1 q' ;;  # ブロックカーソル（ノーマルモード）
        viins|main) echo -ne '\e[5 q' ;;  # ライン状カーソル（インサートモード）
    esac
}
zle -N zle-keymap-select

function zle-line-init {
    echo -ne '\e[5 q'  # 初期状態はインサートモード
}
zle -N zle-line-init
```

---

## 5. 効率的なワークフローパターン

### 5.1 プロジェクト作業環境の構築

```bash
# ── パターン1: tmux + fzf によるプロジェクト切替 ──
# ~/.local/bin/dev-start
#!/bin/bash
SESSION="dev"
PROJECT="${1:-$(pwd)}"

# 既存セッションがあればアタッチ
tmux has-session -t "$SESSION" 2>/dev/null && {
    tmux attach -t "$SESSION"
    exit 0
}

tmux new-session -d -s "$SESSION" -c "$PROJECT"
tmux send-keys "vim ." Enter
tmux split-window -v -p 30 -c "$PROJECT"
tmux send-keys "git status" Enter
tmux split-window -h -c "$PROJECT"
tmux select-pane -t 0
tmux attach -t "$SESSION"

# ── パターン2: 言語別の開発環境 ──

# Node.js プロジェクト
dev-node() {
    local project="${1:-.}"
    tmux new-session -d -s "node" -c "$project" -n "code"
    tmux send-keys "nvim ." Enter
    tmux new-window -t "node" -n "dev" -c "$project"
    tmux send-keys "npm run dev" Enter
    tmux new-window -t "node" -n "test" -c "$project"
    tmux send-keys "npm run test:watch" Enter
    tmux new-window -t "node" -n "shell" -c "$project"
    tmux select-window -t "node:code"
    tmux attach -t "node"
}

# Python プロジェクト
dev-python() {
    local project="${1:-.}"
    tmux new-session -d -s "python" -c "$project" -n "code"
    tmux send-keys "source .venv/bin/activate && nvim ." Enter
    tmux new-window -t "python" -n "repl" -c "$project"
    tmux send-keys "source .venv/bin/activate && ipython" Enter
    tmux new-window -t "python" -n "test" -c "$project"
    tmux send-keys "source .venv/bin/activate && pytest --watch" Enter
    tmux new-window -t "python" -n "shell" -c "$project"
    tmux send-keys "source .venv/bin/activate" Enter
    tmux select-window -t "python:code"
    tmux attach -t "python"
}
```

### 5.2 監視と自動化

```bash
# ── パターン3: 監視ダッシュボード ──
watch -n 5 'echo "=== Docker ===" && docker ps --format "table {{.Names}}\t{{.Status}}" && echo && echo "=== Disk ===" && df -h / && echo && echo "=== Memory ===" && free -h 2>/dev/null || vm_stat'

# ── パターン4: ファイル変更監視 + 自動実行 ──
# entr を使用（brew install entr）
# ファイル変更時にテストを自動実行
fd -e py | entr -c pytest

# ファイル変更時にビルドを自動実行
fd -e ts | entr -c npm run build

# 特定ファイル変更時にコマンド実行
ls *.go | entr -r go run main.go

# watchexec を使用（brew install watchexec）
watchexec -e py -- pytest
watchexec -e rs -- cargo test
watchexec -w src/ -- npm run build

# ── パターン5: 複数サーバーの一括操作 ──
servers=("web1" "web2" "web3")
for s in "${servers[@]}"; do
    echo "=== $s ==="
    ssh "$s" "systemctl status nginx --no-pager" &
done
wait

# ── パターン6: ログの統合監視 ──
# multitail — 複数ログを同時監視
multitail /var/log/nginx/access.log /var/log/nginx/error.log

# tail + awk でリアルタイムフィルタ
tail -f /var/log/app.log | awk '/ERROR/{print "\033[31m" $0 "\033[0m"} /WARN/{print "\033[33m" $0 "\033[0m"}'
```

### 5.3 作業記録と計測

```bash
# ── パターン7: 作業ログの自動記録 ──
# script コマンドで端末操作を全記録
script -q ~/logs/session_$(date +%Y%m%d_%H%M%S).log

# asciinema — より高機能な記録
brew install asciinema
asciinema rec ~/recordings/demo.cast
asciinema play ~/recordings/demo.cast
# asciinema.org にアップロード
asciinema upload ~/recordings/demo.cast

# ── パターン8: コマンド実行時間の計測 ──
time npm run build               # 組み込み time
/usr/bin/time -l npm run build   # macOS: 詳細（メモリ使用量等）
/usr/bin/time -v npm run build   # Linux: 詳細

# hyperfine — 統計的ベンチマーク
hyperfine 'npm run build'
hyperfine --warmup 3 --min-runs 10 'npm run build'
hyperfine 'fd -e py' 'find . -name "*.py"'  # 比較ベンチマーク
hyperfine --export-markdown bench.md 'cmd1' 'cmd2'  # Markdown で出力

# ── パターン9: コマンドの通知 ──
# 長時間コマンドの完了を通知

# macOS の通知
long_command; osascript -e 'display notification "Done!" with title "Terminal"'

# Linux の通知（notify-send）
long_command; notify-send "Terminal" "Command completed"

# 汎用関数
notify() {
    "$@"
    local status=$?
    if command -v osascript &>/dev/null; then
        osascript -e "display notification \"Exit: $status\" with title \"$1 finished\""
    elif command -v notify-send &>/dev/null; then
        notify-send "$1 finished" "Exit: $status"
    fi
    return $status
}
# 使用例: notify npm run build
```

---

## 6. dotfiles 管理

### 6.1 ベアリポジトリ方式

```bash
# Git ベアリポジトリで dotfiles を管理する方法
# シンボリックリンク不要で直接 $HOME の設定ファイルを管理

# ── セットアップ ──
git init --bare "$HOME/.dotfiles"
alias dot='git --git-dir=$HOME/.dotfiles --work-tree=$HOME'
dot config --local status.showUntrackedFiles no

# .zshrc にエイリアスを追加
echo "alias dot='git --git-dir=\$HOME/.dotfiles --work-tree=\$HOME'" >> ~/.zshrc

# ── ファイルの追加 ──
dot add ~/.zshrc
dot add ~/.tmux.conf
dot add ~/.config/starship.toml
dot add ~/.config/bat/config
dot add ~/.config/git/config
dot commit -m "Add dotfiles"

# ── リモートリポジトリに push ──
dot remote add origin git@github.com:username/dotfiles.git
dot push -u origin main

# ── 新しいマシンでの復元 ──
git clone --bare git@github.com:username/dotfiles.git "$HOME/.dotfiles"
alias dot='git --git-dir=$HOME/.dotfiles --work-tree=$HOME'
dot checkout
dot config --local status.showUntrackedFiles no

# checkout でコンフリクトが出る場合（既存ファイルがある場合）:
dot checkout 2>&1 | grep "already exists" | awk '{print $NF}' | xargs -I{} mv {} {}.bak
dot checkout
```

### 6.2 chezmoi（推奨）

```bash
# chezmoi は dotfiles 管理の専用ツール
# テンプレート、暗号化、マシン固有設定をサポート

# ── インストール ──
brew install chezmoi              # macOS
sh -c "$(curl -fsLS get.chezmoi.io)"  # Linux

# ── 初期化 ──
chezmoi init
chezmoi init --apply git@github.com:username/dotfiles.git  # 既存リポジトリから

# ── 基本操作 ──
chezmoi add ~/.zshrc             # ファイルを管理下に追加
chezmoi add ~/.tmux.conf
chezmoi add ~/.config/starship.toml
chezmoi add --encrypt ~/.ssh/config  # 暗号化して追加

chezmoi edit ~/.zshrc            # 管理下のファイルを編集
chezmoi diff                     # 差分を確認
chezmoi apply                    # 変更を $HOME に適用
chezmoi update                   # リモートから取得 + 適用

chezmoi cd                       # dotfiles リポジトリに移動
chezmoi data                     # テンプレートデータを表示
chezmoi doctor                   # 設定の問題をチェック

# ── テンプレート機能 ──
# マシンごとに異なる設定を生成
# ~/.local/share/chezmoi/dot_zshrc.tmpl
# {{ if eq .chezmoi.os "darwin" }}
# export HOMEBREW_PREFIX="/opt/homebrew"
# {{ else if eq .chezmoi.os "linux" }}
# export HOMEBREW_PREFIX="/home/linuxbrew/.linuxbrew"
# {{ end }}
#
# {{ if eq .chezmoi.hostname "work-laptop" }}
# export HTTP_PROXY="http://proxy.corp.example.com:8080"
# {{ end }}

# ── 暗号化 ──
# age で暗号化（GPGより簡単）
# ~/.config/chezmoi/chezmoi.toml
# [age]
#     identity = "~/.config/chezmoi/key.txt"
#     recipient = "age1..."

chezmoi add --encrypt ~/.ssh/config
chezmoi add --encrypt ~/.aws/credentials

# ── Git 操作 ──
chezmoi git add .
chezmoi git commit -- -m "Update dotfiles"
chezmoi git push
```

### 6.3 GNU Stow

```bash
# GNU Stow はシンボリックリンクファームマネージャー
# dotfiles ディレクトリ構造をそのまま $HOME にシンボリックリンク

# ── インストール ──
brew install stow                # macOS
sudo apt install stow            # Ubuntu

# ── ディレクトリ構成 ──
# ~/dotfiles/
# ├── zsh/
# │   └── .zshrc
# ├── tmux/
# │   └── .tmux.conf
# ├── git/
# │   └── .config/
# │       └── git/
# │           └── config
# ├── starship/
# │   └── .config/
# │       └── starship.toml
# └── nvim/
#     └── .config/
#         └── nvim/
#             └── init.lua

# ── 使用方法 ──
cd ~/dotfiles

# パッケージごとにシンボリックリンクを作成
stow zsh                         # ~/.zshrc → ~/dotfiles/zsh/.zshrc
stow tmux                        # ~/.tmux.conf → ~/dotfiles/tmux/.tmux.conf
stow git
stow starship
stow nvim

# 全パッケージを一括
stow */

# シンボリックリンクを削除
stow -D zsh

# 再ストウ（更新）
stow -R zsh

# ドライラン（実行せずに確認）
stow -n zsh
```

---

## 7. ターミナルエミュレータの選択と設定

### 7.1 モダンターミナルエミュレータ

```bash
# ── iTerm2（macOS） ──
# 最も人気のある macOS ターミナル
# https://iterm2.com/
# 特徴:
# - 分割ペイン
# - ホットキーウィンドウ（いつでも呼び出し）
# - シェル統合（コマンド状態の表示）
# - オートコンプリート
# - トリガー（パターンマッチでアクション実行）
# - プロファイル切替

# iTerm2 のおすすめ設定:
# Preferences > General > Closing
#   → "Confirm closing multiple sessions" ON
# Preferences > Profiles > Keys
#   → "Natural Text Editing" プリセット（Option+矢印で単語移動）
# Preferences > Profiles > Terminal
#   → Scrollback lines: 10000
# Preferences > Profiles > Session
#   → "Status bar enabled" ON（CPU、メモリ等を表示）

# ── WezTerm ──
# GPU アクセラレーション、Lua 設定、マルチプレクサ内蔵
# https://wezfurlong.org/wezterm/
# brew install --cask wezterm

# ~/.wezterm.lua
# local wezterm = require 'wezterm'
# return {
#   font = wezterm.font("JetBrains Mono"),
#   font_size = 14.0,
#   color_scheme = "Catppuccin Mocha",
#   enable_tab_bar = true,
#   window_background_opacity = 0.95,
#   keys = {
#     { key = "d", mods = "CMD", action = wezterm.action.SplitHorizontal },
#     { key = "d", mods = "CMD|SHIFT", action = wezterm.action.SplitVertical },
#   },
# }

# ── Alacritty ──
# GPU アクセラレーション、高速、最小限の機能
# https://alacritty.org/
# brew install --cask alacritty

# ~/.config/alacritty/alacritty.toml
# [font]
# size = 14.0
# [font.normal]
# family = "JetBrains Mono"
# [window]
# opacity = 0.95
# [colors]
# # Catppuccin Mocha テーマ

# ── kitty ──
# GPU アクセラレーション、画像表示対応、タイリング
# https://sw.kovidgoyal.net/kitty/
# brew install --cask kitty

# ~/.config/kitty/kitty.conf
# font_family JetBrains Mono
# font_size 14.0
# background_opacity 0.95
# enable_audio_bell no
# tab_bar_style powerline
# map cmd+d new_window_with_cwd
```

### 7.2 フォント

```bash
# ── Nerd Fonts（プログラミングフォント + アイコン） ──
# https://www.nerdfonts.com/

# Homebrew でインストール
brew install --cask font-jetbrains-mono-nerd-font
brew install --cask font-fira-code-nerd-font
brew install --cask font-hack-nerd-font
brew install --cask font-meslo-lg-nerd-font
brew install --cask font-cascadia-code-nerd-font

# 人気フォント:
# - JetBrains Mono Nerd Font — バランスの良いプログラミングフォント
# - Fira Code Nerd Font — リガチャ（合字）対応
# - Hack Nerd Font — 視認性重視
# - MesloLGS NF — Powerlevel10k 推奨
# - CaskaydiaCove Nerd Font — Windows Terminal で人気

# ターミナルのフォント設定でこれらを選択する
# アイコン表示に必要（eza --icons, Starship 等）
```

---

## 8. テキスト処理の高速化

### 8.1 パイプラインパターン

```bash
# ── 頻出パイプラインパターン ──

# ログからエラー行を抽出して件数カウント
rg "ERROR" /var/log/app.log | awk '{print $4}' | sort | uniq -c | sort -rn

# CSVの特定カラムを集計
awk -F',' '{sum += $3} END {print sum}' data.csv

# JSON配列から特定フィールドを抽出
jq '.[] | .name' data.json

# 重複行の削除（順序維持）
awk '!seen[$0]++' file.txt

# 特定パターンの前後N行を表示
rg -C 3 "pattern" file.txt

# ファイルの差分を見やすく表示
diff <(sort file1.txt) <(sort file2.txt) | bat -l diff

# ── xargs の活用 ──
# ファイルを並列処理
fd -e py | xargs -P 4 -I{} python -m py_compile {}

# NULL区切り（ファイル名にスペースがある場合に安全）
fd -0 -e py | xargs -0 wc -l

# 確認付き実行
fd -e tmp | xargs -p rm

# ── プロセス置換 ──
# 2つのコマンドの出力を比較
diff <(curl -s url1) <(curl -s url2)

# 複数のログをマージしてソート
sort -m <(sort log1.txt) <(sort log2.txt) <(sort log3.txt)

# ── tee の活用 ──
# 出力をファイルに保存しつつ画面にも表示
npm run build 2>&1 | tee build.log

# 複数ファイルに同時出力
echo "test" | tee file1.txt file2.txt file3.txt
```

### 8.2 ワンライナー集

```bash
# ── ファイル操作 ──
# 空ファイルを見つける
fd -t f --exec sh -c '[ ! -s {} ] && echo {}'

# 最近変更されたファイル（過去1時間）
fd --changed-within 1h

# ファイル名の一括リネーム
# file_001.txt → file-001.txt（rename コマンド）
rename 's/_/-/g' *.txt

# 拡張子の一括変更
fd -e txt --exec mv {} {.}.md

# ── テキスト処理 ──
# 行番号を追加
nl -ba file.txt

# 特定行を抽出（10〜20行目）
sed -n '10,20p' file.txt

# 行を逆順に
tac file.txt

# ランダムに1行表示
shuf -n 1 file.txt

# カラム入れ替え
awk '{print $2, $1}' file.txt

# タブ区切りをカンマ区切りに
tr '\t' ',' < input.tsv > output.csv

# ── ネットワーク ──
# 特定ポートのプロセスを kill
lsof -ti :3000 | xargs kill -9

# 全てのリスニングポートを表示
lsof -iTCP -sTCP:LISTEN -n -P

# DNS のルックアップ
dig +short example.com

# HTTP レスポンスヘッダーのみ表示
curl -sI https://example.com

# ── Git ワンライナー ──
# 各著者のコミット数
git shortlog -sn --all

# 変更が多いファイルランキング
git log --pretty=format: --name-only | sort | uniq -c | sort -rn | head -20

# 今日のコミット
git log --since="midnight" --oneline

# ブランチ一覧（最終コミット日時付き）
git branch -a --sort=-committerdate --format='%(committerdate:short) %(refname:short)'
```

---

## 9. シェルスクリプトのスニペット集

### 9.1 日常タスクの自動化

```bash
# ── プロジェクトの初期設定 ──
init-project() {
    local name="$1"
    local type="${2:-node}"

    mkdir -p "$name" && cd "$name" || return

    git init
    echo "node_modules/" > .gitignore
    echo ".env" >> .gitignore
    echo "*.log" >> .gitignore

    case "$type" in
        node)
            npm init -y
            echo "# $name" > README.md
            mkdir -p src tests
            ;;
        python)
            python3 -m venv .venv
            echo ".venv/" >> .gitignore
            echo "__pycache__/" >> .gitignore
            echo "# $name" > README.md
            mkdir -p src tests
            touch src/__init__.py tests/__init__.py
            cat > requirements.txt << 'REQS'
pytest>=7.0
black>=23.0
ruff>=0.1.0
REQS
            ;;
        rust)
            cargo init
            ;;
    esac

    git add -A
    git commit -m "Initial commit"
    echo "Project '$name' ($type) initialized!"
}

# ── ディスク使用量レポート ──
disk-report() {
    echo "=== Disk Usage Report ==="
    echo "Date: $(date)"
    echo ""
    echo "--- Top 10 directories ---"
    du -h --max-depth=1 "${1:-.}" 2>/dev/null | sort -rh | head -10
    echo ""
    echo "--- Large files (>100MB) ---"
    fd -t f --size +100m "${1:-.}" 2>/dev/null
    echo ""
    echo "--- Disk space ---"
    df -h "${1:-.}"
}

# ── 定期バックアップ ──
backup-dir() {
    local src="${1:?Source directory required}"
    local dst="${2:-$HOME/backups}"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local name=$(basename "$src")
    local archive="$dst/${name}_${timestamp}.tar.gz"

    mkdir -p "$dst"
    tar czf "$archive" -C "$(dirname "$src")" "$name"
    echo "Backup created: $archive ($(du -h "$archive" | cut -f1))"

    # 30日以上前のバックアップを削除
    find "$dst" -name "${name}_*.tar.gz" -mtime +30 -delete
    echo "Old backups cleaned up."
}

# ── 開発環境のヘルスチェック ──
dev-check() {
    echo "=== Development Environment Check ==="
    echo ""

    local tools=(
        "git:git --version"
        "node:node --version"
        "npm:npm --version"
        "python3:python3 --version"
        "docker:docker --version"
        "kubectl:kubectl version --client --short 2>/dev/null"
        "fzf:fzf --version"
        "rg:rg --version | head -1"
        "fd:fd --version"
        "bat:bat --version | head -1"
        "eza:eza --version | head -1"
    )

    for item in "${tools[@]}"; do
        local name="${item%%:*}"
        local cmd="${item##*:}"
        if command -v "$name" &>/dev/null; then
            local version=$(eval "$cmd" 2>/dev/null)
            printf "  %-12s %s\n" "$name" "$version"
        else
            printf "  %-12s %s\n" "$name" "(not installed)"
        fi
    done
}
```

### 9.2 データ処理ユーティリティ

```bash
# ── CSV 処理 ──

# CSV のカラム名を表示（ヘッダー行）
csv-header() {
    head -1 "$1" | tr ',' '\n' | nl
}

# CSV の特定カラムを抽出
csv-col() {
    local file="$1"
    local col="$2"
    awk -F',' -v c="$col" '{print $c}' "$file"
}

# CSV を整形表示
csv-view() {
    column -s',' -t < "$1" | less -S
}

# ── JSON 処理 ──

# JSON を整形して bat で表示
json-view() {
    if [ -f "$1" ]; then
        jq '.' "$1" | bat -l json
    else
        curl -s "$1" | jq '.' | bat -l json
    fi
}

# JSON のキーパスを一覧表示
json-paths() {
    jq -r 'paths(scalars) | map(tostring) | join(".")' "$1"
}

# ── ログ分析 ──

# アクセスログのステータスコード集計
log-status() {
    awk '{print $9}' "$1" | sort | uniq -c | sort -rn
}

# アクセスログのトップIP
log-top-ip() {
    awk '{print $1}' "$1" | sort | uniq -c | sort -rn | head -${2:-10}
}

# エラーログのパターン分析
log-errors() {
    rg -c "ERROR|FATAL|CRITICAL" "$1"
    echo "---"
    rg "ERROR|FATAL|CRITICAL" "$1" | awk '{$1=$2=$3=""; print}' | sort | uniq -c | sort -rn | head -20
}
```

---

## 10. 環境ごとの生産性設定

### 10.1 macOS 固有の設定

```bash
# ── macOS デフォルト設定（CLI から変更） ──

# Finder で隠しファイルを表示
defaults write com.apple.finder AppleShowAllFiles -bool true

# Dock の自動非表示
defaults write com.apple.dock autohide -bool true

# キーリピートの高速化
defaults write NSGlobalDomain KeyRepeat -int 1
defaults write NSGlobalDomain InitialKeyRepeat -int 10

# スクリーンショットの保存先
defaults write com.apple.screencapture location "$HOME/Screenshots"

# .DS_Store をネットワークドライブに作成しない
defaults write com.apple.desktopservices DSDontWriteNetworkStores true

# ── macOS 固有のコマンド ──
# クリップボード
echo "text" | pbcopy              # クリップボードにコピー
pbpaste                           # クリップボードからペースト
pbpaste | wc -l                   # クリップボードの行数

# 通知
osascript -e 'display notification "Hello" with title "Terminal"'

# ファイルを開く
open .                            # Finder で現在ディレクトリを開く
open -a "Visual Studio Code" .    # VSCode で開く
open https://example.com          # ブラウザで URL を開く

# Spotlight の検索
mdfind "query"                    # Spotlight 検索
mdfind -name "filename"           # ファイル名で検索
mdfind -onlyin ~/projects "TODO"  # 特定ディレクトリで検索

# ディスクの取り出し
diskutil eject /dev/disk2

# Wi-Fi
networksetup -getairportnetwork en0        # 接続中のWi-Fi
networksetup -setairportpower en0 off      # Wi-Fi OFF
networksetup -setairportpower en0 on       # Wi-Fi ON
```

### 10.2 リモートサーバーでの作業効率化

```bash
# ── SSH 設定の最適化 ──
# ~/.ssh/config

# 全ホスト共通設定
# Host *
#     ServerAliveInterval 60
#     ServerAliveCountMax 3
#     AddKeysToAgent yes
#     IdentityFile ~/.ssh/id_ed25519
#     Compression yes

# ── よく使うサーバーのエイリアス ──
# Host web
#     HostName web.example.com
#     User deploy
#     Port 22
#     ForwardAgent yes

# Host db
#     HostName db.example.com
#     User admin
#     LocalForward 5432 localhost:5432

# ── SSH 接続の高速化 ──
# Host *
#     ControlMaster auto
#     ControlPath ~/.ssh/sockets/%r@%h-%p
#     ControlPersist 600

mkdir -p ~/.ssh/sockets

# ── リモート作業の便利スクリプト ──

# リモートサーバーの状態チェック
server-check() {
    local host="$1"
    echo "=== $host ==="
    ssh "$host" '
        echo "Hostname: $(hostname)"
        echo "Uptime: $(uptime)"
        echo "Disk: $(df -h / | tail -1)"
        echo "Memory: $(free -h | grep Mem | awk "{print \$3\"/\"\$2}")"
        echo "Load: $(cat /proc/loadavg)"
        echo "Docker: $(docker ps -q 2>/dev/null | wc -l) containers"
    '
}

# ファイルのリモート編集（ローカルエディタで）
remote-edit() {
    local host="$1"
    local file="$2"
    local tmp="/tmp/remote-edit-$(basename "$file")"
    scp "$host:$file" "$tmp"
    ${EDITOR:-vim} "$tmp"
    scp "$tmp" "$host:$file"
    rm -f "$tmp"
}

# リモートコマンドを全サーバーで並列実行
parallel-ssh() {
    local cmd="$1"
    shift
    for host in "$@"; do
        echo "--- $host ---"
        ssh "$host" "$cmd" &
    done
    wait
}
```

---

## まとめ

| ツール | 代替対象 | 改善点 |
|-------|---------|--------|
| fzf | 手動検索 | ファジー検索でインタラクティブ |
| zoxide | cd | 訪問履歴でスマート移動 |
| bat | cat | シンタックスハイライト |
| eza | ls | カラー・Git・アイコン |
| fd | find | 高速・直感的 |
| ripgrep | grep | 高速・.gitignore尊重 |
| starship | PS1 | 情報豊富なプロンプト |
| delta | diff | シンタックスハイライト付き差分 |
| sd | sed | 直感的な文字列置換 |
| dust | du | ディスク使用量の可視化 |
| procs | ps | カラー表示・ツリー表示 |
| bottom | top | インタラクティブモニタ |
| hyperfine | time | 統計的ベンチマーク |
| tokei | cloc | 高速なコード行数カウント |
| glow | - | ターミナルMarkdownレンダリング |
| difftastic | diff | AST ベースの構造的差分 |

---

## 次に読むべきガイド
→ [[../05-shell-scripting/01-advanced-scripting.md]] -- 高度なシェルスクリプティング
→ [[00-tmux-screen.md]] -- ターミナルマルチプレクサ

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." O'Reilly, 2022.
2. "Modern Unix." github.com/ibraheemdev/modern-unix.
3. "The Art of Command Line." github.com/jlevy/the-art-of-command-line.
4. "Awesome Shell." github.com/alebcay/awesome-shell.
5. "fzf examples." github.com/junegunn/fzf/wiki/examples.
6. "Starship documentation." starship.rs/config/.
