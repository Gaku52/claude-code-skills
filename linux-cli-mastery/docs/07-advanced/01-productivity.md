# CLI 生産性向上

> ツールと設定を最適化し、CLI での作業速度を最大化する。

## この章で学ぶこと

- [ ] fzf, zoxide 等のモダンツールで操作を高速化できる
- [ ] エイリアス・関数でコマンドを短縮できる
- [ ] CLI ワークフローを最適化できる

---

## 1. モダン CLI ツール

```bash
# ── インストール（macOS: Homebrew） ──
brew install fzf zoxide bat eza fd ripgrep starship jq yq tldr

# fzf — ファジーファインダー
# あらゆるリストをインタラクティブに絞り込む

# ファイル検索
fzf                              # カレントディレクトリのファイルを検索
vim $(fzf)                       # 選択したファイルをvimで開く

# コマンド履歴検索
# Ctrl+R                        # fzf で履歴をインタラクティブ検索

# ディレクトリ移動
# Alt+C（またはCtrl+T）         # fzf でディレクトリ選択

# パイプと組み合わせ
ps aux | fzf                     # プロセスを検索
git log --oneline | fzf          # コミットを検索
docker ps | fzf                  # コンテナを検索

# プレビュー付き
fzf --preview 'bat --color=always {}'   # ファイル内容プレビュー
fzf --preview 'head -50 {}'             # 先頭50行プレビュー

# git との連携
alias gb='git branch | fzf | xargs git checkout'
alias gl='git log --oneline | fzf | cut -d" " -f1 | xargs git show'

# ── zoxide — スマートディレクトリ移動 ──
# cd の代替。訪問頻度と最近のアクセスでランキング
eval "$(zoxide init zsh)"        # .zshrc に追加

z project                        # "project" を含むよく行くディレクトリへ
z doc                            # "doc" を含むディレクトリへ
zi                               # fzf連携でインタラクティブ選択

# ── bat — cat の代替 ──
bat file.py                      # シンタックスハイライト + 行番号
bat -l json data.txt             # 言語指定
bat --diff file1 file2           # 差分表示
export BAT_THEME="Dracula"       # テーマ設定

# ── eza — ls の代替 ──
eza                              # カラー表示
eza -la                          # 詳細表示
eza -la --git                    # Git状態表示
eza --tree --level=2             # ツリー表示
eza --icons                      # アイコン表示

# ── fd — find の代替 ──
fd pattern                       # ファイル名でパターン検索
fd -e py                         # .py ファイルのみ
fd -t d                          # ディレクトリのみ
fd -H pattern                    # 隠しファイル含む
fd pattern --exec wc -l          # 見つけたファイルで実行
```

---

## 2. シェル設定の最適化

```bash
# ~/.zshrc（または ~/.bashrc）

# ── エイリアス ──
# ナビゲーション
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'

# ls → eza
alias ls='eza'
alias ll='eza -la --git'
alias lt='eza --tree --level=2'

# cat → bat
alias cat='bat --paging=never'

# grep → ripgrep
alias grep='rg'

# 安全な操作
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Git ショートカット
alias g='git'
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -20'
alias gd='git diff'
alias gco='git checkout'
alias gb='git branch'

# Docker
alias d='docker'
alias dc='docker compose'
alias dps='docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"'

# ── 便利な関数 ──

# ディレクトリ作成 + 移動
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# ファイル/ディレクトリのサイズ表示
sizeof() {
    du -sh "$@" 2>/dev/null | sort -rh
}

# ポートを使っているプロセスを表示
port() {
    lsof -i :"$1"
}

# 指定秒後にアラーム
timer() {
    local seconds="${1:-60}"
    echo "Timer: ${seconds}s"
    sleep "$seconds" && printf '\a' && echo "Time's up!"
}

# JSON整形
json() {
    if [ -t 0 ]; then
        cat "$@" | jq '.'
    else
        jq '.'
    fi
}

# 天気
weather() {
    curl -s "wttr.in/${1:-Tokyo}?format=3"
}
```

---

## 3. Starship プロンプト

```bash
# インストール
brew install starship

# .zshrc に追加
eval "$(starship init zsh)"

# ~/.config/starship.toml で設定
# 最小構成の例:
[character]
success_symbol = "[❯](green)"
error_symbol = "[❯](red)"

[directory]
truncation_length = 3
truncate_to_repo = true

[git_branch]
format = "[$symbol$branch]($style) "

[git_status]
format = '([\[$all_status$ahead_behind\]]($style) )'

[nodejs]
format = "[$symbol($version)]($style) "
detect_files = ["package.json"]

[python]
format = "[$symbol$pyenv_prefix($version)]($style) "

[cmd_duration]
min_time = 5000
format = "took [$duration]($style) "
```

---

## 4. キーボードショートカット

```bash
# Readline / Zsh のキーバインド

# カーソル移動
# Ctrl+A    → 行頭
# Ctrl+E    → 行末
# Ctrl+F    → 1文字前進（→と同じ）
# Ctrl+B    → 1文字後退（←と同じ）
# Alt+F     → 1単語前進
# Alt+B     → 1単語後退

# 編集
# Ctrl+U    → カーソルから行頭まで削除
# Ctrl+K    → カーソルから行末まで削除
# Ctrl+W    → 直前の単語を削除
# Alt+D     → 次の単語を削除
# Ctrl+Y    → 削除した内容をペースト
# Ctrl+T    → カーソル前後の文字を入れ替え

# 履歴
# Ctrl+R    → 履歴の逆方向検索（fzf連携推奨）
# Ctrl+P    → 前のコマンド（↑と同じ）
# Ctrl+N    → 次のコマンド（↓と同じ）
# !!        → 直前のコマンドを再実行
# !$        → 直前のコマンドの最後の引数
# !^        → 直前のコマンドの最初の引数
# !:n       → 直前のコマンドのn番目の引数

# 制御
# Ctrl+C    → 現在のコマンドを中断
# Ctrl+Z    → 現在のコマンドを一時停止
# Ctrl+D    → EOFを送信（シェル終了）
# Ctrl+L    → 画面クリア

# Zsh 固有
# Tab Tab   → 補完候補一覧
# Ctrl+X Ctrl+E → エディタでコマンド編集（$EDITOR）
```

---

## 5. 効率的なワークフローパターン

```bash
# パターン1: プロジェクト作業環境の一発構築
# ~/.local/bin/dev-start
#!/bin/bash
SESSION="dev"
PROJECT="${1:-$(pwd)}"

tmux new-session -d -s "$SESSION" -c "$PROJECT"
tmux send-keys "vim ." Enter
tmux split-window -v -p 30 -c "$PROJECT"
tmux send-keys "git status" Enter
tmux split-window -h -c "$PROJECT"
tmux select-pane -t 0
tmux attach -t "$SESSION"

# パターン2: 監視ダッシュボード
watch -n 5 'echo "=== Docker ===" && docker ps --format "table {{.Names}}\t{{.Status}}" && echo && echo "=== Disk ===" && df -h / && echo && echo "=== Memory ===" && free -h'

# パターン3: 複数サーバーの一括操作
servers=("web1" "web2" "web3")
for s in "${servers[@]}"; do
    echo "=== $s ==="
    ssh "$s" "systemctl status nginx --no-pager" &
done
wait

# パターン4: 作業ログの自動記録
# script コマンドで端末操作を全記録
script -q ~/logs/session_$(date +%Y%m%d_%H%M%S).log

# パターン5: コマンド実行時間の計測
time npm run build               # 組み込み time
/usr/bin/time -v npm run build   # 詳細（メモリ使用量等）
hyperfine 'npm run build'        # ベンチマーク（複数回計測）
```

---

## 6. dotfiles 管理

```bash
# Git で dotfiles を管理する方法

# 方法1: ベアリポジトリ
git init --bare "$HOME/.dotfiles"
alias dot='git --git-dir=$HOME/.dotfiles --work-tree=$HOME'
dot config --local status.showUntrackedFiles no

dot add ~/.zshrc
dot add ~/.tmux.conf
dot add ~/.config/starship.toml
dot commit -m "Add dotfiles"
dot push

# 方法2: chezmoi（推奨）
# brew install chezmoi
chezmoi init
chezmoi add ~/.zshrc
chezmoi add ~/.tmux.conf
chezmoi cd                       # dotfiles リポジトリに移動
chezmoi apply                    # 変更を適用
chezmoi update                   # リモートから取得 + 適用

# 方法3: GNU Stow
# ~/dotfiles/zsh/.zshrc → ~/.zshrc にシンボリックリンク
cd ~/dotfiles
stow zsh
stow tmux
stow starship
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

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." O'Reilly, 2022.
2. "Modern Unix." github.com/ibraheemdev/modern-unix.
