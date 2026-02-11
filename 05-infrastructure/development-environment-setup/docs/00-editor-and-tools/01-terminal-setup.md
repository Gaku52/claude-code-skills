# ターミナル設定

> モダンなターミナルエミュレータとシェル環境を構築し、コマンドライン作業の効率を最大化するための実践ガイド。

## この章で学ぶこと

1. iTerm2 / Windows Terminal のインストールと高度なカスタマイズ方法
2. zsh / fish / PowerShell の設定とプロンプトカスタマイズ（Starship）
3. tmux によるセッション管理とペイン分割の実践テクニック

---

## 1. ターミナルエミュレータの選定

### 1.1 主要ターミナル比較

| 特徴 | iTerm2 (macOS) | Windows Terminal | Alacritty | Warp |
|------|----------------|------------------|-----------|------|
| OS | macOS | Windows | クロスプラットフォーム | macOS / Linux |
| GPU アクセラレーション | 部分的 | あり | あり | あり |
| タブ/ペイン | あり | あり | なし (tmux併用) | あり |
| 設定形式 | GUI + Plist | JSON | TOML | GUI |
| 検索機能 | 高機能 | あり | 基本的 | AI搭載 |
| 価格 | 無料 | 無料 | 無料 | Freemium |

### 1.2 iTerm2 セットアップ (macOS)

```bash
# インストール
brew install --cask iterm2

# カラースキームのインポート
# https://iterm2colorschemes.com/ から .itermcolors をダウンロード
# Preferences → Profiles → Colors → Color Presets → Import
```

推奨設定:

```
iTerm2 推奨設定:
┌─────────────────────────────────────────┐
│ Preferences → General                    │
│   ✅ Closing → Confirm "Quit iTerm2"    │
│                                           │
│ Preferences → Profiles → General        │
│   Working Directory: "Reuse previous"    │
│                                           │
│ Preferences → Profiles → Text           │
│   Font: JetBrains Mono Nerd Font 14pt   │
│   ✅ Use ligatures                       │
│                                           │
│ Preferences → Profiles → Window         │
│   Transparency: 5-10%                    │
│   Columns: 120, Rows: 35                │
│                                           │
│ Preferences → Profiles → Keys           │
│   Left Option Key: Esc+                  │
│   (単語単位の移動に必要)                   │
└─────────────────────────────────────────┘
```

### 1.3 Windows Terminal セットアップ

```jsonc
// settings.json (Windows Terminal)
{
  "defaultProfile": "{your-powershell-guid}",
  "profiles": {
    "defaults": {
      "font": {
        "face": "JetBrains Mono Nerd Font",
        "size": 12
      },
      "colorScheme": "One Half Dark",
      "opacity": 95,
      "useAcrylic": true,
      "padding": "8"
    },
    "list": [
      {
        "name": "PowerShell 7",
        "source": "Windows.Terminal.PowershellCore",
        "startingDirectory": "%USERPROFILE%"
      },
      {
        "name": "Ubuntu (WSL)",
        "source": "Windows.Terminal.Wsl"
      }
    ]
  },
  "actions": [
    { "command": "toggleFocusMode", "keys": "f11" },
    { "command": { "action": "splitPane", "split": "horizontal" }, "keys": "alt+shift+-" },
    { "command": { "action": "splitPane", "split": "vertical" }, "keys": "alt+shift+=" }
  ]
}
```

---

## 2. シェル設定

### 2.1 シェル比較

| 特徴 | zsh | fish | PowerShell 7 |
|------|-----|------|-------------|
| POSIX 互換 | はい | いいえ | いいえ |
| デフォルトOS | macOS | なし | Windows |
| 補完機能 | プラグイン必要 | 組み込み | 組み込み |
| スクリプト互換性 | bash とほぼ同じ | 独自構文 | .NET ベース |
| 学習コスト | 低 | 低 | 中 |
| プラグインエコシステム | 非常に豊富 | 豊富 | 成長中 |

### 2.2 zsh 設定

```bash
# zsh がデフォルトでない場合
chsh -s $(which zsh)

# .zshrc の基本設定
cat << 'EOF' >> ~/.zshrc
# ─── 基本設定 ───
export LANG=ja_JP.UTF-8
export EDITOR="code --wait"
export VISUAL="code --wait"

# ─── 履歴設定 ───
HISTFILE=~/.zsh_history
HISTSIZE=50000
SAVEHIST=50000
setopt HIST_IGNORE_DUPS      # 重複コマンドを無視
setopt HIST_IGNORE_ALL_DUPS  # 古い重複を削除
setopt HIST_REDUCE_BLANKS    # 余分な空白を除去
setopt SHARE_HISTORY         # セッション間で共有
setopt INC_APPEND_HISTORY    # 即座に追記

# ─── 補完設定 ───
autoload -Uz compinit && compinit
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'  # 大文字小文字無視
zstyle ':completion:*' menu select                     # メニュー選択
zstyle ':completion:*' list-colors ''                  # 色付き

# ─── エイリアス ───
alias ll='ls -la'
alias la='ls -A'
alias ..='cd ..'
alias ...='cd ../..'
alias g='git'
alias gs='git status'
alias gd='git diff'
alias gc='git commit'
alias gp='git push'
alias k='kubectl'
EOF
```

### 2.3 fish 設定

```fish
# インストール
# macOS
brew install fish

# fish をデフォルトシェルに設定
echo $(which fish) | sudo tee -a /etc/shells
chsh -s $(which fish)

# ~/.config/fish/config.fish
set -gx LANG ja_JP.UTF-8
set -gx EDITOR "code --wait"

# エイリアス (fish は abbr を推奨)
abbr -a g git
abbr -a gs "git status"
abbr -a gd "git diff"
abbr -a gc "git commit"
abbr -a ll "ls -la"
abbr -a .. "cd .."

# Fisher (プラグインマネージャー) のインストール
curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher

# 推奨プラグイン
fisher install jethrokuan/z        # ディレクトリジャンプ
fisher install PatrickF1/fzf.fish  # fzf 統合
fisher install jorgebucaran/autopair.fish  # 括弧自動補完
```

---

## 3. Starship プロンプト

### 3.1 インストールと基本設定

```bash
# インストール
curl -sS https://starship.rs/install.sh | sh

# シェルに追加
# zsh: ~/.zshrc の末尾に追加
eval "$(starship init zsh)"

# fish: ~/.config/fish/config.fish に追加
starship init fish | source

# PowerShell: $PROFILE に追加
Invoke-Expression (&starship init powershell)
```

### 3.2 設定ファイル

```toml
# ~/.config/starship.toml

# プロンプト全体の設定
format = """
$directory\
$git_branch\
$git_status\
$nodejs\
$python\
$rust\
$docker_context\
$cmd_duration\
$line_break\
$character"""

# ディレクトリ表示
[directory]
truncation_length = 3
truncation_symbol = ".../"
style = "bold cyan"

# Git ブランチ
[git_branch]
format = "[$symbol$branch(:$remote_branch)]($style) "
symbol = " "
style = "bold purple"

# Git ステータス
[git_status]
format = '([$all_status$ahead_behind]($style) )'
style = "bold red"
conflicted = "="
ahead = "⇡${count}"
behind = "⇣${count}"
diverged = "⇕⇡${ahead_count}⇣${behind_count}"
untracked = "?${count}"
stashed = "*${count}"
modified = "!${count}"
staged = "+${count}"

# Node.js
[nodejs]
format = "[$symbol($version)]($style) "
symbol = " "
style = "bold green"

# Python
[python]
format = "[$symbol$pyenv_prefix($version)( \\($virtualenv\\))]($style) "
symbol = " "
style = "bold yellow"

# コマンド実行時間
[cmd_duration]
min_time = 2_000  # 2秒以上で表示
format = "[$duration]($style) "
style = "bold yellow"

# プロンプト文字
[character]
success_symbol = "[❯](bold green)"
error_symbol = "[❯](bold red)"
```

### 3.3 プロンプト表示例

```
プロンプト表示イメージ:

  ~/.../my-project  main !2 +1  v20.11.0  3s
  ❯ _

  ├── ディレクトリ (短縮表示)
  │   ├── Git ブランチ名
  │   │   ├── Git ステータス (変更2, ステージ1)
  │   │   │   ├── Node.js バージョン
  │   │   │   │   └── コマンド実行時間
  │   │   │   │
  └───┴───┴───┴── 全てが1行に収まるコンパクト表示
```

---

## 4. tmux

### 4.1 インストールと基本設定

```bash
# インストール
brew install tmux        # macOS
sudo apt install tmux    # Ubuntu

# ~/.tmux.conf
cat << 'EOF' > ~/.tmux.conf
# ─── プレフィックスキー変更 ───
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# ─── 基本設定 ───
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"
set -g mouse on
set -g history-limit 50000
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
set -sg escape-time 0

# ─── ペイン分割 ───
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# ─── ペイン移動 (vim風) ───
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# ─── リサイズ ───
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# ─── ステータスバー ───
set -g status-position top
set -g status-style "bg=#1e1e2e,fg=#cdd6f4"
set -g status-left "#[fg=#1e1e2e,bg=#89b4fa,bold] #S "
set -g status-right "#[fg=#cdd6f4] %Y-%m-%d %H:%M "
set -g status-left-length 30
EOF
```

### 4.2 tmux レイアウト

```
tmux 典型的な開発レイアウト:

┌─────────────────────────────────────────┐
│ Session: my-project                [top] │
├──────────────────┬──────────────────────┤
│                  │                      │
│   エディタ       │   テスト実行         │
│   (vim/code)     │   (npm test --watch) │
│                  │                      │
│                  │                      │
│                  ├──────────────────────┤
│                  │                      │
│                  │   サーバーログ       │
│                  │   (npm run dev)      │
│                  │                      │
├──────────────────┴──────────────────────┤
│ Window: 1:code  2:server  3:db   [tabs] │
└─────────────────────────────────────────┘

操作:
  Ctrl+a |    → 縦分割
  Ctrl+a -    → 横分割
  Ctrl+a h/j/k/l → ペイン移動
  Ctrl+a c    → 新規ウィンドウ
  Ctrl+a 1-9  → ウィンドウ切替
  Ctrl+a d    → デタッチ
  tmux attach → 再アタッチ
```

### 4.3 TPM（tmux Plugin Manager）

```bash
# TPM インストール
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# ~/.tmux.conf に追加
cat << 'EOF' >> ~/.tmux.conf

# ─── プラグイン ───
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-resurrect'   # セッション復元
set -g @plugin 'tmux-plugins/tmux-continuum'    # 自動保存
set -g @plugin 'catppuccin/tmux'                # テーマ

set -g @continuum-restore 'on'

# TPM 初期化 (この行は必ず最後に置く)
run '~/.tmux/plugins/tpm/tpm'
EOF

# プラグインインストール: tmux 内で Ctrl+a I
```

---

## 5. 便利ツール群

### 5.1 モダン CLI ツール

```bash
# 一括インストール (macOS)
brew install \
  bat        `# cat の代替 (シンタックスハイライト)` \
  eza        `# ls の代替 (アイコン・Git対応)` \
  fd         `# find の代替 (高速)` \
  ripgrep    `# grep の代替 (超高速)` \
  fzf        `# ファジーファインダー` \
  zoxide     `# cd の代替 (学習型)` \
  delta      `# diff の代替 (美麗表示)` \
  tldr       `# man の代替 (実例ベース)` \
  jq         `# JSON パーサー` \
  httpie     `# curl の代替 (人間向け)`

# エイリアス設定
alias cat='bat'
alias ls='eza --icons'
alias ll='eza --icons -la'
alias tree='eza --icons --tree'
alias find='fd'
alias grep='rg'
```

### 5.2 fzf 統合

```bash
# fzf インストール & シェル統合
brew install fzf
$(brew --prefix)/opt/fzf/install

# Ctrl+R: 履歴検索
# Ctrl+T: ファイル検索
# Alt+C:  ディレクトリ移動

# カスタム設定 (~/.zshrc)
export FZF_DEFAULT_OPTS='
  --height 40%
  --layout=reverse
  --border
  --preview "bat --style=numbers --color=always --line-range :500 {}"
'
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
```

---

## 6. アンチパターン

### 6.1 素の bash をカスタマイズせずに使い続ける

```
❌ アンチパターン: デフォルトの bash/zsh を設定なしで使用

問題:
  - 補完機能が貧弱で入力ミスが増える
  - 履歴検索が非効率
  - ディレクトリ移動に時間がかかる
  - Git ブランチ状況が見えない

✅ 正しいアプローチ:
  - Starship でプロンプトを情報豊富にする
  - fzf + zoxide で移動を高速化
  - abbr/alias でコマンド短縮
  - シンタックスハイライトプラグイン導入
```

### 6.2 tmux のセッション管理を使わない

```
❌ アンチパターン: プロジェクトごとに新しいターミナルタブを大量に開く

問題:
  - タブが増えすぎて管理不能
  - SSH 切断でプロセスが全て終了
  - 環境の再構築に毎回時間がかかる

✅ 正しいアプローチ:
  - tmux セッションをプロジェクト単位で作成
  - tmux-resurrect で環境を永続化
  - 名前付きセッションで整理: tmux new -s project-name
```

---

## 7. FAQ

### Q1: zsh と fish、どちらを選ぶべき？

**A:** POSIX 互換性が必要な場合は zsh。既存のシェルスクリプトをそのまま使えるのは大きな利点。一方、設定なしで最初から快適に使いたいなら fish がおすすめ。fish の自動補完とシンタックスハイライトは設定不要で動作する。ただし bash 向けスクリプトの互換性は低い。

### Q2: Nerd Font は本当に必要？

**A:** Starship やモダン CLI ツール（eza 等）でアイコン表示を使うなら必須。以下でインストールする。

```bash
brew install --cask font-jetbrains-mono-nerd-font
```

Nerd Font がないと豆腐文字（□）が表示される。ターミナルのフォント設定で "JetBrains Mono Nerd Font" を選択すること。

### Q3: macOS で iTerm2 と Warp、どちらがよい？

**A:** 安定性と実績を重視するなら iTerm2。AI 補完やモダン UI を求めるなら Warp。Warp は AI によるコマンド候補表示が強力だが、Rust 製で拡張性は iTerm2 に劣る。チーム標準にするなら iTerm2 の方が無難。

---

## 8. まとめ

| 項目 | macOS 推奨 | Windows 推奨 | Linux 推奨 |
|------|-----------|-------------|-----------|
| ターミナル | iTerm2 | Windows Terminal | Alacritty |
| シェル | zsh | PowerShell 7 | zsh / fish |
| プロンプト | Starship | Starship | Starship |
| マルチプレクサ | tmux | tmux (WSL) | tmux |
| フォント | JetBrains Mono NF | JetBrains Mono NF | JetBrains Mono NF |
| ファジーファインダー | fzf | fzf | fzf |

---

## 次に読むべきガイド

- [00-vscode-setup.md](./00-vscode-setup.md) — VS Code との統合
- [02-git-config.md](./02-git-config.md) — Git の詳細設定（diff/merge ツール連携）
- [../01-runtime-and-package/00-version-managers.md](../01-runtime-and-package/00-version-managers.md) — ランタイムバージョン管理

---

## 参考文献

1. **iTerm2 Documentation** — https://iterm2.com/documentation.html — iTerm2 の全機能解説。
2. **Starship: Cross-Shell Prompt** — https://starship.rs/config/ — Starship の設定リファレンス。
3. **tmux 2: Productive Mouse-Free Development** (Brian P. Hogan) — https://pragprog.com/titles/bhtmux2/ — tmux のバイブル的書籍。
4. **Modern Unix** — https://github.com/ibraheemdev/modern-unix — モダン CLI ツールのキュレーションリスト。
