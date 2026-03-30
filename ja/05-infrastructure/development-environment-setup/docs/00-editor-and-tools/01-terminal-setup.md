# ターミナル設定

> モダンなターミナルエミュレータとシェル環境を構築し、コマンドライン作業の効率を最大化するための実践ガイド。

## この章で学ぶこと

1. iTerm2 / Windows Terminal / Alacritty / Warp のインストールと高度なカスタマイズ方法
2. zsh / fish / PowerShell の設定とプロンプトカスタマイズ（Starship）
3. tmux によるセッション管理とペイン分割の実践テクニック
4. モダン CLI ツール群の導入と統合設定
5. ターミナル環境のバックアップと再構築手順


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [VS Code セットアップ](./00-vscode-setup.md) の内容を理解していること

---

## 1. ターミナルエミュレータの選定

### 1.1 主要ターミナル比較

| 特徴 | iTerm2 (macOS) | Windows Terminal | Alacritty | Warp | Kitty | WezTerm |
|------|----------------|------------------|-----------|------|-------|---------|
| OS | macOS | Windows | クロスプラットフォーム | macOS / Linux | クロスプラットフォーム | クロスプラットフォーム |
| GPU アクセラレーション | 部分的 | あり | あり | あり | あり | あり |
| タブ/ペイン | あり | あり | なし (tmux併用) | あり | あり | あり |
| 設定形式 | GUI + Plist | JSON | TOML | GUI | conf | Lua |
| 検索機能 | 高機能 | あり | 基本的 | AI搭載 | あり | あり |
| 画像表示 | あり | 限定的 | なし | あり | あり (icat) | あり |
| リガチャ | あり | あり | なし | あり | あり | あり |
| 価格 | 無料 | 無料 | 無料 | Freemium | 無料 | 無料 |
| 描画エンジン | Metal | DirectX | OpenGL/Metal | Metal | OpenGL | OpenGL |
| メモリ使用量 | 中 | 中 | 低 | 高 | 低 | 中 |

### 1.2 iTerm2 セットアップ (macOS)

```bash
# インストール
brew install --cask iterm2

# カラースキームのインポート
# https://iterm2colorschemes.com/ から .itermcolors をダウンロード
# Preferences → Profiles → Colors → Color Presets → Import

# 人気カラースキーム
# - Catppuccin Mocha (ダークテーマ、目に優しい)
# - Tokyo Night (落ち着いたダーク)
# - Dracula (高コントラストダーク)
# - One Half Dark (VS Code 風)
# - Solarized Dark (古典的名作)
# - Nord (青系のクールなテーマ)
```

推奨設定:

```
iTerm2 推奨設定:
┌─────────────────────────────────────────┐
│ Preferences → General                    │
│   ✅ Closing → Confirm "Quit iTerm2"    │
│   ✅ Selection → Copy to pasteboard     │
│      on selection                        │
│   ✅ Magic → Enable Python API          │
│                                           │
│ Preferences → Appearance                 │
│   Theme: Minimal (モダン外観)            │
│   Tab bar location: Top                  │
│   Status bar location: Bottom            │
│                                           │
│ Preferences → Profiles → General        │
│   Working Directory: "Reuse previous"    │
│   Title: Name + Job                      │
│                                           │
│ Preferences → Profiles → Text           │
│   Font: JetBrains Mono Nerd Font 14pt   │
│   ✅ Use ligatures                       │
│   ✅ Anti-aliased                        │
│   Use thin strokes: Retina              │
│                                           │
│ Preferences → Profiles → Window         │
│   Transparency: 5-10%                    │
│   Blur: 10                               │
│   Columns: 120, Rows: 35                │
│   Style: Normal                          │
│                                           │
│ Preferences → Profiles → Terminal       │
│   Scrollback lines: 10000               │
│   ✅ Unlimited scrollback               │
│   ✅ Save lines to scrollback in        │
│      alternate screen mode              │
│                                           │
│ Preferences → Profiles → Session        │
│   ✅ Status bar enabled                 │
│   Configure: CPU / Memory / Network     │
│                                           │
│ Preferences → Profiles → Keys           │
│   Left Option Key: Esc+                  │
│   (単語単位の移動に必要)                   │
│   Right Option Key: Esc+                 │
│                                           │
│ Preferences → Keys → Key Bindings       │
│   ⌘← : Send Hex Codes: 0x01 (行頭)     │
│   ⌘→ : Send Hex Codes: 0x05 (行末)     │
│   ⌥← : Send Escape Sequence: b (単語左)│
│   ⌥→ : Send Escape Sequence: f (単語右)│
└─────────────────────────────────────────┘
```

#### iTerm2 の高度な機能

```bash
# ─── トリガー設定 (自動ハイライト) ───
# Preferences → Profiles → Advanced → Triggers
# Regular Expression: ERROR|FATAL|CRITICAL
# Action: Highlight Text
# Parameters: Red background

# ─── プロファイル自動切替 ───
# SSHしたサーバーごとに背景色を変える
# Preferences → Profiles → Advanced → Automatic Profile Switching
# ホスト名パターン: *.production.* → "Production" プロファイル (赤背景)
# ホスト名パターン: *.staging.* → "Staging" プロファイル (黄背景)

# ─── Shell Integration (非常に便利) ───
# iTerm2 Shell Integration をインストール
curl -L https://iterm2.com/shell_integration/install_shell_integration.sh | bash

# Shell Integration の機能:
# - コマンドの成功/失敗をプロンプト横に表示
# - 直前のコマンド出力をクリック選択
# - imgcat でターミナル内に画像表示
# - it2copy / it2paste でクリップボード操作
# - コマンド履歴のタイムスタンプと実行時間

# ─── imgcat で画像表示 ───
imgcat screenshot.png

# ─── Badge 設定 (ペイン識別用) ───
# Preferences → Profiles → General → Badge
# \(session.hostname) を設定すると、
# 各ペインにホスト名が薄く表示される
```

#### iTerm2 のキーボードショートカット

```
iTerm2 必須ショートカット:
┌──────────────────────────────────────────┐
│ ウィンドウ/タブ操作                       │
│   ⌘T        新しいタブ                   │
│   ⌘N        新しいウィンドウ              │
│   ⌘W        タブを閉じる                 │
│   ⌘1-9      タブ切替                     │
│   ⌘←→       前後のタブ                   │
│                                           │
│ ペイン操作                                │
│   ⌘D        縦分割                       │
│   ⌘⇧D      横分割                       │
│   ⌘⌥←→↑↓   ペイン移動                   │
│   ⌘⇧Enter   ペイン最大化/復帰            │
│                                           │
│ 検索                                      │
│   ⌘F        検索                         │
│   ⌘⇧F      全タブ検索                   │
│   ⌘⌥B      タイムスタンプ付き戻り       │
│                                           │
│ その他                                    │
│   ⌘;        オートコンプリート           │
│   ⌘⇧H      ペースト履歴                 │
│   ⌘⌥E      全ペインに同時入力           │
│   ⌘/        カーソル位置ハイライト       │
│   ⌘U        透過トグル                   │
└──────────────────────────────────────────┘
```

### 1.3 Windows Terminal セットアップ

```jsonc
// settings.json (Windows Terminal)
// 場所: %LOCALAPPDATA%\Packages\Microsoft.WindowsTerminal_8wekyb3d8bbwe\LocalState\settings.json
{
  "$help": "https://aka.ms/terminal-documentation",
  "$schema": "https://aka.ms/terminal-profiles-schema",
  "defaultProfile": "{your-powershell-guid}",
  "copyOnSelect": true,
  "copyFormatting": "none",
  "trimBlockSelection": true,
  "wordDelimiters": " /\\()\"'-.,:;<>~!@#$%^&*|+=[]{}~?",

  "profiles": {
    "defaults": {
      "font": {
        "face": "JetBrains Mono Nerd Font",
        "size": 12,
        "weight": "normal"
      },
      "colorScheme": "One Half Dark",
      "opacity": 95,
      "useAcrylic": true,
      "acrylicOpacity": 0.85,
      "padding": "8",
      "cursorShape": "bar",
      "cursorColor": "#FFFFFF",
      "antialiasingMode": "cleartype",
      "scrollbarState": "hidden",
      "bellStyle": "none",
      "snapOnInput": true,
      "altGrAliasing": true
    },
    "list": [
      {
        "name": "PowerShell 7",
        "source": "Windows.Terminal.PowershellCore",
        "startingDirectory": "%USERPROFILE%",
        "icon": "ms-appx:///ProfileIcons/pwsh.png",
        "commandline": "pwsh.exe -NoLogo"
      },
      {
        "name": "Ubuntu (WSL)",
        "source": "Windows.Terminal.Wsl",
        "startingDirectory": "~",
        "colorScheme": "Catppuccin Mocha"
      },
      {
        "name": "Git Bash",
        "commandline": "C:\\Program Files\\Git\\bin\\bash.exe --login -i",
        "startingDirectory": "%USERPROFILE%",
        "icon": "C:\\Program Files\\Git\\mingw64\\share\\git\\git-for-windows.ico"
      },
      {
        "name": "Azure Cloud Shell",
        "source": "Windows.Terminal.Azure"
      }
    ]
  },

  // カスタムカラースキーム
  "schemes": [
    {
      "name": "Catppuccin Mocha",
      "foreground": "#CDD6F4",
      "background": "#1E1E2E",
      "cursorColor": "#F5E0DC",
      "selectionBackground": "#585B70",
      "black": "#45475A",
      "red": "#F38BA8",
      "green": "#A6E3A1",
      "yellow": "#F9E2AF",
      "blue": "#89B4FA",
      "purple": "#F5C2E7",
      "cyan": "#94E2D5",
      "white": "#BAC2DE",
      "brightBlack": "#585B70",
      "brightRed": "#F38BA8",
      "brightGreen": "#A6E3A1",
      "brightYellow": "#F9E2AF",
      "brightBlue": "#89B4FA",
      "brightPurple": "#F5C2E7",
      "brightCyan": "#94E2D5",
      "brightWhite": "#A6ADC8"
    }
  ],

  "actions": [
    { "command": "toggleFocusMode", "keys": "f11" },
    { "command": "toggleFullscreen", "keys": "alt+enter" },
    { "command": { "action": "splitPane", "split": "horizontal" }, "keys": "alt+shift+-" },
    { "command": { "action": "splitPane", "split": "vertical" }, "keys": "alt+shift+=" },
    { "command": { "action": "moveFocus", "direction": "left" }, "keys": "alt+h" },
    { "command": { "action": "moveFocus", "direction": "down" }, "keys": "alt+j" },
    { "command": { "action": "moveFocus", "direction": "up" }, "keys": "alt+k" },
    { "command": { "action": "moveFocus", "direction": "right" }, "keys": "alt+l" },
    { "command": { "action": "resizePane", "direction": "left" }, "keys": "alt+shift+h" },
    { "command": { "action": "resizePane", "direction": "down" }, "keys": "alt+shift+j" },
    { "command": { "action": "resizePane", "direction": "up" }, "keys": "alt+shift+k" },
    { "command": { "action": "resizePane", "direction": "right" }, "keys": "alt+shift+l" },
    { "command": { "action": "newTab" }, "keys": "ctrl+shift+t" },
    { "command": "find", "keys": "ctrl+shift+f" },
    { "command": { "action": "switchToTab", "index": 0 }, "keys": "alt+1" },
    { "command": { "action": "switchToTab", "index": 1 }, "keys": "alt+2" },
    { "command": { "action": "switchToTab", "index": 2 }, "keys": "alt+3" }
  ]
}
```

### 1.4 Alacritty セットアップ

```bash
# インストール
brew install --cask alacritty    # macOS
sudo apt install alacritty       # Ubuntu
cargo install alacritty          # ソースからビルド
```

```toml
# ~/.config/alacritty/alacritty.toml

# ─── ウィンドウ設定 ───
[window]
dimensions = { columns = 120, lines = 35 }
padding = { x = 8, y = 8 }
decorations = "Buttonless"
opacity = 0.95
startup_mode = "Windowed"
dynamic_title = true

# ─── フォント設定 ───
[font]
size = 14.0

[font.normal]
family = "JetBrains Mono Nerd Font"
style = "Regular"

[font.bold]
family = "JetBrains Mono Nerd Font"
style = "Bold"

[font.italic]
family = "JetBrains Mono Nerd Font"
style = "Italic"

[font.bold_italic]
family = "JetBrains Mono Nerd Font"
style = "Bold Italic"

# ─── カーソル設定 ───
[cursor]
style = { shape = "Beam", blinking = "On" }
vi_mode_style = { shape = "Block", blinking = "Off" }
blink_interval = 500
blink_timeout = 5

# ─── スクロール設定 ───
[scrolling]
history = 10000
multiplier = 3

# ─── カラースキーム (Catppuccin Mocha) ───
[colors.primary]
background = "#1E1E2E"
foreground = "#CDD6F4"
dim_foreground = "#CDD6F4"
bright_foreground = "#CDD6F4"

[colors.cursor]
text = "#1E1E2E"
cursor = "#F5E0DC"

[colors.vi_mode_cursor]
text = "#1E1E2E"
cursor = "#B4BEFE"

[colors.search.matches]
foreground = "#1E1E2E"
background = "#A6ADC8"

[colors.search.focused_match]
foreground = "#1E1E2E"
background = "#A6E3A1"

[colors.normal]
black = "#45475A"
red = "#F38BA8"
green = "#A6E3A1"
yellow = "#F9E2AF"
blue = "#89B4FA"
magenta = "#F5C2E7"
cyan = "#94E2D5"
white = "#BAC2DE"

[colors.bright]
black = "#585B70"
red = "#F38BA8"
green = "#A6E3A1"
yellow = "#F9E2AF"
blue = "#89B4FA"
magenta = "#F5C2E7"
cyan = "#94E2D5"
white = "#A6ADC8"

# ─── キーバインド ───
key = "N"
mods = "Command"
action = "SpawnNewInstance"

key = "Return"
mods = "Command"
action = "ToggleFullscreen"

# tmux との統合 (Ctrl+a をそのまま送信)
key = "A"
mods = "Control"
chars = "\u0001"
```

### 1.5 Warp ターミナルの特徴と設定

```bash
# インストール
brew install --cask warp

# Warp の独自機能:
# 1. AI Command Search: 自然言語でコマンド検索
#    例: "find large files" → find . -type f -size +100M
#
# 2. Blocks: コマンドと出力がブロック単位で管理
#    - 各ブロックを個別にコピー/共有可能
#    - 出力の折りたたみ
#
# 3. Workflows: よく使うコマンドシーケンスを保存
#    - パラメータ付きテンプレート
#    - チームで共有可能
#
# 4. Warp Drive: クラウド同期
#    - 設定の同期
#    - ワークフローの共有
```

```yaml
# ~/.warp/themes/custom.yaml
# カスタムテーマ定義
accent: "#89B4FA"
background: "#1E1E2E"
foreground: "#CDD6F4"
details: "darker"
terminal_colors:
  normal:
    black: "#45475A"
    red: "#F38BA8"
    green: "#A6E3A1"
    yellow: "#F9E2AF"
    blue: "#89B4FA"
    magenta: "#F5C2E7"
    cyan: "#94E2D5"
    white: "#BAC2DE"
  bright:
    black: "#585B70"
    red: "#F38BA8"
    green: "#A6E3A1"
    yellow: "#F9E2AF"
    blue: "#89B4FA"
    magenta: "#F5C2E7"
    cyan: "#94E2D5"
    white: "#A6ADC8"
```

---

## 2. シェル設定

### 2.1 シェル比較

| 特徴 | zsh | fish | PowerShell 7 | bash | nushell |
|------|-----|------|-------------|------|---------|
| POSIX 互換 | はい | いいえ | いいえ | はい | いいえ |
| デフォルトOS | macOS | なし | Windows | Linux | なし |
| 補完機能 | プラグイン必要 | 組み込み | 組み込み | 基本的 | 組み込み |
| スクリプト互換性 | bash とほぼ同じ | 独自構文 | .NET ベース | 標準 | 独自構文 |
| 学習コスト | 低 | 低 | 中 | 低 | 中 |
| プラグインエコシステム | 非常に豊富 | 豊富 | 成長中 | 限定的 | 成長中 |
| 起動速度 | プラグイン依存 | 高速 | 遅い | 高速 | 高速 |
| 構造化データ | なし | なし | オブジェクト | なし | テーブル |
| パイプ | テキスト | テキスト | オブジェクト | テキスト | 構造化 |

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
export PAGER="less -R"
export LESS="-i -M -R -S -W -z-4"

# XDG Base Directory 準拠
export XDG_CONFIG_HOME="$HOME/.config"
export XDG_DATA_HOME="$HOME/.local/share"
export XDG_CACHE_HOME="$HOME/.cache"
export XDG_STATE_HOME="$HOME/.local/state"

# ─── 履歴設定 ───
HISTFILE=~/.zsh_history
HISTSIZE=100000
SAVEHIST=100000
setopt HIST_IGNORE_DUPS      # 重複コマンドを無視
setopt HIST_IGNORE_ALL_DUPS  # 古い重複を削除
setopt HIST_REDUCE_BLANKS    # 余分な空白を除去
setopt SHARE_HISTORY         # セッション間で共有
setopt INC_APPEND_HISTORY    # 即座に追記
setopt HIST_EXPIRE_DUPS_FIRST # 古い重複から期限切れ
setopt HIST_FIND_NO_DUPS     # 検索時に重複を除外
setopt HIST_SAVE_NO_DUPS     # 保存時に重複を除外
setopt EXTENDED_HISTORY       # タイムスタンプ記録

# ─── ディレクトリ移動 ───
setopt AUTO_CD               # ディレクトリ名だけで cd
setopt AUTO_PUSHD            # cd 時にスタックに追加
setopt PUSHD_IGNORE_DUPS     # 重複をスタックに入れない
setopt PUSHD_MINUS           # +/- の意味を逆にする
DIRSTACKSIZE=20              # スタックサイズ

# ─── 補完設定 ───
autoload -Uz compinit && compinit
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'  # 大文字小文字無視
zstyle ':completion:*' menu select                     # メニュー選択
zstyle ':completion:*' list-colors ''                  # 色付き
zstyle ':completion:*' use-cache yes                   # 補完キャッシュ
zstyle ':completion:*' cache-path "$XDG_CACHE_HOME/zsh/.zcompcache"
zstyle ':completion:*:descriptions' format '%B%d%b'    # 説明の書式
zstyle ':completion:*:warnings' format 'No matches for: %d'
zstyle ':completion:*' group-name ''                   # グループ名表示
zstyle ':completion:*:*:kill:*' menu yes select        # kill の補完
zstyle ':completion:*:*:kill:*:processes' list-colors '=(#b) #([0-9]#)*=0=01;31'

# ─── キーバインド (Emacs モード) ───
bindkey -e
bindkey '^[[A' history-search-backward    # ↑で前方一致検索
bindkey '^[[B' history-search-forward     # ↓で前方一致検索
bindkey '^[b' backward-word               # Alt+b で単語戻り
bindkey '^[f' forward-word                # Alt+f で単語進み
bindkey '^U' backward-kill-line           # Ctrl+U で行頭まで削除
bindkey '^K' kill-line                    # Ctrl+K で行末まで削除

# ─── エイリアス ───
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias ..='cd ..'
alias ...='cd ../..'
alias ....='cd ../../..'
alias g='git'
alias gs='git status'
alias gd='git diff'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline -20'
alias k='kubectl'
alias d='docker'
alias dc='docker compose'
alias tf='terraform'
alias py='python3'
alias pip='pip3'

# ─── 便利関数 ───
# mkcd: ディレクトリ作成 & 移動
mkcd() { mkdir -p "$1" && cd "$1" }

# extract: 統合解凍コマンド
extract() {
  if [ -f "$1" ]; then
    case "$1" in
      *.tar.bz2) tar xjf "$1" ;;
      *.tar.gz)  tar xzf "$1" ;;
      *.tar.xz)  tar xJf "$1" ;;
      *.bz2)     bunzip2 "$1" ;;
      *.gz)      gunzip "$1" ;;
      *.tar)     tar xf "$1" ;;
      *.tbz2)    tar xjf "$1" ;;
      *.tgz)     tar xzf "$1" ;;
      *.zip)     unzip "$1" ;;
      *.Z)       uncompress "$1" ;;
      *.7z)      7z x "$1" ;;
      *.rar)     unrar x "$1" ;;
      *)         echo "Cannot extract '$1'" ;;
    esac
  else
    echo "'$1' is not a valid file"
  fi
}

# port: 指定ポートを使っているプロセスを表示
port() { lsof -i :"$1" }

# weather: 天気予報表示
weather() { curl "wttr.in/${1:-Tokyo}?lang=ja" }
EOF
```

#### zsh プラグイン管理 (zinit)

```bash
# ─── zinit (高速プラグインマネージャー) ───
# インストール
bash -c "$(curl --fail --show-error --silent --location https://raw.githubusercontent.com/zdharma-continuum/zinit/HEAD/scripts/install.sh)"

# ~/.zshrc にプラグイン追加
cat << 'EOF' >> ~/.zshrc

# ─── zinit プラグイン ───
# 遅延読み込みで起動高速化
zinit light zsh-users/zsh-autosuggestions          # コマンド自動候補
zinit light zsh-users/zsh-syntax-highlighting       # シンタックスハイライト
zinit light zsh-users/zsh-completions               # 追加補完定義

# 履歴検索の強化
zinit light zsh-users/zsh-history-substring-search

# ─── Oh My Zsh のスニペット利用 (必要な部分だけ) ───
zinit snippet OMZP::git                # git エイリアス
zinit snippet OMZP::docker             # docker 補完
zinit snippet OMZP::docker-compose     # docker compose 補完
zinit snippet OMZP::kubectl            # kubectl 補完
zinit snippet OMZP::aws                # AWS CLI 補完
zinit snippet OMZP::terraform          # Terraform 補完
zinit snippet OMZP::npm                # npm 補完

# ─── 自動候補の設定 ───
ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=243'  # 候補の色
ZSH_AUTOSUGGEST_STRATEGY=(history completion)  # 候補の優先順
ZSH_AUTOSUGGEST_BUFFER_MAX_SIZE=20  # 最大文字数
bindkey '^ ' autosuggest-accept  # Ctrl+Space で候補確定
EOF
```

#### zsh の起動速度計測と最適化

```bash
# 起動時間を計測
time zsh -i -c exit

# 詳細なプロファイリング
# .zshrc の先頭に追加:
# zmodload zsh/zprof
# .zshrc の末尾に追加:
# zprof

# 目標: 200ms 以下
# 主な遅延原因:
# - nvm の初期化 (~300ms) → fnm に変更で解決
# - compinit の重複呼び出し → キャッシュで解決
# - Oh My Zsh の全体読み込み → zinit で必要部分だけ

# compinit キャッシュ最適化
autoload -Uz compinit
if [[ -n ${ZDOTDIR:-$HOME}/.zcompdump(#qN.mh+24) ]]; then
  compinit
else
  compinit -C  # キャッシュを使用 (24時間以内なら)
fi
```

### 2.3 fish 設定

```fish
# インストール
# macOS
brew install fish

# Ubuntu
sudo apt-add-repository ppa:fish-shell/release-3
sudo apt update
sudo apt install fish

# fish をデフォルトシェルに設定
echo $(which fish) | sudo tee -a /etc/shells
chsh -s $(which fish)

# ~/.config/fish/config.fish
set -gx LANG ja_JP.UTF-8
set -gx EDITOR "code --wait"
set -gx VISUAL "code --wait"

# XDG Base Directory
set -gx XDG_CONFIG_HOME $HOME/.config
set -gx XDG_DATA_HOME $HOME/.local/share
set -gx XDG_CACHE_HOME $HOME/.cache

# パス設定
fish_add_path ~/.local/bin
fish_add_path ~/.cargo/bin

# エイリアス (fish は abbr を推奨)
abbr -a g git
abbr -a gs "git status"
abbr -a gd "git diff"
abbr -a gc "git commit"
abbr -a gp "git push"
abbr -a gl "git log --oneline -20"
abbr -a ll "ls -la"
abbr -a .. "cd .."
abbr -a ... "cd ../.."
abbr -a d docker
abbr -a dc "docker compose"
abbr -a k kubectl
abbr -a py python3

# ─── fish の独自機能 ───
# abbr と alias の違い:
# - abbr: 入力時に展開される (履歴に元のコマンドが残る)
# - alias: 実行時に変換される (履歴に alias 名が残る)
# → abbr 推奨: 学習効果があり、他の環境でも対応できる

# Fisher (プラグインマネージャー) のインストール
curl -sL https://raw.githubusercontent.com/jorgebucaran/fisher/main/functions/fisher.fish | source && fisher install jorgebucaran/fisher

# 推奨プラグイン
fisher install jethrokuan/z               # ディレクトリジャンプ
fisher install PatrickF1/fzf.fish         # fzf 統合
fisher install jorgebucaran/autopair.fish  # 括弧自動補完
fisher install meaningful-ooo/sponge      # 失敗コマンド履歴除外
fisher install jorgebucaran/nvm.fish      # Node.js バージョン管理
fisher install laughedelic/pisces          # ペア文字補完

# ─── カスタム関数 ───
# ~/.config/fish/functions/mkcd.fish
function mkcd
    mkdir -p $argv[1]; and cd $argv[1]
end

# ~/.config/fish/functions/port.fish
function port
    lsof -i :$argv[1]
end
```

### 2.4 PowerShell 7 設定 (Windows / クロスプラットフォーム)

```powershell
# インストール
# Windows
winget install Microsoft.PowerShell
# macOS
brew install powershell/tap/powershell
# Linux
sudo apt install powershell

# プロファイル場所の確認
echo $PROFILE
# 通常: ~/Documents/PowerShell/Microsoft.PowerShell_profile.ps1

# ─── $PROFILE の設定 ───
# モジュールインストール
Install-Module posh-git -Scope CurrentUser
Install-Module PSReadLine -Scope CurrentUser -Force
Install-Module Terminal-Icons -Scope CurrentUser
Install-Module PSFzf -Scope CurrentUser
Install-Module z -Scope CurrentUser

# $PROFILE に追加
@'
# ─── モジュール読み込み ───
Import-Module posh-git
Import-Module Terminal-Icons
Import-Module PSFzf
Import-Module z

# ─── PSReadLine 設定 ───
Set-PSReadLineOption -PredictionSource HistoryAndPlugin
Set-PSReadLineOption -PredictionViewStyle ListView
Set-PSReadLineOption -EditMode Emacs
Set-PSReadLineOption -HistorySearchCursorMovesToEnd
Set-PSReadLineKeyHandler -Key UpArrow -Function HistorySearchBackward
Set-PSReadLineKeyHandler -Key DownArrow -Function HistorySearchForward
Set-PSReadLineKeyHandler -Key Tab -Function MenuComplete
Set-PSReadLineKeyHandler -Key Ctrl+d -Function DeleteChar

# ─── エイリアス ───
Set-Alias -Name g -Value git
Set-Alias -Name k -Value kubectl
Set-Alias -Name ll -Value Get-ChildItem
Set-Alias -Name which -Value Get-Command

# ─── Starship プロンプト ───
Invoke-Expression (&starship init powershell)

# ─── fzf 設定 ───
Set-PsFzfOption -PSReadlineChordProvider 'Ctrl+t' -PSReadlineChordReverseHistory 'Ctrl+r'
'@ | Out-File -FilePath $PROFILE -Encoding utf8
```

---

## 3. Starship プロンプト

### 3.1 インストールと基本設定

```bash
# インストール
curl -sS https://starship.rs/install.sh | sh

# またはパッケージマネージャー経由
brew install starship          # macOS
sudo snap install starship     # Ubuntu
winget install Starship.Starship  # Windows

# シェルに追加
# zsh: ~/.zshrc の末尾に追加
eval "$(starship init zsh)"

# fish: ~/.config/fish/config.fish に追加
starship init fish | source

# PowerShell: $PROFILE に追加
Invoke-Expression (&starship init powershell)

# bash: ~/.bashrc の末尾に追加
eval "$(starship init bash)"
```

### 3.2 設定ファイル

```toml
# ~/.config/starship.toml

# プロンプト全体の設定
format = """
$username\
$hostname\
$directory\
$git_branch\
$git_status\
$git_state\
$nodejs\
$python\
$rust\
$golang\
$java\
$docker_context\
$kubernetes\
$terraform\
$aws\
$cmd_duration\
$line_break\
$jobs\
$character"""

# 右プロンプト
right_format = """$time"""

# 空行を挿入
add_newline = true

# コマンドタイムアウト
command_timeout = 1000

# ユーザー名 (SSH時のみ表示)
[username]
show_always = false
style_user = "bold blue"
style_root = "bold red"
format = "$user@"

# ホスト名 (SSH時のみ表示)
[hostname]
ssh_only = true
format = "$ssh_symbol$hostname "
style = "bold green"

# ディレクトリ表示
[directory]
truncation_length = 3
truncation_symbol = ".../"
style = "bold cyan"
read_only = " 🔒"
home_symbol = "~"
truncate_to_repo = true

# ディレクトリの置換 (長いパスを短縮)
[directory.substitutions]
"Documents" = "DOC"
"Downloads" = "DL"
"src/components" = "comp"

# Git ブランチ
[git_branch]
format = "$symbol$branch(:$remote_branch) "
symbol = " "
style = "bold purple"
truncation_length = 30
truncation_symbol = "..."

# Git ステータス
[git_status]
format = '($all_status$ahead_behind )'
style = "bold red"
conflicted = "="
ahead = "⇡${count}"
behind = "⇣${count}"
diverged = "⇕⇡${ahead_count}⇣${behind_count}"
untracked = "?${count}"
stashed = "*${count}"
modified = "!${count}"
staged = "+${count}"
renamed = "»${count}"
deleted = "✘${count}"

# Git 操作中の状態
[git_state]
format = '\($state( $progress_current of $progress_total)\) '
rebase = "REBASING"
merge = "MERGING"
revert = "REVERTING"
cherry_pick = "CHERRY-PICKING"
bisect = "BISECTING"

# Node.js
[nodejs]
format = "$symbol($version) "
symbol = " "
style = "bold green"
detect_files = ["package.json", ".node-version", ".nvmrc"]
detect_folders = ["node_modules"]

# Python
[python]
format = "$symbol$pyenv_prefix($version)( \\($virtualenv\\)) "
symbol = " "
style = "bold yellow"
detect_extensions = ["py"]
detect_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]

# Rust
[rust]
format = "$symbol($version) "
symbol = " "
style = "bold red"

# Go
[golang]
format = "$symbol($version) "
symbol = " "
style = "bold cyan"

# Java
[java]
format = "$symbol($version) "
symbol = " "
style = "bold orange"

# Docker
[docker_context]
format = "$symbol$context "
symbol = " "
style = "bold blue"
only_with_files = true

# Kubernetes
[kubernetes]
format = "$symbol$context( \\($namespace\\)) "
symbol = "⎈ "
style = "bold blue"
disabled = false
detect_files = ["k8s", "kubernetes"]

[kubernetes.context_aliases]
"arn:aws:eks:*:*:cluster/production" = "PROD"
"arn:aws:eks:*:*:cluster/staging" = "STG"

# Terraform
[terraform]
format = "$symbol$workspace "
symbol = "💠 "
style = "bold purple"

# AWS
[aws]
format = "$symbol($profile)(\\($region\\)) "
symbol = "☁️ "
style = "bold yellow"

[aws.region_aliases]
ap-northeast-1 = "tokyo"
us-east-1 = "virginia"
eu-west-1 = "ireland"

# コマンド実行時間
[cmd_duration]
min_time = 2_000  # 2秒以上で表示
format = "$duration "
style = "bold yellow"
show_milliseconds = false
show_notifications = true
min_time_to_notify = 30_000  # 30秒以上で通知

# プロンプト文字
[character]
success_symbol = "❯"
error_symbol = "❯"
vimcmd_symbol = "❮"

# バックグラウンドジョブ
[jobs]
symbol = "✦ "
threshold = 1
format = "$symbol$number "

# 時刻表示
[time]
disabled = false
format = "$time"
style = "dimmed white"
time_format = "%H:%M"
```

### 3.3 プロンプト表示例

```
プロンプト表示イメージ:

  ~/.../my-project  main !2 +1  v20.11.0  3s         14:30
  ❯ _

  ├── ディレクトリ (短縮表示)
  │   ├── Git ブランチ名
  │   │   ├── Git ステータス (変更2, ステージ1)
  │   │   │   ├── Node.js バージョン
  │   │   │   │   ├── コマンド実行時間
  │   │   │   │   │              └── 時刻 (右寄せ)
  │   │   │   │   │
  └───┴───┴───┴───┴── 全てが1行に収まるコンパクト表示

  SSH 接続時の表示:
  gaku@production ~/.../deploy  main  🐳docker  ⎈PROD(default)
  ❯ _

  Python プロジェクト:
  ~/.../ml-project  feature/model !1  3.12.3 (venv)  15s
  ❯ _
```

### 3.4 プリセットの活用

```bash
# Starship には各種プリセットが用意されている
# 一覧表示
starship preset --list

# プリセット適用
starship preset nerd-font-symbols -o ~/.config/starship.toml
starship preset tokyo-night -o ~/.config/starship.toml
starship preset pastel-powerline -o ~/.config/starship.toml

# プリセットをベースにカスタマイズすることも可能
```

---

## 4. tmux

### 4.1 インストールと基本設定

```bash
# インストール
brew install tmux        # macOS
sudo apt install tmux    # Ubuntu

# バージョン確認 (3.2+ 推奨)
tmux -V

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
set -g focus-events on
set -g set-clipboard on
set -g display-time 4000
set -g display-panes-time 1500

# ─── コピーモード (vi キーバインド) ───
setw -g mode-keys vi
bind -T copy-mode-vi v send -X begin-selection
bind -T copy-mode-vi y send -X copy-pipe-and-cancel "pbcopy"  # macOS
# bind -T copy-mode-vi y send -X copy-pipe-and-cancel "xclip -selection clipboard"  # Linux
bind -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "pbcopy"
bind -T copy-mode-vi Escape send -X cancel

# ─── ペイン分割 ───
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
bind _ split-window -v -c "#{pane_current_path}" -p 30  # 下30%
unbind '"'
unbind %

# ─── ペイン移動 (vim風) ───
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# ─── Alt+矢印でペイン移動 (プレフィックス不要) ───
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# ─── リサイズ ───
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# ─── ウィンドウ操作 ───
bind c new-window -c "#{pane_current_path}"
bind -n S-Left previous-window   # Shift+← で前のウィンドウ
bind -n S-Right next-window      # Shift+→ で次のウィンドウ
bind -r < swap-window -t -1 \; previous-window  # ウィンドウ入替
bind -r > swap-window -t +1 \; next-window

# ─── セッション操作 ───
bind S choose-session             # セッション一覧
bind R command-prompt -I "#{session_name}" "rename-session '%%'"

# ─── 設定リロード ───
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# ─── ステータスバー ───
set -g status-position top
set -g status-interval 5
set -g status-style "bg=#1e1e2e,fg=#cdd6f4"
set -g status-left-length 40
set -g status-right-length 80
set -g status-left "#[fg=#1e1e2e,bg=#89b4fa,bold] #S #[fg=#89b4fa,bg=#1e1e2e]"
set -g status-right "#[fg=#a6adc8] #(whoami)@#H  %Y-%m-%d %H:%M "

# ─── ウィンドウ表示 ───
setw -g window-status-format "#[fg=#6c7086] #I:#W "
setw -g window-status-current-format "#[fg=#1e1e2e,bg=#a6e3a1,bold] #I:#W "
setw -g window-status-separator ""

# ─── ペイン境界線 ───
set -g pane-border-style "fg=#313244"
set -g pane-active-border-style "fg=#89b4fa"

# ─── メッセージ ───
set -g message-style "fg=#cdd6f4,bg=#313244,bold"
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
  Ctrl+a [    → コピーモード (vi移動)
  Ctrl+a z    → ペイン最大化/復帰
  Ctrl+a !    → ペインをウィンドウに分離
  Ctrl+a S    → セッション一覧
```

### 4.3 tmux スクリプト（プロジェクト用セッション自動作成）

```bash
#!/bin/bash
# ~/.local/bin/tmux-project.sh
# プロジェクト用 tmux セッションを自動構築

PROJECT_DIR="${1:-.}"
SESSION_NAME=$(basename "$PROJECT_DIR")

# 既にセッションがあればアタッチ
tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
  tmux attach -t "$SESSION_NAME"
  exit 0
}

# 新規セッション作成
tmux new-session -d -s "$SESSION_NAME" -c "$PROJECT_DIR"

# ウィンドウ 1: エディタ
tmux rename-window -t "$SESSION_NAME:1" "editor"
tmux send-keys -t "$SESSION_NAME:1" "code ." C-m

# ウィンドウ 2: 開発サーバー + テスト
tmux new-window -t "$SESSION_NAME" -n "dev" -c "$PROJECT_DIR"
tmux split-window -h -t "$SESSION_NAME:2" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION_NAME:2.1" "npm run dev" C-m
tmux send-keys -t "$SESSION_NAME:2.2" "npm test -- --watch" C-m

# ウィンドウ 3: Git / 作業用
tmux new-window -t "$SESSION_NAME" -n "git" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION_NAME:3" "git status" C-m

# ウィンドウ 4: DB / ログ
tmux new-window -t "$SESSION_NAME" -n "misc" -c "$PROJECT_DIR"
tmux split-window -v -t "$SESSION_NAME:4" -c "$PROJECT_DIR"

# ウィンドウ 1 に戻る
tmux select-window -t "$SESSION_NAME:1"

# アタッチ
tmux attach -t "$SESSION_NAME"
```

```bash
# エイリアスとして登録
alias tp='~/.local/bin/tmux-project.sh'

# 使用例
tp ~/projects/my-app    # my-app セッション自動構築
tp                      # カレントディレクトリでセッション作成
```

### 4.4 TPM（tmux Plugin Manager）

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
set -g @plugin 'tmux-plugins/tmux-yank'         # クリップボード統合
set -g @plugin 'tmux-plugins/tmux-pain-control' # ペイン操作強化
set -g @plugin 'tmux-plugins/tmux-sessionist'   # セッション操作強化
set -g @plugin 'catppuccin/tmux'                # テーマ
set -g @plugin 'tmux-plugins/tmux-cpu'          # CPU使用率表示
set -g @plugin 'tmux-plugins/tmux-battery'      # バッテリー表示

# ─── Resurrect 設定 ───
set -g @resurrect-capture-pane-contents 'on'
set -g @resurrect-strategy-vim 'session'
set -g @resurrect-strategy-nvim 'session'
set -g @resurrect-processes '~vim ~nvim ~less ~more ~man ~top ~htop'

# ─── Continuum 設定 ───
set -g @continuum-restore 'on'
set -g @continuum-save-interval '15'  # 15分ごとに自動保存

# ─── Catppuccin テーマ設定 ───
set -g @catppuccin_flavor 'mocha'
set -g @catppuccin_window_status_style "rounded"
set -g @catppuccin_status_left_separator "█"
set -g @catppuccin_status_right_separator "█"

# TPM 初期化 (この行は必ず最後に置く)
run '~/.tmux/plugins/tpm/tpm'
EOF

# プラグインインストール: tmux 内で Ctrl+a I
# プラグイン更新: tmux 内で Ctrl+a U
# プラグイン削除: リストから削除後 Ctrl+a alt+u
```

### 4.5 tmux の便利なコマンド集

```bash
# ─── セッション管理 ───
tmux new -s work                     # "work" セッション作成
tmux new -s work -d                  # デタッチ状態で作成
tmux ls                              # セッション一覧
tmux attach -t work                  # セッションにアタッチ
tmux kill-session -t work            # セッション削除
tmux kill-server                     # 全セッション削除
tmux switch -t work                  # セッション切替

# ─── ウィンドウ/ペイン情報 ───
tmux list-windows                    # ウィンドウ一覧
tmux list-panes                      # ペイン一覧
tmux display-panes                   # ペイン番号表示

# ─── コマンド送信 (スクリプトから) ───
tmux send-keys -t work:1 "npm start" C-m
tmux send-keys -t work:2.1 "npm test" C-m

# ─── レイアウト変更 ───
# Ctrl+a Space で順番に切替
# even-horizontal : 等幅横並び
# even-vertical   : 等幅縦並び
# main-horizontal : メイン上 + 下に分割
# main-vertical   : メイン左 + 右に分割
# tiled           : タイル状
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
  httpie     `# curl の代替 (人間向け)` \
  dust       `# du の代替 (ビジュアル)` \
  duf        `# df の代替 (ビジュアル)` \
  bottom     `# top の代替 (リッチUI)` \
  procs      `# ps の代替 (モダン表示)` \
  sd         `# sed の代替 (直感的)` \
  tokei      `# コード行数カウンター` \
  hyperfine  `# ベンチマークツール` \
  gping      `# ping のグラフ表示版` \
  dog        `# dig の代替 (DNS)` \
  xh         `# HTTPieのRust版 (超高速)`

# Ubuntu の場合
sudo apt install bat fd-find ripgrep fzf jq httpie
# 注: Ubuntu では bat → batcat, fd → fdfind としてインストールされる
# エイリアスが必要:
alias bat='batcat'
alias fd='fdfind'

# エイリアス設定 (~/.zshrc)
alias cat='bat --paging=never'
alias ls='eza --icons'
alias ll='eza --icons -la --git'
alias lt='eza --icons --tree --level=3'
alias tree='eza --icons --tree'
alias find='fd'
alias grep='rg'
alias du='dust'
alias df='duf'
alias top='btm'
alias ps='procs'
alias sed='sd'
alias dig='dog'
alias ping='gping'
```

### 5.2 各ツールの詳細設定

```bash
# ─── bat の設定 ───
# テーマ一覧
bat --list-themes

# ~/.config/bat/config
cat << 'EOF' > ~/.config/bat/config
--theme="Catppuccin Mocha"
--style="numbers,changes,header,grid"
--italic-text=always
--map-syntax "*.conf:INI"
--map-syntax ".ignore:Git Ignore"
--map-syntax "*.npmrc:INI"
--pager="less -RF"
EOF

# ─── eza の高度な使い方 ───
# Git ステータス付き一覧
eza -la --git --icons --group-directories-first
# ツリー表示 (3階層、.gitignore 除外)
eza --tree --level=3 --icons --git-ignore
# ファイルサイズ順
eza -la --sort=size --reverse --icons
# 最近変更されたファイル
eza -la --sort=modified --icons | head -20

# ─── ripgrep の設定 ───
# ~/.config/ripgrep/config (RIPGREP_CONFIG_PATH で指定)
export RIPGREP_CONFIG_PATH="$HOME/.config/ripgrep/config"
cat << 'EOF' > ~/.config/ripgrep/config
--smart-case
--hidden
--glob=!.git
--glob=!node_modules
--glob=!.next
--glob=!dist
--glob=!*.min.js
--glob=!*.map
--colors=line:fg:yellow
--colors=path:fg:green
--colors=match:bg:yellow
--colors=match:fg:black
--max-columns=200
--max-columns-preview
EOF

# ─── zoxide の設定 ───
# ~/.zshrc に追加
eval "$(zoxide init zsh)"
# 使い方:
# z foo      → "foo" を含む最近のディレクトリにジャンプ
# z foo bar  → "foo" と "bar" 両方を含むディレクトリ
# zi foo     → fzf でインタラクティブ選択
# zoxide query --list  → データベース内容表示

# ─── delta の設定 ───
# ~/.gitconfig の [delta] セクション
# (02-git-config.md で詳述)
```

### 5.3 fzf 統合

```bash
# fzf インストール & シェル統合
brew install fzf
$(brew --prefix)/opt/fzf/install

# Ctrl+R: 履歴検索
# Ctrl+T: ファイル検索
# Alt+C:  ディレクトリ移動

# カスタム設定 (~/.zshrc)
export FZF_DEFAULT_OPTS='
  --height 60%
  --layout=reverse
  --border=rounded
  --preview-window=right:60%:wrap
  --preview "bat --style=numbers --color=always --line-range :500 {}"
  --bind "ctrl-d:half-page-down,ctrl-u:half-page-up"
  --bind "ctrl-y:execute-silent(echo {} | pbcopy)+abort"
  --color=bg+:#313244,bg:#1e1e2e,spinner:#f5e0dc,hl:#f38ba8
  --color=fg:#cdd6f4,header:#f38ba8,info:#cba6f7,pointer:#f5e0dc
  --color=marker:#f5e0dc,fg+:#cdd6f4,prompt:#cba6f7,hl+:#f38ba8
'
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_ALT_C_COMMAND='fd --type d --hidden --follow --exclude .git'

# ─── fzf カスタム関数 ───

# fkill: プロセスをインタラクティブに kill
fkill() {
  local pid
  pid=$(ps -ef | sed 1d | fzf -m --header='Select process to kill' | awk '{print $2}')
  if [ -n "$pid" ]; then
    echo "$pid" | xargs kill -${1:-9}
  fi
}

# fbr: ブランチをインタラクティブに切替
fbr() {
  local branches branch
  branches=$(git --no-pager branch -vv) &&
  branch=$(echo "$branches" | fzf +m --header='Select branch') &&
  git checkout $(echo "$branch" | awk '{print $1}' | sed "s/.* //")
}

# flog: コミットログをインタラクティブに表示
flog() {
  git log --oneline --graph --color=always |
  fzf --ansi --preview 'git show --color=always {1}' \
    --bind 'enter:execute(git show --color=always {1} | less -R)'
}

# fenv: 環境変数をインタラクティブに検索
fenv() {
  local var
  var=$(env | sort | fzf --header='Select environment variable') &&
  echo "$var"
}

# fdoc: Docker コンテナをインタラクティブに操作
fdoc() {
  local container
  container=$(docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}" |
    sed 1d | fzf --header='Select container') &&
  docker exec -it $(echo "$container" | awk '{print $1}') /bin/sh
}
```

---

## 6. ターミナル環境のバックアップと再構築

### 6.1 dotfiles リポジトリ

```bash
# dotfiles リポジトリの構成
dotfiles/
├── .zshrc
├── .config/
│   ├── starship.toml
│   ├── alacritty/
│   │   └── alacritty.toml
│   ├── bat/
│   │   └── config
│   ├── ripgrep/
│   │   └── config
│   └── fish/
│       └── config.fish
├── .tmux.conf
├── .gitconfig
├── .ssh/
│   └── config
├── Brewfile
└── setup.sh

# ─── setup.sh (新マシンセットアップスクリプト) ───
#!/bin/bash
set -euo pipefail

echo "=== 開発環境セットアップ開始 ==="

# Homebrew
if ! command -v brew &>/dev/null; then
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Brewfile からインストール
brew bundle install

# シンボリックリンク作成
DOTFILES_DIR="$(cd "$(dirname "$0")" && pwd)"
ln -sf "$DOTFILES_DIR/.zshrc" ~/.zshrc
ln -sf "$DOTFILES_DIR/.tmux.conf" ~/.tmux.conf
ln -sf "$DOTFILES_DIR/.gitconfig" ~/.gitconfig
mkdir -p ~/.config
ln -sf "$DOTFILES_DIR/.config/starship.toml" ~/.config/starship.toml
ln -sf "$DOTFILES_DIR/.config/bat" ~/.config/bat
ln -sf "$DOTFILES_DIR/.config/ripgrep" ~/.config/ripgrep

# zinit プラグインインストール
zsh -c 'source ~/.zshrc'

# TPM (tmux Plugin Manager)
if [ ! -d ~/.tmux/plugins/tpm ]; then
  git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm
fi

# Nerd Font
brew install --cask font-jetbrains-mono-nerd-font

echo "=== セットアップ完了 ==="
echo "ターミナルを再起動してください"
```

### 6.2 GNU Stow を使った管理

```bash
# GNU Stow: シンボリックリンクの自動管理
brew install stow

# ディレクトリ構成
dotfiles/
├── zsh/
│   └── .zshrc              → ~/.zshrc
├── tmux/
│   └── .tmux.conf          → ~/.tmux.conf
├── git/
│   └── .gitconfig          → ~/.gitconfig
├── starship/
│   └── .config/
│       └── starship.toml   → ~/.config/starship.toml
└── alacritty/
    └── .config/
        └── alacritty/
            └── alacritty.toml → ~/.config/alacritty/alacritty.toml

# Stow で一括リンク
cd ~/dotfiles
stow zsh tmux git starship alacritty

# 個別に管理
stow zsh          # zsh の設定だけリンク
stow -D tmux      # tmux のリンクを解除
stow -R starship  # starship のリンクを再作成
```

---

## 7. アンチパターン

### 7.1 素の bash をカスタマイズせずに使い続ける

```
❌ アンチパターン: デフォルトの bash/zsh を設定なしで使用

問題:
  - 補完機能が貧弱で入力ミスが増える
  - 履歴検索が非効率
  - ディレクトリ移動に時間がかかる
  - Git ブランチ状況が見えない
  - 繰り返し作業の自動化ができない

✅ 正しいアプローチ:
  - Starship でプロンプトを情報豊富にする
  - fzf + zoxide で移動を高速化
  - abbr/alias でコマンド短縮
  - シンタックスハイライトプラグイン導入
  - 自動補完プラグインで入力ミス削減
```

### 7.2 tmux のセッション管理を使わない

```
❌ アンチパターン: プロジェクトごとに新しいターミナルタブを大量に開く

問題:
  - タブが増えすぎて管理不能
  - SSH 切断でプロセスが全て終了
  - 環境の再構築に毎回時間がかかる
  - コンテキストスイッチのコスト増大

✅ 正しいアプローチ:
  - tmux セッションをプロジェクト単位で作成
  - tmux-resurrect で環境を永続化
  - 名前付きセッションで整理: tmux new -s project-name
  - tmux-project.sh スクリプトで環境を自動構築
```

### 7.3 dotfiles を管理しない

```
❌ アンチパターン: 設定ファイルをバージョン管理せず手動管理

問題:
  - マシン買い替え時に環境再構築で丸一日消費
  - チームメンバーと設定の共有ができない
  - 設定変更の履歴が追えない
  - 複数マシン間で設定が一致しない

✅ 正しいアプローチ:
  - dotfiles リポジトリを Git で管理
  - GNU Stow でシンボリックリンクを自動化
  - Brewfile でツール一覧を管理
  - setup.sh で新マシンセットアップを自動化
  - プライベートリポジトリに保存 (SSH鍵やトークンは除外)
```

### 7.4 重すぎるプラグイン構成

```
❌ アンチパターン: Oh My Zsh を全プラグイン有効で使用

問題:
  - シェル起動に 2-5秒かかる
  - ターミナルを開くたびにストレス
  - 使っていないプラグインがメモリを消費
  - アップデートで予期せぬ破壊的変更

✅ 正しいアプローチ:
  - zinit で必要なプラグインだけ遅延読み込み
  - Oh My Zsh のスニペット機能で必要部分だけ取得
  - 定期的に zprof で起動時間をプロファイリング
  - 目標起動時間: 200ms 以下
  - 不要なプラグインは積極的に削除
```

---

## 8. トラブルシューティング

### 8.1 よくある問題と解決策

```bash
# ─── 文字化け (豆腐文字 □) ───
# 原因: Nerd Font がインストールされていない
# 解決:
brew install --cask font-jetbrains-mono-nerd-font
# ターミナルのフォント設定で "JetBrains Mono Nerd Font" を選択

# ─── zsh の起動が遅い ───
# 原因の特定:
zmodload zsh/zprof  # .zshrc 先頭に追加
zprof               # .zshrc 末尾に追加
# → 遅いプラグインを特定して遅延読み込みに変更

# ─── tmux の色がおかしい ───
# 原因: TERM 設定の不一致
# 解決:
# .tmux.conf:
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"
# .zshrc:
export TERM="xterm-256color"

# ─── tmux 内で pbcopy が動かない ───
# macOS の場合:
brew install reattach-to-user-namespace
# .tmux.conf に追加:
# set -g default-command "reattach-to-user-namespace -l $SHELL"
# ※ tmux 2.6+ では不要な場合が多い

# ─── SSH 接続先で Starship が表示されない ───
# SSH 先にも Starship をインストールする必要がある
# または、SSH 先では PROMPT_COMMAND を使った軽量プロンプトにフォールバック

# ─── fzf の Ctrl+R が動かない ───
# fzf のシェル統合を再インストール
$(brew --prefix)/opt/fzf/install --all
# .zshrc の読み込み順序を確認 (fzf は zinit の後に)

# ─── eza で Git ステータスが表示されない ───
# .git ディレクトリがない場所で実行している
# または git がインストールされていない
git --version  # 確認
```

### 8.2 パフォーマンスチューニング

```bash
# ─── シェル起動時間のベンチマーク ───
# 10回平均を計測
for i in $(seq 1 10); do time zsh -i -c exit; done

# hyperfine で精密計測
hyperfine 'zsh -i -c exit' --warmup 3

# ─── tmux のメモリ使用量監視 ───
tmux list-sessions -F '#{session_name}: #{session_windows} windows, #{session_attached} attached'
# 不要なセッションは定期的に削除

# ─── ディスク使用量チェック ───
# zinit プラグインのサイズ
du -sh ~/.local/share/zinit/plugins/* | sort -rh | head -10
# tmux プラグインのサイズ
du -sh ~/.tmux/plugins/* | sort -rh
```

---

## 9. FAQ

### Q1: zsh と fish、どちらを選ぶべき？

**A:** POSIX 互換性が必要な場合は zsh。既存のシェルスクリプトをそのまま使えるのは大きな利点。一方、設定なしで最初から快適に使いたいなら fish がおすすめ。fish の自動補完とシンタックスハイライトは設定不要で動作する。ただし bash 向けスクリプトの互換性は低い。チーム開発でシェルスクリプトを共有する場合は zsh が無難。個人の作業効率を最優先するなら fish は非常に快適。なお、fish は POSIX 非互換のため、`&&` の代わりに `; and` を使う等の構文の違いがある（fish 3.0 以降は `&&` もサポート）。

### Q2: Nerd Font は本当に必要？

**A:** Starship やモダン CLI ツール（eza 等）でアイコン表示を使うなら必須。以下でインストールする。

```bash
brew install --cask font-jetbrains-mono-nerd-font
```

Nerd Font がないと豆腐文字（□）が表示される。ターミナルのフォント設定で "JetBrains Mono Nerd Font" を選択すること。代替として "FiraCode Nerd Font" や "Hack Nerd Font" も人気がある。VS Code のターミナルでも Nerd Font の設定が必要：`"terminal.integrated.fontFamily": "JetBrains Mono Nerd Font"` を settings.json に追加する。

### Q3: macOS で iTerm2 と Warp、どちらがよい？

**A:** 安定性と実績を重視するなら iTerm2。AI 補完やモダン UI を求めるなら Warp。Warp は AI によるコマンド候補表示が強力だが、Rust 製で拡張性は iTerm2 に劣る。チーム標準にするなら iTerm2 の方が無難。ただし、Warp の Blocks 機能（コマンドと出力をブロック単位で管理）は非常に便利で、長い出力の中から特定のコマンド結果を素早く見つけられる。最近は Alacritty + tmux の組み合わせも人気が高まっている。GPU アクセラレーションで描画が高速で、tmux のキーバインドに統一できるメリットがある。

### Q4: tmux と iTerm2 のペイン分割、どちらを使うべき？

**A:** SSH でリモートサーバーを使う頻度が高いなら tmux 一択。tmux はサーバー側で動作するため、SSH 切断後もセッションが維持される。ローカル開発のみなら iTerm2 のペイン分割でも十分。ただし、tmux に慣れると環境を問わず同じ操作感で使えるため、長期的には tmux の習得を推奨する。iTerm2 と tmux を併用する場合は、iTerm2 の tmux integration モード（`tmux -CC`）も検討する価値がある。

### Q5: シェルの起動時間はどれくらいが適切？

**A:** 目安は 200ms 以下。500ms を超えると体感的にストレスを感じ始める。`time zsh -i -c exit` で計測し、200ms を超えている場合は以下を順に試す。1) nvm を fnm に置き換える（最も効果大）、2) compinit のキャッシュを有効化、3) Oh My Zsh を zinit に移行、4) 不要なプラグインの削除。`hyperfine 'zsh -i -c exit'` でより正確な計測ができる。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 10. まとめ

| 項目 | macOS 推奨 | Windows 推奨 | Linux 推奨 |
|------|-----------|-------------|-----------|
| ターミナル | iTerm2 / Warp | Windows Terminal | Alacritty / Kitty |
| シェル | zsh | PowerShell 7 | zsh / fish |
| プロンプト | Starship | Starship | Starship |
| マルチプレクサ | tmux | tmux (WSL) | tmux |
| フォント | JetBrains Mono NF | JetBrains Mono NF | JetBrains Mono NF |
| ファジーファインダー | fzf | fzf | fzf |
| ディレクトリジャンプ | zoxide | zoxide | zoxide |
| プラグイン管理 (zsh) | zinit | - | zinit |
| dotfiles 管理 | GNU Stow + Git | GNU Stow + Git | GNU Stow + Git |
| cat 代替 | bat | bat | bat |
| ls 代替 | eza | eza | eza |
| grep 代替 | ripgrep | ripgrep | ripgrep |
| find 代替 | fd | fd | fd |

---

## 次に読むべきガイド

- [00-vscode-setup.md](./00-vscode-setup.md) -- VS Code との統合
- [02-git-config.md](./02-git-config.md) -- Git の詳細設定（diff/merge ツール連携）
- [../01-runtime-and-package/00-version-managers.md](../01-runtime-and-package/00-version-managers.md) -- ランタイムバージョン管理

---

## 参考文献

1. **iTerm2 Documentation** -- https://iterm2.com/documentation.html -- iTerm2 の全機能解説。Shell Integration の設定方法も記載。
2. **Starship: Cross-Shell Prompt** -- https://starship.rs/config/ -- Starship の設定リファレンス。全モジュールの詳細設定。
3. **tmux 2: Productive Mouse-Free Development** (Brian P. Hogan) -- https://pragprog.com/titles/bhtmux2/ -- tmux のバイブル的書籍。
4. **Modern Unix** -- https://github.com/ibraheemdev/modern-unix -- モダン CLI ツールのキュレーションリスト。
5. **zinit Documentation** -- https://zdharma-continuum.github.io/zinit/wiki/ -- zinit の公式ドキュメント。遅延読み込みの詳細設定。
6. **Alacritty Configuration** -- https://alacritty.org/config-alacritty.html -- Alacritty の設定リファレンス。TOML 形式。
7. **fish shell Documentation** -- https://fishshell.com/docs/current/ -- fish シェルの公式ドキュメント。独自構文の解説。
8. **GNU Stow** -- https://www.gnu.org/software/stow/ -- シンボリックリンク管理ツールの公式ドキュメント。
9. **fzf Examples** -- https://github.com/junegunn/fzf/wiki/Examples -- fzf のカスタム関数・統合例の大規模コレクション。
10. **Catppuccin** -- https://catppuccin.com/ -- 人気カラースキーム。全ツール向けテーマ提供。
