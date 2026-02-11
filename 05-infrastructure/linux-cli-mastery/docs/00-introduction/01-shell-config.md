# シェル設定

> シェルの設定をカスタマイズすることで、生産性は劇的に向上する。

## この章で学ぶこと

- [ ] .bashrc / .zshrc の役割を理解する
- [ ] エイリアスと環境変数を設定できる
- [ ] プロンプトのカスタマイズ方法を知る

---

## 1. 設定ファイルの読み込み順序

```
bash の場合:
  ログインシェル:     /etc/profile → ~/.bash_profile → ~/.bashrc
  非ログインシェル:   ~/.bashrc

zsh の場合:
  全て:     ~/.zshenv（常に読まれる）
  ログイン: ~/.zprofile → ~/.zshrc → ~/.zlogin
  非ログイン: ~/.zshrc

  実務的には:
  → 環境変数: ~/.zshenv または ~/.bash_profile
  → エイリアス、関数: ~/.zshrc または ~/.bashrc
  → PATH設定: ~/.zshenv（zsh）または ~/.bash_profile（bash）
```

---

## 2. 環境変数とエイリアス

```bash
# 環境変数
export EDITOR="vim"              # デフォルトエディタ
export LANG="ja_JP.UTF-8"       # ロケール
export PATH="$HOME/.local/bin:$PATH"  # PATHに追加

# エイリアス（コマンドの別名）
alias ll='ls -lah'
alias la='ls -A'
alias ..='cd ..'
alias ...='cd ../..'
alias g='git'
alias gs='git status'
alias gc='git commit'
alias gp='git push'
alias dc='docker compose'
alias k='kubectl'
alias py='python3'
alias myip='curl -s ifconfig.me'

# 危険なコマンドの安全化
alias rm='rm -i'               # 確認付き削除
alias cp='cp -i'               # 確認付きコピー
alias mv='mv -i'               # 確認付き移動

# 関数（複雑なエイリアス）
mkcd() {
    mkdir -p "$1" && cd "$1"
}

# 特定のディレクトリで特定のNode.jsバージョンを使う等
```

---

## 3. プロンプトのカスタマイズ

```bash
# bash のプロンプト変数 PS1
# \u: ユーザー名, \h: ホスト名, \w: カレントディレクトリ
export PS1='\u@\h:\w\$ '

# Git ブランチ表示付き（bash）
parse_git_branch() {
    git branch 2>/dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/ (\1)/'
}
export PS1='\u:\w$(parse_git_branch)\$ '

# zsh: Starship（モダンなプロンプト）
# brew install starship
# eval "$(starship init zsh)"  # .zshrc に追加

# Oh My Zsh（zsh のフレームワーク）
# sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# テーマ、プラグイン、補完が充実
```

---

## 4. 便利な設定

```bash
# ~/.zshrc に追加推奨

# 履歴設定
HISTSIZE=100000
SAVEHIST=100000
HISTFILE=~/.zsh_history
setopt SHARE_HISTORY          # 複数ターミナルで履歴共有
setopt HIST_IGNORE_DUPS       # 重複を無視
setopt HIST_IGNORE_SPACE      # スペース始まりを記録しない

# 補完設定
autoload -Uz compinit && compinit
zstyle ':completion:*' matcher-list 'm:{a-z}={A-Z}'  # 大小文字無視

# ディレクトリ移動
setopt AUTO_CD                # ディレクトリ名だけで cd
setopt AUTO_PUSHD             # cd で自動pushd
setopt PUSHD_IGNORE_DUPS

# 便利なツール
# brew install fzf             # ファジー検索
# brew install zoxide          # スマートcd（z コマンド）
# brew install bat             # cat の改良版
# brew install eza             # ls の改良版
# brew install ripgrep         # grep の高速版
# brew install fd              # find の高速版
```

---

## まとめ

| 設定 | ファイル | 用途 |
|------|---------|------|
| 環境変数 | .zshenv / .bash_profile | PATH, EDITOR等 |
| エイリアス | .zshrc / .bashrc | コマンドの短縮 |
| プロンプト | .zshrc / .bashrc | 表示のカスタマイズ |
| 履歴 | .zshrc / .bashrc | 履歴の保存・共有設定 |

---

## 次に読むべきガイド
→ [[02-man-and-help.md]] — マニュアルとヘルプ

---

## 参考文献
1. Robbins, A. "bash Pocket Reference." 2nd Ed, O'Reilly, 2016.
