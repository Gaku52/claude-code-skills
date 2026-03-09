# ターミナルマルチプレクサ（tmux, screen）

> tmux は1つのターミナルで複数のセッション・ウィンドウ・ペインを管理する。SSH切断後も作業が継続する。

## この章で学ぶこと

- [ ] tmux のセッション・ウィンドウ・ペインを操作できる
- [ ] SSH 切断後もプロセスを継続できる
- [ ] tmux をカスタマイズして生産性を上げる
- [ ] tmux スクリプトで作業環境を自動構築できる
- [ ] tmux プラグインを活用できる
- [ ] screen の基本操作を理解する（レガシー環境対応）


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解

---

## 1. tmux の基本概念

### 1.1 構造の理解

```
tmux の構造:

  Server（バックグラウンドプロセス）
  └── Session（作業の単位。SSH切断後も維持）
      ├── Window 0（タブのようなもの）
      │   ├── Pane 0（画面分割の各領域）
      │   └── Pane 1
      └── Window 1
          └── Pane 0

プレフィックスキー: Ctrl+b（デフォルト）
  → 全ての tmux コマンドは Ctrl+b の後にキーを押す
```

### 1.2 tmux が必要な場面

```bash
# tmux が必要な場面:
# 1. SSH接続でサーバー作業 → 切断してもプロセスが継続
# 2. 複数のターミナルを1画面で管理 → 画面分割
# 3. ペアプログラミング → セッション共有
# 4. 長時間実行するジョブの管理 → デタッチ/アタッチ
# 5. 開発環境の一括構築 → スクリプト化

# tmux のインストール
# macOS
brew install tmux

# Ubuntu/Debian
sudo apt install tmux

# RHEL/Fedora
sudo dnf install tmux

# バージョン確認
tmux -V
```

---

## 2. セッション管理

### 2.1 セッションの基本操作

```bash
# セッション操作
tmux                             # 新規セッション作成
tmux new -s work                 # 名前付きセッション
tmux new -s work -d              # バックグラウンドで作成
tmux new -s work -n editor       # 最初のウィンドウ名を指定
tmux ls                          # セッション一覧
tmux list-sessions               # 同上（フルコマンド）
tmux attach -t work              # セッションにアタッチ
tmux attach -t 0                 # 番号でアタッチ
tmux attach                      # 最後のセッションにアタッチ
tmux kill-session -t work        # セッション削除
tmux kill-session -a             # 現在以外の全セッション削除
tmux kill-session -a -t work     # work以外の全セッション削除
tmux kill-server                 # 全セッション削除

# セッションの存在確認
tmux has-session -t work 2>/dev/null && echo "exists" || echo "not found"
```

### 2.2 セッション内のキーバインド

```bash
# セッション内操作（Ctrl+b + キー）
# Ctrl+b d    → デタッチ（セッションから離脱。プロセスは継続）
# Ctrl+b s    → セッション一覧・切り替え（ツリー表示）
# Ctrl+b $    → セッション名変更
# Ctrl+b (    → 前のセッション
# Ctrl+b )    → 次のセッション
# Ctrl+b L    → 最後にアクティブだったセッションに切り替え
```

### 2.3 セッション管理のベストプラクティス

```bash
# プロジェクトごとにセッションを作成
tmux new -s frontend -d
tmux new -s backend -d
tmux new -s database -d

# セッション間の移動
# Ctrl+b s でセッション一覧を表示してから選択
# または Ctrl+b ( / ) で順次切り替え

# SSH先でのセッション管理パターン
# 接続時: 既存セッションがあればアタッチ、なければ新規作成
tmux attach -t main 2>/dev/null || tmux new -s main

# エイリアスとして設定
alias ta='tmux attach -t main 2>/dev/null || tmux new -s main'
```

---

## 3. ウィンドウ操作

### 3.1 基本操作

```bash
# ウィンドウ（タブ相当）
# Ctrl+b c    → 新規ウィンドウ作成
# Ctrl+b ,    → ウィンドウ名変更
# Ctrl+b w    → ウィンドウ一覧（プレビュー付き）
# Ctrl+b n    → 次のウィンドウ
# Ctrl+b p    → 前のウィンドウ
# Ctrl+b 0-9  → 番号でウィンドウ切り替え
# Ctrl+b &    → ウィンドウを閉じる（確認あり）
# Ctrl+b f    → ウィンドウ検索
# Ctrl+b l    → 最後にアクティブだったウィンドウに切り替え
```

### 3.2 コマンドラインからのウィンドウ操作

```bash
# コマンドラインからの操作
tmux new-window                  # 新しいウィンドウ
tmux new-window -n logs          # 名前付きウィンドウ
tmux new-window -t work:         # 特定セッションにウィンドウ追加
tmux select-window -t 2          # ウィンドウ2に移動
tmux select-window -t work:logs  # セッション:ウィンドウ名で指定
tmux rename-window editor        # ウィンドウ名変更

# ウィンドウの入れ替え
tmux swap-window -s 0 -t 1       # ウィンドウ0と1を入れ替え
tmux move-window -s work:1 -t dev:  # セッション間でウィンドウ移動

# ウィンドウでコマンドを実行
tmux new-window -n editor "vim ."
tmux new-window -n server "npm run dev"
```

### 3.3 ウィンドウのレイアウト管理

```bash
# ステータスバーでのウィンドウ表示
# [0] editor* [1] server [2] logs
# * が付いているのが現在のウィンドウ
# - が付いているのが直前のウィンドウ

# ウィンドウの自動リネーム
# デフォルトでは実行中のコマンド名がウィンドウ名になる
# 無効にする場合:
# set-option -g allow-rename off
```

---

## 4. ペイン操作（画面分割）

### 4.1 基本的なペイン操作

```bash
# ペインの分割
# Ctrl+b %    → 左右に分割（垂直分割）
# Ctrl+b "    → 上下に分割（水平分割）

# ペインの移動
# Ctrl+b ←↑→↓  → 矢印キーでペイン移動
# Ctrl+b o      → 次のペインへ
# Ctrl+b ;      → 直前のペインへ
# Ctrl+b q      → ペイン番号表示（番号を押して移動）

# ペインのサイズ変更
# Ctrl+b Ctrl+←↑→↓  → 矢印方向にリサイズ
# Ctrl+b z            → ペインをズーム（全画面切替）

# ペインのレイアウト
# Ctrl+b Space        → レイアウト切り替え（均等分割等）
# Ctrl+b {            → ペインを前に移動
# Ctrl+b }            → ペインを後ろに移動

# ペインを閉じる
# Ctrl+b x            → 現在のペインを閉じる（確認あり）
# exit または Ctrl+d   → シェルを終了してペインを閉じる

# ペインをウィンドウに昇格
# Ctrl+b !            → 現在のペインを新しいウィンドウに
```

### 4.2 コマンドラインからのペイン操作

```bash
# コマンドラインでの分割
tmux split-window -h             # 水平（左右）分割
tmux split-window -v             # 垂直（上下）分割
tmux split-window -h -p 30       # 右側30%で分割
tmux split-window -v -p 20       # 下側20%で分割
tmux split-window -h -l 40       # 右側40カラムで分割

# 分割してコマンド実行
tmux split-window -h "tail -f /var/log/syslog"
tmux split-window -v -p 30 "htop"

# ペインの選択
tmux select-pane -t 0            # ペイン0を選択
tmux select-pane -L              # 左のペインに移動
tmux select-pane -R              # 右のペインに移動
tmux select-pane -U              # 上のペインに移動
tmux select-pane -D              # 下のペインに移動

# ペインのリサイズ
tmux resize-pane -L 5            # 左に5カラム
tmux resize-pane -R 5            # 右に5カラム
tmux resize-pane -U 5            # 上に5行
tmux resize-pane -D 5            # 下に5行
tmux resize-pane -Z              # ズームトグル

# ペインの入れ替え
tmux swap-pane -s 0 -t 1         # ペイン0と1を入れ替え
tmux swap-pane -U                # 上のペインと入れ替え
tmux swap-pane -D                # 下のペインと入れ替え

# ペインをウィンドウ間で移動
tmux join-pane -s work:1 -t work:0   # ウィンドウ1のペインをウィンドウ0に結合
tmux break-pane                       # 現在のペインを新しいウィンドウに

# レイアウトの指定
tmux select-layout even-horizontal   # 均等水平分割
tmux select-layout even-vertical     # 均等垂直分割
tmux select-layout main-horizontal   # メイン（上）+ サブ（下段横並び）
tmux select-layout main-vertical     # メイン（左）+ サブ（右段縦並び）
tmux select-layout tiled             # タイル状（均等グリッド）
```

### 4.3 ペインの同期（全ペインに同時入力）

```bash
# 全ペインへの同時入力（同じコマンドを複数サーバーで実行）
# Ctrl+b : → setw synchronize-panes on
# Ctrl+b : → setw synchronize-panes off

# トグルで切り替え
# .tmux.conf に以下を追加:
# bind S setw synchronize-panes

# 使い方:
# 1. 複数ペインでそれぞれSSH接続
# 2. Ctrl+b S で同期ON
# 3. コマンド入力（全ペインに反映）
# 4. Ctrl+b S で同期OFF
```

---

## 5. コピーモード

### 5.1 基本操作

```bash
# コピーモード（スクロール・テキスト選択）
# Ctrl+b [    → コピーモード開始

# コピーモード内の操作（vi風）:
# q           → コピーモード終了
# ↑↓←→ / hjkl → カーソル移動
# Ctrl+u/d    → ページアップ/ダウン
# Ctrl+b/f    → ページアップ/ダウン（emacs風）
# g / G       → 先頭/末尾
# /pattern    → 前方検索
# ?pattern    → 後方検索
# n / N       → 次/前の検索結果
# Space       → 選択開始
# Enter       → コピー（選択終了）
# w / b       → 単語単位で移動
# 0 / $       → 行頭 / 行末

# Ctrl+b ]    → ペースト
```

### 5.2 vi モードの設定と高度な操作

```bash
# vi モードを有効にする（~/.tmux.conf）:
setw -g mode-keys vi

# vi モードでのコピー操作
# Ctrl+b [     → コピーモード開始
# v            → 選択開始（vi風、設定が必要）
# y            → ヤンク（コピー）
# Ctrl+b ]     → ペースト

# tmux.conf に追加する設定（vi風コピー）
# bind-key -T copy-mode-vi v send-keys -X begin-selection
# bind-key -T copy-mode-vi y send-keys -X copy-selection-and-cancel
# bind-key -T copy-mode-vi r send-keys -X rectangle-toggle

# システムクリップボードとの連携（macOS）
# bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "pbcopy"
# bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "pbcopy"

# システムクリップボードとの連携（Linux / X11）
# bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "xclip -selection clipboard"

# システムクリップボードとの連携（Linux / Wayland）
# bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "wl-copy"
```

### 5.3 マウスによるコピー

```bash
# マウスモードを有効にすると:
# - マウスでペインを選択
# - マウスでペインをリサイズ
# - マウスドラッグでテキスト選択
# - マウスホイールでスクロール

# set -g mouse on  # ~/.tmux.conf に追加

# マウスで選択したテキストのコピー設定
# macOS + iTerm2 の場合:
# Option キーを押しながらドラッグで従来の選択

# tmux 内でのマウスコピーの改善設定
# bind-key -T copy-mode-vi MouseDragEnd1Pane send-keys -X copy-pipe-and-cancel "pbcopy"
```

---

## 6. tmux の設定（~/.tmux.conf）

### 6.1 基本設定

```bash
# ~/.tmux.conf

# プレフィックスキーの変更（Ctrl+a が人気）
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# マウスサポート
set -g mouse on

# vi風キーバインド
setw -g mode-keys vi

# ペイン分割のキーバインド改善
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# ペイン移動（vim風）
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# ペインリサイズ
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# ウィンドウ番号を1から開始
set -g base-index 1
setw -g pane-base-index 1

# ウィンドウ番号の自動リナンバリング
set -g renumber-windows on

# 256色対応
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"

# ステータスバー
set -g status-style 'bg=#333333 fg=#ffffff'
set -g status-left '#[fg=green]#S '
set -g status-right '#[fg=yellow]%Y-%m-%d %H:%M'
set -g status-left-length 30

# 設定の再読み込み
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# 履歴バッファサイズ
set -g history-limit 50000

# エスケープ時間の短縮（vim用）
set -sg escape-time 0

# キーリピート時間
set -g repeat-time 500

# 新しいウィンドウは現在のパスで開く
bind c new-window -c "#{pane_current_path}"

# ウィンドウのアクティビティ通知
setw -g monitor-activity on
set -g visual-activity off
```

### 6.2 外観のカスタマイズ

```bash
# ステータスバーの詳細カスタマイズ
set -g status-position bottom
set -g status-justify left
set -g status-interval 5

# 左側: セッション名
set -g status-left '#[fg=green,bold]#S #[fg=white]| '
set -g status-left-length 30

# 右側: 日時、ホスト名
set -g status-right '#[fg=cyan]#H #[fg=white]| #[fg=yellow]%Y-%m-%d #[fg=white]%H:%M '
set -g status-right-length 50

# ウィンドウリストのスタイル
setw -g window-status-format '#[fg=white] #I:#W '
setw -g window-status-current-format '#[fg=black,bg=green,bold] #I:#W '

# ペインボーダーのスタイル
set -g pane-border-style 'fg=#444444'
set -g pane-active-border-style 'fg=green'

# メッセージのスタイル
set -g message-style 'fg=white bg=black bold'

# コピーモードのスタイル
setw -g mode-style 'fg=black bg=yellow'

# クロックモードの色
setw -g clock-mode-colour green
```

### 6.3 高度なキーバインド設定

```bash
# Alt + 矢印キーでペイン移動（プレフィックス不要）
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# Shift + 矢印キーでウィンドウ切り替え（プレフィックス不要）
bind -n S-Left previous-window
bind -n S-Right next-window

# ペインの同期トグル
bind S setw synchronize-panes

# ペインの結合と分離
bind j join-pane -s !           # 直前のウィンドウのペインを結合
bind J break-pane               # ペインを新しいウィンドウに分離

# ウィンドウの入れ替え
bind -r < swap-window -t -1\; select-window -t -1
bind -r > swap-window -t +1\; select-window -t +1

# コマンドプロンプト
# Ctrl+b :    → tmux コマンドを直接入力
# 例: :new-window -n logs "tail -f /var/log/syslog"
# 例: :setw synchronize-panes on
# 例: :resize-pane -D 10

# セッション内でのウィンドウ検索
# Ctrl+b f    → ウィンドウ名で検索
```

---

## 7. tmux プラグインマネージャ（TPM）

### 7.1 TPM のインストールと使い方

```bash
# TPM（Tmux Plugin Manager）のインストール
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# ~/.tmux.conf に追加
# プラグインリスト
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'

# TPMの初期化（.tmux.conf の最後に配置）
run '~/.tmux/plugins/tpm/tpm'

# プラグインのインストール
# Ctrl+b I     → プラグインをインストール
# Ctrl+b U     → プラグインを更新
# Ctrl+b alt+u → プラグインを削除（リストから削除後に実行）
```

### 7.2 おすすめプラグイン

```bash
# tmux-resurrect: セッションの保存・復元
set -g @plugin 'tmux-plugins/tmux-resurrect'
# Ctrl+b Ctrl+s → セッションを保存
# Ctrl+b Ctrl+r → セッションを復元

# tmux-continuum: 自動保存・自動復元
set -g @plugin 'tmux-plugins/tmux-continuum'
set -g @continuum-restore 'on'
set -g @continuum-save-interval '15'  # 15分ごとに自動保存

# tmux-yank: システムクリップボードとの連携
set -g @plugin 'tmux-plugins/tmux-yank'

# tmux-open: コピーモードでURLやファイルを開く
set -g @plugin 'tmux-plugins/tmux-open'
# コピーモードで選択後:
# o → デフォルトプログラムで開く
# Ctrl+o → エディタで開く
# S → 検索エンジンで検索

# tmux-fzf: fzf でセッション/ウィンドウ/ペインを選択
set -g @plugin 'sainnhe/tmux-fzf'
# Ctrl+b F → fzf メニュー

# tmux-fingers: 画面上のURLやパスを選択してコピー
set -g @plugin 'Morantron/tmux-fingers'
# Ctrl+b F → ハイライトモード

# dracula テーマ
set -g @plugin 'dracula/tmux'
set -g @dracula-plugins "cpu-usage ram-usage time"
set -g @dracula-show-left-icon session

# catppuccin テーマ
set -g @plugin 'catppuccin/tmux'
set -g @catppuccin_flavour 'mocha'
```

### 7.3 完全な .tmux.conf の例

```bash
# ~/.tmux.conf - 完全な設定例

# === 基本設定 ===
set -g prefix C-a
unbind C-b
bind C-a send-prefix

set -g mouse on
setw -g mode-keys vi
set -g base-index 1
setw -g pane-base-index 1
set -g renumber-windows on
set -g history-limit 50000
set -sg escape-time 0
set -g repeat-time 500
set -g focus-events on
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"

# === キーバインド ===
# ペイン分割
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# ペイン移動（vim風）
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# ペインリサイズ
bind -r H resize-pane -L 5
bind -r J resize-pane -D 5
bind -r K resize-pane -U 5
bind -r L resize-pane -R 5

# Alt + 矢印でペイン移動
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# Shift + 矢印でウィンドウ切り替え
bind -n S-Left previous-window
bind -n S-Right next-window

# 新しいウィンドウは現在のパスで開く
bind c new-window -c "#{pane_current_path}"

# 設定再読み込み
bind r source-file ~/.tmux.conf \; display "Reloaded!"

# ペイン同期トグル
bind S setw synchronize-panes

# ウィンドウ入れ替え
bind -r < swap-window -t -1\; select-window -t -1
bind -r > swap-window -t +1\; select-window -t +1

# === コピーモード ===
bind-key -T copy-mode-vi v send-keys -X begin-selection
bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "pbcopy"
bind-key -T copy-mode-vi r send-keys -X rectangle-toggle

# === 外観 ===
set -g status-position bottom
set -g status-style 'bg=#1e1e2e fg=#cdd6f4'
set -g status-left '#[fg=#a6e3a1,bold] #S #[fg=#cdd6f4]| '
set -g status-left-length 30
set -g status-right '#[fg=#89b4fa]#H #[fg=#cdd6f4]| #[fg=#f9e2af]%Y-%m-%d %H:%M '
set -g status-right-length 50
setw -g window-status-format '#[fg=#6c7086] #I:#W '
setw -g window-status-current-format '#[fg=#1e1e2e,bg=#a6e3a1,bold] #I:#W '
set -g pane-border-style 'fg=#313244'
set -g pane-active-border-style 'fg=#a6e3a1'
set -g message-style 'fg=#cdd6f4 bg=#1e1e2e bold'

# === プラグイン ===
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-sensible'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'
set -g @plugin 'tmux-plugins/tmux-yank'

set -g @continuum-restore 'on'
set -g @continuum-save-interval '15'

# TPM初期化（最後に配置）
run '~/.tmux/plugins/tpm/tpm'
```

---

## 8. tmux の実践パターン

### 8.1 開発用レイアウト

```bash
# パターン1: 開発用レイアウト
tmux new -s dev
# ペイン分割: エディタ（上大） + ターミナル（下左） + ログ（下右）
# Ctrl+b "    → 上下分割
# 下ペインで Ctrl+b % → 左右分割

# 手動での操作手順:
# 1. tmux new -s dev
# 2. Ctrl+b " (上下分割)
# 3. Ctrl+b ↓ (下ペインに移動)
# 4. Ctrl+b % (左右分割)
# 5. Ctrl+b ↑ (上ペインに移動)
# 6. vim .  (エディタを開く)
```

### 8.2 スクリプトでレイアウト自動構築

```bash
#!/bin/bash
# dev-session.sh - 開発用セッションの自動構築

SESSION="dev"
PROJECT_DIR="${1:-$(pwd)}"

# 既存セッションがあればアタッチ
tmux has-session -t "$SESSION" 2>/dev/null && {
    tmux attach -t "$SESSION"
    exit 0
}

# 新規セッション作成
tmux new-session -d -s "$SESSION" -n "editor" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:editor" "vim ." Enter

# サーバーウィンドウ
tmux new-window -t "$SESSION" -n "server" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:server" "npm run dev" Enter

# ログウィンドウ
tmux new-window -t "$SESSION" -n "logs" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:logs" "tail -f /var/log/app.log" Enter

# ターミナルウィンドウ（git操作等）
tmux new-window -t "$SESSION" -n "terminal" -c "$PROJECT_DIR"
tmux send-keys -t "$SESSION:terminal" "git status" Enter

# 最初のウィンドウを選択
tmux select-window -t "$SESSION:editor"

# アタッチ
tmux attach -t "$SESSION"
```

### 8.3 分割レイアウト付きセッション

```bash
#!/bin/bash
# monitor-session.sh - サーバー監視用セッション

SESSION="monitor"

tmux new-session -d -s "$SESSION" -n "dashboard"

# メインペイン（上半分）: htop
tmux send-keys -t "$SESSION:dashboard" "htop" Enter

# 下半分を左右に分割
tmux split-window -v -p 40 -t "$SESSION:dashboard"
tmux send-keys "watch -n 5 'df -h'" Enter

tmux split-window -h -t "$SESSION:dashboard"
tmux send-keys "watch -n 5 'free -h'" Enter

# ネットワーク監視ウィンドウ
tmux new-window -t "$SESSION" -n "network"
tmux send-keys -t "$SESSION:network" "sudo iftop" Enter

# ログ監視ウィンドウ
tmux new-window -t "$SESSION" -n "logs"
tmux split-window -h -t "$SESSION:logs"
tmux send-keys -t "$SESSION:logs.0" "journalctl -u nginx -f" Enter
tmux send-keys -t "$SESSION:logs.1" "journalctl -u postgresql -f" Enter

# ダッシュボードに戻る
tmux select-window -t "$SESSION:dashboard"
tmux select-pane -t 0

tmux attach -t "$SESSION"
```

### 8.4 SSH先での長時間ジョブ

```bash
# パターン3: SSH先での長時間ジョブ
ssh server
tmux new -s backup
./run_backup.sh
# Ctrl+b d でデタッチ → SSH切断しても安全
# 後日: ssh server → tmux attach -t backup

# 複数サーバーへの同時接続
#!/bin/bash
# multi-server.sh

SESSION="servers"
SERVERS=("web1" "web2" "web3" "db1")

tmux new-session -d -s "$SESSION"

for i in "${!SERVERS[@]}"; do
    server="${SERVERS[$i]}"
    if [[ $i -eq 0 ]]; then
        tmux rename-window -t "$SESSION:0" "$server"
    else
        tmux new-window -t "$SESSION" -n "$server"
    fi
    tmux send-keys -t "$SESSION:$server" "ssh $server" Enter
done

tmux select-window -t "$SESSION:0"
tmux attach -t "$SESSION"
```

### 8.5 ペアプログラミング

```bash
# パターン4: ペアプログラミング
# ユーザーA（セッション作成者）:
tmux new -s pair

# ユーザーB（参加者）:
tmux attach -t pair
# 同じセッションを共有して画面を見ながら作業

# 読み取り専用で参加する場合:
tmux attach -t pair -r

# 別々のウィンドウサイズで共有する場合:
# ユーザーA:
tmux new -s pair
# ユーザーB:
tmux new -s pair-b -t pair
# これにより各ユーザーが独立したウィンドウサイズを持てる
```

### 8.6 tmux コマンドのスクリプティング

```bash
# tmux にコマンドを送信
tmux send-keys -t dev:editor "echo hello" Enter

# 現在のセッション情報を取得
tmux display-message -p '#S'          # セッション名
tmux display-message -p '#W'          # ウィンドウ名
tmux display-message -p '#P'          # ペイン番号
tmux display-message -p '#{pane_current_path}'  # 現在のパス

# ペインの内容をキャプチャ
tmux capture-pane -t 0 -p             # ペイン0の内容を表示
tmux capture-pane -t 0 -p -S -100     # 過去100行分

# ペインの内容をファイルに保存
tmux capture-pane -t 0 -p -S -1000 > /tmp/pane-output.txt

# 条件付きのコマンド実行
if tmux has-session -t dev 2>/dev/null; then
    tmux send-keys -t dev:server "npm restart" Enter
fi
```

---

## 9. tmux のトラブルシューティング

```bash
# === 問題: 256色が表示されない ===
# .tmux.conf に追加:
# set -g default-terminal "tmux-256color"
# set -ag terminal-overrides ",xterm-256color:RGB"
# ターミナルエミュレータの設定も確認

# === 問題: コピーモードでシステムクリップボードに入らない ===
# macOS: reattach-to-user-namespace が必要（古いtmux）
# brew install reattach-to-user-namespace
# 新しい tmux (2.6+) では不要、tmux-yank プラグインを使用

# === 問題: Neovim/Vim で色がおかしい ===
# .tmux.conf:
# set -g default-terminal "tmux-256color"
# set -ag terminal-overrides ",xterm-256color:Tc"
# .vimrc:
# set termguicolors

# === 問題: マウスモードで選択できない ===
# マウスモード有効時は Shift を押しながらドラッグ
# iTerm2: Option を押しながらドラッグ

# === 問題: tmux が起動しない ===
tmux kill-server                 # サーバーを強制終了
rm -f /tmp/tmux-*/default        # ソケットファイルを削除
tmux

# === 問題: 設定が反映されない ===
tmux source-file ~/.tmux.conf    # 設定を再読み込み
# または
# Ctrl+b : → source-file ~/.tmux.conf

# === デバッグ ===
tmux show-options -g             # グローバルオプション一覧
tmux show-options -w             # ウィンドウオプション一覧
tmux list-keys                   # 全キーバインド一覧
tmux list-commands               # 全コマンド一覧
tmux info                        # tmux の詳細情報
```

---

## 10. screen（レガシー環境用）

### 10.1 基本操作

```bash
# screen は tmux の前身。最低限の操作だけ覚えておく

screen                           # 新規セッション
screen -S work                   # 名前付き
screen -ls                       # セッション一覧
screen -r work                   # リアタッチ
screen -d -r work                # デタッチしてからリアタッチ
screen -x work                   # マルチアタッチ（複数ユーザーで共有）
screen -X quit                   # セッション終了

# screen 内操作（Ctrl+a がプレフィックス）
# Ctrl+a d    → デタッチ
# Ctrl+a c    → 新ウィンドウ
# Ctrl+a n    → 次のウィンドウ
# Ctrl+a p    → 前のウィンドウ
# Ctrl+a "    → ウィンドウ一覧
# Ctrl+a A    → ウィンドウ名変更
# Ctrl+a 0-9  → ウィンドウ番号で切り替え
# Ctrl+a |    → 垂直分割
# Ctrl+a S    → 水平分割
# Ctrl+a Tab  → ペイン切り替え
# Ctrl+a X    → 現在のペインを閉じる
# Ctrl+a k    → ウィンドウ閉じる
# Ctrl+a [    → コピーモード
# Ctrl+a ]    → ペースト
# Ctrl+a ?    → ヘルプ
```

### 10.2 screen の設定（~/.screenrc）

```bash
# ~/.screenrc

# スクロールバッファ
defscrollback 10000

# ステータスバーの設定
hardstatus alwayslastline
hardstatus string '%{= kG}[ %{G}%H %{g}][%{= kw}%?%-Lw%?%{r}(%{W}%n*%f%t%?(%u)%?%{r})%{w}%?%+Lw%?%?%= %{g}][%{B} %Y-%m-%d %{W}%c %{g}]'

# ビジュアルベル
vbell on

# エンコーディング
defencoding utf-8
encoding utf-8

# 起動メッセージを表示しない
startup_message off
```

### 10.3 screen より tmux を使うべき理由

```bash
# screen より tmux を使うべき理由:
# - ペイン操作が直感的
# - 設定が簡単で読みやすい
# - アクティブに開発されている
# - プラグインエコシステム（TPM）がある
# - ステータスバーのカスタマイズが容易
# - セッション管理が柔軟
# - コピーモードが強力
# - スクリプティングが容易

# screen が必要な場面:
# - tmux がインストールされていない古いサーバー
# - シリアルコンソール接続（screen /dev/ttyUSB0 115200）
# - 最小限の機能で十分な場合
```

---

## 11. tmux の代替ツール

```bash
# === Zellij ===
# Rust製のモダンなターミナルマルチプレクサ
# https://zellij.dev/
# brew install zellij
# 特徴:
# - デフォルトで直感的なUI
# - 画面下部にキーバインドのヒントが表示
# - WebAssembly プラグインシステム
# - レイアウトファイルによる設定

# === byobu ===
# screen/tmux のラッパー
# sudo apt install byobu
# 特徴:
# - ファンクションキーで操作
# - 自動的にtmuxまたはscreenをバックエンドとして使用
# - ステータスバーにシステム情報を自動表示

# === Wezterm ===
# ターミナルエミュレータ自体にマルチプレクサ機能がある
# https://wezfurlong.org/wezterm/
# - GPU アクセラレーション
# - Lua で設定
# - マルチプレクサ機能内蔵
# - SSH統合

# === kitty ===
# GPU アクセラレーションターミナル
# https://sw.kovidgoyal.net/kitty/
# - タブとウィンドウ分割機能
# - tmux なしでも画面分割が可能
# - 高速なレンダリング
```

---

## 12. tmux Hooks とイベント駆動

### 12.1 Hook の基本

```bash
# tmux hooks はイベント発生時にコマンドを自動実行する仕組み
# 設定は set-hook コマンドで行う

# ── 利用可能な主要 Hooks ──
# after-new-session      — セッション作成後
# after-new-window       — ウィンドウ作成後
# after-split-window     — ペイン分割後
# after-kill-pane        — ペイン終了後
# after-select-window    — ウィンドウ切り替え後
# after-select-pane      — ペイン切り替え後
# after-resize-pane      — ペインリサイズ後
# after-copy-mode        — コピーモード終了後
# client-attached        — クライアント接続時
# client-detached        — クライアント切断時
# client-resized         — クライアントリサイズ時
# session-closed         — セッション終了時
# window-linked          — ウィンドウがセッションにリンク
# window-renamed         — ウィンドウ名変更時
# pane-exited            — ペイン内プロセス終了時
# pane-focus-in          — ペインにフォーカス時
# pane-focus-out         — ペインからフォーカス離脱時

# ── Hook の設定例 ──

# 新しいウィンドウ作成時にステータスバーの色を一時変更（通知効果）
set-hook -g after-new-window 'set -g status-style "bg=#2e7d32 fg=#ffffff"; run-shell "sleep 1"; set -g status-style "bg=#1e1e2e fg=#cdd6f4"'

# セッション作成後に自動でウィンドウ名を設定
set-hook -g after-new-session 'rename-window "main"'

# ペインフォーカス時にボーダー色を変更（アクティブペインを強調）
set-hook -g pane-focus-in 'select-pane -P "bg=#1a1b26"'
set-hook -g pane-focus-out 'select-pane -P "bg=default"'

# クライアント接続時にログを記録
set-hook -g client-attached 'run-shell "echo $(date): attached >> ~/.tmux-access.log"'
set-hook -g client-detached 'run-shell "echo $(date): detached >> ~/.tmux-access.log"'
```

### 12.2 実践的な Hook パターン

```bash
# ── 自動レイアウト調整 ──
# ウィンドウリサイズ時にレイアウトを自動的に最適化
set-hook -g client-resized 'run-shell "
    width=$(tmux display -p \"#{window_width}\")
    if [ \"$width\" -lt 120 ]; then
        tmux select-layout main-horizontal
    else
        tmux select-layout main-vertical
    fi
"'

# ── ペイン終了時の自動クリーンアップ ──
# 最後のペイン以外が終了したらレイアウトを再調整
set-hook -g after-kill-pane 'select-layout tiled'

# ── ウィンドウ切り替え時のカスタム動作 ──
# ウィンドウ切り替え時に前のウィンドウ名をログ
set-hook -g after-select-window 'run-shell "echo $(date +%H:%M:%S) $(tmux display -p \"#W\") >> /tmp/tmux-window-history.log"'

# ── 作業時間トラッキング ──
# セッション接続・切断の時刻を記録して作業時間を可視化
# ~/.tmux.conf に追加:
set-hook -g client-attached 'run-shell "
    echo \"START $(date +%Y-%m-%d_%H:%M:%S) $(tmux display -p '#S')\" >> ~/.tmux-timetrack.log
"'
set-hook -g client-detached 'run-shell "
    echo \"END   $(date +%Y-%m-%d_%H:%M:%S) $(tmux display -p '#S')\" >> ~/.tmux-timetrack.log
"'

# 作業時間の集計スクリプト
# #!/bin/bash
# awk '/START/{start=$2} /END/{print $3, start, "→", $2}' ~/.tmux-timetrack.log
```

---

## 13. tmux の環境変数とフォーマット文字列

### 13.1 環境変数の管理

```bash
# tmux はセッションごとに独立した環境変数を持つ
# グローバル環境とセッション環境の2層構造

# ── グローバル環境変数 ──
tmux set-environment -g MY_VAR "global_value"
tmux show-environment -g MY_VAR

# ── セッション環境変数 ──
tmux set-environment MY_VAR "session_value"
tmux show-environment MY_VAR

# ── 環境変数の一覧 ──
tmux show-environment -g              # グローバル一覧
tmux show-environment                 # セッション一覧

# ── 環境変数の削除 ──
tmux set-environment -g -u MY_VAR     # グローバルから削除
tmux set-environment -u MY_VAR        # セッションから削除

# ── 環境変数の自動更新 ──
# SSH_AUTH_SOCK 等を新しいクライアント接続時に更新
set -g update-environment "SSH_AUTH_SOCK SSH_CONNECTION DISPLAY XAUTHORITY"

# SSH Agent転送を維持するための設定（重要）
# ~/.tmux.conf:
set -g update-environment "SSH_AUTH_SOCK SSH_AGENT_PID"
# これにより、新しい ssh 接続で tmux に attach した際に
# SSH Agent のソケットが正しく更新される

# 手動で SSH_AUTH_SOCK を更新するスクリプト
# ~/.local/bin/fix-ssh-auth
#!/bin/bash
eval $(tmux show-env -s SSH_AUTH_SOCK 2>/dev/null)
```

### 13.2 フォーマット文字列の活用

```bash
# tmux のフォーマット文字列は #{...} 構文で使用する
# ステータスバー、display-message、if-shell 等で利用可能

# ── 主要なフォーマット変数 ──
# #{session_name}         — セッション名
# #{window_index}         — ウィンドウ番号
# #{window_name}          — ウィンドウ名
# #{pane_index}           — ペイン番号
# #{pane_current_path}    — ペインの現在ディレクトリ
# #{pane_current_command} — ペインで実行中のコマンド
# #{pane_pid}             — ペインのPID
# #{pane_width}           — ペインの幅
# #{pane_height}          — ペインの高さ
# #{window_width}         — ウィンドウの幅
# #{window_height}        — ウィンドウの高さ
# #{client_width}         — クライアントの幅
# #{client_height}        — クライアントの高さ
# #{cursor_x}             — カーソルのX位置
# #{cursor_y}             — カーソルのY位置
# #{pane_in_mode}         — コピーモードかどうか (0 or 1)
# #{window_zoomed_flag}   — ズーム状態かどうか (0 or 1)
# #{session_windows}      — セッションのウィンドウ数
# #{window_panes}         — ウィンドウのペイン数

# ── 条件分岐 ──
# #{?condition,true-value,false-value} 形式で条件分岐
# ズーム状態を表示
set -g status-right '#{?window_zoomed_flag,🔍 ZOOM ,}#H %H:%M'

# コピーモード中に表示を変更
set -g status-left '#{?pane_in_mode,COPY ,}#S '

# ── 文字列操作 ──
# #{=N:variable}   — N文字に切り詰め
# #{b:variable}    — basename
# #{d:variable}    — dirname

# ディレクトリ名をステータスに表示（basenameのみ）
set -g window-status-format '#I:#{b:pane_current_path}'
set -g window-status-current-format '#I:#{b:pane_current_path}*'

# ── display-message でのフォーマット活用 ──
tmux display-message -p "Session: #S | Window: #W (#I) | Pane: #P"
tmux display-message -p "Size: #{pane_width}x#{pane_height}"
tmux display-message -p "Path: #{pane_current_path}"
tmux display-message -p "Command: #{pane_current_command} (PID: #{pane_pid})"

# ── list-windows でカスタムフォーマット ──
tmux list-windows -F '#I: #W (#{window_panes} panes) [#{window_width}x#{window_height}]'
tmux list-panes -F '#P: #{pane_current_command} [#{pane_width}x#{pane_height}] #{pane_current_path}'
tmux list-sessions -F '#S: #{session_windows} windows (#{session_attached} attached)'
```

---

## 14. tmux Popup と高度な表示

### 14.1 Popup ウィンドウ（tmux 3.2+）

```bash
# tmux 3.2 以降で使える popup 機能
# 浮遊ウィンドウ（フローティング）としてコマンドを実行

# ── 基本的な Popup ──
tmux popup                            # デフォルトのシェルをpopupで開く
tmux popup -w 80% -h 60%             # サイズ指定
tmux popup -E "htop"                  # コマンド実行（終了でpopupも閉じる）
tmux popup -E -w 80% -h 80% "lazygit"   # lazygit をポップアップで

# ── キーバインドに登録 ──
# ~/.tmux.conf:

# Ctrl+b g で lazygit をポップアップ
bind g popup -E -w 80% -h 80% -d "#{pane_current_path}" "lazygit"

# Ctrl+b f で fzf ファイル検索 → 選択したファイルを vim で開く
bind f popup -E -w 60% -h 60% -d "#{pane_current_path}" \
    'file=$(fzf --preview "bat --color=always {}"); [ -n "$file" ] && tmux send-keys -t ! "vim $file" Enter'

# Ctrl+b j で jq をインタラクティブに使う（popup内）
bind j popup -E -w 80% -h 80% 'echo "{}" | jq -R "fromjson?" | less'

# Ctrl+b t でポップアップターミナル（簡易的な操作用）
bind t popup -E -w 60% -h 40% -d "#{pane_current_path}"

# Ctrl+b n でメモ帳をポップアップで開く
bind n popup -E -w 60% -h 60% "vim ~/notes/scratch.md"

# Ctrl+b G で git status をクイック表示
bind G popup -E -w 70% -h 50% -d "#{pane_current_path}" \
    "git status && echo '---' && git log --oneline -10; read -p 'Press Enter to close'"

# ── popup のオプション詳細 ──
# -E        — コマンド終了時にpopupを閉じる
# -w WIDTH  — 幅（数値 or パーセント）
# -h HEIGHT — 高さ（数値 or パーセント）
# -x X      — X位置
# -y Y      — Y位置
# -d DIR    — 作業ディレクトリ
# -b BORDER — ボーダースタイル（rounded, double, heavy, simple, none）
# -s STYLE  — ボーダーのスタイル（色など）
# -S STYLE  — ポップアップ内のスタイル
# -T TITLE  — タイトル

# ボーダースタイルの指定
tmux popup -b rounded -s "fg=#a6e3a1" -T "Quick Terminal" -E -w 60% -h 50%
```

### 14.2 メニューシステム（tmux 3.0+）

```bash
# tmux display-menu でインタラクティブなメニューを表示
# ~/.tmux.conf:

# Ctrl+b m でカスタムメニューを表示
bind m display-menu -T "#[align=centre]Actions" \
    "New Window"      w "new-window -c '#{pane_current_path}'" \
    "Kill Window"     x "kill-window" \
    "Horizontal Split" h "split-window -v -c '#{pane_current_path}'" \
    "Vertical Split"   v "split-window -h -c '#{pane_current_path}'" \
    "" \
    "Zoom Pane"       z "resize-pane -Z" \
    "Sync Panes"      s "setw synchronize-panes" \
    "" \
    "Choose Session"  S "choose-session" \
    "Choose Window"   W "choose-window" \
    "" \
    "Reload Config"   r "source-file ~/.tmux.conf; display 'Reloaded'" \
    "Edit Config"     e "popup -E -w 80% -h 80% 'vim ~/.tmux.conf'"

# ペインを右クリックしたときのメニュー
bind -n MouseDown3Pane display-menu -T "#[align=centre]Pane" -t = -x M -y M \
    "Split Horizontal" h "split-window -v -c '#{pane_current_path}'" \
    "Split Vertical"   v "split-window -h -c '#{pane_current_path}'" \
    "Close"           x "kill-pane" \
    "Zoom"            z "resize-pane -Z" \
    "Swap Up"         u "swap-pane -U" \
    "Swap Down"       d "swap-pane -D" \
    "" \
    "Copy Mode"       c "copy-mode"
```

---

## 15. セッション管理の自動化パターン

### 15.1 tmux-sessionizer パターン

```bash
#!/bin/bash
# tmux-sessionizer — プロジェクトディレクトリを選択してセッションを作成/切替
# ThePrimeagen 氏の手法をベースにした実装

# 検索対象ディレクトリ
SEARCH_DIRS=(
    "$HOME/projects"
    "$HOME/work"
    "$HOME/.dotfiles"
)

# fzf でプロジェクトを選択
selected=$(find "${SEARCH_DIRS[@]}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | fzf \
    --preview 'eza -la --git --no-user --no-permissions {} 2>/dev/null || ls -la {}' \
    --preview-window right:50% \
    --header "Select project to open in tmux")

# 選択がなければ終了
[ -z "$selected" ] && exit 0

# セッション名を作成（ディレクトリ名、ドットをアンダースコアに変換）
session_name=$(basename "$selected" | tr '.' '_')

# tmux が動いていない場合
if ! tmux has-session 2>/dev/null; then
    tmux new-session -d -s "$session_name" -c "$selected"
    tmux attach -t "$session_name"
    exit 0
fi

# セッションが既に存在する場合はアタッチ/切替
if tmux has-session -t="$session_name" 2>/dev/null; then
    if [ -z "$TMUX" ]; then
        tmux attach -t "$session_name"
    else
        tmux switch-client -t "$session_name"
    fi
else
    # 新規セッション作成
    if [ -z "$TMUX" ]; then
        tmux new-session -s "$session_name" -c "$selected"
    else
        tmux new-session -d -s "$session_name" -c "$selected"
        tmux switch-client -t "$session_name"
    fi
fi

# このスクリプトをキーバインドに登録:
# ~/.tmux.conf:
# bind C-f popup -E -w 80% -h 60% "~/.local/bin/tmux-sessionizer"
# または tmux 外からも使えるように:
# ~/.zshrc:
# bindkey -s '^f' '~/.local/bin/tmux-sessionizer\n'
```

### 15.2 プロジェクト別セッション設定

```bash
# ~/.config/tmux/projects/web-project.sh
#!/bin/bash
# Web開発プロジェクト用のセッション定義

SESSION="web"
ROOT="$HOME/projects/my-web-app"

tmux_setup() {
    # セッション作成
    tmux new-session -d -s "$SESSION" -n "code" -c "$ROOT"

    # コードウィンドウ（メインの作業場所）
    tmux send-keys -t "$SESSION:code" "nvim ." Enter

    # サーバーウィンドウ（フロント + バック）
    tmux new-window -t "$SESSION" -n "server" -c "$ROOT"
    tmux split-window -h -t "$SESSION:server" -c "$ROOT"
    tmux send-keys -t "$SESSION:server.0" "cd frontend && npm run dev" Enter
    tmux send-keys -t "$SESSION:server.1" "cd backend && npm run dev" Enter

    # DB・キャッシュウィンドウ
    tmux new-window -t "$SESSION" -n "data" -c "$ROOT"
    tmux split-window -h -t "$SESSION:data" -c "$ROOT"
    tmux send-keys -t "$SESSION:data.0" "docker compose up db redis" Enter
    tmux send-keys -t "$SESSION:data.1" "lazydocker" Enter

    # テストウィンドウ
    tmux new-window -t "$SESSION" -n "test" -c "$ROOT"
    tmux send-keys -t "$SESSION:test" "npm run test:watch" Enter

    # Git ウィンドウ
    tmux new-window -t "$SESSION" -n "git" -c "$ROOT"
    tmux send-keys -t "$SESSION:git" "lazygit" Enter

    # コードウィンドウに戻る
    tmux select-window -t "$SESSION:code"
}

# 既存セッションがあればアタッチ
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux attach -t "$SESSION"
else
    tmux_setup
    tmux attach -t "$SESSION"
fi
```

### 15.3 セッションの自動保存・復元

```bash
# tmux-resurrect と tmux-continuum による自動保存

# ── tmux-resurrect の設定 ──
# ~/.tmux.conf:
set -g @plugin 'tmux-plugins/tmux-resurrect'

# 保存対象の拡張
set -g @resurrect-capture-pane-contents 'on'
set -g @resurrect-strategy-vim 'session'     # vim のセッションも復元
set -g @resurrect-strategy-nvim 'session'    # neovim のセッションも復元

# 追加プログラムの復元
set -g @resurrect-processes 'ssh mosh "~rails s" "~rails c" "~mix phx.server"'

# 手動保存: Ctrl+b Ctrl+s
# 手動復元: Ctrl+b Ctrl+r

# ── tmux-continuum の設定 ──
# ~/.tmux.conf:
set -g @plugin 'tmux-plugins/tmux-continuum'

set -g @continuum-restore 'on'          # tmux 起動時に自動復元
set -g @continuum-save-interval '10'    # 10分ごとに自動保存
set -g @continuum-boot 'on'             # システム起動時に tmux を自動起動

# macOS で iTerm2 を使う場合:
set -g @continuum-boot-options 'iterm'

# ── 保存ファイルの場所 ──
# ~/.tmux/resurrect/ に保存される
ls -la ~/.tmux/resurrect/
# last → 最新の保存ファイルへのシンボリックリンク
# tmux_resurrect_YYYYMMDDTHHMMSS.txt

# 手動でバックアップ
cp ~/.tmux/resurrect/last ~/.tmux/resurrect/backup-$(date +%Y%m%d).txt
```

### 15.4 tmuxinator によるセッション管理

```bash
# tmuxinator はYAMLでセッション定義を管理するツール
# gem install tmuxinator

# ── プロジェクト作成 ──
tmuxinator new myproject

# ── YAML設定ファイル ──
# ~/.config/tmuxinator/myproject.yml
name: myproject
root: ~/projects/myproject
on_project_start: docker compose up -d
on_project_stop: docker compose down

windows:
  - editor:
      layout: main-vertical
      panes:
        - nvim .
        - git status
  - server:
      layout: even-horizontal
      panes:
        - npm run dev
        - npm run dev:api
  - logs:
      layout: even-vertical
      panes:
        - tail -f logs/app.log
        - tail -f logs/error.log
  - console:
      panes:
        - # 空のシェル

# ── tmuxinator コマンド ──
tmuxinator start myproject       # セッション開始
tmuxinator stop myproject        # セッション停止
tmuxinator list                  # プロジェクト一覧
tmuxinator edit myproject        # 設定編集
tmuxinator delete myproject      # プロジェクト削除
tmuxinator copy myproject newprj # プロジェクト複製
tmuxinator doctor                # 設定の問題をチェック
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

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |
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

| 操作 | tmux キー | コマンド |
|------|----------|---------|
| セッション作成 | - | tmux new -s name |
| デタッチ | Ctrl+b d | - |
| アタッチ | - | tmux attach -t name |
| 水平分割 | Ctrl+b " | split-window -v |
| 垂直分割 | Ctrl+b % | split-window -h |
| ペイン移動 | Ctrl+b 矢印 | select-pane -[LRUD] |
| ペインズーム | Ctrl+b z | resize-pane -Z |
| ウィンドウ作成 | Ctrl+b c | new-window |
| ウィンドウ切替 | Ctrl+b 0-9 | select-window -t N |
| コピーモード | Ctrl+b [ | - |
| ウィンドウ一覧 | Ctrl+b w | - |
| セッション一覧 | Ctrl+b s | tmux ls |
| 設定再読み込み | Ctrl+b r | source-file ~/.tmux.conf |
| コマンド入力 | Ctrl+b : | - |

---

## 次に読むべきガイド

---

## 参考文献
1. Hogan, B. "tmux 2: Productive Mouse-Free Development." Pragmatic Bookshelf, 2016.
2. "tmux Wiki." github.com/tmux/tmux/wiki.
3. "Awesome tmux." github.com/rothgar/awesome-tmux.
4. "tmux man page." man7.org/linux/man-pages/man1/tmux.1.html.
5. Barrett, D. "Efficient Linux at the Command Line." Ch.8, O'Reilly, 2022.
