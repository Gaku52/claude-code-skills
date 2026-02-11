# ターミナルマルチプレクサ（tmux, screen）

> tmux は1つのターミナルで複数のセッション・ウィンドウ・ペインを管理する。SSH切断後も作業が継続する。

## この章で学ぶこと

- [ ] tmux のセッション・ウィンドウ・ペインを操作できる
- [ ] SSH 切断後もプロセスを継続できる
- [ ] tmux をカスタマイズして生産性を上げる

---

## 1. tmux の基本概念

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

---

## 2. セッション管理

```bash
# セッション操作
tmux                             # 新規セッション作成
tmux new -s work                 # 名前付きセッション
tmux new -s work -d              # バックグラウンドで作成
tmux ls                          # セッション一覧
tmux attach -t work              # セッションにアタッチ
tmux attach -t 0                 # 番号でアタッチ
tmux kill-session -t work        # セッション削除
tmux kill-server                 # 全セッション削除

# セッション内操作（Ctrl+b + キー）
# Ctrl+b d    → デタッチ（セッションから離脱。プロセスは継続）
# Ctrl+b s    → セッション一覧・切り替え
# Ctrl+b $    → セッション名変更
# Ctrl+b (    → 前のセッション
# Ctrl+b )    → 次のセッション
```

---

## 3. ウィンドウ操作

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
```

---

## 4. ペイン操作（画面分割）

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

---

## 5. コピーモード

```bash
# コピーモード（スクロール・テキスト選択）
# Ctrl+b [    → コピーモード開始

# コピーモード内の操作（vi風）:
# q           → コピーモード終了
# ↑↓←→ / hjkl → カーソル移動
# Ctrl+u/d    → ページアップ/ダウン
# g / G       → 先頭/末尾
# /pattern    → 前方検索
# ?pattern    → 後方検索
# n / N       → 次/前の検索結果
# Space       → 選択開始
# Enter       → コピー（選択終了）

# Ctrl+b ]    → ペースト

# vi モードを有効にする（~/.tmux.conf）:
# setw -g mode-keys vi
```

---

## 6. tmux の設定（~/.tmux.conf）

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

# 256色対応
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB"

# ステータスバー
set -g status-style 'bg=#333333 fg=#ffffff'
set -g status-left '#[fg=green]#S '
set -g status-right '#[fg=yellow]%Y-%m-%d %H:%M'

# 設定の再読み込み
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# 履歴バッファサイズ
set -g history-limit 50000

# エスケープ時間の短縮（vim用）
set -sg escape-time 0

# 新しいウィンドウは現在のパスで開く
bind c new-window -c "#{pane_current_path}"
```

---

## 7. tmux の実践パターン

```bash
# パターン1: 開発用レイアウト
tmux new -s dev
# ペイン分割: エディタ（上大） + ターミナル（下左） + ログ（下右）
# Ctrl+b "    → 上下分割
# 下ペインで Ctrl+b % → 左右分割

# パターン2: スクリプトでレイアウト自動構築
#!/bin/bash
SESSION="dev"
tmux new-session -d -s "$SESSION" -n "editor"
tmux send-keys -t "$SESSION:editor" "vim ." Enter

tmux new-window -t "$SESSION" -n "server"
tmux send-keys -t "$SESSION:server" "npm run dev" Enter

tmux new-window -t "$SESSION" -n "logs"
tmux send-keys -t "$SESSION:logs" "tail -f /var/log/app.log" Enter

tmux select-window -t "$SESSION:editor"
tmux attach -t "$SESSION"

# パターン3: SSH先での長時間ジョブ
ssh server
tmux new -s backup
./run_backup.sh
# Ctrl+b d でデタッチ → SSH切断しても安全
# 後日: ssh server → tmux attach -t backup

# パターン4: ペアプログラミング
# ユーザーA: tmux new -s pair
# ユーザーB: tmux attach -t pair
# 同じセッションを共有して画面を見ながら作業
```

---

## 8. screen（レガシー環境用）

```bash
# screen は tmux の前身。最低限の操作だけ覚えておく

screen                           # 新規セッション
screen -S work                   # 名前付き
screen -ls                       # セッション一覧
screen -r work                   # リアタッチ
screen -d -r work                # デタッチしてからリアタッチ

# screen 内操作（Ctrl+a がプレフィックス）
# Ctrl+a d    → デタッチ
# Ctrl+a c    → 新ウィンドウ
# Ctrl+a n    → 次のウィンドウ
# Ctrl+a p    → 前のウィンドウ
# Ctrl+a |    → 垂直分割
# Ctrl+a S    → 水平分割
# Ctrl+a Tab  → ペイン切り替え
# Ctrl+a k    → ウィンドウ閉じる

# screen より tmux を使うべき理由:
# - ペイン操作が直感的
# - 設定が簡単
# - アクティブに開発されている
# - プラグインエコシステム（tpm）
```

---

## まとめ

| 操作 | tmux キー |
|------|----------|
| セッション作成 | tmux new -s name |
| デタッチ | Ctrl+b d |
| アタッチ | tmux attach -t name |
| 水平分割 | Ctrl+b " |
| 垂直分割 | Ctrl+b % |
| ペイン移動 | Ctrl+b 矢印 |
| ウィンドウ作成 | Ctrl+b c |
| ウィンドウ切替 | Ctrl+b 0-9 |
| コピーモード | Ctrl+b [ |

---

## 次に読むべきガイド
→ [[01-productivity.md]] — CLI 生産性向上

---

## 参考文献
1. Hogan, B. "tmux 2: Productive Mouse-Free Development." Pragmatic Bookshelf, 2016.
