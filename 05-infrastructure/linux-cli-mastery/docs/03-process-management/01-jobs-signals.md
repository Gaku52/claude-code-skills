# ジョブ制御とシグナル

> シェルのジョブ制御とシグナルは、プロセスのライフサイクルを操る基本技術。

## この章で学ぶこと

- [ ] フォアグラウンド/バックグラウンドのジョブ制御ができる
- [ ] シグナルの種類と使い方を理解する
- [ ] nohup / disown でセッション切断後も実行を継続できる

---

## 1. ジョブ制御

```bash
# フォアグラウンドとバックグラウンド
sleep 100                        # フォアグラウンド実行（端末ブロック）
sleep 100 &                      # バックグラウンド実行（&で末尾指定）

# ジョブ一覧
jobs                             # ジョブ一覧
jobs -l                          # PID付き一覧

# 出力例:
# [1]+  Running    sleep 100 &
# [2]-  Stopped    vim file.txt
#  ↑     ↑
# ジョブ番号  状態

# ジョブの切り替え
# Ctrl+Z: フォアグラウンドジョブを一時停止（SIGTSTP）
fg                               # 最後のジョブをフォアグラウンドに
fg %1                            # ジョブ番号1をフォアグラウンドに
bg                               # 停止中のジョブをバックグラウンドで再開
bg %2                            # ジョブ番号2をバックグラウンドで再開

# 典型的なワークフロー
vim file.txt                     # vim で編集中
# Ctrl+Z                        # 一時停止
make build                       # ビルド実行
fg                               # vim に戻る
```

### ジョブ指定の書式

```bash
%1                               # ジョブ番号1
%+                               # カレントジョブ（最後に操作したジョブ）
%-                               # 前のジョブ
%string                          # コマンドがstringで始まるジョブ
%?string                         # コマンドにstringを含むジョブ

# 例
fg %vim                          # vim で始まるジョブ
kill %?sleep                     # sleep を含むジョブを終了
```

---

## 2. シグナル

```bash
# シグナル一覧
kill -l                          # 全シグナル一覧

# 主要シグナル
# 番号  名前      デフォルト動作      用途
# ──────────────────────────────────────────
#  1    SIGHUP    終了              設定再読み込み / ハングアップ
#  2    SIGINT    終了              Ctrl+C（割り込み）
#  3    SIGQUIT   コアダンプ+終了   Ctrl+\（コアダンプ付き終了）
#  9    SIGKILL   強制終了          捕捉不可（最終手段）
# 15    SIGTERM   終了              正常終了要求（デフォルト）
# 18    SIGCONT   再開              停止プロセスの再開
# 19    SIGSTOP   停止              捕捉不可の一時停止
# 20    SIGTSTP   停止              Ctrl+Z（端末からの停止）

# シグナル送信
kill 1234                        # SIGTERM（デフォルト）
kill -15 1234                    # SIGTERM（明示的）
kill -9 1234                     # SIGKILL（強制終了）
kill -HUP 1234                   # SIGHUP（設定再読み込み）
kill -0 1234                     # プロセス存在確認（シグナルは送らない）

# 複数プロセスへの送信
kill 1234 5678 9012              # 複数PIDに送信
killall nginx                    # 名前で全プロセスに送信
killall -u gaku python           # ユーザー指定 + 名前
pkill -f "python server.py"      # コマンドライン全体でマッチ
pkill -u root -HUP nginx        # ユーザー + シグナル指定
```

### シグナルの正しい使い方

```bash
# 段階的な終了手順（推奨）
# ステップ1: 正常終了を要求
kill 1234                        # SIGTERM

# ステップ2: 5秒待つ
sleep 5

# ステップ3: まだ生きていれば強制終了
kill -0 1234 2>/dev/null && kill -9 1234

# ワンライナー
kill 1234 && sleep 5 && kill -0 1234 2>/dev/null && kill -9 1234

# なぜ kill -9 を最初に使わないのか？
# - クリーンアップ処理が実行されない（一時ファイル残存）
# - ファイルのフラッシュが行われない（データ損失）
# - ロックの解放が行われない（デッドロックの原因）
# - 子プロセスがゾンビ化する可能性がある
```

### SIGHUP の活用

```bash
# 設定ファイルの再読み込み（プロセスを再起動せずに）
kill -HUP $(pgrep nginx)        # nginx の設定リロード
kill -HUP $(pgrep sshd)         # sshd の設定リロード

# systemd 管理のサービスの場合
sudo systemctl reload nginx      # こちらが推奨
```

---

## 3. セッション切断後の実行継続

```bash
# nohup — ハングアップシグナルを無視
nohup long_task.sh &             # バックグラウンドで実行
# 出力は nohup.out に自動リダイレクト

nohup long_task.sh > output.log 2>&1 &   # 出力先指定
echo $!                          # 直前のバックグラウンドPID

# disown — 実行中のジョブをシェルから切り離す
long_task.sh &                   # バックグラウンドで開始
disown %1                        # ジョブ1をシェルから切り離す
disown -h %1                     # SIGHUPだけ無視（jobsには残る）
disown -a                        # 全ジョブを切り離す

# nohup vs disown
# nohup:  コマンド実行前に指定。出力を自動リダイレクト
# disown: 実行後に適用可能。出力リダイレクトは自分で行う

# setsid — 新しいセッションで実行
setsid long_task.sh              # 新セッションリーダーとして実行
```

### tmux / screen との比較

```
セッション切断対策の比較:

  nohup    → 簡単だがプロセスに再接続不可
  disown   → 実行後に適用可能。再接続不可
  tmux     → セッション管理。切断後も再接続可能 ← 推奨
  screen   → tmuxと同等。古くからある

実務では tmux を使うのが最も柔軟:
  tmux new -s work               # セッション作成
  # 作業実行...
  # Ctrl+b d                     # デタッチ（切断しても安全）
  tmux attach -t work            # 再接続
```

---

## 4. trap — シグナルハンドラ

```bash
# スクリプト内でシグナルを捕捉する
#!/bin/bash

# 一時ファイルのクリーンアップ
TMPFILE=$(mktemp)
trap 'rm -f "$TMPFILE"; echo "Cleaned up"; exit' EXIT INT TERM

# EXIT:  スクリプト終了時（正常・異常問わず）
# INT:   Ctrl+C 受信時
# TERM:  kill（SIGTERM）受信時

echo "Working with $TMPFILE..."
# ... 処理 ...

# trap の活用パターン
# パターン1: ロックファイル管理
LOCKFILE="/tmp/myapp.lock"
trap 'rm -f "$LOCKFILE"' EXIT
echo $$ > "$LOCKFILE"

# パターン2: エラー時の通知
trap 'echo "Error at line $LINENO" >&2' ERR

# パターン3: シグナルの無視
trap '' HUP                      # SIGHUPを無視
trap '' INT                      # Ctrl+Cを無視

# パターン4: デフォルト動作に戻す
trap - INT                       # SIGINTをデフォルトに戻す
```

---

## 5. 実践パターン

```bash
# パターン1: 暴走プロセスの対処
# CPU 90%以上のプロセスを表示
ps aux --sort=-%cpu | awk 'NR<=1 || $3>90'
# 確認してから kill
kill $(ps aux --sort=-%cpu | awk 'NR==2 {print $2}')

# パターン2: 全子プロセスの終了
pkill -P 1234                    # 親PID 1234の子プロセスを全て終了

# パターン3: タイムアウト付き実行
timeout 60 long_command          # 60秒でタイムアウト
timeout -s KILL 60 long_command  # 60秒でSIGKILL

# パターン4: バックグラウンドタスクの待機
task1 &
task2 &
task3 &
wait                             # 全バックグラウンドジョブの完了を待つ
echo "All tasks completed"

wait %1                          # 特定ジョブの完了を待つ

# パターン5: 並列処理
for file in *.csv; do
    process_file "$file" &
done
wait
echo "All files processed"
```

---

## まとめ

| 操作 | コマンド |
|------|---------|
| バックグラウンド実行 | command & |
| 一時停止 | Ctrl+Z |
| フォアグラウンドに戻す | fg %N |
| バックグラウンドで再開 | bg %N |
| 正常終了要求 | kill PID（SIGTERM） |
| 強制終了（最終手段） | kill -9 PID（SIGKILL） |
| 切断後も継続 | nohup / disown / tmux |
| シグナル捕捉 | trap 'handler' SIGNAL |

---

## 次に読むべきガイド
→ [[../04-networking/00-curl-wget.md]] — ネットワークツール

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.8-9, O'Reilly, 2022.
