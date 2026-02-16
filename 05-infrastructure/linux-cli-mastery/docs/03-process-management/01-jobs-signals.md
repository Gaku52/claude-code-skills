# ジョブ制御とシグナル

> シェルのジョブ制御とシグナルは、プロセスのライフサイクルを操る基本技術。
> バックグラウンド処理、プロセス間通信、堅牢なスクリプト作成のすべてに関わる重要概念である。

## この章で学ぶこと

- [ ] フォアグラウンド/バックグラウンドのジョブ制御ができる
- [ ] シグナルの種類と使い方を理解する
- [ ] nohup / disown でセッション切断後も実行を継続できる
- [ ] trap でシグナルハンドラを設定し、堅牢なスクリプトを書ける
- [ ] wait / timeout で並列処理とタイムアウトを制御できる
- [ ] プロセスグループとセッションの概念を理解する

---

## 1. ジョブ制御

### 1.1 フォアグラウンドとバックグラウンド

```bash
# フォアグラウンド実行（デフォルト）
# → 端末がブロックされ、コマンドが完了するまで入力を受け付けない
sleep 100                        # フォアグラウンド実行

# バックグラウンド実行（& を末尾に付ける）
# → 端末は空き、別のコマンドを実行可能
sleep 100 &                      # バックグラウンド実行
# [1] 12345                      ← ジョブ番号とPID が表示される

# 複数のバックグラウンドジョブ
sleep 60 &                       # [1] 12345
sleep 120 &                      # [2] 12346
find / -name "*.log" > /tmp/logs.txt 2>/dev/null &  # [3] 12347

# バックグラウンドジョブの標準出力/エラー
# バックグラウンドジョブの出力は端末に混ざって表示される
# → リダイレクトしておくのがベストプラクティス
long_task > output.log 2>&1 &    # 出力をファイルにリダイレクト

# バックグラウンドで実行しつつPIDを記録
long_task &
PID=$!                           # $! = 直前のバックグラウンドプロセスのPID
echo "Started with PID: $PID"
```

### 1.2 ジョブ一覧と状態確認

```bash
# ジョブ一覧
jobs                             # 現在のシェルのジョブ一覧
jobs -l                          # PID付き一覧
jobs -r                          # 実行中のジョブのみ
jobs -s                          # 停止中のジョブのみ
jobs -p                          # PIDのみ表示

# 出力例:
# [1]+  Running    sleep 100 &
# [2]-  Stopped    vim file.txt
# [3]   Running    find / -name "*.log" > /tmp/logs.txt 2>/dev/null &
#  ↑ ↑   ↑
# ジョブ番号  状態
#    + = カレントジョブ（最後に操作した/起動したジョブ）
#    - = 前のジョブ

# ジョブの状態:
# Running:   実行中（バックグラウンド）
# Stopped:   一時停止中（Ctrl+Z で停止）
# Done:      完了（次のプロンプト表示時に通知）
# Terminated: シグナルで終了
# Killed:    SIGKILLで強制終了
# Exit N:    終了コード N で終了
```

### 1.3 ジョブの切り替え操作

```bash
# Ctrl+Z: フォアグラウンドジョブを一時停止（SIGTSTP送信）
vim file.txt                     # vim で編集中
# Ctrl+Z                        # vim が一時停止（Stoppedになる）
# [1]+  Stopped    vim file.txt

# fg: ジョブをフォアグラウンドに戻す
fg                               # カレントジョブ（+のついたジョブ）
fg %1                            # ジョブ番号1をフォアグラウンドに
fg %vim                          # vim で始まるジョブを指定
fg %?file                        # "file" を含むジョブを指定

# bg: 停止中のジョブをバックグラウンドで再開
bg                               # カレントジョブをバックグラウンドで再開
bg %2                            # ジョブ番号2をバックグラウンドで再開

# 典型的なワークフロー
vim file.txt                     # vim で編集中
# Ctrl+Z                        # 一時停止
make build                       # ビルド実行
fg                               # vim に戻る

# 典型的なワークフロー2: フォアグラウンドをバックグラウンドに移す
long_running_command              # フォアグラウンドで実行してしまった
# Ctrl+Z                        # 一時停止
bg                               # バックグラウンドで再開
# これで端末が使える

# 典型的なワークフロー3: 複数のエディタを切り替え
vim file1.txt                    # 編集
# Ctrl+Z
vim file2.txt                    # 別のファイルを編集
# Ctrl+Z
jobs                             # ジョブ確認
fg %1                            # file1 に戻る
```

### 1.4 ジョブ指定の書式

```bash
# ジョブの指定方法
%1                               # ジョブ番号1
%2                               # ジョブ番号2
%%                               # カレントジョブ（%+ と同じ）
%+                               # カレントジョブ（最後に操作したジョブ）
%-                               # 前のジョブ（カレントの1つ前）
%string                          # コマンドがstringで始まるジョブ
%?string                         # コマンドにstringを含むジョブ

# 使用例
fg %vim                          # vim で始まるジョブをフォアグラウンドに
bg %2                            # ジョブ2をバックグラウンドで再開
kill %?sleep                     # sleep を含むジョブを終了
kill %%                          # カレントジョブを終了
kill %1 %2 %3                   # 複数ジョブを終了
wait %1                          # ジョブ1の完了を待つ

# 注意: ジョブ番号はシェルごとに独立
# 別の端末/シェルのジョブにはアクセスできない
# → PID を使う場合は kill コマンドを使う
```

### 1.5 バックグラウンドジョブの注意点

```bash
# 注意1: バックグラウンドジョブの出力は端末に混ざる
long_task &
# → 出力がプロンプトに割り込む可能性
# 対策: リダイレクト
long_task > /tmp/output.log 2>&1 &

# 注意2: バックグラウンドジョブの入力
# バックグラウンドジョブが端末からの入力を要求すると停止する
cat &                            # 入力待ちで自動停止
# [1]+  Stopped    cat

# 注意3: シェルを終了するとバックグラウンドジョブにSIGHUPが送られる
# → nohup や disown を使う（後述）

# 注意4: bash の huponexit オプション
shopt -s huponexit               # シェル終了時に全ジョブにSIGHUP送信（デフォルトOFF）
shopt -u huponexit               # SIGHUP送信しない

# 注意5: スクリプト内でのバックグラウンドジョブ
#!/bin/bash
task1 &
task2 &
wait                             # 全バックグラウンドジョブの完了を待つ
echo "全タスク完了"
# wait を忘れると、スクリプトがジョブの完了前に終了する
```

---

## 2. シグナル

### 2.1 シグナルの基礎

```bash
# シグナルとは: カーネルからプロセスへの非同期通知メカニズム
# プロセスを制御する（終了、停止、再開など）ための仕組み

# シグナル一覧
kill -l                          # 全シグナル一覧（名前）
kill -l 15                       # シグナル番号→名前（TERM）
kill -l TERM                     # シグナル名→番号（15）

# シグナルの配送:
# 1. ユーザーがシグナルを送信（kill コマンド、Ctrl+C など）
# 2. カーネルがプロセスにシグナルを配送
# 3. プロセスがシグナルを処理:
#    a. デフォルト動作を実行（終了、停止など）
#    b. カスタムハンドラを実行（trap で設定）
#    c. シグナルを無視（一部のシグナルのみ）
#    ※ SIGKILL と SIGSTOP は捕捉も無視もできない
```

### 2.2 主要シグナルの詳細

```bash
# ┌────────┬───────────┬──────────────────┬─────────────────────────────────┐
# │ 番号   │ 名前      │ デフォルト動作   │ 用途・説明                      │
# ├────────┼───────────┼──────────────────┼─────────────────────────────────┤
# │  1     │ SIGHUP    │ 終了             │ ハングアップ / 設定再読み込み   │
# │  2     │ SIGINT    │ 終了             │ Ctrl+C（割り込み）             │
# │  3     │ SIGQUIT   │ コアダンプ+終了  │ Ctrl+\（デバッグ用終了）       │
# │  6     │ SIGABRT   │ コアダンプ+終了  │ abort() 呼び出し               │
# │  9     │ SIGKILL   │ 強制終了         │ 捕捉不可（最終手段）           │
# │ 11     │ SIGSEGV   │ コアダンプ+終了  │ セグメンテーション違反         │
# │ 13     │ SIGPIPE   │ 終了             │ 壊れたパイプへの書き込み       │
# │ 14     │ SIGALRM   │ 終了             │ alarm() タイマー満了           │
# │ 15     │ SIGTERM   │ 終了             │ 正常終了要求（kill のデフォルト）│
# │ 17     │ SIGCHLD   │ 無視             │ 子プロセスの状態変更           │
# │ 18     │ SIGCONT   │ 再開             │ 停止プロセスの再開             │
# │ 19     │ SIGSTOP   │ 停止             │ 捕捉不可の一時停止             │
# │ 20     │ SIGTSTP   │ 停止             │ Ctrl+Z（端末からの停止）       │
# │ 21     │ SIGTTIN   │ 停止             │ バックグラウンドプロセスの入力  │
# │ 22     │ SIGTTOU   │ 停止             │ バックグラウンドプロセスの出力  │
# │ 28     │ SIGWINCH  │ 無視             │ 端末のウィンドウサイズ変更     │
# │ 10     │ SIGUSR1   │ 終了             │ ユーザー定義シグナル1          │
# │ 12     │ SIGUSR2   │ 終了             │ ユーザー定義シグナル2          │
# └────────┴───────────┴──────────────────┴─────────────────────────────────┘

# 注意: シグナル番号はOS/アーキテクチャによって異なる場合がある
# → スクリプトでは名前で指定するのが安全（kill -TERM, kill -HUP）

# SIGKILL（9）と SIGSTOP（19）の特殊性:
# - 捕捉（trap）できない
# - 無視できない
# - ブロックできない
# → カーネルが直接処理する
```

### 2.3 シグナルの送信

```bash
# kill コマンド（名前が紛らわしいが、任意のシグナルを送信するコマンド）
kill 1234                        # SIGTERM（デフォルト）
kill -15 1234                    # SIGTERM（番号で指定）
kill -TERM 1234                  # SIGTERM（名前で指定）
kill -s TERM 1234                # SIGTERM（-s オプション）

kill -9 1234                     # SIGKILL（強制終了）
kill -KILL 1234                  # SIGKILL（名前で指定）

kill -HUP 1234                   # SIGHUP（設定再読み込み）
kill -USR1 1234                  # SIGUSR1（ユーザー定義）
kill -USR2 1234                  # SIGUSR2（ユーザー定義）

kill -CONT 1234                  # SIGCONT（停止プロセスの再開）
kill -STOP 1234                  # SIGSTOP（強制停止）

kill -0 1234                     # シグナルは送らない（プロセス存在確認）
if kill -0 1234 2>/dev/null; then
    echo "PID 1234 は存在する"
else
    echo "PID 1234 は存在しない"
fi

# 複数プロセスへの送信
kill 1234 5678 9012              # 複数PIDに送信
kill -TERM 1234 5678             # 複数PIDにTERM

# killall — プロセス名でシグナル送信
killall nginx                    # nginx という名前の全プロセスにTERM
killall -9 nginx                 # 全 nginx プロセスを強制終了
killall -u gaku python           # ユーザー gaku の python プロセスを終了
killall -i nginx                 # 確認プロンプト付き（1つずつ）
killall -v nginx                 # 詳細表示
killall -w nginx                 # 全プロセスが終了するまで待機
killall -e "python3 server.py"   # 完全一致（exactマッチ）
killall -s HUP nginx             # シグナル指定
killall -o 1h nginx              # 1時間以上前に起動したもの
killall -y 30m nginx             # 30分以内に起動したもの

# pkill — 柔軟なパターンマッチングでシグナル送信
pkill nginx                      # nginx にマッチするプロセスにTERM
pkill -9 nginx                   # 強制終了
pkill -f "python server.py"      # コマンドライン全体でマッチ
pkill -u root nginx              # ユーザー指定
pkill -u root -HUP nginx         # ユーザー + シグナル指定
pkill -P 1234                    # 親PID 1234 の子プロセスにTERM
pkill -t pts/0                   # 端末 pts/0 のプロセスにTERM
pkill -g 1234                    # プロセスグループ 1234 にTERM
pkill -x nginx                   # 完全一致
pkill --signal HUP nginx         # シグナル指定（長いオプション形式）
pkill -c nginx                   # マッチしたプロセス数を表示
pkill -e nginx                   # 終了させたプロセスを表示

# プロセスグループ全体にシグナルを送信
kill -TERM -1234                 # PID の前にマイナスをつける → プロセスグループ
kill -- -1234                    # -- で負の数と区別
```

### 2.4 シグナルの正しい使い方（段階的終了）

```bash
# === 推奨される段階的な終了手順 ===

# ステップ1: 正常終了を要求（SIGTERM）
kill 1234                        # SIGTERM
# → プロセスはクリーンアップ処理を実行して終了するチャンス

# ステップ2: 数秒待つ
sleep 5

# ステップ3: まだ生きていれば強制終了（SIGKILL）
kill -0 1234 2>/dev/null && kill -9 1234

# ワンライナー版
kill 1234; sleep 5; kill -0 1234 2>/dev/null && kill -9 1234

# スクリプト版（関数化）
graceful_kill() {
    local pid=$1
    local timeout=${2:-10}  # デフォルト10秒

    # プロセスが存在するか確認
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "PID $pid は既に存在しない"
        return 0
    fi

    # SIGTERM を送信
    echo "PID $pid に SIGTERM を送信..."
    kill "$pid"

    # timeout 秒待機
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "PID $pid が正常終了"
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # まだ生きていれば SIGKILL
    echo "タイムアウト。PID $pid に SIGKILL を送信..."
    kill -9 "$pid"
    sleep 1

    if kill -0 "$pid" 2>/dev/null; then
        echo "警告: PID $pid が SIGKILL でも終了しない（D状態の可能性）"
        return 1
    fi

    echo "PID $pid を強制終了"
    return 0
}

# 使用例
graceful_kill 1234
graceful_kill 1234 30    # 30秒のタイムアウト

# === なぜ kill -9 を最初に使わないのか？ ===
#
# kill -9 (SIGKILL) の問題点:
# 1. クリーンアップ処理が実行されない
#    → 一時ファイルが残る
#    → PIDファイルが残る
#    → ソケットファイルが残る
#
# 2. ファイルのフラッシュが行われない
#    → バッファ内のデータが失われる
#    → ログの最後の数行が失われる
#    → データベースの未コミットトランザクションが失われる
#
# 3. ロックの解放が行われない
#    → ファイルロックが残り、次回起動時にデッドロック
#    → 共有メモリセグメントが残る
#
# 4. 子プロセスへの影響
#    → 子プロセスがゾンビ化する可能性
#    → 子プロセスの終了処理が行われない
#
# 5. trap ハンドラが実行されない
#    → スクリプトのクリーンアップコードが無視される
```

### 2.5 キーボードショートカットとシグナル

```bash
# 端末のキーボードショートカットとシグナルの対応
# Ctrl+C  → SIGINT  (2)   割り込み（プロセスの終了）
# Ctrl+\  → SIGQUIT (3)   終了（コアダンプ付き）
# Ctrl+Z  → SIGTSTP (20)  一時停止
# Ctrl+D  → EOF（シグナルではない、標準入力の終端）

# ショートカットの確認・変更
stty -a                          # 全端末設定の表示
# intr = ^C; quit = ^\; erase = ^?; kill = ^U; eof = ^D;
# susp = ^Z; ...

# キーバインドの変更（一時的）
stty intr ^X                     # Ctrl+X で SIGINT に変更
stty susp ^Y                     # Ctrl+Y で SIGTSTP に変更
stty intr ^C                     # 元に戻す

# Ctrl+C を無効化（Ctrl+C で終了させたくないスクリプト内で）
stty -isig                       # 全シグナルキーを無効化
stty isig                        # 元に戻す
```

### 2.6 SIGHUP の活用

```bash
# SIGHUP の2つの用途:
# 1. 端末が切断されたときの通知（元の意味: Hang Up）
# 2. デーモンへの設定再読み込み要求（慣例的な使い方）

# 設定ファイルの再読み込み（プロセスを再起動せずに）
kill -HUP $(pgrep -o nginx)      # nginx の設定リロード
kill -HUP $(cat /var/run/nginx.pid)  # PIDファイルから
kill -HUP $(pgrep sshd | head -1) # sshd の設定リロード

# systemd 管理のサービスの場合（こちらが推奨）
sudo systemctl reload nginx      # nginx設定リロード
sudo systemctl reload sshd       # sshd設定リロード
sudo systemctl reload postgresql # PostgreSQL設定リロード

# SIGHUP で設定再読み込みをサポートする主要デーモン:
# nginx:     ワーカープロセスの graceful restart
# Apache:    設定再読み込み
# sshd:      設定再読み込み
# rsyslog:   設定再読み込み
# logrotate: ログローテーション時に使用
# PostgreSQL: postgresql.conf の再読み込み
# HAProxy:   設定再読み込み

# SIGHUP で再読み込みされる内容（nginx の場合）:
# - nginx.conf とインクルードファイル
# - 新しいワーカープロセスが新設定で起動
# - 古いワーカープロセスは現在の接続を処理してから終了
# - マスタープロセスは再起動しない
# → ダウンタイムなしで設定変更が可能
```

---

## 3. セッション切断後の実行継続

### 3.1 nohup

```bash
# nohup: SIGHUP を無視してコマンドを実行
# SSH接続が切れてもプロセスが終了しない

# 基本的な使い方
nohup long_task.sh &             # バックグラウンドで実行
# 出力は nohup.out に自動リダイレクト（カレントディレクトリ）
# nohup.out に書けない場合は ~/nohup.out にフォールバック

# 出力先を明示指定（推奨）
nohup long_task.sh > /var/log/task.log 2>&1 &
echo $!                          # PID を表示

# PIDを記録
nohup long_task.sh > /tmp/task.log 2>&1 &
echo $! > /tmp/task.pid
# 後で確認: cat /tmp/task.pid

# nohup の動作:
# 1. SIGHUP を無視（SIG_IGN）に設定
# 2. 標準出力が端末の場合、nohup.out にリダイレクト
# 3. 標準エラー出力が端末の場合、標準出力にリダイレクト
# 4. コマンドを exec する

# nohup の確認方法
cat /proc/$(cat /tmp/task.pid)/status | grep SigIgn
# SigIgn: 0000000000000001  ← SIGHUP(1) が無視されている
```

### 3.2 disown

```bash
# disown: 実行中のジョブをシェルのジョブテーブルから切り離す
# nohup を付け忘れて実行してしまった場合に有用

# 基本的な使い方
long_task.sh &                   # バックグラウンドで開始
disown %1                        # ジョブ1をシェルから切り離す
# → シェル終了時にSIGHUPが送信されなくなる

# SIGHUP だけ無視（ジョブリストには残る）
long_task.sh &
disown -h %1                     # SIGHUPだけ無視（jobsには残る）

# 全ジョブを切り離す
disown -a                        # 全ジョブを切り離す

# 最後のバックグラウンドジョブを切り離す
long_task.sh &
disown                           # 引数なし = 最後のジョブ

# 典型的なワークフロー: nohup を忘れた場合の対処
long_running_command              # フォアグラウンドで実行中
# Ctrl+Z                        # 一時停止
bg                               # バックグラウンドで再開
disown %1                        # シェルから切り離す
# → SSH接続が切れてもプロセスは継続する

# 注意: disown は出力をリダイレクトしない
# → 出力が端末に残る場合、端末を閉じるとSIGPIPEが送信される可能性
# → 可能であればリダイレクトしてから disown する
long_task.sh > /tmp/output.log 2>&1 &
disown %1
```

### 3.3 nohup vs disown の比較

```bash
# ┌────────────┬──────────────────────────────────┬────────────────────────────────┐
# │ 特徴       │ nohup                            │ disown                         │
# ├────────────┼──────────────────────────────────┼────────────────────────────────┤
# │ 実行タイミング │ コマンド実行前に指定            │ 実行後に適用可能               │
# │ SIGHUP     │ 無視                             │ シェルからの送信を防ぐ         │
# │ 出力       │ nohup.out に自動リダイレクト      │ リダイレクトしない             │
# │ ジョブリスト │ 通常通り表示                     │ 切り離されるとリストから消える  │
# │ 再接続     │ 不可                             │ 不可                           │
# │ 主な用途   │ 計画的な長時間タスク              │ nohup 忘れの救済               │
# └────────────┴──────────────────────────────────┴────────────────────────────────┘
```

### 3.4 setsid

```bash
# setsid: 新しいセッションでコマンドを実行
# 端末から完全に切り離される

setsid long_task.sh              # 新セッションリーダーとして実行
setsid -f long_task.sh           # フォーク後に新セッション作成

# setsid の動作:
# 1. fork() で子プロセスを作成
# 2. 子プロセスで setsid() を呼び出し
# 3. 新しいセッションの作成（制御端末を持たない）
# 4. コマンドを exec
#
# nohup/disown との違い:
# - 完全に新しいセッションを作成（端末から完全分離）
# - デーモン化に近い動作
# - プロセスグループも新しくなる
```

### 3.5 tmux / screen との比較

```
セッション切断対策の比較:

┌────────────┬────────────────────────────────────────────┐
│ 方法       │ 特徴                                       │
├────────────┼────────────────────────────────────────────┤
│ nohup      │ 簡単。出力は nohup.out へ。再接続不可      │
│ disown     │ 実行後に適用可能。再接続不可               │
│ setsid     │ 完全にセッション分離。再接続不可           │
│ tmux       │ セッション管理。切断後も再接続可能 ← 推奨  │
│ screen     │ tmuxと同等。古くからあり互換性高い         │
│ systemd    │ サービスとして管理。ログ、再起動対応       │
└────────────┴────────────────────────────────────────────┘

実務での使い分け:
  一時的なコマンド実行   → nohup（最も簡単）
  nohup を忘れた場合     → disown（救済策）
  対話的な長時間作業     → tmux ← 最も推奨
  デーモン的なサービス   → systemd / supervisor
```

### 3.6 tmux でのセッション管理

```bash
# tmux の基本操作（ジョブ制御の文脈で）

# セッション作成
tmux new -s work                 # "work" という名前でセッション作成

# 作業実行
# ... 長時間タスクを実行 ...

# デタッチ（切断してもセッションは維持）
# Ctrl+b d                      # デタッチ

# 再接続（SSH再接続後でも可能）
tmux attach -t work              # "work" セッションに再接続
tmux a -t work                   # 省略形

# セッション一覧
tmux ls                          # セッション一覧

# セッション終了
tmux kill-session -t work        # セッションを終了

# ウィンドウ操作
# Ctrl+b c                      # 新しいウィンドウ
# Ctrl+b n                      # 次のウィンドウ
# Ctrl+b p                      # 前のウィンドウ
# Ctrl+b 0-9                    # 番号でウィンドウ切替
# Ctrl+b w                      # ウィンドウ一覧

# ペイン操作
# Ctrl+b %                      # 垂直分割
# Ctrl+b "                      # 水平分割
# Ctrl+b 矢印キー               # ペイン間移動
# Ctrl+b z                      # ペインのズーム切替

# スクリプトでの tmux 活用
tmux new-session -d -s build 'make all && echo Done'
# バックグラウンドでセッションを作成してコマンド実行
# 後で tmux a -t build で結果を確認可能
```

---

## 4. trap — シグナルハンドラ

### 4.1 基本構文

```bash
# trap 'コマンド' シグナル [シグナル...]

# 基本的な使い方
trap 'echo "Ctrl+C を受信しました"' INT

# 複数のシグナルを捕捉
trap 'echo "終了します"' INT TERM

# シグナルを無視
trap '' INT                      # SIGINTを無視（Ctrl+Cが効かなくなる）
trap '' HUP                      # SIGHUPを無視

# デフォルト動作に戻す
trap - INT                       # SIGINTをデフォルトに戻す
trap - HUP TERM                  # 複数シグナルをリセット

# 現在のtrap設定を表示
trap -p                          # 全trap設定
trap -p INT                      # 特定シグナルの設定
```

### 4.2 EXIT トラップ（最重要パターン）

```bash
#!/bin/bash
# EXIT トラップ: スクリプト終了時に必ず実行される
# 正常終了でも異常終了でも呼ばれる（SIGKILLを除く）

# パターン1: 一時ファイルのクリーンアップ
TMPFILE=$(mktemp)
TMPDIR=$(mktemp -d)
trap 'rm -f "$TMPFILE"; rm -rf "$TMPDIR"; echo "クリーンアップ完了"' EXIT

echo "一時ファイル: $TMPFILE"
echo "一時ディレクトリ: $TMPDIR"
# ... 処理 ...
# スクリプト終了時に自動的にクリーンアップ

# パターン2: ロックファイル管理
LOCKFILE="/tmp/myapp.lock"
if [ -f "$LOCKFILE" ]; then
    echo "別のインスタンスが実行中です（ロックファイル: $LOCKFILE）" >&2
    exit 1
fi
trap 'rm -f "$LOCKFILE"' EXIT
echo $$ > "$LOCKFILE"
# ... 処理 ...

# パターン3: PIDファイル管理
PIDFILE="/var/run/myapp.pid"
trap 'rm -f "$PIDFILE"' EXIT
echo $$ > "$PIDFILE"

# パターン4: サービスの停止処理
trap 'echo "シャットダウン中..."; stop_service; echo "完了"' EXIT

# パターン5: SSH接続のクリーンアップ
trap 'ssh -O exit user@server 2>/dev/null' EXIT
ssh -M -S /tmp/ssh_mux_%h_%p_%r user@server
```

### 4.3 ERR トラップ

```bash
#!/bin/bash
# ERR トラップ: コマンドが非ゼロ終了コードを返したときに実行される
# set -e と組み合わせて使うことが多い

# パターン1: エラー発生箇所の表示
trap 'echo "エラー発生: 行 $LINENO コマンド \"$BASH_COMMAND\" 終了コード $?" >&2' ERR

# set -e と組み合わせ
set -e
trap 'echo "行 $LINENO でエラー: $BASH_COMMAND" >&2' ERR

# パターン2: エラー時のスタックトレース
trap 'echo "Error at ${BASH_SOURCE[0]}:${LINENO} in ${FUNCNAME[0]:-main}"' ERR

# パターン3: 詳細なエラーハンドリング
error_handler() {
    local exit_code=$?
    local line_no=$1
    local command=$2
    echo "=============================" >&2
    echo "エラー発生!" >&2
    echo "  行番号: $line_no" >&2
    echo "  コマンド: $command" >&2
    echo "  終了コード: $exit_code" >&2
    echo "  スクリプト: ${BASH_SOURCE[1]}" >&2
    echo "=============================" >&2

    # コールスタック表示
    local i=0
    echo "コールスタック:" >&2
    while caller $i; do
        ((i++))
    done 2>/dev/null >&2
}
trap 'error_handler $LINENO "$BASH_COMMAND"' ERR

# 注意: ERR トラップはサブシェルには伝播しない（デフォルト）
# set -E で ERR トラップをサブシェルに伝播させる
set -eE
trap 'echo "Error at line $LINENO" >&2' ERR
```

### 4.4 DEBUG トラップ

```bash
#!/bin/bash
# DEBUG トラップ: 各コマンドの実行前に呼ばれる

# パターン1: 実行コマンドの追跡（デバッグ用）
trap 'echo "DEBUG: $BASH_COMMAND (行 $LINENO)"' DEBUG

echo "step 1"
echo "step 2"
# 出力:
# DEBUG: echo "step 1" (行 5)
# step 1
# DEBUG: echo "step 2" (行 6)
# step 2

# パターン2: 実行時間の計測
LAST_TIME=$(date +%s%N)
trap '
    NOW=$(date +%s%N)
    ELAPSED=$(( (NOW - LAST_TIME) / 1000000 ))
    if [ $ELAPSED -gt 100 ]; then
        echo "SLOW: ${ELAPSED}ms - $BASH_COMMAND" >&2
    fi
    LAST_TIME=$NOW
' DEBUG
```

### 4.5 RETURN トラップ

```bash
#!/bin/bash
# RETURN トラップ: 関数やsourceから戻るときに呼ばれる

trap 'echo "関数から戻りました"' RETURN

my_function() {
    echo "関数内"
    return 0
}

my_function
# 出力:
# 関数内
# 関数から戻りました
```

### 4.6 trap の実践パターン集

```bash
#!/bin/bash
# === 堅牢なスクリプトのテンプレート ===

set -euo pipefail

# グローバル変数
SCRIPT_NAME=$(basename "$0")
TMPDIR=""
LOCKFILE=""
CLEANUP_DONE=false

# クリーンアップ関数
cleanup() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true

    echo "[$SCRIPT_NAME] クリーンアップ中..."

    # 一時ディレクトリの削除
    if [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ]; then
        rm -rf "$TMPDIR"
    fi

    # ロックファイルの削除
    if [ -n "$LOCKFILE" ] && [ -f "$LOCKFILE" ]; then
        rm -f "$LOCKFILE"
    fi

    # バックグラウンドジョブの終了
    jobs -p | xargs -r kill 2>/dev/null

    echo "[$SCRIPT_NAME] クリーンアップ完了"
}

# エラーハンドラ
error_handler() {
    local exit_code=$?
    local line_no=$1
    echo "[$SCRIPT_NAME] エラー: 行 $line_no 終了コード $exit_code" >&2
    cleanup
    exit "$exit_code"
}

# トラップの設定
trap cleanup EXIT
trap 'error_handler $LINENO' ERR
trap 'echo "[$SCRIPT_NAME] 割り込みを受信"; exit 130' INT
trap 'echo "[$SCRIPT_NAME] 終了要求を受信"; exit 143' TERM

# 初期化
TMPDIR=$(mktemp -d "/tmp/${SCRIPT_NAME}.XXXXXX")
LOCKFILE="/tmp/${SCRIPT_NAME}.lock"

if [ -f "$LOCKFILE" ]; then
    EXISTING_PID=$(cat "$LOCKFILE")
    if kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "エラー: 別のインスタンスが実行中 (PID: $EXISTING_PID)" >&2
        exit 1
    else
        echo "警告: 古いロックファイルを削除します" >&2
        rm -f "$LOCKFILE"
    fi
fi
echo $$ > "$LOCKFILE"

# メイン処理
echo "[$SCRIPT_NAME] 開始 (PID: $$, TMPDIR: $TMPDIR)"
# ... ここにメイン処理 ...
echo "[$SCRIPT_NAME] 正常完了"
```

```bash
# === Ctrl+C で中断可能なループ ===
#!/bin/bash

RUNNING=true
trap 'RUNNING=false; echo "中断要求を受信..."' INT

echo "処理を開始します (Ctrl+C で中断)"
count=0
while $RUNNING && [ $count -lt 100 ]; do
    echo "処理中... ($count/100)"
    sleep 1
    count=$((count + 1))
done

if $RUNNING; then
    echo "全処理が完了しました"
else
    echo "処理が中断されました ($count/100 完了)"
fi
```

```bash
# === SIGUSR1/SIGUSR2 を使ったプロセス間通信 ===
#!/bin/bash

# ワーカースクリプト（worker.sh）
STATS_REQUESTS=0
PAUSED=false

# SIGUSR1 → 統計情報を出力
trap 'echo "統計: リクエスト数=$STATS_REQUESTS, 一時停止=$PAUSED"' USR1

# SIGUSR2 → 一時停止/再開のトグル
trap '
    if $PAUSED; then
        PAUSED=false
        echo "再開"
    else
        PAUSED=true
        echo "一時停止"
    fi
' USR2

echo "ワーカー開始 (PID: $$)"
while true; do
    if ! $PAUSED; then
        # ... 実際の処理 ...
        STATS_REQUESTS=$((STATS_REQUESTS + 1))
    fi
    sleep 1
done

# 制御側:
# kill -USR1 <PID>   # 統計表示
# kill -USR2 <PID>   # 一時停止/再開
```

---

## 5. wait / timeout — 並列処理の制御

### 5.1 wait — バックグラウンドジョブの完了待ち

```bash
# 全バックグラウンドジョブの完了を待つ
task1 &
task2 &
task3 &
wait                             # 全ジョブが終了するまでブロック
echo "全タスク完了"

# 特定のPIDの完了を待つ
task1 &
PID1=$!
task2 &
PID2=$!

wait $PID1
echo "task1 完了 (終了コード: $?)"
wait $PID2
echo "task2 完了 (終了コード: $?)"

# 特定ジョブの完了を待つ
task1 &
wait %1
echo "ジョブ1 完了"

# どれか1つの完了を待つ（bash 4.3+）
task1 &
PID1=$!
task2 &
PID2=$!
task3 &
PID3=$!

wait -n                          # 最初に終了したジョブを待つ
echo "最初のジョブが完了 (終了コード: $?)"

# wait -n で完了したPIDを取得（bash 5.1+）
wait -n -p DONE_PID $PID1 $PID2 $PID3
echo "PID $DONE_PID が完了"
```

### 5.2 並列処理パターン

```bash
# パターン1: 単純な並列実行
#!/bin/bash
for file in *.csv; do
    process_file "$file" &
done
wait
echo "全ファイル処理完了"

# パターン2: 並列数を制限した並列処理
#!/bin/bash
MAX_PARALLEL=4
count=0

for file in *.csv; do
    process_file "$file" &
    count=$((count + 1))

    if [ $count -ge $MAX_PARALLEL ]; then
        wait -n              # 1つ完了するのを待つ
        count=$((count - 1))
    fi
done
wait                         # 残りの全ジョブを待つ

# パターン3: 結果の収集
#!/bin/bash
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PIDS=()
for i in $(seq 1 10); do
    (
        result=$(some_task "$i")
        echo "$result" > "$TMPDIR/result_$i"
    ) &
    PIDS+=($!)
done

# 全完了を待ち、エラーチェック
ERRORS=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        ERRORS=$((ERRORS + 1))
    fi
done

echo "完了: 成功=$((10 - ERRORS)), 失敗=$ERRORS"
cat "$TMPDIR"/result_* | sort

# パターン4: xargs による並列処理
find . -name "*.jpg" | xargs -P 4 -I {} convert {} -resize 800x600 resized_{}
# -P 4: 4並列

# パターン5: GNU parallel による並列処理
# parallel がインストールされている場合
find . -name "*.csv" | parallel -j 4 process_file {}
seq 100 | parallel -j 8 'curl -s "https://api.example.com/item/{}" > /tmp/item_{}.json'
```

### 5.3 timeout — タイムアウト付き実行

```bash
# timeout: 指定時間でコマンドを自動終了

# 基本的な使い方
timeout 60 long_command          # 60秒でタイムアウト（SIGTERM）
timeout 30s curl -s https://example.com  # 30秒でタイムアウト
timeout 5m make build            # 5分でタイムアウト
timeout 2h rsync -avz src/ dst/  # 2時間でタイムアウト

# 時間の単位:
# s: 秒（デフォルト）
# m: 分
# h: 時間
# d: 日

# タイムアウト時のシグナルを指定
timeout -s KILL 60 long_command  # 60秒後にSIGKILL
timeout --signal=HUP 30 daemon  # 30秒後にSIGHUP

# 段階的タイムアウト（-k オプション）
timeout -k 10 60 long_command
# 60秒後に SIGTERM を送信
# それでも終了しなければ 10秒後に SIGKILL を送信

# 終了コードの確認
timeout 5 sleep 10
echo $?                          # 124 = タイムアウトで終了
# 終了コード:
# 124: SIGTERM でタイムアウト
# 137: SIGKILL でタイムアウト（128 + 9）
# それ以外: コマンド自身の終了コード

# タイムアウトかどうかの判定
timeout 5 some_command
EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "タイムアウトしました"
elif [ $EXIT_CODE -eq 0 ]; then
    echo "正常完了"
else
    echo "エラー終了 (コード: $EXIT_CODE)"
fi

# フォアグラウンドで実行（--foreground）
timeout --foreground 60 interactive_command
# 対話的なコマンドに使う場合

# 実践例: APIのタイムアウト付きヘルスチェック
timeout 5 curl -s -o /dev/null -w "%{http_code}" https://api.example.com/health
if [ $? -eq 124 ]; then
    echo "API応答タイムアウト"
fi

# 実践例: タイムアウト付きファイル待機
timeout 60 bash -c 'while [ ! -f /tmp/ready.flag ]; do sleep 1; done'
if [ $? -eq 124 ]; then
    echo "ファイルが作成されませんでした（タイムアウト）"
fi
```

---

## 6. プロセスグループとセッション

### 6.1 概念の理解

```bash
# プロセスの階層構造:
#
# セッション（SID）
#   └─ プロセスグループ（PGID）
#       └─ プロセス（PID）
#           └─ スレッド（TID）
#
# セッション: ログインから始まる一連のプロセスの集合
# プロセスグループ: パイプラインなどで関連するプロセスの集合
# セッションリーダー: セッションを開始したプロセス（ログインシェル）

# 確認方法
ps -eo pid,ppid,pgid,sid,tty,cmd | head -20

# 例: cat file | grep pattern | sort
# この3つのプロセスは同じプロセスグループに属する
# → Ctrl+C で3つ全部にSIGINTが送られる

# 現在のプロセスの情報
echo "PID: $$"                   # プロセスID
echo "PPID: $PPID"              # 親プロセスID
ps -p $$ -o pid,ppid,pgid,sid   # グループ・セッション情報

# プロセスグループIDの確認
ps -eo pid,pgid,cmd | grep nginx

# プロセスグループにシグナル送信
kill -TERM -$(ps -o pgid= -p 1234 | tr -d ' ')
# PGID の前にマイナスをつけて、グループ全体にシグナル送信
```

### 6.2 フォアグラウンド/バックグラウンドプロセスグループ

```bash
# 端末には1つのフォアグラウンドプロセスグループと
# 0個以上のバックグラウンドプロセスグループがある

# フォアグラウンドプロセスグループ:
# - 端末からの入力を受け取れる
# - Ctrl+C, Ctrl+Z のシグナルを受け取る
# - 1つの端末に1つだけ

# バックグラウンドプロセスグループ:
# - 端末からの入力を受け取れない（試みるとSIGTTIN/SIGTTOUで停止）
# - Ctrl+C, Ctrl+Z のシグナルを受け取らない
# - 複数存在可能

# fg/bg はフォアグラウンドプロセスグループの切り替え
```

---

## 7. 実践パターン集

### 7.1 暴走プロセスの対処

```bash
# CPU 90%以上のプロセスを表示
ps aux --sort=-%cpu | awk 'NR<=1 || $3>90'

# 確認してから kill
kill $(ps aux --sort=-%cpu | awk 'NR==2 {print $2}')

# 特定ユーザーの全プロセスを終了（慎重に）
pkill -u problematic_user

# プロセスを段階的に終了
graceful_kill() {
    local pid=$1
    kill -TERM "$pid" 2>/dev/null || return 0
    for i in $(seq 1 10); do
        sleep 1
        kill -0 "$pid" 2>/dev/null || return 0
    done
    kill -KILL "$pid" 2>/dev/null
}
```

### 7.2 全子プロセスの終了

```bash
# 親PIDの子プロセスを全て終了
pkill -P 1234                    # 直接の子プロセスのみ

# 子孫プロセス全体を終了（再帰的）
kill_descendants() {
    local pid=$1
    local children=$(pgrep -P "$pid")
    for child in $children; do
        kill_descendants "$child"
    done
    kill -TERM "$pid" 2>/dev/null
}
kill_descendants 1234

# プロセスグループ全体を終了（より簡単）
kill -TERM -1234                 # PGIDの全プロセスを終了
```

### 7.3 バックグラウンドタスクの管理

```bash
# バックグラウンドタスクの完了を待ち、結果を収集
#!/bin/bash

declare -A TASK_PIDS

# タスク起動
for server in web1 web2 web3 db1 db2; do
    ssh "$server" "uptime" > "/tmp/uptime_${server}.txt" 2>&1 &
    TASK_PIDS[$server]=$!
done

# 結果収集
FAILED=()
for server in "${!TASK_PIDS[@]}"; do
    pid=${TASK_PIDS[$server]}
    if wait "$pid"; then
        echo "OK: $server - $(cat /tmp/uptime_${server}.txt)"
    else
        echo "FAIL: $server"
        FAILED+=("$server")
    fi
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "失敗したサーバー: ${FAILED[*]}"
    exit 1
fi
```

### 7.4 タイムアウト付きリトライ

```bash
#!/bin/bash
# retry_with_timeout.sh - タイムアウト付きリトライ

retry_command() {
    local max_retries=${1:-3}
    local timeout_sec=${2:-30}
    local retry_delay=${3:-5}
    shift 3

    local attempt=0
    while [ $attempt -lt $max_retries ]; do
        attempt=$((attempt + 1))
        echo "試行 $attempt/$max_retries: $*"

        if timeout "$timeout_sec" "$@"; then
            echo "成功 (試行 $attempt)"
            return 0
        fi

        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo "タイムアウト (${timeout_sec}秒)"
        else
            echo "失敗 (終了コード: $exit_code)"
        fi

        if [ $attempt -lt $max_retries ]; then
            echo "${retry_delay}秒後にリトライ..."
            sleep "$retry_delay"
        fi
    done

    echo "全 $max_retries 回失敗"
    return 1
}

# 使用例
retry_command 3 10 5 curl -s https://api.example.com/health
```

### 7.5 デーモン化スクリプト

```bash
#!/bin/bash
# simple_daemon.sh - シンプルなデーモン化スクリプト

PIDFILE="/var/run/myapp.pid"
LOGFILE="/var/log/myapp.log"

start() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "既に実行中 (PID: $(cat "$PIDFILE"))"
        return 1
    fi

    echo "起動中..."
    nohup /usr/local/bin/myapp > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "起動完了 (PID: $(cat "$PIDFILE"))"
}

stop() {
    if [ ! -f "$PIDFILE" ]; then
        echo "PIDファイルがありません"
        return 1
    fi

    local pid=$(cat "$PIDFILE")
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "プロセスが存在しません (PID: $pid)"
        rm -f "$PIDFILE"
        return 1
    fi

    echo "停止中 (PID: $pid)..."
    kill "$pid"

    local timeout=30
    while [ $timeout -gt 0 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        timeout=$((timeout - 1))
    done

    if kill -0 "$pid" 2>/dev/null; then
        echo "SIGKILL で強制停止..."
        kill -9 "$pid"
    fi

    rm -f "$PIDFILE"
    echo "停止完了"
}

status() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "実行中 (PID: $(cat "$PIDFILE"))"
    else
        echo "停止中"
        [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
    fi
}

restart() {
    stop
    sleep 2
    start
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    restart) restart ;;
    status)  status ;;
    *)       echo "使い方: $0 {start|stop|restart|status}" ;;
esac
```

---

## まとめ

| 操作 | コマンド | 備考 |
|------|---------|------|
| バックグラウンド実行 | command & | PID は $! で取得 |
| 一時停止 | Ctrl+Z | SIGTSTP 送信 |
| フォアグラウンドに戻す | fg %N | |
| バックグラウンドで再開 | bg %N | |
| 正常終了要求 | kill PID | SIGTERM（デフォルト） |
| 強制終了（最終手段） | kill -9 PID | SIGKILL（クリーンアップなし） |
| 設定再読み込み | kill -HUP PID | SIGHUP |
| プロセス名で終了 | pkill -f "pattern" | 正規表現対応 |
| 切断後も継続（事前） | nohup command & | 出力は nohup.out |
| 切断後も継続（事後） | disown %N | ジョブリストから除外 |
| シグナル捕捉 | trap 'handler' SIGNAL | EXIT が最重要 |
| 全ジョブ完了待ち | wait | スクリプトの並列処理 |
| タイムアウト付き実行 | timeout 60 command | 124=タイムアウト |

---

## 次に読むべきガイド
→ [[../04-networking/00-curl-wget.md]] — ネットワークツール

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.8-9, O'Reilly, 2022.
2. Shotts, W. "The Linux Command Line." Ch.10-11, No Starch Press, 2019.
3. Cooper, M. "Advanced Bash-Scripting Guide." Ch.15 (Signals), tldp.org.
4. Kerrisk, M. "The Linux Programming Interface." Ch.20-22 (Signals), No Starch Press, 2010.
5. "signal(7) — Linux manual page." man7.org.
