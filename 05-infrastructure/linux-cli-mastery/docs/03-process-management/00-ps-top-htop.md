# プロセス監視（ps, top, htop）

> プロセスの状態を把握することは、トラブルシューティングの第一歩。
> サーバーの負荷分析、メモリリークの検出、異常プロセスの特定 — すべてはプロセス監視から始まる。

## この章で学ぶこと

- [ ] ps でプロセス一覧を取得・フィルタリングできる
- [ ] top / htop でリアルタイム監視ができる
- [ ] プロセスの状態・リソース使用量を読み解ける
- [ ] pgrep / pstree でプロセスの検索・構造把握ができる
- [ ] /proc ファイルシステムからプロセス情報を取得できる
- [ ] 監視スクリプトを作成して自動化できる

---

## 1. ps — プロセスのスナップショット

### 1.1 基本的な使い方

```bash
# ps はプロセスの「ある瞬間」のスナップショットを取得するコマンド
# BSD形式とUNIX（System V）形式の2種類の書式がある

# BSD形式（ダッシュなし）
ps aux                          # 全プロセス表示
ps axjf                         # ツリー表示（プロセス階層）

# UNIX形式（ダッシュあり）
ps -ef                          # 全プロセス表示
ps -eF                          # 拡張フォーマットで全プロセス

# 違い:
# ps aux → USER, PID, %CPU, %MEM, VSZ, RSS, TTY, STAT, START, TIME, COMMAND
# ps -ef → UID, PID, PPID, C, STIME, TTY, TIME, CMD

# 自分のプロセスのみ
ps u                            # 現在のユーザーの端末に関連するプロセス
ps ux                           # 現在のユーザーの全プロセス

# 特定ユーザーのプロセス
ps -u gaku                      # ユーザー gaku のプロセス
ps -u root                      # root のプロセス
ps -U gaku                      # 実UID で検索
ps -u gaku -f                   # フルフォーマット
```

### 1.2 出力列の詳細解説

```bash
# ps aux の出力例:
# USER   PID  %CPU %MEM    VSZ   RSS TTY  STAT START   TIME COMMAND
# root     1   0.0  0.1 169344 13256 ?    Ss   Jan01   0:15 /sbin/init
# gaku  1234   5.2  2.3 524288 37120 pts/0 Sl+  14:30  0:42 node server.js

# 各列の詳細:
# USER:    プロセスの実効ユーザー
# PID:     プロセスID（一意の識別子）
# %CPU:    CPU使用率（プロセスのライフタイムにおける平均）
# %MEM:    物理メモリ使用率
# VSZ:     仮想メモリサイズ（KB）— プロセスがアクセスできる全メモリ空間
# RSS:     常駐セットサイズ（KB）— 実際に物理メモリに存在するサイズ
# TTY:     制御端末（? = デーモン、pts/N = 疑似端末）
# STAT:    プロセス状態（後述の詳細参照）
# START:   開始時刻（24時間以内は時刻、それ以外は日付）
# TIME:    累積CPU時間（プロセスが実際にCPUを使用した合計時間）
# COMMAND: 実行コマンド

# VSZ と RSS の違い:
#   VSZ（Virtual Size）: mmap されたファイル、共有ライブラリ、未使用の割当メモリも含む
#   RSS（Resident Set Size）: 実際に物理メモリに存在するページのサイズ
#   通常 VSZ >> RSS（VSZが大きくても実害は少ないことが多い）
#   RSS が大きいプロセスが実際のメモリ消費者

# ps -ef の出力例:
# UID        PID  PPID  C STIME TTY          TIME CMD
# root         1     0  0 Jan01 ?        00:00:15 /sbin/init

# PPID: 親プロセスID（このプロセスを生成したプロセス）
# C:    CPU利用率（短期間の数値）
# STIME: 開始時刻
```

### 1.3 STAT（プロセス状態）の完全ガイド

```bash
# STAT フィールドは1文字の基本状態 + 追加フラグで構成される

# === 基本状態（1文字目） ===
# R: Running     — 実行中またはCPU実行キューに入っている
# S: Sleeping    — 割り込み可能なスリープ（I/O完了やシグナルを待機）
# D: Disk sleep  — 割り込み不可のスリープ（I/O待ち）
#                  → kill -9 でも終了できない！ ディスクやNFSの問題が原因
# Z: Zombie      — 終了済みだが親がwait()していない
#                  → 親プロセスのバグが原因。親を終了させれば消える
# T: Stopped     — シグナルで停止（SIGSTOP/SIGTSTP）
# t: Tracing     — デバッガ（strace等）によるトレース中
# I: Idle        — カーネルのアイドルスレッド（Linux 4.14+）
# X: Dead        — 表示されることはない（終了処理中の一瞬）

# === 追加フラグ（2文字目以降） ===
# s: セッションリーダー（ログインシェルなど）
# l: マルチスレッド
# +: フォアグラウンドプロセスグループのメンバー
# <: 高優先度（nice値が負）
# N: 低優先度（nice値が正）
# L: メモリ内にロックされたページがある
# W: スワップアウトされている（Linux 2.6以降では使われない）

# よく見るSTATの組み合わせと意味:
# Ss   → スリープ中のセッションリーダー（sshd, init など）
# Ssl  → スリープ中のセッションリーダー、マルチスレッド（systemd など）
# R+   → 実行中のフォアグラウンドプロセス
# S+   → スリープ中のフォアグラウンドプロセス（vim, less など）
# Sl   → スリープ中のマルチスレッドプロセス（Java, Node.js など）
# S<   → 高優先度でスリープ中
# SN   → 低優先度でスリープ中
# Z+   → ゾンビ状態のフォアグラウンドプロセス
# D+   → I/O待ちのフォアグラウンドプロセス

# STATで状態を絞り込む
ps aux | awk '$8 ~ /Z/'          # ゾンビプロセスのみ
ps aux | awk '$8 ~ /D/'          # I/O待ちプロセスのみ（ディスク問題の兆候）
ps aux | awk '$8 ~ /R/'          # 実行中のプロセスのみ
ps aux | awk '$8 ~ /T/'          # 停止中のプロセスのみ
```

### 1.4 カスタム出力（-o / --format）

```bash
# 特定の列だけ表示（-o / --format）
ps -eo pid,ppid,user,%cpu,%mem,stat,cmd --sort=-%cpu | head -20

# よく使うカスタムフォーマット
ps -eo pid,ppid,user,%cpu,%mem,rss,vsz,stat,etime,cmd --sort=-%mem | head -20

# 利用可能なフォーマットキーワード（主要なもの）
# pid     プロセスID
# ppid    親プロセスID
# pgid    プロセスグループID
# sid     セッションID
# uid     ユーザーID
# user    ユーザー名
# gid     グループID
# group   グループ名
# %cpu    CPU使用率
# %mem    メモリ使用率
# rss     常駐メモリサイズ（KB）
# vsz     仮想メモリサイズ（KB）
# sz      物理ページ数
# stat    プロセス状態
# state   プロセス状態（1文字）
# pri     優先度
# ni      nice値
# tty     制御端末
# time    累積CPU時間
# etime   経過時間（プロセス起動からの時間）
# etimes  経過時間（秒数）
# cmd     コマンド（引数なし）
# args    コマンド（引数付き）
# comm    コマンド名のみ
# wchan   カーネル関数名（待機中の場所）
# lstart  起動時刻（詳細形式）
# nlwp    スレッド数

# 特定プロセスの詳細
ps -p 1234 -o pid,ppid,%cpu,%mem,rss,etime,lstart,cmd
# etime:  経過時間（DD-HH:MM:SS形式）
# lstart: 起動時刻（Wed Jan 15 14:30:00 2024形式）

# カスタムヘッダー
ps -eo pid=PID,user=USER,%cpu=CPU,%mem=MEM,cmd=COMMAND --sort=-%cpu | head -10

# ヘッダーなし
ps -eo pid,%cpu,%mem,cmd --sort=-%cpu --no-headers | head -10

# スレッド表示
ps -eLo pid,tid,user,%cpu,%mem,cmd | head -20
# -L: スレッドを個別行で表示
# tid: スレッドID

# プロセスのnice値とスケジューリング情報
ps -eo pid,ni,pri,cls,cmd --sort=-ni | head -20
# ni:  nice値（-20〜19、低いほど高優先度）
# pri: カーネル内部優先度
# cls: スケジューリングクラス（TS=タイムシェア、FF=FIFO、RR=ラウンドロビン）
```

### 1.5 ソートオプション

```bash
# ソート指定（--sort）
ps aux --sort=-%cpu                 # CPU使用率の高い順（降順）
ps aux --sort=-%mem                 # メモリ使用率の高い順
ps aux --sort=-rss                  # 常駐メモリの大きい順
ps aux --sort=-vsz                  # 仮想メモリの大きい順
ps aux --sort=start_time            # 起動時刻の古い順
ps aux --sort=-start_time           # 起動時刻の新しい順
ps aux --sort=pid                   # PID順（昇順）
ps aux --sort=-etime               # 経過時間の長い順

# 複数キーでのソート
ps aux --sort=-%cpu,-%mem           # CPU順、同値ならメモリ順

# 特定のコマンドでフィルタ
ps -C nginx                         # コマンド名で検索
ps -C nginx -o pid,%cpu,%mem,cmd    # フォーマット指定付き
ps -C nginx,node,python -o pid,cmd  # 複数コマンド名
```

### 1.6 パイプとの組み合わせ

```bash
# nginx プロセスの検索（grep パターン）
ps aux | grep nginx | grep -v grep
# grep -v grep: grep 自身を除外

# pgrep の方がスマート（推奨）
pgrep -la nginx                  # PID + コマンドライン全体
pgrep -l nginx                   # PID + プロセス名
pgrep -u root -l                 # rootのプロセス一覧
pgrep -c nginx                   # プロセス数のカウント
pgrep -f "node.*server"          # コマンドライン全体で正規表現マッチ
pgrep -P 1234                    # 親PID 1234 の子プロセス
pgrep -n nginx                   # 最新のnginxプロセスのPID
pgrep -o nginx                   # 最古のnginxプロセスのPID
pgrep -x nginx                   # 完全一致（部分一致を防ぐ）

# pidof（完全一致のPID取得）
pidof nginx                      # 全プロセスのPID（スペース区切り）
pidof -s nginx                   # 1つだけ取得

# プロセスの親子関係
pstree -p                        # PID付きツリー
pstree -p 1234                   # 特定プロセスの子孫
pstree -u                        # UID変更を表示
pstree -a                        # コマンドライン引数も表示
pstree -h                        # カレントプロセスをハイライト
pstree -H 1234                   # 指定PIDをハイライト
pstree -s 1234                   # 指定PIDの祖先を表示
pstree -c                        # 同一プロセスを折りたたまない
pstree -g                        # プロセスグループIDを表示

# 実用パターン: 特定サービスのプロセスツリー
pstree -p $(pgrep -o nginx)      # nginx のマスタープロセスからのツリー
```

---

## 2. top — リアルタイム監視

### 2.1 画面構成の詳細解説

```bash
top                              # 基本起動

# top 画面の構成（5つのセクション）
# ┌──────────────────────────────────────────────────────────┐
# │ top - 14:30:00 up 30 days, 2:15, 3 users, load avg: ... │ ← (1) サマリ行
# │ Tasks: 256 total, 1 running, 254 sleeping, 1 stopped    │ ← (2) タスク行
# │ %Cpu(s): 3.2 us, 1.1 sy, 0.0 ni, 95.5 id, 0.1 wa...   │ ← (3) CPU行
# │ MiB Mem:  16384.0 total,  8192.0 free,  4096.0 used..  │ ← (4) メモリ行
# │ MiB Swap:  8192.0 total,  8192.0 free,     0.0 used..  │ ← (5) スワップ行
# ├──────────────────────────────────────────────────────────┤
# │  PID USER  PR  NI    VIRT    RES    SHR S  %CPU  %MEM.. │ ← プロセス一覧
# └──────────────────────────────────────────────────────────┘
```

### 2.2 サマリ行の読み方

```
(1) サマリ行:
top - 14:30:00 up 30 days, 2:15, 3 users, load average: 1.50, 2.00, 1.80
      ↑          ↑                ↑         ↑      ↑      ↑     ↑
      現在時刻    稼働時間          ユーザ数   1分平均  5分平均  15分平均

(2) タスク行:
Tasks: 256 total, 1 running, 254 sleeping, 0 stopped, 1 zombie
       ↑          ↑           ↑             ↑          ↑
       総数       実行中      スリープ中     停止中     ゾンビ
       ※ zombie > 0 の場合は親プロセスの問題を調査すべき

(3) CPU行:
%Cpu(s):  3.2 us,  1.1 sy,  0.0 ni, 95.5 id,  0.1 wa,  0.0 hi,  0.1 si,  0.0 st
          ↑        ↑        ↑        ↑         ↑        ↑        ↑        ↑
          user     system   nice     idle      iowait   hw-irq   sw-irq   steal

各値の意味:
  us (user):     ユーザー空間のプロセス（nice値変更なし）
  sy (system):   カーネル空間の処理
  ni (nice):     nice値を変更したユーザープロセス
  id (idle):     アイドル（何もしていない）
  wa (iowait):   I/O 完了待ち ← ディスクボトルネックの指標
  hi (hardware): ハードウェア割り込み処理
  si (software): ソフトウェア割り込み処理（ネットワーク処理など）
  st (steal):    仮想化環境でホストに奪われた時間 ← クラウドで重要

注目すべきパターン:
  wa が高い → ディスクI/O がボトルネック（SSD化、I/Oスケジューラ調整を検討）
  sy が高い → カーネル処理が多い（コンテキストスイッチ過多、システムコール多発）
  st が高い → VM のCPU リソース不足（インスタンスタイプの変更を検討）
  us が高い → アプリケーションがCPUを消費（プロファイリングで原因特定）

(4) メモリ行:
MiB Mem:  16384.0 total,   2048.0 free,   8192.0 used,   6144.0 buff/cache
                            ↑              ↑               ↑
                            完全に空き      プロセス使用     バッファ/キャッシュ

  ※ Linux はメモリをキャッシュに積極活用するため、free が少なくても問題ない
  ※ 実際の空き = free + buff/cache の大部分
  ※ 「avail Mem」（利用可能メモリ）がより正確な空き容量

(5) スワップ行:
MiB Swap:  8192.0 total,   8192.0 free,      0.0 used.    10240.0 avail Mem
                                               ↑            ↑
                                               スワップ使用量  利用可能メモリ

  ※ Swap used > 0 が継続的に増加 → メモリ不足の兆候
  ※ avail Mem が物理メモリの10%以下 → メモリ増設を検討
```

### 2.3 プロセス一覧の列

```bash
# プロセス一覧の各列:
#  PID  USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
#  1234 gaku      20   0  524288  37120  15360 S   5.2   2.3   0:42.50 node

# PID:     プロセスID
# USER:    所有者
# PR:      カーネル内部優先度（rt = リアルタイム）
# NI:      nice値（-20〜19）
# VIRT:    仮想メモリ（VSZ相当）
# RES:     常駐メモリ（RSS相当）
# SHR:     共有メモリ（ライブラリなど）
# S:       状態（R/S/D/Z/T）
# %CPU:    CPU使用率（直近の更新間隔での値）
# %MEM:    物理メモリ使用率
# TIME+:   累積CPU時間（1/100秒単位）
# COMMAND: コマンド名

# top と ps の %CPU の違い:
#   top: 直近の更新間隔（デフォルト3秒）における瞬間的なCPU使用率
#   ps:  プロセスのライフタイム全体での平均CPU使用率
#   → top の方がリアルタイムな負荷を反映する
```

### 2.4 top の対話的操作キー

```bash
# top 実行中に使えるキー（覚えるべきもの）

# === ソート ===
# P:  CPU使用率順でソート（デフォルト）
# M:  メモリ使用率順でソート
# T:  累積CPU時間順でソート
# N:  PID順でソート
# R:  現在のソートを逆順に

# === 表示切替 ===
# 1:  CPU をコア別に表示/集約表示（マルチコアの確認に必須）
# c:  コマンド名 / フルコマンドライン切替
# H:  スレッド表示ON/OFF
# V:  ツリー表示（プロセスの親子関係）
# e:  メモリ単位切替（KB→MB→GB→TB→PB）
# E:  サマリ行のメモリ単位切替
# m:  メモリ行の表示形式切替（数値/バー表示）
# t:  タスク/CPU行の表示形式切替
# l:  ロードアベレージ行のON/OFF
# 0:  ゼロ値の表示/非表示

# === フィルタリング ===
# u:  ユーザーでフィルタ（ユーザー名を入力）
# o:  フィルタ条件追加（例: %CPU>10, COMMAND=nginx）
# O:  フィルタ条件追加（大文字小文字区別なし）
# =:  フィルタをクリア

# === アクション ===
# k:  プロセスを kill（PIDとシグナルを入力）
# r:  プロセスの nice 値を変更（renice）
# d:  更新間隔を変更（秒数を入力）
# s:  更新間隔を変更（同上）

# === 設定 ===
# f:  表示列の選択・順序変更
# W:  現在の設定を ~/.toprc に保存
# q:  終了

# フィルタの例:
# o を押して以下を入力:
#   %CPU>5.0          → CPU 5%以上のプロセスのみ
#   COMMAND=java       → java を含むプロセスのみ
#   %MEM>10.0          → メモリ 10%以上
#   USER=gaku          → ユーザー gaku のみ
#   !COMMAND=kworker   → kworker を除外
```

### 2.5 top のコマンドラインオプション

```bash
# バッチモード（スクリプト用）
top -bn1                         # 1回だけ出力して終了
top -bn1 | head -20              # 上位20行のみ
top -bn1 -o %MEM | head -20     # メモリ順で1回出力

# 更新間隔の指定
top -d 1                         # 1秒間隔で更新
top -d 0.5                       # 0.5秒間隔

# 特定ユーザーのプロセスのみ
top -u gaku
top -u root

# 特定PIDのみ監視
top -p 1234                      # 1つのプロセス
top -p 1234,5678,9012            # 複数プロセス

# スレッド表示
top -H                           # スレッドを個別に表示
top -H -p 1234                   # 特定プロセスのスレッドを監視

# セキュア（安全）モード
top -s                           # kill, renice などの操作を無効化

# バッチモードの活用例
# CPUトップ10を記録
top -bn1 -o %CPU | head -17 > /tmp/cpu_snapshot_$(date +%H%M%S).txt

# 5秒間隔で10回記録（50秒分の推移）
top -bn10 -d 5 -o %CPU | head -17 > /tmp/cpu_trend.txt

# 特定プロセスのCPU使用率の推移を記録
while true; do
    echo "$(date +%H:%M:%S) $(top -bn1 -p 1234 | tail -1 | awk '{print $9, $10}')"
    sleep 5
done >> /tmp/process_cpu_trend.log
```

### 2.6 load average の深い理解

```
load average: 1.50, 2.00, 1.80
              ↑     ↑     ↑
              1分   5分   15分

意味: 「実行中(R) + 実行待ち(R in queue) + I/O待ち(D)のプロセスの指数移動平均」

重要: Linux の load average は I/O 待ち（D状態）のプロセスも含む
  → 他のUNIX（FreeBSD等）とは異なる
  → ディスクI/Oが多い環境ではCPUに余裕があっても load が高くなる

判断基準（CPUコア数との比較）:
  コア数の確認:
    nproc                         # コア数
    lscpu | grep "^CPU(s):"      # 詳細
    cat /proc/cpuinfo | grep processor | wc -l  # 論理コア数

  4コアマシンの場合:
    load avg < 4.0  → 正常（余裕あり）
    load avg ≈ 4.0  → フル稼働（ギリギリ）
    load avg > 4.0  → 過負荷（CPUキューに待ちが発生）
    load avg > 8.0  → 深刻な過負荷（応答遅延の可能性）
    load avg > 16.0 → 危機的状況（サービス影響あり）

  load avg の傾向分析:
    1分 > 5分 > 15分  → 負荷が上昇中（要注意）
    1分 < 5分 < 15分  → 負荷が下降中（改善傾向）
    1分 ≈ 5分 ≈ 15分  → 安定状態

  load が高い場合の調査手順:
    1. top の CPU行（us, sy, wa, st）を確認
       - wa が高い → ディスクI/Oが原因
       - us が高い → アプリケーションが原因
       - sy が高い → カーネル処理が原因
       - st が高い → 仮想化リソース不足
    2. ps aux --sort=-%cpu | head -10  でCPU消費プロセスを特定
    3. iostat -x 1  でディスクI/Oを確認
    4. vmstat 1     でシステム全体の状態を確認
```

---

## 3. htop — モダンなプロセスモニタ

### 3.1 インストールと起動

```bash
# インストール
# macOS:
brew install htop

# Ubuntu/Debian:
sudo apt install htop

# CentOS/RHEL:
sudo yum install htop
# または
sudo dnf install htop

# Arch Linux:
sudo pacman -S htop

# 基本起動
htop
```

### 3.2 画面構成

```bash
# htop の画面構成
# ┌──────────────────────────────────────────────┐
# │ CPU[||||||||||||       35%]  Tasks: 142, 1 run│  ← CPUメーター
# │ CPU[|||              12%]   Load: 1.50 2.00  │  ← マルチコア個別表示
# │ CPU[||||||           25%]   Uptime: 30 days  │
# │ CPU[|                 5%]                    │
# │ Mem[|||||||||||||||  4.2G/16.0G]             │  ← メモリメーター
# │ Swp[                 0K/8.0G]               │  ← スワップメーター
# ├──────────────────────────────────────────────┤
# │  PID USER    PRI  NI  VIRT  RES  SHR S CPU% │  ← プロセス一覧
# │  1234 gaku    20   0  512M  36M  15M S  5.2 │     カラー表示
# │  5678 root    20   0  256M  18M  12M S  2.1 │     ツリー表示対応
# │  ...                                         │     マウス操作対応
# ├──────────────────────────────────────────────┤
# │ F1Help F2Setup F3Search F4Filter F5Tree F6So │  ← ファンクションキー
# │ F7Nice- F8Nice+ F9Kill F10Quit              │
# └──────────────────────────────────────────────┘

# CPUメーターの色の意味（デフォルト）
#   緑色:  ユーザープロセス（通常の負荷）
#   赤色:  カーネルプロセス（システム負荷）
#   青色:  低優先度プロセス（nice値が高い）
#   水色:  仮想化スチール時間
#   黄色:  I/O待ち時間

# メモリメーターの色の意味
#   緑色:  使用中メモリ
#   青色:  バッファ
#   黄色:  キャッシュ
```

### 3.3 htop の操作キー

```bash
# === 基本操作 ===
# F1 / h:     ヘルプ画面
# F2 / S:     設定画面（メーター、カラー、表示列などを変更）
# F3 / /:     インクリメンタル検索
# F4 / \:     フィルタ（表示するプロセスを文字列でフィルタ）
# F5 / t:     ツリー表示ON/OFF
# F6 / >:     ソート列の選択
# F7 / ]:     nice値を下げる（優先度を上げる）
# F8 / [:     nice値を上げる（優先度を下げる）
# F9 / k:     シグナル送信（プロセスをkill）
# F10 / q:    終了

# === 表示操作 ===
# Space:      プロセスをマーク（複数選択）
# c:          マークしたプロセスにタグ付け
# U:          全マーク解除
# u:          ユーザーでフィルタ
# H:          ユーザースレッドの表示/非表示
# K:          カーネルスレッドの表示/非表示
# p:          プロセスのフルパス表示
# m:          メモリソートの切替
# T:          CPU時間ソート

# === 高度な操作 ===
# l:          プロセスが開いているファイル一覧（lsof）
# s:          プロセスのシステムコール追跡（strace）
# e:          プロセスの環境変数表示
# w:          プロセスを /proc/PID/wchan で確認
# i:          プロセスのI/O情報
# M:          ライブラリマッピング表示（メモリマップ）

# === 検索とフィルタの違い ===
# F3 (検索): プロセス名でカーソルを移動（次を検索: F3 を再度押す）
# F4 (フィルタ): マッチするプロセスのみ表示（他は非表示）
# → フィルタの方が見やすい
```

### 3.4 htop のコマンドラインオプション

```bash
# ユーザーフィルタ
htop -u gaku                     # ユーザー gaku のプロセスのみ

# 特定PIDのみ
htop -p 1234,5678                # 指定PIDのみ表示

# ツリーモードで起動
htop -t                          # ツリー表示をデフォルトに

# 更新間隔（単位: 1/10秒）
htop -d 10                       # 1秒間隔（10 × 0.1秒）
htop -d 50                       # 5秒間隔

# ソート列を指定して起動
htop --sort-key=PERCENT_CPU      # CPU使用率順
htop --sort-key=PERCENT_MEM      # メモリ使用率順
htop --sort-key=M_RESIDENT       # RSS順

# モノクロ表示
htop -C                          # カラーなし

# 遅延カウントを指定
htop --delay=20                  # 2秒間隔
```

### 3.5 htop の設定（F2）

```bash
# F2 で設定画面を開く

# Meters（メーター設定）
# ヘッダー部分に表示するメーターを追加・削除・配置変更
# 追加可能なメーター:
#   - CPU使用率（全体/個別）
#   - メモリ使用率
#   - スワップ使用率
#   - タスク数
#   - ロードアベレージ
#   - 稼働時間
#   - バッテリー
#   - ホスト名
#   - クロック
#   - ディスクI/O
#   - ネットワークI/O

# Display options（表示オプション）
# - ツリー表示
# - シャドウ表示
# - カウント表示
# - プロセスパスの表示方法

# Colors（カラースキーム）
# - デフォルト
# - モノクロ
# - ブラック・オン・ホワイト
# - Light Terminal
# - MC

# Columns（表示列設定）
# プロセス一覧に表示する列を追加・削除・順序変更
# 設定は ~/.config/htop/htoprc に保存される
```

---

## 4. その他の監視ツール

### 4.1 glances — 統合システムモニタ

```bash
# インストール
pip install glances
# または
brew install glances                # macOS
sudo apt install glances            # Ubuntu

# 基本起動
glances

# glances の特徴:
# - CPU + メモリ + ディスク + ネットワーク + プロセスを一画面で表示
# - アラート機能（閾値超過で色が変わる）
# - Web UI モード
# - API モード（RESTful API でデータ取得）
# - CSV/JSON エクスポート

# Web UI モード（リモートからブラウザで監視）
glances -w                        # http://localhost:61208 で接続
glances -w --bind 0.0.0.0         # 全インターフェースでリッスン

# クライアント/サーバーモード
glances -s                        # サーバーとして起動
glances -c server-ip              # クライアントとして接続

# CSV エクスポート
glances --export csv --export-csv-file /tmp/glances.csv

# JSON エクスポート
glances --stdout cpu.total,mem.percent,load
```

### 4.2 btop — 美しいリソースモニタ

```bash
# インストール
brew install btop                 # macOS
sudo apt install btop             # Ubuntu 22.04+
sudo snap install btop            # Snap

# 基本起動
btop

# btop の特徴:
# - 美しいグラフィカル表示（CPU/メモリ/ネットワーク/ディスクのグラフ）
# - マウス操作対応
# - テーマ変更可能
# - プロセスのフィルタリング/ソート
# - 設定ファイル: ~/.config/btop/btop.conf
```

### 4.3 /proc ファイルシステム（Linux）

```bash
# /proc はカーネルが提供する仮想ファイルシステム
# プロセスとシステムの詳細情報にアクセスできる

# === プロセス別情報（/proc/PID/） ===

# プロセスの詳細情報
cat /proc/1234/status
# Name:   nginx
# State:  S (sleeping)
# Tgid:   1234
# Pid:    1234
# PPid:   1
# VmPeak: 524288 kB    ← 仮想メモリのピーク値
# VmSize: 524288 kB    ← 現在の仮想メモリサイズ
# VmRSS:  37120 kB     ← 常駐メモリサイズ
# VmSwap: 0 kB         ← スワップされたサイズ
# Threads: 4           ← スレッド数

# コマンドライン
cat /proc/1234/cmdline | tr '\0' ' '
# /usr/sbin/nginx -g daemon off;

# 環境変数
cat /proc/1234/environ | tr '\0' '\n'
# HOME=/root
# PATH=/usr/local/sbin:/usr/local/bin:...

# ファイルディスクリプタ一覧
ls -l /proc/1234/fd
# lrwx------ 1 root root 64 ... 0 -> /dev/null
# l-wx------ 1 root root 64 ... 1 -> /var/log/nginx/access.log
# l-wx------ 1 root root 64 ... 2 -> /var/log/nginx/error.log
# lrwx------ 1 root root 64 ... 3 -> socket:[12345]

# ファイルディスクリプタ数
ls /proc/1234/fd | wc -l

# メモリマップ
cat /proc/1234/maps | head -20
# 各行: 開始-終了 パーミッション オフセット デバイス inode パス名

# プロセスのリソース制限
cat /proc/1234/limits
# Max open files      65536     65536     files

# プロセスのI/O統計
cat /proc/1234/io
# rchar:  読み取りバイト数
# wchar:  書き込みバイト数
# read_bytes:  実際のディスク読み取り
# write_bytes: 実際のディスク書き込み

# プロセスの cgroup 情報
cat /proc/1234/cgroup

# === システム全体の情報 ===
cat /proc/loadavg                # ロードアベレージ
cat /proc/meminfo                # メモリ情報の詳細
cat /proc/cpuinfo                # CPU情報
cat /proc/uptime                 # 稼働時間（秒）
cat /proc/version                # カーネルバージョン
cat /proc/stat                   # CPU統計
cat /proc/diskstats              # ディスクI/O統計
cat /proc/net/dev                # ネットワークI/O統計
```

### 4.4 lsof — 開いているファイル/ソケット

```bash
# lsof (List Open Files): プロセスが開いているファイルを一覧表示
# Linuxでは「すべてがファイル」→ ソケット、パイプも対象

# 特定PIDが開いているファイル
lsof -p 1234

# ポートを使っているプロセス
lsof -i :8080                    # ポート8080
lsof -i :80,443                  # 複数ポート
lsof -i TCP:3000                 # TCPのポート3000
lsof -i UDP:53                   # UDPのポート53
lsof -i TCP                      # 全TCP接続
lsof -i -P -n                   # 名前解決しない（高速）

# ユーザーが開いているファイル
lsof -u gaku
lsof -u gaku -c python           # ユーザー + コマンド名

# ディレクトリ内のファイルを開いているプロセス
lsof +D /var/log                 # /var/log 内のファイル
lsof +d /var/log                 # /var/log 直下のみ（再帰なし）

# 削除されたがまだ開いているファイル（ディスク解放されない原因）
lsof +L1
# 解決策: プロセスを再起動するか、/proc/PID/fd/N を truncate

# ネットワーク接続の確認
lsof -i -P -n | grep LISTEN     # リッスンしているポート一覧
lsof -i -P -n | grep ESTABLISHED # 確立済み接続一覧

# 特定ファイルを開いているプロセス
lsof /var/log/syslog             # 特定ファイル

# NFS のロック問題調査
lsof -N                          # NFSファイルを開いているプロセス
```

### 4.5 vmstat / iostat / mpstat

```bash
# vmstat — 仮想メモリ統計（システム全体の概要）
vmstat 1 5                       # 1秒間隔で5回表示
# procs -----------memory---------- ---swap-- -----io---- -system-- ------cpu-----
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs us sy id wa st
#  1  0      0 8192000 256000 4096000  0    0    10    20  500  800  3  1 95  1  0

# r: 実行待ちプロセス数（CPUコア数以上なら過負荷）
# b: I/O待ちプロセス数（D状態）
# si/so: スワップイン/スワップアウト（0以外が継続ならメモリ不足）
# bi/bo: ディスクI/O（ブロック/秒）
# in: 割り込み数/秒
# cs: コンテキストスイッチ数/秒

# iostat — ディスクI/O統計
iostat -x 1 5                    # 拡張統計、1秒間隔で5回
# Device  r/s   w/s  rkB/s  wkB/s  await  %util
# sda     50.0  30.0 2000.0 1500.0  5.00  40.0

# %util: ディスク使用率（100%に近いとボトルネック）
# await: 平均I/O待ち時間（ミリ秒）

# mpstat — CPU別統計
mpstat -P ALL 1 5                # 全CPU、1秒間隔で5回
# 特定のCPUコアだけに負荷が偏っていないか確認
```

---

## 5. 実践パターン

### 5.1 CPUボトルネックの調査

```bash
# ステップ1: 全体像の把握
top -bn1 | head -5
# load average と CPU行を確認

# ステップ2: CPU消費プロセスの特定
ps aux --sort=-%cpu | head -10
# または
top -bn1 -o %CPU | head -15

# ステップ3: 特定プロセスの詳細確認
ps -p 1234 -o pid,ppid,%cpu,%mem,etime,ni,stat,cmd
# etime: どのくらい前から動いているか
# ni: nice値（優先度）

# ステップ4: スレッドレベルの確認
ps -p 1234 -L -o pid,tid,%cpu,cmd | sort -k3 -rn | head -10
# どのスレッドがCPUを消費しているか

# ステップ5: strace でシステムコールを確認（上級）
strace -p 1234 -c -T             # システムコールの統計
strace -p 1234 -e trace=write    # write システムコールのみ
```

### 5.2 メモリリークの調査

```bash
# パターン1: メモリ使用量の継続監視
watch -n 5 'ps -p 1234 -o pid,rss,vsz,%mem,etime'
# 5秒ごとにメモリ使用量を表示
# RSS が時間とともに増加し続ける → メモリリークの疑い

# パターン2: メモリ使用量の記録
while true; do
    echo "$(date +%Y-%m-%d_%H:%M:%S) $(ps -p 1234 -o rss=,vsz=,%mem= --no-headers)"
    sleep 60
done >> /tmp/memory_monitor_1234.log

# パターン3: メモリ上位プロセスの定期記録
while true; do
    echo "=== $(date) ==="
    ps aux --sort=-rss | head -11
    echo ""
    sleep 300
done >> /tmp/memory_top_processes.log

# パターン4: システム全体のメモリ使用量の推移
while true; do
    echo "$(date +%H:%M:%S) $(free -m | awk '/Mem:/ {printf "used:%sMB free:%sMB avail:%sMB", $3, $4, $7}')"
    sleep 10
done >> /tmp/system_memory.log

# パターン5: /proc からの詳細メモリ情報
cat /proc/1234/status | grep -E "^Vm|^Rss|^Threads"
cat /proc/1234/smaps_rollup       # メモリマッピングのサマリ

# パターン6: pmap でメモリマップを確認
pmap -x 1234 | tail -5           # プロセスのメモリマップ
pmap -x 1234 | sort -k3 -rn | head -10  # サイズ順
```

### 5.3 ゾンビプロセスの対処

```bash
# ゾンビプロセスの発見
ps aux | awk '$8 ~ /Z/ {print}'
# または
ps -eo pid,ppid,stat,cmd | grep -E "Z"

# ゾンビの数を確認
ps aux | awk '$8 ~ /Z/' | wc -l

# ゾンビの親プロセスを特定
ps -eo pid,ppid,stat,cmd | awk '$3 ~ /Z/ {print "Zombie PID:", $1, "Parent PID:", $2}'

# 親プロセスの情報を確認
ps -p <親PID> -o pid,cmd,stat

# 対処法1: 親プロセスに SIGCHLD を送る
kill -SIGCHLD <親PID>

# 対処法2: 親プロセスを終了する
kill <親PID>
# 親が終了すると、ゾンビは init(PID 1) に引き取られて自動的にクリーンアップ

# 対処法3: 大量のゾンビの場合
# ゾンビの親PIDを集計
ps -eo ppid,stat | grep Z | awk '{print $1}' | sort | uniq -c | sort -rn

# 注意: ゾンビプロセスは kill -9 では消えない（既に死んでいる）
# ゾンビはプロセステーブルのエントリのみ消費（CPU/メモリはほぼゼロ）
# 少数のゾンビは問題ないが、大量に増え続ける場合は親プロセスのバグ
```

### 5.4 ポート使用状況の調査

```bash
# ポートを使用しているプロセスを特定（複数の方法）

# lsof（macOS/Linux 共通）
lsof -i :3000                    # ポート3000
lsof -i TCP:3000 -P -n           # TCP限定、名前解決なし

# ss（Linux、netstat より高速）
ss -tlnp | grep 3000             # TCPリッスン
ss -tunlp                        # TCP/UDP リッスン全一覧
# -t: TCP  -u: UDP  -l: LISTEN  -n: 数値表示  -p: プロセス表示

# netstat（古い方法、レガシー環境用）
netstat -tlnp | grep 3000

# fuser（ポートを使っているプロセスを直接特定）
fuser 3000/tcp                   # ポート3000/TCPのPID
fuser -v 3000/tcp                # 詳細表示
fuser -k 3000/tcp                # ポート3000/TCPを使っているプロセスをkill
```

### 5.5 包括的なシステム診断スクリプト

```bash
#!/bin/bash
# system_health_check.sh - システムヘルスチェックスクリプト

echo "=============================================="
echo "システムヘルスチェック: $(date)"
echo "ホスト名: $(hostname)"
echo "=============================================="

echo ""
echo "--- ロードアベレージ ---"
uptime
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
echo "CPUコア数: $CORES"
LOAD1=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}' || uptime | awk -F'[,:]' '{print $(NF-2)}' | tr -d ' ')
echo "Load/Core比: $(echo "$LOAD1 / $CORES" | bc -l 2>/dev/null | head -c 5)"

echo ""
echo "--- メモリ使用量 ---"
free -m 2>/dev/null || vm_stat 2>/dev/null
echo ""

echo "--- ディスク使用量 ---"
df -h | grep -vE "tmpfs|devtmpfs|overlay"
echo ""

echo "--- CPU消費プロセス Top5 ---"
ps aux --sort=-%cpu | head -6 2>/dev/null || ps aux | sort -k3 -rn | head -6
echo ""

echo "--- メモリ消費プロセス Top5 ---"
ps aux --sort=-rss | head -6 2>/dev/null || ps aux | sort -k6 -rn | head -6
echo ""

echo "--- ゾンビプロセス ---"
ZOMBIES=$(ps aux 2>/dev/null | awk '$8 ~ /Z/' | wc -l)
echo "ゾンビ数: $ZOMBIES"
if [ "$ZOMBIES" -gt 0 ]; then
    ps aux | awk '$8 ~ /Z/ {print}'
fi

echo ""
echo "--- D状態（I/O待ち）プロセス ---"
DSTATE=$(ps aux 2>/dev/null | awk '$8 ~ /D/' | wc -l)
echo "D状態プロセス数: $DSTATE"
if [ "$DSTATE" -gt 0 ]; then
    ps aux | awk '$8 ~ /D/ {print}'
fi

echo ""
echo "--- リッスンポート ---"
ss -tlnp 2>/dev/null | head -20 || lsof -i -P -n 2>/dev/null | grep LISTEN | head -20

echo ""
echo "--- ESTABLISHED接続数 ---"
ss -tn state established 2>/dev/null | wc -l || netstat -tn 2>/dev/null | grep ESTABLISHED | wc -l

echo ""
echo "--- ディスクI/O ---"
iostat -x 1 1 2>/dev/null | tail -10

echo ""
echo "--- 最近のOOMキル ---"
dmesg 2>/dev/null | grep -i "out of memory\|oom" | tail -5

echo ""
echo "=============================================="
echo "チェック完了"
echo "=============================================="
```

### 5.6 プロセスリソース監視スクリプト

```bash
#!/bin/bash
# process_monitor.sh - 特定プロセスの継続監視
# 使い方: ./process_monitor.sh <PID> [間隔秒] [出力ファイル]

PID="${1:?使い方: $0 <PID> [間隔秒] [出力ファイル]}"
INTERVAL="${2:-10}"
OUTPUT="${3:-/tmp/process_monitor_${PID}.csv}"

if ! kill -0 "$PID" 2>/dev/null; then
    echo "エラー: PID $PID が存在しません" >&2
    exit 1
fi

PROCESS_NAME=$(ps -p "$PID" -o comm= 2>/dev/null)
echo "監視開始: PID=$PID ($PROCESS_NAME), 間隔=${INTERVAL}秒"
echo "出力ファイル: $OUTPUT"
echo "Ctrl+C で停止"
echo ""

# CSV ヘッダー
echo "timestamp,pid,cpu_pct,mem_pct,rss_kb,vsz_kb,threads,fd_count,state" > "$OUTPUT"

trap 'echo ""; echo "監視終了: $(wc -l < "$OUTPUT") レコード記録"; exit 0' INT TERM

while kill -0 "$PID" 2>/dev/null; do
    TIMESTAMP=$(date +%Y-%m-%d_%H:%M:%S)

    # ps からプロセス情報を取得
    PS_DATA=$(ps -p "$PID" -o %cpu=,%mem=,rss=,vsz=,nlwp=,stat= --no-headers 2>/dev/null)
    if [ -z "$PS_DATA" ]; then
        echo "プロセス $PID が終了しました"
        break
    fi

    CPU=$(echo "$PS_DATA" | awk '{print $1}')
    MEM=$(echo "$PS_DATA" | awk '{print $2}')
    RSS=$(echo "$PS_DATA" | awk '{print $3}')
    VSZ=$(echo "$PS_DATA" | awk '{print $4}')
    THREADS=$(echo "$PS_DATA" | awk '{print $5}')
    STATE=$(echo "$PS_DATA" | awk '{print $6}')

    # ファイルディスクリプタ数
    FD_COUNT=$(ls /proc/"$PID"/fd 2>/dev/null | wc -l)

    echo "$TIMESTAMP,$PID,$CPU,$MEM,$RSS,$VSZ,$THREADS,$FD_COUNT,$STATE" >> "$OUTPUT"

    # 画面にもサマリを表示
    printf "\r%s CPU:%s%% MEM:%s%% RSS:%sKB Threads:%s FDs:%s State:%s" \
        "$TIMESTAMP" "$CPU" "$MEM" "$RSS" "$THREADS" "$FD_COUNT" "$STATE"

    sleep "$INTERVAL"
done

echo ""
echo "監視終了: $(wc -l < "$OUTPUT") レコード → $OUTPUT"
```

### 5.7 アラート付き監視スクリプト

```bash
#!/bin/bash
# alert_monitor.sh - 閾値超過時にアラートを出す監視
# 使い方: ./alert_monitor.sh

CPU_THRESHOLD=80     # CPU使用率の閾値（%）
MEM_THRESHOLD=90     # メモリ使用率の閾値（%）
LOAD_THRESHOLD_RATIO=2  # load average / コア数の閾値
CHECK_INTERVAL=30    # チェック間隔（秒）
LOG_FILE="/tmp/alert_monitor.log"

CORES=$(nproc 2>/dev/null || echo 4)
LOAD_THRESHOLD=$(echo "$CORES * $LOAD_THRESHOLD_RATIO" | bc)

log_alert() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ALERT: $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1" >> "$LOG_FILE"
}

echo "監視開始（CPU>${CPU_THRESHOLD}%, MEM>${MEM_THRESHOLD}%, Load>${LOAD_THRESHOLD}）"
echo "ログファイル: $LOG_FILE"

while true; do
    # CPU チェック
    CPU_HOGGERS=$(ps aux --sort=-%cpu --no-headers | awk -v thresh="$CPU_THRESHOLD" '$3 > thresh {print $2, $3"%", $11}')
    if [ -n "$CPU_HOGGERS" ]; then
        log_alert "CPU閾値超過:"
        echo "$CPU_HOGGERS" | while read -r line; do
            log_alert "  $line"
        done
    fi

    # メモリチェック
    MEM_USED_PCT=$(free 2>/dev/null | awk '/Mem:/ {printf "%.0f", $3/$2*100}')
    if [ -n "$MEM_USED_PCT" ] && [ "$MEM_USED_PCT" -gt "$MEM_THRESHOLD" ]; then
        log_alert "メモリ使用率: ${MEM_USED_PCT}%"
        ps aux --sort=-rss --no-headers | head -5 | while read -r line; do
            log_alert "  Top RSS: $(echo "$line" | awk '{print $2, $6"KB", $11}')"
        done
    fi

    # Load Average チェック
    LOAD1=$(cat /proc/loadavg 2>/dev/null | awk '{print $1}')
    if [ -n "$LOAD1" ]; then
        OVER=$(echo "$LOAD1 > $LOAD_THRESHOLD" | bc -l 2>/dev/null)
        if [ "$OVER" = "1" ]; then
            log_alert "Load Average: $LOAD1（閾値: $LOAD_THRESHOLD）"
        fi
    fi

    # ゾンビチェック
    ZOMBIE_COUNT=$(ps aux 2>/dev/null | awk '$8 ~ /Z/' | wc -l)
    if [ "$ZOMBIE_COUNT" -gt 5 ]; then
        log_alert "ゾンビプロセス: ${ZOMBIE_COUNT}個"
    fi

    log_info "チェック完了 (CPU:OK MEM:${MEM_USED_PCT:-?}% Load:${LOAD1:-?})"
    sleep "$CHECK_INTERVAL"
done
```

---

## 6. コマンド比較表

```
┌──────────────┬──────────────┬─────────────┬──────────────────┐
│ 機能         │ ps           │ top         │ htop             │
├──────────────┼──────────────┼─────────────┼──────────────────┤
│ 更新         │ スナップショット │ リアルタイム│ リアルタイム     │
│ 表示形式     │ テキスト     │ TUI         │ カラーTUI        │
│ ソート       │ --sort オプション│ 対話キー    │ 対話キー/マウス  │
│ フィルタ     │ grep/awk     │ u/o キー    │ F4キー           │
│ ツリー表示   │ axjf / f     │ V キー      │ F5キー           │
│ kill         │ 別コマンド   │ k キー      │ F9キー           │
│ カスタマイズ │ -o オプション│ f キー      │ F2設定画面       │
│ スクリプト用 │ 最適         │ -bn1 で可   │ 不向き           │
│ マウス操作   │ なし         │ なし        │ 対応             │
│ スレッド     │ -L / -T      │ H キー      │ H キー           │
│ インストール │ 標準搭載     │ 標準搭載    │ 追加インストール │
└──────────────┴──────────────┴─────────────┴──────────────────┘

使い分けガイド:
  スクリプト・自動化       → ps（出力が安定、パイプに適する）
  対話的なトラブルシュート → htop（最も使いやすい）
  htop がない環境         → top（どこにでもある）
  特定プロセスの詳細調査   → ps -p PID -o ...
  サーバーの定期監視       → top -bn1（cron + ログ記録）
```

---

## まとめ

| コマンド | 用途 | よく使うオプション |
|---------|------|-------------------|
| ps aux | プロセス一覧（スナップショット） | --sort, -o, -p, -u, -C |
| ps aux --sort=-%cpu | CPU順でソート | head -N で上位N件 |
| pgrep -la name | プロセス名で検索 | -f（全コマンドライン）, -c（カウント） |
| pstree -p | プロセスツリー | -s（祖先表示）, -a（引数表示） |
| top | リアルタイム監視（標準） | -bn1（バッチ）, -u（ユーザー）, -p（PID） |
| htop | リアルタイム監視（高機能） | -t（ツリー）, -u（ユーザー）, -p（PID） |
| lsof -i :port | ポート使用プロセス特定 | -P -n（高速化） |
| vmstat 1 | システム全体の統計 | r, b 列に注目 |

### 調査フローチャート

```
サーバーが遅い
  ├→ uptime: load average を確認
  │   ├→ load < コア数 → CPUは余裕あり → アプリ・ネットワーク調査
  │   └→ load > コア数 → CPUボトルネック
  │       ├→ top: us が高い → ps --sort=-%cpu → アプリ最適化
  │       ├→ top: wa が高い → iostat -x → ディスクI/O改善
  │       ├→ top: sy が高い → vmstat → コンテキストスイッチ多発
  │       └→ top: st が高い → VM リソース増強
  ├→ free -m: メモリ確認
  │   ├→ avail > 10% → メモリは余裕あり
  │   └→ avail < 10% → メモリ不足
  │       └→ ps --sort=-rss → メモリ消費プロセス特定
  └→ df -h: ディスク確認
      └→ 使用率 > 90% → 不要ファイル削除 or ディスク拡張
```

---

## 次に読むべきガイド
→ [[01-jobs-signals.md]] — ジョブ制御とシグナル

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.8, O'Reilly, 2022.
2. Gregg, B. "Systems Performance: Enterprise and the Cloud." 2nd Ed, Addison-Wesley, 2020.
3. Evi Nemeth et al. "UNIX and Linux System Administration Handbook." 5th Ed, Addison-Wesley, 2017.
4. "proc(5) — Linux manual page." man7.org.
5. "htop — an interactive process viewer." htop.dev.
