# プロセス監視（ps, top, htop）

> プロセスの状態を把握することは、トラブルシューティングの第一歩。

## この章で学ぶこと

- [ ] ps でプロセス一覧を取得・フィルタリングできる
- [ ] top / htop でリアルタイム監視ができる
- [ ] プロセスの状態・リソース使用量を読み解ける

---

## 1. ps — プロセスのスナップショット

```bash
# 基本: ps [オプション]（BSD形式とUNIX形式がある）

# よく使うパターン
ps aux                          # 全プロセス表示（BSD形式）
ps -ef                          # 全プロセス表示（UNIX形式）
ps aux --sort=-%mem             # メモリ使用量順
ps aux --sort=-%cpu             # CPU使用量順
ps -p 1234                      # 特定PIDのプロセス
ps -u gaku                      # 特定ユーザーのプロセス
ps -C nginx                     # コマンド名で検索
ps axjf                         # ツリー表示（プロセス階層）

# 出力列の意味
# USER   PID  %CPU %MEM    VSZ   RSS TTY  STAT START   TIME COMMAND
# root     1   0.0  0.1 169344 13256 ?    Ss   Jan01   0:15 /sbin/init
#
# USER:  実行ユーザー
# PID:   プロセスID
# %CPU:  CPU使用率
# %MEM:  メモリ使用率
# VSZ:   仮想メモリサイズ（KB）
# RSS:   実メモリ使用量（KB）
# TTY:   制御端末（?=デーモン）
# STAT:  プロセス状態
# TIME:  累積CPU時間

# STAT の読み方
# S: スリープ（待機中）  R: 実行中  D: I/O待ち（kill不可）
# Z: ゾンビ              T: 停止    I: アイドル（カーネルスレッド）
# 追加フラグ:
# s: セッションリーダー  l: マルチスレッド  +: フォアグラウンド
# <: 高優先度            N: 低優先度
```

### カスタム出力

```bash
# 特定の列だけ表示
ps -eo pid,ppid,user,%cpu,%mem,stat,cmd --sort=-%cpu | head -20

# 特定プロセスの詳細
ps -p 1234 -o pid,ppid,%cpu,%mem,etime,cmd

# etime: 経過時間（プロセスの稼働時間）
```

### パイプとの組み合わせ

```bash
# nginx プロセスの検索
ps aux | grep nginx | grep -v grep

# pgrep の方がスマート
pgrep -la nginx                  # PID + コマンドライン
pgrep -u root -l                 # rootのプロセス一覧
pgrep -c nginx                   # プロセス数のカウント

# プロセスの親子関係
pstree -p                        # PID付きツリー
pstree -p 1234                   # 特定プロセスの子孫
```

---

## 2. top — リアルタイム監視

```bash
top                              # 基本起動

# top 画面の構成
# ┌─────────────────────────────────────────────┐
# │ top - 14:30:00 up 30 days, load average: ... │  ← システム概要
# │ Tasks: 256 total, 1 running, 255 sleeping    │  ← プロセス数
# │ %Cpu(s): 3.2 us, 1.1 sy, 0.0 ni, 95.5 id   │  ← CPU使用率
# │ MiB Mem:  16384.0 total, 8192.0 free, ...    │  ← メモリ
# │ MiB Swap:  8192.0 total, 8192.0 free, ...    │  ← スワップ
# ├─────────────────────────────────────────────┤
# │  PID USER  PR  NI  VIRT  RES  SHR S %CPU .. │  ← プロセス一覧
# └─────────────────────────────────────────────┘

# CPU行の読み方
# us: ユーザー空間    sy: カーネル空間    ni: nice値変更
# id: アイドル        wa: I/O待ち         hi: ハードウェア割込
# si: ソフト割込      st: 仮想化スチール

# top の操作キー
# P: CPU順ソート     M: メモリ順ソート
# T: 時間順ソート    N: PID順ソート
# k: プロセスkill    r: nice値変更
# 1: CPU個別表示     c: コマンドフルパス
# f: 表示列選択      H: スレッド表示
# q: 終了

# バッチモード（スクリプト用）
top -bn1 | head -20              # 1回だけ出力
top -bn1 -o %MEM | head -20     # メモリ順で1回出力
```

### load average の読み方

```
load average: 1.50, 2.00, 1.80
              ↑     ↑     ↑
              1分   5分   15分

意味: 「CPUを待っている（実行中+実行待ち）プロセスの平均数」

判断基準（CPUコア数との比較）:
  4コアマシンの場合:
    load avg < 4.0  → 正常（余裕あり）
    load avg = 4.0  → フル稼働（ギリギリ）
    load avg > 4.0  → 過負荷（待ち発生）
    load avg > 8.0  → 深刻な過負荷

  コア数の確認:
    nproc                         # コア数
    lscpu | grep "^CPU(s):"      # 詳細
```

---

## 3. htop — モダンなプロセスモニタ

```bash
# インストール
# macOS: brew install htop
# Ubuntu: sudo apt install htop

htop                             # 基本起動

# htop の画面構成
# ┌─────────────────────────────────────┐
# │ CPU [||||||||       25%]  Mem [|||  ]│  ← グラフィカルなメーター
# │ CPU [||             8%]   Swp [     ]│
# ├─────────────────────────────────────┤
# │  PID USER  PRI  NI  VIRT  RES  ...  │  ← プロセス一覧
# │  → ツリー表示、カラー、マウス対応   │
# ├─────────────────────────────────────┤
# │ F1Help F2Setup F3Search F5Tree F9Kill│  ← ファンクションキー
# └─────────────────────────────────────┘

# htop の操作キー
# F1: ヘルプ         F2: 設定
# F3: 検索           F4: フィルタ
# F5: ツリー表示     F6: ソート列選択
# F9: シグナル送信   F10: 終了
# Space: マーク      U: ユーザーフィルタ
# t: ツリー切替      H: ユーザースレッド表示
# /: インクリメンタル検索

# コマンドラインオプション
htop -u gaku                     # ユーザーフィルタ
htop -p 1234,5678                # 特定PIDのみ
htop -t                          # ツリーモードで起動
htop -d 10                       # 更新間隔1秒（単位: 1/10秒）
```

---

## 4. その他の監視ツール

```bash
# glances — 統合システムモニタ
# brew install glances
glances                          # CPU + メモリ + ディスク + ネットワーク

# btop — 美しいリソースモニタ
# brew install btop
btop                             # グラフィカルな表示

# /proc ファイルシステム（Linux）
cat /proc/1234/status            # プロセスの詳細情報
cat /proc/1234/cmdline           # コマンドライン
cat /proc/1234/fd                # 開いているファイル
ls -l /proc/1234/fd              # ファイルディスクリプタ一覧
cat /proc/loadavg                # ロードアベレージ
cat /proc/meminfo                # メモリ情報
cat /proc/cpuinfo                # CPU情報

# lsof — 開いているファイル/ソケット一覧
lsof -p 1234                     # 特定PIDが開いているファイル
lsof -i :8080                    # ポート8080を使っているプロセス
lsof -u gaku                     # ユーザーが開いているファイル
lsof +D /var/log                 # ディレクトリ内のファイルを開いているプロセス
```

---

## 5. 実践パターン

```bash
# パターン1: CPUを大量消費しているプロセスを特定
ps aux --sort=-%cpu | head -5
# または
top -bn1 -o %CPU | head -15

# パターン2: メモリリークの疑いがあるプロセスを監視
watch -n 5 'ps -p 1234 -o pid,rss,vsz,%mem,etime'
# 5秒ごとにメモリ使用量を表示

# パターン3: ゾンビプロセスの発見
ps aux | awk '$8 ~ /Z/ {print}'
# または
ps -eo pid,ppid,stat,cmd | grep -E "^.*Z"

# パターン4: ポートを使用しているプロセスを特定
lsof -i :3000                    # macOS / Linux
ss -tlnp | grep 3000             # Linux（ssの方が高速）
netstat -tlnp | grep 3000        # 古い方法

# パターン5: プロセスのリソース使用量を継続記録
while true; do
  echo "$(date +%H:%M:%S) $(ps -p 1234 -o %cpu,%mem,rss --no-headers)"
  sleep 10
done >> /tmp/process_monitor.log
```

---

## まとめ

| コマンド | 用途 |
|---------|------|
| ps aux | プロセス一覧（スナップショット） |
| ps aux --sort=-%cpu | CPU順でソート |
| pgrep -la name | プロセス名で検索 |
| top | リアルタイム監視（標準） |
| htop | リアルタイム監視（高機能） |
| lsof -i :port | ポート使用プロセス特定 |

---

## 次に読むべきガイド
→ [[01-jobs-signals.md]] — ジョブ制御とシグナル

---

## 参考文献
1. Barrett, D. "Efficient Linux at the Command Line." Ch.8, O'Reilly, 2022.
