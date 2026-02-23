# プロセス間通信（IPC）

> プロセスは独立したメモリ空間を持つため、データをやり取りするには明示的な通信手段が必要。

## この章で学ぶこと

- [ ] 主要なIPC手法を比較できる
- [ ] パイプ、ソケット、共有メモリの使い分けを理解する
- [ ] シグナルの仕組みを知る
- [ ] メッセージキューとセマフォの実装を理解する
- [ ] 実務で適切なIPC手法を選択できる
- [ ] 同期と排他制御の基本を説明できる

---

## 1. IPC手法の全体像

### 1.1 なぜIPCが必要か

```
プロセスの独立性:
  各プロセスは独立した仮想アドレス空間を持つ
  → プロセスAがアドレス0x1000に書き込んでも
    プロセスBのアドレス0x1000には影響しない
  → セキュリティと安定性の基盤

  しかし、プロセス間でデータをやり取りする必要がある場面は多い:
  - シェルのパイプライン（ls | grep | sort）
  - Webサーバー ↔ アプリケーションサーバー ↔ データベース
  - ブラウザのマルチプロセスアーキテクチャ
  - マイクロサービス間の通信
  - プロセスの制御（シグナル）

  → OSが提供するIPC機構を使って通信する

IPCの分類:
  ┌─────────────────────────────────────────────────┐
  │ 1. データ転送型                                   │
  │    パイプ、メッセージキュー、ソケット             │
  │    → カーネルがデータをコピー                     │
  │    → 同期が組み込まれている                       │
  │                                                   │
  │ 2. 共有メモリ型                                   │
  │    共有メモリ、mmap                               │
  │    → 物理メモリを共有（カーネルコピーなし）       │
  │    → 同期は自前で実装する必要あり                 │
  │                                                   │
  │ 3. 通知型                                         │
  │    シグナル                                       │
  │    → データ転送なし、イベント通知のみ             │
  │                                                   │
  │ 4. 同期型                                         │
  │    セマフォ、ミューテックス、条件変数             │
  │    → データ転送なし、アクセス制御のみ             │
  └─────────────────────────────────────────────────┘
```

### 1.2 IPC手法の比較表

```
IPC手法の比較:

┌────────────────┬──────────┬──────────┬───────────┬──────────┬──────────┐
│ 手法           │ 方向     │ 速度     │ 関連性    │ 用途例   │ 永続性   │
├────────────────┼──────────┼──────────┼───────────┼──────────┼──────────┤
│ パイプ         │ 片方向   │ 中       │ 親子      │ ls | grep│ 一時的   │
│ 名前付きパイプ │ 片方向   │ 中       │ 無関係可  │ ログ収集 │ ファイル │
│ シグナル       │ 通知のみ │ 速       │ 無関係可  │ kill     │ 一時的   │
│ メッセージキュー│ 双方向  │ 中       │ 無関係可  │ タスク   │ カーネル │
│ 共有メモリ     │ 双方向   │ 最速     │ 無関係可  │ DB       │ カーネル │
│ ソケット(TCP)  │ 双方向   │ 中〜遅   │ 無関係可  │ ネットワーク│ 一時的 │
│ Unixドメイン   │ 双方向   │ 速       │ 同一マシン│ Docker   │ ファイル │
│ ソケット       │          │          │           │          │          │
│ mmap           │ 双方向   │ 最速     │ 無関係可  │ ファイル │ ファイル │
│ eventfd        │ 通知     │ 速       │ 無関係可  │ イベント │ 一時的   │
│ D-Bus          │ 双方向   │ 中       │ 無関係可  │ デスクトップ│ デーモン│
└────────────────┴──────────┴──────────┴───────────┴──────────┴──────────┘

速度の比較（概算、64バイトメッセージ）:
┌────────────────┬───────────────┬──────────────┐
│ IPC手法        │ レイテンシ    │ スループット │
├────────────────┼───────────────┼──────────────┤
│ 共有メモリ     │ 〜50ns        │ 〜10 GB/s    │
│ パイプ         │ 〜1μs         │ 〜3 GB/s     │
│ Unixドメイン   │ 〜2μs         │ 〜2 GB/s     │
│ メッセージキュー│ 〜5μs        │ 〜500 MB/s   │
│ TCP loopback   │ 〜10μs        │ 〜1 GB/s     │
│ D-Bus          │ 〜100μs       │ 〜50 MB/s    │
└────────────────┴───────────────┴──────────────┘
※ 環境・メッセージサイズにより大きく変動
```

### 1.3 IPC手法の選択指針

```
用途別の推奨IPC:

  同一マシン・親子プロセス:
  → パイプ（最もシンプル、シェル連携に最適）

  同一マシン・無関係プロセス・高速通信:
  → 共有メモリ + セマフォ（最速、大量データ向き）
  → Unixドメインソケット（双方向、API充実）

  同一マシン・無関係プロセス・メッセージ単位:
  → POSIX メッセージキュー（優先度付き、境界明確）
  → Unixドメインソケット（SOCK_DGRAM）

  異なるマシン間:
  → TCPソケット（最も汎用的）
  → gRPC（構造化データ、ストリーミング対応）

  プロセス制御（起動/停止/再読込）:
  → シグナル（SIGTERM, SIGHUP等）

  デスクトップアプリケーション連携:
  → D-Bus（Linux）/ XPC（macOS）/ COM（Windows）

  マイクロサービス間通信:
  → gRPC / HTTP REST / メッセージブローカー（RabbitMQ, Kafka）
```

---

## 2. パイプ

### 2.1 無名パイプ（Pipe）

```
無名パイプ:
  親子プロセス間の単方向バイトストリーム
  → fork() の前にpipe()で作成
  → fork() 後に親と子で読み書き端を分担

  $ ls -la | grep ".md" | wc -l

  ls ──→ [パイプ] ──→ grep ──→ [パイプ] ──→ wc
  stdout    stdin     stdout     stdin

  カーネル内の実装:
  ┌──────────────────────────────────────────────────┐
  │ パイプはカーネル内のリングバッファ                │
  │                                                    │
  │ 書き込み側 ──→ [カーネルバッファ] ──→ 読み込み側  │
  │                   ↑                                │
  │                 通常64KB（Linux）                   │
  │                 /proc/sys/fs/pipe-max-size で変更可 │
  │                                                    │
  │ バッファが満杯 → 書き込み側がブロック（バックプレッシャー）│
  │ バッファが空   → 読み込み側がブロック              │
  │ 全書き込み側がクローズ → 読み込み側にEOF          │
  │ 全読み込み側がクローズ → 書き込み側にSIGPIPE      │
  └──────────────────────────────────────────────────┘

  アトミック書き込み:
  → PIPE_BUF（通常4096バイト）以下の書き込みはアトミック
  → 複数プロセスが同時に書き込んでもデータが混ざらない
  → PIPE_BUFを超える書き込みはアトミック性の保証なし
```

```c
// パイプのCプログラム例
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    int pipefd[2];  // pipefd[0] = 読み込み端, pipefd[1] = 書き込み端

    if (pipe(pipefd) == -1) {
        perror("pipe");
        exit(EXIT_FAILURE);
    }

    pid_t pid = fork();
    if (pid == -1) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        // 子プロセス: 読み込み側
        close(pipefd[1]);  // 書き込み端を閉じる

        char buf[256];
        ssize_t n;
        while ((n = read(pipefd[0], buf, sizeof(buf) - 1)) > 0) {
            buf[n] = '\0';
            printf("子プロセスが受信: %s\n", buf);
        }

        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    } else {
        // 親プロセス: 書き込み側
        close(pipefd[0]);  // 読み込み端を閉じる

        const char *msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);  // EOFを送信
        wait(NULL);        // 子プロセスの終了を待つ
    }

    return 0;
}
```

```c
// 双方向通信には2本のパイプが必要
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    int pipe_parent_to_child[2];  // 親→子
    int pipe_child_to_parent[2];  // 子→親

    pipe(pipe_parent_to_child);
    pipe(pipe_child_to_parent);

    pid_t pid = fork();
    if (pid == 0) {
        // 子プロセス
        close(pipe_parent_to_child[1]);
        close(pipe_child_to_parent[0]);

        char buf[256];
        read(pipe_parent_to_child[0], buf, sizeof(buf));
        printf("子が受信: %s\n", buf);

        const char *reply = "Hello from child!";
        write(pipe_child_to_parent[1], reply, strlen(reply));

        close(pipe_parent_to_child[0]);
        close(pipe_child_to_parent[1]);
        exit(0);
    } else {
        // 親プロセス
        close(pipe_parent_to_child[0]);
        close(pipe_child_to_parent[1]);

        const char *msg = "Hello from parent!";
        write(pipe_parent_to_child[1], msg, strlen(msg));

        char buf[256];
        ssize_t n = read(pipe_child_to_parent[0], buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("親が受信: %s\n", buf);

        close(pipe_parent_to_child[1]);
        close(pipe_child_to_parent[0]);
    }
    return 0;
}
```

### 2.2 名前付きパイプ（FIFO）

```
名前付きパイプ（FIFO）:
  ファイルシステム上に名前を持つパイプ
  → 無関係なプロセス間で通信可能
  → ファイルパスを知っていれば接続可能

  作成:
  $ mkfifo /tmp/myfifo
  $ ls -la /tmp/myfifo
  prw-r--r-- 1 user user 0 ... /tmp/myfifo
  ↑ 'p' = パイプ

  使用:
  # ターミナル1（読み込み側）:
  $ cat /tmp/myfifo          # ブロック: 書き込み側が来るまで待つ

  # ターミナル2（書き込み側）:
  $ echo "hello" > /tmp/myfifo  # 書き込み → ターミナル1に"hello"表示

  特性:
  - 無名パイプと同じカーネルバッファを使用
  - オープンは両端が揃うまでブロック（デフォルト）
  - O_NONBLOCK で非ブロッキング可能
  - 永続的（明示的にrm するまでファイルが残る）
  - シンプルなIPC に適している

  制限:
  - 単方向（双方向には2つのFIFOが必要）
  - バイトストリーム（メッセージ境界なし）
  - ネットワーク越しは不可
```

```python
# Pythonでの名前付きパイプの例

# サーバー側（読み込み）
import os

fifo_path = "/tmp/myfifo"

# FIFOの作成（既に存在する場合はスキップ）
if not os.path.exists(fifo_path):
    os.mkfifo(fifo_path)

print("クライアントの接続を待っています...")
with open(fifo_path, 'r') as fifo:
    while True:
        line = fifo.readline()
        if not line:
            break
        print(f"受信: {line.strip()}")

# クライアント側（書き込み）
with open(fifo_path, 'w') as fifo:
    fifo.write("Hello from client!\n")
    fifo.flush()
```

### 2.3 パイプの実務活用パターン

```
パイプの活用パターン:

  1. ログ処理パイプライン:
  $ tail -f /var/log/syslog | grep "error" | tee errors.log | mail admin

  2. データ変換パイプライン:
  $ cat data.csv | cut -d',' -f1,3 | sort | uniq -c | sort -rn | head -20

  3. 並列処理:
  $ cat urls.txt | xargs -P 10 -I{} curl -s {} > /dev/null

  4. プロセス置換（Bash拡張）:
  $ diff <(ls dir1) <(ls dir2)     # 2つのコマンドの出力を比較
  $ paste <(cut -f1 a.tsv) <(cut -f2 b.tsv)  # 列の結合

  5. コプロセス（Bash 4.0+）:
  $ coproc bc        # bcを双方向パイプで起動
  $ echo "2+3" >&${COPROC[1]}   # bcに入力
  $ read result <&${COPROC[0]}   # 結果を読む
  $ echo $result     # 5

  パイプのパフォーマンス考慮:
  ┌─────────────────────────────────────────────────┐
  │ - パイプのバッファサイズ: デフォルト64KB          │
  │ - fcntl(fd, F_SETPIPE_SZ, size) で変更可能      │
  │ - 最大1MB（/proc/sys/fs/pipe-max-size）         │
  │ - splice() でゼロコピー転送可能                  │
  │ - vmsplice() でユーザー空間からゼロコピー        │
  │ - tee() でパイプ間のゼロコピー分岐              │
  └─────────────────────────────────────────────────┘
```

---

## 3. シグナル

### 3.1 シグナルの基礎

```
シグナル: プロセスへの非同期通知メカニズム

  シグナルの特徴:
  - ソフトウェア割り込みの一種
  - 最小限の情報（シグナル番号のみ）を伝達
  - 非同期（いつ届くか予測不能）
  - カーネルがプロセスに配送

  シグナルの発生源:
  1. ユーザー操作: Ctrl+C（SIGINT）、Ctrl+Z（SIGTSTP）
  2. カーネル: SIGSEGV（不正メモリアクセス）、SIGFPE（ゼロ除算）
  3. 他のプロセス: kill() システムコール
  4. タイマー: alarm()、setitimer()
  5. 子プロセス終了: SIGCHLD

  主要なシグナル一覧:
  ┌──────────┬──────┬──────────────────────┬──────────┐
  │ シグナル  │ 番号 │ 動作                  │ デフォルト│
  ├──────────┼──────┼──────────────────────┼──────────┤
  │ SIGHUP   │ 1    │ 端末切断 / 設定再読込 │ 終了     │
  │ SIGINT   │ 2    │ Ctrl+C（割り込み）    │ 終了     │
  │ SIGQUIT  │ 3    │ Ctrl+\（コアダンプ）  │ コアダンプ│
  │ SIGILL   │ 4    │ 不正命令              │ コアダンプ│
  │ SIGTRAP  │ 5    │ トレース/ブレークポイント│ コアダンプ│
  │ SIGABRT  │ 6    │ abort()              │ コアダンプ│
  │ SIGBUS   │ 7    │ バスエラー            │ コアダンプ│
  │ SIGFPE   │ 8    │ 浮動小数点例外        │ コアダンプ│
  │ SIGKILL  │ 9    │ 強制終了（捕捉不可）  │ 終了     │
  │ SIGUSR1  │ 10   │ ユーザー定義1         │ 終了     │
  │ SIGSEGV  │ 11   │ セグメンテーション違反│ コアダンプ│
  │ SIGUSR2  │ 12   │ ユーザー定義2         │ 終了     │
  │ SIGPIPE  │ 13   │ パイプ破壊            │ 終了     │
  │ SIGALRM  │ 14   │ alarm()タイマー       │ 終了     │
  │ SIGTERM  │ 15   │ 正常終了要求          │ 終了     │
  │ SIGCHLD  │ 17   │ 子プロセス終了通知    │ 無視     │
  │ SIGCONT  │ 18   │ 再開                  │ 再開     │
  │ SIGSTOP  │ 19   │ 一時停止（捕捉不可）  │ 停止     │
  │ SIGTSTP  │ 20   │ Ctrl+Z（端末停止）    │ 停止     │
  │ SIGWINCH │ 28   │ ウィンドウサイズ変更  │ 無視     │
  └──────────┴──────┴──────────────────────┴──────────┘

  SIGKILL(9) と SIGSTOP(19) は捕捉・無視・ブロックが一切不可能
  → カーネルが直接処理する（ユーザー空間のハンドラなし）
  → プロセスのハングアップに対する最終手段
```

### 3.2 シグナルの送信と捕捉

```bash
# シグナルの送信方法

# kill コマンド（名前に反してシグナル全般を送信）
kill -TERM 1234           # PID 1234にSIGTERM送信
kill -9 1234              # PID 1234をSIGKILL（強制終了）
kill -HUP $(cat /var/run/nginx.pid)  # nginx設定再読込
kill -USR1 $(pidof dd)    # ddの進捗表示

# killall: プロセス名で指定
killall -TERM firefox     # firefoxの全プロセスにSIGTERM
killall -HUP syslogd      # syslogd に設定再読込

# pkill: パターンマッチで指定
pkill -f "python server.py"  # コマンドライン全体でマッチ
pkill -u username            # 特定ユーザーの全プロセス

# プロセスグループへの送信
kill -TERM -1234             # PGID 1234の全プロセスに送信
kill -TERM 0                 # 自分と同じプロセスグループに送信

# セッション全体への送信
pkill -s 1234                # セッションID 1234の全プロセス
```

```c
// C言語でのシグナルハンドリング（sigaction推奨）
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

// volatile sig_atomic_t: シグナルハンドラと安全に共有できる型
volatile sig_atomic_t got_sigterm = 0;
volatile sig_atomic_t got_sighup = 0;

void sigterm_handler(int signum) {
    got_sigterm = 1;
    // 注意: シグナルハンドラ内ではasync-signal-safe関数のみ使用可能
    // printf(), malloc(), free() 等は使用不可
    // write() は使用可能
    const char msg[] = "SIGTERM received\n";
    write(STDOUT_FILENO, msg, sizeof(msg) - 1);
}

void sighup_handler(int signum) {
    got_sighup = 1;
}

int main() {
    // sigaction構造体の設定
    struct sigaction sa_term, sa_hup;

    // SIGTERM ハンドラ
    memset(&sa_term, 0, sizeof(sa_term));
    sa_term.sa_handler = sigterm_handler;
    sigemptyset(&sa_term.sa_mask);       // ハンドラ実行中にブロックする追加シグナルなし
    sa_term.sa_flags = 0;                // フラグなし
    sigaction(SIGTERM, &sa_term, NULL);

    // SIGHUP ハンドラ
    memset(&sa_hup, 0, sizeof(sa_hup));
    sa_hup.sa_handler = sighup_handler;
    sigemptyset(&sa_hup.sa_mask);
    sa_hup.sa_flags = SA_RESTART;        // シグナルで中断されたシステムコールを自動再開
    sigaction(SIGHUP, &sa_hup, NULL);

    // SIGPIPE を無視（ソケット/パイプ切断時の予期しない終了を防ぐ）
    struct sigaction sa_pipe;
    memset(&sa_pipe, 0, sizeof(sa_pipe));
    sa_pipe.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa_pipe, NULL);

    printf("PID: %d. SIGTERM or SIGHUP を送信してください\n", getpid());

    while (!got_sigterm) {
        if (got_sighup) {
            printf("設定を再読み込みしています...\n");
            got_sighup = 0;
            // 設定ファイルの再読込処理
        }
        sleep(1);
    }

    printf("正常終了処理中...\n");
    // クリーンアップ処理（ファイルクローズ、一時ファイル削除等）
    return 0;
}
```

### 3.3 シグナルの実務パターン

```
SIGTERM vs SIGKILL の正しい使い方:

  正しい手順:
  1. SIGTERM を送信（graceful shutdown の機会を与える）
  2. 数秒待つ（タイムアウト）
  3. 応答がなければ SIGKILL を送信（最終手段）

  ┌──────────────────────────────────────────────┐
  │ SIGTERM (15):                                 │
  │ → プロセスが捕捉して後処理可能               │
  │ → DBのトランザクションをコミット/ロールバック │
  │ → 一時ファイルの削除                          │
  │ → 接続のクローズ                              │
  │ → ログの出力                                  │
  │                                                │
  │ SIGKILL (9):                                   │
  │ → 即座に強制終了（ハンドラ実行なし）          │
  │ → 一時ファイルが残る可能性                    │
  │ → DB のトランザクションが中途半端に           │
  │ → 共有メモリが解放されない可能性              │
  │ → 最終手段としてのみ使用すべき                │
  └──────────────────────────────────────────────┘

実務でよく使うシグナルパターン:

  1. Webサーバーのgraceful restart:
     kill -HUP $(pidof nginx)
     → 新しい設定で新ワーカーを起動
     → 既存接続は既存ワーカーが処理完了まで担当
     → 既存ワーカーは処理完了後に終了

  2. ログローテーション:
     kill -USR1 $(pidof nginx)
     → ログファイルを再オープン
     → logrotate と連携

  3. Dockerのstop処理:
     docker stop container_name
     → SIGTERM送信 → 10秒待ち → SIGKILL
     → --stop-timeout でタイムアウト変更可能

  4. systemdのサービス停止:
     systemctl stop myservice
     → SIGTERM送信 → TimeoutStopSec待ち → SIGKILL
     → 設定ファイルでカスタマイズ可能

  5. Kubernetesのpod終了:
     kubectl delete pod mypod
     → preStop hook実行
     → SIGTERM送信
     → terminationGracePeriodSeconds（デフォルト30秒）待ち
     → SIGKILL
```

```python
# Pythonでのシグナルハンドリング（実務的な例）
import signal
import sys
import os
import time
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class GracefulShutdown:
    """グレースフルシャットダウンを管理するクラス"""

    def __init__(self):
        self.shutdown_requested = False
        self.connections = []

        # シグナルハンドラの登録
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_sighup)

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"{sig_name} を受信。シャットダウンを開始します...")
        self.shutdown_requested = True

    def _handle_sighup(self, signum, frame):
        logger.info("SIGHUP を受信。設定を再読み込みします...")
        self.reload_config()

    def reload_config(self):
        """設定ファイルの再読み込み"""
        logger.info("設定ファイルを再読み込み中...")
        # config = load_config("/etc/myapp/config.yaml")

    def cleanup(self):
        """クリーンアップ処理"""
        logger.info("接続をクローズしています...")
        for conn in self.connections:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"接続クローズ失敗: {e}")

        logger.info("一時ファイルを削除しています...")
        # cleanup_temp_files()

        logger.info("クリーンアップ完了")

    def run(self):
        """メインループ"""
        logger.info(f"PID: {os.getpid()} で起動しました")

        while not self.shutdown_requested:
            try:
                # メイン処理
                self.process_work()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"エラー: {e}")

        self.cleanup()
        logger.info("正常終了しました")

    def process_work(self):
        """実際のビジネスロジック"""
        pass

if __name__ == "__main__":
    app = GracefulShutdown()
    app.run()
```

### 3.4 シグナルの注意点と制限

```
シグナルの注意点:

  1. async-signal-safe 関数のみハンドラ内で使用可能:
     ┌──────────────────────────────────────────┐
     │ 使用可能（async-signal-safe）:            │
     │ write(), read(), _exit(), signal()        │
     │ open(), close(), fork(), execve()         │
     │ kill(), raise(), sigaction()              │
     │                                            │
     │ 使用不可（unsafe）:                        │
     │ printf(), fprintf() → バッファ破損の危険  │
     │ malloc(), free() → ヒープ破損の危険       │
     │ syslog() → 内部ロックでデッドロック       │
     │ pthread_mutex_lock() → デッドロック        │
     └──────────────────────────────────────────┘

  2. シグナルの損失:
     → 同じシグナルが連続で発生すると、保留中の分が失われる
     → POSIXの標準シグナルはキューイングされない（1つだけ保留）
     → リアルタイムシグナル（SIGRTMIN〜SIGRTMAX）はキューイングされる

  3. マルチスレッドでのシグナル:
     → シグナルはプロセス全体に送られる
     → どのスレッドが受け取るかは不定
     → pthread_sigmask() で特定スレッドのみ受信するように制御
     → 推奨: 専用のシグナル処理スレッドを作る

  4. signalfd（Linux固有）:
     → シグナルをファイルディスクリプタとして受信
     → epoll/select と統合可能
     → シグナルハンドラの問題を回避

  5. 代替手段としてのself-pipe trick:
     → パイプを作成し、シグナルハンドラ内でwrite()
     → メインループでselect/poll/epollで監視
     → シグナルハンドラの制限を回避するテクニック
```

---

## 4. 共有メモリとmmap

### 4.1 POSIX共有メモリ

```
共有メモリ:
  複数プロセスが同じ物理メモリ領域をマッピング
  → IPC最速（カーネルを経由しない直接アクセス）
  → 同期は自前で行う必要あり（セマフォ、ミューテックス等）

  Process A             Process B
  ┌──────────┐         ┌──────────┐
  │仮想メモリ │         │仮想メモリ │
  │  ┌─────┐ │         │ ┌─────┐  │
  │  │共有 │─│────┐────│─│共有 │  │
  │  │領域 │ │    │    │ │領域 │  │
  │  └─────┘ │    │    │ └─────┘  │
  └──────────┘    │    └──────────┘
                  ↓
            物理メモリ上の
            同一領域

  POSIX共有メモリAPI:
  shm_open()    → 共有メモリオブジェクトの作成/オープン
  ftruncate()   → サイズの設定
  mmap()        → メモリマッピング
  munmap()      → マッピング解除
  shm_unlink()  → 共有メモリオブジェクトの削除

  System V共有メモリ（レガシー）:
  shmget()  → 共有メモリセグメントの作成/取得
  shmat()   → アタッチ（マッピング）
  shmdt()   → デタッチ
  shmctl()  → 制御（削除等）
  → POSIX版の方が現代的で推奨
```

```c
// POSIX共有メモリ + セマフォの実装例

// 共通ヘッダ（shared_data.h）
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME "/my_shared_mem"
#define SEM_NAME "/my_semaphore"
#define SHM_SIZE 4096

typedef struct {
    int counter;
    char message[256];
    int ready;  // フラグ
} SharedData;

// プロデューサー（書き込み側）
int producer() {
    // 共有メモリの作成
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedData));

    SharedData *data = mmap(NULL, sizeof(SharedData),
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    // セマフォの作成
    sem_t *sem = sem_open(SEM_NAME, O_CREAT, 0666, 1);

    for (int i = 0; i < 100; i++) {
        sem_wait(sem);  // ロック取得
        data->counter = i;
        snprintf(data->message, sizeof(data->message), "Message #%d", i);
        data->ready = 1;
        sem_post(sem);  // ロック解放
        usleep(10000);  // 10ms待機
    }

    munmap(data, sizeof(SharedData));
    sem_close(sem);
    return 0;
}

// コンシューマー（読み込み側）
int consumer() {
    int fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    SharedData *data = mmap(NULL, sizeof(SharedData),
                           PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    sem_t *sem = sem_open(SEM_NAME, 0);

    while (1) {
        sem_wait(sem);
        if (data->ready) {
            printf("Counter: %d, Message: %s\n", data->counter, data->message);
            // data->ready = 0;  // 読み取り専用なので書けない
        }
        sem_post(sem);
        usleep(10000);
    }

    munmap(data, sizeof(SharedData));
    sem_close(sem);
    return 0;
}
```

### 4.2 mmap（メモリマッピング）

```
mmap:
  ファイルまたは匿名メモリをプロセスの仮想アドレス空間にマッピング

  用途:
  1. ファイルI/Oの高速化
  2. プロセス間でファイルを共有
  3. 匿名マッピング（大きなメモリ確保）
  4. 共有ライブラリのロード

  通常のI/O vs mmap:
  ┌──────────────────────────────────────────────────┐
  │ 通常のread():                                     │
  │ ディスク → カーネルバッファ → ユーザーバッファ    │
  │ → 2回のメモリコピー                               │
  │                                                    │
  │ mmap:                                              │
  │ ディスク → ページキャッシュ ← ユーザー空間から直接 │
  │ → コピー0回（ページキャッシュを直接参照）          │
  │                                                    │
  │ ただし:                                            │
  │ - 小さなファイルではオーバーヘッド（ページテーブル）│
  │ - ランダムアクセスパターンではmmapが有利            │
  │ - シーケンシャルアクセスではread()も十分高速        │
  └──────────────────────────────────────────────────┘

  mmapのフラグ:
  MAP_SHARED:   変更を元のファイルに反映。他プロセスと共有
  MAP_PRIVATE:  Copy-on-Write。変更は自プロセスのみ
  MAP_ANONYMOUS: ファイルなし。ゼロ初期化メモリ（大きなmalloc）
  MAP_FIXED:    指定アドレスにマッピング（危険、通常使わない）
  MAP_HUGETLB:  ラージページを使用
  MAP_POPULATE: マッピング時に事前にページフォルト（ページイン）
```

```c
// mmapによるファイルの読み書き例
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

int main() {
    const char *filepath = "/tmp/mmap_example.dat";

    // ファイルの作成と初期データの書き込み
    int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, 0666);
    const char *initial = "Hello, mmap world!\n";
    write(fd, initial, strlen(initial));

    // ファイルサイズの取得
    struct stat st;
    fstat(fd, &st);

    // メモリマッピング
    char *mapped = mmap(NULL, st.st_size,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    close(fd);  // マッピング後はfdを閉じてもOK

    // マッピング経由でデータを読む
    printf("Read via mmap: %s", mapped);

    // マッピング経由でデータを変更（ファイルに反映される）
    memcpy(mapped, "HELLO", 5);

    // 変更を確実にディスクに書き込む
    msync(mapped, st.st_size, MS_SYNC);

    // マッピング解除
    munmap(mapped, st.st_size);

    // ファイルを通常のI/Oで確認
    fd = open(filepath, O_RDONLY);
    char buf[256];
    read(fd, buf, sizeof(buf));
    printf("Read via read(): %s", buf);
    close(fd);

    unlink(filepath);
    return 0;
}
```

```python
# Pythonでのmmap活用例

import mmap
import os

# 大きなファイルの効率的な検索
def search_large_file(filepath, pattern):
    """mmapを使った大きなファイルの高速検索"""
    with open(filepath, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pattern_bytes = pattern.encode()
            offset = 0
            results = []
            while True:
                pos = mm.find(pattern_bytes, offset)
                if pos == -1:
                    break
                # 行番号を計算
                line_num = mm[:pos].count(b'\n') + 1
                results.append((line_num, pos))
                offset = pos + 1
            return results

# プロセス間共有メモリ（Python 3.8+）
from multiprocessing import shared_memory

# 書き込み側
shm = shared_memory.SharedMemory(name='my_shared', create=True, size=1024)
shm.buf[0:5] = b'Hello'
print(f"共有メモリ名: {shm.name}")

# 読み込み側（別プロセス）
shm2 = shared_memory.SharedMemory(name='my_shared')
print(bytes(shm2.buf[0:5]))  # b'Hello'
shm2.close()

# クリーンアップ
shm.close()
shm.unlink()
```

---

## 5. ソケット

### 5.1 Unixドメインソケット

```
Unixドメインソケット:
  同一マシン内のプロセス間通信
  → TCPソケットより高速（ネットワークスタック不要）
  → ファイルシステム上にソケットファイルを作成
  → ファイルパーミッションでアクセス制御

  用途: Docker, PostgreSQL, MySQL, nginx, systemd, D-Bus

  $ ls -la /var/run/docker.sock
  srw-rw---- 1 root docker 0 ... /var/run/docker.sock
  ↑ 's' = ソケット

  TCPソケットとの比較:
  ┌──────────────────┬───────────────┬───────────────┐
  │ 特性             │ Unixドメイン  │ TCP loopback  │
  ├──────────────────┼───────────────┼───────────────┤
  │ レイテンシ       │ 〜2μs         │ 〜10μs        │
  │ スループット     │ 〜6 GB/s      │ 〜4 GB/s      │
  │ 認証             │ UID/GID取得可 │ IP/Portのみ   │
  │ アクセス制御     │ ファイルPerm  │ ファイアウォール│
  │ ネットワーク対応 │ 不可          │ 可能          │
  │ fd passing       │ 可能          │ 不可          │
  │ credentials      │ 可能          │ 不可          │
  └──────────────────┴───────────────┴───────────────┘

  ソケットタイプ:
  SOCK_STREAM:  TCP的。コネクション型、順序保証、バイトストリーム
  SOCK_DGRAM:   UDP的。コネクションレス、メッセージ境界あり
  SOCK_SEQPACKET: 順序保証 + メッセージ境界あり（SCTPに相当）

  Unixドメインソケット固有の機能:
  1. ファイルディスクリプタの受け渡し（fd passing）:
     → プロセス間でファイルディスクリプタを共有
     → systemd のソケットアクティベーションで使用
     → sendmsg() / recvmsg() の SCM_RIGHTS

  2. ピア資格情報の取得:
     → 接続相手のUID, GID, PIDを取得
     → SO_PEERCRED ソケットオプション
     → D-Busの認証に使用

  3. 抽象ソケット（Linux固有）:
     → ファイルシステムに実体を持たない
     → 名前が'\0'で始まる
     → 自動クリーンアップ（プロセス終了で消える）
```

```python
# Unixドメインソケットサーバー/クライアントの例

import socket
import os

SOCKET_PATH = "/tmp/my_unix_socket"

# サーバー
def unix_server():
    # 既存のソケットファイルを削除
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    # ソケットファイルのパーミッション設定
    os.chmod(SOCKET_PATH, 0o666)

    print(f"サーバー起動: {SOCKET_PATH}")
    try:
        while True:
            conn, addr = server.accept()
            print(f"接続受付")

            # ピア資格情報の取得（Linux固有）
            # creds = conn.getsockopt(socket.SOL_SOCKET,
            #                        socket.SO_PEERCRED, 12)

            data = conn.recv(1024)
            print(f"受信: {data.decode()}")

            conn.sendall(b"ACK: " + data)
            conn.close()
    finally:
        server.close()
        os.unlink(SOCKET_PATH)

# クライアント
def unix_client(message):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)

    client.sendall(message.encode())
    response = client.recv(1024)
    print(f"応答: {response.decode()}")

    client.close()
```

### 5.2 TCPソケット

```
TCPソケット:
  ネットワーク越しの通信にも使用可能
  → 最も汎用的なIPC（マシン間通信対応）
  → HTTP, gRPC, データベースプロトコルの基盤

  ソケットプログラミングの流れ:
  ┌──────────────────────────────────────────┐
  │ サーバー           クライアント           │
  │                                          │
  │ socket()           socket()              │
  │ bind()                                   │
  │ listen()                                 │
  │ accept() ←──────── connect()             │
  │ │                  │                     │
  │ recv()  ←──────── send()                │
  │ send()  ──────→  recv()                │
  │ │                  │                     │
  │ close()            close()               │
  └──────────────────────────────────────────┘

  高性能サーバーの実装パターン:
  1. マルチプロセス（fork/prefork）:
     → 接続ごとにfork
     → プロセス分離でクラッシュ耐性
     → Apache prefork MPM

  2. マルチスレッド:
     → 接続ごとにスレッド
     → メモリ共有で効率的
     → Java の Thread-per-request

  3. I/O多重化（select/poll/epoll/kqueue）:
     → 1スレッドで多数の接続を処理
     → イベント駆動
     → nginx, Node.js, Redis

  4. io_uring（Linux 5.1+）:
     → カーネルとの通信を非同期化
     → システムコールのオーバーヘッドを大幅削減
     → 最新の高性能サーバーで採用
```

```python
# I/O多重化を使った高性能サーバー（select/epoll の例）
import selectors
import socket
import types

# Linux: epoll, macOS: kqueue を自動選択
sel = selectors.DefaultSelector()

def accept_connection(sock):
    conn, addr = sock.accept()
    print(f"接続受付: {addr}")
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b"", outb=b"")
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)

def service_connection(key, mask):
    sock = key.fileobj
    data = key.data

    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(1024)
        if recv_data:
            data.outb += recv_data  # エコーバック
        else:
            print(f"切断: {data.addr}")
            sel.unregister(sock)
            sock.close()
            return

    if mask & selectors.EVENT_WRITE:
        if data.outb:
            sent = sock.send(data.outb)
            data.outb = data.outb[sent:]

def run_server(host='0.0.0.0', port=12345):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(100)
    sock.setblocking(False)
    sel.register(sock, selectors.EVENT_READ, data=None)

    print(f"サーバー起動: {host}:{port}")
    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_connection(key.fileobj)
                else:
                    service_connection(key, mask)
    except KeyboardInterrupt:
        print("終了")
    finally:
        sel.close()

if __name__ == "__main__":
    run_server()
```

---

## 6. メッセージキュー

### 6.1 POSIXメッセージキュー

```
POSIXメッセージキュー:
  メッセージ単位でデータを送受信
  → メッセージ境界が明確（パイプとの違い）
  → 優先度付きメッセージ
  → カーネル内に永続化（プロセス終了後も存続）

  パイプとの比較:
  ┌──────────────────┬──────────────┬──────────────┐
  │ 特性             │ パイプ       │ メッセージキュー│
  ├──────────────────┼──────────────┼──────────────┤
  │ データ形式       │ バイトストリーム│ メッセージ   │
  │ 境界             │ なし         │ あり          │
  │ 優先度           │ なし         │ あり          │
  │ 複数読み取り     │ データ分散   │ 1つに配送    │
  │ 永続性           │ プロセス終了で消滅│ 明示削除まで│
  │ 方向             │ 単方向       │ 単方向（双方向可）│
  │ サイズ制限       │ バッファサイズ│ メッセージ数 │
  └──────────────────┴──────────────┴──────────────┘

  API:
  mq_open()    → キューの作成/オープン
  mq_send()    → メッセージ送信（優先度指定可）
  mq_receive() → メッセージ受信（最高優先度から）
  mq_notify()  → メッセージ到着時の非同期通知
  mq_close()   → キューのクローズ
  mq_unlink()  → キューの削除
```

```c
// POSIXメッセージキューの例
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mqueue.h>
#include <errno.h>

#define QUEUE_NAME "/my_task_queue"
#define MAX_MSG_SIZE 256
#define MAX_MSGS 10

// プロデューサー
void producer() {
    struct mq_attr attr = {
        .mq_flags = 0,
        .mq_maxmsg = MAX_MSGS,
        .mq_msgsize = MAX_MSG_SIZE,
        .mq_curmsgs = 0
    };

    mqd_t mq = mq_open(QUEUE_NAME, O_CREAT | O_WRONLY, 0666, &attr);
    if (mq == (mqd_t)-1) {
        perror("mq_open");
        exit(1);
    }

    // 優先度付きメッセージの送信
    mq_send(mq, "高優先度タスク", strlen("高優先度タスク"), 10);  // 優先度10
    mq_send(mq, "低優先度タスク", strlen("低優先度タスク"), 1);   // 優先度1
    mq_send(mq, "中優先度タスク", strlen("中優先度タスク"), 5);   // 優先度5

    mq_close(mq);
}

// コンシューマー
void consumer() {
    mqd_t mq = mq_open(QUEUE_NAME, O_RDONLY);
    char buffer[MAX_MSG_SIZE + 1];
    unsigned int priority;

    // 優先度の高い順に受信される
    while (1) {
        ssize_t bytes = mq_receive(mq, buffer, MAX_MSG_SIZE + 1, &priority);
        if (bytes == -1) {
            if (errno == EAGAIN) break;
            perror("mq_receive");
            break;
        }
        buffer[bytes] = '\0';
        printf("受信(優先度%u): %s\n", priority, buffer);
    }
    // 出力順: 高優先度タスク(10) → 中優先度タスク(5) → 低優先度タスク(1)

    mq_close(mq);
    mq_unlink(QUEUE_NAME);
}
```

### 6.2 System Vメッセージキュー（レガシー）

```
System Vメッセージキュー:
  POSIXメッセージキューの前身
  → msgget(), msgsnd(), msgrcv(), msgctl()
  → メッセージタイプによるフィルタリングが可能
  → レガシーだが多くのシステムで利用可能

  主な違い:
  - System V: メッセージタイプ（long型）でフィルタリング
  - POSIX: 優先度（unsigned int）でソート
  - POSIX の方がAPIが洗練されている
  - POSIX はmq_notify()で非同期通知が可能
```

---

## 7. セマフォと同期

### 7.1 セマフォの基礎

```
セマフォ:
  共有リソースへのアクセスを制御する同期プリミティブ

  2種類のセマフォ:
  1. バイナリセマフォ（ミューテックスと同等）:
     値は0または1
     → 排他制御（1つのプロセスのみアクセス可能）

  2. カウンティングセマフォ:
     値はN（>0）
     → N個の同時アクセスを許可
     → リソースプール（DB接続プール、スレッドプール）

  操作:
  wait()（P操作, down, acquire）:
    if (semaphore > 0):
        semaphore -= 1  # リソース獲得
    else:
        block()          # ブロック（待機）

  post()（V操作, up, release）:
    semaphore += 1       # リソース解放
    wakeup_one()         # 待機中のプロセスを1つ起こす

  重要: wait/post はアトミック操作（カーネルが保証）

  生産者-消費者問題（セマフォで解決）:
  ┌──────────────────────────────────────┐
  │ empty = N  // 空きスロット数         │
  │ full = 0   // データ数               │
  │ mutex = 1  // 排他制御               │
  │                                      │
  │ Producer:          Consumer:          │
  │   wait(empty)        wait(full)      │
  │   wait(mutex)        wait(mutex)     │
  │   produce()          consume()       │
  │   post(mutex)        post(mutex)     │
  │   post(full)         post(empty)     │
  └──────────────────────────────────────┘
```

### 7.2 ミューテックスとプロセス間ロック

```
プロセス間ミューテックス:
  共有メモリ上にpthread_mutexを配置して
  プロセス間でロックを共有

  設定:
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  // 共有メモリ上のmutex変数に対して
  pthread_mutex_init(shared_mutex, &attr);

  ファイルロック（flock / fcntl）:
  → ファイルを介したプロセス間排他制御
  → データベース（SQLite）でよく使用

  flock() — アドバイザリロック:
  LOCK_SH: 共有ロック（複数プロセスが同時取得可能）
  LOCK_EX: 排他ロック（1プロセスのみ）
  LOCK_UN: アンロック

  fcntl() — レコードロック:
  → ファイルの一部のみロック可能
  → バイト範囲ロック
```

```python
# Pythonでのファイルロック（排他制御）の例
import fcntl
import os
import time

class FileLock:
    """ファイルベースのプロセス間排他制御"""

    def __init__(self, lockfile):
        self.lockfile = lockfile
        self.fd = None

    def acquire(self, timeout=10):
        self.fd = open(self.lockfile, 'w')
        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.fd.write(str(os.getpid()))
                self.fd.flush()
                return True
            except BlockingIOError:
                if time.time() - start > timeout:
                    self.fd.close()
                    raise TimeoutError(f"ロック取得タイムアウト: {self.lockfile}")
                time.sleep(0.1)

    def release(self):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            try:
                os.unlink(self.lockfile)
            except OSError:
                pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()

# 使用例
with FileLock("/tmp/my_app.lock"):
    print("排他ロック取得。クリティカルセクション実行中...")
    time.sleep(5)
    print("完了")
```

---

## 8. eventfdとその他のLinux固有IPC

### 8.1 eventfd

```
eventfd（Linux 2.6.22+）:
  軽量なイベント通知メカニズム
  → 64ビットカウンターをファイルディスクリプタとして提供
  → セマフォライクな使い方が可能
  → epoll/select と統合可能
  → パイプより軽量（カーネルバッファ不要）

  API:
  eventfd(initval, flags)  → fdを返す
  write(fd, &value, 8)     → カウンターに加算
  read(fd, &value, 8)      → カウンターを読み取りリセット

  flags:
  EFD_NONBLOCK:  非ブロッキング
  EFD_SEMAPHORE: セマフォモード（readで1ずつ減少）

  用途:
  - スレッド間のイベント通知
  - epollベースのイベントループでの停止通知
  - KVM（仮想化）の割り込み通知
```

### 8.2 D-Bus

```
D-Bus:
  デスクトップLinuxの標準IPC
  → メッセージバスデーモンを介した通信
  → オブジェクト指向のメッセージパッシング
  → 型安全なインターフェース定義

  アーキテクチャ:
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ App A    │  │ App B    │  │ App C    │
  └────┬─────┘  └────┬─────┘  └────┬─────┘
       │             │             │
  ┌────┴─────────────┴─────────────┴────┐
  │         D-Bus デーモン                │
  │  (dbus-daemon / dbus-broker)         │
  └─────────────────────────────────────┘

  2つのバス:
  - System Bus: システム全体（1つ）
    → NetworkManager, udev, systemd
  - Session Bus: ユーザーセッションごと（ログインごとに1つ）
    → デスクトップ通知、メディアプレーヤー制御

  使用例:
  # デスクトップ通知の送信
  gdbus call --session \
    --dest=org.freedesktop.Notifications \
    --object-path=/org/freedesktop/Notifications \
    --method=org.freedesktop.Notifications.Notify \
    "MyApp" 0 "" "タイトル" "本文" [] {} 5000

  # NetworkManagerの状態取得
  gdbus call --system \
    --dest=org.freedesktop.NetworkManager \
    --object-path=/org/freedesktop/NetworkManager \
    --method=org.freedesktop.DBus.Properties.Get \
    "org.freedesktop.NetworkManager" "State"
```

---

## 9. 実務でのIPC選択ガイド

### 9.1 パフォーマンス最適化

```
IPCのパフォーマンスチューニング:

  パイプ:
  → バッファサイズを大きくする（fcntl F_SETPIPE_SZ）
  → splice() でゼロコピー転送
  → vmsplice() でユーザー空間からのゼロコピー

  共有メモリ:
  → hugepages を使用（TLBミス削減）
  → メモリアライメントを考慮（キャッシュラインバウンシング防止）
  → ロックフリーデータ構造の使用（CASベース）

  ソケット:
  → TCP_NODELAY（Nagleアルゴリズム無効化、レイテンシ削減）
  → SO_SNDBUF / SO_RCVBUF（バッファサイズ拡大）
  → SO_REUSEPORT（マルチコアでのロードバランス）
  → io_uring でシステムコール削減

  メッセージキュー:
  → メッセージサイズを適切に設定（過大な割り当てを避ける）
  → キュー深さの監視（mq_getattr）

  ゼロコピー技術:
  ┌──────────────────────────────────────────────┐
  │ sendfile():   ファイル → ソケット             │
  │ splice():     パイプ → ソケット / パイプ      │
  │ vmsplice():   ユーザー空間 → パイプ           │
  │ mmap():       ファイル → ユーザー空間         │
  │ io_uring:     全般的な非同期I/O（ゼロコピー対応）│
  │                                                │
  │ → カーネル⟷ユーザー空間のコピーを排除        │
  │ → 大量データ転送で効果大                       │
  └──────────────────────────────────────────────┘
```

### 9.2 セキュリティ考慮事項

```
IPCのセキュリティ:

  パイプ:
  → fork()でのみ共有されるため比較的安全
  → 名前付きパイプはファイルパーミッションで制御

  共有メモリ:
  → /dev/shm のパーミッション設定
  → Sensitive データの場合は暗号化を検討
  → プロセス終了時のクリーンアップを確実に

  Unixドメインソケット:
  → ソケットファイルのパーミッション設定
  → SO_PEERCRED でクライアントの UID/GID/PID を確認
  → systemdのソケットアクティベーションでの権限分離

  TCPソケット:
  → TLS暗号化（ネットワーク越しは必須）
  → ファイアウォールでのアクセス制限
  → 認証・認可の実装

  メッセージキュー:
  → /dev/mqueue のパーミッション設定
  → メッセージサイズの制限（DoS防止）

  コンテナ環境:
  ┌──────────────────────────────────────────────┐
  │ Docker のデフォルトIPC設定:                    │
  │ - 各コンテナに独立したIPC名前空間             │
  │ - --ipc=host でホストのIPC名前空間を共有      │
  │ - --ipc=container:id で他コンテナと共有       │
  │                                                │
  │ Kubernetes:                                    │
  │ - Pod内のコンテナはIPC名前空間を共有          │
  │ - Pod間のIPCはネットワーク経由が基本          │
  └──────────────────────────────────────────────┘
```

---

## 10. 実践演習

### 演習1: [基礎] — パイプの実践

```bash
# 以下のパイプラインの各段階の出力を確認せよ
ps aux | awk '{print $1}' | sort | uniq -c | sort -rn | head -5

# 名前付きパイプを使って2つのターミナル間で通信せよ
# ターミナル1:
mkfifo /tmp/chat
while true; do
    if read line < /tmp/chat; then
        echo "受信: $line"
    fi
done

# ターミナル2:
echo "Hello from terminal 2" > /tmp/chat
echo "Second message" > /tmp/chat
```

### 演習2: [応用] — シグナルハンドリング

```python
import signal
import sys
import os
import time

def handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\nシグナル {sig_name}({signum}) を受信。終了処理中...")
    # クリーンアップ処理
    print("一時ファイルの削除...")
    print("接続のクローズ...")
    sys.exit(0)

def reload_handler(signum, frame):
    print(f"\n設定再読み込み（SIGHUP受信）")

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGHUP, reload_handler)

print(f"PID: {os.getpid()}")
print("Ctrl+C, kill -TERM, kill -HUP で操作してください")

while True:
    print(".", end="", flush=True)
    time.sleep(1)
```

### 演習3: [応用] — 共有メモリによるプロセス間通信

```python
# Python 3.8+ の multiprocessing.shared_memory を使った例
from multiprocessing import shared_memory, Process
import numpy as np
import time

def producer(shm_name):
    """共有メモリにデータを書き込むプロセス"""
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((10,), dtype=np.int64, buffer=shm.buf)

    for i in range(100):
        arr[i % 10] = i
        time.sleep(0.01)

    shm.close()

def consumer(shm_name):
    """共有メモリからデータを読み取るプロセス"""
    time.sleep(0.1)  # プロデューサーの開始を待つ
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((10,), dtype=np.int64, buffer=shm.buf)

    for _ in range(50):
        print(f"Current values: {arr.tolist()}")
        time.sleep(0.02)

    shm.close()

if __name__ == "__main__":
    # 共有メモリの作成
    shm = shared_memory.SharedMemory(create=True, size=80)  # 10 * 8bytes

    p1 = Process(target=producer, args=(shm.name,))
    p2 = Process(target=consumer, args=(shm.name,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    shm.close()
    shm.unlink()
```

### 演習4: [発展] — Unixドメインソケットのチャットシステム

```python
# サーバー（マルチクライアント対応）
import socket
import selectors
import os

SOCKET_PATH = "/tmp/chat_server.sock"
sel = selectors.DefaultSelector()
clients = {}

def accept(sock):
    conn, _ = sock.accept()
    conn.setblocking(False)
    sel.register(conn, selectors.EVENT_READ, data="client")
    clients[conn] = f"User{len(clients)+1}"
    broadcast(f"[{clients[conn]}が参加しました]", exclude=conn)

def broadcast(message, exclude=None):
    for client in list(clients.keys()):
        if client != exclude:
            try:
                client.sendall(message.encode() + b"\n")
            except (BrokenPipeError, ConnectionResetError):
                disconnect(client)

def disconnect(conn):
    name = clients.pop(conn, "Unknown")
    sel.unregister(conn)
    conn.close()
    broadcast(f"[{name}が退出しました]")

def handle_client(conn):
    try:
        data = conn.recv(1024)
        if data:
            name = clients.get(conn, "Unknown")
            broadcast(f"{name}: {data.decode().strip()}", exclude=conn)
        else:
            disconnect(conn)
    except ConnectionResetError:
        disconnect(conn)

if os.path.exists(SOCKET_PATH):
    os.unlink(SOCKET_PATH)

server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
server.bind(SOCKET_PATH)
server.listen(10)
server.setblocking(False)
sel.register(server, selectors.EVENT_READ, data="server")
os.chmod(SOCKET_PATH, 0o666)

print(f"チャットサーバー起動: {SOCKET_PATH}")
while True:
    events = sel.select(timeout=None)
    for key, mask in events:
        if key.data == "server":
            accept(key.fileobj)
        else:
            handle_client(key.fileobj)
```

---

## 11. FAQ

### Q1: DockerはなぜUnixドメインソケットを使うのか？

Docker CLIとDockerデーモンは同一マシン上で通信するため、TCPよりも高速で認証管理が容易なUnixドメインソケットが最適。ソケットファイルのパーミッションでアクセス制御でき、ネットワークスタックのオーバーヘッドもない。加えて、SO_PEERCREDで接続元のUID/GIDを確認でき、dockerグループに属するユーザーのみアクセスを許可する仕組みが実現できる。Docker API をリモートから使う場合はTCPソケット（2375/2376番ポート）に切り替えることも可能だが、TLS設定が必要。

### Q2: パイプとメッセージキューの違いは？

パイプは**バイトストリーム**（区切りなし）。メッセージキューは**メッセージ単位**（境界あり、優先度付き）。パイプは使い捨て（プロセス終了で消滅）だが、メッセージキューは明示的に削除するまで存続する。パイプは親子プロセス間（無名パイプ）が基本で、名前付きパイプを使えば無関係なプロセス間でも通信可能。メッセージキューは最初から無関係なプロセス間で使えるよう設計されている。パフォーマンス面ではパイプがやや高速。

### Q3: 共有メモリの同期にはどの方法がベストか？

用途による。排他制御だけなら**プロセス間mutex**（PTHREAD_PROCESS_SHARED）が最もシンプル。同時アクセス数の制御が必要なら**セマフォ**。高性能が必要なら**ロックフリーデータ構造**（Compare-and-Swap ベース）を検討する。ファイルロック（flock/fcntl）は実装が簡単だが性能はやや劣る。実務ではまずmutexで実装し、プロファイリングで問題が見つかったらロックフリーを検討するのが良い。

### Q4: gRPCとRESTの違いは？

gRPCはHTTP/2上のProtocol Buffersベースのリモートプロシージャコール。バイナリ形式で高速、双方向ストリーミング対応、型安全。REST（HTTP/JSON）はテキスト形式で人間可読、ブラウザから直接アクセス可能、ツールが豊富。マイクロサービス内部の通信ではgRPC、外部API公開ではRESTが一般的。ただしgRPC-Webの登場でブラウザからのgRPC利用も可能になっている。

### Q5: io_uringはIPCにどう影響するのか？

io_uring（Linux 5.1+）は非同期I/Oの新しいインターフェースで、システムコールのオーバーヘッドを大幅に削減する。ソケット通信やパイプのI/Oをio_uring経由で行うことで、高負荷環境でのIPCスループットが改善される。特にsend/recvのバッチ処理、ゼロコピー送信、マルチショットaccept等の機能がサーバーアプリケーションのIPCパフォーマンスを向上させる。

---

## 12. まとめ

| IPC手法 | 速度 | 用途 |
|---------|------|------|
| パイプ | 中 | シェルコマンド連携、親子プロセス間 |
| 名前付きパイプ | 中 | 無関係プロセス間の簡易通信 |
| シグナル | 速 | プロセス制御（終了、再読込、停止）|
| メッセージキュー | 中 | メッセージ単位の通信、優先度付き |
| 共有メモリ | 最速 | 大量データ共有、高性能IPC |
| Unixドメインソケット | 速 | ローカルサービス通信 |
| TCPソケット | 中〜遅 | ネットワーク通信 |
| eventfd | 速 | 軽量イベント通知 |
| D-Bus | 中 | デスクトップアプリ連携 |

---

## 次に読むべきガイド
→ [[../02-memory-management/00-virtual-memory.md]] — 仮想メモリ

---

## 参考文献
1. Kerrisk, M. "The Linux Programming Interface." No Starch Press, Ch.43-57, 2010.
2. Stevens, W. R. "UNIX Network Programming, Vol.2: IPC." Prentice Hall, 1998.
3. Stevens, W. R. & Rago, S. A. "Advanced Programming in the UNIX Environment." 3rd Ed, Addison-Wesley, 2013.
4. Love, R. "Linux System Programming." 2nd Ed, O'Reilly, Ch.9-10, 2013.
5. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Ch.30-33, 2018.
6. Corbet, J. et al. "Linux Device Drivers." 3rd Ed, O'Reilly, 2005.
7. Axboe, J. "Efficient IO with io_uring." Kernel Documentation, 2019.
8. Tanenbaum, A. S. & Bos, H. "Modern Operating Systems." 4th Ed, Ch.2, 2014.
