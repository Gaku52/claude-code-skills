# スレッドと並行性

> スレッドは「軽量プロセス」であり、同一プロセス内でメモリを共有しながら並行実行される実行単位である。現代のほぼ全てのアプリケーション――Webサーバ、データベース、GUI、ゲームエンジン――はマルチスレッドで構築されている。この章ではスレッドの基本概念から、ユーザレベル/カーネルレベルの実装モデル、同期プリミティブ、デッドロック、スレッドプールまでを体系的に解説する。

## この章で学ぶこと

- [ ] スレッドとプロセスの構造的な違いを図解付きで説明できる
- [ ] ユーザースレッドとカーネルスレッドの動作原理とトレードオフを理解する
- [ ] 競合状態の発生メカニズムとその検出・対策を実装できる
- [ ] Mutex、セマフォ、条件変数、RWLockの使い分けができる
- [ ] デッドロックのCoffman条件を理解し、回避戦略を実装できる
- [ ] スレッドプールの設計原理とサイジングの考え方を説明できる
- [ ] 現代の並行性モデル（async/await, アクターモデル, CSP）との関係を理解する

---

## 1. スレッドの基本概念

### 1.1 プロセスとスレッドの構造比較

プロセスとスレッドの最も根本的な違いは**メモリ空間の共有範囲**にある。プロセスはOSが提供する独立したアドレス空間を持ち、他プロセスのメモリに直接アクセスできない。一方、スレッドは同一プロセス内の兄弟スレッドと**コード領域、データ領域、ヒープ、ファイルディスクリプタ**を共有する。

```
プロセスのメモリレイアウト（シングルスレッド vs マルチスレッド）:

  シングルスレッドプロセス:          マルチスレッドプロセス:
  ┌────────────────────────┐       ┌──────────────────────────────────┐
  │ カーネル空間            │       │ カーネル空間                     │
  │ (ページテーブル等)       │       │ (スレッドごとにTCBを管理)        │
  ├────────────────────────┤       ├──────────────────────────────────┤
  │ スタック (1つ)          │       │ スタック Th3 ← 各スレッド固有    │
  │                        │       ├──────────────────────────────────┤
  │                        │       │ スタック Th2 ← 各スレッド固有    │
  │                        │       ├──────────────────────────────────┤
  │                        │       │ スタック Th1 ← 各スレッド固有    │
  ├────────────────────────┤       ├──────────────────────────────────┤
  │          ↓             │       │          ↓                       │
  │  (空き領域)            │       │  (空き領域)                      │
  │          ↑             │       │          ↑                       │
  ├────────────────────────┤       ├──────────────────────────────────┤
  │ ヒープ (malloc/new)    │       │ ヒープ  ← 全スレッドで共有       │
  ├────────────────────────┤       ├──────────────────────────────────┤
  │ BSS  (未初期化グローバル)│       │ BSS   ← 全スレッドで共有       │
  ├────────────────────────┤       ├──────────────────────────────────┤
  │ データ (初期化グローバル)│       │ データ ← 全スレッドで共有       │
  ├────────────────────────┤       ├──────────────────────────────────┤
  │ テキスト (機械語命令)   │       │ テキスト ← 全スレッドで共有     │
  └────────────────────────┘       └──────────────────────────────────┘
```

**なぜスレッドは各自のスタックを持つのか？** 各スレッドは独立した実行コンテキスト（ローカル変数、関数呼び出し履歴、リターンアドレス）を必要とするためである。スタックを共有すると、あるスレッドの関数呼び出しが別スレッドのローカル変数を上書きしてしまう。

### 1.2 スレッドが共有するものと固有のもの

| リソース | 共有/固有 | 理由 |
|---------|----------|------|
| コードセクション | 共有 | 同じプログラムを実行するため |
| データセクション（グローバル変数） | 共有 | プロセス全体の状態を保持するため |
| ヒープ | 共有 | 動的メモリ確保はスレッド間で見える必要がある |
| 開いているファイル | 共有 | ファイルディスクリプタテーブルはプロセス単位 |
| シグナルハンドラ | 共有 | プロセス単位でシグナルの処理方法を定義 |
| スタック | **固有** | 各スレッドの実行コンテキストを保持するため |
| プログラムカウンタ（PC） | **固有** | 各スレッドは異なる命令を実行中であるため |
| レジスタセット | **固有** | CPUの演算状態はスレッドごとに異なるため |
| スレッドID（TID） | **固有** | OSがスレッドを識別するために必要 |
| シグナルマスク | **固有** | スレッドごとにブロックするシグナルを制御するため |
| errno | **固有** | システムコールのエラーが他スレッドに影響しないため |

### 1.3 スレッドの利点と代償

スレッドがプロセスfork()より圧倒的に軽量である理由を、OSの内部動作から理解する。

**生成コストの差**: `fork()` はプロセスのアドレス空間のコピー（Copy-on-Writeを使用しても、ページテーブルのコピーやVMAの複製が必要）、ファイルディスクリプタテーブルの複製、シグナル設定の複製等を行う。一方、`pthread_create()` は新しいスタック領域の確保とTCB（Thread Control Block）の作成のみで完了する。Linuxカーネルの `clone()` システムコールでは、共有するリソースをフラグで指定でき、スレッド生成時は `CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND` 等を指定することで、メモリ空間やファイルテーブルのコピーをスキップする。

**コンテキストスイッチの差**: 同一プロセス内のスレッド間スイッチでは、アドレス空間が同じであるためTLB（Translation Lookaside Buffer）のフラッシュが不要である。TLBフラッシュはメモリアクセスのレイテンシに直接影響するため、この差は大きい。

**通信コストの差**: プロセス間通信（IPC）ではパイプ、ソケット、共有メモリ等のOSメカニズムが必要であり、データのコピーやシステムコールのオーバーヘッドが生じる。スレッド間通信は共有メモリへの直接アクセスで済むため、ポインタの受け渡しだけで完了する。

| 観点 | プロセス | スレッド |
|------|---------|---------|
| 生成コスト | 大（アドレス空間コピー） | 小（スタック+TCBのみ） |
| コンテキストスイッチ | 重い（TLBフラッシュ発生） | 軽い（TLBフラッシュ不要） |
| メモリ使用量 | 大（独立アドレス空間） | 小（スタックのみ追加） |
| 通信 | IPC必要（コピー発生） | 共有メモリ直接アクセス |
| 障害分離 | 高（独立空間で保護） | 低（1スレッドの暴走で全体崩壊） |
| デバッグ容易性 | 比較的容易 | 困難（非決定的な実行順序） |
| セキュリティ | 高（メモリ保護） | 低（全メモリにアクセス可能） |

### 1.4 POSIX Threads (pthreads) による基本操作

以下は、pthreadsを使ったスレッドの生成、実行、合流の完全な例である。

```c
/* thread_basics.c - スレッドの基本操作を示す完全な例
 *
 * コンパイル: gcc -Wall -pthread -o thread_basics thread_basics.c
 * 実行:       ./thread_basics
 *
 * なぜ -pthread フラグが必要か:
 *   リンカに libpthread をリンクするよう指示する。
 *   pthreads の関数（pthread_create 等）はこのライブラリに含まれている。
 *   -lpthread でも動作するが、-pthread の方がコンパイラフラグも設定するため推奨。
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

/* スレッドに渡す引数構造体 */
typedef struct {
    int thread_id;
    int iterations;
} thread_arg_t;

/*
 * スレッドのエントリポイント関数。
 * 引数: void* 型で受け取り、内部でキャストする（POSIX APIの規約）。
 * 戻り値: void* 型で返し、pthread_join() で受け取れる。
 *
 * なぜ void* なのか:
 *   型安全性を犠牲にする代わりに、任意の型のデータを渡せる
 *   汎用性を実現するためのC言語的設計。
 */
void* worker(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    printf("Thread %d: 開始 (iterations=%d)\n", targ->thread_id, targ->iterations);

    long sum = 0;
    for (int i = 0; i < targ->iterations; i++) {
        sum += i;
    }

    printf("Thread %d: 完了 (sum=%ld)\n", targ->thread_id, sum);

    /* ヒープに戻り値を確保して返す。
     * スタック上のローカル変数を返してはならない。
     * なぜなら、スレッド終了後にスタックは解放されるため。 */
    long* result = malloc(sizeof(long));
    if (result == NULL) {
        perror("malloc");
        return NULL;
    }
    *result = sum;
    return (void*)result;
}

int main(void) {
    const int NUM_THREADS = 4;
    pthread_t threads[NUM_THREADS];
    thread_arg_t args[NUM_THREADS];

    printf("メインスレッド: %d個のワーカースレッドを生成\n", NUM_THREADS);

    /* スレッド生成 */
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].iterations = (i + 1) * 1000000;

        /*
         * pthread_create の引数:
         *   1. &threads[i]  - スレッドIDを格納する変数
         *   2. NULL         - スレッド属性（NULLでデフォルト）
         *   3. worker       - スレッドが実行する関数
         *   4. &args[i]     - 関数に渡す引数
         *
         * 戻り値: 0なら成功、非0ならエラーコード
         */
        int ret = pthread_create(&threads[i], NULL, worker, &args[i]);
        if (ret != 0) {
            fprintf(stderr, "pthread_create failed: %d\n", ret);
            exit(EXIT_FAILURE);
        }
    }

    /* 全スレッドの完了を待機（join） */
    for (int i = 0; i < NUM_THREADS; i++) {
        void* retval;
        /*
         * pthread_join はスレッドの終了を待ち、戻り値を受け取る。
         * なぜ join が必要か:
         *   1. スレッドの完了を保証する（結果の利用前に）
         *   2. スレッドのリソース（TCB等）を解放する
         *   join しないスレッドはゾンビ状態になり、リソースリークの原因となる。
         */
        int ret = pthread_join(threads[i], &retval);
        if (ret != 0) {
            fprintf(stderr, "pthread_join failed: %d\n", ret);
            exit(EXIT_FAILURE);
        }
        if (retval != NULL) {
            printf("メインスレッド: Thread %d の結果 = %ld\n", i, *(long*)retval);
            free(retval);  /* worker() で malloc したメモリを解放 */
        }
    }

    printf("メインスレッド: 全スレッド完了\n");
    return 0;
}
```

**想定される実行結果**（スレッドの実行順序は非決定的であるため、出力順は実行のたびに変わりうる）:

```
メインスレッド: 4個のワーカースレッドを生成
Thread 0: 開始 (iterations=1000000)
Thread 2: 開始 (iterations=3000000)
Thread 1: 開始 (iterations=2000000)
Thread 3: 開始 (iterations=4000000)
Thread 0: 完了 (sum=499999500000)
Thread 1: 完了 (sum=1999999000000)
Thread 2: 完了 (sum=4499998500000)
Thread 3: 完了 (sum=7999998000000)
メインスレッド: Thread 0 の結果 = 499999500000
メインスレッド: Thread 1 の結果 = 1999999000000
メインスレッド: Thread 2 の結果 = 4499998500000
メインスレッド: Thread 3 の結果 = 7999998000000
メインスレッド: 全スレッド完了
```

---

## 2. スレッドモデル ―― ユーザレベル/カーネルレベルの実装

### 2.1 スレッドの実装階層

スレッドの実装には大きく3つのモデルがある。どのモデルを採用するかにより、性能特性、マルチコア活用の可否、ブロッキングの挙動が大きく変わる。

```
スレッド実装の3モデル比較:

┌─────────────────────────────────────────────────────────────────────┐
│ (1) N:1 モデル（ユーザレベルスレッド）                               │
│                                                                     │
│  ユーザ空間  ┌────────────────────────────────────────┐             │
│             │  スレッドライブラリ（ランタイム）         │             │
│             │  ┌──┐ ┌──┐ ┌──┐ ┌──┐                   │             │
│             │  │T1│ │T2│ │T3│ │T4│  ← ユーザスレッド  │             │
│             │  └──┘ └──┘ └──┘ └──┘                   │             │
│             │         │ スケジューラ                   │             │
│             │         ↓                               │             │
│  ────────── │ ─────────────────────────── ── ── ── ── │             │
│  カーネル空間│     ┌──────┐                             │             │
│             │     │ KT 1 │ ← カーネルスレッド(1個のみ)  │             │
│             │     └──────┘                             │             │
│             └────────────────────────────────────────┘             │
│  特徴: カーネルはスレッドの存在を知らない                            │
│  長所: コンテキストスイッチが超高速（syscall不要、数十ns）            │
│  短所: 1つがブロック → 全スレッドがブロック                          │
│        マルチコアを活用不可（カーネルスレッドは1つ）                  │
│  例: GNU Portable Threads, 初期のJava Green Threads                 │
├─────────────────────────────────────────────────────────────────────┤
│ (2) 1:1 モデル（カーネルレベルスレッド）                             │
│                                                                     │
│  ユーザ空間   ┌──┐ ┌──┐ ┌──┐ ┌──┐  ← ユーザスレッド               │
│              │T1│ │T2│ │T3│ │T4│                                   │
│              └──┘ └──┘ └──┘ └──┘                                   │
│               │    │    │    │    ← 1対1のマッピング                │
│               ↓    ↓    ↓    ↓                                     │
│  ──────────────────────────────────                                 │
│  カーネル空間 ┌──┐ ┌──┐ ┌──┐ ┌──┐                                  │
│              │K1│ │K2│ │K3│ │K4│  ← カーネルスレッド               │
│              └──┘ └──┘ └──┘ └──┘                                   │
│  特徴: 各ユーザスレッドがカーネルスレッドと1対1で対応                │
│  長所: マルチコアを活用可能、個別にブロッキング可能                  │
│  短所: 生成・切替にsyscall必要（数百ns〜数us）                      │
│        スレッド数に上限あり（カーネルリソースの制約）                 │
│  例: Linux NPTL, Windows Threads, macOS pthreads                    │
├─────────────────────────────────────────────────────────────────────┤
│ (3) M:N モデル（ハイブリッド）                                      │
│                                                                     │
│  ユーザ空間  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐  ← M個のユーザスレッド │
│             │T1│ │T2│ │T3│ │T4│ │T5│ │T6│                         │
│             └──┘ └──┘ └──┘ └──┘ └──┘ └──┘                         │
│              ↓  ↗  ↓    ↓  ↗  ↓                                   │
│  ────────────────────────────────────                               │
│  カーネル空間 ┌──┐      ┌──┐       ← N個のカーネルスレッド          │
│              │K1│      │K2│       (N < M)                          │
│              └──┘      └──┘                                        │
│  特徴: ユーザ空間のスケジューラが M→N のマッピングを動的に管理       │
│  長所: マルチコア活用可能、大量のスレッドを低コストで生成可能         │
│  短所: 実装が非常に複雑（スケジューラの設計が難しい）                │
│  例: Go goroutine, Erlang process, 旧Solaris LWP                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 なぜLinuxは1:1モデルを選んだのか

Linuxは当初、LinuxThreads という POSIX準拠度の低いスレッド実装を使用していた。その後、2つの競合する実装が提案された:

1. **NGPT (Next Generation POSIX Threads)**: IBMが開発したM:Nモデル
2. **NPTL (Native POSIX Threads Library)**: Red Hatが開発した1:1モデル

最終的にNPTL（1:1モデル）が採用された理由は以下の通りである:

- **Linuxの `clone()` が高速**: Linuxカーネルのスレッド生成は十分に最適化されており、ユーザ空間でのスケジューリングの複雑さに見合うほどの性能差がM:Nモデルでは得られなかった
- **M:Nモデルの複雑さ**: ユーザ空間スケジューラとカーネルスケジューラの二重管理は、優先度逆転やシグナル配送の問題を引き起こす
- **POSIX準拠の容易さ**: 1:1モデルの方がPOSIXセマンティクスを正確に実装しやすい

### 2.3 GoのGoroutine ―― M:Nモデルの成功例

Goは M:N モデルを採用し、数十万〜数百万の goroutine を少数のOSスレッド上で効率的に多重化する。成功の要因は以下である:

- **言語レベルでの統合**: ランタイムに組み込まれており、ライブラリとして後付けしたものではない
- **協調的プリエンプション**: Go 1.14以降、ゴルーチンは関数呼び出し時のスタックチェックに加え、非同期シグナルによるプリエンプションもサポート
- **Growable Stack**: 初期スタックサイズは2KB（OSスレッドの1〜8MBに比べ極めて小さい）で、必要に応じて動的に拡張される
- **Netpoller統合**: I/Oブロッキング時にゴルーチンをパーキングし、別のゴルーチンを実行する

```
Go ランタイムの GMP モデル:

  G (Goroutine): 実行単位。数百万個生成可能。
  M (Machine):   OSスレッド。通常はCPUコア数程度。
  P (Processor): 論理プロセッサ。GOMAXPROCSで設定。

  ┌─────────────────────────────────────────────┐
  │               Global Run Queue              │
  │  [G5] [G6] [G7] [G8] ...                   │
  └──────────────────┬──────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │  P0     │  │  P1     │  │  P2     │ ← 論理プロセッサ
   │ LocalQ: │  │ LocalQ: │  │ LocalQ: │
   │ [G1]   │  │ [G3]   │  │ [G9]   │
   │ [G2]   │  │ [G4]   │  │        │
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        ↓            ↓            ↓
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │  M0     │  │  M1     │  │  M2     │ ← OSスレッド
   │ (CPU 0) │  │ (CPU 1) │  │ (CPU 2) │
   └─────────┘  └─────────┘  └─────────┘

  Work Stealing:
    P2のキューが空になると、P0やP1のキューから
    Goroutineを「盗んで」くる → 負荷分散を実現
```

### 2.4 Javaにおけるスレッド（1:1 から Virtual Threads へ）

```java
/* JavaThreadDemo.java - Java のスレッド基本操作
 *
 * コンパイル・実行: javac JavaThreadDemo.java && java JavaThreadDemo
 *
 * Java のスレッドは長年 1:1 モデル（OS スレッドと直接対応）であった。
 * Java 21 (2023) で Virtual Threads が正式導入され、M:N モデルに移行した。
 */

import java.util.ArrayList;
import java.util.List;

public class JavaThreadDemo {

    /* 共有カウンタ: volatile だけでは不十分な理由を示す */
    private static int counter = 0;
    private static final Object lock = new Object();

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Platform Threads (従来の 1:1 スレッド) ===");
        platformThreadDemo();

        System.out.println("\n=== Virtual Threads (Java 21+ の M:N スレッド) ===");
        virtualThreadDemo();
    }

    static void platformThreadDemo() throws InterruptedException {
        counter = 0;
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < 4; i++) {
            final int id = i;
            /*
             * Thread.ofPlatform() は Java 21+ の新しいAPI。
             * 従来の new Thread() と同等だが、設定が宣言的に書ける。
             * 内部では OS の pthread_create() が呼ばれる。
             */
            Thread t = Thread.ofPlatform()
                .name("worker-" + id)
                .start(() -> {
                    for (int j = 0; j < 100_000; j++) {
                        /*
                         * synchronized ブロック: Java の組み込み Mutex。
                         * モニターロックとも呼ばれ、JVM が管理する。
                         * なぜ synchronized が必要か:
                         *   counter++ は read-modify-write の3ステップであり、
                         *   アトミックではないため、保護なしでは競合状態が発生する。
                         */
                        synchronized (lock) {
                            counter++;
                        }
                    }
                });
            threads.add(t);
        }

        for (Thread t : threads) {
            t.join(); /* スレッドの完了を待機 */
        }

        System.out.println("Counter = " + counter + " (期待値: 400000)");
    }

    static void virtualThreadDemo() throws InterruptedException {
        counter = 0;
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < 10_000; i++) {
            /*
             * Thread.ofVirtual(): Java 21 の Virtual Threads。
             * OS スレッドではなく、JVM が管理する軽量スレッド。
             * Go の goroutine と同様の M:N モデルで動作する。
             *
             * なぜ Virtual Threads が導入されたか:
             *   OS スレッドは1つあたり ~1MB のスタックを消費するため、
             *   数万スレッドの生成は非現実的だった。
             *   Virtual Threads は ~1KB から開始し、必要に応じて拡張される。
             */
            Thread t = Thread.ofVirtual()
                .start(() -> {
                    synchronized (lock) {
                        counter++;
                    }
                });
            threads.add(t);
        }

        for (Thread t : threads) {
            t.join();
        }

        System.out.println("Counter = " + counter + " (期待値: 10000)");
        System.out.println("10,000 Virtual Threads を生成・完了");
    }
}
```

---

## 3. 同期プリミティブの詳細

### 3.1 競合状態（Race Condition）の発生メカニズム

競合状態は、複数のスレッドが共有データに対して**非アトミック**な操作を同時に行うことで発生する。CPU命令レベルで見ると、`counter++` のような単純な操作でさえ、複数の命令（load, add, store）に分解されるため、割り込みのタイミングにより予期しない結果となる。

```
counter++ の命令レベル分解（x86アーキテクチャ）:

  C言語:  counter++;
  asm:    mov eax, [counter]   ; (1) メモリからレジスタへロード
          add eax, 1           ; (2) レジスタ上でインクリメント
          mov [counter], eax   ; (3) レジスタからメモリへストア

  2つのスレッドが同時に実行した場合のインターリーブ:

  時刻   Thread A               Thread B               counter (メモリ)
  ─────┬──────────────────────┬──────────────────────┬──────────
   t1  │ mov eax, [counter]   │                      │ 0
       │ eax_A = 0            │                      │
  ─────┼──────────────────────┼──────────────────────┼──────────
   t2  │                      │ mov eax, [counter]   │ 0
       │                      │ eax_B = 0            │
  ─────┼──────────────────────┼──────────────────────┼──────────
   t3  │ add eax, 1           │                      │ 0
       │ eax_A = 1            │                      │
  ─────┼──────────────────────┼──────────────────────┼──────────
   t4  │                      │ add eax, 1           │ 0
       │                      │ eax_B = 1            │
  ─────┼──────────────────────┼──────────────────────┼──────────
   t5  │ mov [counter], eax   │                      │ 1 ← Aが書込
  ─────┼──────────────────────┼──────────────────────┼──────────
   t6  │                      │ mov [counter], eax   │ 1 ← Bが上書き!
  ─────┴──────────────────────┴──────────────────────┴──────────

  結果: 2回インクリメントしたはずなのに counter = 1（Lost Update）
```

### 3.2 Mutex（相互排除ロック）

Mutex は最も基本的な同期プリミティブである。「鍵のかかった部屋」に例えられ、一度に1つのスレッドだけがクリティカルセクションに入れる。

**なぜ Mutex が必要か**: 競合状態を防ぐには、read-modify-write 操作の**原子性（atomicity）**を保証する必要がある。Mutex はクリティカルセクションの入り口でロックを取得し、出口でロックを解放することで、そのセクションを実行できるスレッドを1つに限定する。

```c
/* mutex_demo.c - Mutex による競合状態の防止
 *
 * コンパイル: gcc -Wall -pthread -o mutex_demo mutex_demo.c
 * 実行:       ./mutex_demo
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS  4
#define ITERATIONS   1000000

/* 共有カウンタとMutex */
static long counter_unsafe = 0;
static long counter_safe   = 0;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
/*
 * PTHREAD_MUTEX_INITIALIZER はスタティック初期化マクロ。
 * 動的初期化の pthread_mutex_init() と同等だが、
 * グローバル/static変数にのみ使用可能。
 * なぜ便利か: init/destroy の呼び出しが不要でコードが簡潔になる。
 */

void* unsafe_worker(void* arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        counter_unsafe++;  /* 保護なし: 競合状態が発生する */
    }
    return NULL;
}

void* safe_worker(void* arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        /*
         * pthread_mutex_lock: ロックを取得。既に他スレッドがロック中なら待機。
         * 内部ではfutex (Fast Userspace muTEX) syscallを使用:
         *   1. まずアトミック操作でロック取得を試みる（高速パス）
         *   2. 失敗した場合のみカーネルに待機を依頼（低速パス）
         * この2段階設計により、競合がない場合はsyscallを回避できる。
         */
        pthread_mutex_lock(&mtx);
        counter_safe++;
        pthread_mutex_unlock(&mtx);
    }
    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    /* 保護なしの場合 */
    printf("--- 保護なし ---\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, unsafe_worker, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("期待値: %d, 結果: %ld (競合により減少する可能性が高い)\n",
           NUM_THREADS * ITERATIONS, counter_unsafe);

    /* Mutex保護ありの場合 */
    printf("\n--- Mutex保護あり ---\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, safe_worker, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("期待値: %d, 結果: %ld (常に一致する)\n",
           NUM_THREADS * ITERATIONS, counter_safe);

    return 0;
}
```

### 3.3 セマフォ

セマフォは Mutex を一般化したもので、**N個のスレッドが同時にクリティカルセクションに入れる**。内部カウンタの値がN（許可数）を表す。

- **P操作 (wait/down)**: カウンタをデクリメント。0の場合はブロック。
- **V操作 (signal/up)**: カウンタをインクリメント。待機中のスレッドがあれば起床させる。

| 用途 | カウンタ初期値 | 説明 |
|------|-------------|------|
| バイナリセマフォ | 1 | Mutexと同等（ただし所有者の概念がない） |
| カウンティングセマフォ | N | 同時アクセス数を制限（接続プール等） |
| シグナリング | 0 | スレッド間の通知に使用（生産者→消費者） |

**Mutex とセマフォの根本的な違い**: Mutex には「所有権」の概念がある。ロックを取得したスレッドのみがアンロックできる。セマフォにはこの制約がなく、あるスレッドが wait し、別のスレッドが signal できる。この特性により、セマフォは生産者-消費者パターンのシグナリングに適している。

### 3.4 条件変数（Condition Variable）

条件変数は「ある条件が成立するまで効率的に待機する」ための仕組みである。Mutex と組み合わせて使用する。

**なぜ Mutex だけでは不十分か**: Mutex はクリティカルセクションの排他制御を提供するが、「バッファが空でなくなるまで待つ」のような条件待ちには不向きである。ビジーウェイト（条件をループで確認し続ける）はCPU時間を浪費する。条件変数は、条件が不成立の場合にスレッドをスリープさせ、条件変更時に起床させることで、CPUの浪費を回避する。

```python
"""producer_consumer.py - 条件変数を使った生産者-消費者パターン

このパターンは以下のような場面で使われる:
  - Webサーバのリクエストキュー
  - ログ書き込みバッファ
  - 動画エンコードのフレームバッファ

実行: python3 producer_consumer.py
"""
import threading
import time
import random
from collections import deque

BUFFER_SIZE = 5  # バッファの最大サイズ

buffer = deque()
lock = threading.Lock()
not_empty = threading.Condition(lock)  # バッファが空でないことを通知
not_full  = threading.Condition(lock)  # バッファが満杯でないことを通知

produced_count = 0
consumed_count = 0

def producer(producer_id: int, num_items: int) -> None:
    """アイテムを生産してバッファに追加する。

    バッファが満杯の場合は not_full 条件変数で待機する。
    なぜ while ループで条件を確認するか（if ではなく）:
      Spurious Wakeup（偽の起床）の可能性があるため。
      OSの実装上、signal/broadcast なしに条件変数から起床することがある。
      while ループにより、起床後に再度条件を確認することで安全性を保つ。
    """
    global produced_count
    for i in range(num_items):
        item = f"P{producer_id}-Item{i}"
        with not_full:
            while len(buffer) >= BUFFER_SIZE:
                print(f"  Producer {producer_id}: バッファ満杯、待機中...")
                not_full.wait()  # ロック解放 + 待機（アトミックに実行）

            buffer.append(item)
            produced_count += 1
            print(f"  Producer {producer_id}: 生産 '{item}' "
                  f"(バッファサイズ: {len(buffer)})")
            not_empty.notify()  # 消費者を1つ起床

        time.sleep(random.uniform(0.01, 0.05))  # 生産時間をシミュレート


def consumer(consumer_id: int, num_items: int) -> None:
    """バッファからアイテムを取り出して消費する。

    バッファが空の場合は not_empty 条件変数で待機する。
    """
    global consumed_count
    for _ in range(num_items):
        with not_empty:
            while len(buffer) == 0:
                print(f"  Consumer {consumer_id}: バッファ空、待機中...")
                not_empty.wait()

            item = buffer.popleft()
            consumed_count += 1
            print(f"  Consumer {consumer_id}: 消費 '{item}' "
                  f"(バッファサイズ: {len(buffer)})")
            not_full.notify()  # 生産者を1つ起床

        time.sleep(random.uniform(0.02, 0.08))  # 消費時間をシミュレート


def main() -> None:
    NUM_ITEMS_PER_PRODUCER = 5
    NUM_ITEMS_PER_CONSUMER = 5

    producers = [
        threading.Thread(target=producer, args=(i, NUM_ITEMS_PER_PRODUCER))
        for i in range(2)
    ]
    consumers = [
        threading.Thread(target=consumer, args=(i, NUM_ITEMS_PER_CONSUMER))
        for i in range(2)
    ]

    print("=== 生産者-消費者パターン開始 ===")
    print(f"バッファサイズ上限: {BUFFER_SIZE}")
    print(f"生産者: 2, 消費者: 2\n")

    for t in producers + consumers:
        t.start()
    for t in producers + consumers:
        t.join()

    print(f"\n=== 完了 ===")
    print(f"生産数: {produced_count}, 消費数: {consumed_count}")


if __name__ == "__main__":
    main()
```

### 3.5 読み書きロック（RWLock）

読み書きロックは「読み取りは複数同時OK、書き込みは排他的」という制約を実現する。読み取りが圧倒的に多いワークロード（例: キャッシュ、設定データ、DNS テーブル）で効果的である。

```
RWLock の状態遷移図:

         ┌─────────────────────────────────────────┐
         │                                         │
         ↓                                         │
    ┌──────────┐    読み取りロック取得     ┌──────────────┐
    │          │ ──────────────────→   │              │
    │  FREE    │                       │ READ_LOCKED  │
    │ (未ロック) │ ←──────────────────   │ (reader数≧1) │
    │          │    最後のreaderが解放   │              │
    └──────────┘                       └──────────────┘
         │                                    │
         │  書き込みロック取得                   │ 書き込みロック取得
         │  (reader数が0の場合のみ)              │ → 全reader解放まで待機
         ↓                                    ↓
    ┌──────────────┐
    │              │  ← 読み取りロック取得はブロック
    │ WRITE_LOCKED │  ← 他の書き込みロック取得もブロック
    │ (writer=1)   │
    │              │
    └──────────────┘

  並行性の比較:
    Mutex:   [R] [R] [W] [R] [R] [W]  ← 全てが直列
    RWLock:  [R R R] [W] [R R] [W]    ← 読み取りは並行可能
```

### 3.6 同期プリミティブ比較表

| プリミティブ | 同時アクセス数 | 所有権 | 用途 | オーバーヘッド |
|------------|-------------|--------|------|-------------|
| Mutex | 1 | あり（ロック取得者のみ解放可） | クリティカルセクションの排他制御 | 低（futex最適化） |
| スピンロック | 1 | あり | 極短時間のクリティカルセクション（カーネル内部） | 最低（ビジーウェイト） |
| セマフォ | N（可変） | なし（誰でもsignal可） | リソースプール、シグナリング | 中 |
| 条件変数 | - | なし（Mutexと併用） | 条件待ち（生産者-消費者等） | 中 |
| RWLock | 読み取り:無制限、書き込み:1 | あり | 読み取り中心のワークロード | 中〜高 |
| バリア | - | なし | 全スレッドの同期点（フェーズ境界） | 中 |

---

## 4. デッドロックの詳細分析

### 4.1 Coffman条件（デッドロックの必要十分条件）

デッドロックが発生するには、以下の4条件が**全て同時に**成立する必要がある。逆に言えば、1つでも崩せばデッドロックは回避できる。

1. **相互排除（Mutual Exclusion）**: リソースが排他的に使用される（1スレッドのみがロックを保持）
2. **保持と待機（Hold and Wait）**: あるリソースを保持したまま、別のリソースの取得を待つ
3. **非横取り（No Preemption）**: 他スレッドが保持中のリソースを強制的に奪えない
4. **循環待ち（Circular Wait）**: スレッド間でリソースの待ち関係が循環する

```
デッドロックの循環待ち図:

  Thread A                    Thread B
  ┌─────────┐                ┌─────────┐
  │ lock(X) │ ← 保持         │ lock(Y) │ ← 保持
  │         │                │         │
  │ lock(Y) │ ── 待機 ──→   │ lock(X) │ ── 待機 ──┐
  └─────────┘       │        └─────────┘           │
                    │                               │
                    └──────── 循環! ────────────────┘

  リソース割当グラフ（RAG）:
    T_A ──→ R_Y ──→ T_B ──→ R_X ──→ T_A
    (要求)  (保持)  (要求)  (保持)   ← サイクル検出!
```

### 4.2 デッドロック回避戦略

| 戦略 | 崩す条件 | 方法 | 制約 |
|------|---------|------|------|
| ロック順序の統一 | 循環待ち | 全スレッドが同じ順序でロック取得 | 全ロックの順序を事前に決定する必要 |
| タイムアウト付きロック | 保持と待機 | trylock + タイムアウトで諦める | リトライロジックが必要 |
| 一括取得 | 保持と待機 | 全てのロックを一度に取得 | 並行性が低下する |
| ロックフリーアルゴリズム | 相互排除 | CAS等のアトミック操作を使用 | 実装が非常に困難 |
| 銀行家のアルゴリズム | （予防） | リソース割当前に安全性を検証 | 事前にリソース要求量を知る必要 |

### 4.3 デッドロックの実演と回避（C言語）

```c
/* deadlock_demo.c - デッドロックの発生と回避を示す
 *
 * コンパイル: gcc -Wall -pthread -o deadlock_demo deadlock_demo.c
 * 実行:       ./deadlock_demo
 *
 * このプログラムは2つのシナリオを示す:
 *   1. デッドロックが発生するコード（ロック順序が不統一）
 *   2. デッドロックを回避するコード（ロック順序を統一）
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

/* 2つの共有リソースに対応するMutex */
static pthread_mutex_t lock_account_a = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lock_account_b = PTHREAD_MUTEX_INITIALIZER;

static int balance_a = 1000;
static int balance_b = 1000;

/* ===== デッドロックが発生するバージョン ===== */

void* transfer_a_to_b_UNSAFE(void* arg) {
    (void)arg;
    printf("[UNSAFE] Thread 1: lock_account_a を取得\n");
    pthread_mutex_lock(&lock_account_a);

    /* sleep で割り込みタイミングを作る。
     * 本番コードでは sleep がなくてもスケジューラの都合で
     * 同じインターリーブが発生しうる。 */
    usleep(100000);  /* 0.1秒 */

    printf("[UNSAFE] Thread 1: lock_account_b を待機...\n");
    pthread_mutex_lock(&lock_account_b);

    balance_a -= 100;
    balance_b += 100;
    printf("[UNSAFE] Thread 1: 送金完了 A=%d, B=%d\n", balance_a, balance_b);

    pthread_mutex_unlock(&lock_account_b);
    pthread_mutex_unlock(&lock_account_a);
    return NULL;
}

void* transfer_b_to_a_UNSAFE(void* arg) {
    (void)arg;
    printf("[UNSAFE] Thread 2: lock_account_b を取得\n");
    pthread_mutex_lock(&lock_account_b);  /* ← 順序が逆! */

    usleep(100000);

    printf("[UNSAFE] Thread 2: lock_account_a を待機...\n");
    pthread_mutex_lock(&lock_account_a);  /* ← デッドロック! */

    balance_b -= 200;
    balance_a += 200;
    printf("[UNSAFE] Thread 2: 送金完了 A=%d, B=%d\n", balance_a, balance_b);

    pthread_mutex_unlock(&lock_account_a);
    pthread_mutex_unlock(&lock_account_b);
    return NULL;
}

/* ===== デッドロックを回避するバージョン ===== */

/*
 * 戦略: ロック順序の統一
 * Mutex のアドレスを比較し、常に小さい方を先にロックする。
 * これにより循環待ちが発生し得なくなる。
 *
 * なぜアドレス比較が有効か:
 *   全てのMutexに一意の全順序を定義できるため。
 *   全スレッドがこの順序に従えば、循環待ちのサイクルは形成されない。
 */
void transfer_safe(pthread_mutex_t* from_lock, int* from_balance,
                   pthread_mutex_t* to_lock, int* to_balance,
                   int amount, const char* label) {
    /* アドレスの小さい方を先にロック */
    pthread_mutex_t* first  = (from_lock < to_lock) ? from_lock : to_lock;
    pthread_mutex_t* second = (from_lock < to_lock) ? to_lock : from_lock;

    printf("[SAFE] %s: first lock を取得\n", label);
    pthread_mutex_lock(first);
    usleep(100000);
    printf("[SAFE] %s: second lock を取得\n", label);
    pthread_mutex_lock(second);

    *from_balance -= amount;
    *to_balance   += amount;
    printf("[SAFE] %s: 送金完了 A=%d, B=%d\n",
           label, balance_a, balance_b);

    pthread_mutex_unlock(second);
    pthread_mutex_unlock(first);
}

void* safe_a_to_b(void* arg) {
    (void)arg;
    transfer_safe(&lock_account_a, &balance_a,
                  &lock_account_b, &balance_b, 100, "Thread 1");
    return NULL;
}

void* safe_b_to_a(void* arg) {
    (void)arg;
    transfer_safe(&lock_account_b, &balance_b,
                  &lock_account_a, &balance_a, 200, "Thread 2");
    return NULL;
}

/* ===== タイムアウト付きロックによる回避 ===== */

void* transfer_with_timeout(void* arg) {
    (void)arg;
    int retries = 0;
    const int MAX_RETRIES = 5;

    while (retries < MAX_RETRIES) {
        pthread_mutex_lock(&lock_account_b);
        usleep(100000);

        /*
         * pthread_mutex_trylock: ロック取得を試み、失敗したら即座にEBUSYを返す。
         * ブロックしないため、デッドロックを回避できる。
         * ただし、持っているロックを全て解放してリトライする必要がある。
         */
        int ret = pthread_mutex_trylock(&lock_account_a);
        if (ret == 0) {
            /* ロック取得成功 */
            balance_b -= 200;
            balance_a += 200;
            printf("[TIMEOUT] 送金成功 (リトライ %d回) A=%d, B=%d\n",
                   retries, balance_a, balance_b);
            pthread_mutex_unlock(&lock_account_a);
            pthread_mutex_unlock(&lock_account_b);
            return NULL;
        }

        /* ロック取得失敗: 保持中のロックを解放してバックオフ */
        pthread_mutex_unlock(&lock_account_b);
        retries++;
        printf("[TIMEOUT] ロック取得失敗、リトライ #%d\n", retries);
        usleep(rand() % 100000);  /* ランダムバックオフ */
    }

    printf("[TIMEOUT] 最大リトライ回数に到達、送金失敗\n");
    return NULL;
}

int main(void) {
    srand((unsigned)time(NULL));

    /* --- 安全なバージョンを実行 --- */
    printf("=== ロック順序統一による安全な送金 ===\n");
    balance_a = 1000;
    balance_b = 1000;
    pthread_t t1, t2;
    pthread_create(&t1, NULL, safe_a_to_b, NULL);
    pthread_create(&t2, NULL, safe_b_to_a, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("最終残高: A=%d, B=%d (合計=%d、常に2000)\n\n",
           balance_a, balance_b, balance_a + balance_b);

    /*
     * 注意: UNSAFE バージョンは意図的にデッドロックするため、
     * コメントアウトしている。試す場合は kill コマンドで強制終了する必要がある。
     *
     * printf("=== デッドロックする送金（注意: ハングする可能性あり） ===\n");
     * balance_a = 1000; balance_b = 1000;
     * pthread_create(&t1, NULL, transfer_a_to_b_UNSAFE, NULL);
     * pthread_create(&t2, NULL, transfer_b_to_a_UNSAFE, NULL);
     * pthread_join(t1, NULL);
     * pthread_join(t2, NULL);
     */

    return 0;
}
```

---

## 5. スレッドプール

### 5.1 スレッドプールが必要な理由

リクエストのたびにスレッドを生成・破棄するのは、以下の理由で非効率である:

1. **スレッド生成のコスト**: OSスレッドの生成はスタック領域の確保、カーネル構造体（TCB）の作成、スケジューラへの登録を含むため、数十マイクロ秒のオーバーヘッドがある
2. **スレッド数の爆発**: 高負荷時に大量のリクエストが到着すると、スレッドが際限なく生成され、メモリ枯渇やスケジューリングオーバーヘッドの増大を招く
3. **スレッドの再利用**: 一度生成したスレッドを使い回すことで、生成・破棄のコストを償却できる

```
スレッドプールの動作モデル:

  タスクキュー                  ワーカースレッド群
  ┌─────────────────────┐     ┌─────────────────────────┐
  │                     │     │  Worker 0: [実行中]      │
  │ [Task 5] ← 新規追加 │     │  Worker 1: [待機中]      │
  │ [Task 4]            │────→│  Worker 2: [実行中]      │
  │ [Task 3]            │     │  Worker 3: [待機中]      │
  │                     │     │  Worker 4: [実行中]      │
  └─────────────────────┘     └─────────────────────────┘
         ↑                           │
         │                           ↓
    submit(task)               結果/コールバック

  ライフサイクル:
    1. プール初期化: N個のワーカースレッドを事前に生成
    2. タスク投入:   submit(task) でキューに追加
    3. タスク実行:   空きワーカーがキューからタスクを取り出し実行
    4. タスク完了:   ワーカーは次のタスクを待機（スレッドは破棄しない）
    5. プール停止:   全ワーカーを停止し、リソースを解放

  ワーカースレッドの内部ループ:
    while (pool->running) {
        task = queue_pop(pool->queue);  // 空ならブロック
        task->function(task->arg);      // タスクを実行
    }
```

### 5.2 スレッドプールのサイジング

スレッドプールのサイズ（ワーカースレッド数）はパフォーマンスに直結する。

- **小さすぎる**: CPUが遊んでしまい、スループットが低下する
- **大きすぎる**: コンテキストスイッチのオーバーヘッドが増大し、メモリを浪費する

**CPU密度の高いタスク（CPU-bound）**の場合:
- 想定される最適スレッド数 = CPUコア数 (N)
- 理由: CPU密度の高いタスクはCPUを常に使用するため、コア数以上のスレッドはコンテキストスイッチのオーバーヘッドを増やすだけである

**I/O密度の高いタスク（I/O-bound）**の場合:
- 想定される最適スレッド数 = N * (1 + W/C)
  - N: CPUコア数
  - W: I/O待ち時間
  - C: 計算時間
- 例: I/O待ちが計算の9倍なら、N * 10 スレッド

| ワークロード | 推奨スレッド数 | 根拠 |
|------------|-------------|------|
| 画像処理、暗号化 | CPUコア数 | CPUがボトルネック。スレッド増はスイッチコスト増 |
| Webサーバ（DB問合せ）| コア数 * 10〜50 | I/O待ち中にCPUが空くため |
| ファイルダウンロード | コア数 * 50〜200 | ネットワークI/O待ちが大半 |

### 5.3 Pythonによるスレッドプール実装

```python
"""thread_pool.py - スレッドプールの基本実装

動作環境: Python 3.8+
実行: python3 thread_pool.py

この実装では以下を学ぶ:
  - 条件変数を使ったタスクキューの実装
  - ワーカースレッドのライフサイクル管理
  - グレースフルシャットダウンの方法
"""
import threading
import time
import random
from collections import deque
from typing import Callable, Any, Optional


class ThreadPool:
    """固定サイズのスレッドプール実装。

    なぜ固定サイズか:
      動的にスレッド数を変更する実装は複雑になる。
      固定サイズでもサイジングを適切に行えば十分な性能が得られる。
      Java の ThreadPoolExecutor は動的サイズをサポートするが、
      core/max/keepAlive の設定が複雑になりバグの温床になりやすい。
    """

    def __init__(self, num_workers: int) -> None:
        if num_workers <= 0:
            raise ValueError("ワーカー数は1以上でなければならない")

        self._num_workers = num_workers
        self._task_queue: deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._running = True
        self._workers: list[threading.Thread] = []

        # ワーカースレッドを生成・開始
        for i in range(num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"PoolWorker-{i}",
                daemon=True,  # メインスレッド終了時に自動終了
            )
            t.start()
            self._workers.append(t)

        print(f"ThreadPool: {num_workers}個のワーカースレッドを開始")

    def _worker_loop(self, worker_id: int) -> None:
        """ワーカースレッドのメインループ。

        タスクが到着するまで条件変数で待機し、
        到着したら取り出して実行する。
        """
        while True:
            with self._not_empty:
                # タスクが来るか、シャットダウン指示が出るまで待機
                while len(self._task_queue) == 0 and self._running:
                    self._not_empty.wait()

                # シャットダウン指示かつキューが空なら終了
                if not self._running and len(self._task_queue) == 0:
                    print(f"  Worker {worker_id}: シャットダウン")
                    return

                func, args, kwargs = self._task_queue.popleft()

            # ロック外でタスクを実行（長時間のタスクでもキューをブロックしない）
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"  Worker {worker_id}: タスク実行中にエラー: {e}")

    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """タスクをキューに追加する。"""
        with self._not_empty:
            if not self._running:
                raise RuntimeError("シャットダウン済みのプールにタスクを追加できない")
            self._task_queue.append((func, args, kwargs))
            self._not_empty.notify()  # 待機中のワーカーを1つ起床

    def shutdown(self, wait: bool = True) -> None:
        """プールをシャットダウンする。

        wait=True の場合、キュー内の全タスクの完了を待つ。
        なぜグレースフルシャットダウンが重要か:
          - 実行中のタスクが途中で中断されるとデータ不整合の原因になる
          - キュー内の未処理タスクが失われる
        """
        print("ThreadPool: シャットダウン開始")
        with self._not_empty:
            self._running = False
            self._not_empty.notify_all()  # 全ワーカーを起床して終了させる

        if wait:
            for t in self._workers:
                t.join()
        print("ThreadPool: シャットダウン完了")


# === デモ用のタスク関数 ===

def simulate_io_task(task_id: int) -> None:
    """I/O密度の高いタスクをシミュレート"""
    worker_name = threading.current_thread().name
    print(f"  [{worker_name}] タスク {task_id}: 開始（I/Oシミュレート）")
    time.sleep(random.uniform(0.1, 0.5))  # I/O待ちを模擬
    print(f"  [{worker_name}] タスク {task_id}: 完了")


def simulate_cpu_task(task_id: int) -> None:
    """CPU密度の高いタスクをシミュレート"""
    worker_name = threading.current_thread().name
    print(f"  [{worker_name}] タスク {task_id}: 開始（CPU計算）")
    total = sum(i * i for i in range(100_000))  # CPU計算
    print(f"  [{worker_name}] タスク {task_id}: 完了 (result={total})")


if __name__ == "__main__":
    pool = ThreadPool(num_workers=3)

    print("\n--- I/Oタスクを投入 ---")
    for i in range(8):
        pool.submit(simulate_io_task, i)

    time.sleep(3)  # タスクの完了を待つ

    print("\n--- CPUタスクを投入 ---")
    for i in range(4):
        pool.submit(simulate_cpu_task, i)

    time.sleep(2)

    pool.shutdown(wait=True)
```

### 5.4 Java の ExecutorService（標準ライブラリのスレッドプール）

```java
/* ExecutorServiceDemo.java - Java標準のスレッドプール
 *
 * コンパイル・実行: javac ExecutorServiceDemo.java && java ExecutorServiceDemo
 *
 * Java の java.util.concurrent パッケージは産業品質のスレッドプールを提供する。
 * Doug Lea による設計で、高度に最適化されている。
 */

import java.util.concurrent.*;
import java.util.ArrayList;
import java.util.List;

public class ExecutorServiceDemo {

    public static void main(String[] args) throws Exception {
        System.out.println("=== FixedThreadPool ===");
        fixedPoolDemo();

        System.out.println("\n=== Future によるタスク結果の取得 ===");
        futureDemo();

        System.out.println("\n=== CachedThreadPool vs FixedThreadPool ===");
        comparisonDemo();
    }

    static void fixedPoolDemo() throws InterruptedException {
        /*
         * newFixedThreadPool(4): 4つのワーカースレッドを持つプール。
         * タスクが4つ以上投入されると、キューに入り順番を待つ。
         *
         * なぜ固定サイズが推奨されるか:
         *   CachedThreadPool は無制限にスレッドを生成するため、
         *   大量のタスクが一度に投入されるとOOM(OutOfMemory)の危険がある。
         *   FixedThreadPool はスレッド数の上限が保証される。
         */
        ExecutorService pool = Executors.newFixedThreadPool(4);

        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            pool.execute(() -> {
                String name = Thread.currentThread().getName();
                System.out.printf("  [%s] Task %d: 開始%n", name, taskId);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                System.out.printf("  [%s] Task %d: 完了%n", name, taskId);
            });
        }

        pool.shutdown();
        /*
         * shutdown(): 新規タスクの受付を停止し、既存タスクの完了を待つ。
         * shutdownNow(): 実行中のタスクに割り込みを送り、即座に停止を試みる。
         *
         * awaitTermination(): 指定時間内に全タスクが完了するまで待機。
         */
        boolean finished = pool.awaitTermination(10, TimeUnit.SECONDS);
        System.out.println("全タスク完了: " + finished);
    }

    static void futureDemo() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(3);
        List<Future<Long>> futures = new ArrayList<>();

        for (int i = 1; i <= 5; i++) {
            final int n = i * 10_000_000;
            /*
             * submit(Callable): 戻り値を持つタスクを投入。
             * Future オブジェクトを返し、get() で結果を取得できる。
             * get() はタスク完了まで呼び出し元をブロックする。
             */
            Future<Long> future = pool.submit(() -> {
                long sum = 0;
                for (int j = 0; j < n; j++) {
                    sum += j;
                }
                return sum;
            });
            futures.add(future);
        }

        for (int i = 0; i < futures.size(); i++) {
            /* get() はブロッキング呼び出し。タイムアウト付きの get(timeout, unit) も利用可能。 */
            long result = futures.get(i).get();
            System.out.printf("  Task %d 結果: %,d%n", i, result);
        }

        pool.shutdown();
        pool.awaitTermination(10, TimeUnit.SECONDS);
    }

    static void comparisonDemo() {
        /*
         * CachedThreadPool:
         *   - スレッド数の上限なし（Integer.MAX_VALUE）
         *   - 60秒間アイドルのスレッドは自動破棄
         *   - 短命で大量のタスクに向くが、暴走の危険あり
         *
         * FixedThreadPool:
         *   - スレッド数が固定
         *   - タスクキューは無制限（LinkedBlockingQueue）
         *   - 安定した負荷に向く
         *
         * WorkStealingPool (Java 8+):
         *   - ForkJoinPool ベース
         *   - 各ワーカーが独自のキューを持ち、他ワーカーからタスクを盗む
         *   - 不均一なタスクサイズに向く
         */
        System.out.println("  CachedThreadPool:     上限なし、短命タスク向き");
        System.out.println("  FixedThreadPool:      固定サイズ、安定負荷向き");
        System.out.println("  WorkStealingPool:     Work Stealing、不均一タスク向き");
        System.out.println("  ScheduledThreadPool:  定期実行タスク向き");
        System.out.println("  VirtualThreadPerTask: Java 21+ 仮想スレッド版");
    }
}
```

---

## 6. 現代の並行性モデル

スレッドは並行プログラミングの基盤であるが、直接操作すると競合状態やデッドロックのリスクが高い。そこで、より高レベルの抽象化を提供する並行性モデルが発展してきた。

### 6.1 async/await（協調的マルチタスク）

```
async/await の動作モデル:

  イベントループ（シングルスレッド）:
  ┌───────────────────────────────────────────────────┐
  │                                                   │
  │   ┌──────┐    ┌──────┐    ┌──────┐               │
  │   │Task A│    │Task B│    │Task C│  ← コルーチン  │
  │   └──┬───┘    └──┬───┘    └──┬───┘               │
  │      │           │           │                    │
  │      ↓           │           │                    │
  │   [実行中]        │           │   t=0              │
  │      │           │           │                    │
  │   await IO ──→  │           │   t=1 (Aが中断)     │
  │      │           ↓           │                    │
  │      │        [実行中]        │   t=2 (Bが実行)     │
  │      │           │           │                    │
  │      │        await IO ──→  │   t=3 (Bが中断)     │
  │      │           │           ↓                    │
  │      │           │        [実行中]  t=4 (Cが実行)   │
  │      ↓           │           │                    │
  │   IO完了         │           │   t=5 (Aが再開)     │
  │   [実行中]        │           │                    │
  │      │           ↓           │                    │
  │   完了       IO完了         │   t=6               │
  │              [実行中]        │                    │
  │                 │           │                    │
  │              完了          │   t=7               │
  │                             ↓                    │
  │                          完了   t=8               │
  └───────────────────────────────────────────────────┘

  ポイント:
  - OSスレッドは1つだけ → 競合状態が原理的に発生しない
  - await で明示的に制御を返す（協調的）
  - I/O待ちの間に別タスクを実行 → I/O密度の高い処理に最適
  - CPU密度の高い処理には不向き（シングルスレッドのため）
```

### 6.2 アクターモデル

```
アクターモデル（Erlang/Elixir, Akka）:

  ┌──────────┐   メッセージ    ┌──────────┐
  │ Actor A  │ ──────────→   │ Actor B  │
  │          │               │          │
  │ 状態: S1 │   メッセージ    │ 状態: S2 │
  │ mailbox: │ ←──────────   │ mailbox: │
  │ [m1, m2] │               │ [m3]     │
  └──────────┘               └──────────┘
       │                          │
       │  メッセージ                │  メッセージ
       ↓                          ↓
  ┌──────────┐               ┌──────────┐
  │ Actor C  │               │ Actor D  │
  │ 状態: S3 │               │ 状態: S4 │
  └──────────┘               └──────────┘

  原則:
  1. 各アクターは独立した状態を持つ（共有メモリなし）
  2. アクター間の通信はメッセージパッシングのみ
  3. メッセージは非同期に送信され、mailbox にキューイングされる
  4. アクターは一度に1つのメッセージのみ処理する

  利点:
  - 共有メモリがないため、ロックが不要 → デッドロックのリスクが大幅に低減
  - アクター単位での障害分離が可能（Erlang の "Let it crash" 哲学）
  - 分散システムへの自然な拡張（ネットワーク越しのメッセージ送信）
```

### 6.3 CSP（Communicating Sequential Processes）

GoのgoroutineとchannelはCSPモデルを採用している。「メモリを共有して通信するな、通信してメモリを共有せよ」というGoの格言は、このモデルの本質を表している。

### 6.4 並行性モデル比較表

| モデル | 共有メモリ | 通信方式 | スケジューリング | 代表的な言語/フレームワーク |
|--------|----------|---------|---------------|----------------------|
| スレッド + ロック | あり | 共有変数 | プリエンプティブ（OS） | C/C++, Java, Python |
| async/await | なし（通常） | Future/Promise | 協調的（イベントループ） | JavaScript, Python asyncio, Rust tokio |
| アクターモデル | なし | メッセージパッシング | プリエンプティブ（ランタイム） | Erlang/Elixir, Akka (Scala) |
| CSP | なし（推奨） | チャネル | M:Nハイブリッド | Go (goroutine + channel) |
| STM | あり（トランザクション） | トランザクショナルメモリ | 楽観的並行制御 | Haskell, Clojure |
| データ並列 | あり（制御下） | 暗黙的同期 | コンパイラ/ランタイム | CUDA, OpenMP, SIMD |

---

## 7. アンチパターン

### 7.1 アンチパターン1: ロックの粒度が大きすぎる（Giant Lock）

```python
"""antipattern_giant_lock.py - ロック粒度が大きすぎるアンチパターン

問題:
  単一のロックで全てのデータを保護すると、並行性が失われる。
  あるスレッドが accounts['A'] を操作中に、
  無関係な accounts['B'] へのアクセスもブロックされてしまう。
"""
import threading
import time

# === アンチパターン: 単一のグローバルロック ===

class BankAccountGiantLock:
    """全アカウントを1つのロックで保護する（並行性が低い）。

    なぜこれが問題か:
      100個のアカウントがあっても、同時に1つのスレッドしか
      どのアカウントにもアクセスできない。
      これは事実上シングルスレッド実行と同じである。
    """
    def __init__(self):
        self.accounts = {'A': 1000, 'B': 1000, 'C': 1000}
        self.lock = threading.Lock()  # 1つのロックで全アカウントを保護

    def transfer(self, from_acc, to_acc, amount):
        with self.lock:  # 全アカウントがブロックされる
            time.sleep(0.01)  # I/Oをシミュレート
            self.accounts[from_acc] -= amount
            self.accounts[to_acc] += amount


# === 改善: 細粒度ロック（アカウントごとにロック） ===

class BankAccountFineLock:
    """アカウントごとに個別のロックを持つ（並行性が高い）。

    改善のポイント:
      accounts['A'] と accounts['C'] への操作は
      同時に進行できるようになる。
      ただし、デッドロック防止のためロック順序の統一が必要。
    """
    def __init__(self):
        self.accounts = {'A': 1000, 'B': 1000, 'C': 1000}
        self.locks = {k: threading.Lock() for k in self.accounts}

    def transfer(self, from_acc, to_acc, amount):
        # アカウント名の辞書順でロックを取得（デッドロック防止）
        first, second = sorted([from_acc, to_acc])
        with self.locks[first]:
            with self.locks[second]:
                time.sleep(0.01)
                self.accounts[from_acc] -= amount
                self.accounts[to_acc] += amount


def benchmark(bank, label):
    start = time.time()
    threads = []
    # A→B と C→A の送金を並行に実行
    for _ in range(10):
        t1 = threading.Thread(target=bank.transfer, args=('A', 'B', 10))
        t2 = threading.Thread(target=bank.transfer, args=('C', 'A', 10))
        threads.extend([t1, t2])

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start
    total = sum(bank.accounts.values())
    print(f"  {label}: {elapsed:.3f}秒, 合計残高={total} (常に3000)")


if __name__ == "__main__":
    print("=== ロック粒度の比較 ===")
    benchmark(BankAccountGiantLock(), "Giant Lock  ")
    benchmark(BankAccountFineLock(),  "Fine Lock   ")
    print("\n  Fine Lock の方が高速になる（並行実行が可能なため）")
```

### 7.2 アンチパターン2: 条件変数での if による条件チェック（Spurious Wakeup）

```python
"""antipattern_spurious_wakeup.py - 条件変数の誤った使い方

問題:
  条件変数の wait() から戻った時、条件が成立しているとは限らない。
  OS の実装上、signal/broadcast なしに起床する「偽の起床」が発生しうる。
  if で条件をチェックすると、偽の起床時に不正な状態で処理が進む。
"""
import threading
from collections import deque

buffer = deque()
lock = threading.Lock()
cond = threading.Condition(lock)

# === アンチパターン: if で条件チェック ===

def consumer_BAD():
    """偽の起床を考慮していない危険なコード。

    なぜ危険か:
      wait() から戻った直後に buffer が空の可能性がある。
      1. 偽の起床（Spurious Wakeup）: OSが予期せずスレッドを起床させる
      2. broadcast 後の競合: 他のconsumerが先にアイテムを取ってしまう

      結果: buffer が空の状態で popleft() を呼び、IndexError が発生する。
    """
    with cond:
        if len(buffer) == 0:  # if は1回しかチェックしない
            cond.wait()
        # ここに到達しても buffer が空の可能性がある!
        item = buffer.popleft()  # IndexError の可能性


# === 正しいパターン: while で条件チェック ===

def consumer_GOOD():
    """while ループで条件を再確認する安全なコード。

    なぜ while が正しいか:
      wait() から戻るたびに条件を再チェックする。
      偽の起床や broadcast 後の競合があっても、
      条件が成立するまでループが待機を続ける。
    """
    with cond:
        while len(buffer) == 0:  # while で毎回チェック
            cond.wait()
        # ここに到達した時点で必ず buffer にアイテムがある
        item = buffer.popleft()  # 安全
        return item
```

**教訓**: 条件変数と `wait()` を使う場合、条件チェックは**必ず `while` ループ**で行うこと。これは POSIX 仕様でも明記されている推奨事項である。

---

## 8. エッジケース分析

### 8.1 エッジケース1: スレッドセーフな遅延初期化（Double-Checked Locking）

遅延初期化（Lazy Initialization）は、リソースを最初に使用する時点で初期化するパターンである。マルチスレッド環境では、2つのスレッドが同時に初期化を試みる競合状態が発生しうる。

```java
/* DoubleCheckedLocking.java - スレッドセーフな遅延初期化
 *
 * コンパイル・実行: javac DoubleCheckedLocking.java && java DoubleCheckedLocking
 */

public class DoubleCheckedLocking {

    /*
     * なぜ volatile が必要か:
     *
     * Java のメモリモデルでは、コンパイラやCPUが命令の順序を
     * 最適化（リオーダー）する場合がある。
     * volatile なしの場合、以下のような問題が発生しうる:
     *
     *   1. メモリ確保
     *   2. instance にアドレスを代入  ← リオーダーでここが先になる
     *   3. コンストラクタ実行        ← まだ初期化されていない
     *
     *   別のスレッドがステップ2の後、ステップ3の前に instance をチェックすると、
     *   null でないが未初期化のオブジェクトを受け取る。
     *
     * volatile は happens-before 関係を保証し、
     * リオーダーを防止する。
     */
    private static volatile DoubleCheckedLocking instance;

    private final String data;

    private DoubleCheckedLocking() {
        // 重い初期化処理をシミュレート
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        this.data = "初期化完了 @ " + System.currentTimeMillis();
    }

    public static DoubleCheckedLocking getInstance() {
        /*
         * Double-Checked Locking パターン:
         *
         * 1回目のチェック（ロックなし）: 高速パス
         *   既に初期化済みの場合、ロック取得のオーバーヘッドを回避
         *
         * 2回目のチェック（ロック内）: 安全なチェック
         *   1回目のチェックと lock の間に他スレッドが初期化した場合を検出
         */
        if (instance == null) {              // 1回目のチェック（高速パス）
            synchronized (DoubleCheckedLocking.class) {
                if (instance == null) {      // 2回目のチェック（安全な確認）
                    instance = new DoubleCheckedLocking();
                }
            }
        }
        return instance;
    }

    public String getData() {
        return data;
    }

    public static void main(String[] args) throws InterruptedException {
        Thread[] threads = new Thread[10];
        for (int i = 0; i < threads.length; i++) {
            final int id = i;
            threads[i] = new Thread(() -> {
                DoubleCheckedLocking obj = DoubleCheckedLocking.getInstance();
                System.out.printf("Thread %d: %s (hash=%d)%n",
                    id, obj.getData(), System.identityHashCode(obj));
            });
        }

        for (Thread t : threads) t.start();
        for (Thread t : threads) t.join();

        System.out.println("\n全スレッドが同じインスタンスを取得していることを確認");
        System.out.println("(全ての hash 値が同一であるべき)");
    }
}
```

### 8.2 エッジケース2: fork() と スレッドの危険な組み合わせ

マルチスレッドプログラムで `fork()` を呼ぶと、**呼び出したスレッドのみ**が子プロセスにコピーされる。他のスレッドは消滅する。この時、消滅したスレッドがMutexを保持していた場合、子プロセス内でそのMutexは永久にロックされたままとなる。

```
fork() とスレッドの問題:

  親プロセス:
  ┌────────────────────────────────┐
  │  Thread A      Thread B       │
  │  (fork呼出)    (Mutex X保持中) │
  │                               │
  │  fork() ─────────┐            │
  └────────────────────────────────┘
                     │
                     ↓
  子プロセス:
  ┌────────────────────────────────┐
  │  Thread A'     (Thread Bなし)  │
  │                               │
  │  Mutex X: ロック状態           │
  │  (解放するスレッドが存在しない) │
  │  → lock(X) を呼ぶと永久待機!  │
  └────────────────────────────────┘

  対策:
  1. fork() の後すぐに exec() を呼ぶ（アドレス空間が置き換わる）
  2. pthread_atfork() でフォーク前後のロック管理ハンドラを登録する
  3. マルチスレッドプログラムでは fork() を避け、posix_spawn() を使う
```

**なぜ `posix_spawn()` が推奨されるか**: `posix_spawn()` は fork + exec を単一の操作として実行するため、中間状態（fork直後、exec前）でのロック不整合問題を回避できる。多くの現代のOSでは `posix_spawn()` を vfork + exec として実装しており、パフォーマンスも優れている。

### 8.3 エッジケース3: ABA問題（ロックフリーアルゴリズム）

CAS（Compare-And-Swap）ベースのロックフリーアルゴリズムでは、値が A→B→A と変化した場合、CAS はこの変化を検出できない。

```
ABA問題の例（ロックフリースタック）:

  初期状態: top → [A] → [B] → [C]

  Thread 1: CAS(top, A, B) を実行しようとする
            （A を pop して B を top にする）
            スケジューラにより中断

  Thread 2: A を pop  → top → [B] → [C]
            B を pop  → top → [C]
            A を push → top → [A] → [C]  (Bは消えた!)

  Thread 1: CAS(top, A, ?) を実行
            top が A であることを確認 → CAS成功
            しかし、A の next は B を指している（古い情報）
            → top → [B] → ???  (Bは既に free されている可能性)
            → メモリ破壊!

  対策:
  - タグ付きポインタ: ポインタにバージョン番号を付加
    CAS は (ポインタ, バージョン) のペアで比較するため、ABA を検出可能
  - ハザードポインタ: 使用中のノードを登録し、free を遅延させる
  - エポックベースリクレイメーション: epoch を進めてから古いデータを回収
```

---

## 9. スレッドローカルストレージ（TLS）

スレッドローカルストレージは、各スレッドが**自分だけの**変数コピーを持つ仕組みである。グローバル変数のように見えるが、実際にはスレッドごとに独立した値を保持する。

**なぜ TLS が必要か**: errno のようなグローバル変数は、マルチスレッド環境では問題になる。あるスレッドのシステムコールが errno を設定した直後に、別のスレッドのシステムコールが errno を上書きしてしまう可能性がある。TLS を使えば、各スレッドが独自の errno を持つため、この問題を回避できる。

```c
/* thread_local_demo.c - スレッドローカルストレージの使用例
 *
 * コンパイル: gcc -Wall -pthread -o tls_demo thread_local_demo.c
 * 実行:       ./tls_demo
 */
#include <stdio.h>
#include <pthread.h>

/*
 * _Thread_local (C11) / __thread (GCC拡張) でスレッドローカル変数を宣言。
 * 各スレッドがこの変数の独立したコピーを持つ。
 *
 * 内部実装:
 *   ELF バイナリの .tdata / .tbss セクションに格納される。
 *   各スレッドの TCB に TLS ブロックへのポインタがあり、
 *   fs/gs セグメントレジスタを介してアクセスする。
 */
static _Thread_local int tls_counter = 0;
static int shared_counter = 0;  /* 比較用: 共有変数 */

void* worker(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 1000; i++) {
        tls_counter++;      /* スレッドローカル: 競合なし */
        shared_counter++;   /* 共有変数: 競合状態が発生 */
    }

    printf("Thread %d: tls_counter=%d (常に1000), "
           "shared_counter=%d (競合により不定)\n",
           id, tls_counter, shared_counter);
    return NULL;
}

int main(void) {
    const int N = 4;
    pthread_t threads[N];
    int ids[N];

    for (int i = 0; i < N; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, worker, &ids[i]);
    }
    for (int i = 0; i < N; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("\n最終 shared_counter = %d (期待値4000、競合により減少する可能性)\n",
           shared_counter);
    return 0;
}
```

---

## 10. メモリモデルとメモリバリア

マルチスレッドプログラミングでは、CPUやコンパイラによる**命令のリオーダー**がバグの原因になることがある。

```
CPUのメモリアクセスの最適化とリオーダー:

  プログラム上の順序:          実際のメモリアクセス順序:
  ┌──────────────────────┐    ┌──────────────────────┐
  │ 1. x = 42            │    │ 2. ready = true ← 先! │
  │ 2. ready = true      │    │ 1. x = 42     ← 後!  │
  └──────────────────────┘    └──────────────────────┘

  なぜリオーダーが起きるか:
  - CPU: Store Buffer、キャッシュ階層、Out-of-Order実行
  - コンパイラ: 最適化による命令の並べ替え

  問題になる場合:
    Thread A:                 Thread B:
    x = 42;                   while (!ready) { }
    ready = true;             print(x);  // 42のはず...0かも!

    リオーダーにより ready = true が x = 42 より先に
    メモリに書き込まれると、Thread B は x=0 を読んでしまう。

  メモリバリア（フェンス）:
    x = 42;
    __sync_synchronize();  // フルフェンス: これより前の書き込みが
    ready = true;          // これより後の書き込みより先にメモリに反映

  各言語の対応:
  - C11:   atomic_thread_fence(), _Atomic 型
  - C++11: std::atomic, std::memory_order
  - Java:  volatile, synchronized, java.util.concurrent.atomic
  - Go:    sync/atomic パッケージ, sync.Mutex
```

---

## 11. 実践演習

### 演習1: [基礎] スレッドの生成と競合状態の観測

**目的**: 競合状態が実際に発生することを観測し、Mutex による保護の効果を確認する。

```python
"""exercise_01_race_condition.py - 競合状態の観測

課題:
  1. 以下のコードを lock なしで実行し、counter の値が期待値より小さくなることを確認せよ
  2. lock を有効にして実行し、counter が常に期待値と一致することを確認せよ
  3. ITERATIONS の値を変えて、競合の発生頻度がどう変わるか観察せよ
     (小さい値では競合が観測しにくく、大きい値では顕著になる理由を考えよ)

ヒント:
  小さい ITERATIONS で競合が観測しにくいのは、
  スレッドの実行時間が短すぎて、オーバーラップする確率が低いためである。

実行: python3 exercise_01_race_condition.py
"""
import threading
import time

ITERATIONS = 1_000_000
NUM_THREADS = 4

counter = 0
lock = threading.Lock()
USE_LOCK = False  # True に変更して比較せよ

def worker():
    global counter
    for _ in range(ITERATIONS):
        if USE_LOCK:
            with lock:
                counter += 1
        else:
            counter += 1

def main():
    global counter
    expected = ITERATIONS * NUM_THREADS

    # 10回繰り返して統計を取る
    results = []
    for trial in range(10):
        counter = 0
        threads = [threading.Thread(target=worker) for _ in range(NUM_THREADS)]
        start = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - start
        results.append(counter)
        print(f"  Trial {trial+1:2d}: counter={counter:>10,} "
              f"(期待値: {expected:>10,}) "
              f"差: {expected - counter:>8,}  "
              f"時間: {elapsed:.3f}秒")

    print(f"\n  最小値: {min(results):,}")
    print(f"  最大値: {max(results):,}")
    print(f"  期待値: {expected:,}")
    print(f"  USE_LOCK: {USE_LOCK}")

if __name__ == "__main__":
    main()
```

### 演習2: [応用] デッドロックの再現と3つの回避手法

**目的**: デッドロックを実際に発生させ、3つの異なる回避手法を実装する。

```python
"""exercise_02_deadlock.py - デッドロック回避の3手法

課題:
  1. deadlock_scenario() を実行し、プログラムがハングすることを確認せよ
     (Ctrl+C で中断)
  2. 以下の3つの回避手法をそれぞれ実装せよ:
     a) ロック順序の統一（lock_ordering）
     b) タイムアウト付きロック（timeout_approach）
     c) contextmanager による一括ロック取得（batch_locking）
  3. 各手法の長所と短所を整理せよ

実行: python3 exercise_02_deadlock.py
"""
import threading
import time
import contextlib

lock_x = threading.Lock()
lock_y = threading.Lock()

# === デッドロックが発生するコード ===

def deadlock_worker_1():
    lock_x.acquire()
    print("Worker 1: lock_x 取得")
    time.sleep(0.1)  # 他スレッドが lock_y を取る隙を作る
    print("Worker 1: lock_y を待機...")
    lock_y.acquire()  # デッドロック!
    print("Worker 1: 両方取得")
    lock_y.release()
    lock_x.release()

def deadlock_worker_2():
    lock_y.acquire()
    print("Worker 2: lock_y 取得")
    time.sleep(0.1)
    print("Worker 2: lock_x を待機...")
    lock_x.acquire()  # デッドロック!
    print("Worker 2: 両方取得")
    lock_x.release()
    lock_y.release()

def deadlock_scenario():
    """このまま実行するとデッドロックでハングする"""
    t1 = threading.Thread(target=deadlock_worker_1)
    t2 = threading.Thread(target=deadlock_worker_2)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)
    if t1.is_alive() or t2.is_alive():
        print(">>> デッドロック検出! (3秒でタイムアウト)")

# === 回避手法 a) ロック順序の統一 ===
# TODO: id() を使ってロックのアドレスで順序を決定し、
#       常に小さい方を先にロックする実装を書け

# === 回避手法 b) タイムアウト付きロック ===
# TODO: lock.acquire(timeout=...) を使い、
#       タイムアウト時に全ロックを解放してリトライする実装を書け

# === 回避手法 c) 一括ロック取得 ===
# TODO: 必要な全ロックをソートして一括取得する
#       contextmanager を実装せよ

if __name__ == "__main__":
    print("=== デッドロック再現 ===")
    deadlock_scenario()
```

### 演習3: [発展] スレッドプールの拡張実装

**目的**: セクション5.3で示したスレッドプールを拡張し、実用的な機能を追加する。

```python
"""exercise_03_advanced_pool.py - スレッドプールの拡張

課題:
  以下の機能を ThreadPool クラスに追加せよ:

  1. submit_with_future(): タスクの結果を Future オブジェクトで返す
     - result = pool.submit_with_future(func, args)
     - value = result.get(timeout=5.0)  # ブロッキングで結果を取得

  2. map(): 複数の引数に同じ関数を並列適用
     - results = pool.map(func, [arg1, arg2, arg3])
     - list(results) で全結果を取得

  3. 統計情報の収集:
     - pool.stats() で以下を返す:
       - 完了タスク数
       - キュー内の待機タスク数
       - アクティブワーカー数
       - 平均タスク実行時間

  ヒント:
  - Future は内部で threading.Event を使い、
    set() で結果の到着を通知、wait() で結果を待機する
  - map() は submit_with_future() を内部で使い、
    全ての Future を収集して結果を返すジェネレータとして実装できる

実行: python3 exercise_03_advanced_pool.py
"""
import threading
import time
from collections import deque
from typing import Callable, Any, Iterator

class Future:
    """タスクの非同期結果を表すオブジェクト。

    TODO: 以下を実装せよ
    - __init__: Event と result/exception の初期化
    - set_result(value): 結果をセットして Event を通知
    - set_exception(exc): 例外をセットして Event を通知
    - get(timeout=None): 結果が到着するまで待機し、返す
    """
    pass

class AdvancedThreadPool:
    """拡張版スレッドプール。

    TODO: section 5.3 の ThreadPool をベースに、
    上記3つの機能を追加せよ
    """
    pass

if __name__ == "__main__":
    print("=== 演習3: 拡張スレッドプール ===")
    print("上記の TODO を実装して動作を確認せよ")
    print()
    print("テスト項目:")
    print("  1. submit_with_future で計算結果を取得")
    print("  2. map で複数引数を並列処理")
    print("  3. stats で統計情報を確認")
    print("  4. グレースフルシャットダウンの動作確認")
```

---

## 12. FAQ

### Q1: GIL（Global Interpreter Lock）とは何か？なぜ存在するのか？

CPython（Python の標準実装）には、同時に1つのスレッドしか Python バイトコードを実行できない **GIL** という制約がある。

**なぜ GIL が存在するか**: CPython の参照カウント方式のメモリ管理は、オブジェクトの参照カウンタをスレッドセーフに更新する必要がある。全てのオブジェクトに個別のロックを持たせると、ロック管理のオーバーヘッドが非常に大きくなり、シングルスレッド性能が大幅に低下する。GIL はこの問題を「1つの大きなロック」で解決する妥協策である。

**GIL の影響**:
- **CPU密度の高い処理**: マルチスレッドにしても並列実行されない。`multiprocessing` モジュールを使うべき。
- **I/O密度の高い処理**: I/O 待ちの間は GIL が解放されるため、スレッドでも効果がある。
- **C拡張モジュール**: NumPy 等の C 拡張は GIL を解放して計算するため、マルチスレッドの恩恵を受けられる。

**Python 3.13+ (PEP 703)**: 実験的に GIL を無効化できる `--disable-gil` オプションが導入された。将来的には GIL のないCPython が標準になる可能性がある。

### Q2: スレッドとコルーチンはどう使い分けるべきか？

| 観点 | スレッド | コルーチン (async/await) |
|------|---------|----------------------|
| スケジューリング | OS がプリエンプティブに切替 | プログラマが await で明示的に切替 |
| メモリ消費 | ~1MB/スレッド（スタック） | ~1KB/コルーチン |
| 生成コスト | 数十us（syscall + スタック確保） | 数百ns（オブジェクト生成のみ） |
| 並列実行 | 可能（マルチコア活用） | 不可能（シングルスレッド） |
| 競合状態 | 発生しうる（共有メモリ） | 原理的に発生しない（シングルスレッド） |
| 向いている処理 | CPU密度が高い計算 | I/O密度が高い処理（ネットワーク、DB） |
| デバッグ | 困難（非決定的） | 比較的容易（決定的） |

**使い分けの指針**:
- 数千〜数万の同時接続を捌く → コルーチン（async/await）
- CPU を使い切る計算 → スレッド（またはマルチプロセス）
- 両方必要な場合 → コルーチン + スレッドプールの組み合わせ

### Q3: マルチスレッドとマルチプロセスの使い分けは？

- **マルチスレッド**: メモリ共有が必要、I/O 密度が高い、軽量な並行性が欲しい場合
- **マルチプロセス**: CPU 密度が高い計算、隔離が必要（セキュリティ、安定性）、GIL の制約を回避したい場合（Python）、クラッシュ耐性が必要な場合

### Q4: volatile は同期に使えるか？

**C/C++ の volatile**: コンパイラの最適化（変数のレジスタキャッシュ等）を抑制するだけであり、**スレッド間同期の機能はない**。メモリバリアを発行しないため、volatile だけではマルチスレッドの安全性は保証されない。ハードウェアレジスタへのアクセスやシグナルハンドラ用途に限定して使うべきである。

**Java の volatile**: メモリバリアを含み、happens-before 関係を保証する。単純なフラグの読み書きにはスレッドセーフだが、read-modify-write（i++ 等）はアトミックではない。

### Q5: スレッドの数はいくつが最適か？

タスクの性質に依存する（セクション5.2を参照）。一般的な指針:

- **CPU-bound**: コア数と同数
- **I/O-bound**: コア数 * (1 + 待ち時間/計算時間)
- **混在**: プロファイリングで最適値を探索する

**注意**: スレッド数を増やしすぎると、コンテキストスイッチのオーバーヘッド、メモリ消費、キャッシュ汚染が増大し、逆にスループットが低下する。想定される最適値は、ベンチマークにより検証することが重要である。

---

## 13. よくあるバグとデバッグ手法

### 13.1 データレース検出ツール

| ツール | 言語/環境 | 検出方法 | 使い方 |
|-------|----------|---------|--------|
| ThreadSanitizer (TSan) | C/C++ | コンパイル時計装 | `gcc -fsanitize=thread` |
| Helgrind | C/C++ (Valgrind) | 動的解析 | `valgrind --tool=helgrind ./prog` |
| Go Race Detector | Go | コンパイル時計装 | `go run -race main.go` |
| Java Flight Recorder | Java | サンプリング | JVM オプションで有効化 |

### 13.2 デバッグのベストプラクティス

1. **再現性の確保**: スレッドの実行順序を制御する `sleep()` やバリアを挿入して、特定のインターリーブを強制する
2. **ログの活用**: タイムスタンプ + スレッドID + 操作内容をログに記録する。ただし、ログ自体がスレッドセーフでなければ意味がない
3. **最小再現ケースの作成**: 問題を再現する最小のコードを作成し、ノイズを排除する
4. **不変条件の明示**: 「この変数は lock X を保持中のみ書き換え可能」といった不変条件をコメントで明記し、assertion で検証する

---

## 14. まとめ

| 概念 | ポイント |
|------|---------|
| スレッド | プロセス内の軽量実行単位。メモリ共有による高速通信と、競合状態のリスクが表裏一体 |
| スレッドモデル | 1:1（Linux NPTL）が主流。M:N（Go goroutine）は大量の軽量スレッドに適する |
| 競合状態 | 非アトミックな共有データアクセスで発生。Mutex/セマフォ/条件変数で保護 |
| 同期プリミティブ | Mutex（排他）、セマフォ（N個同時）、条件変数（条件待ち）、RWLock（読み書き分離） |
| デッドロック | Coffman 4条件が全て成立すると発生。ロック順序の統一が最も実践的な回避策 |
| スレッドプール | スレッドの再利用でオーバーヘッドを削減。サイズはワークロード特性に合わせて調整 |
| メモリモデル | CPU/コンパイラのリオーダーに注意。volatile（Java）やアトミック操作で制御 |
| 現代の並行性 | async/await、アクター、CSP が高レベルの抽象化を提供 |

---

## 15. 次に読むべきガイド

- [[02-scheduling.md]] -- CPU スケジューリング（スレッドがどのようにCPU時間を割り当てられるか）
- [[03-ipc.md]] -- プロセス間通信（スレッド間通信との比較）
- [[04-synchronization.md]] -- 同期の詳細（バリア、リーダライタ問題、哲学者の食事問題）

---

## 16. 参考文献

1. Silberschatz, A., Galvin, P. B., & Gagne, G. (2018). *Operating System Concepts* (10th ed.). Wiley. Chapter 4: Threads & Concurrency. -- スレッドの基本概念とモデルの標準的な教科書。
2. Herlihy, M. & Shavit, N. (2020). *The Art of Multiprocessor Programming* (2nd ed.). Morgan Kaufmann. -- 同期プリミティブ、ロックフリーアルゴリズム、メモリモデルの理論と実装を網羅。
3. Butenhof, D. R. (1997). *Programming with POSIX Threads*. Addison-Wesley. -- pthreads API の決定版リファレンス。条件変数の使い方やキャンセルの詳細が秀逸。
4. Tanenbaum, A. S. & Bos, H. (2014). *Modern Operating Systems* (4th ed.). Pearson. Chapter 2.2: Threads. -- ユーザレベル/カーネルレベルスレッドの実装の違いを詳細に解説。
5. Drepper, U. (2003). "The Native POSIX Thread Library for Linux." Red Hat Technical Report. -- Linux NPTL の設計と、1:1 モデルを選択した根拠を記した歴史的文書。
6. Pike, R. (2012). "Go Concurrency Patterns." Google I/O Talk. -- Go の goroutine と channel を使った CSP パターンの実践的な解説。
7. Lea, D. (2005). *Concurrent Programming in Java: Design Principles and Patterns* (3rd ed.). Addison-Wesley. -- Java の java.util.concurrent パッケージの設計者による解説。
