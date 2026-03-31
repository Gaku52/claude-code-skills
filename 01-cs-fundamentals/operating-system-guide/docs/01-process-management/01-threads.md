# Threads and Concurrency

> A thread is a "lightweight process" -- a unit of execution that runs concurrently within the same process while sharing memory. Virtually all modern applications -- web servers, databases, GUIs, game engines -- are built with multithreading. This chapter systematically covers thread fundamentals, user-level/kernel-level implementation models, synchronization primitives, deadlocks, and thread pools.

## What You Will Learn in This Chapter

- [ ] Explain the structural differences between threads and processes with diagrams
- [ ] Understand the operating principles and trade-offs of user threads and kernel threads
- [ ] Implement race condition detection, prevention, and resolution mechanisms
- [ ] Distinguish between and properly use Mutexes, semaphores, condition variables, and RWLocks
- [ ] Understand the Coffman conditions for deadlock and implement avoidance strategies
- [ ] Explain thread pool design principles and sizing considerations
- [ ] Understand the relationship with modern concurrency models (async/await, actor model, CSP)


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [Process Concepts and Management](./00-processes.md)

---

## 1. Thread Fundamentals

### 1.1 Structural Comparison of Processes and Threads

The most fundamental difference between processes and threads lies in the **scope of memory space sharing**. A process has an independent address space provided by the OS and cannot directly access the memory of other processes. In contrast, threads within the same process share **the code section, data section, heap, and file descriptors** with sibling threads.

```
Process memory layout (single-threaded vs multithreaded):

  Single-threaded process:          Multithreaded process:
  +------------------------+       +----------------------------------+
  | Kernel space           |       | Kernel space                     |
  | (page table, etc.)    |       | (manages TCB per thread)         |
  +------------------------+       +----------------------------------+
  | Stack (1)              |       | Stack Th3 <- per-thread private  |
  |                        |       +----------------------------------+
  |                        |       | Stack Th2 <- per-thread private  |
  |                        |       +----------------------------------+
  |                        |       | Stack Th1 <- per-thread private  |
  +------------------------+       +----------------------------------+
  |          |             |       |          |                       |
  |  (free space)          |       |  (free space)                    |
  |          ^             |       |          ^                       |
  +------------------------+       +----------------------------------+
  | Heap (malloc/new)      |       | Heap  <- shared by all threads   |
  +------------------------+       +----------------------------------+
  | BSS (uninit. globals)  |       | BSS   <- shared by all threads   |
  +------------------------+       +----------------------------------+
  | Data (init. globals)   |       | Data  <- shared by all threads   |
  +------------------------+       +----------------------------------+
  | Text (machine code)    |       | Text  <- shared by all threads   |
  +------------------------+       +----------------------------------+
```

**Why does each thread have its own stack?** Each thread requires an independent execution context (local variables, call history, return addresses). If the stack were shared, one thread's function call would overwrite another thread's local variables.

### 1.2 What Threads Share vs. What Is Private

| Resource | Shared/Private | Reason |
|---------|---------------|--------|
| Code section | Shared | Executing the same program |
| Data section (global variables) | Shared | Holds process-wide state |
| Heap | Shared | Dynamic memory allocation must be visible across threads |
| Open files | Shared | The file descriptor table is per-process |
| Signal handlers | Shared | Signal handling methods are defined per-process |
| Stack | **Private** | Holds each thread's execution context |
| Program counter (PC) | **Private** | Each thread is executing different instructions |
| Register set | **Private** | CPU arithmetic state differs per thread |
| Thread ID (TID) | **Private** | Required for the OS to identify threads |
| Signal mask | **Private** | Controls which signals are blocked per thread |
| errno | **Private** | Prevents system call errors from affecting other threads |

### 1.3 Advantages and Costs of Threads

Understanding from the OS's internal operations why threads are overwhelmingly more lightweight than process fork().

**Difference in creation cost**: `fork()` performs a copy of the process's address space (even with Copy-on-Write, page table and VMA duplication is needed), duplication of the file descriptor table, duplication of signal settings, etc. In contrast, `pthread_create()` only needs to allocate a new stack region and create a TCB (Thread Control Block). In the Linux kernel's `clone()` system call, flags specify which resources to share; for thread creation, `CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND` etc. are specified to skip copying of the memory space and file table.

**Difference in context switch cost**: When switching between threads within the same process, since the address space is the same, TLB (Translation Lookaside Buffer) flushing is unnecessary. Since TLB flushing directly impacts memory access latency, this difference is significant.

**Difference in communication cost**: Inter-process communication (IPC) requires OS mechanisms such as pipes, sockets, or shared memory, incurring data copying and system call overhead. Inter-thread communication can be done through direct access to shared memory, completing with just pointer passing.

| Aspect | Process | Thread |
|--------|---------|--------|
| Creation cost | High (address space copy) | Low (stack + TCB only) |
| Context switch | Heavy (TLB flush required) | Light (no TLB flush) |
| Memory usage | High (independent address space) | Low (only stack added) |
| Communication | IPC required (copying involved) | Direct shared memory access |
| Fault isolation | High (protected by independent space) | Low (one thread's crash can collapse all) |
| Debugging ease | Relatively easy | Difficult (non-deterministic execution order) |
| Security | High (memory protection) | Low (all memory accessible) |

### 1.4 Basic Operations with POSIX Threads (pthreads)

The following is a complete example of thread creation, execution, and joining using pthreads.

```c
/* thread_basics.c - A complete example demonstrating basic thread operations
 *
 * Compile: gcc -Wall -pthread -o thread_basics thread_basics.c
 * Run:     ./thread_basics
 *
 * Why the -pthread flag is needed:
 *   It instructs the linker to link libpthread.
 *   pthread functions (pthread_create, etc.) are in this library.
 *   -lpthread also works, but -pthread is recommended as it also sets compiler flags.
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

/* Argument structure passed to threads */
typedef struct {
    int thread_id;
    int iterations;
} thread_arg_t;

/*
 * Entry point function for a thread.
 * Argument: received as void* and cast internally (POSIX API convention).
 * Return value: returned as void*, receivable via pthread_join().
 *
 * Why void*?
 *   A C-language design that sacrifices type safety in exchange for
 *   the versatility of being able to pass data of any type.
 */
void* worker(void* arg) {
    thread_arg_t* targ = (thread_arg_t*)arg;
    printf("Thread %d: Started (iterations=%d)\n", targ->thread_id, targ->iterations);

    long sum = 0;
    for (int i = 0; i < targ->iterations; i++) {
        sum += i;
    }

    printf("Thread %d: Completed (sum=%ld)\n", targ->thread_id, sum);

    /* Allocate the return value on the heap.
     * Do not return a local variable on the stack,
     * because the stack is freed after the thread exits. */
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

    printf("Main thread: Creating %d worker threads\n", NUM_THREADS);

    /* Thread creation */
    for (int i = 0; i < NUM_THREADS; i++) {
        args[i].thread_id = i;
        args[i].iterations = (i + 1) * 1000000;

        /*
         * pthread_create arguments:
         *   1. &threads[i]  - Variable to store thread ID
         *   2. NULL         - Thread attributes (NULL for defaults)
         *   3. worker       - Function for the thread to execute
         *   4. &args[i]     - Argument to pass to the function
         *
         * Return value: 0 on success, non-zero error code on failure
         */
        int ret = pthread_create(&threads[i], NULL, worker, &args[i]);
        if (ret != 0) {
            fprintf(stderr, "pthread_create failed: %d\n", ret);
            exit(EXIT_FAILURE);
        }
    }

    /* Wait for all threads to complete (join) */
    for (int i = 0; i < NUM_THREADS; i++) {
        void* retval;
        /*
         * pthread_join waits for thread termination and receives the return value.
         * Why join is necessary:
         *   1. Guarantees thread completion (before using results)
         *   2. Frees thread resources (TCB, etc.)
         *   Threads not joined become zombie-like, causing resource leaks.
         */
        int ret = pthread_join(threads[i], &retval);
        if (ret != 0) {
            fprintf(stderr, "pthread_join failed: %d\n", ret);
            exit(EXIT_FAILURE);
        }
        if (retval != NULL) {
            printf("Main thread: Thread %d result = %ld\n", i, *(long*)retval);
            free(retval);  /* Free the memory malloc'd in worker() */
        }
    }

    printf("Main thread: All threads complete\n");
    return 0;
}
```

**Expected execution result** (thread execution order is non-deterministic, so output order may vary between runs):

```
Main thread: Creating 4 worker threads
Thread 0: Started (iterations=1000000)
Thread 2: Started (iterations=3000000)
Thread 1: Started (iterations=2000000)
Thread 3: Started (iterations=4000000)
Thread 0: Completed (sum=499999500000)
Thread 1: Completed (sum=1999999000000)
Thread 2: Completed (sum=4499998500000)
Thread 3: Completed (sum=7999998000000)
Main thread: Thread 0 result = 499999500000
Main thread: Thread 1 result = 1999999000000
Main thread: Thread 2 result = 4499998500000
Main thread: Thread 3 result = 7999998000000
Main thread: All threads complete
```

---

## 2. Thread Models -- User-Level/Kernel-Level Implementations

### 2.1 Thread Implementation Layers

There are three major models for thread implementation. The performance characteristics, ability to use multiple cores, and blocking behavior vary significantly depending on which model is adopted.

```
Comparison of the 3 thread implementation models:

+---------------------------------------------------------------------+
| (1) N:1 Model (User-Level Threads)                                   |
|                                                                     |
|  User space  +--------------------------------------------+         |
|              |  Thread library (runtime)                   |         |
|              |  +--+ +--+ +--+ +--+                       |         |
|              |  |T1| |T2| |T3| |T4|  <- User threads      |         |
|              |  +--+ +--+ +--+ +--+                       |         |
|              |         | Scheduler                         |         |
|              |         v                                   |         |
|  ----------- | --------------------------------- -- -- --  |         |
|  Kernel space|     +------+                                |         |
|              |     | KT 1 | <- Kernel thread (only 1)      |         |
|              |     +------+                                |         |
|              +--------------------------------------------+         |
|  Features: The kernel is unaware of the existence of threads        |
|  Pros: Ultra-fast context switch (no syscall, tens of ns)          |
|  Cons: If one blocks -> all threads block                           |
|        Cannot utilize multiple cores (only one kernel thread)       |
|  Examples: GNU Portable Threads, early Java Green Threads           |
+---------------------------------------------------------------------+
| (2) 1:1 Model (Kernel-Level Threads)                                |
|                                                                     |
|  User space   +--+ +--+ +--+ +--+  <- User threads                 |
|               |T1| |T2| |T3| |T4|                                  |
|               +--+ +--+ +--+ +--+                                  |
|                |    |    |    |    <- 1-to-1 mapping                 |
|                v    v    v    v                                     |
|  ------------------------------------------                         |
|  Kernel space +--+ +--+ +--+ +--+                                   |
|               |K1| |K2| |K3| |K4|  <- Kernel threads               |
|               +--+ +--+ +--+ +--+                                  |
|  Features: Each user thread maps 1-to-1 with a kernel thread       |
|  Pros: Can utilize multiple cores, individual blocking possible     |
|  Cons: Requires syscall for creation/switching (hundreds of ns~us)  |
|        Thread count limited (kernel resource constraints)           |
|  Examples: Linux NPTL, Windows Threads, macOS pthreads              |
+---------------------------------------------------------------------+
| (3) M:N Model (Hybrid)                                              |
|                                                                     |
|  User space  +--+ +--+ +--+ +--+ +--+ +--+  <- M user threads      |
|              |T1| |T2| |T3| |T4| |T5| |T6|                         |
|              +--+ +--+ +--+ +--+ +--+ +--+                         |
|               v  /  v    v  /  v                                    |
|  ------------------------------------------                         |
|  Kernel space +--+      +--+       <- N kernel threads              |
|               |K1|      |K2|       (N < M)                          |
|               +--+      +--+                                        |
|  Features: User-space scheduler dynamically manages M->N mapping    |
|  Pros: Can utilize multiple cores, large number of threads at low cost |
|  Cons: Extremely complex implementation (scheduler design is hard)  |
|  Examples: Go goroutine, Erlang process, old Solaris LWP            |
+---------------------------------------------------------------------+
```

### 2.2 Why Linux Chose the 1:1 Model

Linux initially used LinuxThreads, a thread implementation with poor POSIX compliance. Subsequently, two competing implementations were proposed:

1. **NGPT (Next Generation POSIX Threads)**: An M:N model developed by IBM
2. **NPTL (Native POSIX Threads Library)**: A 1:1 model developed by Red Hat

NPTL (the 1:1 model) was ultimately adopted for the following reasons:

- **Linux's `clone()` is fast**: The Linux kernel's thread creation is sufficiently optimized that the M:N model could not achieve performance gains worth the complexity of user-space scheduling
- **Complexity of the M:N model**: Dual management by user-space and kernel schedulers causes priority inversion and signal delivery problems
- **Ease of POSIX compliance**: The 1:1 model makes it easier to accurately implement POSIX semantics

### 2.3 Go's Goroutines -- A Success Story of the M:N Model

Go adopts the M:N model, efficiently multiplexing hundreds of thousands to millions of goroutines on a small number of OS threads. The factors behind its success are:

- **Language-level integration**: Built into the runtime, not bolted on as a library
- **Cooperative preemption**: Since Go 1.14, goroutines support asynchronous signal-based preemption in addition to stack checks at function calls
- **Growable stack**: Initial stack size is 2KB (extremely small compared to the 1-8MB of OS threads), dynamically expanding as needed
- **Netpoller integration**: When blocking on I/O, the goroutine is parked and another goroutine is executed

```
Go runtime's GMP model:

  G (Goroutine): Execution unit. Millions can be created.
  M (Machine):   OS thread. Typically around the number of CPU cores.
  P (Processor): Logical processor. Set by GOMAXPROCS.

  +---------------------------------------------+
  |               Global Run Queue              |
  |  [G5] [G6] [G7] [G8] ...                   |
  +------------------+--------------------------+
                     |
        +------------+------------+
        v            v            v
   +---------+  +---------+  +---------+
   |  P0     |  |  P1     |  |  P2     | <- Logical processors
   | LocalQ: |  | LocalQ: |  | LocalQ: |
   | [G1]   |  | [G3]   |  | [G9]   |
   | [G2]   |  | [G4]   |  |        |
   +----+----+  +----+----+  +----+----+
        |            |            |
        v            v            v
   +---------+  +---------+  +---------+
   |  M0     |  |  M1     |  |  M2     | <- OS threads
   | (CPU 0) |  | (CPU 1) |  | (CPU 2) |
   +---------+  +---------+  +---------+

  Work Stealing:
    When P2's queue becomes empty, it "steals"
    goroutines from P0's or P1's queue -> achieves load balancing
```

### 2.4 Threads in Java (From 1:1 to Virtual Threads)

```java
/* JavaThreadDemo.java - Basic thread operations in Java
 *
 * Compile and run: javac JavaThreadDemo.java && java JavaThreadDemo
 *
 * Java threads were the 1:1 model (directly corresponding to OS threads) for many years.
 * Java 21 (2023) officially introduced Virtual Threads, transitioning to an M:N model.
 */

import java.util.ArrayList;
import java.util.List;

public class JavaThreadDemo {

    /* Shared counter: demonstrates why volatile alone is insufficient */
    private static int counter = 0;
    private static final Object lock = new Object();

    public static void main(String[] args) throws InterruptedException {
        System.out.println("=== Platform Threads (traditional 1:1 threads) ===");
        platformThreadDemo();

        System.out.println("\n=== Virtual Threads (Java 21+ M:N threads) ===");
        virtualThreadDemo();
    }

    static void platformThreadDemo() throws InterruptedException {
        counter = 0;
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < 4; i++) {
            final int id = i;
            /*
             * Thread.ofPlatform() is a new API in Java 21+.
             * Equivalent to the traditional new Thread(), but allows declarative configuration.
             * Internally, the OS's pthread_create() is called.
             */
            Thread t = Thread.ofPlatform()
                .name("worker-" + id)
                .start(() -> {
                    for (int j = 0; j < 100_000; j++) {
                        /*
                         * synchronized block: Java's built-in Mutex.
                         * Also called a monitor lock, managed by the JVM.
                         * Why synchronized is necessary:
                         *   counter++ is a 3-step read-modify-write operation,
                         *   which is not atomic, so a race condition occurs without protection.
                         */
                        synchronized (lock) {
                            counter++;
                        }
                    }
                });
            threads.add(t);
        }

        for (Thread t : threads) {
            t.join(); /* Wait for thread completion */
        }

        System.out.println("Counter = " + counter + " (expected: 400000)");
    }

    static void virtualThreadDemo() throws InterruptedException {
        counter = 0;
        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < 10_000; i++) {
            /*
             * Thread.ofVirtual(): Java 21's Virtual Threads.
             * Not OS threads, but lightweight threads managed by the JVM.
             * Operates on an M:N model similar to Go's goroutines.
             *
             * Why Virtual Threads were introduced:
             *   Since an OS thread consumes ~1MB of stack,
             *   creating tens of thousands of threads was impractical.
             *   Virtual Threads start at ~1KB and expand as needed.
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

        System.out.println("Counter = " + counter + " (expected: 10000)");
        System.out.println("10,000 Virtual Threads created and completed");
    }
}
```

---

## 3. Synchronization Primitives in Detail

### 3.1 Race Condition Occurrence Mechanism

A race condition occurs when multiple threads simultaneously perform **non-atomic** operations on shared data. At the CPU instruction level, even a simple operation like `counter++` is decomposed into multiple instructions (load, add, store), leading to unexpected results depending on interrupt timing.

```
Instruction-level decomposition of counter++ (x86 architecture):

  C:    counter++;
  asm:  mov eax, [counter]   ; (1) Load from memory to register
        add eax, 1           ; (2) Increment in register
        mov [counter], eax   ; (3) Store from register to memory

  Interleaving when two threads execute simultaneously:

  Time   Thread A               Thread B               counter (memory)
  ------+----------------------+----------------------+----------
   t1   | mov eax, [counter]   |                      | 0
        | eax_A = 0            |                      |
  ------+----------------------+----------------------+----------
   t2   |                      | mov eax, [counter]   | 0
        |                      | eax_B = 0            |
  ------+----------------------+----------------------+----------
   t3   | add eax, 1           |                      | 0
        | eax_A = 1            |                      |
  ------+----------------------+----------------------+----------
   t4   |                      | add eax, 1           | 0
        |                      | eax_B = 1            |
  ------+----------------------+----------------------+----------
   t5   | mov [counter], eax   |                      | 1 <- A writes
  ------+----------------------+----------------------+----------
   t6   |                      | mov [counter], eax   | 1 <- B overwrites!
  ------+----------------------+----------------------+----------

  Result: counter = 1 despite two increments (Lost Update)
```

### 3.2 Mutex (Mutual Exclusion Lock)

A Mutex is the most basic synchronization primitive. It can be compared to a "locked room" -- only one thread can enter the critical section at a time.

**Why Mutex is necessary**: To prevent race conditions, the **atomicity** of read-modify-write operations must be guaranteed. A Mutex acquires a lock at the entrance to a critical section and releases it at the exit, limiting the section to execution by a single thread.

```c
/* mutex_demo.c - Preventing race conditions with Mutex
 *
 * Compile: gcc -Wall -pthread -o mutex_demo mutex_demo.c
 * Run:     ./mutex_demo
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define NUM_THREADS  4
#define ITERATIONS   1000000

/* Shared counter and Mutex */
static long counter_unsafe = 0;
static long counter_safe   = 0;
static pthread_mutex_t mtx = PTHREAD_MUTEX_INITIALIZER;
/*
 * PTHREAD_MUTEX_INITIALIZER is a static initialization macro.
 * Equivalent to dynamic initialization with pthread_mutex_init(),
 * but can only be used for global/static variables.
 * Why convenient: No need for init/destroy calls, making code more concise.
 */

void* unsafe_worker(void* arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        counter_unsafe++;  /* No protection: race condition occurs */
    }
    return NULL;
}

void* safe_worker(void* arg) {
    (void)arg;
    for (int i = 0; i < ITERATIONS; i++) {
        /*
         * pthread_mutex_lock: Acquires the lock. Blocks if another thread holds it.
         * Internally uses the futex (Fast Userspace muTEX) syscall:
         *   1. First attempts atomic lock acquisition (fast path)
         *   2. Only calls the kernel for waiting if that fails (slow path)
         * This two-stage design avoids syscalls when there is no contention.
         */
        pthread_mutex_lock(&mtx);
        counter_safe++;
        pthread_mutex_unlock(&mtx);
    }
    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];

    /* Without protection */
    printf("--- Without protection ---\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, unsafe_worker, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Expected: %d, Result: %ld (likely less due to races)\n",
           NUM_THREADS * ITERATIONS, counter_unsafe);

    /* With Mutex protection */
    printf("\n--- With Mutex protection ---\n");
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, safe_worker, NULL);
    }
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }
    printf("Expected: %d, Result: %ld (always matches)\n",
           NUM_THREADS * ITERATIONS, counter_safe);

    return 0;
}
```

### 3.3 Semaphores

A semaphore is a generalization of a Mutex that allows **N threads to enter the critical section simultaneously**. The internal counter value represents N (the number of permits).

- **P operation (wait/down)**: Decrement the counter. Block if it is 0.
- **V operation (signal/up)**: Increment the counter. Wake a waiting thread if any.

| Use Case | Initial Counter Value | Description |
|----------|---------------------|-------------|
| Binary semaphore | 1 | Equivalent to a Mutex (but without the ownership concept) |
| Counting semaphore | N | Limits concurrent access (connection pools, etc.) |
| Signaling | 0 | Used for inter-thread notification (producer -> consumer) |

**Fundamental difference between Mutex and semaphore**: A Mutex has the concept of "ownership" -- only the thread that acquired the lock can unlock it. A semaphore has no such constraint; one thread can wait and a different thread can signal. This property makes semaphores suitable for signaling in producer-consumer patterns.

### 3.4 Condition Variables

A condition variable is a mechanism for "efficiently waiting until a certain condition is met." It is used in combination with a Mutex.

**Why Mutex alone is insufficient**: A Mutex provides mutual exclusion for critical sections, but is not suited for conditional waits like "wait until the buffer is non-empty." Busy waiting (continuously checking the condition in a loop) wastes CPU time. A condition variable puts the thread to sleep when the condition is not met and wakes it when the condition changes, avoiding CPU waste.

```python
"""producer_consumer.py - Producer-consumer pattern using condition variables

This pattern is used in scenarios such as:
  - Web server request queues
  - Log write buffers
  - Video encoding frame buffers

Run: python3 producer_consumer.py
"""
import threading
import time
import random
from collections import deque

BUFFER_SIZE = 5  # Maximum buffer size

buffer = deque()
lock = threading.Lock()
not_empty = threading.Condition(lock)  # Notifies that the buffer is not empty
not_full  = threading.Condition(lock)  # Notifies that the buffer is not full

produced_count = 0
consumed_count = 0

def producer(producer_id: int, num_items: int) -> None:
    """Produce items and add them to the buffer.

    Waits on the not_full condition variable when the buffer is full.
    Why check the condition with a while loop (not if):
      Because of the possibility of Spurious Wakeup.
      Due to OS implementation details, a condition variable may wake
      without signal/broadcast. The while loop ensures the condition
      is re-checked after waking, maintaining safety.
    """
    global produced_count
    for i in range(num_items):
        item = f"P{producer_id}-Item{i}"
        with not_full:
            while len(buffer) >= BUFFER_SIZE:
                print(f"  Producer {producer_id}: Buffer full, waiting...")
                not_full.wait()  # Release lock + wait (executed atomically)

            buffer.append(item)
            produced_count += 1
            print(f"  Producer {producer_id}: Produced '{item}' "
                  f"(buffer size: {len(buffer)})")
            not_empty.notify()  # Wake one consumer

        time.sleep(random.uniform(0.01, 0.05))  # Simulate production time


def consumer(consumer_id: int, num_items: int) -> None:
    """Take items from the buffer and consume them.

    Waits on the not_empty condition variable when the buffer is empty.
    """
    global consumed_count
    for _ in range(num_items):
        with not_empty:
            while len(buffer) == 0:
                print(f"  Consumer {consumer_id}: Buffer empty, waiting...")
                not_empty.wait()

            item = buffer.popleft()
            consumed_count += 1
            print(f"  Consumer {consumer_id}: Consumed '{item}' "
                  f"(buffer size: {len(buffer)})")
            not_full.notify()  # Wake one producer

        time.sleep(random.uniform(0.02, 0.08))  # Simulate consumption time


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

    print("=== Producer-Consumer Pattern Started ===")
    print(f"Buffer size limit: {BUFFER_SIZE}")
    print(f"Producers: 2, Consumers: 2\n")

    for t in producers + consumers:
        t.start()
    for t in producers + consumers:
        t.join()

    print(f"\n=== Complete ===")
    print(f"Produced: {produced_count}, Consumed: {consumed_count}")


if __name__ == "__main__":
    main()
```

### 3.5 Read-Write Lock (RWLock)

A read-write lock enforces the constraint "multiple simultaneous reads are OK, but writes are exclusive." It is effective for workloads that are overwhelmingly read-heavy (e.g., caches, configuration data, DNS tables).

```
RWLock state transition diagram:

         +---------------------------------------------+
         |                                             |
         v                                             |
    +----------+    Acquire read lock       +--------------+
    |          | ----------------------->   |              |
    |  FREE    |                           | READ_LOCKED  |
    | (unlocked)| <-----------------------  | (readers>=1) |
    |          |    Last reader releases    |              |
    +----------+                           +--------------+
         |                                    |
         |  Acquire write lock                 | Acquire write lock
         |  (only if reader count is 0)        | -> Wait until all readers release
         v                                    v
    +--------------+
    |              |  <- Read lock acquisition is blocked
    | WRITE_LOCKED |  <- Other write lock acquisitions are also blocked
    | (writer=1)   |
    |              |
    +--------------+

  Concurrency comparison:
    Mutex:   [R] [R] [W] [R] [R] [W]  <- All serialized
    RWLock:  [R R R] [W] [R R] [W]    <- Reads can be concurrent
```

### 3.6 Synchronization Primitive Comparison Table

| Primitive | Concurrent Access | Ownership | Use Case | Overhead |
|-----------|-----------------|-----------|----------|----------|
| Mutex | 1 | Yes (only locker can unlock) | Critical section mutual exclusion | Low (futex optimized) |
| Spinlock | 1 | Yes | Very short critical sections (kernel internals) | Minimal (busy wait) |
| Semaphore | N (variable) | No (anyone can signal) | Resource pools, signaling | Medium |
| Condition variable | - | No (used with Mutex) | Conditional waits (producer-consumer, etc.) | Medium |
| RWLock | Read: unlimited, Write: 1 | Yes | Read-heavy workloads | Medium-High |
| Barrier | - | No | Synchronization point for all threads (phase boundary) | Medium |

---

## 4. Detailed Analysis of Deadlock

### 4.1 Coffman Conditions (Necessary and Sufficient Conditions for Deadlock)

For deadlock to occur, the following four conditions must **all hold simultaneously**. Conversely, breaking even one of them prevents deadlock.

1. **Mutual Exclusion**: Resources are used exclusively (only one thread holds a lock)
2. **Hold and Wait**: Holding one resource while waiting to acquire another
3. **No Preemption**: Resources held by another thread cannot be forcibly taken
4. **Circular Wait**: A circular dependency exists in the wait-for relationship among threads

```
Circular wait diagram for deadlock:

  Thread A                    Thread B
  +---------+                +---------+
  | lock(X) | <- holding     | lock(Y) | <- holding
  |         |                |         |
  | lock(Y) | -- waiting --> | lock(X) | -- waiting --+
  +---------+       |        +---------+              |
                    |                                  |
                    +-------- Cycle! ------------------+

  Resource Allocation Graph (RAG):
    T_A --> R_Y --> T_B --> R_X --> T_A
    (request) (held) (request) (held)  <- Cycle detected!
```

### 4.2 Deadlock Avoidance Strategies

| Strategy | Breaks Which Condition | Method | Constraint |
|----------|----------------------|--------|------------|
| Uniform lock ordering | Circular wait | All threads acquire locks in the same order | Lock order must be predetermined |
| Lock with timeout | Hold and wait | trylock + give up on timeout | Retry logic required |
| Bulk acquisition | Hold and wait | Acquire all locks at once | Concurrency decreases |
| Lock-free algorithms | Mutual exclusion | Use atomic operations like CAS | Extremely difficult to implement |
| Banker's algorithm | (Prevention) | Verify safety before resource allocation | Resource requirements must be known in advance |

### 4.3 Deadlock Demonstration and Avoidance (C)

```c
/* deadlock_demo.c - Demonstrates deadlock occurrence and avoidance
 *
 * Compile: gcc -Wall -pthread -o deadlock_demo deadlock_demo.c
 * Run:     ./deadlock_demo
 *
 * This program demonstrates two scenarios:
 *   1. Code that causes deadlock (inconsistent lock ordering)
 *   2. Code that avoids deadlock (uniform lock ordering)
 */
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

/* Mutexes corresponding to two shared resources */
static pthread_mutex_t lock_account_a = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lock_account_b = PTHREAD_MUTEX_INITIALIZER;

static int balance_a = 1000;
static int balance_b = 1000;

/* ===== Version that causes deadlock ===== */

void* transfer_a_to_b_UNSAFE(void* arg) {
    (void)arg;
    printf("[UNSAFE] Thread 1: Acquired lock_account_a\n");
    pthread_mutex_lock(&lock_account_a);

    /* Sleep creates an interleaving opportunity.
     * In production code, the same interleaving can occur even without
     * sleep due to scheduler decisions. */
    usleep(100000);  /* 0.1 seconds */

    printf("[UNSAFE] Thread 1: Waiting for lock_account_b...\n");
    pthread_mutex_lock(&lock_account_b);

    balance_a -= 100;
    balance_b += 100;
    printf("[UNSAFE] Thread 1: Transfer complete A=%d, B=%d\n", balance_a, balance_b);

    pthread_mutex_unlock(&lock_account_b);
    pthread_mutex_unlock(&lock_account_a);
    return NULL;
}

void* transfer_b_to_a_UNSAFE(void* arg) {
    (void)arg;
    printf("[UNSAFE] Thread 2: Acquired lock_account_b\n");
    pthread_mutex_lock(&lock_account_b);  /* <- Reverse order! */

    usleep(100000);

    printf("[UNSAFE] Thread 2: Waiting for lock_account_a...\n");
    pthread_mutex_lock(&lock_account_a);  /* <- Deadlock! */

    balance_b -= 200;
    balance_a += 200;
    printf("[UNSAFE] Thread 2: Transfer complete A=%d, B=%d\n", balance_a, balance_b);

    pthread_mutex_unlock(&lock_account_a);
    pthread_mutex_unlock(&lock_account_b);
    return NULL;
}

/* ===== Version that avoids deadlock ===== */

/*
 * Strategy: Uniform lock ordering
 * Compare Mutex addresses and always lock the smaller one first.
 * This makes it impossible for circular wait to occur.
 *
 * Why address comparison works:
 *   A unique total order can be defined for all Mutexes.
 *   If all threads follow this order, a circular wait cycle cannot form.
 */
void transfer_safe(pthread_mutex_t* from_lock, int* from_balance,
                   pthread_mutex_t* to_lock, int* to_balance,
                   int amount, const char* label) {
    /* Lock the one with the smaller address first */
    pthread_mutex_t* first  = (from_lock < to_lock) ? from_lock : to_lock;
    pthread_mutex_t* second = (from_lock < to_lock) ? to_lock : from_lock;

    printf("[SAFE] %s: Acquiring first lock\n", label);
    pthread_mutex_lock(first);
    usleep(100000);
    printf("[SAFE] %s: Acquiring second lock\n", label);
    pthread_mutex_lock(second);

    *from_balance -= amount;
    *to_balance   += amount;
    printf("[SAFE] %s: Transfer complete A=%d, B=%d\n",
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

/* ===== Avoidance via lock with timeout ===== */

void* transfer_with_timeout(void* arg) {
    (void)arg;
    int retries = 0;
    const int MAX_RETRIES = 5;

    while (retries < MAX_RETRIES) {
        pthread_mutex_lock(&lock_account_b);
        usleep(100000);

        /*
         * pthread_mutex_trylock: Attempts to acquire the lock; returns EBUSY
         * immediately on failure. Does not block, so deadlock is avoided.
         * However, all held locks must be released before retrying.
         */
        int ret = pthread_mutex_trylock(&lock_account_a);
        if (ret == 0) {
            /* Lock acquisition succeeded */
            balance_b -= 200;
            balance_a += 200;
            printf("[TIMEOUT] Transfer succeeded (retries: %d) A=%d, B=%d\n",
                   retries, balance_a, balance_b);
            pthread_mutex_unlock(&lock_account_a);
            pthread_mutex_unlock(&lock_account_b);
            return NULL;
        }

        /* Lock acquisition failed: release held lock and back off */
        pthread_mutex_unlock(&lock_account_b);
        retries++;
        printf("[TIMEOUT] Lock acquisition failed, retry #%d\n", retries);
        usleep(rand() % 100000);  /* Random backoff */
    }

    printf("[TIMEOUT] Maximum retries reached, transfer failed\n");
    return NULL;
}

int main(void) {
    srand((unsigned)time(NULL));

    /* --- Execute the safe version --- */
    printf("=== Safe transfer with uniform lock ordering ===\n");
    balance_a = 1000;
    balance_b = 1000;
    pthread_t t1, t2;
    pthread_create(&t1, NULL, safe_a_to_b, NULL);
    pthread_create(&t2, NULL, safe_b_to_a, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("Final balances: A=%d, B=%d (total=%d, always 2000)\n\n",
           balance_a, balance_b, balance_a + balance_b);

    /*
     * Note: The UNSAFE version intentionally deadlocks,
     * so it is commented out. If you try it, you will need to
     * force-terminate with the kill command.
     *
     * printf("=== Transfer that deadlocks (Warning: may hang) ===\n");
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

## 5. Thread Pools

### 5.1 Why Thread Pools Are Necessary

Creating and destroying a thread for each request is inefficient for the following reasons:

1. **Thread creation cost**: OS thread creation involves stack allocation, kernel structure (TCB) creation, and scheduler registration, incurring overhead of tens of microseconds
2. **Thread count explosion**: Under high load, a flood of requests causes unlimited thread creation, leading to memory exhaustion and increased scheduling overhead
3. **Thread reuse**: Reusing once-created threads amortizes the creation/destruction cost

```
Thread pool operation model:

  Task queue                  Worker thread pool
  +---------------------+     +-------------------------+
  |                     |     |  Worker 0: [executing]   |
  | [Task 5] <- newly added|  |  Worker 1: [waiting]     |
  | [Task 4]            |--->|  Worker 2: [executing]   |
  | [Task 3]            |     |  Worker 3: [waiting]     |
  |                     |     |  Worker 4: [executing]   |
  +---------------------+     +-------------------------+
         ^                           |
         |                           v
    submit(task)               Results/callbacks

  Lifecycle:
    1. Pool init:       Pre-create N worker threads
    2. Task submission: submit(task) adds to queue
    3. Task execution:  An idle worker takes a task from the queue and executes it
    4. Task completion: Worker waits for the next task (thread is NOT destroyed)
    5. Pool shutdown:   Stop all workers and release resources

  Worker thread internal loop:
    while (pool->running) {
        task = queue_pop(pool->queue);  // Block if empty
        task->function(task->arg);      // Execute task
    }
```

### 5.2 Thread Pool Sizing

The size of a thread pool (number of worker threads) directly impacts performance.

- **Too small**: CPUs idle, and throughput drops
- **Too large**: Context switch overhead increases and memory is wasted

**For CPU-intensive tasks (CPU-bound)**:
- Optimal thread count = number of CPU cores (N)
- Reason: CPU-intensive tasks continuously use the CPU, so threads beyond the core count only increase context switch overhead

**For I/O-intensive tasks (I/O-bound)**:
- Optimal thread count = N * (1 + W/C)
  - N: number of CPU cores
  - W: I/O wait time
  - C: computation time
- Example: If I/O wait is 9x the computation, N * 10 threads

| Workload | Recommended Thread Count | Rationale |
|----------|------------------------|-----------|
| Image processing, encryption | CPU core count | CPU is the bottleneck. More threads = more switch cost |
| Web server (DB queries) | Cores * 10-50 | CPU is idle during I/O waits |
| File downloads | Cores * 50-200 | Network I/O wait dominates |

### 5.3 Thread Pool Implementation in Python

```python
"""thread_pool.py - Basic thread pool implementation

Environment: Python 3.8+
Run: python3 thread_pool.py

This implementation teaches:
  - Task queue implementation using condition variables
  - Worker thread lifecycle management
  - Graceful shutdown
"""
import threading
import time
import random
from collections import deque
from typing import Callable, Any, Optional


class ThreadPool:
    """Fixed-size thread pool implementation.

    Why fixed size:
      Dynamically changing thread count makes the implementation complex.
      With proper sizing, fixed-size pools deliver sufficient performance.
      Java's ThreadPoolExecutor supports dynamic sizing, but the
      core/max/keepAlive configuration is complex and a common source of bugs.
    """

    def __init__(self, num_workers: int) -> None:
        if num_workers <= 0:
            raise ValueError("Number of workers must be 1 or more")

        self._num_workers = num_workers
        self._task_queue: deque = deque()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._running = True
        self._workers: list[threading.Thread] = []

        # Create and start worker threads
        for i in range(num_workers):
            t = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                name=f"PoolWorker-{i}",
                daemon=True,  # Auto-terminate when main thread exits
            )
            t.start()
            self._workers.append(t)

        print(f"ThreadPool: Started {num_workers} worker threads")

    def _worker_loop(self, worker_id: int) -> None:
        """Main loop for a worker thread.

        Waits on the condition variable until a task arrives,
        then takes it and executes it.
        """
        while True:
            with self._not_empty:
                # Wait until a task arrives or shutdown is signaled
                while len(self._task_queue) == 0 and self._running:
                    self._not_empty.wait()

                # Shutdown signaled and queue empty -> exit
                if not self._running and len(self._task_queue) == 0:
                    print(f"  Worker {worker_id}: Shutting down")
                    return

                func, args, kwargs = self._task_queue.popleft()

            # Execute the task outside the lock (so long tasks don't block the queue)
            try:
                func(*args, **kwargs)
            except Exception as e:
                print(f"  Worker {worker_id}: Error during task execution: {e}")

    def submit(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        """Add a task to the queue."""
        with self._not_empty:
            if not self._running:
                raise RuntimeError("Cannot submit tasks to a shut-down pool")
            self._task_queue.append((func, args, kwargs))
            self._not_empty.notify()  # Wake one waiting worker

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool.

        If wait=True, waits for all tasks in the queue to complete.
        Why graceful shutdown is important:
          - Interrupting running tasks can cause data inconsistency
          - Unprocessed tasks in the queue are lost
        """
        print("ThreadPool: Shutdown initiated")
        with self._not_empty:
            self._running = False
            self._not_empty.notify_all()  # Wake all workers to let them exit

        if wait:
            for t in self._workers:
                t.join()
        print("ThreadPool: Shutdown complete")


# === Demo task functions ===

def simulate_io_task(task_id: int) -> None:
    """Simulate an I/O-intensive task"""
    worker_name = threading.current_thread().name
    print(f"  [{worker_name}] Task {task_id}: Started (I/O simulation)")
    time.sleep(random.uniform(0.1, 0.5))  # Simulate I/O wait
    print(f"  [{worker_name}] Task {task_id}: Completed")


def simulate_cpu_task(task_id: int) -> None:
    """Simulate a CPU-intensive task"""
    worker_name = threading.current_thread().name
    print(f"  [{worker_name}] Task {task_id}: Started (CPU computation)")
    total = sum(i * i for i in range(100_000))  # CPU computation
    print(f"  [{worker_name}] Task {task_id}: Completed (result={total})")


if __name__ == "__main__":
    pool = ThreadPool(num_workers=3)

    print("\n--- Submitting I/O tasks ---")
    for i in range(8):
        pool.submit(simulate_io_task, i)

    time.sleep(3)  # Wait for task completion

    print("\n--- Submitting CPU tasks ---")
    for i in range(4):
        pool.submit(simulate_cpu_task, i)

    time.sleep(2)

    pool.shutdown(wait=True)
```

### 5.4 Java's ExecutorService (Standard Library Thread Pool)

```java
/* ExecutorServiceDemo.java - Java standard thread pool
 *
 * Compile and run: javac ExecutorServiceDemo.java && java ExecutorServiceDemo
 *
 * Java's java.util.concurrent package provides production-quality thread pools.
 * Designed by Doug Lea, highly optimized.
 */

import java.util.concurrent.*;
import java.util.ArrayList;
import java.util.List;

public class ExecutorServiceDemo {

    public static void main(String[] args) throws Exception {
        System.out.println("=== FixedThreadPool ===");
        fixedPoolDemo();

        System.out.println("\n=== Retrieving task results with Future ===");
        futureDemo();

        System.out.println("\n=== CachedThreadPool vs FixedThreadPool ===");
        comparisonDemo();
    }

    static void fixedPoolDemo() throws InterruptedException {
        /*
         * newFixedThreadPool(4): A pool with 4 worker threads.
         * If more than 4 tasks are submitted, they enter the queue and wait.
         *
         * Why fixed size is recommended:
         *   CachedThreadPool creates threads without limit, so submitting
         *   a large number of tasks at once risks OOM (OutOfMemory).
         *   FixedThreadPool guarantees an upper bound on thread count.
         */
        ExecutorService pool = Executors.newFixedThreadPool(4);

        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            pool.execute(() -> {
                String name = Thread.currentThread().getName();
                System.out.printf("  [%s] Task %d: Started%n", name, taskId);
                try {
                    Thread.sleep(100);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                System.out.printf("  [%s] Task %d: Completed%n", name, taskId);
            });
        }

        pool.shutdown();
        /*
         * shutdown(): Stops accepting new tasks and waits for existing tasks to finish.
         * shutdownNow(): Sends interrupts to running tasks and attempts immediate stop.
         *
         * awaitTermination(): Waits until all tasks complete within the specified time.
         */
        boolean finished = pool.awaitTermination(10, TimeUnit.SECONDS);
        System.out.println("All tasks complete: " + finished);
    }

    static void futureDemo() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(3);
        List<Future<Long>> futures = new ArrayList<>();

        for (int i = 1; i <= 5; i++) {
            final int n = i * 10_000_000;
            /*
             * submit(Callable): Submits a task with a return value.
             * Returns a Future object; the result is obtained via get().
             * get() blocks the caller until the task completes.
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
            /* get() is a blocking call. A timeout version get(timeout, unit) is also available. */
            long result = futures.get(i).get();
            System.out.printf("  Task %d result: %,d%n", i, result);
        }

        pool.shutdown();
        pool.awaitTermination(10, TimeUnit.SECONDS);
    }

    static void comparisonDemo() {
        /*
         * CachedThreadPool:
         *   - No upper bound on thread count (Integer.MAX_VALUE)
         *   - Idle threads are auto-destroyed after 60 seconds
         *   - Suited for many short-lived tasks, but risk of runaway
         *
         * FixedThreadPool:
         *   - Fixed thread count
         *   - Unbounded task queue (LinkedBlockingQueue)
         *   - Suited for stable workloads
         *
         * WorkStealingPool (Java 8+):
         *   - ForkJoinPool-based
         *   - Each worker has its own queue; steals tasks from others
         *   - Suited for uneven task sizes
         */
        System.out.println("  CachedThreadPool:     No limit, for short-lived tasks");
        System.out.println("  FixedThreadPool:      Fixed size, for stable workloads");
        System.out.println("  WorkStealingPool:     Work stealing, for uneven tasks");
        System.out.println("  ScheduledThreadPool:  For periodic tasks");
        System.out.println("  VirtualThreadPerTask: Java 21+ virtual thread version");
    }
}
```

---

## 6. Modern Concurrency Models

Threads are the foundation of concurrent programming, but directly manipulating them carries high risks of race conditions and deadlocks. As a result, higher-level abstraction concurrency models have evolved.

### 6.1 async/await (Cooperative Multitasking)

```
async/await operation model:

  Event loop (single-threaded):
  +---------------------------------------------------+
  |                                                   |
  |   +------+    +------+    +------+                |
  |   |Task A|    |Task B|    |Task C|  <- Coroutines |
  |   +--+---+    +--+---+    +--+---+                |
  |      |           |           |                     |
  |      v           |           |                     |
  |   [executing]    |           |   t=0               |
  |      |           |           |                     |
  |   await IO -->   |           |   t=1 (A suspends)  |
  |      |           v           |                     |
  |      |        [executing]    |   t=2 (B executes)  |
  |      |           |           |                     |
  |      |        await IO -->   |   t=3 (B suspends)  |
  |      |           |           v                     |
  |      |           |        [executing]  t=4 (C)     |
  |      v           |           |                     |
  |   IO complete    |           |   t=5 (A resumes)   |
  |   [executing]    |           |                     |
  |      |           v           |                     |
  |   complete   IO complete     |   t=6               |
  |              [executing]     |                     |
  |                 |           |                      |
  |              complete       |   t=7                |
  |                             v                      |
  |                          complete   t=8            |
  +---------------------------------------------------+

  Key points:
  - Only 1 OS thread -> race conditions cannot occur in principle
  - Explicitly yields control at await (cooperative)
  - Executes other tasks during I/O waits -> optimal for I/O-intensive processing
  - Not suitable for CPU-intensive processing (single-threaded)
```

### 6.2 Actor Model

```
Actor model (Erlang/Elixir, Akka):

  +----------+   Message       +----------+
  | Actor A  | ------------->  | Actor B  |
  |          |                 |          |
  | State: S1|   Message       | State: S2|
  | mailbox: | <-------------  | mailbox: |
  | [m1, m2] |                 | [m3]     |
  +----------+                 +----------+
       |                            |
       |  Message                   |  Message
       v                            v
  +----------+                 +----------+
  | Actor C  |                 | Actor D  |
  | State: S3|                 | State: S4|
  +----------+                 +----------+

  Principles:
  1. Each actor has independent state (no shared memory)
  2. Communication between actors is only through message passing
  3. Messages are sent asynchronously and queued in mailboxes
  4. An actor processes only one message at a time

  Advantages:
  - No shared memory means no locks needed -> greatly reduced deadlock risk
  - Fault isolation at the actor level (Erlang's "Let it crash" philosophy)
  - Natural extension to distributed systems (message sending across networks)
```

### 6.3 CSP (Communicating Sequential Processes)

Go's goroutines and channels adopt the CSP model. Go's maxim "Don't communicate by sharing memory; share memory by communicating" captures the essence of this model.

### 6.4 Concurrency Model Comparison Table

| Model | Shared Memory | Communication | Scheduling | Representative Languages/Frameworks |
|-------|--------------|---------------|------------|--------------------------------------|
| Threads + Locks | Yes | Shared variables | Preemptive (OS) | C/C++, Java, Python |
| async/await | No (typically) | Future/Promise | Cooperative (event loop) | JavaScript, Python asyncio, Rust tokio |
| Actor model | No | Message passing | Preemptive (runtime) | Erlang/Elixir, Akka (Scala) |
| CSP | No (recommended) | Channels | M:N hybrid | Go (goroutine + channel) |
| STM | Yes (transactional) | Transactional memory | Optimistic concurrency | Haskell, Clojure |
| Data parallel | Yes (controlled) | Implicit sync | Compiler/runtime | CUDA, OpenMP, SIMD |

---

## 7. Anti-patterns

### 7.1 Anti-pattern 1: Lock Granularity Too Large (Giant Lock)

```python
"""antipattern_giant_lock.py - Anti-pattern of overly coarse lock granularity

Problem:
  Protecting all data with a single lock eliminates concurrency.
  While one thread operates on accounts['A'],
  access to unrelated accounts['B'] is also blocked.
"""
import threading
import time

# === Anti-pattern: Single global lock ===

class BankAccountGiantLock:
    """Protects all accounts with one lock (low concurrency).

    Why this is a problem:
      Even with 100 accounts, only one thread at a time can
      access any account.
      This is effectively single-threaded execution.
    """
    def __init__(self):
        self.accounts = {'A': 1000, 'B': 1000, 'C': 1000}
        self.lock = threading.Lock()  # One lock protects all accounts

    def transfer(self, from_acc, to_acc, amount):
        with self.lock:  # All accounts are blocked
            time.sleep(0.01)  # Simulate I/O
            self.accounts[from_acc] -= amount
            self.accounts[to_acc] += amount


# === Improvement: Fine-grained locking (per-account locks) ===

class BankAccountFineLock:
    """Each account has its own lock (high concurrency).

    Improvement:
      Operations on accounts['A'] and accounts['C'] can now
      proceed simultaneously.
      However, lock ordering must be unified to prevent deadlock.
    """
    def __init__(self):
        self.accounts = {'A': 1000, 'B': 1000, 'C': 1000}
        self.locks = {k: threading.Lock() for k in self.accounts}

    def transfer(self, from_acc, to_acc, amount):
        # Acquire locks in dictionary order of account names (deadlock prevention)
        first, second = sorted([from_acc, to_acc])
        with self.locks[first]:
            with self.locks[second]:
                time.sleep(0.01)
                self.accounts[from_acc] -= amount
                self.accounts[to_acc] += amount


def benchmark(bank, label):
    start = time.time()
    threads = []
    # Execute A->B and C->A transfers concurrently
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
    print(f"  {label}: {elapsed:.3f}s, total balance={total} (always 3000)")


if __name__ == "__main__":
    print("=== Lock Granularity Comparison ===")
    benchmark(BankAccountGiantLock(), "Giant Lock  ")
    benchmark(BankAccountFineLock(),  "Fine Lock   ")
    print("\n  Fine Lock is faster (concurrent execution is possible)")
```

### 7.2 Anti-pattern 2: Checking Conditions with `if` in Condition Variables (Spurious Wakeup)

```python
"""antipattern_spurious_wakeup.py - Incorrect use of condition variables

Problem:
  When returning from a condition variable's wait(), the condition may not hold.
  Due to OS implementation details, "spurious wakeups" can occur without
  signal/broadcast. Using if to check the condition means processing
  may proceed in an invalid state during a spurious wakeup.
"""
import threading
from collections import deque

buffer = deque()
lock = threading.Lock()
cond = threading.Condition(lock)

# === Anti-pattern: Condition check with if ===

def consumer_BAD():
    """Dangerous code that doesn't account for spurious wakeups.

    Why dangerous:
      Buffer may be empty immediately after returning from wait().
      1. Spurious Wakeup: The OS unexpectedly wakes the thread
      2. Race after broadcast: Another consumer may have already taken the item

      Result: popleft() is called on an empty buffer, raising IndexError.
    """
    with cond:
        if len(buffer) == 0:  # if checks only once
            cond.wait()
        # Buffer may still be empty here!
        item = buffer.popleft()  # Possible IndexError


# === Correct pattern: Condition check with while ===

def consumer_GOOD():
    """Safe code that re-checks the condition with a while loop.

    Why while is correct:
      Re-checks the condition each time wait() returns.
      Even with spurious wakeups or races after broadcast,
      the loop continues waiting until the condition holds.
    """
    with cond:
        while len(buffer) == 0:  # while checks every time
            cond.wait()
        # At this point, the buffer definitely has an item
        item = buffer.popleft()  # Safe
        return item
```

**Lesson**: When using condition variables with `wait()`, **always** check conditions with a `while` loop. This is a recommended practice also specified in the POSIX specification.

---

## 8. Edge Case Analysis

### 8.1 Edge Case 1: Thread-Safe Lazy Initialization (Double-Checked Locking)

Lazy initialization is a pattern that initializes a resource at the point of first use. In a multithreaded environment, a race condition can occur when two threads attempt initialization simultaneously.

```java
/* DoubleCheckedLocking.java - Thread-safe lazy initialization
 *
 * Compile and run: javac DoubleCheckedLocking.java && java DoubleCheckedLocking
 */

public class DoubleCheckedLocking {

    /*
     * Why volatile is needed:
     *
     * In Java's memory model, the compiler or CPU may optimize (reorder)
     * the instruction order. Without volatile, the following problem can occur:
     *
     *   1. Allocate memory
     *   2. Assign address to instance  <- Reordered to happen first
     *   3. Execute constructor          <- Not yet initialized
     *
     *   If another thread checks instance after step 2 but before step 3,
     *   it receives a non-null but uninitialized object.
     *
     * volatile guarantees happens-before relationships and
     * prevents reordering.
     */
    private static volatile DoubleCheckedLocking instance;

    private final String data;

    private DoubleCheckedLocking() {
        // Simulate heavy initialization
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        this.data = "Initialization complete @ " + System.currentTimeMillis();
    }

    public static DoubleCheckedLocking getInstance() {
        /*
         * Double-Checked Locking pattern:
         *
         * 1st check (without lock): Fast path
         *   If already initialized, avoids lock acquisition overhead
         *
         * 2nd check (inside lock): Safe check
         *   Detects if another thread initialized between the 1st check and lock
         */
        if (instance == null) {              // 1st check (fast path)
            synchronized (DoubleCheckedLocking.class) {
                if (instance == null) {      // 2nd check (safe confirmation)
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

        System.out.println("\nConfirm all threads obtained the same instance");
        System.out.println("(All hash values should be identical)");
    }
}
```

### 8.2 Edge Case 2: The Dangerous Combination of fork() and Threads

When calling `fork()` in a multithreaded program, **only the calling thread** is copied into the child process. Other threads vanish. If a vanished thread was holding a Mutex, that Mutex remains permanently locked in the child process.

```
Problem with fork() and threads:

  Parent process:
  +--------------------------------+
  |  Thread A      Thread B       |
  |  (calls fork)  (holding Mutex X)|
  |                               |
  |  fork() ---------+            |
  +--------------------------------+
                     |
                     v
  Child process:
  +--------------------------------+
  |  Thread A'     (no Thread B)  |
  |                               |
  |  Mutex X: locked state        |
  |  (no thread exists to release it)|
  |  -> lock(X) waits forever!    |
  +--------------------------------+

  Countermeasures:
  1. Call exec() immediately after fork() (replaces the address space)
  2. Register lock management handlers with pthread_atfork() for before/after fork
  3. Avoid fork() in multithreaded programs; use posix_spawn() instead
```

**Why `posix_spawn()` is recommended**: `posix_spawn()` executes fork + exec as a single operation, avoiding lock inconsistency problems in the intermediate state (after fork, before exec). Many modern OSes implement `posix_spawn()` as vfork + exec, which also provides superior performance.

### 8.3 Edge Case 3: The ABA Problem (Lock-Free Algorithms)

In CAS (Compare-And-Swap) based lock-free algorithms, if a value changes A->B->A, CAS cannot detect this change.

```
ABA problem example (lock-free stack):

  Initial state: top -> [A] -> [B] -> [C]

  Thread 1: Attempts CAS(top, A, B)
            (pop A and make B the top)
            Preempted by scheduler

  Thread 2: pop A  -> top -> [B] -> [C]
            pop B  -> top -> [C]
            push A -> top -> [A] -> [C]  (B is gone!)

  Thread 1: Executes CAS(top, A, ?)
            Confirms top is A -> CAS succeeds
            But A's next points to B (stale information)
            -> top -> [B] -> ???  (B may already be freed)
            -> Memory corruption!

  Countermeasures:
  - Tagged pointers: Append a version number to the pointer
    CAS compares (pointer, version) pairs, so ABA is detectable
  - Hazard pointers: Register in-use nodes to delay free
  - Epoch-based reclamation: Advance the epoch before reclaiming old data
```

---

## 9. Thread-Local Storage (TLS)

Thread-local storage is a mechanism where each thread has **its own** copy of a variable. It appears like a global variable but actually holds an independent value per thread.

**Why TLS is necessary**: Global variables like errno become problematic in multithreaded environments. One thread's system call might set errno, and immediately another thread's system call could overwrite it. With TLS, each thread has its own errno, avoiding this problem.

```c
/* thread_local_demo.c - Thread-local storage usage example
 *
 * Compile: gcc -Wall -pthread -o tls_demo thread_local_demo.c
 * Run:     ./tls_demo
 */
#include <stdio.h>
#include <pthread.h>

/*
 * _Thread_local (C11) / __thread (GCC extension) declares a thread-local variable.
 * Each thread holds an independent copy of this variable.
 *
 * Internal implementation:
 *   Stored in the .tdata / .tbss sections of the ELF binary.
 *   Each thread's TCB has a pointer to its TLS block,
 *   accessed via the fs/gs segment register.
 */
static _Thread_local int tls_counter = 0;
static int shared_counter = 0;  /* For comparison: shared variable */

void* worker(void* arg) {
    int id = *(int*)arg;

    for (int i = 0; i < 1000; i++) {
        tls_counter++;      /* Thread-local: no contention */
        shared_counter++;   /* Shared variable: race condition occurs */
    }

    printf("Thread %d: tls_counter=%d (always 1000), "
           "shared_counter=%d (non-deterministic due to races)\n",
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

    printf("\nFinal shared_counter = %d (expected 4000, may be less due to races)\n",
           shared_counter);
    return 0;
}
```

---

## 10. Memory Models and Memory Barriers

In multithreaded programming, **instruction reordering** by CPUs and compilers can be a source of bugs.

```
CPU memory access optimization and reordering:

  Program order:                    Actual memory access order:
  +----------------------+         +----------------------+
  | 1. x = 42            |         | 2. ready = true <- first! |
  | 2. ready = true      |         | 1. x = 42     <- second! |
  +----------------------+         +----------------------+

  Why reordering occurs:
  - CPU: Store Buffer, cache hierarchy, Out-of-Order execution
  - Compiler: Instruction reordering for optimization

  When it becomes a problem:
    Thread A:                 Thread B:
    x = 42;                   while (!ready) { }
    ready = true;             print(x);  // Should be 42...might be 0!

    If reordering causes ready = true to be written to memory before
    x = 42, Thread B reads x=0.

  Memory barrier (fence):
    x = 42;
    __sync_synchronize();  // Full fence: writes before this are
    ready = true;          // guaranteed to be visible before writes after

  Language support:
  - C11:   atomic_thread_fence(), _Atomic types
  - C++11: std::atomic, std::memory_order
  - Java:  volatile, synchronized, java.util.concurrent.atomic
  - Go:    sync/atomic package, sync.Mutex
```

---

## 11. Practical Exercises

### Exercise 1: [Basic] Thread Creation and Observing Race Conditions

**Objective**: Observe that race conditions actually occur and confirm the effectiveness of Mutex protection.

```python
"""exercise_01_race_condition.py - Observing race conditions

Tasks:
  1. Run the following code without lock and confirm the counter is less than expected
  2. Enable the lock and confirm the counter always matches the expected value
  3. Vary the ITERATIONS value and observe how race frequency changes
     (Consider why races are harder to observe with small values and become
      prominent with large values)

Hint:
  Races are harder to observe with small ITERATIONS because the thread
  execution time is too short for overlapping to be likely.

Run: python3 exercise_01_race_condition.py
"""
import threading
import time

ITERATIONS = 1_000_000
NUM_THREADS = 4

counter = 0
lock = threading.Lock()
USE_LOCK = False  # Change to True to compare

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

    # Run 10 trials for statistics
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
              f"(expected: {expected:>10,}) "
              f"diff: {expected - counter:>8,}  "
              f"time: {elapsed:.3f}s")

    print(f"\n  Min: {min(results):,}")
    print(f"  Max: {max(results):,}")
    print(f"  Expected: {expected:,}")
    print(f"  USE_LOCK: {USE_LOCK}")

if __name__ == "__main__":
    main()
```

### Exercise 2: [Intermediate] Reproducing Deadlock and 3 Avoidance Techniques

**Objective**: Actually cause deadlock, then implement three different avoidance techniques.

```python
"""exercise_02_deadlock.py - 3 deadlock avoidance techniques

Tasks:
  1. Run deadlock_scenario() and confirm the program hangs
     (Interrupt with Ctrl+C)
  2. Implement each of the following 3 avoidance techniques:
     a) Uniform lock ordering (lock_ordering)
     b) Lock with timeout (timeout_approach)
     c) Bulk lock acquisition with contextmanager (batch_locking)
  3. Summarize the pros and cons of each technique

Run: python3 exercise_02_deadlock.py
"""
import threading
import time
import contextlib

lock_x = threading.Lock()
lock_y = threading.Lock()

# === Code that causes deadlock ===

def deadlock_worker_1():
    lock_x.acquire()
    print("Worker 1: Acquired lock_x")
    time.sleep(0.1)  # Create a gap for the other thread to acquire lock_y
    print("Worker 1: Waiting for lock_y...")
    lock_y.acquire()  # Deadlock!
    print("Worker 1: Acquired both")
    lock_y.release()
    lock_x.release()

def deadlock_worker_2():
    lock_y.acquire()
    print("Worker 2: Acquired lock_y")
    time.sleep(0.1)
    print("Worker 2: Waiting for lock_x...")
    lock_x.acquire()  # Deadlock!
    print("Worker 2: Acquired both")
    lock_x.release()
    lock_y.release()

def deadlock_scenario():
    """Running this as-is will hang due to deadlock"""
    t1 = threading.Thread(target=deadlock_worker_1)
    t2 = threading.Thread(target=deadlock_worker_2)
    t1.start()
    t2.start()
    t1.join(timeout=3)
    t2.join(timeout=3)
    if t1.is_alive() or t2.is_alive():
        print(">>> Deadlock detected! (timed out after 3 seconds)")

# === Avoidance technique a) Uniform lock ordering ===
# TODO: Use id() to compare lock addresses and always lock
#       the smaller one first

# === Avoidance technique b) Lock with timeout ===
# TODO: Use lock.acquire(timeout=...) and release all locks
#       on timeout, then retry

# === Avoidance technique c) Bulk lock acquisition ===
# TODO: Implement a contextmanager that sorts and bulk-acquires
#       all needed locks

if __name__ == "__main__":
    print("=== Deadlock Reproduction ===")
    deadlock_scenario()
```

### Exercise 3: [Advanced] Extended Thread Pool Implementation

**Objective**: Extend the thread pool from Section 5.3 to add practical features.

```python
"""exercise_03_advanced_pool.py - Extended thread pool

Tasks:
  Add the following features to the ThreadPool class:

  1. submit_with_future(): Return task results as a Future object
     - result = pool.submit_with_future(func, args)
     - value = result.get(timeout=5.0)  # Blocking result retrieval

  2. map(): Apply the same function to multiple arguments in parallel
     - results = pool.map(func, [arg1, arg2, arg3])
     - list(results) to get all results

  3. Statistics collection:
     - pool.stats() returns:
       - Completed task count
       - Waiting tasks in queue
       - Active worker count
       - Average task execution time

  Hints:
  - Future uses threading.Event internally,
    set() to notify result arrival, wait() to await result
  - map() can use submit_with_future() internally,
    collecting all Futures and returning results as a generator

Run: python3 exercise_03_advanced_pool.py
"""
import threading
import time
from collections import deque
from typing import Callable, Any, Iterator

class Future:
    """Object representing the asynchronous result of a task.

    TODO: Implement the following
    - __init__: Initialize Event and result/exception
    - set_result(value): Set the result and notify Event
    - set_exception(exc): Set the exception and notify Event
    - get(timeout=None): Wait for result arrival and return it
    """
    pass

class AdvancedThreadPool:
    """Extended thread pool.

    TODO: Based on the ThreadPool from section 5.3,
    add the above 3 features
    """
    pass

if __name__ == "__main__":
    print("=== Exercise 3: Extended Thread Pool ===")
    print("Implement the TODOs above and verify behavior")
    print()
    print("Test items:")
    print("  1. Retrieve computation results with submit_with_future")
    print("  2. Process multiple arguments in parallel with map")
    print("  3. Check statistics with stats")
    print("  4. Verify graceful shutdown behavior")
```

---

## 12. FAQ

### Q1: What is the GIL (Global Interpreter Lock)? Why does it exist?

CPython (the standard Python implementation) has a constraint called the **GIL** that allows only one thread to execute Python bytecode at a time.

**Why the GIL exists**: CPython's reference counting memory management needs to update object reference counters in a thread-safe manner. Giving each object its own lock would make lock management overhead very large, significantly degrading single-threaded performance. The GIL is a compromise that solves this with "one big lock."

**Impact of the GIL**:
- **CPU-intensive processing**: Multithreading does not run in parallel. Use the `multiprocessing` module instead.
- **I/O-intensive processing**: The GIL is released during I/O waits, so threads are still effective.
- **C extension modules**: C extensions like NumPy release the GIL during computation, so they can benefit from multithreading.

**Python 3.13+ (PEP 703)**: A `--disable-gil` option has been experimentally introduced to disable the GIL. In the future, a GIL-free CPython may become the standard.

### Q2: How should threads and coroutines be used differently?

| Aspect | Threads | Coroutines (async/await) |
|--------|---------|-------------------------|
| Scheduling | OS preemptive switching | Programmer explicitly switches at await |
| Memory consumption | ~1MB/thread (stack) | ~1KB/coroutine |
| Creation cost | Tens of us (syscall + stack alloc) | Hundreds of ns (object creation only) |
| Parallel execution | Possible (multicore utilization) | Not possible (single-threaded) |
| Race conditions | Can occur (shared memory) | Cannot occur in principle (single-threaded) |
| Best for | CPU-intensive computation | I/O-intensive processing (network, DB) |
| Debugging | Difficult (non-deterministic) | Relatively easy (deterministic) |

**Usage guidelines**:
- Handling thousands to tens of thousands of concurrent connections -> coroutines (async/await)
- Maxing out CPU computation -> threads (or multiprocessing)
- Both needed -> combine coroutines + thread pool

### Q3: When to use multithreading vs. multiprocessing?

- **Multithreading**: When memory sharing is needed, I/O-intensive, lightweight concurrency desired
- **Multiprocessing**: CPU-intensive computation, isolation needed (security, stability), avoiding GIL constraints (Python), crash resilience required

### Q4: Can volatile be used for synchronization?

**C/C++ volatile**: Only suppresses compiler optimizations (caching variables in registers, etc.) and **provides no thread synchronization functionality**. Since it does not issue memory barriers, volatile alone does not guarantee multithreaded safety. It should be limited to hardware register access and signal handler uses.

**Java volatile**: Includes memory barriers and guarantees happens-before relationships. Thread-safe for simple flag reads/writes, but read-modify-write (i++, etc.) is not atomic.

### Q5: What is the optimal number of threads?

Depends on the nature of the task (see Section 5.2). General guidelines:

- **CPU-bound**: Same number as CPU cores
- **I/O-bound**: Cores * (1 + wait time / computation time)
- **Mixed**: Find the optimal value through profiling

**Caution**: Creating too many threads increases context switch overhead, memory consumption, and cache pollution, which can actually decrease throughput. The expected optimal value should be verified through benchmarking.

---

## 13. Common Bugs and Debugging Techniques

### 13.1 Data Race Detection Tools

| Tool | Language/Environment | Detection Method | Usage |
|------|---------------------|-----------------|-------|
| ThreadSanitizer (TSan) | C/C++ | Compile-time instrumentation | `gcc -fsanitize=thread` |
| Helgrind | C/C++ (Valgrind) | Dynamic analysis | `valgrind --tool=helgrind ./prog` |
| Go Race Detector | Go | Compile-time instrumentation | `go run -race main.go` |
| Java Flight Recorder | Java | Sampling | Enable via JVM options |

### 13.2 Debugging Best Practices

1. **Ensure reproducibility**: Insert `sleep()` or barriers to control thread execution order and force specific interleavings
2. **Leverage logging**: Record timestamp + thread ID + operation in logs. However, logging itself must be thread-safe to be meaningful
3. **Create minimal reproduction cases**: Create the smallest code that reproduces the problem and eliminate noise
4. **Explicitly state invariants**: Document invariants like "this variable can only be modified while holding lock X" in comments and verify with assertions

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and observing its behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 14. Summary

| Concept | Key Points |
|---------|-----------|
| Threads | Lightweight execution units within a process. Fast communication via shared memory and risk of race conditions are two sides of the same coin |
| Thread models | 1:1 (Linux NPTL) is mainstream. M:N (Go goroutine) is suited for massive lightweight threads |
| Race conditions | Occur from non-atomic shared data access. Protected with Mutexes/semaphores/condition variables |
| Sync primitives | Mutex (exclusion), Semaphore (N concurrent), Condition variable (conditional wait), RWLock (read-write separation) |
| Deadlock | Occurs when all 4 Coffman conditions hold. Uniform lock ordering is the most practical avoidance strategy |
| Thread pool | Reduces overhead by reusing threads. Size should be tuned to workload characteristics |
| Memory model | Beware of CPU/compiler reordering. Control with volatile (Java) and atomic operations |
| Modern concurrency | async/await, actors, CSP provide higher-level abstractions |

---

## 15. Suggested Next Reading


---

## 16. References

1. Silberschatz, A., Galvin, P. B., & Gagne, G. (2018). *Operating System Concepts* (10th ed.). Wiley. Chapter 4: Threads & Concurrency. -- The standard textbook for thread fundamentals and models.
2. Herlihy, M. & Shavit, N. (2020). *The Art of Multiprocessor Programming* (2nd ed.). Morgan Kaufmann. -- Comprehensive coverage of synchronization primitives, lock-free algorithms, and memory model theory and implementation.
3. Butenhof, D. R. (1997). *Programming with POSIX Threads*. Addison-Wesley. -- The definitive reference for the pthreads API. Excellent coverage of condition variable usage and cancellation details.
4. Tanenbaum, A. S. & Bos, H. (2014). *Modern Operating Systems* (4th ed.). Pearson. Chapter 2.2: Threads. -- Detailed explanation of differences in user-level/kernel-level thread implementations.
5. Drepper, U. (2003). "The Native POSIX Thread Library for Linux." Red Hat Technical Report. -- A historical document describing the design of Linux NPTL and the rationale for choosing the 1:1 model.
6. Pike, R. (2012). "Go Concurrency Patterns." Google I/O Talk. -- Practical explanation of CSP patterns using Go's goroutines and channels.
7. Lea, D. (2005). *Concurrent Programming in Java: Design Principles and Patterns* (3rd ed.). Addison-Wesley. -- Commentary by the designer of Java's java.util.concurrent package.

---

## Suggested Next Reading

- [CPU Scheduling](./02-scheduling.md) - Proceed to the next topic

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://ja.wikipedia.org/) - Overview of technical concepts
