# Inter-Process Communication (IPC)

> Because processes have independent memory spaces, explicit communication mechanisms are required to exchange data between them.

## Learning Objectives

- [ ] Compare and contrast major IPC methods
- [ ] Understand the appropriate use of pipes, sockets, and shared memory
- [ ] Understand how signals work
- [ ] Understand the implementation of message queues and semaphores
- [ ] Select the appropriate IPC method for practical scenarios
- [ ] Explain the basics of synchronization and mutual exclusion

## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [CPU Scheduling](./02-scheduling.md)

---

## 1. Overview of IPC Methods

### 1.1 Why IPC Is Needed

```
Process Isolation:
  Each process has its own independent virtual address space
  -> When Process A writes to address 0x1000,
     it does not affect address 0x1000 in Process B
  -> This is the foundation of security and stability

  However, there are many situations where processes need to exchange data:
  - Shell pipelines (ls | grep | sort)
  - Web server <-> Application server <-> Database
  - Browser multi-process architecture
  - Communication between microservices
  - Process control (signals)

  -> Communication is done using IPC mechanisms provided by the OS

IPC Categories:
  +-----------------------------------------------------+
  | 1. Data Transfer                                     |
  |    Pipes, message queues, sockets                    |
  |    -> Kernel copies the data                         |
  |    -> Synchronization is built in                    |
  |                                                      |
  | 2. Shared Memory                                     |
  |    Shared memory, mmap                               |
  |    -> Physical memory is shared (no kernel copy)     |
  |    -> Synchronization must be implemented manually   |
  |                                                      |
  | 3. Notification                                      |
  |    Signals                                           |
  |    -> No data transfer, event notification only      |
  |                                                      |
  | 4. Synchronization                                   |
  |    Semaphores, mutexes, condition variables           |
  |    -> No data transfer, access control only          |
  +-----------------------------------------------------+
```

### 1.2 IPC Method Comparison Table

```
IPC Method Comparison:

+----------------+----------+----------+-----------+----------+----------+
| Method         | Direction| Speed    | Relation  | Use Case | Persist  |
+----------------+----------+----------+-----------+----------+----------+
| Pipe           | Unidirec | Medium   | Parent-   | ls | grep| Temporary|
|                | tional   |          | child     |          |          |
| Named Pipe     | Unidirec | Medium   | Unrelated | Log      | File     |
|                | tional   |          | OK        | collect  |          |
| Signal         | Notif    | Fast     | Unrelated | kill     | Temporary|
|                | only     |          | OK        |          |          |
| Message Queue  | Bidirec  | Medium   | Unrelated | Task     | Kernel   |
|                | tional   |          | OK        | dispatch |          |
| Shared Memory  | Bidirec  | Fastest  | Unrelated | DB       | Kernel   |
|                | tional   |          | OK        |          |          |
| Socket (TCP)   | Bidirec  | Med-Slow | Unrelated | Network  | Temporary|
|                | tional   |          | OK        |          |          |
| Unix Domain    | Bidirec  | Fast     | Same      | Docker   | File     |
| Socket         | tional   |          | machine   |          |          |
| mmap           | Bidirec  | Fastest  | Unrelated | File     | File     |
|                | tional   |          | OK        | mapping  |          |
| eventfd        | Notif    | Fast     | Unrelated | Events   | Temporary|
| D-Bus          | Bidirec  | Medium   | Unrelated | Desktop  | Daemon   |
|                | tional   |          | OK        |          |          |
+----------------+----------+----------+-----------+----------+----------+

Speed Comparison (approximate, 64-byte messages):
+----------------+---------------+--------------+
| IPC Method     | Latency       | Throughput   |
+----------------+---------------+--------------+
| Shared Memory  | ~50ns         | ~10 GB/s     |
| Pipe           | ~1us          | ~3 GB/s      |
| Unix Domain    | ~2us          | ~2 GB/s      |
| Message Queue  | ~5us          | ~500 MB/s    |
| TCP loopback   | ~10us         | ~1 GB/s      |
| D-Bus          | ~100us        | ~50 MB/s     |
+----------------+---------------+--------------+
* Varies significantly depending on environment and message size
```

### 1.3 IPC Method Selection Guidelines

```
Recommended IPC by Use Case:

  Same machine, parent-child processes:
  -> Pipe (simplest, ideal for shell integration)

  Same machine, unrelated processes, high-speed communication:
  -> Shared memory + semaphore (fastest, suitable for large data)
  -> Unix domain socket (bidirectional, rich API)

  Same machine, unrelated processes, message-oriented:
  -> POSIX message queue (priority support, clear boundaries)
  -> Unix domain socket (SOCK_DGRAM)

  Different machines:
  -> TCP socket (most versatile)
  -> gRPC (structured data, streaming support)

  Process control (start/stop/reload):
  -> Signals (SIGTERM, SIGHUP, etc.)

  Desktop application integration:
  -> D-Bus (Linux) / XPC (macOS) / COM (Windows)

  Microservice communication:
  -> gRPC / HTTP REST / Message brokers (RabbitMQ, Kafka)
```

---

## 2. Pipes

### 2.1 Anonymous Pipes

```
Anonymous Pipe:
  A unidirectional byte stream between parent and child processes
  -> Created with pipe() before fork()
  -> After fork(), parent and child share read/write ends

  $ ls -la | grep ".md" | wc -l

  ls --> [pipe] --> grep --> [pipe] --> wc
  stdout    stdin     stdout     stdin

  Kernel-level implementation:
  +------------------------------------------------------+
  | A pipe is a ring buffer inside the kernel             |
  |                                                       |
  | Write end --> [Kernel buffer] --> Read end             |
  |                   ^                                   |
  |                 Typically 64KB (Linux)                 |
  |                 Configurable via                       |
  |                 /proc/sys/fs/pipe-max-size             |
  |                                                       |
  | Buffer full  -> Write end blocks (backpressure)       |
  | Buffer empty -> Read end blocks                       |
  | All write ends closed -> Read end receives EOF        |
  | All read ends closed  -> Write end receives SIGPIPE   |
  +------------------------------------------------------+

  Atomic writes:
  -> Writes of PIPE_BUF (typically 4096 bytes) or less are atomic
  -> Data does not get interleaved even when multiple processes write
     simultaneously
  -> Writes exceeding PIPE_BUF have no atomicity guarantee
```

```c
// C program example using a pipe
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    int pipefd[2];  // pipefd[0] = read end, pipefd[1] = write end

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
        // Child process: read side
        close(pipefd[1]);  // Close write end

        char buf[256];
        ssize_t n;
        while ((n = read(pipefd[0], buf, sizeof(buf) - 1)) > 0) {
            buf[n] = '\0';
            printf("Child received: %s\n", buf);
        }

        close(pipefd[0]);
        exit(EXIT_SUCCESS);
    } else {
        // Parent process: write side
        close(pipefd[0]);  // Close read end

        const char *msg = "Hello from parent!";
        write(pipefd[1], msg, strlen(msg));

        close(pipefd[1]);  // Send EOF
        wait(NULL);        // Wait for child to finish
    }

    return 0;
}
```

```c
// Bidirectional communication requires two pipes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main() {
    int pipe_parent_to_child[2];  // Parent -> Child
    int pipe_child_to_parent[2];  // Child -> Parent

    pipe(pipe_parent_to_child);
    pipe(pipe_child_to_parent);

    pid_t pid = fork();
    if (pid == 0) {
        // Child process
        close(pipe_parent_to_child[1]);
        close(pipe_child_to_parent[0]);

        char buf[256];
        read(pipe_parent_to_child[0], buf, sizeof(buf));
        printf("Child received: %s\n", buf);

        const char *reply = "Hello from child!";
        write(pipe_child_to_parent[1], reply, strlen(reply));

        close(pipe_parent_to_child[0]);
        close(pipe_child_to_parent[1]);
        exit(0);
    } else {
        // Parent process
        close(pipe_parent_to_child[0]);
        close(pipe_child_to_parent[1]);

        const char *msg = "Hello from parent!";
        write(pipe_parent_to_child[1], msg, strlen(msg));

        char buf[256];
        ssize_t n = read(pipe_child_to_parent[0], buf, sizeof(buf) - 1);
        buf[n] = '\0';
        printf("Parent received: %s\n", buf);

        close(pipe_parent_to_child[1]);
        close(pipe_child_to_parent[0]);
    }
    return 0;
}
```

### 2.2 Named Pipes (FIFO)

```
Named Pipe (FIFO):
  A pipe with a name on the file system
  -> Enables communication between unrelated processes
  -> Any process that knows the file path can connect

  Creation:
  $ mkfifo /tmp/myfifo
  $ ls -la /tmp/myfifo
  prw-r--r-- 1 user user 0 ... /tmp/myfifo
  ^ 'p' = pipe

  Usage:
  # Terminal 1 (read side):
  $ cat /tmp/myfifo          # Blocks: waits until a writer connects

  # Terminal 2 (write side):
  $ echo "hello" > /tmp/myfifo  # Write -> "hello" appears in Terminal 1

  Characteristics:
  - Uses the same kernel buffer as anonymous pipes
  - Open blocks until both ends are connected (by default)
  - Can be made non-blocking with O_NONBLOCK
  - Persistent (the file remains until explicitly removed with rm)
  - Suitable for simple IPC

  Limitations:
  - Unidirectional (two FIFOs needed for bidirectional communication)
  - Byte stream (no message boundaries)
  - Cannot be used across a network
```

```python
# Named pipe example in Python

# Server side (reader)
import os

fifo_path = "/tmp/myfifo"

# Create FIFO (skip if it already exists)
if not os.path.exists(fifo_path):
    os.mkfifo(fifo_path)

print("Waiting for client connection...")
with open(fifo_path, 'r') as fifo:
    while True:
        line = fifo.readline()
        if not line:
            break
        print(f"Received: {line.strip()}")

# Client side (writer)
with open(fifo_path, 'w') as fifo:
    fifo.write("Hello from client!\n")
    fifo.flush()
```

### 2.3 Practical Pipe Usage Patterns

```
Pipe Usage Patterns:

  1. Log processing pipeline:
  $ tail -f /var/log/syslog | grep "error" | tee errors.log | mail admin

  2. Data transformation pipeline:
  $ cat data.csv | cut -d',' -f1,3 | sort | uniq -c | sort -rn | head -20

  3. Parallel processing:
  $ cat urls.txt | xargs -P 10 -I{} curl -s {} > /dev/null

  4. Process substitution (Bash extension):
  $ diff <(ls dir1) <(ls dir2)     # Compare output of two commands
  $ paste <(cut -f1 a.tsv) <(cut -f2 b.tsv)  # Join columns

  5. Coprocess (Bash 4.0+):
  $ coproc bc        # Launch bc with bidirectional pipes
  $ echo "2+3" >&${COPROC[1]}   # Send input to bc
  $ read result <&${COPROC[0]}   # Read the result
  $ echo $result     # 5

  Pipe Performance Considerations:
  +-----------------------------------------------------+
  | - Default pipe buffer size: 64KB                     |
  | - Adjustable via fcntl(fd, F_SETPIPE_SZ, size)      |
  | - Maximum 1MB (/proc/sys/fs/pipe-max-size)           |
  | - Zero-copy transfer available via splice()          |
  | - Zero-copy from user space via vmsplice()           |
  | - Zero-copy branching between pipes via tee()        |
  +-----------------------------------------------------+
```

---

## 3. Signals

### 3.1 Signal Basics

```
Signal: An asynchronous notification mechanism for processes

  Signal characteristics:
  - A type of software interrupt
  - Carries minimal information (signal number only)
  - Asynchronous (unpredictable delivery timing)
  - Delivered by the kernel to the process

  Signal sources:
  1. User actions: Ctrl+C (SIGINT), Ctrl+Z (SIGTSTP)
  2. Kernel: SIGSEGV (invalid memory access), SIGFPE (division by zero)
  3. Other processes: kill() system call
  4. Timers: alarm(), setitimer()
  5. Child process termination: SIGCHLD

  Major signals:
  +----------+------+----------------------+----------+
  | Signal   | No.  | Action               | Default  |
  +----------+------+----------------------+----------+
  | SIGHUP   | 1    | Terminal hangup /     | Terminate|
  |          |      | config reload         |          |
  | SIGINT   | 2    | Ctrl+C (interrupt)    | Terminate|
  | SIGQUIT  | 3    | Ctrl+\ (core dump)    | Core dump|
  | SIGILL   | 4    | Illegal instruction   | Core dump|
  | SIGTRAP  | 5    | Trace/breakpoint      | Core dump|
  | SIGABRT  | 6    | abort()               | Core dump|
  | SIGBUS   | 7    | Bus error             | Core dump|
  | SIGFPE   | 8    | Floating-point excep  | Core dump|
  | SIGKILL  | 9    | Force kill (uncatch)  | Terminate|
  | SIGUSR1  | 10   | User-defined 1        | Terminate|
  | SIGSEGV  | 11   | Segmentation fault    | Core dump|
  | SIGUSR2  | 12   | User-defined 2        | Terminate|
  | SIGPIPE  | 13   | Broken pipe           | Terminate|
  | SIGALRM  | 14   | alarm() timer         | Terminate|
  | SIGTERM  | 15   | Graceful termination  | Terminate|
  | SIGCHLD  | 17   | Child process exited  | Ignore   |
  | SIGCONT  | 18   | Resume                | Resume   |
  | SIGSTOP  | 19   | Suspend (uncatchable) | Stop     |
  | SIGTSTP  | 20   | Ctrl+Z (terminal stop)| Stop     |
  | SIGWINCH | 28   | Window size change    | Ignore   |
  +----------+------+----------------------+----------+

  SIGKILL(9) and SIGSTOP(19) cannot be caught, ignored, or blocked at all
  -> Handled directly by the kernel (no user-space handlers)
  -> The last resort for dealing with hung processes
```

### 3.2 Sending and Catching Signals

```bash
# Methods for sending signals

# kill command (despite the name, sends signals in general)
kill -TERM 1234           # Send SIGTERM to PID 1234
kill -9 1234              # SIGKILL PID 1234 (force kill)
kill -HUP $(cat /var/run/nginx.pid)  # Reload nginx config
kill -USR1 $(pidof dd)    # Show dd progress

# killall: specify by process name
killall -TERM firefox     # Send SIGTERM to all firefox processes
killall -HUP syslogd      # Reload syslogd config

# pkill: specify by pattern match
pkill -f "python server.py"  # Match against the entire command line
pkill -u username            # All processes of a specific user

# Send to a process group
kill -TERM -1234             # Send to all processes in PGID 1234
kill -TERM 0                 # Send to own process group

# Send to an entire session
pkill -s 1234                # All processes in session ID 1234
```

```c
// Signal handling in C (sigaction is recommended)
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

// volatile sig_atomic_t: a type safe to share with signal handlers
volatile sig_atomic_t got_sigterm = 0;
volatile sig_atomic_t got_sighup = 0;

void sigterm_handler(int signum) {
    got_sigterm = 1;
    // Note: Only async-signal-safe functions may be used inside signal handlers
    // printf(), malloc(), free(), etc. are NOT safe
    // write() IS safe
    const char msg[] = "SIGTERM received\n";
    write(STDOUT_FILENO, msg, sizeof(msg) - 1);
}

void sighup_handler(int signum) {
    got_sighup = 1;
}

int main() {
    // Configure sigaction structs
    struct sigaction sa_term, sa_hup;

    // SIGTERM handler
    memset(&sa_term, 0, sizeof(sa_term));
    sa_term.sa_handler = sigterm_handler;
    sigemptyset(&sa_term.sa_mask);       // No additional signals blocked during handler
    sa_term.sa_flags = 0;                // No flags
    sigaction(SIGTERM, &sa_term, NULL);

    // SIGHUP handler
    memset(&sa_hup, 0, sizeof(sa_hup));
    sa_hup.sa_handler = sighup_handler;
    sigemptyset(&sa_hup.sa_mask);
    sa_hup.sa_flags = SA_RESTART;        // Auto-restart interrupted system calls
    sigaction(SIGHUP, &sa_hup, NULL);

    // Ignore SIGPIPE (prevent unexpected termination on socket/pipe disconnect)
    struct sigaction sa_pipe;
    memset(&sa_pipe, 0, sizeof(sa_pipe));
    sa_pipe.sa_handler = SIG_IGN;
    sigaction(SIGPIPE, &sa_pipe, NULL);

    printf("PID: %d. Send SIGTERM or SIGHUP\n", getpid());

    while (!got_sigterm) {
        if (got_sighup) {
            printf("Reloading configuration...\n");
            got_sighup = 0;
            // Configuration file reload logic here
        }
        sleep(1);
    }

    printf("Performing graceful shutdown...\n");
    // Cleanup (close files, delete temp files, etc.)
    return 0;
}
```

### 3.3 Practical Signal Patterns

```
SIGTERM vs SIGKILL - Proper Usage:

  Correct procedure:
  1. Send SIGTERM (give the process a chance for graceful shutdown)
  2. Wait a few seconds (timeout)
  3. If no response, send SIGKILL (last resort)

  +----------------------------------------------+
  | SIGTERM (15):                                 |
  | -> Process can catch it and perform cleanup   |
  | -> Commit/rollback DB transactions            |
  | -> Delete temporary files                     |
  | -> Close connections                          |
  | -> Write log entries                          |
  |                                               |
  | SIGKILL (9):                                  |
  | -> Immediate forced termination (no handler)  |
  | -> Temp files may remain                      |
  | -> DB transactions may be left incomplete     |
  | -> Shared memory may not be freed             |
  | -> Should only be used as a last resort       |
  +----------------------------------------------+

Common signal patterns in practice:

  1. Web server graceful restart:
     kill -HUP $(pidof nginx)
     -> Starts new workers with new configuration
     -> Existing connections are served by existing workers
     -> Existing workers terminate after completing their requests

  2. Log rotation:
     kill -USR1 $(pidof nginx)
     -> Reopens log files
     -> Works in coordination with logrotate

  3. Docker stop behavior:
     docker stop container_name
     -> Sends SIGTERM -> Waits 10 seconds -> Sends SIGKILL
     -> Timeout adjustable via --stop-timeout

  4. systemd service stop:
     systemctl stop myservice
     -> Sends SIGTERM -> Waits TimeoutStopSec -> Sends SIGKILL
     -> Customizable in service configuration files

  5. Kubernetes pod termination:
     kubectl delete pod mypod
     -> Executes preStop hook
     -> Sends SIGTERM
     -> Waits terminationGracePeriodSeconds (default 30 seconds)
     -> Sends SIGKILL
```

```python
# Signal handling in Python (practical example)
import signal
import sys
import os
import time
import logging
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class GracefulShutdown:
    """A class to manage graceful shutdown"""

    def __init__(self):
        self.shutdown_requested = False
        self.connections = []

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGHUP, self._handle_sighup)

    def _handle_signal(self, signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}. Initiating shutdown...")
        self.shutdown_requested = True

    def _handle_sighup(self, signum, frame):
        logger.info("Received SIGHUP. Reloading configuration...")
        self.reload_config()

    def reload_config(self):
        """Reload configuration file"""
        logger.info("Reloading configuration file...")
        # config = load_config("/etc/myapp/config.yaml")

    def cleanup(self):
        """Cleanup processing"""
        logger.info("Closing connections...")
        for conn in self.connections:
            try:
                conn.close()
            except Exception as e:
                logger.error(f"Failed to close connection: {e}")

        logger.info("Deleting temporary files...")
        # cleanup_temp_files()

        logger.info("Cleanup complete")

    def run(self):
        """Main loop"""
        logger.info(f"Started with PID: {os.getpid()}")

        while not self.shutdown_requested:
            try:
                # Main processing
                self.process_work()
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error: {e}")

        self.cleanup()
        logger.info("Shut down gracefully")

    def process_work(self):
        """Actual business logic"""
        pass

if __name__ == "__main__":
    app = GracefulShutdown()
    app.run()
```

### 3.4 Signal Caveats and Limitations

```
Signal caveats:

  1. Only async-signal-safe functions may be used inside handlers:
     +------------------------------------------+
     | Safe (async-signal-safe):                 |
     | write(), read(), _exit(), signal()        |
     | open(), close(), fork(), execve()         |
     | kill(), raise(), sigaction()              |
     |                                           |
     | Unsafe:                                   |
     | printf(), fprintf() -> risk of buffer     |
     |   corruption                              |
     | malloc(), free() -> risk of heap          |
     |   corruption                              |
     | syslog() -> deadlock from internal locks  |
     | pthread_mutex_lock() -> deadlock          |
     +------------------------------------------+

  2. Signal loss:
     -> When the same signal fires in rapid succession, pending
        instances may be lost
     -> Standard POSIX signals are not queued (only one is held pending)
     -> Real-time signals (SIGRTMIN-SIGRTMAX) ARE queued

  3. Signals in multithreaded programs:
     -> Signals are sent to the process as a whole
     -> Which thread receives it is indeterminate
     -> Use pthread_sigmask() to control which thread handles signals
     -> Recommended: create a dedicated signal handling thread

  4. signalfd (Linux-specific):
     -> Receive signals as file descriptors
     -> Can be integrated with epoll/select
     -> Avoids signal handler pitfalls

  5. The self-pipe trick as an alternative:
     -> Create a pipe; write() inside the signal handler
     -> Monitor the pipe in the main loop with select/poll/epoll
     -> A technique to work around signal handler restrictions
```

---

## 4. Shared Memory and mmap

### 4.1 POSIX Shared Memory

```
Shared Memory:
  Multiple processes map the same physical memory region
  -> Fastest IPC (direct access without going through the kernel)
  -> Synchronization must be handled manually (semaphores, mutexes, etc.)

  Process A             Process B
  +----------+         +----------+
  |Virtual   |         |Virtual   |
  |Memory    |         |Memory    |
  | +------+ |         | +------+ |
  | |Shared|-|----+----|-|Shared| |
  | |Region| |    |    | |Region| |
  | +------+ |    |    | +------+ |
  +----------+    |    +----------+
                  v
            Same region in
            physical memory

  POSIX Shared Memory API:
  shm_open()    -> Create/open a shared memory object
  ftruncate()   -> Set the size
  mmap()        -> Map into memory
  munmap()      -> Unmap
  shm_unlink()  -> Delete the shared memory object

  System V Shared Memory (legacy):
  shmget()  -> Create/get a shared memory segment
  shmat()   -> Attach (map)
  shmdt()   -> Detach
  shmctl()  -> Control (delete, etc.)
  -> The POSIX version is more modern and recommended
```

```c
// POSIX shared memory + semaphore implementation example

// Common header (shared_data.h)
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
    int ready;  // Flag
} SharedData;

// Producer (writer)
int producer() {
    // Create shared memory
    int fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(SharedData));

    SharedData *data = mmap(NULL, sizeof(SharedData),
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    // Create semaphore
    sem_t *sem = sem_open(SEM_NAME, O_CREAT, 0666, 1);

    for (int i = 0; i < 100; i++) {
        sem_wait(sem);  // Acquire lock
        data->counter = i;
        snprintf(data->message, sizeof(data->message), "Message #%d", i);
        data->ready = 1;
        sem_post(sem);  // Release lock
        usleep(10000);  // Wait 10ms
    }

    munmap(data, sizeof(SharedData));
    sem_close(sem);
    return 0;
}

// Consumer (reader)
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
            // data->ready = 0;  // Cannot write since mapped as read-only
        }
        sem_post(sem);
        usleep(10000);
    }

    munmap(data, sizeof(SharedData));
    sem_close(sem);
    return 0;
}
```

### 4.2 mmap (Memory Mapping)

```
mmap:
  Maps a file or anonymous memory into a process's virtual address space

  Use cases:
  1. Faster file I/O
  2. Sharing files between processes
  3. Anonymous mapping (large memory allocation)
  4. Loading shared libraries

  Regular I/O vs mmap:
  +------------------------------------------------------+
  | Regular read():                                       |
  | Disk -> Kernel buffer -> User buffer                  |
  | -> 2 memory copies                                   |
  |                                                       |
  | mmap:                                                 |
  | Disk -> Page cache <- Direct access from user space   |
  | -> 0 copies (directly references the page cache)      |
  |                                                       |
  | However:                                              |
  | - Overhead for small files (page table setup)         |
  | - mmap is advantageous for random access patterns     |
  | - read() is also fast enough for sequential access    |
  +------------------------------------------------------+

  mmap flags:
  MAP_SHARED:    Reflect changes to the original file. Shared with others
  MAP_PRIVATE:   Copy-on-Write. Changes visible only to the calling process
  MAP_ANONYMOUS: No file. Zero-initialized memory (used by large malloc)
  MAP_FIXED:     Map at a specified address (dangerous, rarely used)
  MAP_HUGETLB:   Use huge pages
  MAP_POPULATE:  Pre-fault pages at mapping time (page in immediately)
```

```c
// mmap file read/write example
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

int main() {
    const char *filepath = "/tmp/mmap_example.dat";

    // Create file and write initial data
    int fd = open(filepath, O_RDWR | O_CREAT | O_TRUNC, 0666);
    const char *initial = "Hello, mmap world!\n";
    write(fd, initial, strlen(initial));

    // Get file size
    struct stat st;
    fstat(fd, &st);

    // Memory map the file
    char *mapped = mmap(NULL, st.st_size,
                       PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    close(fd);  // OK to close fd after mapping

    // Read data via mapping
    printf("Read via mmap: %s", mapped);

    // Modify data via mapping (reflected to the file)
    memcpy(mapped, "HELLO", 5);

    // Ensure changes are written to disk
    msync(mapped, st.st_size, MS_SYNC);

    // Unmap
    munmap(mapped, st.st_size);

    // Verify with regular I/O
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
# mmap usage examples in Python

import mmap
import os

# Efficient search of a large file
def search_large_file(filepath, pattern):
    """Fast search of a large file using mmap"""
    with open(filepath, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            pattern_bytes = pattern.encode()
            offset = 0
            results = []
            while True:
                pos = mm.find(pattern_bytes, offset)
                if pos == -1:
                    break
                # Calculate line number
                line_num = mm[:pos].count(b'\n') + 1
                results.append((line_num, pos))
                offset = pos + 1
            return results

# Inter-process shared memory (Python 3.8+)
from multiprocessing import shared_memory

# Writer
shm = shared_memory.SharedMemory(name='my_shared', create=True, size=1024)
shm.buf[0:5] = b'Hello'
print(f"Shared memory name: {shm.name}")

# Reader (separate process)
shm2 = shared_memory.SharedMemory(name='my_shared')
print(bytes(shm2.buf[0:5]))  # b'Hello'
shm2.close()

# Cleanup
shm.close()
shm.unlink()
```

---

## 5. Sockets

### 5.1 Unix Domain Sockets

```
Unix Domain Socket:
  Inter-process communication within the same machine
  -> Faster than TCP sockets (no network stack overhead)
  -> Creates a socket file on the file system
  -> Access controlled via file permissions

  Use cases: Docker, PostgreSQL, MySQL, nginx, systemd, D-Bus

  $ ls -la /var/run/docker.sock
  srw-rw---- 1 root docker 0 ... /var/run/docker.sock
  ^ 's' = socket

  Comparison with TCP sockets:
  +------------------+---------------+---------------+
  | Property         | Unix Domain   | TCP loopback  |
  +------------------+---------------+---------------+
  | Latency          | ~2us          | ~10us         |
  | Throughput       | ~6 GB/s       | ~4 GB/s       |
  | Authentication   | UID/GID avail | IP/Port only  |
  | Access control   | File perms    | Firewall      |
  | Network support  | No            | Yes           |
  | fd passing       | Yes           | No            |
  | Credentials      | Yes           | No            |
  +------------------+---------------+---------------+

  Socket types:
  SOCK_STREAM:   TCP-like. Connection-oriented, ordered, byte stream
  SOCK_DGRAM:    UDP-like. Connectionless, message boundaries preserved
  SOCK_SEQPACKET: Ordered + message boundaries (similar to SCTP)

  Unix domain socket unique features:
  1. File descriptor passing (fd passing):
     -> Share file descriptors between processes
     -> Used in systemd's socket activation
     -> sendmsg() / recvmsg() with SCM_RIGHTS

  2. Peer credential retrieval:
     -> Obtain the UID, GID, and PID of the connecting peer
     -> SO_PEERCRED socket option
     -> Used in D-Bus authentication

  3. Abstract sockets (Linux-specific):
     -> No physical file on the file system
     -> Name starts with '\0'
     -> Auto-cleanup (disappears when the process exits)
```

```python
# Unix domain socket server/client example

import socket
import os

SOCKET_PATH = "/tmp/my_unix_socket"

# Server
def unix_server():
    # Remove existing socket file
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(5)

    # Set socket file permissions
    os.chmod(SOCKET_PATH, 0o666)

    print(f"Server started: {SOCKET_PATH}")
    try:
        while True:
            conn, addr = server.accept()
            print(f"Connection accepted")

            # Retrieve peer credentials (Linux-specific)
            # creds = conn.getsockopt(socket.SOL_SOCKET,
            #                        socket.SO_PEERCRED, 12)

            data = conn.recv(1024)
            print(f"Received: {data.decode()}")

            conn.sendall(b"ACK: " + data)
            conn.close()
    finally:
        server.close()
        os.unlink(SOCKET_PATH)

# Client
def unix_client(message):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(SOCKET_PATH)

    client.sendall(message.encode())
    response = client.recv(1024)
    print(f"Response: {response.decode()}")

    client.close()
```

### 5.2 TCP Sockets

```
TCP Socket:
  Can be used for communication across a network
  -> Most versatile IPC (supports inter-machine communication)
  -> Foundation for HTTP, gRPC, database protocols

  Socket programming flow:
  +------------------------------------------+
  | Server               Client              |
  |                                          |
  | socket()             socket()            |
  | bind()                                   |
  | listen()                                 |
  | accept() <---------- connect()           |
  | |                    |                   |
  | recv()   <---------- send()              |
  | send()   ----------> recv()              |
  | |                    |                   |
  | close()              close()             |
  +------------------------------------------+

  High-performance server implementation patterns:
  1. Multi-process (fork/prefork):
     -> Fork per connection
     -> Process isolation provides crash resilience
     -> Apache prefork MPM

  2. Multi-threaded:
     -> Thread per connection
     -> Efficient with shared memory
     -> Java's Thread-per-request

  3. I/O multiplexing (select/poll/epoll/kqueue):
     -> Single thread handles many connections
     -> Event-driven
     -> nginx, Node.js, Redis

  4. io_uring (Linux 5.1+):
     -> Asynchronous communication with the kernel
     -> Dramatically reduces system call overhead
     -> Adopted by modern high-performance servers
```

```python
# High-performance server using I/O multiplexing (select/epoll example)
import selectors
import socket
import types

# Automatically selects epoll on Linux, kqueue on macOS
sel = selectors.DefaultSelector()

def accept_connection(sock):
    conn, addr = sock.accept()
    print(f"Connection accepted: {addr}")
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
            data.outb += recv_data  # Echo back
        else:
            print(f"Disconnected: {data.addr}")
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

    print(f"Server started: {host}:{port}")
    try:
        while True:
            events = sel.select(timeout=None)
            for key, mask in events:
                if key.data is None:
                    accept_connection(key.fileobj)
                else:
                    service_connection(key, mask)
    except KeyboardInterrupt:
        print("Shutting down")
    finally:
        sel.close()

if __name__ == "__main__":
    run_server()
```

---

## 6. Message Queues

### 6.1 POSIX Message Queues

```
POSIX Message Queue:
  Send and receive data in message units
  -> Clear message boundaries (unlike pipes)
  -> Priority-based messages
  -> Persisted in the kernel (survives process termination)

  Comparison with pipes:
  +------------------+--------------+--------------+
  | Property         | Pipe         | Message Queue|
  +------------------+--------------+--------------+
  | Data format      | Byte stream  | Messages     |
  | Boundaries       | None         | Yes          |
  | Priority         | None         | Yes          |
  | Multiple readers | Data split   | Delivered to |
  |                  |              | one reader   |
  | Persistence      | Destroyed on | Until        |
  |                  | process exit | explicit del |
  | Direction        | Unidirectional| Unidirection|
  |                  |              | (bidir poss) |
  | Size limit       | Buffer size  | Message count|
  +------------------+--------------+--------------+

  API:
  mq_open()    -> Create/open a queue
  mq_send()    -> Send a message (with optional priority)
  mq_receive() -> Receive a message (highest priority first)
  mq_notify()  -> Asynchronous notification on message arrival
  mq_close()   -> Close the queue
  mq_unlink()  -> Delete the queue
```

```c
// POSIX message queue example
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mqueue.h>
#include <errno.h>

#define QUEUE_NAME "/my_task_queue"
#define MAX_MSG_SIZE 256
#define MAX_MSGS 10

// Producer
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

    // Send messages with priority
    mq_send(mq, "High priority task", strlen("High priority task"), 10);   // Priority 10
    mq_send(mq, "Low priority task", strlen("Low priority task"), 1);      // Priority 1
    mq_send(mq, "Medium priority task", strlen("Medium priority task"), 5); // Priority 5

    mq_close(mq);
}

// Consumer
void consumer() {
    mqd_t mq = mq_open(QUEUE_NAME, O_RDONLY);
    char buffer[MAX_MSG_SIZE + 1];
    unsigned int priority;

    // Messages are received in order of highest priority
    while (1) {
        ssize_t bytes = mq_receive(mq, buffer, MAX_MSG_SIZE + 1, &priority);
        if (bytes == -1) {
            if (errno == EAGAIN) break;
            perror("mq_receive");
            break;
        }
        buffer[bytes] = '\0';
        printf("Received (priority %u): %s\n", priority, buffer);
    }
    // Output order: High priority task(10) -> Medium priority task(5) -> Low priority task(1)

    mq_close(mq);
    mq_unlink(QUEUE_NAME);
}
```

### 6.2 System V Message Queues (Legacy)

```
System V Message Queue:
  Predecessor to POSIX message queues
  -> msgget(), msgsnd(), msgrcv(), msgctl()
  -> Supports filtering by message type
  -> Legacy but available on many systems

  Key differences:
  - System V: Filter by message type (long)
  - POSIX: Sort by priority (unsigned int)
  - POSIX has a more refined API
  - POSIX supports asynchronous notification via mq_notify()
```

---

## 7. Semaphores and Synchronization

### 7.1 Semaphore Basics

```
Semaphore:
  A synchronization primitive to control access to shared resources

  Two types of semaphores:
  1. Binary semaphore (equivalent to a mutex):
     Value is 0 or 1
     -> Mutual exclusion (only one process can access)

  2. Counting semaphore:
     Value is N (> 0)
     -> Allows N concurrent accesses
     -> Resource pools (DB connection pool, thread pool)

  Operations:
  wait() (P operation, down, acquire):
    if (semaphore > 0):
        semaphore -= 1  # Acquire resource
    else:
        block()          # Block (wait)

  post() (V operation, up, release):
    semaphore += 1       # Release resource
    wakeup_one()         # Wake one waiting process

  Important: wait/post are atomic operations (guaranteed by the kernel)

  Producer-Consumer problem (solved with semaphores):
  +--------------------------------------+
  | empty = N  // Number of free slots   |
  | full = 0   // Number of data items   |
  | mutex = 1  // Mutual exclusion       |
  |                                      |
  | Producer:          Consumer:          |
  |   wait(empty)        wait(full)      |
  |   wait(mutex)        wait(mutex)     |
  |   produce()          consume()       |
  |   post(mutex)        post(mutex)     |
  |   post(full)         post(empty)     |
  +--------------------------------------+
```

### 7.2 Mutexes and Inter-Process Locking

```
Inter-process mutex:
  Place a pthread_mutex in shared memory to share
  a lock between processes

  Setup:
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  // Initialize the mutex variable in shared memory
  pthread_mutex_init(shared_mutex, &attr);

  File locks (flock / fcntl):
  -> Inter-process mutual exclusion via files
  -> Commonly used by databases (SQLite)

  flock() -- Advisory lock:
  LOCK_SH: Shared lock (multiple processes can hold simultaneously)
  LOCK_EX: Exclusive lock (one process only)
  LOCK_UN: Unlock

  fcntl() -- Record lock:
  -> Can lock only part of a file
  -> Byte-range locking
```

```python
# File lock (mutual exclusion) example in Python
import fcntl
import os
import time

class FileLock:
    """File-based inter-process mutual exclusion"""

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
                    raise TimeoutError(f"Lock acquisition timeout: {self.lockfile}")
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

# Usage
with FileLock("/tmp/my_app.lock"):
    print("Exclusive lock acquired. Executing critical section...")
    time.sleep(5)
    print("Done")
```

---

## 8. eventfd and Other Linux-Specific IPC

### 8.1 eventfd

```
eventfd (Linux 2.6.22+):
  A lightweight event notification mechanism
  -> Provides a 64-bit counter as a file descriptor
  -> Can be used in a semaphore-like manner
  -> Integrates with epoll/select
  -> Lighter than pipes (no kernel buffer needed)

  API:
  eventfd(initval, flags)  -> Returns an fd
  write(fd, &value, 8)     -> Adds to the counter
  read(fd, &value, 8)      -> Reads and resets the counter

  Flags:
  EFD_NONBLOCK:  Non-blocking
  EFD_SEMAPHORE: Semaphore mode (read decrements by 1)

  Use cases:
  - Event notification between threads
  - Shutdown notification in epoll-based event loops
  - Interrupt notification in KVM (virtualization)
```

### 8.2 D-Bus

```
D-Bus:
  The standard IPC for desktop Linux
  -> Communication through a message bus daemon
  -> Object-oriented message passing
  -> Type-safe interface definitions

  Architecture:
  +----------+  +----------+  +----------+
  | App A    |  | App B    |  | App C    |
  +----+-----+  +----+-----+  +----+-----+
       |             |             |
  +----+-------------+-------------+----+
  |         D-Bus Daemon                 |
  |  (dbus-daemon / dbus-broker)         |
  +--------------------------------------+

  Two buses:
  - System Bus: System-wide (one per system)
    -> NetworkManager, udev, systemd
  - Session Bus: Per user session (one per login)
    -> Desktop notifications, media player control

  Usage examples:
  # Send a desktop notification
  gdbus call --session \
    --dest=org.freedesktop.Notifications \
    --object-path=/org/freedesktop/Notifications \
    --method=org.freedesktop.Notifications.Notify \
    "MyApp" 0 "" "Title" "Body text" [] {} 5000

  # Get NetworkManager state
  gdbus call --system \
    --dest=org.freedesktop.NetworkManager \
    --object-path=/org/freedesktop/NetworkManager \
    --method=org.freedesktop.DBus.Properties.Get \
    "org.freedesktop.NetworkManager" "State"
```

---

## 9. IPC Selection Guide for Practice

### 9.1 Performance Optimization

```
IPC Performance Tuning:

  Pipes:
  -> Increase buffer size (fcntl F_SETPIPE_SZ)
  -> Zero-copy transfer with splice()
  -> Zero-copy from user space with vmsplice()

  Shared memory:
  -> Use hugepages (reduces TLB misses)
  -> Consider memory alignment (prevent cache line bouncing)
  -> Use lock-free data structures (CAS-based)

  Sockets:
  -> TCP_NODELAY (disable Nagle's algorithm, reduce latency)
  -> SO_SNDBUF / SO_RCVBUF (increase buffer sizes)
  -> SO_REUSEPORT (load balancing across cores)
  -> Reduce system calls with io_uring

  Message queues:
  -> Set message size appropriately (avoid over-allocation)
  -> Monitor queue depth (mq_getattr)

  Zero-copy techniques:
  +----------------------------------------------+
  | sendfile():   File -> Socket                  |
  | splice():     Pipe -> Socket / Pipe           |
  | vmsplice():   User space -> Pipe              |
  | mmap():       File -> User space              |
  | io_uring:     General async I/O (zero-copy)   |
  |                                               |
  | -> Eliminates copies between kernel and       |
  |    user space                                 |
  | -> Highly effective for large data transfers  |
  +----------------------------------------------+
```

### 9.2 Security Considerations

```
IPC Security:

  Pipes:
  -> Relatively safe since shared only via fork()
  -> Named pipes are controlled via file permissions

  Shared memory:
  -> Set permissions on /dev/shm
  -> Consider encryption for sensitive data
  -> Ensure cleanup on process termination

  Unix domain sockets:
  -> Set socket file permissions
  -> Verify client UID/GID/PID via SO_PEERCRED
  -> Privilege separation with systemd socket activation

  TCP sockets:
  -> TLS encryption (essential over a network)
  -> Restrict access via firewall
  -> Implement authentication and authorization

  Message queues:
  -> Set permissions on /dev/mqueue
  -> Limit message size (DoS prevention)

  Container environments:
  +----------------------------------------------+
  | Docker's default IPC settings:                |
  | - Each container has its own IPC namespace    |
  | - --ipc=host shares the host's IPC namespace  |
  | - --ipc=container:id shares with another      |
  |   container                                   |
  |                                               |
  | Kubernetes:                                   |
  | - Containers within a Pod share IPC namespace |
  | - IPC between Pods is typically network-based |
  +----------------------------------------------+
```

---

## 10. Practical Exercises

### Exercise 1: [Basic] -- Pipe Practice

```bash
# Examine the output at each stage of the following pipeline
ps aux | awk '{print $1}' | sort | uniq -c | sort -rn | head -5

# Communicate between two terminals using a named pipe
# Terminal 1:
mkfifo /tmp/chat
while true; do
    if read line < /tmp/chat; then
        echo "Received: $line"
    fi
done

# Terminal 2:
echo "Hello from terminal 2" > /tmp/chat
echo "Second message" > /tmp/chat
```

### Exercise 2: [Intermediate] -- Signal Handling

```python
import signal
import sys
import os
import time

def handler(signum, frame):
    sig_name = signal.Signals(signum).name
    print(f"\nReceived signal {sig_name}({signum}). Performing shutdown...")
    # Cleanup processing
    print("Deleting temporary files...")
    print("Closing connections...")
    sys.exit(0)

def reload_handler(signum, frame):
    print(f"\nReloading configuration (received SIGHUP)")

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)
signal.signal(signal.SIGHUP, reload_handler)

print(f"PID: {os.getpid()}")
print("Use Ctrl+C, kill -TERM, or kill -HUP to interact")

while True:
    print(".", end="", flush=True)
    time.sleep(1)
```

### Exercise 3: [Intermediate] -- Inter-Process Communication via Shared Memory

```python
# Using multiprocessing.shared_memory (Python 3.8+)
from multiprocessing import shared_memory, Process
import numpy as np
import time

def producer(shm_name):
    """Process that writes data to shared memory"""
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((10,), dtype=np.int64, buffer=shm.buf)

    for i in range(100):
        arr[i % 10] = i
        time.sleep(0.01)

    shm.close()

def consumer(shm_name):
    """Process that reads data from shared memory"""
    time.sleep(0.1)  # Wait for producer to start
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray((10,), dtype=np.int64, buffer=shm.buf)

    for _ in range(50):
        print(f"Current values: {arr.tolist()}")
        time.sleep(0.02)

    shm.close()

if __name__ == "__main__":
    # Create shared memory
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

### Exercise 4: [Advanced] -- Unix Domain Socket Chat System

```python
# Server (multi-client support)
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
    broadcast(f"[{clients[conn]} has joined]", exclude=conn)

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
    broadcast(f"[{name} has left]")

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

print(f"Chat server started: {SOCKET_PATH}")
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

### Q1: Why does Docker use Unix domain sockets?

Docker CLI and the Docker daemon communicate on the same machine, so Unix domain sockets are the optimal choice because they are faster than TCP and make authentication management straightforward. Access control is achieved through socket file permissions, and there is no network stack overhead. Additionally, SO_PEERCRED can verify the UID/GID of the connecting user, enabling a mechanism where only users belonging to the docker group are granted access. When you need to use the Docker API remotely, you can switch to a TCP socket (ports 2375/2376), though TLS configuration is required.

### Q2: What is the difference between pipes and message queues?

Pipes are **byte streams** (no delimiters). Message queues are **message-oriented** (with boundaries and priority support). Pipes are disposable (destroyed when the process exits), while message queues persist until explicitly deleted. Pipes are fundamentally for parent-child processes (anonymous pipes), though named pipes enable communication between unrelated processes. Message queues are designed from the ground up for communication between unrelated processes. In terms of performance, pipes are slightly faster.

### Q3: What is the best synchronization method for shared memory?

It depends on the use case. For simple mutual exclusion, an **inter-process mutex** (PTHREAD_PROCESS_SHARED) is the simplest approach. If you need to control the number of concurrent accesses, use **semaphores**. For high performance, consider **lock-free data structures** (Compare-and-Swap based). File locks (flock/fcntl) are easy to implement but perform slightly worse. In practice, start with a mutex and consider switching to lock-free approaches only after profiling reveals a problem.

### Q4: What is the difference between gRPC and REST?

gRPC is a remote procedure call framework based on Protocol Buffers over HTTP/2. It uses binary format for speed, supports bidirectional streaming, and is type-safe. REST (HTTP/JSON) uses text format that is human-readable, directly accessible from browsers, and benefits from a rich ecosystem of tools. gRPC is commonly used for internal microservice communication, while REST is the standard for public-facing APIs. However, with the advent of gRPC-Web, using gRPC from browsers has also become possible.

### Q5: How does io_uring affect IPC?

io_uring (Linux 5.1+) is a new asynchronous I/O interface that dramatically reduces system call overhead. By performing socket and pipe I/O through io_uring, IPC throughput improves in high-load environments. In particular, features such as batched send/recv, zero-copy sends, and multi-shot accept improve IPC performance for server applications.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying how things work.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world practice?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary

| IPC Method | Speed | Use Case |
|---------|------|------|
| Pipe | Medium | Shell command chaining, parent-child processes |
| Named Pipe | Medium | Simple communication between unrelated processes |
| Signal | Fast | Process control (terminate, reload, stop) |
| Message Queue | Medium | Message-unit communication, priority support |
| Shared Memory | Fastest | Large data sharing, high-performance IPC |
| Unix Domain Socket | Fast | Local service communication |
| TCP Socket | Medium-Slow | Network communication |
| eventfd | Fast | Lightweight event notification |
| D-Bus | Medium | Desktop application integration |

---

## Recommended Next Reads

---

## References
1. Kerrisk, M. "The Linux Programming Interface." No Starch Press, Ch.43-57, 2010.
2. Stevens, W. R. "UNIX Network Programming, Vol.2: IPC." Prentice Hall, 1998.
3. Stevens, W. R. & Rago, S. A. "Advanced Programming in the UNIX Environment." 3rd Ed, Addison-Wesley, 2013.
4. Love, R. "Linux System Programming." 2nd Ed, O'Reilly, Ch.9-10, 2013.
5. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Ch.30-33, 2018.
6. Corbet, J. et al. "Linux Device Drivers." 3rd Ed, O'Reilly, 2005.
7. Axboe, J. "Efficient IO with io_uring." Kernel Documentation, 2019.
8. Tanenbaum, A. S. & Bos, H. "Modern Operating Systems." 4th Ed, Ch.2, 2014.
