# I/O Scheduling

> The I/O scheduler reorders requests to the disk, minimizing head movement distance to improve throughput.

## Learning Objectives

- [ ] Understand the necessity of I/O scheduling
- [ ] Know the differences between major schedulers
- [ ] Understand I/O optimization in the SSD era
- [ ] Grasp the structure of the Linux block layer
- [ ] Understand asynchronous I/O technologies including io_uring
- [ ] Master practical I/O tuning techniques


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Major File System Implementations](./01-fs-implementations.md)

---

## 1. Why I/O Scheduling Is Necessary

### 1.1 HDD Access Characteristics

```
HDD access time components:

  Total access time = Seek time + Rotational latency + Transfer time

  Seek Time:
    Time for head movement
    -> 3-10ms (average), worst case 15ms+
    -> Adjacent track: 0.5ms-1ms
    -> Full stroke: 10ms-20ms
    <- The biggest bottleneck for I/O performance

  Rotational Latency:
    Wait time for the target sector to arrive under the head
    -> Depends on rotation speed
    -> 7200 RPM: average 4.17ms (half of 1 rotation = 8.33ms)
    -> 15000 RPM: average 2.0ms
    -> 5400 RPM: average 5.56ms

  Transfer Time:
    Actual data read/write
    -> Typically < 1ms (tens to hundreds of MB/s)
    -> Small proportion of total time

  Concrete calculation example:
  Random 4KB read on 7200RPM HDD:
    Seek: 8ms (average)
    Rotational latency: 4.17ms (average)
    Transfer: 0.01ms (4KB / 200MB/s)
    Total: ~ 12.18ms

  -> ~82 random I/Os per second = 82 IOPS
  -> Sequential reads can exceed 200MB/s

  Comparison: SSD random read:
    Latency: 0.05-0.1ms
    -> 10,000-100,000+ IOPS
    -> 100-1000x faster than HDD
```

### 1.2 Effect of Scheduling

```
Optimizing request order dramatically reduces seek time:

  Track positions on disk:
  0     50    100   150   200   250   300
  |-----|-----|-----|-----|-----|-----|

  Request queue (arrival order):
  Positions: 98, 183, 37, 122, 14, 124, 65, 67

  Current head position: 53

  * FCFS (First Come First Served, no optimization):
  53 -> 98 -> 183 -> 37 -> 122 -> 14 -> 124 -> 65 -> 67
  Movement: 45 + 85 + 146 + 85 + 108 + 110 + 59 + 2 = 640

  53-->98------>183
            <--37-->122
       <--14-->124
           <--65>67

  * SSTF (Shortest Seek Time First):
  53 -> 65 -> 67 -> 37 -> 14 -> 98 -> 122 -> 124 -> 183
  Movement: 12 + 2 + 30 + 23 + 84 + 24 + 2 + 59 = 236
  -> 63% reduction compared to FCFS!

  Problem: Starvation
  -> Requests far from the current head position may never be serviced

  * SCAN (Elevator Algorithm):
  Head moves in one direction while servicing, reverses at the end
  53 -> 37 -> 14 -> [0] -> 65 -> 67 -> 98 -> 122 -> 124 -> 183
  Movement: 16 + 23 + 14 + 65 + 2 + 31 + 24 + 2 + 59 = 236
  -> Prevents starvation

  * C-SCAN (Circular SCAN):
  Services in one direction only, jumps to opposite end at boundary
  -> More uniform wait times

  * LOOK / C-LOOK:
  Improved versions of SCAN/C-SCAN
  -> Reverses at the last request position instead of disk edge
  -> Reduces unnecessary movement

  Algorithm comparison:
  +----------+----------+----------+----------+
  | Algorithm| Movement | Starvation| Wait Time|
  +----------+----------+----------+----------+
  | FCFS     | Maximum  | None     | Uneven   |
  | SSTF     | Small    | Possible | Uneven   |
  | SCAN     | Medium   | None     | Somewhat |
  |          |          |          | even     |
  | C-SCAN   | Medium   | None     | Even     |
  | LOOK     | Small    | None     | Somewhat |
  |          |          |          | even     |
  | C-LOOK   | Small    | None     | Even     |
  +----------+----------+----------+----------+
```

### 1.3 Position of I/O Scheduling in the Stack

```
Position of the scheduler in the Linux I/O stack:

  Application
     | read() / write()
     v
  VFS (Virtual File System)
     |
     v
  File System (ext4, XFS, Btrfs)
     | Block I/O request generation
     v
  Page Cache
     | Cache hit -> completes here
     | Cache miss v
     v
  +-------------------------------------+
  | Block Layer                          |
  | +----------------------------------+|
  | | I/O Scheduler                    ||
  | | -> Request reordering & merging  ||
  | +----------------------------------+|
  | +----------------------------------+|
  | | Multi-Queue Block Layer          ||
  | | (blk-mq)                        ||
  | +----------------------------------+|
  +-------------------------------------+
     |
     v
  Device Driver
     |
     v
  Hardware (HDD / SSD / NVMe)
```

---

## 2. Linux I/O Schedulers

### 2.1 Legacy Single-Queue Schedulers (Pre-Kernel 4.x)

```
Legacy I/O schedulers (for reference):

  1. noop:
     No scheduling (FIFO)
     -> Only merging performed, no reordering
     -> For SSDs, virtual environments

  2. deadline:
     Assigns a deadline to each request
     -> Read: 500ms, Write: 5000ms
     -> Prioritizes requests that exceed their deadline
     -> Balances starvation prevention and responsiveness

  3. CFQ (Completely Fair Queuing):
     Creates a queue per process for fair dispatching
     -> Linux 2.6.18-4.x default
     -> For desktops
     -> Cannot leverage SSD performance due to single queue

  Problems with single queue:
  +----------------------------------------------+
  |                                              |
  |   CPU0 -+                                    |
  |   CPU1 -+-- Single request queue --> Device  |
  |   CPU2 -+     ^ Lock contention              |
  |   CPU3 -+                                    |
  |                                              |
  | -> Low scalability in multi-core environments|
  | -> Bottleneck with fast devices like NVMe    |
  +----------------------------------------------+
```

### 2.2 Multi-Queue Block Layer (blk-mq)

```
blk-mq (Multi-Queue Block Layer, Linux 3.13+):

  Design philosophy:
  -> Support multi-core CPUs + high-speed storage (NVMe)
  -> Place a software queue per CPU core
  -> Efficient mapping to hardware queues
  -> Significant reduction of lock contention

  +------------------------------------------------------+
  | blk-mq structure                                      |
  |                                                       |
  |  CPU0 -> [SW Queue 0]-+                               |
  |  CPU1 -> [SW Queue 1]-+-- [HW Queue 0] -> Device     |
  |  CPU2 -> [SW Queue 2]-+-- [HW Queue 1] -> Device     |
  |  CPU3 -> [SW Queue 3]-+-- [HW Queue N] -> Device     |
  |                                                       |
  |  SW Queue: CPU-local (no lock needed)                 |
  |  HW Queue: Corresponds to device hardware queues      |
  |  -> NVMe: up to 64K HW queues                        |
  |  -> SATA: typically 1 HW queue                        |
  +------------------------------------------------------+

  Benefits of blk-mq:
  - No lock contention with independent queues per CPU core
  - Fully utilizes NVMe parallelism
  - NUMA-node-aware placement
  - Low latency (polling mode support)

  Linux 5.0 and later:
  -> Legacy single-queue schedulers completely removed
  -> All devices migrated to blk-mq base
```

### 2.3 Current Linux I/O Schedulers

```
Current Linux I/O schedulers (blk-mq based):

  1. mq-deadline (Multi-Queue Deadline):
  +----------------------------------------------+
  | Overview:                                     |
  | - blk-mq version of the legacy deadline       |
  |   scheduler                                   |
  | - Assigns deadlines to requests               |
  | - Read priority (500ms), write (5000ms)       |
  |                                               |
  | Operation:                                    |
  | 1. Manages requests in 2 queues               |
  |    - Sort queue: by sector number (SCAN-like) |
  |    - Deadline queue: by expiration time        |
  | 2. Normally processes from sort queue          |
  |    (seek optimization)                        |
  | 3. Prioritizes requests past deadline          |
  | 4. Prioritizes reads over writes               |
  |    (improves interactivity)                   |
  |                                               |
  | Parameters:                                   |
  | /sys/block/sda/queue/iosched/                 |
  |   read_expire:     500  (ms, read deadline)   |
  |   write_expire:    5000 (ms, write deadline)  |
  |   writes_starved:  2    (read priority)       |
  |   fifo_batch:      16   (batch size)          |
  |   front_merges:    1    (front merge enabled) |
  |                                               |
  | Use cases: General HDD, DB servers,           |
  |            virtualization hosts                |
  | Best for: When latency guarantee is important |
  +----------------------------------------------+

  2. BFQ (Budget Fair Queueing):
  +----------------------------------------------+
  | Overview:                                     |
  | - Successor to CFQ (blk-mq based)            |
  | - Allocates I/O "budget" per process          |
  | - Fair distribution of bandwidth and latency  |
  |                                               |
  | Operation:                                    |
  | 1. Assigns a queue to each process            |
  | 2. Sets "budget" (serviceable sector count)   |
  | 3. Budget consumed -> switch to next process  |
  | 4. Auto-detects interactive processes and     |
  |    prioritizes them                           |
  | 5. Priority control via weighting             |
  |                                               |
  | Parameters:                                   |
  | /sys/block/sda/queue/iosched/                 |
  |   slice_idle:     8 (ms, idle wait time)      |
  |   low_latency:    1 (low latency mode)        |
  |   timeout_sync:   125 (ms, sync timeout)      |
  |   max_budget:     0 (0=auto, sector count)    |
  |   strict_guarantees: 0                        |
  |                                               |
  | cgroup integration:                           |
  | -> Integration with I/O controller            |
  | -> I/O bandwidth limitation per container     |
  | -> Proportional allocation (weight-based)     |
  |                                               |
  | Use cases: Desktop, multimedia                |
  | Best for: Interactivity-critical with slow    |
  |           storage                             |
  |                                               |
  | Note: CPU overhead is high                    |
  | -> For fast NVMe, none is more appropriate    |
  +----------------------------------------------+

  3. none (noop):
  +----------------------------------------------+
  | Overview:                                     |
  | - No scheduling                               |
  | - No reordering of requests                   |
  | - Only merging (combining adjacent requests)  |
  |                                               |
  | Operation:                                    |
  | -> Processes requests as-is in FIFO order     |
  | -> Merges adjacent block requests             |
  | -> Minimal CPU overhead                       |
  |                                               |
  | Use cases:                                    |
  | - SSD / NVMe (no physical seeks)             |
  | - Virtual machines (host OS already schedules)|
  | - Software RAID (RAID controller optimizes)   |
  | - Hardware RAID (RAID card optimizes)         |
  |                                               |
  | -> For fast devices, scheduler CPU overhead   |
  |    can become a bottleneck                    |
  | -> none achieves the highest throughput       |
  +----------------------------------------------+
```

### 2.4 Configuring and Checking Schedulers

```bash
# Check current scheduler
cat /sys/block/sda/queue/scheduler
# [mq-deadline] bfq none
# -> The one in [] is the current scheduler

# Check scheduler for all block devices
for dev in /sys/block/*/queue/scheduler; do
  echo "$(dirname $(dirname $dev) | xargs basename): $(cat $dev)"
done
# sda: [mq-deadline] bfq none
# nvme0n1: [none] mq-deadline bfq

# Change scheduler (temporary)
echo "mq-deadline" | sudo tee /sys/block/sda/queue/scheduler
echo "none" | sudo tee /sys/block/nvme0n1/queue/scheduler

# Change scheduler (persistent, udev rules)
# /etc/udev/rules.d/60-ioschedulers.rules
# For HDDs
ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/rotational}=="1", \
  ATTR{queue/scheduler}="mq-deadline"

# For SSDs
ACTION=="add|change", KERNEL=="sd[a-z]", ATTR{queue/rotational}=="0", \
  ATTR{queue/scheduler}="none"

# For NVMe
ACTION=="add|change", KERNEL=="nvme[0-9]*", \
  ATTR{queue/scheduler}="none"

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Check if device is rotational
cat /sys/block/sda/queue/rotational
# 1 = HDD (rotational disk)
# 0 = SSD (non-rotational)

# Check scheduler parameters
ls /sys/block/sda/queue/iosched/
# -> read_expire, write_expire, writes_starved, ...

# Change parameters
echo 300 | sudo tee /sys/block/sda/queue/iosched/read_expire
echo 3000 | sudo tee /sys/block/sda/queue/iosched/write_expire
```

### 2.5 I/O Scheduling in the SSD Era

```
SSD/NVMe characteristics and scheduling:

  HDD vs SSD I/O characteristics:
  +------------------+------------------+------------------+
  | Item             | HDD              | SSD (NVMe)       |
  +------------------+------------------+------------------+
  | Random read      | 100-200 IOPS     | 100K-1M+ IOPS   |
  | Sequential read  | 100-200 MB/s     | 3-7 GB/s         |
  | Latency          | 5-15 ms          | 0.02-0.1 ms      |
  | Parallelism      | 1 (single head)  | Up to 4M(64Kx64K)|
  | Seek impact      | Large            | None             |
  | Recommended      | mq-deadline      | none             |
  | scheduler        |                  |                  |
  +------------------+------------------+------------------+

  NVMe parallel I/O structure:
  +------------------------------------------+
  | NVMe structure                            |
  |                                           |
  | NVMe Controller                           |
  | +-- Submission Queue 0 -> Completion Queue 0|
  | +-- Submission Queue 1 -> Completion Queue 1|
  | +-- Submission Queue 2 -> Completion Queue 2|
  | +-- ...                                   |
  | +-- SQ 65535 ---------> CQ 65535          |
  |                                           |
  | Up to 65536 commands per queue            |
  | -> Theoretical max: 64K x 64K = ~4B      |
  |    parallel I/Os                          |
  |                                           |
  | Actual usage:                             |
  | -> 1 queue pair per CPU core              |
  | -> 8-core CPU -> 8 queue pairs            |
  | -> ~1024 entries per queue                |
  +------------------------------------------+

  Why none is optimal for SSDs:
  1. No physical seeks -> reordering is meaningless
  2. Ultra-high parallelism -> queue order doesn't affect performance
  3. Scheduler CPU overhead becomes relatively large
  4. Device performs its own scheduling (FTL)
  5. Scheduler delay offsets the device's low latency

  Exceptions:
  - SATA SSDs (queue depth limited to 32): mq-deadline may be effective
  - Using BFQ on desktop:
    -> Prevents heavy background I/O from hurting interactivity
    -> Example: Application launch speed during large file copy
```

---

## 3. I/O Optimization Techniques

### 3.1 Page Cache

```
Page Cache:
  Mechanism to cache read disk data in memory
  -> Subsequent accesses read from memory (ultra-fast)
  -> Linux uses most of memory for page cache

  Operating principle:
  +------------------------------------------+
  | read() processing flow:                   |
  |                                           |
  | 1. Search page cache                      |
  |    +-- Hit -> Return directly from memory |
  |    |          (us)                        |
  |    +-- Miss -> Read from disk (ms)        |
  |               +-- Store in cache          |
  |                  +-- Return data          |
  |                                           |
  | write() processing flow:                  |
  |                                           |
  | 1. Write to page cache                    |
  | 2. Mark page as "dirty"                   |
  | 3. write() returns immediately            |
  | 4. Kernel thread (pdflush/writeback)      |
  |    writes to disk in background           |
  +------------------------------------------+

  Page cache management:
  +------------------------------------------+
  | Memory usage (4GB RAM example)            |
  |                                          |
  | +------------------------------------+   |
  | | Application usage: 1.5GB           |   |
  | +------------------------------------+   |
  | | Page cache: 2.0GB                  |   |
  | +------------------------------------+   |
  | | Kernel/reserved: 0.5GB             |   |
  | +------------------------------------+   |
  |                                          |
  | -> Even if "free" command shows low      |
  |    Available, it's fine if page cache    |
  |    is large                              |
  | -> Cache is automatically released       |
  |    when apps request memory              |
  +------------------------------------------+
```

```bash
# Check page cache status
free -h
# total   used   free   shared  buff/cache  available
# 16Gi   4.5Gi  1.2Gi  256Mi   10.3Gi       11.0Gi
# -> buff/cache of 10.3GB is page cache + buffers
# -> available of 11.0GB is actually usable memory

# More detailed information
cat /proc/meminfo | grep -E "^(MemTotal|MemFree|Buffers|Cached|Dirty|Writeback)"
# MemTotal:       16384000 kB
# MemFree:         1200000 kB
# Buffers:          256000 kB
# Cached:         10000000 kB   <- Page cache
# Dirty:             32000 kB   <- Data not yet written to disk
# Writeback:             0 kB   <- Data being written

# Clear page cache (for testing/benchmarking)
# Warning: Do not run in production!
sudo sync                                    # Write dirty pages
echo 1 | sudo tee /proc/sys/vm/drop_caches  # Clear page cache
echo 2 | sudo tee /proc/sys/vm/drop_caches  # Clear dentry/inode cache
echo 3 | sudo tee /proc/sys/vm/drop_caches  # Clear both

# Dirty page settings (control write-out timing)
# Dirty ratio (proportion of dirty pages to total memory)
cat /proc/sys/vm/dirty_ratio           # Default: 20
cat /proc/sys/vm/dirty_background_ratio # Default: 10
cat /proc/sys/vm/dirty_expire_centisecs # Default: 3000 (30 seconds)
cat /proc/sys/vm/dirty_writeback_centisecs # Default: 500 (5 seconds)

# dirty_ratio: write() blocks when this ratio is exceeded
# dirty_background_ratio: background write-out starts when this ratio is exceeded

# DB server settings (write out more frequently)
sudo sysctl -w vm.dirty_ratio=5
sudo sysctl -w vm.dirty_background_ratio=2
sudo sysctl -w vm.dirty_expire_centisecs=500

# Large memory server settings (specify in bytes)
sudo sysctl -w vm.dirty_bytes=268435456           # 256MB
sudo sysctl -w vm.dirty_background_bytes=67108864 # 64MB
```

### 3.2 Read-ahead

```
Read-ahead:
  Detects sequential reads and pre-reads data
  -> Prepares data before the application requests it
  -> Dramatically speeds up sequential reads

  Operating principle:
  +------------------------------------------+
  | Application read pattern:                 |
  |                                          |
  | Time 1: read(offset=0, size=4KB)         |
  | Time 2: read(offset=4KB, size=4KB)       |
  | Time 3: read(offset=8KB, size=4KB)       |
  |   ^ Sequential pattern detected!         |
  |                                          |
  | Kernel read-ahead:                        |
  | -> Pre-read 128KB from offset=12KB       |
  | -> Next read() from app hits cache       |
  |                                          |
  | Adaptive read-ahead:                      |
  | - Dynamically adjusts read-ahead size    |
  | - Initially: small size                  |
  | - After confirming sequential: gradually  |
  |   increase                               |
  | - Maximum: up to readahead_kb value      |
  +------------------------------------------+
```

```bash
# Check read-ahead size
cat /sys/block/sda/queue/read_ahead_kb
# Default: 128 (128KB)

# Change read-ahead size
echo 256 | sudo tee /sys/block/sda/queue/read_ahead_kb   # 256KB
echo 2048 | sudo tee /sys/block/sda/queue/read_ahead_kb  # 2MB

# Recommended settings:
# HDD: 256-1024KB (when sequential reads are frequent)
# SSD: 128-256KB (default is sufficient)
# RAID: 1024-4096KB (match stripe size)
# Database: smaller (frequent random I/O)

# Setting with blockdev
sudo blockdev --getra /dev/sda    # Get read-ahead size (in sectors)
sudo blockdev --setra 512 /dev/sda # 256KB (512 sectors x 512B)

# Per-application read-ahead control
# posix_fadvise() system call
# POSIX_FADV_SEQUENTIAL: Declare sequential access
# POSIX_FADV_RANDOM:     Declare random access
# POSIX_FADV_WILLNEED:   Will access in near future
# POSIX_FADV_DONTNEED:   No longer need (cache release hint)
```

### 3.3 io_uring (Linux 5.1+)

```
io_uring: High-performance asynchronous I/O interface

  Comparison with traditional async I/O methods:
  +--------------+------------------------------------+
  | Method       | Characteristics                     |
  +--------------+------------------------------------+
  | Synchronous  | read()/write() blocks              |
  | I/O          | Simple but low parallelism         |
  | (blocking)   |                                    |
  +--------------+------------------------------------+
  | select/poll  | Check FD readiness                 |
  |              | O(n) overhead proportional to FD    |
  |              | count                              |
  +--------------+------------------------------------+
  | epoll        | Event-driven                       |
  |              | O(1) to get ready FDs              |
  |              | But read/write itself is synchronous|
  +--------------+------------------------------------+
  | aio          | Kernel async I/O                   |
  | (Linux AIO)  | Only supports Direct I/O           |
  |              | Buffered I/O not supported         |
  |              | Complex API                        |
  +--------------+------------------------------------+
  | io_uring     | Fully async I/O                    |
  | (Linux 5.1+) | Minimal system call overhead       |
  |              | Buffered I/O supported             |
  |              | Zero-copy support                  |
  |              | Polling mode support               |
  |              | -> Highest performance I/O          |
  |              |    interface                       |
  +--------------+------------------------------------+

  How io_uring works:
  +------------------------------------------------------+
  |                                                       |
  |  User space          Kernel space                     |
  |                                                       |
  |  +--------------+   +--------------+                  |
  |  | Submission   |   |              |                  |
  |  | Queue (SQ)   |-->|  I/O         |                  |
  |  |  Submit      |   |  Processing  |                  |
  |  |  requests    |   |  Engine      |                  |
  |  +--------------+   +------+-------+                  |
  |                            |                          |
  |  +--------------+          |                          |
  |  | Completion   |<---------+                          |
  |  | Queue (CQ)   |                                    |
  |  |  Receive     |                                    |
  |  |  completions |                                    |
  |  +--------------+                                    |
  |                                                       |
  |  SQ and CQ are shared memory between kernel and      |
  |  user space                                           |
  |  -> Submit requests and get results without system    |
  |     calls                                             |
  |  -> io_uring_enter() only when needed (not needed     |
  |     in polling mode)                                  |
  +------------------------------------------------------+

  io_uring features (by Linux version):
  5.1:  Basic read/write
  5.4:  Network I/O (accept, connect, recv, send)
  5.5:  splice, tee
  5.6:  Fixed buffer registration, IO polling optimization
  5.7:  Linked operations, timeouts
  5.10: Restricted mode, permission control
  5.11: shutdown, renameat, unlinkat
  5.12: mkdirat, symlinkat
  5.15: sendmsg_zc (zero-copy send)
  6.0:  send_zc, recv_zc
  6.1:  futex operations
```

```c
// Basic io_uring usage example (C language)
#include <liburing.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

int main() {
    struct io_uring ring;
    struct io_uring_sqe *sqe;
    struct io_uring_cqe *cqe;
    char buf[4096];
    int fd;

    // Initialize io_uring (queue depth 128)
    io_uring_queue_init(128, &ring, 0);

    // Open file
    fd = open("test.txt", O_RDONLY);

    // Get a Submission Queue Entry
    sqe = io_uring_get_sqe(&ring);

    // Prepare a read request
    io_uring_prep_read(sqe, fd, buf, sizeof(buf), 0);

    // Submit the request
    io_uring_submit(&ring);

    // Wait for completion
    io_uring_wait_cqe(&ring, &cqe);

    printf("Read %d bytes\n", cqe->res);

    // Consume the completion entry
    io_uring_cqe_seen(&ring, cqe);

    // Cleanup
    close(fd);
    io_uring_queue_exit(&ring);
    return 0;
}

// Compile: gcc -o io_uring_example io_uring_example.c -luring
```

### 3.4 Direct I/O

```
Direct I/O:
  Access disk directly bypassing the page cache

  Normal I/O (Buffered I/O):
  App -> Page Cache -> Disk
  -> Kernel manages cache
  -> Optimal for most applications

  Direct I/O:
  App -> Disk (bypassing page cache)
  -> Application manages its own cache
  -> Databases are the primary users

  +--------------------------------------------+
  | Buffered I/O:                               |
  | App -> [Page Cache] -> Disk                 |
  |       Accelerated by cache                  |
  |       Double buffering (app + OS)           |
  |                                             |
  | Direct I/O:                                 |
  | App -> -> -> -> -> -> Disk                  |
  |       No cache                              |
  |       Application manages its own cache     |
  +--------------------------------------------+

  Requirements:
  - Specify O_DIRECT flag to open()
  - Buffer address must be aligned to block size
  - Read/write size must be multiple of block size
  - File offset must be multiple of block size

  Primary users:
  - MySQL InnoDB (innodb_flush_method = O_DIRECT)
  - PostgreSQL (limited effectiveness, usually not used)
  - Oracle Database
  - QEMU/KVM (virtual disk I/O)

  Configuration examples:
  # MySQL
  [mysqld]
  innodb_flush_method = O_DIRECT

  # QEMU
  qemu-system-x86_64 -drive file=disk.img,cache=none
  # cache=none: Uses O_DIRECT
  # cache=writeback: Uses page cache
  # cache=writethrough: Synchronous writes
```

### 3.5 mmap I/O

```
mmap (Memory-Mapped I/O):
  Map a file into the process address space

  +--------------------------------------------+
  | read/write vs mmap:                         |
  |                                             |
  | read():                                     |
  | 1. Issue system call                        |
  | 2. Copy data to kernel space                |
  | 3. Copy data to user space                  |
  | -> 2 data copies                            |
  |                                             |
  | mmap():                                     |
  | 1. Set up page table (first time only)      |
  | 2. Page fault on access                     |
  | 3. Directly map page cache pages            |
  | -> No data copy                             |
  +--------------------------------------------+

  Advantages of mmap:
  - No data copy required (zero-copy)
  - Efficient random access
  - Shareable across processes (MAP_SHARED)
  - Executable file loading

  Disadvantages of mmap:
  - Page fault overhead
  - Inefficient when accessing only part of a large file
  - Error handling is difficult (SIGBUS)
  - TLB miss overhead

  Use cases:
  - Databases (LMDB, SQLite mmap mode)
  - Executable file loading (ELF text segment)
  - Shared memory (IPC)
  - Configuration file reading
```

### 3.6 I/O Priority and cgroup

```
I/O priority control:

  ionice command:
  -> Set the I/O scheduling class and priority of a process

  Classes:
  1 (Realtime):  Highest priority. Risk of starvation
  2 (Best-effort): Default. Priority 0-7 (0 is highest)
  3 (Idle):       Processed only when no other I/O exists

  Usage examples:
  # Run backup at low priority
  ionice -c 3 rsync -a /data /backup/

  # Run database at high priority
  ionice -c 1 -n 0 mysqld

  # Check current I/O priority
  ionice -p $(pgrep -f mysqld)

  # Change priority of a running process
  ionice -c 2 -n 4 -p 1234

cgroup v2 I/O control:
  -> I/O limitation per container/process group

  # I/O bandwidth limit
  # cgroup v2 (systemd)
  systemctl set-property myservice.service IOWriteBandwidthMax="/dev/sda 50M"
  systemctl set-property myservice.service IOReadBandwidthMax="/dev/sda 100M"

  # I/O weight (proportional allocation)
  systemctl set-property myservice.service IOWeight=100
  # Range: 1-10000, default: 100

  # Docker I/O limits
  docker run --device-write-bps /dev/sda:50mb \
             --device-read-bps /dev/sda:100mb \
             --blkio-weight 500 \
             myapp

  # Kubernetes I/O limits (requires cgroup v2)
  # Controlled via Guaranteed QoS class
```

---

## 4. I/O Monitoring and Troubleshooting

### 4.1 I/O Monitoring Tools

```bash
# === iostat: Device-level I/O statistics ===
iostat -x 1 5     # Extended stats, 1-second interval, 5 times
# Device  r/s   w/s   rkB/s  wkB/s  rrqm/s  wrqm/s  %util  await
# sda     50.0  30.0  200.0  120.0  5.0     10.0    45.0   8.50

# Key metrics:
# r/s, w/s:      Read/write IOPS
# rkB/s, wkB/s:  Read/write throughput
# await:         Average I/O wait time (ms)
# %util:         Device utilization (near 100% means saturated)
# avgqu-sz:      Average queue length
# svctm:         Average service time (deprecated, inaccurate)

# === iotop: Process-level I/O statistics ===
sudo iotop -o     # Show only processes performing I/O
sudo iotop -b     # Batch mode (for scripts)
sudo iotop -a     # Cumulative display

# === blktrace: Block-level detailed trace ===
# Start trace
sudo blktrace -d /dev/sda -o trace

# Analyze trace
blkparse -i trace -d output.bin

# BPF-based tracing (lighter weight)
sudo biosnoop-bpfcc             # Display I/O latency
sudo biotop-bpfcc               # Top display of I/O
sudo biolatency-bpfcc           # I/O latency histogram

# === pidstat: Per-process I/O statistics ===
pidstat -d 1      # I/O stats, 1-second interval
# PID   kB_rd/s  kB_wr/s  kB_ccwr/s  Command
# 1234  500.0    200.0    0.0        mysqld

# === vmstat: System-wide I/O overview ===
vmstat 1
# bi: Block input (reads, blocks/s)
# bo: Block output (writes, blocks/s)
# wa: I/O wait time percentage (%)
```

### 4.2 Diagnosing I/O Performance Problems

```
I/O performance problem diagnosis flow:

  1. Symptom check:
  +------------------------------------------+
  | $ iostat -x 1                             |
  |                                           |
  | %util > 90%                               |
  | -> Device is saturated                    |
  |                                           |
  | await > 50ms (HDD) / > 5ms (SSD)         |
  | -> High I/O latency                       |
  |                                           |
  | avgqu-sz > 10                             |
  | -> Long queue (processing can't keep up)  |
  +------------------------------------------+

  2. Identify the cause:
  +------------------------------------------+
  | $ iotop -o                                |
  | -> Identify processes performing heavy I/O|
  |                                           |
  | $ sudo biosnoop-bpfcc                     |
  | -> Details of individual I/O requests     |
  |                                           |
  | $ cat /proc/<pid>/io                      |
  | -> I/O statistics for a specific process  |
  |   rchar: Read request byte count          |
  |   wchar: Write request byte count         |
  |   syscr: read system call count           |
  |   syscw: write system call count          |
  |   read_bytes: Actual disk reads           |
  |   write_bytes: Actual disk writes         |
  +------------------------------------------+

  3. Remediation:
  +------------------------------------------+
  | Heavy random I/O:                         |
  | -> Upgrade to SSD/NVMe                    |
  | -> Add memory (expand page cache)         |
  | -> Optimize application access patterns   |
  |                                           |
  | Slow sequential I/O:                      |
  | -> Increase read_ahead_kb                 |
  | -> Optimize stripe size (RAID)            |
  | -> Adjust block size                      |
  |                                           |
  | Write backlog:                            |
  | -> Adjust dirty_ratio / dirty_bytes       |
  | -> Check journal size                     |
  | -> Check fsync frequency                  |
  |                                           |
  | Specific process monopolizing I/O:        |
  | -> Lower priority with ionice             |
  | -> Limit bandwidth with cgroup            |
  | -> Use BFQ scheduler                      |
  +------------------------------------------+
```

### 4.3 I/O Benchmarks

```bash
# === fio: Standard storage benchmark tool ===

# Sequential read
fio --name=seq_read \
    --rw=read \
    --bs=1M \
    --size=4G \
    --numjobs=1 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=64

# Random read (4K, simulating database workload)
fio --name=rand_read_4k \
    --rw=randread \
    --bs=4k \
    --size=4G \
    --numjobs=8 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=32 \
    --group_reporting

# Random write (4K)
fio --name=rand_write_4k \
    --rw=randwrite \
    --bs=4k \
    --size=4G \
    --numjobs=8 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=32 \
    --group_reporting

# Mixed workload (70% read / 30% write)
fio --name=mixed \
    --rw=randrw \
    --rwmixread=70 \
    --bs=4k \
    --size=4G \
    --numjobs=8 \
    --runtime=60 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=32 \
    --group_reporting

# Using io_uring engine (Linux 5.1+)
fio --name=io_uring_test \
    --rw=randread \
    --bs=4k \
    --size=4G \
    --numjobs=4 \
    --runtime=60 \
    --time_based \
    --ioengine=io_uring \
    --direct=1 \
    --iodepth=128 \
    --fixedbufs=1 \
    --registerfiles=1 \
    --sqthread_poll=1

# Check latency distribution
fio --name=latency \
    --rw=randread \
    --bs=4k \
    --size=1G \
    --numjobs=1 \
    --runtime=30 \
    --time_based \
    --ioengine=libaio \
    --direct=1 \
    --iodepth=1 \
    --lat_percentiles=1 \
    --percentile_list=50:90:95:99:99.9:99.99

# How to read results:
# IOPS:     I/O operations per second
# BW:       Bandwidth (throughput)
# lat:      Latency (avg, min, max, percentiles)
# clat:     Completion latency
# slat:     Submission latency
```

---

## 5. Advanced I/O Technologies

### 5.1 Zero-Copy

```
Zero-Copy:
  Eliminate unnecessary data copies for efficiency

  Traditional file transfer (before sendfile):
  +------------------------------------------+
  | read(fd, buf, size):                      |
  | Disk -> Kernel buffer -> User buffer      |
  |                                           |
  | write(sockfd, buf, size):                 |
  | User buffer -> Kernel buffer -> NIC       |
  |                                           |
  | -> 4 copies, 2 context switches           |
  +------------------------------------------+

  Zero-copy with sendfile():
  +------------------------------------------+
  | sendfile(sockfd, fd, offset, size):       |
  | Disk -> Kernel buffer -> NIC              |
  |                                           |
  | -> 2 copies (DMA), no user space transit  |
  | -> 1 system call                          |
  +------------------------------------------+

  splice() / tee():
  -> Zero-copy data transfer using pipes
  -> Transfer page references within kernel (no data copy)

  io_uring zero-copy send:
  -> Complete zero-copy with send_zc operation
  -> Kernel directly references user buffer via page pinning

  Usage examples:
  - Nginx: sendfile on; (static file serving)
  - Kafka: Zero-copy message delivery
  - Video streaming servers
```

### 5.2 I/O Polling

```
I/O Polling:
  CPU polls for I/O completion instead of interrupts

  Interrupt method:
  CPU --- Other work ---- <-Interrupt-- Process
  -> Context switch overhead
  -> Interrupt processing becomes excessive at high IOPS

  Polling method:
  CPU --- Polling --- Completion detected --- Process
  -> No context switch
  -> Minimal latency
  -> CPU usage approaches 100%

  io_uring polling modes:
  +------------------------------------------+
  | IORING_SETUP_IOPOLL:                      |
  | -> Polls for device completion            |
  | -> Most effective with NVMe               |
  | -> Requires Direct I/O                    |
  |                                           |
  | IORING_SETUP_SQPOLL:                      |
  | -> Kernel thread polls the SQ             |
  | -> App issues I/O without system calls    |
  | -> Entirely in user space                 |
  |                                           |
  | Combining both:                           |
  | -> Maximum I/O performance (but high CPU) |
  +------------------------------------------+

  NVMe polling mode:
  # Enable via kernel parameter
  # /sys/block/nvme0n1/queue/io_poll
  echo 1 | sudo tee /sys/block/nvme0n1/queue/io_poll

  # Set polling delay
  echo 0 | sudo tee /sys/block/nvme0n1/queue/io_poll_delay
  # -1: Disabled, 0: Poll immediately, >0: Delay then poll (ns)
```

### 5.3 I/O Barriers and Flushes

```
Write barriers and data persistence:

  Problem:
  -> Disks have write caches
  -> Even if OS writes to disk, data may remain in cache
  -> Power loss can lose data in cache
  -> Journal integrity can be corrupted

  Solutions:
  1. Write Barrier:
     -> Guarantees writes before barrier have reached disk
     -> Writes after barrier execute after those before barrier

  2. FUA (Force Unit Access):
     -> Specific writes bypass disk cache and write directly
     -> Supported on NVMe/SAS

  3. fsync() / fdatasync():
     -> Persist file data to disk
     -> fsync: data + metadata
     -> fdatasync: data + only necessary metadata

  4. sync():
     -> Write all dirty pages from all files to disk

  +------------------------------------------+
  | Persistence guarantee levels:             |
  |                                          |
  | write()        : Up to page cache        |
  |                  -> Data loss possible    |
  |                     on power failure      |
  |                                          |
  | fdatasync()    : Up to disk cache        |
  |                  -> Safe with BBU         |
  |                                          |
  | fsync()        : To disk (incl. metadata)|
  |                  -> Most secure           |
  |                                          |
  | O_SYNC         : fsync on every write()  |
  |                  -> Large performance hit |
  |                                          |
  | O_DSYNC        : fdatasync on every      |
  |                  write()                  |
  |                  -> Faster than O_SYNC    |
  +------------------------------------------+

  Barrier control:
  # Barrier enabled (default, recommended)
  mount -o barrier=1 /dev/sda1 /mnt

  # Barrier disabled (only for RAID controllers with BBU)
  mount -o barrier=0 /dev/sda1 /mnt
  # -> BBU (Battery Backup Unit) protects cache

  # Check disk write cache
  sudo hdparm -W /dev/sda
  # /dev/sda: write-caching = 1 (on)

  # Disable write cache (safety-focused)
  sudo hdparm -W 0 /dev/sda
```

---

## Practical Exercises

### Exercise 1: [Basic] -- Check and Change I/O Schedulers

```bash
# Check current scheduler
cat /sys/block/sda/queue/scheduler

# Check all devices' schedulers
for dev in /sys/block/*/queue/scheduler; do
  echo "$(dirname $(dirname $dev) | xargs basename): $(cat $dev)"
done

# Change scheduler (temporary)
echo "bfq" | sudo tee /sys/block/sda/queue/scheduler
cat /sys/block/sda/queue/scheduler

# Check scheduler parameters
ls /sys/block/sda/queue/iosched/
for param in /sys/block/sda/queue/iosched/*; do
  echo "$(basename $param): $(cat $param)"
done

# Check device type
for dev in /sys/block/*/queue/rotational; do
  name=$(dirname $(dirname $dev) | xargs basename)
  type=$(cat $dev)
  echo "$name: $([ $type -eq 1 ] && echo 'HDD' || echo 'SSD')"
done
```

### Exercise 2: [Intermediate] -- Measure Page Cache Effectiveness

```bash
# Create a large test file
dd if=/dev/urandom of=/tmp/testfile bs=1M count=512

# Clear cache
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

# First read (from disk)
echo "=== First read (from disk) ==="
time dd if=/tmp/testfile of=/dev/null bs=1M
# -> Takes several seconds

# Second read (from page cache)
echo "=== Second read (from cache) ==="
time dd if=/tmp/testfile of=/dev/null bs=1M
# -> Under 1 second

# Check cache status
cat /proc/meminfo | grep -E "^(Cached|Buffers)"

# Check cache status for a specific file (fincore, Linux 4.2+)
fincore /tmp/testfile
# -> RES: Size cached
# -> PAGES: Number of cached pages

# Cleanup
rm /tmp/testfile
```

### Exercise 3: [Advanced] -- I/O Performance Analysis

```bash
# Monitor I/O situation with iostat
# Terminal 1: Generate I/O load
fio --name=load --rw=randrw --bs=4k --size=1G \
    --numjobs=4 --runtime=60 --time_based \
    --ioengine=libaio --direct=1 --iodepth=32

# Terminal 2: Monitoring
iostat -x 1 | tee /tmp/iostat.log

# Analyze key metrics
# High await -> I/O latency problem
# %util near 100% -> Device saturated
# Large avgqu-sz -> Long queue
# rrqm/s, wrqm/s -> I/O merge effectiveness

# Process-level analysis with iotop
sudo iotop -o -d 1

# Detailed analysis with BPF tools (bcc-tools package)
# I/O latency histogram
sudo biolatency-bpfcc 10 1
# -> Check latency distribution

# Details of individual I/O requests
sudo biosnoop-bpfcc
# TIME     COMM     PID  DISK  T  SECTOR  BYTES  LAT(ms)

# I/O pattern visualization
sudo bitesize-bpfcc
# -> Check I/O size distribution
```

### Exercise 4: [Advanced] -- Scheduler Performance Comparison

```bash
# Benchmark comparison across schedulers
for sched in mq-deadline bfq none; do
  echo "=== Scheduler: $sched ==="
  echo "$sched" | sudo tee /sys/block/sda/queue/scheduler

  fio --name=${sched}_test \
      --rw=randread --bs=4k --size=1G \
      --numjobs=4 --runtime=30 --time_based \
      --ioengine=libaio --direct=1 --iodepth=32 \
      --group_reporting \
      --output-format=terse \
    | awk -F';' '{print "IOPS:", $8, "BW:", $7, "lat:", $40}'

  echo ""
done
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Check configuration file path and format |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Check execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read stack traces to identify the location of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Stepwise verification**: Verify hypotheses using log output and debuggers
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator to log function inputs and outputs"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Problems

When performance problems occur, follow this diagnostic procedure:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Examine connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

Below is a summary of decision criteria for technology selection.

| Criterion | When prioritized | When acceptable to compromise |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin screens, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Growing services | Internal tools, fixed users |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+-----------------------------------------------------+
|           Architecture Selection Flow                |
+-----------------------------------------------------+
|                                                      |
|  (1) Team size?                                      |
|    +-- Small (1-5 people) -> Monolith                |
|    +-- Large (10+ people) -> Go to (2)               |
|                                                      |
|  (2) Deployment frequency?                           |
|    +-- Weekly or less -> Monolith + module separation |
|    +-- Daily/multiple -> Go to (3)                   |
|                                                      |
|  (3) Team independence?                              |
|    +-- High -> Microservices                         |
|    +-- Medium -> Modular monolith                    |
|                                                      |
+-----------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A method that is fast in the short term may become technical debt long-term
- Conversely, excessive design incurs high short-term costs and delays projects

**2. Consistency vs Flexibility**
- A unified technology stack has low learning costs
- Diverse technology adoption enables best-fit solutions but increases operational costs

**3. Level of Abstraction**
- High abstraction offers high reusability but can make debugging difficult
- Low abstraction is intuitive but prone to code duplication

```python
# Design decision recording template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe background and challenges"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "+" if c['type'] == 'positive' else "!"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```
---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What mistakes do beginners commonly make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before proceeding to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Scheduler | Characteristics | Use Cases |
|------------|------|------|
| mq-deadline | Deadline guarantee, starvation prevention | HDD, DB servers |
| BFQ | Fairness-focused, interactivity priority | Desktop, multimedia |
| none | No scheduling, minimal overhead | SSD/NVMe, virtual machines |

| I/O Technology | Characteristics | Use Cases |
|---------|------|------|
| Page cache | Memory cache of read data | General |
| Read-ahead | Pre-reading for sequential reads | Log processing, streaming |
| io_uring | High-performance async I/O | High-IOPS applications |
| Direct I/O | Page cache bypass | Databases |
| Zero-copy | Eliminate data copies | Web servers, streaming |
| Polling | I/O completion detection without interrupts | Ultra-low latency requirements |

---

## Recommended Next Guides

---

## References
1. Love, R. "Linux Kernel Development." 3rd Ed, Ch.14, 2010.
2. Bovet, D. & Cesati, M. "Understanding the Linux Kernel." 3rd Ed, O'Reilly, 2005.
3. Axboe, J. "Efficient IO with io_uring." Kernel.dk, 2019.
4. Arpaci-Dusseau, R. & Arpaci-Dusseau, A. "Operating Systems: Three Easy Pieces." Ch.37, 2018.
5. Gregg, B. "Systems Performance." 2nd Ed, Addison-Wesley, 2020.
6. Gregg, B. "BPF Performance Tools." Addison-Wesley, 2019.
7. Linux Block Layer Documentation. https://www.kernel.org/doc/html/latest/block/
8. io_uring Documentation. https://kernel.dk/io_uring.pdf
9. Bjorling, M. et al. "Linux Block IO: Introducing Multi-queue SSD Access on Multi-core Systems." SYSTOR, 2013.
