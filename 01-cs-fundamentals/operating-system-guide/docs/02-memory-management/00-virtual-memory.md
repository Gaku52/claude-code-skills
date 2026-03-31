# Virtual Memory

> Virtual memory is one of the most important OS abstractions, "going beyond physical memory limits to provide each process with an independent, vast address space."

## Learning Objectives

- [ ] Understand the necessity and mechanisms of virtual memory
- [ ] Explain address translation (MMU, page tables)
- [ ] Know the role of the TLB and optimization techniques
- [ ] Trace the operation of page faults and demand paging
- [ ] Compare and evaluate page replacement algorithms
- [ ] Grasp the overall picture of Linux virtual memory management
- [ ] Perform virtual memory performance tuning


## Prerequisites

Understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Why Virtual Memory Is Needed

### 1.1 Problems in a World Without Virtual Memory

```
Memory management before virtual memory:

  1950s-1960s:
  -> Programs used physical addresses directly
  -> Only one program could use the entire memory
  -> Multiprogramming was impossible

  1960s Base Register Approach:
  -> Set a base address for each process
  -> Address = Base + Offset
  -> Problem: Memory fragmentation, insufficient protection

  5 problems without virtual memory:

  Problem 1: Insufficient Memory
  +--------------------------------------+
  | Physical Memory: 4GB                 |
  | Process A: Needs 3GB                 |
  | Process B: Needs 2GB                 |
  | -> Cannot run simultaneously!        |
  |                                      |
  | With virtual memory:                 |
  | Provide each process with 8GB of     |
  | virtual space                        |
  | Only actually used portions placed   |
  | in physical memory                   |
  | The rest is swapped out to disk      |
  +--------------------------------------+

  Problem 2: Memory Fragmentation
  +----+  +--+  +------+  +----+  +--+
  | A  |  |  |  |  B   |  | C  |  |  |
  +----+  +--+  +------+  +----+  +--+
          ^ this gap               ^ here too
  -> Total free space is sufficient but contiguous regions are lacking

  Problem 3: Lack of Memory Protection
  -> Process A writes to an incorrect address
  -> Corrupts Process B's memory
  -> Entire system crashes

  Problem 4: Program Relocation
  -> Programs assume they will be loaded at specific addresses
  -> Address conflicts between multiple programs
  -> Programmers need to manage addresses manually

  Problem 5: Memory Efficiency
  -> 100 processes using the same library
  -> 100 copies exist in physical memory (wasteful)
  -> With virtual memory, a single copy can be shared
```

### 1.2 Virtual Memory Solution

```
Basic concept of virtual memory:
  Provide each process with an independent virtual address space
  -> No need to be aware of physical memory layout
  -> MMU (Memory Management Unit) handles address translation

  Process A           Process B
  +----------+       +----------+
  |Virt Addr |       |Virt Addr |
  |0x0000    |       |0x0000    |  <- Same address but
  |  :       |       |  :       |    different physical memory
  |0xFFFF    |       |0xFFFF    |
  +----+-----+       +----+-----+
       | MMU             | MMU
       v                  v
  +------------------------------+
  |     Physical Memory (RAM)     |
  | [A's data][B's data][free]    |
  +------------------------------+
       | Swap
  +------------------------------+
  |     Disk (Swap Area)          |
  +------------------------------+

  4 important functions of virtual memory:

  1. Address Translation:
     Virtual address -> Physical address mapping
     -> Managed by page tables
     -> MMU performs fast hardware translation

  2. Memory Protection:
     Complete isolation between processes
     -> Process A cannot access Process B's memory
     -> Read/Write/Execute permissions set per page
     -> Separation of kernel space and user space

  3. Demand Paging:
     Load into physical memory only when needed
     -> Do not load all code at program startup
     -> Only accessed pages are placed in physical memory
     -> Efficient use of memory

  4. Memory Sharing:
     Multiple processes share the same physical pages
     -> Shared libraries (libc, etc.)
     -> Copy-on-Write (optimization during fork)
     -> Explicit shared memory (IPC)

  Virtual address space layout (x86-64 Linux):
  +--------------------------------------------------+
  | 0xFFFFFFFFFFFFFFFF                                |
  | +--------------------------------------------+   |
  | | Kernel Space (upper half)                  |   |
  | | -> Not accessible from user mode           |   |
  | | -> Common mapping across all processes      |   |
  | +--------------------------------------------+   |
  | | 0xFFFF800000000000 --- Kernel start         |   |
  | |                                              |   |
  | | Non-canonical address region (unusable)      |   |
  | |                                              |   |
  | | 0x00007FFFFFFFFFFF --- User space upper limit|   |
  | +--------------------------------------------+   |
  | | Stack (grows downward)                      |   |
  | | v                                            |   |
  | |                                              |   |
  | | ^                                            |   |
  | | mmap region (shared libraries, large malloc) |   |
  | | ^                                            |   |
  | |                                              |   |
  | | ^                                            |   |
  | | Heap (grows upward)                          |   |
  | | --- brk                                      |   |
  | | BSS (uninitialized data)                     |   |
  | | Data segment (initialized global variables)  |   |
  | | Text segment (program code)                  |   |
  | | 0x0000000000400000 --- Program start         |   |
  | |                                              |   |
  | | NULL pointer trap region                     |   |
  | +--------------------------------------------+   |
  | 0x0000000000000000                                |
  +--------------------------------------------------+

  User space: 128TB (47 bits)
  Kernel space: 128TB (47 bits)
  -> With 5-level page tables enabled: 64PB each (56 bits)
```

### 1.3 ASLR (Address Space Layout Randomization)

```
ASLR: Randomization of address space layout

  Purpose:
  -> Prevent security attacks (buffer overflow, etc.)
  -> Change stack, heap, and library addresses at each startup
  -> Make it impossible for attackers to predict specific addresses

  Randomized regions:
  +------------------------------------------+
  | Stack:     ~8MB random offset            |
  | mmap area: Randomized within ~1TB range  |
  | Heap:      ~2MB random offset            |
  | Program:   Only when PIE (Position       |
  |            Independent Executable) is on  |
  +------------------------------------------+

  How to check:
  $ cat /proc/sys/kernel/randomize_va_space
  0 = Disabled
  1 = Stack and mmap only
  2 = All (default)

  $ cat /proc/self/maps    # Addresses change with each execution

  PIE (Position Independent Executable):
  -> Compile with -fPIE -pie
  -> Randomizes the executable's own address
  -> Enabled by default since GCC 6+ / Ubuntu 17.10+

  KASLR (Kernel ASLR):
  -> Randomizes the kernel load address
  -> Makes kernel vulnerability exploitation harder
  -> Enabled by default since Linux 4.12+
```

---

## 2. How Paging Works

### 2.1 Pages and Frames

```
Paging:
  Divides virtual memory into fixed-size "pages"
  Divides physical memory into same-size "frames"

  Page (virtual memory side) -> Frame (physical memory side)

  Common page sizes:
  +--------------+--------------+----------------------+
  | Page Size    | Architecture | Use Case             |
  +--------------+--------------+----------------------+
  | 4KB          | x86, ARM     | Standard (default)   |
  | 16KB         | ARM64(opt.)  | Apple Silicon         |
  | 64KB         | ARM64(opt.)  | Some ARM Linux        |
  | 2MB          | x86(hugepage)| Large pages (DB, JVM) |
  | 1GB          | x86(hugepage)| Very large memory     |
  +--------------+--------------+----------------------+

  Why 4KB?
  - Too small: Page table becomes enormous
  - Too large: Internal fragmentation increases
  - 4KB is a historically good balance (from VAX in the 1980s)
  - Matches disk sector size (512B -> 4KB)

  Apple Silicon (M1 and later) page size:
  -> macOS uses 16KB pages
  -> iOS also uses 16KB pages
  -> 4x TLB coverage compared to 4KB
  -> Watch for alignment when porting from x86
```

### 2.2 Address Translation

```
Virtual address -> Physical address translation:

  Virtual address (e.g., 32-bit):
  +--------------+-----------+
  | Page Number  | Offset    |
  | (20 bits)    | (12 bits) |
  +------+-------+-----+-----+
         |              |
         v              |
  +--------------+      |
  | Page Table   |      |
  | [0] -> Frame 5|     |
  | [1] -> Frame 2|     |
  | [2] -> Disk   |     | <- Page fault
  | [3] -> Frame 8|     |
  +------+-------+      |
         |              |
         v              v
  +--------------+-----------+
  | Frame Number | Offset    |
  +--------------+-----------+
  = Physical address

  Calculation example:
  Virtual address = 0x00003A7F
  Page size = 4KB = 2^12 = 4096 bytes

  Page number = 0x00003A7F >> 12 = 0x3 = 3
  Offset = 0x00003A7F & 0xFFF = 0xA7F

  Page table: Page 3 -> Frame 8
  Physical address = (8 << 12) | 0xA7F = 0x8A7F

Page Table Entry (PTE):
  +----------------------------------------------+
  | Bit fields:                                   |
  |                                                |
  | [63]    NX (No Execute): Execution prohibited |
  | [62:52] Reserved / Software use               |
  | [51:12] Physical frame number (40 bits)       |
  | [11:9]  Software use                          |
  | [8]     G (Global): Retained during TLB flush |
  | [7]     PS/PAT: Page size / PAT               |
  | [6]     D (Dirty): Written flag               |
  | [5]     A (Accessed): Referenced flag          |
  | [4]     PCD: Cache disable                     |
  | [3]     PWT: Write-through cache               |
  | [2]     U/S (User/Supervisor): User/Kernel     |
  | [1]     R/W (Read/Write): Read/Write permission|
  | [0]     P (Present): Page exists in phys. mem  |
  +----------------------------------------------+

  Key flag purposes:
  P=0: Page fault occurs
       -> Demand paging, swap out
  R/W=0: Page fault on write attempt
       -> Copy-on-Write implementation
  U/S=0: Exception on user-mode access
       -> Kernel memory protection
  NX=1: Exception on execution attempt
       -> DEP (Data Execution Prevention)
       -> Prevents shellcode execution on the stack
  D=1: Page has been modified
       -> Needs to be written back to disk on swap out
  A=1: Page has been referenced
       -> Used in page replacement algorithms
```

### 2.3 Multi-level Page Tables

```
Why multi-level is needed:

  Single-level page table (32-bit, 4KB pages):
  -> 2^20 = ~1 million entries
  -> 4 bytes per entry -> 4MB table
  -> 4MB per process (enormous with many processes)
  -> Most entries are unused
     (Typical processes use only a few MB to a few hundred MB)

  Two-level page table (32-bit):
  +--------+--------+-----------+
  | PD(10) | PT(10) |Offset(12) |
  +---+----+---+----+-----------+
      |        |
      v        v
  [Page Directory] -> [Page Table] -> Physical frame

  -> Page tables are not created for unused regions
  -> A process using only 1MB needs just a few page tables

x86-64 four-level page table:

  Virtual address (48-bit):
  +-----+-----+-----+-----+------------+
  |PML4 |PDPT | PD  | PT  | Offset     |
  |(9)  |(9)  |(9)  |(9)  |(12)        |
  +--+--+--+--+--+--+--+--+------------+
     |     |     |     |
     v     v     v     v
  [PML4]->[PDPT]->[PD]->[PT]-> Physical frame

  Each level: 512 entries (9 bits, 8 bytes/entry -> 4KB/table)
  -> 512 x 512 x 512 x 512 x 4KB = 256TB virtual space

  Role of each level:
  +-------+--------------------------------------+
  | PML4  | Top level. Pointed to by CR3 register|
  |       | 512 entries -> up to 512 PDPTs       |
  |       | Each entry covers 512GB range         |
  +-------+--------------------------------------+
  | PDPT  | 512 entries -> up to 512 PDs          |
  |       | Each entry covers 1GB range           |
  |       | 1GB large pages are possible           |
  +-------+--------------------------------------+
  | PD    | 512 entries -> up to 512 PTs           |
  |       | Each entry covers 2MB range            |
  |       | 2MB large pages are possible           |
  +-------+--------------------------------------+
  | PT    | 512 entries -> up to 512 phys. frames  |
  |       | Each entry is a 4KB page               |
  +-------+--------------------------------------+

  Memory saving example:
  Process using 100MB:
  -> 4-level: PML4(1) + PDPT(1) + PD(1) + PT(25) = 28 tables = 112KB
  -> 1-level: 256TB/4KB = 64 billion entries x 8B = 512GB (impossible)

5-level page table (Intel LA57, Linux 4.14+):

  Virtual address (57-bit):
  +-----+-----+-----+-----+-----+------------+
  |PML5 |PML4 |PDPT | PD  | PT  | Offset     |
  |(9)  |(9)  |(9)  |(9)  |(9)  |(12)        |
  +-----+-----+-----+-----+-----+------------+

  -> 128PB (petabytes) virtual address space
  -> For server environments with huge memory
  -> Not yet needed for general-purpose use
```

---

## 3. TLB (Translation Lookaside Buffer)

### 3.1 TLB Basics

```
Problem: Walking the page table every time is slow
  -> 4 levels = 4 memory accesses
  -> A single memory access becomes 5 times! (400% overhead)

TLB: A cache of the page table (CPU-integrated hardware)

  Virtual address
       |
       +---> [TLB] ---> Hit! -> Physical address (1 cycle)
       |        |
       |     Miss
       |        v
       +---> [Page table walk] -> Physical address + Register in TLB
            (4 memory accesses, hundreds of cycles)

  TLB internal structure:
  +--------------------------------------------------+
  | TLB Entry:                                        |
  | +----------+----------+-------+--------+          |
  | | Virt PN  | Phys FN  | ASID  | Flags  |          |
  | | (tag)    | (data)   |       | R/W/X  |          |
  | +----------+----------+-------+--------+          |
  |                                                    |
  | Associative memory (Content-Addressable Memory):   |
  | -> Searches all entries simultaneously by Virt PN  |
  | -> Returns result in 1 cycle                       |
  | -> High hardware cost (limits entry count)         |
  +--------------------------------------------------+

  Typical TLB sizes:
  +--------------------------------------------------+
  | Intel Core i7:                                    |
  |   L1 iTLB: 128 entries (4KB pages), 8-way         |
  |   L1 dTLB: 64 entries (4KB pages), 4-way          |
  |   L2 sTLB: 1536 entries (4KB pages), 12-way       |
  |   + Large page TLB (2MB/1GB): dozens of entries   |
  |                                                    |
  | Apple M1:                                          |
  |   L1 iTLB: 192 entries (16KB pages)                |
  |   L1 dTLB: 160 entries (16KB pages)                |
  |   L2 TLB: 3072 entries                             |
  |                                                    |
  | -> Very small (hundreds to thousands of entries)   |
  | -> Yet hit rate exceeds 99% (benefit of locality)  |
  +--------------------------------------------------+

  TLB Coverage:
  L1 dTLB 64 entries x 4KB = 256KB
  L2 sTLB 1536 entries x 4KB = 6MB
  -> TLB misses increase when working set exceeds 6MB

  With large pages (2MB):
  L1 dTLB 32 entries x 2MB = 64MB
  -> TLB coverage expands significantly!
```

### 3.2 TLB Management

```
TLB miss cost:
  Hit: 1 cycle (~0.3ns @3GHz)
  L2 TLB hit: ~5 cycles (~1.5ns)
  Miss (page table walk):
    Page table in L1 cache: ~20 cycles
    Page table in L2 cache: ~50 cycles
    Page table in memory: ~200-400 cycles

  -> Maintaining TLB hit rate is the key to performance

Context switch and TLB:

  Problem: Process A's TLB entries are invalid for Process B
  -> TLB needs to be invalidated during context switch

  Method 1: TLB Flush (invalidate all entries)
  -> Simple but TLB misses are frequent after switch
  -> Increases context switch cost

  Method 2: ASID (Address Space Identifier)
  -> Assign a process ID to each TLB entry
  -> Entries with different ASIDs are automatically ignored
  -> No TLB flush needed!
  -> Called PCID (Process Context Identifier) on x86
  -> Linux 4.14+ leverages PCID

  +----------------------------------------------+
  | No TLB flush (PCID enabled):                  |
  | Process A(PCID=1): [VPN=3->PFN=5, PCID=1]    |
  | Process B(PCID=2): [VPN=3->PFN=8, PCID=2]    |
  |                                                |
  | -> Same VPN=3 but distinguished by PCID        |
  | -> TLB is preserved during context switch      |
  | -> PCID importance increased due to             |
  |    Spectre/Meltdown mitigations                |
  +----------------------------------------------+

KPTI (Kernel Page Table Isolation):
  Countermeasure for Meltdown vulnerability (2018)
  -> Remove kernel page mappings in user mode
  -> Switch page tables during context switch
  -> TLB flush required -> Mitigated by PCID
  -> Performance impact: 0.1%-5% (workload dependent)
```

### 3.3 Large Pages (Huge Pages)

```
Advantages of large pages:

  4KB pages:
  4GB memory = 1,048,576 pages = 1 million TLB entries needed
  -> TLB has at most a few thousand entries -> Frequent misses

  2MB large pages:
  4GB memory = 2,048 pages = 2 thousand TLB entries sufficient
  -> TLB misses dramatically reduced (~1/500)

  1GB large pages:
  4GB memory = 4 pages = 4 TLB entries sufficient
  -> TLB misses almost zero

  Performance improvement:
  +----------------------------------------------+
  | Workload             | 4KB     | 2MB(huge)   |
  +----------------------+---------+-------------+
  | PostgreSQL (large DB)| Baseline| 5-15% better|
  | Redis (in-memory DB) | Baseline| 3-8% better |
  | JVM (large heap)     | Baseline| 10-20% better|
  | Scientific computing | Baseline| 5-30% better|
  | DPDK (networking)    | Baseline| Required    |
  +----------------------+---------+-------------+

  Large page configuration on Linux:

  1. Transparent Huge Pages (THP):
     -> Kernel automatically uses large pages
     -> No application changes required

  2. HugePages (static):
     -> Reserve large pages in advance
     -> Uses hugetlbfs
     -> Recommended for databases
```

```bash
# Large page configuration and verification

# Check Transparent Huge Pages status
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never

# Set THP to madvise mode (recommended)
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
# -> Use large pages only when explicitly requested by application

# Reserve static HugePages
echo 1024 > /proc/sys/vm/nr_hugepages        # 2MB x 1024 = 2GB reserved
cat /proc/meminfo | grep -i huge
# HugePages_Total:    1024
# HugePages_Free:     1024
# HugePages_Rsvd:        0
# Hugepagesize:       2048 kB

# Reserve 1GB HugePages (boot parameter)
# GRUB_CMDLINE_LINUX="hugepagesz=1G hugepages=4"
# -> 1GB x 4 = 4GB of 1GB large pages reserved

# Mount hugetlbfs
mount -t hugetlbfs none /mnt/huge

# Using HugePages with PostgreSQL
# postgresql.conf:
# huge_pages = try
# shared_buffers = 8GB
```

```c
// Requesting Transparent Huge Pages via madvise
#include <sys/mman.h>
#include <string.h>

int main() {
    size_t size = 256 * 1024 * 1024;  // 256MB

    // Allocate a region aligned to 2MB boundary
    void *ptr = mmap(NULL, size,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS,
                    -1, 0);

    // Request the kernel to use large pages
    madvise(ptr, size, MADV_HUGEPAGE);

    // Use memory...
    memset(ptr, 0, size);

    munmap(ptr, size);
    return 0;
}
```

---

## 4. Page Faults and Demand Paging

### 4.1 How Demand Paging Works

```
Demand Paging:
  Do not load all pages when a program starts
  -> Place in physical memory only when first accessed
  -> "Lazy" memory management (Lazy Allocation)

  At program startup:
  1. Read ELF header and set up virtual address mappings
  2. Do not actually place pages in physical memory (PTE P=0)
  3. Set program counter to entry point
  4. Page fault occurs on first instruction fetch
  5. Kernel loads that page from disk
  6. From then on, only necessary pages are loaded per access

  Advantages:
  - Fast startup (no need to load all code)
  - Good memory efficiency (unused pages don't consume physical memory)
  - Large programs can run with little memory

  If a 100MB program actually uses only 20MB:
  -> Can run with only 20MB of physical memory
  -> Remaining 80MB doesn't consume physical memory
```

### 4.2 Types and Handling of Page Faults

```
Page fault processing flow:

  1. CPU accesses a virtual address
  2. MMU references the page table
  3. PTE Present bit is 0 (or access permission violation)
  4. Page fault exception (#PF) is raised
  5. CPU saves fault address to CR2 register
  6. Kernel's page fault handler is invoked

  3 types of page faults:

  1. Minor Page Fault (Minor / Soft):
     -> No disk I/O required
     -> Only allocation and zero-clearing of a new physical page
     -> Example: First access to malloc-allocated region
     -> Cost: ~1us

  2. Major Page Fault (Major / Hard):
     -> Disk I/O required
     -> Read from swap area or file
     -> Example: Access to a swapped-out page
     -> Cost: ~1-10ms (HDD), ~0.1ms (SSD)

  3. Invalid Access:
     -> Access permission violation or unmapped address
     -> Kernel sends SIGSEGV (Segmentation Fault)
     -> Process crashes

  Kernel decision flow:
  +-----------------------------------------------------+
  | Page fault occurs                                    |
  |    |                                                 |
  |    +- Is address within VMA (Virtual Memory Area)?   |
  |    |  NO -> SIGSEGV (invalid access)                 |
  |    |  YES v                                          |
  |    +- Are access permissions correct?                |
  |    |  NO -> SIGSEGV (permission violation)            |
  |    |  YES v                                          |
  |    +- Is the page being accessed for the first time? |
  |    |  YES -> Allocate a new frame (Minor)            |
  |    |  NO v                                           |
  |    +- Is the page swapped out?                       |
  |    |  YES -> Read from swap (Major)                  |
  |    |  NO v                                           |
  |    +- Write to a Copy-on-Write page?                 |
  |    |  YES -> Copy the page (Minor)                   |
  |    |  NO v                                           |
  |    +- File-mapped page?                              |
  |       YES -> Read from file (Major)                  |
  +-----------------------------------------------------+
```

### 4.3 Copy-on-Write (COW)

```
Copy-on-Write:
  An optimization for fork(). Parent and child processes share
  the same physical pages, and copying occurs only when a
  write happens.

  Immediately after fork():
  Parent                 Child
  +----------+          +----------+
  | VPN 0 ---|--+  +---|-- VPN 0  |
  | VPN 1 ---|--+--+---|-- VPN 1  |
  | VPN 2 ---|--+--+---|-- VPN 2  |
  +----------+  |  |   +----------+
                v  v
              +----------+
              |Phys. Mem.| <- Shared (set to read-only)
              | FN 5     |
              | FN 8     |
              | FN 12    |
              +----------+

  Child process writes to VPN 1:
  1. Page fault occurs (write to read-only page)
  2. Kernel determines it is Copy-on-Write
  3. Allocates a new physical frame
  4. Copies contents of original page
  5. Updates child process PTE to new frame
  6. Restores both PTEs to read/write
  7. Re-executes the write

  Parent                 Child
  +----------+          +----------+
  | VPN 0 ---|--+  +---|-- VPN 0  |  <- Still shared
  | VPN 1 ---|--+--|---|-- VPN 1  |  <- Copied
  | VPN 2 ---|--+--+---|-- VPN 2  |  <- Still shared
  +----------+  |  |   +----------+
                v  v        |
              +----------+  |
              | FN 5     |  |  <- Shared
              | FN 8     |  |  <- Parent only
              | FN 12    |  |  <- Shared
              +----------+  |
                            v
                         +----------+
                         | FN 20    |  <- Copy for child's VPN 1
                         +----------+

  Advantages:
  - fork() is nearly instantaneous (only page table copy)
  - Avoids unnecessary copies in fork() + exec() pattern
  - Read-only data is permanently shared
```

### 4.4 Page Replacement Algorithms

```
When physical memory is full:
-> Which page to evict?
-> Page replacement algorithms come into play

1. FIFO (First-In, First-Out):
   Evict the oldest page
   -> Simple to implement
   -> Drawback: Frequently used pages are evicted if old
   -> Belady's Anomaly: Faults increase with more memory

   Example of Belady's Anomaly:
   Reference string: 1 2 3 4 1 2 5 1 2 3 4 5
   3 frames: 9 page faults
   4 frames: 10 page faults <- Worse despite more memory!

2. OPT (Optimal Algorithm / Belady's Algorithm):
   Evict the page that will be used furthest in the future
   -> Theoretically optimal but requires future knowledge, so not implementable
   -> Used as a baseline for evaluating other algorithms

3. LRU (Least Recently Used):
   Evict the page that has not been used for the longest time
   -> Based on temporal locality (recently used items will be used again)
   -> Performance close to OPT
   -> High implementation cost (exact LRU needs timestamps or a stack)

   Implementation cost of exact LRU:
   Method 1: Timestamps
   -> Update timestamp on every access -> High overhead
   Method 2: Stack
   -> Move to top of stack on every access -> O(n)
   -> Both are difficult to implement in hardware

4. Clock (Clock Algorithm / Second Chance):
   Approximation of LRU. Uses the Accessed bit.

   +---+   +---+   +---+   +---+
   |P1 |-->|P2 |-->|P3 |-->|P4 |-->...
   |A=1|   |A=0|   |A=1|   |A=0|
   +---+   +---+   +---+   +---+
     ^ clock hand

   Algorithm:
   1. Check Accessed bit of page at clock hand
   2. A=1 -> Set A=0 and move to next (second chance)
   3. A=0 -> Evict that page

   -> Adopted by Linux (improved version: two lists active/inactive)

5. LFU (Least Frequently Used):
   Evict the page with the lowest access frequency
   -> Counter-based
   -> Problem: Temporarily heavily used pages remain
   -> Addressed with decaying counters

6. Working Set Algorithm:
   Keep the working set (set of pages accessed within a recent
   time window) in memory
   -> Evict pages outside the working set
   -> Effective for preventing thrashing

Comparison:
+--------------+----------+----------+----------+----------+
| Algorithm    | Page     | Impl.    | Belady's | Practical|
|              | Faults   | Cost     | Anomaly  |          |
+--------------+----------+----------+----------+----------+
| OPT         | Minimum  | Impossible| None    | Benchmark|
| LRU         | Low      | High     | None     | Approx.  |
| Clock       | Somewhat | Low      | None     | Linux    |
|             | low      |          |          |          |
| FIFO        | High     | Lowest   | Yes      | Educational|
| LFU         | Low      | Medium   | None     | Caching  |
+--------------+----------+----------+----------+----------+
```

### 4.5 Linux Page Reclaim

```
Linux Page Reclaim:

  Two LRU lists:
  +----------------------------------------------+
  | Active List:                                   |
  | -> Recently accessed pages                     |
  | -> Pages here are basically not evicted        |
  |                                                |
  | Inactive List:                                 |
  | -> Pages not accessed for a while              |
  | -> Eviction candidates are selected from here  |
  |                                                |
  | Page flow:                                     |
  | New -> Inactive -> Access -> Active            |
  |                 -> Long-term unaccessed -> Evict|
  |                                                |
  | Active -> Long-term unaccessed -> Inactive      |
  +----------------------------------------------+

  Further split into 4 lists (Linux 2.6.28+):
  - Active Anonymous:   Stack, heap (swap target)
  - Inactive Anonymous: Same (eviction candidates)
  - Active File:        File cache (written back to file)
  - Inactive File:      Same (eviction candidates)

  kswapd:
  -> Background page reclaim daemon
  -> Starts when free pages fall below low watermark
  -> Reclaims pages from Inactive list
  -> Sleeps after reclaiming up to high watermark

  Direct Reclaim:
  -> When no free pages available during allocation
  -> Process itself performs page reclaim (blocking)
  -> Significant performance impact

  Watermarks:
  +----------------------------------------------+
  | Free memory amount                             |
  | |                                              |
  | |                                              |
  | | --- high watermark --- kswapd stops          |
  | |                                              |
  | | --- low watermark --- kswapd starts          |
  | |                                              |
  | | --- min watermark --- direct reclaim starts  |
  | |                                              |
  | +                                              |
  +----------------------------------------------+

  vm.swappiness parameter:
  -> Controls reclaim ratio of Anonymous vs File pages
  -> 0: Avoid swapping as much as possible (prefer File reclaim)
  -> 60: Default (balanced)
  -> 100: Treat Anonymous and File equally
  -> 200: Aggressively swap Anonymous (when using zswap)
```

---

## 5. Thrashing and OOM

### 5.1 Thrashing

```
Thrashing:
  A state where physical memory is extremely insufficient
  and page faults occur frequently
  -> Most time is spent on page I/O
  -> CPU utilization drops extremely low
  -> System effectively halts

  Vicious cycle of thrashing:
  1. Physical memory shortage
  2. Page faults become frequent
  3. I/O wait time increases
  4. CPU utilization drops
  5. OS decides "CPU is idle -> let's add more processes"
  6. Memory becomes even more insufficient
  7. Page faults increase further
  8. -> Continuous deterioration

  +----------------------------------------+
  | CPU Utilization                         |
  | |          /\                           |
  | |         /  \                          |
  | |        /    \                         |
  | |       /      \  <- Thrashing begins   |
  | |      /        \                       |
  | |     /          \                      |
  | |    /            \                     |
  | |   /              \                    |
  | |--/----------------\------            |
  | +                                      |
  |   Few     Many    Very many            |
  |          Number of processes            |
  +----------------------------------------+

  Countermeasures:
  1. Add memory (fundamental solution)
  2. Optimize swap (use SSD, zswap/zram)
  3. Limit number of processes
  4. Free memory via OOM Killer
  5. Monitor and control working sets
  6. Memory limits with cgroups
```

### 5.2 OOM Killer

```
OOM Killer (Out-Of-Memory Killer):
  A Linux mechanism that forcibly terminates processes to
  free memory when memory is completely exhausted

  Trigger conditions:
  -> Physical memory + swap are completely used up
  -> Page reclaim cannot secure enough memory
  -> Kernel fails to allocate memory

  OOM Score calculation:
  +----------------------------------------------+
  | /proc/<pid>/oom_score:                         |
  | Value 0-1000. Higher = more likely to be killed|
  |                                                |
  | Factors considered:                            |
  | - Memory usage (most important)                |
  | - Swap usage                                   |
  | - Process lifetime                             |
  | - root privileges (-30 bonus)                  |
  | - Adjustment via oom_score_adj                  |
  |                                                |
  | /proc/<pid>/oom_score_adj:                     |
  | Adjust score from -1000 to +1000               |
  | -1000: Exempt from OOM Killer (for DB servers) |
  | +1000: Kill with highest priority               |
  |     0: Default                                  |
  +----------------------------------------------+

  Protecting important processes:
  # Protect PostgreSQL from OOM Killer
  echo -1000 > /proc/$(pidof postgres)/oom_score_adj

  # OOM configuration in systemd
  # /etc/systemd/system/myservice.service
  [Service]
  OOMScoreAdjust=-900
  MemoryMax=4G
  MemoryHigh=3G

  Checking OOM events:
  # Check OOM Killer logs in dmesg
  dmesg | grep -i "out of memory"
  dmesg | grep -i "oom"
  # "Out of memory: Kill process 1234 (java) score 850"
```

```bash
# Memory management monitoring commands

# Memory usage overview
free -h
# total        used        free      shared  buff/cache   available
# Mem:   32Gi       8.0Gi       2.0Gi       256Mi        22Gi        23Gi
# Swap:  8.0Gi       0B        8.0Gi

# available: Amount of memory available for new processes
# -> free + buff/cache (reclaimable portion)

# Detailed memory information
cat /proc/meminfo

# Important fields:
# MemTotal:       Total memory
# MemFree:        Completely free memory
# MemAvailable:   Available memory (estimated)
# Buffers:        Block device buffers
# Cached:         Page cache
# SwapTotal:      Total swap
# SwapFree:       Free swap
# Active:         Active list
# Inactive:       Inactive list
# AnonPages:      Anonymous pages
# Mapped:         mmap'ed pages
# Shmem:          Shared memory
# HugePages_Total: Number of large pages

# Check page faults
/usr/bin/time -v ls 2>&1 | grep "page faults"
# Minor (reclaiming a frame): 234
# Major (requiring I/O): 0

# Process memory information
cat /proc/self/status | grep -E "^(Vm|Rss|Threads)"
# VmPeak:     Maximum virtual memory size
# VmSize:     Current virtual memory size
# VmRSS:      Resident memory (physical memory actually in use)
# VmSwap:     Swap usage

# Process memory map
cat /proc/self/maps
pmap -x $$          # With extended information
```

---

## 6. Virtual Memory in Practice

### 6.1 Checking Memory Usage

```bash
# Detailed process memory map
cat /proc/self/maps
# Address range          Perms  Offset  Device inode Path
# 55a0b1200000-55a0b1201000 r--p  00000000 08:01 123   /usr/bin/bash
# 55a0b1201000-55a0b12e8000 r-xp  00001000 08:01 123   /usr/bin/bash
# 55a0b12e8000-55a0b1320000 r--p  000e8000 08:01 123   /usr/bin/bash
# 55a0b1320000-55a0b1324000 r--p  0011f000 08:01 123   /usr/bin/bash
# 55a0b1324000-55a0b132d000 rw-p  00123000 08:01 123   /usr/bin/bash
# 55a0b27d0000-55a0b2910000 rw-p  00000000 00:00 0     [heap]
# 7f5c2a000000-7f5c2c000000 rw-p  00000000 00:00 0     (anonymous)
# ...
# 7fff45600000-7fff45621000 rw-p  00000000 00:00 0     [stack]
# 7fff45764000-7fff45768000 r--p  00000000 00:00 0     [vvar]
# 7fff45768000-7fff4576a000 r-xp  00000000 00:00 0     [vdso]

# Permission meanings:
# r = read, w = write, x = execute
# p = private (COW), s = shared

# For macOS
vmmap $$                     # Process memory map
vm_stat                      # System-wide memory statistics

# smaps: Detailed memory usage information (Linux)
cat /proc/self/smaps_rollup
# Rss:               1234 kB    # Physical memory usage
# Pss:                890 kB    # Proportional share of shared memory
# Pss_Anon:           456 kB    # Anonymous
# Pss_File:           434 kB    # File mapping
# Referenced:        1200 kB    # Referenced pages
# Swap:                 0 kB    # Swap usage
```

### 6.2 Manual Address Translation Calculation

```
Exercise: With 4KB page size and 32-bit virtual address space:

Virtual address 0x00003A7F access:

Step 1: Split into page number and offset
  4KB = 2^12 -> Offset is 12 bits
  Virtual address = 0x00003A7F = 0000 0000 0000 0000 0011 1010 0111 1111

  Page number = upper 20 bits = 0x00003 = 3
  Offset = lower 12 bits = 0xA7F = 2687

Step 2: Page table lookup
  Page 3 -> Frame 7
  Physical address = (7 << 12) | 0xA7F = 0x7A7F

Step 3: Processing when this entry is not in TLB
  1. TLB miss occurs
  2. MMU starts page table walk
  3. Obtain PML4 table address from CR3 register (for x86-64)
  4. Traverse each level's table to get physical frame number
  5. Register new entry in TLB
  6. Evict old TLB entry if necessary
  7. Re-execute memory access with physical address

Exercise 2: 64-bit address translation (x86-64)

  Virtual address: 0x00007F5C2A001234

  48-bit virtual address: 0x7F5C2A001234
  Bit decomposition:
  PML4  = bits 47-39 = 0x0FE = 254
  PDPT  = bits 38-30 = 0x170 = 368
  PD    = bits 29-21 = 0x150 = 336
  PT    = bits 20-12 = 0x001 = 1
  Offset = bits 11-0  = 0x234 = 564

  -> PML4[254] -> PDPT[368] -> PD[336] -> PT[1] -> Physical frame -> + 0x234
```

### 6.3 Performance Tuning

```bash
# Virtual memory tuning parameters

# Swap aggressiveness (0=minimum, 100=maximum, 200=maximum with zswap)
cat /proc/sys/vm/swappiness          # Default: 60
echo 10 > /proc/sys/vm/swappiness   # Recommended value for DB servers

# Dirty page write timing
cat /proc/sys/vm/dirty_ratio              # Default: 20 (%)
cat /proc/sys/vm/dirty_background_ratio   # Default: 10 (%)
# dirty_background_ratio: Background write starts
# dirty_ratio: Synchronous write forced (process blocks)

# Recommended settings (SSD, large memory)
echo 5 > /proc/sys/vm/dirty_background_ratio
echo 10 > /proc/sys/vm/dirty_ratio

# Overcommit control
cat /proc/sys/vm/overcommit_memory
# 0: Heuristic (default)
# 1: Always allow overcommit
# 2: Disable overcommit (swap + ratio x physical)
cat /proc/sys/vm/overcommit_ratio   # Default: 50 (%)

# Watermark adjustment
cat /proc/sys/vm/min_free_kbytes     # Minimum free memory
echo 262144 > /proc/sys/vm/min_free_kbytes  # Set to 256MB

# THP configuration
cat /sys/kernel/mm/transparent_hugepage/enabled
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
cat /sys/kernel/mm/transparent_hugepage/defrag
echo defer+madvise > /sys/kernel/mm/transparent_hugepage/defrag

# NUMA memory policy
numactl --hardware      # Check NUMA topology
numactl --interleave=all ./my_program    # Distribute memory evenly
numactl --membind=0 --cpunodebind=0 ./my_program  # Pin to node 0

# zswap (compressed swap cache)
echo 1 > /sys/module/zswap/parameters/enabled
echo lz4 > /sys/module/zswap/parameters/compressor
echo 20 > /sys/module/zswap/parameters/max_pool_percent

# zram (compressed RAM disk swap)
modprobe zram
echo 4G > /sys/block/zram0/disksize
mkswap /dev/zram0
swapon -p 100 /dev/zram0   # Add to swap with high priority
```

---

## 7. Advanced Topics in Virtual Memory

### 7.1 Memory Overcommit

```
Linux memory overcommit:

  malloc() success != physical memory allocation

  +----------------------------------------------+
  | Process A: malloc(1GB) -> Success (virtual only)|
  | Process B: malloc(1GB) -> Success (virtual only)|
  | Process C: malloc(1GB) -> Success (virtual only)|
  |                                                |
  | Physical memory: Only 2GB                      |
  | -> 3GB of malloc succeeded!                    |
  |                                                |
  | Physical memory is not allocated until accessed|
  | -> If all processes try to use 1GB each...     |
  | -> OOM Killer activates                        |
  +----------------------------------------------+

  Why overcommit?
  - Many programs don't use all their malloc'd memory
  - No need to fully copy parent process memory during fork() (COW)
  - Works well with actual usage patterns

  Cases where overcommit is problematic:
  - Java: Allocates large heap at startup (-Xmx8g, etc.)
  - Redis: fork + COW (during background saves)
  - Scientific computing: Bulk allocation of large arrays

  overcommit_memory settings:
  0 (default): Heuristic determination
    -> Obviously unreasonable requests are rejected
    -> Requests larger than "free memory + swap" may be rejected
  1: Always allow
    -> Used by JVM, etc. (for large malloc at startup)
  2: Strict limit
    -> CommitLimit = swap + physical x ratio
    -> malloc fails when Committed_AS exceeds this
    -> Guarantees OOM Killer won't fire
```

### 7.2 KSM (Kernel Same-page Merging)

```
KSM: Automatically shares pages with identical content
  -> Especially effective in virtualization environments (multiple VMs running same OS)
  -> Memory deduplication

  +----------------------------------------------+
  | Before KSM:                                    |
  | VM1: [Page A: "Hello"] [Page B: "World"]      |
  | VM2: [Page C: "Hello"] [Page D: "Linux"]      |
  | -> 4 pages of physical memory used             |
  |                                                |
  | After KSM:                                     |
  | VM1: [Page A: "Hello" -+  [Page B: "World"]   |
  | VM2: [Page C: "Hello" -+  [Page D: "Linux"]   |
  | -> 3 pages ("Hello" is shared)                 |
  |                                                |
  | On write: COW creates a copy                   |
  +----------------------------------------------+

  Configuration:
  echo 1 > /sys/kernel/mm/ksm/run
  cat /sys/kernel/mm/ksm/pages_shared       # Number of shared pages
  cat /sys/kernel/mm/ksm/pages_sharing      # Number of pages participating in sharing
  cat /sys/kernel/mm/ksm/pages_unshared     # Number of unique pages

  Note: KSM has CPU overhead (periodically compares pages)
  -> There is also a side-channel attack risk (timing attacks)
  -> May be disabled for security in cloud environments
```

### 7.3 MGLRU (Multi-generational LRU, Linux 6.1+)

```
MGLRU: New page reclaim framework

  Problems with traditional Active/Inactive lists:
  -> Two lists alone lack precision
  -> Can only distinguish page "temperature" (access frequency) in 2 levels
  -> Scanning overhead is large

  MGLRU improvements:
  -> Manages with 4 generation lists (gen0-gen3)
  -> Tracks page "temperature" more granularly
  -> Efficient scanning (reference checking via page table walks)

  +----------------------------------------------+
  | Gen 3 (newest): Recently accessed pages        |
  | Gen 2:          Pages accessed a bit ago       |
  | Gen 1:          Pages accessed quite a while ago|
  | Gen 0 (oldest): Pages not accessed for a long time|
  |                  -> Reclaim candidates           |
  |                                                |
  | Pages are promoted to higher generations on access|
  | All pages are demoted to lower generations over time|
  +----------------------------------------------+

  Enabling:
  echo Y > /sys/kernel/mm/lru_gen/enabled

  Performance improvements:
  -> Developed and tested on Chrome OS
  -> Significant performance improvement under memory pressure
  -> Especially effective in low-memory environments (4GB or less)
  -> Also adopted by Android
```


---

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that meets the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Also create test code

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main logic for data processing"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Retrieve processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "An exception should have been raised"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation with the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add an item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Delete by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Statistics information"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup:             {slow_time/fast_time:.0f}x")

benchmark()
```

**Key Points:**
- Be conscious of algorithm computational complexity
- Choose appropriate data structures
- Measure effectiveness with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read the stack trace and identify the occurrence location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Step-by-step verification**: Use log output and debuggers to verify hypotheses
5. **Fix and regression test**: After fixing, also run tests on related areas

```python
# Debugging utilities
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
    """Decorator that logs function input/output"""
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

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Check connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-------------|-----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB delay | EXPLAIN, slow query log | Indexes, query optimization |
---

## 8. FAQ

### Q1: What is swap?

When physical memory is insufficient, swap moves infrequently used pages to disk (swap area). This allows processes to use more memory than physical RAM, but disk I/O is extremely slow (100,000x slower than RAM: RAM 100ns vs HDD 10ms). SSDs improve this by 10-100x, but fundamentally you should add more RAM. Recently, techniques like zswap (compressed cache) and zram (compressed RAM disk) have become widespread for mitigating swap impact.

### Q2: What is the OOM Killer?

Linux Out-Of-Memory Killer. A mechanism where the OS forcibly terminates processes to free memory when memory is completely exhausted. Each process is assigned an oom_score (0-1000), and processes with higher scores are killed first. Adjustable from -1000 to +1000 via `/proc/<pid>/oom_score_adj`. Setting -1000 exempts from OOM Killer (recommended for database servers, etc.). Configurable via OOMScoreAdjust directive in systemd.

### Q3: Why do 64-bit systems only use 48 bits for addresses?

Current x86-64 uses a 48-bit virtual address space (256TB). The full 64 bits (16EB = 16 exabytes) is not currently needed, and the page table hierarchy would become too deep (6-7 levels). Intel 5-Level Paging (LA57) can extend to 57 bits (128PB), supported since Linux 4.14+. The design allows for gradual expansion as demand grows. AMD also supports 5-level paging through SVME (Secure Virtual Machine Extensions).

### Q4: What is the difference between RSS, VSS, and PSS?

- VSS (Virtual Set Size): The entire virtual memory of a process. Includes malloc'd but unused regions.
- RSS (Resident Set Size): Total pages present in physical memory. Double-counts shared libraries.
- PSS (Proportional Set Size): Divides shared pages by the number of processes. The most accurate "actual usage."
- USS (Unique Set Size): Only pages unique to that process. The amount freed when the process terminates.

Example: 10 processes sharing libc.so, libc = 2MB
- RSS: 2MB counted for each process (total 20MB)
- PSS: 200KB per process (2MB/10) counted (total 2MB)

### Q5: Why should memory limits be set for Docker containers?

Containers can use all host memory by default. Without memory limits, one container can exhaust memory and OOM Killer may forcibly terminate other containers. It is recommended to set limits with cgroups memory.max and use memory.high for early warning and reclaim. In Kubernetes, set via resources.limits. Containers exceeding limits are immediately OOM killed, so setting appropriate values is important.

### Q6: What is the relationship between memory leaks and virtual memory?

When a memory leak occurs, unused memory accumulates without being freed. From a virtual memory perspective:
1. VSS keeps increasing (malloc continues to succeed)
2. RSS also increases when accessed
3. Physical memory pressure increases -> Swap increases -> Performance degrades
4. Eventually OOM Killer activates
Due to virtual memory overcommit, malloc continues to succeed during early leak stages, which can delay problem discovery.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is most important. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently used in daily development work. It becomes especially important during code reviews and architecture design.

---

## 9. Summary

| Concept | Key Point |
|---------|-----------|
| Virtual Memory | Provides each process with an independent address space |
| Paging | Maps virtual to physical in 4KB units |
| Multi-level Page Table | x86-64 uses 4 levels (48-bit), 5 levels (57-bit) |
| TLB | Page table cache. Hit rate 99%+ |
| Large Pages | Expand TLB coverage with 2MB/1GB |
| Demand Paging | Place in physical memory only when needed |
| Copy-on-Write | Optimization during fork. Copy on write |
| Page Replacement | Clock (approximate LRU). Linux uses Active/Inactive lists |
| Thrashing | Frequent page faults from memory shortage |
| OOM Killer | Forcibly terminates processes when memory is exhausted |
| Overcommit | malloc success != physical memory allocation |

---

## Recommended Next Guides

---

## References
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Ch.9-10, 2018.
2. Gorman, M. "Understanding the Linux Virtual Memory Manager." Prentice Hall, 2004.
3. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Ch.13-24, 2018.
4. Bovet, D. & Cesati, M. "Understanding the Linux Kernel." 3rd Ed, O'Reilly, 2005.
5. Love, R. "Linux Kernel Development." 3rd Ed, Ch.15-16, 2010.
6. Corbet, J. "Multi-generational LRU: the next generation." LWN.net, 2022.
7. Intel. "Intel 64 and IA-32 Architectures Software Developer's Manual." Volume 3A, Ch.4, 2024.
8. Tanenbaum, A. S. & Bos, H. "Modern Operating Systems." 4th Ed, Ch.3, 2014.
