# Memory Allocation Strategies

> **The design of a memory allocator is one of the most critical foundational technologies that determines the overall throughput, latency, and stability of a system.**
> This chapter provides a comprehensive explanation covering everything from the principles of dynamic memory allocation to the internal architecture of production-level allocators, garbage collection, and memory leak analysis.

---

## Learning Objectives

- [ ] Accurately explain the structural differences between stack and heap and their memory layout
- [ ] Compare allocation algorithms including First Fit / Best Fit / Worst Fit / Buddy System
- [ ] Understand the internal architecture of ptmalloc2, jemalloc, tcmalloc, and mimalloc
- [ ] Grasp the major garbage collection algorithms (Mark & Sweep, Generational, Reference Counting) at the implementation level
- [ ] Practice detection and mitigation of memory leaks and fragmentation
- [ ] Understand from a compiler perspective why Rust's ownership model guarantees memory safety
- [ ] Explain kernel-level memory management (sbrk, mmap, SLAB allocator) and their interactions
- [ ] Understand memory allocation constraints in real-time systems and embedded environments


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Paging -- Virtual Memory, Page Tables, TLB, Page Replacement, Swapping](./01-paging.md)

---

## Table of Contents

1. [Process Memory Layout](#1-process-memory-layout)
2. [Fundamentals of Dynamic Memory Allocation](#2-fundamentals-of-dynamic-memory-allocation)
3. [Allocation Algorithms in Detail](#3-allocation-algorithms-in-detail)
4. [Major Memory Allocators](#4-major-memory-allocators)
5. [Kernel-Level Memory Management](#5-kernel-level-memory-management)
6. [Deep Dive into Garbage Collection](#6-deep-dive-into-garbage-collection)
7. [Memory Fragmentation and Optimization](#7-memory-fragmentation-and-optimization)
8. [Memory Leaks and Debugging](#8-memory-leaks-and-debugging)
9. [Language Runtimes and Memory Models](#9-language-runtimes-and-memory-models)
10. [Anti-Patterns and Design Principles](#10-anti-patterns-and-design-principles)
11. [Practical Exercises (3 Levels)](#11-practical-exercises-3-levels)
12. [FAQ](#12-faq)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. Process Memory Layout

### 1.1 Overview of the Virtual Address Space

Every process is given an independent virtual address space by the OS. A typical Linux x86-64 process memory layout is as follows.

```
  Virtual Address Space (Linux x86-64, 48-bit)

  High Address (0x7FFF_FFFF_FFFF)
  +---------------------------------------------+
  |           Kernel Space                       |  <- Inaccessible from user processes
  |       (Upper half, ~128TB)                   |
  +---------------------------------------------+  0x7FFF_FFFF_FFFF
  |                                              |
  |           Stack                              |  <- Grows from high to low addresses
  |           [Local variables, arguments,       |     Default limit: 8MB (ulimit -s)
  |            return addresses]                 |
  |              v Growth direction              |
  +---------------------------------------------+
  |                                              |
  |        Memory-Mapped Region                  |  <- mmap(), shared libraries,
  |        (Memory-Mapped Region)                |     large malloc (>128KB)
  |                                              |
  +---------------------------------------------+
  |              ^ Growth direction              |
  |           Heap                               |  <- Managed by malloc/free
  |           [Dynamically allocated data]       |     Extended by brk/sbrk
  +---------------------------------------------+  <- Program Break
  |           BSS Segment                        |  <- Uninitialized global/static variables
  |           (Zero-initialized)                 |
  +---------------------------------------------+
  |           Data Segment                       |  <- Initialized global/static variables
  |           (Initialized data)                 |
  +---------------------------------------------+
  |           Text Segment                       |  <- Executable code (read-only)
  |           (Program code)                     |     String literals are stored here too
  +---------------------------------------------+  0x0000_0040_0000 (typical start address)
  |           NULL Guard Page                    |  <- Detects NULL pointer dereferences
  +---------------------------------------------+
  Low Address (0x0000_0000_0000)
```

### 1.2 Detailed Characteristics of Each Segment

| Segment | Contents | Permissions | Size | Lifetime |
|:---:|:---|:---:|:---:|:---:|
| Text | Machine instructions, string literals | R-X | Fixed | Entire process |
| Data | Initialized global/static variables | RW- | Fixed | Entire process |
| BSS | Uninitialized global/static variables | RW- | Fixed | Entire process |
| Heap | Dynamically allocated via malloc/new | RW- | Variable | Until explicitly freed |
| mmap | Shared libraries, large allocations | Variable | Variable | Until munmap |
| Stack | Local variables, arguments, return addresses | RW- | Limited | Function scope |

### 1.3 Structural Differences Between Stack and Heap

Both the stack and heap store data at runtime, but their management mechanisms are fundamentally different.

```
  +----------- Stack ---------------+    +----------- Heap ----------------+
  |                                 |    |                                 |
  |  +----------------------+      |    |  Free list management:          |
  |  | main() frame         | <-SP |    |                                 |
  |  |  local_a = 10        |      |    |  +----+  +------+  +----+      |
  |  |  local_b = 20        |      |    |  |Used|->| Free |->|Used|->... |
  |  +----------------------+      |    |  | 8B |  | 64B  |  |32B |      |
  |  | foo() frame           |      |    |  +----+  +------+  +----+      |
  |  |  buf[256]            |      |    |                                 |
  |  |  saved_rbp           |      |    |  malloc(24) request:            |
  |  |  return_addr         |      |    |  -> Split the 64B free block    |
  |  +----------------------+      |    |  -> Allocate 24B+8B(header)=32B |
  |  | bar() frame           |      |    |  -> Remaining 32B becomes new  |
  |  |  tmp = 99            |      |    |     free block                  |
  |  +----------------------+      |    |                                 |
  |         v Growth direction     |    |  free() operation:              |
  |                                 |    |  -> Return block to free list  |
  |  Operations: push/pop (O(1))   |    |  -> Coalesce with adjacent     |
  |  Managed by: Hardware (SP reg) |    |     free blocks                 |
  |  Fragmentation: Does not occur |    |                                 |
  |  Speed: Extremely fast         |    |  Operations: Search+split/merge |
  +---------------------------------+    |  Managed by: Software (allocator)|
                                         |  Fragmentation: Occurs          |
                                         +---------------------------------+
```

**Performance Comparison: Stack vs Heap**

| Property | Stack | Heap |
|:---|:---:|:---:|
| Allocation speed | ~1 ns (SP move only) | ~50-200 ns (search+management) |
| Deallocation speed | ~1 ns (SP restore only) | ~50-100 ns (list operation) |
| Maximum size | 1-8 MB (OS configuration dependent) | Up to physical memory + swap |
| Fragmentation | Does not occur | External/internal fragmentation occurs |
| Thread safety | Independent per thread | Synchronization required (locks, etc.) |
| Cache efficiency | Extremely high (good locality) | Low (tends to be scattered) |
| Lifetime | Automatically tied to function scope | Explicitly managed by programmer |
| Overflow | Stack overflow | OOM (Out of Memory) |

### 1.4 Code Example: Verifying Memory Placement

```c
/* Code Example 1: Verifying process memory layout (Linux) */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Global variables -> Data segment or BSS */
int initialized_global = 42;       /* Data segment (initialized) */
int uninitialized_global;          /* BSS segment (zero-initialized) */
const char *string_literal = "Hello, Memory!";  /* Text segment */

void demonstrate_layout(void) {
    /* Local variables -> Stack */
    int stack_var = 100;
    char stack_array[64];

    /* Dynamic allocation -> Heap */
    int *heap_var = (int *)malloc(sizeof(int) * 256);
    if (!heap_var) {
        perror("malloc failed");
        return;
    }

    /* Static local variable -> Data segment */
    static int static_local = 55;

    printf("=== Memory Layout Verification ===\n");
    printf("Text  (function address):          %p\n", (void *)demonstrate_layout);
    printf("Text  (string literal):            %p\n", (void *)string_literal);
    printf("Data  (initialized global):        %p\n", (void *)&initialized_global);
    printf("Data  (static local):              %p\n", (void *)&static_local);
    printf("BSS   (uninitialized global):      %p\n", (void *)&uninitialized_global);
    printf("Heap  (malloc):                    %p\n", (void *)heap_var);
    printf("Stack (local variable):            %p\n", (void *)&stack_var);
    printf("Stack (array):                     %p\n", (void *)stack_array);
    printf("Program Break (brk):               %p\n", sbrk(0));

    free(heap_var);
}

int main(void) {
    demonstrate_layout();

    /* Check detailed memory map via /proc/self/maps */
    printf("\n=== /proc/self/maps (excerpt) ===\n");
    char cmd[64];
    snprintf(cmd, sizeof(cmd), "cat /proc/%d/maps | head -20", getpid());
    system(cmd);

    return 0;
}
```

**Example output (addresses vary by environment):**

```
=== Memory Layout Verification ===
Text  (function address):          0x55a3b2c01169
Text  (string literal):            0x55a3b2c02008
Data  (initialized global):        0x55a3b2c04010
Data  (static local):              0x55a3b2c04018
BSS   (uninitialized global):      0x55a3b2c04014
Heap  (malloc):                    0x55a3b3a092a0
Stack (local variable):            0x7ffd2e4b3c5c
Stack (array):                     0x7ffd2e4b3c10
Program Break (brk):               0x55a3b3a2a000
```

From the relative ordering of addresses, you can confirm the layout order: Text < Data < BSS < Heap << Stack. When ASLR (Address Space Layout Randomization) is enabled, the addresses change with each execution, but the relative positional relationships are preserved.

---

## 2. Fundamentals of Dynamic Memory Allocation

### 2.1 System Call Interface

User-space memory allocators ultimately acquire physical memory through kernel system calls. The two primary interfaces in Linux are `brk`/`sbrk` and `mmap`.

**brk / sbrk:**

```c
/* brk(): Set the Program Break to the specified address */
int brk(void *addr);

/* sbrk(): Move the Program Break by increment bytes and return the old address */
void *sbrk(intptr_t increment);
```

`brk`/`sbrk` extend the contiguous virtual address space by moving the end of the heap region (Program Break). glibc's `malloc` uses `sbrk` for small to medium-sized allocations.

**mmap:**

```c
/* mmap(): Create a new mapping in the virtual address space */
void *mmap(void *addr, size_t length, int prot, int flags,
           int fd, off_t offset);

/* munmap(): Release a mapping */
int munmap(void *addr, size_t length);
```

`mmap` allocates an independent virtual memory region. glibc's `malloc` uses `mmap(MAP_ANONYMOUS)` by default for allocations exceeding 128KB (MMAP_THRESHOLD). The advantage of `mmap` is that memory can be immediately returned to the kernel via `munmap` (whereas `sbrk` can only shrink from the end of the heap).

### 2.2 Internal Operation of malloc/free

```c
/* Code Example 2: malloc/free header structure (simplified) */

/*
 * A management header is placed immediately before the pointer returned by malloc.
 * This header stores the block size and used/free flags.
 *
 *    Memory block returned by malloc(24):
 *
 *    +------------------+---------------------------+
 *    |  Header (8-16B)  |    User data (24B)        |
 *    |  size | flags    |    <- malloc() return val  |
 *    +------------------+---------------------------+
 *    ^                  ^
 *    Actual start       Pointer returned to user
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* Simple allocator block header */
typedef struct block_header {
    size_t size;              /* Block size (including header) */
    int is_free;              /* 0: in use, 1: free */
    struct block_header *next; /* Pointer to next block */
} block_header_t;

#define HEADER_SIZE sizeof(block_header_t)
#define ALIGN(size) (((size) + 7) & ~7)  /* 8-byte alignment */

static block_header_t *free_list_head = NULL;

/* Simple malloc implementation (First Fit) */
void *simple_malloc(size_t size) {
    if (size == 0) return NULL;

    size_t aligned_size = ALIGN(size);
    size_t total_size = HEADER_SIZE + aligned_size;

    /* Search the free list for a suitable block (First Fit) */
    block_header_t *current = free_list_head;
    block_header_t *prev = NULL;

    while (current != NULL) {
        if (current->is_free && current->size >= total_size) {
            /* Found a free block of sufficient size */

            /* Split if possible */
            if (current->size >= total_size + HEADER_SIZE + 8) {
                block_header_t *new_block =
                    (block_header_t *)((char *)current + total_size);
                new_block->size = current->size - total_size;
                new_block->is_free = 1;
                new_block->next = current->next;

                current->size = total_size;
                current->next = new_block;
            }

            current->is_free = 0;
            return (void *)((char *)current + HEADER_SIZE);
        }
        prev = current;
        current = current->next;
    }

    /* No suitable block in free list -> extend with sbrk */
    block_header_t *new_block = (block_header_t *)sbrk(total_size);
    if (new_block == (void *)-1) {
        return NULL;  /* Out of memory */
    }

    new_block->size = total_size;
    new_block->is_free = 0;
    new_block->next = NULL;

    if (prev != NULL) {
        prev->next = new_block;
    } else {
        free_list_head = new_block;
    }

    return (void *)((char *)new_block + HEADER_SIZE);
}

/* Simple free implementation */
void simple_free(void *ptr) {
    if (ptr == NULL) return;

    block_header_t *header =
        (block_header_t *)((char *)ptr - HEADER_SIZE);
    header->is_free = 1;

    /* Coalescing adjacent free blocks */
    block_header_t *current = free_list_head;
    while (current != NULL) {
        if (current->is_free && current->next != NULL
            && current->next->is_free) {
            /* Merge two adjacent free blocks */
            current->size += current->next->size;
            current->next = current->next->next;
            continue;  /* Possibility of further merging */
        }
        current = current->next;
    }
}
```

### 2.3 Alignment Requirements

Modern processors require data to be placed on specific byte boundaries. Improper alignment causes performance degradation or hardware exceptions.

| Data Type | Typical Alignment | Reason |
|:---|:---:|:---|
| char | 1 byte | Byte-level access |
| short | 2 bytes | 16-bit bus width |
| int, float | 4 bytes | 32-bit register |
| long, double, pointer | 8 bytes | 64-bit register |
| SSE/AVX vector | 16 / 32 bytes | SIMD instruction requirements |
| Cache line | 64 bytes | False sharing prevention |

The pointer returned by `malloc` is adjusted to meet the platform's maximum alignment requirement (typically 16 bytes). C11's `aligned_alloc` or POSIX's `posix_memalign` can be used to specify arbitrary alignment (powers of 2).

```c
/* Allocation with specified alignment */
#include <stdlib.h>
#include <stdio.h>

int main(void) {
    /* 64-byte alignment (cache line boundary) */
    void *ptr = NULL;
    int ret = posix_memalign(&ptr, 64, 1024);
    if (ret != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        return 1;
    }
    printf("64-byte aligned pointer: %p (%%64 == %lu)\n",
           ptr, (unsigned long)ptr % 64);
    free(ptr);

    /* C11 aligned_alloc (size must be a multiple of alignment) */
    void *ptr2 = aligned_alloc(32, 1024);
    printf("32-byte aligned pointer: %p (%%32 == %lu)\n",
           ptr2, (unsigned long)ptr2 % 32);
    free(ptr2);

    return 0;
}
```

---

## 3. Allocation Algorithms in Detail

### 3.1 Free List-Based Algorithms

A free list is a data structure that manages unused memory blocks as a linked list. The difference between each algorithm lies in which block is selected from the free list when a new allocation request arrives.

```
  Free list state (numbers represent block sizes):

  HEAD -> [32B free] -> [64B used] -> [128B free] -> [16B free] -> [256B free] -> NULL

  When malloc(50) is requested:

  * First Fit:  Selects [128B free] <- First fitting block found from the beginning
  * Best Fit:   Skips [64B ...] -> Not [128B free]... full search
                  Smallest fitting = [128B free] selected
  * Worst Fit:  Selects [256B free] <- Selects the largest free block
  * Next Fit:   Starts from where the previous search ended

  First Fit operation:
  +----+    +----+    +----------+    +----+    +--------+
  |32B | -> |64B | -> |  128B    | -> |16B | -> | 256B   |
  |free|    |used|    |  free    |    |free|    | free   |
  +----+    +----+    +----------+    +----+    +--------+
                          ^
                      Split here!
                      50B->used | 78B->free
```

### 3.2 Algorithm Comparison Table

| Algorithm | Search Time | External Fragmentation | Internal Fragmentation | Implementation Complexity | Characteristics |
|:---:|:---:|:---:|:---:|:---:|:---|
| First Fit | O(n) avg fast | Medium | Low | Low | Small blocks accumulate at the head of list |
| Next Fit | O(n) distributed | High | Low | Low | Search is distributed but fragmentation worsens |
| Best Fit | O(n) full search | Low | High (small remnants) | Low | Many tiny free blocks are generated |
| Worst Fit | O(n) full search | High | Low | Low | Large blocks are exhausted quickly |
| Buddy System | O(log n) | Medium | High (powers of 2) | Medium | Fast coalescing, adopted by Linux kernel |
| Segregated Fit | O(1) same-size | Low | Medium | High | Separate lists per size class |
| TLSF | O(1) guaranteed | Low | Medium | High | For real-time systems |

### 3.3 Buddy System

The Buddy System is employed by the Linux kernel's physical page allocator. It manages memory by dividing it into blocks of power-of-2 sizes.

```
  Buddy System: Allocating 20B from a 128B memory pool

  Initial state:
  +---------------------------------------------------+
  |                   128B (order 7)                    |
  +---------------------------------------------------+

  Step 1: Split 128B -> 64B + 64B
  +-------------------------+-------------------------+
  |       64B (order 6)     |       64B (order 6)     |
  +-------------------------+-------------------------+

  Step 2: Split left 64B -> 32B + 32B
  +------------+------------+-------------------------+
  | 32B (o5)   | 32B (o5)   |       64B (order 6)     |
  +------------+------------+-------------------------+

  Step 3: Allocate left 32B (allocate 32B for a 20B request)
  +------------+------------+-------------------------+
  |# 32B used #| 32B free   |       64B free           |
  +------------+------------+-------------------------+
  <- 32B for 20B -> internal fragmentation 12B (37.5%)

  On deallocation: If buddy (adjacent same-size block) is free, merge
  32B + 32B -> 64B -> 64B + 64B -> 128B (fully restored)
```

**Buddy System Advantages and Disadvantages:**

- Advantage: Coalescing is O(1) fast (buddy address can be calculated via bit operations)
- Advantage: External fragmentation is limited (managed in buddy-sized units)
- Disadvantage: Internal fragmentation is large (request size rounded up to power of 2)
- Disadvantage: 33B allocation requires 64B (internal fragmentation ~48%)

### 3.4 Segregated Free Lists

A technique adopted by most modern high-performance allocators. Maintains individual free lists for each size class.

```
  Segregated Free Lists structure:

  Size class    Free list
  +----------+
  |   8B     | -> [free] -> [free] -> [free] -> NULL
  +----------+
  |  16B     | -> [free] -> [free] -> NULL
  +----------+
  |  32B     | -> NULL  (no free blocks -> split from upper class)
  +----------+
  |  64B     | -> [free] -> [free] -> [free] -> [free] -> NULL
  +----------+
  |  128B    | -> [free] -> NULL
  +----------+
  |  256B    | -> [free] -> NULL
  +----------+
  |  512B    | -> NULL
  +----------+
  |  ...     |
  +----------+
  |  large   | -> Managed by tree or sorted list
  +----------+

  For malloc(20):
  -> Retrieve from size class 32B list (O(1))
  -> If list is empty, take one from 64B class and split
```

---

## 4. Major Memory Allocators

### 4.1 ptmalloc2 (glibc Standard Allocator)

ptmalloc2 is a multi-threaded extension of Doug Lea's dlmalloc and serves as the default allocator in nearly all Linux distributions.

**Internal Architecture:**

```
  ptmalloc2 structure:

  +-----------------------------------------------------+
  |                   ptmalloc2                          |
  |                                                      |
  |  +----------+  +----------+  +----------+          |
  |  | Arena 0  |  | Arena 1  |  | Arena N  |  ...     |
  |  | (main)   |  | (thread) |  | (thread) |          |
  |  |          |  |          |  |          |          |
  |  | +------+ |  | +------+ |  | +------+ |          |
  |  | |fast  | |  | |fast  | |  | |fast  | |          |
  |  | |bins  | |  | |bins  | |  | |bins  | |          |
  |  | |(LIFO)| |  | |(LIFO)| |  | |(LIFO)| |          |
  |  | +------+ |  | +------+ |  | +------+ |          |
  |  | |small | |  | |small | |  | |small | |          |
  |  | |bins  | |  | |bins  | |  | |bins  | |          |
  |  | |(FIFO)| |  | |(FIFO)| |  | |(FIFO)| |          |
  |  | +------+ |  | +------+ |  | +------+ |          |
  |  | |large | |  | |large | |  | |large | |          |
  |  | |bins  | |  | |bins  | |  | |bins  | |          |
  |  | |(sort)| |  | |(sort)| |  | |(sort)| |          |
  |  | +------+ |  | +------+ |  | +------+ |          |
  |  | |unsort| |  | |unsort| |  | |unsort| |          |
  |  | |bin   | |  | |bin   | |  | |bin   | |          |
  |  | +------+ |  | +------+ |  | +------+ |          |
  |  +----------+  +----------+  +----------+          |
  |                                                      |
  |  Size thresholds:                                    |
  |    fastbin:  <= 160B (64bit)  -> LIFO, lock-free    |
  |    smallbin: <= 512B          -> FIFO, exact size   |
  |    largebin: > 512B           -> Sorted tree        |
  |    mmap:     > 128KB          -> Directly via mmap()|
  +-----------------------------------------------------+
```

**ptmalloc2 Allocation Flow:**

1. Size is within fastbin range -> Retrieve from fastbin (lock-free, fastest)
2. Size is within smallbin range -> Retrieve from smallbin
3. Scan unsorted bin -> Retrieve if suitable block found, otherwise classify
4. Retrieve from largebin using Best Fit
5. Split from top chunk
6. Extend heap with sbrk(), or directly allocate via mmap()

### 4.2 jemalloc (FreeBSD / Redis / Firefox)

An allocator designed by Jason Evans for FreeBSD. Excels in low fragmentation and thread scalability.

**Key Design Features:**

- **Thread cache (tcache):** Each thread has a local cache, so small allocations require no locks
- **Arenas:** Typically creates CPU cores x 4 arenas, distributing threads across them
- **Size classes:** Three tiers: Small (<=14336B), Large (<=4MB), Huge (>4MB)
- **Extents:** Basic unit of memory management. Managed at page granularity
- **Statistics:** Detailed memory usage statistics available via `malloc_stats_print()`

```c
/* Code Example 3: Retrieving jemalloc statistics */
#include <jemalloc/jemalloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    /* Allocate memory of various sizes */
    void *ptrs[1000];
    for (int i = 0; i < 1000; i++) {
        ptrs[i] = malloc(rand() % 4096 + 1);
    }

    /* Free half (to induce fragmentation) */
    for (int i = 0; i < 1000; i += 2) {
        free(ptrs[i]);
        ptrs[i] = NULL;
    }

    /* Output jemalloc statistics */
    malloc_stats_print(NULL, NULL, NULL);

    /* Retrieve specific statistics */
    size_t allocated, active, resident;
    size_t sz = sizeof(size_t);

    mallctl("stats.allocated", &allocated, &sz, NULL, 0);
    mallctl("stats.active", &active, &sz, NULL, 0);
    mallctl("stats.resident", &resident, &sz, NULL, 0);

    printf("\n=== Memory Usage ===\n");
    printf("Allocated: %zu bytes\n", allocated);
    printf("Active:    %zu bytes\n", active);
    printf("Resident:  %zu bytes\n", resident);
    printf("Fragmentation rate: %.2f%%\n",
           (1.0 - (double)allocated / active) * 100);

    /* Free the rest */
    for (int i = 1; i < 1000; i += 2) {
        free(ptrs[i]);
    }

    return 0;
}
/* Compile: gcc -o test test.c -ljemalloc */
```

### 4.3 tcmalloc (Google)

Thread-Caching Malloc developed by Google. Specializes in fast allocation of small objects.

**Architecture:**

```
  tcmalloc architecture:

  +----------------------------------------------------+
  |                   tcmalloc                          |
  |                                                     |
  |  +-------------+ +-------------+ +------------+   |
  |  | Thread Cache | | Thread Cache | |Thread Cache|   |
  |  |  (per-thread)| |  (per-thread)| |(per-thread)|   |
  |  |  Lock-free   | |  Lock-free   | |Lock-free   |   |
  |  +------+------+ +------+------+ +-----+------+   |
  |         |               |              |            |
  |         v               v              v            |
  |  +--------------------------------------------+    |
  |  |         Central Free List                   |    |
  |  |     (Separate list per size class)          |    |
  |  |     Protected by spinlock                   |    |
  |  +--------------------+------------------------+    |
  |                       |                             |
  |                       v                             |
  |  +--------------------------------------------+    |
  |  |         Page Heap                           |    |
  |  |     (Page-level large block management)     |    |
  |  |     span: collection of contiguous pages    |    |
  |  +--------------------------------------------+    |
  |                                                     |
  |  Small objects (<=256KB):                           |
  |    Thread Cache -> Central List -> Page Heap        |
  |  Large objects (>256KB):                            |
  |    Allocated directly from Page Heap                |
  +----------------------------------------------------+
```

### 4.4 mimalloc (Microsoft Research)

A state-of-the-art allocator released by Microsoft Research in 2019. Demonstrates performance exceeding other allocators in benchmarks.

**Key Design Features:**

- **Segment-based:** Acquires large memory blocks (segments) and internally divides them into pages
- **Free list sharding:** Separates local free lists from thread free lists
- **Intra-page management:** Places objects of the same size class on the same page
- **Deferred free:** `free` calls from other threads are added to the thread free list and processed later by the owning thread

### 4.5 Allocator Comprehensive Comparison

| Property | ptmalloc2 | jemalloc | tcmalloc | mimalloc |
|:---|:---:|:---:|:---:|:---:|
| Developer | glibc | FreeBSD/Meta | Google | Microsoft |
| Thread cache | Per-arena | tcache | Thread Cache | Local list |
| Small object speed | Medium | High | Extremely high | Extremely high |
| Large object speed | Medium | High | Medium | High |
| Memory usage efficiency | Medium | High | Medium | High |
| Fragmentation resistance | Low-Medium | High | Medium | Extremely high |
| Statistics/Debug | Limited | Very comprehensive | Comprehensive | Comprehensive |
| Memory return to OS | Slow | Good | Good | Good |
| Notable adopters | Linux standard | Redis, Firefox | Chrome, gRPC | .NET, Research |
| License | LGPL | BSD-2 | Apache 2.0 | MIT |
| Real-time suitability | Low | Medium | Medium | Medium-High |

**Allocator Selection Guidelines:**

- **Default (no specific reason):** ptmalloc2 (use the OS standard as-is)
- **Many small objects like Redis:** jemalloc (less fragmentation)
- **Multi-threaded with primarily small objects:** tcmalloc (thread cache is effective)
- **Latest benchmark performance:** mimalloc (high performance overall)
- **Memory usage visualization needed:** jemalloc (most comprehensive statistics)

---

## 5. Kernel-Level Memory Management

### 5.1 Physical Page Allocator (Buddy Allocator)

The Linux kernel manages physical memory in page units (typically 4KB). The Buddy Allocator manages contiguous physical page blocks from 2^0 to 2^10 pages (4KB to 4MB).

```
  Linux Buddy Allocator structure (/proc/buddyinfo):

  Order:    0     1     2     3     4     5     6     7     8     9    10
  Size:    4KB   8KB  16KB  32KB  64KB 128KB 256KB 512KB  1MB   2MB   4MB
          +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
  Zone    |     |     |     |     |     |     |     |     |     |     |     |
  DMA     |  3  |  1  |  0  |  0  |  2  |  1  |  0  |  1  |  1  |  1  |  3  |
          +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
  DMA32   | 12  |  8  |  6  |  4  |  3  |  2  |  1  |  1  |  0  |  0  |  1  |
          +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
  Normal  |1024 | 512 | 256 | 128 |  64 |  32 |  16 |   8 |   4 |   2 |   1 |
          +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

  Allocation request: 32KB (order 3) needed
  1. Check the free list for order 3
  2. If none available, take a 64KB (order 4) block and split
  3. 64KB -> 32KB + 32KB (one allocated, one goes to order 3 free list)

  On deallocation:
  1. Check if buddy (pair split from the same parent) is free
  2. If free, merge to higher order -> recursively attempt merging
```

### 5.2 SLAB Allocator

The Buddy Allocator manages memory at page granularity (minimum 4KB), which is wasteful for the small objects (tens to hundreds of bytes of structures) frequently used within the kernel. The SLAB Allocator was designed by Bonwick (1994) to solve this problem.

```
  SLAB Allocator's 3-layer structure:

  +------------------------------------------------------------+
  |                    Cache                                    |
  |  (One per object type: task_struct, inode, dentry, etc.)   |
  |                                                             |
  |  +---------------------------------------------+           |
  |  |  slabs_full     (All objects in use)         |           |
  |  |  +------+ +------+ +------+                 |           |
  |  |  | slab |>| slab |>| slab |> ...            |           |
  |  |  +------+ +------+ +------+                 |           |
  |  +---------------------------------------------+           |
  |  |  slabs_partial  (Some objects in use)        |           |
  |  |  +------+ +------+                          |           |
  |  |  | slab |>| slab |> ...  <- Allocate here   |           |
  |  |  +------+ +------+                          |           |
  |  +---------------------------------------------+           |
  |  |  slabs_empty    (All objects free)           |           |
  |  |  +------+                                    |           |
  |  |  | slab |> ...  <- Reclaimed under memory    |           |
  |  |  +------+          pressure                  |           |
  |  +---------------------------------------------+           |
  |                                                             |
  |  Internal structure of a single SLAB:                      |
  |  +----+----+----+----+----+----+----+----+                 |
  |  |obj |obj |obj |obj | ...|obj |obj |mgmt|                 |
  |  | 0  | 1  | 2  | 3  |    |n-1 | n  |info|                 |
  |  +----+----+----+----+----+----+----+----+                 |
  |  <- 1 page (4KB) or multiple pages ->                      |
  |  Each obj is the same size (e.g., task_struct ~ 6KB)       |
  +------------------------------------------------------------+
```

The Linux kernel has successors to SLAB: **SLUB** (Unqueued SLAB, default since 2.6.22) and **SLOB** (simplified version for embedded systems).

| Implementation | Features | Use Case |
|:---:|:---|:---|
| SLAB | Original. Object coloring, per-CPU cache | Legacy servers |
| SLUB | Simple design, metadata stored within objects | Current default |
| SLOB | Minimal memory overhead | Embedded (few MB RAM) |

### 5.3 Observing Kernel Memory via /proc

```bash
# Code Example 4: Commands for checking kernel memory information

# --- Overall physical memory status ---
cat /proc/meminfo
# MemTotal:       16384000 kB    <- Total physical memory
# MemFree:         2048000 kB    <- Completely free memory
# MemAvailable:    8192000 kB    <- Available (including cache)
# Buffers:          512000 kB    <- Block device buffers
# Cached:          4096000 kB    <- Page cache
# Slab:             256000 kB    <- SLAB allocator total

# --- Buddy Allocator status ---
cat /proc/buddyinfo
# Node 0, zone   Normal  1024  512  256  128  64  32  16  8  4  2  1

# --- SLAB detailed statistics ---
cat /proc/slabinfo | head -20
# name            <active_objs> <num_objs> <objsize> ...
# task_struct          512        520       6016     ...
# inode_cache         2048       2060        592     ...
# dentry              4096       4100        192     ...

# --- Per-process memory map ---
cat /proc/self/maps | head -10
# 55a3b2c00000-55a3b2c02000 r--p  ... /path/to/binary   (text)
# 55a3b2c02000-55a3b2c04000 r-xp  ... /path/to/binary   (code)
# 7ffd2e490000-7ffd2e4b2000 rw-p  ... [stack]           (stack)

# --- Top 10 processes by memory usage ---
ps aux --sort=-%mem | head -11

# --- Monitor memory activity with vmstat ---
vmstat 1 5
# procs ---memory--- ---swap-- -----io---- -system-- ------cpu-----
#  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs ...
```

### 5.4 Demand Paging and Overcommit

Linux permits memory overcommit by default. Even if `malloc` succeeds, the actual physical page is not allocated until the first write (Demand Paging). When physical memory is exhausted, the OOM Killer forcibly terminates a process.

```
  Demand Paging flow:

  malloc(4096)
       |
       v
  Virtual page is reserved (no physical page allocated yet)
  Page table entry: Present=0
       |
       v
  *ptr = 42;  <- First write
       |
       v
  Page fault occurs (Present=0)
       |
       v
  Kernel allocates a physical page
  Updates page table: Present=1, PFN=physical frame number
       |
       v
  Write completes

  vm.overcommit_memory settings:
  0 (default): Heuristic overcommit
  1: Always allow overcommit (malloc always succeeds)
  2: Disallow overcommit (limit is swap + RAM x ratio)
```

---

## 6. Deep Dive into Garbage Collection

### 6.1 Basic Classification of GC

Garbage Collection (GC) is a mechanism that automatically reclaims memory no longer used by a program. It fundamentally prevents bugs such as memory leaks, double-free, and Use-After-Free that are unavoidable with manual memory management (malloc/free).

**GC Basic Classification Table:**

| Classification Axis | Options | Description |
|:---|:---|:---|
| Collection timing | Stop-the-World / Concurrent / Incremental | Presence/extent of application pause |
| Reachability determination | Tracing / Reference counting | Method of determining garbage |
| Movement | Compacting (with movement) / Non-compacting | Handling of fragmentation |
| Generational management | Generational / Non-generational | Exploiting object lifetime |
| Target scope | Full GC / Partial GC (Minor/Major) | Granularity of collection scope |

### 6.2 Mark & Sweep

The most fundamental tracing GC algorithm.

```
  Mark & Sweep operation:

  [Mark Phase]
  Mark all objects reachable from the root set
  (stack, global variables, registers)

  Root
   |
   v
  [A]-->[B]-->[C]      [G]-->[H]
   |      |                     ^
   v      v                     |
  [D]    [E]-->[F]      [I]-->[J]
                          ^
  Reachable from root:     Unreachable (garbage):
  A, B, C, D, E, F        G, H, I, J

  After marking:
  [A+]-->[B+]-->[C+]   [G ]-->[H ]
   |       |                     ^
   v       v                     |
  [D+]   [E+]-->[F+]   [I ]-->[J ]

  [Sweep Phase]
  Scan the entire heap and reclaim unmarked objects

  +----+----+----+----+----+----+----+----+----+----+
  | A+ | G  | B+ | H  | D+ | I  | E+ | J  | C+ | F+ |
  +----+----+----+----+----+----+----+----+----+----+
                     v Sweep
  +----+----+----+----+----+----+----+----+----+----+
  | A  |free| B  |free| D  |free| E  |free| C  | F  |
  +----+----+----+----+----+----+----+----+----+----+
  * Clear marks, add unmarked to free list

  Issues:
  - Stop-the-World: Application pauses during marking
  - Fragmentation: Reclaimed memory is scattered (-> mitigated by Compaction)
  - Full heap scan: Sweep takes time if heap is large
```

### 6.3 Generational GC

**Weak Generational Hypothesis:** Most objects die young. Based on this hypothesis, objects are managed by generation.

```
  Generational GC structure (Java HotSpot JVM example):

  +------------------------------------------------------------+
  |                     Java Heap                               |
  |                                                             |
  |  +------------------------------+  +-------------------+  |
  |  |       Young Generation       |  |  Old Generation   |  |
  |  |                              |  |                   |  |
  |  |  +-------+-------+--------+ |  |                   |  |
  |  |  | Eden  |  S0   |   S1   | |  |  Tenured gen.     |  |
  |  |  |       |(From) |  (To)  | |  |  (long-lived obj.)|  |
  |  |  | New   |Survivor|Survivor| |  |                   |  |
  |  |  | alloc |       |        | |  |                   |  |
  |  |  +-------+-------+--------+ |  |                   |  |
  |  |   <- Minor GC (frequent,   |  |  <- Major GC      |  |
  |  |      fast)                  |  |    (infrequent,   |  |
  |  +------------------------------+  |     slow)        |  |
  |                                     +-------------------+  |
  |                                                             |
  |  Object lifecycle:                                         |
  |  1. Born in Eden                                           |
  |  2. Survives Minor GC -> Survivor (alternates between S0/S1)|
  |  3. Survives a certain number of Minor GCs -> Promoted to Old|
  |  4. Old Generation becomes full -> Major GC (Full GC)      |
  +------------------------------------------------------------+

  Write Barrier:
  Records when a reference from Old -> Young is written
  -> Avoids scanning the entire Old during Minor GC
  -> Managed via Card Table or Remembered Set
```

**Evolution of Java GC:**

| GC Implementation | Generational | Pause Time | Throughput | Features |
|:---|:---:|:---:|:---:|:---|
| Serial GC | Yes | Long | Low | Single-threaded, for small-scale |
| Parallel GC | Yes | Medium | High | Multi-threaded parallel collection |
| CMS | Yes | Short | Medium | Concurrent Mark Sweep, removed in JDK 14 |
| G1 GC | Region | Short | High | Region-based, default since JDK 9 |
| ZGC | Region | Ultra-short (<1ms) | High | Colored pointers, TB-scale heap support |
| Shenandoah | Region | Ultra-short | High | Red Hat developed, concurrent compaction |

### 6.4 Reference Counting

Each object maintains a count of "the number of pointers referencing it." When the count reaches 0, the object can be immediately reclaimed.

```python
# Code Example 5: Reference counting and circular references in Python

import sys
import gc

# --- Checking reference counts ---
a = [1, 2, 3]
print(f"Reference count: {sys.getrefcount(a)}")  # 2 (a + getrefcount argument)

b = a  # Add a reference
print(f"Reference count: {sys.getrefcount(a)}")  # 3

del b  # Remove a reference
print(f"Reference count: {sys.getrefcount(a)}")  # 2

# --- Circular reference problem ---
class Node:
    def __init__(self, name):
        self.name = name
        self.ref = None
    def __del__(self):
        print(f"  Node({self.name}) was collected")

# Create circular reference
node_a = Node("A")
node_b = Node("B")
node_a.ref = node_b  # A -> B
node_b.ref = node_a  # B -> A  <- Circular reference!

# Even after removing external references, reference count won't reach 0
del node_a
del node_b
# -> Not collected by reference counting alone!

# Python's GC (generational tracing) detects and collects circular references
print("Before gc.collect():")
collected = gc.collect()
print(f"Objects collected by gc.collect(): {collected}")

# --- GC generational statistics ---
print(f"\nGC generational thresholds: {gc.get_threshold()}")  # (700, 10, 10)
print(f"GC generational counts: {gc.get_count()}")
# Generation 0: GC after 700 allocations
# Generation 1: GC every 10 generation 0 GCs
# Generation 2: GC every 10 generation 1 GCs

# --- Avoiding circular references with weak references ---
import weakref

class SafeNode:
    def __init__(self, name):
        self.name = name
        self._ref = None

    @property
    def ref(self):
        if self._ref is not None:
            return self._ref()
        return None

    @ref.setter
    def ref(self, node):
        self._ref = weakref.ref(node) if node else None

safe_a = SafeNode("A")
safe_b = SafeNode("B")
safe_a.ref = safe_b
safe_b.ref = safe_a  # Weak reference so no circular reference

del safe_a  # safe_b._ref() now returns None
```

### 6.5 Go's GC (Tri-color Mark and Sweep)

Go employs a non-generational concurrent tracing GC. The tri-color marking algorithm minimizes application pause times.

```
  Tri-color Marking:

  White: Unvisited (collection target after GC completes)
  Gray:  Visited but children not yet explored
  Black: Visited and all children explored

  Initial state: All objects are white
  +--+  +--+  +--+  +--+  +--+  +--+
  |Aw|->|Bw|->|Cw|  |Dw|  |Ew|->|Fw|
  +--+  +--+  +--+  +--+  +--+  +--+
  Roots: A, E

  Step 1: Color roots gray
  +--+  +--+  +--+  +--+  +--+  +--+
  |Ag|->|Bw|->|Cw|  |Dw|  |Eg|->|Fw|
  +--+  +--+  +--+  +--+  +--+  +--+

  Step 2: Color gray objects' children gray, self becomes black
  +--+  +--+  +--+  +--+  +--+  +--+
  |Ab|->|Bg|->|Cw|  |Dw|  |Eb|->|Fg|
  +--+  +--+  +--+  +--+  +--+  +--+

  Step 3: Repeat
  +--+  +--+  +--+  +--+  +--+  +--+
  |Ab|->|Bb|->|Cg|  |Dw|  |Eb|->|Fb|
  +--+  +--+  +--+  +--+  +--+  +--+

  Final: Complete when no gray remains
  +--+  +--+  +--+  +--+  +--+  +--+
  |Ab|->|Bb|->|Cb|  |Dw|  |Eb|->|Fb|
  +--+  +--+  +--+  +--+  +--+  +--+
  Collect white(D)!  w=white(collect) g=gray b=black(alive)

  Go GC characteristics:
  - Stop-the-World is tens of us to hundreds of us
  - Write barrier maintains consistency during concurrent marking
  - GOGC environment variable adjusts GC frequency (default 100 = fires when heap doubles)
```

### 6.6 Rust's Ownership Model

Rust guarantees memory safety through compile-time Ownership and Borrowing rules without using GC.

```
  Rust's 3 Ownership Rules:

  1. Each value has a single owner
  2. When the owner goes out of scope, the value is automatically dropped
  3. Ownership is either moved or borrowed

  Ownership Move:
  +----------+          +----------+
  | let s1 = |          | let s2 = |
  | String:: |          | s1;      |
  | from("hi")          |          |
  +----+-----+          +----+-----+
       |                     |
       |  move               |
       v                     v
  +----------+          +----------+
  | s1 (invalid)|       | s2 (valid) |
  | ptr: --  |          | ptr: --+ |
  | len: 2   |          | len: 2 | |
  | cap: 2   |          | cap: 2 | |
  +----------+          +--------+-+
                                 |
                                 v
                          +----------+
                          | Heap     |
                          | "hi"     |
                          +----------+

  Borrowing:
  - Immutable borrow (&T): Multiple simultaneous OK, no modification
  - Mutable borrow (&mut T): Only one, modification allowed
  - Immutable and mutable borrows cannot coexist
  -> Prevents data races at compile time
```

---

## 7. Memory Fragmentation and Optimization

### 7.1 External Fragmentation vs Internal Fragmentation

```
  [External Fragmentation]
  Total free memory is sufficient, but contiguous space is lacking

  malloc(120) request:
  +----+--------+----+------+----+----------+
  |Used| Free   |Used| Free |Used| Free     |
  |    | 64B    |    | 48B  |    | 80B      |
  +----+--------+----+------+----+----------+
  Free total: 64 + 48 + 80 = 192B >= 120B but
  no contiguous 120B available -> allocation fails!

  [Internal Fragmentation]
  Wasted space within an allocated block

  malloc(20) request -> Allocator assigns a 32B block
  +--------------------------------+
  | User data 20B | Remainder 12B  |  <- 12B is wasted (internal fragmentation)
  +--------------------------------+

  Buddy System case:
  malloc(33) -> 64B allocated -> 31B wasted (48% internal fragmentation)
```

### 7.2 Compaction

A technique that resolves external fragmentation by moving in-use objects to create contiguous free space. Widely used in language runtimes with GC (Java, .NET, Go).

**Types of Compaction:**

| Method | Description | Advantages | Disadvantages |
|:---|:---|:---|:---|
| Arbitrary relocation | Move objects to any position | Optimal placement possible | High reference update cost |
| Sliding | Slide objects in one direction | Preserves relative order | Requires 2 passes |
| Copying | Copy live objects to a different region | Fast (1 pass) | Requires 2x memory |

### 7.3 Memory Pool Pattern

In scenarios with frequent allocation/deallocation, using a dedicated memory pool instead of a general-purpose allocator can significantly improve performance.

```c
/* Code Example 6: Fixed-size memory pool implementation */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct memory_pool {
    size_t block_size;      /* Size of each block */
    size_t pool_size;       /* Number of blocks in the pool */
    uint8_t *memory;        /* Memory region */
    uint8_t *free_list;     /* Head of the free list */
    size_t allocated_count; /* Number of allocated blocks */
} memory_pool_t;

/* Pool initialization */
memory_pool_t *pool_create(size_t block_size, size_t pool_size) {
    /* Block size must be at least pointer size */
    if (block_size < sizeof(void *)) {
        block_size = sizeof(void *);
    }

    memory_pool_t *pool = (memory_pool_t *)malloc(sizeof(memory_pool_t));
    pool->block_size = block_size;
    pool->pool_size = pool_size;
    pool->allocated_count = 0;

    /* Allocate a contiguous memory region */
    pool->memory = (uint8_t *)malloc(block_size * pool_size);

    /* Build the free list (pointer to next block at the head of each block) */
    pool->free_list = pool->memory;
    for (size_t i = 0; i < pool_size - 1; i++) {
        uint8_t *current = pool->memory + i * block_size;
        uint8_t *next = pool->memory + (i + 1) * block_size;
        *(uint8_t **)current = next;
    }
    /* The last block's next is NULL */
    *(uint8_t **)(pool->memory + (pool_size - 1) * block_size) = NULL;

    return pool;
}

/* Block acquisition O(1) */
void *pool_alloc(memory_pool_t *pool) {
    if (pool->free_list == NULL) {
        return NULL;  /* Pool exhausted */
    }
    void *block = pool->free_list;
    pool->free_list = *(uint8_t **)pool->free_list;
    pool->allocated_count++;
    return block;
}

/* Block return O(1) */
void pool_free(memory_pool_t *pool, void *block) {
    *(uint8_t **)block = pool->free_list;
    pool->free_list = (uint8_t *)block;
    pool->allocated_count--;
}

/* Pool destruction */
void pool_destroy(memory_pool_t *pool) {
    free(pool->memory);
    free(pool);
}

/* Usage example: Game engine particle system */
typedef struct particle {
    float x, y, z;
    float vx, vy, vz;
    float lifetime;
    int active;
} particle_t;

int main(void) {
    /* Create a pool for 10000 particles */
    memory_pool_t *particle_pool =
        pool_create(sizeof(particle_t), 10000);

    printf("Pool created: block_size=%zu, capacity=%zu\n",
           particle_pool->block_size, particle_pool->pool_size);

    /* Rapidly allocate particles */
    particle_t *particles[100];
    for (int i = 0; i < 100; i++) {
        particles[i] = (particle_t *)pool_alloc(particle_pool);
        particles[i]->x = (float)i;
        particles[i]->active = 1;
    }

    printf("Allocated: %zu\n", particle_pool->allocated_count);

    /* Return particles */
    for (int i = 0; i < 100; i++) {
        pool_free(particle_pool, particles[i]);
    }

    printf("After return: %zu\n", particle_pool->allocated_count);

    pool_destroy(particle_pool);
    return 0;
}
```

### 7.4 Huge Pages

With standard 4KB pages, applications that use large amounts of memory (databases, JVM) suffer frequent TLB misses. Huge Pages (2MB / 1GB) reduce the number of TLB entries needed, mitigating address translation overhead.

```bash
# Checking and configuring Huge Pages
cat /proc/meminfo | grep Huge
# HugePages_Total:     128
# HugePages_Free:       64
# HugePages_Rsvd:       32
# Hugepagesize:       2048 kB

# Check Transparent Huge Pages (THP) status
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never

# Databases (Redis, PostgreSQL) often disable THP
# -> To avoid latency variance (from THP compaction processing)
```

| Item | Regular Page (4KB) | Huge Page (2MB) | Giant Page (1GB) |
|:---|:---:|:---:|:---:|
| Page size | 4 KB | 2 MB | 1 GB |
| 1 TLB entry covers | 4 KB | 2 MB | 1 GB |
| Entries to map 1GB | 262,144 | 512 | 1 |
| Use case | General purpose | DB, JVM | HPC, large-scale DB |

---

## 8. Memory Leaks and Debugging

### 8.1 Classification of Memory Leaks

A memory leak is a phenomenon where "the reference to allocated memory is lost, making it impossible to free." This becomes a critical issue for long-running server processes.

| Leak Type | Cause | Detection Difficulty | Languages |
|:---|:---|:---:|:---|
| Direct leak | Forgetting free/delete | Low | C, C++ |
| Indirect leak | Unreachable memory via leaked pointer | Medium | C, C++ |
| Circular reference | Mutual references prevent count from reaching 0 | Medium | Python, Swift, JS |
| Event listener leak | Missing removeEventListener call | High | JavaScript |
| Closure leak | Closure captures large scope | High | JS, Python, Go |
| Cache leak | Cache that accumulates data without limit | High | All languages |
| Thread-local leak | Accumulation within thread pools | High | Java, Go |

### 8.2 Detection with Valgrind / AddressSanitizer

```c
/* Sample code for leak detection (leak_example.c) */
#include <stdlib.h>
#include <string.h>

void direct_leak(void) {
    /* Direct leak: malloc without free */
    int *data = (int *)malloc(sizeof(int) * 100);
    data[0] = 42;
    /* Forgot free(data)! */
}

void indirect_leak(void) {
    /* Indirect leak: losing the head of a linked list */
    struct node {
        int value;
        struct node *next;
    };

    struct node *head = (struct node *)malloc(sizeof(struct node));
    head->value = 1;
    head->next = (struct node *)malloc(sizeof(struct node));
    head->next->value = 2;
    head->next->next = NULL;

    /* Only free head -> head->next leaks */
    /* Correct: traverse the entire list and free */
    free(head);  /* head->next is an indirect leak */
}

int main(void) {
    direct_leak();
    indirect_leak();
    return 0;
}
```

```bash
# --- Detect memory leaks with Valgrind ---
gcc -g -O0 leak_example.c -o leak_example
valgrind --leak-check=full --show-leak-kinds=all ./leak_example

# Valgrind output example:
# ==12345== HEAP SUMMARY:
# ==12345==   in use at exit: 416 bytes in 2 blocks
# ==12345==   total heap usage: 3 allocs, 1 frees, 432 bytes allocated
# ==12345==
# ==12345== 400 bytes in 1 blocks are definitely lost
# ==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/...)
# ==12345==    by 0x401156: direct_leak (leak_example.c:6)
# ==12345==    by 0x4011A3: main (leak_example.c:32)
# ==12345==
# ==12345== 16 bytes in 1 blocks are indirectly lost
# ==12345==    at 0x4C2FB0F: malloc (in /usr/lib/valgrind/...)
# ==12345==    by 0x401178: indirect_leak (leak_example.c:17)

# --- Detect memory errors with AddressSanitizer (ASan) ---
gcc -fsanitize=address -fno-omit-frame-pointer -g leak_example.c -o leak_asan
./leak_asan

# --- Use LeakSanitizer standalone ---
gcc -fsanitize=leak -g leak_example.c -o leak_lsan
./leak_lsan
```

### 8.3 JavaScript Memory Leak Patterns

```javascript
// Code Example 7: Typical memory leak patterns in JavaScript

// --- Pattern 1: Unremoved event listeners ---
class LeakyComponent {
    constructor() {
        this.data = new Array(1000000).fill('x'); // Large data
        // Register event listener
        this.handler = () => this.handleResize();
        window.addEventListener('resize', this.handler);
    }

    handleResize() {
        console.log('resize', this.data.length);
    }

    // Leak if destroy is not called
    destroy() {
        window.removeEventListener('resize', this.handler);
    }
}

// --- Pattern 2: Capture by closure ---
function createLeak() {
    const hugeArray = new Array(1000000).fill('leak');

    // hugeArray is retained as long as this function lives
    return function() {
        // Even without directly using hugeArray,
        // variables in the same scope may be captured
        console.log('closure alive');
    };
}

// --- Pattern 3: Unbounded cache ---
const cache = new Map();
function addToCache(key, value) {
    // Cache grows indefinitely
    cache.set(key, value);
}

// Fix: Use WeakMap or LRU cache
const weakCache = new WeakMap(); // Entry disappears when key is GC'd

// Fix: LRU cache (with maximum size limit)
class LRUCache {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.cache = new Map();
    }

    get(key) {
        if (this.cache.has(key)) {
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value); // Move to end
            return value;
        }
        return undefined;
    }

    set(key, value) {
        if (this.cache.has(key)) {
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // Delete the oldest entry
            const oldestKey = this.cache.keys().next().value;
            this.cache.delete(oldestKey);
        }
        this.cache.set(key, value);
    }
}

// --- Debugging steps with Chrome DevTools ---
// 1. Memory tab -> "Take heap snapshot"
// 2. Perform operations
// 3. Take another snapshot
// 4. Check differences in "Comparison" view
// 5. Identify what is retaining objects in the Retainers panel
```

### 8.4 Memory Profiling with Go's pprof

```go
// Code Example 8: Go memory profiling
package main

import (
    "fmt"
    "net/http"
    _ "net/http/pprof" // Enable pprof endpoints
    "runtime"
    "time"
)

// Cache that intentionally accumulates memory
var leakyCache = make(map[string][]byte)

func simulateWork() {
    for i := 0; ; i++ {
        // Unlimited addition to cache (cause of leak)
        key := fmt.Sprintf("key-%d", i)
        leakyCache[key] = make([]byte, 1024) // Accumulates 1KB at a time

        if i%1000 == 0 {
            var m runtime.MemStats
            runtime.ReadMemStats(&m)
            fmt.Printf("Alloc=%v MiB, Sys=%v MiB, NumGC=%v\n",
                m.Alloc/1024/1024,
                m.Sys/1024/1024,
                m.NumGC,
            )
        }
        time.Sleep(time.Millisecond)
    }
}

func main() {
    // Start pprof HTTP server
    go func() {
        fmt.Println("pprof: http://localhost:6060/debug/pprof/")
        http.ListenAndServe(":6060", nil)
    }()

    simulateWork()
}

// Profiling commands:
// go tool pprof http://localhost:6060/debug/pprof/heap
// (pprof) top 10        <- Top 10 by memory usage
// (pprof) web           <- Display graph in browser
// (pprof) list main     <- Source code annotated display
```

---

## 9. Language Runtimes and Memory Models

### 9.1 Memory Management Approach Comparison by Language

Memory management approaches differ significantly by programming language. Below is a comparison of major languages.

| Language | Management Method | Heap Allocation | Deallocation Timing | Stack Allocation | Safety Guarantee |
|:---|:---|:---|:---|:---|:---|
| C | Manual (malloc/free) | Explicit | Explicit (free) | Automatic | None |
| C++ | Manual + RAII | new/make_unique | Destructor | Automatic | Partial via RAII |
| Rust | Ownership + Borrowing | Box, Vec, etc. | On scope exit | Automatic | Compile-time guarantee |
| Java | Generational GC | new | GC collects | JIT-optimized | Runtime guarantee |
| Go | Concurrent GC | make, new | GC collects | Escape analysis | Runtime guarantee |
| Python | Ref counting + GC | Implicit | Ref count=0 or GC | None (all heap) | Runtime guarantee |
| JavaScript | Generational GC (V8) | Implicit | GC collects | JIT-optimized | Runtime guarantee |
| Swift | ARC | Implicit | Ref count=0 | Value types on stack | Compile-time ARC |
| Zig | Manual + Allocator abstraction | Via allocator | Explicit (defer) | Automatic | Compile-time checks |

### 9.2 C++ RAII and Smart Pointers

C++ improves the safety of manual memory management through the RAII (Resource Acquisition Is Initialization) pattern.

```cpp
// Code Example 9: Memory management with C++ smart pointers
#include <memory>
#include <vector>
#include <iostream>

class Resource {
public:
    Resource(int id) : id_(id) {
        std::cout << "Resource " << id_ << " acquired\n";
    }
    ~Resource() {
        std::cout << "Resource " << id_ << " released\n";
    }
    void use() { std::cout << "Using Resource " << id_ << "\n"; }
private:
    int id_;
};

void demonstrate_smart_pointers() {
    // --- unique_ptr: Exclusive ownership (no copy, move only) ---
    {
        auto r1 = std::make_unique<Resource>(1);
        r1->use();
        // Automatically freed at scope exit
        // auto r2 = r1;  // Compile error! Cannot copy
        auto r2 = std::move(r1);  // Move is possible
        // r1 is now nullptr
    } // r2's destructor releases the Resource

    // --- shared_ptr: Shared ownership (reference counting) ---
    {
        auto r3 = std::make_shared<Resource>(3);
        std::cout << "ref count: " << r3.use_count() << "\n"; // 1

        {
            auto r4 = r3;  // Share
            std::cout << "ref count: " << r3.use_count() << "\n"; // 2
        } // r4 scope ends: count 2->1

        std::cout << "ref count: " << r3.use_count() << "\n"; // 1
    } // r3 scope ends: count 1->0 -> released

    // --- weak_ptr: Preventing circular references ---
    {
        auto parent = std::make_shared<Resource>(5);
        std::weak_ptr<Resource> weak_ref = parent;

        if (auto locked = weak_ref.lock()) {
            locked->use();  // Still valid
        }

        parent.reset();  // Release

        if (auto locked = weak_ref.lock()) {
            locked->use();  // Won't reach here
        } else {
            std::cout << "Resource already released\n";
        }
    }

    // --- unique_ptr + custom deleter ---
    {
        auto file_deleter = [](FILE *f) {
            if (f) {
                std::cout << "Closing file\n";
                fclose(f);
            }
        };
        std::unique_ptr<FILE, decltype(file_deleter)>
            file(fopen("/tmp/test.txt", "w"), file_deleter);
        if (file) {
            fprintf(file.get(), "Hello RAII\n");
        }
    } // File is automatically closed
}

int main() {
    demonstrate_smart_pointers();
    return 0;
}
```

### 9.3 Go's Escape Analysis

The Go compiler uses Escape Analysis to automatically determine whether a variable should be placed on the stack or the heap.

```go
// Go escape analysis examples
package main

import "fmt"

// Placed on stack (reference doesn't escape the function)
func stackAlloc() int {
    x := 42
    return x  // Value copy -> x stays on stack
}

// Placed on heap (pointer escapes the function = escape)
func heapAlloc() *int {
    x := 42
    return &x  // Pointer escapes -> x placed on heap
}

// Escapes via interface
func interfaceEscape() {
    x := 42
    fmt.Println(x)  // Println(interface{}) -> x escapes
}

// Command to check escape analysis:
// go build -gcflags="-m -m" main.go
// ./main.go:13:2: x escapes to heap:
// ./main.go:13:2:   flow: ~r0 = &x:
// ./main.go:13:2:     from &x (address-of) at ./main.go:14:9
```

### 9.4 Memory Allocation in Real-Time Systems

In real-time systems, the Worst-Case Execution Time (WCET) of memory allocation must be guaranteed.

| Requirement | General-Purpose System | Real-Time System |
|:---|:---|:---|
| Allocation time | Average high speed is sufficient | Worst-case time must be guaranteed |
| GC pauses | Tolerable | Intolerable |
| Memory fragmentation | Gradual increase acceptable, reboot to recover | Zero fragmentation required for long-term operation |
| Allocator used | malloc, jemalloc, etc. | Memory pool, TLSF |
| Behavior on failure | OOM Killer, etc. | Fail-safe operation is mandatory |

**TLSF (Two-Level Segregated Fit):**

An allocator designed for real-time systems where both allocation and deallocation are guaranteed O(1). It performs free list search in constant time using a two-level index of bitmaps and segment lists. Adopted in mission-critical systems such as aerospace, automotive, and medical devices.

---

## 10. Anti-Patterns and Design Principles

### 10.1 Anti-Pattern 1: The "malloc and forget" Pattern

**Problem:** Dynamically allocating memory within a function and forgetting to free it on error paths, etc.

```c
/* Anti-pattern: Acquiring multiple resources with incomplete release */
int process_data_BAD(const char *filename) {
    FILE *file = fopen(filename, "r");
    if (!file) return -1;

    char *buffer = (char *)malloc(4096);
    if (!buffer) {
        /* Forgot to close file! */
        return -1;
    }

    int *results = (int *)malloc(sizeof(int) * 1000);
    if (!results) {
        /* Forgot to free buffer! */
        /* Also forgot to close file! */
        return -1;
    }

    /* ... processing ... */

    free(results);
    free(buffer);
    fclose(file);
    return 0;
}

/* Fix: Unified cleanup using goto (Linux kernel style) */
int process_data_GOOD(const char *filename) {
    int ret = -1;
    FILE *file = NULL;
    char *buffer = NULL;
    int *results = NULL;

    file = fopen(filename, "r");
    if (!file) goto cleanup;

    buffer = (char *)malloc(4096);
    if (!buffer) goto cleanup;

    results = (int *)malloc(sizeof(int) * 1000);
    if (!results) goto cleanup;

    /* ... processing ... */
    ret = 0;

cleanup:
    free(results);   /* free(NULL) is safe */
    free(buffer);    /* free(NULL) is safe */
    if (file) fclose(file);
    return ret;
}
```

**Lessons:**
- In C, the `goto cleanup` pattern is recommended (also adopted in Linux kernel coding conventions)
- In C++, RAII (smart pointers, file streams) solves this automatically
- `free(NULL)` is defined as safe by the C standard, so NULL checks are unnecessary

### 10.2 Anti-Pattern 2: The "malloc a large temp buffer every iteration" Pattern

**Problem:** Calling malloc/free for a large buffer in every loop iteration, causing unnecessary overhead.

```c
/* Anti-pattern: Repeated allocation within a loop */
void process_items_BAD(int *items, int count) {
    for (int i = 0; i < count; i++) {
        /* Allocate and free 4KB temp buffer each time */
        char *tmp = (char *)malloc(4096);
        snprintf(tmp, 4096, "Processing item %d", items[i]);
        /* ... processing using tmp ... */
        free(tmp);
        /* -> If count is 1 million, malloc/free is called 1 million times */
    }
}

/* Fix 1: Allocate once outside the loop */
void process_items_GOOD1(int *items, int count) {
    char *tmp = (char *)malloc(4096);
    if (!tmp) return;

    for (int i = 0; i < count; i++) {
        snprintf(tmp, 4096, "Processing item %d", items[i]);
        /* ... processing using tmp ... */
    }
    free(tmp);
}

/* Fix 2: Use VLA or stack for fixed sizes */
void process_items_GOOD2(int *items, int count) {
    char tmp[4096];  /* Allocated on stack (fixed size) */

    for (int i = 0; i < count; i++) {
        snprintf(tmp, sizeof(tmp), "Processing item %d", items[i]);
        /* ... processing using tmp ... */
    }
    /* Automatically freed */
}
```

**Lessons:**
- malloc/free in loops incurs significant overhead (system calls, lock contention, fragmentation)
- Allocate buffers once outside the loop when possible
- Small fixed-size buffers (~few KB) are faster on the stack
- Consider memory pools when large buffers are needed

### 10.3 Design Principles Summary

| Principle | Description | Applicable Scenarios |
|:---|:---|:---|
| RAII | Tie resource acquisition to object initialization | C++, Rust |
| Clear ownership | Limit memory ownership to a single owner | All languages |
| Buffer reuse | Avoid frequent allocations by reusing buffers | Hot paths |
| Memory pool | Manage same-size allocations in a dedicated pool | Games, servers |
| Arena allocator | Bulk allocate per phase, bulk deallocate | Compilers, parsers |
| Copy-on-write | Share memory until actual modification occurs | fork, strings |
| Zero-copy | Pass references instead of copying data | Network, I/O |

---

## 11. Practical Exercises (3 Levels)

### Exercise 1: [Basic] Verifying Memory Layout

**Task:** Identify which memory segment each variable in the following C program is placed in.

```c
#include <stdlib.h>
#include <string.h>

int global_initialized = 42;        /* -> ? */
int global_uninitialized;           /* -> ? */
const char *literal = "hello";      /* -> ? (what about the pointer itself?) */

void function_example(int param) {  /* param -> ? */
    int local = 10;                 /* -> ? */
    static int persistent = 5;     /* -> ? */
    int *dynamic = malloc(100);    /* dynamic -> ?, *dynamic -> ? */
    char buf[256];                 /* -> ? */

    free(dynamic);
}
```

<details>
<summary>Show Answer</summary>

| Variable | Segment | Reason |
|:---|:---|:---|
| `global_initialized` | Data | Initialized global variable |
| `global_uninitialized` | BSS | Uninitialized global variable (zero-initialized) |
| `literal` (pointer itself) | Data | Initialized global pointer |
| `"hello"` (string body) | Text (rodata) | String literal (read-only) |
| `param` | Stack | Function argument |
| `local` | Stack | Local variable |
| `persistent` | Data | Static variable (initialized) |
| `dynamic` (pointer itself) | Stack | Local variable |
| `*dynamic` (pointed-to data) | Heap | Dynamically allocated via malloc |
| `buf[256]` | Stack | Local array |

</details>

### Exercise 2: [Intermediate] Detecting and Fixing Memory Leaks

**Task:** There are 3 memory leaks in the following code. Identify and fix all of them.

```c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    char *name;
    int *scores;
    int score_count;
} Student;

Student *create_student(const char *name, int count) {
    Student *s = (Student *)malloc(sizeof(Student));
    s->name = strdup(name);
    s->scores = (int *)malloc(sizeof(int) * count);
    s->score_count = count;
    return s;
}

void process_students(void) {
    Student *alice = create_student("Alice", 5);
    Student *bob = create_student("Bob", 3);

    /* Set Alice's scores */
    for (int i = 0; i < alice->score_count; i++) {
        alice->scores[i] = 80 + i;
    }

    /* Early return if error occurs */
    if (alice->scores[0] < 50) {
        free(alice);  /* Leak! name and scores not freed */
        return;       /* bob also leaks! */
    }

    /* Change Bob's name */
    bob->name = strdup("Robert");  /* Leak! Original "Bob" is lost */

    free(alice->scores);
    free(alice->name);
    free(alice);
    free(bob->scores);
    free(bob->name);
    free(bob);
}
```

<details>
<summary>Show Answer</summary>

**Leak locations:**

1. **On early return, `alice->name` and `alice->scores` are not freed**
2. **On early return, all of `bob` (name, scores, body) is not freed**
3. **`bob->name = strdup("Robert")` causes the original `"Bob"` memory to leak**

**Fixed version:**

```c
void destroy_student(Student *s) {
    if (s) {
        free(s->scores);
        free(s->name);
        free(s);
    }
}

void process_students_fixed(void) {
    Student *alice = create_student("Alice", 5);
    Student *bob = create_student("Bob", 3);

    for (int i = 0; i < alice->score_count; i++) {
        alice->scores[i] = 80 + i;
    }

    if (alice->scores[0] < 50) {
        destroy_student(alice);  /* Fix 1: Fully release */
        destroy_student(bob);    /* Fix 2: Release bob too */
        return;
    }

    /* Fix 3: Free original name before replacement */
    free(bob->name);
    bob->name = strdup("Robert");

    destroy_student(alice);
    destroy_student(bob);
}
```

</details>

### Exercise 3: [Advanced] Designing a Custom Allocator

**Task:** Design and implement an arena allocator that satisfies the following requirements.

**Requirements:**
1. Allocate a large memory block (arena) in bulk during initialization
2. `arena_alloc(size)` performs "bump" allocation from the arena (just advance a pointer)
3. Individual `free` is unnecessary. `arena_reset()` resets the entire arena at once
4. `arena_destroy()` releases the entire arena
5. Guarantee alignment (8-byte boundary)

**Hint:** This pattern is widely used in compiler AST construction and per-request server processing.

<details>
<summary>Show Sample Answer</summary>

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define ARENA_ALIGN 8
#define ALIGN_UP(x, a) (((x) + (a) - 1) & ~((a) - 1))

typedef struct arena {
    uint8_t *base;      /* Base of the arena */
    size_t size;        /* Total size of the arena */
    size_t offset;      /* Next allocation position */
} arena_t;

/* Arena initialization */
arena_t *arena_create(size_t size) {
    arena_t *arena = (arena_t *)malloc(sizeof(arena_t));
    if (!arena) return NULL;

    arena->base = (uint8_t *)malloc(size);
    if (!arena->base) {
        free(arena);
        return NULL;
    }
    arena->size = size;
    arena->offset = 0;
    return arena;
}

/* Bump allocation O(1) */
void *arena_alloc(arena_t *arena, size_t size) {
    size_t aligned_offset = ALIGN_UP(arena->offset, ARENA_ALIGN);
    if (aligned_offset + size > arena->size) {
        return NULL;  /* Arena exhausted */
    }
    void *ptr = arena->base + aligned_offset;
    arena->offset = aligned_offset + size;
    return ptr;
}

/* Full reset O(1) -- no individual free needed */
void arena_reset(arena_t *arena) {
    arena->offset = 0;
}

/* Arena destruction */
void arena_destroy(arena_t *arena) {
    if (arena) {
        free(arena->base);
        free(arena);
    }
}

/* Usage example: Compiler AST construction */
typedef struct ast_node {
    int type;
    struct ast_node *left;
    struct ast_node *right;
    int value;
} ast_node_t;

int main(void) {
    arena_t *arena = arena_create(1024 * 1024); /* 1MB arena */

    /* Rapidly allocate AST nodes from the arena */
    ast_node_t *root = (ast_node_t *)arena_alloc(arena, sizeof(ast_node_t));
    root->type = 1;
    root->value = 42;
    root->left = (ast_node_t *)arena_alloc(arena, sizeof(ast_node_t));
    root->right = (ast_node_t *)arena_alloc(arena, sizeof(ast_node_t));

    printf("Usage: %zu / %zu bytes (%.1f%%)\n",
           arena->offset, arena->size,
           (double)arena->offset / arena->size * 100);

    /* Request processing complete -> bulk reset (no individual free needed!) */
    arena_reset(arena);
    printf("After reset: %zu / %zu bytes\n", arena->offset, arena->size);

    arena_destroy(arena);
    return 0;
}
```

**Arena Allocator Advantages:**
- Allocation: Pointer addition only -> O(1), lock-free
- Deallocation: Bulk reset only -> O(1), no individual free needed
- Fragmentation: Does not occur (sequential use of contiguous space)
- Cache efficiency: Extremely high (contiguous memory access)

</details>

---

## 12. FAQ

### Q1: Is malloc thread-safe?

glibc's ptmalloc2 is thread-safe. It uses mutexes internally to protect arenas (memory pools) and reduces lock contention by assigning different arenas to different threads. However, in high-thread-count environments, arena contention can become a bottleneck. jemalloc and tcmalloc can execute most allocation operations lock-free through thread-local caches, achieving high concurrency. If performance is an issue, the allocator can be swapped using `LD_PRELOAD`.

```bash
# Run with jemalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so ./my_program

# Run with tcmalloc
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so ./my_program
```

### Q2: Can Rust's ownership completely prevent memory leaks?

No. Memory leaks can still occur in Rust. There are multiple patterns that cause leaks, including circular references with `Rc<T>`, intentional drop suppression with `std::mem::forget`, explicit leaking with `Box::leak`, and memory held by non-terminating threads. However, what Rust guarantees is **memory safety** (prevention of Use-After-Free, double-free, and data races). Memory leaks are not classified as a safety issue. Circular references should be addressed using `Weak<T>`.

### Q3: Is there a way to explicitly free memory in GC languages?

In most GC languages, you can request GC execution as a hint, but immediate deallocation cannot be guaranteed. Java's `System.gc()` is merely a hint "please run GC," and the JVM may ignore it. Go's `runtime.GC()` executes GC synchronously, but should not be called routinely. Python's `gc.collect()` collects circular references, but objects with a reference count of 0 are immediately collected at the point of `del`. In practice, setting references to large objects to `null` / `None` / `nil` to make them GC-collectible is recommended.

### Q4: What is the criterion for choosing between mmap and brk/sbrk?

glibc's malloc uses `MMAP_THRESHOLD` (128KB) as the default threshold. Allocations of 128KB or less extend the heap with `sbrk`, while allocations exceeding 128KB allocate an independent virtual memory region with `mmap(MAP_ANONYMOUS)`. The advantage of `mmap` is that memory can be immediately returned to the kernel via `munmap`. On the other hand, heap space extended by `sbrk` can only shrink from the end; if blocks in the middle are in use, memory cannot be returned to the kernel. This threshold can be changed with `mallopt(M_MMAP_THRESHOLD, size)`.

### Q5: Which process does the OOM Killer terminate?

Linux's OOM Killer selects a process based on `/proc/[pid]/oom_score`. Higher scores mean higher likelihood of being terminated. The score is primarily based on the process's memory usage, but can be adjusted via `oom_score_adj` (-1000 to +1000). Critical processes (databases, etc.) can be set to `oom_score_adj = -1000` to exclude them from the OOM Killer. However, protecting too many processes may prevent the OOM Killer from terminating appropriate processes during an OOM event, potentially causing the entire system to hang.

```bash
# Check the current process's OOM score
cat /proc/self/oom_score

# Protect an important process from OOM Killer
echo -1000 > /proc/$(pidof my_database)/oom_score_adj
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

## 13. Summary

### Overall Overview

This chapter comprehensively covered memory allocation strategies, starting from process memory layout, through algorithms, production-level allocators, kernel physical memory management, garbage collection, fragmentation countermeasures, debugging techniques, and the characteristics of each language runtime.

### Comprehensive Concept Summary

| Concept | Key Points |
|:---|:---|
| Memory layout | 6 regions: text, data, BSS, heap, mmap, stack |
| Stack vs Heap | Stack: O(1) allocation, auto-release / Heap: flexible but high management cost |
| Allocation algorithms | First Fit (fast), Best Fit (less fragmentation), Buddy (kernel), Segregated (modern) |
| ptmalloc2 | Linux standard. Arena + hierarchy of fastbin/smallbin/largebin |
| jemalloc | Low fragmentation + comprehensive statistics. Adopted by Redis, Firefox |
| tcmalloc | Extremely fast small object allocation via thread cache. Adopted by Chrome |
| mimalloc | Segment-based + free list sharding. High overall performance |
| Buddy Allocator | Power-of-2 block management. Linux kernel physical page allocation |
| SLAB/SLUB | Kernel small object management. Cache per object type |
| Mark & Sweep | Determine garbage by reachability from root. Basic but causes STW |
| Generational GC | Based on weak generational hypothesis. Minor GC (frequent) + Major GC (infrequent) |
| Reference counting | Immediate reclamation possible but weak against circular references. Adopted by Python, Swift (ARC) |
| Tri-color marking | Foundation of concurrent GC. White/gray/black marking. Adopted by Go |
| Rust ownership | No GC. Compile-time memory safety guarantee. Zero cost |
| External fragmentation | Total free sufficient but contiguous space lacking. Mitigated by compaction |
| Internal fragmentation | Waste within allocated blocks. Mitigated by fine-grained size classes |
| Memory pool | O(1) fixed-size allocation. For games, servers |
| Arena allocator | Bump allocation + bulk deallocation. For compilers, request processing |
| RAII | Tie resource acquisition and release to object lifetime |
| Demand Paging | Defer physical page allocation until actual access |

### Next Steps

To further deepen your knowledge of memory allocation, we recommend proceeding to the following topics:

- **Virtual Memory and Paging:** Page tables, TLB, page replacement algorithms
- **Memory-Mapped I/O:** Accelerating file access via mmap
- **NUMA Architecture:** Optimizing memory placement in multi-socket systems
- **Persistent Memory (PMEM):** Non-volatile memory programming with Intel Optane, etc.

---

## Recommended Next Guides


---

## 14. References

1. Silberschatz, A., Galvin, P. B., & Gagne, G. *Operating System Concepts*, 10th Edition, Chapter 9: Main Memory. Wiley, 2018. --- Textbook-level explanation of memory management. Systematically covers contiguous allocation, paging, and segmentation.

2. Tanenbaum, A. S., & Bos, H. *Modern Operating Systems*, 4th Edition, Chapter 3: Memory Management. Pearson, 2014. --- Detailed explanation of Buddy System, SLAB allocator, and virtual memory.

3. Evans, J. "A Scalable Concurrent malloc(3) Implementation for FreeBSD." BSDCan, 2006. --- Original paper detailing jemalloc's design philosophy and implementation.

4. Ghemawat, S. & Menage, P. "TCMalloc: Thread-Caching Malloc." Google Performance Tools Documentation. --- tcmalloc architecture, thread cache and page heap design.

5. Leijen, D., Zorn, B., & de Moura, L. "Mimalloc: Free List Sharding in Action." Microsoft Research, 2019. --- mimalloc design principles and benchmarks demonstrating the effectiveness of free list sharding.

6. Bonwick, J. "The Slab Allocator: An Object-Caching Kernel Memory Allocator." USENIX Summer 1994 Technical Conference. --- Original SLAB allocator paper. Proposed the concept of object caching.

7. Jones, R., Hosking, A., & Moss, E. *The Garbage Collection Handbook: The Art of Automatic Memory Management*. CRC Press, 2011. --- Comprehensive reference on GC algorithms. Covers mark & sweep, generational, and concurrent GC.

8. Klabnik, S. & Nichols, C. *The Rust Programming Language*. No Starch Press, 2019. Chapter 4: Understanding Ownership. --- Official explanation of Rust ownership, borrowing, and lifetimes.

9. Love, R. *Linux Kernel Development*, 3rd Edition, Chapter 12: Memory Management. Addison-Wesley, 2010. --- Explanation of Linux kernel's Buddy Allocator, SLAB, and vmalloc.

10. Masmano, M., Ripoll, I., Crespo, A., & Real, J. "TLSF: A New Dynamic Memory Allocator for Real-Time Systems." 16th Euromicro Conference on Real-Time Systems (ECRTS), 2004. --- Original TLSF allocator paper. O(1) guaranteed allocation algorithm.

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://ja.wikipedia.org/) - Overview of technical concepts
