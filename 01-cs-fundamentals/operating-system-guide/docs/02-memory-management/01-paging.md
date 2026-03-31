# Paging -- Virtual Memory, Page Tables, TLB, Page Replacement, and Swapping

> **Paging** is a memory management scheme that divides physical memory into fixed-size "frames" and virtual address space into same-size "pages," dynamically mapping both through page tables. Nearly all modern general-purpose OSes are based on paging, supporting mechanisms such as process isolation, shared memory, demand paging, and swapping.

## Learning Objectives

- [ ] Diagram the translation process from virtual addresses to physical addresses
- [ ] Explain the structure and trade-offs of single-level and multi-level page tables
- [ ] Quantitatively discuss the role of the TLB (Translation Lookaside Buffer) and the penalty on miss
- [ ] Compare and implement page replacement algorithms such as LRU, Clock, and LFU
- [ ] Understand the relationship between swapping and demand paging and connect it to Linux memory management
- [ ] Analyze the impact of page size selection on system performance


## Prerequisites

Understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Virtual Memory](./00-virtual-memory.md)

---

## 1. Why Paging Is Needed

### 1.1 Limitations of Segmentation

Segmentation, which divides programs into logical units (code, data, stack, heap), provides intuitive memory management, but because segments are **variable-length**, it causes **External Fragmentation**.

```
Example of external fragmentation:

Physical memory (100KB):
+--------+
| A: 20KB|  <- Process A
+--------+
|Free:10KB| <- Leftover from A? No, a different gap
+--------+
| B: 30KB|  <- Process B
+--------+
|Free:15KB| <- Trace of C being freed
+--------+
| D: 25KB|  <- Process D
+--------+

Total free = 10KB + 15KB = 25KB
However, no contiguous 25KB region exists.
-> Cannot place 25KB Process E!

Solution 1: Compaction (memory relocation)
  -> All processes must be stopped and memory moved, which is very costly

Solution 2: Paging (fixed-size division)
  -> Eliminates external fragmentation in principle
```

### 1.2 Basic Idea of Paging

In paging, the virtual address space is divided into **Pages** and physical memory into **Frames**, both of fixed size. The typical page size is **4KB (4096 bytes = 2^12 bytes)**.

```
Basic structure of paging:

Virtual Address Space              Physical Memory
+----------+               +----------+
| Page 0   |-------------->| Frame 5  |
+----------+               +----------+
| Page 1   |------+        | Frame 1  |
+----------+      |        +----------+
| Page 2   |--+   |        | Frame 2  |<--+
+----------+  |   |        +----------+   |
| Page 3   |  |   +------->| Frame 3  |   |
+----------+  |            +----------+   |
| Page 4   |  |            | Frame 4  |   |
+----------+  +----------->| Frame 6  |   |
| Page 5   |---------------+----------+   |
+----------+               | Frame 7  |   |
|  ...     |               |  ...     |   |
| Page N   |---------------+-----------+--+
+----------+               +----------+

  The mapping from virtual pages to physical frames
  is maintained by the "page table."

  Contiguous virtual pages do NOT need to be physically contiguous
  -> No external fragmentation
```

**Why 4KB:**

| Criterion | Small Page (e.g., 512B) | Large Page (e.g., 64KB) |
|-----------|------------------------|------------------------|
| Internal fragmentation | Average 256B (small) | Average 32KB (large) |
| Page table size | Becomes enormous | Small |
| Disk I/O efficiency | Inefficient (many small transfers) | Efficient (bulk transfer) |
| Memory utilization | High (little waste) | Low (much unused space) |
| TLB coverage | Narrow | Wide |

4KB has been adopted as standard since the 1990s as a well-balanced value between small internal fragmentation and page table size/I/O efficiency. However, in modern large-memory environments, **Huge Pages (2MB / 1GB)** are also widely used.

---

## 2. Translation from Virtual Address to Physical Address

### 2.1 Address Translation Mechanism

A virtual address is decomposed into a **Virtual Page Number (VPN)** and an **Offset**.

```
32-bit virtual address (page size = 4KB = 2^12):

  31                    12 11                0
  +-----------------------+-----------------+
  |  VPN (20 bits)        | Offset (12 bits)|
  +-----------------------+-----------------+
       2^20 = 1,048,576 pages     4096 bytes

Translation steps:
  1. Extract VPN from virtual address: VPN = VA >> 12
  2. Convert VPN -> PFN (Physical Frame Number) via page table
  3. Construct physical address: PA = (PFN << 12) | Offset

Concrete example: Virtual address 0x00403A7C
  +-----------------------+-----------------+
  |  VPN = 0x00403        | Offset = 0xA7C  |
  +-----------------------+-----------------+
  Page table maps VPN 0x00403 -> PFN 0x0007B
  Physical address = 0x0007BA7C
  +-----------------------+-----------------+
  |  PFN = 0x0007B        | Offset = 0xA7C  |
  +-----------------------+-----------------+
```

### 2.2 Page Table Entry (PTE)

Each page table entry holds not only the frame number but also control bits related to protection and status.

```
x86 Page Table Entry (32-bit):

  31              12 11  9  8   7   6   5   4   3   2   1   0
  +-----------------+-----+---+---+---+---+---+---+---+---+---+
  | PFN (20 bits)   |Avail| G |PAT| D | A |PCD|PWT|U/S|R/W| P |
  +-----------------+-----+---+---+---+---+---+---+---+---+---+

  P   (Present)     : 1 = page exists in physical memory
                      0 = swapped out or unallocated
  R/W (Read/Write)  : 1 = writable, 0 = read-only
  U/S (User/Super)  : 1 = user-mode accessible, 0 = kernel only
  A   (Accessed)    : Automatically set by MMU on access (used in Clock algorithm)
  D   (Dirty)       : Automatically set on write (used for swap-out decisions)
  G   (Global)      : Not flushed from TLB on context switch
  PCD (Page Cache Disable) : Disable caching (used for MMIO regions)
  PWT (Page Write Through) : Write-through cache control

Why the Dirty bit is important:
  -> When swapping out a page, if Dirty = 0, no need to write back to disk
  -> Reduces I/O and greatly improves page replacement efficiency
```

---

## 3. Multi-level Page Tables

### 3.1 Problem with Single-Level Page Tables

With a 32-bit address space and 4KB pages, the page table has 2^20 = approximately 1 million entries. At 4 bytes per entry, **4MB of page table is needed per process**. However, the virtual address space actually used by a process is only a small portion, so the vast majority of entries are invalid (P=0), resulting in extreme memory waste.

### 3.2 Two-Level Page Table (x86 32-bit)

```
Two-level page table (x86 32-bit):

Virtual address (32-bit):
  31          22 21          12 11           0
  +-------------+--------------+-------------+
  | Dir (10 bits)| Table(10 bits)|Offset(12 bits)|
  +------+------+------+-------+-------------+
         |             |
         v             |
  +------------+       |
  | Page       |       |
  | Directory  |       |
  | (1024 items)|      |
  | +--------+ |       |
  | | Entry0 | |       |
  | +--------+ |       |
  | | Entry1 |-+---+   |
  | +--------+ |   |   |
  | |  ...   | |   |   |
  | +--------+ |   |   |
  +------------+   |   |
         CR3       v   |
                +------------+
                | Page       |
                | Table      |
                | (1024 items)|
                | +--------+ |    +----------+
                | | Entry0 | |    | Physical |
                | +--------+ |    | Frame    |
                | | Entry1 |-+--->| (4KB)    |
                | +--------+ |    |          |
                | |  ...   | |    +----------+
                | +--------+ |
                +------------+

Memory saving mechanism:
  Page tables for unused virtual address regions are
  not allocated (page directory entries are invalidated)

  Example: If a process uses only 8MB
    Page directory: 4KB (always needed)
    Page tables: 4KB x 2 = 8KB (8MB / 4MB per table)
    Total: 12KB  <- Dramatic savings compared to 4MB for single-level
```

### 3.3 Four-Level Page Table (x86-64)

In 64-bit environments, the virtual address space becomes vast, so a 4-level page table is used. However, only 48 bits (256TB) are actually used.

```
Four-level page table (x86-64, 48-bit virtual address):

Virtual address (64-bit):
  63    48 47    39 38    30 29    21 20    12 11     0
  +------+--------+--------+--------+--------+--------+
  |Sign  | PML4   |  PDPT  |  PD    |  PT    |Offset  |
  |ext.  | (9bit) | (9bit) | (9bit) | (9bit) |(12bit) |
  +------+---+----+---+----+---+----+---+----+--------+
             |        |        |        |
             v        v        v        v
  CR3 -> [PML4] -> [PDPT] -> [PD] -> [PT] -> Physical frame
         512 items  512 items 512 items 512 items

  Each table is 512 entries x 8 bytes = 4KB (fits in one page)

  Address coverage:
    1 PTE       = 4KB
    1 PT        = 512 x 4KB    = 2MB
    1 PD        = 512 x 2MB    = 1GB
    1 PDPT      = 512 x 1GB    = 512GB
    1 PML4      = 512 x 512GB  = 256TB

  Why 48 bits is sufficient:
    256TB of virtual address space is practically sufficient today, and
    making the page table hierarchy too deep increases address translation overhead.
    Intel has also defined 5-level (57-bit, LA57), extensible to 128PB.
```

### 3.4 Multi-level Page Table Comparison

| Property | 1-Level | 2-Level (x86-32) | 4-Level (x86-64) |
|----------|---------|-------------------|-------------------|
| Virtual address width | 32bit | 32bit | 48bit |
| Table entries | 2^20 | 2^10 x 2^10 | 4 stages x 2^9 |
| Minimum memory consumption | 4MB fixed | ~12KB | ~16KB |
| Maximum memory consumption | 4MB fixed | 4MB + 4KB | Theoretically huge |
| Memory references for addr. translation | 1 | 2 | 4 |
| Handling of empty pages | All entries held | Tables can be omitted | Tables can be omitted |
| Example OS adoption | Educational OS | Windows XP (32bit) | Linux, Windows 10/11 |

---

## 4. TLB (Translation Lookaside Buffer)

### 4.1 Why the TLB Is Needed

With a 4-level page table, one memory access requires **4 page table references + 1 data access = 5 total** memory accesses. This would degrade performance to 1/5. The TLB is a fast associative memory (CAM: Content-Addressable Memory) that caches recent address translation results, solving this problem.

```
Address translation speedup with TLB:

                    +---------+
Virtual Address ------>|  TLB    |
     |              | (fast)  |
     |              |VPN->PFN |
     |              +----+----+
     |                   |
     |          +--------+--------+
     |          |                 |
     |       TLB Hit          TLB Miss
     |     (1 cycle)       (tens to hundreds of cycles)
     |          |                 |
     |          v                 v
     |    Obtain physical   +----------+
     |    address instantly | Page     |
     |                     | Table    |
     |                     | Walk     |
     |                     | (4 refs) |
     |                     +----+-----+
     |                          |
     |                          v
     |                    Register result in TLB
     |                    + obtain physical address
     v
  Memory Access

Effective access time when TLB hit rate is 99%:
  TLB hit  = 1ns (TLB lookup) + 100ns (memory access) = 101ns
  TLB miss = 1ns + 4x100ns (page walk) + 100ns = 501ns
  Effective time = 0.99 x 101 + 0.01 x 501 = 99.99 + 5.01 = 105ns
  Overhead = (105 - 100) / 100 = 5%

  -> With 99% hit rate, overhead is only 5%
  -> If hit rate drops to 90%: 0.9x101 + 0.1x501 = 141ns -> 41% increase
```

### 4.2 TLB Structure

```
TLB Entry:
+-------+------+---+---+---+----+-----+
|  VPN  |  PFN | V | D | G |ASID|Prot |
+-------+------+---+---+---+----+-----+

  VPN  : Virtual Page Number (search key)
  PFN  : Physical Frame Number (search result)
  V    : Valid bit (whether this entry is valid)
  D    : Dirty bit (whether a write has occurred)
  G    : Global bit (shared across all processes, retained on context switch)
  ASID : Address Space Identifier (process identifier)
         -> ASID avoids TLB flush on context switch
  Prot : Protection bits (read/write/execute)

Typical TLB sizes:
  L1 ITLB (instruction) : 64-128 entries, 4-way set associative
  L1 DTLB (data)        : 64-72 entries, 4-way set associative
  L2 STLB (unified)     : 1024-2048 entries, 8-12-way

  Why is the TLB so small?
    The TLB is composed of CAM that searches all entries in parallel,
    and increasing entries causes power consumption and area to surge,
    while also reducing search speed. It can maintain high hit rates
    even when small, thanks to program "locality."
```

### 4.3 ASID (Address Space Identifier)

When a context switch occurs, the new process's page table has different VPN -> PFN mappings. Without ASID, all TLB entries must be flushed (invalidated), but with ASID, entries per process can be distinguished, avoiding the flush.

```
Context switch optimization with ASID:

Without ASID:
  Process A running -> TLB: [VPN=0x100->PFN=0x5, VPN=0x200->PFN=0x8, ...]
  Context switch -> Full TLB flush (all entries invalidated)
  Process B running -> TLB cold start (all misses)

With ASID:
  Process A (ASID=1) running -> TLB: [(ASID=1,VPN=0x100)->PFN=0x5, ...]
  Context switch -> No TLB flush needed
  Process B (ASID=2) running -> TLB: [(ASID=2,VPN=0x100)->PFN=0x3, ...]
  Return to Process A -> ASID=1 entries may still be present
  -> TLB hits are expected, suppressing performance degradation
```

---

## 5. Demand Paging and Virtual Memory

### 5.1 How Demand Paging Works

In demand paging, pages are not loaded into physical memory at process startup. Instead, they are loaded only when actually accessed. This achieves faster startup times and reduced memory usage.

```
Demand paging flow:

  1. Process startup: Create page table with all pages "invalid (P=0)"
  2. CPU accesses a virtual address
  3. MMU references page table -> P=0 -> Page fault exception raised
  4. OS page fault handler starts:
     a. Verify access is legitimate (SIGSEGV if segmentation violation)
     b. Secure a free physical frame (execute page replacement if none)
     c. Load page contents from disk
     d. Update page table: Set PFN and P=1
     e. Update TLB
  5. Re-execute interrupted instruction -> Page now exists, normal access

  +---------+     Page fault           +----------+
  |   CPU   | ---------------------->  |    OS    |
  |         |                         | Handler  |
  | Resume  | <---------------------- |          |
  +---------+   Page table updated    +----+-----+
                                         |
                                         v
                                    +----------+
                                    | Disk     |
                                    | (Swap    |
                                    |  area)   |
                                    +----------+
```

### 5.2 Types of Page Faults

| Type | Cause | OS Response | Cost |
|------|-------|------------|------|
| Minor (Soft) | Page is in physical memory but table not set | Just set PTE | A few microseconds |
| Major (Hard) | Page is on disk | Read from disk | A few ms (SSD) to tens of ms (HDD) |
| Invalid | Access to invalid address | Send SIGSEGV to process | Process termination |

**Page fault cost analysis:**

```
Major page fault cost estimates:
  SSD random read: ~100us = 100,000ns
  HDD random read: ~10ms  = 10,000,000ns
  Memory access   : ~100ns

  Effective memory access time with HDD swap:
    Given page fault rate p:
    EAT = (1-p) x 100ns + p x 10,000,000ns

    If p = 1/1000 (0.1%):
    EAT = 0.999 x 100 + 0.001 x 10,000,000 = 99.9 + 10,000 = 10,099.9ns
    -> Memory access becomes 100x slower

    To keep performance degradation under 10%:
    110 > (1-p)x100 + p x 10,000,000
    10 > p x 9,999,900
    p < 0.000001 = 0.0001%
    -> Less than 1 page fault per 1 million accesses is needed

  This is why page replacement algorithm performance is critically important.
```

### 5.3 Copy-on-Write (COW)

The `fork()` system call creates a complete copy of a process, but copying all pages is very costly. COW shares the same physical pages between parent and child immediately after `fork()`, and creates a copy only when either writes.

```
Copy-on-Write mechanism:

Immediately after fork():
  Parent Process       Child Process
  Page Table           Page Table
  +---------+        +---------+
  |VPN 0->F3|        |VPN 0->F3|  <- Same frame shared
  |VPN 1->F7|        |VPN 1->F7|  <- R/W changed to Read-Only
  |VPN 2->F1|        |VPN 2->F1|
  +---------+        +---------+
                Physical Memory
              +----------+
         F1   |Shared data|
         F3   |Shared data|
         F7   |Shared data|
              +----------+

When child writes to VPN 1:
  1. Page fault occurs (write to Read-Only page)
  2. OS determines it is COW
  3. Allocates new frame F9
  4. Copies F7 contents to F9
  5. Updates child PTE to VPN 1->F9 (R/W)
  6. Restores parent PTE to VPN 1->F7 (R/W) (ref count=1)

  Parent Process       Child Process
  +---------+        +---------+
  |VPN 0->F3|        |VPN 0->F3|  <- Still shared
  |VPN 1->F7|        |VPN 1->F9|  <- Separation complete
  |VPN 2->F1|        |VPN 2->F1|  <- Still shared
  +---------+        +---------+
```

---

## 6. Page Replacement Algorithms

When physical memory is full and a new page needs to be loaded, an existing page must be evicted. Page replacement algorithms determine which page to evict.

### 6.1 Optimal Algorithm (OPT / Belady's Algorithm)

Replaces the page that will not be referenced for the longest time in the future. **Theoretically optimal**, but since future access patterns are unpredictable, it cannot be implemented. Used as a baseline for evaluating other algorithms' performance.

### 6.2 FIFO (First-In, First-Out)

Replaces the oldest loaded page. Simple to implement, but may evict frequently used pages. Additionally, **Belady's Anomaly** can occur: a phenomenon where page faults increase despite adding more frames.

### 6.3 LRU (Least Recently Used)

Replaces the page that has not been referenced for the longest time. Based on the locality principle that past access patterns predict the future. Shows performance close to OPT, but exact implementation requires accurate recording of access order and is costly.

### 6.4 Clock Algorithm (Second-Chance)

A practical algorithm that approximates LRU. Frames are arranged in a circular list, each with a reference bit (Accessed bit).

```
Clock algorithm operation:

  Frames arranged in a circle (clock hand rotates):

          Hand
          v
    +---+   +---+
    |F0 |   |F1 |
    |A=1|   |A=0|  <- A=0 so replacement candidate
    +---+   +---+
   /               \
  +---+           +---+
  |F5 |           |F2 |
  |A=1|           |A=1|
  +---+           +---+
   \               /
    +---+   +---+
    |F4 |   |F3 |
    |A=0|   |A=1|
    +---+   +---+

  When replacement is needed:
  1. Check the frame at hand position
  2. If A=1, set A=0 and advance hand to next (give Second Chance)
  3. If A=0, select that frame for replacement
  4. Advance hand to next position

  In the above example:
    Hand->F0 (A=1): Set A=0 and advance
    Hand->F1 (A=0): * Replace F1!

  Enhanced Clock (NRU: Not Recently Used):
    Classify into 4 classes by (A, D) combination:
    (0,0): Not recently referenced, not modified -> Highest priority for replacement
    (0,1): Not recently referenced, but modified -> Write-back needed
    (1,0): Recently referenced, not modified
    (1,1): Recently referenced and modified -> Last to replace
```

### 6.5 Algorithm Comparison Table

| Algorithm | Page Fault Rate | Implementation Cost | Belady's Anomaly | Practicality |
|-----------|----------------|--------------------|--------------------|--------------|
| OPT | Minimum (theoretically optimal) | Impossible | None | Benchmark use |
| FIFO | High | Very low | Yes | Simple systems |
| LRU | Close to OPT | High (full implementation) | None | Conceptually important |
| Clock | Close to LRU | Low | None | Adopted in Linux/BSD |
| LFU | Varies | Medium | None | Specific workloads |
| LRU-K | Very low | Medium-high | None | Databases (PostgreSQL) |

---

## 7. Code Examples

### Code Example 1: Virtual Address Decomposition and Translation Simulation (C)

```c
/*
 * virtual_address_translation.c
 *
 * Simulation that decomposes virtual addresses into VPN and offset,
 * and translates to physical addresses using a simple page table.
 *
 * Compile: gcc -Wall -o vat virtual_address_translation.c
 * Run: ./vat
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_SIZE       4096        /* 4KB = 2^12 */
#define PAGE_SHIFT      12          /* log2(PAGE_SIZE) */
#define NUM_PAGES       (1 << 20)   /* 2^20 = 1M pages (32bit) */
#define PT_SIZE         1024        /* Small table for simulation */

/* Page table entry control bits */
#define PTE_PRESENT     (1 << 0)
#define PTE_WRITABLE    (1 << 1)
#define PTE_USER        (1 << 2)
#define PTE_ACCESSED    (1 << 3)
#define PTE_DIRTY       (1 << 4)

typedef struct {
    uint32_t entry;  /* PFN (upper 20 bits) + flags (lower 12 bits) */
} PageTableEntry;

typedef struct {
    PageTableEntry entries[PT_SIZE];
    int num_entries;
} PageTable;

/* Get frame number from PTE */
static inline uint32_t pte_get_pfn(PageTableEntry pte) {
    return pte.entry >> PAGE_SHIFT;
}

/* Set frame number and flags in PTE */
static inline PageTableEntry pte_make(uint32_t pfn, uint32_t flags) {
    PageTableEntry pte;
    pte.entry = (pfn << PAGE_SHIFT) | (flags & 0xFFF);
    return pte;
}

/* Check PTE flags */
static inline int pte_is_present(PageTableEntry pte) {
    return pte.entry & PTE_PRESENT;
}

/* Initialize page table */
void page_table_init(PageTable *pt) {
    memset(pt->entries, 0, sizeof(pt->entries));
    pt->num_entries = PT_SIZE;
}

/* Add mapping */
void page_table_map(PageTable *pt, uint32_t vpn, uint32_t pfn, uint32_t flags) {
    if (vpn >= (uint32_t)pt->num_entries) {
        fprintf(stderr, "Error: VPN %u exceeds page table size\n", vpn);
        return;
    }
    pt->entries[vpn] = pte_make(pfn, flags | PTE_PRESENT);
    printf("  Mapped: VPN 0x%05X -> PFN 0x%05X (flags: ", vpn, pfn);
    if (flags & PTE_WRITABLE) printf("W ");
    if (flags & PTE_USER)     printf("U ");
    printf(")\n");
}

/* Translate virtual address */
int translate_address(PageTable *pt, uint32_t virtual_addr,
                      uint32_t *physical_addr) {
    uint32_t vpn    = virtual_addr >> PAGE_SHIFT;
    uint32_t offset = virtual_addr & (PAGE_SIZE - 1);

    printf("\n--- Address Translation ---\n");
    printf("Virtual Address : 0x%08X\n", virtual_addr);
    printf("  VPN           : 0x%05X (page %u)\n", vpn, vpn);
    printf("  Offset        : 0x%03X (%u bytes)\n", offset, offset);

    if (vpn >= (uint32_t)pt->num_entries) {
        printf("  Result        : FAULT (VPN out of range)\n");
        return -1;
    }

    PageTableEntry pte = pt->entries[vpn];
    if (!pte_is_present(pte)) {
        printf("  Result        : PAGE FAULT (page not present)\n");
        return -1;
    }

    uint32_t pfn = pte_get_pfn(pte);
    *physical_addr = (pfn << PAGE_SHIFT) | offset;

    /* Set Accessed bit (mimicking hardware behavior) */
    pt->entries[vpn].entry |= PTE_ACCESSED;

    printf("  PFN           : 0x%05X (frame %u)\n", pfn, pfn);
    printf("  Physical Addr : 0x%08X\n", *physical_addr);
    printf("  Result        : SUCCESS\n");
    return 0;
}

int main(void) {
    PageTable pt;
    page_table_init(&pt);

    printf("=== Page Table Setup ===\n");
    page_table_map(&pt, 0x00000, 0x00005, PTE_WRITABLE | PTE_USER);
    page_table_map(&pt, 0x00001, 0x00003, PTE_WRITABLE | PTE_USER);
    page_table_map(&pt, 0x00002, 0x0000B, PTE_USER);  /* Read-only */
    page_table_map(&pt, 0x00010, 0x0007B, PTE_WRITABLE | PTE_USER);

    uint32_t pa;

    /* Normal translations */
    translate_address(&pt, 0x00000A7C, &pa);  /* VPN=0, offset=0xA7C */
    translate_address(&pt, 0x00001500, &pa);  /* VPN=1, offset=0x500 */
    translate_address(&pt, 0x00010FF0, &pa);  /* VPN=0x10, offset=0xFF0 */

    /* Page fault: unmapped page */
    translate_address(&pt, 0x00003000, &pa);  /* VPN=3, no mapping */

    /* Page fault: out of range */
    translate_address(&pt, 0xFFFFF000, &pa);

    return 0;
}
```

### Code Example 2: LRU Page Replacement Simulation (Python)

```python
"""
lru_page_replacement.py

Simulation of the LRU (Least Recently Used) page replacement algorithm.
Calculates page fault count for a page reference string
and visualizes frame state at each step.

Run: python3 lru_page_replacement.py
"""

from collections import OrderedDict
from typing import List, Tuple


class LRUPageReplacer:
    """Implementation of the LRU page replacement algorithm.

    Uses OrderedDict to efficiently manage access order.
    The most recently accessed page is at the end,
    and the oldest page is at the front.
    """

    def __init__(self, num_frames: int):
        """
        Args:
            num_frames: Number of available physical frames
        """
        if num_frames <= 0:
            raise ValueError("Number of frames must be a positive integer")
        self.num_frames = num_frames
        self.frames: OrderedDict[int, bool] = OrderedDict()
        self.page_faults = 0
        self.history: List[Tuple[int, list, bool]] = []

    def access_page(self, page: int) -> bool:
        """Access a page.

        Args:
            page: Page number to access

        Returns:
            True: If a page fault occurred
            False: If the page was already in a frame (hit)
        """
        fault = False

        if page in self.frames:
            # Hit: Move page to end (most recent)
            self.frames.move_to_end(page)
        else:
            # Miss: Page fault
            fault = True
            self.page_faults += 1

            if len(self.frames) >= self.num_frames:
                # Frames full -> Evict LRU page (front)
                evicted_page, _ = self.frames.popitem(last=False)

            # Add new page to end
            self.frames[page] = True

        # Record state in history
        self.history.append((page, list(self.frames.keys()), fault))
        return fault

    def simulate(self, reference_string: List[int]) -> int:
        """Simulate the entire page reference string.

        Args:
            reference_string: Page reference string

        Returns:
            Total page fault count
        """
        for page in reference_string:
            self.access_page(page)
        return self.page_faults

    def print_trace(self) -> None:
        """Display simulation trace."""
        print(f"\n{'='*60}")
        print(f"LRU Page Replacement Simulation (Frames: {self.num_frames})")
        print(f"{'='*60}")
        print(f"{'Step':>4} | {'Page':>4} | {'Frames':<25} | {'Result'}")
        print(f"{'-'*4:>4}-+-{'-'*4:>4}-+-{'-'*25:<25}-+-{'-'*10}")

        for i, (page, frames, fault) in enumerate(self.history):
            frames_str = str(frames)
            result = "FAULT" if fault else "HIT"
            print(f"{i+1:>4} | {page:>4} | {frames_str:<25} | {result}")

        print(f"\nTotal page faults: {self.page_faults}")
        hit_count = len(self.history) - self.page_faults
        hit_rate = hit_count / len(self.history) * 100 if self.history else 0
        print(f"Hit rate: {hit_rate:.1f}%")


def compare_algorithms(reference_string: List[int],
                       num_frames: int) -> None:
    """Compare FIFO and LRU."""

    # --- FIFO ---
    from collections import deque
    fifo_frames: deque = deque()
    fifo_set: set = set()
    fifo_faults = 0
    for page in reference_string:
        if page not in fifo_set:
            fifo_faults += 1
            if len(fifo_frames) >= num_frames:
                old = fifo_frames.popleft()
                fifo_set.discard(old)
            fifo_frames.append(page)
            fifo_set.add(page)

    # --- LRU ---
    lru = LRUPageReplacer(num_frames)
    lru_faults = lru.simulate(reference_string)

    # --- OPT ---
    opt_faults = 0
    opt_frames: list = []
    for i, page in enumerate(reference_string):
        if page not in opt_frames:
            opt_faults += 1
            if len(opt_frames) >= num_frames:
                # Find the page referenced furthest in the future
                farthest = -1
                victim = -1
                for f in opt_frames:
                    try:
                        next_use = reference_string[i+1:].index(f)
                    except ValueError:
                        next_use = float('inf')
                    if next_use > farthest:
                        farthest = next_use
                        victim = f
                opt_frames.remove(victim)
            opt_frames.append(page)

    print(f"\n{'='*50}")
    print(f"Algorithm Comparison (Frames: {num_frames})")
    print(f"Reference String: {reference_string}")
    print(f"{'='*50}")
    print(f"{'Algorithm':<12} | {'Page Faults':>12} | {'Hit Rate':>10}")
    print(f"{'-'*12}-+-{'-'*12:>12}-+-{'-'*10}")
    total = len(reference_string)
    print(f"{'OPT':<12} | {opt_faults:>12} | "
          f"{(total-opt_faults)/total*100:>9.1f}%")
    print(f"{'LRU':<12} | {lru_faults:>12} | "
          f"{(total-lru_faults)/total*100:>9.1f}%")
    print(f"{'FIFO':<12} | {fifo_faults:>12} | "
          f"{(total-fifo_faults)/total*100:>9.1f}%")


if __name__ == "__main__":
    # Textbook reference string
    ref_string = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3, 2, 1, 2, 0, 1, 7, 0, 1]

    lru = LRUPageReplacer(num_frames=3)
    lru.simulate(ref_string)
    lru.print_trace()

    compare_algorithms(ref_string, num_frames=3)
    compare_algorithms(ref_string, num_frames=4)
```

### Code Example 3: Clock Algorithm Implementation (C)

```c
/*
 * clock_algorithm.c
 *
 * Implementation of the Clock (Second-Chance) page replacement algorithm.
 * Approximates LRU using a circular buffer and reference bits.
 *
 * Compile: gcc -Wall -o clock clock_algorithm.c
 * Run: ./clock
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_FRAMES 64

typedef struct {
    int page_number;    /* Page number (-1 = empty) */
    int reference_bit;  /* Reference bit (corresponds to Accessed bit) */
    int dirty_bit;      /* Dirty bit */
} Frame;

typedef struct {
    Frame frames[MAX_FRAMES];
    int num_frames;
    int hand;           /* Clock hand (index of next frame to inspect) */
    int used_frames;    /* Currently used frame count */
    int page_faults;
    int writes_back;    /* Number of disk write-backs */
} ClockReplacer;

void clock_init(ClockReplacer *cr, int num_frames) {
    if (num_frames > MAX_FRAMES) {
        fprintf(stderr, "Error: num_frames exceeds MAX_FRAMES\n");
        exit(1);
    }
    cr->num_frames = num_frames;
    cr->hand = 0;
    cr->used_frames = 0;
    cr->page_faults = 0;
    cr->writes_back = 0;
    for (int i = 0; i < num_frames; i++) {
        cr->frames[i].page_number = -1;
        cr->frames[i].reference_bit = 0;
        cr->frames[i].dirty_bit = 0;
    }
}

/* Search for a page in frames */
int clock_find_page(ClockReplacer *cr, int page) {
    for (int i = 0; i < cr->num_frames; i++) {
        if (cr->frames[i].page_number == page) {
            return i;
        }
    }
    return -1;
}

/* Page access processing */
int clock_access(ClockReplacer *cr, int page, int is_write) {
    int idx = clock_find_page(cr, page);

    if (idx >= 0) {
        /* Hit: Set reference bit to 1 */
        cr->frames[idx].reference_bit = 1;
        if (is_write) {
            cr->frames[idx].dirty_bit = 1;
        }
        return 0; /* No page fault */
    }

    /* Page fault */
    cr->page_faults++;

    if (cr->used_frames < cr->num_frames) {
        /* Free frame available */
        for (int i = 0; i < cr->num_frames; i++) {
            if (cr->frames[i].page_number == -1) {
                cr->frames[i].page_number = page;
                cr->frames[i].reference_bit = 1;
                cr->frames[i].dirty_bit = is_write ? 1 : 0;
                cr->used_frames++;
                return 1;
            }
        }
    }

    /* Select replacement target using Clock algorithm */
    while (1) {
        if (cr->frames[cr->hand].reference_bit == 0) {
            /* Found replacement target */
            int evicted = cr->frames[cr->hand].page_number;
            if (cr->frames[cr->hand].dirty_bit) {
                cr->writes_back++;
                printf("    [Write-back] Page %d written to disk\n", evicted);
            }

            cr->frames[cr->hand].page_number = page;
            cr->frames[cr->hand].reference_bit = 1;
            cr->frames[cr->hand].dirty_bit = is_write ? 1 : 0;

            printf("    [Evict] Page %d replaced by Page %d at Frame %d\n",
                   evicted, page, cr->hand);

            cr->hand = (cr->hand + 1) % cr->num_frames;
            return 1;
        }
        /* Second Chance: Clear reference bit and move to next */
        cr->frames[cr->hand].reference_bit = 0;
        cr->hand = (cr->hand + 1) % cr->num_frames;
    }
}

/* Display frame state */
void clock_print_state(ClockReplacer *cr) {
    printf("  Frames: [");
    for (int i = 0; i < cr->num_frames; i++) {
        if (i > 0) printf(", ");
        if (cr->frames[i].page_number == -1) {
            printf("  -  ");
        } else {
            printf("P%d(%c%c)", cr->frames[i].page_number,
                   cr->frames[i].reference_bit ? 'R' : '-',
                   cr->frames[i].dirty_bit ? 'D' : '-');
        }
    }
    printf("]  Hand->%d\n", cr->hand);
}

int main(void) {
    ClockReplacer cr;
    clock_init(&cr, 4);

    /* Page reference string: (page number, whether write) */
    int refs[] =    {1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5};
    int writes[] =  {0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    int n = sizeof(refs) / sizeof(refs[0]);

    printf("=== Clock (Second-Chance) Algorithm ===\n");
    printf("Frames: 4, Reference String Length: %d\n\n", n);

    for (int i = 0; i < n; i++) {
        int fault = clock_access(&cr, refs[i], writes[i]);
        printf("Step %2d: Access Page %d (%s) -> %s\n",
               i + 1, refs[i],
               writes[i] ? "WRITE" : "READ",
               fault ? "FAULT" : "HIT");
        clock_print_state(&cr);
        printf("\n");
    }

    printf("=== Summary ===\n");
    printf("Total page faults : %d\n", cr.page_faults);
    printf("Total write-backs : %d\n", cr.writes_back);
    printf("Hit rate          : %.1f%%\n",
           (double)(n - cr.page_faults) / n * 100.0);

    return 0;
}
```

### Code Example 4: Memory-Mapped File Using Linux mmap (C)

```c
/*
 * mmap_demo.c
 *
 * Demo of mapping a file to memory using mmap()
 * and reading/writing the file through normal memory access.
 * The benefits of demand paging can be directly observed.
 *
 * Compile: gcc -Wall -o mmap_demo mmap_demo.c
 * Run: ./mmap_demo
 *
 * Environment: Linux / macOS (POSIX compliant)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <errno.h>

#define FILE_PATH "/tmp/mmap_demo.dat"
#define FILE_SIZE (4096 * 4)  /* 16KB = 4 pages */

/* Create a demo data file */
int create_demo_file(const char *path, size_t size) {
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return -1;
    }

    /* Set file size. ftruncate extends the file to specified size.
       Why needed: mmap cannot map beyond the file size */
    if (ftruncate(fd, size) < 0) {
        perror("ftruncate");
        close(fd);
        return -1;
    }

    return fd;
}

int main(void) {
    printf("=== mmap Demo: Memory-Mapped File I/O ===\n\n");

    /* Step 1: Create file */
    printf("[1] Creating demo file: %s (%d bytes = %d pages)\n",
           FILE_PATH, FILE_SIZE, FILE_SIZE / 4096);
    int fd = create_demo_file(FILE_PATH, FILE_SIZE);
    if (fd < 0) return 1;

    /* Step 2: Map file to memory
       MAP_SHARED: Changes are reflected in the file (visible to other processes)
       MAP_PRIVATE creates a private COW copy */
    printf("[2] Mapping file to memory with mmap()\n");
    char *mapped = mmap(NULL, FILE_SIZE, PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* After mmap, the mapping remains valid even if fd is closed
       Because: The kernel manages the file's reference count */
    close(fd);
    printf("    File descriptor closed (mapping still valid)\n");

    /* Step 3: Memory write = File write */
    printf("[3] Writing data through memory mapping\n");
    const char *messages[] = {
        "Page 0: Hello from mmap!",
        "Page 1: This is page-aligned data.",
        "Page 2: Memory-mapped I/O is efficient.",
        "Page 3: No read()/write() syscalls needed."
    };

    for (int i = 0; i < 4; i++) {
        /* Write to start of each page.
           On first access, demand paging causes a page fault
           and a physical frame is allocated */
        char *page_start = mapped + (i * 4096);
        snprintf(page_start, 4096, "%s", messages[i]);
        printf("    Wrote to page %d (offset %d): \"%s\"\n",
               i, i * 4096, messages[i]);
    }

    /* Step 4: Memory read = File read */
    printf("[4] Reading data through memory mapping\n");
    for (int i = 0; i < 4; i++) {
        char *page_start = mapped + (i * 4096);
        printf("    Page %d: \"%s\"\n", i, page_start);
    }

    /* Step 5: Force sync to disk with msync
       Why needed: The kernel buffers writes for performance.
       Calling msync ensures changes are reliably written to disk */
    printf("[5] Syncing changes to disk with msync()\n");
    if (msync(mapped, FILE_SIZE, MS_SYNC) < 0) {
        perror("msync");
    }

    /* Step 6: Unmap */
    printf("[6] Unmapping memory\n");
    if (munmap(mapped, FILE_SIZE) < 0) {
        perror("munmap");
    }

    /* Step 7: Verify with normal read() */
    printf("[7] Verifying with normal read()\n");
    fd = open(FILE_PATH, O_RDONLY);
    if (fd >= 0) {
        char buf[64];
        ssize_t n = read(fd, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            printf("    Read from file: \"%s\"\n", buf);
        }
        close(fd);
    }

    /* Cleanup */
    unlink(FILE_PATH);
    printf("\n=== Demo Complete ===\n");

    return 0;
}
```

### Code Example 5: Retrieving Paging Statistics (Python / Linux)

```python
"""
paging_stats.py

A tool that reads paging-related statistics from the Linux /proc
filesystem and displays them in an easy-to-understand format.

Checks virtual memory usage, page fault counts, swap usage,
TLB flush counts, etc.

Run: python3 paging_stats.py
Environment: Linux only
"""

import os
import sys
from typing import Dict, Optional


def read_proc_file(path: str) -> Optional[str]:
    """Read a file from procfs."""
    try:
        with open(path, 'r') as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as e:
        print(f"  Warning: Cannot read {path}: {e}")
        return None


def parse_meminfo() -> Dict[str, int]:
    """
    Parse /proc/meminfo and return memory info as a dictionary.
    All values are in KB.
    """
    content = read_proc_file('/proc/meminfo')
    if content is None:
        return {}

    info = {}
    for line in content.strip().split('\n'):
        parts = line.split(':')
        if len(parts) == 2:
            key = parts[0].strip()
            # Extract numeric part. "1234 kB" -> 1234
            value_str = parts[1].strip().split()[0]
            try:
                info[key] = int(value_str)
            except ValueError:
                pass
    return info


def parse_vmstat() -> Dict[str, int]:
    """
    Parse /proc/vmstat and return virtual memory statistics as a dictionary.
    """
    content = read_proc_file('/proc/vmstat')
    if content is None:
        return {}

    stats = {}
    for line in content.strip().split('\n'):
        parts = line.split()
        if len(parts) == 2:
            try:
                stats[parts[0]] = int(parts[1])
            except ValueError:
                pass
    return stats


def parse_process_status(pid: int) -> Dict[str, str]:
    """
    Parse /proc/[pid]/status and return process memory info.
    """
    content = read_proc_file(f'/proc/{pid}/status')
    if content is None:
        return {}

    status = {}
    for line in content.strip().split('\n'):
        parts = line.split(':', 1)
        if len(parts) == 2:
            status[parts[0].strip()] = parts[1].strip()
    return status


def format_kb(kb: int) -> str:
    """Convert KB to human-readable format."""
    if kb >= 1048576:
        return f"{kb / 1048576:.1f} GB"
    elif kb >= 1024:
        return f"{kb / 1024:.1f} MB"
    else:
        return f"{kb} KB"


def print_memory_overview(meminfo: Dict[str, int]) -> None:
    """Display memory overview."""
    print("=" * 60)
    print("MEMORY OVERVIEW")
    print("=" * 60)

    total = meminfo.get('MemTotal', 0)
    free = meminfo.get('MemFree', 0)
    available = meminfo.get('MemAvailable', 0)
    buffers = meminfo.get('Buffers', 0)
    cached = meminfo.get('Cached', 0)

    used = total - free - buffers - cached
    usage_pct = (used / total * 100) if total > 0 else 0

    print(f"  Total     : {format_kb(total)}")
    print(f"  Used      : {format_kb(used)} ({usage_pct:.1f}%)")
    print(f"  Free      : {format_kb(free)}")
    print(f"  Available : {format_kb(available)}")
    print(f"  Buffers   : {format_kb(buffers)}")
    print(f"  Cached    : {format_kb(cached)}")

    # Page size information
    page_size = os.sysconf('SC_PAGE_SIZE')
    print(f"\n  Page Size : {page_size} bytes ({page_size // 1024} KB)")
    print(f"  Total Pages: {total * 1024 // page_size:,}")


def print_swap_info(meminfo: Dict[str, int]) -> None:
    """Display swap information."""
    print(f"\n{'='*60}")
    print("SWAP INFORMATION")
    print("=" * 60)

    swap_total = meminfo.get('SwapTotal', 0)
    swap_free = meminfo.get('SwapFree', 0)
    swap_used = swap_total - swap_free
    swap_cached = meminfo.get('SwapCached', 0)

    if swap_total == 0:
        print("  Swap is not configured")
        return

    usage_pct = (swap_used / swap_total * 100) if swap_total > 0 else 0
    print(f"  Total  : {format_kb(swap_total)}")
    print(f"  Used   : {format_kb(swap_used)} ({usage_pct:.1f}%)")
    print(f"  Free   : {format_kb(swap_free)}")
    print(f"  Cached : {format_kb(swap_cached)}")


def print_paging_stats(vmstat: Dict[str, int]) -> None:
    """Display paging-related statistics."""
    print(f"\n{'='*60}")
    print("PAGING STATISTICS (since boot)")
    print("=" * 60)

    # Page faults
    pgfault = vmstat.get('pgfault', 0)
    pgmajfault = vmstat.get('pgmajfault', 0)
    pgminfault = pgfault - pgmajfault

    print(f"  Page Faults (total) : {pgfault:>15,}")
    print(f"    Minor faults      : {pgminfault:>15,}")
    print(f"    Major faults      : {pgmajfault:>15,}")
    if pgfault > 0:
        major_pct = pgmajfault / pgfault * 100
        print(f"    Major fault ratio : {major_pct:>14.4f}%")

    # Page in/out
    pgpgin = vmstat.get('pgpgin', 0)
    pgpgout = vmstat.get('pgpgout', 0)
    print(f"\n  Pages In  (from disk) : {pgpgin:>12,} KB")
    print(f"  Pages Out (to disk)   : {pgpgout:>12,} KB")

    # Swap in/out
    pswpin = vmstat.get('pswpin', 0)
    pswpout = vmstat.get('pswpout', 0)
    print(f"\n  Swap In  : {pswpin:>12,} pages")
    print(f"  Swap Out : {pswpout:>12,} pages")


def print_process_memory(pid: int) -> None:
    """Display memory information for a specific process."""
    print(f"\n{'='*60}")
    print(f"PROCESS MEMORY (PID: {pid})")
    print("=" * 60)

    status = parse_process_status(pid)
    if not status:
        print(f"  Cannot read process {pid} info")
        return

    print(f"  Name     : {status.get('Name', 'Unknown')}")
    print(f"  VmSize   : {status.get('VmSize', 'N/A'):>12} (virtual memory size)")
    print(f"  VmRSS    : {status.get('VmRSS', 'N/A'):>12} (physical memory usage)")
    print(f"  VmSwap   : {status.get('VmSwap', 'N/A'):>12} (swap usage)")
    print(f"  VmPeak   : {status.get('VmPeak', 'N/A'):>12} (virtual memory peak)")
    print(f"  VmData   : {status.get('VmData', 'N/A'):>12} (data segment)")
    print(f"  VmStk    : {status.get('VmStk', 'N/A'):>12} (stack segment)")
    print(f"  VmLib    : {status.get('VmLib', 'N/A'):>12} (shared libraries)")

    # Page fault info from /proc/[pid]/stat
    stat_content = read_proc_file(f'/proc/{pid}/stat')
    if stat_content:
        fields = stat_content.split()
        if len(fields) > 11:
            minflt = int(fields[9])
            majflt = int(fields[11])
            print(f"\n  Minor faults: {minflt:>12,}")
            print(f"  Major faults: {majflt:>12,}")


def main():
    if sys.platform != 'linux':
        print("This tool is designed for Linux systems.")
        print("Demonstrating with simulated data...\n")

        # Demo display for non-Linux systems
        print("=" * 60)
        print("SIMULATED PAGING STATISTICS")
        print("=" * 60)
        print("  On a Linux system, this tool reads from:")
        print("    /proc/meminfo    - Memory usage overview")
        print("    /proc/vmstat     - Virtual memory statistics")
        print("    /proc/[pid]/stat - Per-process page fault counters")
        print("\n  Key metrics to monitor:")
        print("    - Major page faults: High values indicate thrashing")
        print("    - Swap usage: Non-zero means physical memory is insufficient")
        print("    - Minor/Major ratio: Should be >99% minor faults")
        return

    meminfo = parse_meminfo()
    vmstat = parse_vmstat()

    print_memory_overview(meminfo)
    print_swap_info(meminfo)
    print_paging_stats(vmstat)
    print_process_memory(os.getpid())

    print(f"\n{'='*60}")
    print("HUGE PAGES")
    print("=" * 60)
    hp_total = meminfo.get('HugePages_Total', 0)
    hp_free = meminfo.get('HugePages_Free', 0)
    hp_size = meminfo.get('Hugepagesize', 0)
    print(f"  Total     : {hp_total}")
    print(f"  Free      : {hp_free}")
    print(f"  Page Size : {format_kb(hp_size)}")

    thp = meminfo.get('AnonHugePages', 0)
    print(f"  Transparent Huge Pages (anon): {format_kb(thp)}")


if __name__ == '__main__':
    main()
```

---

## 8. Swapping and Thrashing

### 8.1 How Swapping Works

When physical memory is insufficient, the OS evacuates infrequently used pages to a **swap area** on disk, freeing physical frames. This process is called **swap out**, and the process of reading back an evacuated page from disk when it is accessed again is called **swap in**.

```
Swapping flow:

  Physical Memory                 Swap Area (Disk)
  +----------+              +------------------+
  | Frame 0  | <-(in use)    |                  |
  +----------+              |                  |
  | Frame 1  | --swap out--> | Page X contents  |
  +----------+              |                  |
  | Frame 2  | <-(in use)    | Page Y contents  |
  +----------+              |                  |
  | Frame 3  | <-(new page)  | Page Z contents  |
  +----------+              |                  |
  |   ...    |              +------------------+
  +----------+

  Swap-out decision criteria:
    1. Prioritize pages with Dirty bit = 0 (no write-back needed)
    2. Prioritize pages with reference bit = 0 (not recently used)
    3. Kernel pages are not swapped
    4. Locked pages (mlock) are not swapped
```

### 8.2 Linux Swap Management

```
Linux swap configuration:

  Check with /proc/swaps:
    Filename    Type        Size       Used    Priority
    /dev/sda2   partition   8388604    102400  -2
    /swapfile   file        4194300    0       -3

  swappiness parameter (/proc/sys/vm/swappiness):
    Value range: 0-200 (default: 60)

    0   : Avoid swapping as much as possible (preferentially release file cache)
    60  : Balanced default value
    100 : Treat page cache and swap equally
    200 : Aggressively swap

  Why adjust swappiness:
    Database server -> swappiness=10-20
      Reason: DBs have their own cache; page cache is unnecessary.
              Swapping causes latency spikes.

    Desktop -> swappiness=60 (default)
      Reason: Balance between app responsiveness and file cache is important.

    Memory-rich server -> swappiness=1
      Reason: Swap is better than OOM Killer; only needed as a safety valve.
```

### 8.3 Thrashing

Thrashing is a state where physical memory is extremely insufficient, page faults are frequent, and the CPU spends most of its time swapping pages in and out, with actual processing barely progressing.

```
Thrashing mechanism:

  CPU Usage
  100% |
       |        +------+
       |       /|      |\
       |      / |      | \
   50% |     /  |      |  \
       |    /   |      |   \___________
       |   /    |      |    Thrashing
       |  /     |      |   (CPU waits
       | /      |      |    on I/O)
    0% |/-------+------+---------------
       +------------------------------>
        Few                    Many
             Concurrent Processes

  Countermeasures:
    1. Working Set Model:
       Track each process's "working set" (set of recently referenced pages),
       and swap out processes whose working sets cannot fit in physical memory

    2. PFF (Page Fault Frequency):
       Monitor each process's page fault frequency;
       allocate additional frames if above threshold,
       reclaim frames if below threshold

    3. OOM Killer (Linux):
       When memory is completely exhausted, forcibly terminate
       the process consuming the most memory to secure physical memory
```

---

## 9. Huge Pages (Large Pages)

### 9.1 Why Huge Pages Are Needed

With regular 4KB pages, applications that use large amounts of memory (databases, virtual machine monitors, scientific computing) experience frequent TLB misses. Using Huge Pages expands the memory region covered by a single TLB entry, dramatically reducing TLB misses.

| Page Size | Coverage per TLB Entry | Coverage with 1024-entry TLB |
|-----------|------------------------|------------------------------|
| 4KB | 4KB | 4MB |
| 2MB | 2MB | 2GB |
| 1GB | 1GB | 1TB |

```
Huge Pages configuration (Linux):

  Static Huge Pages (hugetlbfs):
    # Reserve 1024 2MB pages (total 2GB)
    echo 1024 > /proc/sys/vm/nr_hugepages

    # Verify
    cat /proc/meminfo | grep -i huge
    HugePages_Total:    1024
    HugePages_Free:     1024
    Hugepagesize:       2048 kB

    # Usage from applications
    # shmget + SHM_HUGETLB, or mmap + MAP_HUGETLB

  Transparent Huge Pages (THP):
    # Kernel automatically merges 4KB pages into 2MB pages
    cat /sys/kernel/mm/transparent_hugepage/enabled
    [always] madvise never

    # madvise mode: Only when app explicitly requests with MADV_HUGEPAGE
    echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

  Why THP may be disabled:
    THP's page merging (khugepaged) consumes CPU in the background,
    which can cause performance degradation for latency-sensitive
    applications (like Redis). Therefore, Redis's official documentation
    recommends disabling THP.
```

---

## 10. Linux Memory Management Architecture

### 10.1 Overview

```
Linux memory management stack:

  +----------------------------------------------------+
  |                  User Space                         |
  |  malloc() / free() / new / delete / mmap()         |
  |                    v                                |
  |  +------------------------------------------+      |
  |  | User-space Allocator                      |      |
  |  |   glibc ptmalloc2 / jemalloc / tcmalloc  |      |
  |  |   -> Free list management, thread cache   |      |
  |  +------------------------------------------+      |
  |                    v                                |
  |  brk() / sbrk() : Heap area expansion               |
  |  mmap()         : New virtual memory area allocation |
  +----------------------------------------------------+
  |                  Kernel Space                        |
  |  +------------------------------------------+      |
  |  | VMA (Virtual Memory Area) Management      |      |
  |  |   Red-black tree / list of vm_area_struct |      |
  |  |   -> Describes process virtual addr space |      |
  |  +------------------------------------------+      |
  |                    v                                |
  |  +------------------------------------------+      |
  |  | Page Fault Handler                        |      |
  |  |   do_page_fault() -> handle_mm_fault()    |      |
  |  |   -> Demand paging, COW processing        |      |
  |  +------------------------------------------+      |
  |                    v                                |
  |  +------------------------------------------+      |
  |  | Page Allocator (Buddy System)             |      |
  |  |   Manages physical pages in 2^n blocks    |      |
  |  |   Check with /proc/buddyinfo              |      |
  |  +------------------------------------------+      |
  |                    v                                |
  |  +------------------------------------------+      |
  |  | SLUB Allocator                            |      |
  |  |   Small memory blocks for kernel objects  |      |
  |  |   Check with /proc/slabinfo              |      |
  |  +------------------------------------------+      |
  |                    v                                |
  |  +------------------------------------------+      |
  |  | Page Reclamation                          |      |
  |  |   kswapd daemon / direct reclaim          |      |
  |  |   LRU lists: active / inactive            |      |
  |  |   -> Managed by Clock-type algorithms     |      |
  |  +------------------------------------------+      |
  +----------------------------------------------------+
```

### 10.2 Buddy System

The Buddy System is an algorithm that manages physical page allocation and deallocation, operating in blocks of power-of-2 sizes.

```
Buddy System operation example:

  Initial state: 64 contiguous pages

  order=6 (64 pages)
  +-----------------------------------------------------------+
  |                         64                                 |
  +-----------------------------------------------------------+

  Request for 8 pages -> Need an order=3 block:
  1. Split order=6 into two order=5 (32 pages)
  2. Split order=5 into two order=4 (16 pages)
  3. Split order=4 into two order=3 (8 pages)

  order=3   order=3   order=4      order=5
  +--------+--------+------------+--------------------------+
  | In use  | Free   |   Free      |        Free              |
  |  (8)   |  (8)   |   (16)     |        (32)              |
  +--------+--------+------------+--------------------------+

  Release 8 pages -> Merge with buddy (adjacent same-size free block):
  order=3 + order=3 -> order=4
  order=4 + order=4 -> order=5
  order=5 + order=5 -> order=6 (restored to original)

  Why Buddy System:
    - Merge check is O(1): Buddy's address can be computed with bit operations
    - Suppresses external fragmentation: Maintains large blocks through merging
    - Check free blocks per order with /proc/buddyinfo
```

### 10.3 OOM Killer

```
OOM Killer operation:

  Last resort when memory is completely exhausted.
  Assigns oom_score to each process and terminates the one with highest score.

  Factors in oom_score calculation:
    - Process's physical memory usage (higher usage = higher score)
    - Swap usage
    - Process runtime (longer = lower score)
    - nice value

  Checking and control:
    # Check OOM score for a specific process
    cat /proc/<pid>/oom_score

    # Exclude from OOM targets (-1000 = fully excluded)
    echo -1000 > /proc/<pid>/oom_score_adj

    # Increase OOM priority (1000 = highest priority to kill)
    echo 1000 > /proc/<pid>/oom_score_adj

  Why OOM Killer is needed:
    Due to Linux's overcommit (malloc success != physical memory allocation),
    physical memory can run out when all processes start using memory simultaneously.
    OOM Killer is a safety valve to prevent the entire system from hanging.
```

---

## 11. Inverted Page Tables and Hash Page Tables

### 11.1 Inverted Page Table

Normal page tables cover the entire virtual address space per process, which can become extremely large in 64-bit environments. Inverted page tables reverse the concept, maintaining **one entry per physical frame**.

```
Inverted page table:

  Normal page table (forward):
    Holds VPN -> PFN table per process
    Entry count = number of virtual pages (can be enormous)

  Inverted page table:
    One for the entire system, holds PFN -> (PID, VPN) table
    Entry count = number of physical frames (proportional to physical memory)

  +------------+
  | Frame 0    | -> (PID=5, VPN=0x100)
  +------------+
  | Frame 1    | -> (PID=3, VPN=0x200)
  +------------+
  | Frame 2    | -> (PID=5, VPN=0x300)
  +------------+
  | Frame 3    | -> (PID=7, VPN=0x050)
  +------------+
  |    ...     |
  +------------+

  Translation: (PID, VPN) -> PFN
    Linear search of all entries -> O(N), slow
    -> Use hash table alongside for O(1)

  Adopted by: PowerPC (IBM POWER), IA-64 (Itanium)

  Advantages:
    - Proportional to physical memory size, so memory consumption is predictable
    - Table size doesn't explode in 64-bit environments

  Disadvantages:
    - Hash collision handling needed
    - Shared memory implementation is complex (multiple VPNs correspond to one frame)
```

---

## 12. Memory-Mapped Files (mmap)

### 12.1 How mmap Works

The `mmap()` system call maps a file or anonymous memory region into a process's virtual address space. Memory accesses to the mapped region are automatically translated to file I/O by the kernel.

```
Types of mmap:

  +-----------------+------------------------------------------+
  | Type            | Description                               |
  +-----------------+------------------------------------------+
  | MAP_SHARED      | Changes are reflected in the file.        |
  | (file)          | Can be shared across multiple processes.   |
  |                 | Widely used in databases (SQLite, LMDB).  |
  +-----------------+------------------------------------------+
  | MAP_PRIVATE     | Changes are process-local (COW).          |
  | (file)          | Used for shared library .text sections.    |
  +-----------------+------------------------------------------+
  | MAP_ANONYMOUS   | Anonymous memory not backed by a file.     |
  | + MAP_PRIVATE   | Used for large malloc allocations.         |
  +-----------------+------------------------------------------+
  | MAP_ANONYMOUS   | Inter-process shared memory.               |
  | + MAP_SHARED    | Shared between parent and child after fork.|
  +-----------------+------------------------------------------+

  mmap vs read()/write():

  +--------------+------------------+------------------+
  | Item          | read()/write()   | mmap()           |
  +--------------+------------------+------------------+
  | Data copy     | Kernel -> User   | Zero-copy        |
  |              | (2 copies)       | (only page table |
  |              |                  |  setup)          |
  +--------------+------------------+------------------+
  | Random       | lseek + read     | Pointer arith.   |
  | access       | (system call)    | (user space)     |
  +--------------+------------------+------------------+
  | Small files  | Efficient        | mmap overhead    |
  |              |                  | relatively large |
  +--------------+------------------+------------------+
  | Large files  | Buffer mgmt      | Very efficient   |
  |              | needed           |                  |
  +--------------+------------------+------------------+
```

---

## 13. Code Examples (continued)

### Code Example 6: Working Set Estimation (Python)

```python
"""
working_set_estimator.py

Estimates Working Set Size (WSS) from a page reference string.
The working set is the set of pages accessed within a time window delta,
and is a fundamental concept for preventing thrashing.

Run: python3 working_set_estimator.py
"""

from typing import List, Set, Tuple
import random


def compute_working_set(reference_string: List[int],
                        window_size: int) -> List[Tuple[int, Set[int], int]]:
    """Compute the working set at each time point.

    Args:
        reference_string: Page reference string
        window_size: Working set time window delta

    Returns:
        List of (time, working set, WSS size) for each time point
    """
    results = []

    for t in range(len(reference_string)):
        # Look at accesses from the past window_size references
        start = max(0, t - window_size + 1)
        window = reference_string[start:t + 1]
        ws = set(window)
        results.append((t, ws, len(ws)))

    return results


def analyze_working_set(reference_string: List[int],
                        window_sizes: List[int]) -> None:
    """Analyze working sets with different window sizes."""
    print("=" * 70)
    print("Working Set Analysis")
    print(f"Reference String ({len(reference_string)} accesses):")
    print(f"  {reference_string}")
    print("=" * 70)

    for delta in window_sizes:
        results = compute_working_set(reference_string, delta)

        # Compute average WSS
        avg_wss = sum(wss for _, _, wss in results) / len(results)
        max_wss = max(wss for _, _, wss in results)
        min_wss = min(wss for _, _, wss in results)

        print(f"\nWindow size (delta) = {delta}")
        print(f"  Average WSS: {avg_wss:.2f} pages")
        print(f"  Max WSS    : {max_wss} pages")
        print(f"  Min WSS    : {min_wss} pages")

        # Details for each time point (only for short reference strings)
        if len(reference_string) <= 20:
            print(f"\n  {'Time':>4} | {'Page':>4} | {'Working Set':<25} | {'WSS':>3}")
            print(f"  {'-'*4}-+-{'-'*4}-+-{'-'*25}-+-{'-'*3}")
            for t, ws, wss in results:
                ws_str = str(sorted(ws))
                print(f"  {t:>4} | {reference_string[t]:>4} | "
                      f"{ws_str:<25} | {wss:>3}")


def simulate_thrashing(total_frames: int, num_processes: int,
                       wss_per_process: int) -> None:
    """Thrashing simulation.

    Demonstrates how thrashing occurs when the total working set
    of all processes exceeds the number of physical frames.
    """
    total_wss = num_processes * wss_per_process

    print(f"\n{'='*60}")
    print("Thrashing Simulation")
    print(f"{'='*60}")
    print(f"  Physical frames     : {total_frames}")
    print(f"  Processes           : {num_processes}")
    print(f"  WSS per process     : {wss_per_process} pages")
    print(f"  Total WSS demand    : {total_wss} pages")
    print(f"  Overcommit ratio    : {total_wss / total_frames:.2f}x")

    if total_wss <= total_frames:
        print(f"\n  Status: STABLE")
        print(f"  All working sets fit in physical memory.")
        print(f"  Expected page fault rate: LOW (compulsory faults only)")
    elif total_wss <= total_frames * 1.5:
        print(f"\n  Status: WARNING - Moderate swapping expected")
        print(f"  Some processes may experience elevated page faults.")
        print(f"  Expected performance degradation: 20-50%")
    else:
        print(f"\n  Status: THRASHING - Severe performance degradation")
        print(f"  Working sets cannot fit in memory.")
        print(f"  CPU will spend most time on page fault handling.")
        print(f"  Recommendation: Reduce processes or add memory.")

        # Which processes should be suspended
        max_active = total_frames // wss_per_process
        print(f"\n  Maximum concurrent processes: {max_active}")
        print(f"  Processes to suspend: {num_processes - max_active}")


if __name__ == "__main__":
    # Page reference string with locality
    ref_string = [1, 2, 3, 2, 1, 3, 4, 5, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 1, 2]

    analyze_working_set(ref_string, window_sizes=[3, 5, 8])

    # Thrashing simulation
    simulate_thrashing(total_frames=1000, num_processes=5, wss_per_process=150)
    simulate_thrashing(total_frames=1000, num_processes=10, wss_per_process=150)
    simulate_thrashing(total_frames=1000, num_processes=20, wss_per_process=150)
```

### Code Example 7: Page Table Walk Simulation (Python)

```python
"""
page_table_walk.py

Simulates the x86-64 4-level page table walk.
Decomposes a virtual address into indices for each level and
displays the process of traversing page tables to translate
to a physical address.

Run: python3 page_table_walk.py
"""

from typing import Optional, Dict, Tuple


class PageTableEntry:
    """Class representing a page table entry."""

    def __init__(self, pfn: int = 0, present: bool = False,
                 writable: bool = True, user: bool = True,
                 accessed: bool = False, dirty: bool = False,
                 huge: bool = False):
        self.pfn = pfn
        self.present = present
        self.writable = writable
        self.user = user
        self.accessed = accessed
        self.dirty = dirty
        self.huge = huge  # Huge Page (2MB / 1GB)

    def __repr__(self) -> str:
        flags = []
        if self.present:  flags.append("P")
        if self.writable: flags.append("W")
        if self.user:     flags.append("U")
        if self.accessed: flags.append("A")
        if self.dirty:    flags.append("D")
        if self.huge:     flags.append("H")
        return f"PTE(PFN=0x{self.pfn:05X}, flags={'|'.join(flags)})"


class FourLevelPageTable:
    """Simulates an x86-64 4-level page table.

    Hierarchy:
      PML4 (Page Map Level 4)       -> 9 bits (bits 47-39)
      PDPT (Page Directory Pointer)  -> 9 bits (bits 38-30)
      PD   (Page Directory)          -> 9 bits (bits 29-21)
      PT   (Page Table)              -> 9 bits (bits 20-12)
      Offset                         -> 12 bits (bits 11-0)
    """

    PAGE_SHIFT = 12
    ENTRIES_PER_TABLE = 512  # 2^9
    INDEX_BITS = 9

    def __init__(self):
        # Represent each table as a dict of dicts
        # tables[level][table_pfn][index] = PageTableEntry
        self.tables: Dict[int, Dict[int, Dict[int, PageTableEntry]]] = {
            4: {},  # PML4
            3: {},  # PDPT
            2: {},  # PD
            1: {},  # PT
        }
        self.cr3 = 0x1000  # PML4 physical address (frame number)
        self._init_pml4()
        self.walk_count = 0

    def _init_pml4(self):
        """Initialize PML4 table."""
        self.tables[4][self.cr3] = {}

    def _ensure_table(self, level: int, table_pfn: int):
        """Create table if it doesn't exist."""
        if table_pfn not in self.tables[level]:
            self.tables[level][table_pfn] = {}

    def map_page(self, virtual_addr: int, physical_frame: int,
                 writable: bool = True, user: bool = True) -> None:
        """Map a virtual page to a physical frame."""
        indices = self._extract_indices(virtual_addr)

        # Level 4 (PML4) -> Level 3 (PDPT) entry
        current_table_pfn = self.cr3
        next_pfn = physical_frame + 0x1000  # Frames for intermediate tables

        for level in range(4, 1, -1):
            idx = indices[level]
            self._ensure_table(level, current_table_pfn)

            if idx not in self.tables[level][current_table_pfn]:
                # Allocate new intermediate table
                new_table_pfn = next_pfn
                next_pfn += 1
                self.tables[level][current_table_pfn][idx] = PageTableEntry(
                    pfn=new_table_pfn, present=True,
                    writable=True, user=True
                )
                self._ensure_table(level - 1, new_table_pfn)

            entry = self.tables[level][current_table_pfn][idx]
            current_table_pfn = entry.pfn

        # Level 1 (PT) final entry
        idx = indices[1]
        self._ensure_table(1, current_table_pfn)
        self.tables[1][current_table_pfn][idx] = PageTableEntry(
            pfn=physical_frame, present=True,
            writable=writable, user=user
        )

    def _extract_indices(self, virtual_addr: int) -> Dict[int, int]:
        """Extract indices for each level from a virtual address."""
        offset = virtual_addr & 0xFFF
        pt_idx  = (virtual_addr >> 12) & 0x1FF
        pd_idx  = (virtual_addr >> 21) & 0x1FF
        pdpt_idx = (virtual_addr >> 30) & 0x1FF
        pml4_idx = (virtual_addr >> 39) & 0x1FF

        return {
            4: pml4_idx,
            3: pdpt_idx,
            2: pd_idx,
            1: pt_idx,
            0: offset
        }

    def walk(self, virtual_addr: int, verbose: bool = True
             ) -> Optional[int]:
        """Execute a 4-level page table walk."""
        self.walk_count += 1
        indices = self._extract_indices(virtual_addr)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Page Table Walk #{self.walk_count}")
            print(f"Virtual Address: 0x{virtual_addr:016X}")
            print(f"{'='*60}")
            print(f"  PML4 index : {indices[4]:>3} (0x{indices[4]:03X})")
            print(f"  PDPT index : {indices[3]:>3} (0x{indices[3]:03X})")
            print(f"  PD   index : {indices[2]:>3} (0x{indices[2]:03X})")
            print(f"  PT   index : {indices[1]:>3} (0x{indices[1]:03X})")
            print(f"  Offset     : {indices[0]:>3} (0x{indices[0]:03X})")
            print()

        level_names = {4: "PML4", 3: "PDPT", 2: "PD  ", 1: "PT  "}
        current_table_pfn = self.cr3
        memory_accesses = 0

        for level in range(4, 0, -1):
            idx = indices[level]
            memory_accesses += 1

            if (current_table_pfn not in self.tables[level] or
                idx not in self.tables[level][current_table_pfn]):
                if verbose:
                    print(f"  [{level_names[level]}] Table@0x{current_table_pfn:05X}"
                          f"[{idx}] -> PAGE FAULT (not present)")
                return None

            entry = self.tables[level][current_table_pfn][idx]

            if not entry.present:
                if verbose:
                    print(f"  [{level_names[level]}] Table@0x{current_table_pfn:05X}"
                          f"[{idx}] -> PAGE FAULT (P=0)")
                return None

            if verbose:
                print(f"  [{level_names[level]}] Table@0x{current_table_pfn:05X}"
                      f"[{idx}] -> {entry}")

            # Set Accessed bit
            entry.accessed = True

            if level > 1:
                current_table_pfn = entry.pfn
            else:
                # Final level: compute physical address
                physical_addr = (entry.pfn << self.PAGE_SHIFT) | indices[0]
                memory_accesses += 1  # Data access

                if verbose:
                    print(f"\n  Physical Address: 0x{physical_addr:016X}")
                    print(f"  Memory accesses for translation: {memory_accesses}")
                return physical_addr

        return None


if __name__ == "__main__":
    pt = FourLevelPageTable()

    # Map some pages
    print("Setting up page mappings...")
    pt.map_page(0x0000_0040_0000, 0x00100)  # Virtual 0x400000 -> Physical frame 0x100
    pt.map_page(0x0000_0040_1000, 0x00200)  # Virtual 0x401000 -> Physical frame 0x200
    pt.map_page(0x0000_7FFF_F000, 0x00300)  # Stack region virtual address
    pt.map_page(0x0000_0000_1000, 0x00050)  # Low address

    # Execute page table walks
    pt.walk(0x0000_0040_0A7C)  # VPN=0x400, offset=0xA7C
    pt.walk(0x0000_0040_1500)  # VPN=0x401, offset=0x500
    pt.walk(0x0000_7FFF_F100)  # Near stack
    pt.walk(0x0000_0000_1234)  # Low address

    # Unmapped address -> page fault
    pt.walk(0x0000_DEAD_BEEF)
```

---

## 14. Anti-patterns

### Anti-pattern 1: Excessive Use of mlock

```
Problem:
  "Page faults are scary, so I'll lock all memory with mlock()"

  mlock() makes the specified memory region non-swappable,
  always keeping it in physical memory.
  Appropriate for real-time systems or protecting encryption keys,
  but excessive use is dangerous.

Why it's a problem:
  1. Reduces physical memory available to other processes
  2. Makes OOM Killer more likely to fire
  3. Locked memory is not freed even when not actually in use
  4. Conflicts with cgroup memory limits in container environments

Correct approach:
  - Only mlock data whose presence in swap would be a security risk,
    such as encryption keys or authentication tokens
  - Limit the amount of memory a process can lock with RLIMIT_MEMLOCK
  - If the goal is reducing page faults, consider Huge Pages or
    madvise(MADV_WILLNEED)
```

```c
/* Anti-pattern example */
void *buf = malloc(HUGE_SIZE);
mlock(buf, HUGE_SIZE);  /* Locking all memory -> dangerous */

/* Correct approach */
void *secret_key = malloc(KEY_SIZE);
mlock(secret_key, KEY_SIZE);  /* Lock only the minimum necessary */
/* After use */
memset(secret_key, 0, KEY_SIZE);  /* Zero-clear */
munlock(secret_key, KEY_SIZE);
free(secret_key);
```

### Anti-pattern 2: Virtual Address Space Exhaustion in 32-bit Processes

```
Problem:
  "I have 8GB of memory but malloc fails"

Cause:
  A 32-bit process's virtual address space is limited to 4GB.
  Excluding kernel space (Linux default: 1GB), the user space
  is only 3GB.

  Furthermore, the following consume virtual addresses:
  - Stack (default 8MB)
  - Shared libraries (.so / .dll)
  - Regions allocated with mmap
  - Heap fragmentation

  As a result, malloc returns NULL even when physical memory
  has free space, because virtual address space is insufficient.

Why it happens:
  mmap consumes virtual addresses, and without munmap, virtual
  addresses fragment. Especially when repeatedly allocating and
  freeing medium-sized blocks (128KB-1MB), virtual address space
  fragmentation progresses.

Correct approach:
  1. Migrate to 64-bit build (virtual address space: 128TB+)
  2. If 32-bit is needed, on Linux change 3G/1G split to 3.5G/0.5G
  3. Use memory pools to reduce mmap/munmap calls
  4. Suppress fragmentation with jemalloc or tcmalloc
```

---

## 15. Edge Case Analysis

### Edge Case 1: Page Placement in NUMA Environments

```
NUMA (Non-Uniform Memory Access) environment:

  +-------------+         +-------------+
  |   CPU 0     |         |   CPU 1     |
  |  +-------+  |         |  +-------+  |
  |  | Core0 |  |         |  | Core2 |  |
  |  | Core1 |  |         |  | Core3 |  |
  |  +-------+  |         |  +-------+  |
  |      |      |  QPI/   |      |      |
  |  +-------+  |  UPI    |  +-------+  |
  |  |Local  |  |<------->|  |Local  |  |
  |  |Memory |  |  link   |  |Memory |  |
  |  |(Node0)|  |         |  |(Node1)|  |
  |  +-------+  |         |  +-------+  |
  +-------------+         +-------------+

  Local memory access : ~100ns
  Remote memory access: ~150-300ns (1.5-3x slower)

Problem:
  When the page replacement algorithm selects a "free frame,"
  allocating a frame from a remote node significantly degrades
  performance.

  Example: If a process running on CPU 0 is allocated a frame
  from Node 1, every memory access goes through the QPI/UPI link,
  becoming 1.5-3x slower.

Linux countermeasures:
  - Default policy (local): Allocate frames from the local node
    where the page fault occurred
  - Controllable with numactl / set_mempolicy():
    numactl --membind=0 ./app   # Use only Node 0 memory
    numactl --interleave=all ./app  # Distribute evenly across all nodes

  Monitoring:
    numastat -p <pid>  # Process NUMA memory usage
    cat /proc/buddyinfo  # Free frames per node
```

### Edge Case 2: Memory Pressure and COW Storm After fork()

```
Problem:
  When a process using large amounts of memory (e.g., Redis 10GB) calls fork(),
  COW causes parent and child to share the same physical pages.
  However, if the parent continues writing, a copy occurs for each written page,
  temporarily nearly doubling memory usage.

  Redis 10GB + high write rate:
    Just after fork(): 10GB (shared)
    All pages written to: Up to 20GB needed
    -> If physical memory is only 16GB, OOM Killer fires

  This phenomenon is called a "COW storm" and frequently occurs
  during Redis RDB persistence or BGSAVE.

Countermeasures:
  1. Configure overcommit_memory:
     echo 1 > /proc/sys/vm/overcommit_memory
     -> Always allow fork() (OOM risk exists but BGSAVE won't fail)

  2. Ensure sufficient swap space:
     Absorb temporary memory increase with swap

  3. Avoid using Huge Pages:
     COW copy unit becomes 2MB, increasing cost

  4. Redis 7.0+ also considers fork-less persistence methods
```

---

## 16. Practical Exercises

### Exercise 1 [Basic]: Page Table Size Calculation

```
Problem:
  With 32-bit virtual address space, 4KB page size, PTE = 4 bytes:

  (1) Calculate the size of a single-level page table.
  (2) With a 2-level page table, calculate the minimum page table
      memory needed if a process uses 4MB of memory.
  (3) Explain why multi-level is efficient, using specific numbers.

Solution:
  (1) Page count = 2^32 / 2^12 = 2^20 = 1,048,576 pages
      Table size = 1,048,576 x 4 bytes = 4MB

  (2) 4MB = 1024 pages = 1 page table (1024 entries)
      Page directory: 4KB (1024 entries x 4 bytes)
      Page table: 4KB x 1 (4MB fits exactly in one page directory entry's
                 coverage of 4MB)
      Total: 4KB + 4KB = 8KB

  (3) Single-level: 4MB needed for every process (even if using only 1 byte)
      2-level: 8KB to 4MB+4KB depending on usage
      With 100 processes:
        Single-level: 100 x 4MB = 400MB consumed by page tables alone
        2-level (each using 4MB): 100 x 8KB = 800KB -> 500x savings
```

### Exercise 2 [Applied]: TLB Hit Rate and Effective Access Time

```
Problem:
  Calculate the effective memory access time under these conditions:

  Conditions:
    TLB access time: 1ns
    Memory access time: 100ns
    Number of page table levels: 4
    TLB hit rate: 95%

  (1) Access time on TLB hit
  (2) Access time on TLB miss
  (3) Effective Access Time (EAT)
  (4) How many times faster compared to no TLB
  (5) What is the EAT if hit rate improves to 99%

Solution:
  (1) TLB hit: 1ns (TLB) + 100ns (memory) = 101ns

  (2) TLB miss: 1ns (TLB) + 4 x 100ns (page walk) + 100ns (memory)
              = 1 + 400 + 100 = 501ns

  (3) EAT = 0.95 x 101 + 0.05 x 501
          = 95.95 + 25.05 = 121ns

  (4) Without TLB: 4 x 100 + 100 = 500ns
      Speedup: 500 / 121 = 4.13x

  (5) EAT(99%) = 0.99 x 101 + 0.01 x 501
               = 99.99 + 5.01 = 105ns
      -> 4% hit rate improvement yields 13% performance improvement
```

### Exercise 3 [Advanced]: Comparative Simulation of Page Replacement Algorithms

```
Problem:
  For the following page reference string, calculate page fault counts
  for each algorithm with 3 frames, and compare:

  Reference string: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5

  (1) Find FIFO page fault count and show frame state at each step
  (2) Find LRU page fault count and show frame state at each step
  (3) Find OPT page fault count and show frame state at each step
  (4) Find FIFO results with 4 frames and verify whether
      Belady's Anomaly occurs

Solution (FIFO, frames=3):
  Step 1: Page 1 -> [1, -, -] FAULT
  Step 2: Page 2 -> [1, 2, -] FAULT
  Step 3: Page 3 -> [1, 2, 3] FAULT
  Step 4: Page 4 -> [4, 2, 3] FAULT (replace 1)
  Step 5: Page 1 -> [4, 1, 3] FAULT (replace 2)
  Step 6: Page 2 -> [4, 1, 2] FAULT (replace 3)
  Step 7: Page 5 -> [5, 1, 2] FAULT (replace 4)
  Step 8: Page 1 -> [5, 1, 2] HIT
  Step 9: Page 2 -> [5, 1, 2] HIT
  Step10: Page 3 -> [3, 1, 2] FAULT (replace 5)  *FIFO order: 5 is oldest
  Step11: Page 4 -> [3, 4, 2] FAULT (replace 1)
  Step12: Page 5 -> [3, 4, 5] FAULT (replace 2)
  FIFO faults: 10

  * LRU and OPT can be verified with Code Example 2 in this chapter
```

---

## 17. FAQ

### Q1: What does malloc() actually do? What is its relationship to paging?

`malloc()` is a C library function (such as glibc's ptmalloc2) and is not a direct system call. Its internal operation is as follows:

1. **Small allocations (< ~128KB)**: Extends the heap with `brk()` / `sbrk()`. glibc internally manages free lists and reuses freed memory.
2. **Large allocations (>= ~128KB)**: Allocates a new virtual memory region with `mmap(MAP_ANONYMOUS | MAP_PRIVATE)`. On free, returns it to the OS with `munmap()`.
3. **Physical memory is allocated on access**: `malloc()` returns only a virtual address. The actual physical frame is allocated through demand paging when that address is first accessed.

This is why "malloc success != physical memory allocation." Linux's overcommit mechanism allows `malloc()` to allocate virtual memory exceeding the total of physical memory + swap.

### Q2: Is kernel space memory paged?

Kernel space memory management differs from user space:

- **Kernel code and data**: Loaded into physical memory at boot and normally not swapped. Kernel page tables are permanently mapped.
- **SLUB allocator objects**: Small objects within the kernel (`struct task_struct`, etc.) are managed by Buddy System + SLUB and are not subject to swap.
- **Page cache**: Cache pages used for file reads/writes are reclaimed (freed) under memory pressure. However, this is not "swap out" but rather writing back to the file and then freeing the frame.
- **vmalloc region**: Allocates virtually contiguous memory within the kernel. Physically discontinuous, but mapped contiguously via page tables.

In summary, the kernel's own code and data structures are not swapped, but page cache managed by the kernel is subject to reclamation.

### Q3: How is virtual machine (VM) memory managed?

In virtualization environments, paging occurs in **two stages**:

```
Two-stage address translation (Nested Paging / EPT):

  Guest OS:
    Guest Virtual Address (GVA)
      -> Guest page table
    Guest Physical Address (GPA)

  Hypervisor:
    Guest Physical Address (GPA)
      -> Nested page table (EPT / NPT)
    Host Physical Address (HPA)

  GVA -> GPA -> HPA two-stage translation

  Cost on TLB miss:
    Guest 4 levels x Host 4 levels = up to 24 memory accesses
    -> Hardware support (Intel EPT / AMD NPT) caches GVA->HPA
      directly in TLB, minimizing performance degradation
```

Furthermore, hypervisors can dynamically adjust VM memory amounts using **ballooning** technology. The balloon driver "consumes memory" within the guest OS, applying memory pressure to the guest OS and causing it to swap out unnecessary pages. Reclaimed frames are redistributed to other VMs.

### Q4: Can the page size be changed?

On x86-64, page sizes are fixed by hardware to 4KB / 2MB / 1GB, and the OS cannot choose arbitrary sizes. However, ARM architecture allows selecting base page sizes of 4KB / 16KB / 64KB at boot time, and Apple Silicon (macOS / iOS) uses 16KB pages.

Impact of 16KB pages:
- TLB coverage expands 4x (covers a wider range with the same number of TLB entries)
- Internal fragmentation increases to a maximum of 16KB-1 (average 8KB)
- I/O efficiency improves (16KB transferred per page fault)
- Compatibility issues with software assuming 4KB may occur

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is most important. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this used in practice?

Knowledge of this topic is frequently used in daily development work. It becomes especially important during code reviews and architecture design.

---

## 18. Summary

| Concept | Key Point |
|---------|-----------|
| Paging | Fixed-size (4KB) division. No external fragmentation. Foundation of modern memory management |
| Multi-level page table | Omits tables for unused regions, saving memory |
| TLB | Address translation cache. Achieves practical performance at 99% hit rate |
| Demand paging | Allocates physical pages only on access. Reduces startup time and memory usage |
| COW (Copy-on-Write) | Efficiency optimization for fork(). Shares pages until write occurs |
| Page replacement | LRU is theoretically near-optimal, but Clock is used in practice due to implementation cost |
| Swapping | Uses disk as an extension of virtual memory. Cost of Major page faults is enormous |
| Thrashing | Severe performance degradation when working sets don't fit in physical memory |
| Huge Pages | Expands TLB coverage. Effective for large-memory applications |
| NUMA | Physical memory placement affects performance. Local-node-first page placement is important |

---

## 19. Glossary

| Term | English | Description |
|------|---------|-------------|
| Page | Page | Fixed-size unit of virtual address space (typically 4KB) |
| Frame | Frame | Fixed-size unit of physical memory |
| VPN | Virtual Page Number | Page number portion of virtual address |
| PFN | Physical Frame Number | Frame number portion of physical address |
| PTE | Page Table Entry | One entry in the page table |
| TLB | Translation Lookaside Buffer | Address translation cache |
| ASID | Address Space Identifier | Process identifier in TLB entries |
| COW | Copy-on-Write | Copy on write |
| OOM | Out of Memory | Memory exhaustion state |
| WSS | Working Set Size | Size of the working set |
| NUMA | Non-Uniform Memory Access | Non-uniform memory access |
| THP | Transparent Huge Pages | Transparent large pages |

---

## Recommended Next Guides


---

## References

1. Silberschatz, A., Galvin, P. B., & Gagne, G. "Operating System Concepts." 10th Edition, Chapter 9-10 (Virtual Memory), Wiley, 2018.
   - The definitive textbook comprehensively covering paging, demand paging, and page replacement algorithm theory.

2. Bovet, D. P. & Cesati, M. "Understanding the Linux Kernel." 3rd Edition, Chapter 2, 8-9, O'Reilly, 2005.
   - Detailed description of Linux kernel memory management implementation (Buddy System, SLUB, page fault handler).

3. Gorman, M. "Understanding the Linux Virtual Memory Manager." Prentice Hall, 2004. (https://www.kernel.org/doc/gorman/)
   - Comprehensive explanation of Linux's virtual memory subsystem. Provides implementation-level information on page reclaim, swap, and NUMA.

4. Intel Corporation. "Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3A: System Programming Guide." Chapter 4 (Paging), 2024.
   - Hardware specification for x86-64 paging mechanisms (4-level page tables, TLB, EPT).

5. Love, R. "Linux Kernel Development." 3rd Edition, Chapter 15 (The Process Address Space), Addison-Wesley, 2010.
   - Accessible explanation of Linux kernel memory management from a developer's perspective. Includes VMA, demand paging, and COW implementation.

6. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Chapter 18-23, Arpaci-Dusseau Books, 2018. (https://pages.cs.wisc.edu/~remzi/OSTEP/)
   - Free online textbook that progressively explains paging, TLB, page replacement, and swapping. Strongly recommended for beginners.
