# Memory Hierarchy

> "Memory is fast but small; storage is large but slow" -- this constraint governs all of computer architecture.

## Learning Objectives

- [ ] Systematically explain the characteristics (speed, capacity, cost) of each level of the memory hierarchy
- [ ] Understand cache mapping schemes, replacement algorithms, and write policies
- [ ] Practice cache-friendly programming that leverages the principle of locality
- [ ] Explain the operating principles of virtual memory, paging, and the TLB
- [ ] Acquire practical knowledge of NUMA, Huge Pages, and memory profiling
- [ ] Understand the internal structure and access characteristics of the storage hierarchy (SSD/HDD)

## Prerequisites

- Basic understanding of binary (bits, bytes, address representation)
- Basic reading ability in C or Python

---

## 1. The Memory Hierarchy Pyramid

### 1.1 Overview

A computer's memory system is organized as a hierarchical structure based on tradeoffs among speed, capacity, and cost. Higher levels are faster, smaller, and more expensive; lower levels are slower, larger, and cheaper. This design exploits the "locality" property exhibited by programs, enabling a small amount of fast memory to make a large memory system "appear" to operate at high speed.

```
Memory hierarchy pyramid (representative specs as of 2025):

  Fast, Expensive, Small
  ^
  |  +-----------------+
  |  |   Register      | <- ~0.3ns, ~1KB, inside CPU
  |  |                 |    16 integer regs (x86-64) + SIMD
  |  +-----------------+
  |  |  L1 Cache       | <- ~1ns (3-4 cycles), 64KB x 2
  |  |  (L1i/L1d)      |    Split into instruction and data caches
  |  +-----------------+
  |  |  L2 Cache       | <- ~3-10ns (10-20 cycles), 256KB-2MB
  |  |  (L2 Unified)   |    Dedicated per core
  |  +-----------------+
  |  |  L3 Cache       | <- ~10-30ns (30-70 cycles), 8-96MB
  |  |  (L3/LLC)       |    Shared across all cores, inclusive or non-inclusive
  |  +-----------------+
  |  |  Main Memory    | <- ~50-100ns, 8-256GB, DDR5-5600, etc.
  |  |  (DRAM/RAM)     |    Volatile, random-access capable
  |  +-----------------+
  |  |  NVMe SSD       | <- ~10-100us, 256GB-8TB
  |  |  (Flash NAND)   |    Non-volatile, block-level access
  |  +-----------------+
  |  |  SATA SSD / HDD | <- ~50us-10ms, 256GB-20TB
  |  |                  |    HDD dominated by mechanical seek latency
  |  +-----------------+
  |  |  Tape / Cloud   | <- ~seconds-minutes, PB-scale, archival
  |  |  (Cold Storage)  |    Lowest cost, offline access
  |  +-----------------+
  v
  Slow, Cheap, Large
```

To understand why this hierarchical structure "works," consider the following intuitive analogy. Imagine yourself working at a desk at home.

- **Registers** = Documents in your hand (instant access, but you can only hold a few)
- **L1 Cache** = Papers on your desk (reach out and grab them)
- **L2 Cache** = Desk drawer (requires a little searching, but quick to find)
- **L3 Cache** = Bookshelf in the same room (need to stand up and walk over)
- **DRAM** = Filing cabinet in the next room (walk over to retrieve)
- **SSD** = Document room in the same building (need to take the elevator)
- **HDD** = Warehouse in an adjacent building (need to go outside)
- **Tape** = Warehouse in another city (request by mail)

### 1.2 Latency Comparison (Jeff Dean's Numbers -- 2025 Revised)

| Operation | Latency | Human-Scale Equivalent (L1 = 1 second) |
|-----------|---------|----------------------------------------|
| L1 cache reference | 1 ns | 1 second |
| Branch mispredict | 3 ns | 3 seconds |
| L2 cache reference | 4 ns | 4 seconds |
| L3 cache reference | 12 ns | 12 seconds |
| Mutex lock/unlock | 17 ns | 17 seconds |
| DRAM reference (main memory) | 100 ns | 1 min 40 sec |
| Compress 1KB with Zstd | 3 us | 50 minutes |
| Send 1KB over 1 Gbps network | 10 us | 2.8 hours |
| NVMe SSD random 4KB read | 16 us | 4.4 hours |
| Read 1MB sequentially from NVMe SSD | 49 us | 13.6 hours |
| HDD seek | 2 ms | 23.1 days |
| Read 1MB sequentially from HDD | 825 us | 9.5 days |
| TCP packet round trip (same DC) | 500 us | 5.8 days |
| TCP packet round trip (Tokyo to US West Coast) | 150 ms | 4.8 years |

> **Key insight**: Main memory (DRAM) is roughly 100x slower than L1 cache. An HDD is approximately 2 million times slower than L1. This enormous speed gap is the fundamental motivation behind memory hierarchy design.

### 1.3 Bandwidth Comparison

Along with latency (time per access), bandwidth (data transferred per unit time) is an important performance metric.

| Level | Latency | Bandwidth | Typical Capacity | Approx. Cost per GB |
|-------|---------|-----------|-----------------|---------------------|
| Register | ~0.3ns (1 cycle) | Depends on CPU internal bus width | ~1KB | -- |
| L1 Cache | 1ns (3-4 cycles) | ~1TB/s | 64KB x 2 | -- |
| L2 Cache | 3-10ns (10-20 cycles) | ~500GB/s | 256KB-2MB | -- |
| L3 Cache | 10-30ns (30-70 cycles) | ~200GB/s | 8-96MB | -- |
| DDR5-5600 DRAM | 50-100ns | 45-90GB/s (dual-channel) | 16-256GB | ~$2.5 |
| NVMe SSD (PCIe 4.0) | 10-100us | 3.5-7GB/s | 256GB-8TB | ~$0.07 |
| NVMe SSD (PCIe 5.0) | 10-80us | 10-14GB/s | 512GB-4TB | ~$0.10 |
| SATA SSD | 50-100us | ~560MB/s | 256GB-4TB | ~$0.05 |
| HDD (7200rpm) | 3-10ms | 100-250MB/s | 1-20TB | ~$0.015 |
| Tape (LTO-9) | sec-min | 400MB/s (sequential) | 18TB/cartridge | ~$0.004 |

### 1.4 Why the Hierarchical Structure Is "Economical"

If the entire memory system were built from L1 cache-equivalent SRAM, a 128GB memory system would cost hundreds of thousands of dollars. By adopting a hierarchical structure, a small amount of SRAM (tens of MB) combined with a large amount of inexpensive DRAM (tens of GB) keeps costs to a few hundred dollars while handling most accesses at cache speed.

This works because program behavior is "local." In a typical program, only a small portion of the entire address space (the working set) is actually accessed during any short time period, and as long as that working set fits in the cache, the entire system appears to operate at cache speed.

---

## 2. Registers and SRAM

### 2.1 Registers -- The CPU's Fastest Memory

Registers are the smallest and fastest storage elements built directly into the CPU. They are connected directly to the ALU (Arithmetic Logic Unit), enabling data reads and writes with no wire delay.

**x86-64 Architecture Register Layout:**

```
General-purpose registers (64-bit x 16):
+---------------------------------------------------+
| RAX  RBX  RCX  RDX  RSI  RDI  RBP  RSP           |
| R8   R9   R10  R11  R12  R13  R14  R15            |
+---------------------------------------------------+
| SIMD/Vector registers:                            |
| XMM0-XMM15  (128-bit x 16) ... SSE               |
| YMM0-YMM15  (256-bit x 16) ... AVX/AVX2          |
| ZMM0-ZMM31  (512-bit x 32) ... AVX-512           |
+---------------------------------------------------+
| Special registers:                                |
| RIP (instruction pointer)                         |
| RFLAGS (flags register)                           |
| CR0-CR4 (control registers)                       |
| CS, DS, SS, ES, FS, GS (segment registers)       |
+---------------------------------------------------+

Total capacity: GP 16x8B=128B + SIMD(AVX-512) 32x64B=2048B + special ~ a few KB
```

**Comparison with ARM (AArch64) Architecture:**

| Feature | x86-64 | AArch64 (ARM v8) |
|---------|--------|------------------|
| GP registers | 16 | 31 |
| Register width | 64-bit | 64-bit |
| SIMD registers | ZMM 32 (AVX-512) | V0-V31 32 (NEON/SVE) |
| Characteristics | CISC, variable-length instructions | RISC, fixed-length instructions |

### 2.2 SRAM vs DRAM -- Two Memory Technologies

SRAM (Static RAM) used in caches and DRAM (Dynamic RAM) used for main memory have fundamentally different cell structures.

```
SRAM cell (6-transistor configuration):
+-----------------------------+
|       VDD                    |
|    +--+--+                   |
|  +-| P1  |-+  +-| P2 |-+    |
|  | +-----+ |  | +-----+ |   |
|  |         |  |         |   |
|  +-| N1 |--+--+-| N2 |--+   |
|  | +-----+ |  | +-----+ |   |
|  |         |  |         |   |
|  |    GND  |  |    GND  |   |
|  |         |  |         |   |
| BL   +----+  +----+   BL'  |
|  |   | Access     |   |    |
|  +---| Transistor |---+    |
|      +------+-----+        |
|         Word Line           |
+-----------------------------+
- 6 transistors hold 1 bit
- Data is stable as long as power is supplied
- No refresh required -> fast access possible
- Large cell size -> high cost per bit

DRAM cell (1 transistor + 1 capacitor):
+-----------------------------+
|  Bit Line                    |
|     |                        |
|  +--+--+                     |
|  | Access|                   |
|  | Trans.|                   |
|  +--+--+                     |
|     |                        |
|  +--+--+                     |
|  | Cap  | <- Bit represented |
|  |  C   |   by charge        |
|  +--+--+   (charged=1,       |
|    GND      discharged=0)    |
+-----------------------------+
- 1 transistor + 1 capacitor per bit
- Capacitor charge leaks over time
- Periodic refresh required (~64ms interval)
- Small cell size -> easy to scale to large capacity
```

**SRAM vs DRAM Comparison:**

| Feature | SRAM | DRAM |
|---------|------|------|
| Cell structure | 6 transistors | 1 transistor + 1 capacitor |
| Access speed | ~1-2ns | ~50-100ns |
| Refresh | Not required | Required (~64ms interval) |
| Density | Low (large cell) | High (small cell) |
| Power consumption | Low (standby) | Consumes power for refresh |
| Cost/bit | High (~30-50x DRAM) | Low |
| Primary use | CPU caches (L1/L2/L3) | Main memory |
| Manufacturing process | Compatible with logic process | Dedicated process |

---

## 3. How Caches Work

### 3.1 Why Caches Are Needed -- The Memory Wall Problem

A serious and widening gap exists between CPU processing speed and memory response time. This is known as the "Memory Wall Problem."

```
CPU-memory speed gap (Memory Wall Problem):

  Relative
  performance
  |
  |   CPU performance     /
  |                  /
  |              /        <- ~50-60% per year (Moore's Law era)
  |            /              Slowed to ~20%/year since 2010s
  |          /
  |        /
  |      /
  |    /     <---- This gap is the "Memory Wall"
  |  /
  |/
  |-- -- -- -- -- -- Memory bandwidth
  |---------------- Memory latency <- ~7% improvement per year
  |
  +------------------------------------------------ Year
   1980        1990        2000        2010        2025

  1980: 1 CPU cycle ~ 1 memory cycle
  2000: 1 CPU cycle ~ 100 memory cycles
  2025: 1 CPU cycle ~ 200-300 memory cycles

  -> Without caches, the CPU would spend >99% of its time waiting for memory
```

The essence of this problem is that DRAM latency improvements have not kept up with CPU speed gains. DRAM bandwidth has improved relatively well (DDR5 is roughly 2x DDR4), but latency improvement is minimal. Caches are a mechanism to "hide" this speed gap by exploiting locality.

### 3.2 Basic Cache Operation

A cache holds partial copies of main memory in fast SRAM. When the CPU accesses a memory address, it first checks the cache; if the data is present (cache hit), it is retrieved quickly; if absent (cache miss), data is fetched from a lower level and stored in the cache.

```
Basic cache operation flow:

  CPU requests data at address A
  |
  v
  Search L1 cache
  |
  +-- Hit -> Return data to CPU (~1ns)
  |           * Fastest path
  |
  +-- Miss -> Search L2 cache
              |
              +-- Hit -> Store in L1 and return to CPU (~4ns)
              |
              +-- Miss -> Search L3 cache
                          |
                          +-- Hit -> Store in L2, L1 and return (~12ns)
                          |
                          +-- Miss -> Access DRAM
                                      |
                                      +-- Store in L3, L2, L1 and return (~100ns)
                                           * ~100x penalty compared to L1
```

### 3.3 Cache Lines -- The Minimum Unit of Data Transfer

Data transfer between cache and memory occurs not byte by byte but in fixed-size blocks called "cache lines" (typically 64 bytes).

```
Cache line structure (64 bytes):

  One cache line:
  +------+-------+---------------------------------------------+
  |Valid | Tag   |              Data (64 bytes)                 |
  | (1b) |(upper)|  byte0  byte1  byte2  ...  byte62  byte63   |
  +------+-------+---------------------------------------------+

  Memory address decomposition (64-byte line, 256-set, 8-way cache):
  +-------------------+----------+----------+
  |       Tag          |  Index   |  Offset  |
  |  (remaining upper  | (8 bits) | (6 bits) |
  |   bits)            |          |          |
  +-------------------+----------+----------+
                         |           |
                         |           +- Byte position within the cache line
                         |              (64 bytes = 2^6 -> 6 bits)
                         |
                         +- Which set to store in
                            (256 sets = 2^8 -> 8 bits)

  Example: int array a[16] contiguous in memory
  +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+------+
  |a[0]|a[1]|a[2]|a[3]|a[4]|a[5]|a[6]|a[7]|a[8]|a[9]|... |... |... |... |... |a[15]|
  +----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+------+
  |<------------ 1 cache line (64B) ----------->|<------- next cache line -------->|
  int is 4 bytes, so 16 ints fit in 1 cache line
  -> Accessing a[0] loads a[0]-a[15] into the cache at once
  -> Subsequent accesses to a[1]-a[15] are all cache hits
```

### 3.4 Cache Mapping Schemes

There are three schemes that determine where in the cache a memory address is stored.

```
Three cache mapping schemes:

1. Direct-Mapped:
   Each memory address can only be stored in one specific cache line

   Memory blocks:  0  1  2  3  4  5  6  7  8  9 10 11
   Cache lines:    +--+
                   | 0| <- blocks 0, 4, 8, ... stored here
                   | 1| <- blocks 1, 5, 9, ...
                   | 2| <- blocks 2, 6, 10, ...
                   | 3| <- blocks 3, 7, 11, ...
                   +--+
   Location = block number mod number of cache lines

   Pros: Simple and fast index computation, small hardware
   Cons: High conflict misses (e.g., alternating access to blocks 0 and 4
         causes a cache miss every time = thrashing)

2. Fully Associative:
   Any memory block can be stored in any cache location

   Cache lines:    +--+
                   |  | <- any block can go here
                   |  | <- any block can go here
                   |  | <- any block can go here
                   |  | <- any block can go here
                   +--+
   Pros: No conflict misses
   Cons: Must compare all tags simultaneously
         -> Large circuit area and power; impractical for large caches
   Use cases: TLB (small number of entries), some small caches

3. Set-Associative: * The modern CPU standard
   Cache is divided into multiple sets, each with N lines (ways)
   The set is determined by the address; which way within the set is flexible

   8-Way Set-Associative example:
   +------------------------------------------------------+
   |Set 0: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]|
   |Set 1: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]|
   |Set 2: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]|
   |  ...                                                   |
   |Set N: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]|
   +------------------------------------------------------+

   Target set = block number mod number of sets
   Which way within the set is determined by the replacement policy (LRU, etc.)

   Pros: Balance between direct-mapped speed and fully associative flexibility
   Typical values in modern CPUs:
     L1: 8-way, 64 sets (64 x 8 x 64B = 32KB)
     L2: 4-8-way
     L3: 12-16-way
```

### 3.5 Cache Replacement Policies

When the cache is full, the algorithm that determines which line to evict to make room for new data is called the "replacement policy."

| Policy | Mechanism | Pros | Cons | Use Cases |
|--------|-----------|------|------|-----------|
| **LRU** (Least Recently Used) | Evict the least recently used line | Effective for temporal locality | High hardware cost with many ways | L1/L2 (with few ways) |
| **Pseudo-LRU** | Approximate LRU using a tree structure | Lower cost than LRU | Not strictly LRU | L1/L2/L3 (mainstream in modern CPUs) |
| **RRIP** (Re-Reference Interval Prediction) | Predict re-reference interval for eviction | Scan-resistant | Slightly complex implementation | L3 (Intel) |
| **Random** | Select randomly | Simplest | Far from optimal | Some ARM processors |
| **FIFO** | Evict the oldest line | Simple | May evict recently used data | Software caches |

### 3.6 Cache Write Policies

There are two policies for maintaining consistency between cache and main memory on writes.

```
Write policies:

1. Write-Through:
   On write, both cache and main memory are updated immediately

   CPU -> Write -> [L1 Cache] -> simultaneously -> [DRAM]
                   (update)                        (update)

   Pros: Cache and memory always agree (consistency guaranteed)
   Cons: Memory access on every write (slow)
   Mitigation: Write buffer temporarily buffers writes

2. Write-Back: * The mainstream in modern CPUs
   On write, only the cache is updated; main memory is updated on eviction

   CPU -> Write -> [L1 Cache] (dirty bit = 1)
                   (update)

   On eviction:
   [L1 Cache] (dirty bit == 1) -> write back to [DRAM]

   Pros: Significantly reduces memory accesses for write-heavy workloads
   Cons: Cache and memory temporarily out of sync
         Complex cache coherency in multicore systems
```

### 3.7 Cache Coherency -- A Multicore Challenge

In multicore processors, each core has its own L1/L2 cache, so the same memory address may exist in multiple caches with different values. This is called the "cache coherency problem."

```
Cache coherency problem example:

  Core 0                    Core 1
  +----------+              +----------+
  | L1 Cache |              | L1 Cache |
  | X = 42   |              | X = 42   |  <- Initial state: both have X=42
  +----+-----+              +----+-----+
       |                         |
       |  Core 0 updates X = 100
       |  +----------+
       |  | X = 100  | <- Core 0's cache is updated
       |  +----------+
       |                    +----------+
       |                    | X = 42   | <- Core 1 still holds the old value!
       |                    +----------+
       |                         |
       +--------+----------------+
                |
  +-------------------------+
  | Main memory: X = 42     | <- Write-back, so still the old value
  +-------------------------+

  -> Core 1 reading X returns 42 (stale value) = data inconsistency!
```

The **MESI protocol** (and its extensions) is used to solve this problem.

| State | Name | Meaning |
|-------|------|---------|
| **M** | Modified | Only this cache has the latest value; memory is stale |
| **E** | Exclusive | Only this cache has a copy; matches memory |
| **S** | Shared | Multiple caches have copies; matches memory |
| **I** | Invalid | This cache line is invalid (unusable) |

In the MESI protocol, when a core writes to data in Shared state, it invalidates the corresponding cache lines in all other cores (invalidation). This maintains consistency but incurs overhead from bus communication (snooping) in multicore environments.

> **False Sharing**: Even when different cores access "different variables," if those variables reside on the same cache line, the MESI protocol triggers unnecessary invalidations, causing severe performance degradation. This is an important anti-pattern in multithreaded programming (detailed in the Anti-Patterns section below).

---

## 4. The Principle of Locality

### 4.1 Overview

The memory hierarchy works efficiently because program memory access patterns are "local." This property is called the "Principle of Locality." There are two forms of locality.

### 4.2 Temporal Locality

"Data accessed recently is likely to be accessed again in the near future."

```python
# Temporal locality example: loop counters and accumulator variables
def compute_sum(data: list[int]) -> int:
    total = 0                      # total: very high temporal locality
    count = 0                      # count: very high temporal locality
    for i in range(len(data)):     # i: high temporal locality
        total += data[i]
        count += 1
    return total // count if count > 0 else 0
    # total, count, i are accessed repeatedly across all loop iterations
    # -> The compiler assigns them to registers (register allocation optimization)
    # -> If they don't fit in registers, they remain in L1 cache
```

Examples of data with high temporal locality:
- Loop counters, accumulator variables
- Code of frequently called functions
- Global variables, root nodes of frequently accessed data structures

### 4.3 Spatial Locality

"After accessing a given address, nearby addresses are likely to be accessed in the near future."

```python
# Spatial locality example: sequential array access
def process_image(pixels: list[int], width: int, height: int) -> None:
    # Row-major sequential access -> high spatial locality
    for y in range(height):
        for x in range(width):
            pixels[y * width + x] = transform(pixels[y * width + x])
    # pixels[0], pixels[1], pixels[2], ... are contiguous in memory
    # -> One cache line load (64B) covers 16 ints
    # -> Cache miss rate = 1/16 = 6.25% (theoretical)
```

Examples of access patterns with high spatial locality:
- Sequential array traversal
- Struct field access (fields are contiguous in memory)
- Sequential instruction execution (program counter increment)

### 4.4 The 3C Classification of Cache Misses

Cache misses are classified into three categories by cause (3C classification: Compulsory, Capacity, Conflict).

| Type | English Name | Cause | Countermeasure |
|------|-------------|-------|----------------|
| **Compulsory (Cold) Miss** | Compulsory (Cold) Miss | First access to that data. Data does not exist in the cache | Hardware prefetching, software prefetch instructions |
| **Capacity Miss** | Capacity Miss | Working set exceeds cache capacity | Reduce working set, optimize data structures, cache blocking |
| **Conflict Miss** | Conflict Miss | Different addresses map to the same set and evict each other | Increase associativity, adjust data placement, padding |

### 4.5 Code Example: Quantifying Locality

```c
/* Row-major vs. column-major access: cache miss rate differences */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096  /* 4096 x 4096 = 16M elements, 64MB (int) */

int matrix[N][N];

/* Row-major access (spatial locality present) */
long long sum_row_major(void) {
    long long sum = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += matrix[i][j];   /* matrix[i][0], [i][1], [i][2]... are contiguous */
    return sum;                     /* Cache miss rate: ~1/16 = 6.25% */
}

/* Column-major access (no spatial locality) */
long long sum_col_major(void) {
    long long sum = 0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            sum += matrix[i][j];   /* matrix[0][j], [1][j], [2][j]... stride N */
    return sum;                     /* Cache miss rate: ~100% (for large N) */
}

int main(void) {
    /* Initialize array */
    srand(42);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = rand() % 100;

    clock_t start, end;

    start = clock();
    long long s1 = sum_row_major();
    end = clock();
    printf("Row-major: sum=%lld, time=%.3fs\n",
           s1, (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    long long s2 = sum_col_major();
    end = clock();
    printf("Col-major: sum=%lld, time=%.3fs\n",
           s2, (double)(end - start) / CLOCKS_PER_SEC);

    /* Typical results:
     * Row-major: ~0.05s
     * Col-major: ~0.30s (6x slower)
     * The gap widens as N grows (exceeding L3 cache)
     */
    return 0;
}
```

---

## 5. Virtual Memory

### 5.1 Overview and Purpose

Virtual memory is an OS/hardware-cooperative mechanism that provides each process with an independent, contiguous address space, enabling efficient physical memory management and inter-process protection.

Three primary purposes of virtual memory:

1. **Abstraction**: Each process behaves as if it has its own vast, contiguous address space
2. **Protection**: Prevents one process from reading or writing another process's memory
3. **Efficiency**: Hides physical memory fragmentation and allocates physical memory only to pages actually in use

```
Virtual memory conceptual diagram:

  Process A's virtual address space        Physical memory (RAM)
  +----------------------------+       +--------------------+
  | 0x0000_0000: Code (.text)  |------>| Frame 5: ...       |
  | 0x0040_0000: Data (.data)  |------>| Frame 8: ...       |
  | 0x0080_0000: Heap          |------>| Frame 12: ...      |
  | 0x00C0_0000: (unmapped)    |       | Frame 13: (free)   |
  | ...                        |       | Frame 14: ...      |
  | 0x7FFF_0000: Stack         |------>| Frame 20: ...      |
  +----------------------------+       |                     |
                                       |                     |
  Process B's virtual address space    |                     |
  +----------------------------+       |                     |
  | 0x0000_0000: Code (.text)  |------>| Frame 2: ...       |
  | 0x0040_0000: Data (.data)  |------>| Frame 7: ...       |
  | 0x0080_0000: Heap          |------>| Frame 15: ...      |
  | ...                        |       |                     |
  | 0x7FFF_0000: Stack         |------>| Frame 22: ...      |
  +----------------------------+       +--------------------+
                                              |
  Both processes use the same virtual         |
  address (0x0000_0000), but they are         v
  mapped to different physical frames     +----------+
                                          | SSD/HDD  |
  When physical memory runs out           | (Swap)   |
  -> page out --------------------------> +----------+
```

### 5.2 How Paging Works

Both the virtual address space and physical memory are divided into fixed-size blocks called "pages" (typically 4KB). The "page table" holds the mapping from virtual pages to physical frames.

```
x86-64 4-level page table:

Virtual address (48 bits effective):
+---------+---------+---------+---------+--------------+
| PML4    |  PDPT   |   PD    |   PT    |  Page Offset |
| (9bit)  | (9bit)  | (9bit)  | (9bit)  |   (12bit)    |
| [47:39] | [38:30] | [29:21] | [20:12] |   [11:0]     |
+----+----+----+----+----+----+----+----+--------------+
     |         |         |         |
     v         v         v         v
  +------+ +------+ +------+ +------+
  |PML4  |>|PDPT  |>|Page  |>|Page  |> Physical frame
  |Table | |Table | |Dir   | |Table |   + Offset
  |(512  | |(512  | |(512  | |(512  |
  |entries| |entries| |entries| |entries|
  +------+ +------+ +------+ +------+

  Each table: 512 entries x 8 bytes = 4KB (fits in one page)
  Virtual address space: 2^48 = 256TB
  Physical page size: 4KB (2^12)

  Cost of a page table walk:
  Up to 4 memory accesses = 4 x 100ns = 400ns
  -> Too slow, so the TLB is used to accelerate this
```

### 5.3 TLB (Translation Lookaside Buffer)

The TLB is a dedicated, fast associative memory that caches the translation results from virtual page numbers (VPN) to physical frame numbers (PFN).

```
Address translation mechanism:

  Virtual address
  +------------------+------------+
  | Virtual Page #   | Offset     |
  | (VPN)            | (12bit)    |
  +------+-----------+------------+
         |
         v
  +--------------+    TLB miss    +----------------------+
  |     TLB      | ------------> | Page table walk       |
  |  (fast       |               | (4-level memory refs) |
  |  associative |               |                       |
  |  memory)     |               | Result stored in TLB  |
  |              |               +----------+-----------+
  |  L1 iTLB:   |                          |
  |   64-128    |                          |
  |   entries   |                          |
  |  L1 dTLB:   |  Page fault              |
  |   64-128    |  (page not in            |
  |   entries   |   physical memory)       |
  |  L2 TLB:    |         |                |
  |   1024-2048 |         v                |
  |   entries   |  OS loads page from      |
  +------+------+  disk                    |
         | TLB hit                         |
         v                                 v
  Physical address
  +------------------+------------+
  | Physical Frame # | Offset     |
  | (PFN)            | (12bit)    |
  +------------------+------------+

  TLB hit cost: ~1ns (integrated into the pipeline)
  TLB miss cost: ~10-100ns (page table walk)
  Page fault cost: ~1ms (SSD) / ~10ms (HDD)

  TLB coverage = TLB entries x page size
  Example: 1024 entries x 4KB = 4MB
  Example: 1024 entries x 2MB (Huge Pages) = 2GB <- significant improvement
```

### 5.4 Page Faults

A page fault is an exception (hardware interrupt) that occurs when there is no physical frame corresponding to a virtual page.

```
Page fault handling flow:

  1. CPU accesses virtual address VA
  2. TLB miss -> page table walk
  3. Page table entry's Present bit = 0
  4. * Page fault exception occurs
  5. CPU suspends current instruction and transfers control to the OS
     page fault handler
  6. OS determines:
     +-- Invalid access (segmentation fault)
     |   -> Send SIGSEGV and terminate the process
     |
     +-- Demand paging (first access)
     |   -> Allocate a new physical frame, zero-fill, and return
     |
     +-- Page has been swapped out
     |   -> Load page from disk (~1ms SSD / ~10ms HDD)
     |
     +-- Copy-on-Write (CoW)
         -> Copy the page and make it writable
  7. Update the page table (Present=1, set PFN)
  8. Add new entry to the TLB
  9. Re-execute the suspended instruction

  Cost analysis:
  - Minor page fault (no disk I/O): ~1-10us
  - Major page fault (disk I/O required): ~1ms (SSD) / ~10ms (HDD)
  -> A major page fault is 10,000-100,000x slower than a normal memory access
  -> Frequent page faults (thrashing) effectively halt the system
```

### 5.5 Page Replacement Algorithms

When physical memory is full, page replacement algorithms determine which page to swap out to make room for a new one.

| Algorithm | Overview | Characteristics |
|-----------|----------|-----------------|
| **OPT** (Optimal) | Evict the page that will not be used for the longest time in the future | Theoretically optimal but impossible to implement. Used as a benchmark |
| **LRU** (Least Recently Used) | Evict the page not referenced for the longest time | Exploits temporal locality. Exact implementation is expensive |
| **Clock** (Second Chance) | Circular list with reference bits. Evict pages with reference bit = 0 | LRU approximation. Widely used in Linux, etc. |
| **LFU** (Least Frequently Used) | Evict the page with the fewest references | Considers long-term frequency. Old-but-once-frequent pages may persist |

---

## 6. Storage Hierarchy: SSD and HDD

### 6.1 HDD (Hard Disk Drive) Structure and Characteristics

An HDD is a mechanical storage device that records data on magnetic disks.

```
HDD internal structure:

  +----------------------------------+
  |         Spindle motor            |
  |              |                   |
  |    +---------+---------+         |
  |    |    +----+----+    |         |
  |    |    | Platter  |    | <- Magnetic disks (multiple platters)
  |    |    | (rotates)|    |    7200rpm = 120 rotations/sec
  |    |    +----------+    |         |
  |    |         ^          |         |
  |    |    +----+----+    |         |
  |    |    |  Head    |    | <- Read/write head
  |    |    +----------+    |         |
  |    |         ^          |         |
  |    |    +----+----+    |         |
  |    |    |   Arm    |----+         |
  |    |    +----------+              |
  |    |         ^                    |
  |    |    Actuator                  |
  |    +------------------------------
  +----------------------------------+

  Access time breakdown:
  +--------------------------------------+
  | Seek time     | Rotational | Transfer |
  | (head move)   | latency    | time     |
  | ~3-10ms       | ~2-4ms     | ~0.01ms  |
  | * dominant    | * important| relatively|
  |               |            | small     |
  +--------------------------------------+

  For a 7200rpm HDD:
  - Average seek time: ~4-8ms
  - Average rotational latency: 1/(7200/60)/2 = ~4.17ms
  - Average access time: ~8-12ms
  - Sequential read bandwidth: 100-250MB/s
  - Random 4KB read: ~100 IOPS
```

### 6.2 SSD (Solid State Drive) Structure and Characteristics

An SSD is a semiconductor storage device using NAND flash memory. With no moving parts, random access is orders of magnitude faster than HDD.

```
SSD internal architecture:

  +----------------------------------------------+
  |                SSD Controller                 |
  |  +--------+ +--------+ +--------------+      |
  |  | FTL    | | Wear   | | ECC Engine   |      |
  |  |(Flash  | |Leveling| |              |      |
  |  |Transl.)| |        | |              |      |
  |  +--------+ +--------+ +--------------+      |
  |                |                              |
  |    +-----------+----------+                   |
  |    |           |          |                   |
  |  +-+--+     +-+--+    +-+--+                  |
  |  |Ch 0|    |Ch 1|    |Ch N|  <- Channels      |
  |  +-+--+    +-+--+    +-+--+                   |
  |    |          |          |                     |
  |  +-+--+    +-+--+    +-+--+                   |
  |  |NAND|    |NAND|    |NAND|  <- NAND chips     |
  |  |Die |    |Die |    |Die |                    |
  |  +----+    +----+    +----+                    |
  +----------------------------------------------+

  NAND flash characteristics:
  - Read: page-level (4-16KB)
  - Write: page-level (4-16KB)
  - Erase: block-level (256KB-several MB) * larger unit than reads/writes
  - Limited erase cycles (TLC: ~1000-3000, QLC: ~100-1000)

  Write Amplification:
  - A 4KB logical write may require a block erase + rewrite of
    256KB+ of physical writes
  - The FTL (Flash Translation Layer) minimizes this
```

**SSD vs HDD Comparison:**

| Feature | NVMe SSD (PCIe 4.0) | SATA SSD | HDD (7200rpm) |
|---------|---------------------|----------|---------------|
| Random read | ~16us | ~50us | ~8ms |
| Random write | ~16us | ~50us | ~8ms |
| Sequential read | ~7GB/s | ~560MB/s | ~200MB/s |
| Sequential write | ~5GB/s | ~530MB/s | ~200MB/s |
| Random 4K IOPS (read) | ~500K-1M | ~90K | ~100 |
| Random 4K IOPS (write) | ~500K-1M | ~80K | ~100 |
| Power (active) | 5-10W | 2-5W | 5-10W |
| Power (idle) | ~30mW | ~30mW | 3-6W |
| Vibration resistance | High | High | Low (moving parts) |
| Lifespan | TBW-dependent | TBW-dependent | MTBF ~1M hours |
| Price per 1TB | ~$70-100 | ~$50-70 | ~$15-25 |

---

## 7. NUMA (Non-Uniform Memory Access)

### 7.1 NUMA Architecture Overview

In multi-socket servers, each CPU socket has its own "local memory," and accessing another socket's memory (remote access) incurs additional latency via the interconnect. An architecture with such non-uniform memory access characteristics is called NUMA.

```
NUMA architecture (2-socket server example):

  +---------------------------+    +---------------------------+
  |       NUMA Node 0         |    |       NUMA Node 1         |
  |                           |    |                           |
  |  +------+  +------+      |    |  +------+  +------+      |
  |  |Core0 |  |Core1 |      |    |  |Core8 |  |Core9 |      |
  |  | L1/L2|  | L1/L2|      |    |  | L1/L2|  | L1/L2|      |
  |  +--+---+  +--+---+      |    |  +--+---+  +--+---+      |
  |     |         |           |    |     |         |           |
  |  +------+  +------+      |    |  +------+  +------+      |
  |  |Core2 |  |Core3 |      |    |  |Core10|  |Core11|      |
  |  | L1/L2|  | L1/L2|      |    |  | L1/L2|  | L1/L2|      |
  |  +--+---+  +--+---+      |    |  +--+---+  +--+---+      |
  |     +----+-----+          |    |     +----+-----+          |
  |          |                |    |          |                |
  |     +----+----+           |    |     +----+----+           |
  |     | L3 Cache|           |    |     | L3 Cache|           |
  |     | (shared)|           |    |     | (shared)|           |
  |     +----+----+           |    |     +----+----+           |
  |          |                |    |          |                |
  |  +-------+-------+       |    |  +-------+-------+       |
  |  | Local Memory  |       |    |  | Local Memory  |       |
  |  | DDR5 128GB    |       |    |  | DDR5 128GB    |       |
  |  | Access: ~80ns |       |    |  | Access: ~80ns |       |
  |  +-------+-------+       |    |  +-------+-------+       |
  +----------+----------------+    +----------+----------------+
             |      UPI / CXL link          |
             +------------------------------+
                Remote access: ~130-160ns
                (~1.5-2x the latency of local access)
```

### 7.2 NUMA-Aware Programming

```c
/* Linux NUMA-aware memory allocation */
#include <numa.h>
#include <numaif.h>

void numa_aware_allocation(void) {
    /* Check if NUMA is available */
    if (numa_available() < 0) {
        fprintf(stderr, "NUMA is not available\n");
        return;
    }

    /* Check number of nodes */
    int num_nodes = numa_max_node() + 1;
    printf("NUMA nodes: %d\n", num_nodes);

    /* Allocate memory on a specific NUMA node */
    size_t size = 1024 * 1024 * 1024;  /* 1GB */
    void *local_mem = numa_alloc_onnode(size, 0);  /* Allocate on node 0 */

    /* Bind this thread to node 0's CPUs */
    struct bitmask *cpumask = numa_allocate_cpumask();
    numa_node_to_cpus(0, cpumask);
    numa_sched_setaffinity(0, cpumask);

    /* Accesses to local_mem are at local speed (~80ns) */
    memset(local_mem, 0, size);

    numa_free(local_mem, size);
    numa_free_cpumask(cpumask);
}
```

---

## 8. Huge Pages and Practical Memory Management

### 8.1 The Need for Huge Pages

With standard 4KB pages, an enormous number of TLB entries are needed to cover large amounts of memory. Using Huge Pages (2MB or 1GB) allows a wider memory range to be covered with the same number of TLB entries.

```
TLB coverage comparison:

  Standard pages (4KB):
  1024 TLB entries x 4KB = 4MB coverage
  -> High TLB miss rate against a 64GB memory space

  Huge Pages (2MB):
  1024 TLB entries x 2MB = 2GB coverage
  -> Significantly reduced TLB miss rate even with 64GB

  Huge Pages (1GB):
  4 TLB entries x 1GB = 4GB coverage
  -> Optimal for database and HPC workloads
```

### 8.2 Configuring Huge Pages on Linux

```bash
# Check Transparent Huge Pages (THP) status
cat /sys/kernel/mm/transparent_hugepage/enabled
# [always] madvise never

# Reserve explicit Huge Pages (2MB x 1024 = 2GB)
echo 1024 > /proc/sys/vm/nr_hugepages

# Check Huge Pages usage
cat /proc/meminfo | grep -i huge
# HugePages_Total:    1024
# HugePages_Free:     1024
# HugePages_Rsvd:        0
# HugePages_Surp:        0
# Hugepagesize:       2048 kB
```

```c
/* Using Huge Pages in C */
#include <sys/mman.h>
#include <stdio.h>

int main(void) {
    size_t huge_page_size = 2 * 1024 * 1024;  /* 2MB */
    size_t alloc_size = 256 * huge_page_size;   /* 512MB */

    /* Request Huge Pages with MAP_HUGETLB */
    void *ptr = mmap(NULL, alloc_size,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                     -1, 0);

    if (ptr == MAP_FAILED) {
        perror("mmap with MAP_HUGETLB failed");
        /* Fallback: allocate with standard pages */
        ptr = mmap(NULL, alloc_size,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS,
                   -1, 0);
        if (ptr == MAP_FAILED) {
            perror("mmap fallback failed");
            return 1;
        }
        /* Request THP via madvise */
        madvise(ptr, alloc_size, MADV_HUGEPAGE);
    }

    printf("Allocated %zu MB with Huge Pages\n", alloc_size / (1024 * 1024));

    /* Use the memory */
    memset(ptr, 0, alloc_size);

    munmap(ptr, alloc_size);
    return 0;
}
```

---

## 9. Cache-Friendly Programming

### 9.1 Data Structure Layout: AoS vs SoA

The memory layout of data structures has a decisive impact on cache efficiency. It is important to choose a layout where "only the needed data ends up in the cache line."

```c
/*
 * AoS (Array of Structures) vs SoA (Structure of Arrays)
 * Example: Particle system in a game engine
 */

/* ===== AoS: Array of Structures ===== */
struct ParticleAoS {
    float x, y, z;        /* Position: 12 bytes (frequently used) */
    float vx, vy, vz;     /* Velocity: 12 bytes (frequently used) */
    float r, g, b, a;     /* Color:    16 bytes (rendering only) */
    float lifetime;        /* Lifetime: 4 bytes  (occasional) */
    int   texture_id;      /* Texture:  4 bytes  (rendering only) */
    char  name[16];        /* Name:     16 bytes (debug only) */
};  /* Total: 64 bytes = exactly 1 cache line */

struct ParticleAoS particles_aos[100000];

/* Position update: */
void update_positions_aos(int n, float dt) {
    for (int i = 0; i < n; i++) {
        particles_aos[i].x += particles_aos[i].vx * dt;
        particles_aos[i].y += particles_aos[i].vy * dt;
        particles_aos[i].z += particles_aos[i].vz * dt;
    }
    /* Problem: Each particle is 64 bytes. The position update only needs
     * x,y,z,vx,vy,vz = 24 bytes, but the unnecessary 40 bytes of
     * name, texture_id, etc. also occupy the cache line.
     * -> Cache utilization: 24/64 = 37.5%
     */
}

/* ===== SoA: Structure of Arrays ===== */
struct ParticleSystemSoA {
    float *x,  *y,  *z;       /* Position */
    float *vx, *vy, *vz;      /* Velocity */
    float *r,  *g,  *b, *a;   /* Color */
    float *lifetime;           /* Lifetime */
    int   *texture_id;         /* Texture */
    /* name managed separately */
};

struct ParticleSystemSoA psys;

/* Position update: */
void update_positions_soa(int n, float dt) {
    for (int i = 0; i < n; i++) {
        psys.x[i] += psys.vx[i] * dt;
        psys.y[i] += psys.vy[i] * dt;
        psys.z[i] += psys.vz[i] * dt;
    }
    /* Advantage: x[], vx[] are in contiguous memory. 16 floats fit in one
     * cache line. Unnecessary color, name, texture_id don't enter the cache.
     * -> Cache utilization: nearly 100%
     * -> Also easy to vectorize with SIMD (AVX2/AVX-512)
     */
}
```

### 9.2 Loop Blocking (Tiling)

In large-scale matrix operations, naive implementations produce strided accesses that don't fit in the cache. Blocking (tiling) divides data into cache-sized blocks for processing.

```c
/*
 * Cache-optimized matrix multiplication C = A x B
 * NxN matrices (float, row-major storage)
 */

/* ===== Naive implementation (frequent cache misses) ===== */
void matmul_naive(int N, float *A, float *B, float *C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
                /*     ^^^^^^^^^^   ^^^^^^^^^^
                 *     Row-wise:OK  Col-wise:BAD!
                 *
                 * A[i*N+k]: k increments by 1 -> adjacent float -> spatial locality
                 * B[k*N+j]: k increments by N -> stride N floats -> strided access
                 *   For N=1024, stride = 4096 bytes = 64 cache lines
                 *   -> Nearly every B access is a cache miss
                 */
            C[i*N + j] = sum;
        }
}

/* ===== Blocked (tiled) implementation ===== */
void matmul_blocked(int N, float *A, float *B, float *C) {
    /* Block size: 3 blocks must fit in L1 cache (32KB)
     * BLOCK^2 x 4bytes x 3 matrices <= 32KB
     * BLOCK ~ sqrt(32768 / 12) ~ 52 -> round to 64 */
    int BLOCK = 64;

    for (int ii = 0; ii < N; ii += BLOCK)
        for (int jj = 0; jj < N; jj += BLOCK)
            for (int kk = 0; kk < N; kk += BLOCK)
                /* Inner loop: multiply BLOCK x BLOCK sub-matrices
                 * Sub-blocks of A, B, C fit in L1 cache */
                for (int i = ii; i < ii+BLOCK && i < N; i++)
                    for (int j = jj; j < jj+BLOCK && j < N; j++) {
                        float sum = C[i*N + j];
                        for (int k = kk; k < kk+BLOCK && k < N; k++)
                            sum += A[i*N+k] * B[k*N+j];
                        C[i*N + j] = sum;
                    }
    /* Performance improvement for N=1024: 3-8x faster than naive
     * Performance improvement for N=4096: 5-15x faster than naive
     * Cache miss rate: naive ~25% -> blocked ~1-3%
     */
}
```

### 9.3 Prefetching

Hardware prefetchers detect sequential access patterns and preload data into the cache. However, for irregular access patterns, explicit software prefetching can be effective.

```c
/* Software prefetch example */
#include <immintrin.h>  /* _mm_prefetch */

/* Applying prefetch to linked list traversal */
struct Node {
    int data;
    struct Node *next;
};

long long sum_list_prefetch(struct Node *head) {
    long long sum = 0;
    struct Node *curr = head;
    while (curr != NULL) {
        /* Prefetch 2 nodes ahead (hide latency) */
        if (curr->next && curr->next->next) {
            _mm_prefetch((const char *)curr->next->next, _MM_HINT_T0);
        }
        sum += curr->data;
        curr = curr->next;
    }
    return sum;
    /* Without prefetch: ~100ns per node (DRAM latency)
     * With prefetch: significant improvement if prefetch completes in time
     * However, effectiveness depends on inter-node distance and access patterns */
}
```

### 9.4 Cache Efficiency Comparison of Data Structures

The choice of data structure directly impacts cache performance. Here is a comparison of cache characteristics for major data structures.

| Data Structure | Memory Layout | Spatial Locality | Cache Efficiency | Usage Guidelines |
|----------------|--------------|-----------------|-----------------|-----------------|
| Array / std::vector | Contiguous | Very high | Best | First choice when sequential access dominates |
| std::deque | Block-contiguous | High | Good | When insertion/deletion at both ends is needed |
| B-Tree / B+Tree | Contiguous within nodes | Medium-high | Good | Disk-based indexes, large sorted data |
| Hash table (open addressing) | Contiguous | Medium-high | Good | When fast key lookup is needed |
| Hash table (chaining) | Scattered | Low | Poor | Pointer chasing in chains causes frequent misses |
| Red-black tree / std::map | Scattered | Low | Poor | Heavy pointer chasing. Consider sorted array + binary search instead |
| Linked list / std::list | Scattered | Very low | Worst | Should almost never be used on modern hardware |

```c
/*
 * Guidelines for cache-efficient data structure selection:
 *
 * 1. Sequential access dominant -> array/vector as first choice
 *    std::list is a "rarely use" best practice on modern hardware
 *
 * 2. Search dominant -> sorted array + binary search or hash (open addressing)
 *    std::map (red-black tree) has poor cache efficiency due to pointer chasing
 *
 * 3. Keep node size <= cache line size (64B)
 *    Separate unnecessary fields into a separate struct (Hot/Cold splitting)
 *
 * 4. Use memory pools to place objects of the same type in proximity
 *    Prevents locality degradation from malloc fragmentation
 */
```

### 9.5 Memory Alignment and Padding

Struct field ordering and alignment affect cache efficiency and memory usage.

```c
/* Examples of memory waste from struct padding */

/* Bad layout: excessive padding */
struct BadLayout {
    char   a;       /* 1 byte + 7 bytes padding */
    double b;       /* 8 bytes */
    char   c;       /* 1 byte + 3 bytes padding */
    int    d;       /* 4 bytes */
    char   e;       /* 1 byte + 7 bytes padding */
    double f;       /* 8 bytes */
};
/* sizeof(BadLayout) = 40 bytes (actual data 23 bytes, padding 17 bytes) */

/* Good layout: order by size descending */
struct GoodLayout {
    double b;       /* 8 bytes */
    double f;       /* 8 bytes */
    int    d;       /* 4 bytes */
    char   a;       /* 1 byte */
    char   c;       /* 1 byte */
    char   e;       /* 1 byte + 1 byte padding */
};
/* sizeof(GoodLayout) = 24 bytes (actual data 23 bytes, padding 1 byte) */

/*
 * For an array of 100,000 structs:
 * BadLayout:  40 x 100,000 = 4,000,000 bytes (3.81MB)
 * GoodLayout: 24 x 100,000 = 2,400,000 bytes (2.29MB)
 * -> 40% memory savings + more elements per cache line
 *
 * How to check:
 * gcc/clang: -Wpadded option outputs padding warnings
 * pahole command: visualizes struct layout
 */
```

### 9.6 Loop Unrolling and Cache Interaction

Loop unrolling is an optimization technique that affects both the CPU pipeline and the cache.

```c
/* Loop unrolling for pipeline efficiency improvement */

/* Without unrolling */
float dot_product_basic(float *a, float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];  /* Dependency chain: sum update serialized every iteration */
    }
    return sum;
}

/* 4x unrolled: split dependency chain into 4 */
float dot_product_unrolled(float *a, float *b, int n) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    int i;
    for (i = 0; i + 3 < n; i += 4) {
        sum0 += a[i]   * b[i];    /* 4 independent accumulations */
        sum1 += a[i+1] * b[i+1];  /* -> CPU can execute all 4 in parallel */
        sum2 += a[i+2] * b[i+2];
        sum3 += a[i+3] * b[i+3];
    }
    float sum = sum0 + sum1 + sum2 + sum3;
    for (; i < n; i++) sum += a[i] * b[i]; /* Remainder loop */
    return sum;
    /*
     * Benefits:
     * 1. Loop overhead (branch, counter update) reduced to 1/4
     * 2. 4 independent dependency chains improve pipeline efficiency
     * 3. Prefetcher's stream detection becomes easier
     *
     * Note: Modern compilers (-O2/-O3) perform automatic unrolling
     *       Manual unrolling should be based on profiling results
     */
}
```

---

## 10. Memory Allocators and the Cache

### 10.1 Problems with Standard Allocators (malloc/free)

Standard `malloc`/`free` is a general-purpose memory allocator, but in long-running programs, memory fragmentation progresses and spatial locality degrades.

```
malloc behavior and fragmentation over time:

  Memory becomes fragmented over time:
  +----+--+----+--+----+--+----+--+----+
  |Used|  |Used|  |Used|  |Used|  |Used|
  +----+--+----+--+----+--+----+--+----+
  -> Related objects are scattered across memory
  -> Spatial locality degrades -> cache efficiency worsens

  Countermeasure: Use purpose-specific memory allocators
```

### 10.2 Arena Allocators (Pool Allocators)

Game engines and database engines use specialized memory allocators to improve cache efficiency.

```c
/*
 * Simple arena allocator implementation example
 * Places objects of the same type in contiguous memory to maximize spatial locality
 */
#include <stdlib.h>
#include <stdint.h>

typedef struct Arena {
    uint8_t *memory;     /* Allocated memory block */
    size_t   capacity;   /* Total size */
    size_t   offset;     /* Next allocation position */
} Arena;

Arena arena_create(size_t capacity) {
    Arena arena;
    arena.memory = (uint8_t *)aligned_alloc(64, capacity);
    arena.capacity = capacity;
    arena.offset = 0;
    return arena;
}

void *arena_alloc(Arena *arena, size_t size) {
    size_t aligned_size = (size + 7) & ~7;  /* 8-byte boundary alignment */
    if (arena->offset + aligned_size > arena->capacity) return NULL;
    void *ptr = arena->memory + arena->offset;
    arena->offset += aligned_size;
    return ptr;
}

void arena_reset(Arena *arena) {
    arena->offset = 0;  /* Free all objects at once O(1) */
}

void arena_destroy(Arena *arena) {
    free(arena->memory);
}

/*
 * Advantages:
 * 1. Placed in contiguous memory -> maximized spatial locality
 * 2. No free needed -> no fragmentation
 * 3. arena_reset for O(1) bulk deallocation
 *
 * Use cases: Per-frame memory in games, AST construction in parsers,
 *            Per-request memory management in web servers
 */
```

---

## 11. Memory Profiling in Practice

### 11.1 Measuring Cache Misses with Linux perf

```bash
# Get cache miss statistics
perf stat -e cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses \
         ./my_program

# Example output:
#  1,234,567,890  cache-references
#     12,345,678  cache-misses       #  1.00% of all cache refs
#  5,678,901,234  L1-dcache-loads
#    567,890,123  L1-dcache-load-misses  # 10.00% of all L1-dcache loads

# Identify code locations where cache misses occur
perf record -e cache-misses ./my_program
perf report
```

### 11.2 Simulation with Valgrind (Cachegrind)

```bash
# Simulate cache behavior with Cachegrind
valgrind --tool=cachegrind ./my_program

# Example output:
# ==12345== I   refs:      1,234,567,890
# ==12345== I1  misses:          123,456
# ==12345== LLi misses:           12,345
# ==12345== I1  miss rate:          0.01%
# ==12345== LLi miss rate:         0.00%
# ==12345==
# ==12345== D   refs:        567,890,123  (345,678,901 rd + 222,211,222 wr)
# ==12345== D1  misses:       56,789,012  ( 34,567,890 rd +  22,221,122 wr)
# ==12345== LLd misses:        5,678,901  (  3,456,789 rd +   2,222,112 wr)
# ==12345== D1  miss rate:           10.0% (       10.0%   +        10.0%)
# ==12345== LLd miss rate:            1.0% (        1.0%   +         1.0%)

# Per-source-line cache miss information
cg_annotate cachegrind.out.12345
```

---

## 12. Anti-Patterns

### 12.1 Anti-Pattern 1: False Sharing

False Sharing is a phenomenon in multithreaded programs where logically independent variables are placed on the same cache line, causing unnecessary invalidation via the MESI protocol and severely degrading performance.

```c
/* ===== False Sharing Anti-Pattern ===== */
#include <pthread.h>
#include <stdio.h>
#include <time.h>

#define NUM_THREADS 4
#define ITERATIONS 100000000

/* Bad example: counters on the same cache line */
struct BadCounters {
    long count[NUM_THREADS];  /* 4 longs contiguous = 32 bytes < 64 bytes
                               * -> All on the same cache line */
};

/* Good example: padding separates cache lines */
struct GoodCounters {
    struct {
        long count;
        char padding[64 - sizeof(long)];  /* 64-byte alignment */
    } per_thread[NUM_THREADS];
};

struct BadCounters  bad_counters  = {0};
struct GoodCounters good_counters = {0};

void *bad_worker(void *arg) {
    int id = *(int *)arg;
    for (long i = 0; i < ITERATIONS; i++) {
        bad_counters.count[id]++;
        /* Thread 0 updates count[0]
         * -> count[1], [2], [3] are on the same cache line, so
         *   Thread 1,2,3's cache lines are invalidated
         * -> All threads reload from L3 or DRAM every time */
    }
    return NULL;
}

void *good_worker(void *arg) {
    int id = *(int *)arg;
    for (long i = 0; i < ITERATIONS; i++) {
        good_counters.per_thread[id].count++;
        /* Each thread's counter is on a separate cache line
         * -> No impact on other threads' cache lines
         * -> Each thread operates independently from L1 cache */
    }
    return NULL;
}

/* Typical performance difference:
 * Bad  (with False Sharing): ~8 seconds (4 threads)
 * Good (no False Sharing):   ~0.5 seconds (4 threads)
 * -> 16x performance difference!
 * The gap widens as thread count increases
 */
```

### 12.2 Anti-Pattern 2: Pointer Chasing

Access patterns that follow pointers to random memory locations, such as linked lists and tree structures, have low spatial locality and extremely poor cache efficiency.

```c
/* ===== Pointer Chasing Anti-Pattern ===== */

/* Bad example: linked list traversal */
struct LinkedNode {
    int value;
    struct LinkedNode *next;  /* Points to random heap locations */
};

long long sum_linked_list(struct LinkedNode *head) {
    long long sum = 0;
    struct LinkedNode *curr = head;
    while (curr) {
        sum += curr->value;  /* <- Nearly every access is a cache miss (~100ns) */
        curr = curr->next;   /*   next points to scattered memory locations */
    }
    return sum;
    /* For 1 million elements:
     * Worst case: 1M x 100ns = 100ms (all DRAM accesses)
     * Array: 1M x 4B = 4MB -> fits in L3, ~5ms
     */
}

/* Good example: array-based contiguous data structure */
long long sum_array(int *data, int n) {
    long long sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];  /* <- Cache miss only once every 16 accesses */
    }
    return sum;
    /* 1 million elements: ~5ms (even faster if it fits in cache) */
}

/* Compromise: Unrolled Linked List */
#define BLOCK_SIZE 256
struct UnrolledNode {
    int values[BLOCK_SIZE];    /* Contiguous access within block -> high locality */
    int count;
    struct UnrolledNode *next; /* Pointer chasing only between blocks */
};
/* Cache misses reduced to once per BLOCK_SIZE elements */
```

### 12.3 Anti-Pattern 3: Oversized Working Set

When the working set (the memory region actually accessed in a short time) greatly exceeds cache capacity, capacity misses become frequent and performance degrades dramatically.

```python
# ===== Impact of working set overflow =====

import time
import array

def benchmark_working_set(sizes_mb):
    """Compare performance with different working set sizes"""
    for size_mb in sizes_mb:
        n = size_mb * 1024 * 1024 // 4  # int is 4 bytes
        data = array.array('i', range(n))

        # Random access (worst case)
        import random
        indices = list(range(n))
        random.shuffle(indices)
        sample = indices[:min(1_000_000, n)]

        start = time.time()
        total = 0
        for idx in sample:
            total += data[idx]
        elapsed = time.time() - start

        ns_per_access = elapsed * 1e9 / len(sample)
        print(f"  {size_mb:6d} MB: {ns_per_access:8.1f} ns/access")

# Typical output:
#       1 MB:      3.5 ns/access  <- Fits in L2 cache
#       4 MB:      5.2 ns/access  <- Fits in L3 cache
#      32 MB:     12.0 ns/access  <- Near L3 cache capacity
#     128 MB:     85.0 ns/access  <- DRAM access dominant
#    1024 MB:    110.0 ns/access  <- Fully DRAM-dependent
#
# -> Performance drops sharply once L3 cache capacity is exceeded
#   This is the phenomenon known as the "Cache Cliff"
```

---

## 13. Practical Exercises

### Exercise 1 (Fundamentals): Developing Latency Intuition

Using Jeff Dean's numbers, estimate the approximate time for the following operations.

**Problems:**

1. Linear search through a 1000-element int array (4KB) within L1 cache: total latency
2. Binary search on a 1-million-element sorted array within L3 cache: latency (comparisons x L3 latency)
3. Sequential read of a 100MB file from NVMe SSD: elapsed time
4. Sequential read of the same 100MB from HDD: elapsed time

**Approximate answers:**

1. 1000 elements fit in L1 -> 1000 x 1ns = 1us. With branch misprediction effects, actually ~2-3us
2. log2(1,000,000) ~ 20 comparisons. Assuming L3 access per comparison -> 20 x 12ns = 240ns. With TLB misses and branch mispredictions, actually ~500ns-1us
3. 100MB / 7GB/s (PCIe 4.0) ~ 14ms. Adding initial seek of 16us, approximately 14ms
4. 100MB / 200MB/s ~ 500ms. Adding initial seek of 8ms, approximately 508ms

### Exercise 2 (Application): Measuring Cache Efficiency

Implement the following in your preferred programming language and compare performance differences.

**Problems:**

1. Implement row-major sum vs. column-major sum on an NxN matrix and compare execution times for N=1000, 4000, 8000
2. Compare traversal speed of 1 million elements between a contiguous-memory array (ArrayList/Vector) and a linked list
3. Compare the speed of a "position-only update" process between AoS and SoA layouts

**Measurement tips:**
- Take at least 5 measurements and use the median
- For JIT languages (Java, C#, etc.), provide sufficient warmup
- If possible, check cache miss rates with `perf stat`

### Exercise 3 (Advanced): System-Wide Memory Analysis

Investigate the following about an application you are developing (or an open-source application).

**Problems:**

1. Estimate the application's working set size (using `top`, `ps`, `/proc/[pid]/smaps`, etc.)
2. Determine whether the working set fits in L3 cache, and discuss the impact if it doesn't
3. Measure L1/L2/L3 cache miss rates and TLB miss rates with `perf stat`
4. Identify scenarios where page faults may occur and discuss the potential impact of applying Huge Pages
5. If there are multithreaded sections, investigate the possibility of False Sharing

---

## 14. FAQ

### Q1: How does GC (garbage collection) affect the cache?

**A**: GC has a significant impact on cache efficiency. The main effects are as follows.

- **Mark & Sweep GC**: Scans all reachable objects, traversing the entire heap. This process replaces nearly all cache contents (cache pollution). The larger the working set, the more severe the impact
- **Generational GC**: Frequently collects only the young generation, rarely collecting the old generation. The young generation is typically small (a few MB to tens of MB) and often fits in L3 cache, so cache impact is limited
- **Compaction GC**: Moves objects in memory to eliminate fragmentation. Locality of references may improve immediately after compaction, but the cache is polluted during the move
- **Concurrent GC (ZGC, Shenandoah, etc.)**: GC threads run concurrently with application threads, so GC thread memory accesses can pollute the application's cache. However, Stop-the-World time is short, so latency jitter is small

### Q2: Does virtual memory always hurt performance?

**A**: Under normal conditions, virtual memory overhead is hidden by the TLB and is essentially negligible. Problematic cases include:

- **Frequent TLB misses**: When the working set exceeds TLB coverage (4KB x 1024 entries = 4MB), TLB misses increase. Using Huge Pages (2MB/1GB) is an effective countermeasure
- **Page faults (thrashing)**: A state where frequent page faults effectively halt the system. Adding physical memory is the most reliable solution
- **Multi-level page table walks**: x86-64's 4-level page table requires up to 4 memory accesses on TLB miss. This is accelerated by the hardware PTW (Page Table Walker)
- **The mmap trap**: mmap-ing a large file causes minor page faults on first access. These can be avoided with the MAP_POPULATE flag or madvise(MADV_WILLNEED)

### Q3: Why is Apple Silicon's Unified Memory so fast?

**A**: In traditional PC architecture, CPU-dedicated DDR DRAM and GPU-dedicated GDDR/HBM are physically separate, requiring data transfer between CPU and GPU via the PCIe bus (~32GB/s).

Apple Silicon (M1/M2/M3/M4 series) unified memory architecture:

- CPU, GPU, and Neural Engine (NPU) all directly reference the same LPDDR5 memory (no copying required)
- All components share the full LPDDR5 bandwidth (~200-400GB/s)
- CPU computation results can be used by the GPU with zero copy, making CPU-to-GPU data transfer latency effectively zero
- For AI inference workloads (LLMs, etc.), the entire model can be kept in memory and processed cooperatively across CPU/GPU/NPU

### Q4: What is the difference between DDR5 and DDR4?

**A**: DDR5 is the successor to DDR4, with significantly improved bandwidth.

| Feature | DDR4-3200 | DDR5-5600 |
|---------|-----------|-----------|
| Clock speed | 1600MHz | 2800MHz |
| Data rate | 3200MT/s | 5600MT/s |
| Bandwidth (1 channel) | 25.6GB/s | 44.8GB/s |
| Channel config | 1 channel/DIMM | 2 channels/DIMM |
| Voltage | 1.2V | 1.1V |
| Burst length | 8 | 16 |
| Latency (CAS) | CL22 (~13.75ns) | CL36 (~12.86ns) |

Latency (CAS latency) is nearly the same, but bandwidth has improved by approximately 1.75x. This embodies the essence of the Memory Wall problem: "DRAM latency hardly improves."

### Q5: What is cache warmup?

**A**: Immediately after application startup or a context switch, cache contents are invalid (cold state), so all initial memory accesses result in cache misses. Cache warmup refers to the process during which working set data is loaded into the cache and the hit rate reaches a steady state.

In benchmark measurement, it is important to include a warmup phase to stabilize the cache state before starting measurement. Measurements without warmup include the effects of cold misses and do not accurately reflect steady-state performance.

---

## 15. Advanced Topics

### 15.1 Latest Trends in Memory Technology

| Technology | Overview | Use Cases |
|------------|----------|-----------|
| **HBM (High Bandwidth Memory)** | DRAM dies stacked with TSV (Through-Silicon Via). ~1TB/s bandwidth | GPUs (H100/A100), HPC |
| **CXL (Compute Express Link)** | PCIe-based memory protocol. Enables memory pooling | Data centers, memory expansion |
| **Persistent Memory (Intel Optane)** | Non-volatile memory with near-DRAM speed. Byte-addressable | Databases, logging |
| **MRAM (Magnetoresistive RAM)** | Non-volatile memory using magnetoresistance. Near-SRAM speed | Embedded caches |
| **Processing-in-Memory (PIM)** | Computation executed within memory. Minimizes data movement | AI inference, graph processing |

### 15.2 The Future of Cache Hierarchies

Modern processors use a standard 3-level L1/L2/L3 hierarchy, but the following changes are underway.

- **L4 cache emergence**: Intel Meteor Lake includes an eDRAM-based L4 cache (128MB) shared between GPU and CPU
- **3D V-Cache**: AMD's 3D V-Cache technology stacks L3 cache on top of the die, achieving up to 128MB of L3 cache. Significant performance improvement for cache-sensitive workloads like gaming
- **Adaptive cache policies**: Research into machine learning-based dynamic cache replacement policies is progressing

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge from this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## 16. Summary

| Concept | Key Points |
|---------|------------|
| Memory hierarchy | Register -> L1 -> L2 -> L3 -> DRAM -> SSD -> HDD (speed vs. capacity tradeoff) |
| SRAM vs DRAM | SRAM is 6T, fast (for caches); DRAM is 1T1C, large capacity (for main memory) |
| Cache line | 64-byte unit data transfer. The fundamental unit for exploiting spatial locality |
| Mapping scheme | Set-associative is the modern standard (compromise between direct-mapped and fully associative) |
| Write policy | Write-back is mainstream. MESI protocol used for cache coherency |
| Principle of locality | Temporal (recently used data) + spatial (nearby data) are the keys to cache efficiency |
| 3C classification | Compulsory / Capacity / Conflict -- three types of cache misses |
| Virtual memory | Process isolation + efficient physical memory management. Implemented with page tables + TLB |
| Huge Pages | 2MB/1GB pages expand TLB coverage. Effective for large-scale applications |
| NUMA | In multi-socket servers, ignoring memory placement can cause 50%+ performance degradation |
| AoS vs SoA | Choosing a layout based on data access patterns determines cache efficiency |
| Blocking | An optimization technique that processes matrix operations in cache-sized blocks |
| False Sharing | A hidden cause of performance degradation in multithreading. Avoided with cache line padding |

---

## 17. Recommended Next Guides


---

## 18. References

1. Bryant, R. E. & O'Hallaron, D. R. *Computer Systems: A Programmer's Perspective.* 3rd Edition, Pearson, 2015.
   - Comprehensive coverage of memory hierarchy and virtual memory. The standard undergraduate textbook
2. Hennessy, J. L. & Patterson, D. A. *Computer Architecture: A Quantitative Approach.* 6th Edition, Morgan Kaufmann, 2017.
   - Quantitative analysis of cache design and memory technology. The standard graduate-level textbook
3. Drepper, U. "What Every Programmer Should Know About Memory." 2007. https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
   - A practical guide to DRAM, caches, NUMA, and profiling. Essential reading
4. Dean, J. & Barroso, L. A. "The Tail at Scale." *Communications of the ACM*, 56(2):74-80, 2013.
   - Examines the impact of latency in large-scale distributed systems. The source of "Numbers Everyone Should Know"
5. Intel Corporation. *Intel 64 and IA-32 Architectures Optimization Reference Manual.* 2024.
   - The official reference for cache and memory optimization on x86 processors
6. Levinthal, D. *Performance Analysis Guide for Intel Core i7 Processor and Intel Xeon 5500 Processors.* Intel, 2009.
   - A practical guide to cache performance analysis using perf counters
7. Fog, A. "Optimizing software in C++: An optimization guide for Windows, Linux and Mac platforms." https://www.agner.org/optimize/
   - A practical collection of data structure layout and cache optimization techniques

---

## Recommended Next Guides

- [Storage Systems](./02-storage-systems.md) - Proceed to the next topic

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://ja.wikipedia.org/) - Technical concept overviews
