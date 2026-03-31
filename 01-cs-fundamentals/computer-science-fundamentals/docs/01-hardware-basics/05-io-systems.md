# I/O Systems

> I/O (Input/Output) is the window connecting computers to the outside world, and in many applications, the bottleneck lies not in the CPU but in I/O. A systematic understanding of buses, interrupts, DMA, device drivers, and I/O scheduling is the first step toward high-performance system design.

## Learning Objectives

- [ ] Explain the basic concepts of I/O (polling, interrupts, DMA)
- [ ] Understand the hierarchical structure and bandwidth calculation of bus architectures
- [ ] Explain the complete flow of interrupt handling (hardware interrupts, software interrupts, MSI/MSI-X)
- [ ] Understand the operating principles of DMA, bounce buffers, and scatter-gather DMA
- [ ] Explain device driver design patterns and the Linux kernel module mechanism
- [ ] Compare I/O scheduling algorithms (CFQ, Deadline, mq-deadline, BFQ, none)
- [ ] Make informed decisions about the evolution and use cases of asynchronous I/O (select, poll, epoll, kqueue, io_uring)
- [ ] Understand the differences between memory-mapped I/O and port-mapped I/O
- [ ] Master I/O performance bottleneck analysis and optimization techniques

## Prerequisites


---

## 1. Overview of I/O Systems

### 1.1 The Role of I/O

In computer systems, I/O (Input/Output) is the collective term for mechanisms that exchange data between the CPU/memory and external devices. All keyboard and mouse input, display output, storage read/write operations, and network communications fall under I/O.

In modern systems, the fundamental challenge is that I/O device speeds are overwhelmingly slower than CPU computation speeds. To efficiently absorb this speed gap, multi-layered mechanisms such as buses, interrupts, DMA, and scheduling have evolved.

```
Overall structure of I/O systems:

  +------------------------------------------------------------+
  |  Application Layer                                          |
  |  +------+ +------+ +------+ +------+                       |
  |  | Web  | |  DB  | | File | | Game |                       |
  |  |Server| |Engine| | Ops  | |Engine|                       |
  |  +--+---+ +--+---+ +--+---+ +--+---+                      |
  |     +--------+--------+--------+                            |
  |                   | System calls (read/write/ioctl)         |
  +-------------------+----------------------------------------+
  |  Kernel Layer     |                                         |
  |     +-------------v--------------+                          |
  |     | VFS (Virtual File System)  |                          |
  |     +-------------+--------------+                          |
  |     +-------------v--------------+                          |
  |     | I/O Scheduler              | <- Request reordering    |
  |     +-------------+--------------+                          |
  |     +-------------v--------------+                          |
  |     | Device Driver              | <- HW-specific ops       |
  |     +-------------+--------------+                          |
  |     +-------------v--------------+                          |
  |     | Interrupt Handler / DMA    | <- Data transfer control |
  |     +-------------+--------------+                          |
  +-------------------+----------------------------------------+
  |  Hardware Layer    |                                         |
  |     +-------------v--------------+                          |
  |     | Bus (PCIe / USB / SATA)    | <- Physical data path    |
  |     +-------------+--------------+                          |
  |     +-------------v--------------+                          |
  |     | Device Controller          |                          |
  |     +-------------+--------------+                          |
  |     +-------------v--------------+                          |
  |     | I/O Device (SSD/NIC/GPU)   |                          |
  |     +----------------------------+                          |
  +------------------------------------------------------------+
```

### 1.2 Device Speed Hierarchy

I/O device speeds differ by orders of magnitude depending on the device type. This speed gap must always be kept in mind during system design.

| Device | Bandwidth | Latency | IOPS (approx.) |
|--------|-----------|---------|-----------------|
| CPU Registers | Several TB/s | < 1ns | - |
| L1 Cache | ~1TB/s | ~1ns | - |
| DDR5 Memory | ~50GB/s | ~100ns | - |
| NVMe SSD (PCIe 5.0) | ~14GB/s | ~10us | ~2,000,000 |
| SATA SSD | ~550MB/s | ~50us | ~100,000 |
| HDD (7200RPM) | ~200MB/s | ~5ms | ~150 |
| 10GbE NIC | ~1.25GB/s | ~10us | - |
| USB 3.2 Gen 2 | ~1.25GB/s | ~1ms | - |
| Keyboard | Tens of B/s | ~50ms | - |

As this table clearly shows, there is a speed difference of over 10 orders of magnitude between CPU registers and a keyboard. The role of the I/O subsystem is to conceal this disparity and make the overall system operate efficiently.

### 1.3 I/O Addressing Modes

There are two fundamental addressing modes for CPU communication with I/O devices.

```
(A) Port-Mapped I/O (PMIO)

  CPU                    Memory Space       I/O Space
  +------+              +------+           +------+
  |      |--mem instr-> | 0x00 |           |      |
  |      |  (MOV etc.)  | 0x01 |           |      |
  |      |              | ...  |           |      |
  |      |              | 0xFF |           |      |
  |      |              +------+           |      |
  |      |                                 |      |
  |      |--I/O instr->                    | 0x00 |
  |      |  (IN/OUT)                       | 0x01 |
  |      |                                 | ...  |
  +------+                                 +------+

  Characteristics:
  - Memory space and I/O space are completely separate
  - Uses x86-specific IN/OUT instructions
  - I/O address space is 0x0000-0xFFFF (64KB)
  - Used by legacy devices (serial port: 0x3F8, keyboard: 0x60)

(B) Memory-Mapped I/O (MMIO)

  CPU                    Unified Address Space
  +------+              +--------------+
  |      |              | 0x00000000   | <- Memory region
  |      |--mem instr-> | ...          |
  |      |  (MOV etc.)  | 0x7FFFFFFF   |
  |      |              +--------------+
  |      |              | 0x80000000   | <- Device register region
  |      |              | (GPU VRAM)   |
  |      |              | (NIC regs)   |
  |      |              | 0xFFFFFFFF   |
  |      |              +--------------+
  +------+

  Characteristics:
  - Memory and I/O share the same address space
  - Access devices with standard memory instructions (MOV, LDR/STR, etc.)
  - Adopted by most modern devices (via PCIe BAR)
  - Must set uncacheable attributes to avoid CPU cache effects
```

**PMIO vs MMIO Comparison:**

| Item | PMIO | MMIO |
|------|------|------|
| Address Space | Dedicated I/O space (64KB) | Part of memory space |
| Access Instructions | IN/OUT (x86 only) | General-purpose instructions like MOV |
| Address Width | Fixed 16-bit | Architecture-dependent (up to 64-bit) |
| Caching | Automatically uncacheable | Explicit uncacheable setting required |
| Architecture | x86 only | All architectures |
| Primary Use | Legacy devices | PCIe devices, modern I/O |
| Performance | Slower (dedicated instruction overhead) | Faster (optimizable with general instructions) |

---

## 2. Bus Architecture

### 2.1 Basic Bus Concepts

A bus is a shared communication path for transferring data within a computer. The term "bus" derives from the Latin "omnibus" (for everyone), signifying a communication path shared by multiple components.

A bus consists of three types of signal lines.

```
Three signal lines of a bus:

  +------------------------------------------------------+
  |                                                      |
  |  Data Bus                                            |
  |  ============================================        |
  |  Signal line group that transfers data itself        |
  |  Width: 8/16/32/64 bits (bits transferable at once)  |
  |                                                      |
  |  Address Bus                                         |
  |  ============================================        |
  |  Signal line group specifying source/destination     |
  |  Width: 32 bits -> 4GB, 64 bits -> 16EB address space|
  |                                                      |
  |  Control Bus                                         |
  |  ============================================        |
  |  Control signals for read/write direction,           |
  |  interrupt requests, clock, etc.                     |
  |  Signal examples: R/W, IRQ, CLK, RESET, READY       |
  |                                                      |
  +------------------------------------------------------+

  Bandwidth calculation:
    Bandwidth = Bus width (bits) x Clock frequency x Transfer rate factor

    Example: PCIe 5.0 x16
    Bandwidth = 16 lanes x 32GT/s x 128b/130b encoding
              ~ 63 GB/s (unidirectional)
              ~ 126 GB/s (bidirectional total)
```

### 2.2 Bus Hierarchy

In modern PCs, not all devices share a single bus; instead, a hierarchical structure exists based on speed tiers.

```
Modern PC bus hierarchy (typical configuration from 2024 onward):

  +--------------------------------------------------------------+
  |                        CPU                                    |
  |  +------+  +------+  +------------------------------+       |
  |  | Cores|  | Cores|  | Integrated Memory Controller |       |
  |  | 0-7  |  | 8-15 |  | DDR5: ~89.6 GB/s            |       |
  |  +--+---+  +--+---+  +----------+-------------------+       |
  |     +----+-----+                 |                            |
  |          | Internal Interconnect |                            |
  |  +-------v-----------------------v-----------------------+   |
  |  |            PCIe Root Complex                          |   |
  |  +---+--------------+--------------+---------------------+   |
  +------+              |              |-------------------------+
         |              |              |
    PCIe 5.0 x16    PCIe 5.0 x4    PCIe 4.0 x4
    (63 GB/s)       (16 GB/s)      (8 GB/s)
         |              |              |
    +----v----+    +----v----+    +----v----+
    |  GPU    |    |  NVMe   |    |  NVMe   |
    |(RTX5090)|    |  SSD    |    |  SSD    |
    +---------+    +---------+    +---------+

  +-------------------------------------------------------------+
  |                    Chipset (PCH)                              |
  |  +-----------------------------------------------------+    |
  |  |  PCIe 4.0/3.0 Switch                                |    |
  |  +--+--------+--------+--------+--------+-----------+  |    |
  |     |        |        |        |        |              |    |
  |  SATA III  USB 3.2  2.5GbE  Audio    Additional PCIe  |    |
  |  (600MB/s) (20Gbps) (312MB/s)        slots             |    |
  |     |        |        |        |        |              |    |
  |  +--v--+ +--v--+ +--v--+ +--v----+ +--v--+            |    |
  |  |HDD  | |USB  | |NIC  | |Sound  | |Expan|            |    |
  |  |/SSD | |Dev  | |     | |Card   | |Card |            |    |
  |  +-----+ +-----+ +-----+ +-------+ +-----+            |    |
  +-------------------------------------------------------------+
```

### 2.3 PCIe in Detail

PCIe (Peripheral Component Interconnect Express) is the standard specification for modern I/O buses, replacing the traditional parallel bus (PCI) with serial point-to-point connections.

**PCIe Bandwidth by Generation:**

| Generation | Year | Transfer Rate | x1 Bandwidth | x16 Bandwidth | Encoding |
|------------|------|--------------|-------------|---------------|----------|
| PCIe 1.0 | 2003 | 2.5 GT/s | 250 MB/s | 4 GB/s | 8b/10b |
| PCIe 2.0 | 2007 | 5 GT/s | 500 MB/s | 8 GB/s | 8b/10b |
| PCIe 3.0 | 2010 | 8 GT/s | 984 MB/s | 15.75 GB/s | 128b/130b |
| PCIe 4.0 | 2017 | 16 GT/s | 1.97 GB/s | 31.5 GB/s | 128b/130b |
| PCIe 5.0 | 2019 | 32 GT/s | 3.94 GB/s | 63 GB/s | 128b/130b |
| PCIe 6.0 | 2022 | 64 GT/s | 7.56 GB/s | 121 GB/s | PAM4+FEC |

An important concept in PCIe is the BAR (Base Address Register). BARs are a mechanism for PCIe devices to inform the system of the memory space they use. The OS reads these BARs to configure MMIO regions.

```c
/* Example of reading PCIe BAR (Linux kernel driver) */
#include <linux/pci.h>

static int my_pci_probe(struct pci_dev *pdev,
                        const struct pci_device_id *id)
{
    int ret;
    void __iomem *bar0;
    resource_size_t bar0_start, bar0_len;

    /* Enable the PCI device */
    ret = pci_enable_device(pdev);
    if (ret)
        return ret;

    /* Get BAR0 resource */
    bar0_start = pci_resource_start(pdev, 0);
    bar0_len   = pci_resource_len(pdev, 0);

    /* Map BAR to memory space (MMIO) */
    bar0 = ioremap(bar0_start, bar0_len);
    if (!bar0) {
        pci_disable_device(pdev);
        return -ENOMEM;
    }

    /* Read/write device registers */
    u32 status = ioread32(bar0 + DEVICE_STATUS_REG);
    iowrite32(0x1, bar0 + DEVICE_CONTROL_REG);

    /* Note: Use ioread/iowrite instead of normal pointer
       dereferences (for memory barriers and endian considerations) */

    return 0;
}
```

### 2.4 USB, SATA, and Other Bus Standards

Various bus standards exist for different purposes beyond PCIe.

| Bus Standard | Topology | Max Bandwidth | Primary Use |
|-------------|----------|--------------|-------------|
| USB 2.0 | Tree (hub) | 480 Mbps | Mouse, keyboard |
| USB 3.2 Gen 2x2 | Tree | 20 Gbps | External SSD |
| USB4 / Thunderbolt 4 | Tunneling | 40 Gbps | External GPU, dock |
| SATA III | Point-to-point | 6 Gbps | Internal SSD/HDD |
| NVMe (PCIe 5.0 x4) | PCIe | ~14 GB/s | High-speed internal SSD |
| CXL 3.0 | PCIe physical layer | PCIe 6.0 compliant | Memory pooling |
| InfiniBand HDR | Switched | 200 Gbps | HPC, data centers |

---

## 3. Three I/O Methods

### 3.1 Programmed I/O (Polling)

Programmed I/O is a method where the CPU actively and repeatedly checks device status. It is also called "Polling."

```
Polling operation:

  CPU                         Device Controller
  |                           |
  |--"Write command"--------->| (1) CPU issues command to device
  |                           |
  |--"Read status register"-->| (2) CPU checks busy flag
  |<-"BUSY"------------------| -> Not yet complete
  |                           |
  |--"Read status register"-->| (3) Check again (busy wait)
  |<-"BUSY"------------------| -> Still not complete
  |                           |
  |  ... (repeat) ...         | <- Wasting CPU cycles
  |                           |
  |--"Read status register"-->| (n) Detect completion
  |<-"DONE"------------------|
  |                           |
  |--"Read data register"---->| (n+1) Retrieve data
  |<-- Data ------------------|
  |                           |

  Busy wait loop pseudocode:
    while (read_status_register() & BUSY_FLAG) {
        /* CPU spins idly */
    }
    data = read_data_register();
```

**Advantages of Polling:**
- Extremely simple implementation; no interrupt controller required
- Minimum latency (no interrupt processing overhead)
- Predictable timing (suitable for real-time systems)

**Disadvantages of Polling:**
- Consumes massive CPU cycles (busy wait)
- CPU utilization approaches 100% when the device is slow
- Other processes cannot execute in multitasking environments

**When Polling is Appropriate:**
Polling is still actively used for specific purposes in modern systems. In high-performance mode for NVMe SSDs, the interrupt overhead (several microseconds) is non-negligible relative to I/O completion time (~10us), so polling mode (io_poll) is used to minimize latency. DPDK (Data Plane Development Kit) also adopts polling for network packet processing.

### 3.2 Interrupt-Driven I/O

An interrupt is a mechanism by which a device asynchronously notifies the CPU that "processing is complete." The CPU can execute other tasks while waiting for device completion.

```
Interrupt-driven I/O operation:

  CPU                     Interrupt Controller   Device
  |                       (APIC)                |
  |--"Issue command"----------------------------->| (1)
  |                                              |
  | [Execute other processes]                    | (2) CPU does other work
  | [Task A -> Task B -> ...]                    |
  |                                              |
  |                       |<-- IRQ signal --------| (3) Device signals completion
  |                       |                      |
  |<-- Interrupt notify --|                      | (4) APIC forwards to CPU
  |                       |                      |
  | [Save current state]  |                      | (5) Context save
  | [Lookup interrupt     |                      | (6) Get handler address
  |  vector]              |                      |     from IDT
  | [Execute ISR]         |                      |
  |   +- Read data        |                      | (7) Device ops in ISR
  |   +- Copy to buffer   |                      |
  |   +- Send EOI -------->|                      | (8) Interrupt completion
  |                       |                      |
  | [Restore state]       |                      | (9) Return to original task
  | [Resume original task]|                      |
  |                       |                      |
```

### 3.3 Types of Interrupts

Interrupts are classified into several types based on their source and purpose.

```
Interrupt classification:

  Interrupt
  +-- Hardware Interrupts (External Interrupts)
  |   +-- Maskable Interrupts (INTR)
  |   |   +-- Level-triggered: Interrupt active while signal level is High
  |   |   +-- Edge-triggered: Interrupt on rising edge of signal
  |   +-- Non-Maskable Interrupts (NMI)
  |       +-- Memory parity errors, hardware failures, etc.
  |
  +-- Software Interrupts (Internal Interrupts)
  |   +-- Exceptions
  |   |   +-- Fault: Recoverable (e.g., page fault)
  |   |   +-- Trap: Intentional (e.g., INT 0x80, syscall)
  |   |   +-- Abort: Unrecoverable (e.g., double fault)
  |   +-- System Calls (INT 0x80 / SYSCALL instruction)
  |
  +-- Message Signaled Interrupts (MSI/MSI-X)
      +-- MSI: Interrupt notification via memory write from PCI device
      |   -> Up to 32 interrupt vectors
      +-- MSI-X: Extended version of MSI
          -> Up to 2048 interrupt vectors
          -> Can assign individual interrupts to each NVMe/NIC queue
```

### 3.4 Detailed Interrupt Processing Flow (x86_64)

```
Interrupt processing flow on x86_64:

  (1) Device generates interrupt signal
         |
         v
  (2) Local APIC accepts interrupt
      - Check priority (TPR: Task Priority Register)
      - Hold pending if lower priority than currently executing interrupt
         |
         v
  (3) CPU accepts interrupt after completing current instruction
      - Automatically saves RFLAGS, CS, RIP to stack
      - Also switches RSP when privilege level changes
         |
         v
  (4) Reference IDT (Interrupt Descriptor Table)
      - Interrupt vector number -> IDT entry
      - Obtain ISR address from entry

      IDT structure:
      +----------+-----------------------------+
      | Vector # | Purpose                     |
      +----------+-----------------------------+
      | 0        | #DE: Divide-by-zero         |
      | 1        | #DB: Debug exception        |
      | 2        | NMI: Non-maskable interrupt  |
      | 6        | #UD: Invalid opcode         |
      | 13       | #GP: General protection     |
      | 14       | #PF: Page fault             |
      | 32-255   | External / User-defined     |
      +----------+-----------------------------+
         |
         v
  (5) Execute ISR (Interrupt Service Routine)
      - Top Half: Minimal processing (interrupts disabled)
        - Read device registers
        - Clear interrupt flags
        - Schedule Bottom Half
      - Bottom Half: Deferrable processing
        - Executed via softirq / tasklet / workqueue
        - Runs with interrupts enabled
         |
         v
  (6) Send EOI (End of Interrupt) to APIC
         |
         v
  (7) IRET instruction restores original context
      - Restores RIP, CS, RFLAGS from stack
```

### 3.5 Interrupt Handler Registration in the Linux Kernel

```c
/* Interrupt handler implementation example in Linux kernel */
#include <linux/interrupt.h>
#include <linux/module.h>

#define MY_IRQ 17  /* Interrupt number */

/* Top Half: Runs in interrupt context (must complete quickly) */
static irqreturn_t my_isr_top(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;
    u32 status;

    /* Check device interrupt status */
    status = ioread32(dev->regs + IRQ_STATUS_REG);
    if (!(status & MY_DEVICE_IRQ_MASK))
        return IRQ_NONE;  /* This interrupt is not from our device */

    /* Clear interrupt flag (allow device to re-interrupt) */
    iowrite32(status, dev->regs + IRQ_ACK_REG);

    /* Save received data to device-local buffer */
    dev->pending_data = ioread32(dev->regs + DATA_REG);
    dev->irq_count++;

    /* Schedule Bottom Half */
    tasklet_schedule(&dev->my_tasklet);

    return IRQ_HANDLED;
}

/* Bottom Half: Runs in near-process context */
static void my_tasklet_handler(unsigned long data)
{
    struct my_device *dev = (struct my_device *)data;

    /* Perform time-consuming processing here */
    process_received_data(dev->pending_data);
    wake_up_interruptible(&dev->wait_queue);
}

/* Register interrupt during driver initialization */
static int my_driver_init(struct my_device *dev)
{
    int ret;

    tasklet_init(&dev->my_tasklet, my_tasklet_handler,
                 (unsigned long)dev);

    /* IRQF_SHARED: Can share IRQ line with other devices
       4th argument: Pointer for device identification */
    ret = request_irq(MY_IRQ, my_isr_top,
                      IRQF_SHARED, "my_device", dev);
    if (ret) {
        pr_err("Failed to request IRQ %d\n", MY_IRQ);
        return ret;
    }

    return 0;
}

/* Release interrupt during driver cleanup */
static void my_driver_exit(struct my_device *dev)
{
    free_irq(MY_IRQ, dev);
    tasklet_kill(&dev->my_tasklet);
}
```

### 3.6 DMA (Direct Memory Access)

DMA (Direct Memory Access) is a mechanism that transfers data directly between memory and I/O devices without CPU involvement. It dramatically reduces CPU load for large data transfers.

```
DMA transfer flow:

  CPU              DMA Controller (DMAC)        Device       Memory
  |                |                            |             |
  | (1) DMA setup  |                            |             |
  | -src address-->|                            |             |
  | -dst address-->|                            |             |
  | -byte count -->|                            |             |
  | -start cmd  -->|                            |             |
  |                |                            |             |
  | (2) CPU works  | (3) DMA uses bus            |             |
  |   on other     | --"Data request"----------->|             |
  |   tasks        | <--- Data block ------------|             |
  | [Task A]       | --"Memory write"------------------------->|
  | [Task B]       |                            |             |
  | [Task C]       | --"Data request"----------->|             |
  |                | <--- Data block ------------|             |
  |                | --"Memory write"------------------------->|
  |                |                            |             |
  |                | (4) Transfer complete       |             |
  | <-- Completion |                            |             |
  |    interrupt   |                            |             |
  |                |                            |             |
  | (5) Post-      |                            |             |
  |  processing    |                            |             |
  |  in ISR        |                            |             |

  DMA transfer modes:
  +------------------------------------------------------+
  | (A) Block Transfer Mode                               |
  |   Monopolizes the bus for bulk transfer. For large    |
  |   data. CPU cannot access bus during transfer.        |
  |                                                      |
  | (B) Cycle Stealing Mode                               |
  |   "Steals" bus cycles not used by the CPU.           |
  |   Time-shares bus with CPU. For small data.          |
  |                                                      |
  | (C) Burst Mode                                        |
  |   Transfers consecutive addresses at high speed.     |
  |   Coordinates with DDR burst transfers.              |
  |   The dominant DMA mode in modern systems.           |
  +------------------------------------------------------+
```

### 3.7 Scatter-Gather DMA

Scatter-Gather DMA (SG-DMA) is a technology that performs scatter writes or gather reads across physically non-contiguous memory regions in a single DMA operation.

```
Standard DMA vs Scatter-Gather DMA:

(A) Standard DMA: Requires contiguous physical memory

  Physical Memory:
  +----+----+----+----+----+----+----+----+
  |Used|Used|Free|Free|Free|Free|Used|Used|
  +----+----+----+----+----+----+----+----+
                ^                ^
                +-- 4 contiguous --+
                    pages
                Usable as DMA buffer

  Problem: Difficult to allocate contiguous regions when memory is fragmented

(B) Scatter-Gather DMA: Non-contiguous is OK

  Physical Memory:
  +----+----+----+----+----+----+----+----+
  |Used| SG |Used| SG |Used|Used| SG | SG |
  +----+--+-+----+--+-+----+----+--+-+--+-+
          |         |              |    |
          v         v              v    v
  SG List (Scatter-Gather List):
  +-------------------------------------+
  | Entry 0: addr=0x1000, len=4096     | -> Page 1
  | Entry 1: addr=0x3000, len=4096     | -> Page 3
  | Entry 2: addr=0x6000, len=4096     | -> Page 6
  | Entry 3: addr=0x7000, len=4096     | -> Page 7
  +-------------------------------------+

  The DMA controller processes the SG list sequentially,
  scattering data transfers to non-contiguous physical pages
```

### 3.8 DMA and Cache Coherency

Since DMA accesses memory directly, bypassing the CPU cache, cache coherency issues arise.

```
Cache coherency problem:

  (Problem 1) CPU reads stale cache after DMA write

    CPU Cache: [Data A (stale)]    <- CPU reads this
                                   ^ Cache hit
    Memory:    [Data B (DMA updated)] <- DMA wrote new data

  (Problem 2) CPU write remains in cache, DMA reads stale memory

    CPU Cache: [Data C (latest)]   <- Not yet written back to memory
    Memory:    [Data D (stale)]    <- DMA reads this

  Solutions:
  +------------------------------------------------------------+
  | (A) Cache Invalidate                                        |
  |   Invalidate cache lines before DMA read                    |
  |   -> CPU reloads from memory on next access                 |
  |                                                            |
  | (B) Cache Flush/Clean                                       |
  |   Write back cache contents to memory before DMA write      |
  |   -> DMA can read the latest data                           |
  |                                                            |
  | (C) Uncacheable Memory                                      |
  |   Allocate DMA buffer with uncacheable attribute             |
  |   -> No coherency issues, but performance degrades          |
  |                                                            |
  | (D) Hardware Cache-Coherent DMA                             |
  |   PCIe device references CPU cache (snooping)               |
  |   -> ARM's Cache Coherent Interconnect (CCI)                |
  |   -> On x86, PCIe is basically cache-coherent               |
  +------------------------------------------------------------+
```

### 3.9 Comparison of the Three Methods

| Item | Polling | Interrupt | DMA |
|------|---------|-----------|-----|
| CPU Load | Extremely high (busy wait) | Medium (only during ISR) | Low (only setup and completion) |
| Latency | Minimum (instant detection) | Medium (1-10us) | Medium-high (setup overhead) |
| Throughput | Low | Medium | High |
| Implementation Complexity | Low | Medium | High |
| Use Cases | Ultra-low latency, DPDK, NVMe io_poll | General I/O, keyboard | Bulk transfers, disk, NIC |
| Hardware Requirements | Minimal | Interrupt controller | DMA controller |

---

## 4. Device Drivers

### 4.1 The Role of Device Drivers

A device driver is a software module that mediates between the OS kernel and hardware devices. It abstracts away hardware details and provides a unified interface to the kernel. The fact that approximately 70% of the Linux kernel source code consists of device drivers speaks to the importance and diversity of drivers.

```
Device driver positioning:

  +----------------------------------------------+
  |  User Space                                   |
  |  +--------------------------------------+    |
  |  | Application                          |    |
  |  | open(), read(), write(), ioctl(),    |    |
  |  | mmap(), close()                      |    |
  |  +----------------+---------------------+    |
  |                   | System calls               |
  +-------------------+---------------------------+
  |  Kernel Space                                  |
  |                   v                            |
  |  +--------------------------------------+    |
  |  | VFS (Virtual File System)            |    |
  |  | Unified interface layer              |    |
  |  | -> "Everything is a file" impl       |    |
  |  +---------+-----------+----------------+    |
  |            |           |                     |
  |    +-------v--+  +----v----------+           |
  |    | Block    |  | Character     |           |
  |    | Device   |  | Device Layer  |           |
  |    +-------+--+  +----+----------+           |
  |            |           |                     |
  |    +-------v--+  +----v----------+           |
  |    |I/O Sched |  | TTY/Input    |           |
  |    |  uler    |  | Subsystem    |           |
  |    +-------+--+  +----+----------+           |
  |            |           |                     |
  |    +-------v-----------v----------+          |
  |    | Device Drivers               |          |
  |    | +--------+ +--------+ +-----+|          |
  |    | |SATA Drv| |NVMe Drv| |USB Drv|         |
  |    | +--------+ +--------+ +-----+|          |
  |    +---------------+---------------+          |
  |                    |                          |
  +--------------------+--------------------------+
                       v
  +----------------------------------------------+
  |  Hardware (SSD, NIC, GPU, etc.)               |
  +----------------------------------------------+
```

### 4.2 UNIX Device Classification

UNIX classifies devices into three categories.

| Category | Characteristics | Access Unit | Examples |
|----------|----------------|-------------|---------|
| Character Device | Sequential access, no buffering | Byte-level | Terminal, serial port, mouse |
| Block Device | Random access, kernel buffering | Block-level (512B/4KB) | HDD, SSD, USB memory |
| Network Device | Packet-based send/receive, socket API | Packet-level | Ethernet NIC, Wi-Fi |

```
Example /dev directory structure:

  $ ls -la /dev/sd* /dev/tty* /dev/null /dev/zero 2>/dev/null | head -20

  brw-rw---- 1 root disk    8,  0  /dev/sda      <- Block device (b)
  brw-rw---- 1 root disk    8,  1  /dev/sda1     <- Partition
  crw-rw-rw- 1 root root    1,  3  /dev/null     <- Character device (c)
  crw-rw-rw- 1 root root    1,  5  /dev/zero     <- Character device
  crw--w---- 1 root tty     4,  0  /dev/tty0     <- Terminal device

  Device numbers (Major, Minor):
  - Major number: Identifies the driver (e.g., 8 = SCSI disk)
  - Minor number: Identifies the specific device (e.g., 0 = first disk)
```

### 4.3 Linux Kernel Module Implementation Example

In Linux, device drivers can be dynamically loaded and unloaded as kernel modules (Loadable Kernel Module: LKM).

```c
/* Simple character device driver example */
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "mychardev"
#define BUF_SIZE 1024

static dev_t dev_num;
static struct cdev my_cdev;
static struct class *my_class;
static char kernel_buf[BUF_SIZE];
static int buf_len = 0;

/* open: Called when the device file is opened */
static int my_open(struct inode *inode, struct file *filp)
{
    pr_info("mychardev: device opened\n");
    return 0;
}

/* read: Called when user reads data from the device */
static ssize_t my_read(struct file *filp, char __user *buf,
                       size_t count, loff_t *offset)
{
    int bytes_to_read;

    if (*offset >= buf_len)
        return 0;  /* EOF */

    bytes_to_read = min((int)count, buf_len - (int)*offset);

    /* Copy from kernel space -> user space
       Must not copy directly via pointer (security) */
    if (copy_to_user(buf, kernel_buf + *offset, bytes_to_read))
        return -EFAULT;

    *offset += bytes_to_read;
    return bytes_to_read;
}

/* write: Called when user writes data to the device */
static ssize_t my_write(struct file *filp, const char __user *buf,
                        size_t count, loff_t *offset)
{
    int bytes_to_write = min((int)count, BUF_SIZE - 1);

    /* Copy from user space -> kernel space */
    if (copy_from_user(kernel_buf, buf, bytes_to_write))
        return -EFAULT;

    kernel_buf[bytes_to_write] = '\0';
    buf_len = bytes_to_write;

    pr_info("mychardev: received %d bytes\n", bytes_to_write);
    return bytes_to_write;
}

/* release: Called when the device file is closed */
static int my_release(struct inode *inode, struct file *filp)
{
    pr_info("mychardev: device closed\n");
    return 0;
}

/* file_operations struct: Connects VFS to the driver */
static const struct file_operations my_fops = {
    .owner   = THIS_MODULE,
    .open    = my_open,
    .read    = my_read,
    .write   = my_write,
    .release = my_release,
};

/* Module initialization */
static int __init my_init(void)
{
    int ret;

    /* Dynamically allocate device number */
    ret = alloc_chrdev_region(&dev_num, 0, 1, DEVICE_NAME);
    if (ret < 0)
        return ret;

    /* Initialize cdev struct and register file_operations */
    cdev_init(&my_cdev, &my_fops);
    ret = cdev_add(&my_cdev, dev_num, 1);
    if (ret < 0) {
        unregister_chrdev_region(dev_num, 1);
        return ret;
    }

    /* Auto-create /dev/mychardev (udev integration) */
    my_class = class_create(THIS_MODULE, DEVICE_NAME);
    device_create(my_class, NULL, dev_num, NULL, DEVICE_NAME);

    pr_info("mychardev: registered with major=%d minor=%d\n",
            MAJOR(dev_num), MINOR(dev_num));
    return 0;
}

/* Module cleanup */
static void __exit my_exit(void)
{
    device_destroy(my_class, dev_num);
    class_destroy(my_class);
    cdev_del(&my_cdev);
    unregister_chrdev_region(dev_num, 1);
    pr_info("mychardev: unregistered\n");
}

module_init(my_init);
module_exit(my_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Example character device driver");
```

Usage:
```bash
# Build and load kernel module
$ make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
$ sudo insmod mychardev.ko

# Verify the device
$ ls -la /dev/mychardev
crw------- 1 root root 237, 0 ... /dev/mychardev

# Write to and read from the device
$ echo "Hello, kernel!" | sudo tee /dev/mychardev
$ sudo cat /dev/mychardev
Hello, kernel!

# Check kernel logs
$ dmesg | tail -5
mychardev: registered with major=237 minor=0
mychardev: device opened
mychardev: received 15 bytes
mychardev: device closed

# Unload the module
$ sudo rmmod mychardev
```

### 4.4 User-Space Drivers (UIO / VFIO)

While traditional drivers operate in kernel space, implementing drivers in user space has been gaining attention in recent years.

```
Kernel Driver vs User-Space Driver:

  (A) Traditional kernel driver:
  +--------------+
  | User Space   |  Application
  |              |      | syscall
  +--------------+------ Kernel Boundary ------
  | Kernel Space |  Driver code
  |              |      | MMIO/DMA
  +--------------+
  | Hardware     |  Device
  +--------------+

  Pros: Full HW feature access
  Cons: Bugs can cause kernel panic, difficult to develop

  (B) User-space driver (UIO/VFIO):
  +--------------+
  | User Space   |  Application + Driver
  |              |      | mmap() for direct device register access
  +--------------+------ Kernel Boundary ------
  | Kernel Space |  Thin UIO/VFIO stub (interrupt notification only)
  +--------------+
  | Hardware     |  Device
  +--------------+

  Pros: Bugs only cause process crash, easier debugging
  Cons: Context switch overhead

  Notable user-space driver frameworks:
  - DPDK: High-speed packet processing (Intel-led)
  - SPDK: High-speed storage I/O
  - VFIO: Device passthrough to VMs
```

---

## 5. I/O Scheduling

### 5.1 The Need for I/O Scheduling

The I/O scheduler is responsible for efficiently reordering and merging multiple I/O requests from applications. Especially for devices like HDDs where seek time is dominant, optimizing request order can dramatically improve throughput.

```
Without vs with I/O scheduling:

  Request positions on disk: [100] [500] [120] [480] [130] [510]

  (A) Without scheduling (FIFO):
    Head movement: 100->500->120->480->130->510
    Total movement: 400 + 380 + 360 + 350 + 380 = 1870 tracks
    -> Head moves back and forth (inefficient)

  (B) With scheduling (SCAN):
    Head movement: 100->120->130->480->500->510
    Total movement: 20 + 10 + 350 + 20 + 10 = 410 tracks
    -> Processes in one direction sequentially (efficient)

    Improvement: (1870 - 410) / 1870 ~ 78% reduction
```

### 5.2 Classical Disk Scheduling Algorithms

```
Major disk scheduling algorithms:

(1) FCFS (First Come First Served)
    Processes in arrival order. Fair but inefficient.

    Request queue: 98, 183, 37, 122, 14, 124, 65, 67
    Initial head position: 53

    Processing order: 53->98->183->37->122->14->124->65->67
    Total movement: 45+85+146+85+108+110+59+2 = 640

(2) SSTF (Shortest Seek Time First)
    Processes the request nearest to current head position next.

    Processing order: 53->65->67->37->14->98->122->124->183
    Total movement: 12+2+30+23+84+24+2+59 = 236

    Problem: Starvation can occur
    -> Requests at the edges may never be processed

(3) SCAN (Elevator Algorithm)
    Head moves in one direction processing requests.
    Reverses at the end. Similar to elevator movement.

    Processing order: 53->37->14->0->65->67->98->122->124->183
    Total movement: 16+23+14+65+2+31+24+2+59 = 236

    Advantage: Eliminates SSTF's starvation problem

(4) C-SCAN (Circular SCAN)
    Services in one direction only. Returns to the beginning at the end.
    High uniformity of response time.

(5) LOOK / C-LOOK
    Improved versions of SCAN/C-SCAN. Does not go to the very end;
    reverses/resets at the position of the last request.
```

### 5.3 Modern Linux I/O Schedulers

The Linux kernel provides multiple I/O schedulers tailored to device characteristics. Since kernel 5.0, multi-queue (blk-mq) based schedulers are the standard.

| Scheduler | Target Device | Algorithm | Characteristics |
|-----------|--------------|-----------|-----------------|
| **none** | NVMe SSD | None (FIFO) | Minimum overhead. Device has FTL |
| **mq-deadline** | SATA SSD / HDD | Deadline | Read priority, deadline guarantee, starvation prevention |
| **bfq** | Desktop | Budget Fair Queueing | Low latency, fairness-focused |
| **kyber** | High-speed SSD | Token-based | Lightweight, 3 queues for read/write/discard |

```bash
# Check current I/O scheduler
$ cat /sys/block/sda/queue/scheduler
[mq-deadline] kyber bfq none

# For NVMe SSD (typically none)
$ cat /sys/block/nvme0n1/queue/scheduler
[none] mq-deadline kyber bfq

# Change I/O scheduler
$ echo "bfq" | sudo tee /sys/block/sda/queue/scheduler

# Check I/O queue depth
$ cat /sys/block/nvme0n1/queue/nr_requests
1023

# I/O scheduler statistics
$ cat /sys/block/sda/queue/stat
# Read: completed  merges  sectors  time(ms)
# Write: completed  merges  sectors  time(ms)
```

### 5.4 mq-deadline Scheduler in Detail

mq-deadline (multi-queue Deadline) is a scheduler that sets a deadline for each I/O request, preventing starvation while maximizing throughput.

```
mq-deadline internal structure:

  Requests from applications
          |
          v
  +-------------------------------------------+
  | Software Queues (per-CPU)                  |
  | +------------+ +------------+              |
  | | CPU 0 Queue| | CPU 1 Queue| ...          |
  | +-----+------+ +-----+------+              |
  |       +------+-------+                     |
  |              v                             |
  | +------------------------------------+    |
  | | mq-deadline Scheduler              |    |
  | |                                    |    |
  | | +- Sorted Queue (by sector) -----+ |    |
  | | | [LBA:100] [LBA:200] [LBA:300]  | |    |
  | | +--------------------------------+ |    |
  | |                                    |    |
  | | +- FIFO Queue (by arrival) ------+ |    |
  | | | [deadline:T1] [T2] [T3]        | |    |
  | | +--------------------------------+ |    |
  | |                                    |    |
  | | Dispatch decision:                 |    |
  | | 1. Expired requests get highest    |    |
  | |    priority                        |    |
  | | 2. Otherwise select from sorted    |    |
  | |    queue                           |    |
  | | 3. Read deadline = 500ms (default) |    |
  | | 4. Write deadline = 5000ms         |    |
  | | -> Prioritize reads (improve       |    |
  | |    interactivity)                  |    |
  | +------------------------------------+    |
  |              |                             |
  +--------------+-----------------------------+
                 v
  +-------------------------------------------+
  | Hardware Dispatch Queue                    |
  | -> To device driver                        |
  +-------------------------------------------+
```

### 5.5 BFQ (Budget Fair Queueing)

BFQ is a scheduler developed as the successor to CFQ (Completely Fair Queueing). It allocates a "budget" (processable sectors) to each process and achieves fair I/O distribution. It excels at interactive performance in desktop environments.

Key features of BFQ:
- Dynamically adjusts I/O budget per process
- Provides idle time to favor sequential I/O
- Automatically prioritizes light I/O processes (GUI apps, etc.)
- Reduces the impact of heavy background I/O (cp, rsync, etc.)

---

## 6. Evolution of Asynchronous I/O

### 6.1 Synchronous I/O vs Asynchronous I/O

```
(A) Synchronous I/O (Blocking):

  Thread 1:  --[read()]----------wait-----------[data received]-->
  Thread 2:  --[read()]----------wait-----------[data received]-->
  Thread 3:  --[read()]----------wait-----------[data received]-->

  Problem: 1 connection = 1 thread -> 10,000 connections = 10,000 threads
  -> Memory: 10,000 x 8MB (stack) = 80GB
  -> Massive context switch cost

(B) Asynchronous I/O (Non-blocking + Event Multiplexing):

  Thread 1:  --[register request]--[other work]--[event received]--[process]-->
              Manages 10,000 connections with 1 thread

  Advantages: Memory-efficient, fewer context switches
  Implementation evolution: select -> poll -> epoll -> io_uring
```

### 6.2 select / poll (Legacy Approaches)

```c
/* Basic usage of select */
#include <sys/select.h>

int main(void)
{
    fd_set read_fds;
    struct timeval timeout;
    int max_fd, nready;

    /* Initialize 1024-FD bitmap each time */
    FD_ZERO(&read_fds);
    FD_SET(sock_fd, &read_fds);
    max_fd = sock_fd;

    timeout.tv_sec = 5;
    timeout.tv_usec = 0;

    /* Ask kernel to check all FD states
       -> O(n) scan proportional to FD count */
    nready = select(max_fd + 1, &read_fds, NULL, NULL, &timeout);

    if (nready > 0 && FD_ISSET(sock_fd, &read_fds)) {
        /* Data available to read */
        read(sock_fd, buf, sizeof(buf));
    }

    return 0;
}

/*
  Limitations of select:
  - FD_SETSIZE = 1024 (fixed at compile time)
  - Copies fd_set to kernel each time (O(n))
  - Kernel scans all FDs internally (O(n))
  - Copies resulting fd_set back to user space (O(n))
  -> Performance degrades linearly as connections increase
*/
```

### 6.3 How epoll Works

epoll was introduced in Linux 2.6 as a high-performance I/O event notification mechanism. It solves the fundamental problems of select and achieves O(1) event notification.

```
epoll internal operation:

  User Space                        Kernel Space
  +-----------------+                +----------------------+
  | Application     |                | epoll Instance       |
  |                 | epoll_create() |                      |
  | (1) Create      |--------------->| +------------------+ |
  |   epoll inst    |                | | Red-Black Tree   | |
  |                 |                | | (monitored FDs)  | |
  |                 | epoll_ctl()    | +------------------+ |
  | (2) Register FD |--------------->|                      |
  |  (ADD/MOD/DEL)  |                | +------------------+ |
  |                 |                | | Ready List       | |
  |                 | epoll_wait()   | | (ready FD list)  | |
  | (3) Wait for    |--------------->| +------------------+ |
  |     events      |                |                      |
  |     (block)     |                | Device interrupt     |
  |                 |                | -> Callback adds to  |
  |                 |                |   Ready List         |
  |                 |                |                      |
  | (4) Return only |<-- Ready FDs --| Return from Ready    |
  |     ready FDs   |                | List                 |
  +-----------------+                +----------------------+

  Fundamental difference from select:
  +------------------------------------------------------------+
  | select: "Check all FDs for me" each time                    |
  |   -> Kernel scans 10,000 FDs every time                     |
  |   -> O(n) x number of calls                                 |
  |                                                            |
  | epoll: "Tell me only about FDs that changed"                |
  |   -> Kernel adds to Ready List via callbacks                |
  |   -> epoll_wait just returns the Ready List                 |
  |   -> O(1) event notification                                |
  +------------------------------------------------------------+
```

**Example of a high-performance TCP server using epoll:**

```c
/* Echo server using epoll */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#include <errno.h>

#define MAX_EVENTS 1024
#define BUF_SIZE   4096
#define PORT       8080

/* Set socket to non-blocking */
static void set_nonblocking(int fd)
{
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main(void)
{
    int listen_fd, epoll_fd, nfds, i;
    struct epoll_event ev, events[MAX_EVENTS];
    struct sockaddr_in addr;

    /* Create listening socket */
    listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(PORT);

    bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(listen_fd, SOMAXCONN);
    set_nonblocking(listen_fd);

    /* Create epoll instance */
    epoll_fd = epoll_create1(0);

    /* Register listening socket with epoll */
    ev.events = EPOLLIN;
    ev.data.fd = listen_fd;
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, listen_fd, &ev);

    printf("Echo server listening on port %d\n", PORT);

    /* Event loop */
    for (;;) {
        /* Wait for ready FDs (timeout: -1 = wait indefinitely) */
        nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);

        for (i = 0; i < nfds; i++) {
            if (events[i].data.fd == listen_fd) {
                /* Accept new connection */
                int client_fd = accept(listen_fd, NULL, NULL);
                if (client_fd < 0) continue;

                set_nonblocking(client_fd);
                ev.events = EPOLLIN | EPOLLET;  /* Edge-triggered */
                ev.data.fd = client_fd;
                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &ev);

            } else {
                /* Process data from client */
                char buf[BUF_SIZE];
                ssize_t n = read(events[i].data.fd, buf, sizeof(buf));

                if (n <= 0) {
                    /* Connection closed or error */
                    epoll_ctl(epoll_fd, EPOLL_CTL_DEL,
                              events[i].data.fd, NULL);
                    close(events[i].data.fd);
                } else {
                    /* Echo back */
                    write(events[i].data.fd, buf, n);
                }
            }
        }
    }

    close(listen_fd);
    close(epoll_fd);
    return 0;
}
```

### 6.4 epoll Trigger Modes

| Mode | Behavior | Characteristics | Use Case |
|------|----------|----------------|----------|
| Level-Triggered (LT) | Continues notifying as long as data remains | Compatible with select. Less prone to missing data | Default. General use |
| Edge-Triggered (ET) | Notifies only once when state changes | Highly efficient but must be careful about leftover data | High-performance servers (Nginx) |

When using edge-triggered mode, upon receiving a notification, you must loop reading all data until EAGAIN is returned. Failing to do so results in remaining data never triggering another notification, causing the connection to hang.

### 6.5 io_uring (Linux 5.1+)

io_uring is a revolutionary asynchronous I/O interface introduced in Linux kernel 5.1 in 2019. It overcomes the limitations of traditional epoll and AIO (Linux AIO), providing near-zero-copy performance for both file I/O and network I/O through a unified interface.

```
io_uring architecture:

  +---------------------------------------------------------+
  |                  Shared Memory Region                     |
  |                                                         |
  |  Submission Queue (SQ)           Completion Queue (CQ)  |
  |  +---+---+---+---+---+         +---+---+---+---+---+  |
  |  |SQE|SQE|SQE|   |   |         |CQE|CQE|   |   |   |  |
  |  | 0 | 1 | 2 |   |   |         | 0 | 1 |   |   |   |  |
  |  +-^-+---+---+---+---+         +---+---+-|-+---+---+  |
  |    |                                      |             |
  |    | User submits                  Kernel |writes result|
  |    | (no syscall needed!)                 |             |
  |    |                                      v             |
  +----|--------------------------------------|---------+---+
       |                                      |
  +----+----------+                    +------+--------+
  | User Space    |                    | User Space    |
  | Create SQEs   |                    | Read CQEs and |
  | and submit    |                    | process results|
  | to the ring   |                    |               |
  +---------------+                    +---------------+

  SQE (Submission Queue Entry) structure:
  +------------------------------------+
  | opcode:   IORING_OP_READ          | <- Operation type
  | fd:       File descriptor          |
  | addr:     Buffer address           |
  | len:      Transfer byte count      |
  | offset:   File offset              |
  | user_data: User identifier         | <- Correlates with CQE
  +------------------------------------+

  CQE (Completion Queue Entry) structure:
  +------------------------------------+
  | user_data: User identifier         | <- Same value as SQE
  | res:       Result (bytes or error) |
  +------------------------------------+
```

**Revolutionary aspects of io_uring:**

1. **Reduced system calls:** Submission to the SQ completes with just a shared memory write from user space. In `SQPOLL` mode, a kernel thread automatically monitors the SQ, making even `io_uring_enter()` unnecessary.

2. **Batch processing:** Multiple I/O requests can be submitted to the SQ at once. Where traditionally 1 request = 1 system call, io_uring can process hundreds of requests with 0-1 system calls.

3. **Unified interface:** File reads/writes, network send/receive, timers, file sync (fsync), and more can all be handled through the same ring buffer.

### 6.6 Comprehensive Comparison of I/O Multiplexing Methods

| Method | Introduced | Complexity | FD Limit | Capabilities | Primary Users |
|--------|-----------|-----------|----------|-------------|--------------|
| **select** | 1983 | O(n) | 1024 | Basic multiplexing | Legacy systems |
| **poll** | 1986 | O(n) | None | Extended select | Small servers |
| **epoll** | 2002 | O(1) | None | Event-driven | Nginx, Redis, Node.js |
| **kqueue** | 2000 | O(1) | None | epoll equivalent (BSD) | macOS, FreeBSD |
| **IOCP** | 2000 | O(1) | None | Proactor model | Windows (.NET) |
| **io_uring** | 2019 | O(1) | None | Zero-copy, unified API | High-perf DB, storage |

---

## 7. I/O Optimization in Practice

### 7.1 I/O Performance Measurement Tools

Understanding the Linux tools for identifying I/O bottlenecks.

```bash
# (1) iostat: Device-level I/O statistics
$ iostat -xz 1
Device  r/s    w/s   rkB/s   wkB/s  rrqm/s  wrqm/s  %util  await
sda     150    50    6000    2000   10      30       65%    4.2
nvme0n1 5000   3000  200000  120000 0       0        40%    0.1

# Key metrics to watch:
# - %util: Device utilization. Near 100% means saturated
# - await: Average I/O wait time (ms). High means bottleneck
# - r/s, w/s: Read/write I/O count per second

# (2) blktrace + blkparse: Detailed block I/O tracing
$ sudo blktrace -d /dev/sda -o - | blkparse -i -
  8,0  1  1  0.000000000  1234  Q  R  100 + 8  [myapp]
  8,0  1  2  0.000001000  1234  G  R  100 + 8  [myapp]
  8,0  1  3  0.000005000  1234  D  R  100 + 8  [myapp]
  8,0  1  4  0.004200000  1234  C  R  100 + 8  [0]

# Q=Queued, G=Get request, D=Dispatched, C=Completed

# (3) strace: System call tracing
$ strace -e trace=read,write,open,close -c ./myapp
% time     seconds  usecs/call     calls    errors  syscall
------ ----------- ----------- --------- --------- --------
 85.30    1.234567          12    102400           read
 10.20    0.147654           8     18000           write
  4.50    0.065123          65      1000           open

# (4) perf: I/O-related event profiling
$ sudo perf record -e block:block_rq_insert,block:block_rq_complete -a
$ sudo perf report
```

### 7.2 Zero-Copy Techniques

Traditional file transmission copies data multiple times between kernel and user buffers. Zero-copy techniques eliminate these unnecessary copies.

```
Traditional read() + write() (4 copies):

  Disk -> [DMA] -> Kernel Buffer -> [CPU] -> User Buffer
                                                  |
  Socket <- [DMA] <- Kernel Buffer <- [CPU] <- User Buffer

  Copy count: 4
  Context switches: 4 (read x 2 + write x 2)

sendfile() zero-copy (2 copies):

  Disk -> [DMA] -> Kernel Buffer --[CPU]--> Socket Buffer
                                                 |
  Socket <- [DMA] <-----------------------------+

  Copy count: 2 (bypasses user space)
  Context switches: 2

splice() / sendfile() + DMA Scatter-Gather:

  Disk -> [DMA] -> Kernel Buffer --reference info--> Socket Buffer
                         |                              |
                         +--------- [DMA] -----------> NIC

  Copy count: 0 (no CPU copy, DMA only)
  Context switches: 2
```

### 7.3 Node.js Event Loop and I/O

```javascript
/*
 * Internal structure of Node.js event loop
 * libuv -> epoll (Linux) / kqueue (macOS) / IOCP (Windows)
 */

const fs = require('fs');
const http = require('http');

/*
 * File I/O: Executed in libuv's thread pool
 * (because epoll does not support file I/O)
 */
fs.readFile('/path/to/large-file', (err, data) => {
    /* A thread pool worker executes read()
       After completion, callback is pushed to event queue */
    if (err) throw err;
    console.log(`Read ${data.length} bytes`);
});

/*
 * Network I/O: Directly multiplexed via epoll/kqueue
 * Does not use the thread pool (non-blocking sockets)
 */
const server = http.createServer((req, res) => {
    /* Can handle tens of thousands of concurrent connections with 1 thread */
    res.writeHead(200, { 'Content-Type': 'text/plain' });
    res.end('Hello, World!\n');
});
server.listen(3000);

/*
 * Event loop phases:
 *
 *   +---------------------------+
 *   |      timers               |  <- setTimeout, setInterval
 *   +---------------------------+
 *   |      pending callbacks    |  <- I/O callbacks (partial)
 *   +---------------------------+
 *   |      idle, prepare        |  <- Internal processing
 *   +---------------------------+
 *   |      poll                 |  <- epoll_wait() for I/O
 *   |      (get I/O events)     |     Execute new I/O callbacks
 *   +---------------------------+
 *   |      check                |  <- setImmediate()
 *   +---------------------------+
 *   |      close callbacks      |  <- socket.on('close', ...)
 *   +---------------------------+
 *          ^                |
 *          +----------------+  (loop)
 *
 * process.nextTick() interleaves between each phase
 */
```

---

## 8. Anti-Patterns and Solutions

### 8.1 Anti-Pattern 1: Careless Use of Synchronous I/O

```
Problem: Synchronous file reads per request in a web server

  +------------------------------------------------------+
  | BAD: Blocking with synchronous I/O                     |
  |                                                      |
  |  Request 1: --[read(file)]---- 50ms wait ---->process |
  |  Request 2: ----------------- waiting ------------>    |
  |  Request 3: ----------------- waiting ------------>    |
  |                                                      |
  |  At 100 requests/sec, average response time: 2.5s     |
  |  -> With only 1 thread, I/O wait stalls everything    |
  +------------------------------------------------------+

  Root cause:
  - read() / write() are blocking by default
  - With few threads, I/O waits stall processing
  - Adding threads increases memory and context switch overhead

  +------------------------------------------------------+
  | GOOD: Async I/O + event-driven                         |
  |                                                      |
  |  Request 1: --[async read]--> [other work] --> done   |
  |  Request 2: --[async read]--> [other work] --> done   |
  |  Request 3: --[async read]--> [other work] --> done   |
  |                                                      |
  |  Handle tens of thousands of requests concurrently     |
  |  with 1 thread                                        |
  |  -> Model adopted by Nginx, Node.js, Go               |
  +------------------------------------------------------+

  Solutions:
  1. Use non-blocking I/O + epoll/kqueue
  2. Leverage async/await patterns (Python asyncio, Rust tokio)
  3. Offload I/O operations to thread pools (Java NIO, libuv)
```

### 8.2 Anti-Pattern 2: Forgetting DMA Buffer Cache Management

```
Problem: Driver implementation ignoring DMA buffer cache coherency

  +------------------------------------------------------+
  | BAD: kmalloc + virt_to_phys for direct DMA address    |
  |                                                      |
  |   buf = kmalloc(4096, GFP_KERNEL);                   |
  |   dma_addr = virt_to_phys(buf);  /* Dangerous! */    |
  |   /* Cache coherency not guaranteed */                |
  |   /* IOMMU not considered, no bounce buffer */        |
  |   /* 32-bit DMA devices cannot access >4GB */         |
  |                                                      |
  | Symptoms:                                            |
  | - Data corruption occurs "occasionally" (hard to      |
  |   reproduce)                                         |
  | - Crashes only on specific hardware configurations    |
  | - Issues manifest only under high load                |
  +------------------------------------------------------+

  +------------------------------------------------------+
  | GOOD: Use the DMA API correctly                        |
  |                                                      |
  |   /* Allocate coherent DMA buffer */                  |
  |   buf = dma_alloc_coherent(dev, 4096,                |
  |                            &dma_handle, GFP_KERNEL); |
  |                                                      |
  |   /* Or use streaming DMA mapping */                  |
  |   dma_handle = dma_map_single(dev, buf, 4096,        |
  |                                DMA_FROM_DEVICE);     |
  |   /* After I/O completion */                          |
  |   dma_unmap_single(dev, dma_handle, 4096,            |
  |                     DMA_FROM_DEVICE);                 |
  |                                                      |
  | Benefits:                                            |
  | - Cache coherency automatically guaranteed            |
  | - IOMMU integration (essential for virtualization)    |
  | - Automatic bounce buffer handling                     |
  +------------------------------------------------------+
```

### 8.3 Anti-Pattern 3: Inappropriate I/O Scheduler Selection

```
Problem: Using BFQ scheduler for NVMe SSDs

  +------------------------------------------------------+
  | BAD: NVMe SSD + BFQ                                   |
  |                                                      |
  | NVMe SSD characteristics:                             |
  | - Small difference between random and sequential      |
  |   access                                             |
  | - Hardware has multiple queues (up to 65535)          |
  | - Internal FTL performs I/O optimization               |
  |                                                      |
  | BFQ overhead:                                        |
  | - Per-process budget calculation                       |
  | - Request sorting and merging                         |
  | - -> Unnecessary processing for SSDs degrades         |
  |   performance                                        |
  | - -> CPU becomes bottleneck in high-IOPS environments |
  +------------------------------------------------------+

  +------------------------------------------------------+
  | GOOD: Select scheduler matching device characteristics |
  |                                                      |
  | NVMe SSD     -> none (no scheduler)                   |
  | SATA SSD     -> mq-deadline                           |
  | HDD          -> mq-deadline or bfq                    |
  | Desktop use  -> bfq (interactivity-focused)           |
  | Server use   -> mq-deadline (throughput-focused)       |
  +------------------------------------------------------+
```

---

## 9. Practice Exercises

### Exercise 1: Selecting I/O Methods (Fundamentals)

For each device and usage scenario below, select the optimal I/O method (polling, interrupt, DMA) and explain your reasoning.

1. Keyboard input (user typing characters)
2. NVMe SSD random 4KB reads (1 million IOPS environment)
3. 10Gbps network reception (large file transfer)
4. Periodic temperature sensor reading (1-second interval, embedded system)
5. Large framebuffer transfer from GPU (4K 60fps)

**Model Answers:**

1. **Keyboard -> Interrupt-driven**
   Reason: Input frequency is low (tens to hundreds per second), so polling would waste CPU cycles. With interrupts, the CPU only responds when input occurs. DMA is unnecessary since the transfer data volume is extremely small (1 to a few bytes).

2. **NVMe SSD high IOPS -> Polling (io_poll)**
   Reason: I/O completion time is ~10us while interrupt overhead is 1-5us. The interrupt cost is relatively large, so polling minimizes latency. Linux's `io_poll` flag corresponds to this use case.

3. **10Gbps bulk transfer -> DMA + Interrupts**
   Reason: The transfer data volume is massive (~1.25GB/s), and CPU-mediated copying cannot saturate the bandwidth. DMA directly transfers from NIC to memory, with completion notified via interrupt. Additionally, NAPI (Linux) interrupts only on packet arrival, then switches to polling -- a hybrid approach.

4. **Temperature sensor -> Polling**
   Reason: Periodic reading at 1-second intervals; driving polling via a timer interrupt is sufficient. The responsiveness of interrupts is unnecessary. In embedded systems where interrupt controller resources are limited, polling is rational.

5. **GPU framebuffer -> DMA**
   Reason: 4K 60fps frame data reaches 3840x2160x4Bx60 ~ 1.99GB/s. CPU-mediated transfer would be bandwidth-insufficient, so transfer occurs via PCIe Bus Master DMA. The GPU autonomously controls DMA and notifies completion via VSync interrupt.

### Exercise 2: Implementing an epoll Server (Applied)

Implement a chat server using epoll that meets the following requirements.

- Support up to 10,000 simultaneous connections
- Use edge-triggered mode
- Broadcast messages from any client to all clients
- Log connections/disconnections

Hint: In edge-triggered mode, read in a loop until EAGAIN. Use a hash table or array for connection list management.

### Exercise 3: I/O Performance Analysis (Advanced)

Perform I/O bottleneck analysis using the following steps.

1. Measure baseline performance of the target storage with `fio`
```bash
# Sequential read
fio --name=seq-read --rw=read --bs=1M --size=1G \
    --numjobs=1 --ioengine=libaio --direct=1 --runtime=30

# Random read (4KB)
fio --name=rand-read --rw=randread --bs=4k --size=1G \
    --numjobs=4 --iodepth=32 --ioengine=libaio --direct=1 --runtime=30

# Comparison with io_uring engine
fio --name=io-uring --rw=randread --bs=4k --size=1G \
    --numjobs=4 --iodepth=32 --ioengine=io_uring --direct=1 --runtime=30
```

2. Check the system call distribution of the target application with `strace -c`
3. Check device utilization and I/O wait time with `iostat -xz 1`
4. Identify the bottleneck cause and propose improvements

---

## 10. FAQ

### Q1: What does "everything is a file" mean?

**A**: It is one of UNIX's design philosophies. Almost all resources -- devices (`/dev/sda`), process information (`/proc/`), kernel parameters (`/sys/`), network sockets, etc. -- are abstracted as "files" and accessed through a unified API of `open()`, `read()`, `write()`, `close()`.

This philosophy allows file operation knowledge to transfer directly to device operations, and facilitates inter-program cooperation through pipes and redirection. Plan 9 (a successor research OS to UNIX) took this further, even performing network communication through the filesystem.

### Q2: What is the relationship between async/await and epoll?

**A**: `async/await` is syntactic sugar in programming languages. Under the hood, an epoll-based (Linux) or kqueue-based (macOS) event loop is running. When `await` is called, execution of the current function is suspended, and the target I/O operation is registered with epoll. When epoll_wait() detects I/O completion, the suspended function resumes.

Specific mappings:
- Python asyncio -> Wraps epoll_wait() (Linux) / kqueue (macOS)
- Rust tokio -> Wraps epoll (Linux) / kqueue (macOS) / IOCP (Windows)
- Go goroutine -> Internally uses netpoller (Go's own epoll/kqueue wrapper)
- Node.js -> Uses libuv (abstraction library for epoll/kqueue/IOCP)

### Q3: When should io_uring be used?

**A**: io_uring is particularly effective in the following cases.

1. **High-throughput storage:** When you want to maximize NVMe SSD performance. Traditional libaio incurs a system call per request, while io_uring reduces system calls through batch submission.
2. **Database engines:** RocksDB, ScyllaDB, TiKV, and others have adopted io_uring to reduce write latency.
3. **File servers:** When handling large volumes of file I/O concurrently.

For typical web applications, however, epoll (Node.js, Nginx) is often sufficient. The benefits of io_uring become significant when I/O operations reach hundreds of thousands per second or more. Additionally, note that io_uring has security concerns, and some Linux distributions (e.g., Ubuntu) restrict its use by unprivileged users by default.

### Q4: Is an I/O scheduler unnecessary for NVMe SSDs?

**A**: In most cases, `none` (no scheduler) is optimal for NVMe SSDs. There are three reasons.

First, NVMe SSDs have no seek time unlike HDDs, so there is no performance gain from reordering requests. Second, NVMe SSDs have an internal FTL (Flash Translation Layer) that performs its own I/O optimization. Third, NVMe supports up to 65535 hardware queues, making it more efficient to dispatch directly to hardware queues without a software scheduler.

However, in multi-tenant environments (cloud VMs, etc.) where I/O fairness is required, `mq-deadline` or `bfq` may be configured.

### Q5: What is the difference between Windows IOCP and epoll?

**A**: They differ fundamentally in design philosophy.

epoll is based on the "reactor model." You ask the kernel "notify me when I/O is possible," then the application performs the I/O itself upon notification (`epoll_wait()` -> `read()`).

IOCP is based on the "proactor model." You ask the kernel "perform this I/O for me," and the kernel notifies you after completing the I/O (receive the completed result with `GetQueuedCompletionStatus()`).

IOCP makes application-side code more concise, but the OS internal implementation is more complex. There is no significant performance difference; it depends on platform choice.

---

## 11. Advanced Topics

### 11.1 IOMMU (I/O Memory Management Unit)

The IOMMU is a hardware unit that virtualizes I/O device memory access, translating between addresses visible to devices and physical addresses. It is known as Intel VT-d or AMD-Vi.

```
IOMMU role:

  (A) Without IOMMU:
  Device --physical address--> Memory
  -> Device can access any physical address (dangerous)
  -> Malicious device can read/write kernel memory

  (B) With IOMMU:
  Device --I/O virtual address--> IOMMU --physical address--> Memory
  -> IOMMU performs address translation and access control
  -> Device can only access permitted regions

  IOMMU address translation table:
  +------------------------------------------------------+
  |  Device (BDF: Bus/Device/Function)                    |
  |     |                                                 |
  |     v                                                 |
  |  +----------------------+                             |
  |  | Context Table        |  References per-device      |
  |  | (Root Table Entry)   |  page tables                |
  |  +----------+-----------+                             |
  |             v                                         |
  |  +----------------------+                             |
  |  | I/O Page Table       |  Similar to CPU page table  |
  |  | IOVA -> Phys Address |  4-level walk               |
  |  +----------+-----------+                             |
  |             v                                         |
  |  Only permitted physical memory regions accessible    |
  +------------------------------------------------------+

  Primary IOMMU uses:
  1. DMA remapping: Restrict device DMA access (security)
  2. Interrupt remapping: Prevent MSI/MSI-X interrupt forgery
  3. Device passthrough: Directly assign devices to VMs (VFIO)
  4. SG-DMA simplification: Provide contiguous IOVA addresses
```

### 11.2 NAPI (New API) -- Linux Hybrid Network Reception

NAPI is a hybrid approach combining interrupts and polling for network reception in the Linux kernel. It dramatically improves packet processing efficiency under high load.

```
NAPI operating modes:

  (1) Low load: Interrupt-driven
  +----------------------------------------------------+
  |  Packet arrives -> Interrupt -> ISR processes packet |
  |  Packet arrives -> Interrupt -> ISR processes packet |
  |                                                    |
  |  Long intervals between packets -> Interrupts       |
  |  are sufficient                                    |
  +----------------------------------------------------+

  (2) High load: Switch to polling
  +----------------------------------------------------+
  |  Packet arrives -> Interrupt -> Disable interrupts  |
  |  -> Start NAPI polling mode                         |
  |                                                    |
  |  +---------------------------------------------+   |
  |  |  napi_poll() loop:                           |   |
  |  |    while (budget > 0) {                      |   |
  |  |      packet = Get directly from NIC (no IRQ) |   |
  |  |      Process packet                          |   |
  |  |      budget--;                               |   |
  |  |    }                                         |   |
  |  |    if (more packets remaining)               |   |
  |  |      -> Schedule next poll                   |   |
  |  |    else                                      |   |
  |  |      -> Re-enable interrupts (return to low  |   |
  |  |         load)                                |   |
  |  +---------------------------------------------+   |
  |                                                    |
  |  Benefit: Avoids interrupt storm                    |
  |  Per-packet interrupt unnecessary -> Higher          |
  |  throughput                                        |
  +----------------------------------------------------+

  Performance comparison (10GbE, 1500-byte packets):
  - Interrupt only: Max packet receive rate ~ 1M packets/s
    (interrupt overhead dominates)
  - NAPI:          Max packet receive rate ~ 14.8M packets/s
    (achieves near-theoretical maximum)
```

### 11.3 Virtualized I/O -- virtio

In virtualized environments (KVM/QEMU, etc.), a dedicated interface is needed for I/O communication between guest and host OS. virtio is the standard specification for paravirtualized I/O, enabling efficient data transfer.

```
Three approaches to virtualized I/O:

  (A) Full emulation:
  +----------------------+    +----------------------+
  | Guest OS             |    | Host OS / Hypervisor |
  | Existing driver  --> |    |                      |
  | [Virtual HW] --VMEXIT->|-->| [Device Emulator]   |
  |                      |    | --> Physical device   |
  +----------------------+    +----------------------+
  Performance: Low (frequent VMEXITs)
  Compatibility: High (existing drivers work as-is)

  (B) Paravirtualization (virtio):
  +----------------------+    +----------------------+
  | Guest OS             |    | Host OS / Hypervisor |
  | virtio driver -->    |    |                      |
  | [Shared Virtqueue] --|--->| [virtio Backend]     |
  | (ring buffer)        |    | --> Physical device   |
  +----------------------+    +----------------------+
  Performance: High (minimal VMEXITs, batch processing)
  Compatibility: Requires virtio drivers

  (C) Device passthrough (VFIO + IOMMU):
  +----------------------+    +----------------------+
  | Guest OS             |    | Hypervisor           |
  | Native driver        |    | (control plane only) |
  |      |               |    |                      |
  |      +-- via IOMMU --|---->| Directly to physical|
  |                      |    |  device              |
  +----------------------+    +----------------------+
  Performance: Near-native (bypasses hypervisor)
  Constraint: Device dedicated to 1 VM, live migration difficult

  Virtqueue structure:
  +----------------------------------------------------+
  |  Descriptor Table                                  |
  |  +-----+-------------+------+-------+              |
  |  | idx | addr (GPA)  | len  | flags |              |
  |  +-----+-------------+------+-------+              |
  |  |  0  | 0x1000      | 1500 | NEXT  |              |
  |  |  1  | 0x2000      | 4096 | WRITE |              |
  |  | ... | ...         | ...  | ...   |              |
  |  +-----+-------------+------+-------+              |
  |                                                    |
  |  Available Ring (Guest -> Host)                     |
  |  +------+---+---+---+---+                          |
  |  | idx  | 0 | 1 | 2 |   | <- Guest notifies       |
  |  +------+---+---+---+---+   available descriptors  |
  |                                                    |
  |  Used Ring (Host -> Guest)                          |
  |  +------+---+---+---+---+                          |
  |  | idx  | 0 | 1 |   |   | <- Host notifies        |
  |  +------+---+---+---+---+   completed descriptors  |
  +----------------------------------------------------+
```

### 11.4 CXL (Compute Express Link)

CXL (Compute Express Link) is a new interconnect specification built on the PCIe physical layer, enabling cache-coherent communication between CPUs, memory, and accelerators.

```
Three CXL protocols:

  +----------------------------------------------------+
  |                    CXL                              |
  |  +----------------------------------------------+  |
  |  | CXL.io    : PCIe-compatible device discovery  |  |
  |  |             and configuration                 |  |
  |  |             (essentially PCIe itself)         |  |
  |  +----------------------------------------------+  |
  |  | CXL.cache : Device accesses host memory with  |  |
  |  |             cache coherency                   |  |
  |  |             (GPU/FPGA coherent with CPU cache)|  |
  |  +----------------------------------------------+  |
  |  | CXL.mem   : Host accesses device memory as    |  |
  |  |             regular memory                    |  |
  |  |             (memory pooling, expansion)       |  |
  |  +----------------------------------------------+  |
  |                    |                                |
  |                    v                                |
  |  +----------------------------------------------+  |
  |  |           PCIe Physical Layer (PHY)           |  |
  |  +----------------------------------------------+  |
  +----------------------------------------------------+

  CXL use cases:
  - Type 1: Accelerators (FPGA/SmartNIC)
    -> CXL.io + CXL.cache
  - Type 2: GPU/AI accelerators
    -> CXL.io + CXL.cache + CXL.mem (share device-attached memory)
  - Type 3: Memory expansion devices
    -> CXL.io + CXL.mem (large-scale memory pool)
```

### 11.5 I/O and CPU Coordination -- Hybrid Approaches

In modern high-performance I/O systems, dynamically switching between polling, interrupts, and DMA based on conditions is the mainstream approach.

```
Hybrid I/O example (Linux NVMe driver):

  I/O request issued
       |
       v
  +---------------------------------------------+
  | Decision: Check request size and queue depth |
  |                                             |
  | if (request_size < 4KB && queue_depth < 4) {|
  |   -> Synchronous polling (io_poll)           |
  |   -> Prioritize latency minimization         |
  | }                                           |
  | else if (request_size >= 4KB) {             |
  |   -> DMA + interrupt                         |
  |   -> Prioritize throughput maximization       |
  | }                                           |
  | else {                                      |
  |   -> Interrupt coalescing                     |
  |   -> Merge multiple completions into 1 IRQ   |
  |   -> Reduce interrupt overhead                |
  | }                                           |
  +---------------------------------------------+

  Interrupt Coalescing:
  +----------------------------------------------------+
  | Without coalescing:                                |
  |  Complete -> IRQ -> Complete -> IRQ -> Complete ->  |
  |  IRQ                                               |
  |  (3 interrupts)                                    |
  |                                                    |
  | Timer-based coalescing (100us):                    |
  |  Complete -> Complete -> Complete -> [100us] -> IRQ |
  |  (1 interrupt handles 3 completions)               |
  |                                                    |
  | Count-based coalescing (16 items):                 |
  |  Complete x 16 -> IRQ                              |
  |  (1 interrupt handles 16 completions)              |
  |                                                    |
  | Trade-off:                                         |
  | - Higher coalescing -> Lower IRQ overhead, higher   |
  |   latency                                          |
  | - Lower coalescing -> Higher IRQ overhead, lower    |
  |   latency                                          |
  +----------------------------------------------------+
```

### 11.6 I/O System Implementation Example in Rust

Rust's type system and ownership model are well-suited for safe I/O programming. Below is an example of async I/O using the tokio runtime.

```rust
/* Async TCP echo server with Rust + tokio */
use tokio::net::TcpListener;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    println!("Server listening on port 8080");

    loop {
        // accept() wraps epoll_wait()
        let (mut socket, addr) = listener.accept().await?;
        println!("New connection from: {}", addr);

        // Spawn a task per connection (not a thread)
        // Can handle tens of thousands of concurrent connections with 1 thread
        tokio::spawn(async move {
            let mut buf = [0u8; 4096];

            loop {
                // read() internally registers with epoll -> switches to other tasks
                let n = match socket.read(&mut buf).await {
                    Ok(0) => {
                        println!("Connection closed: {}", addr);
                        return;
                    }
                    Ok(n) => n,
                    Err(e) => {
                        eprintln!("Read error: {}", e);
                        return;
                    }
                };

                // Echo back
                if let Err(e) = socket.write_all(&buf[..n]).await {
                    eprintln!("Write error: {}", e);
                    return;
                }
            }
        });
    }
}

/*
 * tokio internal operation:
 * 1. TcpListener::accept() -> epoll_ctl(EPOLL_CTL_ADD, listen_fd)
 * 2. socket.read() -> Actually a non-blocking read()
 *    -> If EAGAIN, register with epoll and switch to another task
 *    -> If data available, return immediately
 * 3. tokio::spawn() -> Add to task queue (not an OS thread)
 * 4. Internal runtime loop:
 *    loop {
 *      events = epoll_wait(...)
 *      for event in events {
 *        Wake the task corresponding to this event
 *      }
 *      Execute ready tasks
 *    }
 */
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently applied in everyday development work, and is particularly important during code reviews and architecture design.

---

## 12. Summary

| Concept | Key Point |
|---------|-----------|
| I/O Addressing | PMIO is legacy; MMIO is the modern standard. Mapped via PCIe BAR |
| Bus Architecture | PCIe is standard. Bandwidth = lanes x transfer rate x encoding efficiency |
| Polling | Simple and low-latency but wastes CPU. Still used in DPDK, NVMe io_poll |
| Interrupts | General-purpose method using CPU efficiently. Split into Top Half / Bottom Half |
| DMA | Bulk transfer without CPU. Note SG-DMA and cache coherency |
| Device Drivers | Unified interface via VFS. Dynamically loadable as LKM |
| I/O Scheduler | NVMe->none, SATA SSD/HDD->mq-deadline, Desktop->bfq |
| epoll | O(1) event notification, solves C10K problem. Foundation of Nginx/Redis/Node.js |
| io_uring | Zero-copy async I/O. Shared ring buffer reduces syscalls |
| Zero-Copy | sendfile/splice eliminates CPU copies. Essential for high-bandwidth transfers |
| IOMMU | Address translation and access control for device DMA. Essential for virtualization |
| NAPI | Hybrid of interrupts and polling. Effective under high network load |
| virtio | Standard for paravirtualized I/O. Efficient VM I/O via shared ring buffer |
| CXL | Cache-coherent connection on PCIe. Foundation for memory pooling |

---

## Recommended Next Reading


---

## References

1. Love, R. *Linux Kernel Development.* 3rd Edition, Addison-Wesley, 2010. -- The definitive reference comprehensively covering Linux kernel interrupt handling, device drivers, and memory management.
2. Arpaci-Dusseau, R. H. and Arpaci-Dusseau, A. C. *Operating Systems: Three Easy Pieces (OSTEP).* Chapter 36: I/O Devices, Chapter 37: Hard Disk Drives. https://pages.cs.wisc.edu/~remzi/OSTEP/ -- An OS textbook adopted by universities worldwide. The I/O device chapters are ideal for understanding polling, interrupts, and DMA.
3. Axboe, J. "Efficient I/O with io_uring." Linux kernel documentation, 2019. https://kernel.dk/io_uring.pdf -- Technical documentation by io_uring designer Jens Axboe himself. Details the ring buffer design philosophy and benchmark results.
4. Stevens, W. R. and Rago, S. A. *Advanced Programming in the UNIX Environment.* 3rd Edition, Addison-Wesley, 2013. -- The definitive work on UNIX I/O programming. Historical context and implementation details of select/poll/epoll.
5. Corbet, J., Rubini, A., and Kroah-Hartman, G. *Linux Device Drivers.* 3rd Edition, O'Reilly, 2005. https://lwn.net/Kernel/LDD3/ -- The bible of Linux device driver development. Covers character devices, block devices, DMA, and interrupt implementations. Available online for free.
6. Tanenbaum, A. S. and Bos, H. *Modern Operating Systems.* 4th Edition, Pearson, 2014. -- Excellent coverage of I/O software layered structure (interrupt handler -> device driver -> device-independent software -> user space).
7. Patterson, D. A. and Hennessy, J. L. *Computer Organization and Design: The Hardware/Software Interface.* 6th Edition, Morgan Kaufmann, 2020. -- Explains bus protocols, I/O interfaces, and DMA mechanisms from a hardware perspective.
