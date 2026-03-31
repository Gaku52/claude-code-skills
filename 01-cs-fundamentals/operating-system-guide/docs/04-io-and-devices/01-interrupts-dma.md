# Interrupts and DMA — A Complete Overview of CPU-Device Communication

> **Interrupts** are the fundamental mechanism for asynchronous communication between the CPU and external devices, and **DMA (Direct Memory Access)** is a technology that enables high-speed data transfer without CPU intervention. A proper understanding of these two concepts is essential for OS kernel development, device driver design, embedded systems development, and performance tuning.

---

## What You Will Learn in This Chapter

- [ ] Systematically understand the classification of interrupts (hardware interrupts, software interrupts, exceptions)
- [ ] Explain the structure and lookup procedure of the Interrupt Descriptor Table (IDT)
- [ ] Understand the top-half/bottom-half split design in the Linux kernel
- [ ] Trace the complete flow of DMA transfer initialization, execution, and completion notification
- [ ] Explain the roles of scatter/gather DMA and IOMMU
- [ ] Understand the position of modern I/O technologies such as MSI/MSI-X, RDMA, and NVMe
- [ ] Practice performance optimization through interrupt affinity and IRQ balancing
- [ ] Recognize common anti-patterns and design countermeasures


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding the content of [Device Drivers](./00-device-drivers.md)

---

## Overall Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Space                                  │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│   │ App (A)  │  │ App (B)  │  │ App (C)  │  │ App (D)  │          │
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│        │ syscall      │ syscall     │ read()      │ write()        │
├────────┼──────────────┼─────────────┼─────────────┼────────────────┤
│        ▼              ▼             ▼             ▼                 │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              VFS (Virtual File System)                    │     │
│   └──────────────────────┬───────────────────────────────────┘     │
│                          ▼                                          │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │           Block / Character Device Layer                  │     │
│   │  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │     │
│   │  │ I/O        │  │ Interrupt   │  │ DMA          │      │     │
│   │  │ Scheduler  │  │ Handler     │  │ Engine       │      │     │
│   │  └─────┬──────┘  └──────┬──────┘  └──────┬───────┘      │     │
│   └────────┼────────────────┼────────────────┼───────────────┘     │
│            ▼                ▼                ▼                       │
│   ┌──────────────────────────────────────────────────────────┐     │
│   │              Hardware Abstraction Layer                    │     │
│   │   ┌──────┐  ┌───────┐  ┌────────┐  ┌──────────┐        │     │
│   │   │ PIC/ │  │ APIC  │  │ IOMMU  │  │ DMA      │        │     │
│   │   │ 8259 │  │       │  │        │  │ Controller│        │     │
│   │   └──┬───┘  └───┬───┘  └───┬────┘  └────┬─────┘        │     │
│   └──────┼──────────┼──────────┼────────────┼───────────────┘     │
│          ▼          ▼          ▼            ▼                       │
│                      Kernel Space                                   │
├─────────────────────────────────────────────────────────────────────┤
│                     Hardware Bus (PCIe / AHB / AXI)                 │
│   ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐     │
│   │Keyboard│  │  NIC   │  │  NVMe  │  │  GPU   │  │ Timer  │     │
│   │        │  │        │  │  SSD   │  │        │  │        │     │
│   └────────┘  └────────┘  └────────┘  └────────┘  └────────┘     │
└─────────────────────────────────────────────────────────────────────┘
```

As this diagram shows, when a user-space application issues an I/O request, it passes through the VFS to reach the device layer, and ultimately the communication with hardware is controlled through interrupts and DMA. This chapter dives deep into this entire mechanism.

---

## 1. Fundamental Concepts of Interrupts

### 1.1 Why Are Interrupts Necessary?

There is an enormous speed gap between the CPU and I/O devices.

| Operation | Approximate Cycles | Approximate Time |
|------|---------------------|-----------------|
| CPU register access | 1 cycle | 0.3 ns |
| L1 cache hit | 4 cycles | 1.2 ns |
| L3 cache hit | 40 cycles | 12 ns |
| Main memory access | 200 cycles | 60 ns |
| SSD (NVMe) read | — | 10-100 us |
| HDD seek | — | 3-10 ms |
| Network round-trip (LAN) | — | 0.1-1 ms |
| Network round-trip (WAN) | — | 10-100 ms |

Without interrupts, the CPU would have no choice but to repeatedly check the device readiness through **polling**. With polling, CPU cycles are wasted until the device responds.

```
Polling vs Interrupts:

  [Polling Method]
  CPU: Check → Not ready → Check → Not ready → ... → Ready → Process
       ^^^^^^^^   ^^^^^^^^   ^^^^^^^^
       Wasted CPU cycles (busy wait)

  [Interrupt Method]
  CPU: Issue I/O request → Execute other tasks → ... → Receive interrupt → Process
                           ^^^^^^^^^^^^^^^^^^^^^
                           CPU used effectively
```

However, interrupts are not a universal solution. In high-frequency I/O scenarios (such as packet reception on a 10Gbps NIC), the overhead of interrupts themselves can become a bottleneck. This problem is addressed by polling mode (NAPI) and hybrid approaches discussed later.

### 1.2 Three Major Categories of Interrupts

Interrupts are classified into three types based on their source and nature.

```
┌──────────────────────────────────────────────────────────────┐
│                    Interrupt Classification                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │ 1. Hardware Interrupts (External Interrupts) │            │
│  │    Source: External devices                   │            │
│  │    ┌────────────────┬───────────────────┐    │            │
│  │    │ Maskable       │ Non-Maskable      │    │            │
│  │    │ (INTR)         │ (NMI)             │    │            │
│  │    │ Can be disabled│ Cannot be ignored │    │            │
│  │    │ with CLI       │ Memory parity     │    │            │
│  │    │ instruction    │ errors, watchdog  │    │            │
│  │    │ Keyboard,      │ timer,            │    │            │
│  │    │ disk complete, │ hardware failures │    │            │
│  │    │ NIC receive    │                   │    │            │
│  │    └────────────────┴───────────────────┘    │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │ 2. Software Interrupts (Traps)               │            │
│  │    Source: Program instructions               │            │
│  │    - INT n instruction (x86: int 0x80)        │            │
│  │    - SYSCALL / SYSENTER instruction           │            │
│  │    - SVC instruction (ARM)                    │            │
│  │    - Debug breakpoint (INT 3)                 │            │
│  └─────────────────────────────────────────────┘            │
│                                                              │
│  ┌─────────────────────────────────────────────┐            │
│  │ 3. Exceptions                                │            │
│  │    Source: CPU internal                       │            │
│  │    ┌──────────┬──────────┬──────────┐        │            │
│  │    │ Fault    │ Trap     │ Abort    │        │            │
│  │    │ Re-      │ Next     │ Non-     │        │            │
│  │    │ executable│ instruc-│ recover- │        │            │
│  │    │ Page     │ tion     │ able     │        │            │
│  │    │ fault    │ Overflow │ Double   │        │            │
│  │    │ GPF      │ Debug    │ fault    │        │            │
│  │    │          │ trap     │ Machine  │        │            │
│  │    │          │          │ check    │        │            │
│  │    └──────────┴──────────┴──────────┘        │            │
│  └─────────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────┘
```

#### Detailed Sub-classification of Exceptions

| Category | Return Address | Recoverability | Examples | x86 Vector Number |
|------|--------|-----------|--------|--------------|
| Fault | Faulting instruction | Re-executable | Page Fault (#PF) | 14 |
| Fault | Faulting instruction | Re-executable | General Protection Fault (#GP) | 13 |
| Fault | Faulting instruction | Re-executable | Division by Zero (#DE) | 0 |
| Trap | Next instruction | Continuable | Breakpoint (#BP) | 3 |
| Trap | Next instruction | Continuable | Overflow (#OF) | 4 |
| Abort | None | Non-recoverable | Double Fault (#DF) | 8 |
| Abort | None | Non-recoverable | Machine Check (#MC) | 18 |

### 1.3 x86/x86-64 Interrupt Vector Table

In the x86 architecture, interrupts are dispatched to handlers through the **IDT (Interrupt Descriptor Table)**. The IDT can hold up to 256 entries, each defining a handler address and its attributes.

```
x86-64 IDT Entry Structure (16 bytes / Gate Descriptor):

  Bit Position    Field
  ┌───────────────────────────────────────┐
  │ 127:96  Reserved (upper DWORD)       │
  │  95:64  Offset[63:32]                 │
  │  63:48  Offset[31:16]                 │
  │  47     P (Present bit)               │
  │  46:45  DPL (Privilege Level 0-3)     │
  │  44     0 (fixed)                     │
  │  43:40  Gate Type                     │
  │         0xE = Interrupt Gate           │
  │         0xF = Trap Gate                │
  │  39:35  Reserved                      │
  │  34:32  IST (Interrupt Stack Table)   │
  │  31:16  Segment Selector              │
  │  15:0   Offset[15:0]                  │
  └───────────────────────────────────────┘

  IDTR Register:
  ┌──────────────────────────────┐
  │ Base Address (64bit) │ Limit │  ← Loaded with LIDT instruction
  └──────────────────────────────┘
```

**Difference between Interrupt Gate and Trap Gate:**
- Interrupt Gate: Automatically clears the IF flag (Interrupt Flag) when the handler executes. Subsequent interrupts are disabled.
- Trap Gate: Does not modify the IF flag. Interrupts are still accepted during handler execution.

Due to this difference, hardware interrupt handlers typically use Interrupt Gates, while software interrupts (system calls) generally use Trap Gates.

---

## 2. Detailed Flow of Interrupt Processing

### 2.1 From Hardware Interrupt Generation to Return

```
 Device          PIC/APIC           CPU                   Memory
   │                │                │                      │
   │─IRQ signal────→│                │                      │
   │                │─INTR signal──→│                      │
   │                │                │                      │
   │                │                │◆ Complete current    │
   │                │                │  instruction         │
   │                │                │                      │
   │                │                │◆ PUSH RFLAGS, CS,   │
   │                │                │  RIP to stack ──────→│
   │                │                │                      │
   │                │←INTA(ack)─────│                      │
   │                │                │                      │
   │                │─Vector num───→│                      │
   │                │                │                      │
   │                │                │◆ Get handler from   │
   │                │                │  IDT[vector num] ←──│
   │                │                │                      │
   │                │                │◆ Check privilege     │
   │                │                │  level. If Ring3→    │
   │                │                │  Ring0, load RSP0    │
   │                │                │  from TSS            │
   │                │                │                      │
   │                │                │◆ Begin handler       │
   │                │                │  execution           │
   │                │                │  (top half)          │
   │                │                │                      │
   │                │                │◆ EOI (End of        │
   │                │←Send EOI──────│  Interrupt)          │
   │                │                │                      │
   │                │                │◆ Return via IRET    │
   │                │                │  POP RIP, CS, RFLAGS│
   │                │                │  and resume original │
   │                │                │  execution           │
   │                │                │                      │
```

### 2.2 Code Example: Skeleton of an x86-64 Interrupt Handler (C + Inline Assembly)

Below is a schematic code example of Linux kernel-style interrupt handler registration and implementation.

```c
/* Code Example 1: Basic structure of an x86-64 interrupt handler */

#include <linux/interrupt.h>
#include <linux/module.h>

#define MY_DEVICE_IRQ  11

/* Interrupt context information */
struct my_device_data {
    unsigned long irq_count;
    spinlock_t    lock;
    void __iomem *base_addr;
    /* Device-specific ring buffer, etc. */
};

/*
 * Top half: Main interrupt handler
 * - Executes with interrupts disabled
 * - Performs only minimal processing
 * - Sleep is prohibited (only GFP_ATOMIC may be used)
 */
static irqreturn_t my_device_isr(int irq, void *dev_id)
{
    struct my_device_data *data = dev_id;
    u32 status;

    /* Read the device's interrupt status register */
    status = ioread32(data->base_addr + STATUS_REG);

    /* Check if this is our device's interrupt (shared IRQ support) */
    if (!(status & MY_DEVICE_IRQ_PENDING))
        return IRQ_NONE;  /* Interrupt from another device */

    /* Clear the device-side interrupt (ACK) */
    iowrite32(status, data->base_addr + STATUS_REG);

    spin_lock(&data->lock);
    data->irq_count++;
    spin_unlock(&data->lock);

    /* Schedule the bottom half */
    tasklet_schedule(&my_device_tasklet);

    return IRQ_HANDLED;
}

/*
 * Bottom half: Deferred processing
 * - Executes with interrupts enabled
 * - Can perform relatively lengthy processing
 */
static void my_device_tasklet_fn(unsigned long arg)
{
    struct my_device_data *data = (struct my_device_data *)arg;

    /* Process received data, buffer copies, etc. */
    process_received_data(data);

    /* Notify user space */
    wake_up_interruptible(&data->wait_queue);
}

DECLARE_TASKLET(my_device_tasklet, my_device_tasklet_fn, 0);

/*
 * Register IRQ during device initialization
 */
static int my_device_probe(struct pci_dev *pdev,
                           const struct pci_device_id *id)
{
    struct my_device_data *data;
    int ret;

    data = devm_kzalloc(&pdev->dev, sizeof(*data), GFP_KERNEL);
    if (!data)
        return -ENOMEM;

    spin_lock_init(&data->lock);

    /*
     * request_irq() flags:
     *   IRQF_SHARED   - Can share IRQ with other devices
     *   IRQF_ONESHOT  - One-shot for threaded IRQ
     */
    ret = request_irq(pdev->irq, my_device_isr,
                      IRQF_SHARED, "my_device", data);
    if (ret) {
        dev_err(&pdev->dev, "Failed to register IRQ %d: %d\n",
                pdev->irq, ret);
        return ret;
    }

    dev_info(&pdev->dev, "Registered IRQ %d\n", pdev->irq);
    return 0;
}

/*
 * Release IRQ during device removal
 */
static void my_device_remove(struct pci_dev *pdev)
{
    struct my_device_data *data = pci_get_drvdata(pdev);
    free_irq(pdev->irq, data);
}
```

### 2.3 Evolution of Interrupt Controllers

```
┌─────────────────────────────────────────────────────────────────┐
│           Generational Evolution of Interrupt Controllers        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Generation 1: 8259A PIC (1980s)                                │
│  ┌─────────────────────────────────────────┐                   │
│  │  Master 8259A ──── Slave 8259A          │                   │
│  │  IRQ0: Timer       IRQ8:  RTC           │                   │
│  │  IRQ1: Keyboard    IRQ9:  Redirect      │                   │
│  │  IRQ2: → Cascade   IRQ10: (free)        │                   │
│  │  IRQ3: COM2        IRQ11: (free)        │                   │
│  │  IRQ4: COM1        IRQ12: PS/2 Mouse    │                   │
│  │  IRQ5: LPT2/Sound  IRQ13: FPU          │                   │
│  │  IRQ6: Floppy      IRQ14: Primary IDE   │                   │
│  │  IRQ7: LPT1        IRQ15: Secondary IDE │                   │
│  │                                          │                   │
│  │  Limitations: Max 15 IRQ lines,          │                   │
│  │  fixed priority                          │                   │
│  └─────────────────────────────────────────┘                   │
│                          ↓                                      │
│  Generation 2: APIC (1990s - present)                           │
│  ┌─────────────────────────────────────────┐                   │
│  │  I/O APIC ←→ System Bus ←→ Local APIC   │                   │
│  │                             (one per CPU)│                   │
│  │                                          │                   │
│  │  Improvements:                           │                   │
│  │  - 224 IRQ vectors (32-255)             │                   │
│  │  - Multiprocessor support                │                   │
│  │  - Programmable priority                 │                   │
│  │  - Interrupt delivery to specific CPUs  │                   │
│  └─────────────────────────────────────────┘                   │
│                          ↓                                      │
│  Generation 3: MSI / MSI-X (2000s - present)                    │
│  ┌─────────────────────────────────────────┐                   │
│  │  Device notifies interrupt via memory    │                   │
│  │  write                                   │                   │
│  │  - No dedicated IRQ pins needed          │                   │
│  │  - MSI: Up to 32 vectors                 │                   │
│  │  - MSI-X: Up to 2048 vectors             │                   │
│  │  - Standard for PCIe devices              │                   │
│  │  - Required for NVMe, high-speed NICs    │                   │
│  └─────────────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Linux Kernel Top-Half/Bottom-Half Design

### 3.1 Design Principles

During execution of an interrupt handler (top half), interrupts on the same IRQ line are disabled. This means that if a handler runs for an extended time, subsequent interrupts can be lost or overall system responsiveness degrades.

To solve this problem, the Linux kernel separates interrupt processing into two stages.

| Property | Top Half | Bottom Half |
|------|------------|------------|
| Execution context | Interrupt context | softirq/tasklet: interrupt context, workqueue: process context |
| Interrupt state | Same IRQ disabled | Interrupts enabled |
| Sleep allowed | No | Only in workqueue |
| Memory allocation | GFP_ATOMIC only | GFP_KERNEL allowed in workqueue |
| Execution timing | Immediate | Deferred (though softirq is fast) |
| Typical processing | Send ACK, set flags, schedule bottom half | Data copy, protocol processing, user notification |

### 3.2 Comparison of Three Bottom-Half Mechanisms

```
 ┌──────────────────────────────────────────────────────────────┐
 │       Comparison of Bottom-Half Execution Mechanisms          │
 │                                                              │
 │  ┌──────────┐    ┌──────────┐    ┌──────────────┐           │
 │  │ softirq  │    │ tasklet  │    │  workqueue   │           │
 │  │          │    │          │    │              │           │
 │  │ Statically│    │ Can be   │    │ Runs on     │           │
 │  │ defined  │    │ created  │    │ kernel      │           │
 │  │ (10 types)│    │ dynamic- │    │ thread      │           │
 │  │          │    │ ally     │    │              │           │
 │  │ Can run  │    │ Same     │    │ Sleep       │           │
 │  │ on multi-│    │ tasklet  │    │ allowed     │           │
 │  │ ple CPUs │    │ is       │    │ mutex       │           │
 │  │ simultan-│    │ serialized│    │ usable      │           │
 │  │ eously   │    │          │    │              │           │
 │  │ High perf│    │ Middle   │    │ High        │           │
 │  │ NET_RX   │    │ ground   │    │ flexibility │           │
 │  │ etc.     │    │          │    │ USB etc.    │           │
 │  └──────────┘    └──────────┘    └──────────────┘           │
 │                                                              │
 │  Performance: softirq > tasklet >> workqueue                 │
 │  Flexibility: workqueue > tasklet > softirq                  │
 │  Complexity:  softirq > tasklet > workqueue                  │
 └──────────────────────────────────────────────────────────────┘
```

### 3.3 Code Example: Using Threaded IRQ

Since Linux 2.6.30, `request_threaded_irq()` allows executing the bottom half as a kernel thread.

```c
/* Code Example 2: Handler separation using threaded IRQ */

#include <linux/interrupt.h>

/*
 * Hard IRQ handler (top half)
 * Minimal processing only -- device ACK and determination
 */
static irqreturn_t my_hard_irq(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;
    u32 status = ioread32(dev->regs + IRQ_STATUS);

    if (!(status & DEVICE_IRQ_FLAG))
        return IRQ_NONE;

    /* Acknowledge the device interrupt */
    iowrite32(status, dev->regs + IRQ_ACK);

    /* Request threaded handler execution */
    return IRQ_WAKE_THREAD;
}

/*
 * Thread handler (bottom half)
 * Runs in process context -- sleep is allowed
 */
static irqreturn_t my_thread_fn(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;

    /* Heavy processing can be done here */
    mutex_lock(&dev->data_mutex);

    /* Process DMA completion data */
    process_dma_buffer(dev);

    /* I2C/SPI communication (may involve sleeping) */
    update_device_config(dev);

    mutex_unlock(&dev->data_mutex);

    return IRQ_HANDLED;
}

static int my_device_init(struct platform_device *pdev)
{
    struct my_device *dev = platform_get_drvdata(pdev);
    int irq = platform_get_irq(pdev, 0);

    /*
     * request_threaded_irq():
     *   2nd arg: Hard IRQ handler (can be NULL -> defaults to IRQ_WAKE_THREAD)
     *   3rd arg: Thread handler
     *   IRQF_ONESHOT: Do not re-enable IRQ until thread handler completes
     */
    return request_threaded_irq(irq,
                                my_hard_irq,
                                my_thread_fn,
                                IRQF_ONESHOT | IRQF_SHARED,
                                "my_device",
                                dev);
}
```

### 3.4 Linux softirq List

The Linux kernel statically defines the following 10 types of softirq (in priority order).

| Number | Name | Purpose |
|------|------|------|
| 0 | HI_SOFTIRQ | High-priority tasklet |
| 1 | TIMER_SOFTIRQ | Timer callbacks |
| 2 | NET_TX_SOFTIRQ | Network transmit |
| 3 | NET_RX_SOFTIRQ | Network receive |
| 4 | BLOCK_SOFTIRQ | Block I/O completion |
| 5 | IRQ_POLL_SOFTIRQ | IRQ polling |
| 6 | TASKLET_SOFTIRQ | Normal-priority tasklet |
| 7 | SCHED_SOFTIRQ | Scheduler load balancing |
| 8 | HRTIMER_SOFTIRQ | High-resolution timer |
| 9 | RCU_SOFTIRQ | RCU processing |

---

## 4. DMA (Direct Memory Access) Mechanism

### 4.1 Basic Principles of DMA

DMA is a technology that transfers data directly between devices and main memory without CPU intervention. The CPU only needs to set the transfer parameters (source address, destination address, transfer size) on the DMA controller; the actual data transfer is performed by the DMA controller.

```
Complete DMA Transfer Sequence:

  CPU                 DMA Controller        Device           Memory
   │                      │                   │                │
   │ (1) Allocate DMA     │                   │                │
   │     buffer            │                   │                │
   │────────────────────────────────────────────────────────→│
   │                      │                   │                │
   │ (2) Set transfer     │                   │                │
   │     parameters        │                   │                │
   │  src_addr = 0xFE000  │                   │                │
   │  dst_addr = 0x80000  │                   │                │
   │  length   = 4096     │                   │                │
   │  direction = DEV→MEM │                   │                │
   │─────────────────────→│                   │                │
   │                      │                   │                │
   │ (3) Start DMA        │                   │                │
   │─────────────────────→│                   │                │
   │                      │                   │                │
   │ (4) CPU moves to     │                   │                │
   │     another task     │                   │                │
   │  ...                 │                   │                │
   │                      │ (5) Acquires bus  │                │
   │                      │  as bus master    │                │
   │                      │                   │                │
   │                      │←─ Read data ─────│                │
   │                      │                   │                │
   │                      │── Write to mem ──────────────────→│
   │                      │                   │                │
   │                      │  (6) Repeat for   │                │
   │                      │  transfer size    │                │
   │                      │                   │                │
   │                      │ (7) Transfer      │                │
   │                      │     complete       │                │
   │←── Interrupt ────────│                   │                │
   │                      │                   │                │
   │ (8) Completion       │                   │                │
   │  processing           │                   │                │
   │  (buffer release etc.)│                   │                │
   │                      │                   │                │
```

### 4.2 Types of DMA Mapping

The Linux kernel provides several DMA memory mapping methods for different use cases.

| Method | API | Use Case | Characteristics |
|------|-----|------|------|
| Coherent mapping | `dma_alloc_coherent()` | Buffers frequently accessed by both device and CPU | Cache coherency is automatically maintained. Suitable for ring buffers, DMA descriptors |
| Streaming mapping | `dma_map_single()` | Unidirectional temporary transfer | CPU-side cache management required. High performance but API usage requires care |
| Scatter/gather | `dma_map_sg()` | Transfer of non-contiguous memory regions | Physically non-contiguous pages can be transferred in a single DMA operation. Optimal for network buffers |

### 4.3 Code Example: Using the Linux DMA API

```c
/* Code Example 3: Allocating and using a DMA coherent buffer */

#include <linux/dma-mapping.h>
#include <linux/pci.h>

struct my_dma_device {
    struct pci_dev    *pdev;
    void              *dma_buf_virt;   /* CPU-side virtual address */
    dma_addr_t         dma_buf_phys;   /* Device-side DMA address */
    size_t             buf_size;
};

static int setup_dma_buffer(struct my_dma_device *dev)
{
    /* Set DMA mask -- for a 32-bit DMA capable device */
    if (dma_set_mask_and_coherent(&dev->pdev->dev, DMA_BIT_MASK(32))) {
        dev_err(&dev->pdev->dev, "32-bit DMA not supported\n");
        return -EIO;
    }

    dev->buf_size = PAGE_SIZE * 4;  /* 16KB */

    /*
     * dma_alloc_coherent():
     *   - Allocates a buffer accessible from both CPU and device
     *   - Cache coherency is automatically maintained
     *   - Return value: CPU-side virtual address
     *   - dma_buf_phys: DMA address used by the device
     */
    dev->dma_buf_virt = dma_alloc_coherent(&dev->pdev->dev,
                                            dev->buf_size,
                                            &dev->dma_buf_phys,
                                            GFP_KERNEL);
    if (!dev->dma_buf_virt) {
        dev_err(&dev->pdev->dev, "Failed to allocate DMA buffer\n");
        return -ENOMEM;
    }

    dev_info(&dev->pdev->dev,
             "DMA buffer allocated: virt=%p phys=%pad size=%zu\n",
             dev->dma_buf_virt, &dev->dma_buf_phys, dev->buf_size);

    return 0;
}

static void cleanup_dma_buffer(struct my_dma_device *dev)
{
    if (dev->dma_buf_virt) {
        dma_free_coherent(&dev->pdev->dev,
                          dev->buf_size,
                          dev->dma_buf_virt,
                          dev->dma_buf_phys);
        dev->dma_buf_virt = NULL;
    }
}

/*
 * Streaming DMA usage example:
 * Unidirectional transfer (device -> memory)
 */
static int start_dma_read(struct my_dma_device *dev,
                           void *buffer, size_t len)
{
    dma_addr_t dma_handle;

    /*
     * dma_map_single():
     *   - DMA maps an existing kernel buffer
     *   - DMA_FROM_DEVICE: Direction where device writes to memory
     */
    dma_handle = dma_map_single(&dev->pdev->dev,
                                 buffer, len,
                                 DMA_FROM_DEVICE);

    if (dma_mapping_error(&dev->pdev->dev, dma_handle)) {
        dev_err(&dev->pdev->dev, "DMA mapping failed\n");
        return -EIO;
    }

    /* Set DMA address and length on the device */
    iowrite32(lower_32_bits(dma_handle),
              dev->regs + DMA_SRC_ADDR_LO);
    iowrite32(upper_32_bits(dma_handle),
              dev->regs + DMA_SRC_ADDR_HI);
    iowrite32(len, dev->regs + DMA_LENGTH);

    /* Start DMA transfer */
    iowrite32(DMA_START, dev->regs + DMA_CONTROL);

    return 0;
}

/*
 * Cleanup after DMA completion
 * (called from interrupt handler)
 */
static void finish_dma_read(struct my_dma_device *dev,
                             void *buffer, size_t len,
                             dma_addr_t dma_handle)
{
    /*
     * dma_unmap_single():
     *   - Releases the DMA mapping
     *   - Performs CPU cache invalidation
     *   - After this, buffer contents can be read by the CPU
     */
    dma_unmap_single(&dev->pdev->dev,
                      dma_handle, len,
                      DMA_FROM_DEVICE);

    /* Process buffer contents here */
}
```

### 4.4 Scatter/Gather DMA

In network packets and file I/O, data to be transferred is often scattered across physically non-contiguous memory pages. Scatter/gather DMA is a technology that transfers these non-contiguous memory regions in a single DMA operation.

```
Scatter/Gather DMA Concept:

  [Standard DMA -- Requires contiguous memory]

  Physical memory:
  ┌──────┬──────┬──────┬──────┬──────┐
  │ Used │ FREE │ Used │ FREE │ Used │
  └──────┴──────┴──────┴──────┴──────┘
                 ↓
  Must copy to create contiguous region (CPU overhead)

  [Scatter/Gather DMA -- Non-contiguous OK]

  SG List (Scatter/Gather List):
  ┌──────────────────────┐
  │ Entry 0:             │     Physical memory
  │  addr=0x1000 len=512 │──→ ┌──────┐
  │ Entry 1:             │     │ Data │ page A
  │  addr=0x5000 len=256 │──→ ├──────┤
  │ Entry 2:             │     │      │
  │  addr=0x9000 len=768 │──→ │ Data │ page C
  └──────────────────────┘     ├──────┤
                               │      │
    DMA controller processes   │ Data │ page E
    SG list sequentially,      └──────┘
    no CPU intervention needed
```

This technology is particularly effective in the following scenarios:

- **Networking**: When packet headers and payloads are in separate buffers (zero-copy transmit)
- **Storage**: When reading/writing multiple file system blocks at once
- **Virtualization**: When guest OS physical memory is non-contiguous on the host

---

## 5. IOMMU — DMA Address Virtualization and Protection

### 5.1 The Need for IOMMU

DMA can access memory without CPU intervention, which poses a significant security risk. A malicious device (or a buggy driver) performing DMA transfers to arbitrary physical memory addresses could corrupt kernel memory or leak data.

**IOMMU (Input/Output Memory Management Unit)** virtualizes device DMA addresses and restricts access to only permitted memory regions. It serves a role for I/O devices similar to what the MMU does for the CPU.

```
Role of IOMMU:

  CPU side:                              Device side:
  ┌─────┐    ┌─────┐                 ┌────────┐    ┌───────┐
  │ CPU │───→│ MMU │───→Physical Mem←──│ IOMMU  │←───│Device │
  └─────┘    └─────┘                 └────────┘    └───────┘
              │                        │
              ▼                        ▼
         Page tables              I/O page tables
         (virtual→physical)       (DMA addr→physical)

  MMU:   CPU virtual address → physical address translation
  IOMMU: Device DMA address → physical address translation

  ┌──────────────────────────────────────────────┐
  │  Device accesses DMA addr 0x2000             │
  │         ↓                                    │
  │  IOMMU references I/O page table             │
  │         ↓                                    │
  │  [Allowed]  → Translate to physical addr     │
  │               0xA8000 and transfer            │
  │  [Denied]   → DMA Fault → notify kernel      │
  └──────────────────────────────────────────────┘
```

### 5.2 Major Uses of IOMMU

| Use Case | Description |
|------|------|
| DMA protection | Restricts the physical memory regions accessible by a device |
| Device virtualization passthrough | Uses VT-d/AMD-Vi to translate guest physical addresses to host physical addresses when directly assigning devices to VMs (PCI passthrough) |
| DMA remapping | Allows devices limited to 32-bit DMA to access memory beyond 4GB (avoiding bounce buffers) |
| Interrupt remapping | Validates interrupts from devices to prevent malicious interrupt injection |

### 5.3 IOMMU Configuration in Linux

```bash
# Code Example 4: IOMMU-related kernel parameters and verification commands

# Kernel boot parameters (GRUB configuration)
# Enable Intel VT-d
GRUB_CMDLINE_LINUX="intel_iommu=on"

# Enable AMD-Vi
GRUB_CMDLINE_LINUX="amd_iommu=on"

# Check IOMMU groups
# Display which IOMMU group each device belongs to
for d in /sys/kernel/iommu_groups/*/devices/*; do
    n=$(echo "$d" | rev | cut -d/ -f1 | rev)
    g=$(echo "$d" | rev | cut -d/ -f3 | rev)
    echo "IOMMU Group $g: $(lspci -nns "$n")"
done

# Verify IOMMU initialization in dmesg
dmesg | grep -i iommu
# Example output:
# [    0.123456] DMAR: IOMMU enabled
# [    0.234567] DMAR: Intel(R) Virtualization Technology for Directed I/O

# Check interrupt distribution in /proc/interrupts
cat /proc/interrupts | head -20

# Check affinity of a specific IRQ
cat /proc/irq/24/smp_affinity
# Example output: f  (delivered to CPU 0-3)

# Set IRQ affinity (deliver only to CPU 2)
echo 4 > /proc/irq/24/smp_affinity
# Bitmask: 4 = 0100 → CPU 2
```

---

## 6. Modern I/O Technologies

### 6.1 MSI / MSI-X (Message Signaled Interrupts)

In traditional interrupt methods, devices used dedicated IRQ pins (physical wiring) to notify the CPU of interrupts. With MSI/MSI-X, devices notify interrupts by writing data to a specific memory address.

```
Traditional method vs MSI/MSI-X:

  [Traditional (Pin-based interrupts)]
  ┌────────┐  IRQ pin  ┌────────┐  INTR  ┌─────┐
  │ Device ├──────────→│ I/O    ├───────→│ CPU │
  │   A    │           │ APIC   │        │     │
  └────────┘           │        │        └─────┘
  ┌────────┐  IRQ pin  │        │
  │ Device ├──────────→│        │  Problems:
  │   B    │           └────────┘  - Limited pins (24)
  └────────┘                       - Performance loss with shared IRQs
                                   - Difficult to dynamically change
                                     delivery targets

  [MSI-X]
  ┌────────┐                          ┌─────────┐
  │ Device │── Memory write ─────────→│ Local   │
  │   A    │  (addr=0xFEE00xxx,       │ APIC    │
  │        │   data=vector_num)       │ (CPU 0) │
  │  Up to │                          └─────────┘
  │  2048  │── Memory write ─────────→┌─────────┐
  │ vectors│  (different addr/data)   │ Local   │
  └────────┘                          │ APIC    │
                                      │ (CPU 3) │
  Benefits:                           └─────────┘
  - Up to 2048 vectors per device
  - Each vector can be delivered to a different CPU
  - No IRQ sharing needed → faster handlers
  - NVMe: Assigns MSI-X vector per queue
```

### 6.2 Performance Benefits of MSI-X

| Property | Pin-based Interrupts | MSI | MSI-X |
|------|-------------------|-----|-------|
| Vector count | 1 (shared) | Up to 32 | Up to 2048 |
| IRQ sharing | Required | Not required | Not required |
| CPU targeting | Limited | Limited | Free per vector |
| Latency | High | Low | Low |
| Multi-queue support | Not possible | Limited | Full support |
| PCIe compatibility | Legacy | Standard | Recommended |

### 6.3 NVMe (Non-Volatile Memory Express)

NVMe is a PCIe-native storage protocol designed to maximize the performance of SSDs (NAND Flash / 3D XPoint). While the legacy AHCI (Advanced Host Controller Interface) was designed around HDD rotational latency, NVMe efficiently handles massive parallel I/O.

```
AHCI vs NVMe Architecture Comparison:

  [AHCI (SATA-based)]
  ┌──────┐     1 queue              ┌──────────┐
  │ CPU  │────(max 32 commands)───→│ SATA SSD │
  │      │     depth: 32           │          │
  └──────┘                         └──────────┘
  ↑ All commands serialized to 1 queue = bottleneck

  [NVMe (PCIe-based)]
  ┌──────┐  Submission Q 0 ────→┌──────────┐
  │ CPU  │  Submission Q 1 ────→│ NVMe SSD │
  │ Core │  Submission Q 2 ────→│          │
  │  0   │                      │ Internal │
  │      │  Completion Q 0 ←───│ Control- │
  └──────┘  Completion Q 1 ←───│ ler      │
  ┌──────┐  Submission Q 3 ────→│          │
  │ CPU  │  Submission Q 4 ────→│ Flash    │
  │ Core │                      │ Channels │
  │  1   │  Completion Q 2 ←───│ x8-16    │
  └──────┘                      └──────────┘

  Up to 65,535 queues x 65,536 entries/queue
  Dedicated Submission/Completion queue pairs per CPU core
  MSI-X vectors also assigned per queue → lock-free design
```

| Comparison Item | AHCI (SATA) | NVMe |
|----------|-------------|------|
| Queue count | 1 | Up to 65,535 |
| Queue depth | 32 | Up to 65,536 |
| Host interface | SATA (6 Gbps) | PCIe Gen4 x4 (64 Gbps) |
| Interrupt method | Pin-based / MSI | MSI-X (per queue) |
| Command submission | 4 register writes | 1 doorbell register write |
| CPU utilization | High | Low |
| IOPS (4K Random Read) | ~100,000 | ~1,000,000+ |

### 6.4 RDMA (Remote Direct Memory Access)

RDMA is a technology that directly accesses the memory of a remote machine over a network. In conventional network communication, data passes through multiple stages: application -> kernel -> NIC driver -> NIC -> network -> NIC -> NIC driver -> kernel -> application, involving multiple copies and context switches. RDMA achieves this with "zero-copy" and "OS bypass."

```
Conventional Network Communication vs RDMA:

  [Conventional TCP/IP Communication]
  App → [copy] → Kernel TCP/IP Stack → [copy] → NIC Driver → NIC
                     ↓ Many interrupts / context switches
  NIC → NIC Driver → [copy] → Kernel TCP/IP Stack → [copy] → App

  Total of 4 memory copies + multiple context switches

  [RDMA]
  App ──────→ RNIC (RDMA NIC) ──────→ Network
                ↓ Hardware offload         ↓
  Network ──→ RNIC ──────→ Writes directly to app's memory

  Zero-copy, OS bypass, no CPU involvement

  RDMA Verbs:
  ┌─────────────────────────────────────────────┐
  │ RDMA Read  : Read remote memory             │
  │ RDMA Write : Write to remote memory         │
  │ Send/Recv  : Message passing                │
  │ Atomic     : Remote CAS/Fetch-Add           │
  └─────────────────────────────────────────────┘
```

The three major RDMA transport technologies are:

| Technology | Physical Layer | Bandwidth | Latency | Use Case |
|------|--------|--------|-----------|------|
| InfiniBand | Dedicated fabric | 200-400 Gbps (HDR/NDR) | < 1 us | HPC, AI clusters |
| RoCEv2 | Ethernet | 25-400 Gbps | 1-2 us | Data centers |
| iWARP | Ethernet + TCP | 10-100 Gbps | 5-10 us | General purpose |

---

## 7. Interrupt Affinity and Performance Tuning

### 7.1 IRQ Balancing Challenges

On multi-core systems, which CPU processes an interrupt significantly impacts performance. By default, the Linux `irqbalance` daemon distributes interrupts across CPUs, but high-performance workloads may require manual tuning.

```
Interrupt Affinity Design Considerations:

  ┌─────────────────────────────────────────────────────┐
  │           NUMA Node 0          NUMA Node 1          │
  │  ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐          │
  │  │CPU 0 │ │CPU 1 │     │CPU 2 │ │CPU 3 │          │
  │  └──┬───┘ └──┬───┘     └──┬───┘ └──┬───┘          │
  │     │        │            │        │               │
  │  ┌──┴────────┴──┐     ┌──┴────────┴──┐            │
  │  │  L3 Cache    │     │  L3 Cache    │            │
  │  │  Memory      │     │  Memory      │            │
  │  │  Controller  │     │  Controller  │            │
  │  └──────┬───────┘     └──────┬───────┘            │
  │         │                    │                     │
  │  ┌──────┴───────┐     ┌─────┴────────┐            │
  │  │ PCIe Root    │     │ PCIe Root    │            │
  │  │ Complex      │     │ Complex      │            │
  │  └──────┬───────┘     └──────┬───────┘            │
  │         │                    │                     │
  │      ┌──┴──┐             ┌───┴──┐                 │
  │      │ NIC │             │ NVMe │                 │
  │      │eth0 │             │nvme0 │                 │
  │      └─────┘             └──────┘                 │
  └─────────────────────────────────────────────────────┘

  Best Practices:
  - NIC interrupts → Deliver to CPUs on the same NUMA node as the NIC
  - NVMe interrupts → Deliver to CPUs on the same NUMA node as the NVMe
  - Cross-NUMA memory access increases latency
```

### 7.2 Code Example: Interrupt Affinity Configuration Script

```bash
# Code Example 5: Script to manually set NIC interrupt affinity

#!/bin/bash
# nic_irq_affinity.sh — Pin NIC interrupts to NUMA-local CPUs

DEVICE="eth0"
NUMA_NODE=$(cat /sys/class/net/${DEVICE}/device/numa_node)

echo "=== Setting interrupt affinity for ${DEVICE} ==="
echo "NUMA Node: ${NUMA_NODE}"

# Get list of CPUs belonging to the NUMA node
CPULIST=$(cat /sys/devices/system/node/node${NUMA_NODE}/cpulist)
echo "Local CPUs: ${CPULIST}"

# Stop irqbalance (conflicts with manual settings)
systemctl stop irqbalance 2>/dev/null

# Get list of IRQ numbers for the device
IRQS=$(grep "${DEVICE}" /proc/interrupts | awk '{print $1}' | tr -d ':')

CPU_IDX=0
CPUS=($(echo "${CPULIST}" | tr ',' ' ' | tr '-' ' '))

for IRQ in ${IRQS}; do
    # Assign CPUs in round-robin fashion
    TARGET_CPU=${CPUS[$((CPU_IDX % ${#CPUS[@]}))]}

    # Calculate affinity mask
    MASK=$(printf "%x" $((1 << TARGET_CPU)))

    echo "  IRQ ${IRQ} → CPU ${TARGET_CPU} (mask: ${MASK})"
    echo "${MASK}" > /proc/irq/${IRQ}/smp_affinity

    CPU_IDX=$((CPU_IDX + 1))
done

echo "=== Configuration complete ==="

# Verify
echo ""
echo "Current interrupt distribution:"
grep "${DEVICE}" /proc/interrupts
```

### 7.3 NAPI — Network Interrupt Optimization

On high-speed networks (10GbE and above), if an interrupt is generated for every packet, the interrupt overhead itself overwhelms the CPU (**interrupt storm**). Linux NAPI (New API) solves this problem with a hybrid approach combining interrupts and polling.

```
NAPI Operation Flow:

  Packet arrival rate: Low                  High
  ←─────────────────────────────────────────→

  [Interrupt Mode]          [Polling Mode]
  Interrupt per             Disable interrupts and
  packet                    CPU polls NIC periodically

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  (1) Packet arrives → NIC generates interrupt    │
  │         ↓                                        │
  │  (2) Interrupt handler: napi_schedule()          │
  │      → Disable further interrupts                │
  │         ↓                                        │
  │  (3) softirq (NET_RX_SOFTIRQ) starts            │
  │         ↓                                        │
  │  (4) NAPI poll function repeatedly checks        │
  │      NIC's ring buffer (polling)                 │
  │         ↓                                        │
  │  (5) Process up to budget (typically 64 packets) │
  │         ↓                                        │
  │  (6a) More packets remain → return to (4)        │
  │  (6b) Packets exhausted → napi_complete_done()   │
  │       → Re-enable interrupts → return to (1)     │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

This design enables automatic switching: leveraging the low latency of interrupts under light load, and maximizing throughput through polling under heavy load.

---

## 8. Anti-Patterns and Countermeasures

### 8.1 Anti-Pattern 1: Long-Running Processing Inside Interrupt Handlers

```
[Problematic Code]

static irqreturn_t bad_isr(int irq, void *dev_id)
{
    /* DANGEROUS: Acquiring mutex inside interrupt handler */
    mutex_lock(&data->big_lock);         /* Can sleep → BUG */

    /* DANGEROUS: Large data copy operation */
    memcpy(user_buf, dma_buf, 1048576);  /* 1MB copy → long duration */

    /* DANGEROUS: Normal kernel memory allocation */
    buf = kmalloc(65536, GFP_KERNEL);    /* Can sleep → BUG */

    mutex_unlock(&data->big_lock);
    return IRQ_HANDLED;
}
```

**Problems:**
- Sleep is not allowed in interrupt context. `mutex_lock()` and `GFP_KERNEL` may sleep.
- Long processing blocks other interrupts and degrades overall system responsiveness.
- In the worst case, the watchdog timer fires and resets the system.

**Countermeasures:**
- Perform only minimal processing (ACK, flag setting) in the top half.
- Delegate heavy processing to the bottom half (tasklet, workqueue, threaded IRQ).
- Use only `spin_lock()` and `GFP_ATOMIC` in interrupt context.

```
[Corrected Code]

static irqreturn_t good_isr(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;
    u32 status;

    status = ioread32(dev->regs + IRQ_STATUS);
    if (!(status & MY_IRQ_FLAG))
        return IRQ_NONE;

    /* Minimal processing: ACK and flag setting only */
    iowrite32(status, dev->regs + IRQ_ACK);

    spin_lock(&dev->lock);
    dev->pending_status |= status;
    spin_unlock(&dev->lock);

    /* Heavy processing goes to the bottom half */
    return IRQ_WAKE_THREAD;
}

static irqreturn_t good_thread_fn(int irq, void *dev_id)
{
    struct my_device *dev = dev_id;

    /* Process context: mutex, GFP_KERNEL, sleep all allowed */
    mutex_lock(&dev->big_lock);
    memcpy(user_buf, dma_buf, 1048576);
    buf = kmalloc(65536, GFP_KERNEL);
    process_data(dev);
    mutex_unlock(&dev->big_lock);

    return IRQ_HANDLED;
}
```

### 8.2 Anti-Pattern 2: DMA Buffer Cache Coherency Violation

```
[Problematic Code]

/* Reading data received via streaming DMA */
dma_handle = dma_map_single(dev, buffer, len, DMA_FROM_DEVICE);

/* Tell the device to start DMA transfer */
start_device_dma(dev, dma_handle, len);

/* Wait for transfer completion */
wait_for_completion(&dev->dma_done);

/* DANGEROUS: CPU accesses buffer before unmap */
process_data(buffer);    /* Stale data may remain in cache */

/* Unmap too late */
dma_unmap_single(dev, dma_handle, len, DMA_FROM_DEVICE);
```

**Problems:**
- `dma_unmap_single()` performs CPU cache invalidation.
- If the CPU reads the buffer before unmap, it may read stale data remaining in the cache.
- This type of bug only manifests on certain architectures (such as ARM, where cache coherency is weak) and is hard to detect on x86.

**Countermeasures:**
- Always access the buffer after `dma_unmap_single()`.
- For repeated DMA, use `dma_sync_single_for_cpu()` / `dma_sync_single_for_device()`.

```
[Corrected Code]

dma_handle = dma_map_single(dev, buffer, len, DMA_FROM_DEVICE);
start_device_dma(dev, dma_handle, len);
wait_for_completion(&dev->dma_done);

/* Correct order: unmap first, then access */
dma_unmap_single(dev, dma_handle, len, DMA_FROM_DEVICE);
process_data(buffer);    /* Cache invalidated → correct data */

/* Alternatively, sync while maintaining the mapping */
dma_sync_single_for_cpu(dev, dma_handle, len, DMA_FROM_DEVICE);
process_data(buffer);    /* Synced → correct data */

/* Before letting the device DMA again */
dma_sync_single_for_device(dev, dma_handle, len, DMA_FROM_DEVICE);
start_device_dma(dev, dma_handle, len);
```

### 8.3 Anti-Pattern 3: NUMA-Unaware IRQ Placement

```
Problematic configuration:

  NUMA Node 0                   NUMA Node 1
  ┌──────────┐                 ┌──────────┐
  │ CPU 0-7  │                 │ CPU 8-15 │
  │          │                 │          │
  │ Memory   │   QPI/UPI      │ Memory   │
  │ (local)  │←─────────────→│ (local)  │
  └────┬─────┘                 └────┬─────┘
       │                            │
  ┌────┴─────┐                      │
  │ 10GbE   │                      │
  │ NIC     │                      │
  └──────────┘                      │

  Problem: NIC interrupts processed on CPU 8-15 (Node 1)
  → Packet data arrives in Node 0's memory
  → CPU 8-15 accesses Node 0's memory
  → NUMA remote access increases latency by 50-100%
```

**Countermeasures:**
- Set IRQ affinity to CPUs on the same NUMA node as the device.
- Run applications on the same NUMA node (`numactl --cpunodebind=0 --membind=0`).
- Use `irqbalance`'s `--banirq` option to prevent automatic migration of specific IRQs.

---

## 9. Interrupts and DMA in Embedded Systems

### 9.1 ARM Architecture Interrupt Controller (GIC)

ARM processors use the **GIC (Generic Interrupt Controller)** for interrupt management. GICv3/v4 is also used in server-class ARM processors and includes interrupt delivery capabilities for virtualization.

```
ARM GICv3 Architecture:

  ┌─────────────────────────────────────────────────┐
  │                    GIC                           │
  │                                                 │
  │  ┌──────────────┐   ┌───────────────────────┐  │
  │  │ Distributor  │   │   Redistributor       │  │
  │  │              │   │   (CPU interface)      │  │
  │  │ SPI mgmt    │   │   ┌──────┐ ┌──────┐   │  │
  │  │ (shared     │──→│   │CPU 0│ │CPU 1│   │  │
  │  │  interrupts)│   │   │Re-  │ │Re-  │   │  │
  │  │              │   │   │dist │ │dist │   │  │
  │  │ Priority    │   │   └──┬───┘ └──┬───┘   │  │
  │  │ control     │   └──────┼────────┼───────┘  │
  │  │ Affinity    │          │        │          │
  │  └──────┬───────┘   ┌─────┴──┐ ┌───┴────┐     │
  │         │           │CPU Core│ │CPU Core│     │
  │         │           │   0    │ │   1    │     │
  │         │           └────────┘ └────────┘     │
  └─────────┼─────────────────────────────────────┘
            │
  Interrupt Types:
  ┌────────────────────────────────────────────┐
  │ SGI (Software Generated Interrupt)         │
  │   ID 0-15: Inter-CPU communication (IPI)   │
  │                                            │
  │ PPI (Private Peripheral Interrupt)         │
  │   ID 16-31: CPU-local timers, etc.         │
  │                                            │
  │ SPI (Shared Peripheral Interrupt)          │
  │   ID 32-1019: External device interrupts   │
  │                                            │
  │ LPI (Locality-specific Peripheral Int.)    │
  │   ID 8192+: MSI/MSI-X equivalent (GICv3+) │
  └────────────────────────────────────────────┘
```

### 9.2 Interrupt Definition via Device Tree in Embedded Linux

In embedded Linux, hardware configuration is described using Device Trees. Interrupt connection relationships are also defined in the Device Tree.

```dts
/* Interrupt definition example in Device Tree */
/ {
    interrupt-controller@f9010000 {
        compatible = "arm,gic-400";
        #interrupt-cells = <3>;
        interrupt-controller;
        reg = <0xf9010000 0x10000>,
              <0xf9020000 0x20000>;
    };

    /* UART device interrupt definition */
    uart0: serial@e0000000 {
        compatible = "xlnx,xuartps";
        reg = <0xe0000000 0x1000>;
        /*
         * interrupts = <type irq_num flags>;
         *   type:  0 = SPI, 1 = PPI
         *   flags: 1 = edge-rising, 4 = level-high
         */
        interrupts = <0 27 4>;  /* SPI #27, level-high */
        interrupt-parent = <&intc>;
        clocks = <&clkc 23>;
    };

    /* DMA controller definition */
    dma0: dma@f8003000 {
        compatible = "arm,pl330", "arm,primecell";
        reg = <0xf8003000 0x1000>;
        interrupts = <0 13 4>,   /* fault */
                     <0 14 4>,   /* ch0 done */
                     <0 15 4>,   /* ch1 done */
                     <0 16 4>,   /* ch2 done */
                     <0 17 4>;   /* ch3 done */
        #dma-cells = <1>;
    };

    /* Device definition that uses DMA */
    spi0: spi@e0006000 {
        compatible = "xlnx,zynq-spi-r1p6";
        reg = <0xe0006000 0x1000>;
        interrupts = <0 26 4>;
        dmas = <&dma0 0>, <&dma0 1>;
        dma-names = "tx", "rx";
    };
};
```

### 9.3 Interrupt Processing in RTOS

In Real-Time Operating Systems (RTOS), guaranteeing interrupt response time is the top priority. Linux's `PREEMPT_RT` patch and FreeRTOS interrupt management achieve real-time behavior through different approaches.

| Property | Standard Linux | PREEMPT_RT Linux | FreeRTOS |
|------|-----------|-----------------|----------|
| Interrupt latency | Tens of us | Single-digit us | Hundreds of ns - single-digit us |
| Interrupt handlers | Hard IRQ + softirq | Nearly all threaded | ISR + deferred processing tasks |
| Priority inversion prevention | None (normally) | Priority inheritance mutex | Priority inheritance mutex |
| spinlock | Disables interrupts | rt_mutex (preemptible) | Critical section |
| Deterministic behavior | Not guaranteed | Nearly guaranteed | Guaranteed |

---

## 10. Exercises

### 10.1 Beginner: Understanding the Interrupt Processing Flow

**Problem:** Arrange the following steps of interrupt processing in chronological order for the scenario below.

When a user presses the "A" key on the keyboard:

```
Options (rearrange in correct order):
(a) Keyboard controller issues IRQ 1
(b) CPU completes the currently executing instruction
(c) Obtain keyboard interrupt handler address from IDT[33]
(d) PUSH RFLAGS, CS, RIP to kernel stack
(e) I/O APIC sends vector number 33 to CPU
(f) Keyboard handler reads the scan code
(g) Send EOI (End of Interrupt) to Local APIC
(h) Return to interrupted process via IRET instruction
(i) Convert scan code to key code via tasklet/workqueue
(j) Deliver input event to user-space input subsystem
```

**Example Answer:**
Correct order: (a) -> (e) -> (b) -> (d) -> (c) -> (f) -> (schedule i) -> (g) -> (h) -> (execute i) -> (j)

Detailed explanation:
1. **(a)** Keyboard controller generates scan code and issues IRQ 1
2. **(e)** I/O APIC converts IRQ 1 to vector number 33 and delivers it to the target CPU's Local APIC
3. **(b)** CPU completes the currently executing instruction (accepts interrupt at instruction boundary)
4. **(d)** CPU automatically pushes RFLAGS, CS, RIP to the kernel stack (loads RSP0 from TSS if Ring 3 -> Ring 0)
5. **(c)** References IDT[33] to obtain handler address, clears IF flag
6. **(f)** Top half: reads scan code from I/O port 0x60, stores in buffer
7. Schedules the bottom half (tasklet)
8. **(g)** Writes EOI to Local APIC to notify completion of interrupt processing
9. **(h)** IRET instruction pops RIP, CS, RFLAGS and returns to the interrupted process
10. **(i)** softirq/tasklet starts, converts scan code to key code
11. **(j)** Delivered to user space through Linux input subsystem (evdev)

### 10.2 Intermediate: DMA Driver Design

**Problem:** Design pseudocode for a simple DMA driver that meets the following requirements.

Requirements:
- Receive 4KB of data from a PCIe device via DMA
- Use streaming DMA mapping
- Process DMA completion with threaded IRQ
- Include error handling

**Design Points:**

```c
/*
 * Answer skeleton (showing the design approach)
 */

/* 1. Device initialization */
static int my_probe(struct pci_dev *pdev, ...)
{
    /* (a) Enable PCI device */
    pci_enable_device(pdev);

    /* (b) Enable bus mastering (required for DMA) */
    pci_set_master(pdev);

    /* (c) Set DMA mask */
    dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));

    /* (d) Allocate receive buffer */
    buf = kmalloc(4096, GFP_KERNEL);

    /* (e) Register threaded IRQ */
    request_threaded_irq(pdev->irq, hard_isr, thread_fn,
                         IRQF_ONESHOT, "my_dma", dev);
}

/* 2. Start DMA receive */
static int start_receive(struct my_device *dev)
{
    /* (a) Create streaming DMA mapping */
    dev->dma_addr = dma_map_single(&dev->pdev->dev,
                                    dev->buf, 4096,
                                    DMA_FROM_DEVICE);

    /* (b) Check mapping error -- mandatory! */
    if (dma_mapping_error(&dev->pdev->dev, dev->dma_addr))
        return -EIO;

    /* (c) Set DMA address and size on the device */
    iowrite32(lower_32_bits(dev->dma_addr), dev->regs + DMA_ADDR_LO);
    iowrite32(upper_32_bits(dev->dma_addr), dev->regs + DMA_ADDR_HI);
    iowrite32(4096, dev->regs + DMA_SIZE);

    /* (d) Start DMA */
    iowrite32(DMA_START_BIT, dev->regs + DMA_CTRL);
    return 0;
}

/* 3. Top half: ACK only */
static irqreturn_t hard_isr(int irq, void *dev_id)
{
    u32 status = ioread32(dev->regs + IRQ_STATUS);
    if (!(status & DMA_COMPLETE_BIT))
        return IRQ_NONE;
    iowrite32(status, dev->regs + IRQ_ACK);
    return IRQ_WAKE_THREAD;
}

/* 4. Bottom half: Data processing */
static irqreturn_t thread_fn(int irq, void *dev_id)
{
    /* (a) Release DMA mapping → cache invalidation */
    dma_unmap_single(&dev->pdev->dev,
                      dev->dma_addr, 4096,
                      DMA_FROM_DEVICE);

    /* (b) Process data (sleep allowed here) */
    process_received_data(dev->buf, 4096);

    /* (c) Start next receive */
    start_receive(dev);

    return IRQ_HANDLED;
}
```

### 10.3 Advanced: Interrupt Load Analysis and Optimization

**Problem:** Given the following `perf` output from a server equipped with a 10Gbps NIC, identify the problem and propose an optimization strategy.

```
# perf top -C 0 output (CPU 0 only)

  45.3%  [kernel]  [k] native_queued_spin_lock_slowpath
  12.8%  [kernel]  [k] _raw_spin_lock
   8.2%  [kernel]  [k] net_rx_action
   6.5%  [kernel]  [k] mlx5e_napi_poll
   4.1%  [kernel]  [k] __netif_receive_skb_core
   3.7%  [kernel]  [k] ip_rcv
   2.9%  [kernel]  [k] tcp_v4_rcv

# /proc/interrupts (excerpt)
            CPU0       CPU1       CPU2       CPU3
  98:  125847293          0          0          0  IR-PCI-MSI  mlx5_comp0
  99:  118234567          0          0          0  IR-PCI-MSI  mlx5_comp1
 100:          0          0          0          0  IR-PCI-MSI  mlx5_comp2
 101:          0          0          0          0  IR-PCI-MSI  mlx5_comp3
```

**Analysis and Solution Direction:**

1. **Problem Identification:**
   - All NIC interrupts are concentrated on CPU 0 (mlx5_comp0, comp1 only on CPU 0)
   - `native_queued_spin_lock_slowpath` at 45.3% -> severe lock contention
   - Only 2 out of 4 NIC queues are active, both pinned to CPU 0

2. **Optimization Strategy:**
   - **(a) Redistribute IRQ affinity:** Spread each queue's interrupts to different CPUs
   - **(b) Enable RPS (Receive Packet Steering):** Distribute packet processing across CPUs at the software level
   - **(c) Configure XPS (Transmit Packet Steering):** Distribute the transmit side across CPUs as well
   - **(d) Check NIC queue count:** Verify available queues with `ethtool -l eth0` and enable all 4 queues
   - **(e) Verify NUMA affinity:** Confirm that CPUs from the NUMA node where the NIC is connected are being used

```bash
# Optimization command examples
# Distribute all NIC queue IRQs to different CPUs
echo 1 > /proc/irq/98/smp_affinity   # CPU 0
echo 2 > /proc/irq/99/smp_affinity   # CPU 1
echo 4 > /proc/irq/100/smp_affinity  # CPU 2
echo 8 > /proc/irq/101/smp_affinity  # CPU 3

# Enable RPS (distribute to CPU 0-3)
echo f > /sys/class/net/eth0/queues/rx-0/rps_cpus
echo f > /sys/class/net/eth0/queues/rx-1/rps_cpus
```

---

## 11. FAQ (Frequently Asked Questions)

### Q1: Is it safe to use `printk()` inside an interrupt handler?

**A:** `printk()` itself can be called from interrupt context. Internally, it only writes to a lock-free ring buffer in the kernel, so it does not sleep. However, the following points require attention:

- Calling `printk()` in high-frequency interrupts can cause ring buffer overflow and console output overhead, significantly degrading performance.
- For debugging purposes, use `printk_ratelimited()` or `dev_dbg_ratelimited()` to limit output frequency.
- In production code, remove `printk()` from the top half and use statistical counters or tracepoints instead.

### Q2: Should I use `request_irq()` or `request_threaded_irq()`?

**A:** General guidelines are as follows:

| Scenario | Recommended API | Reason |
|--------|---------|------|
| Very short processing (register reads/writes only) | `request_irq()` | Minimal overhead |
| Devices on buses that involve sleeping (I2C/SPI) | `request_threaded_irq()` | Sleep is needed in the handler |
| Moderate processing volume | `request_threaded_irq()` | Good balance of flexibility and safety |
| High-performance networking/storage | `request_irq()` + NAPI/softirq | Latency minimization required |

Since Linux 4.x, `request_threaded_irq()` is recommended for new drivers. This also automatically ensures compatibility with RT (PREEMPT_RT) kernels.

### Q3: Is there a size limit for DMA buffers?

**A:** Technically, the DMA upper limit is determined by the architecture's address space, but practical constraints include:

- **Coherent DMA (`dma_alloc_coherent()`):** Depends on contiguous physical memory allocation in the kernel, so the typical limit is a few MB. Larger blocks can be allocated immediately after boot, but may fail as memory fragmentation progresses. Using CMA (Contiguous Memory Allocator) enables allocations up to several hundred MB.
- **Streaming DMA (`dma_map_single()`):** Since it only maps an already-allocated buffer, size constraints are lenient as long as the buffer itself can be allocated.
- **Scatter/Gather DMA:** Can use non-contiguous pages, so it is not constrained by physically contiguous memory. Large transfers are possible depending on the number of SG list entries.

For large DMA requirements, increase the CMA reserved area or use `dma_alloc_attrs()` with `DMA_ATTR_FORCE_CONTIGUOUS`.

### Q4: Can interrupts be lost (lost interrupt)?

**A:** This can theoretically occur with edge-triggered interrupts.

- **Edge-triggered:** Detects the interrupt on the signal's rising edge (Low -> High). If a second interrupt occurs while the handler is running and the signal returns to Low before the handler completes, the second interrupt may be lost.
- **Level-triggered:** Continuously signals the interrupt as long as the signal remains High. Since the signal is maintained until the device explicitly clears it, interrupt loss is unlikely.

PCIe MSI/MSI-X is memory-write-based and depends on bus reliability. Normally interrupts are not lost, but it can happen in extreme situations such as IOMMU failures. A robust design includes polling the device state periodically with a watchdog timer on the driver side.

### Q5: Can interrupts be handled from user space?

**A:** This is possible using Linux's **UIO (Userspace I/O)** framework and **VFIO (Virtual Function I/O)**.

- **UIO:** By performing `read()` / `select()` on `/dev/uioX`, interrupt occurrence can be waited for in user space. High-performance packet processing frameworks like DPDK use this approach.
- **VFIO:** Exposes devices to user space with IOMMU memory protection. Also used for device passthrough in virtualization environments.

However, interrupt "detection" is performed within the kernel, and it is communicated to user space as event notifications. It is important to note that this is not purely kernel-bypass hardware interrupt processing.

---


## FAQ

### Q1: What is the most important point when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge used in practice?

This topic's knowledge is frequently applied in everyday development work. It becomes especially important during code reviews and architecture design.

---

## 12. Summary — Key Concept Overview

### Complete Technology Map

| Concept | Essence | Key Points |
|------|------|-------------|
| Hardware interrupts | Asynchronous notifications from devices to CPU | Maskable/non-maskable distinction, dispatch via IDT |
| Software interrupts | Intentional interrupts from programs | System calls (syscall/int 0x80), debug traps |
| Exceptions | Errors generated internally by the CPU | Fault (re-executable) / Trap (next instruction) / Abort (non-recoverable) |
| Top half | Immediate interrupt processing | Minimal processing, no sleep, fast |
| Bottom half | Deferred interrupt processing | softirq / tasklet / workqueue / threaded IRQ |
| DMA | Data transfer without CPU intervention | Coherent / streaming / scatter-gather |
| IOMMU | DMA address virtualization/protection | VT-d / AMD-Vi, foundation for virtualization passthrough |
| MSI/MSI-X | Memory-write-based interrupts | PCIe device standard, multi-queue support |
| NVMe | High-speed I/O protocol for SSDs | 65K queues x 65K entries, MSI-X integration |
| RDMA | Direct memory access over network | Zero-copy, OS bypass, for HPC/AI |
| NAPI | Interrupt + polling hybrid | Interrupt storm countermeasure for high-speed NICs |
| IRQ affinity | Interrupt CPU delivery target control | NUMA awareness, basic performance optimization |

### Design Decision Flowchart

```
Selecting the interrupt processing method for a device driver:

  Does interrupt processing
  require sleeping?
      │
      ├── YES → request_threaded_irq()
      │          Use IRQF_ONESHOT
      │
      └── NO ──→ Is processing short?
                     │
                     ├── YES → request_irq() only
                     │          (complete in top half)
                     │
                     └── NO ──→ Is high throughput
                                  needed?
                                    │
                                    ├── YES → request_irq()
                                    │          + softirq/NAPI
                                    │
                                    └── NO ──→ request_irq()
                                                + tasklet

  Selecting the DMA buffer method:

  Do CPU and device
  access simultaneously?
      │
      ├── YES → dma_alloc_coherent()
      │          (ring buffers, descriptors)
      │
      └── NO ──→ Is memory contiguous?
                     │
                     ├── YES → dma_map_single()
                     │          (streaming DMA)
                     │
                     └── NO ──→ dma_map_sg()
                                 (scatter/gather)
```

---

## Recommended Next Guides


---

## References

1. Corbet, J., Rubini, A., & Kroah-Hartman, G. *Linux Device Drivers*, 3rd Edition. O'Reilly Media, 2005. — The standard reference for Linux device drivers. Particularly excellent coverage of interrupt handling (Chapter 10) and DMA (Chapter 15).
2. Love, R. *Linux Kernel Development*, 3rd Edition. Addison-Wesley, 2010. — Chapter 7 "Interrupts and Interrupt Handlers" provides detailed discussion of the top-half/bottom-half design philosophy.
3. Bovet, D. P. & Cesati, M. *Understanding the Linux Kernel*, 3rd Edition. O'Reilly Media, 2005. — Explains the Interrupt Descriptor Table (IDT) and APIC internals from a low-level perspective.
4. Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer's Manual*, Volume 3: System Programming Guide, Chapter 6 "Interrupt and Exception Handling". — The official specification for x86/x86-64 interrupt architecture.
5. ARM Limited. *ARM Generic Interrupt Controller Architecture Specification (GICv3/GICv4)*. — The official GIC specification for embedded Linux developers.
6. NVM Express Workgroup. *NVM Express Base Specification*, Revision 2.0. — The official NVMe protocol specification defining queue structures and interrupt methods.
7. Mellanox Technologies (NVIDIA). *RDMA Aware Networks Programming User Manual*. — A practical guide to RDMA programming covering the InfiniBand/RoCE Verbs API.
