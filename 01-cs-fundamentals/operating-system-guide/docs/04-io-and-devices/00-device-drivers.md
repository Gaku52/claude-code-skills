# Device Drivers

> A device driver is an "interpreter that translates hardware dialects into the common language of the OS."
> Standing between the kernel and hardware, it provides a unified interface,
> enabling application developers to be freed from the complexity specific to each device.

---

## Learning Objectives

- [ ] Understand the role and design philosophy of device drivers
- [ ] Know the differences between character devices, block devices, and network devices
- [ ] Understand the mechanism and lifecycle of Linux kernel modules
- [ ] Compare I/O control methods (polling, interrupts, DMA)
- [ ] Read and comprehend the internal structure of device drivers (file_operations, probe/remove)
- [ ] Explain the advantages and disadvantages of user-space drivers (UIO / VFIO)
- [ ] Avoid typical anti-patterns in driver development
- [ ] Understand hardware description through Device Tree and ACPI


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. Fundamental Concepts of Device Drivers

### 1.1 What Is a Device Driver?

A device driver is a software module that operates within the operating system's kernel (or in a space that cooperates with the kernel) and controls a specific hardware device. It mediates between the abstract interfaces provided by the OS (file operations, network stack, etc.) and the concrete register operations and protocol processing of physical devices.

The fundamental reasons why device drivers are necessary can be summarized in the following three points.

1. **Absorbing Hardware Diversity**: Even for the same functionality (e.g., storage), register layouts, command sets, and timing specifications differ by vendor and model. The driver absorbs these differences and provides a unified API to upper layers.

2. **Managing Privileged Operations**: Direct access to hardware registers and DMA configuration require kernel privilege (Ring 0). The driver safely encapsulates these privileged operations.

3. **Coordinating Resource Sharing**: The driver handles mutual exclusion, buffer management, and priority control when multiple processes access the same device.

```
┌─────────────────────────────────────────────────────────┐
│                     User Space                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  App A    │  │  App B    │  │  App C    │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │ open/read    │ write       │ ioctl              │
├───────┼──────────────┼─────────────┼─────────────────────┤
│       ▼              ▼             ▼     Kernel Space    │
│  ┌─────────────────────────────────────────────┐        │
│  │        System Call Interface                 │        │
│  └──────────────────┬──────────────────────────┘        │
│                     ▼                                    │
│  ┌─────────────────────────────────────────────┐        │
│  │     VFS (Virtual File System)                │        │
│  └──────┬───────────┬──────────────┬───────────┘        │
│         ▼           ▼              ▼                     │
│  ┌──────────┐ ┌──────────┐  ┌──────────────┐           │
│  │char drv  │ │blk drv   │  │net drv       │           │
│  │(tty,input)│ │(SCSI,NVMe)│ │(e1000,iwlwifi)│          │
│  └────┬─────┘ └────┬─────┘  └──────┬───────┘           │
├───────┼─────────────┼───────────────┼────────────────────┤
│       ▼             ▼               ▼   Hardware         │
│  ┌──────────┐ ┌──────────┐  ┌──────────────┐           │
│  │Keyboard  │ │SSD/HDD   │  │NIC           │           │
│  │Mouse     │ │NVMe      │  │Wi-Fi Adapter │           │
│  └──────────┘ └──────────┘  └──────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### 1.2 Device Classification System

The Linux kernel classifies devices into three major categories. This classification inherits the UNIX tradition while being extended to accommodate modern hardware requirements.

#### Character Device

A device accessed sequentially as a byte stream. The data unit is bytes, and it fundamentally performs sequential read/write operations. It appears as a device file under the `/dev` directory and is identified by a combination of major and minor numbers.

Representative examples:
- `/dev/tty*` -- Terminal devices
- `/dev/input/event*` -- Input devices (keyboard, mouse)
- `/dev/random`, `/dev/urandom` -- Random number generators
- `/dev/null`, `/dev/zero` -- Special devices
- `/dev/video0` -- Video capture (V4L2)
- `/dev/snd/*` -- Sound devices (ALSA)

#### Block Device

A device that allows random access in fixed-size block units (typically 512 bytes or 4096 bytes). It is accessed through the kernel's Block Layer and benefits from request optimization by the I/O scheduler and buffer cache.

Representative examples:
- `/dev/sda`, `/dev/sdb` -- SCSI/SATA disks
- `/dev/nvme0n1` -- NVMe storage
- `/dev/mmcblk0` -- eMMC/SD cards
- `/dev/loop0` -- Loopback device
- `/dev/dm-0` -- Device Mapper (LVM, encryption)

#### Network Device

A device that sends and receives data in packet units. Unlike the other two types, it is not represented as a file under `/dev` and is accessed through the socket interface. It is managed with the `ip` command or `ifconfig` command.

Representative examples:
- `eth0`, `ens33` -- Wired Ethernet
- `wlan0`, `wlp2s0` -- Wireless LAN
- `lo` -- Loopback interface
- `docker0`, `br0` -- Bridge interfaces

### 1.3 Device Classification Comparison Table

| Property | Character Device | Block Device | Network Device |
|:---------|:-----------------|:-------------|:---------------|
| Access Unit | Byte (stream) | Block (512B/4KB) | Packet (variable length) |
| Access Pattern | Primarily sequential | Random access possible | Packet send/receive |
| `/dev` Entry | Yes (shown as `c`) | Yes (shown as `b`) | None |
| Buffer Cache | None (typically) | Yes (page cache) | SKB (Socket Buffer) |
| I/O Scheduler | None | Yes (mq-deadline, etc.) | None (Qdisc) |
| Registration API | `register_chrdev_region()` | `register_blkdev()` | `register_netdev()` |
| Primary ops Structure | `file_operations` | `block_device_operations` | `net_device_ops` |
| Representative Devices | tty, input, GPU | HDD, SSD, NVMe | Ethernet, Wi-Fi |
| Seek Operation | Not possible (in most cases) | Possible | No concept |

---

## 2. Linux Kernel Module Mechanism

### 2.1 What Is a Kernel Module?

A Linux Kernel Module (Loadable Kernel Module: LKM) is a unit of code that can be dynamically loaded and unloaded without recompiling or rebooting the kernel. Many device drivers are implemented as kernel modules and are loaded into memory only when needed.

This mechanism provides the following benefits.

- The kernel image size can be kept to a minimum
- Appropriate drivers can be automatically loaded when a device is connected (udev integration)
- The driver development and testing cycle is shortened (no reboot required)
- Unused drivers can be unloaded to free memory

### 2.2 Minimal Kernel Module Implementation

The following is the complete source code of the most basic Linux kernel module. As a starting point for kernel module development, it demonstrates that a pair of initialization and exit functions is required.

```c
/* hello_driver.c -- Minimal kernel module */
#include <linux/module.h>    /* MODULE_LICENSE, module_init/exit */
#include <linux/kernel.h>    /* printk, KERN_INFO */
#include <linux/init.h>      /* __init, __exit macros */

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Example Author");
MODULE_DESCRIPTION("Minimal kernel module example");
MODULE_VERSION("1.0");

/*
 * Initialization function called when the module is loaded
 * __init macro: indicates that this function's memory can be freed after initialization completes
 * Return value: 0 = success, negative value = error (negated errno value)
 */
static int __init hello_init(void)
{
    pr_info("hello_driver: module loaded\n");
    pr_info("hello_driver: kernel version = %s\n", UTS_RELEASE);

    /*
     * Device initialization processing would go here:
     *   - Allocate device numbers (alloc_chrdev_region)
     *   - Initialize and register cdev structure (cdev_init, cdev_add)
     *   - Create device class (class_create)
     *   - Create device node (device_create)
     *   - Initialize hardware (register configuration, etc.)
     */

    return 0;  /* Success */
}

/*
 * Exit function called when the module is unloaded
 * __exit macro: indicates this function can be omitted if the module is built-in
 */
static void __exit hello_exit(void)
{
    pr_info("hello_driver: module unloaded\n");

    /*
     * Cleanup processing goes here (reverse order of init):
     *   - device_destroy
     *   - class_destroy
     *   - cdev_del
     *   - unregister_chrdev_region
     */
}

/* Register entry points with the kernel */
module_init(hello_init);
module_exit(hello_exit);
```

The `Makefile` for building is as follows.

```makefile
# Makefile for hello_driver kernel module
obj-m += hello_driver.o

# KDIR: Path to the kernel source tree (usually headers alone are sufficient)
KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

# Load/unload the module (for testing)
load:
	sudo insmod hello_driver.ko

unload:
	sudo rmmod hello_driver

# Check kernel logs
log:
	dmesg | tail -20
```

### 2.3 Module Management Command System

```
┌──────────────────────────────────────────────────────┐
│         Module Management Command Relationships       │
│                                                      │
│   insmod hello.ko          modprobe hello            │
│      │ (single load)           │ (dependency         │
│      ▼                        ▼  resolution + load)  │
│   ┌────────────────────────────────┐                 │
│   │     Kernel Module Loader       │                 │
│   │  ┌───────────────────────┐     │                 │
│   │  │ Module Verification    │     │                 │
│   │  │ (signature check,     │     │                 │
│   │  │  version check)       │     │                 │
│   │  └───────┬───────────────┘     │                 │
│   │          ▼                      │                 │
│   │  ┌───────────────────────┐     │                 │
│   │  │ Symbol Resolution      │     │                 │
│   │  │ (exported symbols of  │     │                 │
│   │  │  dependent modules)   │     │                 │
│   │  └───────┬───────────────┘     │                 │
│   │          ▼                      │                 │
│   │  ┌───────────────────────┐     │                 │
│   │  │ init function call     │     │                 │
│   │  └───────────────────────┘     │                 │
│   └────────────────────────────────┘                 │
│                                                      │
│   rmmod hello              modprobe -r hello         │
│      │ (single unload)       │ (unload with deps)    │
│      ▼                        ▼                      │
│   exit function call -> memory freed                 │
│                                                      │
│   lsmod           -> formatted display of            │
│                      /proc/modules                   │
│   modinfo hello   -> display module metadata         │
│   depmod -a       -> rebuild modules.dep             │
│                      dependency database             │
└──────────────────────────────────────────────────────┘
```

Usage examples of the main commands:

```bash
# List loaded modules and their reference counts
$ lsmod | head -10
Module                  Size  Used by
nvidia_drm             77824  10
nvidia_modeset       1236992  18 nvidia_drm
nvidia              56467456  656 nvidia_modeset
snd_hda_intel          57344  4

# Check detailed module information
$ modinfo e1000e
filename:       /lib/modules/6.1.0/kernel/drivers/net/ethernet/intel/e1000e/e1000e.ko
version:        3.2.6-k
license:        GPL v2
description:    Intel(R) PRO/1000 Network Driver
author:         Intel Corporation
depends:
retpoline:      Y

# Load with automatic dependency resolution
$ sudo modprobe snd_hda_intel

# Load with parameters
$ sudo modprobe usbcore autosuspend=-1

# Check module parameters
$ cat /sys/module/usbcore/parameters/autosuspend

# Unload along with dependent modules
$ sudo modprobe -r snd_hda_intel
```

### 2.4 udev and Hotplug

When a device is connected, the kernel issues a uevent, and the user-space daemon `udevd` automatically creates device nodes and loads drivers according to rules.

```
Flow from device connection to driver loading:

  USB device inserted
       │
       ▼
  ┌──────────────────┐
  │ USB Core Driver   │  Device detected within the kernel
  │ (usb-core)       │
  └────────┬─────────┘
           │ kobject_uevent() issues kernel event
           ▼
  ┌──────────────────┐
  │ netlink socket   │  Kernel -> user-space notification
  └────────┬─────────┘
           ▼
  ┌──────────────────┐
  │ udevd            │  Receives uevent
  │ (systemd-udevd)  │
  └────────┬─────────┘
           │ Evaluates rules in /etc/udev/rules.d/
           ▼
  ┌──────────────────────────────────────┐
  │ Actions based on rule evaluation:     │
  │  1. Create /dev/xxx device node       │
  │  2. Set permissions and ownership     │
  │  3. Create symbolic links             │
  │  4. Load driver via modprobe          │
  │  5. Execute external script via RUN=  │
  └──────────────────────────────────────┘
```

udev rule examples:

```bash
# /etc/udev/rules.d/99-usb-serial.rules
# Rule to assign a fixed device name to a USB-serial converter

# For devices with Vendor ID=0403 (FTDI), Product ID=6001,
# create a symlink /dev/ttyFTDI,
# assign to group dialout, and set permissions to 0666
SUBSYSTEM=="tty", ATTRS{idVendor}=="0403", ATTRS{idProduct}=="6001", \
    SYMLINK+="ttyFTDI", GROUP="dialout", MODE="0666"

# Assign unique names to devices with specific serial numbers
SUBSYSTEM=="tty", ATTRS{serial}=="A50285BI", SYMLINK+="sensor_gps"
SUBSYSTEM=="tty", ATTRS{serial}=="FTXYZ123", SYMLINK+="sensor_imu"
```

---

## 3. I/O Control Methods in Detail

### 3.1 Three I/O Control Methods

The methods by which an operating system exchanges data with hardware devices are broadly classified into three categories. Each method has different trade-offs in CPU load, latency, and throughput, and they are chosen based on the device characteristics and use case.

#### (a) Polling (Programmed I/O / Polling)

A method where the CPU repeatedly reads the device's status register to detect operation completion. It is the simplest implementation, but it continues to consume CPU cycles while waiting, hence it is also called busy waiting.

```c
/*
 * Pseudo-code for polling I/O
 * Using serial port single-byte reception as an example
 */
#define UART_LSR    0x3FD   /* Line Status Register */
#define UART_RBR    0x3F8   /* Receiver Buffer Register */
#define LSR_DR      0x01    /* Data Ready bit */

uint8_t uart_read_polling(void)
{
    /* Repeatedly check the status register (busy wait) */
    while (!(inb(UART_LSR) & LSR_DR)) {
        /* CPU spins here continuously
         * Cannot execute other tasks
         * Risk of infinite loop if device does not respond
         */
        cpu_relax();  /* Hint instruction for spin loops */
    }

    /* Read the data register once data is ready */
    return inb(UART_RBR);
}
```

#### (b) Interrupt-Driven I/O

A method where the device sends an interrupt signal (IRQ) to the CPU upon completion of an operation or the occurrence of a noteworthy event. Since the CPU can execute other tasks while waiting for interrupts, CPU efficiency is significantly improved compared to polling.

```c
/*
 * Pseudo-code for interrupt-driven I/O
 * Using a serial port interrupt handler as an example
 */
#include <linux/interrupt.h>

/* Receive buffer (circular buffer) */
static DECLARE_KFIFO(rx_fifo, unsigned char, 1024);
static DECLARE_WAIT_QUEUE_HEAD(rx_waitq);

/*
 * Interrupt handler (top half)
 * Called by the kernel when an IRQ occurs
 * Note: sleeping is not allowed in interrupt context
 */
static irqreturn_t uart_irq_handler(int irq, void *dev_id)
{
    uint8_t status = inb(UART_LSR);

    if (!(status & LSR_DR))
        return IRQ_NONE;  /* This interrupt is not for our device */

    /* Read data and store in FIFO */
    while (status & LSR_DR) {
        uint8_t data = inb(UART_RBR);
        kfifo_put(&rx_fifo, data);
        status = inb(UART_LSR);
    }

    /* Wake up waiting processes */
    wake_up_interruptible(&rx_waitq);

    return IRQ_HANDLED;  /* Interrupt has been handled */
}

/* Register IRQ handler during driver initialization */
static int uart_probe(struct platform_device *pdev)
{
    int ret;

    ret = request_irq(IRQ_UART, uart_irq_handler,
                      IRQF_SHARED,       /* Shared IRQ support */
                      "my_uart",          /* Display name in /proc/interrupts */
                      pdev);              /* dev_id: passed to handler */
    if (ret) {
        dev_err(&pdev->dev, "Failed to request IRQ %d\n", IRQ_UART);
        return ret;
    }

    return 0;
}
```

#### (c) DMA (Direct Memory Access)

A method where the DMA controller (or a bus-mastering-capable device itself) performs direct data transfer between the device and main memory without involving the CPU. Since it minimizes CPU load for bulk data transfers, it is an indispensable technology for scenarios requiring high throughput, such as disk I/O, network communication, and audio/video streaming.

```c
/*
 * Pseudo-code for DMA transfer
 * Using data read from a block device as an example
 */
#include <linux/dma-mapping.h>

static int setup_dma_transfer(struct device *dev, void *buffer, size_t len)
{
    dma_addr_t dma_handle;

    /*
     * Create DMA mapping
     * Convert CPU virtual address to bus address accessible by the device
     * If an IOMMU is present, address translation occurs via the IOMMU
     */
    dma_handle = dma_map_single(dev, buffer, len, DMA_FROM_DEVICE);
    if (dma_mapping_error(dev, dma_handle)) {
        dev_err(dev, "DMA mapping failed\n");
        return -ENOMEM;
    }

    /*
     * Instruct the device to perform DMA transfer
     * - Source/destination bus address
     * - Transfer size
     * - Transfer direction
     * Write to device-specific registers
     */
    writel(dma_handle, dev_regs + DMA_ADDR_REG);
    writel(len, dev_regs + DMA_LEN_REG);
    writel(DMA_START | DMA_DIR_READ, dev_regs + DMA_CTRL_REG);

    /* CPU can perform other processing without waiting for transfer completion
     * Transfer completion is notified via interrupt */
    return 0;
}

/* DMA completion interrupt handler */
static irqreturn_t dma_complete_handler(int irq, void *dev_id)
{
    struct device *dev = dev_id;

    /* Release DMA mapping (cache synchronization is also performed) */
    dma_unmap_single(dev, dma_handle, len, DMA_FROM_DEVICE);

    /* Notify upper layer of transfer completion */
    complete(&dma_completion);

    return IRQ_HANDLED;
}
```

### 3.2 I/O Control Method Comparison Table

| Property | Polling | Interrupt-Driven | DMA |
|:---------|:--------|:-----------------|:----|
| CPU Load | Very high (busy wait) | Low (event-driven) | Minimal (CPU not involved during transfer) |
| Latency | Minimum (immediate detection) | Moderate (interrupt delay) | Moderate (setup + interrupt) |
| Throughput | Low (CPU-bound) | Moderate | High (bus bandwidth limit) |
| Implementation Complexity | Simplest | Moderate (handler design) | High (address translation, cache coherency) |
| Use Cases | Embedded, BIOS, short waits | General I/O operations | Bulk data transfer (disk, NIC) |
| Multitasking Suitability | Not suitable | Suitable | Most suitable |
| Device Examples | GPIO, simple sensors | Keyboard, mouse | NVMe SSD, 10GbE NIC |
| Hardware Requirements | None | IRQ lines | DMA controller/bus master |
| Cache Coherency | No issues | No issues | Must be considered (dma_sync_*) |

---

## 4. Internal Structure of Device Drivers

### 4.1 Character Device Driver Structure

Linux character device drivers expose an interface to user space through the `file_operations` structure. This structure contains function pointers such as `open`, `read`, `write`, `ioctl`, and `release` (close), dispatching file operations from the VFS (Virtual File System) layer to device-specific processing.

```c
/*
 * simplechar.c -- Complete implementation of an educational character device driver
 * Exposes a memory buffer as a device
 */
#include <linux/module.h>
#include <linux/fs.h>         /* file_operations, register_chrdev_region */
#include <linux/cdev.h>       /* cdev structure */
#include <linux/device.h>     /* device_create, class_create */
#include <linux/uaccess.h>    /* copy_to_user, copy_from_user */
#include <linux/mutex.h>

#define DEVICE_NAME   "simplechar"
#define CLASS_NAME    "simple"
#define BUFFER_SIZE   4096

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Educational character device driver");

/* Driver private data */
struct simplechar_dev {
    struct cdev    cdev;          /* Kernel cdev structure */
    struct class   *class;        /* Device class */
    struct device  *device;       /* Device */
    dev_t          devno;         /* Device number (major:minor) */
    struct mutex   lock;          /* Mutex for mutual exclusion */
    char           buffer[BUFFER_SIZE];  /* Data buffer */
    size_t         data_len;      /* Valid data length in buffer */
    int            open_count;    /* Open reference count */
};

static struct simplechar_dev *sc_dev;

/* --- file_operations callback implementations --- */

static int sc_open(struct inode *inode, struct file *filp)
{
    struct simplechar_dev *dev;

    /* Retrieve device structure from cdev within inode using container_of */
    dev = container_of(inode->i_cdev, struct simplechar_dev, cdev);
    filp->private_data = dev;  /* Used in subsequent read/write/release */

    mutex_lock(&dev->lock);
    dev->open_count++;
    pr_info("%s: opened (count=%d)\n", DEVICE_NAME, dev->open_count);
    mutex_unlock(&dev->lock);

    return 0;
}

static ssize_t sc_read(struct file *filp, char __user *ubuf,
                        size_t count, loff_t *f_pos)
{
    struct simplechar_dev *dev = filp->private_data;
    ssize_t retval;

    mutex_lock(&dev->lock);

    /* If file offset exceeds data length, return EOF */
    if (*f_pos >= dev->data_len) {
        retval = 0;  /* EOF */
        goto out;
    }

    /* Limit read amount to remaining data */
    if (*f_pos + count > dev->data_len)
        count = dev->data_len - *f_pos;

    /*
     * copy_to_user: copy from kernel space to user space
     * User-space pointer validation is also performed internally
     * Return value: number of bytes that could not be copied (0 = success)
     */
    if (copy_to_user(ubuf, dev->buffer + *f_pos, count)) {
        retval = -EFAULT;  /* Invalid user-space address */
        goto out;
    }

    *f_pos += count;  /* Update file offset */
    retval = count;
    pr_info("%s: read %zu bytes from offset %lld\n",
            DEVICE_NAME, count, *f_pos - count);

out:
    mutex_unlock(&dev->lock);
    return retval;
}

static ssize_t sc_write(struct file *filp, const char __user *ubuf,
                         size_t count, loff_t *f_pos)
{
    struct simplechar_dev *dev = filp->private_data;
    ssize_t retval;

    mutex_lock(&dev->lock);

    /* Prevent buffer overflow */
    if (*f_pos + count > BUFFER_SIZE)
        count = BUFFER_SIZE - *f_pos;

    if (count == 0) {
        retval = -ENOSPC;  /* No space left on device */
        goto out;
    }

    if (copy_from_user(dev->buffer + *f_pos, ubuf, count)) {
        retval = -EFAULT;
        goto out;
    }

    *f_pos += count;
    if (*f_pos > dev->data_len)
        dev->data_len = *f_pos;

    retval = count;
    pr_info("%s: wrote %zu bytes at offset %lld\n",
            DEVICE_NAME, count, *f_pos - count);

out:
    mutex_unlock(&dev->lock);
    return retval;
}

static int sc_release(struct inode *inode, struct file *filp)
{
    struct simplechar_dev *dev = filp->private_data;

    mutex_lock(&dev->lock);
    dev->open_count--;
    pr_info("%s: released (count=%d)\n", DEVICE_NAME, dev->open_count);
    mutex_unlock(&dev->lock);

    return 0;
}

/*
 * file_operations structure -- Connection point between VFS and the driver
 * Operations not defined here fall back to VFS default behavior
 */
static const struct file_operations sc_fops = {
    .owner   = THIS_MODULE,    /* Module reference count management */
    .open    = sc_open,
    .read    = sc_read,
    .write   = sc_write,
    .release = sc_release,
    /* .llseek, .unlocked_ioctl, .poll, .mmap, etc. can also be defined */
};

/* --- Module initialization and exit --- */

static int __init sc_init(void)
{
    int ret;

    /* Allocate device structure */
    sc_dev = kzalloc(sizeof(*sc_dev), GFP_KERNEL);
    if (!sc_dev)
        return -ENOMEM;

    mutex_init(&sc_dev->lock);

    /* Step 1: Dynamic allocation of device numbers */
    ret = alloc_chrdev_region(&sc_dev->devno, 0, 1, DEVICE_NAME);
    if (ret < 0) {
        pr_err("%s: alloc_chrdev_region failed\n", DEVICE_NAME);
        goto err_alloc_region;
    }
    pr_info("%s: registered with major=%d, minor=%d\n",
            DEVICE_NAME, MAJOR(sc_dev->devno), MINOR(sc_dev->devno));

    /* Step 2: Initialize and register cdev structure */
    cdev_init(&sc_dev->cdev, &sc_fops);
    sc_dev->cdev.owner = THIS_MODULE;
    ret = cdev_add(&sc_dev->cdev, sc_dev->devno, 1);
    if (ret < 0) {
        pr_err("%s: cdev_add failed\n", DEVICE_NAME);
        goto err_cdev_add;
    }

    /* Step 3: Create device class (/sys/class/simple/) */
    sc_dev->class = class_create(CLASS_NAME);
    if (IS_ERR(sc_dev->class)) {
        ret = PTR_ERR(sc_dev->class);
        goto err_class;
    }

    /* Step 4: Create device node (/dev/simplechar) */
    sc_dev->device = device_create(sc_dev->class, NULL,
                                    sc_dev->devno, NULL, DEVICE_NAME);
    if (IS_ERR(sc_dev->device)) {
        ret = PTR_ERR(sc_dev->device);
        goto err_device;
    }

    pr_info("%s: driver initialized successfully\n", DEVICE_NAME);
    return 0;

/* Cleanup on error (reverse order of initialization) */
err_device:
    class_destroy(sc_dev->class);
err_class:
    cdev_del(&sc_dev->cdev);
err_cdev_add:
    unregister_chrdev_region(sc_dev->devno, 1);
err_alloc_region:
    kfree(sc_dev);
    return ret;
}

static void __exit sc_exit(void)
{
    /* Cleanup in reverse order of initialization */
    device_destroy(sc_dev->class, sc_dev->devno);
    class_destroy(sc_dev->class);
    cdev_del(&sc_dev->cdev);
    unregister_chrdev_region(sc_dev->devno, 1);
    kfree(sc_dev);
    pr_info("%s: driver removed\n", DEVICE_NAME);
}

module_init(sc_init);
module_exit(sc_exit);
```

### 4.2 Driver Initialization Sequence

The following illustrates each step of the driver initialization in the code above.

```
Driver Initialization Sequence (success path):

  module_init(sc_init) called
       │
       ▼
  ┌─────────────────────────────┐
  │ 1. kzalloc()                │  Allocate memory for driver structure
  │    allocate sc_dev           │
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 2. alloc_chrdev_region()    │  Assign major/minor numbers
  │    devno = (major, minor)   │  Registered in /proc/devices
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 3. cdev_init() + cdev_add() │  Bind fops to cdev
  │    Route VFS operations      │  Registered in cdev_map
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 4. class_create()           │  Create /sys/class/simple/
  │    Create sysfs entries      │  Monitored by udev
  └──────────┬──────────────────┘
             ▼
  ┌─────────────────────────────┐
  │ 5. device_create()          │  Issue uevent
  │    Create /dev/simplechar    │  udev creates device node
  └──────────┬──────────────────┘
             ▼
        Initialization complete (return 0)

  * If any step fails, resources allocated up to that
    point are freed in reverse order (goto error handling)
```

### 4.3 Key Members of the file_operations Structure

`file_operations` is the core structure of Linux drivers and is a table of callback functions invoked by the VFS. The following summarizes the key members and their purposes.

| Member | System Call | Purpose |
|:-------|:-----------|:--------|
| `.owner` | -- | Module reference count management (`THIS_MODULE`) |
| `.open` | `open(2)` | Initialization processing when device is opened |
| `.release` | `close(2)` | Cleanup when the last fd is closed |
| `.read` | `read(2)` | Read data from device to user space |
| `.write` | `write(2)` | Write data from user space to device |
| `.unlocked_ioctl` | `ioctl(2)` | Device-specific control commands |
| `.compat_ioctl` | `ioctl(2)` | Compatible ioctl from 32-bit processes to 64-bit kernel |
| `.poll` | `poll(2)`/`select(2)` | Notification of I/O ready state (asynchronous I/O) |
| `.mmap` | `mmap(2)` | Map device memory to user space |
| `.llseek` | `lseek(2)` | Change file offset |
| `.fasync` | `fcntl(2)` | Asynchronous notification setup (SIGIO) |
| `.flush` | `close(2)` | Per-fd close processing (before all fds are closed) |

---

## 5. Platform Bus and Device Tree

### 5.1 Bus Model and Device-Driver Matching

The Linux kernel has an abstract bus model for binding devices with drivers. For self-describing (discoverable) buses like PCI and USB, devices report their own vendor ID and product ID, and the kernel automatically selects the matching driver.

On the other hand, devices on SoCs (System on Chip) used in embedded systems often lack self-description capabilities. The "platform bus" is a virtual bus designed to handle such devices.

```
Overview of the Bus Model:

  ┌──────────────────────────────────────────────┐
  │            Linux Device Model                 │
  │                                              │
  │  bus_type --- device_driver --- device        │
  │   (bus)        (driver)        (device)      │
  │                                              │
  │  Matching:                                   │
  │    When bus->match(dev, drv) returns true,   │
  │    drv->probe(dev) is called                 │
  │                                              │
  ├─────────────┬───────────────┬────────────────┤
  │  PCI Bus    │   USB Bus     │ Platform Bus   │
  │             │               │                │
  │ vendor_id + │ vendor_id +   │ compatible     │
  │ device_id   │ product_id    │ string or name │
  │ matching    │ matching      │ matching       │
  │             │               │                │
  │ Auto-detect │ Hotplug       │ Static         │
  │ possible    │ supported     │ description    │
  │ (enumera-   │ (device       │ required       │
  │  tion)      │  descriptors) │ (Device Tree   │
  │             │               │  or ACPI)      │
  └─────────────┴───────────────┴────────────────┘
```

### 5.2 Device Tree

The Device Tree is a data format that describes hardware configuration in a tree structure. It is widely used in architectures such as ARM, RISC-V, and PowerPC. Instead of hardcoding hardware information in the kernel source code, it is supplied externally as a DTB (Device Tree Blob).

```
Device Tree Source (.dts) example:

/dts-v1/;

/ {
    compatible = "vendor,board-name";
    model = "Example Development Board";

    /* Devices inside the SoC */
    soc {
        compatible = "simple-bus";
        #address-cells = <1>;
        #size-cells = <1>;

        /* UART Controller */
        uart0: serial@10010000 {
            compatible = "ns16550a";          /* Used for driver matching */
            reg = <0x10010000 0x100>;         /* Register base address and size */
            interrupts = <10>;                /* IRQ number */
            clock-frequency = <48000000>;     /* Clock frequency */
            status = "okay";                  /* Enable the device */
        };

        /* GPIO Controller */
        gpio0: gpio@10020000 {
            compatible = "vendor,gpio-controller";
            reg = <0x10020000 0x40>;
            #gpio-cells = <2>;
            gpio-controller;
            interrupt-controller;
            #interrupt-cells = <2>;
        };

        /* Temperature sensor on I2C bus */
        i2c0: i2c@10030000 {
            compatible = "vendor,i2c-controller";
            reg = <0x10030000 0x100>;
            #address-cells = <1>;
            #size-cells = <0>;

            temperature-sensor@48 {
                compatible = "ti,tmp102";
                reg = <0x48>;                 /* I2C slave address */
            };
        };
    };

    /* LEDs on the board */
    leds {
        compatible = "gpio-leds";
        led-heartbeat {
            gpios = <&gpio0 5 0>;             /* Use pin 5 of GPIO0 */
            linux,default-trigger = "heartbeat";
        };
    };
};
```

Compiling and deploying the Device Tree:

```bash
# Compile Device Tree Source (.dts) to binary (.dtb)
$ dtc -I dts -O dtb -o board.dtb board.dts

# Decompile an existing .dtb to inspect its contents
$ dtc -I dtb -O dts -o decompiled.dts /boot/dtbs/board.dtb

# Inspect the Device Tree of the running system
$ ls /proc/device-tree/
$ cat /proc/device-tree/model

# Device Tree Overlay (apply a diff to an existing DTB)
$ dtc -I dts -O dtb -o overlay.dtbo overlay.dts
$ sudo dtoverlay overlay.dtbo
```

### 5.3 ACPI (Advanced Configuration and Power Interface)

On x86/x64 platforms, ACPI is used as the standard specification for hardware description instead of Device Tree. ACPI tables are provided by the firmware (BIOS/UEFI) and parsed by the kernel at boot time. They contain bytecode called AML (ACPI Machine Language), which is executed by the ACPI interpreter within the kernel.

```bash
# Inspect ACPI tables
$ sudo acpidump | head -30
$ sudo acpidump -b     # Dump in binary format
$ iasl -d DSDT.aml     # Disassemble AML

# List ACPI devices
$ ls /sys/bus/acpi/devices/

# Check power management states
$ cat /sys/bus/acpi/devices/*/power_state
```

---

## 6. User-Space Drivers

### 6.1 Motivation for User-Space Drivers

Traditional kernel drivers have the following challenges.

- Risk of kernel crashes: Driver bugs can crash the entire system (kernel panic)
- Difficulty of debugging: General-purpose debuggers like gdb cannot be directly used in kernel space
- Long development cycles: Each change requires module build, load, and test
- License constraints: GPL kernel modules may require GPL-compatible licensing

To solve these issues, frameworks exist that allow part or all of a device driver to operate in user space.

### 6.2 UIO (Userspace I/O)

UIO is a framework that places a minimal stub driver on the kernel side and implements the actual device control logic as a user-space process.

```c
/*
 * UIO user-space driver example
 * Access device registers through /dev/uio0
 */
#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <stdint.h>
#include <unistd.h>

int main(void)
{
    int fd;
    volatile uint32_t *regs;
    uint32_t irq_count;

    /* Open the UIO device */
    fd = open("/dev/uio0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/uio0");
        return 1;
    }

    /*
     * Map device registers into user space
     * offset 0 = BAR0 (PCI) or register region
     * mmap enables direct access to physical registers
     */
    regs = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                MAP_SHARED, fd, 0);
    if (regs == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    /* Read/write registers (device-specific operations) */
    printf("Device ID: 0x%08x\n", regs[0]);  /* Read register 0 */
    regs[1] = 0x00000001;                     /* Write to register 1 */

    /*
     * Wait for interrupt
     * Block on read(), return when interrupt occurs
     * Return value is the interrupt occurrence count
     */
    while (1) {
        ssize_t n = read(fd, &irq_count, sizeof(irq_count));
        if (n != sizeof(irq_count)) {
            perror("read (IRQ wait)");
            break;
        }
        printf("Interrupt #%u received\n", irq_count);

        /* Interrupt handling (logic can be freely written in user space) */
        uint32_t status = regs[2];  /* Read status register */
        regs[3] = status;           /* Clear interrupt */
    }

    munmap((void *)regs, 4096);
    close(fd);
    return 0;
}
```

### 6.3 VFIO (Virtual Function I/O)

VFIO is a framework that leverages the IOMMU to safely pass through devices to user space. It is used for PCI passthrough where devices are directly assigned to guest OSes in virtualized environments, and in high-performance networking stacks like DPDK (Data Plane Development Kit).

```
Comparison of UIO and VFIO:

  UIO:
  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │User-space │────>│ /dev/uio0    │────>│ Device   │
  │ driver    │     │ (mmap+read)  │     │ registers│
  └──────────┘     └──────────────┘     └──────────┘
       │                                       ▲
       │  DMA buffers are                      │
       │  allocated with hugepages   Direct    │
       └───────────────────────memory access───┘
       * No IOMMU protection -> dangerous if address is specified incorrectly

  VFIO:
  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │User-space │────>│ /dev/vfio/*  │────>│ IOMMU    │
  │ driver    │     │ (ioctl+mmap) │     │          │
  └──────────┘     └──────────────┘     └────┬─────┘
                                              │
                        IOMMU validates and    │ Address
                        translates DMA         │ translation
                        addresses              │ + protection
                                              ▼
                                        ┌──────────┐
                                        │ Device   │
                                        └──────────┘
       * Invalid DMA addresses are blocked by the IOMMU
```

| Property | UIO | VFIO |
|:---------|:----|:-----|
| IOMMU Requirement | Not required | Required |
| DMA Access Protection | None (DMA to any address possible) | Yes (restricted by IOMMU) |
| Interrupt Handling | Blocking read() | Via eventfd |
| Virtualization Support | Limited | Optimal for PCI passthrough |
| Multi-device Isolation | Not possible | Isolation by IOMMU group |
| Performance | High | High (IOMMU translation overhead is small) |
| Security | Low (root privileges required) | High (protected by IOMMU) |
| Typical Users | Industrial/embedded devices | DPDK, SPDK, virtualization |

### 6.4 FUSE (Filesystem in Userspace)

FUSE is a framework for implementing file systems in user space and can be considered a type of user-space driver. Numerous projects leverage FUSE, including SSHFS, NTFS-3G, and GlusterFS.

---

## 7. Advanced Topics in Interrupt Handling

### 7.1 Top Half and Bottom Half

The interrupt handler (top half) runs in the context of hardware interrupt processing and is subject to the following strict constraints.

- **No sleeping**: Functions that may sleep, such as mutex_lock() and kmalloc(GFP_KERNEL), cannot be called
- **Minimize execution time**: Since execution occurs with interrupts disabled, lengthy processing causes delays to other interrupts
- **No user-space access**: copy_to_user()/copy_from_user() cannot be used

To relax these constraints, a design that splits interrupt processing into a "top half (minimal processing that must be done immediately)" and a "bottom half (heavy processing that can be done later)" is common.

```
Interrupt Processing Split Model:

  Hardware interrupt occurs
       │
       ▼
  ┌─────────────────────────────────┐
  │ Top Half (Hard IRQ Context)      │
  │                                 │
  │  - Identify interrupt cause      │  <- Interrupts disabled
  │  - Clear device interrupt flag   │  <- Sleeping not allowed
  │  - Minimal data evacuation       │  <- Must complete as fast
  │  - Schedule bottom half          │     as possible
  │                                 │
  │  return IRQ_HANDLED;            │
  └──────────┬──────────────────────┘
             │ Schedule
             ▼
  ┌─────────────────────────────────┐
  │ Bottom Half (Deferred Execution) │
  │                                 │
  │ ┌───────────┐ ┌───────────┐     │
  │ │ softirq   │ │ tasklet   │     │  <- Interrupts enabled
  │ │ (high     │ │ (medium   │     │  <- But sleeping not
  │ │  priority)│ │  priority)│     │     allowed
  │ └───────────┘ └───────────┘     │
  │                                 │
  │ ┌───────────┐ ┌──────────────┐  │
  │ │ workqueue │ │ threaded IRQ │  │  <- Process context
  │ │ (general) │ │ (recommended)│  │  <- Sleeping allowed
  │ └───────────┘ └──────────────┘  │
  └─────────────────────────────────┘
```

### 7.2 Bottom Half Mechanism Comparison

| Mechanism | Context | Sleeping | Priority | Use Cases |
|:----------|:--------|:---------|:---------|:----------|
| softirq | Soft interrupt | Not allowed | Highest | Networking (NET_RX), block I/O |
| tasklet | Soft interrupt | Not allowed | High | General short deferred processing |
| workqueue | Process | Allowed | Normal | All processing requiring sleep |
| threaded IRQ | Process (kernel thread) | Allowed | High | Recommended approach for modern drivers |

### 7.3 Threaded IRQ (Threaded Interrupts)

In modern Linux kernels, threaded interrupts via `request_threaded_irq()` are recommended. After performing minimal processing in the hard IRQ handler, the remaining processing is executed as a kernel thread. Since it runs in process context, sleeping is allowed, and the use of mutexes, waiting for DMA completion, and similar operations can be written naturally.

```c
/*
 * Threaded interrupt implementation example
 */
static irqreturn_t sensor_hard_irq(int irq, void *dev_id)
{
    struct sensor_dev *sdev = dev_id;

    /*
     * Hard IRQ handler (top half)
     * Perform only minimal processing
     */
    sdev->irq_status = readl(sdev->regs + IRQ_STATUS_REG);

    if (!(sdev->irq_status & IRQ_PENDING))
        return IRQ_NONE;  /* Not this device's interrupt */

    /* Mask interrupts (suppress until thread handler completes) */
    writel(0, sdev->regs + IRQ_ENABLE_REG);

    return IRQ_WAKE_THREAD;  /* Wake the thread handler */
}

static irqreturn_t sensor_thread_fn(int irq, void *dev_id)
{
    struct sensor_dev *sdev = dev_id;
    int ret;

    /*
     * Thread handler (bottom half)
     * Runs in process context -> sleeping is allowed
     */

    /* Read sensor data via I2C (involves sleeping) */
    mutex_lock(&sdev->lock);
    ret = i2c_smbus_read_word_data(sdev->i2c_client, DATA_REG);
    if (ret >= 0) {
        sdev->last_value = ret;
        sysfs_notify(&sdev->dev->kobj, NULL, "value");
    }
    mutex_unlock(&sdev->lock);

    /* Re-enable interrupts */
    writel(IRQ_ENABLE, sdev->regs + IRQ_ENABLE_REG);

    return IRQ_HANDLED;
}

/* During driver initialization */
static int sensor_probe(struct platform_device *pdev)
{
    int ret;

    ret = request_threaded_irq(
        sdev->irq,
        sensor_hard_irq,     /* Hard IRQ handler (top half) */
        sensor_thread_fn,     /* Thread handler (bottom half) */
        IRQF_ONESHOT,         /* Mask interrupt until thread completes */
        "sensor_drv",
        sdev
    );

    return ret;
}
```

### 7.4 MSI/MSI-X (Message Signaled Interrupts)

In the PCIe era, MSI (Message Signaled Interrupts) or MSI-X (Extended MSI) are used instead of traditional pin-based interrupts (INTx). MSI is a method that signals interrupts via memory writes and offers the following advantages.

- No physical IRQ lines needed (no problems from sharing)
- Multiple interrupt vectors can be assigned to a single device (MSI-X: up to 2048)
- Low latency (memory write only)
- Interrupt ordering is guaranteed

---

## 8. Power Management and Suspend/Resume

### 8.1 Runtime Power Management

Device drivers need to actively participate in system power management. The Linux kernel provides two mechanisms for device-level power management: "runtime PM" and "system sleep."

Runtime PM is a mechanism that automatically transitions individual devices to a low-power state when they are not in use.

```c
/*
 * Runtime power management implementation example
 */
#include <linux/pm_runtime.h>

static int mydev_probe(struct platform_device *pdev)
{
    struct mydev *dev = platform_get_drvdata(pdev);

    /* Enable runtime PM */
    pm_runtime_set_active(&pdev->dev);
    pm_runtime_enable(&pdev->dev);

    /*
     * Configure auto-suspend
     * Automatically suspend 2 seconds after the last operation
     */
    pm_runtime_set_autosuspend_delay(&pdev->dev, 2000);
    pm_runtime_use_autosuspend(&pdev->dev);

    return 0;
}

/* Call before using the device */
static int mydev_do_io(struct mydev *dev)
{
    int ret;

    /* Transition device to active state (power ON if necessary) */
    ret = pm_runtime_get_sync(dev->dev);
    if (ret < 0) {
        pm_runtime_put_noidle(dev->dev);
        return ret;
    }

    /* Device I/O operation */
    writel(cmd, dev->regs + CMD_REG);

    /* Notify usage completion (start auto-suspend timer) */
    pm_runtime_mark_last_busy(dev->dev);
    pm_runtime_put_autosuspend(dev->dev);

    return 0;
}

/* Runtime suspend/resume callbacks */
static int mydev_runtime_suspend(struct device *dev)
{
    struct mydev *mdev = dev_get_drvdata(dev);

    /* Stop device clock */
    clk_disable_unprepare(mdev->clk);
    /* Turn OFF regulator */
    regulator_disable(mdev->supply);

    return 0;
}

static int mydev_runtime_resume(struct device *dev)
{
    struct mydev *mdev = dev_get_drvdata(dev);
    int ret;

    /* Turn ON regulator */
    ret = regulator_enable(mdev->supply);
    if (ret)
        return ret;
    /* Resume clock */
    ret = clk_prepare_enable(mdev->clk);
    if (ret) {
        regulator_disable(mdev->supply);
        return ret;
    }

    /* Re-initialize device (restore registers, etc.) */
    mydev_hw_init(mdev);

    return 0;
}

static const struct dev_pm_ops mydev_pm_ops = {
    SET_RUNTIME_PM_OPS(mydev_runtime_suspend,
                       mydev_runtime_resume, NULL)
    SET_SYSTEM_SLEEP_PM_OPS(mydev_system_suspend,
                            mydev_system_resume)
};
```

### 8.2 System Sleep (S3/S4)

When the entire system transitions to a suspend state (S3: Sleep / S4: Hibernate), the suspend/resume callbacks of all device drivers are called. Drivers need to perform the following.

**During suspend:**
1. Complete or cancel in-progress I/O operations
2. Stop accepting new I/O requests
3. Save the device's hardware state
4. Disable interrupts
5. Transition the device to a low-power state

**During resume:**
1. Restore device power
2. Restore hardware state (reconfigure registers)
3. Re-enable interrupts
4. Resume accepting I/O requests

### 8.3 Power State Hierarchy

```
System-wide Power States (ACPI-based):

  S0 (Working)     -- Normal operation
  │
  ├── S0ix (Modern Standby) -- Low-power idle (network can be maintained)
  │
  S1 (Standby)     -- CPU stopped, memory retained
  │
  S2 (--)          -- Rarely used
  │
  S3 (Sleep)       -- Only memory powered (Suspend to RAM)
  │
  S4 (Hibernate)   -- Memory contents saved to disk, full power off
  │
  S5 (Soft Off)    -- Software power off

Device-level Power States:

  D0 (Full Power)  -- Fully operational state
  │
  D1 (Light Sleep) -- Low power (some functions stopped)
  │
  D2 (Deep Sleep)  -- Even lower power (partial context loss)
  │
  D3hot            -- Lowest power (bus connection maintained)
  │
  D3cold           -- Full power off (bus connection also severed)
```

---

## 9. sysfs and Device Attributes

### 9.1 Role of sysfs

sysfs is a virtual file system that exports kernel internal objects (devices, drivers, buses, etc.) as a file system tree under `/sys`. Driver developers can expose device configuration values and state to user space through sysfs attributes.

```bash
# Examples of inspecting sysfs structure
$ ls /sys/class/net/eth0/
address  carrier  device  duplex  mtu  operstate  speed  statistics  ...

$ cat /sys/class/net/eth0/mtu
1500

$ cat /sys/class/net/eth0/address
00:1a:2b:3c:4d:5e

# Check/change block device I/O scheduler
$ cat /sys/block/sda/queue/scheduler
[mq-deadline] kyber bfq none

$ echo "bfq" | sudo tee /sys/block/sda/queue/scheduler

# Check device power state
$ cat /sys/devices/pci0000:00/0000:00:1f.0/power/runtime_status
active
```

### 9.2 Implementing Custom sysfs Attributes

The following shows how to expose driver-specific settings and state via sysfs.

```c
/*
 * sysfs attribute implementation example
 * Exposed as /sys/class/simple/simplechar/debug_level
 */

static int debug_level = 0;

/* read: cat /sys/.../debug_level */
static ssize_t debug_level_show(struct device *dev,
                                 struct device_attribute *attr,
                                 char *buf)
{
    return sysfs_emit(buf, "%d\n", debug_level);
}

/* write: echo 3 > /sys/.../debug_level */
static ssize_t debug_level_store(struct device *dev,
                                  struct device_attribute *attr,
                                  const char *buf, size_t count)
{
    int val;
    int ret;

    ret = kstrtoint(buf, 10, &val);
    if (ret)
        return ret;

    if (val < 0 || val > 5)
        return -EINVAL;

    debug_level = val;
    pr_info("debug_level set to %d\n", debug_level);

    return count;
}

/* Register show/store with DEVICE_ATTR_RW macro */
static DEVICE_ATTR_RW(debug_level);

/* Group multiple attributes */
static struct attribute *mydev_attrs[] = {
    &dev_attr_debug_level.attr,
    NULL,
};
ATTRIBUTE_GROUPS(mydev);

/* Specify attribute group when creating device in probe() */
/* class->dev_groups = mydev_groups; */
```

---

## 10. Debugging Techniques

### 10.1 Fundamental Kernel Debugging Tools

| Tool | Purpose | Use Case |
|:-----|:--------|:---------|
| `printk` / `pr_info` | Kernel log output | Basic tracing |
| `dmesg` | Display kernel ring buffer | Checking driver messages |
| `ftrace` | Function tracing | Tracking call paths |
| `perf` | Performance profiling | Identifying bottlenecks |
| `crash` / `kdump` | Kernel crash dump analysis | Post-mortem analysis |
| `/proc/interrupts` | Interrupt statistics | Checking IRQ distribution |
| `/proc/iomem` | I/O memory map | Checking address space |
| `/proc/ioports` | I/O port map | Checking port addresses |
| `strace` | System call tracing | Tracking calls from user space |

### 10.2 Dynamic Debug

The Linux kernel's pr_debug() and dev_dbg() can be toggled ON/OFF at runtime through the dynamic debug mechanism.

```bash
# Enable dynamic debug
# Enable all debug messages in a specific file
$ echo "file mydriver.c +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# Enable debug messages for a specific function
$ echo "func mydev_probe +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# Enable all debug messages for a specific module
$ echo "module mydriver +p" | sudo tee /sys/kernel/debug/dynamic_debug/control

# Check currently enabled debug points
$ cat /sys/kernel/debug/dynamic_debug/control | grep mydriver

# Function tracing with ftrace
$ echo function > /sys/kernel/debug/tracing/current_tracer
$ echo mydev_* > /sys/kernel/debug/tracing/set_ftrace_filter
$ echo 1 > /sys/kernel/debug/tracing/tracing_on
$ cat /sys/kernel/debug/tracing/trace
```

---

## 11. Anti-Pattern Collection

### 11.1 Anti-Pattern 1: Sleeping in an Interrupt Handler

**Problem**: Calling functions that may sleep in interrupt context (top half) causes the kernel to produce a "BUG: scheduling while atomic" error, which in the worst case leads to a kernel panic.

```c
/* BAD: Sleeping in an interrupt handler (never do this) */
static irqreturn_t bad_irq_handler(int irq, void *dev_id)
{
    struct my_dev *dev = dev_id;
    void *buf;

    /*
     * BAD: GFP_KERNEL is a sleepable allocation
     * Since sleeping is not allowed in interrupt context,
     * the kernel may output a BUG message and crash
     */
    buf = kmalloc(4096, GFP_KERNEL);  /* BAD! */

    /*
     * BAD: mutex_lock may sleep
     * If another context holds the lock,
     * the interrupt handler will sleep, leading to deadlock
     */
    mutex_lock(&dev->lock);           /* BAD! */

    /* BAD: copy_to_user may cause a page fault */
    copy_to_user(ubuf, data, len);    /* BAD! */

    mutex_unlock(&dev->lock);
    kfree(buf);
    return IRQ_HANDLED;
}

/* OK: Correct implementation -- use threaded IRQ or workqueue */
static irqreturn_t good_hard_irq(int irq, void *dev_id)
{
    struct my_dev *dev = dev_id;

    /* Minimal processing: read status and clear interrupt */
    dev->irq_status = readl(dev->regs + STATUS_REG);
    writel(dev->irq_status, dev->regs + IRQ_ACK_REG);

    return IRQ_WAKE_THREAD;  /* Delegate to thread handler */
}

static irqreturn_t good_thread_fn(int irq, void *dev_id)
{
    struct my_dev *dev = dev_id;
    void *buf;

    /* OK: GFP_KERNEL can be used in process context */
    buf = kmalloc(4096, GFP_KERNEL);

    /* OK: mutex_lock can also be used */
    mutex_lock(&dev->lock);
    /* Data processing */
    mutex_unlock(&dev->lock);

    kfree(buf);
    return IRQ_HANDLED;
}
```

**Lesson**: The functions that can be safely used in interrupt context are limited. Use `GFP_ATOMIC` for memory allocation, `spin_lock_irqsave()` for mutual exclusion, and delegate heavy processing to bottom halves. The most recommended approach is threaded interrupts via `request_threaded_irq()`.

### 11.2 Anti-Pattern 2: Error Handling with Resource Leaks

**Problem**: When acquiring multiple resources in a driver's initialization function, if an error occurs at an intermediate step, failing to properly release the previously acquired resources causes resource leaks.

```c
/* BAD: Error handling with resource leaks */
static int bad_probe(struct platform_device *pdev)
{
    struct my_dev *dev;
    int ret;

    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->clk = clk_get(&pdev->dev, "main_clk");
    if (IS_ERR(dev->clk))
        return PTR_ERR(dev->clk);    /* BAD: dev memory is not freed */

    ret = clk_prepare_enable(dev->clk);
    if (ret)
        return ret;                   /* BAD: clk and dev are not freed */

    dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dev->regs))
        return PTR_ERR(dev->regs);    /* BAD: clk still enabled, dev not freed */

    /* ... */
    return 0;
}

/* OK: Correct error handling -- goto chain pattern */
static int good_probe(struct platform_device *pdev)
{
    struct my_dev *dev;
    int ret;

    dev = kzalloc(sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    dev->clk = clk_get(&pdev->dev, "main_clk");
    if (IS_ERR(dev->clk)) {
        ret = PTR_ERR(dev->clk);
        goto err_free_dev;            /* Free dev */
    }

    ret = clk_prepare_enable(dev->clk);
    if (ret)
        goto err_put_clk;             /* Put clk and free dev */

    dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dev->regs)) {
        ret = PTR_ERR(dev->regs);
        goto err_disable_clk;         /* Disable/put clk and free dev */
    }

    return 0;

err_disable_clk:
    clk_disable_unprepare(dev->clk);
err_put_clk:
    clk_put(dev->clk);
err_free_dev:
    kfree(dev);
    return ret;
}

/*
 * Even better: Use devm_* (device-managed) APIs
 * Resources acquired with devm_ APIs are automatically released when the driver is removed
 */
static int best_probe(struct platform_device *pdev)
{
    struct my_dev *dev;

    /* devm_kzalloc: Memory allocation bound to device lifecycle */
    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;

    /* devm_clk_get: Auto-released clock acquisition */
    dev->clk = devm_clk_get(&pdev->dev, "main_clk");
    if (IS_ERR(dev->clk))
        return PTR_ERR(dev->clk);  /* devm_kzalloc portion is auto-released */

    /* devm_ioremap_resource: Auto-released I/O memory mapping */
    dev->regs = devm_ioremap_resource(&pdev->dev, res);
    if (IS_ERR(dev->regs))
        return PTR_ERR(dev->regs);  /* All above resources are auto-released */

    return 0;
    /* On remove: all resources acquired via devm_* are auto-released in reverse order */
}
```

**Lesson**: Actively leverage the Linux kernel's `devm_*` (device-managed) APIs. Resources acquired with devm_ APIs are automatically released when the driver is unbound, significantly simplifying error handling goto chains and fundamentally eliminating the risk of resource leaks.

### 11.3 Anti-Pattern 3: Inappropriate Lock Granularity

**Problem**: Protecting the entire driver with a single lock like the BKL (Big Kernel Lock) severely impairs concurrency. Conversely, if lock granularity is too fine, the risk of deadlocks and race conditions increases.

- Granularity too coarse: Protecting the entire driver with one mutex -> all operations are serialized, unable to leverage multi-core performance
- Granularity too fine: A lock per data structure field -> lock ordering management becomes difficult, increasing deadlock risk
- Appropriate granularity: A lock per device instance, or per logically independent data structure

---

## 12. Practical Exercises

### Exercise 1: [Basic] Device Observation and Information Gathering

**Goal**: Understand the relationship between devices and drivers on a Linux system.

```bash
# === Step 1: Check device file types ===
# 'b' = block, 'c' = character
ls -la /dev/sda /dev/null /dev/tty0 /dev/random 2>/dev/null

# How to read major and minor numbers
# crw-rw-rw- 1 root root 1, 3 ... /dev/null
#                         ^  ^
#                    major=1 minor=3

# === Step 2: Block device hierarchy ===
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL

# === Step 3: PCI device and driver mapping ===
lspci -v | head -40
# Check "Kernel driver in use:" to see which driver is being used

# === Step 4: Investigate loaded kernel modules ===
lsmod | sort -k3 -rn | head -20
# Sort by 3rd column (Used by) -> modules with most dependencies appear at top

# === Step 5: Check interrupt distribution ===
cat /proc/interrupts | head -20
# Interrupt counts per CPU (CPU0, CPU1, ...) are displayed

# === Step 6: Trace driver behavior from kernel messages ===
dmesg | grep -i "driver\|probe\|loaded" | tail -20
```

**Task**: Based on the output of the above commands, fill in the following table.

| Device Name | Device Type | Major Number | Driver Used |
|:------------|:-----------|:-------------|:------------|
| /dev/sda | Block | ? | ? |
| /dev/null | Character | ? | ? |
| (NIC name) | Network | -- | ? |

### Exercise 2: [Intermediate] Building and Loading a Kernel Module

**Goal**: Build a minimal kernel module and experience the load/unload lifecycle.

**Prerequisites**: A Linux environment (virtual machine recommended), with `build-essential` and `linux-headers-$(uname -r)` installed.

```bash
# === Step 1: Verify header package ===
ls /lib/modules/$(uname -r)/build/
# OK if Makefile, include/, etc. exist

# === Step 2: Create module source ===
mkdir -p ~/driver_lab && cd ~/driver_lab

# Create hello_driver.c (use the code from Section 2.2 of this chapter)
# Create Makefile (use the Makefile from Section 2.2 of this chapter)

# === Step 3: Build ===
make
# On success, hello_driver.ko is generated

# === Step 4: Check module information ===
modinfo hello_driver.ko

# === Step 5: Load and check logs ===
sudo insmod hello_driver.ko
dmesg | tail -5
lsmod | grep hello

# === Step 6: Unload and check logs ===
sudo rmmod hello_driver
dmesg | tail -5

# === Step 7: Extend to a parameterized module ===
# Try adding the following to hello_driver.c:
#   static int repeat = 1;
#   module_param(repeat, int, 0644);
#   MODULE_PARM_DESC(repeat, "Number of greeting repetitions");
# Loop 'repeat' times in the init function, outputting pr_info
```

**Advanced Task**: Verify that the value of parameter `repeat` can be read from `/sys/module/hello_driver/parameters/repeat`, and confirm that changing the value at runtime alters the behavior.

### Exercise 3: [Advanced] Implementing and Verifying a Character Device Driver

**Goal**: Build the simplechar driver from Section 4.1 and verify read/write operations from user space.

```bash
# === Step 1: Build and load the driver ===
cd ~/driver_lab
# Create simplechar.c (use the code from Section 4.1 of this chapter)
# Modify the obj-m line in Makefile: obj-m += simplechar.o
make
sudo insmod simplechar.ko

# === Step 2: Verify the device node ===
ls -la /dev/simplechar
# crw------- 1 root root 240, 0 ... /dev/simplechar
# (major number is dynamically assigned, so it may differ)

# Change permissions (for testing)
sudo chmod 666 /dev/simplechar

# === Step 3: Write test ===
echo "Hello, kernel driver!" > /dev/simplechar
dmesg | tail -3

# === Step 4: Read test ===
cat /dev/simplechar
# "Hello, kernel driver!" should be displayed

# === Step 5: Read with offset using dd ===
dd if=/dev/simplechar bs=1 skip=7 count=6 2>/dev/null
# "kernel" should be displayed

# === Step 6: Concurrent access test from multiple processes ===
# Terminal 1:
while true; do echo "Writer1: $(date)" > /dev/simplechar; done &
# Terminal 2:
while true; do cat /dev/simplechar; done &
# Verify that mutex-based mutual exclusion is functioning correctly

# === Step 7: Cleanup ===
sudo rmmod simplechar
dmesg | tail -5
```

**Advanced Tasks**:
1. Add `unlocked_ioctl` to implement a buffer clear function (`ioctl(fd, SIMPLECHAR_CLEAR, 0)`)
2. Add `poll` so that data writes can be detected with `select()`/`poll()`
3. Add a `/sys/class/simple/simplechar/buffer_usage` attribute that displays the current buffer usage as a percentage

---

## 13. Cross-OS Driver Model Comparison

| Property | Linux | Windows (WDM/WDF) | macOS (IOKit/DriverKit) | FreeBSD |
|:---------|:------|:-------------------|:------------------------|:--------|
| Driver Language | C (Rust being gradually introduced) | C/C++ | C++/Swift (DriverKit) | C |
| Load Unit | Kernel module (.ko) | Driver package (.sys) | kext / dext | Kernel module (.ko) |
| Device Description | Device Tree / ACPI | INF files | IOKit matching | hints / FDT |
| User-Space Drivers | UIO / VFIO | UMDF | DriverKit (dext) | None (standard) |
| Driver Signing | Optional (required with Secure Boot) | Required (WHQL recommended) | Required (notarization) | Optional |
| Hotplug | udev + uevent | PnP Manager | IOKit matching | devd |
| Power Management | Runtime PM + ACPI | WDF Power Policy | IOPMPowerState | ACPI |
| Debugging | printk, ftrace, kgdb | WinDbg, Driver Verifier | lldb, IOKitDebug | kgdb, DTrace |

---

## 14. Frequently Asked Questions (FAQ)

### Q1: What is the difference between a kernel module and a built-in kernel driver?

**A1**: Functionally, they are equivalent. The difference lies in the timing and method of loading.

- **Kernel built-in**: Compiled into the kernel image (vmlinuz) itself and automatically available at boot time. Configured as `CONFIG_XXX=y` in the kernel `.config`. Drivers needed for mounting the root file system (storage controllers, file systems) typically need to be built-in (when not using initramfs).
- **Kernel module (loadable)**: Placed as `.ko` files under `/lib/modules/` and dynamically loaded by `modprobe` or udev. Configured as `CONFIG_XXX=m` in `.config`. The advantage is that drivers for unused devices do not consume memory.

Most distributions build as many drivers as possible as modules and include the minimum set of modules needed for boot within initramfs.

### Q2: Why can a device driver bug crash the entire system?

**A2**: Traditional kernel drivers operate in the same address space and at the same privilege level as the kernel, so they do not benefit from memory protection. Specifically, this is due to the following reasons.

- **NULL pointer dereference**: A NULL pointer dereference in kernel space causes a page fault, resulting in an unrecoverable oops or kernel panic
- **Buffer overflow**: May corrupt critical kernel data structures
- **Deadlock**: If kernel threads or interrupt handlers are permanently blocked, the entire system becomes unresponsive
- **Invalid memory deallocation**: Use-after-free and double-free can corrupt the kernel's memory allocator

The following approaches are taken to mitigate this problem:
1. Isolation through user-space drivers (UIO, VFIO, DriverKit)
2. Safe kernel extensions via eBPF
3. Memory-safe driver implementation in Rust (Linux 6.1 and later)
4. Microkernel architecture (MINIX 3, seL4)

### Q3: What should I do if no driver is found when connecting new hardware?

**A3**: Investigate and address the issue with the following steps.

1. **Verify device recognition**: Confirm vendor ID and product ID with `lspci -nn` (PCI) or `lsusb -v` (USB)
2. **Check kernel logs**: Check for error messages or unsupported device warnings with `dmesg | tail -30`
3. **Search for a compatible driver**: Search the kernel source using the vendor ID:product ID (e.g., `8086:1533`) to identify the corresponding driver module name
4. **Manually load the module**: Try `sudo modprobe <module_name>`
5. **Check kernel version**: New devices may only be supported in the latest kernel. Verify with `uname -r` and update to a newer kernel if necessary
6. **Introduce an OOT (Out-of-Tree) driver**: Consider vendor-provided drivers or third-party drivers via DKMS
7. **Check firmware**: Some devices require firmware binaries included in the `linux-firmware` package. Check whether the firmware exists under `/lib/firmware/`

### Q4: What are the benefits of getting a driver merged into the Linux kernel (mainlining)?

**A4**: When a driver is merged into the kernel's mainline tree, the following benefits are obtained.

- **Continuous maintenance**: Fixes for kernel API changes are made by the community
- **Extensive testing**: Quality assurance through CI/CD systems and numerous testers
- **Distribution inclusion**: Included in major distribution kernel packages, eliminating the need for users to manually install drivers
- **Security fixes**: When vulnerabilities are discovered, the kernel security team responds promptly

### Q5: Why are GPU drivers special?

**A5**: GPU drivers are special and complex compared to other device drivers in the following ways.

- **DRM/KMS subsystem**: A kernel-side framework responsible for display output control (mode setting, CRTC, encoder, and connector management)
- **User-space components**: Closely cooperates with large user-space libraries such as Mesa (OpenGL/Vulkan implementation) and libdrm
- **Memory management complexity**: VRAM (video memory) management, buffer object management via GEM/TTM, and memory coherency control between CPU and GPU
- **Command submission**: Scheduling job submission to GPU command queues and completion waiting
- **Vendor-specific complexity**: Architectures differ greatly between NVIDIA (proprietary + Nouveau), AMD (amdgpu), and Intel (i915/xe)

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important. Understanding deepens not just through theory alone, but by actually writing code and verifying behavior.

### Q2: What are common mistakes that beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before proceeding to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently utilized in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Point |
|:--------|:----------|
| Role of Device Drivers | Interpreter that translates hardware-specific operations into a unified OS API |
| Character Device | Byte stream. Sequential access via file_operations |
| Block Device | Block units. Via I/O scheduler and page cache |
| Network Device | Packet units. Via socket API. Does not appear in /dev |
| Kernel Module | Dynamically loadable. module_init/module_exit |
| Polling | CPU constantly monitors. Simple but inefficient. Suited for embedded |
| Interrupt-Driven | Event notification. Good CPU efficiency. General-purpose method |
| DMA | Direct memory transfer without CPU involvement. Optimal for bulk data |
| Top Half/Bottom Half | Split interrupt processing. Defer heavy processing |
| Threaded IRQ | Modern recommended approach. Sleeping allowed in process context |
| UIO/VFIO | User-space drivers. Improved safety and development ease |
| devm_* API | Device-managed resources. Auto-release prevents leaks |
| Device Tree | Hardware description. Standard for ARM/RISC-V |
| ACPI | Hardware description and power management standard for x86 |
| Runtime PM | Per-device dynamic power management |

---

## Recommended Next Guides


---

## References

1. Corbet, J., Rubini, A., Kroah-Hartman, G. "Linux Device Drivers." 3rd Edition, O'Reilly Media, 2005. -- The classic book on Linux driver development. While some parts have become outdated due to kernel API changes, the design philosophy and fundamental concepts remain valid. Online version: https://lwn.net/Kernel/LDD3/

2. Love, R. "Linux Kernel Development." 3rd Edition, Addison-Wesley, 2010. -- Comprehensive explanation of kernel internals. Covers prerequisite knowledge for driver development including process management, memory management, interrupt handling, and synchronization mechanisms.

3. The Linux Kernel Documentation -- "Driver Model." https://www.kernel.org/doc/html/latest/driver-api/index.html -- The kernel's official driver API documentation. The most reliable source for up-to-date API references. Contains official guidelines for the device model, DMA mapping, interrupt handling, power management, and more.

4. Kroah-Hartman, G. "Linux Kernel in a Nutshell." O'Reilly Media, 2006. -- A practical guide focused on kernel configuration, building, and module management.

5. Venkateswaran, S. "Essential Linux Device Drivers." Prentice Hall, 2008. -- Systematically explains the implementation of character drivers, block drivers, network drivers, USB drivers, and more.

6. Mauerer, W. "Professional Linux Kernel Architecture." Wiley, 2008. -- Detailed internal explanation of kernel architecture. Deeply explores the implementation of virtual memory, process scheduler, VFS, network stack, and more.

