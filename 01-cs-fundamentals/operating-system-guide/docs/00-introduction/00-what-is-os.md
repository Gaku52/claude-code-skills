# What Is an Operating System?

> An operating system is software that "hides the complexity of hardware and provides a unified interface to applications."

## Learning Objectives

- [ ] Explain the role and basic functions of an OS
- [ ] Understand the difference between kernel space and user space
- [ ] Distinguish between OS architectures (monolithic, microkernel, etc.)
- [ ] Understand the mechanism of system calls and the major syscalls
- [ ] Understand the significance of POSIX and Unix philosophy
- [ ] Know the characteristics and historical background of major OS families


## Prerequisites

The following knowledge will help you better understand this guide:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Why Do We Need an OS?

```
A world without an OS:

  Application -> Directly operates hardware

  Problems:
  +----------------------------------------------+
  | 1. Every app must know hardware details       |
  |    -> Specify disk sector numbers directly?   |
  |    -> Directly manipulate GPU registers?      |
  |                                               |
  | 2. Resource contention                        |
  |    -> Two apps want to use the printer at once|
  |    -> Freely using memory, destroying others  |
  |                                               |
  | 3. No security                                |
  |    -> Any app can access all data             |
  |    -> Malicious programs can do whatever they |
  |       want                                    |
  +----------------------------------------------+

The role of an OS:
  +----------------------------------------------+
  | Applications (browser, editor, etc.)          |
  +----------------------------------------------+
  | OS (Kernel)                                   |
  |  +- Process management:  Distributes CPU time |
  |  +- Memory management:   Safely allocates mem |
  |  +- File system:         Persistent storage   |
  |  +- I/O management:      Unified device API   |
  |  +- Security:            Access control        |
  +----------------------------------------------+
  | Hardware (CPU, memory, disk, NIC, etc.)       |
  +----------------------------------------------+

  Two faces of an OS:
  1. Resource manager: Manages and distributes CPU, memory, disk, etc.
  2. Abstraction layer: Hides hardware complexity and provides simple APIs
```

### 1.1 Specific Functions of an OS

```
Detailed major functions provided by an OS:

  1. Process management:
     +----------------------------------------------------+
     | - Process creation and termination                  |
     | - CPU scheduling (which process runs when)          |
     | - Inter-process communication (pipes, sockets,      |
     |   shared memory)                                    |
     | - Synchronization and mutual exclusion              |
     |   (mutex, semaphore)                                |
     | - Signal handling (SIGKILL, SIGTERM, etc.)          |
     |                                                     |
     | Example: When you launch Chrome                     |
     |   -> OS creates a new process                       |
     |   -> Allocates memory space                         |
     |   -> Distributes CPU time slices                    |
     |   -> Isolates multiple tabs in separate processes   |
     +----------------------------------------------------+

  2. Memory management:
     +----------------------------------------------------+
     | - Virtual memory: Provides address space larger     |
     |   than physical memory                              |
     | - Paging: Manages memory in 4KB page units          |
     | - Memory protection: Isolates memory between        |
     |   processes                                         |
     | - Memory mapping: Maps files directly to memory     |
     | - Swap: Evicts unused pages to disk                 |
     |                                                     |
     | Example: Running multiple apps with 8GB physical RAM|
     |   -> Each process has its own virtual address space  |
     |   -> Can operate even with 20GB total memory demand |
     |   -> Unused pages are swapped to disk               |
     +----------------------------------------------------+

  3. File system:
     +----------------------------------------------------+
     | - File creation, reading/writing, and deletion      |
     | - Hierarchical directory (folder) structure         |
     | - Access permission management (owner, group, other)|
     | - Journaling: Protects data from unexpected power   |
     |   loss                                              |
     | - Mount: Integrates different storage devices       |
     |                                                     |
     | Major file systems:                                 |
     |   ext4: Linux standard (journaling, max 1EB)        |
     |   APFS: macOS/iOS (CoW, encryption, snapshots)      |
     |   NTFS: Windows (ACL, journaling, compression)      |
     |   Btrfs: Next-gen Linux (CoW, RAID, snapshots)      |
     |   ZFS: Enterprise (checksums, RAID-Z, compression)  |
     |   XFS: Large files (high-concurrency I/O)           |
     +----------------------------------------------------+

  4. I/O management:
     +----------------------------------------------------+
     | - Device drivers: Abstract hardware communication   |
     | - Buffering: Buffers to optimize I/O                |
     | - Interrupt handling: Process notifications from     |
     |   devices                                           |
     | - DMA (Direct Memory Access): Data transfer without |
     |   CPU involvement                                   |
     | - I/O scheduling: Optimize disk access              |
     |                                                     |
     | Role of device drivers:                             |
     |   App -> write() -> VFS -> File system              |
     |                      -> Block layer                 |
     |                      -> Device driver               |
     |                      -> Hardware (SSD/HDD)          |
     +----------------------------------------------------+

  5. Network management:
     +----------------------------------------------------+
     | - TCP/IP protocol stack implementation              |
     | - Socket API: Network communication for apps        |
     | - Firewall: Packet filtering                        |
     | - Routing: Determining packet forwarding            |
     |   destination                                       |
     | - Network device drivers                            |
     |                                                     |
     | Linux network stack:                                |
     |   App -> socket() -> TCP/UDP -> IP -> NIC Driver    |
     |   -> iptables/nftables for filtering                |
     |   -> tc (traffic control) for bandwidth control     |
     +----------------------------------------------------+

  6. Security:
     +----------------------------------------------------+
     | - User authentication: Login, password, biometrics  |
     | - Access control: File permissions, capabilities    |
     | - Encryption: Disk encryption, communication        |
     |   encryption                                        |
     | - Auditing: Logging security events                 |
     | - Sandboxing: Restricting app permissions           |
     |                                                     |
     | Linux security modules:                             |
     |   SELinux: Mandatory access control (Red Hat family)|
     |   AppArmor: Pathname-based access control (Ubuntu)  |
     |   seccomp: System call filtering                    |
     |   namespaces + cgroups: Foundation for container    |
     |   isolation                                         |
     +----------------------------------------------------+
```

### 1.2 The Era Before Operating Systems

```
Programming without an OS (1950s):

  Procedure:
  1. Write a program on punch cards
  2. Bring cards to the computer room and hand them
     to the operator
  3. Program is loaded into the machine (wait hours
     to days)
  4. Results are printed on paper and returned
  5. If there's a bug, start over from step 1

  Problems:
  - CPU utilization was extremely low (idle during
    program loading)
  - Programmers had to calculate hardware addresses
    directly
  - I/O device timing had to be managed manually
  - An error would halt the entire machine

  What the introduction of an OS solved:
  +----------------------------------------------------+
  | Batch processing OS (late 1950s):                   |
  | -> Automatically executed jobs one after another     |
  | -> CPU utilization improved dramatically             |
  |                                                     |
  | Multiprogramming OS (1960s):                        |
  | -> Multiple jobs held in memory simultaneously       |
  | -> Ran other jobs while one waited for I/O           |
  |                                                     |
  | Time-sharing OS (late 1960s):                       |
  | -> Multiple users accessed simultaneously via        |
  |    terminals                                         |
  | -> "Interactive" computer usage became possible      |
  +----------------------------------------------------+

Environments without an OS even today:
  - Arduino: A single program directly controls hardware
  - Bare-metal programming: Firmware for embedded devices
  - Bootloaders: GRUB, U-Boot (run before the OS starts)
  -> What you gain without an OS: Low latency, small footprint
  -> What you lose: Multitasking, memory protection, abstraction
```

---

## 2. Kernel and User Space

```
CPU operating modes (x86):

  Ring 0 (kernel mode):
  -> Direct access to all hardware
  -> Can execute privileged instructions
  -> OS kernel runs here

  Ring 3 (user mode):
  -> Cannot directly access hardware
  -> Privileged instructions cause exceptions (traps)
  -> Regular applications run here

  +---------------------------------------+
  |  User space (Ring 3)                  |
  |  +------+ +------+ +------+          |
  |  |Chrome| |VSCode| |Slack | ...      |
  |  +--+---+ +--+---+ +--+---+         |
  |     |        |        |              |
  |=====+========+========+=============|
  |     |  System call (boundary)        |
  |=====+========+========+=============|
  |  Kernel space (Ring 0)               |
  |  +------------------------------+   |
  |  | Process mgmt | Memory mgmt   |   |
  |  | File system  | Networking    |   |
  |  | Device drivers               |   |
  |  +------------------------------+   |
  |  Hardware                            |
  +---------------------------------------+

  Ring 1, 2:
  -> x86 has 4 rings, but most OSes use only
    Ring 0 (kernel) and Ring 3 (user)
  -> VMX root/non-root mode: Additional mode for
    virtualization (VT-x)
```

### 2.1 System Call (syscall) Details

```
System call (syscall):
  The only gateway to invoke kernel functionality
  from user space

  Steps for invoking a system call:
  +---------------------------------------------------+
  | 1. Application calls a library function            |
  |    Example: write(fd, buf, count)                  |
  |                                                    |
  | 2. C library (glibc) sets the syscall number in    |
  |    a register and executes the syscall/int 0x80    |
  |    instruction                                     |
  |                                                    |
  | 3. CPU switches to kernel mode                     |
  |    -> Ring 3 -> Ring 0                             |
  |    -> Stack also switches to kernel stack          |
  |                                                    |
  | 4. Kernel executes the syscall handler             |
  |    -> Kernel function such as sys_write()          |
  |                                                    |
  | 5. Result is stored in a register, and returns     |
  |    to user mode                                    |
  |    -> Ring 0 -> Ring 3                             |
  +---------------------------------------------------+

  Major system calls (Linux):

  Process management:
  +------------------------------------------------------+
  | fork()      -> Copy current process to create child   |
  | exec()      -> Replace current process with another   |
  |                program                                |
  | wait()      -> Wait for child process to terminate    |
  | exit()      -> Terminate the process                  |
  | getpid()    -> Get process ID                         |
  | kill()      -> Send a signal to a process             |
  | clone()     -> Create a thread (Linux-specific)       |
  +------------------------------------------------------+

  File operations:
  +------------------------------------------------------+
  | open()      -> Open a file                            |
  | read()      -> Read from a file                       |
  | write()     -> Write to a file                        |
  | close()     -> Close a file                           |
  | lseek()     -> Move file position                     |
  | stat()      -> Get file information                   |
  | mkdir()     -> Create a directory                     |
  | unlink()    -> Delete a file                          |
  +------------------------------------------------------+

  Memory management:
  +------------------------------------------------------+
  | mmap()      -> Map memory                             |
  | munmap()    -> Unmap memory                           |
  | brk()       -> Change heap size                       |
  | mprotect()  -> Change memory protection attributes    |
  | mlock()     -> Lock memory pages (prevent swap out)   |
  +------------------------------------------------------+

  Networking:
  +------------------------------------------------------+
  | socket()    -> Create a socket                        |
  | bind()      -> Bind an address to a socket            |
  | listen()    -> Start listening for connections         |
  | accept()    -> Accept a connection                    |
  | connect()   -> Connect to a server                    |
  | send()      -> Send data                              |
  | recv()      -> Receive data                           |
  +------------------------------------------------------+

  Cost:
  Switching from user mode to kernel mode is expensive
  (thousands of cycles)
  -> Reducing the number of system calls is key to
     performance

  Acceleration techniques:
  +------------------------------------------------------+
  | vDSO (virtual Dynamic Shared Object):                |
  | -> Virtual shared library provided by the kernel     |
  | -> Executes gettimeofday() etc. in user space        |
  | -> Completely avoids syscall overhead                 |
  |                                                      |
  | io_uring (Linux 5.1+):                               |
  | -> New interface for asynchronous I/O                 |
  | -> Shares ring buffer with the kernel                |
  | -> Dramatically reduces syscall count                |
  | -> Adopted by high-performance web servers and       |
  |    databases                                         |
  |                                                      |
  | Effect of vDSO + io_uring:                           |
  | Traditional: read() -> syscall -> kernel -> result   |
  | io_uring: Submit SQE -> kernel processes async       |
  |           -> Retrieve CQE                            |
  | -> Multiple I/O operations in a single syscall       |
  +------------------------------------------------------+
```

### 2.2 Tracing System Calls in Practice

```bash
# Observe system calls with strace (Linux)
# Observe syscalls when outputting "hello"

$ strace echo "hello"
execve("/usr/bin/echo", ["echo", "hello"], ...) = 0
brk(NULL)                               = 0x55a123456000
access("/etc/ld.so.preload", R_OK)      = -1 ENOENT
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
fstat(3, {st_mode=S_IFREG|0644, ...})   = 0
mmap(NULL, 76888, PROT_READ, MAP_PRIVATE, 3, 0) = 0x7f1234567000
close(3)                                = 0
# ... (loading shared libraries)
write(1, "hello\n", 6)                  = 6
close(1)                                = 0
close(2)                                = 0
exit_group(0)                           = ?
```

```c
// Example of directly invoking system calls in C

#include <unistd.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <stdio.h>

int main() {
    // Method 1: Via library function (normal)
    int fd = open("test.txt", O_RDONLY);

    // Method 2: Direct call via syscall()
    int fd2 = syscall(SYS_openat, AT_FDCWD, "test.txt", O_RDONLY);

    // Method 3: Inline assembly (x86_64)
    // Not normally used, but useful for understanding the mechanism
    long result;
    char *msg = "Hello from syscall!\n";
    __asm__ volatile (
        "mov $1, %%rax\n"    // syscall number: write = 1
        "mov $1, %%rdi\n"    // fd: stdout = 1
        "mov %1, %%rsi\n"    // buffer
        "mov $20, %%rdx\n"   // size
        "syscall\n"          // syscall instruction
        "mov %%rax, %0\n"
        : "=r" (result)
        : "r" (msg)
        : "rax", "rdi", "rsi", "rdx"
    );

    printf("syscall returned: %ld\n", result);
    return 0;
}
```

```python
# Verify system call behavior in Python
import os
import sys

# Check file descriptors
print(f"stdin:  fd={sys.stdin.fileno()}")    # 0
print(f"stdout: fd={sys.stdout.fileno()}")   # 1
print(f"stderr: fd={sys.stderr.fileno()}")   # 2

# os.open() internally calls the open() syscall
fd = os.open("test.txt", os.O_CREAT | os.O_WRONLY, 0o644)
os.write(fd, b"Hello from Python syscall!\n")
os.close(fd)

# Check the current syscall via /proc/self/syscall (Linux)
# Check open files via /proc/self/fd
try:
    fds = os.listdir("/proc/self/fd")
    print(f"Open file descriptors: {fds}")
except FileNotFoundError:
    print("Not running on Linux")

# Get process information
print(f"PID: {os.getpid()}")
print(f"PPID: {os.getppid()}")
print(f"UID: {os.getuid()}")
print(f"GID: {os.getgid()}")
```

---

## 3. Kernel Architecture

```
1. Monolithic kernel:
   All OS functions in a single large binary

   +----------------------------------+
   | Kernel space                      |
   | +----+----+----+----+----+       |
   | |Proc|Mem |FS  |Net |Dri |       |
   | |ess |ory |    |    |ver |       |
   | +----+----+----+----+----+       |
   | All run in the same address space |
   +----------------------------------+

   Pros: Fast (simple function calls)
   Cons: A single bug can crash the whole system; grows large
   Examples: Linux, FreeBSD

   Scale of the Linux kernel:
   +------------------------------------------+
   | Source code: ~30 million lines (2025)     |
   | Committers: Thousands                     |
   | Supported architectures: 30+             |
   | Device drivers: Over 60% of the kernel   |
   | Release cycle: Approximately every 9-10  |
   | weeks                                     |
   +------------------------------------------+

   Linux dynamic modules:
   -> Monolithic, but can dynamically load/unload modules
   -> Add device drivers etc. to the kernel as needed
   -> Managed with lsmod, modprobe, rmmod
   -> Stored in /lib/modules/<kernel-version>/

2. Microkernel:
   Only minimal functionality in the kernel;
   the rest runs in user space

   +----------------------------------+
   | User space                        |
   | +----+ +----+ +----+ +----+      |
   | |FS  | |Net | |Dri | |App |      |
   | |Srv || |Srv || |ver| |    |      |
   | +--+-+ +--+-+ +--+-+ +--+-+      |
   |---|------|------|------|-----     |
   | Kernel: IPC + Scheduling          |
   |         + Memory mgmt (minimal)   |
   +----------------------------------+

   Pros: Stability (if a server crashes, the kernel survives)
   Cons: IPC (inter-process communication) overhead
   Examples: MINIX, QNX, seL4, GNU Hurd

   Features of seL4:
   +----------------------------------------------+
   | - World's first formally verified OS kernel   |
   | - ~8,700 lines of C + 600 lines of assembly  |
   | - Mathematically proven to be correct          |
   | - Used in defense, aviation, automotive fields|
   | - Guaranteed to never produce runtime errors   |
   +----------------------------------------------+

3. Hybrid kernel:
   Monolithic performance + microkernel design philosophy

   Examples: Windows NT, macOS (XNU), DragonFly BSD
   -> In practice, most implementations are close to
      monolithic

   macOS XNU kernel:
   +------------------------------------------------------+
   | XNU = "X is Not Unix"                                 |
   |                                                       |
   | Mach microkernel (message passing, VM)                |
   | + BSD (POSIX API, VFS, networking)                    |
   | + I/O Kit (object-oriented device drivers)            |
   |                                                       |
   | -> Mach's design philosophy + BSD's practicality      |
   |    = Hybrid                                           |
   | -> Open source (darwin-xnu)                           |
   +------------------------------------------------------+

   Windows NT kernel:
   +------------------------------------------------------+
   | HAL (Hardware Abstraction Layer)                       |
   | + Microkernel (scheduler, interrupt handling)          |
   | + Executive (I/O, VM, process management)              |
   | + Win32 subsystem (GUI, API)                           |
   | + WSL2 subsystem (Linux compatibility)                 |
   |                                                       |
   | -> Designed by Dave Cutler (VMS designer)              |
   | -> Initially microkernel-oriented, but functions       |
   |    were pulled into the kernel for performance         |
   +------------------------------------------------------+

4. Unikernel:
   Packs only the app + required OS functions into
   a single image

   +------------------+
   | App + OS funcs   |  <- Single binary
   +------------------+

   Pros: Minimal size, fast boot, smallest attack surface
   Cons: Single app only, difficult to debug
   Examples: MirageOS, Unikraft, OSv

   Use cases:
   - Cloud microservices (minimal footprint)
   - NFV (Network Function Virtualization)
   - CDN edge nodes
   - IoT devices

5. Exokernel:
   Kernel only handles resource protection;
   management is delegated to the application

   +----------------------------------+
   | App + LibOS (implements FS, Net) |
   | -------------------------------- |
   | Exokernel: Resource allocation   |
   | and protection only              |
   +----------------------------------+

   -> Apps can customize OS functions
   -> Still in research, but influences containers
      and unikernels

Comparison:
+--------------+----------+----------+----------+-----------+
| Type         | Perf.    | Stabil.  | Example  | Code size |
+--------------+----------+----------+----------+-----------+
| Monolithic   | Excellent| Fair     | Linux    | Tens of M |
| Micro        | Fair     | Excellent| QNX      | Tens of K |
| Hybrid       | Good     | Good     | Windows  | Millions  |
| Unikernel    | Excellent| Good     | Cloud    | Thousands |
| Exokernel    | Excellent| Fair     | Research | Thousands |
+--------------+----------+----------+----------+-----------+
```

---

## 4. Major OS Families

```
Unix family:
  1969: Unix (AT&T Bell Labs -- Thompson, Ritchie)
    +-- BSD family: FreeBSD, OpenBSD, NetBSD
    |   +-- macOS / iOS (Darwin = Mach + FreeBSD)
    +-- System V family
    |   +-- Solaris, AIX, HP-UX
    +-- Linux (1991, Linus Torvalds)
        +-- Debian family: Ubuntu, Linux Mint, Raspberry Pi OS
        +-- Red Hat family: RHEL, CentOS Stream, Fedora, Rocky, AlmaLinux
        +-- Arch family: Arch Linux, Manjaro, EndeavourOS
        +-- SUSE family: openSUSE, SLES
        +-- Android (Linux kernel + Dalvik/ART)
        +-- Chrome OS (Linux kernel + Chrome browser)
        +-- SteamOS (Linux kernel + Steam)

Windows family:
  MS-DOS (1981)
    +-- Windows 3.1 -> 95 -> 98 -> Me (DOS-based)
  Windows NT (1993)
    +-- NT -> 2000 -> XP -> Vista -> 7 -> 8 -> 10 -> 11

Others:
  z/OS: IBM mainframe (runs COBOL assets)
  VxWorks: Embedded RTOS (also used on Mars rovers)
  FreeRTOS: Lightweight RTOS for IoT (managed by AWS)
  Zephyr: RTOS for IoT (Linux Foundation)
  Fuchsia: Google's next-gen OS (Zircon microkernel)
  HarmonyOS: Huawei's OS (microkernel)
  Redox: Microkernel OS written in Rust

Current market share (2025 estimate):
  Desktop:        Windows 72%, macOS 16%, Linux 4%, Chrome OS 3%
  Server:         Linux 80%+, Windows 15%
  Mobile:         Android 72%, iOS 27%
  Supercomputers: Linux 100% (TOP500)
  Embedded/IoT:   FreeRTOS, Linux, VxWorks, Zephyr are dominant
  Containers:     Linux 99%+ (Docker/K8s depend on the Linux kernel)
```

### 4.1 Choosing a Linux Distribution

```
Recommended distributions by use case:

  Server use:
  +----------------------------------------------------+
  | RHEL/Rocky Linux: Enterprise, long-term support     |
  | Ubuntu Server: Cloud, beginner-friendly             |
  | Debian: Stability-focused, server staple            |
  | Amazon Linux: Optimized for AWS                     |
  | Alpine Linux: For containers (lightweight, <5MB)    |
  +----------------------------------------------------+

  Desktop use:
  +----------------------------------------------------+
  | Ubuntu Desktop: For beginners, abundant resources   |
  | Fedora: Cutting-edge technology, GNOME              |
  | Linux Mint: Best for transitioning from Windows     |
  | Arch Linux: Customization-focused, for advanced     |
  |   users                                             |
  | Pop!_OS: For developers, NVIDIA GPU support         |
  +----------------------------------------------------+

  Package management comparison:
  +-----------+----------+----------------------+
  | Family    | Tool     | Example command       |
  +-----------+----------+----------------------+
  | Debian    | apt      | apt install nginx    |
  | Red Hat   | dnf/yum  | dnf install nginx    |
  | Arch      | pacman   | pacman -S nginx      |
  | SUSE      | zypper   | zypper install nginx |
  | Alpine    | apk      | apk add nginx        |
  | Universal | snap     | snap install firefox |
  | Universal | flatpak  | flatpak install ...  |
  +-----------+----------+----------------------+
```

---

## 5. OS Abstraction

```
Major abstractions provided by an OS:

  Physical Resource   ->   OS Abstraction
  ----------------------------------------
  CPU                 ->   Process/Thread
  Physical memory     ->   Virtual address space
  Disk sectors        ->   Files/Directories
  Network             ->   Sockets
  Display             ->   Windows
  Timer               ->   Time API

  Benefits of abstraction:
  +------------------------------------------------------+
  | 1. Portability: Same program runs on different HW     |
  | 2. Simplicity: Complex operations via simple APIs     |
  | 3. Isolation: Prevents interference between processes |
  | 4. Efficiency: Resources are automatically optimally  |
  |    distributed                                        |
  | 5. Security: Enforces access control                  |
  +------------------------------------------------------+

  "Everything is a file" (Unix philosophy):
  /dev/sda        -> Disk
  /dev/null       -> Bit bucket
  /proc/cpuinfo   -> CPU information
  /dev/urandom    -> Random numbers
  /dev/tty        -> Terminal
  /sys/class/net/ -> Network interface information
  /dev/video0     -> Webcam
  -> Everything can be treated uniformly as files
  -> Unified with 4 operations: read/write/open/close

  Linux virtual file systems:
  +------------------------------------------------------+
  | /proc:                                                |
  |   /proc/<pid>/status    -> Process state              |
  |   /proc/<pid>/maps      -> Memory mappings            |
  |   /proc/<pid>/fd/       -> Open files                 |
  |   /proc/meminfo         -> Memory usage               |
  |   /proc/cpuinfo         -> CPU information            |
  |   /proc/loadavg         -> Load average               |
  |   /proc/net/tcp         -> TCP connection info        |
  |                                                       |
  | /sys:                                                 |
  |   /sys/class/           -> Device classes             |
  |   /sys/block/           -> Block devices              |
  |   /sys/fs/              -> File system information    |
  |   /sys/kernel/          -> Kernel parameters          |
  |                                                       |
  | /dev:                                                 |
  |   /dev/sd*              -> SCSI disks                 |
  |   /dev/nvme*            -> NVMe devices               |
  |   /dev/tty*             -> Terminal devices           |
  |   /dev/loop*            -> Loopback devices           |
  +------------------------------------------------------+

  POSIX (Portable Operating System Interface):
  Standard API specification for Unix-like OSes
  -> Programs conforming to POSIX have high portability
  -> Linux, macOS, BSD are broadly POSIX-compliant
  -> Windows provides Linux-compatible environment via WSL2

  What POSIX defines:
  +------------------------------------------+
  | - System call interface                   |
  | - Basic commands (ls, grep, awk, etc.)   |
  | - Shell language (sh)                     |
  | - Thread API (pthread)                    |
  | - Regular expressions                     |
  | - File permission model                   |
  | - Signal handling                         |
  +------------------------------------------+
```

### 5.1 Unix Philosophy in Practice

```
The essence of Unix philosophy (Doug McIlroy):

  1. "Write programs that do one thing and do it well."
  2. "Write programs to work together."
  3. "Build a prototype as soon as possible, discard
     clumsy parts, and rebuild."

  The power of pipes:
  +------------------------------------------------------+
  | # Tally request counts per IP address from access log |
  | cat access.log | awk '{print $1}' | sort | uniq -c | sort -rn | head -10
  |                                                      |
  | Role of each command:                                |
  | cat: Output file contents                            |
  | awk: Extract 1st field (IP address)                  |
  | sort: Sort                                           |
  | uniq -c: Count duplicates                            |
  | sort -rn: Reverse numeric sort                       |
  | head -10: Show top 10                                |
  |                                                      |
  | -> Complex log analysis achieved by combining        |
  |    6 small programs                                  |
  +------------------------------------------------------+

  Plan 9 (Unix successor research OS) innovations:
  -> Extended "everything is a file" to the network
  -> Network connections as file operations via /net/tcp
  -> Processes as file operations via /proc
  -> Mount remote resources via the 9P protocol
  -> This philosophy was inherited by Linux's /proc, /sys
```

---

## 6. OS, Containers, and Virtualization

```
Modern OS features: Container technology

  Linux kernel features that form the foundation
  of containers:

  1. Namespace:
     Isolates resource visibility
     +------------------------------------------+
     | PID namespace:     Process ID isolation   |
     | Network namespace: Network isolation       |
     | Mount namespace:   File system isolation   |
     | UTS namespace:     Hostname isolation      |
     | User namespace:    UID/GID isolation       |
     | IPC namespace:     IPC isolation           |
     | Cgroup namespace:  cgroup isolation        |
     | Time namespace:    Time isolation (5.6+)   |
     +------------------------------------------+

  2. Cgroups (Control Groups):
     Limits resource usage
     +------------------------------------------+
     | CPU: Set usage cap                        |
     | Memory: Set memory usage limit            |
     | I/O: Limit disk I/O bandwidth             |
     | PID: Set process count limit              |
     | -> Used for Docker container resource      |
     |    limits                                 |
     +------------------------------------------+

  3. Union FS (OverlayFS, etc.):
     Layer-based file system
     +------------------------------------------+
     | Read-write layer (container-specific)     |
     | ---------------------                     |
     | Read-only layer 3 (app)                   |
     | ---------------------                     |
     | Read-only layer 2 (libraries)             |
     | ---------------------                     |
     | Read-only layer 1 (base OS)               |
     | -> Docker image layer structure            |
     +------------------------------------------+

  Virtual machine vs. Container:
  +-----------------+------------------------------+
  | Virtual Machine | Container                     |
  +-----------------+------------------------------+
  | Full guest OS   | Shares host OS kernel         |
  | Boot: tens of s | Boot: milliseconds to seconds |
  | Size: GB        | Size: MB                      |
  | High overhead   | Low overhead                  |
  | Isolation: high | Isolation: medium (shared      |
  |                 | kernel)                        |
  | Use: different  | Use: App isolation on the      |
  | OSes            | same OS                        |
  +-----------------+------------------------------+
```

```bash
# Experience container principles manually (Linux)

# 1. Start a process in a new namespace
sudo unshare --pid --fork --mount-proc /bin/bash

# 2. Running ps shows only your own process
ps aux
# PID 1 is bash (isolated)

# 3. Set a memory limit with cgroups
sudo mkdir /sys/fs/cgroup/memory/mycontainer
echo 100M > /sys/fs/cgroup/memory/mycontainer/memory.limit_in_bytes
echo $$ > /sys/fs/cgroup/memory/mycontainer/cgroup.procs

# 4. Create a network namespace
sudo ip netns add testns
sudo ip netns exec testns ip addr
# -> An isolated network with only the loopback interface
```

---

## Hands-On Exercises

### Exercise 1: [Basics] -- Tracing System Calls

```bash
# Observe system calls with strace (Linux)
strace ls /tmp 2>&1 | head -30

# On macOS, use dtruss
sudo dtruss ls /tmp 2>&1 | head -30

# Observation points:
# 1. execve() -> Program start
# 2. openat() -> Opening a file
# 3. getdents() -> Reading directory entries
# 4. write() -> Outputting results
# 5. close() -> Closing a file

# Task: Compare the syscalls of the following commands
# - echo "hello" vs printf "hello"
# - cat file vs less file

# Advanced task: Gather syscall statistics
strace -c ls /tmp 2>&1
# -> Tallies call count and time for each syscall

# Filter specific syscalls only
strace -e trace=open,read,write cat /etc/passwd 2>&1
```

### Exercise 2: [Intermediate] -- Kernel Module Concepts

```bash
# Inspect Linux kernel modules

# List currently loaded modules
lsmod

# Information about a specific module
modinfo ext4

# Module dependencies
modprobe --show-depends usb_storage

# Relationship between /proc/modules and lsmod
cat /proc/modules | head -10
# -> lsmod simply formats and displays /proc/modules
```

```c
// Minimal Linux kernel module (educational)
// Filename: hello_module.c

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Student");
MODULE_DESCRIPTION("Hello World Kernel Module");

static int __init hello_init(void) {
    printk(KERN_INFO "Hello from kernel module!\n");
    return 0;
}

static void __exit hello_exit(void) {
    printk(KERN_INFO "Goodbye from kernel module!\n");
}

module_init(hello_init);
module_exit(hello_exit);

// Makefile:
// obj-m += hello_module.o
// all:
//     make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules
// clean:
//     make -C /lib/modules/$(shell uname -r)/build M=$(PWD) clean

// Build and load:
// make
// sudo insmod hello_module.ko
// dmesg | tail  -> "Hello from kernel module!"
// sudo rmmod hello_module
// dmesg | tail  -> "Goodbye from kernel module!"
```

### Exercise 3: [Advanced] -- Comparing OS Designs

```
Choose the optimal OS architecture for the following
requirements and explain your reasoning:

1. Automotive control system (brakes, steering)
   -> Hint: Real-time performance, safety, and
      certification are critical
   -> QNX (microkernel) or seL4 (formally verified)

2. Web server (high-volume request processing)
   -> Hint: Performance, ecosystem, and operability
      are critical
   -> Linux (monolithic) + io_uring

3. IoT sensor device (battery-powered, minimal resources)
   -> Hint: Footprint and power consumption are critical
   -> FreeRTOS or Zephyr

4. Cloud FaaS (Function as a Service) platform
   -> Hint: Boot speed, isolation, and efficiency
      are critical
   -> Unikernel or Firecracker (microVM)

For each case, discuss whether "monolithic / micro /
unikernel" is appropriate from the perspectives of
performance, safety, and development cost.

Example evaluation criteria:
+----------------------+----------+----------+----------+
| Criteria             | Case 1   | Case 2   | Case 3   |
+----------------------+----------+----------+----------+
| Performance          |          |          |          |
| Safety/Reliability   |          |          |          |
| Development cost     |          |          |          |
| Maintainability      |          |          |          |
| Boot speed           |          |          |          |
| Memory footprint     |          |          |          |
| Ease of certification|          |          |          |
+----------------------+----------+----------+----------+
```

### Exercise 4: [Advanced] -- Exploring OS Internals

```bash
# Explore the internal state of Linux

# 1. CPU information
cat /proc/cpuinfo | grep "model name" | head -1
nproc  # Number of CPU cores

# 2. Memory information
free -h
cat /proc/meminfo | head -10

# 3. Process information
ps aux --sort=-%mem | head -10  # Top 10 by memory usage
ps aux --sort=-%cpu | head -10  # Top 10 by CPU usage

# 4. File system information
df -h            # Disk usage
mount | head -20 # Mount information

# 5. Network information
ss -tlnp        # Listening ports (Linux)
# netstat -tlnp  # Legacy command

# 6. Kernel information
uname -a         # Kernel version
cat /proc/version

# 7. System boot time
uptime
who -b

# Task: Create a script that collects the above
#       information and generates a "server health
#       check report"
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify executing user's permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify where the error occurred
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging or a debugger to verify hypotheses
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

### Diagnosing Performance Issues

Steps for diagnosing performance problems:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O conditions
4. **Check concurrent connections**: Examine connection pool status

| Problem Type | Diagnostic Tool | Countermeasure |
|-------------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

Summary of criteria for making technology choices:

| Criterion | Prioritize when | Can compromise when |
|-----------|----------------|-------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+-----------------------------------------------------+
|           Architecture Selection Flow                |
+-----------------------------------------------------+
|                                                     |
|  (1) Team size?                                     |
|    +-- Small (1-5) -> Monolith                      |
|    +-- Large (10+) -> Go to (2)                     |
|                                                     |
|  (2) Deploy frequency?                              |
|    +-- Weekly or less -> Monolith + module split     |
|    +-- Daily/multiple -> Go to (3)                   |
|                                                     |
|  (3) Team independence?                             |
|    +-- High -> Microservices                         |
|    +-- Medium -> Modular monolith                    |
|                                                     |
+-----------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs. long-term cost**
- A quick approach in the short term can become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and delays projects

**2. Consistency vs. flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit solutions but increases operational costs

**3. Level of abstraction**
- Higher abstraction increases reusability but can make debugging harder
- Lower abstraction is more intuitive but prone to code duplication

```python
# Design decision record template
class ArchitectureDecisionRecord:
    """Create an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and problem"""
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

## Practical Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** Need to release a product quickly with limited resources

**Approach:**
- Choose a simple architecture
- Focus on the minimum necessary features
- Automated tests only for critical paths
- Introduce monitoring early

**Lessons learned:**
- Don't aim for perfection (YAGNI principle)
- Get user feedback early
- Manage technical debt consciously

### Scenario 2: Legacy System Modernization

**Situation:** Incrementally modernizing a system that has been in operation for over 10 years

**Approach:**
- Migrate incrementally using the Strangler Fig pattern
- Create Characterization Tests first if no existing tests exist
- Use an API gateway to allow old and new systems to coexist
- Migrate data incrementally

| Phase | Work | Estimated Duration | Risk |
|-------|------|-------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration start | Sequential migration from peripheral features | 3-6 months | Medium |
| 4. Core migration | Migration of core features | 6-12 months | High |
| 5. Completion | Decommission legacy system | 2-4 weeks | Medium |

### Scenario 3: Large Team Development

**Situation:** 50+ engineers developing the same product

**Approach:**
- Clarify boundaries with domain-driven design
- Set ownership per team
- Manage shared libraries using the Inner Source model
- Design API-first to minimize inter-team dependencies

```python
# API contract definition between teams
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """API contract between teams"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Verify SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage example
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: Performance-Critical Systems

**Situation:** A system requiring millisecond-level response times

**Optimization points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Effect | Implementation Cost | Application |
|-------------------|--------|-------------------|-------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy workloads |
| DB optimization | High | High | Slow queries |
| Code optimization | Low-Med | High | CPU-bound workloads |

---

## Team Development Practices

### Code Review Checklist

Points to check in code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] No performance impact
- [ ] No security issues
- [ ] Documentation has been updated

### Knowledge Sharing Best Practices

| Method | Frequency | Audience | Effect |
|--------|-----------|----------|--------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talk | Weekly | Entire team | Horizontal knowledge sharing |
| ADR (Decision Record) | As needed | Future members | Decision transparency |
| Retrospective | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Critical design | Building consensus |

### Managing Technical Debt

```
Priority matrix:

        High impact
          |
    +-----+-----+
    | Plan | Fix  |
    | for  | imme-|
    | later| diate|
    |      | ly   |
    +-----+-----+
    | Log  | Next |
    | only | Sprint|
    |      |      |
    +-----+-----+
          |
        Low impact
    Low frequency  High frequency
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------------|-----------|---------------|-----------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor auth, strengthened session management | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Configuration issues | Medium | Security headers, principle of least privilege | Configuration scanning |
| Insufficient logging | Medium | Structured logs, audit trails | Log analysis |

### Secure Coding Best Practices

```python
# Secure coding example
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """Security utilities"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """Generate a cryptographically secure token"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """Hash a password"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """Verify a password"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """Sanitize input"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# Usage example
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### Security Checklist

- [ ] All input values are validated
- [ ] Sensitive information is not output to logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scanning has been performed
- [ ] Error messages do not contain internal information
---

## FAQ

### Q1: Is Linux a Unix?

Strictly speaking, "no." Linux was written from scratch without using Unix source code; it is a "Unix-compatible" OS. No AT&T Unix license is required. However, because it is POSIX-compatible and follows Unix philosophy, it is called "Unix-like." The OSes that have obtained official UNIX certification (Single UNIX Specification compliance) include macOS, Solaris, and AIX. Linux has not obtained UNIX certification (it has not applied for it).

### Q2: What is the difference between a kernel and an OS?

The kernel is the core part of an OS (hardware management, process management, etc.). An OS is the collective term for the kernel + shell + utilities + libraries. Strictly speaking, Linux is a kernel name, and distributions like Ubuntu are the complete OS. The term GNU/Linux means the combination of GNU project utilities (gcc, coreutils, bash, etc.) and the Linux kernel.

### Q3: Why is Linux overwhelmingly dominant on servers?

1. Free (zero licensing cost)
2. Open source (freely customizable)
3. Stability (can run for years without downtime)
4. Command-line centric (ideal for remote management)
5. Rich community and ecosystem
6. Container technology (Docker/K8s) is built for Linux
7. Cloud providers (AWS, GCP, Azure) offer Linux as standard
8. Lightweight (no GUI needed, maximizing server resources)

### Q4: What is a real-time OS?

A real-time OS (RTOS) is an OS that guarantees processing will be completed within a specified time. There are hard real-time (strict deadlines: medical devices, automotive controls) and soft real-time (best effort: multimedia playback) types. Linux itself is a general-purpose OS, but applying the PREEMPT_RT patch can achieve soft real-time performance.

### Q5: How does WSL2 work?

WSL2 (Windows Subsystem for Linux 2) is a mechanism for running a full Linux kernel on Windows. It uses Hyper-V virtualization technology to launch a lightweight Linux VM and provides seamless integration with Windows (file sharing, network sharing, GPU sharing). Unlike the original WSL1 (syscall translation approach), a full Linux kernel runs, so all Linux programs work.

### Q6: Where should I start to build my own OS?

1. **Learn OS theory**: "Operating Systems: Three Easy Pieces" (free online textbook)
2. **xv6**: MIT's educational OS (simple Unix implementation, x86/RISC-V)
3. **OSDev Wiki**: Community resource for OS development
4. **Writing an OS in Rust**: Blog series by Philipp Oppermann
5. **"30 Days to Make Your Own OS"**: By Hidemi Kawai (Japanese, x86)

---

## Summary

| Concept | Key Point |
|---------|----------|
| Role of an OS | Resource management + hardware abstraction |
| Kernel | Runs in Ring 0. Can access all hardware |
| System call | The only gateway from user space to kernel. Costs thousands of cycles |
| Architecture | Monolithic (Linux) vs. Micro (QNX) vs. Hybrid (Windows) |
| Unix philosophy | Everything is a file. Combine small tools |
| POSIX | Standard API spec for Unix-like OSes. Key to portability |
| Containers | Implemented with Namespace + Cgroups + Union FS |
| Virtualization | VM (full isolation) vs. Container (lightweight isolation) |

---

## Recommended Next Guides

---

## References
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Wiley, 2018.
2. Tanenbaum, A. "Modern Operating Systems." 4th Ed, Pearson, 2014.
3. Arpaci-Dusseau, R. & A. "Operating Systems: Three Easy Pieces." 2018.
4. Love, R. "Linux Kernel Development." 3rd Ed, Addison-Wesley, 2010.
5. Kerrisk, M. "The Linux Programming Interface." No Starch Press, 2010.
6. McKusick, M. et al. "The Design and Implementation of the FreeBSD Operating System." 2nd Ed, 2014.
7. Russinovich, M. et al. "Windows Internals." 7th Ed, Microsoft Press, 2017.
8. Klein, G. et al. "seL4: Formal Verification of an OS Kernel." SOSP, 2009.
