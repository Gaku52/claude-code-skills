# History and Evolution of Operating Systems

> The history of operating systems is a history of "layered abstraction" -- a record of the struggle to free humans from the complexity of hardware.

## What You Will Learn in This Chapter

- [ ] Learn the major milestones in the evolution of operating systems
- [ ] Understand the progression from batch processing to time-sharing to GUI
- [ ] Trace the roots of modern operating systems
- [ ] Understand why the design philosophies of each era emerged
- [ ] Gain detailed knowledge of the historical background of Unix/Linux/Windows


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [What is an OS](./00-what-is-os.md)

---

## 1. Timeline of OS Evolution

### 1.1 1940s-1950s: The Pre-OS Era

```
The Dawn of Computing:

  1943-1945: ENIAC (Electronic Numerical Integrator and Computer)
  +----------------------------------------------+
  | - World's first general-purpose electronic    |
  |   computer (US Army)                          |
  | - Weighed 30 tons, 18,000 vacuum tubes        |
  | - Programming = physically rewiring           |
  | - The concept of an OS did not exist           |
  | - Operators (many were female mathematicians)  |
  |   operated it manually                         |
  +----------------------------------------------+

  1949: EDSAC (Electronic Delay Storage Automatic Calculator)
  -> Practical implementation of stored-program concept
  -> Programs stored in memory and executed
  -> Implementation of von Neumann architecture

  Typical computer usage in the early 1950s:
  +----------------------------------------------+
  | 1. Programmer writes program on punch cards    |
  | 2. Hands the card deck to the operator         |
  | 3. Operator feeds cards into the reader         |
  | 4. Computer executes one job                    |
  | 5. Results printed on printer                   |
  | 6. Preparation for next job (tape swap, etc.)   |
  |    takes several minutes                        |
  | 7. During this time, the expensive CPU          |
  |    is completely idle                            |
  |                                                |
  | Problem: Extremely low CPU utilization           |
  | -> An expensive machine worth $1/second sits idle|
  | -> Human operations are the bottleneck           |
  +----------------------------------------------+

  The first OS was born to solve this problem.
```

### 1.2 Late 1950s: The Era of Batch Processing

```
1950s: The Era of Batch Processing
  +------------------------------------------+
  | Punch cards -> Computer -> Output results  |
  | Humans feed programs one by one             |
  | No OS (operator manages manually)           |
  |                                            |
  | 1956: GM-NAA I/O (the first OS)           |
  | -> Batch processing system for IBM 704     |
  | -> Jointly developed by General Motors      |
  |    and North American Aviation              |
  | -> Automatically executes jobs sequentially |
  | -> Switches jobs without human intervention |
  +------------------------------------------+

  How a batch processing OS works:
  +----------------------------------------------+
  | +---------+  +---------+  +---------+        |
  | | Job 1   |->| Job 2   |->| Job 3   |-> ...  |
  | |(FORTRAN)|  |(COBOL)  |  |(Assemb.)|        |
  | +---------+  +---------+  +---------+        |
  |                                              |
  | Resident Monitor:                            |
  | - A small program residing in memory          |
  | - Automates job loading and execution         |
  | - Jobs described in JCL (Job Control Language)|
  |                                              |
  | JCL example:                                 |
  | $JOB                                         |
  | $FORTRAN                                     |
  | (FORTRAN source code)                        |
  | $LOAD                                        |
  | $RUN                                         |
  | (input data)                                 |
  | $END                                         |
  +----------------------------------------------+

  1959: SHARE Operating System (SOS)
  -> Batch processing OS for IBM 709
  -> Developed by user community SHARE

  Limitations of batch processing OS:
  - CPU completely idle during I/O wait
  - No interactive operation possible
  - Debugging extremely difficult (results returned the next day)
  -> Multiprogramming was developed to solve this
```

### 1.3 1960s: Multiprogramming and Time-Sharing

```
1960s: Multiprogramming and Time-Sharing
  +------------------------------------------+
  | 1961: CTSS (Compatible Time-Sharing System)|
  | -> MIT. The first time-sharing OS          |
  | -> Multiple users using a computer         |
  |    simultaneously                          |
  | -> Developed by Fernando Corbato           |
  | -> Corbato received the Turing Award in 2014|
  |                                            |
  | Innovations of CTSS:                       |
  | - Interactive programming from terminals    |
  | - Errors could be fixed on the spot         |
  | - Prototype of file systems                 |
  | - Prototype of email systems                |
  +------------------------------------------+

  Principle of multiprogramming:
  +----------------------------------------------+
  | Problem: While Job A waits for I/O,           |
  |          the CPU does nothing                  |
  |                                              |
  | Traditional (single programming):             |
  | Job A: ######______######______              |
  | CPU:   ######______######______              |
  | (#=I/O wait  _=CPU idle)                      |
  |                                              |
  | Multiprogramming:                             |
  | Job A: ######______######______              |
  | Job B: ______######______######              |
  | CPU:   ######################## <- Always busy!|
  |                                              |
  | -> CPU utilization dramatically improved       |
  | -> Multiple jobs held in memory simultaneously |
  | -> Memory protection becomes necessary         |
  |    (isolation between jobs)                    |
  +----------------------------------------------+

  1964: Multics (MULTiplexed Information and Computing Service)
  +----------------------------------------------------+
  | -> Joint project by MIT + Bell Labs + GE            |
  | -> Too ambitious, commercially unsuccessful         |
  | -> However, had enormous influence on Unix           |
  |                                                    |
  | Innovative concepts of Multics:                     |
  | - Virtual memory: segmentation + paging              |
  | - Dynamic linking: prototype of shared libraries     |
  | - Hierarchical file system: tree structure of dirs   |
  | - Access Control Lists (ACL): security model         |
  | - Ring-based protection: Ring 0-7 privilege levels   |
  | - PLAN language: PL/I-based system language          |
  |                                                    |
  | Reasons for failure:                                |
  | - Excessively complex design                         |
  | - Insufficient hardware performance (GE 645)         |
  | - Project management issues                          |
  | - Bell Labs withdrew in 1969                         |
  |   -> Thompson & Ritchie envisioned a "simple Multics"|
  |   -> This led to the birth of Unix                   |
  +----------------------------------------------------+

  1964: IBM OS/360
  +------------------------------------------+
  | -> First general-purpose OS (common across |
  |    multiple machine models)                |
  | -> Ran on the entire IBM System/360 family |
  | -> Subject of Fred Brooks' "Mythical       |
  |    Man-Month"                              |
  |                                            |
  | Lessons from OS/360:                       |
  | - "The Mythical Man-Month": Adding people  |
  |   to a late project makes it later         |
  | - Millions of lines of code, thousands     |
  |   of developers                            |
  | - Made the importance of software           |
  |   engineering recognized                    |
  | - Full of bugs, but commercially            |
  |   very successful                           |
  |                                            |
  | IBM's compatibility strategy:              |
  | - One OS covering small to large machines   |
  | - Established the concept of "software      |
  |   compatibility"                            |
  | - Predecessor of today's x86 compatibility  |
  +------------------------------------------+

  1965: THE (Technische Hogeschool Eindhoven)
  -> Hierarchically structured OS by Dijkstra
  -> Invention of semaphores (synchronization primitives)
  -> Advocacy for structured programming
  -> The first example of treating OS design as software engineering
```

### 1.4 1970s: The Birth of Unix

```
1970s: The Birth of Unix
  +------------------------------------------+
  | 1969: Unix (Ken Thompson & Dennis Ritchie) |
  | -> Pursued "simplicity" as a reaction to   |
  |    Multics                                 |
  | -> Initially written in assembly for PDP-7  |
  |                                            |
  | 1971: Unix Version 1 (PDP-11)             |
  | -> Deployed to Bell Labs' patent department |
  |    for text processing                      |
  |                                            |
  | 1973: Unix rewritten in C                  |
  | -> Revolutionary idea: "writing an OS in    |
  |    a high-level language"                   |
  | -> Portability dramatically improved         |
  | -> Dennis Ritchie designed the C language    |
  |                                            |
  | 1974: "The UNIX Time-Sharing System" paper  |
  | -> Published in ACM Communications          |
  | -> Huge impact on academia                   |
  |                                            |
  | Unix Philosophy:                           |
  | 1. Do one thing well                        |
  | 2. Communicate via text streams             |
  | 3. Combine small tools                      |
  | 4. Build prototypes as early as possible     |
  | 5. Prioritize portability                    |
  +------------------------------------------+

  The Unix Split (Unix Wars):
  +----------------------------------------------------+
  | 1977: BSD (Berkeley Software Distribution)          |
  | -> University of California, Berkeley               |
  | -> Led by Bill Joy (later Sun co-founder)           |
  | -> TCP/IP networking implementation (DARPA funded)  |
  | -> Developed vi editor and csh shell                |
  | -> Improved virtual memory (paged VM)               |
  |                                                    |
  | 1983: System V Release 1 (AT&T)                    |
  | -> AT&T's commercial Unix                           |
  | -> System V vs BSD "Unix Wars"                      |
  |                                                    |
  | Differences between the two lineages:               |
  | +--------------+------------------+                |
  | | BSD          | System V         |                |
  | +--------------+------------------+                |
  | | TCP/IP net   | STREAMS          |                |
  | | csh          | sh (Bourne Shell)|                |
  | | vi           | ed               |                |
  | | Improved VM  | IPC (msg, shm)   |                |
  | | BSD License  | Commercial License|               |
  | +--------------+------------------+                |
  |                                                    |
  | -> Eventually standardized by POSIX (1988)          |
  | -> Integrated the best parts of both                |
  +----------------------------------------------------+

  1979: Unix Version 7
  -> Called "the last true Unix"
  -> The most influential version
  -> Included tools like awk, make, tar, cpio
  -> Became the foundation for many Unix-like OSes
```

### 1.5 1980s: PCs and GUIs

```
1980s: PCs and GUIs
  +------------------------------------------+
  | 1981: MS-DOS (Microsoft)                  |
  | -> For IBM PC. Command line                |
  | -> Acquired QDOS developed by Tim Paterson |
  | -> CP/M-compatible OS (Gary Kildall)       |
  | -> Single-tasking, no memory protection     |
  | -> However, became dominant along with      |
  |    the spread of IBM PCs                    |
  |                                            |
  | 1983: GNU Project (Richard Stallman)       |
  | -> Aimed to create a free Unix-compatible OS|
  | -> GNU Compiler (gcc)                       |
  | -> GNU Utilities (coreutils)                |
  | -> Emacs editor                             |
  | -> Creation of the GPL license              |
  | -> Kernel (GNU Hurd) was never completed    |
  | -> Combined with Linux kernel to form       |
  |    GNU/Linux                                |
  |                                            |
  | 1984: Macintosh (Apple)                    |
  | -> First commercially successful GUI OS     |
  | -> Inspired by Xerox PARC's Alto            |
  | -> Steve Jobs visited PARC and got the idea |
  | -> Mouse operation, windows, icons          |
  | -> 128KB memory, 9-inch monochrome display  |
  | -> "1984" Super Bowl commercial             |
  |                                            |
  | 1987: MINIX (Andrew Tanenbaum)             |
  | -> Educational microkernel OS               |
  | -> Appendix to the textbook "Operating      |
  |    Systems: Design and Implementation"      |
  | -> Influenced Linus                         |
  | -> Tanenbaum-Torvalds debate (1992):       |
  |    "Linux is monolithic and obsolete"        |
  |    -> Ultimately Linux conquered the world   |
  +------------------------------------------+

  Genealogy of GUIs:
  +----------------------------------------------------+
  | 1973: Xerox Alto                                    |
  | -> World's first GUI computer                        |
  | -> Mouse, windows, WYSIWYG, Ethernet                |
  | -> Not commercialized (research use only)            |
  |                                                    |
  | 1981: Xerox Star (8010)                             |
  | -> Commercial version of Alto, but too expensive     |
  |    and failed                                        |
  |                                                    |
  | 1984: Apple Macintosh                               |
  | -> First PC to popularize GUI for consumers          |
  |                                                    |
  | 1985: Windows 1.0 (Microsoft)                       |
  | -> GUI shell on top of MS-DOS                        |
  | -> Tiled windows (no overlapping)                    |
  | -> Commercially almost a failure                     |
  |                                                    |
  | 1987: Windows 2.0                                   |
  | -> Achieved overlapping windows                      |
  | -> Apple sued for copyright infringement             |
  |                                                    |
  | 1990: Windows 3.0                                   |
  | -> First commercially successful Windows             |
  | -> 386 protected mode support                        |
  | -> "Solitaire" included                              |
  +----------------------------------------------------+
```

### 1.6 1990s: Linux, Windows, Web

```
1990s: Linux, Windows, Web
  +------------------------------------------+
  | 1991: Linux (Linus Torvalds)              |
  | -> Started as a "hobby OS"                 |
  | -> Student at University of Helsinki,      |
  |    Finland                                 |
  | -> Open source under GPL license           |
  | -> Developers worldwide contributed         |
  |                                            |
  | Linux's first post (August 25, 1991):     |
  | "Hello everybody out there using minix -   |
  |  I'm doing a (free) operating system       |
  |  (just a hobby, won't be big and           |
  |  professional like gnu) for 386(486) AT    |
  |  clones."                                  |
  |                                            |
  | Rapid growth of Linux:                     |
  | 1991: v0.01 (10,000 lines)                |
  | 1994: v1.0 (176,000 lines)                |
  | 1996: v2.0 (multiprocessor support)        |
  | 2003: v2.6 (millions of lines)             |
  | 2025: v6.x (approximately 30 million lines)|
  +------------------------------------------+

  +------------------------------------------+
  | 1993: Windows NT                          |
  | -> Full 32-bit OS (designed by Dave Cutler)|
  | -> Foundation of modern Windows             |
  | -> "Proper OS" design by VMS architect      |
  |                                            |
  | Design goals of Windows NT:                |
  | - Complete 32-bit architecture              |
  | - Memory protection (process isolation)     |
  | - Preemptive multitasking                   |
  | - Multiprocessor support                    |
  | - Portability (x86, Alpha, MIPS, PPC)       |
  | - POSIX, OS/2 subsystem support             |
  | - NTFS (journaling file system)             |
  | - Win32 API                                 |
  +------------------------------------------+

  +------------------------------------------+
  | 1995: Windows 95                          |
  | -> Popularized GUIs, Start Menu            |
  | -> Brought PCs to ordinary households       |
  | -> Plug & Play                              |
  | -> Long filename support                    |
  | -> A cultural phenomenon with lines on      |
  |    launch day                               |
  | -> Rolling Stones "Start Me Up"             |
  |                                            |
  | 1998: Windows 98                           |
  | -> USB support, Internet Explorer 4         |
  |    integration                              |
  | -> Still DOS-based (unstable)               |
  |                                            |
  | 2000: Windows 2000 / Windows Me            |
  | -> NT line: stable, enterprise-oriented     |
  | -> Me line: last DOS-based, notoriously     |
  |    unstable                                 |
  |                                            |
  | 2001: Windows XP                           |
  | -> Merger of NT and DOS lines               |
  | -> NT kernel even for consumers             |
  | -> 13 years of support (until 2014)         |
  | -> The longest-used version of Windows      |
  +------------------------------------------+

  1990s Open Source Movement:
  +----------------------------------------------+
  | 1991: Linux Kernel (GPL)                      |
  | 1993: Debian Project launched                  |
  | 1993: FreeBSD 1.0 released                    |
  | 1994: Red Hat Linux 1.0                       |
  | 1995: Apache HTTP Server                      |
  | 1998: "Open Source" term coined                |
  | 1998: Netscape open-sourced -> Mozilla         |
  | 1999: GNOME 1.0, KDE 1.0                     |
  |                                               |
  | "The Cathedral and the Bazaar"                |
  | (Eric Raymond, 1997):                         |
  | -> Cathedral model: A few designers develop    |
  |    in a closed manner                          |
  | -> Bazaar model: Many developers develop openly|
  | -> Linux is a success story of the bazaar model|
  | -> "Given enough eyeballs, all bugs are shallow"|
  +----------------------------------------------+
```

### 1.7 2000s Onward: Mobile and Cloud

```
2000s~: Mobile and Cloud
  +------------------------------------------+
  | 2001: Mac OS X (macOS)                    |
  | -> NeXTSTEP + FreeBSD = Darwin             |
  | -> Unix-based commercial desktop OS         |
  | -> Aqua GUI (beautiful water-like UI)       |
  | -> Unix commands available via Terminal.app  |
  | -> Officially obtained UNIX 03 certification|
  |                                            |
  | Apple's OS Strategy:                       |
  | 1997: NeXT acquisition, Steve Jobs returns  |
  | 2001: Mac OS X 10.0 "Cheetah"             |
  | 2001-2019: Big cat names (-> changed to     |
  |            place names)                     |
  | 2020: macOS Big Sur (macOS 11)             |
  | 2020: Apple Silicon (M1) support           |
  +------------------------------------------+

  +------------------------------------------+
  | 2007: iPhone OS (iOS)                     |
  | -> Dawn of the mobile OS era               |
  | -> Multi-touch interface                    |
  | -> App Store ecosystem (from 2008)          |
  | -> Trigger for the smartphone revolution    |
  |                                            |
  | Technical features of iOS:                 |
  | - XNU kernel (shared foundation with macOS) |
  | - Sandbox model (app isolation)             |
  | - Objective-C -> Swift                      |
  | - Metal GPU API (OpenGL ES -> Metal)        |
  | - Core ML (on-device AI inference)          |
  +------------------------------------------+

  +------------------------------------------+
  | 2008: Android                             |
  | -> Built on the Linux kernel                |
  | -> World's largest mobile OS                |
  | -> Google acquired Android, Inc. in 2005    |
  | -> Open source (AOSP)                       |
  |                                            |
  | Android Architecture:                      |
  | +----------------------------+             |
  | | Applications               |             |
  | | (Java/Kotlin -> APK)       |             |
  | +----------------------------+             |
  | | Android Framework          |             |
  | | (Activity, Service, etc.)  |             |
  | +----------------------------+             |
  | | ART (Android Runtime)      |             |
  | | (Dalvik VM -> ART)         |             |
  | +----------------------------+             |
  | | HAL (Hardware Abstraction)  |             |
  | +----------------------------+             |
  | | Linux Kernel               |             |
  | | (Binder IPC, ashmem, etc.) |             |
  | +----------------------------+             |
  +------------------------------------------+

  +----------------------------------------------+
  | 2013: Docker                                  |
  | -> Revolutionized OS virtualization with       |
  |    container technology                        |
  | -> Developed by Solomon Hykes at dotCloud       |
  | -> Made Linux namespace + cgroups user-friendly |
  |                                               |
  | Impact of Docker:                              |
  | - "Build, Ship, Run Anywhere"                  |
  | - Accelerated DevOps                            |
  | - Popularized microservice architecture         |
  | - Emergence of Kubernetes (2014, Google)        |
  | - Became the standard for cloud-native          |
  |   development                                   |
  |                                               |
  | 2014: Kubernetes (K8s)                         |
  | -> Rooted in Google's internal systems          |
  |    Borg/Omega                                   |
  | -> Became the standard for container             |
  |    orchestration                                 |
  | -> CNCF (Cloud Native Computing Foundation)     |
  +----------------------------------------------+

  +------------------------------------------+
  | 2020: Apple Silicon (M1)                  |
  | -> ARM + macOS overturns PC performance    |
  |    expectations                            |
  | -> Achieves both high performance and low  |
  |    power consumption                       |
  | -> Success story of x86 to ARM migration   |
  | -> Rosetta 2 translates and runs x86       |
  |    binaries                                |
  | -> Unified memory architecture              |
  |                                            |
  | 2024-2025: The AI PC Era                   |
  | -> PCs with built-in NPU (Neural           |
  |    Processing Unit)                         |
  | -> Apple M4, Qualcomm Snapdragon X Elite   |
  | -> Intel Meteor Lake / Arrow Lake           |
  | -> OS-level AI feature integration          |
  | -> Windows: Copilot, Recall                |
  | -> macOS: Apple Intelligence               |
  +------------------------------------------+
```

---

## 2. Evolution of Key Concepts

### 2.1 Evolution of Multitasking

```
Evolution of Multitasking:

  Batch Processing (1950s):
  Job1 -------> Job2 -------> Job3
  -> Execute one at a time, sequentially

  Multiprogramming (1960s):
  Job1 ##__##__##
  Job2 __##__##__
  -> Execute other jobs during I/O wait

  Time-Sharing (1960s):
  User1 #__#__#__
  User2 _#__#__#_
  User3 __#__#__#
  -> Alternately assign short time slices to each user

  Cooperative Multitasking (1980s-90s):
  -> Processes voluntarily yield the CPU
  -> Windows 3.x, Classic Mac OS
  -> If one process runs away, the entire system freezes

  Preemptive Multitasking (1990s~):
  -> OS forcibly switches processes
  -> A runaway process does not affect others
  -> The modern standard
  -> Unix (1969), Windows NT (1993), macOS (2001)

  Real-Time Multitasking:
  -> Guarantee task completion within a deadline
  -> Rate Monotonic Scheduling, Earliest Deadline First
  -> VxWorks, QNX, FreeRTOS
```

### 2.2 Evolution of Memory Management

```
Evolution of Memory Management:

  1. Fixed Partitions (1950s-60s):
     +----------------+
     | OS             | Fixed size
     +----------------+
     | Partition 1    | 32KB
     +----------------+
     | Partition 2    | 64KB
     +----------------+
     | Partition 3    | 128KB
     +----------------+
     -> Place jobs in partitions that match their size
     -> Internal fragmentation (wasted unused portions)

  2. Variable Partitions (1960s):
     -> Dynamically create partitions to match job size
     -> External fragmentation (scattered gaps)
     -> Solved by compaction (rearranging jobs)

  3. Paging (1960s-70s):
     +------------------------------------------+
     | Virtual Address Space  Physical Memory    |
     | +----+               +----+              |
     | |Page0| -----------> |Frame5|            |
     | +----+               +----+              |
     | |Page1| -----------> |Frame2|            |
     | +----+               +----+              |
     | |Page2| --> Disk      |Frame8|            |
     | +----+               +----+              |
     | |Page3| -----------> Frame12            |
     | +----+                                  |
     | -> Managed in 4KB page units              |
     | -> Contiguous virtual addresses map to     |
     |    non-contiguous physical memory           |
     | -> Translation via page table               |
     +------------------------------------------+

  4. Virtual Memory (1960s~present):
     -> Physical memory + disk (swap) = huge address space
     -> Demand paging: only read pages into memory when used
     -> Page fault: accessed page is not in physical memory
       -> OS reads from disk and continues
     -> Each process has an independent, vast address space

  5. Modern Memory Management Techniques:
     +----------------------------------------------+
     | Huge Pages: Large 2MB/1GB pages                |
     | -> Reduce TLB misses (for databases, VMs)      |
     |                                               |
     | NUMA-aware: Conscious of physical memory       |
     |   location                                     |
     | -> Important for multi-socket servers           |
     |                                               |
     | Memory Ballooning: Dynamically adjust VM memory|
     | -> Memory efficiency in virtualized environments|
     |                                               |
     | KSM (Kernel Same-page Merging):               |
     | -> Merge pages with identical content           |
     | -> Memory savings in virtualized environments   |
     |                                               |
     | zram/zswap: In-memory compressed swap           |
     | -> Save memory without disk I/O                 |
     +----------------------------------------------+
```

### 2.3 Evolution of File Systems

```
Evolution of File Systems:

  1950s: Magnetic tape (sequential access only)
  1960s: FAT (for floppy disks, flat structure)
  1970s: Unix File System (inodes, hierarchical structure)
  1980s: FAT16 (MS-DOS), HFS (Macintosh)
  1990s: ext2 (Linux), NTFS (Windows NT)
  2000s: ext3/ext4 (journaling), ZFS (Sun)
  2010s: Btrfs (Linux), APFS (Apple)
  2020s: bcachefs (merged into Linux 6.7)

  The Innovation of Journaling File Systems:
  +----------------------------------------------+
  | Problem: Power loss during write              |
  |   -> File system corruption                    |
  |                                               |
  | Solution: Write the journal (change log) first |
  | 1. Record changes in the journal               |
  | 2. Write actual data                           |
  | 3. Mark journal entry as complete               |
  |                                               |
  | On power loss:                                 |
  | - Replay journal to restore consistency         |
  | - No fsck needed (fast boot)                    |
  |                                               |
  | ext3/ext4, NTFS, XFS, JFS support journaling   |
  +----------------------------------------------+

  Copy-on-Write (CoW) File Systems:
  +----------------------------------------------+
  | ZFS, Btrfs, APFS:                             |
  | - Write to a new location instead of overwriting|
  | - Consistency is always maintained              |
  | - Snapshots can be created instantly            |
  | - Data deduplication is possible                |
  | - Checksums detect data corruption              |
  |                                               |
  | ZFS Features (definitive enterprise storage):  |
  | - 128-bit address space (virtually unlimited)   |
  | - RAID-Z (parity-based redundancy)              |
  | - Online data compression                       |
  | - ARC (Adaptive Replacement Cache)              |
  | - Encryption, deduplication, snapshots           |
  +----------------------------------------------+
```

### 2.4 Evolution of Security

```
Evolution of OS Security:

  1960s: Password authentication (first introduced in CTSS)
  1970s: Unix permission model (owner/group/other, rwx)
  1980s: Access Control Lists (ACL)
  1990s: Firewalls, encrypted file systems
  2000s: SELinux (NSA), ASLR, DEP/NX
  2010s: Sandboxing, container isolation, UEFI Secure Boot
  2020s: Zero trust, security monitoring with eBPF

  Modern OS Security Features:
  +----------------------------------------------------+
  | ASLR (Address Space Layout Randomization):          |
  | -> Randomize memory addresses                        |
  | -> Make buffer overflow attacks more difficult        |
  |                                                    |
  | DEP/NX (Data Execution Prevention):                 |
  | -> Prohibit code execution in data regions           |
  | -> Prevent shellcode execution on the stack          |
  |                                                    |
  | Stack Canary:                                       |
  | -> Place a canary value on the stack                 |
  | -> Detect buffer overflows                           |
  |                                                    |
  | Secure Boot:                                        |
  | -> Verify signatures at each stage of boot process   |
  | -> Prevent bootkits (malware at boot time)           |
  |                                                    |
  | eBPF:                                               |
  | -> Safely execute programs within the kernel         |
  | -> Networking, security, tracing                     |
  | -> Foundation for security tools like Cilium, Falco  |
  +----------------------------------------------------+
```

---

## 3. OS Genealogy Chart

```
  Multics (1964)
    |
    +---> Unix (1969) ------------------------------------+
    |      +-- BSD (1977) --> FreeBSD --> macOS/iOS       |
    |      |                -> OpenBSD (security-focused)
    |      |                -> NetBSD (portability-focused)|
    |      +-- System V --> Solaris, AIX, HP-UX          |
    |      +-- Philosophy --> GNU (1983)                  |
    |                          +-- + Linux (1991)         |
    |                               +-- Ubuntu            |
    |                               +-- RHEL              |
    |                               +-- Android           |
    |                               +-- Chrome OS         |
    |                               +-- SteamOS           |
    |
  CP/M (1974) --> MS-DOS (1981) --> Windows 95/98/Me     |
                                                         |
  VMS (1977) --> Windows NT (1993) --> Win 2000/XP       |
                                   --> Win 7/10/11       |
                                                         |
  Xerox Alto (1973) --> Macintosh (1984)                  |
                    --> Windows GUI                       |
                                                         |
  NeXTSTEP (1989) --> Mac OS X (2001) --> macOS          |
                                                         |
  Emerging trends:                                        |
  Linux Kernel --> Android (2008)                         |
  Linux Kernel --> Chrome OS (2011)                       |
  Zircon (microkernel) --> Fuchsia (Google, 2016-)        |
  Redox (Rust) --> Next-generation Unix-like OS            |

Influence relationships between OSes:
+----------------------------------------------------+
| Multics -> Unix: Pursuit of simplicity               |
| Unix -> BSD: Academic improvements, networking        |
| Unix -> Linux: Reimplementation as open source        |
| BSD -> macOS: To Apple via NeXTSTEP                   |
| VMS -> Windows NT: Dave Cutler brought design ideas   |
| Xerox Alto -> Mac, Windows: GUI concepts              |
| Plan 9 -> Linux /proc, /sys: Virtual file systems     |
| MINIX -> Linux: Linus's starting point                |
| Linux -> Docker/K8s: Foundation for container tech     |
| Linux -> Android: Foundation for mobile OS             |
+----------------------------------------------------+
```

---

## 4. The Future of Operating Systems

```
Ongoing Transformations:

  1. AI-Integrated OS:
     +----------------------------------------------+
     | - Windows: Copilot, Windows Recall             |
     | - macOS: Apple Intelligence                    |
     | - Linux: ML inference optimization             |
     |   (CUDA, ROCm integration)                     |
     |                                               |
     | OS-level AI features:                          |
     | - System operation via natural language         |
     | - Intelligent notification management           |
     | - Cross-app context understanding               |
     | - Predictive resource management                |
     +----------------------------------------------+

  2. Rust-based OS:
     +----------------------------------------------+
     | - Rust support added to Linux kernel (6.1+)    |
     | - Redox OS: Microkernel OS written in Rust      |
     | - Android: Some new code migrating to Rust      |
     | -> Fundamental improvement in memory safety     |
     | -> Eliminates C memory bugs (cause of 70%       |
     |    of vulnerabilities)                           |
     +----------------------------------------------+

  3. WebAssembly (Wasm) OS:
     +----------------------------------------------+
     | - WASI may become the successor to POSIX       |
     | - Wasm as an alternative to containers          |
     | - Fermyon Spin, Wasmtime                        |
     | -> Portable and lightweight application          |
     |    environment                                   |
     +----------------------------------------------+

  4. Unikernels and MicroVMs:
     +----------------------------------------------+
     | - AWS Firecracker: MicroVM (Lambda/Fargate)    |
     | - gVisor: User-space kernel (Google)            |
     | - Kata Containers: VM-based containers          |
     | -> New balance between security and efficiency   |
     +----------------------------------------------+
```

---

## Hands-On Exercises

### Exercise 1: [Basic] -- Checking OS Information

```bash
# Check your OS information
# Linux:
uname -a
cat /etc/os-release
cat /proc/version
hostnamectl  # systemd environment

# macOS:
sw_vers
uname -a
system_profiler SPSoftwareDataType

# Record the kernel version, build date, and architecture

# Check kernel configuration (Linux)
cat /boot/config-$(uname -r) | grep CONFIG_PREEMPT
# -> Check the preemption model

# Check boot logs
dmesg | head -50
journalctl -b | head -50  # systemd environment

# Hardware information
lscpu          # CPU info
lsblk          # Block devices
lspci          # PCI devices
lsusb          # USB devices
```

### Exercise 2: [Applied] -- Practicing Unix Philosophy

```bash
# Combine small tools with pipes to accomplish the following:

# 1. Get shell usage statistics from /etc/passwd
cat /etc/passwd | cut -d: -f7 | sort | uniq -c | sort -rn

# 2. Aggregate access log by status code
cat access.log | awk '{print $9}' | sort | uniq -c | sort -rn

# 3. Top 10 file sizes in a directory
du -sh * 2>/dev/null | sort -rh | head -10

# 4. Process memory usage ranking
ps aux --sort=-%mem | awk 'NR<=11{printf "%-10s %5s %s\n", $1, $4, $11}'

# 5. Network connection state counts (Linux)
ss -tan | awk 'NR>1{print $1}' | sort | uniq -c | sort -rn

# 6. Come up with and execute 3 similar pipelines of your own
# Hint: Use ps, netstat, df, du, wc, grep, awk, find, xargs
```

### Exercise 3: [Applied] -- OS History Research Report

```
Write a report of 2000+ words on one of the following topics:

1. The Unix Wars (BSD vs System V): background and impact
   - Technical differences
   - Licensing issues
   - Unification through POSIX
   - Impact on the modern era

2. The Linux vs MINIX debate (Tanenbaum-Torvalds debate)
   - Monolithic vs microkernel debate
   - Arguments from both sides
   - Evaluation 30 years later
   - Impact on modern OS design

3. The evolution of Windows (MS-DOS -> Windows 11)
   - Architecture transitions
   - GUI evolution
   - NT kernel design philosophy
   - Convergence with Linux through WSL2

4. Apple's OS strategy (Classic Mac OS -> macOS -> iOS)
   - Significance of the NeXT acquisition
   - Transition to Unix foundation
   - Integration of iOS and macOS (Apple Silicon)
   - Strength of the ecosystem
```

### Exercise 4: [Advanced] -- Experiencing Each Generation of OS via Emulators

```
Experience the following OSes via emulators and record
the changes in user experience:

1. SIMH (System Simulator):
   -> Experience Unix V6 on PDP-11
   -> File editing with the ed editor
   -> Shell operations (Thompson shell)

2. DOSBox:
   -> Experience MS-DOS 6.22
   -> Command-line operations (dir, copy, type, edit)
   -> Feel the 640KB memory constraint

3. PCem / 86Box:
   -> Experience the Windows 3.1 GUI
   -> File Manager, Paintbrush

4. QEMU:
   -> Boot an early version of Linux (v0.01 is impractical,
      but 1.0 series)
   -> Verify the operation of a minimal Linux configuration

5. Browser-based emulators:
   -> Experience Windows 95, Linux, etc. at copy.sh/v86/
   -> Various OS emulations at archive.org

For each OS, record:
- Boot time
- Available commands/operations
- Differences in memory management
- Presence or absence of multitasking
- Comparison with modern OSes
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Data volume growth | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Check execution user permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check error messages**: Read the stack trace and identify the location of the error
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Incremental verification**: Verify hypotheses using log output or debuggers
5. **Fix and regression test**: After fixing, run tests on related areas as well

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
    """Decorator that logs function inputs and outputs"""
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

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check disk and network I/O conditions
4. **Check concurrent connections**: Check connection pool status

| Problem Type | Diagnostic Tools | Countermeasures |
|-------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper release of references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the criteria for making technology selections.

| Criteria | Prioritize When | Acceptable to Compromise When |
|---------|----------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed users |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Choosing an Architecture Pattern

```
+---------------------------------------------------+
|          Architecture Selection Flow                |
+---------------------------------------------------+
|                                                   |
|  (1) Team size?                                    |
|    +- Small (1-5 people) -> Monolith               |
|    +- Large (10+ people) -> Go to (2)              |
|                                                   |
|  (2) Deployment frequency?                         |
|    +- Weekly or less -> Monolith + module split     |
|    +- Daily/multiple times -> Go to (3)            |
|                                                   |
|  (3) Independence between teams?                   |
|    +- High -> Microservices                         |
|    +- Moderate -> Modular monolith                  |
|                                                   |
+---------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A quick short-term approach may become technical debt in the long run
- Conversely, over-engineering has high short-term costs and may delay the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables the right tool for the job but increases operational costs

**3. Level of Abstraction**
- Higher abstraction provides better reusability but can make debugging more difficult
- Lower abstraction is more intuitive but tends to produce code duplication

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
        """Describe the background and challenges"""
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
- Automated testing only for the critical path
- Introduce monitoring from the early stages

**Lessons Learned:**
- Do not aim for perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt consciously

### Scenario 2: Legacy System Modernization

**Situation:** Gradually modernize a system that has been in operation for over 10 years

**Approach:**
- Migrate incrementally using the Strangler Fig pattern
- If there are no existing tests, create Characterization Tests first
- Use an API gateway to allow new and old systems to coexist
- Perform data migration in stages

| Phase | Work Content | Estimated Duration | Risk |
|-------|-------------|-------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Start Migration | Migrate peripheral features sequentially | 3-6 months | Medium |
| 4. Core Migration | Migrate core features | 6-12 months | High |
| 5. Completion | Decommission legacy system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50+ engineers developing the same product

**Approach:**
- Clarify boundaries with Domain-Driven Design
- Set ownership for each team
- Manage shared libraries using Inner Source approach
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
        """Check SLA compliance"""
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

**Optimization Points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Effect | Implementation Cost | Application |
|--------------------|--------|-------------------|-------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Asynchronous processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | When queries are slow |
| Code optimization | Low-Medium | High | When CPU-bound |
---

## FAQ

### Q1: Why are there so many Linux distributions?

Since the Linux kernel is open source, anyone can create a distribution by combining the kernel with their own package management and configuration. Diversity emerged because optimizations differ depending on the purpose (RHEL for servers, Ubuntu for desktops, Kali for security, Edubuntu for education, etc.). This is proof of "freedom" and simultaneously a source of confusion for beginners. Major distributions are consolidating to around 10.

### Q2: Why is macOS Unix-based?

When Apple acquired NeXT (1997), NeXTSTEP (Mach + BSD), developed by Steve Jobs, became the foundation. Darwin (the macOS kernel) consists of the Mach microkernel + FreeBSD components and has officially obtained UNIX 03 certification. Classic Mac OS (System 1-9) was a proprietary OS, but it lacked modern memory protection and preemptive multitasking and had reached its limits. By adopting NeXTSTEP, macOS was able to achieve both a stable Unix foundation and a beautiful GUI.

### Q3: What was the difference between Windows NT and Windows 95?

Windows 95 was a 16/32-bit hybrid built on MS-DOS and was unstable. Windows NT, on the other hand, was a full 32-bit OS designed from scratch by VMS architect Dave Cutler, with memory protection, preemptive multitasking, and NTFS. The NT line was positioned for enterprise use (NT 4.0, 2000), while the consumer line (95, 98, Me) was separate. Windows XP (2001) merged these two lines, bringing the benefits of the NT kernel to consumers.

### Q4: What is the difference between BSD and Linux?

Technically, both are Unix-like OSes, but the main differences are:
1. **Kernel**: BSD develops the kernel and userland as a unified whole. Linux is kernel-only (userland comes from GNU and others)
2. **License**: BSD License (permissive) vs GPL (copyleft)
3. **Development Model**: BSD has a controlled core development team. Linux is distributed (Linus makes final decisions)
4. **Drivers**: Linux has more extensive hardware support
5. **Use Cases**: FreeBSD excels for network appliances (Netflix, WhatsApp), OpenBSD for security-critical applications

### Q5: What will the next generation of OS look like?

Several directions are being discussed:
1. **Revival of microkernels**: seL4 (formally verified), Fuchsia (Zircon)
2. **Rust-based OS**: Memory safety guaranteed at the language level
3. **AI integration**: OS understands user intent and optimizes autonomously
4. **Wasm OS**: Portable OS environments based on WebAssembly
5. **Quantum OS**: OS for managing quantum computer resources (research stage)
6. **Brain-computer OS**: OS for processing input from BCI (future)

---

## Summary

| Era | Innovation | Representative OS | Design Philosophy |
|-----|-----------|-------------------|-------------------|
| 1950s | Batch processing | GM-NAA I/O | Improve CPU utilization |
| 1960s | Time-sharing | Multics, CTSS | Interactive computing |
| 1970s | Birth of Unix, C language | Unix | Simplicity, portability |
| 1980s | PCs, GUIs | MS-DOS, Macintosh | Personal computing |
| 1990s | Open source | Linux, Windows NT | Freedom, robustness |
| 2000s | Mobile, cloud | iOS, Android, Docker | Ubiquity, virtualization |
| 2020s | AI integration, ARM | Apple Silicon, AI PC | Efficiency, intelligence |

---

## Recommended Next Reading

---

## References
1. Ritchie, D. & Thompson, K. "The UNIX Time-Sharing System." CACM, 1974.
2. Raymond, E. "The Art of Unix Programming." Addison-Wesley, 2003.
3. Campbell-Kelly, M. "From Airline Reservations to Sonic the Hedgehog: A History of the Software Industry." MIT Press, 2003.
4. Brinch Hansen, P. "Classic Operating Systems: From Batch Processing to Distributed Systems." Springer, 2001.
5. Brooks, F. "The Mythical Man-Month." Anniversary Edition, Addison-Wesley, 1995.
6. Ceruzzi, P. "A History of Modern Computing." MIT Press, 2003.
7. Salus, P. "A Quarter Century of UNIX." Addison-Wesley, 1994.
8. DiBona, C. et al. "Open Sources: Voices from the Open Source Revolution." O'Reilly, 1999.
