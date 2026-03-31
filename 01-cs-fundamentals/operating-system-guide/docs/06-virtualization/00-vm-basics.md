# Virtual Machine Fundamentals

> Virtualization is a technology that "runs multiple independent virtual machines on a single physical machine," forming the foundation of cloud computing.

## Learning Objectives

- [ ] Distinguish between types of virtualization
- [ ] Understand how hypervisors work
- [ ] Know the major virtualization technologies
- [ ] Understand the mechanisms of CPU virtualization, memory virtualization, and I/O virtualization
- [ ] Practice KVM/QEMU configuration and operations
- [ ] Understand how live migration works
- [ ] Perform virtualization performance tuning
- [ ] Know about nested virtualization and its applications


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. History and Overview of Virtualization

### 1.1 History of Virtualization

```
History of Virtualization:

  1960s:
  ┌──────────────────────────────────────────────────┐
  │ IBM System/360 Model 67 (1966):                   │
  │ → World's first virtual machine monitor           │
  │ → CP-40 / CP-67: Prototype of the VMM            │
  │ → Run multiple OSes simultaneously on one         │
  │   mainframe                                        │
  │                                                    │
  │ IBM VM/370 (1972):                                │
  │ → Commercial virtualization platform              │
  │ → Provided independent virtual machines to        │
  │   each user                                        │
  │ → Ran CMS (Conversational Monitor System)         │
  └──────────────────────────────────────────────────┘

  1990s-2000s:
  ┌──────────────────────────────────────────────────┐
  │ VMware (1999):                                    │
  │ → Commercialized virtualization on x86            │
  │ → Handled privileged instructions via binary      │
  │   translation                                      │
  │ → VMware Workstation, ESX Server                  │
  │                                                    │
  │ Xen (2003):                                       │
  │ → Developed at the University of Cambridge        │
  │ → High performance via paravirtualization         │
  │ → Early foundation of Amazon EC2                  │
  │                                                    │
  │ Intel VT-x / AMD-V (2005-2006):                   │
  │ → Hardware-assisted virtualization                │
  │ → Added virtualization-specific instructions      │
  │   to the CPU                                       │
  │ → Achieved high performance even with full        │
  │   virtualization                                   │
  └──────────────────────────────────────────────────┘

  2007 to Present:
  ┌──────────────────────────────────────────────────┐
  │ KVM (2007, Linux 2.6.20):                         │
  │ → Implemented as a Linux kernel module            │
  │ → Turns Linux itself into a hypervisor            │
  │ → Grew to be the standard virtualization          │
  │   platform for the cloud                           │
  │                                                    │
  │ Cloud Era (2006-):                                │
  │ → AWS EC2 (2006): Xen → Nitro (KVM)             │
  │ → Google Compute Engine: KVM                      │
  │ → Azure: Hyper-V                                  │
  │ → Virtualization became the foundation of cloud   │
  │   computing                                        │
  │                                                    │
  │ Lightweight Virtualization (2018-):               │
  │ → Firecracker: microVM                            │
  │ → Cloud Hypervisor: Rust-based lightweight VMM   │
  │ → High security, low overhead as QEMU alternatives│
  └──────────────────────────────────────────────────┘
```

### 1.2 Popek & Goldberg Virtualization Requirements

```
Theoretical Foundation of Virtualization (1974):
  Three requirements for a Virtual Machine Monitor (VMM) defined by
  Popek & Goldberg:

  1. Equivalence / Fidelity:
     Programs running on the VMM behave the same as on bare metal
     → With some exceptions: timing, resource availability

  2. Efficiency:
     Most guest instructions execute directly on hardware
     → Direct execution, not emulation

  3. Resource Control / Safety:
     The VMM has complete control over all hardware resources
     → Guests cannot bypass the VMM to access resources

  Difficulty of x86 Virtualization:
  ┌──────────────────────────────────────────────────┐
  │ Problem: x86 had "sensitive but non-privileged"  │
  │ instructions                                      │
  │                                                    │
  │ Privileged instructions: Cause exceptions when    │
  │ executed outside Ring 0                            │
  │ → VMM can trap and handle them                    │
  │                                                    │
  │ Sensitive instructions: Affect system state but   │
  │ execute regardless of privilege level              │
  │ → VMM cannot trap them!                           │
  │                                                    │
  │ Examples of problematic instructions:              │
  │ - SGDT/SIDT: Read descriptor table                │
  │ - SMSW: Read machine status word                  │
  │ - PUSHF/POPF: Manipulate flags register           │
  │                                                    │
  │ Solutions:                                         │
  │ 1. Binary translation (VMware)                    │
  │ 2. Paravirtualization (Xen)                       │
  │ 3. Hardware-assisted (VT-x/AMD-V) ← Fundamental  │
  │    solution                                        │
  └──────────────────────────────────────────────────┘
```

---

## 2. Types of Virtualization

### 2.1 Full Virtualization

```
Full Virtualization:
  Runs unmodified guest OSes

  Method 1: Binary Translation
  ┌──────────────────────────────────────────────────┐
  │ Guest OS Code                                     │
  │       ↓                                           │
  │ Binary Translation Engine                         │
  │   → User-mode instructions: Execute directly     │
  │   → Sensitive instructions: Translate to safe     │
  │     instruction sequences                          │
  │       ↓                                           │
  │ Translated Code                                   │
  │   → Cached for reuse                              │
  │       ↓                                           │
  │ Execute on Physical Hardware                      │
  │                                                    │
  │ Developed and commercialized by VMware            │
  │ → Performance overhead: 10-30%                    │
  │ → Biggest advantage: No guest OS modification     │
  │   required                                         │
  └──────────────────────────────────────────────────┘

  Method 2: Hardware-Assisted Virtualization
  ┌──────────────────────────────────────────────────┐
  │ CPU adds virtualization support instructions:     │
  │ → Intel VT-x (VMX: Virtual Machine Extensions)   │
  │ → AMD-V (SVM: Secure Virtual Machine)            │
  │                                                    │
  │ CPU Operating Modes:                              │
  │ ┌─────────────────────────┐                       │
  │ │ VMX Root Mode (Host)    │                       │
  │ │   Ring 0: Hypervisor    │                       │
  │ │   Ring 3: Host apps     │                       │
  │ └─────────┬───────────────┘                       │
  │           │ VM Entry                              │
  │           ↓                                       │
  │ ┌─────────────────────────┐                       │
  │ │ VMX Non-Root (Guest)    │                       │
  │ │   Ring 0: Guest kernel  │                       │
  │ │   Ring 3: Guest apps    │                       │
  │ └─────────┬───────────────┘                       │
  │           │ VM Exit (on privileged operations)    │
  │           ↓                                       │
  │ VMX Root Mode handles, then VM Entry to resume    │
  │                                                    │
  │ VMCS (Virtual Machine Control Structure):         │
  │ → Structure that saves guest/host state           │
  │ → Automatically loaded/saved on VM Entry/Exit    │
  │ → VM Exit conditions are configurable             │
  └──────────────────────────────────────────────────┘
```

### 2.2 Paravirtualization

```
Paravirtualization:
  Guest OS is modified for a virtual environment

  ┌──────────────────────────────────────────────────┐
  │ Guest OS (Modified)                               │
  │       ↓ Hypercall                                 │
  │ Hypervisor                                        │
  │       ↓ Actual hardware operations                │
  │ Physical Hardware                                 │
  │                                                    │
  │ Hypercall:                                        │
  │ → Direct requests from guest OS to hypervisor    │
  │ → Interface similar to system calls              │
  │ → Enables privileged operations without binary   │
  │   translation                                      │
  │                                                    │
  │ Advantages:                                       │
  │ - Higher performance than binary translation     │
  │ - Efficient interrupt handling                    │
  │ - Fast I/O                                        │
  │                                                    │
  │ Disadvantages:                                    │
  │ - Requires guest OS kernel modification          │
  │ - Cannot be used with unmodifiable OSes like     │
  │   Windows                                          │
  │ - Advantages diminished with the spread of       │
  │   hardware-assisted virtualization                 │
  └──────────────────────────────────────────────────┘

  Paravirtualized Drivers (Virtio):
  → A compromise that paravirtualizes only the driver portion
    instead of the entire guest OS kernel
  → Widely used in modern virtualization
  → Network, storage, memory ballooning, etc.
```

### 2.3 Emulation

```
Emulation:
  Mimics entire hardware in software

  ┌──────────────────────────────────────────────────┐
  │ Emulation vs Virtualization:                      │
  │                                                    │
  │ Emulation:                                        │
  │ → Reproduces a different architecture in software│
  │ → Example: ARM on x86, x86 on MIPS              │
  │ → Very slow (10-100x overhead)                   │
  │ → QEMU (accelerated with JIT translation)        │
  │                                                    │
  │ Virtualization:                                   │
  │ → Runs multiple instances on the same            │
  │   architecture                                     │
  │ → Guest instructions execute directly on CPU     │
  │ → Near-native performance                        │
  │ → KVM + QEMU (only device emulation portion)     │
  │                                                    │
  │ Two Modes of QEMU:                                │
  │ 1. Full System Emulation:                         │
  │    Emulates an entire system                      │
  │    → qemu-system-aarch64 (emulates ARM system)   │
  │                                                    │
  │ 2. User Mode Emulation:                           │
  │    Emulates only user programs                    │
  │    → qemu-aarch64 ./arm_binary                   │
  │    → Used in Docker multi-platform builds        │
  └──────────────────────────────────────────────────┘
```

---

## 3. Hypervisor Architecture

### 3.1 Type 1 Hypervisor (Bare Metal)

```
Type 1 (Bare Metal) Hypervisor:
  Runs directly on hardware

  ┌──────┐ ┌──────┐ ┌──────┐
  │ VM 1 │ │ VM 2 │ │ VM 3 │
  │ OS A │ │ OS B │ │ OS C │
  └──┬───┘ └──┬───┘ └──┬───┘
  ┌──┴────────┴────────┴───┐
  │ Type 1 Hypervisor      │  ← Directly on hardware
  └────────────────────────┘
  ┌────────────────────────┐
  │ Hardware               │
  └────────────────────────┘

  Major Type 1 Hypervisors:
  ┌──────────────────────────────────────────────────┐
  │ VMware ESXi:                                      │
  │ → Commercial enterprise standard                 │
  │ → Unified management via vSphere / vCenter       │
  │ → Live migration via vMotion                     │
  │ → DRS (Distributed Resource Scheduler)            │
  │ → HA (High Availability)                          │
  │ → Dominant market share in enterprise             │
  │   virtualization                                   │
  │                                                    │
  │ Microsoft Hyper-V:                                │
  │ → Integrated with Windows Server                  │
  │ → Foundation technology of Azure                  │
  │ → Managed via System Center                       │
  │ → High affinity with Windows environments        │
  │ → Generation 2 VMs with UEFI, Secure Boot        │
  │                                                    │
  │ Xen:                                              │
  │ → Open source                                     │
  │ → Dom0 (privileged domain) + DomU (guests)       │
  │ → Early foundation of AWS EC2 (now Nitro/KVM)    │
  │ → Citrix Hypervisor (formerly XenServer)          │
  │ → Security foundation of Qubes OS                 │
  │                                                    │
  │ KVM (Kernel-based Virtual Machine):               │
  │ → Linux kernel module                             │
  │ → Turns Linux into a Type 1 hypervisor            │
  │ → Device emulation combined with QEMU            │
  │ → Foundation of AWS EC2 (Nitro), GCE, OpenStack  │
  │ → Most widely used open-source hypervisor         │
  └──────────────────────────────────────────────────┘
```

### 3.2 Type 2 Hypervisor (Hosted)

```
Type 2 (Hosted) Hypervisor:
  Runs as an application on a host OS

  ┌──────┐ ┌──────┐
  │ VM 1 │ │ VM 2 │
  │ OS A │ │ OS B │
  └──┬───┘ └──┬───┘
  ┌──┴────────┴───┐
  │ Hypervisor    │  ← Application on top of host OS
  ├───────────────┤
  │ Host OS       │
  └───────────────┘
  ┌───────────────┐
  │ Hardware      │
  └───────────────┘

  Major Type 2 Hypervisors:
  ┌──────────────────────────────────────────────────┐
  │ VirtualBox (Oracle):                              │
  │ → Open source (GPLv2)                            │
  │ → Cross-platform (Windows/Mac/Linux)             │
  │ → Widely used for dev/test environments          │
  │ → Simple GUI management                          │
  │ → Automatable via VBoxManage CLI                 │
  │ → Snapshots, shared folders support              │
  │                                                    │
  │ VMware Workstation / Fusion:                      │
  │ → Commercial (free for personal use)             │
  │ → Workstation: Windows/Linux host                │
  │ → Fusion: macOS host                             │
  │ → High performance, enterprise features          │
  │ → Unity mode (guest apps on host desktop)        │
  │                                                    │
  │ Parallels Desktop:                                │
  │ → macOS exclusive                                │
  │ → Native Apple Silicon (M1/M2/M3) support       │
  │ → Coherence mode (seamless integration)          │
  │ → Highest performance Mac virtualization         │
  │                                                    │
  │ QEMU:                                             │
  │ → Open source emulator/virtualization tool       │
  │ → Standalone is emulation only                   │
  │ → Hardware-assisted virtualization with KVM      │
  │ → Supports many architectures                    │
  │ → Managed via libvirt / virsh                    │
  └──────────────────────────────────────────────────┘

  On the Classification of KVM:
  ┌──────────────────────────────────────────────────┐
  │ KVM is strictly between Type 1 and Type 2:       │
  │                                                    │
  │ → Operates as part of the Linux kernel            │
  │   → Linux functions as hypervisor (Type 1-like)  │
  │ → But runs on top of a full OS, Linux            │
  │   → Module on host OS (Type 2-like)              │
  │                                                    │
  │ Generally classified as Type 1                    │
  │ (directly accesses HW as a kernel module)         │
  └──────────────────────────────────────────────────┘
```

---

## 4. CPU Virtualization

### 4.1 How Intel VT-x Works

```
Intel VT-x (Virtual Machine Extensions):

  VMX Operation:
  ┌──────────────────────────────────────────────────┐
  │ VMXON: Enable VMX mode                           │
  │       ↓                                           │
  │ Create and configure VMCS                         │
  │       ↓                                           │
  │ VMLAUNCH: First guest launch                      │
  │       ↓                                           │
  │ ┌─── Guest Execution ──┐                          │
  │ │ Normal instrs: Direct│                          │
  │ │ Privileged ops: VM   │                          │
  │ │   Exit               │                          │
  │ └────────┬─────────────┘                          │
  │          ↓                                        │
  │ VMM handling (VM Exit handler)                    │
  │          ↓                                        │
  │ VMRESUME: Resume guest execution                  │
  │          ↓                                        │
  │ (Loop: VM Exit → VMM handling → VMRESUME)        │
  │          ↓                                        │
  │ VMXOFF: Disable VMX mode                          │
  └──────────────────────────────────────────────────┘

  VMCS (Virtual Machine Control Structure):
  ┌──────────────────────────────────────────────────┐
  │ Guest State Area:                                  │
  │ → Guest CPU state (registers, CR, MSR, etc.)     │
  │ → Automatically saved on VM Exit                  │
  │ → Automatically restored on VM Entry              │
  │                                                    │
  │ Host State Area:                                   │
  │ → Host (VMM) CPU state                            │
  │ → Automatically restored on VM Exit               │
  │                                                    │
  │ VM-Execution Control Fields:                       │
  │ → Configure VM Exit conditions                    │
  │ → E.g., Exit on external interrupt, I/O port      │
  │   access                                           │
  │ → Fine-grained control via bitmaps               │
  │                                                    │
  │ VM-Exit Control Fields:                            │
  │ → Configure behavior on VM Exit                   │
  │                                                    │
  │ VM-Entry Control Fields:                           │
  │ → Configure behavior on VM Entry                  │
  │ → E.g., interrupt injection                       │
  │                                                    │
  │ VM-Exit Information Fields:                        │
  │ → Reason for VM Exit (Exit Reason)               │
  │ → E.g., External interrupt, I/O instruction      │
  │ → E.g., EPT violation, CPUID                     │
  └──────────────────────────────────────────────────┘

  Major VM Exit Causes:
  ┌────────────────────────┬────────────────────────────┐
  │ Cause                  │ Description                │
  ├────────────────────────┼────────────────────────────┤
  │ External interrupt     │ External hardware interrupt│
  │ HLT                    │ CPU halt instruction       │
  │ I/O instruction        │ IN/OUT instructions        │
  │ CR access              │ Control register access    │
  │ MSR read/write         │ Model-specific register op │
  │ CPUID                  │ CPU information retrieval  │
  │ EPT violation          │ Memory access violation    │
  │ VMCALL                 │ Hypercall                  │
  │ Task switch            │ Task switch                │
  │ INVLPG                 │ TLB entry invalidation     │
  └────────────────────────┴────────────────────────────┘

  VM Exit Overhead:
  → One VM Exit: Hundreds to thousands of CPU cycles
  → Reducing VM Exits is key to virtualization performance tuning
  → Posted Interrupts: Reduce interrupt-related Exits
  → APIC Virtualization: Reduce APIC operation Exits
```

### 4.2 AMD-V Features

```
AMD-V (AMD Virtualization / SVM):

  Comparison with Intel VT-x:
  ┌─────────────────┬────────────────┬────────────────┐
  │ Feature         │ Intel VT-x     │ AMD-V (SVM)    │
  ├─────────────────┼────────────────┼────────────────┤
  │ Control struct  │ VMCS           │ VMCB           │
  │ VM Entry        │ VMLAUNCH/      │ VMRUN          │
  │                 │ VMRESUME       │                │
  │ VM Exit         │ VM Exit        │ #VMEXIT        │
  │ Enable          │ VMXON          │ EFER.SVME      │
  │ Nested          │ VMCS shadowing │ Native support │
  │ EPT/NPT         │ EPT            │ NPT (RVI)      │
  │ I/O control     │ I/O bitmap     │ I/O permission │
  │ Interrupts      │ Posted Int.    │ AVIC           │
  │ Memory encrypt  │ TDX (separate) │ SEV-SNP        │
  └─────────────────┴────────────────┴────────────────┘

  AMD SEV-SNP (Secure Encrypted Virtualization - SNP):
  → Encrypts VM memory with AES-128
  → Unique encryption key per VM
  → Hypervisor cannot read guest memory
  → SNP: Adds page-level integrity protection
  → Foundation for Confidential VMs in the cloud
```

---

## 5. Memory Virtualization

### 5.1 Address Translation

```
Memory Virtualization Address Spaces:

  Three Stages of Address Translation:
  ┌──────────────────────────────────────────────────┐
  │ Guest Virtual Address (GVA)                       │
  │     ↓ Guest page table                           │
  │ Guest Physical Address (GPA)                      │
  │     ↓ Hypervisor translation                     │
  │ Host Physical Address (HPA)                       │
  │     ↓                                             │
  │ Physical Memory (DRAM)                            │
  └──────────────────────────────────────────────────┘

  Method 1: Shadow Page Tables
  ┌──────────────────────────────────────────────────┐
  │ Hypervisor maintains a direct GVA → HPA mapping  │
  │ as shadow page tables                             │
  │                                                    │
  │ Guest PT (GVA→GPA) + VMM translation (GPA→HPA)  │
  │     → Shadow PT (GVA→HPA)                        │
  │                                                    │
  │ Problems:                                          │
  │ - Shadow table sync needed whenever guest         │
  │   modifies page tables                             │
  │ - Frequent VM Exits degrade performance           │
  │ - Increased memory consumption                    │
  │ - Complex implementation                          │
  └──────────────────────────────────────────────────┘

  Method 2: EPT / NPT (Hardware-Assisted)
  ┌──────────────────────────────────────────────────┐
  │ Intel EPT (Extended Page Tables):                 │
  │ AMD NPT (Nested Page Tables) / RVI:              │
  │                                                    │
  │ GVA → Guest PT → GPA → EPT/NPT → HPA           │
  │                                                    │
  │ → Hardware performs two-stage address translation │
  │   automatically                                    │
  │ → No shadow page tables needed                   │
  │ → Significant reduction in VM Exits              │
  │ → TLB miss penalty: 4 levels x 4 levels = up to │
  │   24 memory accesses (page walk)                  │
  │ → VPID (Virtual Processor ID): Avoids TLB flush │
  │   to improve performance on VM switches          │
  │                                                    │
  │ EPT Structure (4 levels):                         │
  │ EPT PML4 → EPT PDPT → EPT PD → EPT PT → HPA   │
  │ → Must traverse both guest PT and EPT, but       │
  │   hardware handles it automatically              │
  └──────────────────────────────────────────────────┘
```

### 5.2 Memory Efficiency Techniques

```
Memory Efficiency:

  KSM (Kernel Same-page Merging):
  ┌──────────────────────────────────────────────────┐
  │ Shares memory pages with identical content        │
  │                                                    │
  │ Operation:                                        │
  │ 1. ksmd daemon scans pages                       │
  │ 2. Discovers pages with identical content        │
  │ 3. Shares one page (Copy-on-Write)               │
  │ 4. Creates a copy on write                       │
  │                                                    │
  │ Effectiveness:                                    │
  │ - 30-50% memory savings with many same-OS VMs   │
  │ - Particularly effective in desktop VDI           │
  │ - OS kernels, shared libraries are sharing targets│
  │                                                    │
  │ Cautions:                                         │
  │ - CPU overhead from scanning                     │
  │ - Side-channel attack risk (timing attacks)      │
  │ - Recommended to disable in security-sensitive    │
  │   environments                                     │
  │                                                    │
  │ Configuration:                                    │
  │ echo 1 > /sys/kernel/mm/ksm/run                   │
  │ echo 1000 > /sys/kernel/mm/ksm/sleep_millisecs    │
  │ cat /sys/kernel/mm/ksm/pages_sharing              │
  └──────────────────────────────────────────────────┘

  Memory Ballooning:
  ┌──────────────────────────────────────────────────┐
  │ A balloon driver inside the guest OS "inflates"  │
  │ to claim memory                                    │
  │ → Temporarily reduces guest available memory     │
  │ → Host reclaims it for other VMs                 │
  │                                                    │
  │ Inflate:                                          │
  │ Guest memory: [XXXXX     ]                       │
  │ Balloon inflated: [XXXXXXXXXB]                   │
  │ → X=Guest use, B=Balloon, space=Free             │
  │ → Pages in balloon reclaimed by host             │
  │                                                    │
  │ Deflate:                                          │
  │ Balloon deflated: [XXXXXXX   ]                   │
  │ → Guest can use memory again                     │
  │                                                    │
  │ Advantages:                                       │
  │ - Enables memory overcommitment                  │
  │ - Respects guest OS memory management            │
  │ - Guest swaps out unnecessary pages              │
  │                                                    │
  │ Implemented via virtio-balloon driver             │
  └──────────────────────────────────────────────────┘

  Huge Pages:
  ┌──────────────────────────────────────────────────┐
  │ Normal: 4KB pages → Consumes many TLB entries    │
  │ Huge Pages: 2MB or 1GB pages                     │
  │ → Significantly reduces TLB misses               │
  │ → Particularly effective for memory-intensive VMs│
  │                                                    │
  │ Transparent Huge Pages (THP):                     │
  │ → Kernel automatically uses Huge Pages           │
  │ → Automatically applied to VMs                   │
  │                                                    │
  │ Static Huge Pages:                                │
  │ → Reserved in advance                            │
  │ → QEMU: -mem-path /dev/hugepages -mem-prealloc  │
  │ → More reliable but reduces memory flexibility   │
  │                                                    │
  │ Configuration:                                    │
  │ # Reserve Huge Pages                              │
  │ echo 1024 > /sys/kernel/mm/hugepages/             │
  │   hugepages-2048kB/nr_hugepages                   │
  │ # → 1024 x 2MB = 2GB reserved                    │
  │                                                    │
  │ # Verify                                          │
  │ cat /proc/meminfo | grep Huge                     │
  │ # HugePages_Total:    1024                        │
  │ # HugePages_Free:     1024                        │
  │ # Hugepagesize:       2048 kB                     │
  └──────────────────────────────────────────────────┘
```

---

## 6. I/O Virtualization

### 6.1 I/O Virtualization Methods

```
Three Methods of I/O Virtualization:

  1. Emulation (Full Virtualization):
  ┌──────────────────────────────────────────────────┐
  │ Guest → I/O instruction → VM Exit → VMM emulates│
  │                                                    │
  │ Example: QEMU emulates a virtual NE2000 NIC      │
  │ → Guest can use standard drivers                 │
  │ → Very slow (massive VM Exits)                   │
  │ → Highest compatibility                          │
  └──────────────────────────────────────────────────┘

  2. Paravirtualized I/O (Virtio):
  ┌──────────────────────────────────────────────────┐
  │ Guest → Virtio driver → Shared memory ring       │
  │              → Minimal VM Exits → Host handling  │
  │                                                    │
  │ How Virtio Works:                                 │
  │ ┌─────────────────────────────────────┐           │
  │ │ Guest                               │           │
  │ │ ┌─────────────┐                     │           │
  │ │ │ Virtio Driver│                     │           │
  │ │ └──────┬──────┘                     │           │
  │ │        ↓                             │           │
  │ │ ┌──────────────┐                    │           │
  │ │ │ Virtqueue    │ ← Ring buffer      │           │
  │ │ │ (desc/avail/ │                    │           │
  │ │ │  used ring)  │                    │           │
  │ │ └──────┬──────┘                     │           │
  │ └────────┼────────────────────────────┘           │
  │          ↓ Shared memory                          │
  │ ┌────────┼────────────────────────────┐           │
  │ │ Host   ↓                             │           │
  │ │ ┌──────────────┐                    │           │
  │ │ │ Virtio Backend│                    │           │
  │ │ │ (vhost-net etc)│                   │           │
  │ │ └──────────────┘                    │           │
  │ └─────────────────────────────────────┘           │
  │                                                    │
  │ Virtio Device Types:                              │
  │ - virtio-net: Network                             │
  │ - virtio-blk: Block storage                      │
  │ - virtio-scsi: SCSI storage                      │
  │ - virtio-serial: Serial communication            │
  │ - virtio-balloon: Memory ballooning              │
  │ - virtio-gpu: Graphics                           │
  │ - virtio-fs: Filesystem sharing                  │
  │ - virtio-vsock: Host-guest communication         │
  │                                                    │
  │ vhost: Moves Virtio backend to kernel space      │
  │ → Reduces QEMU user-space overhead               │
  │ → vhost-net: In-kernel network backend           │
  │ → vhost-user: User-space backend (e.g., DPDK)   │
  └──────────────────────────────────────────────────┘

  3. Device Passthrough (SR-IOV):
  ┌──────────────────────────────────────────────────┐
  │ SR-IOV (Single Root I/O Virtualization):          │
  │                                                    │
  │ Physical Device (NIC):                            │
  │ ┌─────────────────────────────────┐               │
  │ │ PF (Physical Function)          │               │
  │ │ ┌─────┐ ┌─────┐ ┌─────┐       │               │
  │ │ │ VF1 │ │ VF2 │ │ VF3 │ ...   │               │
  │ │ └──┬──┘ └──┬──┘ └──┬──┘       │               │
  │ └────┼───────┼───────┼───────────┘               │
  │      │       │       │                            │
  │   ┌──┴──┐ ┌──┴──┐ ┌──┴──┐                       │
  │   │ VM1 │ │ VM2 │ │ VM3 │                       │
  │   └─────┘ └─────┘ └─────┘                       │
  │                                                    │
  │ PF (Physical Function): Full physical device      │
  │ VF (Virtual Function): Lightweight virtual device │
  │                                                    │
  │ → Direct access bypassing the hypervisor         │
  │ → Near-native I/O performance                    │
  │ → DMA protected by IOMMU (Intel VT-d / AMD-Vi)  │
  │ → Live migration is difficult (device-dependent) │
  │                                                    │
  │ Performance Comparison:                           │
  │ ┌────────────┬──────────┬──────────┐              │
  │ │ Method     │ Latency  │ Throughput│              │
  │ ├────────────┼──────────┼──────────┤              │
  │ │ Emulated   │ High     │ Low      │              │
  │ │ Virtio     │ Medium   │ High     │              │
  │ │ SR-IOV     │ Low      │ Very High│              │
  │ │ Native     │ Lowest   │ Highest  │              │
  │ └────────────┴──────────┴──────────┘              │
  └──────────────────────────────────────────────────┘
```

---

## 7. KVM/QEMU in Practice

### 7.1 KVM Architecture

```
KVM + QEMU Architecture:

  ┌──────────────────────────────────────────────────┐
  │ Guest OS                                          │
  │ ┌──────────────────┐                              │
  │ │ Application       │                              │
  │ │ Guest Kernel      │                              │
  │ │ virtio drivers    │                              │
  │ └────────┬─────────┘                              │
  │          │                                         │
  │ QEMU Process (User Space)                         │
  │ ┌────────┴─────────┐                              │
  │ │ Device Emulation  │                              │
  │ │ (NIC, Disk, VGA)  │                              │
  │ │ ioctl(KVM_RUN)    │                              │
  │ └────────┬──────────┘                              │
  │          │                                         │
  │ KVM Kernel Module                                  │
  │ ┌────────┴─────────┐                              │
  │ │ /dev/kvm          │                              │
  │ │ VMCS management   │                              │
  │ │ VM Entry/Exit     │                              │
  │ │ EPT management    │                              │
  │ └────────┬─────────┘                              │
  │          │                                         │
  │ Hardware (VT-x/AMD-V + VT-d/AMD-Vi)              │
  └──────────────────────────────────────────────────┘

  Role of Each Component:
  KVM: CPU virtualization, memory virtualization (EPT/NPT)
  QEMU: Device emulation, VM management
  libvirt: VM management API (backend for virsh, virt-manager)
```

### 7.2 Creating and Managing VMs

```bash
# Verify KVM
lsmod | grep kvm
# kvm_intel     xxx  0
# kvm           xxx  1 kvm_intel

# Check if CPU supports virtualization
grep -E 'vmx|svm' /proc/cpuinfo

# Start a VM directly with QEMU
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 4 \
  -m 4096 \
  -drive file=disk.qcow2,if=virtio,format=qcow2 \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -display none \
  -daemonize

# Create a disk image
qemu-img create -f qcow2 disk.qcow2 50G
# qcow2: Copy-on-Write, snapshots, thin provisioning

# Disk image info
qemu-img info disk.qcow2
# image: disk.qcow2
# file format: qcow2
# virtual size: 50 GiB
# disk size: 196 MiB  ← Actual disk usage

# Snapshots
qemu-img snapshot -c snap1 disk.qcow2    # Create
qemu-img snapshot -l disk.qcow2          # List
qemu-img snapshot -a snap1 disk.qcow2    # Restore
qemu-img snapshot -d snap1 disk.qcow2    # Delete
```

### 7.3 Management with libvirt / virsh

```bash
# VM management with libvirt (recommended)

# List VMs
virsh list --all

# Start/stop VMs
virsh start myvm
virsh shutdown myvm      # Request graceful shutdown from guest OS
virsh destroy myvm       # Force stop (equivalent to power off)
virsh reboot myvm

# Create a VM (XML definition)
virsh define myvm.xml
virsh create myvm.xml    # Define and start immediately

# VM information
virsh dominfo myvm
virsh vcpuinfo myvm
virsh domblklist myvm
virsh domiflist myvm

# Console connection
virsh console myvm

# Snapshots
virsh snapshot-create-as myvm snap1 "Initial snapshot"
virsh snapshot-list myvm
virsh snapshot-revert myvm snap1
virsh snapshot-delete myvm snap1

# Dynamic resource changes
virsh setmem myvm 8G --live           # Change memory (live)
virsh setvcpus myvm 8 --live          # Change CPU count (live)

# VM monitoring
virsh domstats myvm                    # Statistics
virt-top                               # Real-time monitor
```

```xml
<!-- VM Definition File Example (myvm.xml) -->
<domain type='kvm'>
  <name>myvm</name>
  <memory unit='GiB'>4</memory>
  <vcpu placement='static'>4</vcpu>

  <cpu mode='host-passthrough' check='none' migratable='on'/>

  <os>
    <type arch='x86_64' machine='q35'>hvm</type>
    <boot dev='hd'/>
  </os>

  <features>
    <acpi/>
    <apic/>
  </features>

  <devices>
    <!-- Virtio disk -->
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' cache='writeback' discard='unmap'/>
      <source file='/var/lib/libvirt/images/myvm.qcow2'/>
      <target dev='vda' bus='virtio'/>
    </disk>

    <!-- Virtio network -->
    <interface type='network'>
      <source network='default'/>
      <model type='virtio'/>
    </interface>

    <!-- VNC console -->
    <graphics type='vnc' port='-1' autoport='yes'/>

    <!-- Virtio memory balloon -->
    <memballoon model='virtio'>
      <stats period='10'/>
    </memballoon>

    <!-- virtio-serial (for guest agent communication) -->
    <channel type='unix'>
      <target type='virtio' name='org.qemu.guest_agent.0'/>
    </channel>
  </devices>
</domain>
```

---

## 8. Live Migration

### 8.1 How Live Migration Works

```
Live Migration:
  A technology to move a VM to a different physical server without stopping it

  Pre-copy Migration (Most Common):
  ┌──────────────────────────────────────────────────┐
  │ Phase 1: Bulk Memory Copy                         │
  │   → Transfer all memory pages to the destination │
  │   → VM continues running                         │
  │                                                    │
  │ Phase 2: Delta Copy (Iterative)                   │
  │   → Transfer pages changed during Phase 1        │
  │     (dirty pages)                                  │
  │   → Transfer pages changed again (repeat)        │
  │   → Repeat until delta is small enough           │
  │                                                    │
  │ Phase 3: Final Sync (Stop-and-Copy)              │
  │   → Pause the VM                                 │
  │   → Transfer remaining dirty pages               │
  │   → Transfer CPU state, device state             │
  │   → Resume VM on destination                     │
  │   → Downtime: tens to hundreds of ms             │
  │                                                    │
  │ Phase 4: Network Switchover                       │
  │   → Update ARP (send RARP packet)               │
  │   → Reachable from outside at the same IP       │
  └──────────────────────────────────────────────────┘

  Post-copy Migration:
  ┌──────────────────────────────────────────────────┐
  │ 1. Transfer CPU state and device state first     │
  │ 2. Start VM immediately on destination           │
  │ 3. Transfer memory pages on-demand when accessed │
  │    → Handle page faults via userfaultfd          │
  │                                                    │
  │ Advantages: Very short downtime                   │
  │ Disadvantages: Performance degradation from       │
  │   page faults; VM crashes on network failure     │
  │ → Hybrid of pre-copy and post-copy also exists   │
  └──────────────────────────────────────────────────┘

  Live Migration Requirements:
  ┌──────────────────────────────────────────────────┐
  │ 1. Shared storage (NFS, Ceph, SAN)               │
  │    → Disk image accessible from both hosts       │
  │    → Not needed for storage migration            │
  │                                                    │
  │ 2. Same CPU features (or compatible)             │
  │    → Ensure compatibility with                   │
  │      cpu mode='host-model'                        │
  │    → Adjust with QEMU CPU feature masks          │
  │                                                    │
  │ 3. Sufficient network bandwidth                  │
  │    → 10Gbps or more recommended                  │
  │    → dirty page rate < transfer rate is required │
  │                                                    │
  │ 4. Same libvirt / QEMU versions                  │
  │    → Ensure protocol compatibility               │
  └──────────────────────────────────────────────────┘
```

```bash
# Executing Live Migration

# Migration with virsh
virsh migrate --live --verbose myvm \
  qemu+ssh://dest-host/system \
  --migrateuri tcp://dest-host:49152

# Migration with bandwidth limit
virsh migrate --live myvm \
  qemu+ssh://dest-host/system \
  --bandwidth 500  # 500 MiB/s

# Migration with compression (saves bandwidth)
virsh migrate --live myvm \
  qemu+ssh://dest-host/system \
  --comp-methods xbzrle

# Storage migration (no shared storage needed)
virsh migrate --live --copy-storage-all myvm \
  qemu+ssh://dest-host/system

# Check migration progress
virsh domjobinfo myvm

# Cancel migration
virsh domjobabort myvm
```

---

## 9. Performance Tuning

### 9.1 CPU Tuning

```bash
# CPU Pinning (pin vCPUs to physical CPU cores)
virsh vcpupin myvm 0 2    # vCPU 0 → Physical core 2
virsh vcpupin myvm 1 3    # vCPU 1 → Physical core 3

# NUMA node placement
virsh numatune myvm --nodeset 0 --mode strict
# → Use memory only from NUMA node 0

# Check CPU affinity
virsh vcpuinfo myvm

# Emulator thread pinning
virsh emulatorpin myvm 0-1
# → Pin QEMU emulator threads to cores 0-1

# IO thread pinning
virsh iothreadpin myvm 1 4
# → Pin IO thread 1 to core 4
```

### 9.2 Memory Tuning

```bash
# Using Huge Pages
# Add to VM definition file:
# <memoryBacking>
#   <hugepages>
#     <page size='2048' unit='KiB'/>
#   </hugepages>
# </memoryBacking>

# NUMA-aware memory placement
# <numatune>
#   <memory mode='strict' nodeset='0'/>
# </numatune>
# <cpu>
#   <numa>
#     <cell id='0' cpus='0-3' memory='4' unit='GiB'/>
#   </numa>
# </cpu>

# Memory locking (prevent swapout)
# <memoryBacking>
#   <locked/>
# </memoryBacking>
```

### 9.3 Storage Tuning

```
Storage Tuning:

  Disk Cache Modes:
  ┌──────────────┬──────────────────────────────────┐
  │ none         │ No host cache. Direct I/O         │
  │              │ → Highest data consistency        │
  │              │ → Recommended (let guest cache)   │
  │ writethrough │ Read cache, immediate write       │
  │              │ → Safe but slow writes            │
  │ writeback    │ Read/write cache                  │
  │              │ → Fast but risk of data loss      │
  │ unsafe       │ Ignores all fsync                 │
  │              │ → Test environments only          │
  │              │   (high data loss risk)            │
  │ directsync   │ Direct I/O + synchronous write   │
  │              │ → Safest but slowest              │
  └──────────────┴──────────────────────────────────┘

  I/O Scheduler Settings:
  ┌──────────────────────────────────────────────────┐
  │ Host side:                                        │
  │   SSD: none (noop) is optimal                    │
  │   HDD: mq-deadline is optimal                    │
  │                                                    │
  │ Guest side:                                       │
  │   virtio-blk: none (noop) is optimal             │
  │   → Scheduling is done on the host side          │
  └──────────────────────────────────────────────────┘

  Disk Formats:
  ┌──────────────┬──────────────────────────────────┐
  │ qcow2        │ Snapshots, compression, encryption│
  │              │ → High flexibility, slightly slow │
  │ raw          │ Simple. Highest performance        │
  │              │ → No dynamic resizing             │
  │ qcow2 +     │ Pre-allocated qcow2               │
  │ preallocation│ → Performance close to raw        │
  └──────────────┴──────────────────────────────────┘
```

---

## 10. Nested Virtualization

```
Nested Virtualization:
  Running VMs inside VMs

  ┌──────────────────────────────────────────────────┐
  │ L0: Physical hardware + Host KVM                 │
  │   └── L1: Guest VM (running KVM inside)          │
  │         └── L2: Nested VM                        │
  │                                                    │
  │ Use Cases:                                        │
  │ - Testing and developing virtualization on cloud │
  │ - VM testing in CI/CD pipelines                  │
  │ - KVM/QEMU development and debugging             │
  │ - Hypervisor security testing                    │
  │ - Education and training environments            │
  │                                                    │
  │ Performance:                                      │
  │ → About 60-80% of L1 performance                │
  │ → Additional overhead from nested VM Exits      │
  │ → VMCS shadowing (Intel) reduces overhead       │
  └──────────────────────────────────────────────────┘
```

```bash
# Enabling Nested Virtualization

# For Intel
cat /sys/module/kvm_intel/parameters/nested
# N → Disabled

# Enable (temporarily)
sudo modprobe -r kvm_intel
sudo modprobe kvm_intel nested=1

# Enable permanently
echo "options kvm_intel nested=1" | \
  sudo tee /etc/modprobe.d/kvm-nested.conf

# For AMD
echo "options kvm_amd nested=1" | \
  sudo tee /etc/modprobe.d/kvm-nested.conf

# Expose VMX/SVM in VM CPU configuration
# <cpu mode='host-passthrough'>
#   <feature policy='require' name='vmx'/>
# </cpu>
```

---

## 11. Cloud Virtualization

### 11.1 AWS Nitro System

```
AWS Nitro System:
  Offloads networking, storage, and security from the main CPU
  to dedicated hardware

  Nitro Components:
  ┌──────────────────────────────────────────────────┐
  │ EC2 Instance                                      │
  │ ┌──────────────────────────────────┐              │
  │ │ Guest VM (Customer Workload)     │              │
  │ │ → Can use nearly 100% of CPU    │              │
  │ └──────────────────────────────────┘              │
  │                                                    │
  │ Nitro Cards (Dedicated ASICs):                    │
  │ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
  │ │ Nitro    │ │ Nitro    │ │ Nitro    │          │
  │ │ Network  │ │ EBS      │ │ Security │          │
  │ │ Card     │ │ Card     │ │ Chip     │          │
  │ └──────────┘ └──────────┘ └──────────┘          │
  │ → VPC, EBS, encryption handled in hardware       │
  │ → No host CPU resource consumption               │
  │                                                    │
  │ Nitro Hypervisor:                                 │
  │ → KVM-based lightweight hypervisor               │
  │ → Replaced the legacy Xen hypervisor             │
  │ → Handles only CPU and memory virtualization     │
  │ → I/O offloaded to Nitro Cards                   │
  │                                                    │
  │ Nitro Enclaves:                                   │
  │ → Highly isolated compute environment            │
  │ → No network connection, no persistent storage   │
  │ → Communicates with parent VM via vsock only     │
  │ → Used for cryptographic key management,          │
  │   sensitive data processing                        │
  └──────────────────────────────────────────────────┘
```

### 11.2 How Cloud Instances Work

```
Cloud Instance Lifecycle:

  User → API → Control Plane
                     │
                ┌─────┴─────┐
                │ Scheduler  │
                └─────┬─────┘
                      │ Determines VM placement
                ┌─────┴─────┐
                │ Physical   │
                │ Server     │
                │ KVM + QEMU │
                │ ┌───┐┌───┐│
                │ │VM1││VM2││
                │ └───┘└───┘│
                └───────────┘

  Scheduler Placement Algorithms:
  ┌──────────────────────────────────────────────────┐
  │ 1. Resource Filtering:                            │
  │    → Hosts meeting CPU, memory, storage reqs     │
  │                                                    │
  │ 2. Affinity / Anti-affinity:                      │
  │    → Place specific VMs on same/different hosts   │
  │                                                    │
  │ 3. NUMA Optimization:                             │
  │    → Optimal placement based on NUMA topology    │
  │                                                    │
  │ 4. Availability Zones:                            │
  │    → Distribute across fault domains             │
  │                                                    │
  │ 5. Cost Optimization:                             │
  │    → Bin packing (efficient resource filling)    │
  │    → or Spread (distributed placement)           │
  └──────────────────────────────────────────────────┘
```

---

## Hands-on Exercises

### Exercise 1: [Beginner] -- Verifying Virtualization Support

```bash
# Check CPU virtualization support
grep -E 'vmx|svm' /proc/cpuinfo | head -1
# flags : ... vmx ...  → Intel VT-x supported

# Check KVM module
lsmod | grep kvm
# kvm_intel     xxxxx  0
# kvm           xxxxx  1 kvm_intel

# Check /dev/kvm
ls -la /dev/kvm
# crw-rw---- 1 root kvm 10, 232 Jan  1 00:00 /dev/kvm
```

### Exercise 2: [Beginner] -- Starting a VM with QEMU

```bash
# Create a disk image
qemu-img create -f qcow2 test.qcow2 10G

# Install with Ubuntu Server ISO
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 2 \
  -m 2048 \
  -drive file=test.qcow2,if=virtio \
  -cdrom ubuntu-server.iso \
  -boot d \
  -vnc :0

# Boot after installation
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -smp 2 \
  -m 2048 \
  -drive file=test.qcow2,if=virtio \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -nographic

# SSH connection
ssh -p 2222 user@localhost
```

### Exercise 3: [Advanced] -- VM Management with virsh

```bash
# Create a VM
virt-install \
  --name testvm \
  --ram 2048 \
  --vcpus 2 \
  --disk size=20,format=qcow2 \
  --os-variant ubuntu22.04 \
  --cdrom /path/to/ubuntu-22.04.iso \
  --network network=default \
  --graphics vnc

# Snapshot management
virsh snapshot-create-as testvm clean-install "Fresh install"
virsh snapshot-list testvm
virsh snapshot-revert testvm clean-install

# Dynamic resource changes
virsh setmem testvm 4G --live
virsh setvcpus testvm 4 --live

# VM statistics
virsh domstats testvm
virt-top
```

### Exercise 4: [Advanced] -- Performance Tuning

```bash
# CPU pinning
virsh vcpupin testvm 0 0
virsh vcpupin testvm 1 1

# Huge Pages setup
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
# Add hugepages to VM definition

# I/O tuning
# Change disk cache mode to none
virsh attach-disk testvm /path/to/disk.qcow2 vdb \
  --driver qemu --subdriver qcow2 --cache none

# Performance measurement
# Inside the guest:
fio --name=seqwrite --rw=write --bs=4k --size=1G --numjobs=4
sysbench cpu --threads=4 run
iperf3 -c host-ip
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important. Understanding deepens not just through theory, but by actually writing code and observing how it works.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this used in real-world practice?

Knowledge of this topic is frequently applied in daily development work. It becomes particularly important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Full Virtualization | Unmodified guest OS. Binary translation or HW-assisted |
| Paravirtualization | Modified guest OS. High performance via hypercalls |
| Type 1 | Bare metal. KVM, ESXi, Xen. Cloud foundation |
| Type 2 | Hosted. VirtualBox, Parallels. Dev environments |
| VT-x/AMD-V | Hardware-assisted. VMX Root/Non-Root modes |
| EPT/NPT | Hardware-assisted memory virtualization. No shadow PT needed |
| Virtio | Paravirtualized I/O. Network, storage acceleration |
| SR-IOV | Direct device passthrough. Near-native I/O performance |
| Live Migration | Pre-copy method. Downtime of tens of ms |
| KSM | Same-page merging. Memory efficiency |
| Nitro | AWS proprietary. HW offload. 100% CPU to guest |
| Nested Virtualization | VM in VM. For development and testing |

---

## Recommended Next Guides

---

## References
1. Portnoy, M. "Virtualization Essentials." 2nd Ed, Sybex, 2016.
2. Popek, G. J. & Goldberg, R. P. "Formal Requirements for Virtualizable Third Generation Architectures." Communications of the ACM, 1974.
3. Agesen, O. et al. "Software and Hardware Techniques for x86 Virtualization." VMware Technical Report, 2012.
4. Kivity, A. et al. "kvm: the Linux Virtual Machine Monitor." Proceedings of the Linux Symposium, 2007.
5. Adams, K. & Agesen, O. "A Comparison of Software and Hardware Techniques for x86 Virtualization." ASPLOS, 2006.
6. Amazon. "AWS Nitro System." AWS Documentation, 2024.
7. Red Hat. "Virtualization Deployment and Administration Guide." RHEL Documentation, 2024.
8. QEMU Project. "QEMU Documentation." qemu.org, 2024.
9. Intel. "Intel 64 and IA-32 Architectures Software Developer's Manual, Volume 3C: System Programming Guide, Part 3." Chapter 23-34 (VMX), 2024.
10. Habib, I. "Virtualization with KVM." Linux Journal, 2008.
