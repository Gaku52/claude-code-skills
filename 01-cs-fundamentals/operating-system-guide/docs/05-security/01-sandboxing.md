# Sandboxing and Isolation

> A sandbox is a technology that "confines untrusted code in a safe playground to minimize its impact on the system."

## Learning Objectives

- [ ] Understand the concepts and design principles of sandboxing
- [ ] Compare major isolation technologies
- [ ] Understand each type of Linux Namespace in detail
- [ ] Practice resource control using cgroups v2
- [ ] Know the isolation mechanisms of containers
- [ ] Understand advanced isolation technologies such as gVisor and Firecracker
- [ ] Compare sandboxes across browsers, mobile OSes, and desktop OSes
- [ ] Design seccomp-bpf filters


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of the content in [Access Control](./00-access-control.md)

---

## 1. Fundamental Concepts of Sandboxing

### 1.1 What is a Sandbox?

```
Sandbox:
  A technology that establishes trust boundaries to restrict program execution environments

  Design Principles:
  ┌──────────────────────────────────────────────────┐
  │ 1. Principle of Least Privilege:                  │
  │    Allow access to only the minimum resources     │
  │    needed                                         │
  │                                                    │
  │ 2. Isolation:                                     │
  │    Processes inside the sandbox do not affect      │
  │    the outside                                     │
  │                                                    │
  │ 3. Mediation:                                     │
  │    Inspect all resource access at checkpoints      │
  │                                                    │
  │ 4. Defense in Depth:                               │
  │    Layer multiple isolation layers for defense      │
  │    → Even if one layer is breached, others protect │
  └──────────────────────────────────────────────────┘

  Classification of Sandboxes:
  ┌──────────────────────────────────────────────────┐
  │ OS Level:                                        │
  │   → Namespaces, cgroups, chroot, jail             │
  │   → Containers (Docker, Podman)                   │
  │   → VMs (KVM, Xen, Hyper-V)                      │
  │                                                    │
  │ Application Level:                                │
  │   → Browser sandbox (Chromium)                    │
  │   → Java SecurityManager (deprecated)             │
  │   → .NET Code Access Security                     │
  │   → WebAssembly (Wasm) sandbox                    │
  │                                                    │
  │ Language Level:                                    │
  │   → Rust's ownership system                       │
  │   → Deno's permission system                      │
  │   → Wasm's linear memory model                    │
  │                                                    │
  │ Hardware Level:                                    │
  │   → Intel SGX / TDX                               │
  │   → ARM TrustZone / CCA                           │
  │   → AMD SEV-SNP                                   │
  └──────────────────────────────────────────────────┘
```

### 1.2 Comparison of Isolation Levels

```
Comparison of Isolation Levels:

  Weak Isolation ←────────────────────→ Strong Isolation
  Process    chroot  namespace   Container  gVisor   VM     TEE
  separation        + cgroup    (Docker)   (sandbox) (KVM)  (SGX)

  Details of Each Level:
  ┌──────────────────────────────────────────────────────────────┐
  │ Level       │ Isolation Target     │ Attack      │ Perf.    │
  │             │                      │ Surface     │ Impact   │
  ├─────────────┼────────────────────┼────────────┼────────────┤
  │ Process     │ Memory space only    │ Very wide  │ None       │
  │ chroot      │ + Filesystem         │ Wide       │ Negligible │
  │ Namespace   │ + PID,Net,IPC, etc.  │ Moderate   │ Minimal    │
  │ Container   │ + seccomp,cap        │ Moderate   │ Minimal    │
  │ gVisor      │ + System calls       │ Narrow     │ 10-30%     │
  │ microVM     │ + Virtualization     │ Narrow     │ 5-15%      │
  │ VM          │ Hardware level       │ Very narrow│ 5-10%      │
  │ TEE         │ + Encrypted memory   │ Minimal    │ 10-30%     │
  └─────────────┴────────────────────┴────────────┴────────────┘

  Escape Difficulty:
  ┌──────────────────────────────────────────────────┐
  │ chroot: Relatively easy                          │
  │   → Escapable if root privileges exist inside    │
  │   → Attacks using mknod, mount, ptrace, etc.     │
  │   → Insufficient as a true security boundary     │
  │                                                    │
  │ Container: Moderate difficulty                    │
  │   → Escape cases via kernel vulnerabilities exist │
  │   → CVE-2019-5736 (runc vulnerability)           │
  │   → CVE-2020-15257 (containerd vulnerability)    │
  │   → Risk can be significantly reduced with proper │
  │     configuration                                 │
  │                                                    │
  │ VM: Very difficult                                │
  │   → Requires a hypervisor vulnerability           │
  │   → Cases like VENOM (CVE-2015-3456) exist       │
  │     but are rare                                   │
  │   → Requires highly advanced techniques           │
  │                                                    │
  │ TEE: Extremely difficult                          │
  │   → Hardware-level protection                     │
  │   → Some information leakage via side-channel     │
  │     attacks has occurred                           │
  │   → Watch out for Spectre/Meltdown-class attacks │
  └──────────────────────────────────────────────────┘
```

---

## 2. chroot and FreeBSD Jail

### 2.1 chroot

```
chroot (Change Root):
  Changes the filesystem root
  → Restricts files visible to a process
  → The oldest isolation technology (1979, Unix V7)

  How chroot Works:
  ┌──────────────────────────────────────────────────┐
  │ Normal process:                                   │
  │   / (true root)                                  │
  │   ├── etc/                                        │
  │   ├── usr/                                        │
  │   ├── home/                                       │
  │   └── var/                                        │
  │                                                    │
  │ chrooted process:                                 │
  │   /srv/jail/ ← This appears as /                 │
  │   ├── etc/   (jail-internal config)               │
  │   ├── usr/   (minimal binaries)                   │
  │   └── tmp/                                        │
  │   → Cannot see outside /srv/jail/                │
  └──────────────────────────────────────────────────┘

  Limitations of chroot:
  - Escapable if root privileges are available
  - Network, process, and IPC are not isolated
  - /proc, /sys are visible if mounted
  - An environment isolation tool, not a security feature
```

```bash
# Building a chroot environment

# 1. Create directory structure
sudo mkdir -p /srv/jail/{bin,lib,lib64,etc,usr/lib,dev,proc}

# 2. Copy required binaries
sudo cp /bin/bash /srv/jail/bin/
sudo cp /bin/ls /srv/jail/bin/
sudo cp /bin/cat /srv/jail/bin/

# 3. Copy dependency libraries
# Check libraries with ldd and copy them
ldd /bin/bash
# linux-vdso.so.1 (0x00007fff...)
# libtinfo.so.6 => /lib/x86_64-linux-gnu/libtinfo.so.6
# libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6
# /lib64/ld-linux-x86-64.so.2

sudo cp /lib/x86_64-linux-gnu/libtinfo.so.6 /srv/jail/lib/
sudo cp /lib/x86_64-linux-gnu/libc.so.6 /srv/jail/lib/
sudo cp /lib64/ld-linux-x86-64.so.2 /srv/jail/lib64/

# 4. Enter the chroot
sudo chroot /srv/jail /bin/bash
# → / now points to /srv/jail
ls /        # bin lib lib64 etc usr dev proc
# → Cannot access outside /srv/jail

# 5. Example of escape (when root privileges are available)
# * For educational purposes only. Do not try on real systems
mkdir /tmp/escape
chroot /tmp/escape
cd ../../../..   # Reach the true root
chroot .         # chroot to the true root
# → chroot is insufficient as a security boundary

# Practical uses of chroot:
# - Isolating build environments (debootstrap + chroot)
# - Isolating BIND DNS server
# - Package build environments
# - Repairing broken systems (rescue mode)
```

### 2.2 FreeBSD Jail

```
FreeBSD Jail:
  Enhanced version of chroot. Also isolates processes, network, and users
  → Introduced in 2000. A precursor to containers
  → Equivalent to Linux Namespaces + cgroups

  Jail Features:
  ┌──────────────────────────────────────────────────┐
  │ Filesystem Isolation:                             │
  │   → Similar to chroot but escape is much harder  │
  │                                                    │
  │ Process Isolation:                                │
  │   → Processes inside Jail cannot see outside      │
  │   → root inside Jail is restricted                │
  │                                                    │
  │ Network Isolation:                                │
  │   → Each Jail is assigned its own IP address      │
  │   → VNET (virtual network stack) support          │
  │                                                    │
  │ User Isolation:                                    │
  │   → root inside Jail cannot access outside        │
  │   → securelevel further restricts privileges      │
  └──────────────────────────────────────────────────┘

  Jail Configuration Example (/etc/jail.conf):
  webserver {
      host.hostname = "web.jail.local";
      ip4.addr = "10.0.0.2";
      path = "/jails/webserver";
      exec.start = "/bin/sh /etc/rc";
      exec.stop = "/bin/sh /etc/rc.shutdown";
      mount.devfs;
      allow.raw_sockets;
  }

  Management Commands:
  # jail -c webserver        # Start a Jail
  # jls                      # List Jails
  # jexec webserver /bin/sh  # Execute command inside Jail
```

---

## 3. Linux Namespaces

### 3.1 Namespace Types and Details

```
Linux Namespaces:
  Isolate OS resources per process
  → Foundation of container technology

  Namespace Types:
  ┌──────────────┬──────────────────────────┬──────────┐
  │ Namespace    │ Isolation Target          │ Kernel   │
  ├──────────────┼──────────────────────────┼──────────┤
  │ PID          │ Process ID space          │ 2.6.24   │
  │ Network      │ Network stack             │ 2.6.29   │
  │ Mount        │ Mount points              │ 2.4.19   │
  │ UTS          │ Hostname and domain name  │ 2.6.19   │
  │ IPC          │ Inter-process communic.   │ 2.6.19   │
  │ User         │ UID/GID mapping           │ 3.8      │
  │ Cgroup       │ cgroup visibility         │ 4.6      │
  │ Time         │ System time (CLOCK_*)     │ 5.6      │
  └──────────────┴──────────────────────────┴──────────┘
```

### 3.2 PID Namespace

```
PID Namespace:
  Isolates the process ID space
  → PIDs start from 1 within the Namespace
  → External PIDs are not visible

  PID Namespace Hierarchy:
  ┌──────────────────────────────────────────────────┐
  │ Host PID Namespace:                               │
  │   PID 1 (systemd/init)                           │
  │   PID 100 (sshd)                                  │
  │   PID 200 (container runtime)                     │
  │     │                                              │
  │     └── Child PID Namespace:                      │
  │           PID 1 (container init) ← Host PID 201  │
  │           PID 2 (app process)    ← Host PID 202  │
  │           PID 3 (worker)         ← Host PID 203  │
  │                                                    │
  │ → Parent Namespace can see child PIDs             │
  │ → Child Namespace cannot see parent PIDs          │
  │ → When PID 1 exits, all processes in the          │
  │   Namespace receive SIGKILL                       │
  └──────────────────────────────────────────────────┘

  Special Behaviors of PID Namespace:
  - PID 1 ignores signals for which no handler is registered
  - Orphan processes are reparented to PID 1
  - Mounting /proc displays accurate process info
```

```bash
# Creating and verifying a PID Namespace

# Start a shell in a new PID Namespace
sudo unshare --pid --fork --mount-proc bash
echo $$      # PID 1
ps aux       # Shows only processes within the Namespace
# USER  PID %CPU %MEM    VSZ   RSS TTY  STAT START   TIME COMMAND
# root    1  0.0  0.0   8532  5240 pts/0 S   12:00   0:00 bash
# root    2  0.0  0.0  10068  3456 pts/0 R+  12:00   0:00 ps aux

# Verify from another terminal
ps aux | grep bash  # Visible under a different PID on the host

# Why --fork is needed:
# unshare itself enters the new Namespace, but
# PID Namespace takes effect from child processes, so
# --fork is needed to create a new process
```

### 3.3 Network Namespace

```
Network Namespace:
  Isolates the entire network stack
  → Interfaces, routing tables, firewalls, sockets
  → Foundation of container network isolation

  Network Namespace Layout:
  ┌──────────────────────────────────────────────────┐
  │ Host Namespace:                                   │
  │   eth0: 192.168.1.100/24                         │
  │   veth-host ──┐                                   │
  │               │ veth pair (virtual ethernet)      │
  │   Container Namespace:                            │
  │   veth-cont ──┘                                   │
  │   eth0: 172.17.0.2/16                            │
  │   lo: 127.0.0.1                                   │
  │   → Independent network stack                    │
  └──────────────────────────────────────────────────┘

  Docker Network Model:
  ┌──────────────────────────────────────────────────┐
  │ Host                                              │
  │ ┌──────────────────────────────────┐              │
  │ │ docker0 bridge (172.17.0.1)     │              │
  │ │  ┌─────────┐  ┌─────────┐       │              │
  │ │  │ veth1   │  │ veth2   │       │              │
  │ │  └────┬────┘  └────┬────┘       │              │
  │ └───────┼────────────┼────────────┘              │
  │         │            │                            │
  │    Container1    Container2                       │
  │    eth0:         eth0:                            │
  │    172.17.0.2    172.17.0.3                       │
  │                                                    │
  │ → External communication via NAT through host eth0│
  │ → iptables MASQUERADE rules                       │
  └──────────────────────────────────────────────────┘
```

```bash
# Network Namespace Operations

# Create a Namespace
sudo ip netns add test-ns

# List Namespaces
ip netns list

# Execute a command within the Namespace
sudo ip netns exec test-ns ip addr
# → Only the lo interface (DOWN state)

# Create and configure a veth pair
sudo ip link add veth-host type veth peer name veth-ns
sudo ip link set veth-ns netns test-ns

# Host-side configuration
sudo ip addr add 10.0.0.1/24 dev veth-host
sudo ip link set veth-host up

# Namespace-side configuration
sudo ip netns exec test-ns ip addr add 10.0.0.2/24 dev veth-ns
sudo ip netns exec test-ns ip link set veth-ns up
sudo ip netns exec test-ns ip link set lo up

# Connectivity test
ping 10.0.0.2                              # Host → Namespace
sudo ip netns exec test-ns ping 10.0.0.1   # Namespace → Host

# Internet access from Namespace (NAT configuration)
sudo ip netns exec test-ns ip route add default via 10.0.0.1
sudo iptables -t nat -A POSTROUTING -s 10.0.0.0/24 -j MASQUERADE
sudo sysctl -w net.ipv4.ip_forward=1

# Independent firewall within the Namespace
sudo ip netns exec test-ns iptables -A INPUT -p tcp --dport 80 -j ACCEPT
sudo ip netns exec test-ns iptables -A INPUT -j DROP

# Delete the Namespace
sudo ip netns delete test-ns
```

### 3.4 Mount Namespace

```
Mount Namespace:
  Isolates mount points
  → Each Namespace has a different filesystem view
  → Foundation of container filesystem isolation

  Mount Namespace Behavior:
  ┌──────────────────────────────────────────────────┐
  │ Host Mount Namespace:                             │
  │   /           (ext4)                              │
  │   /home       (ext4)                              │
  │   /tmp        (tmpfs)                             │
  │                                                    │
  │ Container Mount Namespace:                        │
  │   /           (overlay2 - container image)        │
  │   /etc/hosts  (bind mount - Docker managed)       │
  │   /proc       (procfs)                            │
  │   /sys        (sysfs - read-only)                 │
  │   /dev        (devtmpfs - restricted)             │
  │   /app/data   (volume mount - persistent data)    │
  │                                                    │
  │ → Completely independent from host filesystem     │
  │ → Share only necessary files via bind mount       │
  └──────────────────────────────────────────────────┘

  Propagation Types (Mount Propagation):
  ┌───────────┬──────────────────────────────────────┐
  │ shared    │ Mount events propagate bidirectionally│
  │ slave     │ Propagates only from parent → child  │
  │ private   │ No propagation (default)             │
  │ unbindable│ private + cannot be bind mounted     │
  └───────────┴──────────────────────────────────────┘
```

```bash
# Mount Namespace Operations

# Create a new Mount Namespace
sudo unshare --mount bash

# Mount tmpfs within the Namespace
mount -t tmpfs tmpfs /mnt
echo "isolated" > /mnt/test.txt
cat /mnt/test.txt  # "isolated"

# Verify from another terminal (host side)
ls /mnt/           # Empty → isolated

# Set private mount
mount --make-private /

# Remount /proc inside Namespace (combined with PID Namespace)
sudo unshare --pid --fork --mount bash
mount -t proc proc /proc
ps aux  # Only processes within the Namespace
```

### 3.5 User Namespace

```
User Namespace:
  Isolates UID/GID mappings
  → root (UID 0) inside the Namespace is an unprivileged user outside
  → Foundation technology for rootless containers

  User Namespace Mapping:
  ┌──────────────────────────────────────────────────┐
  │ Host:                                             │
  │   alice (UID 1000)                                │
  │                                                    │
  │ Inside User Namespace:                            │
  │   root (UID 0) ← Actually alice (UID 1000)       │
  │   → Operates as root within the Namespace         │
  │   → Has only alice's permissions on the host      │
  │   → Files created in the Namespace are            │
  │     owned by alice on the host                     │
  └──────────────────────────────────────────────────┘

  Contents of /proc/PID/uid_map:
  # Namespace UID  Host UID  Mapping count
  0                 1000       1
  → UID 0 in the Namespace = UID 1000 on the host

  What User Namespace Enables:
  - Namespace creation by unprivileged users
  - mount operations within the Namespace
  - chroot within the Namespace
  - Rootless container implementation
  - → Significant security benefits
```

```bash
# User Namespace Operations

# Create a User Namespace (possible even as unprivileged user)
unshare --user --map-root-user bash
id
# uid=0(root) gid=0(root) groups=0(root)
# → Appears as root within the Namespace

whoami  # root (inside the Namespace)

# Check UID mapping
cat /proc/self/uid_map
# 0  1000  1
# → UID 0 in the Namespace = UID 1000 on the host

# File creation test
touch /tmp/test-user-ns
ls -la /tmp/test-user-ns
# → On the host, owned by alice (UID 1000)

# Rootless container (Podman)
podman run --rm -it alpine sh
# → Run container without root privileges
# → root inside via User Namespace, regular user outside
```

### 3.6 UTS, IPC, Cgroup, and Time Namespaces

```
UTS Namespace:
  Isolates hostname and domain name
  → Each container can have a different hostname
  → UTS = Unix Time Sharing

IPC Namespace:
  Isolates System V IPC and POSIX message queues
  → Shared memory, semaphores, message queues
  → Prevents data leakage between Namespaces

Cgroup Namespace:
  Isolates cgroup visibility
  → Containers see only their own cgroup tree
  → Host cgroup structure is hidden

Time Namespace (Linux 5.6+):
  Isolates CLOCK_MONOTONIC and CLOCK_BOOTTIME
  → Useful for live migration of containers
  → Host boot time and container boot time can be independent
```

```bash
# Checking each Namespace

# Check current Namespaces
ls -la /proc/self/ns/
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 cgroup -> 'cgroup:[4026531835]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 ipc -> 'ipc:[4026531839]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 mnt -> 'mnt:[4026531840]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 net -> 'net:[4026531992]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 pid -> 'pid:[4026531836]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 user -> 'user:[4026531837]'
# lrwxrwxrwx 1 user user 0 Jan 1 12:00 uts -> 'uts:[4026531838]'

# UTS Namespace (hostname isolation)
sudo unshare --uts bash
hostname container-1
hostname    # container-1
# In another terminal: hostname → Host name is unchanged

# IPC Namespace
sudo unshare --ipc bash
ipcs        # IPC resources start empty
# → Host shared memory and semaphores are not visible

# Create multiple Namespaces simultaneously
sudo unshare --pid --fork --mount-proc \
  --net --uts --ipc --user --map-root-user bash
# → Fully isolated environment (approximately = container)
```

---

## 4. cgroups (Control Groups)

### 4.1 cgroups v1 vs v2

```
cgroups (Control Groups):
  A mechanism to limit, monitor, and isolate resource usage of processes
  → Introduced in kernel 2.6.24 (v1)
  → v2 introduced in kernel 4.5
  → Foundation of container resource limiting

  cgroups v1 vs v2:
  ┌─────────────┬──────────────────┬──────────────────┐
  │ Item        │ v1               │ v2               │
  ├─────────────┼──────────────────┼──────────────────┤
  │ Hierarchy   │ Independent per  │ Unified single   │
  │             │ controller       │ hierarchy        │
  │ Management  │ Complex          │ Simple           │
  │ Pressure    │ None             │ PSI support      │
  │ monitoring  │                  │                  │
  │ Memory mgmt│ Sometimes        │ Accurate         │
  │             │ inaccurate       │                  │
  │ I/O control │ blkio            │ io (improved)    │
  │ Status      │ Legacy           │ Recommended      │
  └─────────────┴──────────────────┴──────────────────┘

  cgroups v2 Hierarchy:
  /sys/fs/cgroup/                   ← Root cgroup
  ├── cgroup.controllers            ← Available controllers
  ├── cgroup.subtree_control        ← Controllers enabled for subtree
  ├── system.slice/                 ← systemd services
  │   ├── nginx.service/
  │   │   ├── cgroup.procs          ← Process ID list
  │   │   ├── memory.max            ← Memory limit
  │   │   ├── memory.current        ← Current memory usage
  │   │   ├── cpu.max               ← CPU limit
  │   │   └── io.max                ← I/O limit
  │   └── postgresql.service/
  └── user.slice/                   ← User sessions
      └── user-1000.slice/
          └── session-1.scope/

  Major Controllers (v2):
  ┌──────────┬──────────────────────────────────────┐
  │ cpu      │ CPU time limiting and weighting       │
  │ cpuset   │ CPU core and memory node assignment   │
  │ memory   │ Memory usage limiting and monitoring  │
  │ io       │ Block I/O limiting                    │
  │ pids     │ Process count limiting                │
  │ rdma     │ RDMA resource limiting                │
  │ hugetlb  │ Huge Pages limiting                   │
  │ misc     │ Other resources (DRM, etc.)           │
  └──────────┴──────────────────────────────────────┘
```

### 4.2 cgroups v2 in Practice

```bash
# Verify cgroups v2
mount | grep cgroup2
# cgroup2 on /sys/fs/cgroup type cgroup2 (rw,nosuid,nodev,noexec,relatime)

# Check available controllers
cat /sys/fs/cgroup/cgroup.controllers
# cpuset cpu io memory hugetlb pids rdma misc

# ========================================
# Setting Memory Limits
# ========================================

# Create a cgroup
sudo mkdir /sys/fs/cgroup/myapp

# Enable controllers for subtree
echo "+memory +cpu +io +pids" | \
  sudo tee /sys/fs/cgroup/cgroup.subtree_control

# Set memory limit
echo 256M | sudo tee /sys/fs/cgroup/myapp/memory.max
echo 200M | sudo tee /sys/fs/cgroup/myapp/memory.high
# memory.max: Hard limit (OOM Killer triggers when exceeded)
# memory.high: Soft limit (throttling when exceeded)

# Limit swap
echo 0 | sudo tee /sys/fs/cgroup/myapp/memory.swap.max

# Add process
echo $$ | sudo tee /sys/fs/cgroup/myapp/cgroup.procs

# Check memory usage
cat /sys/fs/cgroup/myapp/memory.current    # Current usage
cat /sys/fs/cgroup/myapp/memory.stat       # Detailed stats
cat /sys/fs/cgroup/myapp/memory.events     # Events like OOM

# ========================================
# Setting CPU Limits
# ========================================

# CPU time limit (50%)
echo "50000 100000" | sudo tee /sys/fs/cgroup/myapp/cpu.max
# Can use 50000us of CPU time per 100000us period = 50%

# CPU weight (relative priority)
echo 100 | sudo tee /sys/fs/cgroup/myapp/cpu.weight
# Default: 100, Range: 1-10000
# A group with weight=200 gets twice the CPU time as weight=100

# CPU core assignment
echo "0-1" | sudo tee /sys/fs/cgroup/myapp/cpuset.cpus
# Can only use CPU 0 and CPU 1

# ========================================
# Setting I/O Limits
# ========================================

# Check devices
lsblk
# sda  8:0

# I/O bandwidth limit
echo "8:0 rbps=10485760 wbps=5242880" | \
  sudo tee /sys/fs/cgroup/myapp/io.max
# sda read: 10MB/s, write: 5MB/s

# I/O weight
echo "8:0 200" | sudo tee /sys/fs/cgroup/myapp/io.weight
# Default: 100, Range: 1-10000

# ========================================
# Limiting PID Count
# ========================================

# Process count limit (fork bomb protection)
echo 100 | sudo tee /sys/fs/cgroup/myapp/pids.max
# → Up to 100 processes

# Current process count
cat /sys/fs/cgroup/myapp/pids.current

# ========================================
# PSI (Pressure Stall Information) Monitoring
# ========================================

# Check resource pressure
cat /sys/fs/cgroup/myapp/memory.pressure
# some avg10=0.00 avg60=0.00 avg300=0.00 total=0
# full avg10=0.00 avg60=0.00 avg300=0.00 total=0
# → some: some tasks are waiting, full: all tasks are waiting

cat /sys/fs/cgroup/myapp/cpu.pressure
cat /sys/fs/cgroup/myapp/io.pressure

# Set PSI notification (notify when memory pressure exceeds 500ms in 5s)
echo "some 500000 5000000" > /sys/fs/cgroup/myapp/memory.pressure
# → Monitorable via epoll/poll

# Delete the cgroup
echo $$ | sudo tee /sys/fs/cgroup/cgroup.procs  # Move process out
sudo rmdir /sys/fs/cgroup/myapp
```

### 4.3 systemd and cgroups

```
systemd uses cgroups v2 to manage service resources:

  Resource Limits in Service Files:
  ┌──────────────────────────────────────────────────┐
  │ /etc/systemd/system/myapp.service                │
  │                                                    │
  │ [Service]                                         │
  │ # Memory limits                                   │
  │ MemoryMax=512M          # Hard limit              │
  │ MemoryHigh=400M         # Soft limit              │
  │ MemorySwapMax=0         # Disable swap            │
  │                                                    │
  │ # CPU limits                                      │
  │ CPUQuota=200%           # 2 cores worth of CPU    │
  │ CPUWeight=50            # Low priority             │
  │ AllowedCPUs=0-3         # Usable CPU cores        │
  │                                                    │
  │ # I/O limits                                      │
  │ IOWeight=100                                      │
  │ IOReadBandwidthMax=/dev/sda 50M                   │
  │ IOWriteBandwidthMax=/dev/sda 20M                  │
  │                                                    │
  │ # Process count limit                             │
  │ TasksMax=64                                       │
  │                                                    │
  │ # Security hardening                              │
  │ ProtectSystem=strict    # Make / read-only        │
  │ ProtectHome=true        # Hide /home              │
  │ PrivateTmp=true         # Independent /tmp        │
  │ NoNewPrivileges=true    # Deny privilege escalation│
  │ PrivateDevices=true     # Restrict device access  │
  └──────────────────────────────────────────────────┘
```

```bash
# Resource monitoring with systemd
systemd-cgtop                    # cgroup resource usage overview
systemctl status myapp.service   # Service status and cgroup info
systemctl show myapp.service --property=MemoryMax
systemctl show myapp.service --property=CPUQuota

# Change resource limits on a running service
sudo systemctl set-property myapp.service MemoryMax=1G
sudo systemctl set-property myapp.service CPUQuota=150%

# Create a slice (grouping related services)
# /etc/systemd/system/myapp.slice
# [Slice]
# MemoryMax=2G
# CPUQuota=400%
#
# → Specify Slice=myapp.slice in the service file
```

---

## 5. Virtual Machines vs Containers

### 5.1 Architecture Comparison

```
┌──────────────────────────────────────────┐
│ Virtual Machine                          │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │App A │ │App B │ │App C │              │
│ │Guest │ │Guest │ │Guest │              │
│ │ OS   │ │ OS   │ │ OS   │              │
│ └──┬───┘ └──┬───┘ └──┬───┘              │
│ ┌──┴────────┴────────┴───┐              │
│ │ Hypervisor (KVM/Xen)    │              │
│ └──────────────────────────┘              │
│ ┌──────────────────────────┐              │
│ │ Host OS + Hardware       │              │
│ └──────────────────────────┘              │
│ → Full isolation, different OSes possible │
│ → Large overhead, boot takes secs to mins │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ Container                                │
│ ┌──────┐ ┌──────┐ ┌──────┐              │
│ │App A │ │App B │ │App C │              │
│ │Libs  │ │Libs  │ │Libs  │              │
│ └──┬───┘ └──┬───┘ └──┬───┘              │
│ ┌──┴────────┴────────┴───┐              │
│ │ Container Runtime       │              │
│ │ (Docker/containerd)     │              │
│ └──────────────────────────┘              │
│ ┌──────────────────────────┐              │
│ │ Host OS (shared kernel)  │              │
│ └──────────────────────────┘              │
│ → Shared kernel, lightweight              │
│ → Boots in ms, memory efficient           │
└──────────────────────────────────────────┘

Detailed Comparison:
┌────────────┬──────────────────┬──────────────────┐
│ Item       │ VM               │ Container        │
├────────────┼──────────────────┼──────────────────┤
│ Isolation  │ Strong (HW sep.) │ Moderate (OS sep)│
│ Boot       │ Secs to mins     │ ms to secs       │
│ Size       │ GB               │ MB               │
│ Density    │ Tens per host    │ 100s-1000s/host  │
│ OS         │ Different OS ok  │ Shares host OS   │
│ Kernel     │ Independent      │ Shared           │
│ Performance│ Near native      │ Native           │
│ Security   │ High             │ Moderate         │
│ Live       │ Possible         │ Difficult        │
│ Migration  │                  │                  │
│ Use case   │ Multi-tenant     │ Microservices    │
│ Examples   │ EC2, GCE         │ ECS, GKE, EKS   │
└────────────┴──────────────────┴──────────────────┘
```

### 5.2 Intermediate Technologies: gVisor, Firecracker, Kata Containers

```
gVisor (Google):
  User-space kernel
  → Processes application system calls in user space
  → Significantly reduces the attack surface against the host kernel

  gVisor Architecture:
  ┌──────────────────────────────────────────────────┐
  │ Application                                      │
  │       ↓ System call                              │
  │ Sentry (User-space kernel)                       │
  │   → Implements ~200 system calls in user space   │
  │   → Memory management, filesystem, networking    │
  │       ↓ Limited system calls                     │
  │ Gofer (Filesystem proxy)                         │
  │       ↓                                           │
  │ Host Kernel (restricted with seccomp)            │
  │                                                    │
  │ Features:                                         │
  │ - Implemented in Go (memory safe)                │
  │ - OCI compatible (usable with Docker, K8s)       │
  │ - Uses ptrace or KVM as platform                 │
  │ - Performance overhead: 10-30% (workload dep.)   │
  │ - Used in Google Cloud Run                       │
  └──────────────────────────────────────────────────┘

Firecracker (Amazon):
  microVM (ultra-lightweight VM)
  → Foundation of AWS Lambda and Fargate

  Firecracker Architecture:
  ┌──────────────────────────────────────────────────┐
  │ Features:                                         │
  │ - Implemented in Rust (memory safe)              │
  │ - Boot time: under 125ms                         │
  │ - Memory: under 5MB overhead                     │
  │ - Minimal device model (virtio only)             │
  │ - Full VM isolation based on KVM                 │
  │                                                    │
  │ Differences from regular VMs:                     │
  │ - No BIOS/UEFI → Direct kernel boot             │
  │ - No USB, PCI, graphics                          │
  │ - Only virtio-net, virtio-block                  │
  │ - → Very small attack surface                    │
  │                                                    │
  │ Use Cases:                                        │
  │ - Serverless computing                            │
  │ - Multi-tenant environments                       │
  │ - Can run thousands of microVMs per host         │
  └──────────────────────────────────────────────────┘

Kata Containers:
  Runs containers inside lightweight VMs
  → VM isolation + container compatibility

  Kata Containers Architecture:
  ┌──────────────────────────────────────────────────┐
  │ Kubernetes / Docker                               │
  │       ↓ CRI / OCI                                │
  │ Kata Runtime                                      │
  │       ↓                                           │
  │ ┌─────────────────────┐                           │
  │ │ Lightweight VM       │                           │
  │ │ (QEMU/CLH)           │                           │
  │ │ ┌─────────────────┐ │                           │
  │ │ │ Guest Kernel    │ │                           │
  │ │ │ + kata-agent    │ │                           │
  │ │ │ + Container     │ │                           │
  │ │ └─────────────────┘ │                           │
  │ └─────────────────────┘                           │
  │                                                    │
  │ → Creates a VM per Pod                            │
  │ → Maintains OCI compatibility for containers     │
  │ → Appears as a normal container from Kubernetes  │
  │ → Further lightweight with Cloud Hypervisor (CLH)│
  └──────────────────────────────────────────────────┘

Comparison of Isolation Technologies:
┌──────────────┬──────────┬──────────┬─────────────┐
│ Technology   │ Boot Time│ Memory   │ Security    │
├──────────────┼──────────┼──────────┼─────────────┤
│ Docker       │ ~100ms   │ ~10MB    │ Moderate    │
│ gVisor       │ ~150ms   │ ~30MB    │ Mod-High    │
│ Firecracker  │ ~125ms   │ ~5MB     │ High        │
│ Kata         │ ~500ms   │ ~30MB    │ High        │
│ Regular VM   │ ~5s      │ ~500MB   │ Very High   │
└──────────────┴──────────┴──────────┴─────────────┘
```

---

## 6. Application Sandboxes

### 6.1 Browser Sandbox

```
Chromium Sandbox Architecture:
  One of the most widely used sandboxes in the world

  Multi-process Architecture:
  ┌──────────────────────────────────────────────────┐
  │ Browser Process (privileged process)              │
  │ → Handles UI, network, file access               │
  │ → The only high-privilege process                 │
  │                                                    │
  │ Renderer Process (sandboxed)                      │
  │ → Renders HTML/CSS/JS                            │
  │ → Independent process per site (Site Isolation)  │
  │ → System calls restricted via seccomp-bpf        │
  │ → Filesystem isolated via Namespaces             │
  │ → No network access (requests via IPC)           │
  │                                                    │
  │ GPU Process                                       │
  │ → Handles graphics processing                    │
  │ → Moderately sandboxed                           │
  │                                                    │
  │ Plugin Process                                    │
  │ → Runs extensions                                │
  │ → Has its own sandbox                            │
  │                                                    │
  │ Network Service                                   │
  │ → Handles network communication                  │
  │ → Sandboxed                                      │
  └──────────────────────────────────────────────────┘

  Chromium Sandbox Layers:
  ┌──────────────────────────────────────────────────┐
  │ Layer 1: Linux Namespace                          │
  │   → PID Namespace: Hides other processes         │
  │   → Network Namespace: Blocks direct communication│
  │   → User Namespace: Runs as unprivileged user    │
  │                                                    │
  │ Layer 2: seccomp-bpf                              │
  │   → Restricts available system calls to minimum  │
  │   → Blocks open, exec, socket, etc.              │
  │   → Needed I/O delegated to Browser Process      │
  │     via IPC                                       │
  │                                                    │
  │ Layer 3: Filesystem Restrictions                  │
  │   → chroot + pivot_root                           │
  │   → Minimal /proc mount                          │
  │                                                    │
  │ Layer 4: Process Level                            │
  │   → No New Privileges (PR_SET_NO_NEW_PRIVS)      │
  │   → Capabilities dropped                          │
  │                                                    │
  │ → 4 layers of defense minimize vulnerability      │
  │   impact                                           │
  └──────────────────────────────────────────────────┘

  Site Isolation (Spectre Countermeasure):
  → Different sites run in different processes
  → Memory spaces between sites are fully separated
  → Prevents cross-site attacks via Spectre/Meltdown
  → Enabled by default since Chrome 67
```

### 6.2 Mobile OS Sandboxes

```
iOS Sandbox:
  ┌──────────────────────────────────────────────────┐
  │ App Sandbox:                                      │
  │ → Each app runs in an independent container       │
  │ → Cannot access outside its home directory        │
  │                                                    │
  │ Directory Structure:                              │
  │ /var/mobile/Containers/                           │
  │   Bundle/Application/UUID/        ← App binary    │
  │   Data/Application/UUID/          ← App data      │
  │     ├── Documents/                ← User data     │
  │     ├── Library/                  ← Config, cache │
  │     └── tmp/                      ← Temp files    │
  │                                                    │
  │ Security Mechanisms:                              │
  │ 1. Code Signing: All apps require Apple signature │
  │ 2. Entitlements: Per-feature permission declaration│
  │ 3. Sandbox profiles: TrustedBSD MAC-based        │
  │ 4. ASLR: Address space randomization             │
  │ 5. PAC (Pointer Authentication): Pointer signing │
  │ 6. MTE (Memory Tagging, A17+): Memory tags       │
  │                                                    │
  │ Inter-app Communication:                          │
  │ → URL Scheme, App Extensions, Shared Keychain    │
  │ → Requires explicit permission                   │
  └──────────────────────────────────────────────────┘

Android Sandbox:
  ┌──────────────────────────────────────────────────┐
  │ App Isolation:                                    │
  │ → Each app is assigned a unique Linux UID        │
  │ → SELinux type enforcement                       │
  │ → seccomp-bpf system call restrictions           │
  │                                                    │
  │ Security Layers:                                  │
  │ 1. Linux UID separation: Process isolation        │
  │ 2. SELinux: Mandatory access control             │
  │ 3. seccomp: System call restriction              │
  │ 4. Permission model: API access control          │
  │ 5. Verified Boot: Boot integrity verification    │
  │ 6. dm-verity: System partition verification      │
  │                                                    │
  │ Android Permissions:                              │
  │ - Normal: Auto-granted (internet access, etc.)   │
  │ - Dangerous: Requires explicit user permission   │
  │   → Camera, location, contacts, storage, etc.    │
  │ - Signature: Only apps with same signature       │
  │ - Special: Manual permission in system settings  │
  │                                                    │
  │ Android 10+ Storage Sandbox:                      │
  │ → Scoped Storage                                  │
  │ → Apps can freely access only their own directory│
  │ → Other files accessible via MediaStore API      │
  │ → Explicit selection via Storage Access Framework│
  └──────────────────────────────────────────────────┘
```

### 6.3 Desktop OS Sandboxes

```
macOS Sandbox:
  ┌──────────────────────────────────────────────────┐
  │ App Sandbox (mandatory for App Store apps):       │
  │ → Based on TrustedBSD MAC framework              │
  │ → Declares permissions via Entitlements          │
  │                                                    │
  │ Key Entitlements:                                 │
  │ - com.apple.security.app-sandbox: Sandboxing     │
  │ - com.apple.security.files.user-selected.read-only│
  │ - com.apple.security.network.client               │
  │ - com.apple.security.network.server               │
  │ - com.apple.security.device.camera                │
  │ - com.apple.security.device.microphone            │
  │                                                    │
  │ Gatekeeper:                                       │
  │ → Prevents unsigned apps from running            │
  │ → Notarization (pre-scan by Apple)               │
  │                                                    │
  │ SIP (System Integrity Protection):                │
  │ → Protects system files in /System, /usr, /bin   │
  │ → Cannot be modified even by root                │
  └──────────────────────────────────────────────────┘

Windows Sandbox:
  ┌──────────────────────────────────────────────────┐
  │ Windows Sandbox:                                   │
  │ → Disposable lightweight VM (Windows 10 Pro+)    │
  │ → Shares the host kernel (lightweight)           │
  │ → All data erased when closed                    │
  │ → Useful for verifying suspicious files          │
  │                                                    │
  │ WDAC (Windows Defender Application Control):      │
  │ → Application whitelisting                       │
  │ → Code signing-based control                     │
  │                                                    │
  │ AppContainer:                                     │
  │ → Sandbox for UWP apps                           │
  │ → Filesystem, registry, network restrictions     │
  │ → Declares permissions via Capabilities          │
  │                                                    │
  │ Hyper-V Based Protection:                         │
  │ → VBS (Virtualization-Based Security)            │
  │ → HVCI (Hypervisor-protected Code Integrity)     │
  │ → Credential Guard: Protects credentials in      │
  │   an isolated VM                                  │
  └──────────────────────────────────────────────────┘

Linux Sandbox:
  ┌──────────────────────────────────────────────────┐
  │ Flatpak:                                          │
  │ → Sandboxing for desktop applications            │
  │ → Uses bubblewrap (no setuid required)           │
  │ → Portals: APIs for file pickers, etc.           │
  │ → Restrictions: --filesystem, --device, --socket │
  │                                                    │
  │ Snap:                                             │
  │ → Ubuntu application isolation                   │
  │ → AppArmor profile-based                         │
  │ → Installed from Snap Store                      │
  │                                                    │
  │ Firejail:                                         │
  │ → Tool for sandboxing existing applications      │
  │ → Namespace + seccomp + Capabilities              │
  │ → Example: firejail --seccomp firefox            │
  │                                                    │
  │ bubblewrap (bwrap):                               │
  │ → Low-level sandboxing tool                      │
  │ → Foundation of Flatpak, GNOME                   │
  │ → User Namespace-based (no setuid required)      │
  └──────────────────────────────────────────────────┘
```

### 6.4 WebAssembly Sandbox

```
WebAssembly (Wasm) Sandbox:
  ┌──────────────────────────────────────────────────┐
  │ Wasm in the Browser:                              │
  │ → Linear memory model (isolated from host memory)│
  │ → Type safe (resistant to buffer overflows)      │
  │ → No direct filesystem or network access         │
  │ → Communicates externally only via JavaScript API│
  │                                                    │
  │ WASI (WebAssembly System Interface):              │
  │ → Capability-based security                      │
  │ → Can only access explicitly provided file       │
  │   descriptors                                     │
  │ → Expected as the "next" isolation technology    │
  │   after containers                                │
  │                                                    │
  │ Wasm Advantages:                                  │
  │ - Boot time: microsecond level                   │
  │ - Size: kilobytes to megabytes                   │
  │ - Portability: Runs on any platform              │
  │ - Security: Least privilege by default           │
  │                                                    │
  │ Wasm Runtimes:                                    │
  │ - Wasmtime (Bytecode Alliance)                    │
  │ - Wasmer                                          │
  │ - WasmEdge (CNCF)                                 │
  │ - wazero (Go native)                              │
  │                                                    │
  │ Solomon Hykes (Docker founder):                   │
  │ "If WASM+WASI existed in 2008, we wouldn't       │
  │  have needed to create Docker."                   │
  └──────────────────────────────────────────────────┘
```

---

## 7. seccomp-bpf in Detail

### 7.1 Designing seccomp-bpf Filters

```
seccomp-bpf:
  Filters process system calls using BPF
  (Berkeley Packet Filter) programs

  Filter Operation:
  ┌──────────────────────────────────────────────────┐
  │ Application                                      │
  │       ↓ System call                              │
  │ seccomp-bpf filter                               │
  │   → ALLOW: Permit the system call               │
  │   → KILL: SIGKILL the process                   │
  │   → TRAP: Send SIGSYS signal                    │
  │   → ERRNO: Return error number                  │
  │   → TRACE: Notify ptrace                        │
  │   → LOG: Log only (permit)                      │
  │   → USER_NOTIF: Notify user space               │
  │       ↓                                           │
  │ Kernel                                            │
  └──────────────────────────────────────────────────┘
```

```c
/* seccomp-bpf filter implementation example in C */
#include <linux/seccomp.h>
#include <linux/filter.h>
#include <linux/audit.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stddef.h>

/* BPF filter: Only allow write, exit_group, sigreturn */
static struct sock_filter filter[] = {
    /* Verify architecture */
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
             offsetof(struct seccomp_data, arch)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K,
             AUDIT_ARCH_X86_64, 1, 0),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),

    /* Get system call number */
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
             offsetof(struct seccomp_data, nr)),

    /* Allowed system calls */
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_write, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_exit_group, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_rt_sigreturn, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),

    /* Deny everything else */
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
};

int main() {
    struct sock_fprog prog = {
        .len = sizeof(filter) / sizeof(filter[0]),
        .filter = filter,
    };

    /* Set NO_NEW_PRIVS (disable SUID) */
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);

    /* Apply seccomp filter */
    prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog);

    /* From here, only write, exit_group, sigreturn are usable */
    write(1, "Hello, sandboxed world!\n", 24);

    return 0;
}
```

### 7.2 Simplified Configuration with libseccomp

```c
/* Configuration using libseccomp (simpler) */
#include <seccomp.h>
#include <unistd.h>

int main() {
    /* Default action: KILL */
    scmp_filter_ctx ctx = seccomp_init(SCMP_ACT_KILL_PROCESS);

    /* Allowed system calls */
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(rt_sigreturn), 0);

    /* Conditional allow based on arguments */
    /* write only to fd=1 (stdout) and fd=2 (stderr) */
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_EQ, STDOUT_FILENO));
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
                     SCMP_A0(SCMP_CMP_EQ, STDERR_FILENO));

    /* Load the filter */
    seccomp_load(ctx);
    seccomp_release(ctx);

    write(1, "Sandboxed!\n", 11);
    return 0;
}
```

---

## 8. Confidential Computing

```
Confidential Computing:
  Technology that protects data with encryption even while "in use"
  → Conventional: Encryption at rest + encryption in transit
  → Added: Encryption in use

  TEE (Trusted Execution Environment):
  ┌──────────────────────────────────────────────────┐
  │ Intel SGX (Software Guard Extensions):            │
  │ → Enclave: Encrypted memory region               │
  │ → Only the CPU can decrypt                       │
  │ → Unreadable by OS/hypervisor                    │
  │ → Verify authenticity via Remote Attestation     │
  │ → Use cases: Key management, ML model protection │
  │                                                    │
  │ Intel TDX (Trust Domain Extensions):              │
  │ → Encrypts entire VMs (VM-level TEE)             │
  │ → Handles larger workloads than SGX              │
  │ → Adopted in Azure Confidential VMs              │
  │                                                    │
  │ AMD SEV-SNP (Secure Encrypted Virtualization):    │
  │ → Encrypts VM memory with AES                    │
  │ → SNP: Adds integrity protection via Secure      │
  │   Nested Paging                                    │
  │ → Adopted by AWS, Google Cloud                    │
  │                                                    │
  │ ARM CCA (Confidential Compute Architecture):      │
  │ → Introduced in ARMv9                             │
  │ → Realm: Isolated execution environment           │
  │ → Confidential computing for mobile/edge devices │
  └──────────────────────────────────────────────────┘

  Use Cases:
  - Multi-party computation: Joint analysis while protecting data from multiple organizations
  - Medical data analysis: Machine learning on patient data without decryption
  - Blockchain: Confidential execution of smart contracts
  - Finance: Secure processing of customer data
```

---

## Hands-on Exercises

### Exercise 1: [Beginner] -- Verifying Namespaces

```bash
# Check current Namespaces
ls -la /proc/self/ns/

# Execute command in a new Namespace (requires root)
sudo unshare --pid --fork --mount-proc bash
ps aux  # A separate world starting from PID 1

# Compare Namespace IDs
readlink /proc/self/ns/pid
# → Different IDs on host vs inside Namespace
```

### Exercise 2: [Beginner] -- Network Namespace

```bash
# Create and communicate across Network Namespaces
sudo ip netns add ns1
sudo ip netns add ns2

# Connect ns1 and ns2 with a veth pair
sudo ip link add veth1 type veth peer name veth2
sudo ip link set veth1 netns ns1
sudo ip link set veth2 netns ns2

sudo ip netns exec ns1 ip addr add 10.0.0.1/24 dev veth1
sudo ip netns exec ns1 ip link set veth1 up
sudo ip netns exec ns1 ip link set lo up

sudo ip netns exec ns2 ip addr add 10.0.0.2/24 dev veth2
sudo ip netns exec ns2 ip link set veth2 up
sudo ip netns exec ns2 ip link set lo up

# Connectivity test
sudo ip netns exec ns1 ping -c 3 10.0.0.2

# Cleanup
sudo ip netns delete ns1
sudo ip netns delete ns2
```

### Exercise 3: [Advanced] -- Resource Limiting with cgroups

```bash
# Memory limiting with cgroup v2 (Linux)
sudo mkdir /sys/fs/cgroup/test

# Enable controllers
echo "+memory +pids" | sudo tee /sys/fs/cgroup/cgroup.subtree_control

# Memory limit
echo 50M | sudo tee /sys/fs/cgroup/test/memory.max
echo 30M | sudo tee /sys/fs/cgroup/test/memory.high

# PID count limit
echo 10 | sudo tee /sys/fs/cgroup/test/pids.max

# Add process
echo $$ | sudo tee /sys/fs/cgroup/test/cgroup.procs

# Check memory usage
cat /sys/fs/cgroup/test/memory.current
cat /sys/fs/cgroup/test/memory.stat

# Stress test
python3 -c "
data = []
try:
    while True:
        data.append('x' * 1024 * 1024)  # Allocate 1MB at a time
except MemoryError:
    print(f'OOM at {len(data)} MB')
"

# Cleanup
echo $$ | sudo tee /sys/fs/cgroup/cgroup.procs
sudo rmdir /sys/fs/cgroup/test
```

### Exercise 4: [Advanced] -- Building a Fully Isolated Environment

```bash
# Create a simple container with Namespace + cgroup

# 1. Prepare root filesystem
sudo debootstrap --variant=minbase focal /srv/container

# 2. Start with full isolation
sudo unshare --pid --fork --mount-proc \
  --net --uts --ipc --user --map-root-user \
  --mount \
  chroot /srv/container /bin/bash

# 3. Verify inside the isolated environment
hostname isolated-container
ps aux           # Only PID 1
ip addr          # Only lo
whoami           # root (actually an unprivileged user)
cat /etc/os-release  # Ubuntu Focal
```

### Exercise 5: [Production] -- Hardening Docker Container Security

```bash
# Docker container with minimum privileges
docker run --rm -it \
  --cap-drop ALL \
  --cap-add NET_BIND_SERVICE \
  --security-opt no-new-privileges:true \
  --security-opt seccomp=default.json \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  --user 1000:1000 \
  --pids-limit 64 \
  --memory 256m \
  --memory-swap 256m \
  --cpus 0.5 \
  nginx:alpine

# Using gVisor (stronger isolation)
docker run --runtime=runsc --rm -it alpine sh

# Rootless Docker
dockerd-rootless-setuptool.sh install
docker context use rootless
docker run --rm hello-world
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

| Technology | Isolation Level | Use Case |
|------------|----------------|----------|
| chroot | Weak | Simple FS isolation, build environments |
| FreeBSD Jail | Moderate | Server isolation (FreeBSD) |
| Namespace | Moderate | Foundation of containers |
| cgroup | Resource limiting | Containers, multi-tenant |
| seccomp-bpf | System call restriction | Containers, browsers |
| Container | Moderate | Microservices, CI/CD |
| gVisor | Moderate-Strong | Serverless, multi-tenant |
| Firecracker | Strong | Serverless (Lambda/Fargate) |
| Kata | Strong | Security-focused containers |
| VM | Strong | Multi-tenant, different OSes |
| TEE | Strongest | Confidential computing, finance, medical |
| Wasm | Moderate-Strong | Edge, plugins, browsers |

---

## Recommended Next Guides

---

## References
1. Lieberman, H. "Container Security." O'Reilly, 2020.
2. Provos, N. "Preventing Privilege Escalation." USENIX Security, 2003.
3. Rice, L. "Container Security: Fundamental Technology Concepts that Protect Containerized Applications." O'Reilly, 2020.
4. Google. "gVisor: Container Runtime Sandbox." gVisor Documentation, 2024.
5. Amazon. "Firecracker: Secure and Fast microVMs for Serverless Computing." Firecracker Documentation, 2024.
6. Chromium. "Sandbox Design." Chromium Security Documentation, 2024.
7. Kerrisk, M. "Namespaces in Operation." LWN.net, 2013-2014.
8. Rosen, R. "Linux Kernel Networking: Implementation and Theory." Apress, 2014.
9. Confidential Computing Consortium. "A Technical Analysis of Confidential Computing." 2023.
10. Bytecode Alliance. "WebAssembly System Interface (WASI)." 2024.
