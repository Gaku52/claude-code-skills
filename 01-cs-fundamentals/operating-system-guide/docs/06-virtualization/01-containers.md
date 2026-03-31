# Container Technology

> Containers are a technology that "packages an application and its dependencies, enabling it to run identically anywhere."
> By leveraging OS kernel isolation mechanisms, containers start far more lightweight and faster than virtual machines.
> This chapter systematically covers container principles, practical operations, security, and orchestration.

## What You Will Learn in This Chapter

- [ ] Understand the technical mechanisms of containers (Namespace, cgroups, Union FS)
- [ ] Grasp the OCI standard specifications and the layered structure of container runtimes
- [ ] Master the basics of image building and operations with Docker / Podman
- [ ] Acquire techniques for multi-stage builds and security hardening
- [ ] Understand the design principles of container networking and storage
- [ ] Learn the concepts of orchestration centered on Kubernetes
- [ ] Recognize anti-patterns and countermeasures in container operations
- [ ] Understand container usage in CI/CD pipelines


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [VM Basics](./00-vm-basics.md)

---

## 1. History and Background of Container Technology

### 1.1 Evolution from Virtualization to Containers

Container technology did not appear suddenly; it stands on decades of OS-level virtualization history.

```
Container Technology Timeline:

1979  chroot          Introduced in Unix V7. Changes the root directory
2000  FreeBSD Jails   Full isolation environment extending chroot
2001  Linux VServer   Server virtualization patch on Linux
2004  Solaris Zones   Container technology for Solaris
2006  Process Containers → Merged into the Linux kernel as cgroups
2008  LXC (Linux Containers)  Integrated Namespace + cgroups
2013  Docker 0.1      Appeared based on LXC. User-friendly CLI
2014  Kubernetes      Google open-sourced based on internal Borg experience
2015  OCI established Open Container Initiative. Standardization
2015  runc 1.0        OCI-compliant low-level runtime
2017  containerd 1.0  Independent as a CNCF project
2018  Podman 1.0      Daemonless, rootless containers
2020  K8s deprecates Docker shim (recommends containerd/CRI-O)
2022  WebAssembly containers (Spin, wasmCloud) emerge
2024  Kata Containers 3.0  Strong isolation via micro VMs
```

### 1.2 Comparison of Virtual Machines and Containers

```
┌─────────────────────────────────────────────────────────┐
│              Virtual Machines (VM)                        │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │  App A   │ │  App B   │ │  App C   │               │
│  ├──────────┤ ├──────────┤ ├──────────┤               │
│  │Guest OS  │ │Guest OS  │ │Guest OS  │  ← Each VM    │
│  │(Ubuntu)  │ │(CentOS)  │ │(Alpine)  │    has its OS │
│  └──────────┘ └──────────┘ └──────────┘    (multi-GB)  │
│  ┌─────────────────────────────────────┐               │
│  │       Hypervisor (KVM / Xen)        │  ← HW virt.  │
│  ├─────────────────────────────────────┤               │
│  │       Host OS (Linux)               │               │
│  ├─────────────────────────────────────┤               │
│  │       Hardware                      │               │
│  └─────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              Containers                                  │
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐               │
│  │  App A   │ │  App B   │ │  App C   │               │
│  ├──────────┤ ├──────────┤ ├──────────┤               │
│  │  Bins/   │ │  Bins/   │ │  Bins/   │  ← Only       │
│  │  Libs    │ │  Libs    │ │  Libs    │    required    │
│  └──────────┘ └──────────┘ └──────────┘    libs (MB)   │
│  ┌─────────────────────────────────────┐               │
│  │   Container Runtime (containerd)    │  ← Kernel     │
│  ├─────────────────────────────────────┤    shared      │
│  │       Host OS (Linux Kernel)        │               │
│  ├─────────────────────────────────────┤               │
│  │       Hardware                      │               │
│  └─────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────┘
```

**Comparison Table: Virtual Machines vs Containers**

| Attribute | Virtual Machine (VM) | Container |
|------|----------------|---------|
| Isolation level | Hardware level (strong) | Process level (lightweight) |
| Startup time | Tens of seconds to minutes | Milliseconds to seconds |
| Image size | Several GB to tens of GB | Several MB to hundreds of MB |
| Resource efficiency | Low (each VM has a guest OS) | High (shared kernel) |
| Density | Tens of VMs per host | Hundreds to thousands of containers per host |
| Kernel | Own kernel | Shared host kernel |
| Security | Strong isolation | Risk from shared kernel |
| Portability | Limited | High (OCI standard) |
| Live migration | Mature technology | Under development (CRIU) |
| Use case | Heterogeneous OS, strong isolation needed | Microservices, CI/CD |

---

## 2. How Containers Work — Linux Kernel Features

Containers are essentially "a combination of isolation mechanisms provided by the Linux kernel." They are not magical new technology, but a clever combination of existing kernel features.

### 2.1 Namespace

Namespaces are a mechanism for isolating kernel resources. Each Namespace separates the "view" of a specific resource per process.

```
Types and Roles of Linux Namespaces:

┌───────────────┬──────────────────────────────────────────────┐
│ Namespace     │ Isolation Target                              │
├───────────────┼──────────────────────────────────────────────┤
│ PID           │ Process ID space                              │
│               │ PID 1 in the container = container's init     │
│               │ Appears as a different PID from the host      │
├───────────────┼──────────────────────────────────────────────┤
│ NET (Network) │ Network stack (interfaces,                    │
│               │ routing tables, iptables, sockets)            │
│               │ Each container has its own eth0               │
├───────────────┼──────────────────────────────────────────────┤
│ MNT (Mount)   │ Filesystem mount points                       │
│               │ Container-specific filesystem tree             │
├───────────────┼──────────────────────────────────────────────┤
│ UTS           │ Hostname and domain name                      │
│               │ Each container has its own hostname            │
├───────────────┼──────────────────────────────────────────────┤
│ IPC           │ Inter-process communication (semaphores,      │
│               │ message queues, shared memory)                │
├───────────────┼──────────────────────────────────────────────┤
│ User          │ UID/GID mapping                               │
│               │ Root in container = unprivileged user on host  │
├───────────────┼──────────────────────────────────────────────┤
│ Cgroup        │ cgroup root directory view                    │
│               │ Container sees only its own cgroup tree       │
├───────────────┼──────────────────────────────────────────────┤
│ Time          │ System clock (Linux 5.6+)                     │
│               │ Container-specific time settings               │
└───────────────┴──────────────────────────────────────────────┘
```

**Code Example 1: Inspecting Namespaces and Manual Container Creation**

```bash
#!/bin/bash
# === Manual Container Creation Using Namespaces ===

# Check namespaces for the current process
ls -la /proc/$$/ns/
# lrwxrwxrwx 1 root root 0 ... cgroup -> cgroup:[4026531835]
# lrwxrwxrwx 1 root root 0 ... ipc -> ipc:[4026531839]
# lrwxrwxrwx 1 root root 0 ... mnt -> mnt:[4026531840]
# lrwxrwxrwx 1 root root 0 ... net -> net:[4026531992]
# lrwxrwxrwx 1 root root 0 ... pid -> pid:[4026531836]
# lrwxrwxrwx 1 root root 0 ... user -> user:[4026531837]
# lrwxrwxrwx 1 root root 0 ... uts -> uts:[4026531838]

# Create new Namespaces with unshare and launch bash
# Separate PID, UTS, and Mount Namespaces
sudo unshare --pid --uts --mount --fork /bin/bash

# Operations within the new Namespace
hostname container-demo        # UTS Namespace: own hostname
mount -t proc proc /proc       # MNT Namespace: remount proc
ps aux                          # PID Namespace: starts from PID 1

# Verify from the host side in another terminal
# Appears as a normal PID from the host
ps aux | grep "unshare"

# Enter an existing Namespace with nsenter
# (PID is the host-side PID of the container process)
sudo nsenter --target <PID> --pid --uts --mount
```

### 2.2 cgroups (Control Groups)

cgroups provide resource limits, priority control, and monitoring for process groups.

```
cgroups v2 Hierarchy:

/sys/fs/cgroup/
├── cgroup.controllers        # List of available controllers
├── cgroup.subtree_control    # Controllers enabled for subtree
├── system.slice/
│   ├── docker-<container-id>.scope/
│   │   ├── cpu.max           # CPU limit (quota period)
│   │   ├── cpu.weight        # CPU weight (1-10000)
│   │   ├── memory.max        # Memory limit (bytes)
│   │   ├── memory.current    # Current memory usage
│   │   ├── memory.swap.max   # Swap limit
│   │   ├── io.max            # Block I/O limit
│   │   ├── pids.max          # Maximum number of processes
│   │   └── cgroup.procs      # List of PIDs of member processes
│   └── docker-<another-id>.scope/
│       └── ...
└── user.slice/
    └── ...

Resource Control Mechanisms:

  CPU Limit:
  cpu.max = "200000 100000"
  → 200ms of CPU time within a 100ms period
  → Effectively 2 CPU cores available

  Memory Limit:
  memory.max = 536870912    (512 MB)
  memory.swap.max = 0       (Swap disabled)
  → On exceeding: OOM Killer kills processes in the container

  PID Limit:
  pids.max = 512
  → Protection against fork bombs
```

**Code Example 2: Inspecting Resource Limits with cgroups**

```bash
#!/bin/bash
# === Observing Resource Limits with cgroups v2 ===

# Start a Docker container with limits
docker run -d \
  --name cgroup-demo \
  --cpus="1.5" \
  --memory="256m" \
  --memory-swap="256m" \
  --pids-limit=100 \
  nginx:alpine

# Check the cgroup path for the container
CONTAINER_ID=$(docker inspect --format '{{.Id}}' cgroup-demo)
CGROUP_PATH="/sys/fs/cgroup/system.slice/docker-${CONTAINER_ID}.scope"

# Check CPU limit
cat ${CGROUP_PATH}/cpu.max
# Example output: 150000 100000
# → 150ms in a 100ms period = 1.5 cores

# Check memory limit
cat ${CGROUP_PATH}/memory.max
# Example output: 268435456 (256 MB)

# Current memory usage
cat ${CGROUP_PATH}/memory.current

# Check PID limit
cat ${CGROUP_PATH}/pids.max
# Example output: 100

# Monitor resource usage in real time
docker stats cgroup-demo --no-stream
# CONTAINER ID  NAME         CPU %  MEM USAGE / LIMIT  MEM %  NET I/O  ...
# abc123def456  cgroup-demo  0.02%  3.5MiB / 256MiB    1.37%  ...

# Verify memory limit with a stress test
docker run --rm --memory="64m" --memory-swap="64m" \
  alpine:latest sh -c "
    # Try allocating a large amount of memory
    dd if=/dev/zero of=/dev/null bs=1M count=128
  "
# → Gets OOM Killed

# Cleanup
docker rm -f cgroup-demo
```

### 2.3 Union FS (OverlayFS)

Union FS is a filesystem that enables the efficient layered structure of container images.

```
OverlayFS Operating Principles:

  File Read:
  ┌─────────────────────────┐
  │ Upper Layer (Container)  │  1. First check upperdir
  │ (Read-Write)             │     Return if file exists
  └────────────┬────────────┘
               │ If not found, go down
  ┌────────────▼────────────┐
  │ Lower Layer 3 (App)      │  2. Search from top of lower layers
  │ (Read-Only)              │
  └────────────┬────────────┘
               │ If not found, go down
  ┌────────────▼────────────┐
  │ Lower Layer 2 (Runtime)  │  3. Return when found
  │ (Read-Only)              │
  └────────────┬────────────┘
               │ If not found, go down
  ┌────────────▼────────────┐
  │ Lower Layer 1 (Base OS)  │  4. Search to the bottom layer
  │ (Read-Only)              │
  └─────────────────────────┘

  File Write:
  Copy-on-Write (CoW) Strategy
  ┌─────────────────────────┐
  │ Upper Layer              │  Writes always go to upper
  │  /etc/nginx/nginx.conf ←──── On modification: copy from
  │  (modified copy)         │         lower and modify in upper
  └─────────────────────────┘
  ┌─────────────────────────┐
  │ Lower Layer              │  Original file remains unchanged
  │  /etc/nginx/nginx.conf   │  (Can be shared with other containers)
  │  (original, untouched)   │
  └─────────────────────────┘

  File Deletion:
  A whiteout file indicates "deleted"
  Creates .wh.<filename> in upper
  → The file in lower is not actually deleted but becomes invisible
```

### 2.4 seccomp and Capabilities

```
Security Mechanism Layers:

  ┌─────────────────────────────────────────────────┐
  │           Application                            │
  ├─────────────────────────────────────────────────┤
  │ AppArmor / SELinux   MAC (Mandatory Access Ctrl) │
  ├─────────────────────────────────────────────────┤
  │ seccomp-bpf          System call filter          │
  │                      Allow only required calls   │
  │                      from ~300+ syscalls          │
  ├─────────────────────────────────────────────────┤
  │ Capabilities         Fine-grained root privileges│
  │                      CAP_NET_BIND_SERVICE:       │
  │                      Bind to privileged ports    │
  │                      CAP_SYS_ADMIN: mount, etc.  │
  ├─────────────────────────────────────────────────┤
  │ Namespace            Isolate resource visibility  │
  ├─────────────────────────────────────────────────┤
  │ cgroups              Limit resource usage         │
  ├─────────────────────────────────────────────────┤
  │ Linux Kernel                                     │
  └─────────────────────────────────────────────────┘

Capabilities Allowed by Docker Default (partial):
  CAP_CHOWN            Change file ownership
  CAP_DAC_OVERRIDE     Override file access permissions
  CAP_FSETID           Maintain set-user-ID bit
  CAP_FOWNER           File owner-related permissions
  CAP_NET_RAW          Use RAW sockets
  CAP_NET_BIND_SERVICE Bind to privileged ports (< 1024)
  CAP_SYS_CHROOT       Use chroot
  CAP_SETUID           Change process UID
  CAP_SETGID           Change process GID

Capabilities Denied by Docker Default (partial):
  CAP_SYS_ADMIN        Numerous admin operations (mount, etc.)
  CAP_SYS_PTRACE       Trace processes
  CAP_SYS_MODULE       Load kernel modules
  CAP_NET_ADMIN        Change network settings
  CAP_SYS_RAWIO        Direct access to I/O ports
  CAP_SYS_BOOT         Reboot the system
```

---

## 3. OCI Standard Specifications and Container Runtimes

### 3.1 OCI (Open Container Initiative)

OCI was established in 2015 under the Linux Foundation, led by Docker and CoreOS. It defines three standard specifications to ensure container interoperability.

```
OCI Standard Specification Structure:

1. Runtime Specification (runtime-spec)
   Defines how to execute a container
   ├── config.json    Container configuration
   │   ├── ociVersion     OCI version
   │   ├── process         Process information to execute
   │   │   ├── args        Command-line arguments
   │   │   ├── env         Environment variables
   │   │   ├── cwd         Working directory
   │   │   └── user        Execution user
   │   ├── root            Root filesystem
   │   ├── mounts          Mount points
   │   ├── linux           Linux-specific settings
   │   │   ├── namespaces  Namespaces to use
   │   │   ├── resources   cgroups resource limits
   │   │   └── seccomp     seccomp profile
   │   └── hooks           Lifecycle hooks
   └── rootfs/        Root filesystem

2. Image Specification (image-spec)
   Defines the format of container images
   ├── Image Index      Multi-architecture list
   ├── Image Manifest   Reference information for layers and config
   ├── Image Config     Runtime settings (CMD, ENV, EXPOSE, etc.)
   └── Filesystem Layers  Layers in tar+gzip format

3. Distribution Specification (distribution-spec)
   Defines how container images are distributed
   ├── Push     Sending images to a registry
   ├── Pull     Fetching images from a registry
   ├── Content Discovery  Retrieving metadata such as tag lists
   └── Content Management  Deleting images, etc.
```

### 3.2 Container Runtime Layers

```
Container Runtime Architecture:

  User Operations
      │
      ▼
  ┌──────────────────────────────────┐
  │ CLI / API                         │
  │ docker, nerdctl, podman, crictl   │
  └──────────────┬───────────────────┘
                 │
      ▼          ▼
  ┌──────────────────────────────────┐
  │ High-Level Runtime (CRI impl.)   │  Daemon process
  │                                   │  Image management
  │  containerd         CRI-O         │  Snapshots
  │  (Docker/K8s both)  (K8s only)    │  Network management
  └──────────────┬───────────────────┘
                 │ OCI Runtime Spec
                 ▼
  ┌──────────────────────────────────┐
  │ Low-Level Runtime (OCI Runtime)   │  Namespace creation
  │                                   │  cgroups setup
  │  runc           crun              │  Process launch
  │  (Go, standard) (C, fast)         │
  │                                   │
  │  gVisor (runsc)  Kata (kata-rt)   │  Sandboxed
  │  (User-space      (Micro-VM       │  Strong isolation
  │   kernel)          based)          │
  └──────────────────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────┐
  │ Linux Kernel                      │
  │ Namespaces + cgroups + seccomp    │
  └──────────────────────────────────┘
```

**Comparison Table: Low-Level Container Runtimes**

| Runtime | Language | Isolation Method | Startup Speed | Security | Memory Overhead | Use Case |
|-----------|------|---------|---------|-------------|-------------------|------------|
| runc | Go | Namespace+cgroups | Fast | Standard | Minimal | General purpose (default) |
| crun | C | Namespace+cgroups | Fastest | Standard | Minimal | Performance-focused |
| gVisor (runsc) | Go | User-space kernel | Somewhat slow | High | Moderate (tens of MB) | Multi-tenant |
| Kata Containers | Go/Rust | Micro VM | Slow | Highest | Large (tens of MB) | High security |
| youki | Rust | Namespace+cgroups | Fast | Standard | Minimal | Rust ecosystem |
| WasmEdge | C++ | Wasm sandbox | Near fastest | High | Minimal | Edge/serverless |

---

## 4. Container Practice with Docker

### 4.1 Docker Architecture

```
Docker Architecture Overview:

  ┌─────────────────────────────────────────────────────────┐
  │ Client (docker CLI)                                      │
  │                                                         │
  │  docker build    docker run    docker pull    docker ps  │
  └────────────────────────┬────────────────────────────────┘
                           │ REST API (Unix Socket / TCP)
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ Docker Daemon (dockerd)                                  │
  │                                                         │
  │  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐   │
  │  │ Image Mgmt  │  │ Network     │  │ Volume Mgmt   │   │
  │  │ Build/Pull  │  │ bridge/host │  │ Bind/Named    │   │
  │  │ Push/Tag    │  │ overlay/mac │  │ tmpfs         │   │
  │  └─────────────┘  └─────────────┘  └───────────────┘   │
  └────────────────────────┬────────────────────────────────┘
                           │ gRPC
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ containerd                                               │
  │                                                         │
  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
  │  │Snapshots │ │Content   │ │Tasks     │ │Events    │  │
  │  │(OverlayFS)│ │Store    │ │(Process) │ │Stream    │  │
  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
  └────────────────────────┬────────────────────────────────┘
                           │ OCI Runtime Spec
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │ runc                                                     │
  │ → Create Namespace → Configure cgroups → Apply seccomp   │
  │ → pivot_root → exec process                              │
  └─────────────────────────────────────────────────────────┘
```

### 4.2 Systematic Understanding of Dockerfile

**Code Example 3: Production-Quality Dockerfile (Multi-Stage Build)**

```dockerfile
# === Stage 1: Install Dependencies ===
FROM node:20-slim AS deps
WORKDIR /app

# Copy only package.json and lock file first
# → Cache is effective if dependencies haven't changed
COPY package.json package-lock.json ./
RUN npm ci --production && npm cache clean --force

# === Stage 2: Build ===
FROM node:20-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
# Build artifacts: /app/dist/

# === Stage 3: Production Image ===
FROM gcr.io/distroless/nodejs20-debian12 AS production

# Metadata labels (OCI Image Spec compliant)
LABEL org.opencontainers.image.title="my-api-server"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="Production API Server"
LABEL org.opencontainers.image.source="https://github.com/example/my-api"

WORKDIR /app

# Copy only required files (build tools excluded)
COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./

# Environment variables
ENV NODE_ENV=production
ENV PORT=3000

# Port declaration (for documentation purposes)
EXPOSE 3000

# No USER directive needed for distroless images
# (runs as non-root user by default)

# Health check
# Since distroless images have no shell,
# using K8s livenessProbe/readinessProbe is recommended

# Startup command
CMD ["dist/server.js"]
```

```
Multi-Stage Build Image Size Comparison:

  ┌────────────────────────────────────────────────────┐
  │ Single Stage (node:20)                              │
  │ ████████████████████████████████████  1.1 GB       │
  │ [Node.js + npm + build tools + src + node_modules] │
  ├────────────────────────────────────────────────────┤
  │ Single Stage (node:20-slim)                         │
  │ ██████████████████████  650 MB                     │
  │ [Node.js + src + node_modules]                     │
  ├────────────────────────────────────────────────────┤
  │ Multi-Stage (node:20-slim → distroless)             │
  │ ████████  180 MB                                   │
  │ [Node.js runtime + dist + prod node_modules]       │
  ├────────────────────────────────────────────────────┤
  │ Multi-Stage (node:20-slim → alpine)                 │
  │ ███████  150 MB                                    │
  │ [Node.js (musl) + dist + prod node_modules]        │
  └────────────────────────────────────────────────────┘

  Reduction rate: Up to 85% size reduction
  Security: Attack surface also significantly reduced
```

### 4.3 Managing Multiple Containers with Docker Compose

**Code Example 4: Production-Level docker-compose.yml**

```yaml
# docker-compose.yml
# Configuration example: API + DB + Cache + Reverse Proxy

version: "3.9"

services:
  # --- Reverse Proxy ---
  nginx:
    image: nginx:1.25-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      api:
        condition: service_healthy
    networks:
      - frontend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 128M

  # --- API Server ---
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://app:${DB_PASSWORD}@db:5432/myapp
      - REDIS_URL=redis://cache:6379
      - JWT_SECRET=${JWT_SECRET}
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    networks:
      - frontend
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:3000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 10s
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "1.0"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 128M

  # --- Database ---
  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: app
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - db-data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d myapp"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 1G

  # --- Cache ---
  cache:
    image: redis:7-alpine
    command: redis-server --maxmemory 128mb --maxmemory-policy allkeys-lru
    volumes:
      - cache-data:/data
    networks:
      - backend
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M

volumes:
  db-data:
    driver: local
  cache-data:
    driver: local

networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true   # No external access (protects DB/Cache)
```

### 4.4 .dockerignore Best Practices

```
# .dockerignore
# Files to exclude from the build context

# Version control
.git
.gitignore

# Dependencies (reinstalled inside the container)
node_modules
vendor/
__pycache__
*.pyc

# Build artifacts
dist
build
*.o
*.a

# Environment settings / Sensitive information
.env
.env.*
*.pem
*.key
credentials.json

# IDE / Editor
.vscode
.idea
*.swp
*.swo
*~

# Tests / Documentation
tests/
test/
docs/
*.md
LICENSE

# Docker-related
Dockerfile*
docker-compose*
.dockerignore

# OS files
.DS_Store
Thumbs.db
```

---

## 5. Container Networking

### 5.1 Docker Network Drivers

Container networking is built upon Linux virtual network features (veth pairs, bridges, iptables, VXLAN, etc.).

```
Docker Network Types and Communication Paths:

  1. bridge (default)
  ┌──────────────────────────────────────────────────┐
  │ Host                                              │
  │                                                   │
  │  ┌──────────┐    ┌──────────┐                    │
  │  │Container │    │Container │                    │
  │  │  A       │    │  B       │                    │
  │  │ eth0     │    │ eth0     │                    │
  │  └──┬───────┘    └──┬───────┘                    │
  │     │ veth pair     │ veth pair                   │
  │  ┌──▼───────────────▼───────┐                    │
  │  │   docker0 (bridge)       │ 172.17.0.1         │
  │  │   172.17.0.0/16          │                    │
  │  └──────────┬───────────────┘                    │
  │             │ NAT (iptables)                      │
  │  ┌──────────▼───────────────┐                    │
  │  │   eth0 (host NIC)        │ 192.168.1.100      │
  │  └──────────────────────────┘                    │
  └──────────────────────────────────────────────────┘

  2. host
  Container directly uses the host's network stack
  → No port mapping needed, best network performance
  → Risk of port conflicts, no isolation

  3. overlay (Swarm / K8s)
  Build container networks across multiple hosts
  → Achieve L2 connectivity via VXLAN tunneling
  → Service-to-service communication in cluster environments

  4. macvlan
  Assign a unique MAC address to the container
  → Direct connection to the physical network
  → Useful for integration with legacy systems

  5. none
  No network (complete isolation)
  → For batch processing or security purposes
```

**Docker Network Driver Comparison Table**

| Driver | Performance | Isolation | Multi-host | Primary Use |
|---------|------|------|------------|---------|
| bridge | Medium | Yes | No | Development, single-host production |
| host | High | None | No | Performance-critical applications |
| overlay | Medium-Low | Yes | Yes | Swarm/K8s clusters |
| macvlan | High | Yes | No | Legacy integration, direct L2 access |
| ipvlan | High | Yes | No | Environments with MAC address restrictions |
| none | - | Full | - | Security isolation, batch processing |

### 5.2 Container-to-Container Communication Patterns

```
Service Discovery and Communication Patterns:

  Pattern 1: Docker Compose DNS-Based
  ┌─────────────────────────────────────────┐
  │ User-Defined Bridge Network              │
  │                                          │
  │  api ──── "redis://cache:6379" ───► cache│
  │   │                                      │
  │   └───── "postgresql://db:5432" ──► db   │
  │                                          │
  │  Docker's built-in DNS (127.0.0.11)      │
  │  Automatically resolves service name     │
  │  → container IP                          │
  └─────────────────────────────────────────┘

  Pattern 2: K8s Service-Based
  ┌─────────────────────────────────────────┐
  │ Kubernetes Cluster                       │
  │                                          │
  │  Pod A ─── "http://api-svc:3000" ──►    │
  │            ClusterIP Service             │
  │                │ kube-proxy (iptables)   │
  │            ┌───▼────┐  ┌────────┐       │
  │            │ Pod B-1│  │ Pod B-2│       │
  │            │ (api)  │  │ (api)  │       │
  │            └────────┘  └────────┘       │
  │                                          │
  │  CoreDNS: <svc>.<ns>.svc.cluster.local   │
  └─────────────────────────────────────────┘
```

---

## 6. Container Storage and Data Management

### 6.1 Storage Types

```
Docker Storage Options:

  1. Volumes (Recommended)
  ┌──────────────────────────────────────┐
  │ Container                            │
  │ /app/data ──────┐                    │
  └─────────────────┼────────────────────┘
                    │ mount
  ┌─────────────────▼────────────────────┐
  │ /var/lib/docker/volumes/mydata/_data  │
  │ Storage managed by Docker             │
  │ ├── Easy to back up                   │
  │ ├── Works on Linux / Mac / Windows    │
  │ └── Extensible with volume drivers    │
  └──────────────────────────────────────┘

  2. Bind Mounts
  ┌──────────────────────────────────────┐
  │ Container                            │
  │ /app/src ───────┐                    │
  └─────────────────┼────────────────────┘
                    │ mount
  ┌─────────────────▼────────────────────┐
  │ /home/user/project/src               │
  │ Mount any path on the host            │
  │ ├── Convenient for live reload during │
  │ │   development                       │
  │ ├── Depends on host directory layout  │
  │ └── Security risk (host exposure)     │
  └──────────────────────────────────────┘

  3. tmpfs Mounts
  ┌──────────────────────────────────────┐
  │ Container                            │
  │ /app/tmp ───────┐                    │
  └─────────────────┼────────────────────┘
                    │
  ┌─────────────────▼────────────────────┐
  │ In-memory filesystem                  │
  │ ├── Does not write to disk            │
  │ ├── Fast but lost when container stops│
  │ └── Suitable for temp files / secrets │
  └──────────────────────────────────────┘
```

### 6.2 Data Persistence Best Practices

```bash
#!/bin/bash
# === Storage Management Command Examples ===

# Create and use a Named Volume
docker volume create app-data
docker run -d \
  --name db \
  -v app-data:/var/lib/postgresql/data \
  postgres:16

# Volume detail information
docker volume inspect app-data
# Example output:
# [{
#   "CreatedAt": "2024-01-15T10:30:00Z",
#   "Driver": "local",
#   "Mountpoint": "/var/lib/docker/volumes/app-data/_data",
#   "Name": "app-data",
#   "Scope": "local"
# }]

# Backup a Volume
docker run --rm \
  -v app-data:/source:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/app-data-backup.tar.gz -C /source .

# Restore a Volume
docker run --rm \
  -v app-data:/target \
  -v $(pwd):/backup:ro \
  alpine tar xzf /backup/app-data-backup.tar.gz -C /target

# Bulk delete unused Volumes
docker volume prune -f

# Bind Mount (for development environments)
docker run -d \
  --name dev-server \
  -v $(pwd)/src:/app/src:cached \
  -v /app/node_modules \
  node:20-slim npm run dev
# :cached → Improved write performance on macOS
# /app/node_modules → Anonymous Volume to avoid overwriting host's

# tmpfs (for sensitive data)
docker run -d \
  --name secure-app \
  --tmpfs /app/secrets:rw,noexec,nosuid,size=64m \
  --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  my-app:latest

# Read-Only root filesystem + tmpfs
docker run -d \
  --name readonly-app \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid \
  --tmpfs /var/run:rw,noexec,nosuid \
  nginx:alpine
```

---

## 7. Container Security

### 7.1 Threat Model and Security Layers

```
Container Security Defense-in-Depth:

  Attack Surface
  ┌─────────────────────────────────────────────────────┐
  │ Layer 7: Application vulnerabilities                 │
  │   SQLi, XSS, RCE → WAF, input validation, scanning  │
  ├─────────────────────────────────────────────────────┤
  │ Layer 6: Dependency vulnerabilities                   │
  │   Libraries with CVEs → Trivy/Snyk scan, SCA         │
  ├─────────────────────────────────────────────────────┤
  │ Layer 5: Container image                              │
  │   Unnecessary tools/shells → distroless, minimal base │
  │   Root execution → Non-root via USER directive        │
  ├─────────────────────────────────────────────────────┤
  │ Layer 4: Container runtime                            │
  │   Privileged containers → Prohibit --privileged       │
  │   Excessive Capabilities → drop ALL + add only needed │
  │   Syscall abuse → seccomp profiles                    │
  ├─────────────────────────────────────────────────────┤
  │ Layer 3: Host OS                                      │
  │   Kernel vulnerabilities → Patching, gVisor/Kata      │
  │   Docker socket exposure → Prohibit socket mounting   │
  ├─────────────────────────────────────────────────────┤
  │ Layer 2: Network                                      │
  │   Lateral movement → NetworkPolicy, internal network  │
  │   Plaintext communication → mTLS (service mesh)       │
  ├─────────────────────────────────────────────────────┤
  │ Layer 1: Orchestration                                │
  │   RBAC misconfiguration → Principle of least privilege│
  │   Plaintext secrets → Vault, Sealed Secrets           │
  └─────────────────────────────────────────────────────┘
```

### 7.2 Security Hardening in Practice

**Code Example 5: Security-Hardened Docker Execution**

```bash
#!/bin/bash
# === Security-Hardened Container Execution Examples ===

# ---- Basic Security Hardening ----

# 1. Run as non-root user
docker run -d \
  --name secure-nginx \
  --user 1000:1000 \
  nginx:alpine

# 2. Minimize Capabilities
docker run -d \
  --name minimal-caps \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --cap-add=CHOWN \
  --cap-add=SETUID \
  --cap-add=SETGID \
  nginx:alpine

# 3. Read-Only Filesystem
docker run -d \
  --name readonly-web \
  --read-only \
  --tmpfs /var/cache/nginx:rw,noexec,nosuid \
  --tmpfs /var/run:rw,noexec,nosuid \
  --tmpfs /tmp:rw,noexec,nosuid \
  nginx:alpine

# 4. Apply seccomp Profile
docker run -d \
  --name seccomp-app \
  --security-opt seccomp=./custom-seccomp.json \
  my-app:latest

# 5. Apply AppArmor Profile
docker run -d \
  --name apparmor-app \
  --security-opt apparmor=docker-custom \
  my-app:latest

# ---- Comprehensive Security Hardening ----

docker run -d \
  --name hardened-app \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --cap-drop=ALL \
  --cap-add=NET_BIND_SERVICE \
  --security-opt=no-new-privileges:true \
  --security-opt seccomp=./seccomp-profile.json \
  --user 1000:1000 \
  --memory=256m \
  --memory-swap=256m \
  --cpus="0.5" \
  --pids-limit=64 \
  --network=app-net \
  --restart=unless-stopped \
  --health-cmd="wget --spider -q http://localhost:8080/health" \
  --health-interval=30s \
  --health-timeout=5s \
  --health-retries=3 \
  my-app:latest

# ---- Image Scanning ----

# Scan image with Trivy
trivy image --severity HIGH,CRITICAL my-app:latest

# Example output:
# my-app:latest (alpine 3.19.0)
# ================================
# Total: 2 (HIGH: 1, CRITICAL: 1)
#
# ┌───────────────┬──────────────┬──────────┬────────┐
# │   Library     │ Vulnerability│ Severity │ Status │
# ├───────────────┼──────────────┼──────────┼────────┤
# │ openssl       │ CVE-2024-XXX │ CRITICAL │ fixed  │
# │ curl          │ CVE-2024-YYY │ HIGH     │ fixed  │
# └───────────────┴──────────────┴──────────┴────────┘

# Scan the Dockerfile itself with Trivy (detect misconfigurations)
trivy config ./Dockerfile

# Vulnerability analysis with Docker Scout
docker scout cves my-app:latest
docker scout recommendations my-app:latest
```

### 7.3 Podman: Daemonless, Rootless Containers

```
Docker vs Podman Architecture Comparison:

  Docker:
  ┌─────────┐     ┌──────────────────────┐
  │ docker   │────►│ dockerd (root daemon) │
  │ CLI      │     │     │                │
  └─────────┘     │  containerd          │
                  │     │                │
                  │   runc               │
                  └──────────────────────┘
  → Daemon is a SPOF (Single Point of Failure)
  → Risk of running with root privileges
  → Daemon restart affects all containers

  Podman:
  ┌─────────┐
  │ podman   │──── fork/exec ────► runc
  │ CLI      │     (no daemon)
  └─────────┘
  → Daemonless: Each command directly invokes runc
  → Rootless: Can run as a regular user
  → systemd integration: Manage containers as systemd units
  → Compatibility: Nearly identical usage to docker CLI

Podman's Notable Features:
  # Pod (same concept as K8s Pod)
  podman pod create --name my-pod -p 8080:80
  podman run --pod my-pod nginx:alpine
  podman run --pod my-pod php:fpm

  # Generate systemd unit
  podman generate systemd --new --name my-container \
    > ~/.config/systemd/user/my-container.service
  systemctl --user enable --now my-container

  # Generate/apply K8s YAML
  podman generate kube my-pod > pod.yaml
  podman play kube pod.yaml
```

---

## 8. Container Orchestration with Kubernetes

### 8.1 Kubernetes Architecture

```
Detailed Kubernetes Cluster Architecture:

  ┌─────────────────────────────────────────────────────────────┐
  │                    Control Plane                             │
  │                                                             │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
  │  │ kube-apiserver│  │kube-scheduler│  │  kube-controller │  │
  │  │              │  │              │  │  -manager        │  │
  │  │ REST API     │  │ Pod placement│  │ ReplicaSet       │  │
  │  │ AuthN/AuthZ  │  │ Node select. │  │ Deployment       │  │
  │  │ Admission    │  │ Resource-    │  │ Node/Job/...     │  │
  │  └──────┬───────┘  │ aware       │  └──────────────────┘  │
  │         │          └──────────────┘                        │
  │  ┌──────▼───────┐  ┌──────────────┐                        │
  │  │   etcd       │  │ cloud-ctrl-  │                        │
  │  │ Distributed  │  │ manager      │                        │
  │  │ KV Store     │  │ (CSP integ.) │                        │
  │  │ Single source│  └──────────────┘                        │
  │  │ of truth for │                                          │
  │  │ cluster state│                                          │
  │  └──────────────┘                                          │
  └─────────────────────────────────────────────────────────────┘
                           │
                  ─────────┼──────────
                           │
  ┌─────────────────────────────────────────────────────────────┐
  │                    Worker Node                              │
  │                                                             │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
  │  │  kubelet     │  │ kube-proxy   │  │ Container        │  │
  │  │              │  │              │  │ Runtime          │  │
  │  │ Pod mgmt     │  │ Service      │  │ (containerd/     │  │
  │  │ Health check │  │ network      │  │  CRI-O)          │  │
  │  │ Controls     │  │ rule mgmt   │  │                  │  │
  │  │ runtime      │  │ (iptables/   │  │ OCI Runtime      │  │
  │  │ via CRI      │  │  IPVS)       │  │ (runc)           │  │
  │  └──────────────┘  └──────────────┘  └──────────────────┘  │
  │                                                             │
  │  ┌──────────────────────────────────────────────────────┐   │
  │  │ Pod                                                  │   │
  │  │ ┌────────────┐ ┌────────────┐ ┌──────────────────┐  │   │
  │  │ │ Container  │ │ Container  │ │ Pause Container  │  │   │
  │  │ │ (app)      │ │ (sidecar)  │ │ (holds net ns)   │  │   │
  │  │ └────────────┘ └────────────┘ └──────────────────┘  │   │
  │  │          Shared: Network NS, IPC NS, Volume          │   │
  │  └──────────────────────────────────────────────────────┘   │
  └─────────────────────────────────────────────────────────────┘
```

### 8.2 Key Kubernetes Resources

```
K8s Resource Hierarchy and Relationships:

  Deployment
  ├── strategy: RollingUpdate / Recreate
  ├── replicas: 3
  └── ReplicaSet (auto-managed)
      ├── Pod-1
      │   ├── Container (app)
      │   ├── Container (sidecar)
      │   └── Volume
      ├── Pod-2
      │   └── ...
      └── Pod-3
          └── ...

  Service (stable access to Pods)
  ├── ClusterIP    Internal to cluster only (default)
  ├── NodePort     Expose via port on each node (30000-32767)
  ├── LoadBalancer  Auto-provision external LB
  └── ExternalName  Alias for external DNS name

  Ingress (HTTP/HTTPS routing)
  ├── host: api.example.com → api-service:3000
  ├── host: web.example.com → web-service:80
  └── TLS termination

  ConfigMap / Secret (externalize configuration and secrets)
  ├── Inject as environment variables
  ├── Mount as files
  └── Use as command-line arguments

  PersistentVolume (PV) / PersistentVolumeClaim (PVC)
  ├── Dynamic provisioning with StorageClass
  ├── AccessMode: ReadWriteOnce / ReadOnlyMany / ReadWriteMany
  └── Reclaim Policy: Retain / Delete

  HorizontalPodAutoscaler (HPA)
  ├── Scale based on CPU/memory utilization
  ├── Custom metrics (Prometheus integration)
  └── min/max replica count constraints
```

### 8.3 Kubernetes Manifest Example

**Code Example 6: Production-Level Kubernetes Deployment**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: v1.2.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # Max additional Pods during rolling update
      maxUnavailable: 0     # No unavailability during update
  template:
    metadata:
      labels:
        app: api-server
        version: v1.2.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
    spec:
      serviceAccountName: api-server
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: api
          image: registry.example.com/api-server:v1.2.0
          ports:
            - containerPort: 3000
              protocol: TCP
          env:
            - name: NODE_ENV
              value: "production"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: database-url
          resources:
            requests:
              cpu: "250m"       # 0.25 cores
              memory: "128Mi"
            limits:
              cpu: "1000m"      # 1 core
              memory: "512Mi"
          livenessProbe:
            httpGet:
              path: /health/live
              port: 3000
            initialDelaySeconds: 15
            periodSeconds: 20
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 3000
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          startupProbe:
            httpGet:
              path: /health/live
              port: 3000
            failureThreshold: 30
            periodSeconds: 2
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop: ["ALL"]
          volumeMounts:
            - name: tmp
              mountPath: /tmp
            - name: config
              mountPath: /app/config
              readOnly: true
      volumes:
        - name: tmp
          emptyDir:
            medium: Memory
            sizeLimit: 64Mi
        - name: config
          configMap:
            name: api-config
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: api-server
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: api-server
  namespace: production
spec:
  type: ClusterIP
  selector:
    app: api-server
  ports:
    - port: 80
      targetPort: 3000
      protocol: TCP
---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-server
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-server
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Percent
          value: 50
          periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### 8.4 Lightweight Kubernetes Options

```
Kubernetes Distribution Comparison:

  ┌─────────────────────────────────────────────────────────────┐
  │               Full K8s (kubeadm / kops)                     │
  │  ████████████████████████████████████████  Resources: Large  │
  │  Control Plane: 3+ nodes (HA)                               │
  │  Memory: 2GB+ per node                                      │
  │  Use: Large-scale production                                │
  ├─────────────────────────────────────────────────────────────┤
  │               K3s (Rancher)                                 │
  │  ████████████████████  Resources: Medium                    │
  │  Single binary (~70MB)                                      │
  │  Memory: Runs with 512MB                                    │
  │  SQLite / etcd selectable                                   │
  │  Use: Edge, IoT, small-medium scale, CI/CD                 │
  ├─────────────────────────────────────────────────────────────┤
  │               k0s (Mirantis)                                │
  │  ██████████████████  Resources: Medium                      │
  │  Zero-dependency single binary                              │
  │  Memory: Runs with 300MB                                    │
  │  Use: Edge, air-gapped environments                         │
  ├─────────────────────────────────────────────────────────────┤
  │               minikube / kind / k3d                         │
  │  ████████████  Resources: Small                             │
  │  Local development only                                     │
  │  minikube: VM-based, kind: Docker-in-Docker                 │
  │  k3d: K3s in Docker (fastest)                               │
  │  Use: Development, testing, learning                        │
  └─────────────────────────────────────────────────────────────┘
```

---

## 9. Container Usage in CI/CD Pipelines

### 9.1 Container-Based CI/CD Architecture

```
CI/CD Pipeline Using Containers:

  Developer
      │
      │ git push
      ▼
  ┌────────────────────────────────────────────────────────────┐
  │ CI Pipeline (GitHub Actions / GitLab CI / Jenkins)         │
  │                                                            │
  │  Stage 1: Build                                            │
  │  ┌──────────────────────────────────────────┐              │
  │  │ docker build --target builder -t app:ci  │              │
  │  │ → Build source and run tests             │              │
  │  └──────────────────────────────────────────┘              │
  │           │                                                │
  │           ▼                                                │
  │  Stage 2: Test                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │ docker compose -f docker-compose.test.yml│              │
  │  │ → Integration tests (with DB, Redis, etc.)│             │
  │  └──────────────────────────────────────────┘              │
  │           │                                                │
  │           ▼                                                │
  │  Stage 3: Scan                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │ trivy image app:ci                       │              │
  │  │ → Vulnerability scan (fail on CRITICAL)  │              │
  │  └──────────────────────────────────────────┘              │
  │           │                                                │
  │           ▼                                                │
  │  Stage 4: Push                                             │
  │  ┌──────────────────────────────────────────┐              │
  │  │ docker push registry/app:v1.2.0          │              │
  │  │ docker push registry/app:latest          │              │
  │  └──────────────────────────────────────────┘              │
  └────────────────────────────────────────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────────────────────┐
  │ CD Pipeline                                                │
  │                                                            │
  │  ┌────────────────┐    ┌─────────────────────────────┐    │
  │  │ GitOps (ArgoCD) │───►│ Kubernetes Cluster           │    │
  │  │ Manifest sync   │    │ Rolling Update               │    │
  │  └────────────────┘    │ → v1.1.0 → v1.2.0           │    │
  │                        └─────────────────────────────┘    │
  └────────────────────────────────────────────────────────────┘
```

### 9.2 Container CI/CD with GitHub Actions

**Code Example 7: GitHub Actions Workflow**

```yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
    tags: ["v*"]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build test image
        uses: docker/build-push-action@v5
        with:
          context: .
          target: builder
          load: true
          tags: app:test
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run unit tests
        run: docker run --rm app:test npm test

      - name: Run integration tests
        run: |
          docker compose -f docker-compose.test.yml up -d
          docker compose -f docker-compose.test.yml run --rm test
          docker compose -f docker-compose.test.yml down -v

  security-scan:
    needs: build-and-test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build production image
        uses: docker/build-push-action@v5
        with:
          context: .
          load: true
          tags: app:scan

      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: app:scan
          format: sarif
          output: trivy-results.sarif
          severity: CRITICAL,HIGH

      - name: Upload scan results
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif

  publish:
    needs: [build-and-test, security-scan]
    if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4

      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            type=sha

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          platforms: linux/amd64,linux/arm64
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

## 10. Anti-Patterns and Countermeasures

### 10.1 Anti-Pattern 1: "Fat Container"

```
Problem:
  Packing multiple processes into a single container
  "Using a container like a virtual machine"

  Bad Example:
  ┌──────────────────────────────────────┐
  │ Fat Container                        │
  │                                      │
  │  ┌────────┐ ┌────────┐ ┌────────┐   │
  │  │ nginx  │ │ Node.js│ │ cron   │   │
  │  └────────┘ └────────┘ └────────┘   │
  │  ┌────────┐ ┌────────┐              │
  │  │ Redis  │ │ sshd   │              │
  │  └────────┘ └────────┘              │
  │                                      │
  │  All processes managed by supervisord│
  │  → Image size 2GB+                  │
  │  → Complex log management           │
  │  → Cannot scale individually        │
  │  → No fault isolation               │
  └──────────────────────────────────────┘

  Good Example:
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ nginx    │ │ Node.js  │ │ Redis    │
  │ container│ │ container│ │ container│
  │ 25MB     │ │ 180MB    │ │ 30MB     │
  └──────────┘ └──────────┘ └──────────┘
  Each service is independent
  → Individual scaling possible
  → Clear fault isolation
  → High image reusability
  → Logs go to stdout/stderr

Countermeasures:
  - Principle of 1 container = 1 process
  - Separate auxiliary processes using the sidecar pattern
  - Coordinate multiple containers with Docker Compose / K8s
  - Exceptions: initialization scripts, signal handlers are acceptable
```

### 10.2 Anti-Pattern 2: "Latest Tag Dependency"

```
Problem:
  Using the :latest tag in production

  Bad Example:
  # Dockerfile
  FROM node:latest          # Version unknown
  ...

  # K8s Deployment
  image: my-app:latest      # Cannot rollback

  Why it is dangerous:
  ┌──────────────────────────────────────────────────────────┐
  │ Day 1: docker pull node:latest → Node.js 20.10.0        │
  │ Day 2: docker pull node:latest → Node.js 20.11.0 (auto) │
  │ Day 3: docker pull node:latest → Node.js 21.0.0 (break!)│
  │                                                          │
  │ → No build reproducibility                               │
  │ → Different versions in production and development       │
  │ → Difficult to rollback on failures                      │
  │ → K8s imagePullPolicy: Always pulls every time           │
  └──────────────────────────────────────────────────────────┘

  Good Example:
  # Dockerfile
  FROM node:20.11.0-slim     # Exact version pinned
  ...

  # Even better: specify digest
  FROM node:20.11.0-slim@sha256:abc123...

  # K8s Deployment
  image: registry.example.com/my-app:v1.2.0  # Semantic version

Countermeasures:
  - Always pin the base image version
  - Always use tags (v1.2.0) or digests for production deploys
  - Automatically assign version tags in CI/CD
  - Manage base image updates with Dependabot / Renovate
  - Image signing and verification (cosign / Notary)
```

### 10.3 Anti-Pattern 3: "Docker Socket Mounting"

```
Problem:
  Mounting the Docker socket into a container

  Bad Example:
  docker run -v /var/run/docker.sock:/var/run/docker.sock my-tool

  Why it is dangerous:
  → Full control over Docker daemon from inside the container
  → Access to any file on the host (by creating privileged containers)
  → Effectively equivalent to host root privileges

  docker run -v /var/run/docker.sock:/var/run/docker.sock \
    alpine sh -c "
      # Create a container that mounts host's /
      docker run -v /:/host alpine cat /host/etc/shadow
    "
  → A classic container escape technique

Countermeasures:
  - Docker socket mounting should be prohibited in principle
  - Use Docker-in-Docker (DinD) or Kaniko in CI/CD
  - Restrict with PodSecurityPolicy / PodSecurityStandard in K8s
  - If necessary, use Docker Socket Proxy (Tecnativa) to limit API access
```

---

## 11. Exercises

### Exercise 1: Beginner Level — Dockerfile Optimization

```
Task:
  Optimize the following inefficient Dockerfile.

  === Before Optimization ===
  FROM ubuntu:latest
  RUN apt-get update
  RUN apt-get install -y nodejs npm python3 gcc make
  COPY . /app
  WORKDIR /app
  RUN npm install
  RUN npm run build
  EXPOSE 3000
  CMD ["node", "dist/server.js"]

  Optimization criteria:
  1. Base image selection (size reduction)
  2. Leveraging layer cache (faster builds)
  3. Multi-stage build (lighter final image)
  4. Security (non-root execution, removing unnecessary tools)
  5. Creating a .dockerignore

  === Reference Solution ===
  # Stage 1: Build
  FROM node:20-slim AS builder
  WORKDIR /app
  COPY package.json package-lock.json ./
  RUN npm ci
  COPY . .
  RUN npm run build

  # Stage 2: Production
  FROM node:20-slim AS production
  RUN groupadd -r appuser && useradd -r -g appuser appuser
  WORKDIR /app
  COPY --from=builder /app/package.json /app/package-lock.json ./
  RUN npm ci --production && npm cache clean --force
  COPY --from=builder /app/dist ./dist
  USER appuser
  EXPOSE 3000
  HEALTHCHECK --interval=30s --timeout=5s \
    CMD wget --spider -q http://localhost:3000/health || exit 1
  CMD ["node", "dist/server.js"]

  Improvement points:
  - ubuntu:latest → node:20-slim (size reduction, unnecessary packages removed)
  - Intentionally not merging RUN commands (preserves cache granularity)
  - Copy package.json first (leverages dependency cache)
  - Multi-stage removes build tools like gcc/make
  - USER directive for non-root execution
  - Added HEALTHCHECK
```

### Exercise 2: Intermediate Level — Building Microservices with Docker Compose

```
Task:
  Create a docker-compose.yml that meets the following requirements.

  Requirements:
  - Frontend: React app (served via Nginx)
  - Backend: Node.js API (3 replicas)
  - Database: PostgreSQL (persistent data)
  - Cache: Redis
  - Network: Separate frontend and backend networks
  - Security: DB/Redis not accessible from external
  - Health checks: Configured for all services

  Hints:
  - internal option in networks
  - condition in depends_on
  - Resource limits with deploy.resources
  - Named volumes

  Evaluation Criteria:
  [ ] Service dependencies are correct
  [ ] Network isolation is appropriate
  [ ] Data is persisted
  [ ] Health checks are configured for all services
  [ ] Resource limits are set
  [ ] Sensitive information is externalized via environment variables
```

### Exercise 3: Advanced Level — Kubernetes Deployment Design

```
Task:
  Design manifests to deploy the following application on Kubernetes.

  Application Configuration:
  - Web API: 3 replicas, auto-scaling on CPU/memory
  - Worker: 2 replicas, queue processing
  - PostgreSQL: StatefulSet, persistent volumes
  - Redis: Sentinel configuration

  Design Requirements:
  1. Security:
     - Pod Security Standards: restricted
     - NetworkPolicy to restrict inter-service communication
     - Secrets fetched from external secret store
     - All containers non-root, read-only rootfs

  2. Availability:
     - Pod Disruption Budget (PDB)
     - Pod Topology Spread Constraints
     - Rolling Update (maxUnavailable: 0)
     - Liveness / Readiness / Startup Probe

  3. Observability:
     - Prometheus metrics endpoint
     - Structured logging (JSON)
     - Distributed tracing (OpenTelemetry)

  4. Resource Management:
     - Resource Requests / Limits
     - LimitRange / ResourceQuota
     - HPA (CPU 70%, Memory 80% for scale-up)
     - VPA (automatic recommendation adjustment)

  Evaluation Criteria:
  [ ] YAML manifests have correct syntax
  [ ] All security requirements are met
  [ ] All availability requirements are met
  [ ] Zero-downtime deployment is achievable
  [ ] Resource settings are cost-efficient
  [ ] Countermeasures for failure scenarios are considered
```

---

## 12. The Future and New Trends in Containers

### 12.1 WebAssembly (Wasm) Containers

```
Positioning of Wasm Containers:

  Relationship between Isolation Level and Overhead:

  Strong│ VM (KVM/Xen)
  Iso-  │   ● Hundreds of MB, seconds to start
  la-   │
  tion  │ Kata Containers
        │   ● Tens of MB, under 1 second
        │
        │ gVisor
        │   ● Tens of MB, 100ms
        │
        │ Traditional containers (runc)
        │   ● Several MB, 50ms
        │
        │ Wasm containers
  Weak  │   ● Several KB-MB, under 1ms
        └──────────────────────────────►
       Small     Overhead        Large

  Advantages of Wasm:
  - Startup time: Cold start under 1ms
  - Memory: Several KB to several MB
  - Security: Sandbox guaranteed at the language level
  - Portability: CPU architecture independent
  - Multi-language: Rust, Go, C/C++, Python, JS, ...

  Constraints:
  - Limited filesystem access (WASI)
  - Network capabilities are still developing
  - Ecosystem is not yet mature
  - Not suitable for all workloads
```

### 12.2 Container Observability with eBPF

```
eBPF (extended Berkeley Packet Filter):

  Executes sandboxed programs within the kernel
  → Revolutionizes container observability and security

  Traditional approach:
  App → syscall → Kernel → (afterwards) log analysis
                              ↑ High overhead

  eBPF:
  App → syscall → Kernel ← eBPF program (in-kernel)
                              ↑ Real-time, low overhead

  Representative tools:
  ┌─────────────────────────────────────────────────┐
  │ Cilium        K8s networking + security          │
  │               kube-proxy replacement, NetworkPolicy│
  │               L3/L4/L7 visibility                │
  ├─────────────────────────────────────────────────┤
  │ Tetragon      Runtime security                   │
  │               Process execution, file access     │
  │               monitoring, network connection     │
  │               tracking                           │
  ├─────────────────────────────────────────────────┤
  │ Pixie         Application observability          │
  │               HTTP/gRPC/SQL tracing without      │
  │               code changes                       │
  │               Auto-generation of service maps    │
  ├─────────────────────────────────────────────────┤
  │ Falco         Runtime threat detection           │
  │               Suspicious syscall detection       │
  │               Container escape detection         │
  └─────────────────────────────────────────────────┘
```

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

The knowledge from this topic is frequently used in day-to-day development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary

| Concept | Key Points |
|------|---------|
| Namespace | 8 types — PID, NET, MNT, UTS, IPC, User, Cgroup, Time — isolate resource visibility |
| cgroups | Resource limits and monitoring for CPU, memory, I/O, PID count. Unified management in v2 |
| Union FS | CoW layer structure via OverlayFS. Disk savings through sharing read-only layers |
| seccomp | System call filtering. Allows only necessary syscalls from ~300 available |
| OCI | Three specs — Runtime Spec, Image Spec, Distribution Spec — standardize containers |
| Docker | De facto standard for image building + execution. Built on containerd + runc |
| Podman | Daemonless, rootless alternative. Docker CLI compatible |
| Kubernetes | Container orchestration. Pod, Service, Deployment are core concepts |
| Security | Defense-in-depth: non-root, minimal Capabilities, seccomp, read-only rootfs |
| CI/CD | Container-based pipelines for reproducible build, test, and deploy |
| Wasm | Next-generation container alternative candidate. Ultra-lightweight, fast startup but ecosystem still developing |

---

## 14. FAQ (Frequently Asked Questions)

### Q1: Should I use Docker or Podman?

**A**: It depends on the use case. For development environments, Docker Desktop offers high convenience and a rich ecosystem. However, for production environments where security is a priority, Podman's "daemonless, rootless" architecture is advantageous. Additionally, Docker Desktop may incur licensing fees for commercial use (companies with 250+ employees or annual revenue of $10M+). Podman is completely free and open source. Many organizations adopt Docker for development and containerd (via K8s) for production.

### Q2: Will containers completely replace virtual machines?

**A**: They will not. Both are optimized for different use cases and will continue to coexist. Containers are suited for microservices, CI/CD, and stateless workloads. On the other hand, VMs remain necessary for heterogeneous OS environments (Linux and Windows coexistence), multi-tenant environments requiring strong isolation, legacy application migration, and kernel module testing. Hybrid approaches like Kata Containers, which combine VM isolation strength with container operability, also exist.

### Q3: Is Kubernetes necessary for small-scale projects?

**A**: In most cases, no. Kubernetes has high learning and operational costs, and tends to become over-engineering for small projects. Alternatives include:

- **Docker Compose**: The optimal solution when a single server suffices. Simple configuration with low learning cost
- **Managed services**: AWS ECS/Fargate, Google Cloud Run, Azure Container Apps. Run containers without the complexity of K8s
- **K3s**: A lightweight alternative when K8s features are absolutely needed. Runs with 512MB of memory

K8s is appropriate when multiple teams operate dozens or more services and require advanced features like auto-scaling, zero-downtime deployment, and service mesh.

### Q4: Is an Alpine-based image always optimal?

**A**: Not necessarily. Alpine uses musl libc, which can cause compatibility issues with applications that assume glibc. Problems have been reported particularly with Python native extensions and Node.js native addons. You need to weigh the benefit of minimal size against the difficulty of debugging (shell limitations, etc.). As alternatives, Debian slim variants (`node:20-slim`, `python:3.12-slim`) provide a good balance. Distroless images are even smaller and contain no shell, minimizing the attack surface, but require measures like ephemeral containers for debugging.

### Q5: Should I run a database inside a container?

**A**: For development and test environments, it is actively recommended. For production environments, careful consideration is needed. Challenges of containerized databases include data persistence design (Volume management), performance (OverlayFS overhead), backup/restore operations, and HA configuration complexity. Many organizations choose managed DB services (RDS, Cloud SQL, etc.). However, with the maturation of Kubernetes StatefulSet and Operator patterns (CloudNativePG, Crunchy Postgres Operator, etc.), running containerized databases in production is becoming increasingly viable.

---

## 15. Glossary

| Term | Description |
|------|------|
| OCI | Open Container Initiative. Organization that develops container standard specifications |
| CRI | Container Runtime Interface. Interface between K8s and container runtimes |
| CNI | Container Network Interface. Plugin interface for container networking |
| CSI | Container Storage Interface. Plugin interface for container storage |
| CoW | Copy-on-Write. Strategy that copies data only when writing |
| DinD | Docker-in-Docker. Technique for running Docker inside a Docker container |
| distroless | Base images provided by Google containing only the minimum files needed for application execution |
| etcd | Distributed Key-Value store that holds K8s cluster state |
| HPA | Horizontal Pod Autoscaler. Horizontal scaling of Pods based on metrics |
| Init Container | Initialization container that runs before the main containers in a Pod |
| Sidecar | Container in the same Pod that assists the main container (log collection, proxy, etc.) |
| StatefulSet | K8s resource for stateful applications (databases, etc.) |
| Wasm | WebAssembly. Portable binary format that also runs outside browsers |
| WASI | WebAssembly System Interface. System interface specification for Wasm |

---

## Recommended Next Guides


---

## References

1. Luksa, M. "Kubernetes in Action." 2nd Ed, Manning, 2022.
2. Kane, S. et al. "Docker: Up & Running." 3rd Ed, O'Reilly, 2023.
3. Rice, L. "Container Security: Fundamental Technology Concepts that Protect Containerized Applications." O'Reilly, 2020.
4. Hausenblas, M. & Cindy Sridharan. "Cloud Native Infrastructure." O'Reilly, 2017.
5. Burns, B. et al. "Kubernetes: Up and Running." 3rd Ed, O'Reilly, 2022.
6. Linux man pages: namespaces(7), cgroups(7), capabilities(7), seccomp(2).
7. Open Container Initiative Specifications. https://opencontainers.org/
8. CNCF Cloud Native Landscape. https://landscape.cncf.io/
9. NIST SP 800-190 "Application Container Security Guide." 2017.
10. CIS Docker Benchmark. Center for Internet Security, 2023.
