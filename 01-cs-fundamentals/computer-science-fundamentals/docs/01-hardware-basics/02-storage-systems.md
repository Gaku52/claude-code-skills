# Storage Systems

> Data persistence is the foundation of computing, and the evolution of storage technology underpins our digital society.

## What You Will Learn in This Chapter

- [ ] Explain the internal structure and operating principles of HDD/SSD/NVMe
- [ ] Understand the role of file systems and their major implementations
- [ ] Explain the differences between RAID levels and how to choose among them
- [ ] Master storage I/O performance calculations and benchmarking techniques
- [ ] Make informed decisions between cloud storage and on-premises storage
- [ ] Design data protection strategies (backup, replication)

## Prerequisites


---

## 1. HDD (Hard Disk Drive)

### 1.1 Internal Structure

```
Internal structure of an HDD:

  +-------------------------------------+
  |                                     |
  |     +---------------------+         |
  |     |   Platter (Magnetic |         |
  |     |         Disk)       |         |
  |     |   +-------------+  |         |
  |     |   |  ---------  |  |         |
  |     |   | /  Track   \|  |         |
  |     |   ||  +------+  |  |         |
  |     |   || | Spindle| |  |         |
  |     |   ||  +------+  |  |         |
  |     |   | \  Sector  / |  |         |
  |     |   |  ---------  |  |         |
  |     |   +-------------+  |         |
  |     +---------------------+         |
  |                                     |
  |     +-----------------------+       |
  |     |   Arm + Head          |       |
  |     |   <--------* Head     |       |
  |     +-----------------------+       |
  |                                     |
  +-------------------------------------+

  Components:
  - Platter: Aluminum/glass disk coated with magnetic material (both sides used)
  - Spindle: Motor that rotates the platter (5400/7200/10000/15000 RPM)
  - Head: Ultra-small electromagnet that reads/writes magnetic data (floats 10nm above the platter surface)
  - Actuator Arm: Moves the head to the target track
```

### 1.2 Access Time Breakdown

```
HDD Read Time = Seek Time + Rotational Latency + Transfer Time

  Seek Time:
    Time to move the head to the target track
    Average: 3-10ms (Full stroke: 15-20ms)

  Rotational Latency:
    Time waiting for the target sector to rotate under the head
    At 7200RPM: Average 4.17ms (half rotation)
    Calculation: 60s / 7200 rotations / 2 = 4.17ms

  Transfer Time:
    Time to actually read/write the data
    100-200 MB/s -> 5-10us to read 1MB

  Typical random read:
    Seek(5ms) + Rotational Latency(4ms) + Transfer(0.01ms) ~ 9ms
    -> Approximately 110 random I/Os per second (110 IOPS)
```

### 1.3 HDD Technical Details

```
Platter Magnetic Recording Methods:

  * Longitudinal Magnetic Recording (LMR)
    - Magnetization direction: Horizontal to platter surface
    - Limit: Approximately 100-200 Gbit/in^2
    - Dominant until around 2005

  * Perpendicular Magnetic Recording (PMR)
    - Magnetization direction: Perpendicular to platter surface
    - Density: Approximately 500-1000 Gbit/in^2
    - Dominant method since 2005

  * Shingled Magnetic Recording (SMR)
    - Writes tracks overlapping like roof shingles
    - Density: Approximately 25% improvement over PMR
    - Drawback: Extremely slow random writes (adjacent tracks must be rewritten during writes)
    - Use case: Archive and backup applications

  * Heat Assisted Magnetic Recording (HAMR)
    - Uses laser heating to facilitate magnetization
    - Density: 2000+ Gbit/in^2
    - Seagate commercialized in 2024 with 30TB HDDs

  * Microwave Assisted Magnetic Recording (MAMR)
    - Utilizes microwave magnetic resonance
    - Adopted by Western Digital
    - Competing technology with HAMR

HDD Cache Structure:
  +-------------------------------------+
  | HDD Firmware                        |
  | +---------------------------------+ |
  | | DRAM Cache (64-256MB)           | |
  | | - Read buffer                   | |
  | | - Write buffer                  | |
  | | - Read-Ahead cache              | |
  | +---------------------------------+ |
  | +---------------------------------+ |
  | | Command Queuing                 | |
  | | - NCQ (Native Command Queuing)  | |
  | | - Queue depth: Up to 32         | |
  | | - Optimizes access order        | |
  | +---------------------------------+ |
  +-------------------------------------+

  NCQ Effect:
    During random reads, reorders command execution
    to minimize head movement distance
    -> Approximately 20-50% improvement in random IOPS
```

### 1.4 HDD Failures and Countermeasures

```
Typical HDD Failure Patterns:

  1. Head Crash
     - Head contacts the platter surface
     - Causes: Shock, vibration, manufacturing defects
     - Result: Platter surface damage, data loss

  2. Increasing Bad Sectors
     - Magnetic recording weakens due to aging
     - Monitorable via SMART attributes
     - Increasing Reallocated Sectors Count is a danger sign

  3. Spindle Motor Degradation
     - Bearing wear
     - Abnormal noise, startup failures

  4. Firmware Failure
     - Corruption of the SA (Service Area)
     - Can result in inaccessible drives

SMART (Self-Monitoring, Analysis and Reporting Technology):
  Important SMART Attributes:
  | ID  | Name                    | Severity |
  |-----|-------------------------|----------|
  | 5   | Reallocated Sectors     | High     |
  | 10  | Spin Retry Count        | Medium   |
  | 187 | Reported Uncorrectable  | High     |
  | 188 | Command Timeout         | Medium   |
  | 197 | Current Pending Sectors | High     |
  | 198 | Offline Uncorrectable   | High     |
```

```bash
# Check SMART information on Linux
sudo smartctl -a /dev/sda

# Run SMART self-test
sudo smartctl -t short /dev/sda  # Short test (approximately 2 minutes)
sudo smartctl -t long /dev/sda   # Long test (several hours)

# Display SMART attributes
sudo smartctl -A /dev/sda

# On macOS
brew install smartmontools
sudo smartctl -a /dev/disk0
```

---

## 2. SSD (Solid State Drive)

### 2.1 How NAND Flash Works

```
Internal structure of an SSD:

  +-------------------------------------------+
  |  SSD Controller                            |
  |  +------+ +------+ +------+ +-------+    |
  |  | FTL  | | ECC  | | WL   | | GC    |    |
  |  |      | |      | |      | |       |    |
  |  +------+ +------+ +------+ +-------+    |
  |                                            |
  |  +------------------------------------+   |
  |  |    NAND Flash Chips                |   |
  |  |  +----+ +----+ +----+ +----+      |   |
  |  |  |Die0| |Die1| |Die2| |Die3|      |   |
  |  |  +----+ +----+ +----+ +----+      |   |
  |  |  Inside each Die:                  |   |
  |  |  +---------------------------+     |   |
  |  |  | Block 0                   |     |   |
  |  |  | +-----+-----+-----+      |     |   |
  |  |  | |Page0|Page1|Page2| ...   |     |   |
  |  |  | +-----+-----+-----+      |     |   |
  |  |  | Block 1                   |     |   |
  |  |  | ...                       |     |   |
  |  |  +---------------------------+     |   |
  |  +------------------------------------+   |
  |                                            |
  |  +----------------+                        |
  |  | DRAM Cache     | (Mapping table)        |
  |  +----------------+                        |
  +-------------------------------------------+

  FTL: Flash Translation Layer (Logical -> Physical address translation)
  ECC: Error Correcting Code
  WL: Wear Leveling (Write equalization)
  GC: Garbage Collection (Reclaiming invalid blocks)
```

### 2.2 NAND Cell Types and Characteristics

```
Comparison of NAND Cell Types:

  SLC (Single Level Cell): 1 bit/cell
  +------------------+
  | Voltage levels: 2 |  -> "0" or "1"
  | Read: 25us        |
  | Write: 200us      |
  | Erase cycles: 100K|
  | Use: Enterprise high-endurance SSDs
  +------------------+

  MLC (Multi Level Cell): 2 bits/cell
  +------------------+
  | Voltage levels: 4 |  -> "00" "01" "10" "11"
  | Read: 50us        |
  | Write: 600us      |
  | Erase cycles: 3K  |
  | Use: Enterprise SSDs
  +------------------+

  TLC (Triple Level Cell): 3 bits/cell
  +------------------+
  | Voltage levels: 8 |  -> 8 voltage levels represent 3 bits
  | Read: 75us        |
  | Write: 1ms        |
  | Erase cycles: 1K  |
  | Use: Consumer SSDs (current mainstream)
  +------------------+

  QLC (Quad Level Cell): 4 bits/cell
  +-------------------+
  | Voltage levels: 16 |  -> 16 voltage levels represent 4 bits
  | Read: 100us        |
  | Write: 2ms         |
  | Erase cycles: 300  |
  | Use: High-capacity, low-cost SSDs
  +-------------------+

  PLC (Penta Level Cell): 5 bits/cell
  +-------------------+
  | Voltage levels: 32 |  -> 32 voltage levels represent 5 bits
  | Read: 150us        |
  | Write: 3ms+        |
  | Erase cycles: 100  |
  | Use: Archive applications (mass production starting 2025)
  +-------------------+

  Trade-offs:
  Capacity/Cost Efficiency    SLC < MLC < TLC < QLC < PLC
  Speed/Durability            SLC > MLC > TLC > QLC > PLC

  -> The more bits per cell, the cheaper and higher capacity,
     but voltage margins narrow, increasing error rates and latency
```

### 2.3 SSD-Specific Constraints and Optimizations

| Operation | Granularity | Speed |
|-----------|------------|-------|
| Read | Page-level (4-16KB) | ~25us |
| Write | Page-level (4-16KB) | ~250us |
| Erase | **Block-level** (256-512 pages) | ~2ms |

**Important**: SSDs cannot perform "in-place overwrite." A block-level erase is always required before writing.

> **Write Amplification**: To update a single page, the entire block must be read, erased, and rewritten.

> **TRIM**: The OS notifies the SSD that "these blocks are no longer in use," improving GC efficiency.

```
SSD Internal Optimization Mechanisms:

  * FTL (Flash Translation Layer)
    Translates Logical Block Address (LBA) -> Physical Page Address (PPA)

    +----------+         +--------------+
    | OS       |         | NAND Flash   |
    | LBA: 100 |--FTL--->| Die2, Block5,|
    |          |         | Page 42      |
    +----------+         +--------------+

    During writes:
    1. Write to a new free page
    2. Update FTL table (LBA -> new PPA)
    3. Mark old page as invalid
    -> "Append" model rather than "overwrite"

  * Garbage Collection (GC)
    +--------------------------+
    | Block A (GC target)      |
    | +----+----+----+----+   |
    | |Valid|Inv.|Valid|Inv.|   |
    | +----+----+----+----+   |
    +--------------------------+
         | GC execution
    1. Copy valid pages to another block
    2. Erase entire Block A
    3. Reuse as a free block

    GC Timing:
    - Background GC: Runs during idle time
    - Foreground GC: Runs when free blocks are insufficient (causes performance degradation)

  * Wear Leveling
    Equalizes erase counts across all blocks
    - Dynamic WL: Distributes write destinations
    - Static WL: Periodically moves read-only data as well
    -> Maximizes SSD lifespan

  * Over-Provisioning (OP)
    Reserved area hidden from the user (7-28% of total capacity)
    - Reserves free blocks for GC
    - Substitutes for bad blocks
    - Stabilizes performance

    Example: 512GB SSD = Actual NAND capacity ~560GB
             48GB (~9%) is OP area
```

### 2.4 SSD vs HDD

| Metric | HDD | SATA SSD | NVMe SSD |
|--------|-----|----------|----------|
| Sequential Read | 100-200 MB/s | 500 MB/s | 3,500-14,000 MB/s |
| Sequential Write | 100-200 MB/s | 450 MB/s | 3,000-12,000 MB/s |
| Random IOPS | 100-200 | 50,000-100,000 | 500,000-2,000,000 |
| Latency | 3-10 ms | 50-100 us | 10-20 us |
| Power Consumption | 6-8W | 2-3W | 5-8W |
| Lifespan | ~5 years (mechanical wear) | 3-5 years (write limit) | 3-5 years (write limit) |
| Shock Resistance | Low (head crash risk) | High | High |
| Price/TB | ~$15 | ~$50 | ~$60-100 |

### 2.5 SSD Lifespan Calculation and Management

```
How to Calculate SSD Lifespan:

  TBW (Total Bytes Written):
    Metric expressing SSD lifespan in total write volume

    Example: Samsung 990 Pro 2TB
    TBW = 1,200 TB

    If daily write volume is 50GB:
    Lifespan = 1,200TB / (50GB x 365 days) = Approximately 65 years
    -> For general use, there is virtually no need to worry about lifespan

  DWPD (Drive Writes Per Day):
    How many times the full capacity can be written per day during the warranty period

    Example: Enterprise SSD 3.84TB, DWPD=3, 5-year warranty
    Daily write allowance = 3.84TB x 3 = 11.52TB/day
    TBW = 11.52TB x 365 x 5 = 21,024 TB

  SSD Lifespan Monitoring Commands:
```

```bash
# Linux: Check NVMe SSD lifespan information
sudo nvme smart-log /dev/nvme0n1

# Example output:
# percentage_used   : 3%       <- Lifespan consumption rate
# data_units_written : 12345   <- Write volume (512B units x 1000)
# data_units_read    : 67890

# Linux: SATA SSD SMART information
sudo smartctl -a /dev/sda | grep -E "Wear_Leveling|Total_LBAs"

# macOS: Check SSD information with diskutil
diskutil info disk0 | grep -i "smart\|wear\|life"
```

---

## 3. NVMe/PCIe

### 3.1 Protocol Stack

```
Evolution of Storage I/O Protocols:

  SATA (2003):
    CPU -> AHCI -> SATA -> SSD
    * 1 command queue, queue depth 32
    * Maximum bandwidth: 600 MB/s (SATA III)
    * Extension of legacy HDD-oriented protocol

  NVMe over PCIe (2011):
    CPU -> NVMe -> PCIe -> SSD
    * 65,535 command queues, each with queue depth 65,536
    * Maximum bandwidth: 32 GB/s (PCIe 5.0 x4)
    * Protocol designed from scratch for SSDs
    * Lower CPU utilization (reduced interrupts)

  Comparison:
  | Metric          | AHCI/SATA  | NVMe/PCIe    |
  |-----------------|------------|--------------|
  | Queue count     | 1          | 65,535       |
  | Queue depth     | 32         | 65,536       |
  | Bandwidth       | 600 MB/s   | 32+ GB/s     |
  | Latency         | ~100 us    | ~10 us       |
  | CPU efficiency  | Low        | High         |
```

### 3.2 NVMe Detailed Architecture

```
NVMe Submission/Completion Queue:

  +---------------------------------------------+
  |  Host (CPU side)                             |
  |                                              |
  |  Admin Queue:                                |
  |  +-----------------------------+             |
  |  | SQ (Submission Queue)       |             |
  |  | -> Device management cmds   |             |
  |  | CQ (Completion Queue)       |             |
  |  | -> Completion notifications |             |
  |  +-----------------------------+             |
  |                                              |
  |  I/O Queue Pair 1:                           |
  |  +-----------------------------+             |
  |  | SQ1 <- Dedicated to Core 0  |             |
  |  | CQ1 <- Interrupt vector 1   |             |
  |  +-----------------------------+             |
  |                                              |
  |  I/O Queue Pair 2:                           |
  |  +-----------------------------+             |
  |  | SQ2 <- Dedicated to Core 1  |             |
  |  | CQ2 <- Interrupt vector 2   |             |
  |  +-----------------------------+             |
  |  ... (Queue pairs can be created per CPU core)|
  +---------------------------------------------+

  NVMe I/O Processing Flow:
  1. Application calls read()
  2. Driver places command in SQ
  3. Writes to Doorbell register (notifies SSD)
  4. SSD executes the command
  5. Writes completion entry to CQ
  6. MSI-X interrupt notifies the CPU
  7. Driver processes the completion

  -> No locks needed (each CPU core uses its own queue)
  -> Achieves high parallelism
```

### 3.3 NVMe-oF (NVMe over Fabrics)

```
NVMe over Fabrics:

  Local NVMe:
    App -> NVMe -> PCIe -> Local SSD
    Latency: ~10us

  NVMe over Fabrics:
    App -> NVMe -> Network -> Remote SSD
    Latency: ~30-100us

  Supported Transports:
  | Transport       | Latency     | Bandwidth  | Use Case       |
  |-----------------|-------------|------------|----------------|
  | RDMA/RoCE v2    | 30-50 us   | 100+ Gbps | Data center    |
  | TCP              | 50-100 us  | 25+ Gbps  | General purpose|
  | FC (Fibre Ch.)   | 30-50 us   | 32 Gbps   | SAN            |

  Application: Storage disaggregation
  -> Scale compute nodes and storage nodes independently
  -> Foundational cloud technology
```

---

## 4. File Systems

### 4.1 Major File System Comparison

| FS | OS | Max File | Max Volume | Journaling | COW | Characteristics |
|----|-----|----------|-----------|------------|-----|-----------------|
| **ext4** | Linux | 16TB | 1EB | Yes | No | Linux standard, stable |
| **XFS** | Linux | 8EB | 8EB | Yes | No | Optimized for large files |
| **Btrfs** | Linux | 16EB | 16EB | No | Yes | Snapshots, compression |
| **ZFS** | FreeBSD/Linux | 16EB | 256ZB | No | Yes | Strongest data integrity |
| **NTFS** | Windows | 16EB | 256TB | Yes | No | Windows standard |
| **APFS** | macOS/iOS | 8EB | -- | No | Yes | Apple-exclusive, encryption |
| **F2FS** | Android | 3.94TB | 16TB | No | No | SSD/eMMC optimized |

### 4.2 How Journaling Works

```
Journaling (in the case of ext4):

  Normal write (without journaling):
    1. Update metadata
    2. Write data
    -> Power loss midway -> Data inconsistency (FS corruption)

  Journaling:
    1. Write "what is about to be done" to the journal area
    2. Actually update metadata/data
    3. Mark the journal transaction as "complete"

    If power loss occurs:
    -> On boot, read the journal and roll back or replay
       incomplete transactions
    -> FS consistency is guaranteed

  +--------+    +--------+    +--------+
  | 1.Log  |--->| 2.Exec |--->| 3.Done |
  | Journal|    | Data   |    | Commit |
  +--------+    +--------+    +--------+
       ^                            |
       +--- Resume from here on ----+
            power failure
```

```
Comparison of Journaling Modes (ext4):

  * journal (Full journal)
    - Journals both metadata and data
    - Safety: Highest
    - Performance: Lowest (2x write overhead)
    - Use case: Database servers

  * ordered (Default)
    - Journals metadata only
    - Data is written first, then metadata
    - Safety: High
    - Performance: Good
    - Use case: General servers

  * writeback
    - Journals metadata only
    - No ordering guarantee between data and metadata
    - Safety: Low (potential for data corruption)
    - Performance: Highest
    - Use case: Temporary data
```

### 4.3 Copy-on-Write (COW)

```
COW (in the case of Btrfs, ZFS, APFS):

  Traditional FS:
    Data block -> Overwrite in place
    -> Power loss during write causes data corruption

  COW:
    1. Write data to a new block
    2. Update metadata to point to the new block
    3. Release the old block
    -> Even on power loss during write, old data remains intact

  Advantages:
  - Atomic writes (no corruption)
  - Snapshots can be created in O(1) (metadata copy only)
  - Compression and deduplication are straightforward

  Disadvantages:
  - Prone to fragmentation
  - Random write overhead
```

### 4.4 ZFS Detailed Features

```
Key ZFS Features:

  * Storage Pool (zpool)
    +-------------------------------------+
    | ZFS Pool (zpool)                     |
    | +---------------------------------+ |
    | | Dataset: /data                   | |
    | | Dataset: /data/mysql             | |
    | | Dataset: /data/logs              | |
    | | Zvol: /dev/zvol/pool/vm-disk     | |
    | +---------------------------------+ |
    |                                      |
    | vdev: mirror-0  vdev: mirror-1       |
    | +------+------+ +------+------+     |
    | | sda  | sdb  | | sdc  | sdd  |     |
    | +------+------+ +------+------+     |
    +-------------------------------------+

  * Data Integrity (Checksums)
    - SHA-256 checksum on every block
    - Automatic verification on read
    - Detects and repairs silent data corruption (bit rot)

  * Snapshots
    - Creation: Instantaneous (metadata pointer copy only)
    - Capacity: Only consumes delta
    - Rollback: Instantaneous

  * Send/Receive (zfs send/recv)
    - Transfers deltas between snapshots to another pool
    - Ideal for incremental backups
    - Remote replication over WAN

  * Compression
    - LZ4 (default): Fast, low CPU overhead
    - ZSTD: High compression ratio
    - Transparent (invisible to applications)

  * Deduplication (dedup)
    - Stores only one copy of identical data blocks
    - Consumes significant memory (5GB of RAM per 1TB)
    - Effective for VM backups and similar workloads
```

```bash
# ZFS Basic Operations
# Create a pool (mirror configuration)
sudo zpool create tank mirror /dev/sda /dev/sdb

# Create datasets
sudo zfs create tank/data
sudo zfs create tank/data/mysql

# Enable compression
sudo zfs set compression=lz4 tank/data

# Create a snapshot
sudo zfs snapshot tank/data/mysql@before-migration

# List snapshots
sudo zfs list -t snapshot

# Restore from snapshot
sudo zfs rollback tank/data/mysql@before-migration

# Incremental send (remote backup)
sudo zfs send -i @snap1 tank/data@snap2 | ssh remote zfs recv backup/data

# Disk usage status
sudo zpool status
sudo zfs list
```

---

## 5. RAID

### 5.1 RAID Level Comparison

```
RAID 0 (Striping):
  +------+ +------+
  |Disk 0| |Disk 1|
  | A1   | | A2   |  <- Data distributed alternately
  | A3   | | A4   |
  +------+ +------+
  Performance: 2x read/write  Redundancy: None (1 disk death = total loss)

RAID 1 (Mirroring):
  +------+ +------+
  |Disk 0| |Disk 1|
  | A1   | | A1   |  <- Same data replicated
  | A2   | | A2   |
  +------+ +------+
  Performance: 2x read, 1x write  Redundancy: 1 disk failure OK  Capacity: 50%

RAID 5 (Distributed Parity):
  +------+ +------+ +------+
  |Disk 0| |Disk 1| |Disk 2|
  | A1   | | A2   | | Ap   |  <- Parity distributed across disks
  | B1   | | Bp   | | B2   |
  | Cp   | | C1   | | C2   |
  +------+ +------+ +------+
  Performance: (N-1)x read, slow write  Redundancy: 1 disk failure OK  Capacity: (N-1)/N

RAID 6 (Dual Parity):
  RAID 5 + 2 parities -> 2 simultaneous disk failures OK

RAID 10 (1+0: Mirror + Stripe):
  +------+ +------+ +------+ +------+
  |Disk 0| |Disk 1| |Disk 2| |Disk 3|
  | A1   | | A1   | | A2   | | A2   |
  +------+ +------+ +------+ +------+
    Mirror pair 1       Mirror pair 2
  Performance: 4x read, 2x write  Redundancy: 1 per pair OK  Capacity: 50%
```

### 5.2 RAID Selection Guide

| RAID | Capacity Efficiency | Read Performance | Write Performance | Fault Tolerance | Use Case |
|------|-------------------|-----------------|------------------|----------------|----------|
| 0 | 100% | Highest | Highest | None | Temporary/Cache |
| 1 | 50% | Good | Normal | 1 disk | OS, Boot |
| 5 | (N-1)/N | Good | Slow | 1 disk | File server |
| 6 | (N-2)/N | Good | Slowest | 2 disks | Large-scale storage |
| 10 | 50% | Highest | Good | 1 per pair | DB, High performance |

### 5.3 RAID Detailed Explanation and Calculations

```
RAID 5 Parity Calculation:

  Parity = Data1 XOR Data2 XOR Data3 ...

  Example: 3-disk RAID 5
  Disk 0: 10110011
  Disk 1: 01101010
  Parity: 11011001  (= Disk0 XOR Disk1)

  If Disk 1 fails:
  Disk 1 = Disk 0 XOR Parity
         = 10110011 XOR 11011001
         = 01101010  <- Original data recovered!

RAID 5 Write Penalty:

  Operations required to update a single block:
  1. Read old data
  2. Read old parity
  3. Calculate new parity (old_data XOR new_data XOR old_parity)
  4. Write new data
  5. Write new parity
  -> 1 logical write = 2 reads + 2 writes = 4 I/Os

  Write Penalty by RAID Level:
  | RAID | Penalty | Description                      |
  |------|---------|----------------------------------|
  | 0    | 1       | Direct write to stripe           |
  | 1    | 2       | Simultaneous write to mirror     |
  | 5    | 4       | Read-Modify-Write                |
  | 6    | 6       | Dual parity                      |
  | 10   | 2       | Simultaneous write to mirror     |

RAID Performance Calculation Example:

  Conditions: 8 SSDs (100K IOPS each), 70% read : 30% write

  RAID 10:
    Read IOPS = 8 x 100K x 0.7 = 560K IOPS
    Write IOPS = (8/2) x 100K x 0.3 / 2 = 60K IOPS
    Total ~ 620K IOPS

  RAID 5:
    Read IOPS = 7 x 100K x 0.7 = 490K IOPS
    Write IOPS = 7 x 100K x 0.3 / 4 = 52.5K IOPS
    Total ~ 542.5K IOPS
```

### 5.4 Software RAID vs Hardware RAID

```
Hardware RAID:
  +------------------------------+
  | RAID Controller Card         |
  | +----------+ +-----------+  |
  | | Dedicated| | Battery   |  | <- BBU (Battery Backup Unit)
  | | CPU      | | Backup    |  |    Protects cache on power loss
  | | (XOR     | |           |  |
  | |  calc)   | |           |  |
  | +----------+ +-----------+  |
  | +----------------------------+
  | | DRAM Cache (256MB-4GB)    | <- Write cache
  | +----------------------------+
  +------------------------------+
  Advantages: No CPU load, BBU protects cache
  Disadvantages: Expensive, controller itself is a single point of failure

Software RAID:
  Linux mdadm:
    - CPU handles XOR calculations (fast enough on modern CPUs)
    - Free, highly flexible
    - No BBU needed (use UPS instead)
    - Can mix disks from different controllers

  ZFS RAIDZ:
    - RAID-Z1 ~ RAID 5 (1 parity)
    - RAID-Z2 ~ RAID 6 (2 parity)
    - RAID-Z3 (3 parity, for ultra-large capacity)
    - Copy-on-write eliminates RAID 5's "write hole" problem
```

```bash
# Build RAID with Linux mdadm
# Create RAID 1
sudo mdadm --create /dev/md0 --level=1 --raid-devices=2 /dev/sda1 /dev/sdb1

# Create RAID 5
sudo mdadm --create /dev/md1 --level=5 --raid-devices=3 /dev/sda1 /dev/sdb1 /dev/sdc1

# Check RAID status
cat /proc/mdstat
sudo mdadm --detail /dev/md0

# Replace a failed disk
sudo mdadm /dev/md0 --remove /dev/sdb1
sudo mdadm /dev/md0 --add /dev/sdd1

# Monitor rebuild progress
watch cat /proc/mdstat
```

---

## 6. I/O Schedulers

### 6.1 Major Schedulers

| Scheduler | Method | Applicable To |
|-----------|--------|--------------|
| **NOOP/None** | FIFO (First In, First Out) | SSD (hardware optimizes) |
| **Deadline** | Deadline-based | DB, Real-time |
| **CFQ** | Completely Fair Queuing | Desktop (former default) |
| **BFQ** | Budget Fair Queuing | Desktop (low latency) |
| **mq-deadline** | Multi-queue version | NVMe SSD |
| **kyber** | 2-level queue | High-performance NVMe |

### 6.2 I/O Scheduler Details and Selection

```
Operating Principles of Each Scheduler:

  * NOOP/None
    +-----------------------------+
    | Request -> FIFO -> Device   |
    +-----------------------------+
    - No reordering, merging only
    - For SSDs, hardware handles scheduling
    - Sufficient for NVMe

  * mq-deadline
    +---------------------------------+
    | Read queue (deadline: 500ms)    |
    | Write queue (deadline: 5s)      |
    | -> Prioritizes requests near    |
    |    their deadlines              |
    +---------------------------------+
    - Prioritizes reads (response-oriented)
    - Deadlines prevent starvation
    - Optimal for DB servers

  * BFQ (Budget Fair Queuing)
    +---------------------------------+
    | Allocates I/O budget per process|
    | -> High budget for interactive  |
    |    processes                     |
    | -> Low budget for background    |
    |    processes                     |
    +---------------------------------+
    - Smooth desktop experience
    - No stuttering during file copy while playing video

  * kyber
    +---------------------------------+
    | Sync queue (reads) -> Low latency|
    | Async queue (writes) -> High     |
    |   throughput                      |
    | -> Token-based throttling        |
    +---------------------------------+
    - For high-performance NVMe
    - Automatically maintains latency targets
```

```bash
# Check current I/O scheduler
cat /sys/block/nvme0n1/queue/scheduler

# Change I/O scheduler (temporary)
echo "mq-deadline" | sudo tee /sys/block/nvme0n1/queue/scheduler

# Change permanently (udev rule)
echo 'ACTION=="add|change", KERNEL=="nvme*", ATTR{queue/scheduler}="none"' | \
  sudo tee /etc/udev/rules.d/60-scheduler.rules

# Check I/O statistics
iostat -x 1  # Display at 1-second intervals

# Device queue settings
cat /sys/block/nvme0n1/queue/nr_requests    # Queue depth
cat /sys/block/nvme0n1/queue/read_ahead_kb  # Read-ahead size
```

---

## 7. Cloud Storage and Storage Tiers

### 7.1 Cloud Storage Service Comparison

```
AWS Storage Services:

  * EBS (Elastic Block Store)
    +-------------------------------------------+
    | Type          | IOPS    | Throughput    | Use Case   |
    |---------------|---------|---------------|------------|
    | gp3           | 16,000  | 1,000MB/s     | General    |
    | io2           | 64,000  | 1,000MB/s     | DB         |
    | io2 Express   | 256,000 | 4,000MB/s     | High-perf DB|
    | st1           | 500     | 500MB/s       | Logs       |
    | sc1           | 250     | 250MB/s       | Archive    |
    +-------------------------------------------+

  * S3 (Simple Storage Service)
    +-----------------------------------------------+
    | Class           | Avail.  | Price(GB/mo) | Use Case      |
    |-----------------|---------|--------------|---------------|
    | Standard        | 99.99%  | $0.023       | Frequent access|
    | Intelligent     | Auto    | Auto-optimized| Unknown pattern|
    | Standard-IA     | 99.9%   | $0.0125      | Infrequent     |
    | Glacier Instant | 99.9%   | $0.004       | Archive        |
    | Glacier Deep    | 99.99%  | $0.00099     | Long-term      |
    +-----------------------------------------------+

  * Storage Tier Design (Tiering):

    Hot data (frequently accessed):
    +- io2 EBS / gp3 EBS
       +- High cost, highest IOPS

    Warm data (occasionally accessed):
    +- S3 Standard / S3 Standard-IA
       +- Medium cost, ~100ms latency

    Cold data (rarely accessed):
    +- S3 Glacier
       +- Low cost, retrieval takes minutes to hours

    Archive (almost never accessed):
    +- S3 Glacier Deep Archive
       +- Lowest cost, retrieval takes 12 hours
```

### 7.2 Data Protection Strategies

```
3-2-1 Backup Rule:
  3: Maintain 3 copies of data
  2: Store on 2 or more different media types
  1: Keep 1 copy offsite (at a remote location)

  Implementation Example:
  +------------------------------------------+
  | Primary: NVMe SSD (RAID 10)              |
  | | Daily                                  |
  | Secondary: NAS Server (ZFS RAIDZ2)       |
  | | Weekly                                 |
  | Offsite: S3 Glacier / Google Cloud       |
  +------------------------------------------+

RPO (Recovery Point Objective) and RTO (Recovery Time Objective):

  RPO: Acceptable amount of data loss (in time)
  RTO: Acceptable time to system recovery

  | Level              | RPO      | RTO      | Method                    |
  |--------------------|----------|----------|---------------------------|
  | Mission-critical   | 0        | Seconds  | Synchronous replication   |
  | Important systems  | Minutes  | Minutes  | Asynchronous replication  |
  | General operations | Hours    | Hours    | Snapshots                 |
  | Archive            | 1 day    | 1+ days  | Daily backup              |
```

---

## 8. The Future of Storage

| Technology | Characteristics | Status |
|-----------|----------------|--------|
| **CXL** | New CPU-memory protocol, memory pooling | Commercialization began 2024 |
| **Intel Optane** | Characteristics between DRAM and SSD (discontinued) | Production ended, technology transferred to others |
| **PLC NAND** | 5 bits/cell, high capacity, low cost | Mass production underway |
| **DNA Storage** | 215PB of data per gram | Research stage |
| **Glass Storage** | Durability exceeding 1000 years | Microsoft Project Silica |
| **UCIe SSD** | Chiplet-based SSD | 2025 prototype |
| **ZNS (Zoned Namespaces)** | Host-managed SSD writes | For data centers |

### 8.1 ZNS SSD (Zoned Namespaces)

```
Conventional SSD vs ZNS SSD:

  Conventional SSD:
    Host -> LBA -> FTL (inside SSD) -> Physical block
    - FTL is complex (requires large amounts of DRAM)
    - Performance degradation due to GC
    - Capacity loss from over-provisioning

  ZNS SSD:
    Host -> Zone (sequential write region) -> Physical block
    +----------------------------------------+
    | Zone 0: [Written] [Written] [WP->] [Free]      |
    | Zone 1: [Written] [WP->] [Free] [Free]         |
    | Zone 2: [Free] [Free] [Free] [Free]             |
    +----------------------------------------+
    WP = Write Pointer

    - Only sequential writes within a zone
    - Greatly simplified FTL (reduced DRAM)
    - No GC needed (host manages zone resets)
    - No over-provisioning needed
    -> Cost reduction, performance stabilization

    Compatible Software:
    - Linux (blk-zoned)
    - f2fs (native ZNS support)
    - RocksDB (Zenith plugin)
    - Ceph (BlueStore ZNS support)
```

---

## 9. Storage Benchmarking

### 9.1 Benchmarking with fio

```bash
# Install fio (Flexible I/O Tester)
sudo apt install fio  # Ubuntu/Debian
brew install fio       # macOS

# Sequential read benchmark
fio --name=seq_read --filename=/tmp/fio_test \
    --rw=read --bs=1M --size=1G --numjobs=1 \
    --iodepth=32 --ioengine=libaio --direct=1 \
    --runtime=30 --group_reporting

# Sequential write benchmark
fio --name=seq_write --filename=/tmp/fio_test \
    --rw=write --bs=1M --size=1G --numjobs=1 \
    --iodepth=32 --ioengine=libaio --direct=1 \
    --runtime=30 --group_reporting

# Random read benchmark (4KB, IOPS-focused)
fio --name=rand_read --filename=/tmp/fio_test \
    --rw=randread --bs=4k --size=1G --numjobs=4 \
    --iodepth=64 --ioengine=libaio --direct=1 \
    --runtime=30 --group_reporting

# Random write benchmark
fio --name=rand_write --filename=/tmp/fio_test \
    --rw=randwrite --bs=4k --size=1G --numjobs=4 \
    --iodepth=64 --ioengine=libaio --direct=1 \
    --runtime=30 --group_reporting

# Mixed workload (70% read : 30% write)
fio --name=mixed --filename=/tmp/fio_test \
    --rw=randrw --rwmixread=70 --bs=4k --size=1G \
    --numjobs=4 --iodepth=32 --ioengine=libaio \
    --direct=1 --runtime=30 --group_reporting

# With latency histogram
fio --name=latency --filename=/tmp/fio_test \
    --rw=randread --bs=4k --size=1G --numjobs=1 \
    --iodepth=1 --ioengine=libaio --direct=1 \
    --runtime=30 --lat_percentiles=1 \
    --group_reporting
```

### 9.2 How to Read Benchmark Results

```
Interpreting fio Output:

  seq_read: (groupid=0, jobs=1): err= 0: pid=1234
    read: IOPS=3456, BW=3456MiB/s (3623MB/s)
              ~~~~                  ~~~~~~~~
              IOPS count            Effective bandwidth

    clat (usec): min=15, max=892, avg=28.23, stdev=12.41
                 ~~~~            ~~~~~~~~
                 Minimum latency Average latency

    clat percentiles (usec):
     |  1.00th=[   18],  5.00th=[   20], 10.00th=[   21]
     | 50.00th=[   26], 90.00th=[   38], 95.00th=[   45]
     | 99.00th=[   82], 99.50th=[  112], 99.90th=[  245]
     | 99.95th=[  338], 99.99th=[  668]

     -> P99 = 82us: 99% of requests complete within 82us
     -> Focus on tail latency (P99.9+)

  Key Metrics:
  - BW (Bandwidth): Sequential I/O performance
  - IOPS: Random I/O performance
  - clat (Completion Latency): Response time of individual I/Os
  - P99/P99.9: Tail latency (critical for SLA design)
```

---

## 10. Hands-On Exercises

### Exercise 1: I/O Performance Calculation (Fundamentals)

Calculate the time required for the following operations on a 7200RPM HDD:
1. Read a single random 4KB block
2. Read a contiguous 1GB file
3. Read 1000 random 4KB blocks

### Exercise 2: Storage Selection (Applied)

Select the optimal storage and RAID level for the following workloads and explain your reasoning:
1. PostgreSQL database (read-heavy, low latency required)
2. Video streaming service source data storage (high capacity, sequential reads)
3. Web application log collection (write-heavy, sequential append)

### Exercise 3: Benchmarking (Advanced)

Use `fio` or `dd` to measure the storage performance of your own machine:
- Sequential read/write bandwidth
- Random read/write IOPS
- Compare the measured results against theoretical values

### Exercise 4: RAID Calculation (Applied)

Calculate the performance and capacity for the following RAID configurations:
- Disks: 4TB SSD x 8 (each disk: 100K IOPS, 500MB/s sequential)
- Workload: 60% read, 40% write

For each RAID level (0, 1, 5, 6, 10):
1. Effective capacity
2. Theoretical read/write IOPS
3. Theoretical sequential read bandwidth
4. Number of tolerable disk failures

### Exercise 5: Backup Strategy Design (Advanced)

Design a backup strategy for the following systems:
- PostgreSQL: 500GB database, RPO=1 hour, RTO=30 minutes
- User-uploaded images: 10TB, RPO=24 hours, RTO=4 hours
- Access logs: 50GB generated per day, 90-day retention

Design elements:
1. Backup method (full/incremental/differential)
2. Schedule
3. Storage destination
4. Recovery procedure overview
5. Monthly cost estimate (assuming AWS)


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file paths and formats |
| Timeout | Network latency/resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Implement batch processing, add pagination |
| Permission error | Insufficient access permissions | Verify user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read the stack trace to identify the source
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output and debuggers to test hypotheses
5. **Fix and regression test**: After fixing, run tests on related areas

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
            logger.error(f"Exception in: {func.__name__}: {e}")
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

Steps to diagnose when performance issues occur:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Examine disk and network I/O status
4. **Check concurrent connections**: Examine connection pool status

| Issue Type | Diagnostic Tool | Countermeasure |
|-----------|----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the criteria for making technology choices.

| Criterion | When to Prioritize | When to Compromise |
|-----------|-------------------|-------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed users |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+---------------------------------------------------+
|          Architecture Selection Flow               |
+---------------------------------------------------+
|                                                   |
|  (1) Team size?                                   |
|    +- Small (1-5) -> Monolith                     |
|    +- Large (10+) -> Go to (2)                    |
|                                                   |
|  (2) Deployment frequency?                        |
|    +- Once/week or less -> Monolith + Modular     |
|    +- Daily/multiple -> Go to (3)                 |
|                                                   |
|  (3) Team independence?                           |
|    +- High -> Microservices                       |
|    +- Moderate -> Modular Monolith                |
|                                                   |
+---------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A method that is fast in the short term can become technical debt in the long term
- Conversely, over-engineering incurs high short-term costs and can delay projects

**2. Consistency vs Flexibility**
- A unified tech stack reduces learning costs
- Diverse technology adoption enables best-fit choices but increases operational costs

**3. Level of Abstraction**
- High abstraction improves reusability but can make debugging difficult
- Low abstraction is intuitive but prone to code duplication

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
            md += f"- [{icon}] {c['description']}\n"
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
- Focus on the minimum viable set of features
- Automated tests only for critical paths
- Introduce monitoring from the start

**Lessons Learned:**
- Don't pursue perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt consciously

### Scenario 2: Legacy System Modernization

**Situation:** Incrementally modernizing a system that has been running for 10+ years

**Approach:**
- Use the Strangler Fig pattern for gradual migration
- Create Characterization Tests first if existing tests are absent
- Use an API gateway to allow old and new systems to coexist
- Perform data migration in stages

| Phase | Work | Estimated Duration | Risk |
|-------|------|-------------------|------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration start | Migrate peripheral features first | 3-6 months | Medium |
| 4. Core migration | Migrate core functionality | 6-12 months | High |
| 5. Completion | Decommission legacy system | 2-4 weeks | Medium |

### Scenario 3: Development with Large Teams

**Situation:** 50+ engineers developing the same product

**Approach:**
- Use Domain-Driven Design to clarify boundaries
- Set ownership per team
- Manage shared libraries using the Inner Source model
- Design API-first to minimize inter-team dependencies

```python
# Inter-team API contract definition
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
    """Inter-team API contract"""
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

**Optimization Points:**
1. Cache strategy (L1: In-memory, L2: Redis, L3: CDN)
2. Leverage async processing
3. Connection pooling
4. Query optimization and index design

| Optimization Method | Effect | Implementation Cost | Applicable Scenario |
|--------------------|--------|--------------------|--------------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | Heavy I/O wait processing |
| DB optimization | High | High | Slow queries |
| Code optimization | Low-Medium | High | CPU-bound cases |

---

## Leveraging in Team Development

### Code Review Checklist

Key points to verify in code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] No performance impact
- [ ] No security concerns
- [ ] Documentation is updated

### Best Practices for Knowledge Sharing

| Method | Frequency | Target | Benefit |
|--------|-----------|--------|---------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge spread |
| ADR (Decision Records) | Per decision | Future members | Decision transparency |
| Retrospective | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Important designs | Consensus building |

### Managing Technical Debt

```
Priority Matrix:

        High Impact
          |
    +-----+-----+
    | Plan | Fix  |
    | for  | Imme-|
    | Later| diately|
    +-----+-----+
    | Log  | Next |
    | Only | Sprint|
    |      |      |
    +-----+-----+
          |
        Low Impact
    Low Frequency  High Frequency
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|---------------|-----------|----------------|-----------------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor authentication, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scanning |
| Insufficient logging | Medium | Structured logging, audit trails | Log analysis |

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
        """Sanitize input values"""
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
- [ ] Dependency vulnerability scanning is performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Considerations When Upgrading Versions

| Version | Major Changes | Migration Work | Impact Scope |
|---------|--------------|---------------|-------------|
| v1.x -> v2.x | API redesign | Endpoint changes | All clients |
| v2.x -> v3.x | Authentication method change | Token format update | Auth-related |
| v3.x -> v4.x | Data model change | Run migration scripts | DB-related |

### Gradual Migration Procedure

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Gradual migration execution engine"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """Register a migration"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """Execute migration (upgrade)"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"Running: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"Completed: {migration['version']}")
            except Exception as e:
                logger.error(f"Failed: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """Rollback migration"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"Rolling back: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """Check migration status"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### Rollback Plan

Always prepare a rollback plan for migration work:

1. **Data backup**: Take a full backup before migration
2. **Test environment verification**: Pre-verify in an environment equivalent to production
3. **Gradual rollout**: Deploy incrementally with canary releases
4. **Enhanced monitoring**: Shorten metric monitoring intervals during migration
5. **Clear decision criteria**: Define rollback criteria in advance
---

## FAQ

### Q1: How long does an SSD last?

**A**: Expressed as TBW (Total Bytes Written). For typical consumer SSDs:
- 500GB SSD: ~300 TBW (approximately 8 years at 100GB writes per day)
- Under normal use, you will replace the PC itself before the SSD reaches end of life
- Server SSDs have even higher endurance (~10x PBW)

### Q2: Should databases be placed on HDD or SSD?

**A**: For random I/O-heavy databases, **SSD is overwhelmingly superior**:
- HDD: 100 IOPS -> SSD: 100,000+ IOPS (1000x)
- The difference is especially pronounced in index lookups and random JOIN operations
- Cold data (archive) is fine on HDD/S3

### Q3: How do I choose between ZFS and ext4?

**A**:
- **ext4**: Simple, stable, Linux standard. Best for general web servers
- **ZFS**: When data integrity is paramount (NAS, backup, DB). Uses significant memory (1GB of RAM per 1TB recommended)

### Q4: Why is RAID 5 unsuitable for databases?

**A**: Due to RAID 5's write penalty:
- 1 write = 2 reads + 2 writes = 4 I/Os
- Databases have many random writes
- RAID 10 requires only 1 write = 2 I/Os (mirroring only)
- Furthermore, with large-capacity disks, rebuild times can reach tens of hours, increasing the risk of double failure during rebuild

### Q5: Why does NVMe SSD have 65,535 queues?

**A**: For parallel processing with multi-core CPUs:
- Each CPU core can have its own dedicated queue
- I/O requests can be issued simultaneously without lock contention
- Servers commonly have 64+ core CPUs, requiring sufficient queue count
- In practice, several hundred queues are typically used

### Q6: What is SSD over-provisioning?

**A**: A portion of the SSD's actual NAND capacity hidden from the user and used for internal management:
- Substitution for bad blocks
- Working area for GC
- Performance stabilization
- Enterprise SSDs typically use ~28% (1TB labeled SSD has ~1.28TB actual NAND)
- Consumer SSDs typically use ~7%

### Q7: What should I do if I unknowingly purchased an SMR HDD?

**A**: SMR HDDs have no issues with sequential writes, but random writes are extremely slow:
- Not suitable for NAS/RAID (rebuilds can take days)
- Limit usage to backup and archive purposes
- Check CMR/SMR in the manufacturer's spec sheet before purchasing
- Some Seagate Barracuda and WD Blue models use SMR

---

## Summary

| Concept | Key Points |
|---------|-----------|
| HDD | Mechanical, slow random I/O (~100 IOPS), high capacity, low cost |
| SSD | NAND-based, fast random I/O (~100K IOPS), write cycle limitations |
| NVMe | Direct PCIe connection, AHCI->NVMe provides 65,535x parallelism |
| FS | Consistency ensured via journaling (ext4) or COW (ZFS/Btrfs) |
| RAID | 0=Speed, 1=Safety, 5=Balance, 10=Performance+Safety |
| ZNS | Host-managed zone writes improve SSD efficiency |
| Cloud | Tiering optimizes cost (Hot->Cold) |

---

## Recommended Next Guides


---

## References

1. Arpaci-Dusseau, R. H. & Arpaci-Dusseau, A. C. "Operating Systems: Three Easy Pieces." Chapter on Hard Disk Drives and Flash-based SSDs.
2. Cornwell, M. "Anatomy of a Solid-State Drive." ACM Queue, 2012.
3. Agrawal, N. et al. "Design Tradeoffs for SSD Performance." USENIX ATC, 2008.
4. Bonwick, J. & Moore, B. "ZFS: The Last Word in File Systems." Sun Microsystems, 2004.
5. Love, R. "Linux Kernel Development." 3rd Edition, Addison-Wesley, 2010.
6. Bjorling, M. et al. "ZNS: Avoiding the Block Interface Tax for Flash-based SSDs." USENIX ATC, 2021.
7. NVM Express Specification. "NVM Express Base Specification." Revision 2.0.
8. Desnoyers, P. "Analytic Models of SSD Write Performance." ACM TOS, 2014.
9. AWS Documentation. "Amazon EBS Volume Types." https://docs.aws.amazon.com/ebs/
10. Leventhal, A. "A File System All Its Own (ZFS)." ACM Queue, 2013.
