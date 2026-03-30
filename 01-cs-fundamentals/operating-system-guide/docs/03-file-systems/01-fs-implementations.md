# Major Filesystem Implementations

> The choice of filesystem depends on the workload. There is no one-size-fits-all filesystem.

## Learning Objectives

- [ ] Compare the characteristics of major filesystems
- [ ] Select an appropriate filesystem based on the workload
- [ ] Understand the internal structures and algorithms of each filesystem
- [ ] Master filesystem-specific tuning techniques
- [ ] Perform data migration between filesystems


## Prerequisites

Having the following knowledge before reading this guide will deepen your understanding:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding of the content in [Filesystem Basics](./00-fs-basics.md)

---

## 1. ext4 (Linux Standard)

### 1.1 History and Evolution of ext4

```
ext Family Lineage:

  ext (1992): The first Linux filesystem
  -> Maximum 2GB partition
  -> Influenced by UFS (Unix File System)

  ext2 (1993): The first full-featured Linux filesystem
  -> Maximum 4TB partition, 2GB file
  -> No journaling
  -> Stable design used for a long time
  -> Still used in some cases for USB drives

  ext3 (2001): ext2 + journaling
  -> Backward compatible with ext2
  -> Online upgrade from ext2 to ext3 possible
  -> Introduced 3 journaling modes

  ext4 (2008): Major extension of ext3
  -> Maximum 1EB partition, 16TB file
  -> New features: extents, delayed allocation, etc.
  -> Default on Ubuntu, Debian
  -> The most widely used Linux filesystem

  Evolution diagram:
  ext (1992)
   |
   v
  ext2 (1993) -- Added journaling --> ext3 (2001)
                                       |
                                       v
                  Large capacity/extents --> ext4 (2008)
```

### 1.2 Key Features of ext4

```
ext4 (Fourth Extended Filesystem, 2008):
  ext2 -> ext3 (added journaling) -> ext4 (large capacity support)

  Key features:
  +--------------------------------------------------+
  | Capacity Limits                                    |
  | - Maximum volume: 1EB (exabyte)                    |
  | - Maximum file: 16TB                               |
  | - Maximum filename: 255 bytes                      |
  | - Maximum path length: 4096 bytes                  |
  | - Maximum directory entries: approx. 10 million    |
  +--------------------------------------------------+
  | Extents                                            |
  | - Manages contiguous blocks with a single entry    |
  |   (reduces fragmentation)                          |
  | - Extent tree (B-tree structure)                   |
  | - Up to 128MB of contiguous space per extent       |
  +--------------------------------------------------+
  | Delayed Allocation                                 |
  | - Holds writes in page cache                       |
  | - Determines optimal placement at actual disk      |
  |   write time                                       |
  | - Increases probability of contiguous block        |
  |   allocation                                       |
  | - Avoids unnecessary writes for short-lived files  |
  +--------------------------------------------------+
  | Journaling                                         |
  | - Ensures metadata + data consistency              |
  | - 3 modes: journal, ordered, writeback             |
  | - JBD2 (Journaling Block Device 2) engine          |
  +--------------------------------------------------+
  | Directory Indexing                                  |
  | - Fast lookup via HTree (hashed B-tree)            |
  | - Uses half-MD4 hash function                      |
  | - Fast even with millions of files in a directory  |
  +--------------------------------------------------+
  | Multiblock Allocation                              |
  | - Allocates multiple blocks at once                |
  | - Reduces block allocator invocation count         |
  | - Improves contiguous block allocation efficiency  |
  +--------------------------------------------------+
  | Preallocation (fallocate)                          |
  | - Reserves blocks for a file in advance            |
  | - Prevents fragmentation for database files, etc.  |
  | - POSIX fallocate() system call                    |
  +--------------------------------------------------+
  | Flex Block Groups                                  |
  | - Aggregates metadata from multiple block groups   |
  | - Improves metadata locality                       |
  | - Performance improvement for large filesystems    |
  +--------------------------------------------------+

  Advantages: Stability, compatibility, rich tooling, extensive test track record
  Disadvantages: No snapshot support, no data checksums
  Use cases: Desktops, general servers (Ubuntu default)
```

### 1.3 ext4 On-Disk Data Structures

```
ext4 inode structure (256 bytes, extensible):

  Traditional ext2 inode (128 bytes):
  +--------------------------------+
  | i_mode (2)    : File type+perms|
  | i_uid (2)     : Owner ID       |
  |                 (lower 16 bits) |
  | i_size_lo (4) : Size            |
  |                 (lower 32 bits) |
  | i_atime (4)   : Access time    |
  | i_ctime (4)   : Change time    |
  | i_mtime (4)   : Modify time    |
  | i_dtime (4)   : Delete time    |
  | i_gid (2)     : Group ID       |
  | i_links_count (2): Link count  |
  | i_blocks_lo (4): Block count   |
  | i_flags (4)   : Flags          |
  | i_block[15] (60): Data ptrs    |
  | ...                            |
  +--------------------------------+

  ext4 extended fields (128 bytes additional):
  +--------------------------------+
  | i_extra_isize  : Extended size |
  | i_checksum_hi  : Checksum     |
  | i_ctime_extra  : Nanosecond   |
  |                  precision     |
  | i_mtime_extra  : Nanosecond   |
  |                  precision     |
  | i_atime_extra  : Nanosecond   |
  |                  precision     |
  | i_crtime       : Creation time|
  | i_crtime_extra : Creation ns  |
  | i_version_hi   : NFS version  |
  | i_projid       : Project ID   |
  +--------------------------------+

  Extent tree on-disk structure:
  +-------------------------------------+
  | ext4_extent_header:                  |
  |   eh_magic  = 0xF30A                |
  |   eh_entries: Number of entries     |
  |   eh_max:    Maximum entries        |
  |   eh_depth:  Tree depth             |
  |   eh_generation: Generation number  |
  |                                      |
  | When depth=0: ext4_extent[4]        |
  |   ee_block:  Logical block number   |
  |   ee_len:    Block count            |
  |   ee_start:  Physical block number  |
  |                                      |
  | When depth>0: ext4_extent_idx[4]    |
  |   ei_block:  Logical block number   |
  |   ei_leaf:   Child node block       |
  +-------------------------------------+
```

### 1.4 ext4 Tuning

```bash
# Creating an ext4 filesystem (custom settings)
# For a typical server
mkfs.ext4 -L "data" \
  -b 4096 \              # Block size 4KB
  -i 16384 \             # bytes-per-inode (inode density)
  -J size=256 \          # Journal size 256MB
  -O metadata_csum \     # Enable metadata checksums
  -E lazy_itable_init=1 \  # Lazy inode table initialization
  /dev/sda1

# For a mail server (many small files)
mkfs.ext4 -L "mail" \
  -b 4096 \
  -i 4096 \              # Allocate more inodes
  -J size=128 \
  /dev/sda1

# For large files (media server)
mkfs.ext4 -L "media" \
  -b 4096 \
  -i 1048576 \           # Fewer inodes (large files assumed)
  -T largefile4 \        # Large file preset
  /dev/sda1

# Mount option optimization
# /etc/fstab
# General use (recommended settings)
/dev/sda1 / ext4 defaults,noatime,commit=60 0 1

# Database use
/dev/sda1 /var/lib/mysql ext4 defaults,noatime,data=ordered,barrier=1,commit=5 0 2

# Temporary files / build use
/dev/sda1 /tmp ext4 defaults,noatime,data=writeback,barrier=0,commit=120 0 0

# For SSDs
/dev/sda1 / ext4 defaults,noatime,discard,commit=60 0 1
```

```bash
# Tuning ext4 during operation

# Check/change journaling mode
cat /proc/fs/ext4/sda1/options | grep data
# data=ordered

# Adjust commit interval (default 5 seconds)
# Longer improves performance, shorter improves safety
sudo tune2fs -o commit=30 /dev/sda1

# Adjust reserved blocks (default 5%)
# Can be reduced on large disks
sudo tune2fs -m 1 /dev/sda1       # Reduce to 1%
sudo tune2fs -l /dev/sda1 | grep "Reserved"

# Set filesystem label
sudo e2label /dev/sda1 "my-data"

# Set check interval
sudo tune2fs -c 50 /dev/sda1      # Check every 50 mounts
sudo tune2fs -i 6m /dev/sda1      # Check every 6 months

# Enable metadata checksums (ext4 >= 3.18)
sudo tune2fs -O metadata_csum /dev/sda1

# Check fragmentation
sudo e4defrag -c /mount/point
# Fragmentation score: 0-30=low, 30-55=medium, 55-100=high
```

---

## 2. XFS

### 2.1 XFS Overview and History

```
XFS (SGI, 1993 -> Linux):
  Developed by Silicon Graphics for IRIX
  -> Ported to Linux in 2001
  -> Adopted as RHEL 7 default in 2014
  -> Currently one of the most widely used Linux filesystems

  Design philosophy:
  - Designed for large-scale storage (originally for supercomputers)
  - High parallel I/O performance
  - Emphasis on scalability
  - Native 64-bit design (from the beginning)

  Key features:
  +--------------------------------------------------+
  | Capacity Limits                                    |
  | - Maximum volume: 8EB (64-bit)                     |
  | - Maximum file: 8EB                                |
  | - Maximum filename: 255 bytes                      |
  +--------------------------------------------------+
  | B+ tree-based metadata management                  |
  | - All metadata managed by B+ trees                 |
  | - Inode allocation, free block management,         |
  |   directories                                      |
  | - Efficient search and update                      |
  +--------------------------------------------------+
  | Allocation Groups (AG)                             |
  | - Divides the filesystem into independent regions  |
  | - Each AG has its own independent metadata         |
  | - Parallel access possible (concurrent writes to   |
  |   different AGs)                                   |
  | - Reduces lock contention                          |
  +--------------------------------------------------+
  | Delayed Allocation                                 |
  | - Like ext4, delays block allocation at write time |
  | - Optimizes contiguous block allocation            |
  +--------------------------------------------------+
  | Journaling                                         |
  | - Metadata-only journaling                         |
  | - External log device support                      |
  | - Asynchronous delayed logging                     |
  +--------------------------------------------------+
  | Online Operations                                  |
  | - Online resize (expansion only, no shrinking)     |
  | - Online defrag (xfs_fsr)                          |
  | - Online dump/restore (xfsdump/xfsrestore)         |
  +--------------------------------------------------+
```

### 2.2 XFS Internal Structure

```
XFS disk layout:

  +----------------------------------------------+
  | Filesystem (entire)                           |
  +----------+----------+----------+--------------+
  |   AG 0   |   AG 1   |   AG 2   |    AG 3     |
  +----------+----------+----------+--------------+

  Structure of each AG (Allocation Group):
  +----------------------------------------------+
  | AG header area:                               |
  | +------+------+------+------+------+         |
  | | AGF  | AGI  | AGFL |Free  |inode |         |
  | |      |      |      |Space |B+Tree|         |
  | |      |      |      |B+Tree|      |         |
  | +------+------+------+------+------+         |
  |                                               |
  | Data area:                                    |
  | +------------------------------------+        |
  | | inode chunks + data blocks          |        |
  | +------------------------------------+        |
  +----------------------------------------------+

  AGF (AG Free Space): Free block management
  AGI (AG Inode):      Inode management
  AGFL (AG Free List): Block management for AGF/AGI B+ trees

  B+ tree usage:
  +--------------------------------------------+
  | Where XFS uses B+ trees                     |
  +--------------------------------------------+
  | 1. Free block management (by block number)  |
  | 2. Free block management (by size)          |
  | 3. Inode allocation management              |
  | 4. Directory entries                        |
  | 5. Extent maps (file data)                  |
  | 6. Reverse mapping (for reflink)            |
  | 7. Reference counts (for reflink)           |
  +--------------------------------------------+
```

### 2.3 XFS Tuning and Operations

```bash
# Creating an XFS filesystem
mkfs.xfs -L "xfs-data" \
  -b size=4096 \         # Block size
  -d agcount=16 \        # Number of AGs (adjusts parallelism)
  -l size=256m \         # Log size
  /dev/sda1

# Using an external log device (performance improvement)
mkfs.xfs -l logdev=/dev/sdb1,size=256m /dev/sda1
mount -o logdev=/dev/sdb1 /dev/sda1 /mnt

# Checking XFS information
xfs_info /mount/point
# meta-data=/dev/sda1  isize=512  agcount=16, agsize=...
# data     =           bsize=4096 blocks=...
# naming   =version 2  bsize=4096 ascii-ci=0
# log      =internal   bsize=4096 blocks=...

# Mount options
# /etc/fstab
/dev/sda1 /data xfs defaults,noatime,inode64,logbufs=8 0 0

# Performance-related options:
# logbufs=N     : Number of log buffers (2-8, default 8)
# logbsize=N    : Log buffer size (32K-256K)
# nobarrier     : Disable write barriers (for RAID with BBU)
# allocsize=N   : Speculative preallocation size
# inode64       : 64-bit inode numbers (required for large FS)

# Online resize (expansion only)
xfs_growfs /mount/point

# Note: XFS cannot be shrunk
# If shrinking is needed: backup -> recreate -> restore

# Defragmentation
xfs_fsr /mount/point          # Entire filesystem
xfs_fsr /path/to/file         # Specific file
xfs_fsr -v /mount/point       # Verbose output

# Backup and restore (XFS-specific tools)
xfsdump -l 0 -f /backup/dump.xfsdump /mount/point
xfsrestore -f /backup/dump.xfsdump /restore/point

# Repair
xfs_repair /dev/sda1           # Must be unmounted
xfs_repair -L /dev/sda1        # Clear log to zero (last resort)

# XFS metadata dump (for debugging)
xfs_db -r /dev/sda1
xfs_db> sb 0                   # Display superblock
xfs_db> freesp                 # Free space distribution
```

### 2.4 XFS reflink and Copy

```
reflink (reference link):
  Supported since XFS 4.9+. A mechanism for instant file copying.

  Normal copy:
  cp source dest -> Copies all data blocks (time proportional to file size)

  reflink copy:
  cp --reflink source dest -> Copies metadata only (instant)
  -> Shares data blocks
  -> When either is modified, only the changed portion is written
     to a new block (CoW)

  Mechanism:
  +----------------------------------------+
  | Before reflink:                         |
  | source -> [Block A] [Block B] [Block C]|
  |                                         |
  | After reflink:                          |
  | source -> [Block A] [Block B] [Block C]|
  | dest   ->/          /          /        |
  | (reference count = 2)                   |
  |                                         |
  | Modifying Block B in dest:              |
  | source -> [Block A] [Block B] [Block C]|
  | dest   ->/         [Block B'] /         |
  | Block A, C: reference count = 2         |
  | Block B:    reference count = 1         |
  | Block B':   reference count = 1 (new)   |
  +----------------------------------------+

  Usage examples:
  # Create a reflink copy
  $ cp --reflink=auto source.img dest.img   # Use reflink if possible
  $ cp --reflink=always source.img dest.img # Require reflink (error if not possible)

  # Fast cloning of virtual machine images
  $ cp --reflink=always base.qcow2 vm1.qcow2  # Instant
  $ cp --reflink=always base.qcow2 vm2.qcow2  # Instant

  # Check if reflink is enabled
  $ xfs_info /mount/point | grep reflink
  # reflink=1 means enabled

  # Enable reflink at mkfs time (enabled by default, XFS 5.1+)
  $ mkfs.xfs -m reflink=1 /dev/sda1
```

---

## 3. Btrfs

### 3.1 Btrfs Overview

```
Btrfs (B-tree File System, Oracle -> Linux, 2009):
  Pronounced "butter FS"
  -> Developed as a Linux equivalent to ZFS
  -> Default on SUSE Linux Enterprise (15 SP1+)
  -> Default on Fedora 33+ (Workstation edition)

  Design philosophy:
  - Copy-on-Write (CoW) based
  - Checksums for all data and metadata
  - Flexible storage management
  - Efficient support for snapshots and clones
  - Enterprise features native to Linux

  Key features:
  +--------------------------------------------------+
  | Copy-on-Write                                      |
  | - Writes data to a new location instead of         |
  |   overwriting                                      |
  | - Structural guarantee of crash consistency        |
  | - No journal needed                                |
  +--------------------------------------------------+
  | Snapshots                                          |
  | - Instant backup of subvolumes                     |
  | - Read-only / read-write capable                   |
  | - Incremental backup (send/receive)                |
  +--------------------------------------------------+
  | Transparent Compression                            |
  | - zstd (recommended), lzo, zlib                    |
  | - Configurable per file/directory                  |
  | - Automatic compression/decompression on           |
  |   read/write                                       |
  +--------------------------------------------------+
  | Data and Metadata Checksums                        |
  | - CRC32C (default), xxhash, sha256, blake2b       |
  | - Detection of silent data corruption (bit rot)    |
  | - Auto-repair when combined with RAID              |
  +--------------------------------------------------+
  | Built-in RAID                                      |
  | - RAID 0, 1, 10: Stable                            |
  | - RAID 5, 6: Write hole issue (experimental)       |
  | - Profile can be dynamically changed               |
  +--------------------------------------------------+
  | Subvolumes                                         |
  | - Independent namespaces within the filesystem     |
  | - Alternative to partition splitting               |
  | - Individually mountable                           |
  | - Individually snapshottable                       |
  +--------------------------------------------------+
  | Online Operations                                  |
  | - Online resize (both expansion and shrinking)     |
  | - Online defragmentation                           |
  | - Online scrub (data integrity check)              |
  | - Online balance (data redistribution)             |
  +--------------------------------------------------+
  | Deduplication                                      |
  | - Offline: duperemove tool                         |
  | - Efficient deduplication based on reflink         |
  +--------------------------------------------------+
```

### 3.2 Btrfs Internal Structure

```
Btrfs disk layout:

  Everything is managed by B-trees:

  +--------------------------------------------+
  | Superblock (3 copies: 64KB, 64MB, 256GB)   |
  |                                             |
  | +----------------------------------+        |
  | | Tree of Trees (Root Tree)        |        |
  | | -> Manages roots of all B-trees  |        |
  | +------+-------+--------+---------+        |
  |        v       v        v                   |
  |  +--------++--------++--------+             |
  |  |FS Tree ||Extent  ||Checksum|             |
  |  |(files  || Tree   || Tree   |             |
  |  | + dirs)|| (block ||        |             |
  |  |        || mgmt)  ||        |             |
  |  +--------++--------++--------+             |
  |  +--------++--------++--------+             |
  |  |Chunk   ||Device  ||UUID    |             |
  |  | Tree   || Tree   || Tree   |             |
  |  |(logical|| (device||        |             |
  |  | ->     || mgmt)  ||        |             |
  |  | phys)  ||        ||        |             |
  |  +--------++--------++--------+             |
  +--------------------------------------------+

  CoW update:
  +---------------------------------------------+
  | Before update:                               |
  | Root -> A -> [D] [E] [F]                    |
  |           -> [G] [H]                         |
  |                                              |
  | Modifying [E]:                               |
  | 1. Create a copy of E as E' at a new loc     |
  | 2. Create a copy of A as A' (pointing to E') |
  | 3. Update Root to A' (atomic)                |
  |                                              |
  | After update:                                |
  | Root' -> A' -> [D] [E'] [F]                 |
  |             -> [G] [H]                       |
  |                                              |
  | Old Root, A, E can be freed                  |
  | (retained if referenced by a snapshot)       |
  +---------------------------------------------+
```

### 3.3 Btrfs Subvolumes and Snapshots

```bash
# Subvolume management
# Create subvolumes
btrfs subvolume create /mnt/@home
btrfs subvolume create /mnt/@var
btrfs subvolume create /mnt/@snapshots

# List subvolumes
btrfs subvolume list /mnt
# ID 256 gen 100 top level 5 path @home
# ID 257 gen 101 top level 5 path @var
# ID 258 gen 102 top level 5 path @snapshots

# Mount subvolumes individually
# /etc/fstab
/dev/sda1  /home       btrfs  subvol=@home,defaults,noatime,compress=zstd  0  0
/dev/sda1  /var        btrfs  subvol=@var,defaults,noatime                  0  0

# Create snapshots
# Read-only snapshot
btrfs subvolume snapshot -r /mnt/@home /mnt/@snapshots/home-$(date +%Y%m%d)

# Read-write snapshot
btrfs subvolume snapshot /mnt/@home /mnt/@snapshots/home-writable

# Restore a file from a snapshot
cp /mnt/@snapshots/home-20240101/user/important.txt /home/user/

# Roll back to a snapshot (for read-write snapshots)
# Delete the current subvolume and rename the snapshot
btrfs subvolume delete /mnt/@home
btrfs subvolume snapshot /mnt/@snapshots/home-20240101 /mnt/@home

# Delete a snapshot
btrfs subvolume delete /mnt/@snapshots/home-20240101

# Incremental backup (send/receive)
# First time: full backup
btrfs subvolume snapshot -r /mnt/@home /mnt/@snapshots/snap1
btrfs send /mnt/@snapshots/snap1 | btrfs receive /backup/

# Subsequent times: incremental backup
btrfs subvolume snapshot -r /mnt/@home /mnt/@snapshots/snap2
btrfs send -p /mnt/@snapshots/snap1 /mnt/@snapshots/snap2 | btrfs receive /backup/
# -> Only the diff between snap1 and snap2 is transferred

# Incremental backup over SSH
btrfs send -p /mnt/@snapshots/snap1 /mnt/@snapshots/snap2 | \
  ssh backup-server "btrfs receive /backup/"
```

### 3.4 Btrfs Compression and RAID

```bash
# Transparent compression settings

# Enable compression at mount time
mount -o compress=zstd /dev/sda1 /mnt
mount -o compress=zstd:3 /dev/sda1 /mnt   # Specify compression level

# Compression algorithm comparison:
# +-----------+----------+----------+----------+
# | Algorithm | Ratio    | Compress | Decompr  |
# |           |          | Speed    | Speed    |
# +-----------+----------+----------+----------+
# | lzo       | Low      | Fastest  | Fastest  |
# | zstd:1    | Medium   | Fast     | Fast     |
# | zstd:3    | Med-High | Medium   | Fast     | <- Recommended
# | zlib:6    | High     | Slow     | Medium   |
# | zstd:15   | Highest  | Slowest  | Fast     |
# +-----------+----------+----------+----------+

# Disable compression for a specific directory
btrfs property set /mnt/database compression ""
chattr +m /mnt/database    # No-compression flag

# Recompress existing data
btrfs filesystem defragment -r -czstd /mnt/

# Check compression statistics
btrfs filesystem df /mnt
compsize /mnt                # Detailed compression ratio tool

# RAID configuration

# RAID1 (mirroring)
mkfs.btrfs -d raid1 -m raid1 /dev/sda1 /dev/sdb1

# RAID10 (stripe + mirror)
mkfs.btrfs -d raid10 -m raid10 /dev/sd{a,b,c,d}1

# Dynamically change RAID profile
# Convert single -> RAID1
btrfs balance start -dconvert=raid1 -mconvert=raid1 /mnt

# Add a device
btrfs device add /dev/sdc1 /mnt
btrfs balance start /mnt      # Redistribute data

# Remove a device
btrfs device delete /dev/sda1 /mnt

# Replace a device
btrfs replace start /dev/sda1 /dev/sdd1 /mnt

# Check RAID information
btrfs filesystem show /mnt
btrfs filesystem df /mnt
btrfs filesystem usage /mnt
```

### 3.5 Btrfs Operations and Maintenance

```bash
# Scrub (data integrity check)
# -> Verifies checksums of all data
# -> Auto-repair with RAID
btrfs scrub start /mnt
btrfs scrub status /mnt

# Set up periodic scrub (systemd timer)
# /etc/systemd/system/btrfs-scrub.timer
# [Timer]
# OnCalendar=monthly
# [Install]
# WantedBy=timers.target

# Balance (data redistribution)
btrfs balance start /mnt
btrfs balance status /mnt
btrfs balance cancel /mnt      # Cancel

# Detailed usage information
btrfs filesystem usage /mnt
btrfs filesystem df /mnt
btrfs filesystem show

# Quota settings
btrfs quota enable /mnt
btrfs qgroup limit 50G /mnt/@home
btrfs qgroup show /mnt

# Filesystem repair
# Normal check
btrfs check /dev/sda1

# Execute repair (caution: dangerous operation)
btrfs check --repair /dev/sda1

# Rescue mode
btrfs rescue super-recover /dev/sda1
btrfs rescue zero-log /dev/sda1
btrfs rescue chunk-recover /dev/sda1
```

---

## 4. ZFS

### 4.1 ZFS Overview

```
ZFS (Zettabyte File System, Sun Microsystems, 2005):
  Designed as "the last filesystem"
  -> Released with OpenSolaris in 2005
  -> Ported to FreeBSD (native support)
  -> Available on Linux as ZFS on Linux (OpenZFS)
  -> Cannot be integrated into the kernel due to license issues
     (CDDL vs GPL)

  Design philosophy:
  - Integrated management of the entire storage stack
  - Volume manager + filesystem + RAID
  - Enterprise-grade data protection
  - Simplified administration ("set and forget")

  Key features:
  +--------------------------------------------------+
  | 128-bit Addressing                                 |
  | - Virtually unlimited capacity                     |
  |   (256 trillion yobibytes)                         |
  | - Checksums on all data (SHA-256/Fletcher4)        |
  | - Triple copies of metadata                        |
  +--------------------------------------------------+
  | Pool-based Storage Management                      |
  | - zpool: Pool of physical devices                  |
  | - zfs dataset: Logical volume carved from pool     |
  | - No partitioning needed, dynamic resizing         |
  +--------------------------------------------------+
  | RAID-Z (Improved RAID5/6)                          |
  | - RAID-Z1: Tolerates 1 disk failure                |
  | - RAID-Z2: Tolerates 2 disk failures               |
  | - RAID-Z3: Tolerates 3 disk failures               |
  | - No write hole problem (CoW-based)                |
  +--------------------------------------------------+
  | ARC (Adaptive Replacement Cache)                   |
  | - Advanced caching algorithm                       |
  | - Adaptive combination of MRU (Most Recently       |
  |   Used) + MFU (Most Frequently Used)               |
  | - L2ARC: Uses SSD as cache device                  |
  +--------------------------------------------------+
  | ZIL (ZFS Intent Log)                               |
  | - Accelerates synchronous writes                   |
  | - SLOG: Places ZIL on a separate device (SSD)      |
  +--------------------------------------------------+
  | Snapshots and Clones                               |
  | - Instant snapshot creation                        |
  | - Read-write clones                                |
  | - Efficient replication via send/receive            |
  +--------------------------------------------------+
  | Deduplication                                      |
  | - Block-level deduplication                        |
  | - DDT (Dedup Table) kept in memory                 |
  | - Requires large amounts of memory                 |
  |   (approx. 5GB RAM per 1TB)                        |
  +--------------------------------------------------+
  | Compression                                        |
  | - LZ4 (default, recommended), gzip, zstd, lzjb    |
  | - Transparent compression                          |
  | - Can combine compression + deduplication          |
  +--------------------------------------------------+
```

### 4.2 ZFS Basic Operations

```bash
# Installing ZFS (Ubuntu)
sudo apt install zfsutils-linux

# Creating a pool (zpool)

# Simple pool (stripe, no redundancy)
sudo zpool create tank /dev/sdb

# Mirror pool (RAID1 equivalent)
sudo zpool create tank mirror /dev/sdb /dev/sdc

# RAID-Z1 pool (RAID5 equivalent)
sudo zpool create tank raidz1 /dev/sdb /dev/sdc /dev/sdd

# RAID-Z2 pool (RAID6 equivalent)
sudo zpool create tank raidz2 /dev/sd{b,c,d,e}

# Pool with cache and log
sudo zpool create tank raidz1 /dev/sd{b,c,d} \
  cache /dev/sde \      # L2ARC (SSD for read cache)
  log mirror /dev/sdf /dev/sdg  # SLOG (SSD for write log, mirrored)

# Check pool information
zpool status tank
zpool list
zpool iostat -v tank 5     # I/O statistics every 5 seconds

# Create and manage datasets
sudo zfs create tank/home
sudo zfs create tank/data
sudo zfs create tank/backup

# Set properties
sudo zfs set compression=lz4 tank         # Enable compression
sudo zfs set atime=off tank               # Disable atime
sudo zfs set recordsize=1M tank/media     # For large files
sudo zfs set quota=100G tank/home         # Set quota
sudo zfs set reservation=50G tank/data    # Reserve space

# Check properties
zfs get all tank/home
zfs get compression,compressratio tank
zfs list -o name,used,avail,refer,mountpoint
```

### 4.3 ZFS Snapshots and Replication

```bash
# Create a snapshot
sudo zfs snapshot tank/home@daily-$(date +%Y%m%d)

# Recursive snapshot (including all child datasets)
sudo zfs snapshot -r tank@backup-$(date +%Y%m%d)

# List snapshots
zfs list -t snapshot

# Restore a file from a snapshot
# Snapshots are accessible via .zfs/snapshot/
ls /tank/home/.zfs/snapshot/daily-20240101/
cp /tank/home/.zfs/snapshot/daily-20240101/file.txt /tank/home/

# Roll back to a snapshot
sudo zfs rollback tank/home@daily-20240101

# Delete a snapshot
sudo zfs destroy tank/home@daily-20240101

# Bulk delete old snapshots
sudo zfs destroy tank/home@%daily-202301  # Delete all from January 2023

# Replication (send/receive)
# First time: full backup
sudo zfs send tank/home@snap1 | sudo zfs receive backup/home

# Incremental backup
sudo zfs send -i tank/home@snap1 tank/home@snap2 | \
  sudo zfs receive backup/home

# Replication over SSH
sudo zfs send -i tank/home@snap1 tank/home@snap2 | \
  ssh backup-server "sudo zfs receive backup/home"

# Encrypted send (raw send)
sudo zfs send --raw -i tank/home@snap1 tank/home@snap2 | \
  ssh backup-server "sudo zfs receive backup/home"

# Automated snapshots (zfs-auto-snapshot, sanoid/syncoid)
# Example sanoid.conf:
# [tank/home]
#   use_template = production
#   autosnap = yes
#   autoprune = yes
# [template_production]
#   hourly = 24
#   daily = 30
#   monthly = 12
#   yearly = 5
```

### 4.4 ZFS Performance Tuning

```bash
# Check and adjust ARC (cache)
cat /proc/spl/kstat/zfs/arcstats | grep -E "^(size|c_max|hits|misses)"
# Set ARC size limit (/etc/modprobe.d/zfs.conf)
# options zfs zfs_arc_max=8589934592   # 8GB

# Disable ZIL (write log) (not recommended, for testing only)
# sync=disabled disables synchronous writes
sudo zfs set sync=disabled tank/tmp

# Optimize record size
# Database (small block I/O)
sudo zfs set recordsize=8K tank/database

# Large files (media, backups)
sudo zfs set recordsize=1M tank/media

# General use
sudo zfs set recordsize=128K tank/home    # Default

# Check compression effectiveness
zfs get compressratio tank
# NAME  PROPERTY       VALUE  SOURCE
# tank  compressratio  2.50x  -
# -> 2.5x compression ratio = 40% disk savings

# Scrub (data integrity check)
sudo zpool scrub tank
zpool status tank      # Check scrub progress

# Check I/O performance
zpool iostat -v 5      # Every 5 seconds

# Replace a device
sudo zpool replace tank /dev/sdb /dev/sdh

# Add a device (add a mirror)
sudo zpool attach tank /dev/sdb /dev/sdc  # Add sdc as mirror of sdb
```

---

## 5. Other Filesystems

### 5.1 NTFS (Windows)

```
NTFS (New Technology File System, 1993):
  The standard filesystem for Windows NT and later

  Key features:
  - MFT (Master File Table): Manages metadata for all files
  - Journaling: USN (Update Sequence Number) Journal
  - ACL: Windows-specific access control lists
  - Encryption: EFS (Encrypting File System)
  - Compression: NTFS compression (per file/directory)
  - Hard links, symbolic links (Vista and later)
  - Alternate Data Streams (ADS): Multiple data streams per file
  - Quotas: Per-user disk usage limits
  - VSS (Volume Shadow Copy Service): Snapshots

  Capacity:
  - Maximum volume: 256TB (theoretically 16EB)
  - Maximum file: 256TB
  - Cluster size: 4KB (default)

  Accessing from Linux:
  # Kernel driver (read-only)
  mount -t ntfs /dev/sda1 /mnt

  # NTFS-3G (read-write, FUSE-based)
  mount -t ntfs-3g /dev/sda1 /mnt

  # ntfs3 (Linux 5.15+, built into kernel, read-write)
  mount -t ntfs3 /dev/sda1 /mnt

  MFT structure:
  +----------------------------------------+
  | MFT entry (typically 1KB)              |
  | +-------------------------------------+|
  | | $STANDARD_INFORMATION              ||
  | |  -> Timestamps, flags               ||
  | +-------------------------------------+|
  | | $FILE_NAME                          ||
  | |  -> Filename, parent directory ref   ||
  | +-------------------------------------+|
  | | $DATA                               ||
  | |  -> File data                        ||
  | |  -> Small files stored within MFT    ||
  | |  -> Large files reference runs       ||
  | +-------------------------------------+|
  | | $SECURITY_DESCRIPTOR                ||
  | |  -> ACL information                  ||
  | +-------------------------------------+|
  +----------------------------------------+
```

### 5.2 APFS (Apple)

```
APFS (Apple File System, 2017):
  Developed by Apple as the successor to HFS+
  -> Default since macOS High Sierra (10.13)
  -> Adopted since iOS 10.3
  -> Also used on watchOS, tvOS

  Design philosophy:
  - Optimized for SSD/flash storage
  - Encryption-first design
  - Container + volume model

  Key features:
  - CoW (Copy-on-Write)
  - Snapshots: Used for Time Machine backups
  - Encryption: Entire volume / per-file (FileVault 2)
  - Space sharing: Multiple volumes share container capacity
  - Clones: Instant copy of files/directories
  - Nanosecond timestamps (HFS+ had second precision)
  - Crash protection
  - TRIM support (SSD optimization)

  Container model:
  +----------------------------------------+
  | APFS Container (= physical partition)  |
  | +----------++----------++----------+   |
  | | Volume 1 || Volume 2 || Volume 3 |   |
  | | (macOS)  || (Data)   || (VM)     |   |
  | |          ||          ||          |   |
  | +----------++----------++----------+   |
  |   <- Free space shared between vols -> |
  +----------------------------------------+

  Volume layout since macOS Catalina:
  - Macintosh HD (System): Read-only system volume
  - Macintosh HD - Data: User data
  - Preboot, Recovery, VM: System volumes

  Checking with diskutil:
  $ diskutil list
  $ diskutil apfs list
  $ diskutil info /
```

### 5.3 FAT32 / exFAT

```
FAT32 (File Allocation Table, 1996):
  The most widely compatible filesystem

  Features:
  - Read/write on all OSes (Windows, macOS, Linux, embedded)
  - Maximum file: 4GB (biggest limitation)
  - Maximum volume: 2TB (theoretically 8TB)
  - No journaling
  - No ACL (basic permissions only)

  Structure:
  +------+----------+-----------+----------+
  | Boot | FAT 1    | FAT 2     | Data     |
  | Sect | (file    | (backup)  | area     |
  |      | alloc    |           |          |
  |      | table)   |           |          |
  +------+----------+-----------+----------+

  FAT (File Allocation Table):
  A linked list recording the next cluster number for each cluster
  +------------------------------+
  | Cluster 2: -> 3              |
  | Cluster 3: -> 7              |
  | Cluster 4: -> EOF (free)     |
  | Cluster 5: -> 0 (free)      |
  | Cluster 6: -> EOF            |
  | Cluster 7: -> EOF            |
  +------------------------------+
  File A: 2 -> 3 -> 7 -> EOF

  Use cases: SD cards, USB drives (small capacity), embedded systems

exFAT (Extended FAT, 2006):
  Successor to FAT32. Supports large files.

  Features:
  - Maximum file: 16EB (virtually unlimited)
  - Maximum volume: 128PB
  - Removes FAT32's 4GB limitation
  - Microsoft published patents (2019)
  - Native support since Linux kernel 5.4
  - No journaling
  - No ACL

  Use cases:
  - SD cards (official filesystem for SDXC standard)
  - USB drives (large files)
  - External HDDs shared between Windows/Mac
  - Memory cards for digital cameras

  Creation:
  # FAT32
  $ mkfs.vfat -F 32 /dev/sdb1
  # exFAT
  $ mkfs.exfat /dev/sdb1
```

### 5.4 Special Virtual Filesystems

```
tmpfs:
  Filesystem on RAM
  - Mount points: /tmp, /dev/shm, /run
  - Extremely fast but lost on reboot
  - Can also be swapped out
  - Ideal for build caches, temporary files

  Configuration:
  # /etc/fstab
  tmpfs  /tmp      tmpfs  defaults,size=4G,noatime  0  0
  tmpfs  /dev/shm  tmpfs  defaults,size=2G          0  0

procfs:
  Virtual filesystem exposing process and kernel information
  Mount point: /proc

  Important files:
  /proc/cpuinfo        : CPU information
  /proc/meminfo        : Memory information
  /proc/loadavg        : Load average
  /proc/version        : Kernel version
  /proc/<pid>/status   : Process status
  /proc/<pid>/maps     : Memory mappings
  /proc/<pid>/fd/      : File descriptors
  /proc/sys/           : Kernel parameters (sysctl)

sysfs:
  Exposes device and driver information in a tree structure
  Mount point: /sys

  Structure:
  /sys/class/           : Device classes
  /sys/block/           : Block devices
  /sys/devices/         : Physical device hierarchy
  /sys/bus/             : Bus types
  /sys/module/          : Loaded modules
  /sys/fs/              : Filesystem-specific information

debugfs:
  Exposes debug information
  Mount point: /sys/kernel/debug
  -> Used by debug tools such as ftrace, tracing

devtmpfs:
  Automatic creation of device files
  Mount point: /dev
  -> Works with udev to automatically manage device nodes

cgroupfs:
  Control group management
  -> Resource limiting for CPU, memory, I/O
  -> Foundation technology for containers (Docker, systemd)

overlayfs:
  Overlays multiple directories
  -> Implements Docker image layers
  -> lower (read-only) + upper (read-write) = merged (unified view)
```

### 5.5 Network Filesystems

```
NFS (Network File System):
  Standard network sharing for Unix/Linux
  - NFSv3: Stateless (easy recovery)
  - NFSv4: Stateful (improved locking, firewall traversal)
  - NFSv4.1: pNFS (parallel NFS, distributed storage support)
  - NFSv4.2: Server-side copy, hole detection

  # Server configuration (/etc/exports)
  /data  192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)

  # Mount from client
  mount -t nfs server:/data /mnt/nfs

SMB/CIFS (Server Message Block):
  Standard network sharing for Windows
  - Samba provides this on Linux servers
  - SMB1 (CIFS): Legacy, security issues (use discouraged)
  - SMB2: Windows Vista and later
  - SMB3: Encryption, multichannel (Windows 8 / Server 2012)

  # Mount from Linux client
  mount -t cifs //server/share /mnt/smb -o user=username

GlusterFS:
  Distributed filesystem
  -> Combines multiple servers into a single large-capacity filesystem
  -> Replication, striping, distribution

CephFS:
  Distributed filesystem (part of Ceph storage)
  -> POSIX compatible
  -> Highly scalable
  -> Used with OpenStack, Kubernetes

FUSE (Filesystem in Userspace):
  Framework for implementing filesystems in user space
  -> SSHFS: File access over SSH
  -> S3FS: Mount Amazon S3 as a filesystem
  -> NTFS-3G: NTFS read-write support
  -> rclone mount: Mount cloud storage

  # SSHFS example
  sshfs user@server:/remote/path /mnt/sshfs

  # S3FS example
  s3fs mybucket /mnt/s3 -o passwd_file=~/.passwd-s3fs
```

---

## 6. Filesystem Selection Guide

### 6.1 Recommendations by Use Case

```
Filesystem selection by use case:

+-------------------------+----------+------------------------+
| Use Case                | Rec. FS  | Reason                 |
+-------------------------+----------+------------------------+
| Desktop (general)       | ext4     | Stable, compatible,    |
|                         |          | rich tooling           |
| Desktop (Fedora)        | Btrfs    | Snapshot-based restore |
| Enterprise server       | XFS      | High parallelism,      |
|                         |          | large-scale, RHEL std  |
| Database server         | XFS/ext4 | High I/O performance,  |
|                         |          | stability              |
| NAS (home/small-scale)  | Btrfs    | Built-in RAID,         |
|                         |          | snapshots              |
| NAS (enterprise)        | ZFS      | Best data protection   |
| USB drive (shared)      | exFAT    | All-OS compatible,     |
|                         |          | large file support     |
| SD card (32GB or less)  | FAT32    | Maximum compatibility  |
| SD card (64GB or more)  | exFAT    | SDXC standard          |
| CI/CD build directory   | tmpfs    | Ultra-fast on RAM      |
| Container (Docker)      | overlay2 | Best for layer mgmt    |
| VM storage              | XFS+     | Fast cloning           |
|                         | reflink  |                        |
| Backup server           | ZFS/     | Snapshots +            |
|                         | Btrfs    | compression            |
| Media server            | XFS      | Fast transfer of       |
|                         |          | large files            |
| Embedded system         | SquashFS+| Read-only + writable   |
|                         | overlayfs| area                   |
| Boot partition          | ext4/    | GRUB/UEFI compatible   |
|                         | FAT32    |                        |
+-------------------------+----------+------------------------+
```

### 6.2 Comprehensive Comparison Table

```
Detailed comparison of major filesystems:

+--------------+----------+----------+----------+----------+
| Property     | ext4     | XFS      | Btrfs    | ZFS      |
+--------------+----------+----------+----------+----------+
| Max FS       | 1EB      | 8EB      | 16EB     | 256ZiB   |
| Max file     | 16TB     | 8EB      | 16EB     | 16EB     |
| CoW          | No       | No       | Yes      | Yes      |
|              |          | (reflink)|          |          |
| Compression  | No       | No       | Yes      | Yes      |
| Snapshots    | No       | No       | Yes      | Yes      |
| RAID         | No       | No       | Yes      | Yes      |
| Checksums    | meta only| meta only| data+meta| data+meta|
| Shrinking    | Yes      | No       | Yes      | No       |
| reflink      | No       | Yes(4.9+)| Yes      | No       |
| Dedup        | No       | No       | offline  | Yes      |
| Encryption   | fscrypt  | No       | No       | Yes      |
| Quotas       | Yes      | Yes(prj) | Yes      | Yes      |
|              |          |          | (qgroup) |          |
| Stability    | Excellent| Excellent| Good     | Excellent|
| Speed (seq)  | Good     | Excellent| Good     | Good     |
| Speed (rand) | Good     | Excellent| Fair     | Good     |
| Memory usage | Low      | Low      | Medium   | High     |
| Tooling      | Excellent| Good     | Good     | Good     |
| Linux integ. | Excellent| Excellent| Excellent| Fair     |
|              |          |          |          | (DKMS)   |
+--------------+----------+----------+----------+----------+
```

---

## Practical Exercises

### Exercise 1: [Basic] -- Checking Filesystem Information

```bash
# Check current filesystems
df -Th
lsblk -f
cat /etc/fstab

# Detailed mount point information
findmnt --real                # Show only physical filesystems
findmnt -t ext4,xfs,btrfs    # Show specific types only

# Filesystem-specific information
# ext4
sudo dumpe2fs -h /dev/sda1
sudo tune2fs -l /dev/sda1

# XFS
xfs_info /mount/point

# Btrfs
btrfs filesystem show
btrfs filesystem df /mount/point
btrfs filesystem usage /mount/point

# ZFS
zpool status
zfs list
```

### Exercise 2: [Applied] -- Filesystem Selection Decisions

```
Select the optimal filesystem for the following use cases
and explain your reasoning:

1. 100TB NAS storage (regular backup required)
   -> ZFS: Checksums, RAID-Z2/Z3, snapshots,
     efficient backup via send/receive

2. High-frequency DB transactions (MySQL/PostgreSQL)
   -> XFS: High parallel I/O performance, B+ tree metadata
     management, stable performance, RHEL recommended

3. USB drive (shared between Windows/Mac/Linux)
   -> exFAT: All-OS support, supports files over 4GB,
     SDXC card standard compliant

4. CI/CD temporary build directory
   -> tmpfs: Fastest on RAM, auto-clean on reboot,
     eliminates disk I/O bottleneck

5. Developer Linux desktop
   -> Btrfs: Save state before config changes via snapshots,
     SSD space savings via compression, Fedora default

6. Container host storage
   -> XFS + overlay2: Docker recommended configuration,
     efficient image layer management with reflink
```

### Exercise 3: [Advanced] -- Filesystem Benchmark Comparison

```bash
# Prepare test environment (create each FS on loopback devices)
for fs in ext4 xfs btrfs; do
  dd if=/dev/zero of=/tmp/test_${fs}.img bs=1M count=2048
  case $fs in
    ext4)  mkfs.ext4 -F /tmp/test_${fs}.img ;;
    xfs)   mkfs.xfs -f /tmp/test_${fs}.img ;;
    btrfs) mkfs.btrfs -f /tmp/test_${fs}.img ;;
  esac
  mkdir -p /mnt/test_${fs}
  mount -o loop /tmp/test_${fs}.img /mnt/test_${fs}
done

# Benchmark with fio (run on each filesystem)
for fs in ext4 xfs btrfs; do
  echo "=== ${fs} ==="
  fio --name=${fs}_seqwrite \
    --directory=/mnt/test_${fs} \
    --rw=write --bs=4k --size=512M \
    --numjobs=4 --runtime=30 --time_based \
    --group_reporting
done

# Cleanup
for fs in ext4 xfs btrfs; do
  umount /mnt/test_${fs}
  rm /tmp/test_${fs}.img
done
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Check path and format of configuration files |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Data volume increase | Introduce batch processing, implement pagination |
| Permission error | Insufficient access permissions | Check execution user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, transaction management |

### Debugging Procedure

1. **Check error messages**: Read stack traces and identify the location
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use log output and debuggers to test hypotheses
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

Diagnostic procedure when performance issues occur:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check disk and network I/O status
4. **Check concurrent connections**: Check connection pool status

| Problem Type | Diagnostic Tool | Solution |
|-------------|----------------|----------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the decision criteria for making technology choices.

| Criteria | When Prioritized | When Acceptable to Compromise |
|---------|-----------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin screens, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Growing services | Internal tools, fixed users |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+-----------------------------------------------------+
|              Architecture Selection Flow              |
+-----------------------------------------------------+
|                                                       |
|  (1) Team size?                                       |
|    +-- Small (1-5) -> Monolith                        |
|    +-- Large (10+) -> Go to (2)                       |
|                                                       |
|  (2) Deployment frequency?                            |
|    +-- Once a week or less -> Monolith + module split |
|    +-- Daily/multiple times -> Go to (3)              |
|                                                       |
|  (3) Inter-team independence?                         |
|    +-- High -> Microservices                          |
|    +-- Moderate -> Modular Monolith                   |
|                                                       |
+-----------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs. Long-term Cost**
- Methods that are fast in the short term may become technical debt in the long term
- Conversely, over-engineering has high short-term costs and delays the project

**2. Consistency vs. Flexibility**
- A unified tech stack has lower learning costs
- Adopting diverse technologies enables the right tool for the job but increases operational costs

**3. Level of Abstraction**
- High abstraction increases reusability but can make debugging difficult
- Low abstraction is intuitive but prone to code duplication

```python
# Design Decision Record Template
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


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend firmly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| FS | Characteristics | Use Cases |
|----|----------------|-----------|
| ext4 | Stable, general-purpose, most widely used | Desktops, general servers |
| XFS | Large files, high parallelism, B+ tree | DB, media, RHEL |
| Btrfs | CoW, snapshots, compression, RAID | NAS, backups, Fedora/SUSE |
| ZFS | Best data protection, pool management | Enterprise NAS, backups |
| NTFS | Windows standard, ACL | Windows environments |
| APFS | SSD optimized, encryption | macOS, iOS |
| exFAT | All-OS compatible, large file support | USB drives, SD cards |
| tmpfs | On RAM, ultra-fast | Temp files, build caches |

---

## Recommended Next Reading

---

## References
1. Carrier, B. "File System Forensic Analysis." Addison-Wesley, 2005.
2. McDougall, R. & Mauro, J. "Solaris Internals." 2nd Ed, Prentice Hall, 2006.
3. Rodeh, O. et al. "BTRFS: The Linux B-Tree Filesystem." ACM TOS, 2013.
4. Bonwick, J. & Moore, B. "ZFS: The Last Word in File Systems." Sun Microsystems, 2007.
5. Sweeney, A. et al. "Scalability in the XFS File System." USENIX ATC, 1996.
6. Mathur, A. et al. "The New ext4 filesystem." Ottawa Linux Symposium, 2007.
7. Lucas, M. W. "FreeBSD Mastery: ZFS." Tilted Windmill Press, 2015.
8. OpenZFS Documentation. https://openzfs.github.io/openzfs-docs/
9. Btrfs Wiki. https://btrfs.wiki.kernel.org/
10. XFS Documentation. https://xfs.wiki.kernel.org/
