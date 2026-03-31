# File System Fundamentals

> A file system is a mechanism that "converts a sequence of bytes on a disk into a hierarchical structure of files and directories that humans can understand."

## What You Will Learn in This Chapter

- [ ] Understand the basic structure of file systems
- [ ] Be able to explain how inodes and directories work
- [ ] Understand the need for journaling
- [ ] Understand the mechanism and importance of VFS
- [ ] Grasp file system consistency maintenance mechanisms
- [ ] Become proficient in practical file system operations


## Prerequisites

Having the following knowledge will deepen your understanding before reading this guide:

- Basic programming knowledge
- Understanding of related fundamental concepts

---

## 1. File System Structure

### 1.1 Converting Physical Structure to Logical Structure

```
Physical structure of disk -> Logical structure of file system:

  Physical: Contiguous sectors (512B/4KB)
  +--+--+--+--+--+--+--+--+--+--+
  |S0|S1|S2|S3|S4|S5|S6|S7|S8|S9|...
  +--+--+--+--+--+--+--+--+--+--+

  Logical: Hierarchical files and directories
  /
  +-- home/
  |   +-- user/
  |       +-- document.txt
  |       +-- photo.jpg
  +-- etc/
  |   +-- config.yaml
  +-- var/
      +-- log/
          +-- syslog
          +-- auth.log
```

The fundamental role of a file system is to convert the sequence of sectors on a disk into a hierarchical structure of files and directories that humans can intuitively work with. This conversion is achieved through multiple layers.

```
Conversion layer structure:

  User space:  open("/home/user/doc.txt", O_RDONLY)
      |
      v
  VFS layer:   Path name resolution -> dentry cache lookup
      |
      v
  FS-specific:  ext4_lookup() -> inode retrieval
      |
      v
  Block layer:  Block number -> sector number conversion
      |
      v
  Device layer: I/O request issuance -> disk controller
      |
      v
  Physical:     Head movement -> sector read (HDD)
                NAND flash access (SSD)
```

### 1.2 Relationship Between Blocks and Sectors

```
Sector:
  - Smallest physical unit of a disk
  - Traditionally 512 bytes
  - Advanced Format (AF) drives use 4096 bytes (4Kn)
  - 512e drives: physical 4KB, logical 512B emulation

Block:
  - Smallest logical unit of a file system
  - Typically 1KB to 4KB (ext4 default is 4KB)
  - 1 block = N sectors

  Block size selection:
  +----------+----------------------------------+
  | Small    | Less internal fragmentation      |
  | block    | Higher metadata overhead         |
  | (1KB)    | Advantageous for many small files |
  +----------+----------------------------------+
  | Large    | More internal fragmentation      |
  | block    | Lower metadata overhead          |
  | (4KB)    | Better transfer efficiency for   |
  |          | large files                      |
  +----------+----------------------------------+

  Example: Even a 1-byte file consumes 1 block (4KB)
  -> Internal fragmentation

  Check block size:
  $ tune2fs -l /dev/sda1 | grep "Block size"
  Block size:               4096

  $ stat -f / | grep "Block size"  # macOS
```

### 1.3 Basic Layout of ext4

```
ext4 disk layout:

  +------+----------+----------+----------+------+
  |Boot  | Block    | inode    | inode    |Data  |
  |Block | Group    | Table    | Bitmap   |Blocks|
  |      |Descriptor|          | + Block  |      |
  |      |          |          | Bitmap   |      |
  +------+----------+----------+----------+------+

  Detailed block group structure:
  +-----------------------------------------------------+
  |                    Block Group 0                      |
  +------+------+------+------+------+-----------------+
  |Super |Group |Block |inode |inode |    Data         |
  |block |Desc. |Bitmap|Bitmap|Table |    Blocks       |
  |(1blk)|(Nblk)|(1blk)|(1blk)|(Nblk)|  (all remaining)|
  +------+------+------+------+------+-----------------+

  Block Groups 1, 2, ... have the same structure
  (Superblock is stored in some groups as backup)
```

**Superblock** is the most critical structure that holds metadata for the entire file system.

```
Major fields of the Superblock:

  s_inodes_count       : Total number of inodes
  s_blocks_count       : Total number of blocks
  s_free_blocks_count  : Number of free blocks
  s_free_inodes_count  : Number of free inodes
  s_log_block_size     : Block size (expressed as a power of 2)
  s_blocks_per_group   : Number of blocks per block group
  s_inodes_per_group   : Number of inodes per block group
  s_magic              : File system identifier (ext4 = 0xEF53)
  s_state              : File system state (clean/error)
  s_feature_compat     : Compatible feature flags
  s_feature_incompat   : Incompatible feature flags
  s_feature_ro_compat  : Read-only compatible feature flags
  s_uuid               : File system UUID
  s_volume_name        : Volume name
  s_last_mounted       : Last mounted path

Superblock backups:
  - Stored in block groups 0, 1, 3, 5, 7, 9, 25, 27, ...
  - Stored in groups that are powers of 3, 5, 7 (sparse superblock)
  - Recovery is possible even if the main Superblock is corrupted

  Recovery example:
  $ sudo mke2fs -n /dev/sda1  # Check backup locations
  Superblock backups stored on blocks:
    32768, 98304, 163840, 229376, ...

  $ sudo e2fsck -b 32768 /dev/sda1  # Repair from backup
```

### 1.4 Block Group Descriptor

```
Block Group Descriptor (BGD) contents:

  bg_block_bitmap      : Location of block bitmap
  bg_inode_bitmap      : Location of inode bitmap
  bg_inode_table       : Start position of inode table
  bg_free_blocks_count : Number of free blocks in group
  bg_free_inodes_count : Number of free inodes in group
  bg_used_dirs_count   : Number of directories in group

Bitmap mechanism:
  Block bitmap:
  Each bit corresponds to one block (in use=1, free=0)

  Example: Bitmap for 8 blocks
  +-+-+-+-+-+-+-+-+
  |1|1|0|1|0|0|1|0|
  +-+-+-+-+-+-+-+-+
   B0 B1 B2 B3 B4 B5 B6 B7
   U  U  F  U  F  F  U  F
   (U=Used, F=Free)

  Number of blocks manageable with 1 block (4KB) bitmap:
  4096 x 8 = 32768 blocks
  32768 x 4KB = 128MB

  -> One block group can manage up to 128MB
```

### 1.5 Flex Block Groups (ext4 Extension)

```
Flex Block Groups:
  Consolidates metadata from multiple block groups into one location
  -> Improves metadata locality and reduces seek count

  Standard layout:
  +--------++--------++--------++--------+
  |metadata||metadata||metadata||metadata|
  | + data || + data || + data || + data |
  +--------++--------++--------++--------+
   BG0       BG1       BG2       BG3

  Flex Block Groups:
  +------------------++------++------++------+
  | meta0+meta1+     || data || data || data |
  | meta2+meta3      ||      ||      ||      |
  +------------------++------++------++------+
   Flex BG 0 (consolidated metadata)   Data area

  -> Faster metadata reads
  -> Effective for large file systems
```

---

## 2. Inodes and Directories

### 2.1 Inode Structure

```
inode (index node):
  A structure that stores file metadata
  * File name is NOT included! Names are managed by directories

  +----------------------------------+
  | inode #12345                     |
  +----------------------------------+
  | File type: regular file          |
  | Permissions: rwxr-xr-x          |
  | Owner: uid=1000                  |
  | Group: gid=1000                  |
  | Size: 4096 bytes                 |
  | Timestamps:                      |
  |   atime: last access time        |
  |   mtime: last modification time  |
  |   ctime: metadata change time    |
  |   crtime: creation time (ext4)   |
  | Link count: 1                    |
  | Block count: 8 (in 512B units)   |
  | Flags: 0x80000 (extents used)    |
  | Data block pointers:             |
  |   Direct: [B1][B2]...[B12]      |
  |   Indirect: [-> block group]     |
  |   Double indirect: [->> blocks]  |
  |   Triple indirect: [--->> ...]   |
  +----------------------------------+
```

### 2.2 Data Block Pointer Mechanism

```
Traditional block pointer scheme (ext2/ext3):

  inode data pointers:
  +--------------+
  | Direct       | x12 -> 12 x 4KB = 48KB
  | pointers     |
  | [0]-[11]     |
  +--------------+
  | Indirect     | x1 -> 1024 x 4KB = 4MB
  | pointer      |
  | [12]         |
  +--------------+
  | Double       | x1 -> 1024^2 x 4KB = 4GB
  | indirect     |
  | [13]         |
  +--------------+
  | Triple       | x1 -> 1024^3 x 4KB = 4TB
  | indirect     |
  | [14]         |
  +--------------+

  Expansion of indirect pointers:
  inode[12] -> +------+
               | B100 |
               | B101 |
               | B102 |
               | ...  | (1024 entries)
               | B1123|
               +------+

  Problems:
  - Multi-level references needed for large files
  - Individual pointers maintained even for contiguous blocks -> metadata bloat
  - Example: 1GB contiguous file -> 262,144 block pointers needed
```

### 2.3 Extents (ext4)

```
ext4 extents:
  Represent contiguous blocks as "start block + block count"
  -> Significant metadata reduction

  Extent structure:
  +--------------------------------+
  | ee_block: logical block number  |
  | ee_len:   block count           |
  | ee_start: physical block number |
  +--------------------------------+

  Example: 1GB contiguous file
  Traditional: 262,144 block pointers
  Extents: 1 extent (start + length)

  Extent tree:
  +------------------+
  | inode            |
  | +--------------+ |
  | | Header       | |
  | | Extent 1     | | -> Blocks 0-99   -> Physical 1000-1099
  | | Extent 2     | | -> Blocks 100-299 -> Physical 2000-2199
  | | Extent 3     | | -> Blocks 300-599 -> Physical 5000-5299
  | | Index        | | -> Additional tree nodes
  | +--------------+ |
  +------------------+

  Up to 4 extents can be stored directly in the inode
  -> Beyond that, managed with a B-tree structure (extent tree)

  Advantages:
  - Efficient representation of contiguous allocation
  - Significant metadata reduction
  - Faster processing of large files
  - Especially effective when fragmentation is low
```

### 2.4 Directory Implementation

```
Directory:
  A mapping table between file names and inode numbers

  Directory entries for /home/user/:
  +--------------+----------+----------+----------+
  | File name    | inode #  | Entry len| Type     |
  +--------------+----------+----------+----------+
  | .            | 201      | 12       | DIR      |
  | ..           | 100      | 12       | DIR      |
  | document.txt | 12345    | 20       | REG      |
  | photo.jpg    | 12346    | 16       | REG      |
  | scripts/     | 12400    | 16       | DIR      |
  +--------------+----------+----------+----------+

  ext4 directory implementation methods:

  1. Linear directory:
     - Entries stored sequentially
     - For small directories
     - Lookup: O(n)

  2. HTree (Hash Tree) directory:
     - B-tree constructed from file name hashes
     - For large directories (tens of thousands of files or more)
     - Lookup: O(log n)
     - Enabled by default

  HTree structure:
  +-----------------------------+
  | Root node                   |
  | hash < 0x4000 -> block 5   |
  | hash < 0x8000 -> block 12  |
  | hash < 0xC000 -> block 18  |
  | hash >= 0xC000 -> block 25 |
  +-----+-------+-------+-----+
        v       v       v
    +------++------++------+
    |entry ||entry ||entry |
    |list  ||list  ||list  |
    +------++------++------+
```

### 2.5 Path Name Resolution

```
Processing of open("/home/user/document.txt"):

  Step 1: Resolve "/"
  -> Get the root directory inode (usually inode 2)
  -> Check dentry cache

  Step 2: Resolve "home"
  -> Search directory entries from "/" inode
  -> Get inode number corresponding to "home" (e.g., 100)
  -> Permission check on inode 100 (x bit)

  Step 3: Resolve "user"
  -> Search for "user" in directory entries of inode 100
  -> Get inode 201
  -> Permission check

  Step 4: Resolve "document.txt"
  -> Search for "document.txt" in directory entries of inode 201
  -> Get inode 12345
  -> Permission check (r bit)

  Step 5: Open file
  -> Load inode 12345 into memory
  -> Assign a file descriptor
  -> Create and return a file structure

  Important: Permission checks are performed at each step
  -> Directory "x" (execute) bit is required
  -> Without "x", files within the directory cannot be accessed

  Path name resolution optimization:
  - dentry cache: Keeps resolved path->inode mappings in memory
  - Negative dentry: Records non-existent paths (avoids unnecessary lookups)
  - RCU (Read-Copy-Update): Lock-free concurrent access
```

### 2.6 Hard Links and Symbolic Links

```
Hard link:
  Another name pointing to the same inode

  $ echo "hello" > original.txt   # inode 12345
  $ ln original.txt link.txt       # inode 12345 (same)

  +--------------+     +--------------+
  | original.txt |---->|  inode       |
  +--------------+     |  #12345      |
  +--------------+     |  links: 2    |
  | link.txt     |---->|  data: ...   |
  +--------------+     +--------------+

  Characteristics:
  - Same inode number
  - Link count increases
  - Same data accessed from either name
  - Data persists even if one is deleted (while link count > 0)
  - Data blocks freed when link count reaches 0

  Constraints:
  - Hard links to directories are not allowed (to prevent loops)
    -> "." and ".." are exceptions (managed by the kernel)
  - Cannot cross partition/file system boundaries
    -> Inodes are unique only within a file system

Symbolic link (symlink):
  A special file that stores a path

  $ ln -s /path/to/original symlink

  +--------------+     +--------------+     +--------------+
  | symlink      |---->| inode #99999 |     | inode #12345 |
  +--------------+     | type: LINK   |     | type: REG    |
                       | data:        |---->| data: ...    |
                       |"/path/to/    |     +--------------+
                       | original"    |
                       +--------------+

  Characteristics:
  - Has a separate inode
  - Stores a path string (stored directly in the inode if 60 bytes or less)
  - Can link to directories
  - Can cross partition boundaries
  - Becomes a dangling link (broken link) if the target is deleted

  Fast symlink (ext4):
  - If path is 60 bytes or less, stored directly in the data pointer area of the inode
  - No additional disk blocks consumed
  - -> Most symlinks are fast symlinks

  Comparison table:
  +--------------+--------------+--------------+
  |              | Hard link    | Symlink      |
  +--------------+--------------+--------------+
  | inode        | Same         | Different    |
  | Cross FS     | Not possible | Possible     |
  | Directories  | Not possible | Possible     |
  | Target       | Data         | Dangling     |
  | deleted      | persists     | link         |
  | Disk usage   | None         | 1 inode      |
  | Path update  | Not needed   | May be needed|
  +--------------+--------------+--------------+
```

### 2.7 Special Files

```
Special files in Linux:

  1. Device files:
     - Character device (c): /dev/tty, /dev/null
     - Block device (b): /dev/sda, /dev/nvme0n1
     - Identified by major number + minor number

  2. Named pipe (FIFO):
     - mkfifo /tmp/mypipe
     - Used for inter-process communication
     - One side writes, the other reads

  3. UNIX domain socket:
     - Network-style local inter-process communication
     - /var/run/docker.sock, /tmp/.X11-unix/X0

  4. Special virtual files:
     /dev/null   : Discards writes. Reads return immediate EOF
     /dev/zero   : Generates infinite zero bytes
     /dev/random : Cryptographically secure random numbers (blocks on entropy exhaustion)
     /dev/urandom: Non-blocking pseudo-random numbers
     /dev/full   : Returns ENOSPC error on writes (for testing)

  Check file types:
  $ ls -la /dev/null /dev/sda /tmp/mypipe
  crw-rw-rw- 1 root root 1, 3 ... /dev/null     # c=character
  brw-rw---- 1 root disk 8, 0 ... /dev/sda       # b=block
  prw-r--r-- 1 user user 0    ... /tmp/mypipe    # p=pipe
  srwxrwxrwx 1 root root 0    ... /var/run/docker.sock  # s=socket

  $ stat --format '%F' /dev/null
  character special file
```

---

## 3. Journaling

### 3.1 Crash Consistency Problem

```
Problem: What happens if a power failure occurs during a write?

  Steps to create a file:
  1. Find and allocate a free inode in the inode bitmap
  2. Write metadata to the inode
  3. Add an entry to the directory
  4. Allocate a free block in the block bitmap
  5. Write data to the data block

  Inconsistency patterns when power failure occurs:

  Case 1: Power failure after step 2
  -> Inode is allocated but no directory entry exists
  -> Inode exists but is inaccessible = orphan inode
  -> Inode leak (in use but not referenced)

  Case 2: Power failure after step 3
  -> Directory entry exists but no data
  -> File is visible but contains garbage data

  Case 3: Power failure after step 4
  -> Block is allocated but data is not written
  -> Previous file's data may be visible (security risk)

  Case 4: Power failure during file append
  -> Inode size update and data write are inconsistent
  -> Garbage data at the end of the file

  Means to solve these problems:
  1. fsck (File System Check): Inspects all data at boot time
     -> Can take hours to tens of hours for large disks
  2. Journaling: Records changes to a log first
     -> Only need to check the journal at boot (seconds)
  3. CoW: Writes data to a new location instead of overwriting
     -> Guarantees crash consistency by design
```

### 3.2 How Journaling Works

```
Journaling:
  Write changes to the "journal (log)" first

  +----------------------------------------------+
  | 1. Begin transaction                          |
  |    -> Create a log record describing changes  |
  | 2. Write log to journal                       |
  |    -> Pre-change data + post-change data      |
  | 3. Commit the journal                         |
  |    -> Write the commit block                  |
  | 4. Write to actual data area (checkpoint)     |
  | 5. Invalidate the journal log                 |
  +----------------------------------------------+

  Structure of the journal area:
  +------+------+------+------+------+------+------+
  | Desc | Data | Data |Commit| Desc | Data |Commit|
  | Block| Log1 | Log2 | Block| Block| Log3 | Block|
  +------+------+------+------+------+------+------+
  <---- Transaction 1 ------><-- Transaction 2 -->

  When power failure occurs:
  Case 1: Power failure during journal write
  -> No commit block
  -> Discard entire transaction (do not roll forward)
  -> Data area is unchanged, so consistency is maintained

  Case 2: Power failure after commit, during actual data write
  -> Commit block exists
  -> Replay (re-execute) using journal log
  -> Recover data area to correct state

  Case 3: After normal completion
  -> Journal log is no longer needed and is invalidated
```

### 3.3 Journaling Modes (ext4)

```
ext4 journaling modes:

  1. journal mode:
     - Journals both data + metadata
     - Safest but slowest
     - Data written twice (journal + actual area)
     - Use case: When extremely high reliability is required

  2. ordered mode (default):
     - Journals only metadata
     - Data is written before metadata commit
     - Guarantee: When metadata is updated, data is already written
     - Use case: Optimal balance for the vast majority of use cases

  3. writeback mode:
     - Journals only metadata
     - Does not guarantee data write order
     - Fastest but risk of data loss
     - Use case: Temporary files, regeneratable data

  Performance comparison (relative values):
  +----------+------+------+------+
  | Mode     |Safety| Read | Write|
  +----------+------+------+------+
  | journal  | Best | 100  |  60  |
  | ordered  | Good | 100  |  85  |
  | writeback| Fair | 100  | 100  |
  +----------+------+------+------+

  Configuration:
  # /etc/fstab
  /dev/sda1  /  ext4  data=ordered  0  1

  # Specify at mount time
  $ sudo mount -o data=journal /dev/sda1 /mnt

  # Check current mode
  $ cat /proc/fs/ext4/sda1/options | grep data
  data=ordered

  # Check journal status
  $ sudo dumpe2fs /dev/sda1 | grep -i journal
  Journal inode:            8
  Journal backup:           inode blocks
  Journal features:         journal_incompat_revoke journal_64bit
  Journal size:             128M
  Journal length:           32768
  Journal sequence:         0x000c3a10
```

### 3.4 Checkpointing and Journal Management

```
Checkpoint:
  The process of applying journal logs to the actual data area

  +-----------+     +-----------+     +-----------+
  | App       |     | Journal   |     | Data      |
  | write()   |---->| Log entry |---->| Actual    |
  +-----------+     +-----------+     | write     |
                    <- Fast write ->  +-----------+
                                      <- Background ->

  Journal circular buffer:
  +--------------------------------------+
  |                                      |
  |  +--+--+--+--+--+--+--+--+--+      |
  |  |T1|T2|  |  |T5|T6|T7|  |  |      |
  |  +--+--+--+--+--+--+--+--+--+      |
  |       ^              ^               |
  |    Checkpoint       Latest commit    |
  |    completed        position         |
  |    position                          |
  +--------------------------------------+

  T1, T2: Checkpoint completed -> can be reused as free space
  T5-T7: Not yet applied to actual data

  When journal becomes full:
  -> Block new transactions
  -> Force checkpoint execution
  -> Proper journal size configuration is important

  Recommended journal sizes:
  - Small (< 100GB): 64MB
  - Medium (100GB-1TB): 128MB (default)
  - Large (> 1TB): 256MB-1GB

  $ sudo tune2fs -J size=256 /dev/sda1  # Change journal size
```

### 3.5 Copy-on-Write (CoW) File Systems

```
CoW (Copy-on-Write) file systems:
  Btrfs, ZFS use CoW instead of journaling
  -> Write to a new location instead of overwriting data
  -> Atomic updates, fast snapshots

  How CoW works:
  +----------------------------------------------+
  | Before update:                                |
  |   Root -> Node A -> [Leaf 1] [Leaf 2] [Leaf 3]|
  |                                               |
  | When updating Leaf 2:                         |
  | 1. Create a copy of Leaf 2 in a new location  |
  | 2. Write data to the copy                     |
  | 3. Create a copy of Node A (pointing to new   |
  |    Leaf 2)                                    |
  | 4. Create a copy of Root (pointing to new     |
  |    Node A)                                    |
  | 5. Update Superblock to new Root (atomic op)  |
  |                                               |
  | After update:                                 |
  |   Root' -> Node A' -> [Leaf 1] [Leaf 2'] [Leaf 3]|
  |                                               |
  | Old Root, Node A, Leaf 2 can be freed         |
  | (retained if referenced by a snapshot)        |
  +----------------------------------------------+

  CoW advantages:
  - Crash consistency is guaranteed by design
  - Snapshots can be created instantly (just keep old data)
  - Easy rollback

  CoW disadvantages:
  - Write amplification (even small changes copy entire tree path)
  - Prone to fragmentation (data placed in scattered locations)
  - Random write performance may degrade

  Journaling vs CoW:
  +--------------+-----------------+-----------------+
  |              | Journaling      | CoW             |
  +--------------+-----------------+-----------------+
  | Consistency  | Log-based       | Structural      |
  | guarantee    |                 |                 |
  | Snapshots    | Not supported   | Instant creation|
  | Write        | 2x (log+actual) | Path copy cost  |
  | amplification|                 |                 |
  | Fragmentation| Low             | High            |
  | Implementations| ext4, XFS    | Btrfs, ZFS      |
  +--------------+-----------------+-----------------+
```

---

## 4. VFS (Virtual File System)

### 4.1 VFS Overview

```
VFS: Linux's unified file system interface

  Application
     | open(), read(), write(), close()
     v
  +------------------------------------------+
  | VFS (Virtual File System)                |
  | -> Unified API                           |
  | -> dentry cache                          |
  | -> inode cache                           |
  | -> page cache                            |
  +--+----+----+----+----+----+----+--------+
     v    v    v    v    v    v    v
   ext4  XFS  Btrfs NTFS  NFS  procfs tmpfs
   (Actual file system implementations)

  Purpose of VFS:
  - Applications need not be aware of file system types
  - Same open()/read()/write() can access all file systems
  - Easy to add new file systems (just implement the VFS interface)
  - Seamless data copy between file systems
```

### 4.2 Four Main VFS Objects

```
Main VFS data structures:

  1. struct super_block:
     Information about a mounted file system
     +--------------------------------+
     | s_dev:     Device identifier    |
     | s_type:    File system type     |
     | s_op:      Operations table     |
     | s_flags:   Mount flags          |
     | s_root:    Root dentry          |
     | s_fs_info: FS-specific data     |
     +--------------------------------+

  2. struct inode (inode object):
     Information about an individual file (in-memory representation)
     +--------------------------------+
     | i_ino:     inode number         |
     | i_mode:    Access permissions   |
     | i_uid:     Owner ID             |
     | i_gid:     Group ID             |
     | i_size:    File size            |
     | i_op:      inode operations     |
     | i_fop:     File operations      |
     | i_sb:      Owning superblock    |
     | i_mapping: Page cache           |
     +--------------------------------+

  3. struct dentry (directory entry):
     Information about each component of a path name
     +--------------------------------+
     | d_name:    Name                 |
     | d_inode:   Associated inode     |
     | d_parent:  Parent dentry        |
     | d_subdirs: Child dentry list    |
     | d_op:      dentry operations    |
     | d_flags:   Status flags         |
     +--------------------------------+

     dentry cache:
     - Resolves path names without disk I/O
     - Managed by LRU, shrinks under memory pressure
     - Negative dentry: Also records non-existent paths

  4. struct file (file object):
     State of a file opened by a process
     +--------------------------------+
     | f_path:    Path information     |
     | f_inode:   Associated inode     |
     | f_op:      File operations      |
     | f_pos:     Current offset       |
     | f_flags:   Open flags           |
     | f_mode:    Access mode          |
     | f_count:   Reference count      |
     +--------------------------------+
```

### 4.3 VFS Operations Tables

```c
// File operations table (file_operations):
struct file_operations {
    loff_t (*llseek)(struct file *, loff_t, int);
    ssize_t (*read)(struct file *, char __user *, size_t, loff_t *);
    ssize_t (*write)(struct file *, const char __user *, size_t, loff_t *);
    int (*open)(struct inode *, struct file *);
    int (*release)(struct inode *, struct file *);
    int (*fsync)(struct file *, loff_t, loff_t, int);
    int (*mmap)(struct file *, struct vm_area_struct *);
    // ...
};

// inode operations table (inode_operations):
struct inode_operations {
    struct dentry *(*lookup)(struct inode *, struct dentry *, unsigned int);
    int (*create)(struct user_namespace *, struct inode *, struct dentry *,
                  umode_t, bool);
    int (*link)(struct dentry *, struct inode *, struct dentry *);
    int (*unlink)(struct inode *, struct dentry *);
    int (*symlink)(struct inode *, struct dentry *, const char *);
    int (*mkdir)(struct user_namespace *, struct inode *, struct dentry *,
                 umode_t);
    int (*rmdir)(struct inode *, struct dentry *);
    int (*rename)(struct user_namespace *, struct inode *, struct dentry *,
                  struct inode *, struct dentry *, unsigned int);
    // ...
};

// Each file system provides its own implementation:
// For ext4:
const struct file_operations ext4_file_operations = {
    .llseek    = ext4_llseek,
    .read_iter = ext4_file_read_iter,
    .write_iter = ext4_file_write_iter,
    .open      = ext4_file_open,
    .release   = ext4_release_file,
    .fsync     = ext4_sync_file,
    .mmap      = ext4_file_mmap,
};
```

### 4.4 Mount and Unmount

```
Mount: The operation of attaching a file system to the directory tree

  $ mount /dev/sda1 /mnt

  Before mount:
  /
  +-- home/
  +-- mnt/        <- empty directory
  +-- tmp/

  After mount:
  /
  +-- home/
  +-- mnt/        <- contents of /dev/sda1 are visible
  |   +-- data/
  |   +-- config.txt
  +-- tmp/

  Internal mount processing:
  1. Read the file system superblock
  2. Create struct super_block
  3. Read the root inode
  4. Associate with the mount point dentry
  5. Add mount structure to vfsmount tree

  Mount options:
  +------------+--------------------------------------+
  | Option     | Description                          |
  +------------+--------------------------------------+
  | ro         | Read-only                            |
  | rw         | Read-write                           |
  | noatime    | Do not update access time (perf gain)|
  | relatime   | Conditionally update atime (default) |
  | nosuid     | Ignore SUID/SGID bits                |
  | noexec     | Ignore execute permissions           |
  | nodev      | Ignore device files                  |
  | sync       | Synchronous writes (performance hit) |
  | data=      | Specify journaling mode              |
  | discard    | Issue TRIM/UNMAP commands (for SSDs) |
  | barrier=   | Write barrier control                |
  +------------+--------------------------------------+

  /etc/fstab example:
  # <device>      <mount>  <type>  <options>              <dump> <fsck>
  /dev/sda1       /        ext4    defaults,noatime       0      1
  /dev/sda2       /home    ext4    defaults,nosuid        0      2
  /dev/sdb1       /data    xfs     defaults,nobarrier     0      0
  tmpfs           /tmp     tmpfs   defaults,size=4G       0      0
  UUID=xxxx-yyyy  /boot    ext4    defaults               0      2

  Mounting by UUID (recommended):
  -> Device names can change (/dev/sda -> /dev/sdb)
  -> UUID is unique to the file system and does not change
  $ blkid  # Check UUID
```

### 4.5 File Descriptors

```
File descriptor (FD):
  A reference (integer value) to a file opened by a process

  Standard FDs:
  0: stdin  (standard input)
  1: stdout (standard output)
  2: stderr (standard error)
  3+: Files opened by the user

  Relationship of data structures:
  +-------------------------------------------------+
  | Process task_struct                              |
  | +-------------------+                           |
  | | files_struct       |                           |
  | | +---------------+ |                           |
  | | | fd_array       | |                           |
  | | | [0] -> file A  |-|-->  struct file --> inode  |
  | | | [1] -> file B  |-|-->  struct file --> inode  |
  | | | [2] -> file C  |-|-->  struct file --> inode  |
  | | | [3] -> file D  |-|-->  struct file --> inode  |
  | | +---------------+ |                           |
  | +-------------------+                           |
  +-------------------------------------------------+

  FD sharing during fork():
  The parent process FD table is copied
  -> Same struct file is shared (reference count +1)
  -> File offset is also shared

  FD limits:
  $ ulimit -n              # Check soft limit (typically 1024)
  $ ulimit -Hn             # Check hard limit
  $ cat /proc/sys/fs/file-max  # System-wide limit

  # Change limit
  $ ulimit -n 65536        # Change within session
  # /etc/security/limits.conf for persistent settings
  * soft nofile 65536
  * hard nofile 65536
```

---

## 5. File System Integrity and Maintenance

### 5.1 fsck (File System Check)

```
fsck: File system integrity check and repair tool

  Items checked:
  1. Superblock consistency
  2. Accuracy of block bitmap and inode bitmap
  3. Inode link counts
  4. Directory structure consistency
  5. Detection of orphan inodes (unreferenced inodes)
  6. Detection of invalid block pointers
  7. Detection of duplicate block allocations

  Usage:
  # Warning: Must be run with the filesystem unmounted or read-only!
  $ sudo umount /dev/sda1
  $ sudo fsck /dev/sda1

  # ext4 specific
  $ sudo e2fsck -f /dev/sda1       # Force check
  $ sudo e2fsck -p /dev/sda1       # Automatic repair
  $ sudo e2fsck -y /dev/sda1       # Answer yes to all questions

  # XFS specific
  $ sudo xfs_repair /dev/sda1

  # Btrfs
  $ sudo btrfs check /dev/sda1
  $ sudo btrfs scrub start /mnt    # Online check

  fsck on journaling file systems:
  -> Usually not needed (recovery via journal replay)
  -> Only needed if the journal itself is corrupted
  -> Recovery in seconds even on large disks

  fsck on non-journaling file systems:
  -> May need to be run at every boot
  -> Can take hours to tens of hours on large disks
  -> Was the bane of the ext2 era
```

### 5.2 TRIM and SSD Considerations

```
SSD-specific file system considerations:

  TRIM (UNMAP):
  -> Notifies the SSD of deleted blocks
  -> Improves SSD garbage collection efficiency
  -> Important for maintaining write performance

  Mechanism:
  When a file is deleted:
  1. Traditional: File system marks block as "free"
           -> SSD still considers it "in use"
  2. TRIM: File system notifies SSD "this block is no longer needed"
           -> SSD erases in background (faster next write)

  Configuration:
  # Add discard option to /etc/fstab (continuous TRIM)
  /dev/sda1  /  ext4  defaults,discard  0  1

  # Periodic batch TRIM (recommended, less performance impact)
  $ sudo fstrim /                   # Manual execution
  $ sudo fstrim -v /                # Verbose output

  # Schedule periodic execution with systemd timer
  $ sudo systemctl enable fstrim.timer  # Runs weekly

  # Verify TRIM support
  $ lsblk --discard
  NAME   DISC-ALN DISC-GRAN DISC-MAX DISC-ZERO
  sda           0      512B       2G         0
  nvme0n1       0      512B       2T         0

  SSD alignment:
  -> Align partition start to physical page size
  -> Modern tools (fdisk, parted) handle this by default
  -> Improper alignment causes performance degradation

  $ sudo parted /dev/sda align-check optimal 1
  1 aligned

  noatime recommendation:
  -> Updating atime on every file read = unnecessary writes
  -> Impacts SSD lifespan
  -> noatime or relatime recommended
```

### 5.3 File System Defragmentation

```
Fragmentation:

  External fragmentation:
  File blocks placed non-contiguously
  -> HDD: Increased seek time -> performance degradation
  -> SSD: Impact is small but not completely zero

  Fragmentation example:
  Block layout:
  +--+--+--+--+--+--+--+--+--+--+
  |A1|B1|A2|C1|A3|B2|C2|A4|B3|C3|
  +--+--+--+--+--+--+--+--+--+--+
  File A: blocks 0, 2, 4, 7 -> fragmented
  File B: blocks 1, 5, 8 -> fragmented
  File C: blocks 3, 6, 9 -> fragmented

  After defragmentation:
  +--+--+--+--+--+--+--+--+--+--+
  |A1|A2|A3|A4|B1|B2|B3|C1|C2|C3|
  +--+--+--+--+--+--+--+--+--+--+
  -> Contiguous placement improves read performance

  Defragmentation tools for each FS:
  # ext4
  $ sudo e4defrag /path/to/file    # Specific file
  $ sudo e4defrag /mount/point     # Entire mount point
  $ sudo e4defrag -c /mount/point  # Check fragmentation status

  # XFS
  $ sudo xfs_fsr /dev/sda1         # Online defragmentation
  $ sudo xfs_db -r /dev/sda1       # Check fragmentation status

  # Btrfs
  $ sudo btrfs filesystem defragment /path  # Online defragmentation
  $ sudo btrfs filesystem defragment -r /mount  # Recursive

  Techniques to prevent fragmentation:
  - Delayed allocation (ext4's delalloc)
  - Preallocation (fallocate)
  - Maintaining sufficient free space (10% or more recommended)
```

---

## 6. Advanced File System Concepts

### 6.1 Extended Attributes (xattr)

```
Extended attributes:
  Store additional metadata beyond normal permissions

  Namespaces:
  - user.*    : User-defined attributes
  - system.*  : System use (ACLs, etc.)
  - security.*: Security module use (SELinux, etc.)
  - trusted.* : For privileged processes

  Operation commands:
  # Set an attribute
  $ setfattr -n user.description -v "Important document" file.txt

  # Get an attribute
  $ getfattr -n user.description file.txt
  # file: file.txt
  user.description="Important document"

  # List all attributes
  $ getfattr -d file.txt

  # Delete an attribute
  $ setfattr -x user.description file.txt

  # SELinux context (security namespace)
  $ ls -Z file.txt
  unconfined_u:object_r:user_home_t:s0 file.txt

  ACL (Access Control List):
  -> POSIX ACLs are stored as extended attributes
  -> system.posix_acl_access, system.posix_acl_default

  # Set ACL
  $ setfacl -m u:john:rw file.txt   # Grant rw permission to john
  $ setfacl -m g:dev:rx dir/        # Grant rx permission to dev group
  $ getfacl file.txt                # Check ACL
```

### 6.2 Quotas

```
Disk quotas:
  Disk usage limits per user/group

  Types of quotas:
  - Block quota: Limit on usage capacity
  - inode quota: Limit on number of files
  - Soft limit: Can be exceeded within a grace period
  - Hard limit: Absolute upper limit that cannot be exceeded

  Setup procedure:
  # 1. Enable quotas in mount options
  $ sudo mount -o remount,usrquota,grpquota /home

  # 2. Create quota files
  $ sudo quotacheck -cum /home     # User quotas
  $ sudo quotacheck -cgm /home     # Group quotas

  # 3. Enable quotas
  $ sudo quotaon /home

  # 4. Set user quotas
  $ sudo edquota -u username
  # Soft limit: 5GB, Hard limit: 10GB

  # 5. Check usage
  $ sudo repquota -a               # Usage for all users
  $ quota -u username               # Usage for specific user

  ext4 project quotas (per-directory):
  # /etc/projects
  1:/home/project_a
  2:/home/project_b

  # /etc/projid
  project_a:1
  project_b:2

  $ sudo tune2fs -O project /dev/sda1  # Enable project feature
  $ sudo mount -o prjquota /dev/sda1 /home
```

### 6.3 Sparse Files and Holes

```
Sparse files:
  Only blocks with actual data consume disk space
  -> Logical size > Physical size

  Example: Creating a 1TB sparse file
  $ truncate -s 1T sparse_file.img
  $ ls -lh sparse_file.img
  -rw-r--r-- 1 user user 1.0T ... sparse_file.img  # logical 1TB
  $ du -h sparse_file.img
  0       sparse_file.img                            # physical 0

  Hole mechanism:
  +--------------------------------------+
  | Logical blocks:                      |
  | [0] [1] [2] [3] [4] [5] [6] [7]    |
  |  v       v               v          |
  |  data    data             data       |
  |          v                           |
  | Blocks 3-5 are holes (unallocated)  |
  | -> Reading returns 0x00             |
  | -> No disk blocks consumed          |
  +--------------------------------------+

  Detecting and operating on holes:
  # Detect holes using SEEK_HOLE / SEEK_DATA
  $ python3 -c "
  import os
  fd = os.open('sparse_file.img', os.O_RDONLY)
  hole = os.lseek(fd, 0, os.SEEK_HOLE)
  print(f'First hole at: {hole}')
  os.close(fd)
  "

  # Efficiently copy sparse files with cp
  $ cp --sparse=always source dest

  # Efficiently archive sparse files with tar
  $ tar -cSf archive.tar sparse_file.img

  Use cases:
  - Virtual machine disk images (qcow2, VMDK)
  - Database preallocation
  - Core dump files
```

### 6.4 Memory-Mapped Files (mmap)

```
mmap:
  Map file contents directly to virtual memory space

  Normal read/write:
  App -> read() -> kernel -> page cache -> disk
  Data is copied from kernel space -> user space

  mmap:
  App's virtual address directly points to the page cache
  -> No copy needed -> fast

  +--------------------------------------+
  | Process virtual address space        |
  |                                      |
  | +----------------+                   |
  | | text segment   |                   |
  | +----------------+                   |
  | | data segment   |                   |
  | +----------------+                   |
  | | mmap region    |---> page cache -> disk
  | | (file content) |                   |
  | +----------------+                   |
  | | heap           |                   |
  | +----------------+                   |
  | | stack          |                   |
  | +----------------+                   |
  +--------------------------------------+

  Flags:
  - MAP_SHARED:  Changes are reflected to the file
  - MAP_PRIVATE: CoW. Changes are process-local only
  - MAP_ANONYMOUS: No file (used for memory allocation)

  Use cases:
  - Efficient read/write of large files
  - Inter-process shared memory
  - Loading executable files (text segment)
  - Database buffer management

  Caveats:
  - Writing beyond file size causes SIGBUS
  - Address space limitations on 32-bit environments
  - msync() for explicit sync to disk
  - munmap() to release the mapping
```

---

## Practical Exercises

### Exercise 1: [Basic] -- Examining Inodes

```bash
# Examine inodes
ls -li                        # Display inode numbers
stat filename                 # Detailed metadata
df -i                         # inode usage

# Hard links and symbolic links
echo "hello" > original.txt
ln original.txt hardlink.txt
ln -s original.txt symlink.txt
ls -li original.txt hardlink.txt symlink.txt
# -> hardlink has same inode, symlink has a different inode

# Check link count
stat original.txt | grep Links
# Links: 2  <- because there is a hard link

# Check symbolic link target
readlink symlink.txt
readlink -f symlink.txt       # Display as absolute path

# Check for dangling links
rm original.txt
cat symlink.txt               # Error (target does not exist)
cat hardlink.txt              # Reads normally (data still exists)
```

### Exercise 2: [Intermediate] -- Investigating File Systems

```bash
# Check mounted file systems
mount | column -t
df -Th                        # Display with type
findmnt                       # Tree display
findmnt -t ext4               # Show only ext4

# Detailed file system information (Linux, ext4)
sudo dumpe2fs /dev/sda1 | head -30
sudo tune2fs -l /dev/sda1     # Superblock information

# Block group information
sudo dumpe2fs /dev/sda1 | grep -A 5 "Group 0"

# Journal information
sudo dumpe2fs /dev/sda1 | grep -i journal

# inode usage
df -i /                       # inode utilization
for dir in /*; do echo "$(find "$dir" -xdev 2>/dev/null | wc -l) $dir"; done | sort -rn | head

# Check block size
sudo tune2fs -l /dev/sda1 | grep "Block size"
stat -f /                     # File system information
```

### Exercise 3: [Intermediate] -- Creating and Mounting a File System

```bash
# Create a test loopback file system
# 1. Create a file
dd if=/dev/zero of=/tmp/testfs.img bs=1M count=100

# 2. Create ext4 file system
mkfs.ext4 /tmp/testfs.img

# 3. Mount
sudo mkdir -p /mnt/testfs
sudo mount -o loop /tmp/testfs.img /mnt/testfs

# 4. Verify
df -Th /mnt/testfs
ls -la /mnt/testfs
sudo dumpe2fs /tmp/testfs.img | head -20

# 5. Test write
sudo touch /mnt/testfs/testfile
sudo ls -li /mnt/testfs/

# 6. Unmount
sudo umount /mnt/testfs

# XFS file system creation
dd if=/dev/zero of=/tmp/testxfs.img bs=1M count=100
mkfs.xfs /tmp/testxfs.img
sudo mount -o loop /tmp/testxfs.img /mnt/testfs
xfs_info /mnt/testfs

# Btrfs file system creation
dd if=/dev/zero of=/tmp/testbtrfs.img bs=1M count=256
mkfs.btrfs /tmp/testbtrfs.img
sudo mount -o loop /tmp/testbtrfs.img /mnt/testfs
btrfs filesystem show /mnt/testfs
```

### Exercise 4: [Advanced] -- File System Performance Measurement

```bash
# I/O benchmark using fio
# Sequential read
fio --name=seqread --rw=read --bs=4k --size=1G \
    --numjobs=1 --runtime=30 --time_based

# Random read
fio --name=randread --rw=randread --bs=4k --size=1G \
    --numjobs=4 --runtime=30 --time_based

# Sequential write
fio --name=seqwrite --rw=write --bs=4k --size=1G \
    --numjobs=1 --runtime=30 --time_based

# Random write
fio --name=randwrite --rw=randwrite --bs=4k --size=1G \
    --numjobs=4 --runtime=30 --time_based

# Simple benchmark using dd
# Write speed
dd if=/dev/zero of=/tmp/testfile bs=1M count=1024 conv=fdatasync

# Read speed (after clearing cache)
sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"
dd if=/tmp/testfile of=/dev/null bs=1M

# Verify page cache effectiveness
# First read (from disk)
time cat /tmp/testfile > /dev/null
# Second read (from page cache)
time cat /tmp/testfile > /dev/null
```

### Exercise 5: [Advanced] -- Verifying Sparse Files and Extents

```bash
# Create and verify sparse files
truncate -s 10G /tmp/sparse_test
ls -lh /tmp/sparse_test       # Logical size: 10G
du -h /tmp/sparse_test        # Actual disk usage: 0

# Write data to some portions
dd if=/dev/urandom of=/tmp/sparse_test bs=4K count=1 seek=1000
dd if=/dev/urandom of=/tmp/sparse_test bs=4K count=1 seek=2000

du -h /tmp/sparse_test        # Only 8K used

# Check extent information (ext4)
# filefrag: Display file fragmentation and extent information
filefrag -v /tmp/sparse_test

# Directly inspect inode with debugfs
sudo debugfs -R "stat <$(stat -c %i /path/to/file)>" /dev/sda1

# Check disk cache with hdparm
sudo hdparm -t /dev/sda       # Unbuffered read speed
sudo hdparm -T /dev/sda       # Buffer cache read speed
```

---

## 7. File System Troubleshooting

### 7.1 Common Problems and Solutions

```
Problem 1: "No space left on device" but df shows free space
  Cause: inode exhaustion
  Check: $ df -i
  Fix: Delete large quantities of small files, or recreate FS with more inodes

Problem 2: Deleting files does not reduce disk usage
  Cause: A process still has the file open
  Check: $ lsof +D /path/to/dir | grep deleted
  Fix: Restart the process, or identify the FD via /proc/<pid>/fd/

Problem 3: File system has become read-only
  Cause: Automatic protection triggered by file system error detection
  Check: $ dmesg | grep -i "remount"
  Fix: $ sudo fsck /dev/sda1 -> remount after repair

Problem 4: Cannot mount
  Cause: Superblock corruption
  Check: $ sudo file -s /dev/sda1
  Fix: $ sudo e2fsck -b 32768 /dev/sda1  # Repair with backup superblock

Problem 5: Significant performance degradation
  Cause: Fragmentation, journal saturation, cache shortage
  Check:
  $ sudo e4defrag -c /mount/point   # Fragmentation rate
  $ vmstat 1                         # Check I/O wait
  $ iostat -x 1                      # Detailed device I/O
  Fix: Defragment, adjust journal size, add memory
```

### 7.2 Data Recovery

```
Recovering deleted files:

  Why recovery is possible:
  -> File deletion = directory entry deletion + inode deallocation
  -> Data blocks themselves are not immediately overwritten
  -> Recovery is possible until new data is written over them

  Recovery tools:
  # ext4
  $ sudo extundelete /dev/sda1 --restore-all
  $ sudo ext4magic /dev/sda1 -r -d /tmp/recovered

  # General purpose
  $ sudo testdisk /dev/sda1       # Partition recovery
  $ sudo photorec /dev/sda1       # File recovery

  Preventive measures:
  - Regular backups (3-2-1 rule)
  - Use a trash can (trash-cli package)
  - Use trash-put instead of rm
  - Take regular Btrfs/ZFS snapshots
```


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issues | Verify configuration file path and format |
| Timeout | Network latency / resource shortage | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Verify execution user permissions, review settings |
| Data inconsistency | Concurrency conflicts | Introduce locking mechanisms, implement transaction management |

### Debugging Procedure

1. **Check error messages**: Read stack traces to identify the location of occurrence
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Incremental verification**: Verify hypotheses using log output and debuggers
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
    """Decorator that logs function input and output"""
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

1. **Identify the bottleneck**: Measure using profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O wait**: Verify disk and network I/O status
4. **Check concurrent connections**: Verify connection pool status

| Problem type | Diagnostic tools | Countermeasures |
|-------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |
---

## FAQ

### Q1: Why do inodes get exhausted?

When there are a large number of small files, the inode limit can be reached even though there is plenty of disk capacity (e.g., a mail server with massive amounts of mail, npm's node_modules). Check with `df -i`. In ext4, the initial inode count can be specified with mkfs options.

```bash
# Check for inode exhaustion
$ df -i /
Filesystem     Inodes  IUsed  IFree IUse% Mounted on
/dev/sda1      655360 655350     10  100% /

# Create FS with more inodes
$ mkfs.ext4 -N 2000000 /dev/sda1   # Allocate 2 million inodes

# Identify directories with high inode usage
$ for d in /*; do echo "$(find "$d" -xdev 2>/dev/null | wc -l) $d"; done | sort -rn | head -10
```

### Q2: How to choose between ext4, XFS, and Btrfs?

- **ext4**: Most stable. Ideal for desktops and general servers. Max 16TB file size. Ubuntu default. Mature technology with high reliability
- **XFS**: Strong for large files and high-concurrency I/O. RHEL default. Online expansion possible but cannot shrink
- **Btrfs**: Built-in snapshot, compression, and RAID features. SUSE default. Ideal for NAS use, but RAID5/6 is immature

### Q3: What is the difference between noatime and relatime?

```
atime (access time) update policies:

  atime (default, legacy method):
  -> Updates atime on every file read
  -> Read operations cause write I/O
  -> Negatively impacts SSD lifespan

  noatime:
  -> Never updates atime
  -> Most I/O efficient
  -> Issues with software that depends on atime for read/unread detection (e.g., mail)

  relatime (current default):
  -> Updates only when atime < mtime
  -> Or when more than 24 hours have passed since last update
  -> Maintains compatibility with atime-dependent software while reducing I/O
  -> Optimal for most use cases
```

### Q4: What is a file system UUID?

```
UUID (Universally Unique Identifier):
  A 128-bit identifier randomly generated when a file system is created

  Advantages:
  - Device names (/dev/sda1) can change with hardware configuration changes
  - UUID is immutable
  - Using UUID in /etc/fstab enables stable mounting

  How to check:
  $ blkid
  /dev/sda1: UUID="a1b2c3d4-e5f6-7890-abcd-ef1234567890" TYPE="ext4"

  $ ls -la /dev/disk/by-uuid/

  $ lsblk -o NAME,UUID

  Regenerating UUID:
  $ sudo tune2fs -U random /dev/sda1  # ext4
  $ sudo xfs_admin -U generate /dev/sda1  # XFS
```

### Q5: How to perform complete file deletion (secure erase)?

```
Normal deletion:
  -> Data blocks remain (recovery possible)

Secure deletion:
  # shred: Overwrite data
  $ shred -vfz -n 3 sensitive_file
  # -v: verbose, -f: force, -z: overwrite with zeros at end, -n: number of passes

  Note: shred is not reliable on SSDs
  -> FTL (Flash Translation Layer) retains data in other locations
  -> Use TRIM + secure erase

  SSD secure erase:
  $ sudo hdparm --security-set-pass password /dev/sda
  $ sudo hdparm --security-erase password /dev/sda

  Recommended approach:
  -> Use full disk encryption (LUKS, BitLocker) from the start
  -> At disposal, simply destroying the encryption key renders data unreadable
```

---

## Summary

| Concept | Key Points |
|---------|-----------|
| Block/Sector | Smallest FS unit. Block size is a trade-off between performance and space efficiency |
| inode | File metadata. Does not contain name. Optimized with extents |
| Directory | Name-to-inode-number mapping. Fast lookup with HTree |
| Journaling | Prevents inconsistency. Fast recovery from power failure. Three modes |
| CoW | Non-overwriting approach. Supports snapshots. Btrfs/ZFS |
| VFS | Unified API. Four main objects. Transparent access to different file systems |
| File descriptor | Per-process file reference. Pay attention to limit settings |
| mmap | Maps files to virtual memory. Fast without copying |
| TRIM | For SSDs. Notifies deleted blocks. Important for performance maintenance |
| xattr | Extended attributes. Stores ACLs, SELinux contexts, etc. |

---

## Recommended Next Guides

---

## References
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Ch.13-15, 2018.
2. Love, R. "Linux Kernel Development." 3rd Ed, Ch.13, 2010.
3. Bovet, D. & Cesati, M. "Understanding the Linux Kernel." 3rd Ed, O'Reilly, 2005.
4. Tanenbaum, A. S. & Bos, H. "Modern Operating Systems." 4th Ed, Pearson, 2014.
5. McKusick, M. K. et al. "The Design and Implementation of the FreeBSD Operating System." 2nd Ed, 2014.
6. Ts'o, T. "Design and Implementation of ext4." Ottawa Linux Symposium, 2009.
7. Linux Kernel Documentation. "Filesystems." https://www.kernel.org/doc/html/latest/filesystems/
8. Arpaci-Dusseau, R. & Arpaci-Dusseau, A. "Operating Systems: Three Easy Pieces." Ch.39-42, 2018.
