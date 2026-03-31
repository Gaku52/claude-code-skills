# Storage Capacity and Units

> An engineer who cannot precisely explain the difference between "gigabyte" and "gibibyte" cannot design a petabyte-scale system.

## Learning Objectives

- [ ] Explain the definitions of bit and byte, along with their historical background
- [ ] Accurately distinguish between binary prefixes (KiB, MiB, GiB) and decimal prefixes (KB, MB, GB)
- [ ] Estimate typical sizes of various data types (text, images, audio, video)
- [ ] Perform back-of-the-envelope estimation for system design capacity planning
- [ ] Quickly estimate bandwidth, transfer time, and storage cost
- [ ] Explain the design principles of tiered storage

## Prerequisites


---

## 1. Bits and Bytes -- The Smallest Units of Information

### 1.1 Definition of a Bit

A bit (short for binary digit) is the smallest unit in information theory. It is a concept formalized by Claude Shannon in his 1948 paper "A Mathematical Theory of Communication," representing one of two states (0 or 1).

```
The essence of a bit:

  1 bit = the amount of information needed to specify one of 2 states

  Physical implementation examples:
  +---------------------+-----------+-----------+
  | Medium              | State 0   | State 1   |
  +---------------------+-----------+-----------+
  | Voltage (TTL)       | 0-0.8V    | 2.0-5.0V  |
  | Magnetic disk        | S->N pole | N->S pole |
  | Optical disc         | No pit    | Pit       |
  | DRAM                | Discharged| Charged   |
  | Flash (SLC)         | High Vth  | Low Vth   |
  | Optical fiber        | No light  | Light     |
  +---------------------+-----------+-----------+

  Number of states representable by n bits = 2^n
  ------------------------------------------
  1 bit  -> 2 states       (0, 1)
  2 bits -> 4 states       (00, 01, 10, 11)
  3 bits -> 8 states       (000 ... 111)
  8 bits -> 256 states     (00000000 ... 11111111)
  16 bits -> 65,536 states
  32 bits -> 4,294,967,296 states (~4.3 billion)
  64 bits -> 18,446,744,073,709,551,616 states (~1.8 x 10^19)
```

The information content is expressed using logarithms. When an event has probability p, the information gained from learning the event occurred is `-log2(p)` bits.

```python
import math

def information_content(probability):
    """Calculate the information content of an event in bits"""
    if probability <= 0 or probability > 1:
        raise ValueError("Probability must be in range 0 < p <= 1")
    return -math.log2(probability)

# Example: coin flip (probability of heads = 0.5)
print(f"Coin flip:    {information_content(0.5):.1f} bit")  # 1.0 bit

# Example: die roll (probability of a specific face = 1/6)
print(f"Die roll:     {information_content(1/6):.2f} bit")  # 2.58 bit

# Example: one letter (26 letters equally likely = 1/26)
print(f"One letter:   {information_content(1/26):.2f} bit")   # 4.70 bit

# Example: ASCII character (128 chars equally likely = 1/128)
print(f"ASCII char:   {information_content(1/128):.2f} bit")  # 7.00 bit
```

### 1.2 Definition and History of the Byte

A byte is the smallest addressable unit in most modern computers. Today, 1 byte = 8 bits is the de facto standard, but historically, bytes of different sizes existed.

```
Historical evolution of byte size:

  Era         Machine                    Byte size
  -------------------------------------------------
  1950s       IBM 7030 Stretch           Variable
  1956        IBM 7030 design docs       First appearance of "byte"
  1963        IBM System/360             Standardized to 8 bits
  1970s       PDP-8                      12-bit word
  1970s       PDP-10                     36-bit word
  1980s~      IBM PC compatibles         8 bits as de facto standard
  Present     Nearly all architectures   8 bits = 1 byte

  Note: To avoid historical ambiguity, the term "octet" is used
  to explicitly mean 8 bits (especially in networking).
  RFCs use "octet."

  Etymology:
  - "byte" is a portmanteau between "bit" and "bite"
  - Named by Werner Buchholz (IBM) in 1956
  - Changed to "y" to avoid confusion with "bite"
```

### 1.3 Words and Addressing

```
Word size and architecture:

  Word = the basic unit of data that a CPU processes at once

  +------------------+-----------+----------------------+
  | Architecture     | Word width| Address space        |
  +------------------+-----------+----------------------+
  | Intel 8080       | 8 bit     | 64 KB                |
  | Intel 8086       | 16 bit    | 1 MB                 |
  | Intel 80386      | 32 bit    | 4 GB                 |
  | x86-64           | 64 bit    | 16 EB (theoretical)  |
  | x86-64 (current) | 48 bit    | 256 TB               |
  | ARM Cortex-M     | 32 bit    | 4 GB                 |
  | AArch64          | 64 bit    | 256 TB - 4 PB        |
  +------------------+-----------+----------------------+

  Impact of 32-bit vs 64-bit:
  +------------------+--------------+------------------+
  | Item             | 32-bit       | 64-bit           |
  +------------------+--------------+------------------+
  | Max memory       | 4 GB         | 16 EB (theory)   |
  | Pointer size     | 4 bytes      | 8 bytes          |
  | Integer math     | 32-bit       | 64-bit           |
  | Memory efficiency| Good         | Pointers double  |
  | Per process      | 2-3 GB       | 128 TB+          |
  +------------------+--------------+------------------+
```

### 1.4 Nibbles and Other Units

```
Units between bits and bytes:

  Nibble (nybble) = 4 bits = 1 hexadecimal digit
  +----------+-------------------------------+
  | Nibble   | Hex representation            |
  +----------+-------------------------------+
  | 0000     | 0                             |
  | 0101     | 5                             |
  | 1001     | 9                             |
  | 1010     | A                             |
  | 1111     | F                             |
  +----------+-------------------------------+

  1 byte = 2 nibbles = high nibble + low nibble

  Example: 0xA7 = 1010_0111
           ^^^^   ^^^^
           high   low
           nibble nibble

  Other information units:
  +--------------+---------------------------------+
  | Unit         | Description                     |
  +--------------+---------------------------------+
  | trit         | 3 states (1 digit in base 3)    |
  | dit / ban    | 10 states (1 digit in base 10)  |
  | nat          | Information unit based on        |
  |              | natural logarithm               |
  | shannon      | Alternate name for bit (after    |
  |              | Shannon)                        |
  | qubit        | Quantum bit (superposition)      |
  +--------------+---------------------------------+
```

---

## 2. Prefixes -- The Binary vs Decimal Confusion

### 2.1 The Core of the Problem

One of the most deep-rooted confusions in computer science is the prefix for capacity units. Is 1 kilobyte 1,000 bytes or 1,024 bytes? This issue is not merely academic; it has even led to lawsuits over "capacity fraud" when purchasing storage.

```
Source of confusion:

  Decimal prefixes (SI prefixes):
    kilo  = 10^3  = 1,000
    mega  = 10^6  = 1,000,000
    giga  = 10^9  = 1,000,000,000
    tera  = 10^12 = 1,000,000,000,000

  Binary convention:
    "kilo" = 2^10 = 1,024          (2.4% larger)
    "mega" = 2^20 = 1,048,576      (4.9% larger)
    "giga" = 2^30 = 1,073,741,824  (7.4% larger)
    "tera" = 2^40 = ~1.0995 x 10^12 (10.0% larger)

  Scale of difference:
  +-----------+--------------+--------------+----------+
  | Prefix    | Decimal      | Binary       | Diff     |
  +-----------+--------------+--------------+----------+
  | K / Ki    | 1,000        | 1,024        | +2.4%    |
  | M / Mi    | 1,000,000    | 1,048,576    | +4.9%    |
  | G / Gi    | 1,000,000,000| 1,073,741,824| +7.4%    |
  | T / Ti    | 10^12        | 2^40         | +10.0%   |
  | P / Pi    | 10^15        | 2^50         | +12.6%   |
  | E / Ei    | 10^18        | 2^60         | +15.3%   |
  +-----------+--------------+--------------+----------+

  -> The gap widens as units get larger
  -> A 1 TB HDD appears as approximately 931 GiB on the OS
```

### 2.2 IEC Binary Prefixes (Established 1998)

In 1998, the International Electrotechnical Commission (IEC) formally established binary prefixes to resolve the confusion (IEC 60027-2).

```
IEC binary prefixes:

  +----------+--------+---------+------------------+
  | Prefix   | Symbol | Value   | Etymology        |
  +----------+--------+---------+------------------+
  | kibi     | Ki     | 2^10    | kilo + binary    |
  | mebi     | Mi     | 2^20    | mega + binary    |
  | gibi     | Gi     | 2^30    | giga + binary    |
  | tebi     | Ti     | 2^40    | tera + binary    |
  | pebi     | Pi     | 2^50    | peta + binary    |
  | exbi     | Ei     | 2^60    | exa + binary     |
  | zebi     | Zi     | 2^70    | zetta + binary   |
  | yobi     | Yi     | 2^80    | yotta + binary   |
  +----------+--------+---------+------------------+

  Correct notation:
  - RAM 16 GiB (binary: 16 x 2^30 = 17,179,869,184 bytes)
  - HDD 1 TB  (decimal: 1 x 10^12 = 1,000,000,000,000 bytes)
  - SSD 512 GB (decimal: 512 x 10^9 = 512,000,000,000 bytes)
```

### 2.3 Usage by Industry

```
Prefix usage across industries and software:

  Uses decimal prefixes (SI, powers of 1000):
  +-------------------------------------------------+
  | - HDD / SSD manufacturers (capacity labels)     |
  | - Network bandwidth (Mbps, Gbps)                |
  | - Carriers (data plans: 3GB, 20GB)              |
  | - macOS (since 10.6 Snow Leopard)               |
  | - Ubuntu / GNOME                                |
  | - iOS / iPadOS                                  |
  | - HDD manufacturers (Western Digital, Seagate)   |
  +-------------------------------------------------+

  Uses binary prefixes (powers of 1024):
  +-------------------------------------------------+
  | - RAM (memory module capacity)                   |
  | - Windows (File Explorer file size display)      |
  | - Linux kernel (dmesg, etc.)                     |
  | - JEDEC memory standards                         |
  | - Many programming language standard libraries   |
  +-------------------------------------------------+

  Explicitly uses IEC binary prefixes (KiB, MiB, GiB):
  +-------------------------------------------------+
  | - GNU coreutils (ls -lh --si vs default)        |
  | - Wikipedia                                     |
  | - IEEE / IEC standard documents                 |
  | - Some Linux distributions                       |
  | - systemd / journalctl                          |
  +-------------------------------------------------+
```

### 2.4 The "Missing Capacity" Problem

```
The "less capacity than expected" problem when purchasing HDD/SSD:

  Purchased capacity: 1 TB (manufacturer label, decimal)
  = 1,000,000,000,000 bytes

  OS display (binary):
  = 1,000,000,000,000 / 1,073,741,824
  = 931.32 GiB

  -> Approximately 69 GiB (~7%) appears to have "disappeared"

  Further considering filesystem overhead:
  - Partition table:          A few MB
  - Filesystem metadata:      1-5% of total
  - Reserved blocks (ext4):   5% (default)
  - SSD over-provisioning:    7-28%

  Purchased vs usable capacity mapping:
  +----------------+--------------+--------------+
  | Manufacturer   | OS display   | Effective    |
  | label          | (binary)     | capacity     |
  +----------------+--------------+--------------+
  | 128 GB         | 119 GiB      | 110-115 GiB  |
  | 256 GB         | 238 GiB      | 220-230 GiB  |
  | 512 GB         | 476 GiB      | 440-460 GiB  |
  | 1 TB           | 931 GiB      | 860-900 GiB  |
  | 2 TB           | 1,862 GiB    | 1,720-1,800 GiB|
  | 4 TB           | 3,725 GiB    | 3,450-3,600 GiB|
  +----------------+--------------+--------------+

  Litigation history:
  - 2006: Western Digital settled a class-action lawsuit
  - Provided free data recovery software to consumers
  - Since then, many manufacturers add fine print on packaging:
    "1GB = 1,000,000,000 bytes"
```

### 2.5 Unit Conversion Implementation

```python
"""Storage capacity unit conversion utility"""

# --- Binary prefixes (IEC) ---
KiB = 1024
MiB = 1024 ** 2     # 1,048,576
GiB = 1024 ** 3     # 1,073,741,824
TiB = 1024 ** 4     # 1,099,511,627,776
PiB = 1024 ** 5
EiB = 1024 ** 6

# --- Decimal prefixes (SI) ---
KB = 1000
MB = 1000 ** 2      # 1,000,000
GB = 1000 ** 3      # 1,000,000,000
TB = 1000 ** 4      # 1,000,000,000,000
PB = 1000 ** 5
EB = 1000 ** 6

def human_readable_binary(size_bytes: int) -> str:
    """Convert byte count to human-readable binary prefix format"""
    units = [
        (EiB, "EiB"), (PiB, "PiB"), (TiB, "TiB"),
        (GiB, "GiB"), (MiB, "MiB"), (KiB, "KiB"),
    ]
    for threshold, unit in units:
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.2f} {unit}"
    return f"{size_bytes} B"

def human_readable_decimal(size_bytes: int) -> str:
    """Convert byte count to human-readable decimal prefix format"""
    units = [
        (EB, "EB"), (PB, "PB"), (TB, "TB"),
        (GB, "GB"), (MB, "MB"), (KB, "KB"),
    ]
    for threshold, unit in units:
        if size_bytes >= threshold:
            return f"{size_bytes / threshold:.2f} {unit}"
    return f"{size_bytes} B"

# --- Usage examples ---
hdd_capacity = 1 * TB   # Manufacturer-labeled 1 TB
print(f"HDD capacity (decimal): {human_readable_decimal(hdd_capacity)}")
# => HDD capacity (decimal): 1.00 TB
print(f"HDD capacity (binary):  {human_readable_binary(hdd_capacity)}")
# => HDD capacity (binary):  931.32 GiB

ram_capacity = 16 * GiB  # 16 GiB RAM
print(f"RAM capacity (binary):  {human_readable_binary(ram_capacity)}")
# => RAM capacity (binary):  16.00 GiB
print(f"RAM capacity (decimal): {human_readable_decimal(ram_capacity)}")
# => RAM capacity (decimal): 17.18 GB
```

---

## 3. The Complete Picture of SI Prefixes

### 3.1 From Smallest to Largest

```
Complete list of SI prefixes (International System of Units):

  Small side (used primarily in bit rates, etc.):
  +----------+--------+---------+------------------+
  | Prefix   | Symbol | 10^n    | Name             |
  +----------+--------+---------+------------------+
  | quecto   | q      | 10^-30  | quecto           |
  | ronto    | r      | 10^-27  | ronto            |
  | yocto    | y      | 10^-24  | yocto            |
  | zepto    | z      | 10^-21  | zepto            |
  | atto     | a      | 10^-18  | atto             |
  | femto    | f      | 10^-15  | femto            |
  | pico     | p      | 10^-12  | pico             |
  | nano     | n      | 10^-9   | nano             |
  | micro    | u      | 10^-6   | micro            |
  | milli    | m      | 10^-3   | milli            |
  +----------+--------+---------+------------------+

  Large side (directly used for storage capacity):
  +----------+--------+---------+--------------------------+
  | Prefix   | Symbol | 10^n    | Typical usage            |
  +----------+--------+---------+--------------------------+
  | kilo     | K / k  | 10^3    | Text files               |
  | mega     | M      | 10^6    | Photos, MP3              |
  | giga     | G      | 10^9    | Movies, RAM              |
  | tera     | T      | 10^12   | HDDs, annual logs        |
  | peta     | P      | 10^15   | Large-scale DBs, data    |
  |          |        |         | lakes                    |
  | exa      | E      | 10^18   | Entire cloud             |
  | zetta    | Z      | 10^21   | Global annual data gen.  |
  | yotta    | Y      | 10^24   | (theoretical at present) |
  | ronna    | R      | 10^27   | (added 2022)             |
  | quetta   | Q      | 10^30   | (added 2022)             |
  +----------+--------+---------+--------------------------+

  At the 27th General Conference on Weights and Measures (CGPM)
  in November 2022, four new prefixes were added:
  ronna / quetta / ronto / quecto.
```

### 3.2 Intuitive Sense of Data Scale

```
Intuitive understanding of each scale:

  1 B     = One English letter
  1 KB    = A short text file (a few paragraphs)
  1 MB    = One novel as text / 1 minute of MP3
  1 GB    = One movie (SD quality) / about 1.5 CDs
  1 TB    = About 45 million printed pages
  1 PB    = About 2,000 years of MP3 music / Netflix entire catalog
  1 EB    = Amount of data all humans generate in about 2 days (est.)
  1 ZB    = About half a year of global internet traffic
  1 YB    = Several times all data in human history combined

  Cosmic scale analogies:
  +----------+---------------------------------------+
  | Unit     | Analogy                               |
  +----------+---------------------------------------+
  | 1 KB     | A very short letter                   |
  | 1 MB     | One thick book                        |
  | 1 GB     | A single bookshelf in a small library |
  | 1 TB     | An entire university library          |
  | 1 PB     | An entire national library (est.)     |
  | 1 EB     | All libraries in the world combined   |
  | 1 ZB     | Comparable to the number of grains    |
  |          | of sand on Earth                      |
  | 1 YB     | Still far from the number of atoms    |
  |          | in the observable universe             |
  +----------+---------------------------------------+

  Reference: The number of atoms in the observable universe is ~10^80
       1 YB = 10^24 bytes = 8 x 10^24 bits
       -> Still 56 orders of magnitude short of the universe's atoms
```

---

## 4. Intuitive Data Sizes

### 4.1 Text Data

```
Text capacity by encoding:

  ASCII (alphanumeric):
    1 character = 1 byte (7 bits + parity or 0 padding)
    1 page of English (~250 words / ~1,500 chars) = ~1.5 KB
    1 English novel (~80,000 words / ~500,000 chars) = ~500 KB

  UTF-8 (variable length):
    ASCII-compatible chars: 1 byte (U+0000-U+007F)
    Latin extended:         2 bytes (U+0080-U+07FF)
    CJK characters:         3 bytes (U+0800-U+FFFF)
    Emoji:                  4 bytes (U+10000-U+10FFFF)

  UTF-16 (Java internal, Windows internal API):
    BMP characters:        2 bytes
    Supplementary chars:   4 bytes (surrogate pairs)

  UTF-32:
    All characters:        4 bytes (fixed length)

  CJK text capacity (Japanese example):
  +---------------------+------------+------------+
  | Content             | Characters | UTF-8 size |
  +---------------------+------------+------------+
  | 1 tweet             | 140 chars  | ~420 B     |
  | Email body          | 500 chars  | ~1.5 KB    |
  | Blog post           | 3,000 chars| ~9 KB      |
  | Short book          | 100K chars | ~300 KB    |
  | Full-length novel   | 300K chars | ~900 KB    |
  | Manga vol (text)    | 3,000 chars| ~9 KB      |
  | Legal code compil.  | ~9M chars  | ~27 MB     |
  +---------------------+------------+------------+

  Programming data:
  +--------------------------+--------------+
  | Data type                | Typical size |
  +--------------------------+--------------+
  | JSON API response (sm)   | 1-10 KB      |
  | JSON API response (lg)   | 100 KB-1 MB  |
  | 1 log line               | 100 B-1 KB   |
  | App logs per day         | 100 MB-10 GB |
  | RDB record (small)       | 100 B-1 KB   |
  | RDB record (large)       | 1-10 KB      |
  | CSV 1M rows              | 50-500 MB    |
  | 1 source code file       | 1-50 KB      |
  | Mid-size project total   | 1-100 MB     |
  | Linux kernel source      | ~1.3 GB      |
  +--------------------------+--------------+
```

### 4.2 Image Data

```
Image capacity -- format comparison:

  Uncompressed image size calculation:
  Size = Width x Height x Color depth (bytes) + Header

  Example: 1920x1080, 24-bit color (RGB 8 bits each)
  = 1920 x 1080 x 3 = 6,220,800 B = ~5.93 MiB

  Capacity comparison by format (1920x1080 photo):
  +------------------+-----------+----------+----------+
  | Format           | Size      | Ratio    | Feature  |
  +------------------+-----------+----------+----------+
  | BMP (uncompr.)   | 5.93 MB   | 1:1      | Uncompr. |
  | PNG              | 2-4 MB    | 1.5-3:1  | Lossless |
  | JPEG (Q85)       | 300-600KB | 10-20:1  | Lossy    |
  | JPEG (Q50)       | 100-250KB | 25-60:1  | Lossy    |
  | WebP             | 200-400KB | 15-30:1  | Lossy    |
  | AVIF             | 150-300KB | 20-40:1  | Lossy    |
  | JPEG XL          | 150-350KB | 17-40:1  | Both     |
  +------------------+-----------+----------+----------+

  Typical sizes by resolution (JPEG Q85):
  +--------------------+----------+------------+
  | Resolution         | Pixels   | Size       |
  +--------------------+----------+------------+
  | Icon 32x32         | 1 K      | 2-5 KB     |
  | Thumbnail 200x200  | 40 K     | 10-30 KB   |
  | SNS avatar 400x400 | 160 K    | 30-80 KB   |
  | Web image 800x600  | 480 K    | 60-150 KB  |
  | HD 1280x720        | 921 K    | 150-400 KB |
  | Full HD 1920x1080  | 2.07 M   | 300-600 KB |
  | 4K 3840x2160       | 8.29 M   | 1-3 MB     |
  | 8K 7680x4320       | 33.18 M  | 5-15 MB    |
  | RAW 6000x4000      | 24 M     | 25-50 MB   |
  +--------------------+----------+------------+

  Web development image optimization targets:
  +--------------------+-----------------+
  | Use case           | Target size     |
  +--------------------+-----------------+
  | User avatar        | 50-200 KB       |
  | E-commerce product | 100-500 KB      |
  | Hero image         | 200 KB-1 MB     |
  | Landing/Banner     | 200 KB-2 MB     |
  | All images on page | Target: < 1 MB  |
  +--------------------+-----------------+
```

### 4.3 Audio Data

```
Audio capacity:

  Uncompressed audio size calculation:
  Size = Sample rate x Bit depth x Channels x Seconds / 8

  Example: CD quality (44.1 kHz, 16 bit, stereo, 1 second)
  = 44,100 x 16 x 2 x 1 / 8 = 176,400 B = ~172 KiB/s

  By quality level:
  +-------------------+-------+------+------+------------+
  | Quality           | Hz    | bit  | ch   | Rate       |
  +-------------------+-------+------+------+------------+
  | Telephone         | 8k    | 8    | 1    | 8 KB/s     |
  | AM Radio          | 22k   | 16   | 1    | 44 KB/s    |
  | FM Radio          | 32k   | 16   | 2    | 128 KB/s   |
  | CD                | 44.1k | 16   | 2    | 172 KB/s   |
  | DVD Audio         | 96k   | 24   | 2    | 576 KB/s   |
  | Hi-Res            | 192k  | 24   | 2    | 1,152 KB/s |
  +-------------------+-------+------+------+------------+

  Capacity by codec (4-minute track):
  +------------------+----------+--------------+----------+
  | Format           | Bitrate  | 4 min size   | Ratio    |
  +------------------+----------+--------------+----------+
  | WAV (uncompr.)   | 1,411kbps| 42 MB        | 1:1      |
  | FLAC (lossless)  | 700-1000 | 21-30 MB     | 1.5-2:1  |
  | AAC 256kbps      | 256 kbps | 7.7 MB       | 5.5:1    |
  | MP3 320kbps      | 320 kbps | 9.6 MB       | 4.4:1    |
  | MP3 128kbps      | 128 kbps | 3.8 MB       | 11:1     |
  | Opus 128kbps     | 128 kbps | 3.8 MB       | 11:1     |
  | Opus 64kbps      | 64 kbps  | 1.9 MB       | 22:1     |
  | Codec2 (voice)   | 2.4 kbps | 72 KB        | 588:1    |
  +------------------+----------+--------------+----------+

  Streaming service quality settings:
  +------------------+--------------+------------------+
  | Service          | Normal       | High quality     |
  +------------------+--------------+------------------+
  | Spotify          | AAC 128kbps  | OGG 320kbps      |
  | Apple Music      | AAC 256kbps  | ALAC (lossless)  |
  | YouTube Music    | AAC 128kbps  | AAC 256kbps      |
  | Amazon Music HD  | AAC 256kbps  | FLAC 24bit/192k  |
  +------------------+--------------+------------------+
```

### 4.4 Video Data

```
Video capacity:

  Uncompressed video size calculation:
  Size = Width x Height x Color depth(B) x Frame rate x Seconds

  Example: 1080p, 30fps, 24-bit color, 1 second
  = 1920 x 1080 x 3 x 30 = 186,624,000 B = ~178 MiB/s

  -> 1 minute of uncompressed 1080p = ~10.4 GiB
  -> 2-hour movie = ~1.25 TiB
  -> Storage and transmission are impossible without compression

  Capacity by resolution x codec (per minute):
  +----------+-----------+-----------+-----------+
  | Res.     | H.264     | H.265     | AV1       |
  +----------+-----------+-----------+-----------+
  | 480p     | 8 MB      | 5 MB      | 3 MB      |
  | 720p     | 20 MB     | 12 MB     | 8 MB      |
  | 1080p    | 45 MB     | 25 MB     | 17 MB     |
  | 4K       | 130 MB    | 70 MB     | 45 MB     |
  | 8K       | 500 MB    | 250 MB    | 160 MB    |
  +----------+-----------+-----------+-----------+
  * Values are approximate for typical bitrate settings

  Streaming service bandwidth:
  +------------------+--------------+------------------+
  | Service          | Quality      | Bandwidth        |
  +------------------+--------------+------------------+
  | YouTube 360p     | SD           | 0.7 Mbps         |
  | YouTube 720p     | HD           | 2.5 Mbps         |
  | YouTube 1080p    | Full HD      | 5 Mbps           |
  | YouTube 4K       | 4K           | 20 Mbps          |
  | Netflix SD       | SD           | 1 Mbps           |
  | Netflix HD       | HD           | 5 Mbps           |
  | Netflix 4K HDR   | 4K           | 15-25 Mbps       |
  | Zoom audio only  | -            | 0.1 Mbps         |
  | Zoom 720p        | HD           | 1.5 Mbps         |
  | Zoom 1080p       | Full HD      | 3 Mbps           |
  +------------------+--------------+------------------+

  Video storage estimates:
  +------------------------+----------------------+
  | Content                | Size                 |
  +------------------------+----------------------+
  | 30s promo video (web)  | 5-20 MB              |
  | YouTube 10min video    | 100-500 MB (source)  |
  | 1-hour webinar         | 500 MB-2 GB          |
  | 1 movie Blu-ray        | 25-50 GB             |
  | 1 movie 4K UHD         | 50-100 GB            |
  | Netflix full catalog   | Estimated 10 PB+     |
  +------------------------+----------------------+
```

---

## 5. Physical Characteristics of Storage Media

### 5.1 Storage Technology Hierarchy

```
Storage hierarchy (speed vs capacity vs cost):

  Fast, expensive, small capacity
  ^
  |  CPU Registers      : ~hundreds B, < 1ns   , $$$$$$$
  |  L1 Cache           : 32-64KB    , 1ns     , $$$$$$
  |  L2 Cache           : 256KB-1MB  , 4ns     , $$$$$
  |  L3 Cache           : multi MB   , 20ns    , $$$$
  |  Main memory (DDR5) : multi GB-TB, 100ns   , $$$
  |  NVMe SSD           : hundreds GB-TB, 10-100us, $$
  |  SATA SSD           : hundreds GB-TB, 50-200us, $$
  |  HDD                : multi TB-20TB, 2-10ms , $
  |  Tape (LTO-9)       : 18TB/cart  , secs-min, c
  |  Optical disc        : 25-128GB  , seconds , c
  v
  Slow, cheap, large capacity

  +----------------+--------------+------------+----------+
  | Media          | Cap/device   | Throughput | $/GB     |
  +----------------+--------------+------------+----------+
  | DDR5 DIMM      | 16-256 GB   | 50 GB/s    | $3-5     |
  | NVMe Gen5 SSD  | 1-8 TB      | 12 GB/s    | $0.08-0.15|
  | SATA SSD       | 256GB-4 TB  | 550 MB/s   | $0.05-0.10|
  | HDD (CMR)      | 2-24 TB     | 200 MB/s   | $0.015-0.03|
  | HDD (SMR)      | 4-20 TB     | 150 MB/s   | $0.012-0.025|
  | LTO-9 Tape     | 18 TB       | 400 MB/s   | $0.005   |
  | Blu-ray 4-layer| 128 GB      | 72 MB/s    | $0.01    |
  | DNA Storage    | Theory:     | Extremely  | Research |
  |                | 215PB/g     | slow       | stage    |
  +----------------+--------------+------------+----------+
```

### 5.2 IOPS Comparison of Storage Devices

```
IOPS (Input/Output Operations Per Second) comparison:

  +----------------+--------------+--------------+----------+
  | Device         | Read IOPS    | Write IOPS   | Latency  |
  +----------------+--------------+--------------+----------+
  | HDD 7200rpm    | 100-200      | 100-200      | 2-10 ms  |
  | HDD 15000rpm   | 200-400      | 200-400      | 2-4 ms   |
  | SATA SSD       | 50K-100K     | 30K-90K      | 50-200 us|
  | NVMe SSD       | 500K-1.5M    | 200K-1M      | 10-50 us |
  | Intel Optane   | 500K-600K    | 200K-500K    | 7-10 us  |
  | RAM Disk       | Millions     | Millions     | < 1 us   |
  +----------------+--------------+--------------+----------+

  Example: If 1 DB transaction requires 10 random I/Os
  - HDD:      100 IOPS / 10 = 10 TPS
  - SATA SSD: 80,000 IOPS / 10 = 8,000 TPS
  - NVMe SSD: 800,000 IOPS / 10 = 80,000 TPS

  -> Migration to SSD alone can improve DB throughput 100-800x
```

### 5.3 Historical Evolution of Storage Capacity

```
Evolution of storage capacity:

  Year  Device                       Capacity     $/GB (then)
  ---------------------------------------------------------
  1956  IBM 350 (first HDD)         5 MB         $200,000
  1969  IBM 2314                    29 MB        $10,000
  1980  Seagate ST-506 (first 5.25")5 MB         $5,000
  1983  IBM PC/XT                   10 MB        $3,500
  1988  First 100MB HDD             100 MB       $100
  1991  First 1GB HDD               1 GB         $10
  1997  First 10GB HDD              10 GB        $4
  2000  First 100GB HDD             100 GB       $1
  2007  First 1TB HDD               1 TB         $0.30
  2009  First 2TB HDD               2 TB         $0.10
  2019  16TB HDD (Seagate Exos)     16 TB        $0.025
  2023  30TB HDD (SMR)              30 TB        $0.015
  2025  HDD roadmap                 40-50 TB     -

  Kryder's Law:
  "HDD areal density doubles every 13 months"
  -> The storage equivalent of Moore's Law
  -> Has been slowing since the 2010s
  -> May re-accelerate with HAMR/MAMR and other new technologies

  SSD evolution:
  Year  Capacity           Notes
  ----------------------------------------
  2006  32 GB              Early consumer SSDs
  2009  256 GB             SLC/MLC
  2012  1 TB               Samsung 840 EVO series
  2016  4 TB               Samsung 850 EVO
  2018  8 TB               Samsung 870 QVO
  2020  16 TB              Enterprise
  2023  32 TB              Solidigm D5-P5336
  2024  64 TB+             QLC/PLC high-capacity SSDs
```

---

## 6. Capacity Estimation for System Design

### 6.1 Back-of-the-Envelope Estimation Techniques

One of the most important skills in system design interviews and actual infrastructure design is back-of-the-envelope estimation. Exact numbers are unnecessary; what matters is getting the order of magnitude right.

```
Useful approximations for estimation:

  2^10 = 10^3 (thousand)    Error: +2.4%
  2^20 = 10^6 (million)     Error: +4.9%
  2^30 = 10^9 (billion)     Error: +7.4%
  2^40 = 10^12 (trillion)   Error: +10.0%

  Time approximations:
  1 day   = 86,400 sec    = ~10^5 sec (rough estimate is fine)
  1 month = 2,592,000 sec = ~2.5 x 10^6 sec
  1 year  = 31,536,000 sec = ~3 x 10^7 sec = ~pi x 10^7 sec

  Common calculation patterns:
  +------------------------------------------------------+
  | 1. Storage = DAU x actions/day x data size            |
  |    x replica count x retention period                 |
  |                                                       |
  | 2. Bandwidth = QPS x response size                    |
  |                                                       |
  | 3. QPS = DAU x actions/day / 86400                    |
  |    Peak QPS = avg QPS x 2-3 (peak factor)             |
  |                                                       |
  | 4. Read/write ratio = read QPS / write QPS            |
  |    SNS: 100:1 - 1000:1                               |
  |    Messaging: 5:1 - 20:1                              |
  |    E-commerce: 50:1 - 200:1                           |
  +------------------------------------------------------+
```

### 6.2 Capacity Calculation for a Twitter-like Service

```
Capacity estimate for a Twitter-like service:

  Assumptions:
  +--------------------------------------+
  | MAU (Monthly Active Users): 500M     |
  | DAU (Daily Active Users): 300M       |
  | Avg posts per user: 2/day            |
  | Avg text: 200 chars (UTF-8: 400B)    |
  | Metadata: 200B/post                  |
  | Image attachment rate: 20% (avg 200KB)|
  | Video attachment rate: 5% (avg 5MB)  |
  +--------------------------------------+

  Step 1: Daily posts
  300M x 2 = 600M posts/day

  Step 2: Daily data volume
  Text+meta: 600M x 600B      = 360 GB/day
  Images:    600M x 0.2 x 200KB = 24 TB/day
  Video:     600M x 0.05 x 5MB  = 150 TB/day
  Total:     ~174 TB/day

  Step 3: Annual storage
  174 TB x 365 = 63.5 PB/year

  Step 4: Replicas and backups
  3x replicas: 63.5 x 3 = 190.5 PB/year
  Add backups: ~200 PB/year

  Step 5: 5-year retention
  200 PB x 5 = 1 EB (before compression)
  50% compression: ~500 PB

  Step 6: Write QPS
  600M / 86,400 = ~6,944 QPS (average)
  Peak (x3): ~20,833 QPS

  Step 7: Read QPS (100:1 ratio)
  Read average: 694,400 QPS
  Read peak: ~2,083,000 QPS

  Step 8: Bandwidth
  Write: 174 TB / 86,400 = ~2 GB/s = ~16 Gbps
  Read:  2 GB/s x 100 = ~200 GB/s = ~1.6 Tbps

  Conclusion:
  +----------------------------------------+
  | Annual storage growth: ~200 PB          |
  | 5-year total: ~500 PB (compressed)      |
  | Peak write QPS: ~21K                    |
  | Peak read QPS: ~2.1M                    |
  | Required bandwidth: ~1.6 Tbps (read)    |
  +----------------------------------------+
```

### 6.3 Infrastructure Guidelines by Service Scale

```
Infrastructure guidelines by service scale:

  +---------------+----------+-----------+---------+----------+
  | Scale         | DB       | Storage   | BW      | QPS      |
  +---------------+----------+-----------+---------+----------+
  | Personal blog | 100 MB   | 10 GB     | 10 Mbps | 1-10     |
  | Small SaaS    | 1 GB     | 100 GB    | 50 Mbps | 10-100   |
  | Startup       | 10 GB    | 1 TB      | 100Mbps | 100-1K   |
  | Mid-scale     | 1 TB     | 100 TB    | 10 Gbps | 1K-100K  |
  | Large-scale   | 100 TB   | 10 PB     | 1 Tbps  | 100K-10M |
  | FAANG-class   | 10 PB+   | 1 EB+     | 100Tbps | 10M+     |
  +---------------+----------+-----------+---------+----------+
```

---

## 7. Latency Numbers

### 7.1 Latency Numbers Every Programmer Should Know

The "Numbers Every Programmer Should Know" compiled by Jeff Dean at Google is widely known as foundational knowledge for system design.

```
Latency comparison (2024 approximate values):

  Operation                            Time         Analogy
  ---------------------------------------------------------------
  L1 cache reference                   1 ns         One heartbeat
  Branch misprediction                 3 ns
  L2 cache reference                   4 ns
  Mutex lock/unlock                    17 ns
  L3 cache reference                   20 ns
  Main memory reference                100 ns       Blink of an eye
  Snappy compress 1 KB                 2,000 ns
  SSD random read                      16,000 ns    A sneeze
  Read 1 MB sequentially from memory   3,000 ns
  Read 1 MB sequentially from SSD      49,000 ns
  Read 1 MB sequentially from HDD      825,000 ns
  Round trip within same datacenter    500,000 ns    A deep breath
  HDD disk seek                        2,000,000 ns
  Packet CA -> NL -> CA                150,000,000 ns A nap

  Key ratios to remember:
  +-----------------------------+----------+
  | Comparison                  | Factor   |
  +-----------------------------+----------+
  | Memory vs SSD random read   | ~160x    |
  | SSD vs HDD random read      | ~125x    |
  | Memory vs HDD               | ~20,000x |
  | Within-DC vs cross-continent| ~300x    |
  | 1MB memory vs 1MB SSD       | ~16x     |
  | 1MB SSD vs 1MB HDD          | ~17x     |
  +-----------------------------+----------+
```

### 7.2 Latency Converted to "Human Scale"

```
If 1 ns = 1 second:

  L1 cache reference:       1 sec          -> One heartbeat
  L2 cache reference:       4 sec          -> One deep breath
  L3 cache reference:       20 sec         -> Waiting for elevator
  Main memory reference:    1 min 40 sec   -> Brewing a cup of tea
  SSD random read:          4 hr 26 min    -> Two movies
  HDD disk seek:            23 days        -> International trip
  Within-DC comm:           5 days 18 hr   -> A week-long business trip
  Cross-continent comm:     4 yr 9 months  -> College graduation
  5-second page load:       158 years      -> Two lifetimes

  +---------------------------------------------+
  | Lessons from this analogy:                   |
  |                                              |
  | 1. Cache miss: "seconds" become "minutes"    |
  | 2. Disk I/O: "seconds" become "days"         |
  | 3. Network call: "seconds" become "years"    |
  | 4. That's why caching and batching matter     |
  +---------------------------------------------+
```

---

## 8. Bandwidth Calculations

### 8.1 Bits per Second vs Bytes per Second

The most critical caveat with network bandwidth is that network speed is expressed in bit/s while file sizes are expressed in bytes. Getting this conversion wrong causes transfer time estimates to be off by 8x.

```
Bandwidth unit conversion:

  Fundamental: 1 byte = 8 bits
  -> 1 Gbps = 1,000,000,000 bits/s
           = 125,000,000 bytes/s
           = 125 MB/s
           = ~119 MiB/s

  Theoretical vs effective bandwidth by connection:
  +---------------+----------+----------+--------------+
  | Connection    | Theor.   | Effective| Effective    |
  |               | max      | BW       | (MB/s)       |
  +---------------+----------+----------+--------------+
  | 4G LTE        | 150 Mbps | 10-50    | 1.25-6.25    |
  | 5G Sub-6      | 1 Gbps   | 100-500  | 12.5-62.5    |
  | 5G mmWave     | 10 Gbps  | 1-4 Gbps| 125-500      |
  | WiFi 5 (ac)   | 3.5 Gbps | 200-800  | 25-100       |
  | WiFi 6 (ax)   | 9.6 Gbps | 500-2000 | 62.5-250     |
  | WiFi 7 (be)   | 46 Gbps  | 1-8 Gbps| 125-1000     |
  | GbE           | 1 Gbps   | 900 Mbps | 112          |
  | 10GbE         | 10 Gbps  | 9.5 Gbps| 1187         |
  | 25GbE         | 25 Gbps  | 24 Gbps | 3000         |
  | 100GbE        | 100 Gbps | 98 Gbps | 12250        |
  +---------------+----------+----------+--------------+

  Overhead breakdown:
  - Ethernet frame:         ~3% (preamble, IFG, CRC)
  - IP header:              20-60 bytes/packet
  - TCP header:             20-60 bytes/packet
  - TLS:                    tens of bytes + handshake
  - Application layer:      variable
  - Total overhead:         30-50% for small packets, 3-5% for large
```

### 8.2 Transfer Time Calculation

```python
"""Transfer time calculation tool"""

def transfer_time(size_bytes: int, bandwidth_mbps: float,
                  overhead_pct: float = 10.0) -> dict:
    """
    Calculate file transfer time.

    Args:
        size_bytes: File size (bytes)
        bandwidth_mbps: Bandwidth (Mbps)
        overhead_pct: Protocol overhead (%)

    Returns:
        Dictionary (seconds and human-readable format)
    """
    effective_bps = bandwidth_mbps * 1_000_000 * (1 - overhead_pct / 100)
    effective_Bps = effective_bps / 8
    seconds = size_bytes / effective_Bps

    if seconds < 60:
        human = f"{seconds:.1f} sec"
    elif seconds < 3600:
        human = f"{seconds / 60:.1f} min"
    elif seconds < 86400:
        human = f"{seconds / 3600:.1f} hr"
    else:
        human = f"{seconds / 86400:.1f} days"

    return {
        "seconds": round(seconds, 2),
        "human": human,
        "effective_MBps": round(effective_Bps / 1_000_000, 2),
    }

# --- Generate transfer time comparison table ---
files = [
    ("Web page",     2 * 10**6),
    ("1 song",       4 * 10**6),
    ("HD movie",     5 * 10**9),
    ("Game",         50 * 10**9),
    ("Backup",       1 * 10**12),
]

bandwidths = [10, 100, 1000]  # Mbps

print(f"{'File':12s}", end="")
for bw in bandwidths:
    print(f"  {bw} Mbps", end="")
print()
print("-" * 60)

for name, size in files:
    print(f"{name:12s}", end="")
    for bw in bandwidths:
        result = transfer_time(size, bw)
        print(f"  {result['human']:>10s}", end="")
    print()

# Sample output:
# File          10 Mbps  100 Mbps  1000 Mbps
# ------------------------------------------------------------
# Web page      1.8 sec    0.2 sec    0.0 sec
# 1 song        3.6 sec    0.4 sec    0.0 sec
# HD movie     44.4 min    4.4 min    0.4 min
# Game          7.4 hr    44.4 min    4.4 min
# Backup        3.1 days   7.4 hr    44.4 min
```

### 8.3 Sneakernet -- Physical Transfer Bandwidth

```
"Never underestimate the bandwidth of a station wagon full of tapes."
  -- Andrew Tanenbaum

  Physical transfer vs network transfer comparison:

  Scenario: Transfer 1 PB of data

  Via network:
    10 Gbps line: 1 PB / 1.25 GB/s = 800,000 sec = ~9.3 days
    100 Gbps line: 1 PB / 12.5 GB/s = 80,000 sec = ~22 hours

  Physical transfer:
    18TB LTO-9 tapes x 56 = ~1 PB
    Weight: ~15 kg
    Next-day delivery: 86,400 sec
    Effective bandwidth: 1 PB / 86,400 = 11.6 GB/s = ~93 Gbps

  -> Physical transport is 10x faster than a 10 Gbps line
  -> However, latency is 24 hours

  AWS physical transfer services:
  +------------------+----------+-------------------+
  | Service          | Capacity | Use case          |
  +------------------+----------+-------------------+
  | AWS Snowcone     | 8-14 TB  | Edge/IoT          |
  | AWS Snowball     | 80 TB    | Data migration    |
  | AWS Snowball Edge| 80-210TB | Edge computing    |
  | AWS Snowmobile   | 100 PB   | Exabyte-scale     |
  |                  |          | migration         |
  +------------------+----------+-------------------+

  Snowmobile is literally a 45-foot trailer equipped
  with dedicated networking equipment and power infrastructure.
```

### 8.4 Web Performance Budget

```
Web performance budget design:

  Target: Reach LCP within 3 seconds on 3G mobile (1.5 Mbps)

  Transferable data volume:
  3 sec x 1.5 Mbps / 8 = 562 KB

  Distributing this 562 KB across resources:
  +--------------+----------+--------------------------+
  | Resource     | Budget   | Notes                    |
  +--------------+----------+--------------------------+
  | HTML         | 30 KB    | After gzip               |
  | CSS          | 50 KB    | After gzip               |
  | JavaScript   | 200 KB   | After gzip (~800KB raw)  |
  | Fonts        | 80 KB    | woff2 subset             |
  | Images       | 200 KB   | WebP / AVIF              |
  | Total        | 560 KB   | <= 562KB budget          |
  +--------------+----------+--------------------------+

  Hidden cost of JavaScript:
  +------------------------------------------------------+
  | gzip 200KB -> ~800KB uncompressed                     |
  | -> Parse time (mobile CPU): 0.5-2 sec                |
  | -> Compile time: 0.2-1 sec                           |
  | -> Execution time: variable                          |
  |                                                       |
  | Total: transfer 3s + parse/exec 2-5s = 5-8s perceived|
  | -> The "true cost" of JS is not just its size         |
  +------------------------------------------------------+

  Core Web Vitals targets (Google recommended):
  +------+----------+----------+----------+
  | Metric| Good    | Needs Imp| Poor     |
  +------+----------+----------+----------+
  | LCP  | <= 2.5s  | <= 4.0s  | > 4.0s   |
  | INP  | <= 200ms | <= 500ms | > 500ms  |
  | CLS  | <= 0.1   | <= 0.25  | > 0.25   |
  +------+----------+----------+----------+
```

---

## 9. Datacenter-Scale Data

### 9.1 Data Volumes of Major Services

```
Global-scale data volumes (approximate):

  Google:
  +----------------------------------------------+
  | Search index:       100 PB+                   |
  | Gmail:              15 EB+ (3 billion users)  |
  | YouTube:            500 hrs uploaded per minute|
  |    -> 720,000 hrs of new video per day        |
  | Google Photos:      4 billion uploads per day |
  | Google Maps:        20 PB+ (satellite/street) |
  +----------------------------------------------+

  Meta:
  +----------------------------------------------+
  | Facebook + Instagram: hundreds of TB/day      |
  | Instagram:          100M+ photos per day      |
  | WhatsApp:           100B messages per day     |
  | Data warehouse:     300+ PB (Hive)            |
  +----------------------------------------------+

  Netflix:
  +----------------------------------------------+
  | Catalog total:      10 PB+ (multi-res x lang)|
  | CDN delivery:       ~15% of peak net traffic  |
  | Open Connect CDN:   Thousands of servers      |
  |                     worldwide                 |
  +----------------------------------------------+

  Global totals:
  +----------------------------------------------+
  | 2020 annual data generation: ~64 ZB           |
  | 2025 annual data generation: ~180 ZB (proj.)  |
  | 2030 annual data generation: ~600+ ZB (proj.) |
  |                                                |
  | 1 ZB = 10^21 bytes                            |
  |      = 1,000 EB                               |
  |      = 1,000,000 PB                           |
  |      = 1,000,000,000 TB                       |
  +----------------------------------------------+
```

### 9.2 Storage Cost Comparison

```
Storage cost comparison table (approximate):

  +-----------------+--------------+----------------------+
  | Media           | $/TB/month   | Primary use          |
  +-----------------+--------------+----------------------+
  | DRAM Memory     | $3,000       | Cache, in-memory DB  |
  | NVMe SSD        | $50          | Hot data, OLTP       |
  | SATA SSD        | $20          | Warm data            |
  | HDD             | $5           | Cold data, logs      |
  | S3 Standard     | $23          | Cloud hot storage    |
  | S3 IA           | $12.5        | Infrequent access    |
  | S3 Glacier      | $4           | Archive (min-hours)  |
  | S3 Deep Archive | $1           | Long-term (12hr      |
  |                 |              | restore)             |
  | LTO-9 Tape      | $0.5         | Offline archive      |
  +-----------------+--------------+----------------------+

  Cost comparison visualization (log scale):
  DRAM    |####################################| $3000
  NVMe    |##                                  | $50
  SATA SSD|#                                   | $20
  HDD     |                                    | $5
  S3 Std  |#                                   | $23
  S3 IA   |#                                   | $12.5
  Glacier |                                    | $4
  Deep Arc|                                    | $1
  Tape    |                                    | $0.5

  -> DRAM is ~600x the cost of HDD
  -> Proper tiering is key to cost optimization
```

### 9.3 Tiered Storage Design

```
Tiered Storage design principles:

  +-------------------------------------------------+
  |                                                   |
  |   Hot tier (5% of data, 80% of access)           |
  |   +-------------------------------------+        |
  |   | NVMe SSD / Redis / Memcached        |        |
  |   | Last 24 hours of data               |        |
  |   | Access frequency: per second/minute  |        |
  |   +-------------------------------------+        |
  |                                                   |
  |   Warm tier (15% of data, 15% of access)         |
  |   +-------------------------------------+        |
  |   | SATA SSD / S3 Standard / HDD        |        |
  |   | Last 30 days of data                |        |
  |   | Access frequency: hourly/daily      |        |
  |   +-------------------------------------+        |
  |                                                   |
  |   Cold tier (40% of data, 4% of access)          |
  |   +-------------------------------------+        |
  |   | S3 IA / HDD array                   |        |
  |   | Last 1 year of data                 |        |
  |   | Access frequency: a few times/month |        |
  |   +-------------------------------------+        |
  |                                                   |
  |   Archive tier (40% of data, 1% of access)       |
  |   +-------------------------------------+        |
  |   | Glacier / Tape / Deep Archive       |        |
  |   | Data older than 1 year              |        |
  |   | Access frequency: a few times/year  |        |
  |   | (disaster recovery, etc.)           |        |
  |   +-------------------------------------+        |
  |                                                   |
  +-------------------------------------------------+

  Cost optimization example:
  Retaining 100 TB of data

  All S3 Standard: 100 TB x $23/TB = $2,300/month

  With tiering:
    Hot (5TB)      x $50   = $250
    Warm (15TB)    x $23   = $345
    Cold (40TB)    x $12.5 = $500
    Archive (40TB) x $1    = $40
    Total: $1,135/month

  -> Approximately 51% cost reduction
```

---

## 10. Capacity Estimation Implementation Patterns

### 10.1 Capacity Estimation Class

```python
"""Automated capacity estimation for system design"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ServiceEstimation:
    """Class for calculating service capacity estimates"""
    name: str
    dau: int                        # Daily active users
    actions_per_user_per_day: float  # Daily actions per user
    avg_payload_bytes: int           # Average data size per action
    read_write_ratio: int = 100     # Read/write ratio
    peak_factor: float = 3.0        # Peak factor
    replication_factor: int = 3     # Number of replicas
    retention_years: int = 5        # Data retention years
    compression_ratio: float = 0.5  # Compression ratio (0.5 = 50%)

    @property
    def daily_actions(self) -> float:
        return self.dau * self.actions_per_user_per_day

    @property
    def daily_storage_bytes(self) -> float:
        return self.daily_actions * self.avg_payload_bytes

    @property
    def yearly_storage_bytes(self) -> float:
        return self.daily_storage_bytes * 365

    @property
    def total_storage_bytes(self) -> float:
        raw = self.yearly_storage_bytes * self.retention_years
        replicated = raw * self.replication_factor
        compressed = replicated * self.compression_ratio
        return compressed

    @property
    def avg_write_qps(self) -> float:
        return self.daily_actions / 86400

    @property
    def peak_write_qps(self) -> float:
        return self.avg_write_qps * self.peak_factor

    @property
    def avg_read_qps(self) -> float:
        return self.avg_write_qps * self.read_write_ratio

    @property
    def peak_read_qps(self) -> float:
        return self.avg_read_qps * self.peak_factor

    @property
    def write_bandwidth_gbps(self) -> float:
        bytes_per_sec = self.daily_storage_bytes / 86400
        return bytes_per_sec * 8 / 1e9

    def report(self) -> str:
        """Generate an estimation report"""
        def fmt(n):
            if n >= 1e18: return f"{n/1e18:.1f} EB"
            if n >= 1e15: return f"{n/1e15:.1f} PB"
            if n >= 1e12: return f"{n/1e12:.1f} TB"
            if n >= 1e9:  return f"{n/1e9:.1f} GB"
            if n >= 1e6:  return f"{n/1e6:.1f} MB"
            return f"{n:.0f} B"

        lines = [
            f"=== {self.name} Capacity Estimate ===",
            f"DAU:              {self.dau:,}",
            f"Daily actions:    {self.daily_actions:,.0f}",
            f"",
            f"--- Storage ---",
            f"Daily raw data:   {fmt(self.daily_storage_bytes)}",
            f"Yearly raw data:  {fmt(self.yearly_storage_bytes)}",
            f"Total (compr.):   {fmt(self.total_storage_bytes)}",
            f"  ({self.retention_years}yr, x{self.replication_factor} replicas,"
            f" {self.compression_ratio*100:.0f}% compression)",
            f"",
            f"--- QPS ---",
            f"Avg write QPS:    {self.avg_write_qps:,.0f}",
            f"Peak write QPS:   {self.peak_write_qps:,.0f}",
            f"Avg read QPS:     {self.avg_read_qps:,.0f}",
            f"Peak read QPS:    {self.peak_read_qps:,.0f}",
            f"",
            f"--- Bandwidth ---",
            f"Write bandwidth:  {self.write_bandwidth_gbps:.2f} Gbps",
            f"Read bandwidth:   "
            f"{self.write_bandwidth_gbps * self.read_write_ratio:.2f} Gbps",
        ]
        return "\n".join(lines)

# --- Usage examples ---
twitter = ServiceEstimation(
    name="Twitter-like service",
    dau=300_000_000,
    actions_per_user_per_day=2,
    avg_payload_bytes=600,          # Text + metadata
    read_write_ratio=100,
    peak_factor=3.0,
    replication_factor=3,
    retention_years=5,
    compression_ratio=0.5,
)
print(twitter.report())

instagram = ServiceEstimation(
    name="Instagram-like service",
    dau=500_000_000,
    actions_per_user_per_day=3,
    avg_payload_bytes=2_000_000,    # Average 2MB image
    read_write_ratio=200,
    peak_factor=2.5,
    replication_factor=3,
    retention_years=7,
    compression_ratio=0.6,
)
print("\n" + instagram.report())
```

### 10.2 Storage Cost Calculation

```python
"""Tiered storage cost optimization calculation"""

from dataclasses import dataclass, field

@dataclass
class StorageTier:
    """Storage tier definition"""
    name: str
    cost_per_tb_month: float    # $/TB/month
    data_percentage: float      # Percentage of data in this tier
    access_latency: str         # Approximate access latency

@dataclass
class StorageCostCalculator:
    """Storage cost calculator"""
    total_data_tb: float
    tiers: list = field(default_factory=list)

    def add_tier(self, tier: StorageTier):
        self.tiers.append(tier)

    def calculate(self) -> dict:
        total_cost = 0
        details = []
        for tier in self.tiers:
            tier_tb = self.total_data_tb * tier.data_percentage
            tier_cost = tier_tb * tier.cost_per_tb_month
            total_cost += tier_cost
            details.append({
                "name": tier.name,
                "data_tb": tier_tb,
                "cost_month": tier_cost,
                "latency": tier.access_latency,
            })
        return {
            "total_cost_month": total_cost,
            "total_cost_year": total_cost * 12,
            "details": details,
        }

# --- Usage example: 100TB tiered design ---
calc = StorageCostCalculator(total_data_tb=100)
calc.add_tier(StorageTier("NVMe SSD (Hot)",      50.0, 0.05, "< 1ms"))
calc.add_tier(StorageTier("S3 Standard (Warm)",   23.0, 0.15, "< 100ms"))
calc.add_tier(StorageTier("S3 IA (Cold)",         12.5, 0.40, "< 100ms"))
calc.add_tier(StorageTier("Glacier (Archive)",     1.0, 0.40, "min-hours"))

result = calc.calculate()
print(f"Monthly cost:  ${result['total_cost_month']:,.0f}")
print(f"Annual cost:   ${result['total_cost_year']:,.0f}")
for d in result["details"]:
    print(f"  {d['name']:30s} {d['data_tb']:6.1f} TB  "
          f"${d['cost_month']:8,.0f}/mo  Latency: {d['latency']}")

# Output:
# Monthly cost:  $1,135
# Annual cost:   $13,620
#   NVMe SSD (Hot)                    5.0 TB  $     250/mo  Latency: < 1ms
#   S3 Standard (Warm)               15.0 TB  $     345/mo  Latency: < 100ms
#   S3 IA (Cold)                     40.0 TB  $     500/mo  Latency: < 100ms
#   Glacier (Archive)                40.0 TB  $      40/mo  Latency: min-hours
```

---

## 11. Anti-patterns

### 11.1 Anti-pattern 1: Capacity Estimate Collapse from Unit Confusion

```
Anti-pattern: Confusing GB and GiB in estimates
==================================================

  Scenario:
  "Buy 10 x 1TB SSDs to build a 10TB storage cluster"

  Wrong calculation:
  10 drives x 1TB = 10TB -> 10,000 GB of usable capacity

  Correct calculation:
  1. Manufacturer-labeled 1TB = 1,000,000,000,000 bytes (decimal)
  2. OS display: 931 GiB (binary)
  3. Filesystem overhead: -5% -> 884 GiB
  4. SSD over-provisioning: -7% -> 822 GiB
  5. RAID 10 redundancy: 50% effective -> 411 GiB/drive
  6. 10 drives total: 4,110 GiB = ~4.01 TiB

  -> The expected "10TB" is actually ~4 TiB
  -> Estimate is off by approximately 2.5x

  Lessons:
  +---------------------------------------------------------+
  | 1. Always calculate in bytes, convert units last         |
  | 2. Distinguish manufacturer labels (decimal) from        |
  |    OS display (binary)                                   |
  | 3. Always account for FS overhead, redundancy, OP        |
  | 4. Always maintain a 20-30% margin in estimates          |
  +---------------------------------------------------------+
```

### 11.2 Anti-pattern 2: Storage Design Without Considering Growth Rate

```
Anti-pattern: Static capacity design
=====================================

  Scenario:
  "Current data is 500GB, so 1TB DB should be plenty"

  The problem:
  +-------------------------------------------------+
  | With a 10% monthly growth rate:                  |
  |                                                   |
  | Now:        500 GB                               |
  | 3 months:   665 GB                               |
  | 6 months:   885 GB                               |
  | 7 months:   974 GB  <- Approaching 1TB           |
  | 8 months:   1,071 GB <- Capacity exceeded!       |
  | 12 months:  1,569 GB                             |
  | 24 months:  4,926 GB                             |
  |                                                   |
  | -> Capacity shortage in just 8 months             |
  +-------------------------------------------------+

  The power of compound growth (Rule of 72):
  Growth rate x Doubling period = 72

  10%/month: 72 / 10 = 7.2 months to double
  5%/month:  72 / 5  = 14.4 months to double
  2%/month:  72 / 2  = 36 months to double

  Correct approach:
  +---------------------------------------------------------+
  | 1. Analyze the growth rate over the past 6-12 months     |
  | 2. Estimate capacity for at least 18 months ahead        |
  | 3. Set an alert at 70% capacity                          |
  | 4. Scale up/out at 85% capacity                          |
  | 5. Conduct regular capacity reviews (quarterly)          |
  | 6. Define a data retention policy to delete old data     |
  +---------------------------------------------------------+
```

### 11.3 Anti-pattern 3: Overlooking Bit/Byte Conversion

```
Anti-pattern: Network bandwidth unit mistake
=============================================

  Scenario:
  "We have a 1Gbps line, so we can transfer 1GB in 1 second"

  Reality:
  1 Gbps = 1,000,000,000 bits/s
         = 125,000,000 bytes/s
         = 125 MB/s

  Further considering protocol overhead (~10%):
  Effective bandwidth = ~112 MB/s

  Transfer time for a 1 GB file:
  1,000,000,000 / 112,000,000 = ~8.9 seconds

  -> Takes ~9 seconds, not 1 second (9x error)

  Lessons:
  +---------------------------------------------------------+
  | 1. Network = bits/s, Storage = bytes                     |
  | 2. Always divide by 8 when converting (bits -> bytes)    |
  | 3. Account for 10-30% protocol overhead                  |
  | 4. TCP slow start means bandwidth is low initially       |
  | 5. Shared lines compete with other traffic               |
  +---------------------------------------------------------+
```

---

## 12. Practice Exercises

### Exercise 1: Capacity Estimation (Fundamentals)

**Problem**: Design an Instagram-like photo sharing service. Calculate the storage required for 1 year given the following assumptions.

- DAU: 10 million
- Average 3 uploads/day per user
- Average size per photo: 2 MB (original)
- Thumbnails: 3 sizes (large: 200KB, medium: 50KB, small: 10KB)
- Replicas: 3x
- No compression applied

```
Solution guide:

  Step 1: Daily uploads
  10,000,000 x 3 = 30,000,000 photos/day

  Step 2: Total data per photo
  Original: 2,000 KB
  Thumbnails: 200 + 50 + 10 = 260 KB
  Total: 2,260 KB = ~2.26 MB

  Step 3: Daily data volume
  30,000,000 x 2.26 MB = 67,800,000 MB = ~67.8 TB/day

  Step 4: Annual data volume
  67.8 TB x 365 = 24,747 TB = ~24.7 PB/year

  Step 5: Apply replicas
  24.7 PB x 3 = 74.1 PB/year

  Conclusion: Approximately 74 PB/year of storage required
```

### Exercise 2: Cost Optimization (Applied)

**Problem**: Tier the storage from Exercise 1 and compare annual costs.

Conditions:
- Last 30 days: S3 Standard ($23/TB/month)
- 30-180 days: S3 IA ($12.5/TB/month)
- 180+ days: S3 Glacier ($4/TB/month)

```
Solution guide:

  1 year of raw data: 24.7 PB (before replicas)

  Data volume per tier (after 3x replicas):
  Hot (30 days):   67.8 TB x 30 x 3 = 6,102 TB = ~6.1 PB
  Warm (150 days): 67.8 TB x 150 x 3 = 30,510 TB = ~30.5 PB
  Cold (185 days): 67.8 TB x 185 x 3 = 37,629 TB = ~37.6 PB

  Monthly cost:
  Hot:    6,100 TB x $23   = $140,300
  Warm:   30,500 TB x $12.5 = $381,250
  Cold:   37,600 TB x $4    = $150,400
  Total: $671,950/month = ~$8,063,400/year

  Comparison: All S3 Standard
  74,100 TB x $23 = $1,704,300/month = ~$20,451,600/year

  -> Tiering saves ~$12,388,200/year (~60% reduction)
```

### Exercise 3: Bandwidth and Infrastructure Design (Advanced)

**Problem**: For the service in Exercise 1, design the following.

1. Peak write QPS
2. Peak read QPS (read/write ratio = 200:1)
3. Required CDN bandwidth
4. Origin bandwidth with 95% CDN cache hit rate

```
Solution guide:

  Step 1: Write QPS
  Average: 30,000,000 / 86,400 = 347 QPS
  Peak (x3): 1,041 QPS

  Step 2: Read QPS
  Average: 347 x 200 = 69,400 QPS
  Peak: 1,041 x 200 = 208,200 QPS

  Step 3: CDN bandwidth
  Assuming an average 500KB image per read:
  Peak read bandwidth:
  208,200 QPS x 500 KB = 104,100,000 KB/s
  = ~99.3 GB/s = ~794 Gbps

  Step 4: Origin bandwidth
  CDN cache hit rate 95%:
  Requests to origin: 208,200 x 0.05 = 10,410 QPS
  Origin bandwidth: 10,410 x 500 KB = ~4.97 GB/s = ~39.7 Gbps

  Infrastructure proposal:
  +-------------------------------------------------+
  | CDN:                                             |
  |   - Distributed globally                         |
  |   - Total bandwidth: 1 Tbps+                     |
  |   - Cache capacity: hundreds of TB per PoP       |
  |                                                   |
  | Origin:                                           |
  |   - Multiple regions                              |
  |   - Bandwidth: 50+ Gbps per region               |
  |   - S3 + CloudFront architecture                  |
  |                                                   |
  | Upload:                                           |
  |   - Direct S3 upload (presigned URLs)             |
  |   - Async thumbnail generation (Lambda/SQS)       |
  |   - Bandwidth: peak 1,041 x 2MB = ~16.7 Gbps    |
  +-------------------------------------------------+
```

---

## 13. FAQ

### Q1: How do you estimate "how many records fit in 1GB of RAM?"

**A**: Start with record size estimation. For example, if a user record is ID(8B) + name(100B) + email(100B) + metadata(100B) = ~300B/record, then theoretically 1 GiB / 300B = ~3.6 million records can fit. However, the following overheads must be considered:

- Indexes: consume 2-3x the memory of the record data
- Memory allocator fragmentation: 10-30% overhead
- Data structure pointers: tens of bytes per record
- Garbage collector (Java/Go, etc.): requires 1.5-2x the heap memory

Accounting for these, the realistic estimate is about 1/3 to 1/5 of the theoretical value, or approximately 700K-1.2M records.

### Q2: Which is cheaper -- cloud or on-premise storage?

**A**: It depends on scale and usage patterns. General decision criteria are:

- **Under a few TB**: Cloud is favorable. Management cost savings (personnel, facilities, power) are significant.
- **Tens to hundreds of TB**: Case by case. Depends on workload characteristics and growth rate.
- **PB scale and above**: On-premise is often more favorable. Netflix and Dropbox are examples of companies that have repatriated from cloud.

However, on-premise has hidden costs: personnel (operations team), datacenter rent/power/cooling, hardware depreciation (typically 3-5 years), network line costs, and disaster recovery facilities. When these are added, the results can differ significantly from a simple $/TB comparison.

### Q3: Should all data be retained forever?

**A**: No. Establishing a Data Retention Policy is essential for both cost and compliance reasons. Recommended retention periods are:

- **Application logs**: 90 days to 1 year (enough for incident investigation)
- **Access logs**: 1-3 years (per security audit requirements)
- **User data**: Certain period after account deletion (GDPR mandates deletion in principle)
- **Financial transaction data**: 7-10 year retention obligation by law
- **Backups**: Generation management (7 daily, 4 weekly, 12 monthly is common)
- **Metrics/monitoring data**: High-resolution for 30 days, aggregated for 1-2 years

### Q4: How does effective capacity change by RAID level?

**A**: Effective capacity per RAID level is as follows.

```
Effective capacity by RAID level (N drives, C TB each):

  +-----------+--------------+-----------+------------------+
  | RAID      | Eff. capacity| Redundancy| Use case         |
  +-----------+--------------+-----------+------------------+
  | RAID 0    | N x C        | None      | Temporary data   |
  | RAID 1    | C            | Mirror    | OS / Boot        |
  | RAID 5    | (N-1) x C    | 1 failure | Read-heavy       |
  | RAID 6    | (N-2) x C    | 2 failures| High cap/relia.  |
  | RAID 10   | N/2 x C      | Mirror    | DB / High IOPS   |
  |           |              | pairs     |                  |
  +-----------+--------------+-----------+------------------+

  Example: 10 TB HDD x 8 drives
  RAID 0:  80 TB (no redundancy -- not recommended)
  RAID 1:  10 TB (typically only 2 of 8 drives used)
  RAID 5:  70 TB
  RAID 6:  60 TB
  RAID 10: 40 TB
```

### Q5: Why do SSDs have seemingly arbitrary capacities like 120GB or 480GB?

**A**: SSD NAND chips are manufactured in powers of 2 (128 GiB, 256 GiB, 512 GiB). After subtracting the over-provisioning (OP) area, the remaining capacity is rounded in decimal notation for the manufacturer label.

Example: 128 GiB NAND chip
- 7% OP: 128 GiB x 0.93 = 119 GiB = ~128 GB -> sold as "120 GB"
- 12% OP: 128 GiB x 0.88 = 112.6 GiB = ~121 GB -> premium model sold as "128 GB"

OP is a reserved area used for wear leveling, bad block management, garbage collection, and TRIM processing, and is essential for maintaining SSD lifespan and performance.

### Q6: How should data compression ratios be estimated?

**A**: Compression ratios vary significantly by data type. General guidelines are:

```
Compression ratio estimates by data type:

  +--------------------+-----------+----------------------+
  | Data type          | Ratio     | Notes                |
  +--------------------+-----------+----------------------+
  | Plain text         | 60-80%    | High compression     |
  |                    |           | with gzip/zstd       |
  | JSON / XML         | 70-90%    | High redundancy,     |
  |                    |           | compresses well      |
  | Log files          | 80-95%    | Many repeating       |
  |                    |           | patterns             |
  | Source code        | 60-75%    | A type of text       |
  | Database dumps     | 70-85%    | Structural           |
  |                    |           | redundancy           |
  | JPEG / PNG         | 0-5%     | Already compressed   |
  | MP3 / AAC          | 0-3%     | Already compressed   |
  | H.264 / H.265     | 0-2%     | Already compressed   |
  | Encrypted data     | 0%       | Cannot compress      |
  | Random data        | 0%       | Maximum entropy      |
  +--------------------+-----------+----------------------+

  Compression ratio = (1 - compressed size / original size) x 100%

  Important: Already compressed or encrypted data may
  actually increase in size with additional compression.
```

---

## 14. Edge Cases and Caveats

### 14.1 File System Size Limits

```
Size limits of major file systems:

  +----------+-----------------+-----------------+----------+
  | FS       | Max file size   | Max volume      | Notes    |
  +----------+-----------------+-----------------+----------+
  | FAT32    | 4 GiB - 1       | 2 TiB           | USB etc  |
  | exFAT    | 16 EiB          | 128 PiB         | SD/USB   |
  | NTFS     | 16 TiB          | 256 TiB         | Windows  |
  | ext4     | 16 TiB          | 1 EiB           | Linux    |
  | XFS      | 8 EiB           | 8 EiB           | Linux    |
  | Btrfs    | 16 EiB          | 16 EiB          | Linux    |
  | ZFS      | 16 EiB          | 256 ZiB (theory)| Various  |
  | APFS     | 8 EiB           | No limit (prac.)| macOS    |
  +----------+-----------------+-----------------+----------+

  Common issues:
  - Cannot save files > 4GB on FAT32
    -> Solution: Reformat to exFAT
  - Default inode count insufficient on ext4 (many small files)
    -> Solution: Adjust inode density with -i option during mkfs
```

### 14.2 Database Size Estimation

```
RDB storage estimation considerations:

  1. Table data body
     Row count x average bytes per row

  2. Indexes
     B-tree index: 20-100% of table size
     Covering indexes: even larger
     Rule of thumb: total indexes = table size x 1.5-3x

  3. WAL / Redo logs
     PostgreSQL: wal_size parameter (default 1GB)
     MySQL: innodb_log_file_size x innodb_log_files_in_group

  4. MVCC / Vacuum
     PostgreSQL: bloats with dead tuples under high update frequency
     Maximum: can bloat to 2-5x table size

  5. TOAST / LOB
     Large column values are stored in a separate table

  Estimation formula:
  Required storage = Table data
                     x (1 + index multiplier)
                     x (1 + MVCC bloat factor)
                     x (1 + WAL/temp files)
                     + replica storage

  Example: 100GB table data
  = 100 GB x 2.5 (indexes) x 1.3 (MVCC) x 1.1 (WAL)
  = 357.5 GB (per node)
  x 3 (replicas) = 1,072.5 GB = ~1 TB
```

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is paramount. Understanding deepens not just through theory, but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping fundamentals and jumping to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge used in practice?

Knowledge of this topic is frequently used in daily development work. It becomes particularly important during code reviews and architecture design.

---

## 15. Summary

### 15.1 Key Concept Overview

| Concept | Key Point |
|---------|-----------|
| Bit | The smallest unit of information. Represents 2 states (0/1) |
| Byte | 8 bits. The smallest addressable unit |
| Binary prefixes | KiB(2^10), MiB(2^20), GiB(2^30) -- powers of 1024 |
| Decimal prefixes | KB(10^3), MB(10^6), GB(10^9) -- powers of 1000 |
| Text | 1 CJK char (UTF-8) = ~3B. Short book = ~300KB |
| Images | Web photo = ~100-500KB. RAW = 25-50MB |
| Video | 1080p 1 min = ~25-45MB (codec dependent) |
| Estimation | DAU x actions x data size x replicas x retention |
| Cost | RAM: $3000/TB/mo -> SSD: $50 -> HDD: $5 -> Tape: $0.5 |
| Bandwidth | Network = bits/s, Storage = bytes. Divide by 8 |

### 15.2 Cheat Sheet: Memorize for Quick Estimation

```
Instantly usable approximations:

  Time:
    1 day = ~10^5 sec (86,400)
    1 year = ~3 x 10^7 sec (31,536,000)

  Powers of 2:
    2^10 = ~10^3 (1,024)
    2^20 = ~10^6 (1,048,576)
    2^30 = ~10^9 (1,073,741,824)
    2^40 = ~10^12

  Data sizes:
    1 English letter = 1 B
    1 CJK char (UTF-8) = 3 B
    Web image = 100-500 KB
    MP3 1 min = ~1 MB
    HD video 1 min = ~25-45 MB

  Bandwidth:
    1 Gbps = 125 MB/s (divide by 8)
    With overhead, ~100-110 MB/s

  QPS:
    1M DAU x 1 action/day = ~12 QPS
    Peak = average x 2-3
```

---

## 16. Recommended Next Reading


---

## Recommended Next Reading

- [Brain vs Computer Comparison](./06-brain-vs-computer.md) - Proceed to the next topic

---

## References

1. Shannon, C. E. "A Mathematical Theory of Communication." Bell System Technical Journal, 1948. -- The historic paper that laid the foundation of information theory. Formalized the concept of the bit.
2. IEC 60027-2: "Letter symbols to be used in electrical technology -- Part 2: Telecommunications and electronics." International Electrotechnical Commission, 1998 (amended 2005). -- The international standard that formally defined binary prefixes (KiB, MiB, GiB, etc.).
3. Dean, J. and Barroso, L. A. "The Tail at Scale." Communications of the ACM, Vol. 56, No. 2, 2013. -- Google's infrastructure design philosophy. Widely cited as the source for latency numbers.
4. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly Media, 2017. -- Comprehensive coverage of data system design principles. The capacity planning chapters are particularly useful.
5. Xu, A. "System Design Interview -- An Insider's Guide." Byte Code LLC, 2020. -- Chapter 2 "Back-of-the-envelope Estimation" provides a systematic methodology for capacity estimation.
6. Patterson, D. A. and Hennessy, J. L. "Computer Organization and Design: The Hardware/Software Interface." 6th Edition, Morgan Kaufmann, 2020. -- Textbook-level coverage of memory hierarchy and storage technology.
7. AWS Documentation. "Amazon S3 Storage Classes." https://aws.amazon.com/s3/storage-classes/ -- Official reference for specific costs and characteristics in cloud storage tier design.
