# Motherboard and Bus

> The motherboard is the "nervous system" of the computer, governing communication between all components.

## Learning Objectives

- [ ] Describe the major components of a motherboard and their roles
- [ ] Understand the evolution of bus architecture
- [ ] Explain each stage of the boot process
- [ ] Understand PCIe detailed specifications and lane allocation
- [ ] Master USB specification evolution and practical selection criteria
- [ ] Explain the differences from server architecture

## Prerequisites


---

## 1. Motherboard Components

```
Motherboard Layout (Conceptual Diagram):

  ┌──────────────────────────────────────────────────┐
  │  ┌──────────┐          ┌──────────────────────┐ │
  │  │ CPU      │←────────→│ Memory Slots         │ │
  │  │ Socket   │ Memory Bus│ DIMM1 DIMM2 DIMM3  │ │
  │  └────┬─────┘          └──────────────────────┘ │
  │       │                                          │
  │       │ PCIe x16                                 │
  │       ▼                                          │
  │  ┌──────────────────────────────────┐            │
  │  │       PCH (Platform Controller Hub)│           │
  │  │  ┌─────────────────────────────┐  │           │
  │  │  │ PCIe x4  → NVMe SSD Slot   │  │           │
  │  │  │ PCIe x16 → GPU Slot        │  │           │
  │  │  │ SATA     → HDD/SSD         │  │           │
  │  │  │ USB 3.x/4.0 Controller     │  │           │
  │  │  │ Ethernet Controller        │  │           │
  │  │  │ Audio Controller           │  │           │
  │  │  │ Wi-Fi/Bluetooth            │  │           │
  │  │  └─────────────────────────────┘  │           │
  │  └──────────────────────────────────┘            │
  │                                                   │
  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
  │  │ BIOS/UEFI│  │ Power    │  │ I/O Ports    │  │
  │  │ (SPI     │  │ Connector│  │ USB, HDMI,   │  │
  │  │  Flash)  │  │ (ATX)    │  │ Ethernet...  │  │
  │  └──────────┘  └──────────┘  └──────────────┘  │
  └──────────────────────────────────────────────────┘
```

### 1.1 Detailed Breakdown of Major Motherboard Components

```
Details of Each Component:

  ■ CPU Socket
    - Intel LGA (Land Grid Array): Pins on the socket side
      LGA1700 (12th-14th Gen), LGA1851 (Arrow Lake)
    - AMD PGA (Pin Grid Array): Pins on the CPU side (up to AM4)
    - AMD LGA: Transitioned to LGA starting from AM5
    - Server: LGA4094 (AMD SP5), LGA4677 (Intel)

    Socket Compatibility:
    ┌───────────────────────────────────────────┐
    │ Platform         │ Socket   │ Generation    │
    │──────────────────│──────────│───────────────│
    │ Intel Desktop    │ LGA1700  │ 12th-14th Gen │
    │ Intel Desktop    │ LGA1851  │ Arrow Lake+   │
    │ AMD Desktop      │ AM4      │ Ryzen 1000-5000│
    │ AMD Desktop      │ AM5      │ Ryzen 7000+   │
    │ Intel Server     │ LGA4677  │ Sapphire Rapids+│
    │ AMD Server       │ SP5      │ EPYC 9004+    │
    └───────────────────────────────────────────┘

  ■ Memory Slots (DIMM Slots)
    - Typically 2 or 4 slots (desktop)
    - 8-12 slots per CPU on servers
    - DDR5: 288-pin, dual-channel (32 bits per channel)
    - DIMM Types:
      UDIMM: Unbuffered (desktop)
      RDIMM: Registered (server, ECC-capable)
      LRDIMM: Load-Reduced (high-capacity servers)
      SO-DIMM: For laptops (compact form factor)

  ■ VRM (Voltage Regulator Module)
    - Supplies stable voltage to the CPU
    - More phases means greater stability (high-end motherboards have 16-20 phases)
    - Especially important during overclocking
    - MOSFET quality determines VRM quality

    VRM Configuration:
    ┌──────────────────────────────────────────┐
    │ 12V (ATX PSU) → VRM → 1.1V (CPU VCore)  │
    │                                           │
    │ ┌─────┐ ┌─────┐ ┌─────┐    ┌──────┐  │
    │ │Phase│ │Phase│ │Phase│... │ CPU   │  │
    │ │ 1   │ │ 2   │ │ 3   │    │       │  │
    │ └─────┘ └─────┘ └─────┘    └──────┘  │
    │ PWM controller alternates each phase     │
    │ → Stabilizes current, distributes heat   │
    └──────────────────────────────────────────┘

  ■ SPI Flash (BIOS/UEFI ROM)
    - Capacity: 16-32MB (UEFI + microcode)
    - Connected via SPI (Serial Peripheral Interface) bus
    - The first chip read upon power-on
    - Dual BIOS configurations include 2 chips (for fault tolerance)
```

### 1.2 Form Factors

```
Motherboard Form Factors:

  ┌────────────────────────────────────────────────────┐
  │ ATX (305 x 244 mm)                                 │
  │ ┌──────────────────────────────────────────────┐  │
  │ │                                              │  │
  │ │  PCIe x16 x 2-3                             │  │
  │ │  M.2 Slots x 2-4                            │  │
  │ │  DIMM x 4                                    │  │
  │ │  SATA x 4-8                                  │  │
  │ │  USB Headers x multiple                      │  │
  │ │                                              │  │
  │ └──────────────────────────────────────────────┘  │
  │ → Most common, highest expandability               │
  └────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────┐
  │ Micro-ATX (244 x 244 mm)                │
  │ ┌──────────────────────────────────┐    │
  │ │  PCIe x16 x 1-2                  │    │
  │ │  M.2 x 1-2                       │    │
  │ │  DIMM x 2-4                      │    │
  │ │  SATA x 4-6                      │    │
  │ └──────────────────────────────────┘    │
  │ → Cost-effective, moderate size          │
  └──────────────────────────────────────────┘

  ┌────────────────────────────────┐
  │ Mini-ITX (170 x 170 mm)       │
  │ ┌──────────────────────┐      │
  │ │  PCIe x16 x 1        │      │
  │ │  M.2 x 1-2           │      │
  │ │  DIMM x 2            │      │
  │ │  SATA x 2-4          │      │
  │ └──────────────────────┘      │
  │ → Small PCs, HTPCs             │
  └────────────────────────────────┘

  Server Form Factors:
  ┌────────────────────────────────────────────────────────┐
  │ E-ATX (305 x 330 mm)                                   │
  │ → Dual-socket support, DIMM x 8-16                     │
  │                                                         │
  │ EEB (305 x 330 mm)                                     │
  │ → Server standard, numerous PCIe slots                  │
  │                                                         │
  │ OCP (Open Compute Project)                              │
  │ → Open standard for data centers                        │
  └────────────────────────────────────────────────────────┘
```

---

## 2. Bus Types and Evolution

### 2.1 Bus History

| Standard | Era | Bandwidth | Characteristics |
|------|------|--------|------|
| ISA | 1981 | 8 MB/s | Early IBM PC bus |
| PCI | 1992 | 133 MB/s | Shared bus, Plug & Play |
| AGP | 1997 | 2.1 GB/s | GPU-dedicated bus |
| PCI Express 1.0 | 2003 | 250 MB/s/lane | Point-to-point, lane-based |
| PCIe 2.0 | 2007 | 500 MB/s/lane | 2x bandwidth |
| PCIe 3.0 | 2010 | 985 MB/s/lane | 128b/130b encoding |
| PCIe 4.0 | 2017 | 1,969 MB/s/lane | Standard for NVMe SSDs |
| PCIe 5.0 | 2019 | 3,938 MB/s/lane | Server, high-end |
| PCIe 6.0 | 2022 | 7,877 MB/s/lane | PAM4, FEC required |
| PCIe 7.0 | 2025 | 15,754 MB/s/lane | Under development |

### 2.2 PCIe Structure

```
PCIe Lane Configurations:

  PCIe x1:  ──→  1 lane  =  3.9 GB/s (PCIe 5.0)
  PCIe x4:  ────→ 4 lanes = 15.8 GB/s (NVMe SSD)
  PCIe x8:  ────────→ 8 lanes  = 31.5 GB/s
  PCIe x16: ────────────────→ 16 lanes = 63.0 GB/s (GPU)

  Each lane is an independent transmit/receive pair (differential signaling):
  ┌──────┐         ┌──────┐
  │ CPU  │ ──TX──→ │ GPU  │  Transmit
  │      │ ←──RX── │      │  Receive
  └──────┘         └──────┘
  → Full-duplex communication (simultaneous send/receive)
```

### 2.3 PCIe Technical Details

```
PCIe Protocol Layers:

  ┌─────────────────────────────────────┐
  │ Transaction Layer (TLP)             │
  │ - Memory read/write, I/O, config   │
  │ - Packet-based communication       │
  │ - Flow control (credit-based)      │
  ├─────────────────────────────────────┤
  │ Data Link Layer (DLLP)             │
  │ - Error detection via CRC          │
  │ - Retransmission via ACK/NAK      │
  │ - Flow control information exchange│
  ├─────────────────────────────────────┤
  │ Physical Layer                      │
  │ - Differential signal pairs        │
  │ - Encoding                         │
  │   PCIe 1.0-2.0: 8b/10b (20% overhead)     │
  │   PCIe 3.0-5.0: 128b/130b (1.5% overhead) │
  │   PCIe 6.0-7.0: PAM4 + FEC        │
  │ - Lane widths: x1, x2, x4, x8, x16│
  └─────────────────────────────────────┘

Bandwidth Calculation by PCIe Generation:

  PCIe 3.0 x4 (NVMe SSD):
    Transfer rate: 8 GT/s x 4 lanes = 32 GT/s
    Encoding: 128b/130b
    Effective bandwidth: 32 x (128/130) / 8 = 3.938 GB/s
    → Approximately 3.9 GB/s (unidirectional)

  PCIe 5.0 x16 (GPU):
    Transfer rate: 32 GT/s x 16 lanes = 512 GT/s
    Encoding: 128b/130b
    Effective bandwidth: 512 x (128/130) / 8 = 63.0 GB/s
    → Approximately 63 GB/s (unidirectional), 126 GB/s bidirectional

  PCIe 6.0 x16:
    Transfer rate: 64 GT/s x 16 lanes = 1024 GT/s
    Modulation: PAM4 (2 bits/symbol)
    FEC overhead: approximately 3%
    Effective bandwidth: approximately 121 GB/s (unidirectional)
```

### 2.4 PCIe Lane Allocation Examples

```
Intel 14th Gen (Raptor Lake) PCIe Lane Allocation:

  CPU-Direct Lanes (total 20 lanes + 4 DMI):
  ┌─────────────────────────────────────────┐
  │ CPU                                      │
  │ ├── PCIe 5.0 x16 → GPU                  │
  │ ├── PCIe 4.0 x4  → M.2 SSD (1st)       │
  │ └── DMI 4.0 x4   → PCH                  │
  └─────────────────────────────────────────┘

  PCH (Z790) Lanes (total 28 lanes):
  ┌─────────────────────────────────────────┐
  │ PCH (Z790)                               │
  │ ├── PCIe 4.0 x4 → M.2 SSD (2nd)        │
  │ ├── PCIe 3.0 x4 → M.2 SSD (3rd)        │
  │ ├── PCIe 3.0 x16 → Expansion Slots      │
  │ ├── SATA x 8                             │
  │ ├── USB 3.2 x 5                          │
  │ ├── USB 2.0 x 14                        │
  │ ├── Ethernet                             │
  │ └── Audio, Wi-Fi, etc.                   │
  └─────────────────────────────────────────┘

  Note: PCH lanes are shared resources
  → Using an M.2 SSD may disable certain SATA ports
  → Check the motherboard manual for bandwidth sharing details

AMD Ryzen 7000 (AM5) PCIe Lane Allocation:

  CPU-Direct Lanes (total 28 lanes + 4 GMI):
  ┌─────────────────────────────────────────┐
  │ CPU                                      │
  │ ├── PCIe 5.0 x16 → GPU                  │
  │ ├── PCIe 5.0 x4  → M.2 SSD (1st)       │
  │ ├── PCIe 4.0 x4  → M.2 SSD (2nd)       │
  │ ├── USB4 x 2                             │
  │ └── GMI → Chipset                        │
  └─────────────────────────────────────────┘
  → AMD provides more CPU-direct lanes, GPU bifurcation (x8+x8) also possible
```

### 2.5 PCIe Power Delivery

```
PCIe Slot Power Delivery Capability:

  │ Slot     │ PCIe 3.0 │ PCIe 4.0 │ PCIe 5.0 │ PCIe 6.0 │
  │──────────│──────────│──────────│──────────│──────────│
  │ x1       │ 10W      │ 10W      │ 10W      │ 10W      │
  │ x4       │ 25W      │ 25W      │ 25W      │ 25W      │
  │ x8       │ 25W      │ 25W      │ 25W      │ 25W      │
  │ x16      │ 75W      │ 75W      │ 75W      │ 75W      │

  Additional GPU Power Supply:
  ┌─────────────────────────────────────────────┐
  │ Connector           │ Power   │ Use Case     │
  │───────────────────│─────────│────────────────│
  │ PCIe slot only      │ 75W    │ Low-end GPU    │
  │ + 6-pin x1          │ 150W   │ Mid-range      │
  │ + 8-pin x1          │ 225W   │ High-end       │
  │ + 8-pin x2          │ 375W   │ RTX 3090, etc. │
  │ 12VHPWR (600W)      │ 675W   │ RTX 4090       │
  │ 12V-2x6 (600W)      │ 675W   │ RTX 50 series  │
  └─────────────────────────────────────────────┘

  12VHPWR Connector (PCIe 5.0 Power Connector):
  - 16-pin (12-pin power + 4-pin sense)
  - Capable of delivering up to 600W
  - Cable connection issues causing melting have been reported (Gen5-era challenge)
```

---

## 3. USB Standards

### 3.1 USB Standard Comparison

| Standard | Year | Speed | Power Delivery | Connector |
|------|-----|------|---------|---------|
| USB 1.1 | 1998 | 12 Mbps | 2.5W | Type-A/B |
| USB 2.0 | 2000 | 480 Mbps | 2.5W | Type-A/B |
| USB 3.0 | 2008 | 5 Gbps | 4.5W | Type-A (blue)/B |
| USB 3.1 Gen2 | 2013 | 10 Gbps | 100W (PD) | Type-C |
| USB 3.2 Gen2x2 | 2017 | 20 Gbps | 100W (PD) | Type-C |
| USB4 v1 | 2019 | 40 Gbps | 100W (PD) | Type-C |
| USB4 v2 | 2022 | 80 Gbps | 240W (EPR) | Type-C |
| Thunderbolt 5 | 2024 | 120 Gbps | 240W | Type-C |

### 3.2 USB Type-C Unification

```
USB Type-C Connector (24-pin):

  ┌─────────────────────────────────┐
  │ ● ● ● ● ● ● ● ● ● ● ● ● │ Upper row: 12 pins
  │ ● ● ● ● ● ● ● ● ● ● ● ● │ Lower row: 12 pins
  └─────────────────────────────────┘

  Reversible: Can be inserted in either orientation
  Unified: USB, Thunderbolt, DisplayPort, and power delivery in a single cable

  Important caveat: Type-C connector ≠ USB4
  → Cables with the same Type-C appearance may only support USB 2.0 speeds
  → Always verify cable/device specifications
```

### 3.3 USB Power Delivery (PD) Details

```
USB PD Voltage and Current Combinations:

  USB PD 3.1 (SPR: Standard Power Range):
  │ Voltage │ Max Current │ Max Power │ Use Case            │
  │─────────│─────────────│───────────│─────────────────────│
  │ 5V      │ 3A          │ 15W       │ Smartphone charging │
  │ 9V      │ 3A          │ 27W       │ Tablet charging     │
  │ 15V     │ 3A          │ 45W       │ Thin laptops        │
  │ 20V     │ 3A          │ 60W       │ Standard laptops    │
  │ 20V     │ 5A          │ 100W      │ High-performance laptops │

  USB PD 3.1 (EPR: Extended Power Range):
  │ Voltage │ Max Current │ Max Power │ Use Case            │
  │─────────│─────────────│───────────│─────────────────────│
  │ 28V     │ 5A          │ 140W      │ Gaming laptops      │
  │ 36V     │ 5A          │ 180W      │ Mobile workstations │
  │ 48V     │ 5A          │ 240W      │ High-performance devices │

  PD Negotiation:
  ┌──────────┐                    ┌──────────┐
  │ Charger  │ ── CC Line ──→   │ Device   │
  │ (Source) │ ← USB PD Message │ (Sink)   │
  └──────────┘                    └──────────┘

  1. Device connects to charger
  2. Communication begins over CC line (Configuration Channel)
  3. Charger advertises supported voltages/currents (Source Capabilities)
  4. Device requests desired voltage/current (Request)
  5. Charger approves → Voltage switches
  → Fully automatic, no user action required
```

### 3.4 USB Internal Protocol

```
USB Data Transfer Types:

  ■ Control Transfer
    - Device configuration, status retrieval
    - Bidirectional, small size
    - Used by all USB devices

  ■ Bulk Transfer
    - Large data transfers (storage, printers)
    - No bandwidth guarantee, error correction included
    - Utilizes available bandwidth to the maximum

  ■ Isochronous Transfer
    - Real-time data (audio, video)
    - Bandwidth guaranteed, no error correction
    - Prioritizes continuity over latency

  ■ Interrupt Transfer
    - Small periodic data (keyboard, mouse)
    - Polling interval guaranteed
    - Low latency

USB 3.0+ Physical Layer:
  ┌─────────────────────────────────────────┐
  │ USB 2.0 pair (D+/D-):  480 Mbps        │ ← Backward compatible
  │ USB 3.0 TX pair:       5 Gbps          │ ← Added
  │ USB 3.0 RX pair:       5 Gbps          │ ← Added
  └─────────────────────────────────────────┘
  → USB 3.0+ includes USB 2.0 signal lines as well (maintaining compatibility)
  → Type-C cables add additional pins including CC, SBU, VBUS, etc.
```

---

## 4. Boot Process

### 4.1 From Power-On to OS Boot

```
Complete Boot Process Stages:

  ┌──────────────┐
  │ 1. Power On   │ ← PSU sends Power Good signal
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 2. Reset      │ ← CPU jumps to reset vector (0xFFFFFFF0)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 3. BIOS/UEFI │ ← Firmware loaded from SPI flash
  │    Init      │    CPU cache used as temporary RAM (CAR)
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 4. POST      │ ← Power-On Self Test
  │  (Self-Test) │    Memory detection, device init, error checking
  └──────┬───────┘    Beep codes for errors (no memory = continuous beep, etc.)
         ▼
  ┌──────────────┐
  │ 5. Boot      │ ← Searches for boot device (NVMe→USB→Network)
  │    Device    │    UEFI: Searches for ESP (EFI System Partition)
  │    Selection │    BIOS: Reads first 512 bytes of MBR
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 6. Boot      │ ← GRUB, systemd-boot, Windows Boot Manager, etc.
  │    Loader    │    Loads kernel image and initramfs into memory
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 7. Kernel    │ ← Hardware initialization, driver loading
  │    Init      │    Root filesystem mount
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 8. init/     │ ← systemd (PID 1) starts services
  │    systemd   │    Network, logging, GUI, etc.
  └──────┬───────┘
         ▼
  ┌──────────────┐
  │ 9. Login     │ ← System ready for user interaction
  └──────────────┘

  Duration: A few seconds (NVMe + UEFI + SSD) to several minutes (HDD + BIOS)
```

### 4.2 BIOS vs UEFI

| Item | BIOS (Legacy) | UEFI |
|------|-------------|------|
| Established | 1975 (IBM PC) | 2007 (Intel-led) |
| Interface | Text-based | GUI support |
| Boot Driver | 16-bit | 64-bit |
| Partitioning | MBR (max 2TB) | GPT (max 8ZB) |
| Security | None | Secure Boot |
| Boot Speed | Slow | Fast |
| Networking | None | PXE boot standard |

### 4.3 UEFI Details

```
UEFI Boot Detailed Flow:

  1. SEC (Security Phase)
     - CPU initialization (microcode application)
     - Temporary RAM setup (CAR: Cache As RAM)
     - Security verification begins

  2. PEI (Pre-EFI Initialization)
     - Memory controller initialization
     - DRAM training (timing optimization)
     - Physical RAM becomes available

  3. DXE (Driver Execution Environment)
     - Device driver loading
     - Protocol initialization
     - PCI/PCIe device enumeration

  4. BDS (Boot Device Selection)
     - Boot option enumeration
     - ESP (EFI System Partition) discovery
     - Boot via user selection or default

  5. TSL (Transient System Load)
     - Boot loader execution
     - OS kernel loading

  6. RT (Runtime)
     - UEFI runtime services remain available during OS execution
     - Clock, variable store, power management, etc.

ESP (EFI System Partition) Structure:
  /boot/efi/ (FAT32, typically 100-500MB)
  ├── EFI/
  │   ├── BOOT/
  │   │   └── BOOTX64.EFI    ← Default boot entry
  │   ├── ubuntu/
  │   │   └── grubx64.efi    ← Ubuntu's GRUB
  │   ├── Microsoft/
  │   │   └── Boot/
  │   │       └── bootmgfw.efi ← Windows Boot Manager
  │   └── fedora/
  │       └── shimx64.efi    ← Fedora (Secure Boot compatible)
  └── ...

Secure Boot:
  ┌──────────────────────────────────────────────┐
  │ Chain of Trust:                                │
  │                                                │
  │ Platform Key (PK)                              │
  │   └── Key Exchange Key (KEK)                   │
  │       └── db (database of allowed signatures)  │
  │           └── Verify boot loader signature     │
  │               └── Verify kernel signature      │
  │                                                │
  │ → Unsigned code cannot execute at boot time    │
  │ → Early detection of malware                   │
  └──────────────────────────────────────────────┘
```

```bash
# UEFI-related commands (Linux)

# List boot entries
efibootmgr -v

# Change boot order
sudo efibootmgr -o 0001,0002,0003

# Add a new boot entry
sudo efibootmgr -c -d /dev/nvme0n1 -p 1 \
    -l /EFI/ubuntu/grubx64.efi -L "Ubuntu"

# Check Secure Boot status
mokutil --sb-state

# Display UEFI variables
ls /sys/firmware/efi/efivars/

# Analyze boot time with systemd
systemd-analyze
systemd-analyze blame | head -20
systemd-analyze plot > boot.svg

# Check boot log with dmesg
dmesg | head -100
```

---

## 5. Chipset Architecture

### 5.1 History of Evolution

```
Legacy (2000s era):
  ┌─────┐   FSB    ┌───────────┐   ┌──────┐
  │ CPU │←────────→│Northbridge │──→│ GPU  │
  └─────┘          │(MCH)       │   └──────┘
                   └──────┬────┘
                          │
                   ┌──────┴────┐
                   │Southbridge │──→ USB, SATA, Audio
                   │(ICH)       │
                   └───────────┘

Modern (2020s):
  ┌──────────────────────┐
  │        CPU            │
  │  ┌──────────────────┐│
  │  │ Memory Controller ││──→ DDR5 RAM
  │  │ PCIe Controller   ││──→ GPU (PCIe x16)
  │  │                    ││──→ NVMe (PCIe x4)
  │  └──────────────────┘│
  └──────────┬───────────┘
             │ DMI 4.0 (~8GB/s)
             ▼
  ┌──────────────────────┐
  │   PCH (Platform      │
  │   Controller Hub)    │──→ USB, SATA, Audio
  │                      │──→ Wi-Fi, Ethernet
  │                      │──→ Additional PCIe lanes
  └──────────────────────┘

  Evolution: Northbridge functions integrated into CPU
  → Faster memory access (bus bottleneck eliminated)
  → Lower latency GPU connection
```

### 5.2 Intel vs AMD Chipset Comparison

```
Intel Z790 vs AMD X670E Chipset Comparison:

  │ Feature             │ Z790         │ X670E         │
  │─────────────────────│──────────────│───────────────│
  │ CPU-PCH Link        │ DMI 4.0 x4   │ GMI (proprietary) │
  │ CPU-direct PCIe 5.0 │ x16 + x4     │ x16 + x4 + x4│
  │ CPU-direct PCIe 4.0 │ x4           │ None          │
  │ PCH PCIe 4.0        │ x12          │ x12           │
  │ PCH PCIe 3.0        │ x16          │ x8            │
  │ USB 3.2 Gen2x2      │ 5            │ 6             │
  │ USB4                 │ None         │ 2 (CPU-direct)│
  │ SATA                 │ 8            │ 8             │
  │ DDR Support          │ DDR4/DDR5    │ DDR5 only     │
  │ OC Support           │ Supported    │ Supported     │

  AMD X670E uses a dual-chip chipset configuration:
  ┌──────────┐     ┌──────────┐
  │ Promontory│────│ Promontory│
  │ Chip 1    │    │ Chip 2    │
  └──────────┘     └──────────┘
  → Provides more I/O but increases power consumption
```

### 5.3 DMI (Direct Media Interface) Bottleneck

```
Understanding the DMI Bottleneck:

  CPU-direct PCIe: High bandwidth, low latency
  PCH-connected devices: DMI caps the bandwidth

  DMI 4.0 x4 bandwidth:
  = PCIe 4.0 x4 = approximately 8 GB/s

  When all PCH-attached devices access simultaneously:
  NVMe SSD (via PCH):   max 3.5 GB/s
  + USB 3.2 Gen2 x2:    2.5 GB/s
  + 2.5G Ethernet:       0.3 GB/s
  + SATA SSD x2:         1.0 GB/s
  Total: 7.3 GB/s → Fits within DMI bandwidth but with little headroom

  Mitigation:
  - Choose CPU-direct connections for high-bandwidth devices (GPU, primary NVMe)
  - PCH-connected NVMe SSDs should be secondary storage
  - Be cautious when adding network cards to PCIe slots
```

---

## 6. Memory Bus and DDR

### 6.1 DDR Generation Comparison

```
DDR Memory Generation Comparison:

  │ Standard │ Year  │ Transfer Rate │ Bandwidth(1ch)│ Voltage │
  │──────────│───────│───────────────│───────────────│─────────│
  │ DDR3     │ 2007  │ 800-2133      │ 17 GB/s       │ 1.5V    │
  │ DDR4     │ 2014  │ 2133-5333     │ 42.7 GB/s     │ 1.2V    │
  │ DDR5     │ 2020  │ 4800-8800     │ 70.4 GB/s     │ 1.1V    │
  │ DDR5 OC  │ 2024  │ 9200+         │ 73.6 GB/s     │ 1.35V   │

  Key DDR5 Improvements:
  ┌───────────────────────────────────────────────┐
  │ DDR4:                                          │
  │ ┌──────────────────────────────────────────┐  │
  │ │ 1 channel x 64-bit width                 │  │
  │ │ Burst length: 8                           │  │
  │ │ Bank groups: 4                            │  │
  │ │ PMIC: On motherboard                      │  │
  │ └──────────────────────────────────────────┘  │
  │                                                │
  │ DDR5:                                          │
  │ ┌──────────────────────────────────────────┐  │
  │ │ 2 sub-channels x 32-bit width            │  │
  │ │ Burst length: 16                          │  │
  │ │ Bank groups: 8                            │  │
  │ │ PMIC: On-DIMM (voltage regulation on module)│ │
  │ │ On-die ECC: Built-in error correction     │  │
  │ └──────────────────────────────────────────┘  │
  │                                                │
  │ → 2 sub-channels improve bandwidth efficiency  │
  │ → On-module PMIC provides cleaner power supply │
  └───────────────────────────────────────────────┘

Dual-Channel vs Single-Channel:
  Single-channel: 1 DIMM = bandwidth x 1
  Dual-channel:   2 DIMMs = bandwidth x 2
  Quad-channel:   4 DIMMs = bandwidth x 4 (server/HEDT)
  Octa-channel:   8 DIMMs = bandwidth x 8 (server)

  → When using an iGPU, dual-channel can nearly double frame rates
  → Always configure in dual-channel (2 or 4 sticks)
```

---

## 7. Latest Trends

### 7.1 CXL (Compute Express Link)

```
CXL's 3 Protocols:

  CXL.io    — PCIe-based device I/O (legacy compatible)
  CXL.cache — Device caches host memory
  CXL.mem   — Host accesses device memory

  Use Cases:
  ┌──────┐        CXL        ┌──────────────────┐
  │ CPU  │←──────────────────→│ CXL Memory       │
  └──────┘                    │ Expansion        │
                              │ (DRAM expansion, │
                              │  persistent memory)│
                              └──────────────────┘
  → Share RAM across servers (memory pooling)
  → Enables previously impossible TB-scale memory spaces

CXL Versions:
  │ Version    │ Year │ Bandwidth  │ Key Features                 │
  │────────────│──────│────────────│──────────────────────────────│
  │ CXL 1.1    │ 2020 │ PCIe 5.0   │ Basic memory expansion       │
  │ CXL 2.0    │ 2022 │ PCIe 5.0   │ Memory pooling               │
  │ CXL 3.0    │ 2023 │ PCIe 6.0   │ Multi-level switching        │
  │ CXL 3.1    │ 2024 │ PCIe 6.0   │ TSP (security enhancement)   │

CXL in Data Centers:
  Conventional: Fixed DRAM per server
  ┌──────┐ ┌──────┐ ┌──────┐
  │128GB │ │256GB │ │64GB  │  ← Memory waste occurs
  │(50%  │ │(30%  │ │(90%  │
  │used) │ │used) │ │used) │
  └──────┘ └──────┘ └──────┘

  CXL: Shared memory pool
  ┌──────┐ ┌──────┐ ┌──────┐
  │CPU 1 │ │CPU 2 │ │CPU 3 │
  └──┬───┘ └──┬───┘ └──┬───┘
     │        │        │
  ┌──┴────────┴────────┴──┐
  │   CXL Memory Pool      │
  │   Total: 448GB         │
  │   Each CPU uses only   │
  │   what it needs        │
  └────────────────────────┘
  → Significant improvement in memory utilization (TCO reduction)
```

### 7.2 Chiplet Architecture

| Approach | Description | Example |
|-----------|------|-----|
| Monolithic | Single large die | Intel Core (older generations) |
| Chiplet | Multiple small dies connected | AMD EPYC, Apple M2 Ultra |
| UCIe | Standard for inter-chiplet communication | Co-developed by Intel, AMD, ARM |

> Improved yield, mixed process nodes, flexible scaling.

```
Chiplet Interconnect Technologies:

  ■ AMD Infinity Fabric
    ┌────────┐  IF  ┌────────┐
    │ CCD 0  │────→│ CCD 1  │   CCD = Core Complex Die
    │ 8 cores│←────│ 8 cores│
    └───┬────┘     └───┬────┘
        │              │
    ┌───┴──────────────┴───┐
    │      IOD (I/O Die)    │
    │  Memory Controller    │
    │  PCIe Controller      │
    └───────────────────────┘
    → CCD: 5nm (cutting-edge), IOD: 6nm (cheaper) can coexist

  ■ Apple UltraFusion
    ┌────────────────┐  UF  ┌────────────────┐
    │ M2 Max Die 1   │────→│ M2 Max Die 2   │
    │                │←────│                │
    │ GPU 38 cores   │      │ GPU 38 cores   │
    │ CPU 12 cores   │      │ CPU 12 cores   │
    └────────────────┘      └────────────────┘
    Bandwidth: 2.5 TB/s (silicon interposer)
    → Two M2 Max dies used as one M2 Ultra

  ■ UCIe (Universal Chiplet Interconnect Express)
    - Industry-standard chiplet interconnect specification
    - Enables combining chiplets from different vendors
    - Bandwidth: up to 256 GB/s/mm
    - Participants include Intel, AMD, ARM, Samsung, TSMC, etc.
```

---

## 8. Server Architecture

### 8.1 Differences Between Servers and Desktops

```
Server Motherboard Characteristics:

  │ Feature             │ Desktop         │ Server               │
  │─────────────────────│─────────────────│──────────────────────│
  │ CPU Sockets         │ 1               │ 1-2 (dual-socket)    │
  │ Memory Slots        │ 2-4             │ 8-24                 │
  │ Memory Type         │ UDIMM           │ RDIMM/LRDIMM (ECC)  │
  │ Max Memory          │ 128GB           │ 4-6TB                │
  │ PCIe Slots          │ 1-3             │ 6-10                 │
  │ Networking          │ 1GbE-2.5GbE     │ 10/25/100GbE         │
  │ Storage             │ M.2 x 2-3       │ U.2/E1.S x 24+      │
  │ Remote Management   │ None            │ IPMI/BMC/iLO/iDRAC   │
  │ Redundant PSU       │ None            │ Yes (hot-swappable)  │
  │ ECC Memory          │ Typically none  │ Required             │

  IPMI/BMC (Baseboard Management Controller):
  ┌────────────────────────────────────────┐
  │ BMC Chip                               │
  │ - Independent CPU (ARM Cortex, etc.)   │
  │ - Independent network connection       │
  │ - Server management even when OS down  │
  │ - Power ON/OFF                         │
  │ - KVM (Keyboard, Video, Mouse)         │
  │ - Firmware updates                     │
  │ - Hardware monitoring (temp, voltage, fans) │
  │ - Serial console                       │
  └────────────────────────────────────────┘
  → Server management without physical access in data centers
```

### 8.2 NUMA Architecture

```
NUMA (Non-Uniform Memory Access):

  Dual-Socket Server Configuration:
  ┌─────────────────────────────────────────────┐
  │ NUMA Node 0               NUMA Node 1       │
  │ ┌─────────┐               ┌─────────┐     │
  │ │ CPU 0   │               │ CPU 1   │     │
  │ │ 32 cores│──── QPI/UPI ──│ 32 cores│     │
  │ └────┬────┘               └────┬────┘     │
  │      │                         │           │
  │ ┌────┴────┐               ┌────┴────┐     │
  │ │ DDR5    │               │ DDR5    │     │
  │ │ 256GB   │               │ 256GB   │     │
  │ │ Local   │               │ Local   │     │
  │ └─────────┘               └─────────┘     │
  └─────────────────────────────────────────────┘

  Memory Access Latency:
  - Local memory:  80ns
  - Remote memory: 130ns (approximately 1.6x slower)

  → OS and applications must use NUMA-aware memory placement for optimal performance
  → NUMA can be controlled via the numactl command
```

```bash
# Check NUMA information
numactl --hardware

# Run a process on NUMA Node 0
numactl --cpunodebind=0 --membind=0 ./my_application

# Check NUMA balancing
cat /proc/sys/kernel/numa_balancing

# NUMA memory allocation status
numastat -m
```

---

## 9. Hands-On Exercises

### Exercise 1: Reading Specifications (Fundamentals)

Investigate the specs of your own PC/Mac and identify the following:
1. CPU socket/chip type
2. Memory standard (DDR4/DDR5), channel count, and bandwidth
3. Storage interface (NVMe/SATA)
4. USB port types and speeds

### Exercise 2: Bottleneck Analysis (Intermediate)

Identify potential bottlenecks in the following system:
- CPU: AMD Ryzen 9 7950X (16 cores)
- RAM: DDR5-5200 64GB (dual-channel)
- GPU: NVIDIA RTX 4090 (PCIe 4.0 x16)
- Storage: Samsung 990 Pro 2TB (PCIe 4.0 x4)
- Workload: 4K video editing + AI training

### Exercise 3: Observing the Boot Process (Advanced)

Observe the UEFI boot process on a Linux machine:
```bash
# Check boot log
journalctl -b | head -100

# Check UEFI boot variables
efibootmgr -v

# Enumerate PCIe devices
lspci -vvv | head -50

# Enumerate USB devices
lsusb -t
```

### Exercise 4: PCIe Lane Allocation Design (Intermediate)

Design a PCIe lane allocation for a system with the following requirements:
- GPU: RTX 4090 (x16 required)
- NVMe SSD: 2TB x 2 drives (x4 each)
- 10GbE NIC: x4
- Thunderbolt 4 card: x4
- Platform: Intel Z790

Discuss how to resolve lane shortages and the associated trade-offs.

### Exercise 5: Server Configuration Design (Advanced)

Design a server for the following requirements:
- Purpose: PostgreSQL database server
- CPU: Dual-socket preferred
- Memory: 512GB minimum (ECC required)
- Storage: NVMe SSD RAID 10
- Network: 25GbE x 2 (redundant)
- Budget: Within 3 million JPY

Design Items:
1. CPU/platform selection
2. Memory configuration (DIMM count, channel allocation)
3. Storage configuration (drive count, RAID)
4. Network configuration
5. Redundancy strategy


---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|--------|------|--------|
| Initialization error | Configuration file issues | Verify config file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Implement batch processing, add pagination |
| Permission error | Insufficient access rights | Verify executing user permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Implement locking mechanisms, manage transactions |

### Debugging Procedure

1. **Check error messages**: Read the stack trace to identify the point of failure
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Formulate hypotheses**: List possible causes
4. **Verify step by step**: Use logging output or a debugger to verify hypotheses
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

Steps for diagnosing performance issues:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Verify disk and network I/O status
4. **Check connection count**: Verify connection pool status

| Problem Type | Diagnostic Tool | Solution |
|-----------|-----------|------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Proper reference release |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

A summary of decision criteria for technology selection:

| Criterion | When to Prioritize | When Compromise is Acceptable |
|---------|------------|-------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
┌─────────────────────────────────────────────────┐
│          Architecture Selection Flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  (1) Team size?                                 │
│    ├─ Small (1-5 people) → Monolith             │
│    └─ Large (10+ people) → Go to (2)            │
│                                                 │
│  (2) Deployment frequency?                       │
│    ├─ Once a week or less → Monolith + modular  │
│    └─ Daily / multiple times → Go to (3)        │
│                                                 │
│  (3) Inter-team independence?                    │
│    ├─ High → Microservices                       │
│    └─ Moderate → Modular monolith                │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Costs**
- A short-term fast approach may become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and delays the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Adopting diverse technologies enables best-fit solutions but increases operational costs

**3. Level of Abstraction**
- High abstraction improves reusability but can make debugging harder
- Low abstraction is intuitive but leads to code duplication

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
        """Describe background and challenges"""
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
- Focus on the minimum viable feature set
- Automated tests for critical paths only
- Introduce monitoring early

**Lessons Learned:**
- Do not pursue perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt deliberately

### Scenario 2: Legacy System Modernization

**Situation:** Incrementally modernizing a system that has been in operation for 10+ years

**Approach:**
- Gradual migration using the Strangler Fig pattern
- Create Characterization Tests first if existing tests are absent
- Coexist old and new systems via an API gateway
- Execute data migration incrementally

| Phase | Work Content | Estimated Duration | Risk |
|---------|---------|---------|--------|
| 1. Investigation | Current state analysis, dependency mapping | 2-4 weeks | Low |
| 2. Foundation | CI/CD setup, test environment | 4-6 weeks | Low |
| 3. Migration Start | Sequential migration from peripheral features | 3-6 months | Medium |
| 4. Core Migration | Migration of core features | 6-12 months | High |
| 5. Completion | Legacy system decommission | 2-4 weeks | Medium |

### Scenario 3: Large-Team Development

**Situation:** 50+ engineers developing the same product

**Approach:**
- Define clear boundaries using Domain-Driven Design
- Assign ownership per team
- Manage shared libraries via Inner Source
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

### Scenario 4: Performance-Critical System

**Situation:** A system requiring millisecond-level response times

**Optimization Points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Leverage asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization Technique | Impact | Implementation Cost | Applicable Scenario |
|-----------|------|-----------|---------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Async processing | Medium | Medium | I/O-heavy operations |
| DB optimization | High | High | Slow queries |
| Code optimization | Low-Medium | High | CPU-bound cases |

---

## Team Development Practices

### Code Review Checklist

Key points to verify during code reviews related to this topic:

- [ ] Naming conventions are consistent
- [ ] Error handling is appropriate
- [ ] Test coverage is sufficient
- [ ] There is no negative performance impact
- [ ] There are no security issues
- [ ] Documentation is updated

### Knowledge Sharing Best Practices

| Method | Frequency | Audience | Impact |
|------|------|------|------|
| Pair programming | As needed | Complex tasks | Immediate feedback |
| Tech talks | Weekly | Entire team | Horizontal knowledge spread |
| ADR (Decision Records) | As needed | Future members | Decision transparency |
| Retrospectives | Biweekly | Entire team | Continuous improvement |
| Mob programming | Monthly | Critical design | Consensus building |

### Technical Debt Management

```
Priority Matrix:

        Impact High
          │
    ┌─────┼─────┐
    │ Plan │ Act  │
    │ for  │ on   │
    │ later│ now  │
    ├─────┼─────┤
    │ Log  │ Next │
    │ only │Sprint│
    │      │      │
    └─────┼─────┘
          │
        Impact Low
    Frequency Low  Frequency High
```

---

## Security Considerations

### Common Vulnerabilities and Countermeasures

| Vulnerability | Risk Level | Countermeasure | Detection Method |
|--------|------------|------|---------|
| Injection attacks | High | Input validation, parameterized queries | SAST/DAST |
| Authentication flaws | High | Multi-factor auth, session management hardening | Penetration testing |
| Sensitive data exposure | High | Encryption, access control | Security audit |
| Misconfiguration | Medium | Security headers, principle of least privilege | Configuration scanning |
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
- [ ] Sensitive information is not written to logs
- [ ] HTTPS is enforced
- [ ] CORS policy is properly configured
- [ ] Dependency vulnerability scanning is performed
- [ ] Error messages do not contain internal information

---

## Migration Guide

### Notes on Version Upgrades

| Version | Key Changes | Migration Work | Impact Scope |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API design overhaul | Endpoint changes | All clients |
| v2.x → v3.x | Authentication method change | Token format update | Auth-related |
| v3.x → v4.x | Data model change | Run migration scripts | DB-related |

### Step-by-Step Migration Procedure

```python
# Migration script template
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """Step-by-step migration execution engine"""

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
        """Execute migrations (upgrade)"""
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
        """Rollback migrations"""
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

1. **Data backup**: Take a complete backup before migration
2. **Test environment verification**: Validate in a production-equivalent environment beforehand
3. **Gradual rollout**: Deploy incrementally using canary releases
4. **Enhanced monitoring**: Shorten metrics monitoring intervals during migration
5. **Clear criteria**: Define rollback decision criteria in advance
---

## FAQ

### Q1: How should I choose a motherboard?

**A**: Decide in the following order:
1. CPU socket (Intel LGA1700, AMD AM5, etc.)
2. Chipset (feature differences: PCIe lane count, USB count, overclocking support)
3. Form factor (ATX/mATX/Mini-ITX)
4. Number of memory slots and DDR generation
5. Number of M.2/NVMe slots
6. Expandability (PCIe slots, USB, networking)

### Q2: What is the difference between Thunderbolt and USB4?

**A**: Thunderbolt 4/5 is a superset of USB4:
- USB4: Guarantees minimum 20Gbps
- Thunderbolt 4: Guarantees 40Gbps + DP 2.0 + PCIe tunneling
- Thunderbolt 5: 80-120Gbps + 240W power delivery

All use the Type-C connector, but actual performance depends on the cable and device capabilities.

### Q3: Why is PCIe measured in "lanes"?

**A**: For flexible scaling. The number of lanes can be adjusted to match the device's bandwidth requirements:
- NVMe SSD: x4 is sufficient (~8GB/s)
- GPU: x16 for maximum bandwidth (~32GB/s)
- Wi-Fi card: x1 is sufficient (~1GB/s)
The total number of PCIe lanes on a motherboard is determined by the CPU + chipset, and allocation can be configured in the BIOS.

### Q4: How much faster is DDR5 compared to DDR4?

**A**: Bandwidth is approximately 1.5-2x higher, but latency is comparable or slightly worse:
- DDR4-3200: 25.6GB/s bandwidth, CL16 = 10ns
- DDR5-5600: 44.8GB/s bandwidth, CL36 = 12.86ns
- Bandwidth-intensive workloads (video editing, AI) favor DDR5
- Latency-sensitive workloads (gaming) show a smaller difference from DDR4

### Q5: How can you evaluate VRM quality?

**A**: Check the following points:
- Phase count: 12+ phases desirable (16+ for overclocking)
- MOSFET: DrMOS (high-efficiency integrated type) is ideal
- Heatsink: Large heatsinks over the VRM area
- PWM controller: High-quality chips from Renesas/Infineon
- Thermography test results from review sites

### Q6: Is it safe to disable Secure Boot?

**A**: For general Linux use, many distributions already support Secure Boot. Cases where disabling may be necessary:
- Using a custom kernel
- Unsigned drivers (certain NVIDIA drivers, etc.)
- Special dual-boot configurations
From a security standpoint, keeping it enabled is recommended. In enterprise environments, it is often mandatory.

---

## Summary

| Concept | Key Point |
|------|---------|
| Motherboard | The foundation connecting CPU, memory, and PCH (chipset) |
| PCIe | Point-to-point, lane-based, bandwidth doubles per generation |
| USB | Trending toward Type-C unification, accelerated by USB4/Thunderbolt |
| Boot | Power → UEFI → POST → Boot Loader → Kernel → init |
| Evolution | Northbridge → CPU integration, CXL, chiplets |
| DDR5 | 2 sub-channels, improved bandwidth, on-die ECC |
| NUMA | In dual-socket systems, memory placement directly impacts performance |

---

## Recommended Next Guides


---

## References

1. PCI-SIG. "PCI Express Base Specification." Various Revisions.
2. USB Implementers Forum. "Universal Serial Bus Specification." Various Revisions.
3. UEFI Forum. "Unified Extensible Firmware Interface Specification."
4. Intel. "Platform Controller Hub (PCH) Datasheets."
5. CXL Consortium. "Compute Express Link Specification." https://www.computeexpresslink.org/
6. JEDEC. "DDR5 SDRAM Standard (JESD79-5)." 2020.
7. AMD. "AMD EPYC 9004 Series Architecture." Whitepaper.
8. UCIe Consortium. "Universal Chiplet Interconnect Express Specification." 2022.
9. Intel. "12th Gen Intel Core Processor Datasheet." Volume 1.
10. AMD. "AMD Ryzen 7000 Series Platform Technology." 2022.
