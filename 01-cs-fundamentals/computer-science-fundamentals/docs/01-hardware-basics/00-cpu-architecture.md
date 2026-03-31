# CPU Architecture

> The CPU is the computational heart of a computer, repeating the cycle of "Fetch -> Decode -> Execute -> Write Back" billions of times per second. A systematic understanding of modern processor operation -- from ISA design philosophies to cache hierarchies, branch prediction, and out-of-order execution -- forms the foundation for writing high-performance software.

## Learning Objectives

- [ ] Explain the CPU instruction cycle in four stages
- [ ] Understand pipeline processing and its limitations (hazards)
- [ ] Explain the differences in design philosophy between CISC and RISC
- [ ] Understand cache hierarchy mechanisms and cache-friendly programming techniques
- [ ] Explain the principles of branch prediction and the countermeasures programmers should take
- [ ] Understand the basics of superscalar execution, out-of-order execution, and register renaming
- [ ] Explain multi-core and SMT parallel processing models
- [ ] Understand the characteristics of RISC-V and the significance of an open ISA

## Prerequisites

- A conceptual understanding of basic logic gates (AND, OR, NOT) is helpful

---

## 1. Basic Structure of the CPU

### 1.1 Major CPU Components

The CPU is broadly composed of four major components: the "Control Unit," the "Arithmetic Logic Unit (ALU)," the "Register File," and the "Cache." These work closely together to process program instructions at high speed.

```
┌─────────────────────────────────────────────────────────────────┐
│                           CPU                                    │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Control Unit (CU)                        │ │
│  │                                                            │ │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐  │ │
│  │  │Instruction│  │ Program  │  │ Instruction Decoder    │  │ │
│  │  │ Register  │  │ Counter  │  │                        │  │ │
│  │  │ (IR)      │  │ (PC)     │  │ Opcode → Control Signal│  │ │
│  │  └──────────┘  └──────────┘  └────────────────────────┘  │ │
│  │                                                            │ │
│  │  ┌────────────────────────────────────────────────────┐   │ │
│  │  │ Microsequencer: Timing coordination of ctrl signals │   │ │
│  │  └────────────────────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────┐    ┌──────────────────────────────────┐│
│  │  ALU               │    │  Register File                   ││
│  │  (Arithmetic Logic │    │  ┌────┐ ┌────┐ ┌────┐ ┌────┐  ││
│  │   Unit)            │    │  │ R0 │ │ R1 │ │ R2 │ │ R3 │  ││
│  │                     │    │  └────┘ └────┘ └────┘ └────┘  ││
│  │  ┌───────────────┐ │    │  ┌────┐ ┌────┐ ┌────┐ ┌────┐  ││
│  │  │ Integer Adder │ │    │  │ R4 │ │ R5 │ │ R6 │ │ R7 │  ││
│  │  │ Integer Mult. │ │    │  └────┘ └────┘ └────┘ └────┘  ││
│  │  │ Divider       │ │    │  ...                            ││
│  │  │ Bit Shifter   │ │    │  x86-64: 16 GP + many special   ││
│  │  │ Logic Unit    │ │    │  AArch64: 31 GP + SP + ZR       ││
│  │  │ Comparator    │ │    │                                  ││
│  │  └───────────────┘ │    │  ┌──────────────────────────┐   ││
│  │                     │    │  │ Flags Register (RFLAGS)  │   ││
│  │  ┌───────────────┐ │    │  │ ZF: Zero Flag            │   ││
│  │  │ FPU/SIMD      │ │    │  │ CF: Carry Flag           │   ││
│  │  │ (Floating     │ │    │  │ OF: Overflow Flag        │   ││
│  │  │  Point/Vector)│ │    │  │ SF: Sign Flag            │   ││
│  │  └───────────────┘ │    │  └──────────────────────────┘   ││
│  └────────────────────┘    └──────────────────────────────────┘│
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │  Cache Hierarchy                                           │ │
│  │  L1i: 32-64KB (Instruction Cache)  Latency: ~4 cycles     │ │
│  │  L1d: 32-64KB (Data Cache)         Latency: ~4 cycles     │ │
│  │  L2:  256KB-1MB (Unified)          Latency: ~12 cycles    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
└──────────────────────────────┬───────────────────────────────────┘
                               │ Bus (Memory Bus / Ring Bus)
                               ▼
                    ┌──────────────────┐
                    │  L3 Cache        │  Shared / 8-96MB
                    │  Latency:        │  ~40 cycles
                    │  ~40 cycles      │
                    └────────┬─────────┘
                             │
                    ┌────────────────┐
                    │  Main Memory   │  DDR5 / ~100 cycles
                    │  (DRAM)        │
                    └────────────────┘
```

### 1.2 Roles of Each Component

| Component | Role | Access Speed | Typical Capacity |
|-------------|------|------------|-----------|
| **ALU** | Performs arithmetic (add, subtract, multiply, divide) and logic (AND/OR/NOT/XOR) operations | 1 cycle (simple operations) | -- |
| **FPU** | A dedicated unit for floating-point arithmetic | 1-5 cycles | -- |
| **SIMD Unit** | Processes multiple data elements in parallel with a single instruction (SSE, AVX, NEON, etc.) | 1-3 cycles | -- |
| **Control Unit** | Decodes instructions and controls execution, sends control signals to other components | -- | -- |
| **Registers** | Ultra-fast storage inside the CPU (accessible in 1 clock cycle) | ~0.3ns | x86: 16, ARM: 31 |
| **PC (Program Counter)** | Holds the address of the next instruction to execute | -- | -- |
| **IR (Instruction Register)** | Holds the currently executing instruction | -- | -- |
| **L1 Cache** | Fastest cache. Split into instruction (L1i) and data (L1d) | ~1ns | 32-64KB |
| **L2 Cache** | Per-core intermediate cache | ~3-4ns | 256KB-1MB |
| **L3 Cache** | Cache shared across all cores | ~10ns | 8-96MB |

### 1.3 Detailed Register Classification

Registers are the fastest storage region within the CPU and come in several types depending on their purpose.

```
x86-64 Architecture Register Layout:

General-Purpose Registers (64-bit x 16):
┌──────────────────────────────────────────────────────────┐
│  RAX (Accumulator)          │  RBX (Base)                │
│  RCX (Counter)              │  RDX (Data)                │
│  RSI (Source Index)         │  RDI (Destination Index)   │
│  RSP (Stack Pointer)        │  RBP (Base Pointer)        │
│  R8  through R15 (Additional General-Purpose)            │
└──────────────────────────────────────────────────────────┘

SIMD / Floating-Point Registers:
┌──────────────────────────────────────────────────────────┐
│  XMM0-XMM15  (128-bit SSE)                              │
│  YMM0-YMM15  (256-bit AVX)                              │
│  ZMM0-ZMM31  (512-bit AVX-512)                          │
└──────────────────────────────────────────────────────────┘

Control / Status Registers:
┌──────────────────────────────────────────────────────────┐
│  RIP (Instruction Pointer = Program Counter)             │
│  RFLAGS (Flags Register: ZF, CF, OF, SF, PF, AF)        │
│  CR0-CR4 (Control Registers: Paging, Protected Mode, etc.)│
│  CS, DS, SS, ES, FS, GS (Segment Registers)             │
└──────────────────────────────────────────────────────────┘

AArch64 (ARM 64-bit) Architecture Register Layout:
┌──────────────────────────────────────────────────────────┐
│  X0-X30  (64-bit General-Purpose Registers x 31)        │
│  XZR     (Zero Register: always returns 0)              │
│  SP      (Stack Pointer)                                │
│  PC      (Program Counter: not directly writable)       │
│  V0-V31  (128-bit SIMD/FP Registers x 32)              │
│  NZCV    (Condition Flags: Negative, Zero, Carry, oVerflow)│
└──────────────────────────────────────────────────────────┘
```

### 1.4 Partial Register Access (x86-64)

For historical reasons, x86-64 allows access to lower portions of 64-bit registers using alternative names.

```
RAX (64-bit):
┌────────────────────────────────────────────────────────────────┐
│ 63                              32 31              16 15  8 7 0│
│                                   │      EAX         │ AH │AL │
│                                   │                  │  AX    │
│                  RAX                                          │
└────────────────────────────────────────────────────────────────┘

Example:
  RAX = 0x00000000_AABBCCDD
  EAX = 0xAABBCCDD   (lower 32 bits)
  AX  = 0xCCDD       (lower 16 bits)
  AH  = 0xCC         (upper 8 bits of AX)
  AL  = 0xDD         (lower 8 bits of AX)

Note: Writing to EAX zero-extends the upper 32 bits of RAX (x86-64 specification)
      Writing to AX does not modify the upper bits of RAX
```

---

## 2. Instruction Set Architecture (ISA)

### 2.1 What Is an ISA?

The Instruction Set Architecture (ISA) is a specification that defines the interface between software and hardware. An ISA specifies the following elements:

- **Instruction types and formats**: What operations are available and how instructions are encoded
- **Data types**: Supported integer and floating-point widths
- **Register configuration**: Number, width, and purpose of general-purpose registers
- **Memory model**: Addressing modes, endianness, alignment
- **Interrupts and exceptions**: How hardware interrupts and traps are handled
- **Privilege levels**: Distinction between user mode and kernel mode

As long as the ISA is identical, the same binary can run on different microarchitectures (specific hardware implementations). For example, any program can execute on both Intel Core and AMD Ryzen processors as long as both conform to the x86-64 ISA.

### 2.2 Components of an Instruction

A typical machine instruction consists of the following elements:

```
Instruction Format (Conceptual Diagram):

┌──────────┬──────────┬──────────┬──────────┐
│  Opcode  │ Operand1 │ Operand2 │ Operand3 │
│ (Op Type)│ (Dest)   │ (Source1)│ (Source2)│
└──────────┴──────────┴──────────┴──────────┘

Example: ADD R1, R2, R3
     │    │    │    │
     │    │    │    └── Operand3: Source register 2
     │    │    └─────── Operand2: Source register 1
     │    └──────────── Operand1: Destination register
     └───────────────── Opcode: Indicates addition
```

### 2.3 Addressing Modes

The method by which the CPU determines where to obtain an operand's value is called an "addressing mode."

| Mode | Notation (x86) | Description | Use Case |
|--------|-------------|------|------|
| **Immediate** | `mov rax, 42` | Value is embedded in the instruction itself | Setting constants |
| **Register** | `mov rax, rbx` | Uses the value in a register | Variable-to-variable assignment |
| **Direct** | `mov rax, [0x1000]` | Uses the value at a fixed memory address | Global variables |
| **Register Indirect** | `mov rax, [rbx]` | Uses the value at the memory address pointed to by a register | Pointer dereference |
| **Base+Offset** | `mov rax, [rbp-8]` | Base register + constant offset | Local variables |
| **Indexed** | `mov rax, [rbx+rcx*8]` | Base + index x scale | Array access |

```nasm
; Addressing mode examples (x86-64)

; Immediate addressing: Set the constant 42 in RAX
mov rax, 42                ; RAX = 42

; Register addressing: Copy the value of RBX to RAX
mov rax, rbx               ; RAX = RBX

; Register indirect: Load the value at the memory address pointed to by RBX into RAX
mov rax, [rbx]             ; RAX = Memory[RBX]

; Base+Offset: Access a local variable on the stack
mov rax, [rbp-8]           ; RAX = Memory[RBP - 8]

; Indexed: Array element access (8-byte elements)
; Accessing array[i]: RBX=array base, RCX=index
mov rax, [rbx+rcx*8]       ; RAX = Memory[RBX + RCX * 8]

; RIP-relative: Relative address from the current instruction position (position-independent code)
mov rax, [rip+0x1234]      ; RAX = Memory[RIP + 0x1234]
```

### 2.4 Instruction Classification

Instructions provided by an ISA are broadly classified into the following categories:

```
Instruction Classification:

1. Data Transfer Instructions
   ├── Register-to-register: MOV, MOVZX, MOVSX
   ├── Load:     LDR (ARM), MOV reg,[mem] (x86)
   └── Store:    STR (ARM), MOV [mem],reg (x86)

2. Arithmetic Instructions
   ├── Integer:   ADD, SUB, MUL, DIV, INC, DEC, NEG
   └── Floating:  FADD, FSUB, FMUL, FDIV (x87)
                   ADDSS, MULSD (SSE/AVX)

3. Logic Instructions
   ├── Bitwise:  AND, OR, XOR, NOT
   └── Shift:    SHL, SHR, SAR, ROL, ROR

4. Comparison & Branch Instructions
   ├── Compare:     CMP, TEST
   ├── Conditional: JE, JNE, JG, JL, JGE, JLE, ...
   ├── Unconditional: JMP, CALL, RET
   └── Conditional Execution: CMOV (x86), Condition Suffixes (ARM)

5. SIMD / Vector Instructions
   ├── SSE:     ADDPS, MULPS, SHUFPS, ...
   ├── AVX:     VADDPS, VMULPS, VFMADD, ...
   └── NEON:    FADD, FMUL (ARM)

6. System Instructions
   ├── Privileged:      INT, SYSCALL, HLT, WBINVD
   ├── Synchronization: LOCK, MFENCE, LFENCE, SFENCE
   └── Virtualization:  VMXON, VMLAUNCH (Intel VT-x)
```

---

## 3. Instruction Cycle

### 3.1 Basic Four Stages

The CPU processes every instruction through the following four stages (or five stages). This is the most fundamental model; modern CPUs subdivide further (described later), but this four-stage model is essential for conceptual understanding.

```
Instruction Cycle (4-Stage Model):

  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────┐
  │  Fetch   │───→│  Decode  │───→│ Execute  │───→│ Write Back  │
  └─────────┘    └─────────┘    └─────────┘    └─────────────┘
       │               │              │               │
  Read the         Analyze the    Perform ALU     Store the
  instruction from instruction to  computation or  result in a
  memory at the    identify the   memory access   register or
  PC address       opcode and                     memory
                   operands
```

**Details of Each Stage**:

1. **Fetch**: Reads the instruction from the memory address indicated by the Program Counter (PC) and stores it in the Instruction Register (IR). Simultaneously updates the PC to the next instruction address.
2. **Decode**: Analyzes the bit pattern of the instruction stored in IR, identifying the opcode (operation type) and operands (target data). The values of required registers are also read at this stage.
3. **Execute**: The ALU performs the actual computation. If memory access is needed, address calculation is also done here. In a 5-stage pipeline, a separate Memory access stage follows.
4. **Write Back**: Writes the computation result to the destination register or memory. Flag register updates also occur at this stage.

### 3.2 Five-Stage Pipeline Model

In the classic five-stage pipeline (e.g., MIPS architecture), the Execute stage is further split into "computation" and "memory access."

```
5-Stage Pipeline:

  ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
  │  IF   │──→│  ID   │──→│  EX   │──→│  MEM  │──→│  WB   │
  │ Instr │   │ Instr │   │ Exec  │   │Memory │   │ Write │
  │ Fetch │   │Decode/│   │       │   │Access │   │ Back  │
  │       │   │Reg    │   │       │   │       │   │       │
  │       │   │ Read  │   │       │   │       │   │       │
  └───────┘   └───────┘   └───────┘   └───────┘   └───────┘

  IF: Instruction Fetch    - Fetch instruction from instruction memory
  ID: Instruction Decode   - Decode instruction and read register values
  EX: Execute              - Perform computation in the ALU
  MEM: Memory Access       - Read/write data memory (Load/Store only)
  WB: Write Back           - Write result to register
```

### 3.3 Concrete Example: Assembly Instruction Execution Trace

```nasm
; x86-64 assembly: Adding two numbers
; Equivalent to a = b + c in C

; Instruction 1: Load the value of b into RAX
mov rax, [rbp-8]
; IF: Fetch the byte sequence of this instruction from PC=0x401000
; ID: Decoded as MOV instruction. Source=[RBP-8], Destination=RAX
; EX: Address calculation RBP - 8 = 0x7FFFE000
; MEM: Read value from Memory[0x7FFFE000]
; WB: Store the read value into RAX

; Instruction 2: Add the value of c to RAX
add rax, [rbp-16]
; IF: Fetch this instruction from the next PC
; ID: Decoded as ADD instruction. Source=RAX and [RBP-16], Destination=RAX
; EX: Address calculation RBP - 16
; MEM: Read the value of c from memory
; WB: Store result of RAX + c in RAX, update flags

; Instruction 3: Store the result in a
mov [rbp-24], rax
; IF: Fetch this instruction
; ID: Decoded as MOV (store) instruction. Source=RAX, Destination=[RBP-24]
; EX: Address calculation RBP - 24
; MEM: Write the value of RAX to Memory[RBP-24]
; WB: No register write-back
```

### 3.4 Pipeline Depth in Modern CPUs

In actual commercial CPUs, pipelines are far deeper than 5 stages. More stages make it easier to increase clock frequency, but the branch misprediction penalty also increases.

| CPU | Era | Pipeline Stages | Notes |
|-----|------|----------------|------|
| MIPS R2000 | 1985 | 5 stages | Classic textbook pipeline |
| Intel Pentium III | 1999 | 10 stages | |
| Intel Pentium 4 (Prescott) | 2004 | 31 stages | Too deep; large branch miss penalty |
| Intel Core (Skylake) | 2015 | 14-19 stages | Return to moderate depth |
| Apple M1 (Firestorm) | 2020 | ~13 stages | High-IPC design |
| AMD Zen 4 | 2022 | ~19 stages | |

**Lesson**: The Pentium 4's NetBurst architecture adopted a 31-stage deep pipeline, pushing clock frequency to 3.8GHz, but the branch misprediction penalty was severe and IPC suffered. The successor Core architecture returned to approximately 14 stages and significantly improved IPC. This is a historic example demonstrating that "clock frequency alone does not determine performance."

---

## 4. Pipeline Processing

### 4.1 Basic Concept

Pipeline processing improves throughput (number of instructions completed per unit time) by temporally overlapping the stages of multiple instructions. The latency of a single instruction does not change, but efficiency when processing instructions consecutively is dramatically improved.

```
Without Pipelining (Sequential Execution):
  Time →  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
  Instr1: IF ID EX ME WB
  Instr2:                 IF ID EX ME WB
  Instr3:                                IF ID EX ME WB
  → 3 instructions take 15 clock cycles

5-Stage Pipeline:
  Time →  1  2  3  4  5  6  7
  Instr1: IF ID EX ME WB
  Instr2:    IF ID EX ME WB
  Instr3:       IF ID EX ME WB
  → 3 instructions take 7 clock cycles (= 5 + (3-1))

  IF=Fetch  ID=Decode  EX=Execute  ME=Memory  WB=WriteBack

Generalization: Processing N instructions in a K-stage pipeline
  - Without pipeline: N x K cycles
  - With pipeline: K + (N - 1) cycles
  - Speedup ratio: N x K / (K + N - 1)
  - When N is sufficiently large: approximately K-fold speedup
```

### 4.2 Pipeline Ideal vs. Reality

In an ideal pipeline, one instruction completes every cycle (CPI = 1: Cycles Per Instruction). In reality, however, the following factors cause the pipeline to "stall" (halt), degrading performance.

```
Pipeline Stall (Bubble Insertion):

Normal Operation:
  Time →  1  2  3  4  5  6  7  8
  Instr1: IF ID EX ME WB
  Instr2:    IF ID EX ME WB
  Instr3:       IF ID EX ME WB
  Instr4:          IF ID EX ME WB

When a Stall Occurs (Instr2 waits for Instr1's result):
  Time →  1  2  3  4  5  6  7  8  9
  Instr1: IF ID EX ME WB
  Instr2:    IF ID -- EX ME WB         ← Bubble (idle cycle) after ID
  Instr3:       IF -- ID EX ME WB      ← Delayed by 1 cycle
  Instr4:          -- IF ID EX ME WB   ← Delayed by 1 cycle

  "--" represents a bubble (NOP) = a wasted pipeline cycle
```

### 4.3 Pipeline Hazards (3 Types)

Factors that impede pipeline efficiency are called "hazards." Hazards are classified into three major types.

```
┌────────────────────────────────────────────────────────────────┐
│               3 Types of Pipeline Hazards                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Data Hazard                                                │
│     ─────────────────────────────                              │
│     Occurs when a later instruction uses the result of a      │
│     preceding instruction. Three subtypes exist:               │
│                                                                │
│     RAW (Read After Write) - Most common:                      │
│       ADD R1, R2, R3   ; Writes to R1                          │
│       SUB R4, R1, R5   ; Reads R1 ← write not yet complete    │
│                                                                │
│     WAR (Write After Read):                                    │
│       ADD R1, R2, R3   ; Reads R2                              │
│       SUB R2, R4, R5   ; Writes R2 ← dangerous if written first│
│                                                                │
│     WAW (Write After Write):                                   │
│       ADD R1, R2, R3   ; Writes to R1                          │
│       SUB R1, R4, R5   ; Writes to R1 ← order reversal danger │
│                                                                │
│     Countermeasures:                                           │
│     - Forwarding (Bypass): Transfer result from EX before WB   │
│     - Pipeline stall: Wait until dependency is resolved        │
│     - Register renaming: Eliminate WAR/WAW                     │
│                                                                │
│  2. Control Hazard                                             │
│     ─────────────────────────────                              │
│     Occurs when the address of the next instruction to execute │
│     is uncertain due to a branch instruction.                  │
│                                                                │
│     BEQ R1, R2, label  ; Branch to label if R1==R2            │
│     ADD R3, R4, R5     ; ← Execute this or the branch target? │
│                                                                │
│     Countermeasures:                                           │
│     - Branch Prediction: Predict branch target, execute speculatively│
│     - Delayed Branch: Always execute the instruction after branch│
│     - Speculative Execution: Execute based on prediction, roll │
│       back if wrong                                            │
│                                                                │
│  3. Structural Hazard                                          │
│     ─────────────────────────────                              │
│     Multiple instructions try to use the same hardware         │
│     resource in the same cycle.                                │
│                                                                │
│     Example: Fetch and Memory stages use the same memory port  │
│                                                                │
│     Countermeasures:                                           │
│     - Resource duplication: Split L1 cache into L1i and L1d    │
│     - Pipeline stall: Make one wait                            │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.4 Forwarding (Data Bypass)

The most common solution to data hazards is forwarding. The computation result is transferred directly from the Execute stage output to the next instruction's Execute stage, rather than waiting until the Write Back stage.

```
Without Forwarding (2-cycle stall):
  Time →  1  2  3  4  5  6  7  8  9
  ADD R1,R2,R3:  IF ID EX ME WB
  SUB R4,R1,R5:     IF ID -- -- EX ME WB    ← Wait for R1 to be determined

With Forwarding (no stall):
  Time →  1  2  3  4  5  6  7
  ADD R1,R2,R3:  IF ID EX ME WB
  SUB R4,R1,R5:     IF ID EX ME WB
                         ↑
                    EX output transferred directly (bypass path)

  However, Load-Use hazard requires 1 stall cycle even with forwarding:
  Time →  1  2  3  4  5  6  7  8
  LDR R1,[R2]:   IF ID EX ME WB
  ADD R3,R1,R4:     IF ID -- EX ME WB    ← ME result needed for EX
                            ↑
                    ME output transferred to EX (1 stall cycle)
```

### 4.5 Pipeline Performance Metrics

```python
# Pipeline Performance Calculation (Python)

def pipeline_performance(
    num_instructions: int,
    pipeline_stages: int,
    stall_cycles: int = 0,
    branch_miss_rate: float = 0.0,
    branch_penalty: int = 0,
    branch_frequency: float = 0.0
) -> dict:
    """Calculate pipeline performance metrics.

    Args:
        num_instructions: Number of instructions
        pipeline_stages: Number of pipeline stages
        stall_cycles: Total stall cycles due to data hazards
        branch_miss_rate: Branch misprediction rate (0.0-1.0)
        branch_penalty: Penalty per branch miss (in cycles)
        branch_frequency: Fraction of all instructions that are branches (0.0-1.0)

    Returns:
        Dictionary of performance metrics
    """
    # Ideal execution cycles (no hazards)
    ideal_cycles = pipeline_stages + (num_instructions - 1)

    # Penalty cycles due to branch misses
    num_branches = int(num_instructions * branch_frequency)
    branch_stalls = int(num_branches * branch_miss_rate * branch_penalty)

    # Actual execution cycles
    actual_cycles = ideal_cycles + stall_cycles + branch_stalls

    # CPI (Cycles Per Instruction)
    cpi = actual_cycles / num_instructions

    # IPC (Instructions Per Cycle)
    ipc = num_instructions / actual_cycles

    # Without pipeline
    no_pipeline_cycles = num_instructions * pipeline_stages

    # Speedup ratio
    speedup = no_pipeline_cycles / actual_cycles

    return {
        "ideal_cycles": ideal_cycles,
        "actual_cycles": actual_cycles,
        "data_stalls": stall_cycles,
        "branch_stalls": branch_stalls,
        "cpi": round(cpi, 3),
        "ipc": round(ipc, 3),
        "speedup": round(speedup, 2),
        "efficiency": round(ipc / 1.0 * 100, 1),  # Efficiency relative to ideal IPC=1
    }

# Usage example
result = pipeline_performance(
    num_instructions=1000,
    pipeline_stages=5,
    stall_cycles=50,        # 50 stall cycles from data hazards
    branch_miss_rate=0.05,  # 5% branch misprediction rate
    branch_penalty=15,      # 15 cycles penalty per miss
    branch_frequency=0.20   # 20% of instructions are branches
)

# Example result:
# ideal_cycles: 1004
# actual_cycles: 1054 + 150 = 1204
# CPI: 1.204
# IPC: 0.831
# speedup: 4.15x (vs. 5000 cycles without pipeline)
```

---

## 5. Branch Prediction

### 5.1 Why Branch Prediction Matters

Modern CPUs have deep pipelines (14-20 stages), requiring many cycles before a branch instruction's outcome is determined. Stalling the pipeline until the branch target is resolved would cause a major throughput loss. Instead, the branch outcome is "predicted" and instructions are executed speculatively. If the prediction is correct, there is no penalty; if wrong, the pipeline is flushed and execution restarts.

```
Branch Prediction Overview:

  Branch Instruction BEQ R1, R2, target:

  Correct Prediction:
    IF ID EX ME WB     ← Branch instruction
       IF ID EX ME WB  ← Predicted instruction (correct)
          IF ID EX ME WB
    → No penalty, smooth execution continues

  Misprediction:
    IF ID EX ME WB           ← Branch instruction
       IF ID EX ×× ××       ← Predicted instruction (wrong) → discarded
          IF ID ×× ×× ××    ← Predicted instruction (wrong) → discarded
             IF ×× ×× ×× ×× ← Predicted instruction (wrong) → discarded
                IF ID EX ME WB   ← Restart from correct branch target
    → Several wasted cycles (flush penalty)
```

### 5.2 Static and Dynamic Prediction

Branch prediction is broadly classified into "static prediction" and "dynamic prediction."

**Static Prediction** (determined at compile time, hardware-fixed):
- **Always predict not taken**: Always predict that a conditional branch is "not taken." Simple but low accuracy.
- **Backward branches predict taken**: Predict that branches to a loop's return address (backward branches) are "taken." Correct for most loop iterations.
- **Compiler hints**: The compiler can provide hints to the CPU about the branch direction, such as GCC's `__builtin_expect`.

**Dynamic Prediction** (hardware predicts based on execution history):
- **1-bit counter**: Predict the same as the last result. Accuracy ~85%.
- **2-bit saturating counter**: Don't change prediction until two consecutive misses. Accuracy ~90%.
- **Correlating predictor**: Considers the outcomes of other branches. Accuracy ~95%.
- **TAGE predictor**: Uses tables with multiple history lengths for high-accuracy prediction. Accuracy >97%.

### 5.3 Two-Bit Saturating Counter Mechanism

One of the most basic dynamic predictors is the 2-bit saturating counter.

```
2-Bit Saturating Counter State Transitions:

  ┌──────────┐   Taken    ┌──────────┐
  │ 00: Strong│──────────→│ 01: Weak  │
  │ Not Taken │←──────────│ Not Taken │
  └──────────┘  Not Taken └──────────┘
       ↑ Not Taken              │ Taken
       │                        ▼
  ┌──────────┐  Not Taken ┌──────────┐
  │ 10: Weak  │←──────────│ 11: Strong│
  │ Taken     │──────────→│ Taken     │
  └──────────┘   Taken    └──────────┘

  Prediction Method:
    States 00, 01 → Predict "Not Taken"
    States 10, 11 → Predict "Taken"

  Example: Branch for a 10-iteration loop (9 taken, last 1 not taken)
    Iter:       1   2   3   4   5   6   7   8   9   10
    Actual:     T   T   T   T   T   T   T   T   T   NT
    State:      01  10  11  11  11  11  11  11  11  10
    Predicted:  NT  NT  T   T   T   T   T   T   T   T
    Correct:    x   x   O   O   O   O   O   O   O   x
    → 8 correct out of 10 (80%), 1 miss at next loop entry
    → Overall: 8/10 = 80% → Accuracy improves with longer loops
```

### 5.4 Branch Prediction Accuracy Comparison

| Prediction Scheme | Approximate Accuracy | Hardware Cost | Example Usage |
|---------|-----------|------------------|--------|
| Always not taken | ~60% | Near zero | Simplest implementation |
| 1-bit counter | ~85% | Low | Early pipelined CPUs |
| 2-bit saturating counter | ~90% | Low-Medium | Widely used |
| Correlating predictor (gshare) | ~95% | Medium | Pentium Pro onward |
| Tournament predictor | ~96% | Medium-High | Alpha 21264 |
| TAGE predictor | ~97% | High | Modern high-performance CPUs |
| Perceptron predictor | ~97% | High | AMD Zen series |

### 5.5 Impact of Branch Prediction on Programs

```c
// Classic example demonstrating the impact of branch prediction (C)
//
// Sorted vs. unsorted arrays change how predictable the conditional branch is

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SIZE 32768

int compare_int(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int main(void) {
    int data[SIZE];
    srand(42);
    for (int i = 0; i < SIZE; i++)
        data[i] = rand() % 256;

    // --- Test 1: Unsorted array ---
    clock_t start = clock();
    long long sum = 0;
    for (int iter = 0; iter < 100000; iter++) {
        for (int j = 0; j < SIZE; j++) {
            if (data[j] >= 128)   // Branch: random data makes prediction difficult
                sum += data[j];
        }
    }
    clock_t end = clock();
    printf("Unsorted: sum=%lld, time=%.3f sec\n",
           sum, (double)(end - start) / CLOCKS_PER_SEC);

    // --- Test 2: Sorted array ---
    int sorted[SIZE];
    memcpy(sorted, data, sizeof(data));
    qsort(sorted, SIZE, sizeof(int), compare_int);

    start = clock();
    sum = 0;
    for (int iter = 0; iter < 100000; iter++) {
        for (int j = 0; j < SIZE; j++) {
            if (sorted[j] >= 128)  // Branch: first half all false, second half all true
                sum += sorted[j];
        }
    }
    end = clock();
    printf("Sorted:   sum=%lld, time=%.3f sec\n",
           sum, (double)(end - start) / CLOCKS_PER_SEC);

    // Typical results:
    // Unsorted: time=11.5 sec (branch misprediction rate ~50%)
    // Sorted:   time= 3.8 sec (branch misprediction rate ~0%)
    // → ~3x difference for the same amount of computation

    return 0;
}
```

### 5.6 Branchless Programming

Branchless programming is a technique to avoid branch mispredictions by performing equivalent processing without conditional branches.

```c
// Branchless Programming Examples

// With branches (potential for mispredictions)
int max_branch(int a, int b) {
    if (a > b)
        return a;
    else
        return b;
}

// Branchless version (using bit operations)
int max_branchless(int a, int b) {
    // When (a > b): diff < 0 → mask = 0xFFFFFFFF
    // When (a <= b): diff >= 0 → mask = 0x00000000
    int diff = b - a;
    int mask = diff >> 31;  // Arithmetic right shift: fills with sign bit
    return b - (diff & mask);
}

// Leveraging compiler optimization (recommended)
int max_cmov(int a, int b) {
    // Most compilers convert ternary operators to CMOV instructions
    return (a > b) ? a : b;
}

// x86-64 generated assembly:
// max_branch:           max_cmov:
//   cmp edi, esi          cmp edi, esi
//   jle .L2               cmovge eax, edi  ← Conditional move (no branch)
//   mov eax, edi          ret
//   ret
// .L2:
//   mov eax, esi
//   ret

// Conditional array summation (branchless version)
long sum_branchless(const int *data, int size) {
    long sum = 0;
    for (int i = 0; i < size; i++) {
        // When data[i] >= 128: mask = 0xFFFFFFFF, otherwise 0
        int mask = -(data[i] >= 128);  // bool → 0 or 1 → 0 or -1
        sum += (data[i] & mask);
    }
    return sum;
}
```

---

## 6. CISC vs RISC

### 6.1 Differences in Design Philosophy

In the history of computer architecture, instruction set design philosophies diverged into two major schools: CISC (Complex Instruction Set Computer) and RISC (Reduced Instruction Set Computer).

```
CISC (Complex Instruction Set Computer):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Design Philosophy: "Make each instruction do a lot of work"
  Representatives: x86/x64 (Intel, AMD), VAX, MC68000

  Historical Background:
  - In the 1970s-80s, memory was expensive and slow
  - Fewer instructions → Reduced memory usage
  - Complex operations per instruction → Fewer memory accesses
  - Compilers were immature → Hardware supported complex operations

  Characteristics:
  ┌─────────────────────────────────────────────────┐
  │ - Variable-length instructions (x86: 1-15 bytes)│
  │ - Single instruction can do memory access + ALU │
  │ - Many addressing modes                         │
  │ - Internally decomposed into RISC-like          │
  │   micro-operations (uops) via microcode         │
  │ - Large instruction count (1500+)               │
  │ - Emphasizes backward compatibility (40+ years) │
  └─────────────────────────────────────────────────┘


RISC (Reduced Instruction Set Computer):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Design Philosophy: "Keep instructions simple, maximize pipeline efficiency"
  Representatives: ARM, RISC-V, MIPS, SPARC, PowerPC

  Historical Background:
  - In the 1980s, research revealed that 80% of instructions used
    were only 20% of the full instruction set
  - Patterson and Hennessy demonstrated that "simple instructions +
    efficient pipelines" yielded higher performance
  - Advances in compiler technology made complex instructions unnecessary

  Characteristics:
  ┌─────────────────────────────────────────────────┐
  │ - Fixed-length instructions (4 bytes)           │
  │ - Load/Store architecture                       │
  │   (only dedicated instructions access memory)   │
  │ - Simple instructions → optimized for pipelining│
  │ - More registers (ARM: 31, RISC-V: 32)         │
  │ - Fewer instructions (basic 100-300)            │
  │ - Low power consumption                         │
  └─────────────────────────────────────────────────┘
```

### 6.2 Detailed Comparison Table

| Comparison | CISC (x86-64) | RISC (AArch64 / ARM) | RISC-V (RV64G) |
|---------|--------------|---------------------|----------------|
| **Representative CPUs** | Intel Core, AMD Ryzen | Apple M-series, Snapdragon, AWS Graviton | SiFive, Alibaba Xuantie |
| **Instruction Length** | Variable (1-15 bytes) | Fixed (4 bytes) | Fixed (4 bytes, 2 bytes with C extension) |
| **Instruction Count** | 1500+ | ~1000 (including extensions) | ~300 (base + standard extensions) |
| **General-Purpose Registers** | 16 (x64) | 31 (X0-X30) | 32 (x0-x31, x0 is always 0) |
| **Memory Access** | Directly in arithmetic instructions | Load/Store instructions only | Load/Store instructions only |
| **Addressing** | Many modes | Relatively few | Few |
| **Conditional Execution** | CMOV instruction | Conditional branch + CSEL instruction | Conditional branch only (base ISA) |
| **Power Consumption** | High (15-125W TDP) | Low (1-15W) | Design-dependent (ultra-low possible) |
| **Primary Use** | Desktop, Server | Mobile, Embedded, Server | IoT, Embedded, Academic Research |
| **Compatibility** | 40+ years backward compatibility | Changes between versions | ISA is frozen and stable |
| **License** | Intel/AMD monopoly | License fee to ARM Ltd. | Open source, free |
| **Decoding** | Complex (microcode) | Relatively simple | Very simple |

### 6.3 Assembly Comparison of the Same Operation

```nasm
; Operation: memory[C] = memory[A] + memory[B]
; Compared across three architectures

; ============================================================
; x86-64 (CISC)
; ============================================================
; Can directly operate on data in memory with arithmetic instructions
mov eax, [A]       ; Load value from Memory[A] into EAX
add eax, [B]       ; Add Memory[B] directly to EAX (memory + ALU in one instruction)
mov [C], eax       ; Store result to Memory[C]
; → 3 instructions (but variable length: total ~10-18 bytes)

; ============================================================
; AArch64 (ARM RISC)
; ============================================================
; Load/Store architecture: memory access and computation are separate instructions
; Assume addresses are pre-set in other registers
ldr w0, [x3]       ; Load value from Memory[x3] into W0
ldr w1, [x4]       ; Load value from Memory[x4] into W1
add w2, w0, w1     ; W2 = W0 + W1 (register-only computation)
str w2, [x5]       ; Store W2 to Memory[x5]
; → 4 instructions (all 4-byte fixed: total 16 bytes)

; ============================================================
; RISC-V (RV32I)
; ============================================================
; Load/Store architecture similar to ARM
lw   t0, 0(s0)     ; Load value from Memory[s0+0] into t0
lw   t1, 0(s1)     ; Load value from Memory[s1+0] into t1
add  t2, t0, t1    ; t2 = t0 + t1
sw   t2, 0(s2)     ; Store t2 to Memory[s2+0]
; → 4 instructions (all 4-byte fixed: total 16 bytes)

; CISC has fewer instructions, but each is complex to decode with variable length.
; RISC has more instructions, but each is simple and fixed-length,
; allowing the pipeline to operate efficiently.
; Modern CISC (x86) internally decomposes into RISC-like uops for execution.
```

### 6.4 Modern Convergence

Historically, CISC and RISC were discussed as opposing concepts, but in modern processors the boundary between them has become blurred.

```
Convergence of CISC/RISC in Modern CPUs:

  Inside x86 (CISC):
  ┌──────────────────────────────────────────────┐
  │  x86 instr → Front-end → uop (micro-ops)    │
  │               (Decoder)                      │
  │                                              │
  │  ADD EAX,[mem] → uop1: LOAD tmp,[mem]       │
  │                   uop2: ADD EAX,tmp          │
  │                                              │
  │  → Internally decomposed into RISC-like      │
  │    simple instructions, executed by the      │
  │    out-of-order engine                       │
  └──────────────────────────────────────────────┘

  ARM (RISC) Extensions:
  ┌──────────────────────────────────────────────┐
  │  - Instruction set continues to expand and   │
  │    grow more complex                         │
  │  - SVE/SVE2 (Scalable Vector Extensions)     │
  │  - Cryptographic and matrix instructions     │
  │  - Apple M-series has ultra-wide decoders    │
  │    (8 instructions simultaneously)           │
  │  → High-performance design beyond simple RISC│
  └──────────────────────────────────────────────┘

  Conclusion:
  - The pure CISC/RISC distinction has faded
  - x86 is CISC at the ISA level but RISC-like in execution
  - ARM is RISC at the ISA level but highly complex in implementation
  - What matters is "microarchitectural efficiency" rather than "ISA design"
```

### 6.5 The Innovation of RISC-V

RISC-V is an open-source ISA that originated at UC Berkeley in 2010.

```
RISC-V Characteristics and Significance:

  1. Open-Source ISA
     ├── No license fees (ARM costs millions to tens of millions of dollars)
     ├── Anyone can design and manufacture RISC-V compliant CPUs
     └── Specification managed by RISC-V International

  2. Modular Design
     ├── Base instruction set: RV32I / RV64I (minimal integer instructions)
     ├── Standard extensions:
     │   ├── M: Integer multiply/divide
     │   ├── A: Atomic operations
     │   ├── F: Single-precision floating-point
     │   ├── D: Double-precision floating-point
     │   ├── C: Compressed instructions (16-bit instructions)
     │   └── V: Vector extension
     └── Custom extensions: Freely add AI instructions, crypto instructions, etc.

  3. Major Use Cases
     ├── IoT / Embedded: Ultra-compact, ultra-low-power cores
     ├── Academic research: Freely modifiable and experimentable
     ├── China's semiconductor industry: Avoiding ARM sanctions, self-sufficiency
     └── Data centers: Ventana, Tenstorrent, etc. developing server-class chips

  4. Ecosystem Growth
     ├── Linux kernel officially supports RISC-V
     ├── GCC, LLVM support complete
     ├── Android supports RISC-V
     └── Shipped chips: Over 10 billion (as of 2024)
```

---

## 7. Cache Memory in Detail

### 7.1 Memory Hierarchy and Principle of Locality

Program memory access exhibits a statistical bias called "locality." Cache memory exploits this locality to achieve fast data access.

```
Memory Hierarchy Pyramid:

                    ┌───┐
                    │ R │  Registers
               Fast │ e │  ~0.3ns / ~several KB    ← Fastest, smallest
                    │ g │
                    ├───┤
                    │L1 │  L1 Cache
                    │   │  ~1ns / 32-64KB
                    ├───┤
                    │L2 │  L2 Cache
                    │   │  ~3-4ns / 256KB-1MB
                    ├───┤
                    │ L3  │  L3 Cache
                    │     │  ~10ns / 8-96MB
                    ├─────┤
                    │DRAM │  Main Memory
                    │     │  ~50-100ns / 8-128GB
                    ├─────┤
                    │ SSD │  Storage
               Slow │     │  ~100us / 256GB-8TB    ← Slowest, largest
                    └─────┘

  Principle of Locality:
  - Temporal locality: Recently accessed data is likely to be used again soon
    Example: Loop variables, local variables within functions
  - Spatial locality: After accessing an address, nearby addresses are likely to be accessed
    Example: Sequential array elements, sequential instruction execution
```

### 7.2 Cache Structure

```
Cache Internal Structure (Set-Associative):

  8-Way Set-Associative L1 Data Cache (32KB) Example:

  Memory Address Decomposition:
  ┌──────────────┬──────────┬──────────┐
  │   Tag         │ Set      │ Offset   │
  │ (upper bits)  │ Index    │(lower bits)│
  └──────────────┴──────────┴──────────┘

  Cache Internals:
  Set 0: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]
  Set 1: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]
  ...
  Set N: [Way0][Way1][Way2][Way3][Way4][Way5][Way6][Way7]

  Each Entry:
  ┌───────┬──────┬───────────────────────────────┐
  │ Valid │ Tag  │ Data (Cache Line: 64B)         │
  │ (1bit)│      │                               │
  └───────┴──────┴───────────────────────────────┘

  Types of Associativity:
  ┌────────────────┬──────────────┬─────────────────┐
  │ Direct-Mapped  │ N-Way Set    │ Fully Associative│
  │ (1-Way)        │ Associative  │                 │
  │ Fast but many  │ Balanced     │ No conflicts,   │
  │ conflicts      │              │ but slow        │
  └────────────────┴──────────────┴─────────────────┘
```

### 7.3 Cache Write Policies

| Policy | Behavior | Advantage | Disadvantage |
|---------|------|------|------|
| **Write-Through** | Updates both cache and memory on write | Data consistency is maintained | Writes are slow |
| **Write-Back** | Updates only cache; writes back to memory on eviction | Writes are fast | Consistency management is complex |
| **Write-Allocate** | Allocates a cache line on write miss | Subsequent writes become hits | Extra memory read occurs |
| **No-Write-Allocate** | Does not allocate cache on write miss | Avoids wasting cache on transient data | Cannot exploit write locality |

### 7.4 Cache-Friendly Programming

```c
// Cache Efficiency Experiment: Row-Major vs. Column-Major Access

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 4096

// Dynamic allocation (to prevent stack overflow)
static int matrix[N][N];

// Row-major access: Cache-friendly
long sum_row_major(void) {
    long sum = 0;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            sum += matrix[row][col];
            // Memory layout: [0][0], [0][1], [0][2], ...
            // → Access to contiguous addresses
            // → 1 cache line (64B) covers 16 ints (4B each)
            // → Cache hit rate: ~15/16 = 93.75%
        }
    }
    return sum;
}

// Column-major access: Cache-unfriendly
long sum_col_major(void) {
    long sum = 0;
    for (int col = 0; col < N; col++) {
        for (int row = 0; row < N; row++) {
            sum += matrix[row][col];
            // Memory layout: [0][0], [1][0], [2][0], ...
            // → stride = N * sizeof(int) = 16384 byte jumps
            // → Accesses different cache lines each time
            // → Frequent cache misses
        }
    }
    return sum;
}

// Blocking (Tiling): Maintains cache efficiency even for large matrices
#define BLOCK 64  // Block size that fits in L1 cache

long sum_blocked(void) {
    long sum = 0;
    for (int bi = 0; bi < N; bi += BLOCK) {
        for (int bj = 0; bj < N; bj += BLOCK) {
            // Process BLOCK x BLOCK sub-matrix
            // This sub-matrix fits in L1 cache
            int row_end = (bi + BLOCK < N) ? bi + BLOCK : N;
            int col_end = (bj + BLOCK < N) ? bj + BLOCK : N;
            for (int row = bi; row < row_end; row++) {
                for (int col = bj; col < col_end; col++) {
                    sum += matrix[row][col];
                }
            }
        }
    }
    return sum;
}

int main(void) {
    // Initialization
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = (i + j) % 100;

    clock_t start, end;

    start = clock();
    long s1 = sum_row_major();
    end = clock();
    printf("Row-major:  sum=%ld, time=%.4f sec\n",
           s1, (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    long s2 = sum_col_major();
    end = clock();
    printf("Col-major:  sum=%ld, time=%.4f sec\n",
           s2, (double)(end - start) / CLOCKS_PER_SEC);

    start = clock();
    long s3 = sum_blocked();
    end = clock();
    printf("Blocked:    sum=%ld, time=%.4f sec\n",
           s3, (double)(end - start) / CLOCKS_PER_SEC);

    // Typical results (N=4096):
    // Row-major:  time=0.03 sec
    // Col-major:  time=0.25 sec  (8-10x slower)
    // Blocked:    time=0.03 sec  (same as row-major)

    return 0;
}
```

```
Memory Layout and Cache Line Relationship (Detailed Diagram):

  C 2D array int matrix[4][4] memory layout:

  Address:   0x100  0x104  0x108  0x10C  0x110  0x114  0x118  0x11C
  Value:     [0][0] [0][1] [0][2] [0][3] [1][0] [1][1] [1][2] [1][3]
             ←──── Cache Line 0 ────────→ ←──── Cache Line 1 ────────→

  Address:   0x120  0x124  0x128  0x12C  0x130  0x134  0x138  0x13C
  Value:     [2][0] [2][1] [2][2] [2][3] [3][0] [3][1] [3][2] [3][3]
             ←──── Cache Line 2 ────────→ ←──── Cache Line 3 ────────→

  Row-major access (row=0,1,2,3; col=0,1,2,3):
    [0][0]→[0][1]→[0][2]→[0][3] → 4 hits in Cache Line 0
    [1][0]→[1][1]→[1][2]→[1][3] → 4 hits in Cache Line 1
    → Cache misses: 4 (only at line load time)
    → Hit rate: 12/16 = 75% (actually higher since line=64B fits 16 ints)

  Column-major access (col=0,1,2,3; row=0,1,2,3):
    [0][0]→[1][0]→[2][0]→[3][0] → Different cache lines each time
    [0][1]→[1][1]→[2][1]→[3][1] → Different cache lines each time
    → For large N, exceeds cache capacity, causing many cache misses
```

---

## 8. Instruction-Level Parallelism (ILP)

### 8.1 Superscalar Execution

A superscalar processor can simultaneously fetch, decode, and execute multiple instructions per cycle. Nearly all modern high-performance CPUs employ superscalar design.

```
Scalar (1 instruction/cycle):
  Clock:  1    2    3    4    5    6
  Issue:  ADD  MUL  SUB  AND  OR   XOR
  → 6 instructions in 6 cycles, IPC = 1.0

2-Way Superscalar (2 instructions/cycle):
  Clock:  1       2       3
  Port0:  ADD     SUB     OR
  Port1:  MUL     AND     XOR
  → 6 instructions in 3 cycles, IPC = 2.0

Apple M4 Firestorm Core (up to 10 instructions/cycle):
  Clock:  1
  Port0:  ALU instruction
  Port1:  ALU instruction
  Port2:  ALU instruction
  Port3:  ALU instruction
  Port4:  FP/SIMD instruction
  Port5:  FP/SIMD instruction
  Port6:  FP/SIMD instruction
  Port7:  FP/SIMD instruction
  Port8:  LOAD
  Port9:  LOAD
  Port10: STORE
  → Theoretically can issue up to 10 instructions per cycle
     (when there are no dependencies)
```

### 8.2 Out-of-Order Execution (OoO)

A technique that executes instructions not in program order but prioritizes instructions without data dependencies, improving pipeline utilization.

```
In-Order Execution (program order):
  Instr1: a = LOAD [addr1]     ; Memory access: 100 cycles
  Instr2: b = a + 1            ; Depends on Instr1 → wait 100 cycles
  Instr3: c = LOAD [addr2]     ; Unrelated to Instr1,2 but must wait in order
  Instr4: d = c * 2            ; Depends on Instr3
  → Total: ~200+ cycles

Out-of-Order Execution:
  Cycle 1: Issue Instr1 (LOAD [addr1])
  Cycle 1: Issue Instr3 (LOAD [addr2])  ← Issued in parallel with Instr1!
  Cycle ~100: Both Instr1 and Instr3 complete
  Cycle ~101: Execute Instr2 (a + 1)
  Cycle ~101: Execute Instr4 (c * 2)    ← Executed in parallel with Instr2!
  → Total: ~102 cycles (roughly halved)
```

### 8.3 Register Renaming

A technique to eliminate false dependencies such as WAR (Write After Read) and WAW (Write After Write) for efficient out-of-order execution.

```python
# Register Renaming Concept in Pseudocode

# Original code (reuses architectural register R1)
# ADD R1, R2, R3    ; R1 = R2 + R3
# MUL R4, R1, R5    ; R4 = R1 * R5  (RAW: true dependency)
# ADD R1, R6, R7    ; R1 = R6 + R7  (WAW: re-write to R1)
# SUB R8, R1, R9    ; R8 = R1 - R9  (RAW: true dependency)

# After renaming (physical registers P1, P2, ... assigned)
# ADD P10, P2, P3    ; P10 = P2 + P3  (R1 → P10)
# MUL P4, P10, P5   ; P4 = P10 * P5  (RAW dependency preserved)
# ADD P11, P6, P7    ; P11 = P6 + P7  (R1 → P11: different physical register!)
# SUB P8, P11, P9    ; P8 = P11 - P9

# → Instruction groups 1,2 and 3,4 can execute independently
# → WAW dependency eliminated, parallelism increased

# Physical register counts in modern CPUs:
# Intel Skylake: 16 architectural registers → 180 physical registers
# Apple M1: 31 architectural registers → ~300+ physical registers
# → This difference (renaming buffer) determines OoO execution "depth"
```

### 8.4 Speculative Execution and Its Cost

Speculative execution, combining out-of-order execution with branch prediction, is a powerful acceleration technique but carries security risks.

```
Security Issues with Speculative Execution:

  Spectre / Meltdown (disclosed in 2018) Overview:

  1. Speculatively executed instructions read data from
     memory regions that should normally be inaccessible
  2. The speculative execution is ultimately "undone," but
     side effects on the cache (side channel) remain
  3. By measuring cache timing differences,
     secret data can be inferred

  Impact:
  - Nearly all modern CPUs (Intel, AMD, ARM) were affected
  - Mitigations required at the OS, browser, and compiler levels
  - Performance degradation from mitigations: 2-30% (workload-dependent)

  Lessons:
  - Performance and security are in a trade-off relationship
  - Hardware optimizations can threaten software security
  - We live in an era where microarchitectural details directly affect security
```

---

## 9. Multi-Core and Hyper-Threading

### 9.1 The Necessity of Multi-Core

```
Why Multi-Core Became Necessary:

  Clock Frequency Trends:
  1990: 25MHz   │
  1995: 100MHz  │  Frequency increased
  2000: 1GHz    │  following Moore's Law
  2004: 3.8GHz  │  ← Hit the Power Wall!
  2024: ~5.8GHz │  ← Only +50% in 20 years

  Power Wall:
  ┌────────────────────────────────────────────┐
  │ Power Consumption ∝ Voltage^2 x Frequency  │
  │                                            │
  │ Doubling the frequency:                    │
  │   - Voltage also needs to increase (~1.1x) │
  │   - Power ≈ 1.1^2 x 2 = 2.4x              │
  │   - Heat also more than doubles → exceeds  │
  │     cooling limits                         │
  │                                            │
  │ Solution: Instead of increasing frequency, │
  │           increase core count for parallel  │
  │           processing                       │
  └────────────────────────────────────────────┘
```

### 9.2 Multi-Core Configuration

```
Typical Multi-Core CPU Configuration:

  ┌──────────────────────────────────────────────────────┐
  │                    4-Core CPU                         │
  │                                                      │
  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
  │  │  Core 0   │  │  Core 1   │  │  Core 2   │  │  Core 3   │
  │  │ ┌───┐┌───┐│  │ ┌───┐┌───┐│  │ ┌───┐┌───┐│  │ ┌───┐┌───┐│
  │  │ │L1i││L1d││  │ │L1i││L1d││  │ │L1i││L1d││  │ │L1i││L1d││
  │  │ └───┘└───┘│  │ └───┘└───┘│  │ └───┘└───┘│  │ └───┘└───┘│
  │  │  ┌─────┐  │  │  ┌─────┐  │  │  ┌─────┐  │  │  ┌─────┐  │
  │  │  │ L2  │  │  │  │ L2  │  │  │  │ L2  │  │  │  │ L2  │  │
  │  │  └─────┘  │  │  └─────┘  │  │  └─────┘  │  │  └─────┘  │
  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘
  │       │              │              │              │        │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │              Shared L3 Cache (8-96MB)                │   │
  │  └─────────────────────────────────────────────────────┘   │
  │                          │                                  │
  │  ┌─────────────────────────────────────────────────────┐   │
  │  │          Memory Controller                          │   │
  │  └─────────────────────────────────────────────────────┘   │
  └──────────────────────────────────────────────────────────────┘
                             │
                    ┌────────────────┐
                    │  DRAM (DDR5)   │
                    └────────────────┘
```

### 9.3 Amdahl's Law

There is a theoretical upper limit to speedup from multi-core. Amdahl's Law formalizes this.

```
Amdahl's Law:

  Speedup = 1 / ((1 - P) + P/N)

  P = Parallelizable fraction (0.0 ~ 1.0)
  N = Number of processors (cores)

  Example: When 75% of the program is parallelizable
  ┌──────┬──────────┬──────────┬──────────┬──────────┐
  │Cores │ 1        │ 4        │ 16       │ infinity │
  │Speedup│ 1.00x   │ 2.29x    │ 3.16x    │ 4.00x    │
  └──────┴──────────┴──────────┴──────────┴──────────┘

  → No matter how many cores are added, the sequential portion (25%)
    becomes the bottleneck, and the maximum speedup is 1/(1-0.75) = 4x

  Graph (Theoretical limits by parallelizable fraction):
  Speedup
   20x ┤                                         . P=95%
       │                                    .....
   16x ┤                               ....
       │                          ....        .... P=90%
   12x ┤                     ....    .....
       │                .... ....
    8x ┤           .........           ....... P=75%
       │      .....  ......
    4x ┤  ......................              P=50%
       │ .............
    2x ┤..........
       │.......
    1x ┼──┬──┬──┬──┬──┬──┬──┬──→ Cores
       1  2  4  8  16 32 64 128
```

### 9.4 Hyper-Threading (SMT: Simultaneous Multi-Threading)

```
Hyper-Threading (SMT) Mechanism:

  A technology that places two (or four) logical threads on a single
  physical core. Execution resources such as ALU and cache are shared,
  while thread state like register files and PC are duplicated.

  1 Physical Core (2-way SMT):
  ┌──────────────────────────────────────────┐
  │  ┌─────────────┐  ┌─────────────┐       │
  │  │ Thread 0    │  │ Thread 1    │Replicated│
  │  │(Logical CPU0)│  │(Logical CPU1)│ Part  │
  │  │ - PC        │  │ - PC        │       │
  │  │ - Registers │  │ - Registers │       │
  │  │ - TLB part  │  │ - TLB part  │       │
  │  └──────┬──────┘  └──────┬──────┘       │
  │         │                │               │
  │         └───────┬────────┘               │
  │                 ▼                        │
  │  ┌─────────────────────────────────┐     │
  │  │ Shared Resources               │     │
  │  │ - ALU / FPU / SIMD             │     │
  │  │ - L1i / L1d Cache              │     │
  │  │ - L2 Cache                     │     │
  │  │ - Decoder                      │     │
  │  │ - Reorder Buffer               │     │
  │  └─────────────────────────────────┘     │
  └──────────────────────────────────────────┘

  Effect:
  - While one thread waits on memory, the other thread
    can use the ALU → effective resource utilization
  - Throughput improvement: 15-30% (workload-dependent)
  - Not 2x because: execution resources are shared
  - Can be counterproductive: when both threads contend for cache
```

---

## 10. Apple Silicon and Heterogeneous Computing

### 10.1 SoC (System on Chip) Design Philosophy

```
Traditional PC Configuration:
  ┌─────────┐  PCIe   ┌─────────┐
  │ CPU     │─────────│ GPU     │
  │ (Intel) │  Bus    │(NVIDIA) │
  └────┬────┘         └────┬────┘
       │ DDR              │ GDDR
  ┌────┴────┐         ┌────┴────┐
  │CPU RAM  │         │GPU VRAM │ ← Separate memory for CPU and GPU
  │(DDR5)   │         │(GDDR6X) │   Data copy required
  └─────────┘         └─────────┘

Apple Silicon SoC:
  ┌───────────────────────────────────────────────┐
  │               Apple M4 Max                     │
  │                                               │
  │  ┌──────────┐  ┌───────────┐  ┌────────────┐│
  │  │ P-Core   │  │ E-Core    │  │ GPU        ││
  │  │ x12      │  │ x4        │  │ 40 cores   ││
  │  │(Perform.)│  │(Efficiency)│  │            ││
  │  └──────────┘  └───────────┘  └────────────┘│
  │                                               │
  │  ┌──────────┐  ┌───────────┐  ┌────────────┐│
  │  │ Neural   │  │ Media     │  │ Display    ││
  │  │ Engine   │  │ Engine    │  │ Engine     ││
  │  │ 16 cores │  │ H.264/5   │  │            ││
  │  │(ML Infer.)│  │ ProRes    │  │            ││
  │  └──────────┘  └───────────┘  └────────────┘│
  │                                               │
  │  ┌─────────────────────────────────────────┐  │
  │  │         Unified Memory (LPDDR5)         │  │
  │  │         Up to 128GB / BW: 546GB/s       │  │
  │  │                                         │  │
  │  │  CPU, GPU, NPU all directly access      │  │
  │  │  the same memory → No data copy needed  │  │
  │  └─────────────────────────────────────────┘  │
  └───────────────────────────────────────────────┘
```

### 10.2 big.LITTLE (Efficiency Cores + Performance Cores)

| Characteristic | P-Core (Performance) | E-Core (Efficiency) |
|------|---------------------|---------------------|
| Purpose | Maximum single-thread performance | Lightweight processing with low power |
| Clock | High (~4.5GHz) | Low (~2.8GHz) |
| Pipeline Width | Wide (8-10 simultaneous decode) | Narrow (~4 instructions) |
| Reorder Buffer | Large (600+ entries) | Small |
| Power Consumption | High | Low (1/3 to 1/5 of P-Core) |
| Use Cases | Compilation, video editing, gaming | Email, browsing, background tasks |

The OS scheduler (on macOS, QoS: Quality of Service) automatically determines which core to use based on task priority and load.

---

## 11. Anti-Patterns

### Anti-Pattern 1: "Clock Frequency Supremacy"

```
Anti-Pattern: Judging CPU performance solely by clock frequency

  Incorrect Thinking:
    "A 5.8GHz CPU is 1.66x faster than a 3.5GHz CPU"

  Reality:
    Performance = IPC x Clock Frequency x Core Count (effective utilization)

    Intel i9-14900K:  5.8GHz x IPC ~4   = Effective ~23.2 (relative)
    Apple M4 Max:     4.5GHz x IPC ~8   = Effective ~36.0 (relative)
    * IPC values are approximate for conceptual comparison

  Why This Is Wrong:
  1. IPC (Instructions Per Cycle) varies greatly across architectures
  2. Pipeline depth, width, and branch prediction accuracy differ
  3. Cache configuration and memory bandwidth differ
  4. Instruction mix in actual workloads differs

  Correct Evaluation:
  - Refer to benchmark results on actual workloads
  - Standard benchmarks such as Geekbench, SPEC CPU, Cinebench
  - Real-world measurement on the target application is most reliable
```

### Anti-Pattern 2: "Data Structure Design Ignoring Cache"

```
Anti-Pattern: Choosing data structures without considering memory access patterns

  Poor Design (AoS: Array of Structures):
  ┌─────────────────────────────────────────────────┐
  │  struct Particle {                               │
  │      float x, y, z;       // Position (12B)     │
  │      float vx, vy, vz;    // Velocity (12B)     │
  │      float mass;           // Mass (4B)          │
  │      int   type;           // Type (4B)          │
  │      char  name[32];       // Name (32B)         │
  │  };  // Total: 64 bytes                          │
  │                                                  │
  │  Particle particles[10000];                      │
  │                                                  │
  │  // When updating only positions of all particles:│
  │  for (int i = 0; i < 10000; i++) {               │
  │      particles[i].x += particles[i].vx * dt;     │
  │      // Only 4+4=8 bytes of 64 bytes are used    │
  │      // 87.5% of the cache line is wasted        │
  │  }                                               │
  └─────────────────────────────────────────────────┘

  Correct Design (SoA: Structure of Arrays):
  ┌─────────────────────────────────────────────────┐
  │  struct ParticleSystem {                         │
  │      float *x, *y, *z;      // Position arrays  │
  │      float *vx, *vy, *vz;   // Velocity arrays  │
  │      float *mass;            // Mass array       │
  │      int   *type;            // Type array       │
  │  };                                              │
  │                                                  │
  │  // Position update:                             │
  │  for (int i = 0; i < 10000; i++) {               │
  │      ps.x[i] += ps.vx[i] * dt;                  │
  │      // x[] and vx[] are contiguous in memory    │
  │      // 100% cache line utilization              │
  │  }                                               │
  │                                                  │
  │  // Can also process 4 at a time with SIMD (AVX):│
  │  // __m128 vx4 = _mm_load_ps(&ps.vx[i]);         │
  │  // __m128 dx  = _mm_mul_ps(vx4, dt4);            │
  │  // __m128 x4  = _mm_add_ps(..., dx);             │
  └─────────────────────────────────────────────────┘

  Performance Difference: SoA can be 2-8x faster than AoS
  (especially when combined with SIMD optimization)
```

### Anti-Pattern 3: "Excessive Conditional Branches in Loops"

```
Anti-Pattern: Using hard-to-predict conditional branches extensively in loops

  Problematic Code:
  ┌─────────────────────────────────────────────────┐
  │  for (int i = 0; i < N; i++) {                   │
  │      if (data[i] > threshold) {                  │
  │          result[i] = func_a(data[i]);            │
  │      } else if (data[i] > threshold2) {          │
  │          result[i] = func_b(data[i]);            │
  │      } else {                                    │
  │          result[i] = func_c(data[i]);            │
  │      }                                          │
  │  }                                               │
  │  // When data is random, frequent branch misses  │
  └─────────────────────────────────────────────────┘

  Improvements:
  1. Pre-sort the data to make branch patterns more predictable
  2. Use a function pointer table (lookup table)
  3. Use SIMD instructions for branchless processing
  4. Partition data into per-category buckets before processing
```

---

## 12. Practical Exercises

### Exercise 1: Instruction Cycle Trace (Foundational)

For the following pseudo-assembly code, describe the pipeline diagram showing what happens at each stage of a 5-stage pipeline (IF, ID, EX, MEM, WB).

```
LOAD R1, 100    ; Load value at memory address 100 into R1
LOAD R2, 104    ; Load value at memory address 104 into R2
ADD  R3, R1, R2 ; R3 = R1 + R2
STORE R3, 108   ; Store R3's value to memory address 108
```

**Tasks**:
1. Draw the pipeline diagram without hazards
2. Identify where data hazards occur
3. Calculate how many cycles are needed with forwarding

**Solution Hints**:
- The ADD instruction depends on the results of the LOAD instructions (R1, R2) (Load-Use hazard)
- Even with forwarding, LOAD → use requires 1 stall cycle
- The STORE instruction depends on the ADD result (R3)

### Exercise 2: Cache Performance Analysis (Intermediate)

Calculate the cache hit rate under the following conditions.

```
Conditions:
- L1d cache: 32KB, 8-way set-associative, 64B line size
- Access pattern: Sequential access of an int array (4B elements)
- Array size: 8192 elements (32KB)

Q1: Calculate the cache hit rate for a single sequential scan
Q2: What is the hit rate for the second scan when scanning the same array twice?
Q3: What happens when the array size is 64KB (2x cache size)?
```

**Solution Hints**:
- 1 cache line fits 64/4 = 16 ints
- First access always misses → next 15 are hits → Cold miss rate = 1/16
- Whether the entire array fits in cache determines the second scan's hit rate

### Exercise 3: Multi-Core Performance Estimation (Advanced)

Using Amdahl's Law, estimate the parallelization effect of the following program.

```
Program Composition:
- Data loading: 10% of total (sequential only)
- Preprocessing: 15% of total (80% parallelizable)
- Main computation: 60% of total (95% parallelizable)
- Post-processing: 10% of total (70% parallelizable)
- Result output: 5% of total (sequential only)

Q1: Calculate the overall parallelizable fraction P
Q2: Calculate the theoretical speedup for 4, 8, and 64 cores
Q3: When considering parallelization overhead (communication cost,
     synchronization cost), discuss what the realistic speedup would be
```

**Solution Hints**:
- P = 0.10x0 + 0.15x0.80 + 0.60x0.95 + 0.10x0.70 + 0.05x0
- P = 0 + 0.12 + 0.57 + 0.07 + 0 = 0.76
- Amdahl's Law: Speedup = 1 / ((1-P) + P/N)

---

## 13. FAQ

### Q1: Does a higher clock frequency always mean a faster CPU?

**A**: Not necessarily. CPU performance is determined holistically by "IPC (Instructions Per Cycle) x Clock Frequency x Effective Core Utilization." The Apple M4 (~3.5-4.5GHz) can outperform the Intel i9 (~5.8GHz) in single-threaded scenarios because its IPC is significantly higher. Pipeline width (simultaneous decode/issue count), branch prediction accuracy, cache capacity and bandwidth, and memory latency all collectively determine performance.

### Q2: Why is x86 still in use?

**A**: The primary reason is backward compatibility. An enormous body of x86 binary software has accumulated over 40+ years, and it must continue to run without recompilation. Enterprise legacy systems and a great deal of commercial Windows software are x86 binaries. However, migration to ARM-based processors is progressing in the server space (AWS Graviton, Ampere Altra), and Apple's Mac ARM transition with Apple Silicon is complete. Since x86 internally converts to RISC-like micro-operations (uops) for execution, ISA-level complexity does not necessarily translate to execution inefficiency.

### Q3: What makes RISC-V revolutionary?

**A**: RISC-V is an open-source Instruction Set Architecture (ISA) that is revolutionary because anyone can design and manufacture RISC-V compliant CPUs without license fees. ARM ISA licensing costs millions to tens of millions of dollars, but RISC-V has no such cost. Furthermore, its modular design allows selective implementation of needed extensions (multiply/divide, floating-point, vector computation, custom AI instructions, etc.) based on the use case. Linux, GCC, and LLVM officially support RISC-V, and its ecosystem is growing rapidly. As of 2024, over 10 billion RISC-V chips have been shipped, with broad deployment expected from IoT and embedded systems to the server market in the future.

### Q4: How much does a cache miss slow things down?

**A**: It varies significantly depending on which memory hierarchy level is accessed. An L1 cache hit takes approximately 4 cycles (~1ns), whereas an L2 miss accessing L3 takes approximately 40 cycles (~10ns), and an L3 miss accessing DRAM takes approximately 100-200 cycles (~50-100ns). This means there is a 25-50x difference between an L1 hit and a DRAM access. When cache misses occur frequently in a program's hot loop, the CPU pipeline spends most of its time waiting on memory, resulting in performance degradation far beyond what the clock frequency alone would suggest.

### Q5: Why does a deeper pipeline increase the branch misprediction penalty?

**A**: Pipeline depth (number of stages) directly correlates with the number of cycles between when a branch instruction is fetched and when the branch target is determined. For a 5-stage pipeline, the misprediction penalty is approximately 2-3 cycles, while for a 20-stage pipeline, it can reach 10-20 cycles. All speculatively executed instructions must be discarded (flushed) until the branch target is determined, leaving the pipeline empty during that time. Intel's Pentium 4 NetBurst architecture (31 stages) is a well-known historical example of the drawbacks of an excessively deep pipeline.

---


## FAQ

### Q1: What is the most important point when studying this topic?

Gaining practical experience is the most important thing. Understanding deepens not just from theory but from actually writing code and verifying how things work.

### Q2: What common mistakes do beginners make?

Skipping the fundamentals to jump into advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge used in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 14. Summary

| Concept | Key Points |
|------|---------|
| Instruction Cycle | Fetch → Decode → Execute → (Memory) → WriteBack in 4-5 stages |
| ISA | The interface between software and hardware. CISC/RISC design philosophies exist |
| Pipeline | Overlaps instruction stages to improve throughput. Depth involves trade-offs |
| Hazards | 3 types: Data/Control/Structural. Addressed with forwarding and branch prediction |
| Branch Prediction | Dynamic prediction (2-bit counter, TAGE, etc.) achieves >97% accuracy. Misses incur 10-20 cycle penalties |
| CISC vs RISC | x86 (complex, compatible) vs ARM/RISC-V (simple, power-efficient). Modern trend toward convergence |
| Cache | L1/L2/L3 hierarchy. Exploits spatial and temporal locality. Memory access patterns determine performance |
| ILP | Superscalar, OoO execution, register renaming extract instruction-level parallelism |
| Multi-Core | Solution to the Power Wall. Amdahl's Law sets theoretical speedup limits |
| SMT | Multiple logical threads per physical core. 15-30% throughput improvement |
| Apple Silicon | SoC design, unified memory, big.LITTLE. Balances power efficiency and performance |
| Security | Spectre/Meltdown: Speculative execution creates side-channel attacks |

---

## Recommended Next Reading


---

## References

1. Patterson, D. A. & Hennessy, J. L. *Computer Organization and Design: The Hardware/Software Interface.* 6th Edition, Morgan Kaufmann, 2020. -- The definitive textbook for systematically learning instruction cycles, pipelines, and RISC design fundamentals.
2. Hennessy, J. L. & Patterson, D. A. *Computer Architecture: A Quantitative Approach.* 6th Edition, Morgan Kaufmann, 2017. -- An advanced textbook covering quantitative performance analysis of cache hierarchies, branch prediction, ILP, and multiprocessors.
3. Bryant, R. E. & O'Hallaron, D. R. *Computer Systems: A Programmer's Perspective.* 3rd Edition, Pearson, 2015. -- A practical text explaining memory hierarchy, cache optimization, and processor operation principles from a programmer's perspective.
4. Intel Corporation. *Intel 64 and IA-32 Architectures Software Developer Manuals.* https://www.intel.com/ -- The official x86-64 architecture specification. Detailed technical documents on instruction sets and microarchitecture.
5. ARM Limited. *ARM Architecture Reference Manual (ARMv8-A).* https://developer.arm.com/ -- The official AArch64 (ARM 64-bit) architecture specification. Covers instruction sets, exception handling models, and memory models comprehensively.
6. Fog, A. *The Microarchitecture of Intel, AMD and VIA CPUs.* https://agner.org/optimize/ -- A technical document providing detailed analysis of Intel and AMD processor microarchitectures (pipeline configurations, execution units, branch predictors).
7. RISC-V International. *The RISC-V Instruction Set Manual.* https://riscv.org/technical/specifications/ -- The official RISC-V ISA specification. Defines base integer instruction sets through various standard extensions.
8. Kocher, P. et al. "Spectre Attacks: Exploiting Speculative Execution." *IEEE Symposium on Security and Privacy*, 2019. -- The paper reporting speculative execution side-channel vulnerabilities. Essential for understanding modern CPU security issues.
