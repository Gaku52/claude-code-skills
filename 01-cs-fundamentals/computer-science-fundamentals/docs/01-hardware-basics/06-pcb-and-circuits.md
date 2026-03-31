# PCB and Circuits — From Logic Gates to Semiconductor Processes

> "The fact that a universal computer can be built from a single NAND gate
> is one of the most beautiful consequences in computer science."
> — Noam Nisan & Shimon Schocken, *The Elements of Computing Systems*

---

## Learning Objectives

- [ ] Explain the operating principles of MOSFETs and how CMOS logic is constructed
- [ ] Write truth tables for basic logic gates (NOT, AND, OR, NAND, NOR, XOR)
- [ ] Prove the universality (functional completeness) of NAND gates
- [ ] Design combinational circuits (adders, multiplexers, decoders)
- [ ] Explain the operation of sequential circuits (latches, flip-flops, counters)
- [ ] Write basic circuit descriptions in Verilog HDL
- [ ] Understand PCB (Printed Circuit Board) design principles and multilayer structures
- [ ] Explain FPGA building blocks and the concept of reconfigurable computing
- [ ] Provide an overview of the entire semiconductor manufacturing process
- [ ] Understand the physical limits of process node scaling and current technology trends

## Prerequisites

- Basic concepts of Boolean algebra (AND, OR, NOT)
- Elementary concepts of voltage and current (middle school physics level)

## Target Audience

| Level | Reading Approach |
|-------|-----------------|
| Beginner | Read Sections 1–3 carefully and work through the basic exercises |
| Intermediate | Focus on Sections 4–6 covering sequential circuits and HDL |
| Advanced | Concentrate on Sections 7–9 covering FPGA, semiconductor processes, and latest trends |

---

## Table of Contents

1. [Transistors — The Smallest Building Blocks of a Computer](#1-transistors--the-smallest-building-blocks-of-a-computer)
2. [Logic Gates — Physical Implementation of Boolean Algebra](#2-logic-gates--physical-implementation-of-boolean-algebra)
3. [Combinational Circuits — Combining Logic Gates](#3-combinational-circuits--combining-logic-gates)
4. [Sequential Circuits — Memory and State Transitions](#4-sequential-circuits--memory-and-state-transitions)
5. [Circuit Description with Verilog HDL](#5-circuit-description-with-verilog-hdl)
6. [PCB (Printed Circuit Board) Design and Structure](#6-pcb-printed-circuit-board-design-and-structure)
7. [FPGA — Reconfigurable Hardware](#7-fpga--reconfigurable-hardware)
8. [Semiconductor Manufacturing Process](#8-semiconductor-manufacturing-process)
9. [Latest Technology Trends and Future Outlook](#9-latest-technology-trends-and-future-outlook)
10. [Anti-Patterns and Design Pitfalls](#10-anti-patterns-and-design-pitfalls)
11. [Hands-On Exercises](#11-hands-on-exercises)
12. [FAQ](#12-faq)
13. [Summary](#13-summary)
14. [References](#14-references)

---

## 1. Transistors — The Smallest Building Blocks of a Computer

### 1.1 Physics of Semiconductors

The computational power of a computer has its roots in the physical properties of semiconductors. Conductors (metals) conduct electricity well, while insulators (glass) do not. Semiconductors lie between these two, and their conductivity can be controlled by external conditions (voltage, temperature, introduction of impurities).

Silicon (Si) is a tetravalent element that forms four covalent bonds with neighboring atoms in its crystal structure. Pure silicon has very low conductivity, but its electrical properties can be dramatically altered through doping — the intentional introduction of impurities.

```
Principles of Doping:

  N-type semiconductor (doped with Phosphorus P):
  +---------------------------+
  |  Si --- Si --- Si         |
  |  |       |       |        |
  |  Si --- P  --- Si         |  <- Phosphorus is pentavalent: excess electron (e-) becomes free
  |  |       |^      |        |
  |  Si --- Si --- Si         |  Carrier = electron (Negative)
  |         excess e-          |
  +---------------------------+

  P-type semiconductor (doped with Boron B):
  +---------------------------+
  |  Si --- Si --- Si         |
  |  |       |       |        |
  |  Si --- B  --- Si         |  <- Boron is trivalent: electron vacancy (hole) is created
  |  |       |^      |        |
  |  Si --- Si --- Si         |  Carrier = hole (Positive)
  |         hole (O)           |
  +---------------------------+

  PN Junction:
  +----------+----------+
  |  P-type  |  N-type  |
  |  O O O   |  e-e-e-  |
  |  O O     |   e-e-   |
  |          |          |
  |  <- holes|  electrons -> |
  +----------+----------+
       Depletion region (electric field formed)

  -> Forward bias: current ON, reverse bias: current OFF
  -> This is the principle of a diode and the foundational technology for transistors
```

**Why Silicon?** There are multiple reasons why silicon dominates the semiconductor industry:

| Property | Description |
|----------|-------------|
| Abundance | Constitutes about 25% of the Earth's crust. Raw material cost is extremely low |
| Oxide layer | Thermal oxidation forms a stable SiO2 layer, which serves as an excellent insulator |
| Band gap | 1.12 eV. A suitable value for room temperature operation |
| Processability | Established techniques for producing high-purity single crystals |
| Ecosystem | Over 70 years of accumulated research and manufacturing infrastructure |

### 1.2 MOSFET Operating Principles

The MOSFET (Metal-Oxide-Semiconductor Field-Effect Transistor) is the fundamental device in modern digital circuits. It controls channel conduction through the field effect.

```
NMOS Transistor Structure and Operation:

  Cross-section:
           Gate (G)
             |
        +----+----+
        | Metal    |
        +---------+
        |  SiO2   |  <- Oxide layer (insulator, thickness of a few nm)
        +---------+
   +----| Channel  |----+
   | n+ |  (p-type)| n+ |  <- Source/Drain are heavily doped N-type
   |    |          |    |
   | S  | P-substrate| D |
   +----+----------+----+
  Source              Drain

  Operating Modes:
  ---------------------------------------------
  Gate Voltage Vgs    Channel State     Logic Value
  ---------------------------------------------
  0V (Low)           Closed (non-conducting)  OFF -> Logic 0
  Vdd (High)         Open (conducting)        ON  -> Logic 1
  ---------------------------------------------

  Threshold Voltage Vth:
  - Vgs < Vth -> OFF (cutoff region)
  - Vgs > Vth -> ON (saturation/linear region)
  - In advanced processes, Vth is approximately 0.2-0.4V
```

The PMOS transistor operates complementarily to NMOS. Its substrate is N-type, the source and drain are P-type, and it conducts when Low is applied to the gate.

```
NMOS vs PMOS Comparison:

  NMOS:                          PMOS:
  Vgs = High -> ON                Vgs = Low -> ON
  Vgs = Low  -> OFF               Vgs = High -> OFF

  +---------------------------------------------+
  |              CMOS Inverter                    |
  |                                              |
  |  Vdd ---+---                                 |
  |         |                                    |
  |      +--+--+                                 |
  |      |PMOS | <- Input=0: ON -> Output=Vdd(1) |
  |      +--+--+                                 |
  |  Input -+---- Output                          |
  |      +--+--+                                 |
  |      |NMOS | <- Input=1: ON -> Output=0      |
  |      +--+--+                                 |
  |         |                                    |
  |  GND ---+                                    |
  |                                              |
  |  -> PMOS and NMOS operate complementarily    |
  |  -> No through-current from Vdd to GND       |
  |     in steady state                          |
  |  -> Low power consumption = greatest          |
  |     advantage of CMOS                        |
  +---------------------------------------------+
```

### 1.3 Advantages of CMOS Logic

Here is a summary of why CMOS (Complementary MOS) technology has become the standard for modern integrated circuits.

| Property | CMOS | TTL | ECL |
|----------|------|-----|-----|
| Static power consumption | Extremely low | Medium | High |
| Dynamic power consumption | Only during switching | Constant | Constant |
| Noise margin | Large (~40% of Vdd) | Medium | Small |
| Integration density | Extremely high | Low | Low |
| Speed | Fast (advanced process) | Medium | Very fast |
| Manufacturing cost | Low (mass production) | Medium | High |

The greatest advantage of CMOS is that static power consumption is essentially zero. Because PMOS and NMOS operate complementarily, there is no direct path from Vdd to GND when the logic state is stable. Power is consumed only during switching transitions, which makes it possible to integrate billions of transistors on a single chip while keeping the Thermal Design Power (TDP) within a practical range.

### 1.4 Evolution of Transistor Counts

```
Moore's Law and Transistor Count Progression:

  Year    Chip                Transistors        Process
  --------------------------------------------------------
  1971    Intel 4004             2,300           10 um
  1978    Intel 8086            29,000            3 um
  1989    Intel 486          1,200,000            1 um
  1999    Pentium III         9,500,000          250 nm
  2006    Core 2 Duo        291,000,000           65 nm
  2012    Ivy Bridge      1,400,000,000           22 nm
  2020    Apple M1       16,000,000,000            5 nm
  2023    Apple M3 Max   92,000,000,000            3 nm
  2024    Apple M4       28,000,000,000            3 nm

  -> More than 10 orders of magnitude increase over approximately 50 years
  -> The pace of roughly doubling every 2 years is showing signs of slowing,
    but effective integration density continues to improve through
    3D stacking and chiplet technologies
```

---

## 2. Logic Gates — Physical Implementation of Boolean Algebra

### 2.1 Review of Boolean Algebra

Boolean algebra, systematized by George Boole (1854), is a mathematical system dealing with two values: TRUE/FALSE (1/0). Claude Shannon (1937) demonstrated the equivalence between relay circuits and Boolean algebra, establishing the mathematical foundation for logic circuit design.

**Fundamental Laws of Boolean Algebra:**

```
Identity:       A + 0 = A          A . 1 = A
Null:           A + 1 = 1          A . 0 = 0
Idempotent:     A + A = A          A . A = A
Complement:     A + A' = 1         A . A' = 0
Commutative:    A + B = B + A      A . B = B . A
Associative:    (A+B)+C = A+(B+C)  (A.B).C = A.(B.C)
Distributive:   A.(B+C) = A.B+A.C  A+(B.C) = (A+B).(A+C)
Absorption:     A + A.B = A        A . (A+B) = A
De Morgan's:    (A+B)' = A'.B'     (A.B)' = A'+B'
```

De Morgan's laws are particularly important. They show that AND and OR can be interconverted through NOT. This is the mathematical basis for the universality of NAND (and NOR).

### 2.2 CMOS Implementation of Basic Logic Gates

Each logic gate is implemented as a combination of CMOS transistors.

```
Basic Logic Gates — Truth Tables and CMOS Transistor Counts:

+---------+-------------------+-------------+-------------------+
| Gate    | Truth Table       | Transistors | CMOS Configuration|
+---------+-------------------+-------------+-------------------+
| NOT     | 0->1, 1->0       | 2 (1P+1N)   | Inverter          |
| NAND    | 11->0, else->1   | 4 (2P+2N)   | P parallel + N series|
| NOR     | 00->1, else->0   | 4 (2P+2N)   | P series + N parallel|
| AND     | 11->1, else->0   | 6 (2P+3N)   | NAND + NOT        |
| OR      | 00->0, else->1   | 6 (3P+2N)   | NOR + NOT         |
| XOR     | same->0, diff->1 | 8-12        | Compound gate     |
| XNOR    | same->1, diff->0 | 8-12        | XOR + NOT         |
+---------+-------------------+-------------+-------------------+

Note: NAND and NOR require fewer transistors than AND/OR
   -> In actual circuit design, NAND/NOR-based construction is more efficient
```

```
CMOS NAND Gate Circuit Configuration:

  Vdd --- + ------- + ---
           |         |
        +--+--+   +--+--+
  A ----|PMOS |   |PMOS |---- B
        +--+--+   +--+--+
           |         |
           +----+----+
                |
                +-------- Output Y = (A.B)'
                |
             +--+--+
  A ---------|NMOS |
             +--+--+
             +--+--+
  B ---------|NMOS |
             +--+--+
                |
  GND ----------+

  Operation:
  - A=0 or B=0 -> PMOS side conducts -> Y=Vdd(1)
  - A=1 and B=1 -> NMOS side conducts -> Y=GND(0)
  - PMOS is connected in parallel (either one ON conducts)
  - NMOS is connected in series (both must be ON to conduct)
```

### 2.3 Functional Completeness of NAND

The NAND gate is called a "universal gate," and any other logic gate can be constructed using only combinations of NAND gates. This property is called functional completeness.

```
Constructing Other Gates from NAND:

1. NOT(A) = NAND(A, A)

   A --+--[NAND]-- A'
       |
   A --+

2. AND(A, B) = NOT(NAND(A, B)) = NAND(NAND(A,B), NAND(A,B))

   A --+                    +--[NAND]-- A.B
   B --+--[NAND]--+---------+
                  +---------+

3. OR(A, B) = NAND(NOT(A), NOT(B)) = NAND(NAND(A,A), NAND(B,B))

   A --+--[NAND]--+
       |          +--[NAND]-- A+B
   A --+          |
   B --+--[NAND]--+
       |
   B --+

4. XOR(A, B) = constructible with 4 NAND gates

   A --+-----------[NAND]--+
       |                   |
       +--[NAND]--W--+-----+
       |              |    +--[NAND]-- A XOR B
   B --+              |    |
       |              +----+
       +--------------+
                      |
   B --+--------------+
       |
       +------[NAND]--+

   Simplified: W = NAND(A,B)
   XOR = NAND(NAND(A,W), NAND(B,W))
```

**Proof Outline:** Any Boolean function can be expressed in Sum of Products (SOP) form from a Karnaugh map or truth table. SOP is a combination of AND, OR, and NOT, all of which can be constructed from NAND as shown above. Therefore, NAND is functionally complete. NOR can be similarly proven to be functionally complete.

### 2.4 Propagation Delay and Fan-out

Real logic gates have non-ideal characteristics.

**Propagation Delay:** The time from an input change until the output stabilizes. In advanced processes (3nm-5nm), this is on the order of picoseconds (ps), but cascading multiple gate stages accumulates this delay. The critical path (the longest delay path) determines the upper limit of clock frequency.

**Fan-out:** The number of subsequent gate inputs that a single gate output can drive. Excessive fan-out degrades signal quality. In general, standard cell libraries use FO4 (fan-out-of-4 inverter delay) as the basic unit of delay.

**Glitch:** In combinational circuits, a phenomenon where the output temporarily shows an incorrect value due to differences in propagation delays along different paths. In synchronous designs, values are sampled only at clock edges, eliminating the effects of glitches.

---

## 3. Combinational Circuits — Combining Logic Gates

A combinational circuit is a circuit whose output depends only on the current inputs. It contains no internal memory elements and always returns the same output for the same input.

### 3.1 Half Adder and Full Adder

The most basic circuit for arithmetic operations is the adder.

```
Half Adder:

  Inputs: A, B (1 bit each)
  Outputs: Sum, Cout (carry)

  Circuit:
  A --+--[XOR]-- Sum = A XOR B
      |
  B --+
      |
      +--[AND]-- Cout = A . B

  Truth Table:
  +-----+-----+-----+------+
  |  A  |  B  | Sum | Cout |
  +-----+-----+-----+------+
  |  0  |  0  |  0  |  0   |
  |  0  |  1  |  1  |  0   |
  |  1  |  0  |  1  |  0   |
  |  1  |  1  |  0  |  1   |  <- 1+1 = 10 in binary
  +-----+-----+-----+------+
```

```
Full Adder:

  Inputs: A, B, Cin (carry from previous stage)
  Outputs: Sum, Cout

  Circuit (2 half adders + 1 OR gate):

  A --+--[HA1]--S1--+--[HA2]-- Sum = A XOR B XOR Cin
      |             |
  B --+    C1--+    |
               |  Cin --+    C2--+
               |                  |
               +--[OR]------------+-- Cout

  Logic Equations:
  Sum  = A XOR B XOR Cin
  Cout = (A . B) + (Cin . (A XOR B))

  Truth Table:
  +-----+-----+-----+-----+------+
  |  A  |  B  | Cin | Sum | Cout |
  +-----+-----+-----+-----+------+
  |  0  |  0  |  0  |  0  |  0   |
  |  0  |  0  |  1  |  1  |  0   |
  |  0  |  1  |  0  |  1  |  0   |
  |  0  |  1  |  1  |  0  |  1   |
  |  1  |  0  |  0  |  1  |  0   |
  |  1  |  0  |  1  |  0  |  1   |
  |  1  |  1  |  0  |  0  |  1   |
  |  1  |  1  |  1  |  1  |  1   |
  +-----+-----+-----+-----+------+
```

### 3.2 Ripple Carry Adder and Speedup Techniques

The simplest way to construct an N-bit adder is to connect N full adders in series — the Ripple Carry Adder.

```
4-bit Ripple Carry Adder:

  A0 B0   A1 B1   A2 B2   A3 B3
   |  |    |  |    |  |    |  |
   v  v    v  v    v  v    v  v
  +----+  +----+  +----+  +----+
  |FA0 |->|FA1 |->|FA2 |->|FA3 |-> Cout
  +----+  +----+  +----+  +----+
    |        |       |       |
    v        v       v       v
   S0       S1      S2      S3

  Cin=0 -> FA0 -> C1 -> FA1 -> C2 -> FA2 -> C3 -> FA3 -> Cout

  Delay: O(N) — Because the carry must propagate from the least significant to the most significant bit
  -> Delay is too large for 32-bit/64-bit adders
```

The Carry Lookahead Adder (CLA) was developed to solve this delay problem.

```
Carry Lookahead Adder (CLA) Principle:

  For each bit position i:
    Generate: Gi = Ai . Bi      (carry is generated at that bit)
    Propagate: Pi = Ai XOR Bi   (carry from previous stage is propagated)

  Carry Computation:
    C1 = G0 + P0.C0
    C2 = G1 + P1.G0 + P1.P0.C0
    C3 = G2 + P2.G1 + P2.P1.G0 + P2.P1.P0.C0
    C4 = G3 + P3.G2 + P3.P2.G1 + P3.P2.P1.G0 + P3.P2.P1.P0.C0

  -> All carries can be computed in parallel
  -> Delay: O(log N) — significant speedup

  Adder Speed Comparison:
  +----------------------+----------+----------+---------------+
  | Method               | Delay    | Area     | Characteristic|
  +----------------------+----------+----------+---------------+
  | Ripple Carry         | O(N)     | O(N)     | Minimum area  |
  | Carry Lookahead (CLA)| O(log N) | O(N^2)   | High speed    |
  | Carry Select         | O(sqrt N)| O(N)     | Balanced      |
  | Brent-Kung           | O(log N) | O(N logN)| Practical fast|
  | Kogge-Stone          | O(log N) | O(N logN)| Fastest       |
  +----------------------+----------+----------+---------------+
```

### 3.3 Multiplexer (MUX)

A multiplexer is a circuit that selects one of multiple inputs and routes it to the output. It is widely used in CPU data paths for register selection and ALU input switching.

```
2:1 Multiplexer:

  D0 --+
       +--[MUX]-- Y
  D1 --+
        |
  S ----+ (select signal)

  Y = S'.D0 + S.D1

  S=0 -> Y=D0
  S=1 -> Y=D1

4:1 Multiplexer:

  D0 --+
  D1 --+
       +--[MUX]-- Y
  D2 --+
  D3 --+
      | |
  S1 -+ +- S0

  Y = S1'.S0'.D0 + S1'.S0.D1 + S1.S0'.D2 + S1.S0.D3

  -> An N:1 MUX requires log2(N) select signals
  -> Any Boolean function can be implemented with a MUX (functions as a LUT)
```

**Universality of MUX:** An N-input Boolean function can be implemented using a 2^N : 1 MUX. Connect the inputs to the select signals and set the truth table output values on the data inputs. This principle forms the basis of FPGA LUTs (Look-Up Tables).

### 3.4 Decoder and Encoder

```
2:4 Decoder:

  Input A1 A0 -> Output Y3 Y2 Y1 Y0

  +----+----+----+----+----+----+
  | A1 | A0 | Y3 | Y2 | Y1 | Y0 |
  +----+----+----+----+----+----+
  |  0 |  0 |  0 |  0 |  0 |  1 |  <- Select address 0
  |  0 |  1 |  0 |  0 |  1 |  0 |  <- Select address 1
  |  1 |  0 |  0 |  1 |  0 |  0 |  <- Select address 2
  |  1 |  1 |  1 |  0 |  0 |  0 |  <- Select address 3
  +----+----+----+----+----+----+

  Applications:
  - Memory address decoder (address -> memory cell selection)
  - Instruction decoder (opcode -> control signals)
  - 7-segment display driver
```

### 3.5 ALU (Arithmetic Logic Unit)

The ALU is the core of the CPU, performing operations such as addition, subtraction, AND, OR, XOR, and shifts. The ALU is the culmination of the combinational circuits described above.

```
Structure of a Simple ALU:

  A[N-1:0] --+
              +--[ALU]-- Result[N-1:0]
  B[N-1:0] --+           Flags (Zero, Carry, Overflow, Negative)
              |
  Op[2:0] ---+ (operation select)

  Example Operation Codes:
  +--------+------------+----------------------+
  | Op     | Operation  | Internal Impl.       |
  +--------+------------+----------------------+
  | 000    | ADD        | Carry lookahead adder|
  | 001    | SUB        | Adder + 2's complement|
  | 010    | AND        | Bitwise AND          |
  | 011    | OR         | Bitwise OR           |
  | 100    | XOR        | Bitwise XOR          |
  | 101    | SLT        | Compare (Set Less Than)|
  | 110    | SLL        | Shift Left           |
  | 111    | SRL        | Shift Right          |
  +--------+------------+----------------------+

  -> A MUX selects the operation result and outputs it based on the Op code
  -> Flag outputs are used by conditional branch instructions
```

---

## 4. Sequential Circuits — Memory and State Transitions

A sequential circuit is a circuit whose output depends not only on the current inputs but also on past states. It contains feedback loops or memory elements, forming the physical basis of a computer's "memory."

### 4.1 SR Latch

The most basic memory element is the SR Latch (Set-Reset Latch).

```
NOR-type SR Latch:

  S --[NOR]--+-- Q
       ^     |
       |     |
       +-----+ (cross-coupled feedback)
             |
  R --[NOR]--+-- Q'
       ^     |
       |     |
       +-----+

  Operation Table:
  +-----+-----+------+------+----------------+
  |  S  |  R  |  Q   |  Q'  | Operation      |
  +-----+-----+------+------+----------------+
  |  0  |  0  | hold | hold | State hold     |
  |  1  |  0  |  1   |  0   | Set (Q=1)      |
  |  0  |  1  |  0   |  1   | Reset (Q=0)    |
  |  1  |  1  |  ?   |  ?   | Forbidden state|
  +-----+-----+------+------+----------------+

  -> In the forbidden state S=R=1, both Q and Q' become 0 (undefined)
  -> The D Latch and D Flip-Flop were developed to resolve this constraint
```

### 4.2 D Flip-Flop

The D Flip-Flop captures its input only at the edge (rising or falling) of a clock signal. It is the fundamental memory element in modern digital systems.

```
D Flip-Flop (Positive Edge-Triggered):

  D --[Master]--[Slave]-- Q
       ^                   |
  CLK -+                   +-- Q'

  Operation:
  +--------------+------+----------------------------+
  | CLK          | D    | Q                          |
  +--------------+------+----------------------------+
  | Rising edge  | 0    | 0 (captures D input value) |
  | Rising edge  | 1    | 1 (captures D input value) |
  | Otherwise    | X    | Hold (retains previous value)|
  +--------------+------+----------------------------+

  Timing Constraints:
  +---------------------------------------------+
  |                                             |
  |  Setup Time (tsu):                          |
  |    Time D must be stable before CLK edge    |
  |                                             |
  |  Hold Time (th):                            |
  |    Time D must be stable after CLK edge     |
  |                                             |
  |  Propagation Delay (tpd):                   |
  |    Time from CLK edge until Q is valid      |
  |                                             |
  |  ------D stable------+CLK^+---D stable---   |
  |  <--- tsu --->|        |<- th ->             |
  |              |        |<---- tpd ---->|      |
  |                                  Q valid    |
  |                                             |
  |  Clock period >= tpd + combinational delay + tsu |
  |  -> This determines the maximum operating frequency |
  +---------------------------------------------+
```

### 4.3 Registers and Counters

Arranging D Flip-Flops in parallel creates a register (an N-bit memory element), and adding feedback creates a counter.

```
4-bit Register:

  D3 -> [DFF] -> Q3
  D2 -> [DFF] -> Q2    All bits updated simultaneously on shared CLK
  D1 -> [DFF] -> Q1
  D0 -> [DFF] -> Q0
       ^
  CLK -+

4-bit Synchronous Binary Counter:

  CLK -> [DFF0] -> Q0 --+-- Output
              ^        |
              +-[NOT]--+ (T-type flip-flop behavior)

  Toggle condition for each stage: toggle when all lower bits are 1
  Q0: toggles every clock
  Q1: toggles when Q0=1
  Q2: toggles when Q0=Q1=1
  Q3: toggles when Q0=Q1=Q2=1

  Count: 0000 -> 0001 -> 0010 -> ... -> 1111 -> 0000
```

### 4.4 Finite State Machine (FSM)

The Finite State Machine is a general abstract model for sequential circuits, widely used in CPU control units and protocol processing.

```
FSM Structure:

  +--------------------------------------------+
  |  Input --> [Combinational Logic] --> Output |
  |               ^        |                   |
  |               |        v                   |
  |           [Register (state memory)]        |
  |               ^                            |
  |          CLK -+                            |
  +--------------------------------------------+

  Two Types of FSM:
  +--------------+--------------------------------------------+
  | Mealy        | Output = f(current state, input)           |
  |              | Output changes immediately on input change |
  +--------------+--------------------------------------------+
  | Moore        | Output = f(current state)                  |
  |              | Output changes after state transition      |
  |              | (1 clock delay). Fewer glitches, more stable|
  +--------------+--------------------------------------------+
```

**Example: Vending Machine Control FSM (Moore Type):**

```
State Transition Diagram (100 yen product, accepting 10 and 50 yen coins):

  [S0: 0 yen] --10 yen--> [S1: 10 yen] --10 yen--> [S2: 20 yen]
     |                    |                    |
    50 yen               50 yen              50 yen
     |                    |                    |
     v                    v                    v
  [S5: 50 yen] --10 yen--> [S6: 60 yen] --10 yen--> [S7: 70 yen]
     |                    |                    |
    50 yen               50 yen              ...
     |                    |
     v                    v                    Eventually
  [S10: 100 yen]       [S11: 110 yen]       -> [Dispense]
  -> Dispense product  -> Dispense + 10 yen change -> Return to S0
```

---

## 5. Circuit Description with Verilog HDL

### 5.1 Overview of HDL (Hardware Description Language)

HDL is a language for describing circuit structure and behavior in text form. The two major HDLs are Verilog and VHDL.

| Property | Verilog | VHDL |
|----------|---------|------|
| Origin | 1984 (Gateway Design Automation) | 1987 (U.S. Department of Defense) |
| Syntax style | Similar to C | Similar to Ada |
| Type system | Weakly typed | Strongly typed |
| Abstraction level | Gate-level to behavioral | Gate-level to system-level |
| Primary usage regions | North America & Asia | Europe & military |
| Latest standard | SystemVerilog (IEEE 1800) | VHDL-2019 (IEEE 1076) |

### 5.2 Code Example 1: Full Adder in Verilog

```verilog
// full_adder.v — 1-bit full adder
// Inputs: a, b, cin (carry input)
// Outputs: sum, cout (carry output)

module full_adder (
    input  wire a,
    input  wire b,
    input  wire cin,
    output wire sum,
    output wire cout
);

    // Structural description (gate level)
    wire w1, w2, w3;

    // Half adder 1: a + b
    xor g1 (w1, a, b);       // w1 = a XOR b
    and g2 (w2, a, b);       // w2 = a AND b

    // Half adder 2: w1 + cin
    xor g3 (sum, w1, cin);   // sum = w1 XOR cin = a XOR b XOR cin
    and g4 (w3, w1, cin);    // w3 = w1 AND cin

    // Carry output
    or  g5 (cout, w2, w3);   // cout = w2 OR w3

endmodule


// ----- Testbench -----
module full_adder_tb;
    reg a, b, cin;
    wire sum, cout;

    full_adder uut (.a(a), .b(b), .cin(cin), .sum(sum), .cout(cout));

    initial begin
        $display("a b cin | sum cout");
        $display("--------|----------");

        // Exhaust all input patterns
        {a, b, cin} = 3'b000; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b001; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b010; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b011; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b100; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b101; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b110; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        {a, b, cin} = 3'b111; #10;
        $display("%b %b  %b  |  %b    %b", a, b, cin, sum, cout);

        $finish;
    end
endmodule

// Simulation execution (Icarus Verilog):
//   iverilog -o full_adder_tb full_adder.v
//   vvp full_adder_tb
//
// Expected output:
//   a b cin | sum cout
//   --------|----------
//   0 0  0  |  0    0
//   0 0  1  |  1    0
//   0 1  0  |  1    0
//   0 1  1  |  0    1
//   1 0  0  |  1    0
//   1 0  1  |  0    1
//   1 1  0  |  0    1
//   1 1  1  |  1    1
```

### 5.3 Code Example 2: 4-bit Counter in Verilog

```verilog
// counter_4bit.v — Synchronous 4-bit up counter (with reset)
module counter_4bit (
    input  wire       clk,
    input  wire       rst_n,  // Active-low reset
    input  wire       en,     // Count enable
    output reg [3:0]  count
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            count <= 4'b0000;       // Asynchronous reset
        end else if (en) begin
            count <= count + 4'b0001; // Count up
        end
        // When en=0, count is held (implicit)
    end

endmodule


// ----- Testbench -----
module counter_4bit_tb;
    reg        clk, rst_n, en;
    wire [3:0] count;

    counter_4bit uut (
        .clk(clk), .rst_n(rst_n), .en(en), .count(count)
    );

    // Clock generation (period 10ns = 100MHz)
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("counter.vcd");
        $dumpvars(0, counter_4bit_tb);

        // Initialization and reset
        rst_n = 0; en = 0;
        #20;
        rst_n = 1;          // Release reset
        #10;
        en = 1;             // Start counting

        // Count for 20 clocks (one full cycle 0-15 plus extra)
        #200;

        en = 0;             // Stop counting
        #30;
        en = 1;             // Resume counting
        #50;

        $finish;
    end

    // Monitor
    always @(posedge clk) begin
        $display("Time=%0t rst_n=%b en=%b count=%d (%b)",
                 $time, rst_n, en, count, count);
    end
endmodule
```

### 5.4 Code Example 3: Bit-Operation-Only Adder in Python

```python
"""
bit_adder.py — Adder implemented using only bit operations
Achieves addition without using the + operator at all, using only AND, XOR, and shifts.
This is a software reproduction of the ripple carry behavior of full adders.
"""


def add_bitwise(a: int, b: int) -> int:
    """
    Add two integers using only bit operations.
    Loop until carry propagation becomes 0.

    Principle:
      sum = a XOR b        (sum of each bit without carry)
      carry = (a AND b) << 1  (carry shifted left by 1 bit)
      -> Add sum and carry again (repeat until carry is 0)

    >>> add_bitwise(0, 0)
    0
    >>> add_bitwise(5, 3)
    8
    >>> add_bitwise(255, 1)
    256
    """
    # Python has arbitrary-precision integers, so a mask is needed for negative numbers
    MASK = 0xFFFFFFFF  # 32-bit mask
    MAX_INT = 0x7FFFFFFF

    while b != 0:
        # Addition without carry
        sum_without_carry = (a ^ b) & MASK
        # Carry computation
        carry = ((a & b) << 1) & MASK
        # Next iteration
        a = sum_without_carry
        b = carry

    # Interpret as a 32-bit signed integer
    return a if a <= MAX_INT else ~(a ^ MASK)


def add_8bit(a: int, b: int) -> tuple[int, int]:
    """
    Simulation of an 8-bit adder.
    Simulates a full adder bit by bit.

    Returns: tuple of (result, carry_out)

    >>> add_8bit(100, 50)
    (150, 0)
    >>> add_8bit(200, 100)
    (44, 1)
    """
    result = 0
    carry = 0

    for i in range(8):
        # Extract the i-th bit
        bit_a = (a >> i) & 1
        bit_b = (b >> i) & 1

        # Full adder logic
        sum_bit = bit_a ^ bit_b ^ carry
        carry = (bit_a & bit_b) | (carry & (bit_a ^ bit_b))

        # Set in result
        result = result | (sum_bit << i)

    return (result, carry)


def subtract_twos_complement(a: int, b: int, bits: int = 8) -> int:
    """
    Subtraction using two's complement: a - b = a + (~b + 1)

    >>> subtract_twos_complement(10, 3)
    7
    >>> subtract_twos_complement(5, 8)
    -3
    """
    mask = (1 << bits) - 1  # 8-bit mask: 0xFF

    # Two's complement of b = bitwise inversion + 1
    b_complement = (~b & mask)
    result, _ = add_8bit(a & mask, b_complement)
    result, _ = add_8bit(result, 1)
    result = result & mask

    # Interpret as a signed integer
    if result >= (1 << (bits - 1)):
        return result - (1 << bits)
    return result


# ---- Tests ----
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

    print("\n=== Bit-Operation Adder Test ===")
    test_cases = [
        (0, 0), (1, 1), (5, 3), (127, 128),
        (255, 1), (100, 200), (42, 58)
    ]

    for a, b in test_cases:
        result = add_bitwise(a, b)
        result_8bit, carry = add_8bit(a & 0xFF, b & 0xFF)
        print(f"  {a:3d} + {b:3d} = {result:4d}"
              f"  (8bit: {result_8bit:3d}, carry={carry})")

    print("\n=== Two's Complement Subtraction Test ===")
    sub_cases = [(10, 3), (5, 8), (100, 100), (0, 1)]
    for a, b in sub_cases:
        result = subtract_twos_complement(a, b)
        print(f"  {a:3d} - {b:3d} = {result:4d}")
```

---

## 6. PCB (Printed Circuit Board) Design and Structure

### 6.1 What is a PCB?

A PCB (Printed Circuit Board) is a board that mechanically supports and electrically connects electronic components such as integrated circuits (ICs), resistors, capacitors, and connectors. It forms the foundation of all electronic devices (smartphones, PCs, appliances, automotive ECUs).

```
Basic PCB Structure (cross-section of a 4-layer board):

  +---------------------------------------------------------+
  |  #### Component ####  ## Component ##    # IC #          |  <- Mounted components
  +---------------------------------------------------------+
  | ### Copper pattern (Signal Layer L1: Top) ############## |  <- Signal traces
  +---------------------------------------------------------+
  | #### Prepreg (insulating layer) ######################## |  <- FR-4 substrate
  +---------------------------------------------------------+
  | ############ Copper fill (GND Layer L2) ################ |  <- Ground plane
  +---------------------------------------------------------+
  | ########## Core material (insulating layer) ############ |  <- FR-4 substrate
  +---------------------------------------------------------+
  | ############ Copper fill (Power Layer L3) ############## |  <- Power plane
  +---------------------------------------------------------+
  | #### Prepreg (insulating layer) ######################## |  <- FR-4 substrate
  +---------------------------------------------------------+
  | ### Copper pattern (Signal Layer L4: Bottom) ########### |  <- Signal traces
  +---------------------------------------------------------+
  |  ## Solder ##  #### Solder ####                          |  <- Bottom-side mounting
  +---------------------------------------------------------+

  Via (inter-layer connection):
  ------ # -------- <- L1
  ------ | -------- <- Prepreg
  ------ | -------- <- L2 (GND)
  ------ | -------- <- Core
  ------ | -------- <- L3 (Power)
  ------ | -------- <- Prepreg
  ------ # -------- <- L4

  Through-hole via: penetrates all layers
  Blind via:        from outer layer to inner layer (L1<->L2)
  Buried via:       between inner layers only (L2<->L3)
```

### 6.2 PCB Layer Configurations and Design Considerations

| Layers | Purpose | Typical Application |
|--------|---------|-------------------|
| 1 layer | Simplest. Copper pattern on one side | LED lighting, simple sensors |
| 2 layers | Traces on both sides. Standard for hobby electronics | Arduino, simple control boards |
| 4 layers | 2 signal layers + GND/power planes | Common consumer devices, IoT devices |
| 6-8 layers | High-density routing. Impedance control | Smartphones, network equipment |
| 10-16 layers | High-speed signals, complex SoC-mounted boards | Servers, GPU boards |
| 20+ layers | Ultra-high density. HDI (High Density Interconnect) | Cutting-edge smartphones, FPGA evaluation boards |

**Key Design Concepts:**

**Impedance Matching:** At high-speed signals (hundreds of MHz to GHz range), trace impedance significantly affects signal quality. Transmission line impedance is determined by trace width, substrate dielectric constant, and distance to the GND plane. Typically matched to 50 ohm (single-ended) or 100 ohm (differential pair).

**Crosstalk:** A phenomenon where signals leak between parallel traces due to electromagnetic coupling. Can be reduced by increasing trace spacing (3W rule: spacing of at least 3 times the trace width W).

**Power Integrity (PI):** Stability of the power plane. Decoupling capacitors are placed near ICs to bypass high-frequency noise. When splitting power planes, care must be taken not to break return current paths.

**Thermal Design:** Heat dissipation design for high-power components. Thermal vias (groups of vias that conduct heat to the GND plane) and heat sink placement are important.

### 6.3 PCB Manufacturing Process

```
Major Steps in PCB Manufacturing:

  1. Design Data Input (Gerber files)
     |
  2. Inner Layer Pattern Formation
     Apply dry film to substrate -> Exposure -> Development -> Etching -> Stripping
     |
  3. Lamination Press
     Inner layers + prepreg + copper foil -> Press at high temperature and pressure
     |
  4. Drilling
     Through-hole via drilling (mechanical drill or laser)
     |
  5. Plating
     Desmear -> Electroless copper plating -> Electrolytic copper plating
     |
  6. Outer Layer Pattern Formation
     Dry film -> Exposure -> Development -> Etching -> Stripping
     |
  7. Solder Resist
     Insulating coating on board surface (typically green)
     |
  8. Silkscreen Printing
     Markings for component numbers, terminal names, etc.
     |
  9. Surface Treatment
     HASL / ENIG / OSP and other solderability improvement treatments
     |
  10. Outline Processing & Inspection
      Router machining -> Electrical test (continuity/insulation) -> Shipment
```

---

## 7. FPGA — Reconfigurable Hardware

### 7.1 What is an FPGA?

An FPGA (Field-Programmable Gate Array) is a programmable device whose circuit configuration can be freely changed by the user after manufacturing. It occupies a position between ASICs (Application-Specific ICs) and general-purpose processors (CPUs).

```
Positioning of FPGA:

  Flexibility High <-----------------------------> Performance High
  +----------+----------+----------+----------+
  |  CPU     |  GPU     |  FPGA    |  ASIC    |
  |          |          |          |          |
  | General  | Parallel | Circuit  | Fixed    |
  | purpose  | compute  | can be   | circuit  |
  | software | oriented | rewritten| Max perf.|
  | execution|          |          | Max eff. |
  |          |          |          |          |
  | Most     | Data     | Proto-   | For mass |
  | flexible | parallel | typing   | production|
  | Slowest  |          |          | Huge dev cost|
  +----------+----------+----------+----------+

  Development cost:  CPU < GPU < FPGA <<< ASIC
  Performance/unit:  CPU < GPU ~ FPGA < ASIC
  Power efficiency:  CPU < GPU < FPGA < ASIC
  Time to market:    CPU ~ GPU > FPGA >> ASIC
```

### 7.2 Internal Structure of an FPGA

```
Basic Building Blocks of an FPGA:

  +----------------------------------------------+
  |  I/O Block   I/O Block   I/O Block   I/O    |
  |  +------+   +------+   +------+   +------+ |
  |  | IOB  |   | IOB  |   | IOB  |   | IOB  | |
  |  +------+   +------+   +------+   +------+ |
  |                                              |
  |  +------+   +------+   +------+             |
  |  | CLB  |===| CLB  |===| CLB  |  CLB:       |
  |  |      |   |      |   |      |  Configurable|
  |  +======+   +======+   +======+  Logic Block |
  |     ||          ||          ||                |
  |  +======+   +======+   +======+              |
  |  | CLB  |===| CLB  |===| CLB  |  === Routing |
  |  |      |   |      |   |      |  Resources   |
  |  +======+   +======+   +======+              |
  |     ||          ||          ||                |
  |  +======+   +======+   +======+              |
  |  | CLB  |===|BRAM  |===| CLB  |  BRAM:      |
  |  |      |   |      |   |      |  Block RAM   |
  |  +======+   +======+   +======+              |
  |                                              |
  |  +------+   +------+   +------+   +------+ |
  |  | DSP  |   | PLL  |   | DSP  |   |SERDES| |
  |  +------+   +------+   +------+   +------+ |
  |  DSP: Digital Signal Processing block        |
  |  PLL: Phase-Locked Loop (clock generation)   |
  |  SERDES: High-speed serial communication     |
  +----------------------------------------------+

  Internal Structure of a CLB:
  +----------------------------------------+
  |  CLB (Configurable Logic Block)        |
  |                                        |
  |  +------+  +------+                   |
  |  | LUT  |  | LUT  |  LUT: Look-Up     |
  |  |(6-in)|  |(6-in)|  Table            |
  |  +--+---+  +--+---+  -> 64-bit SRAM   |
  |     |         |        implements any  |
  |  +--+---+  +--+---+    6-input Boolean |
  |  | MUX  |  | MUX  |    function       |
  |  +--+---+  +--+---+                   |
  |     |         |                        |
  |  +--+---+  +--+---+                   |
  |  | FF   |  | FF   |  FF: Flip-Flop    |
  |  |(D-FF)|  |(D-FF)|                   |
  |  +--+---+  +--+---+                   |
  |     |         |                        |
  |     v         v                        |
  |   Output    Output                     |
  +----------------------------------------+
```

### 7.3 FPGA Application Areas

| Field | Application Example | Reason FPGA is Chosen |
|-------|--------------------|-----------------------|
| Telecommunications | 5G base stations, packet processing | Low-latency, high-throughput pipeline processing |
| Finance | High-Frequency Trading (HFT) | Sub-microsecond ultra-low latency |
| Image Processing | Medical imaging, industrial cameras | Real-time pipeline processing |
| AI Inference | Data center inference | INT8/FP16 parallel computation, low power |
| Cryptography | Hardware wallets | Secure key management, tamper resistance |
| Automotive | ADAS, autonomous driving | Functional safety (ISO 26262), low latency |
| Space | Satellite control | Radiation-hardened devices, on-orbit reconfiguration |
| Prototyping | ASIC verification | Behavioral verification before actual chip production |

### 7.4 Code Example 4: Simple UART Transmitter in Verilog for FPGA

```verilog
// uart_tx.v — Simple UART Transmitter
// Baud rate: parameterizable
// Frame: 8 data bits, 1 stop bit, no parity (8N1)

module uart_tx #(
    parameter CLK_FREQ  = 50_000_000,  // System clock 50MHz
    parameter BAUD_RATE = 115_200       // Baud rate
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       tx_start,   // Transmit start signal
    input  wire [7:0] tx_data,    // Transmit data
    output reg        tx_out,     // UART TX output
    output reg        tx_busy     // Transmitting flag
);

    // Baud rate divider counter limit
    localparam BAUD_DIV = CLK_FREQ / BAUD_RATE - 1;

    // State definitions
    localparam IDLE  = 2'b00;
    localparam START = 2'b01;
    localparam DATA  = 2'b10;
    localparam STOP  = 2'b11;

    reg [1:0]  state;
    reg [15:0] baud_cnt;    // Baud rate counter
    reg [2:0]  bit_idx;     // Transmit bit index (0-7)
    reg [7:0]  tx_shift;    // Shift register

    // Baud rate timing signal
    wire baud_tick = (baud_cnt == BAUD_DIV);

    // Baud rate counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            baud_cnt <= 16'd0;
        else if (state == IDLE)
            baud_cnt <= 16'd0;
        else if (baud_tick)
            baud_cnt <= 16'd0;
        else
            baud_cnt <= baud_cnt + 16'd1;
    end

    // Main state machine
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE;
            tx_out   <= 1'b1;     // High when idle
            tx_busy  <= 1'b0;
            bit_idx  <= 3'd0;
            tx_shift <= 8'd0;
        end else begin
            case (state)
                IDLE: begin
                    tx_out <= 1'b1;
                    if (tx_start) begin
                        state    <= START;
                        tx_shift <= tx_data;
                        tx_busy  <= 1'b1;
                    end
                end

                START: begin
                    tx_out <= 1'b0;           // Start bit (Low)
                    if (baud_tick) begin
                        state   <= DATA;
                        bit_idx <= 3'd0;
                    end
                end

                DATA: begin
                    tx_out <= tx_shift[0];    // LSB first
                    if (baud_tick) begin
                        tx_shift <= {1'b0, tx_shift[7:1]}; // Right shift
                        if (bit_idx == 3'd7)
                            state <= STOP;
                        else
                            bit_idx <= bit_idx + 3'd1;
                    end
                end

                STOP: begin
                    tx_out <= 1'b1;           // Stop bit (High)
                    if (baud_tick) begin
                        state   <= IDLE;
                        tx_busy <= 1'b0;
                    end
                end
            endcase
        end
    end

endmodule
```

---

## 8. Semiconductor Manufacturing Process

### 8.1 Silicon Wafer Manufacturing

Semiconductor manufacturing begins with sand (SiO2) and requires the highest precision of any industrial process in the world.

```
Silicon Wafer Manufacturing Process:

  1. Raw Material Refining
     +------------+     +--------------+     +--------------+
     | Silica     |---->| Metallurgical|---->| Trichlorosil.|
     | sand (SiO2)|     | Si refining  |     | distillation |
     | (sand)     |     | (purity 98%) |     | (purity 99.9%)|
     +------------+     +--------------+     +------+-------+
                                                    |
     +----------------------------------------------+
     v
  2. Polycrystalline Silicon Production
     +--------------+
     | Siemens      |  -> Purity 99.999999999% (eleven nines)
     | process      |     One of the purest man-made substances
     | CVD deposit  |     in the world
     +------+-------+
            |
  3. Single Crystal Growth (Czochralski method)
     +--------------------------------------+
     |         ^ Pull up + rotate           |
     |     +---+---+                        |
     |     | Seed  |                        |
     |     | crystal|                       |
     |     +---+---+                        |
     |         | Growing single crystal     |
     |         | ingot (300mm dia., 2m len.) |
     |     +---+---------------+            |
     |     |    Molten silicon |            |
     |     |   (~1420 deg C)   |            |
     |     +-------------------+            |
     +--------------------------------------+
     -> Perfectly regular crystal structure at the atomic level

  4. Wafer Processing
     Ingot -> Wire saw slicing -> Lapping
     -> Polishing -> Cleaning -> Inspection
     -> Surface roughness: atomic-level (< 0.5nm RMS)
```

### 8.2 Photolithography

Photolithography is the core process of semiconductor manufacturing, transferring circuit patterns onto the wafer.

```
Basic Steps of Photolithography:

  +----------------------------------------------+
  | Step 1: Film Deposition                       |
  | Deposit thin film (oxide, metal, etc.)        |
  | on wafer                                      |
  |                                               |
  | ########################  <- Deposited layer  |
  | ========================  <- Wafer            |
  +----------------------------------------------+
           |
           v
  +----------------------------------------------+
  | Step 2: Photoresist Coating                   |
  | Apply photosensitive resin uniformly by       |
  | spin coating                                  |
  |                                               |
  | ######################  <- Photoresist        |
  | ########################  <- Deposited layer  |
  | ========================  <- Wafer            |
  +----------------------------------------------+
           |
           v
  +----------------------------------------------+
  | Step 3: Exposure                              |
  | Illuminate through mask (reticle)             |
  |                                               |
  |     Light source (ArF excimer 193nm /         |
  |                    EUV 13.5nm)                |
  |        v  v  v  v  v  v                       |
  |     +--+--+--+--+--+--+                      |
  |     | Mask (circuit pattern)|                  |
  |     | ##    ####    ##     |                  |
  |     +-----------------------+                 |
  |        v       v       v                      |
  |     +----------------------+                  |
  |     | Reduction lens (4:1) |                  |
  |     +----------------------+                  |
  |        v       v       v                      |
  |     ##  ######  ####  ##  <- Exposed areas    |
  |                              are altered      |
  |     ########################                  |
  |     ========================                  |
  +----------------------------------------------+
           |
           v
  +----------------------------------------------+
  | Step 4: Development                           |
  | Positive: dissolve exposed areas              |
  | Negative: dissolve unexposed areas            |
  |                                               |
  |     ##          ####      <- Resist pattern   |
  |     ########################                  |
  |     ========================                  |
  +----------------------------------------------+
           |
           v
  +----------------------------------------------+
  | Step 5: Etching                               |
  | Remove deposited layer where resist is absent |
  |                                               |
  |     ##          ####      <- Resist (protect) |
  |     ##          ####      <- Patterned layer  |
  |     ========================                  |
  +----------------------------------------------+
           |
           v
  +----------------------------------------------+
  | Step 6: Resist Stripping                      |
  | Remove resist that is no longer needed        |
  |                                               |
  |     ##          ####      <- Finished pattern |
  |     ========================                  |
  |                                               |
  | -> This process is repeated tens to over a    |
  |   hundred times to form 3D transistor         |
  |   structures                                  |
  +----------------------------------------------+
```

### 8.3 EUV Lithography

For process nodes of 7nm and below, EUV (Extreme Ultraviolet, wavelength 13.5nm) lithography was introduced to surpass the resolution limits of conventional ArF immersion lithography (wavelength 193nm).

| Item | ArF Immersion | EUV |
|------|--------------|-----|
| Wavelength | 193 nm | 13.5 nm |
| Light source | Excimer laser | Laser-Produced Plasma (LPP) |
| Medium | Lenses (refractive optics) | Mirrors (reflective optics) |
| Environment | Atmospheric/water | Ultra-high vacuum |
| Multi-patterning | Required (SAQP, etc.) | Not required (single exposure) |
| Equipment cost | ~$100M | ~$200-400M |
| Supplier | Multiple | ASML (sole supplier worldwide) |
| Applicable node | Down to 7nm | Primary for 5nm and beyond |

The EUV light source generates 13.5nm light by irradiating droplets of molten tin with a CO2 laser in a vacuum chamber, turning them into plasma. Since this light is absorbed by all materials, lenses cannot be used; multilayer mirrors (Mo/Si reflective mirrors) are used for focusing and imaging.

### 8.4 Evolution of Advanced Process Nodes

```
Evolution of Transistor Structures:

  -- Planar (through ~22nm) --
  +----------------------------+
  |     Gate                   |
  |  +---+---+                 |
  |  | Oxide |                 |  <- Channel is a 2D plane under the gate
  |  +-------+                 |     Gate control on only 1 side
  |--| Chan  |--               |     -> Leakage current increases with scaling
  | S+-------+D               |
  +----------------------------+

  -- FinFET (22nm through ~3nm) --
  +----------------------------+
  |        Gate                |
  |     +--+--+               |
  |  +--+     +--+            |  <- Channel is a 3D fin structure
  |  |  |     |  |            |     Gate controls from 3 sides
  |  |  | Fin |  |            |     -> Leakage current greatly reduced
  |--|  |     |  |--          |     -> First introduced at Intel 22nm (2012)
  | S|  |     |  |D           |
  |  +--+     +--+            |
  +----------------------------+

  -- GAA FET / Nanosheet (2nm onward) --
  +----------------------------+
  |        Gate                |
  |  +-------------+          |
  |  | +---------+ |          |  <- Nanosheets (thin sheet-like channels)
  |  | | Sheet 3 | |          |     Gate controls from all 4 sides
  |  | +---------+ |          |     -> Best gate control
  |  | | Sheet 2 | |          |     -> Samsung 3nm GAA (2022)
  |  | +---------+ |          |     -> Intel 20A/18A planned
  |  | | Sheet 1 | |          |     -> TSMC N2 (2025) introduction
  |--| +---------+ |--        |
  | S+-------------+D         |
  +----------------------------+

  -- CFET (Future) --
  +----------------------------+
  |  +-------------+          |
  |  |   PMOS      |          |  <- PMOS and NMOS stacked vertically
  |  | +---------+ |          |     Area reduced by approximately half
  |  | | P-Sheet | |          |     -> Candidate technology for sub-1nm
  |  +-+---------+-+          |
  |  |   NMOS      |          |
  |  | +---------+ |          |
  |  | | N-Sheet | |          |
  |  | +---------+ |          |
  |  +-------------+          |
  +----------------------------+
```

### 8.5 Manufacturing Cost and Yield

| Process Node | Design Cost (est.) | Wafer Cost/unit | Transistor Density (MTr/mm2) | Major Foundries |
|-------------|-------------------|-----------------|--------------------------|---------------|
| 28nm | ~$50M | ~$3,000 | ~7 | TSMC, Samsung, GlobalFoundries |
| 14nm | ~$100M | ~$5,000 | ~30 | TSMC, Samsung, Intel |
| 7nm | ~$300M | ~$10,000 | ~96 | TSMC, Samsung |
| 5nm | ~$500M | ~$17,000 | ~171 | TSMC, Samsung |
| 3nm | ~$600-800M | ~$20,000+ | ~292 | TSMC |
| 2nm | ~$1B+ | ~$30,000+ | ~400+ | TSMC, Intel, Samsung |

-> Advanced processes reach design costs of hundreds of millions to billions of dollars, requiring mass production to recover the investment. Consequently, the number of fabless companies able to use leading-edge nodes is narrowing to a select few such as Apple, Qualcomm, NVIDIA, and AMD.

---

## 9. Latest Technology Trends and Future Outlook

### 9.1 Chiplet Technology

Instead of scaling up a single monolithic die, the mainstream approach is increasingly to divide functional blocks into separate dies (chiplets) connected via high-speed interconnects.

```
Chiplet Architecture Example (AMD EPYC):

  +------------------------------------------------------+
  |                Package Substrate                      |
  |                                                      |
  |  +---------+  +---------+  +---------+              |
  |  | CCD 0   |  | CCD 1   |  | CCD 2   |   CCD:     |
  |  |(CPU Core |  |(CPU Core |  |(CPU Core |   Core     |
  |  | Die)     |  | Die)     |  | Die)     |   Complex  |
  |  | 5nm      |  | 5nm      |  | 5nm      |   Die      |
  |  +----+-----+  +----+-----+  +----+-----+           |
  |       |             |             |                   |
  |  =====+=============+=============+================  |
  |                Infinity Fabric                        |
  |  =====+===========================================   |
  |       |                                              |
  |  +----+------------------------------+              |
  |  |           IOD                      |   IOD:      |
  |  |  (I/O Die: Memory Controller,     |   I/O Die   |
  |  |   PCIe, USB, etc.)                |   6nm       |
  |  |   6nm process                     |              |
  |  +-----------------------------------+              |
  |                                                      |
  +------------------------------------------------------+

  Advantages:
  - Each chiplet manufactured on the optimal process
    (Compute cores = leading-edge 5nm, I/O = cost-effective 6nm)
  - Improved yield (smaller dies have lower defect rates)
  - Design reusability (same CCD with variable count)
  - Flexible product lineup (differentiated by CCD count)
```

### 9.2 3D Stacking Technology

```
Major 3D Stacking Technologies:

  TSV (Through-Silicon Via):
  +-------------+
  |   Die 2     |
  |  +--+ +--+  |  <- Upper die
  |  |  | |  |  |
  +--+  +-+  +--+  <- Micro-bumps
  |  |  | |  |  |
  |  |TSV| |TSV| |  <- Through-silicon electrodes
  |  |  | |  |  |     (diameter 5-10 um)
  |  +--+ +--+  |
  |   Die 1     |
  +-------------+

  HBM (High Bandwidth Memory):
  +-------------+
  |  DRAM Die 4 |
  +-------------+
  |  DRAM Die 3 |   <- 4-16 DRAM dies stacked
  +-------------+      Connected by TSV between layers
  |  DRAM Die 2 |      Bandwidth: HBM3E at 1.2TB/s+
  +-------------+
  |  DRAM Die 1 |
  +-------------+
  |  Logic Die  |   <- Controller (base die)
  +------+------+
         |
    Interposer (in 2.5D implementation)
```

### 9.3 Post-Moore's Law Era

As the physical limits of scaling approach, multiple parallel approaches are being pursued to achieve continued performance improvement.

| Approach | Overview | Status |
|----------|---------|--------|
| GAA/CFET | Innovation in transistor structures | GAA: in production, CFET: research stage |
| Chiplets | Splitting and reintegrating dies | Production track record at AMD, Intel |
| 3D Stacking | Vertical integration density improvement | Commercialized with HBM. Logic 3D at research stage |
| New Materials | GaN, SiC, 2D materials | GaN/SiC commercialized for power semiconductors |
| Optical Interconnects | Chip-to-chip optical communication | Research to prototype stage |
| Quantum Computing | Quantum mechanical superposition | NISQ era. Limited advantage demonstrated |
| Neuromorphic | Brain-inspired computing | Intel Loihi 2, etc. Research stage |

---

## 10. Anti-Patterns and Design Pitfalls

### Anti-Pattern 1: Misuse of Asynchronous Reset

```
[Mistake] Generating an asynchronous reset from combinational logic output

  // Dangerous example
  wire async_rst = some_combinational_logic;  // Glitches can occur

  always @(posedge clk or posedge async_rst) begin
      if (async_rst)
          q <= 0;
      else
          q <= d;
  end

  Problem:
  - Combinational logic outputs can produce glitches (momentary erroneous pulses)
  - Asynchronous reset reacts to glitches, causing unintended resets
  - Extremely difficult to debug (timing-dependent intermittent failures)

  [Solution] Use a reset synchronizer for the reset signal

  // Correct example: Reset synchronizer
  reg rst_meta, rst_sync;

  always @(posedge clk or negedge rst_n_ext) begin
      if (!rst_n_ext) begin
          rst_meta <= 1'b0;
          rst_sync <= 1'b0;
      end else begin
          rst_meta <= 1'b1;      // Metastability resolution stage
          rst_sync <= rst_meta;  // Stabilization
      end
  end

  // Use rst_sync as the system reset
  // -> Reset de-assertion is synchronized to the clock;
  //    reset assertion is asynchronous (safe)
```

### Anti-Pattern 2: Improper Signal Passing Between Clock Domains

```
[Mistake] Directly connecting signals between different clock domains

  // clk_a domain
  always @(posedge clk_a) begin
      signal_a <= some_logic;
  end

  // clk_b domain — Dangerous! Using signal_a directly
  always @(posedge clk_b) begin
      result <= signal_a;        // Metastability may occur
  end

  Problem:
  - If clk_a and clk_b are asynchronous, setup/hold time violations occur
  - The flip-flop enters a metastable state, producing undefined output
  - Undefined values propagate to downstream circuits, destabilizing the entire system

  Metastable State:
  +---------------------------------------------+
  |   Normal transition:    0 -> 1 (reliably stable) |
  |   Metastable:          0 -> ??? (stalls at    |
  |                         intermediate value)     |
  |                                                |
  |   Voltage                                      |
  |   Vdd ------------------- <- Normal High      |
  |                                                |
  |   Vdd/2 - - - - - - - - - <- Metastable       |
  |          (may stall here for an extended period)|
  |   GND ------------------- <- Normal Low       |
  +---------------------------------------------+

  [Solution] Two-stage flip-flop synchronizer (Double-FF synchronizer)

  // Correct example: CDC (Clock Domain Crossing) synchronizer
  reg signal_meta, signal_sync;

  always @(posedge clk_b or negedge rst_n) begin
      if (!rst_n) begin
          signal_meta <= 1'b0;
          signal_sync <= 1'b0;
      end else begin
          signal_meta <= signal_a;     // Stage 1 (metastability absorption)
          signal_sync <= signal_meta;  // Stage 2 (stabilization)
      end
  end

  // Use signal_sync safely in the clk_b domain
  // -> MTBF (Mean Time Between Failures) is greatly improved
  // -> For multi-bit buses, use FIFOs or Gray code counters
```

### Anti-Pattern 3: Unintended Latch Inference

```
[Mistake] Not covering all conditions in if-else or case statements

  // Dangerous example: case statement without default clause
  always @(*) begin
      case (sel)
          2'b00: out = a;
          2'b01: out = b;
          2'b10: out = c;
          // 2'b11 is missing -> a latch will be inferred
      endcase
  end

  Problem:
  - The synthesis tool determines it must hold the value for undefined conditions
  - A latch (level-sensitive storage element) is generated instead of a flip-flop
  - Unintended latches complicate timing analysis and become a source of bugs

  [Solution] Cover all conditions or always include a default clause

  // Correct example
  always @(*) begin
      case (sel)
          2'b00:   out = a;
          2'b01:   out = b;
          2'b10:   out = c;
          default: out = 1'b0;  // All conditions covered
      endcase
  end
```

---

## 11. Hands-On Exercises

### Exercise 1: Logic Circuit Design (Basic — For Beginners)

**Task A:** Draw circuit diagrams using only NAND gates to construct the following gates:
1. NOT gate (1 input)
2. AND gate (2 inputs)
3. OR gate (2 inputs)
4. XOR gate (2 inputs)

**Task B:** Simplify the following Boolean expression using a Karnaugh map and determine the number of minterms:
```
F(A, B, C, D) = Sigma m(0, 1, 2, 5, 8, 9, 10)
```

**Task C:** Design a circuit using only NAND gates that realizes the following truth table:
```
A B C | Y
------+---
0 0 0 | 1
0 0 1 | 0
0 1 0 | 1
0 1 1 | 1
1 0 0 | 0
1 0 1 | 0
1 1 0 | 1
1 1 1 | 0
```

**Hint:** First convert to SOP (Sum of Products) form, then transform to NAND representation using De Morgan's law.

### Exercise 2: HDL Implementation (Applied — For Intermediate)

**Task A:** Implement an 8-bit ALU in Verilog supporting the following operations:
- ADD (addition)
- SUB (subtraction: two's complement addition)
- AND, OR, XOR (bitwise operations)
- SLL, SRL (logical shifts)
- SLT (Set Less Than: comparison)

Include flag outputs for Zero (result is zero), Carry, Overflow, and Negative.

**Task B:** Create a comprehensive testbench for the ALU from Task A that covers all operations and flags. Pay special attention to boundary values (0x00, 0x7F, 0x80, 0xFF).

**Task C:** Based on the ALU above, design a simple 4-instruction CPU (LOAD, STORE, ADD, JUMP) in Verilog and verify its operation through simulation.

### Exercise 3: Semiconductor and PCB (Advanced — For Experts)

**Task A:** Design a PCB with the following specifications using KiCad (an open-source EDA tool):
- ATmega328P (Arduino-compatible) microcontroller
- USB-C connector (power supply)
- 4 LEDs (for GPIO output verification)
- 2-layer board, within 50mm x 30mm board dimensions

**Task B:** Complete Projects 1-5 of Nand2Tetris ("The Elements of Computing Systems") to progressively build from NAND gates to ALU and CPU.

**Task C:** Using an FPGA evaluation board (Digilent Basys 3, Terasic DE10-Lite, etc.), implement a UART communication module and achieve bidirectional communication with a PC. Use 115200bps baud rate and 8N1 frame format.

---

## 12. FAQ

### Q1: Do software engineers need to learn about circuits and semiconductors?

**A:** It is not mandatory, but it is important background knowledge directly relevant to the following:

- **Why computers use binary** — Because the two states of a transistor (ON/OFF, high/low voltage) can be distinguished most reliably. Noise immunity decreases with three or more values.
- **Why floating-point numbers have errors** — The significand of IEEE 754 has a finite bit width and cannot exactly represent infinite decimals. This stems from hardware bit-width constraints.
- **Why CPU clock frequencies have an upper limit** — Physical constraints including wire delay, gate delay, and power consumption (dynamic power proportional to f x V^2 x C). Clock frequency hit a wall around 2005 (Power Wall), driving the shift to multi-core.
- **Why memory access is slow** — DRAM requires capacitor charge/read operations, making it inherently slower than SRAM (cache). Understanding circuits helps with cache hierarchy design.
- **Why certain parallel patterns are fast** — GPU and FPGA parallel architectures directly exploit the parallelism of combinational circuits.

### Q2: How do you choose between FPGA and ASIC?

**A:** The decision criteria mainly revolve around three axes: "production volume," "development time," and "performance requirements":

| Criterion | Choose FPGA | Choose ASIC |
|-----------|------------|-------------|
| Production volume | Up to tens of thousands | Hundreds of thousands or more |
| Development time | Need market entry within months | Can tolerate 1-2 years of development |
| Performance | Sufficient performance achievable with FPGA | Extreme performance/power efficiency needed |
| Cost | Low volume, no NRE needed | Can recover NRE (hundreds of millions to billions) |
| Modifiability | Post-shipment updates needed | Specifications finalized, no changes needed |
| Risk | Want to minimize initial risk | Have a thoroughly verified design |

A common path is to prototype with FPGA, then migrate to ASIC once mass production is decided (FPGA-to-ASIC migration).

### Q3: Is Moore's Law coming to an end?

**A:** Moore's Law in its original sense — "the number of transistors on a die doubles every two years" — is slowing due to physical limits. However, in the broader sense of "exponential improvement in computing capability," it continues through the following means:

1. **Transistor structure innovation** — FinFET -> GAA -> CFET, expanding control surfaces in the vertical direction
2. **Chiplets and 3D stacking** — Evolving in terms of total transistors per package, rather than transistors per unit area
3. **Architecture improvements** — Performance per watt improvement through specialized accelerators (GPU, TPU, NPU)
4. **New computing paradigms** — Quantum computing, optical computing, neuromorphic computing

The 2nm node is scheduled for mass production in 2025-2026, and 1.4nm (TSMC A14) is on the roadmap for around 2028. The end of physical scaling is still several generations away, but below 1nm, transistor channel lengths reach just tens of atoms, making the battle against quantum mechanical effects (tunneling) a serious challenge.

### Q4: What benefits do software engineers gain from learning PCB design?

**A:** In the fields of IoT, robotics, and embedded systems, the boundary between hardware and software is becoming increasingly blurred. Having basic PCB design knowledge allows you to:

- Communicate more effectively with hardware teams
- Design software-side mitigations for signal integrity, noise, and EMC issues
- Create IoT prototypes yourself (low-cost small-lot manufacturing with KiCad + JLCPCB, etc.)
- Correctly interpret oscilloscope and logic analyzer measurements during debugging

### Q5: Should I learn Verilog or VHDL?

**A:** It is practical to follow the standard of your industry or region. However, you can use the following guidance:

- **Verilog / SystemVerilog** — Dominant in North American and Asian semiconductor companies. C-like syntax with a gentle learning curve. SystemVerilog is also the standard for verification (UVM). When in doubt, choose this.
- **VHDL** — Strong in European aerospace, defense, and space industries. Strong typing catches many errors at compile time. Preferred for safety-critical systems.

In either case, once you master one, transitioning to the other is straightforward. The essential concepts of circuit design are language-independent.

---


## FAQ

### Q1: What is the most important point in learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not only through theory but by actually writing code and verifying behavior.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals to jump into applications. We recommend thoroughly understanding the basic concepts explained in this guide before moving to the next step.

### Q3: How is this applied in practice?

Knowledge of this topic is frequently used in everyday development work. It becomes particularly important during code reviews and architecture design.

---

## 13. Summary

| Concept | Key Point |
|---------|-----------|
| Transistor | ON/OFF of MOSFET is the physical foundation of all computation. CMOS configuration achieves low power |
| Logic Gates | NOT, AND, OR, NAND, NOR, XOR. Any logic function can be constructed from NAND alone (universality) |
| Combinational Circuits | Adders, MUX, decoders, ALU. Depend only on current inputs. Speedup via carry lookahead |
| Sequential Circuits | Latches, flip-flops, counters, FSM. Store state and operate clock-synchronously |
| HDL | Circuits described in Verilog / VHDL. Synthesis tools map to logic gates |
| PCB | Physical foundation for electronic components. Multilayer structure for impedance control and EMI countermeasures |
| FPGA | Programmable logic device. Implements any circuit using LUT + FF + routing |
| Semiconductor Mfg. | Repeated photolithography. EUV enables 5nm and beyond. Evolving from FinFET to GAA |
| Future Outlook | Continued performance improvement via chiplets, 3D stacking, new materials, and quantum computing |

```
Knowledge Hierarchy (Bottom-Up):

  +-------------------------------------------+
  |           Software                        |
  |   OS / Applications / AI Models           |
  +-------------------------------------------+
  |       Instruction Set Architecture (ISA)  |  <- HW/SW boundary
  +-------------------------------------------+
  |         Microarchitecture                 |
  |   CPU Pipeline / Cache / Branch Prediction|
  +-------------------------------------------+
  |           Digital Logic Circuits           |  <- Core of this chapter
  |   ALU / Register File / Control Unit      |
  +-------------------------------------------+
  |           Logic Gates                     |  <- Foundation of this chapter
  |   NAND / NOR / NOT / XOR / MUX           |
  +-------------------------------------------+
  |        CMOS Transistors                   |
  |   NMOS / PMOS / FinFET / GAA             |
  +-------------------------------------------+
  |          Semiconductor Physics            |
  |   PN Junction / Band Gap / Doping        |
  +-------------------------------------------+
```

---

## Appendix A. Code Example 5: Logic Gate Simulator in Python

By building a logic circuit simulation in software, you can gain a deeper understanding of how hardware works. The following is a Python simulator that defines basic gates and constructs and evaluates combinational circuits.

```python
"""
logic_simulator.py — Logic circuit simulator
Define basic gates and combine them to build and evaluate more complex circuits.
An educational tool for verifying the universality of NAND in practice.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
from itertools import product


# ============================================================
# Basic Gate Definitions
# ============================================================

def gate_not(a: int) -> int:
    """NOT gate: inverts the input"""
    return 1 - a

def gate_and(a: int, b: int) -> int:
    """AND gate: 1 only when both inputs are 1"""
    return a & b

def gate_or(a: int, b: int) -> int:
    """OR gate: 1 when either input is 1"""
    return a | b

def gate_nand(a: int, b: int) -> int:
    """NAND gate: negation of AND. Universal gate."""
    return 1 - (a & b)

def gate_nor(a: int, b: int) -> int:
    """NOR gate: negation of OR. Universal gate."""
    return 1 - (a | b)

def gate_xor(a: int, b: int) -> int:
    """XOR gate: exclusive OR"""
    return a ^ b

def gate_xnor(a: int, b: int) -> int:
    """XNOR gate: negation of exclusive OR"""
    return 1 - (a ^ b)


# ============================================================
# Constructing Other Gates from NAND Only (Proof of Universality)
# ============================================================

def nand_not(a: int) -> int:
    """Construct NOT from NAND: NOT(A) = NAND(A, A)"""
    return gate_nand(a, a)

def nand_and(a: int, b: int) -> int:
    """Construct AND from NAND: AND(A,B) = NOT(NAND(A,B))"""
    return nand_not(gate_nand(a, b))

def nand_or(a: int, b: int) -> int:
    """Construct OR from NAND: OR(A,B) = NAND(NOT(A), NOT(B))"""
    return gate_nand(nand_not(a), nand_not(b))

def nand_xor(a: int, b: int) -> int:
    """Construct XOR from NAND (4 NAND gates)"""
    w = gate_nand(a, b)
    return gate_nand(gate_nand(a, w), gate_nand(b, w))


# ============================================================
# Combinational Circuits: Half Adder, Full Adder, N-bit Adder
# ============================================================

@dataclass
class AdderResult:
    """Adder output"""
    sum_bits: list[int]
    carry_out: int

    def to_int(self) -> int:
        """Convert bit list to integer (LSB first)"""
        result = 0
        for i, bit in enumerate(self.sum_bits):
            result |= (bit << i)
        return result

    def __repr__(self) -> str:
        bits_str = ''.join(str(b) for b in reversed(self.sum_bits))
        return f"AdderResult(bits={bits_str}, carry={self.carry_out}, value={self.to_int()})"


def half_adder(a: int, b: int) -> tuple[int, int]:
    """
    Half adder: computes sum with XOR, carry with AND

    >>> half_adder(0, 0)
    (0, 0)
    >>> half_adder(1, 1)
    (0, 1)
    """
    return (gate_xor(a, b), gate_and(a, b))


def full_adder(a: int, b: int, cin: int) -> tuple[int, int]:
    """
    Full adder: constructed from 2 half adders + OR

    >>> full_adder(1, 1, 1)
    (1, 1)
    """
    s1, c1 = half_adder(a, b)
    s2, c2 = half_adder(s1, cin)
    return (s2, gate_or(c1, c2))


def ripple_carry_adder(a_bits: list[int], b_bits: list[int],
                        cin: int = 0) -> AdderResult:
    """
    N-bit ripple carry adder

    >>> r = ripple_carry_adder([1,0,1], [1,1,0])  # 5 + 6
    >>> r.to_int()
    11
    >>> r.carry_out
    0
    """
    n = len(a_bits)
    assert len(b_bits) == n, "Input bit lengths must match"

    sum_bits = []
    carry = cin

    for i in range(n):
        s, carry = full_adder(a_bits[i], b_bits[i], carry)
        sum_bits.append(s)

    return AdderResult(sum_bits=sum_bits, carry_out=carry)


# ============================================================
# Circuit Graph Representation (Netlist)
# ============================================================

@dataclass
class Wire:
    """Wire (signal line)"""
    name: str
    value: int = 0


@dataclass
class Gate:
    """Logic gate node"""
    name: str
    gate_type: str
    inputs: list[str]
    output: str
    func: Callable


@dataclass
class Circuit:
    """Netlist representation of a logic circuit"""
    name: str
    input_names: list[str]
    output_names: list[str]
    gates: list[Gate] = field(default_factory=list)

    def add_gate(self, name: str, gate_type: str,
                 inputs: list[str], output: str,
                 func: Callable) -> None:
        """Add a gate to the circuit"""
        self.gates.append(Gate(name, gate_type, inputs, output, func))

    def evaluate(self, input_values: dict[str, int]) -> dict[str, int]:
        """
        Evaluate the circuit with given input values and return all signal values.
        Simple implementation that evaluates in topological order.
        """
        signals = dict(input_values)
        evaluated = set(input_values.keys())
        remaining = list(self.gates)

        max_iterations = len(self.gates) * 2
        iteration = 0

        while remaining and iteration < max_iterations:
            iteration += 1
            progress = False
            next_remaining = []

            for gate in remaining:
                if all(inp in evaluated for inp in gate.inputs):
                    args = [signals[inp] for inp in gate.inputs]
                    signals[gate.output] = gate.func(*args)
                    evaluated.add(gate.output)
                    progress = True
                else:
                    next_remaining.append(gate)

            remaining = next_remaining
            if not progress and remaining:
                raise ValueError(
                    f"Circuit has feedback loops or undefined signals: "
                    f"{[g.name for g in remaining]}"
                )

        return {name: signals.get(name, 0) for name in self.output_names}

    def truth_table(self) -> list[dict[str, int]]:
        """Generate truth table for all input patterns"""
        rows = []
        for values in product([0, 1], repeat=len(self.input_names)):
            inputs = dict(zip(self.input_names, values))
            outputs = self.evaluate(inputs)
            rows.append({**inputs, **outputs})
        return rows

    def print_truth_table(self) -> None:
        """Print formatted truth table"""
        table = self.truth_table()
        headers = self.input_names + self.output_names
        print(" | ".join(f"{h:>5}" for h in headers))
        print("-" * (7 * len(headers)))
        for row in table:
            print(" | ".join(f"{row[h]:>5}" for h in headers))


# ============================================================
# Usage Example: Building a 2-bit adder as a circuit graph
# ============================================================

def build_2bit_adder() -> Circuit:
    """Build a 2-bit adder at the gate level"""
    circuit = Circuit(
        name="2bit_adder",
        input_names=["a0", "a1", "b0", "b1"],
        output_names=["s0", "s1", "cout"]
    )

    # Bit 0: Half adder (sufficient since Cin=0)
    circuit.add_gate("xor0", "XOR", ["a0", "b0"], "s0", gate_xor)
    circuit.add_gate("and0", "AND", ["a0", "b0"], "c0", gate_and)

    # Bit 1: Full adder
    circuit.add_gate("xor1", "XOR", ["a1", "b1"], "p1", gate_xor)
    circuit.add_gate("and1", "AND", ["a1", "b1"], "g1", gate_and)
    circuit.add_gate("xor2", "XOR", ["p1", "c0"], "s1", gate_xor)
    circuit.add_gate("and2", "AND", ["p1", "c0"], "pc1", gate_and)
    circuit.add_gate("or1",  "OR",  ["g1", "pc1"], "cout", gate_or)

    return circuit


# ============================================================
# Universality Verification
# ============================================================

def verify_nand_universality() -> None:
    """Verify NAND gate universality for all input patterns"""
    print("=== NAND Gate Universality Verification ===\n")

    # NOT verification
    print("NOT: nand_not(a) == gate_not(a)")
    for a in [0, 1]:
        assert nand_not(a) == gate_not(a), f"NOT mismatch: a={a}"
    print("  -> All patterns match\n")

    # AND, OR, XOR verification
    for name, nand_fn, ref_fn in [
        ("AND", nand_and, gate_and),
        ("OR",  nand_or,  gate_or),
        ("XOR", nand_xor, gate_xor),
    ]:
        print(f"{name}: nand_{name.lower()}(a,b) == gate_{name.lower()}(a,b)")
        for a, b in product([0, 1], repeat=2):
            result = nand_fn(a, b)
            expected = ref_fn(a, b)
            assert result == expected, \
                f"{name} mismatch: a={a}, b={b}, got={result}, expected={expected}"
        print("  -> All patterns match\n")

    print("Universality of all gates has been verified.\n")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # Universality verification
    verify_nand_universality()

    # 2-bit adder truth table
    print("=== 2-bit Adder Truth Table ===\n")
    adder = build_2bit_adder()
    adder.print_truth_table()

    # Ripple carry adder test
    print("\n=== 8-bit Ripple Carry Adder ===\n")
    test_pairs = [(100, 55), (200, 100), (255, 1), (0, 0), (127, 128)]
    for a, b in test_pairs:
        a_bits = [(a >> i) & 1 for i in range(8)]
        b_bits = [(b >> i) & 1 for i in range(8)]
        result = ripple_carry_adder(a_bits, b_bits)
        print(f"  {a:3d} + {b:3d} = {result.to_int():3d} (carry={result.carry_out})")
```

---

## Appendix B. Physics of Power Consumption — Why Power Matters

The power consumption of digital circuits is directly linked to software performance. The "Power Wall" problem — where CPU clock frequencies plateaued at around 3-4GHz around 2005 — stems from the physical constraints of power consumption.

### B.1 Three Components of CMOS Power Consumption

```
Total CMOS Circuit Power Consumption:

  P_total = P_dynamic + P_short + P_static

  1. Dynamic Power (P_dynamic):
     P_dynamic = alpha * C_L * V_dd^2 * f

     alpha: Activity factor (switching rate, 0-1)
     C_L:   Load capacitance (wire capacitance + gate capacitance)
     V_dd:  Supply voltage
     f:     Operating frequency

     -> Proportional to the square of voltage -> dramatically reduced by lowering voltage
     -> Proportional to frequency -> increases linearly with clock speed
     -> In advanced processes, V_dd is approximately 0.65-0.8V

  2. Short-Circuit Power (P_short):
     Through-current during switching transitions when PMOS/NMOS are simultaneously ON
     -> Approximately 10-15% of dynamic power
     -> Can be reduced by shortening input signal transition times

  3. Static Power (P_static):
     P_static = I_leak * V_dd

     I_leak: Leakage current (sub-threshold leakage + gate leakage)
     -> Even when a transistor is OFF, quantum mechanical tunneling causes a small current
     -> Leakage increases as devices shrink (gate oxide becomes thinner)
     -> In advanced processes (3nm and below), accounts for 30-50% of total power
```

### B.2 The Power Wall and Its Countermeasures

```
Collapse of Dennard Scaling (1974):

  Ideal (Dennard Scaling):
  +------------------------------------+
  | Process scaling -> Transistor shrinks|
  | -> Voltage decreases proportionally |
  | -> Power density remains constant   |
  | -> Increasing frequency doesn't     |
  |    increase power                   |
  +------------------------------------+

  Reality (~2005 onward):
  +------------------------------------+
  | Voltage stalls due to threshold    |
  |   voltage constraints              |
  | -> Leakage current increases       |
  |    exponentially                   |
  | -> Power density increases         |
  | -> Increasing frequency causes     |
  |    heat to exceed limits           |
  | -> "Power Wall" = frequency wall   |
  +------------------------------------+

  Evolution of Countermeasures:
  Year        Approach                    Example
  -------------------------------------------------
  2005~       Multi-core                  Core 2 Duo
  2007~       Dynamic Voltage/Freq. ctrl  Intel SpeedStep
  2010~       Heterogeneous design        ARM big.LITTLE
  2015~       Domain-specific accelerators GPU, TPU, NPU
  2020~       Chiplets + DVFS             AMD Zen 3/4
  2023~       Ultra-low voltage + 3D      Apple M3 (Efficiency Core)
```

### B.3 Implications for Software Engineers

Understanding the physics of power consumption reveals the direction for software optimization:

| Physical Principle | Software Response |
|-------------------|-------------------|
| Reducing activity factor | Eliminate unnecessary computation, zero-skipping |
| Data movement costs more than computation | Optimize cache efficiency (data locality) |
| SIMD is more power-efficient than scalar | Leverage vectorization |
| Specialized hardware is more efficient than general-purpose | Appropriate use of GPU/TPU/NPU |
| Clock frequency is constant | Improve performance through concurrent/parallel programming |
| Memory access is the dominant power factor | Optimize memory access patterns |

---

## Appendix C. Digital Design Workflow — From RTL to GDS II

### C.1 ASIC Design Flow

```
Digital Circuit Design Flow (for ASIC):

  +-------------------------------------------------------+
  | 1. Specification                                       |
  |    Functional spec -> Architecture design ->           |
  |    Microarchitecture design                            |
  +-----------------------+-------------------------------+
                          v
  +-------------------------------------------------------+
  | 2. RTL Design (Register Transfer Level)                |
  |    Coding in Verilog / SystemVerilog / VHDL            |
  |    -> Logic simulation (functional verification)       |
  |    -> Formal verification (equivalence checking)       |
  |    -> Code coverage analysis                           |
  +-----------------------+-------------------------------+
                          v
  +-------------------------------------------------------+
  | 3. Logic Synthesis                                     |
  |    RTL -> Gate-level netlist                            |
  |    Tools: Synopsys Design Compiler, Cadence Genus      |
  |    -> Optimization based on timing constraints (SDC)   |
  |    -> Area/power/speed tradeoff adjustment             |
  +-----------------------+-------------------------------+
                          v
  +-------------------------------------------------------+
  | 4. Place & Route                                       |
  |    Physical placement of gates -> Routing determination|
  |    Tools: Synopsys ICC2, Cadence Innovus               |
  |    -> Timing closure (all paths meet constraints)      |
  |    -> DRC (Design Rule Check)                          |
  |    -> LVS (Layout vs. Schematic)                       |
  +-----------------------+-------------------------------+
                          v
  +-------------------------------------------------------+
  | 5. Sign-off                                            |
  |    STA (Static Timing Analysis): verify all path delays|
  |    IR Drop Analysis: verify supply voltage drops       |
  |    EM Analysis: verify electromigration                |
  |    -> All verification pass -> GDSII file output       |
  +-----------------------+-------------------------------+
                          v
  +-------------------------------------------------------+
  | 6. Fab-out                                             |
  |    GDSII -> Mask fabrication -> Wafer manufacturing -> |
  |    Test -> Packaging -> Final test -> Shipment         |
  +-------------------------------------------------------+

  Approximate Timeline:
  +----------------+------------------+
  | Phase          | Duration         |
  +----------------+------------------+
  | RTL design/ver.| 6-18 months      |
  | Logic synthesis| 2-4 weeks        |
  | Place & route  | 4-12 weeks       |
  | Sign-off       | 2-4 weeks        |
  | Manufacturing  | 2-3 months       |
  | Test/prod.     | 1-2 months       |
  +----------------+------------------+
  | Total          | 1-2 years        |
  +----------------+------------------+
```

### C.2 FPGA Design Flow

In FPGAs, the ASIC "manufacturing" step is replaced by "configuration (programming)," making iteration dramatically faster.

```
FPGA Design Flow:

  RTL Design -> Simulation -> Synthesis -> Place & Route
  -> Timing Analysis -> Bitstream Generation -> Program FPGA

  Major Toolchains:
  +----------------+---------------------------+
  | Vendor         | Tools                     |
  +----------------+---------------------------+
  | AMD (Xilinx)  | Vivado / Vitis            |
  | Intel (Altera)| Quartus Prime             |
  | Lattice       | Radiant / iCEcube2        |
  | Microchip     | Libero SoC                |
  | Open Source   | Yosys + nextpnr + icestorm|
  +----------------+---------------------------+

  Open Source FPGA Toolchain (for iCE40 FPGAs):
  Verilog -> [Yosys] -> Netlist -> [nextpnr] -> Bitstream
  -> [iceprog] -> Program FPGA

  -> FPGA development possible without commercial tools
  -> Ideal for learning, education, and hobbyist use
```

---

## Appendix D. Practical PCB Design — Workflow with KiCad

### D.1 EDA Tool Comparison

| Tool | License | Features | Suited For |
|------|---------|----------|-----------|
| KiCad | Open Source (GPL) | Free, full-featured, active development | Hobby, education, small/medium projects |
| Altium Designer | Commercial (~$10K/yr) | Industry standard, feature-rich, extensive libraries | Professional design |
| Eagle (Fusion 360) | Commercial/limited free | Autodesk ecosystem | Hobby, small-scale projects |
| OrCAD/Allegro | Commercial | Large-scale design, high-speed signal support | High-speed/RF design |
| EasyEDA | Browser-based/free | Integration with JLCPCB, easy to use | Beginners, prototyping |

### D.2 KiCad Design Flow

```
PCB Design Workflow with KiCad:

  1. Schematic Editor (Eeschema)
     +-- Place symbols (component symbols)
     +-- Wiring (net connections)
     +-- Power/GND connections
     +-- ERC (Electrical Rule Check)
     +-- BOM (Bill of Materials) generation

  2. Footprint Assignment
     +-- Assign PCB footprints to each symbol
     +-- 0402, 0603, SOT-23, QFP, BGA, etc.

  3. PCB Editor (Pcbnew)
     +-- Define board outline
     +-- Component placement (manual + auto-placement)
     +-- Routing rule setup
     |   +-- Minimum trace width/clearance
     |   +-- Impedance control (differential pairs, etc.)
     |   +-- Power/GND trace widening
     +-- Routing (manual / interactive router)
     +-- GND copper fill injection
     +-- DRC (Design Rule Check)
     +-- 3D viewer confirmation

  4. Manufacturing Data Output
     +-- Gerber files (pattern data for each layer)
     +-- Drill files (hole position data)
     +-- BOM + CPL (component placement data)
     +-- -> Order from PCB manufacturers (JLCPCB, PCBWay, Elecrow, etc.)

  Cost estimate (2-layer board, 100mm x 100mm, 5 units):
  JLCPCB: approximately $2-$5 + shipping
  -> Very low cost prototyping is possible even for individual developers
```

---

## Appendix E. Historical Perspective — Physical Evolution of Electronic Computers

```
Evolution of Physical Computing Implementations:

  Era        Technology          Speed       Power      Integration
  --------------------------------------------------------------------
  1940s      Vacuum tubes        kHz range   Enormous   ENIAC: 17,468 tubes
             (electronic         (174kW)
              switches)

  1950s      Transistors         MHz range   Medium     Discrete components
             (point contact     (several kW)
              -> junction type)

  1960s      SSI/MSI ICs         Several MHz Small     Tens to hundreds of
             (Small/Medium       (several W)            devices per chip
              Scale Integration)

  1970s      LSI                 MHz range   Small     Thousands to tens of
             (Large Scale                               thousands of devices
              Integration)                              Intel 4004: 2,300

  1980s      VLSI               10 MHz range Medium    Tens to hundreds of
             (Very Large Scale                          thousands of devices
              Integration)                              Intel 386: 275,000

  1990s      ULSI               100 MHz range Medium   Millions to tens of
                                                        millions of devices
                                                        Pentium: 3.1M

  2000s      SoC                 GHz range   Tens of W  Hundreds of millions
             (System on Chip)                            Core 2: 291M

  2010s      FinFET SoC         Several GHz  Several   Tens to hundreds of
                                W to hundreds billions
                                 of W        Apple A14: 11.8B

  2020s      GAA / Chiplets     Several GHz  Several   Hundreds of billions
                                W to hundreds to trillions
                                 of W        Apple M3 Max: 92B

  -> Over 10 orders of magnitude increase in switching elements over 80 years
  -> Power consumption decreased dramatically from 174kW in the vacuum tube era
     to single-digit watts (per-element power improved by ~15 orders of magnitude)
```

---

## Appendix F. Fundamentals of Debugging and Verification

### F.1 Digital Circuit Debugging Techniques

| Technique | Overview | Applicable Stage |
|-----------|---------|-----------------|
| Waveform Simulation | Check all signal waveforms in a simulator | RTL design stage |
| Assertions | Formally describe design intent for automated verification | RTL to gate-level |
| Formal Verification | Mathematical proof of equivalence and properties | Post-synthesis |
| FPGA Prototyping | High-speed verification on actual silicon | Pre-production |
| Logic Analyzer | Simultaneously observe multiple signals on an actual board | Board debugging |
| Oscilloscope | Observe analog waveforms (signal quality check) | Board debugging |
| JTAG/Boundary Scan | Observe/control IC pin states externally | Board testing |

### F.2 Testbench Design Principles

```
Requirements for a Good Testbench:

  1. Self-checking
     -> Automatically compare expected values with actual outputs
     -> Do not rely on manual waveform visual inspection

  2. Reproducibility
     -> Fix random seeds for reproducibility
     -> Eliminate environment dependencies

  3. Coverage
     -> Exhaustive input patterns (for small circuits)
     -> Random tests + corner cases (for large circuits)
     -> Measure code coverage + functional coverage

  4. Incremental Testing
     -> Unit tests (individual modules)
     -> Integration tests (inter-module interfaces)
     -> System tests (overall behavior)

  5. Documentation
     -> Test item list
     -> Coverage reports
     -> Record of known limitations
```

---

## Appendix G. Glossary

| Term | Full Name | Description |
|------|-----------|-------------|
| ASIC | Application-Specific Integrated Circuit | IC for a specific purpose. Best performance for mass production |
| CMOS | Complementary MOS | Low-power technology combining NMOS and PMOS |
| CLB | Configurable Logic Block | Basic logic block in FPGAs. Composed of LUT + FF |
| DRC | Design Rule Check | Checks for design rule violations in PCB/IC layout |
| EDA | Electronic Design Automation | General term for electronic circuit design tools |
| EUV | Extreme Ultraviolet | 13.5nm wavelength extreme ultraviolet lithography |
| FinFET | Fin Field-Effect Transistor | FET with 3D fin structure. Used from 22nm to ~3nm |
| FPGA | Field-Programmable Gate Array | Programmable logic device |
| FSM | Finite State Machine | Basic model for control circuits |
| GAA | Gate-All-Around | FET structure where the gate surrounds the channel on all sides |
| GDS II | Graphic Data System II | Industry standard file format for IC layout |
| HDL | Hardware Description Language | Language for describing hardware (Verilog, VHDL, etc.) |
| HBM | High Bandwidth Memory | TSV-stacked high-bandwidth memory |
| LUT | Look-Up Table | Truth table implementation memory inside FPGAs |
| MOSFET | Metal-Oxide-Semiconductor FET | Metal-oxide-semiconductor field-effect transistor |
| PCB | Printed Circuit Board | Printed circuit board |
| RTL | Register Transfer Level | Abstraction level for HDL descriptions |
| SoC | System on Chip | Entire system integrated on a single chip |
| STA | Static Timing Analysis | Static timing analysis |
| TSV | Through-Silicon Via | Through-silicon electrode (for 3D stacking) |

---

## Next Guides to Read


---

## 14. References

1. Nisan, N. & Schocken, S. *The Elements of Computing Systems (Nand2Tetris).* MIT Press, 2nd Edition, 2021. — A seminal work on building from NAND gates to an OS bottom-up. Ideal for hands-on practice of this chapter's content.
2. Weste, N. & Harris, D. *CMOS VLSI Design: A Circuits and Systems Perspective.* 4th Edition, Pearson, 2010. — Standard textbook for CMOS circuit design. Covers transistor-level design through physical design.
3. Patterson, D. & Hennessy, J. *Computer Organization and Design: The Hardware/Software Interface.* 6th Edition (RISC-V Edition), Morgan Kaufmann, 2020. — Computer architecture textbook. Details ALU, datapath, and control unit design.
4. Harris, D. & Harris, S. *Digital Design and Computer Architecture.* 2nd Edition, Morgan Kaufmann, 2012. — A textbook covering digital circuits through processor design in one volume. Covers both Verilog and VHDL.
5. Mead, C. & Conway, L. *Introduction to VLSI Systems.* Addison-Wesley, 1980. — A classic on VLSI design. Historical work that established the methodology for chip design.
6. IEEE Standard 1364-2005 (Verilog) / IEEE Standard 1800-2017 (SystemVerilog). — Official Verilog/SystemVerilog specifications.
7. Wolf, W. *Modern VLSI Design: IP-Based Design.* 4th Edition, Pearson, 2008. — Explains modern IP-reuse-based VLSI design methodology.
