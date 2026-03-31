# Binary and Number Systems

> In the world of computers, everything is represented with 0s and 1s. This very "constraint" is the source of digital technology's reliability and universality.

## Learning Objectives

- [ ] Perform conversions between binary, octal, and hexadecimal
- [ ] Master bitwise operations (AND, OR, XOR, NOT, shift)
- [ ] Explain why computers use the binary system
- [ ] Acquire techniques for rapid base conversion
- [ ] Understand practical patterns for applying bitwise operations
- [ ] Correctly write numeric literals in various programming languages

## Prerequisites

- Basic mathematics (arithmetic operations)

---

## 1. Why Binary?

### 1.1 Physical Reasons

```
Transistor = Switch:

  ON  = High voltage = 1
  OFF = Low voltage  = 0

  With only two states, "binary" is natural

  If we tried to use ternary:
  - We would need to precisely distinguish three voltage levels
  - More susceptible to noise (two boundaries)
  - Circuit complexity increases

  Advantages of binary:
  - Noise-resistant (only one threshold)
  - Simple circuits (just ON/OFF)
  - Direct correspondence with logic and bitwise operations
```

### 1.2 Historical Background

The historical path to computers adopting binary lies at the intersection of mathematics, physics, and engineering.

```
Key events by era:

  1679  Leibniz publishes a systematic treatise on binary
        - Advocates "universal computation with 0 and 1"
        - Also noted to be influenced by the Chinese I Ching (yin-yang philosophy)

  1847  George Boole devises Boolean algebra
        - Expresses logic mathematically with two values: TRUE/FALSE
        - Formalizes the basic operations AND, OR, NOT

  1937  Claude Shannon's master's thesis
        - "A Symbolic Analysis of Relay and Switching Circuits"
        - Proves the correspondence between Boolean algebra and electrical circuits
        - Establishes the theoretical foundation for digital circuit design

  1945  Von Neumann's EDVAC report
        - Proposes a binary-based stored-program architecture
        - "Binary representation greatly simplifies circuit design"

  1947  Invention of the transistor (Bell Labs)
        - A highly reliable, compact switching element replacing vacuum tubes
        - ON/OFF two-state operation is extremely stable

  1958  Invention of the integrated circuit (IC)
        - Enables mass integration of transistors
        - Makes parallel processing in binary practical
```

### 1.3 Perspective from Information Theory

```
Shannon's Information Theory (1948):

  The fundamental unit of information = bit (binary digit)

  1 bit = the amount of information needed to "choose one from two equally probable alternatives"

  Information entropy H = -Sigma p(x) log2 p(x)

  Example: Coin toss (heads/tails equally probable)
    H = -(0.5 x log2(0.5) + 0.5 x log2(0.5))
    H = -(0.5 x (-1) + 0.5 x (-1))
    H = 1 bit

  Example: 8-sided die (each face equally probable)
    H = -8 x (1/8 x log2(1/8))
    H = -8 x (1/8 x (-3))
    H = 3 bits

  -> Binary directly represents the "smallest unit" of information
```

### 1.4 Why Not Decimal?

```
Problems with decimal computers:

  1. Circuit complexity
     - A circuit distinguishing 10 states is far more complex than one for 2 states
     - Error rate becomes orders of magnitude higher

  2. Incompatibility with logic operations
     - Boolean logic (TRUE/FALSE) naturally maps to binary
     - Logic operations are inefficient in decimal

  3. Historical attempts
     - ENIAC (1946) was decimal-based
     - A decimal adder requires about 3 times the circuitry of a binary one
     - From EDVAC onward, the shift to binary began

  Actual comparison (adder circuit scale):
     Binary full adder: 2 AND gates + 2 XOR gates + 1 OR gate
     Decimal adder: Tens of gates required + complex carry handling
```

---

## 2. Radix and Positional Notation

### 2.1 General Theory of Positional Notation

```
Base-N notation (radix N), general form:

  Value = Sigma(i=0 to n) d_i x N^i

  Where d_i is the digit at each position (0 <= d_i < N)

  Example: Decimal 4273
    = 4x10^3 + 2x10^2 + 7x10^1 + 3x10^0
    = 4000  + 200   + 70    + 3
    = 4273

  Example: Binary 1101
    = 1x2^3 + 1x2^2 + 0x2^1 + 1x2^0
    = 8    + 4    + 0    + 1
    = 13 (decimal)

  Example: Hexadecimal 0x2AF
    = 2x16^2 + A(10)x16^1 + F(15)x16^0
    = 512   + 160       + 15
    = 687 (decimal)

  Example: Octal 0o755
    = 7x8^2 + 5x8^1 + 5x8^0
    = 448   + 40   + 5
    = 493 (decimal)
```

### 2.2 Base Conversion

```
Decimal <-> Binary <-> Hexadecimal:

  Decimal  Binary          Hex      Octal
  ------------------------------------------
     0    0000 0000      0x00     000
     1    0000 0001      0x01     001
    10    0000 1010      0x0A     012
    42    0010 1010      0x2A     052
   127    0111 1111      0x7F     177
   128    1000 0000      0x80     200
   255    1111 1111      0xFF     377
   256    1 0000 0000    0x100    400

  Conversion method (decimal -> binary): Repeatedly divide by 2, read remainders in reverse
  42 / 2 = 21 remainder 0
  21 / 2 = 10 remainder 1
  10 / 2 =  5 remainder 0
   5 / 2 =  2 remainder 1
   2 / 2 =  1 remainder 0
   1 / 2 =  0 remainder 1
  -> 101010 (2) = 42 (10) check

  Hexadecimal groups 4 bits of binary:
  0010 1010 -> 2  A -> 0x2A check
```

### 2.3 Decimal to Other Bases (Detailed Procedure)

```
[Method 1: Division Method (Decimal -> Base N)]

  Decimal 173 -> Binary:
  173 / 2 = 86 remainder 1
   86 / 2 = 43 remainder 0
   43 / 2 = 21 remainder 1
   21 / 2 = 10 remainder 1
   10 / 2 =  5 remainder 0
    5 / 2 =  2 remainder 1
    2 / 2 =  1 remainder 0
    1 / 2 =  0 remainder 1
  -> 10101101 (2) check

  Decimal 173 -> Hexadecimal:
  173 / 16 = 10 remainder 13 (D)
   10 / 16 =  0 remainder 10 (A)
  -> 0xAD check

  Verification: 0xAD = 10x16 + 13 = 160 + 13 = 173 check

  Decimal 4096 -> Octal:
  4096 / 8 = 512 remainder 0
   512 / 8 =  64 remainder 0
    64 / 8 =   8 remainder 0
     8 / 8 =   1 remainder 0
     1 / 8 =   0 remainder 1
  -> 0o10000 check (= 8^4 = 4096)


[Method 2: Subtraction Method (Decimal -> Binary)]

  Use a table of powers of 2 (recommended to memorize):
  2^0=1, 2^1=2, 2^2=4, 2^3=8, 2^4=16, 2^5=32, 2^6=64, 2^7=128, 2^8=256

  Convert decimal 173 to binary:
  173 >= 128 -> 1 (173-128=45)
   45 >=  64 -> x -> 0
   45 >=  32 -> 1 (45-32=13)
   13 >=  16 -> x -> 0
   13 >=   8 -> 1 (13-8=5)
    5 >=   4 -> 1 (5-4=1)
    1 >=   2 -> x -> 0
    1 >=   1 -> 1 (1-1=0)
  -> 10101101 (2) check

  This method is faster than the division method once you get used to it
```

### 2.4 Binary-Hexadecimal Interconversion (Fast Method)

```
Binary -> Hexadecimal: Group into 4-bit chunks from right to left

  Correspondence table to memorize:
  0000=0  0001=1  0010=2  0011=3
  0100=4  0101=5  0110=6  0111=7
  1000=8  1001=9  1010=A  1011=B
  1100=C  1101=D  1110=E  1111=F

  Example: 1011 1110 0100 1101
           B    E    4    D
  -> 0xBE4D

  Hexadecimal -> Binary: Expand each digit to 4 bits

  Example: 0xCAFE
           C    A    F    E
           1100 1010 1111 1110
  -> 1100 1010 1111 1110


Binary -> Octal: Group into 3-bit chunks from right to left

  Example: 10 101 101
           2  5   5
  -> 0o255

  Octal -> Binary: Expand each digit to 3 bits

  Example: 0o755
           7   5   5
           111 101 101
  -> 111 101 101


Commonly seen conversion patterns in practice:
  0xFF     = 1111 1111               = 255
  0xFFFF   = 1111 1111 1111 1111     = 65535
  0xDEAD   = 1101 1110 1010 1101     = 57005
  0xBEEF   = 1011 1110 1110 1111     = 48879
  0xCAFE   = 1100 1010 1111 1110     = 51966
  0xC0FFEE = 1100 0000 1111 1111 1110 1110 = 12648430
```

### 2.5 Fractional Base Conversion

```
Decimal fraction -> Binary fraction: Multiply by 2 and extract the integer part

  Convert 0.625 to binary:
  0.625 x 2 = 1.25  -> integer part 1
  0.25  x 2 = 0.5   -> integer part 0
  0.5   x 2 = 1.0   -> integer part 1
  0.0   -> done
  -> 0.101 (2) check

  Verification: 0.101 = 1x2^-1 + 0x2^-2 + 1x2^-3
                       = 0.5   + 0     + 0.125
                       = 0.625 check


  Convert 0.1 to binary (example of a repeating fraction):
  0.1 x 2 = 0.2  -> 0
  0.2 x 2 = 0.4  -> 0
  0.4 x 2 = 0.8  -> 0
  0.8 x 2 = 1.6  -> 1
  0.6 x 2 = 1.2  -> 1
  0.2 x 2 = 0.4  -> 0  <- repeats from here
  0.4 x 2 = 0.8  -> 0
  ...
  -> 0.0001100110011... (2) = 0.0(0011) repeating

  This is the root cause of the "0.1 + 0.2 != 0.3" problem!
  -> See 03-floating-point.md for details


Binary fraction -> Decimal fraction:
  0.1101 (2) = 1x2^-1 + 1x2^-2 + 0x2^-3 + 1x2^-4
              = 0.5   + 0.25  + 0     + 0.0625
              = 0.8125 (10)
```

---

## 3. Bitwise Operations

### 3.1 Basic Operations

```python
# Bitwise operations in Python

# AND: 1 only if both are 1
a = 0b1100  # 12
b = 0b1010  # 10
print(bin(a & b))   # 0b1000 = 8

# OR: 1 if either is 1
print(bin(a | b))   # 0b1110 = 14

# XOR: 1 if different
print(bin(a ^ b))   # 0b0110 = 6

# NOT: Bit inversion
print(bin(~a & 0xFF))  # 0b11110011 = 243 (for 8 bits)

# Left shift: Multiply by 2 (shift each bit left)
print(bin(a << 1))  # 0b11000 = 24 (12 x 2)
print(bin(a << 3))  # 0b1100000 = 96 (12 x 8)

# Right shift: Divide by 2 (shift each bit right)
print(bin(a >> 1))  # 0b110 = 6 (12 / 2)
```

### 3.2 Truth Tables

```
AND (Logical Conjunction): 1 only when both are 1
  A  B  | A & B
  ------+------
  0  0  |  0
  0  1  |  0
  1  0  |  0
  1  1  |  1

  Use: Bit masking (extracting specific bits)
  Example: 0b1011_0110 & 0b0000_1111 = 0b0000_0110 (extract lower 4 bits)


OR (Logical Disjunction): 1 if either is 1
  A  B  | A | B
  ------+------
  0  0  |  0
  0  1  |  1
  1  0  |  1
  1  1  |  1

  Use: Setting flags (setting specific bits to 1)
  Example: 0b0000_0001 | 0b0000_0100 = 0b0000_0101 (set bits 0 and 2)


XOR (Exclusive OR): 1 when different
  A  B  | A ^ B
  ------+------
  0  0  |  0
  0  1  |  1
  1  0  |  1
  1  1  |  0

  Important properties of XOR:
  - A ^ A = 0 (XOR with itself is 0)
  - A ^ 0 = A (XOR with 0 is the original value)
  - A ^ B ^ B = A (XORing twice restores the original -> used in cryptography)
  - Commutative: A ^ B = B ^ A
  - Associative: (A ^ B) ^ C = A ^ (B ^ C)


NOT (Bit Inversion):
  A  | ~A
  ---+----
  0  |  1
  1  |  0

  Note: In Python, ~n = -(n+1) (two's complement representation)
  For 8 bits: ~0b0000_1111 = 0b1111_0000


NAND (Negated AND):
  A  B  | ~(A & B)
  ------+---------
  0  0  |  1
  0  1  |  1
  1  0  |  1
  1  1  |  0

  NAND is a universal gate: all logic operations can be constructed from NAND alone
  - NOT(A) = NAND(A, A)
  - AND(A, B) = NOT(NAND(A, B))
  - OR(A, B) = NAND(NOT(A), NOT(B))
```

### 3.3 Shift Operations in Detail

```
Left Shift (<<): Shifts bits left, fills right end with 0s

  0b0000_1100 << 1 = 0b0001_1000  (12 -> 24 = 12 x 2^1)
  0b0000_1100 << 2 = 0b0011_0000  (12 -> 48 = 12 x 2^2)
  0b0000_1100 << 3 = 0b0110_0000  (12 -> 96 = 12 x 2^3)

  General rule: x << n = x x 2^n (when no overflow occurs)


Logical Right Shift (>>>): Shifts bits right, fills left end with 0s
  - Java, JavaScript's >>> operator
  - Treats the value as an unsigned integer

  0b1000_0000 >>> 1 = 0b0100_0000  (128 -> 64)
  0b1000_0000 >>> 2 = 0b0010_0000  (128 -> 32)


Arithmetic Right Shift (>>): Shifts bits right, fills left end with the sign bit
  - C, Python, Java's >> operator (for signed integers)
  - For negative numbers, the left end is filled with 1

  Positive numbers:
  0b0110_0000 >> 1 = 0b0011_0000  (96 -> 48)

  Negative numbers (8-bit signed):
  0b1100_0000 >> 1 = 0b1110_0000  (-64 -> -32)
  0b1100_0000 >> 2 = 0b1111_0000  (-64 -> -16)

  General rule: x >> n = x / 2^n (rounds toward negative infinity)


Shift Operations vs Multiplication/Division Performance:
  - On modern CPUs, multiplication is nearly as fast (1 clock cycle)
  - Compilers automatically replace with shifts during optimization
  - It is fine to write x * 2 for readability
  - However, shifts are still advantageous in embedded systems in some cases
```

### 3.4 Practical Applications of Bitwise Operations

```python
# Practical use cases for bitwise operations

# 1. Flag management (bitfields)
READ    = 0b001  # 1
WRITE   = 0b010  # 2
EXECUTE = 0b100  # 4

permissions = READ | WRITE  # 0b011 = 3
has_read = bool(permissions & READ)     # True
has_execute = bool(permissions & EXECUTE)  # False

# File permissions: chmod 755 = rwxr-xr-x
# 7 = 111, 5 = 101, 5 = 101

# 2. Even/odd check (least significant bit)
is_even = (n & 1) == 0  # Same as n % 2 == 0 but faster

# 3. Power-of-2 check
is_power_of_2 = n > 0 and (n & (n - 1)) == 0
# 8 = 1000, 7 = 0111 -> 1000 & 0111 = 0000 -> True
# 6 = 0110, 5 = 0101 -> 0110 & 0101 = 0100 -> False

# 4. XOR swap (exchange without a temporary variable)
a ^= b
b ^= a
a ^= b
# Theoretically interesting, but in practice temp = a; a = b; b = temp is more readable

# 5. Bit masking
ip_address = 0xC0A80164  # 192.168.1.100
subnet_mask = 0xFFFFFF00  # 255.255.255.0
network = ip_address & subnet_mask  # 192.168.1.0
```

### 3.5 Advanced Bitwise Techniques

```python
# === Bit Count (popcount / Hamming Weight) ===

# Method 1: Naive approach
def popcount_naive(n):
    count = 0
    while n:
        count += n & 1
        n >>= 1
    return count

# Method 2: Brian Kernighan's algorithm (fast)
def popcount_kernighan(n):
    """n & (n-1) clears the lowest set bit"""
    count = 0
    while n:
        n &= n - 1  # Clear the lowest set bit
        count += 1
    return count

# Example: n = 0b1011_0100 (180)
# 1011_0100 & 1011_0011 = 1011_0000 (count=1)
# 1011_0000 & 1010_1111 = 1010_0000 (count=2)
# 1010_0000 & 1001_1111 = 1000_0000 (count=3)
# 1000_0000 & 0111_1111 = 0000_0000 (count=4)
# -> 4 set bits

# Method 3: Python built-in
bin(180).count('1')  # 4

# Method 4: Use CPU instructions directly (fastest)
# x86 POPCNT instruction, ARM VCNT instruction


# === Extract Lowest Set Bit ===
def lowest_set_bit(n):
    """Keep only the lowest set bit"""
    return n & (-n)  # Leverages two's complement property

# Example: n = 0b1010_1000
# -n = 0b0101_1000 (two's complement)
# n & (-n) = 0b0000_1000 -> bit 3 is the lowest set bit


# === Clear Lowest Set Bit ===
def clear_lowest_set_bit(n):
    return n & (n - 1)

# Example: n = 0b1010_1000
# n-1 = 0b1010_0111
# n & (n-1) = 0b1010_0000


# === Toggle Bits (specific range) ===
def toggle_bits(n, mask):
    """Toggle bits specified by the mask"""
    return n ^ mask

# Example: n = 0b1100_0011, mask = 0b0000_1111
# n ^ mask = 0b1100_1100 (lower 4 bits toggled)


# === Average of Two Values (overflow-safe) ===
def average_no_overflow(a, b):
    """Overflow-safe version of (a + b) / 2"""
    return (a & b) + ((a ^ b) >> 1)

# The usual (a + b) / 2 can overflow on a + b
# This method computes: common bits + differing bits / 2


# === Absolute Value (branchless) ===
def abs_branchless(n):
    """Absolute value of a 32-bit signed integer"""
    mask = n >> 31  # 0x00000000 if positive, 0xFFFFFFFF if negative
    return (n ^ mask) - mask

# For negative: mask = -1
# n ^ (-1) = ~n (bit inversion)
# ~n - (-1) = ~n + 1 = -n (two's complement)
```

### 3.6 Algorithmic Applications of Bitwise Operations

```python
# === Bitboard (Chess / Shogi AI) ===
# Represent an 8x8 board as a 64-bit integer

# Chess example: Calculate knight's reachable positions
def knight_moves(position):
    """Calculate knight move targets (bitboard)"""
    # Edge masks to prevent wraparound
    NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
    NOT_H_FILE = 0x7F7F7F7F7F7F7F7F
    NOT_AB_FILE = 0xFCFCFCFCFCFCFCFC
    NOT_GH_FILE = 0x3F3F3F3F3F3F3F3F

    moves = 0
    moves |= (position << 17) & NOT_A_FILE   # Up 2, right 1
    moves |= (position << 15) & NOT_H_FILE   # Up 2, left 1
    moves |= (position << 10) & NOT_AB_FILE  # Up 1, right 2
    moves |= (position <<  6) & NOT_GH_FILE  # Up 1, left 2
    moves |= (position >> 17) & NOT_H_FILE   # Down 2, left 1
    moves |= (position >> 15) & NOT_A_FILE   # Down 2, right 1
    moves |= (position >> 10) & NOT_GH_FILE  # Down 1, left 2
    moves |= (position >>  6) & NOT_AB_FILE  # Down 1, right 2
    return moves


# === Bloom Filter (Probabilistic Data Structure) ===
class BloomFilter:
    """Probabilistic set membership test using a bit array"""

    def __init__(self, size=1024):
        self.size = size
        self.bit_array = 0  # Use Python's arbitrary-precision integers as a bit array

    def _hashes(self, item):
        """Generate multiple hash values"""
        h1 = hash(item) % self.size
        h2 = hash(str(item) + "salt") % self.size
        h3 = hash(str(item) + "pepper") % self.size
        return [h1, h2, h3]

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array |= (1 << h)  # Set corresponding bit

    def might_contain(self, item):
        for h in self._hashes(item):
            if not (self.bit_array & (1 << h)):
                return False  # Definitely not contained
        return True  # Might be contained (possible false positive)


# === Subset Enumeration via Bit Manipulation ===
def enumerate_subsets(s):
    """Enumerate all subsets of bitmask s"""
    subset = s
    subsets = []
    while subset > 0:
        subsets.append(subset)
        subset = (subset - 1) & s
    subsets.append(0)  # Empty set
    return subsets

# Example: s = 0b1010 (set containing elements 2 and 0)
# Subsets: 1010, 1000, 0010, 0000
# -> {2,0}, {2}, {0}, {}


# === Bit Reversal (used in FFT) ===
def reverse_bits(n, bit_width=8):
    """Reverse the order of a bit string"""
    result = 0
    for _ in range(bit_width):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result

# Example: reverse_bits(0b10110010, 8) = 0b01001101
```

### 3.7 Bitwise Operations in Various Languages

```c
// C: Ideal for low-level operations

#include <stdint.h>

// Using unsigned integers is recommended (shift behavior is well-defined)
uint32_t set_bit(uint32_t n, int pos) {
    return n | (1U << pos);
}

uint32_t clear_bit(uint32_t n, int pos) {
    return n & ~(1U << pos);
}

uint32_t toggle_bit(uint32_t n, int pos) {
    return n ^ (1U << pos);
}

int test_bit(uint32_t n, int pos) {
    return (n >> pos) & 1;
}

// GCC built-in functions
int count = __builtin_popcount(0xFF);     // 8
int leading_zeros = __builtin_clz(0x10);  // 27
int trailing_zeros = __builtin_ctz(0x10); // 4
```

```java
// Java: Integer / Long utility methods

int n = 0b1010_1100;

// Bit count
int count = Integer.bitCount(n);        // 4

// Leading/trailing zero count
int leadingZeros = Integer.numberOfLeadingZeros(n);   // 24
int trailingZeros = Integer.numberOfTrailingZeros(n);  // 2

// Bit reversal
int reversed = Integer.reverse(n);

// Highest/lowest set bit
int highest = Integer.highestOneBit(n);   // 128 (0b10000000)
int lowest = Integer.lowestOneBit(n);     // 4   (0b00000100)

// Difference between logical right shift (>>>) and arithmetic right shift (>>)
int neg = -128;                  // 0xFFFFFF80
int arithmetic = neg >> 4;       // 0xFFFFFFF8 = -8 (sign preserved)
int logical = neg >>> 4;         // 0x0FFFFFF8 = 268435448 (zero-filled)
```

```go
// Go: math/bits package

package main

import (
    "fmt"
    "math/bits"
)

func main() {
    var n uint32 = 0b1010_1100

    // Bit count
    fmt.Println(bits.OnesCount32(n))     // 4

    // Leading/trailing zero count
    fmt.Println(bits.LeadingZeros32(n))  // 24
    fmt.Println(bits.TrailingZeros32(n)) // 2

    // Bit length
    fmt.Println(bits.Len32(n))           // 8

    // Bit reversal
    fmt.Println(bits.Reverse32(n))

    // Rotation
    fmt.Println(bits.RotateLeft32(n, 4))
}
```

```rust
// Rust: Built-in methods on types

fn main() {
    let n: u32 = 0b1010_1100;

    // Bit count
    println!("{}", n.count_ones());       // 4
    println!("{}", n.count_zeros());      // 28

    // Leading/trailing zero count
    println!("{}", n.leading_zeros());    // 24
    println!("{}", n.trailing_zeros());   // 2

    // Bit reversal
    println!("{}", n.reverse_bits());

    // Rotation
    println!("{}", n.rotate_left(4));
    println!("{}", n.rotate_right(4));

    // Overflow detection
    let (result, overflowed) = n.overflowing_add(u32::MAX);
    println!("result={}, overflow={}", result, overflowed);

    // Checked arithmetic
    match n.checked_add(u32::MAX) {
        Some(v) => println!("sum = {}", v),
        None => println!("overflow!"),
    }
}
```

---

## 4. Data Size Units

### 4.1 Bit and Byte Hierarchy

```
Bit and byte hierarchy:

  1 bit    = 0 or 1
  1 nibble = 4 bits   = 1 hexadecimal digit
  1 byte   = 8 bits   = 256 possible values (0-255)
  1 word   = 32 or 64 bits (CPU-dependent)

  Storage units:
  +------------+--------------+-----------------------+
  | Unit       | Decimal (SI) | Binary (IEC)          |
  +------------+--------------+-----------------------+
  | Kilo (K)   | 1,000        | 1,024 (2^10) = KiB   |
  | Mega (M)   | 1,000,000    | 1,048,576 (2^20) = MiB|
  | Giga (G)   | 10^9         | 2^30 = GiB            |
  | Tera (T)   | 10^12        | 2^40 = TiB            |
  | Peta (P)   | 10^15        | 2^50 = PiB            |
  | Exa (E)    | 10^18        | 2^60 = EiB            |
  +------------+--------------+-----------------------+

  Note: HDD/SSD manufacturers use SI units; OSes use IEC units
  -> A 1TB SSD shows as 931GB in the OS
  -> 1,000,000,000,000 / 1,073,741,824 ~ 931 GiB
```

### 4.2 Powers of 2 Reference Table

```
Powers of 2 every programmer should memorize:

  2^0  = 1
  2^1  = 2
  2^2  = 4
  2^3  = 8
  2^4  = 16
  2^5  = 32
  2^6  = 64
  2^7  = 128
  2^8  = 256        <- Range of 1 byte
  2^9  = 512
  2^10 = 1,024      <- 1 KiB
  2^11 = 2,048
  2^12 = 4,096      <- Common memory page size
  2^13 = 8,192
  2^14 = 16,384
  2^15 = 32,768
  2^16 = 65,536     <- Range of unsigned short
  2^20 = 1,048,576  <- 1 MiB
  2^24 = 16,777,216 <- 8 bits per RGB channel (TrueColor)
  2^30 = 1,073,741,824 <- ~1 billion ~ 1 GiB
  2^32 = 4,294,967,296 <- Range of unsigned int (~4.3 billion)
  2^40 = 1,099,511,627,776 <- 1 TiB
  2^64 = 18,446,744,073,709,551,616 <- Range of unsigned long long

  Approximation mnemonics:
  2^10 ~ 10^3  (1024 ~ 1000)
  2^20 ~ 10^6  (~1 million)
  2^30 ~ 10^9  (~1 billion)
  2^40 ~ 10^12
  2^50 ~ 10^15
  2^60 ~ 10^18
```

### 4.3 Capacity Estimation in Practice

```
Common data size guidelines:

  Text:
  - 1 ASCII character = 1 byte
  - 1 Japanese character (UTF-8) = 3 bytes
  - 1 page of text (40 lines x 80 characters) ~ 3.2 KB
  - 1 novel (~100,000 characters) ~ 300 KB (UTF-8)
  - All of English Wikipedia ~ 22 GB (as of 2024)

  Images:
  - 1920x1080 uncompressed (24-bit color) = 1920 x 1080 x 3 ~ 6.2 MB
  - Same size JPEG (quality 80) ~ 200-500 KB
  - Same size PNG ~ 1-3 MB
  - 4K image (3840x2160) uncompressed ~ 24.9 MB

  Audio:
  - CD quality (44.1kHz, 16-bit, stereo) = 44100 x 2 x 2 = 176.4 KB/s ~ 10.6 MB/min
  - MP3 128kbps ~ 960 KB/min ~ 1 MB/min
  - FLAC (lossless) ~ 5 MB/min

  Video:
  - 1080p uncompressed (30fps) ~ 186 MB/s ~ 11.2 GB/min
  - 1080p H.264 (streaming quality) ~ 5 Mbps ~ 37.5 MB/min
  - 4K H.265 ~ 15-25 Mbps

  Databases:
  - 1 user record (ID, name, email, etc.) ~ 200-500 bytes
  - 1 million users ~ 200-500 MB
  - Indexes are typically 10-30% of the table size
```

---

## 5. Numeric Literals in Various Programming Languages

### 5.1 Python

```python
# Python: Specify base with prefixes
decimal = 42        # Decimal
binary  = 0b101010  # Binary
octal   = 0o52      # Octal
hexadec = 0x2A      # Hexadecimal
# All equal 42

# Underscore separators (Python 3.6+)
large = 1_000_000_000  # 1 billion
binary_large = 0b1010_1010_1100_1100
hex_large = 0xFF_FF_FF_FF

# Conversion functions
bin(42)   # '0b101010'
oct(42)   # '0o52'
hex(42)   # '0x2a'
int('101010', 2)  # 42 (binary string -> integer)
int('2A', 16)     # 42 (hex string -> integer)
int('52', 8)      # 42 (octal string -> integer)

# Formatting
f"{42:b}"     # '101010'   (binary)
f"{42:o}"     # '52'       (octal)
f"{42:x}"     # '2a'       (hex lowercase)
f"{42:X}"     # '2A'       (hex uppercase)
f"{42:08b}"   # '00101010' (8-digit zero-padded binary)
f"{42:#010x}" # '0x0000002a' (prefixed 10-digit zero-padded hex)

# Python has arbitrary-precision integers
huge = 2 ** 1000  # Computes without issue
print(huge.bit_length())  # 1001 (number of bits)
```

### 5.2 JavaScript / TypeScript

```javascript
// JavaScript: Similar prefixes
const decimal = 42;
const binary  = 0b101010;
const octal   = 0o52;
const hexadec = 0x2A;

// BigInt: Large integers
const big = 9007199254740993n;  // 'n' suffix
const bigHex = 0x1FFFFFFFFFFFFFn;

// Number limits
console.log(Number.MAX_SAFE_INTEGER);  // 9007199254740991 (2^53 - 1)
console.log(Number.MIN_SAFE_INTEGER);  // -9007199254740991
// Precision is lost beyond this range

// Bitwise operations convert to 32-bit integers (caution!)
console.log(0xFFFFFFFF | 0);    // -1 (signed 32-bit)
console.log(0xFFFFFFFF >>> 0);  // 4294967295 (unsigned 32-bit)

// Conversion
(42).toString(2);    // '101010'
(42).toString(8);    // '52'
(42).toString(16);   // '2a'
parseInt('101010', 2);  // 42
parseInt('2A', 16);     // 42

// Handle binary data with TypedArrays
const buffer = new ArrayBuffer(4);
const view = new DataView(buffer);
view.setUint32(0, 0xDEADBEEF);
console.log(view.getUint8(0).toString(16));  // 'de'

// Direct operations with Uint8Array
const bytes = new Uint8Array([0xCA, 0xFE, 0xBA, 0xBE]);
```

### 5.3 Rust

```rust
// Rust: Type annotations + underscore separators
let decimal: u32 = 42;
let binary: u32  = 0b0010_1010;  // Underscores improve readability
let octal: u32   = 0o52;
let hexadec: u32 = 0x2A;
let byte: u8     = b'A';  // ASCII value (65)

// Type suffixes
let a = 42u8;    // u8 type
let b = 42i32;   // i32 type
let c = 42usize; // usize type

// Formatting
println!("{:b}", 42);     // "101010"
println!("{:08b}", 42);   // "00101010"
println!("{:o}", 42);     // "52"
println!("{:x}", 42);     // "2a"
println!("{:X}", 42);     // "2A"
println!("{:#010x}", 42); // "0x0000002a"

// Byte arrays
let bytes: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];

// Convert from string with from_str_radix
let n = u32::from_str_radix("101010", 2).unwrap();   // 42
let m = u32::from_str_radix("2A", 16).unwrap();      // 42

// Bit manipulation methods
let n: u32 = 42;
println!("{}", n.count_ones());      // 3
println!("{}", n.count_zeros());     // 29
println!("{}", n.leading_zeros());   // 26
println!("{}", n.trailing_zeros());  // 1
println!("{}", n.reverse_bits());    // Bit reversal
println!("{}", n.rotate_left(4));    // Left rotation
```

### 5.4 Go

```go
package main

import (
    "fmt"
    "strconv"
)

func main() {
    // Literals
    decimal := 42
    binary := 0b101010     // Go 1.13+
    octal := 0o52          // Go 1.13+ (legacy syntax: 052 also works)
    hexadec := 0x2A

    fmt.Println(decimal, binary, octal, hexadec) // 42 42 42 42

    // Underscore separators
    large := 1_000_000_000

    // Formatting
    fmt.Printf("%b\n", 42)    // 101010
    fmt.Printf("%08b\n", 42)  // 00101010
    fmt.Printf("%o\n", 42)    // 52
    fmt.Printf("%x\n", 42)    // 2a
    fmt.Printf("%X\n", 42)    // 2A
    fmt.Printf("%#x\n", 42)   // 0x2a

    // String conversion
    s := strconv.FormatInt(42, 2)   // "101010"
    n, _ := strconv.ParseInt("101010", 2, 64)  // 42

    _ = large
    _ = s
    _ = n
}
```

### 5.5 C / C++

```c
// C
#include <stdio.h>
#include <stdint.h>

int main() {
    int decimal = 42;
    int binary  = 0b00101010;  // C23 / GCC extension
    int octal   = 052;         // Leading 0 means octal (caution!)
    int hexadec = 0x2A;

    // Formatting
    printf("%d\n", 42);    // 42 (decimal)
    printf("%o\n", 42);    // 52 (octal)
    printf("%x\n", 42);    // 2a (hex lowercase)
    printf("%X\n", 42);    // 2A (hex uppercase)
    printf("%#x\n", 42);   // 0x2a (with prefix)

    // Fixed-width integer types (recommended)
    uint8_t  byte_val = 0xFF;       // Always 8 bits
    uint16_t short_val = 0xFFFF;    // Always 16 bits
    uint32_t int_val = 0xFFFFFFFF;  // Always 32 bits
    uint64_t long_val = 0xFFFFFFFFFFFFFFFFULL;  // Always 64 bits

    // Literal suffixes
    unsigned int u = 42U;
    long l = 42L;
    unsigned long ul = 42UL;
    long long ll = 42LL;
    unsigned long long ull = 42ULL;

    // C23 binary literals
    // int b = 0b1010; // C23 standard

    return 0;
}
```

```cpp
// C++: std::bitset
#include <bitset>
#include <iostream>

int main() {
    std::bitset<8> bits(42);          // "00101010"
    std::bitset<8> mask("11110000");  // Construct from string

    std::cout << bits << std::endl;          // 00101010
    std::cout << bits.count() << std::endl;  // 3 (number of 1s)
    std::cout << bits.size() << std::endl;   // 8
    std::cout << bits.test(1) << std::endl;  // 1 (is bit 1 set?)

    bits.set(0);    // Set bit 0
    bits.reset(3);  // Clear bit 3
    bits.flip(5);   // Toggle bit 5
    bits.flip();    // Toggle all bits

    // C++14 and later: binary literals
    auto b = 0b0010'1010;  // ' as digit separator

    // C++20: std::bit_cast, std::popcount, etc.
    // #include <bit>
    // int pc = std::popcount(42u);  // 3

    return 0;
}
```

---

## 6. Frequently Encountered Number Patterns in Practice

### 6.1 Memory Addresses and Alignment

```
Memory address alignment:

  Many CPUs require or recommend that data be aligned to specific boundaries.

  4-byte alignment:
    Lower 2 bits of the address are 00
    -> Address is a multiple of 4 (0x00, 0x04, 0x08, 0x0C, ...)

  8-byte alignment:
    Lower 3 bits of the address are 000
    -> Address is a multiple of 8 (0x00, 0x08, 0x10, 0x18, ...)

  Alignment calculation (bitwise):
    Round up address to N-byte boundary:
    aligned = (addr + (N - 1)) & ~(N - 1)

    Example: addr = 0x13, N = 8
    aligned = (0x13 + 0x07) & ~0x07
            = 0x1A & 0xFFFFFFF8
            = 0x18

  Alignment check:
    is_aligned = (addr & (N - 1)) == 0

    Example: addr = 0x18, N = 8
    0x18 & 0x07 = 0x00 -> aligned check

  Why alignment matters:
  - Misaligned access causes exceptions on some CPUs
  - Properly aligned access completes in a single memory read
  - Misaligned access requires two memory reads + merge
  - SIMD instructions (SSE, AVX) require 16/32-byte alignment
```

### 6.2 Hashing and Bitwise Operations

```python
# Hash table index calculation

# Method 1: Modulo operation (common)
index = hash(key) % table_size

# Method 2: Bit masking (fast, when table size is a power of 2)
# When table_size = 2^n:
index = hash(key) & (table_size - 1)

# Example: table_size = 1024 = 2^10
# hash(key) & 0x3FF  <- Extract lower 10 bits (0-1023)

# This is the secret behind HashMap/HashSet performance
# -> The reason table size is always kept as a power of 2

# FNV-1a hash (implemented with bitwise operations)
def fnv1a_32(data: bytes) -> int:
    FNV_OFFSET_BASIS = 0x811C9DC5
    FNV_PRIME = 0x01000193
    h = FNV_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFF  # Limit to 32 bits
    return h

# MurmurHash final mixing (bit shifts + XOR)
def murmur_finalizer(h):
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h ^= h >> 16
    return h
```

### 6.3 Color Codes and Bitwise Operations

```python
# RGBA color manipulation (8 bits each = 32 bits total)

# Color composition: 0xAARRGGBB (Alpha, Red, Green, Blue)
color = 0xFF8040C0  # A=255, R=128, G=64, B=192

# Extract each channel
alpha = (color >> 24) & 0xFF  # 255
red   = (color >> 16) & 0xFF  # 128
green = (color >>  8) & 0xFF  # 64
blue  = (color >>  0) & 0xFF  # 192

# Compose color from channels
def rgba(r, g, b, a=255):
    return (a << 24) | (r << 16) | (g << 8) | b

white = rgba(255, 255, 255)       # 0xFFFFFFFF
red_color = rgba(255, 0, 0)       # 0xFFFF0000
transparent = rgba(0, 0, 0, 0)    # 0x00000000

# Alpha blending
def blend(fg, bg, alpha):
    """alpha: 0-255"""
    return ((fg * alpha) + (bg * (255 - alpha))) // 255

# Brightness adjustment
def brighten(color, factor):
    """factor: 0.0-2.0 (1.0 = no change)"""
    a = (color >> 24) & 0xFF
    r = min(int(((color >> 16) & 0xFF) * factor), 255)
    g = min(int(((color >>  8) & 0xFF) * factor), 255)
    b = min(int(((color >>  0) & 0xFF) * factor), 255)
    return (a << 24) | (r << 16) | (g << 8) | b

# Web colors
# #FF8040 -> RGB(255, 128, 64)
hex_str = "FF8040"
r = int(hex_str[0:2], 16)  # 255
g = int(hex_str[2:4], 16)  # 128
b = int(hex_str[4:6], 16)  # 64
```

### 6.4 Bitwise Operations in Network Programming

```python
# IPv4 address manipulation

import struct
import socket

# Dotted notation -> 32-bit integer
def ip_to_int(ip_str):
    """'192.168.1.100' -> 0xC0A80164"""
    parts = ip_str.split('.')
    return (int(parts[0]) << 24) | (int(parts[1]) << 16) | \
           (int(parts[2]) << 8) | int(parts[3])

# 32-bit integer -> Dotted notation
def int_to_ip(ip_int):
    """0xC0A80164 -> '192.168.1.100'"""
    return f"{(ip_int >> 24) & 0xFF}.{(ip_int >> 16) & 0xFF}." \
           f"{(ip_int >> 8) & 0xFF}.{ip_int & 0xFF}"

# Subnet calculation
ip = ip_to_int('192.168.1.100')        # 0xC0A80164
mask = ip_to_int('255.255.255.0')      # 0xFFFFFF00

network = ip & mask                     # 192.168.1.0
broadcast = ip | (~mask & 0xFFFFFFFF)   # 192.168.1.255
host_part = ip & (~mask & 0xFFFFFFFF)   # 0.0.0.100

# Generate mask from CIDR prefix
def cidr_to_mask(prefix_len):
    """24 -> 0xFFFFFF00"""
    return (0xFFFFFFFF << (32 - prefix_len)) & 0xFFFFFFFF

# Get CIDR prefix from mask
def mask_to_cidr(mask):
    """0xFFFFFF00 -> 24"""
    return bin(mask).count('1')

# Calculate number of hosts
def host_count(prefix_len):
    """Number of usable hosts"""
    return (2 ** (32 - prefix_len)) - 2  # Exclude network and broadcast

print(host_count(24))   # 254
print(host_count(16))   # 65534
print(host_count(8))    # 16777214

# Check if two IPs belong to the same subnet
def same_subnet(ip1, ip2, mask):
    return (ip_to_int(ip1) & ip_to_int(mask)) == \
           (ip_to_int(ip2) & ip_to_int(mask))

print(same_subnet('192.168.1.100', '192.168.1.200', '255.255.255.0'))  # True
print(same_subnet('192.168.1.100', '192.168.2.100', '255.255.255.0'))  # False


# TCP flags (bitfields)
TCP_FIN = 0x01  # Connection termination
TCP_SYN = 0x02  # Connection initiation
TCP_RST = 0x04  # Reset
TCP_PSH = 0x08  # Push
TCP_ACK = 0x10  # Acknowledgment
TCP_URG = 0x20  # Urgent

# SYN-ACK packet flags
flags = TCP_SYN | TCP_ACK  # 0x12

# Flag checks
is_syn = bool(flags & TCP_SYN)      # True
is_fin = bool(flags & TCP_FIN)      # False
is_syn_ack = (flags & (TCP_SYN | TCP_ACK)) == (TCP_SYN | TCP_ACK)  # True
```

### 6.5 Bitwise Operations in Cryptography

```python
# XOR cipher (basis of stream ciphers)

def xor_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """XOR encryption (decryption is the same operation)"""
    return bytes(p ^ k for p, k in zip(plaintext, key * (len(plaintext) // len(key) + 1)))

# Property of XOR: A ^ K ^ K = A
message = b"Hello, World!"
key = b"SECRET"
encrypted = xor_encrypt(message, key)
decrypted = xor_encrypt(encrypted, key)
print(decrypted)  # b'Hello, World!'


# Bit rotation (heavily used in cryptographic algorithms)
def rotate_left_32(n, d):
    """32-bit left rotation"""
    return ((n << d) | (n >> (32 - d))) & 0xFFFFFFFF

def rotate_right_32(n, d):
    """32-bit right rotation"""
    return ((n >> d) | (n << (32 - d))) & 0xFFFFFFFF

# SHA-256's Ch and Maj functions
def ch(x, y, z):
    """Choice: if x is 1, take y's bit; otherwise take z's bit"""
    return (x & y) ^ (~x & z)

def maj(x, y, z):
    """Majority: majority vote"""
    return (x & y) ^ (x & z) ^ (y & z)


# Feistel structure (basic structure of block ciphers like DES)
def feistel_round(left, right, round_key):
    """One round of the Feistel structure"""
    new_left = right
    new_right = left ^ f(right, round_key)  # f is the encryption function
    return new_left, new_right

# Inverse transformation (decryption) simply reverses the key order
def feistel_round_inv(left, right, round_key):
    new_right = left
    new_left = right ^ f(left, round_key)
    return new_left, new_right
```

---

## 7. Endianness (Byte Order)

### 7.1 Big-Endian and Little-Endian

```
Memory layout of 32-bit integer 0x12345678:

  Big-Endian (BE): Most significant byte at the lowest address
  Address: 0x00  0x01  0x02  0x03
  Value:   0x12  0x34  0x56  0x78
  -> Natural order for humans (most significant digit on the left)

  Little-Endian (LE): Least significant byte at the lowest address
  Address: 0x00  0x01  0x02  0x03
  Value:   0x78  0x56  0x34  0x12
  -> Adopted by Intel/AMD (x86, x64)

  Why little-endian is dominant:
  - 8-bit and 16-bit data share the same starting address
  - Matches the process of addition starting from the least significant byte
  - Simplifies byte-width conversion

  Endianness by platform:
  +----------------------+------------+
  | Platform             | Endianness |
  +----------------------+------------+
  | x86, x64 (Intel/AMD) | Little     |
  | ARM (default mode)    | Little     |
  | Network (TCP/IP)      | Big        |
  | Java (JVM)            | Big        |
  | RISC-V               | Little     |
  | MIPS                 | Switchable |
  | PowerPC              | Big        |
  +----------------------+------------+

  Network byte order = Big-endian
  Host byte order = CPU-dependent (usually little)
```

### 7.2 Endian Conversion Implementations

```python
import struct

# Python's struct module
# '<' = little-endian, '>' = big-endian, '!' = network (BE)

value = 0x12345678

# Pack (integer -> byte sequence)
be_bytes = struct.pack('>I', value)  # b'\x12\x34\x56\x78'
le_bytes = struct.pack('<I', value)  # b'\x78\x56\x34\x12'

# Unpack (byte sequence -> integer)
be_value = struct.unpack('>I', be_bytes)[0]  # 0x12345678
le_value = struct.unpack('<I', le_bytes)[0]  # 0x12345678

# int.from_bytes / int.to_bytes (Python 3.2+)
n = int.from_bytes(b'\x12\x34\x56\x78', byteorder='big')    # 0x12345678
n = int.from_bytes(b'\x78\x56\x34\x12', byteorder='little') # 0x12345678

data = (0x12345678).to_bytes(4, byteorder='big')    # b'\x12\x34\x56\x78'
data = (0x12345678).to_bytes(4, byteorder='little')  # b'\x78\x56\x34\x12'

# Byte swap (implemented with bitwise operations)
def bswap32(n):
    """32-bit byte swap"""
    return ((n & 0xFF000000) >> 24) | \
           ((n & 0x00FF0000) >>  8) | \
           ((n & 0x0000FF00) <<  8) | \
           ((n & 0x000000FF) << 24)

print(hex(bswap32(0x12345678)))  # 0x78563412

def bswap16(n):
    """16-bit byte swap"""
    return ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8)
```

```c
// Endian conversion in C

#include <arpa/inet.h>  // POSIX

// Network byte order conversion functions
uint32_t net_val = htonl(0x12345678);  // Host TO Network Long
uint16_t net_s   = htons(0x1234);      // Host TO Network Short
uint32_t host_val = ntohl(net_val);    // Network TO Host Long
uint16_t host_s   = ntohs(net_s);     // Network TO Host Short

// GCC built-ins
uint32_t swapped = __builtin_bswap32(0x12345678);  // 0x78563412
uint64_t swapped64 = __builtin_bswap64(value64);
```

```rust
// Endian conversion in Rust

let n: u32 = 0x12345678;

// Byte swap
let swapped = n.swap_bytes();  // 0x78563412

// Endian conversion
let be_bytes = n.to_be_bytes();  // [0x12, 0x34, 0x56, 0x78]
let le_bytes = n.to_le_bytes();  // [0x78, 0x56, 0x34, 0x12]

let from_be = u32::from_be_bytes([0x12, 0x34, 0x56, 0x78]);  // 0x12345678
let from_le = u32::from_le_bytes([0x78, 0x56, 0x34, 0x12]);  // 0x12345678
```

---

## 8. Practice Exercises

### Exercise 1: Base Conversion (Fundamentals)
Perform the following conversions by hand:
1. Decimal `173` -> Binary -> Hexadecimal
2. Hexadecimal `0xDEAD` -> Decimal
3. Binary `1011 0110` -> Decimal -> Octal
4. Decimal `0.6875` -> Binary
5. Octal `0o1777` -> Hexadecimal -> Decimal

### Exercise 2: Bitwise Operation Puzzles (Applied)
Implement the following using only bitwise operations (no arithmetic operations allowed):
1. Addition of two integers
2. Count the number of set bits (1s) in an integer
3. Negate (two's complement)
4. Compare two integers (branchless)

### Exercise 3: IP Address Calculation (Advanced)
Given IPv4 address `192.168.10.50` and subnet mask `255.255.255.0 (/24)`, use bitwise operations to compute the network address and broadcast address.

### Exercise 4: Bitfield Design (Practical)
Design a file permission system. The following requirements must be met:
- Three types of access principals: user, group, and other
- read, write, and execute permissions for each principal
- Represent all permissions in a single integer
- Implement functions for checking, granting, and revoking permissions

### Exercise 5: Endian Conversion (Applied)
Given byte sequence `[0x48, 0x65, 0x6C, 0x6C]`:
1. Interpret as a big-endian 32-bit integer
2. Interpret as a little-endian 32-bit integer
3. Interpret as an ASCII string

### Solution Examples

```python
# Exercise 2-1: Addition using only bitwise operations
def add(a, b):
    """Implement addition using only bitwise operations"""
    while b:
        carry = a & b      # Carry bits
        a = a ^ b           # Addition without carry
        b = carry << 1      # Propagate carry to the next digit
    return a

# Test
print(add(5, 3))    # 8
print(add(100, 200)) # 300

# Detailed walkthrough:
# a=5 (101), b=3 (011)
# Round 1: carry=001, a=110, b=010
# Round 2: carry=010, a=100, b=100
# Round 3: carry=100, a=000, b=1000
# Round 4: carry=000, a=1000, b=0000 -> done
# Result: 1000 = 8 check


# Exercise 2-2: Bit count
def count_bits(n):
    """Brian Kernighan's algorithm"""
    count = 0
    while n:
        n &= n - 1  # Clear the lowest set bit
        count += 1
    return count

print(count_bits(0b1011_0110))  # 5


# Exercise 2-3: Negate (two's complement)
def negate(n):
    """~n + 1 = -n"""
    return add(~n, 1)  # Bit inversion + 1


# Exercise 4: File permissions
class FilePermission:
    # Bit position definitions
    OTHER_EXECUTE = 0  # Bit 0
    OTHER_WRITE   = 1  # Bit 1
    OTHER_READ    = 2  # Bit 2
    GROUP_EXECUTE = 3  # Bit 3
    GROUP_WRITE   = 4  # Bit 4
    GROUP_READ    = 5  # Bit 5
    USER_EXECUTE  = 6  # Bit 6
    USER_WRITE    = 7  # Bit 7
    USER_READ     = 8  # Bit 8

    def __init__(self, mode=0):
        self.mode = mode

    @classmethod
    def from_octal(cls, octal_str):
        """'755' -> FilePermission"""
        return cls(int(octal_str, 8))

    def has_permission(self, bit):
        """Check if the specified bit permission is set"""
        return bool(self.mode & (1 << bit))

    def grant(self, bit):
        """Grant permission"""
        self.mode |= (1 << bit)

    def revoke(self, bit):
        """Revoke permission"""
        self.mode &= ~(1 << bit)

    def toggle(self, bit):
        """Toggle permission"""
        self.mode ^= (1 << bit)

    def __repr__(self):
        chars = ''
        for label, r, w, x in [
            ('u', self.USER_READ, self.USER_WRITE, self.USER_EXECUTE),
            ('g', self.GROUP_READ, self.GROUP_WRITE, self.GROUP_EXECUTE),
            ('o', self.OTHER_READ, self.OTHER_WRITE, self.OTHER_EXECUTE),
        ]:
            chars += 'r' if self.has_permission(r) else '-'
            chars += 'w' if self.has_permission(w) else '-'
            chars += 'x' if self.has_permission(x) else '-'
        return chars

perm = FilePermission.from_octal('755')
print(perm)  # rwxr-xr-x
print(perm.has_permission(FilePermission.USER_WRITE))  # True
print(perm.has_permission(FilePermission.OTHER_WRITE))  # False
```

---

## 9. Debugging and Troubleshooting

### 9.1 Common Mistakes

```python
# Mistake 1: Watch out for C's octal literals
# In C:
# int n = 010;  // This is 8! (octal 10), not decimal 10

# Mistake 2: JavaScript bitwise operations use 32 bits
# JavaScript:
# 0xFFFFFFFF | 0  -> -1 (interpreted as signed 32-bit)
# Correct: 0xFFFFFFFF >>> 0  -> 4294967295

# Mistake 3: Python's ~ operator
n = 5
print(~n)     # -6 (-(n+1))
# To invert within an 8-bit range:
print(~n & 0xFF)  # 250 (0b11111010)

# Mistake 4: Signed right shift
# -1 >> 1 = -1 (arithmetic right shift: sign bit is preserved)
# In Python, integers have infinite precision:
# -1 = ...1111_1111_1111 (all bits are 1)
# >> 1 = ...1111_1111_1111 (still all bits are 1) = -1

# Mistake 5: Bitwise operations on floating-point numbers
# Python: Bitwise operations only work on int types
# Cannot use on float
# 3.14 & 0xFF  -> TypeError

# Mistake 6: Overflow
# In C: uint8_t n = 255; n + 1 = 0 (wraparound)
# In Python, integers never overflow (arbitrary precision)
```

### 9.2 Debugging Tools

```python
# Bit representation visualization helper

def show_bits(n, width=8):
    """Display the bit representation of an integer in a readable format"""
    if n < 0:
        # Display two's complement representation
        n = n & ((1 << width) - 1)
    bits = format(n, f'0{width}b')
    # Group into 4-bit chunks
    grouped = ' '.join(bits[i:i+4] for i in range(0, len(bits), 4))
    print(f"Dec: {n:>{width//3+3}d}  Hex: 0x{n:0{width//4}X}  Bin: {grouped}")

show_bits(42)       # Dec:  42  Hex: 0x2A  Bin: 0010 1010
show_bits(255)      # Dec: 255  Hex: 0xFF  Bin: 1111 1111
show_bits(0, 16)    # Dec:   0  Hex: 0x0000  Bin: 0000 0000 0000 0000

def show_operation(a, b, op_name, op_func, width=8):
    """Visualize the process of a bitwise operation"""
    result = op_func(a, b) & ((1 << width) - 1)
    a_bits = format(a & ((1 << width) - 1), f'0{width}b')
    b_bits = format(b & ((1 << width) - 1), f'0{width}b')
    r_bits = format(result, f'0{width}b')

    print(f"  {a_bits}  ({a})")
    print(f"{op_name} {b_bits}  ({b})")
    print(f"  {'─' * width}")
    print(f"  {r_bits}  ({result})")
    print()

show_operation(0b11001010, 0b10110110, '&', lambda a, b: a & b)
#   11001010  (202)
# & 10110110  (182)
#   --------
#   10000010  (130)

show_operation(0b11001010, 0b10110110, '|', lambda a, b: a | b)
show_operation(0b11001010, 0b10110110, '^', lambda a, b: a ^ b)
```

### 9.3 Investigating Binary Data

```bash
# Hex dump of a file
xxd file.bin | head -20
# 00000000: 504b 0304 1400 0000 0800 ...  PK..........

# hexdump (alternative format)
hexdump -C file.bin | head -20

# Read from a specific byte offset
xxd -s 0x100 -l 32 file.bin

# Binary comparison
xxd file1.bin > /tmp/hex1.txt
xxd file2.bin > /tmp/hex2.txt
diff /tmp/hex1.txt /tmp/hex2.txt

# Analyze binary files with Python
python3 -c "
with open('file.bin', 'rb') as f:
    data = f.read(16)
    print(' '.join(f'{b:02X}' for b in data))
    # Check magic numbers
    magic = {
        b'\\x89PNG': 'PNG image',
        b'\\xff\\xd8\\xff': 'JPEG image',
        b'PK': 'ZIP/XLSX/DOCX',
        b'\\x7fELF': 'ELF executable',
        b'GIF8': 'GIF image',
        b'%PDF': 'PDF file',
    }
    for sig, name in magic.items():
        if data[:len(sig)] == sig:
            print(f'File format: {name}')
"
```


---

## Design Decision Guide

### Selection Criteria Matrix

The following summarizes the decision criteria for making technical choices.

| Criterion | Prioritize When | Acceptable to Compromise When |
|-----------|----------------|------------------------------|
| Performance | Real-time processing, large-scale data | Admin dashboards, batch processing |
| Maintainability | Long-term operations, team development | Prototypes, short-term projects |
| Scalability | Services expecting growth | Internal tools, fixed user base |
| Security | Personal data, financial data | Public data, internal use |
| Development Speed | MVP, time-to-market | Quality-focused, mission-critical |

### Architecture Pattern Selection

```
+--------------------------------------------------+
|         Architecture Selection Flow               |
+--------------------------------------------------+
|                                                    |
|  (1) Team size?                                    |
|    +-- Small (1-5) -> Monolith                     |
|    +-- Large (10+) -> Go to (2)                    |
|                                                    |
|  (2) Deployment frequency?                         |
|    +-- Weekly or less -> Monolith + module split    |
|    +-- Daily/multiple -> Go to (3)                 |
|                                                    |
|  (3) Team independence?                            |
|    +-- High -> Microservices                       |
|    +-- Medium -> Modular monolith                  |
|                                                    |
+--------------------------------------------------+
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze from the following perspectives:

**1. Short-term vs Long-term Cost**
- A short-term fast approach may become technical debt in the long run
- Conversely, over-engineering incurs high short-term costs and can delay the project

**2. Consistency vs Flexibility**
- A unified technology stack has lower learning costs
- Diverse technology adoption enables best-fit solutions but increases operational costs

**3. Level of Abstraction**
- High abstraction provides high reusability but can make debugging difficult
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
        """Describe the background and challenge"""
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

## FAQ

### Q1: Why is hexadecimal so widely used in programming?
**A**: Because 4 bits of binary correspond exactly to 1 hexadecimal digit. `0xFF` immediately tells you `1111 1111`, but `255` does not convey this intuitively. It is convenient for memory addresses, color codes (#FF0000), and byte sequence representation.

### Q2: Did ternary computers ever exist?
**A**: The Soviet Setun (1958) is famous as a ternary computer. Theoretically, base-3 has the highest information efficiency (3 is the integer closest to the natural logarithm base e). However, in practice, the reliability of two-state transistor switching was overwhelmingly superior, and binary became the standard.

### Q3: How often are bitwise operations used in practice?
**A**: It depends on the field. In web development, they are rarely used. In systems programming, networking (IP masks), cryptography, game development, and embedded systems, they are used frequently. Even when "not used," understanding the principles helps with performance optimization and bug analysis.

### Q4: Do endianness differences actually cause bugs?
**A**: Frequently. Especially when reading byte sequences directly as integers in network programming, or when exchanging binary files between different architectures. Always use conversion functions like `htonl()/ntohl()`, and explicitly specify the protocol's endianness.

### Q5: Are bitwise performance improvements still effective on modern CPUs?
**A**: Simple substitutions (`n * 2` -> `n << 1`) are automatically optimized by modern compilers, so manual conversion is unnecessary. However, algorithms that exploit bit-level parallelism (popcount, bitboards, SIMD, etc.) still provide significant benefits. Avoid sacrificing readability for micro-optimizations, but algorithmic use of bitwise operations remains important.

### Q6: Do quantum computers use binary?
**A**: Quantum computers use quantum bits (qubits). Qubits differ from classical bits in that they can exist in a superposition of 0 and 1, but measurement results are always 0 or 1. Since quantum computer outputs are ultimately converted to classical bits, the input/output interface is binary.

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Binary | Directly maps to transistor ON/OFF. High noise resistance |
| Hexadecimal | 4 bits of binary = 1 digit. Standard notation for memory/byte sequences |
| Octal | 3 bits of binary = 1 digit. Used in Unix permissions (755, etc.) |
| Base Conversion | Two approaches: division method and subtraction method. Fractions use the multiplication method |
| Bitwise Operations | AND, OR, XOR, NOT, shift. Used for flag management and optimization |
| Shift Operations | Left shift = x2^n, right shift = /2^n. Note the arithmetic/logical distinction |
| Endianness | Big = MSB first, Little = LSB first |
| Units | Two systems: SI (KB=1000) and IEC (KiB=1024) |
| Practical Applications | IP calculation, color manipulation, cryptography, hashing, game AI |

---

## Recommended Next Reading

---

## References
1. Petzold, C. "Code: The Hidden Language of Computer Hardware and Software." 2nd Edition, Microsoft Press, 2022.
2. Warren, H. S. "Hacker's Delight." 2nd Edition, Addison-Wesley, 2012.
3. Bryant, R. E. & O'Hallaron, D. R. "Computer Systems: A Programmer's Perspective." Chapter 2.
4. Shannon, C. E. "A Mathematical Theory of Communication." Bell System Technical Journal, 1948.
5. Knuth, D. E. "The Art of Computer Programming, Volume 4A: Combinatorial Algorithms, Part 1." Addison-Wesley, 2011.
6. IEEE 754-2019. "IEEE Standard for Floating-Point Arithmetic."
7. RFC 791. "Internet Protocol." IETF, 1981.
